from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from baukit import Trace
from tqdm import tqdm
import einops
from einops import rearrange
from huggingface_hub import HfApi, create_repo, login

def push_sae_to_huggingface(
    save_dir,
    model_save_path,
    cfg_save_path,
    hf_repo_id,
    hf_token=None
):
    """
    Push existing SAE model and config files to Hugging Face Hub.
    
    Args:
        save_dir: Local directory where model files are saved
        model_save_path: Path to the model file relative to save_dir
        cfg_save_path: Path to the config file relative to save_dir
        hf_repo_id: Hugging Face repository ID (e.g., "username/repo-name")
        hf_token: Hugging Face API token (default: None, will use token from cache)
    """
    import os
    
    # Initialize API (this will use cached token if hf_token is None)
    if hf_token:
        login(token=hf_token)
    api = HfApi()
    
    # Create or ensure repo exists
    try:
        create_repo(repo_id=hf_repo_id, exist_ok=True, token=hf_token)
        print(f"Repository {hf_repo_id} ready")
    except Exception as e:
        print(f"Error with repository: {e}")
        return
    
    # Upload model
    local_model_path = os.path.join(save_dir, model_save_path)
    try:
        model_url = api.upload_file(
            path_or_fileobj=local_model_path,
            path_in_repo=model_save_path,
            repo_id=hf_repo_id,
            token=hf_token
        )
        print(f"Uploaded model to {model_url}")
    except Exception as e:
        print(f"Error uploading model: {e}")
    
    # Upload config
    local_cfg_path = os.path.join(save_dir, cfg_save_path)
    try:
        cfg_url = api.upload_file(
            path_or_fileobj=local_cfg_path,
            path_in_repo=cfg_save_path,
            repo_id=hf_repo_id,
            token=hf_token
        )
        print(f"Uploaded config to {cfg_url}")
    except Exception as e:
        print(f"Error uploading config: {e}")

class AutoEncoderTopK(nn.Module):
    """
    The top-k autoencoder architecture and initialization used in https://arxiv.org/abs/2406.04093
    NOTE: (From Adam Karvonen) There is an unmaintained implementation using Triton kernels in the topk-triton-implementation branch.
    We abandoned it as we didn't notice a significant speedup and it added complications, which are noted
    in the AutoEncoderTopK class docstring in that branch.

    With some additional effort, you can traisn a Top-K SAE with the Triton kernels and modify the state dict for compatibility with this class.
    Notably, the Triton kernels currently have the decoder to be stored in nn.Parameter, not nn.Linear, and the decoder weights must also
    be stored in the same shape as the encoder.
    """

    def __init__(self, activation_dim: int, dict_size: int, k: int, data_mean = None, tokens_to_combine=None, embedding=None):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.k = k
        self.tokens_to_combine = tokens_to_combine

        self.encoder = nn.Linear(activation_dim, dict_size)
        self.encoder.bias.data.zero_()

        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.decoder.weight.data = self.encoder.weight.data.clone().T
        self.set_decoder_norm_to_unit_norm()

        self.b_dec = nn.Parameter(torch.zeros(activation_dim))
        self.per_token_bias = embedding


    def encode(self, x: torch.Tensor, return_topk: bool = False):
        post_relu_feat_acts_BF = nn.functional.relu(self.encoder(x - self.b_dec))
        post_topk = post_relu_feat_acts_BF.topk(self.k, sorted=False, dim=-1)

        # We can't split immediately due to nnsight
        tops_acts_BK = post_topk.values
        top_indices_BK = post_topk.indices

        buffer_BF = torch.zeros_like(post_relu_feat_acts_BF)
        encoded_acts_BF = buffer_BF.scatter_(dim=-1, index=top_indices_BK, src=tops_acts_BK)

        if return_topk:
            return encoded_acts_BF, tops_acts_BK, top_indices_BK
        else:
            return encoded_acts_BF

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x) + self.b_dec

    def forward(self, x: torch.Tensor, output_features: bool = False):
        encoded_acts_BF = self.encode(x)
        x_hat_BD = self.decode(encoded_acts_BF)
        if not output_features:
            return x_hat_BD
        else:
            return x_hat_BD, encoded_acts_BF

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        eps = torch.finfo(self.decoder.weight.dtype).eps
        norm = torch.norm(self.decoder.weight.data, dim=0, keepdim=True)
        self.decoder.weight.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        assert self.decoder.weight.grad is not None  # keep pyright happy

        parallel_component = einops.einsum(
            self.decoder.weight.grad,
            self.decoder.weight.data,
            "d_in d_sae, d_in d_sae -> d_sae",
        )
        self.decoder.weight.grad -= einops.einsum(
            parallel_component,
            self.decoder.weight.data,
            "d_sae, d_in d_sae -> d_in d_sae",
        )

    def from_pretrained(path, k: int, device=None, embedding=None):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = torch.load(path, weights_only=True, map_location='cpu')
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        if 'per_token_bias.bias' in state_dict:
            num_tokens, d_model = state_dict["per_token_bias.bias"].shape
            embedding = EmbeddingBias(nn.Embedding(num_tokens, d_model))
            # del state_dict["per_token_bias.bias"]
        else: 
            embedding = None
        autoencoder = AutoEncoderTopK(activation_dim, dict_size, k)
        autoencoder.per_token_bias = embedding
        autoencoder.load_state_dict(state_dict)
        # autoencoder.per_token_bias = embedding
        if device is not None:
            autoencoder.to(device)
        return autoencoder
    
class DotDict(dict):
    """Dictionary that allows accessing keys via dot notation."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class EmbeddingBias(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        num_tokens = embedding.weight.size(0)
        d_model = embedding.weight.size(1)
        self.bias = nn.Parameter(torch.zeros(num_tokens, d_model))
        self.bias.data.copy_(embedding.weight)
        self.bias.requires_grad = True

    def forward(self, x):
        return self.bias[x]

class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, tokenizer, name= None, batch_size=1, max_length=128, total_batches=None):
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.name = name
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.total_batches= total_batches
        d = load_dataset(dataset_name, name, split='train', streaming=True)
        self.dataset = d.map(lambda x: tokenizer(x["text"])).filter(
            lambda x: len(x["input_ids"]) >= max_length).map(
            lambda x: {"input_ids": x["input_ids"][:max_length]},
            )
        self.dataset_iterator = iter(self.dataset)

    def reset_iterator(self):
        self.dataset_iterator = iter(self.dataset)

    def next(self):
      batches = []
      try:
          while len(batches) < self.batch_size:
              batch = next(self.dataset_iterator)
              batches.append(batch)
          # return_batch_size amount of iterables in a torch tensor
          return torch.stack([torch.tensor(batch["input_ids"]) for batch in batches])
      except StopIteration:
          return None


def prepare_streaming_dataset(tokenizer, dataset_name, max_length, batch_size, num_datapoints=None, num_cpu_cores=6, name=None):
    """Create a generator that streams batches from the dataset"""
    split = "train"
    split_text = f"{split}[:{num_datapoints}]" if num_datapoints else split
    
    # Load the dataset
    # dataset = load_dataset(dataset_name, split=split_text, streaming=True)
    dataset = load_dataset(dataset_name, name=name, split=split_text)
    current_batch = []
    
    def process_text(text):
        """Helper function to tokenize text"""
        return tokenizer(tokenizer.bos_token + text)['input_ids']
    
    for item in dataset:
        # Tokenize the text
        input_ids = process_text(item['text'])
        
        # Only keep sequences that are long enough
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]  # Truncate if necessary
            current_batch.append(torch.tensor(input_ids))
            
            # When we have enough samples, yield a batch
            if len(current_batch) == batch_size:
                # Pad the sequences in the batch to the same length
                # padded_batch = pad_sequence(current_batch, batch_first=True)
                yield torch.stack(current_batch)
                current_batch = []
    
    # Yield any remaining samples in the last batch
    if current_batch:
        yield torch.stack(current_batch)


class ResidualMLP(nn.Module):
    def __init__(self, hidden_dim, expansion_size=4, init_scale=0.01):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*expansion_size),
            nn.ReLU(),
            nn.Linear(hidden_dim*expansion_size, hidden_dim)
        )
        
        # Initialize close to zero for residual behavior
        with torch.no_grad():
            for layer in self.mlp:
                if isinstance(layer, nn.Linear):
                    layer.weight.data.normal_(0, init_scale)
                    layer.bias.data.zero_()
    
    def forward(self, x):
        if isinstance(x, tuple):
            return (x[0] + self.mlp(x[0]), x[1])
        return x + self.mlp(x)

def calculate_fvu(x_orig, x_pred):
    """Calculate Fraction of Variance Unexplained"""
    mean = x_orig.mean(dim=0, keepdim=True)
    numerator = torch.mean(torch.sum((x_orig - x_pred)**2, dim=-1))
    denominator = torch.mean(torch.sum((x_orig - mean)**2, dim=-1))
    return numerator / (denominator + 1e-6)


'''Visualization Utils'''


def make_colorbar(min_value, max_value, white = 255, red_blue_ness = 250, positive_threshold = 0.01, negative_threshold = 0.01):
    # Add color bar
    colorbar = ""
    num_colors = 4
    if(min_value < -negative_threshold):
        for i in range(num_colors, 0, -1):
            ratio = i / (num_colors)
            value = round((min_value*ratio),1)
            text_color = "255,255,255" if ratio > 0.5 else "0,0,0"
            colorbar += f'<span style="background-color:rgba(255, {int(red_blue_ness-(red_blue_ness*ratio))},{int(red_blue_ness-(red_blue_ness*ratio))},1); color:rgb({text_color})">&nbsp{value}&nbsp</span>'
    # Do zero
    colorbar += f'<span style="background-color:rgba({white},{white},{white},1);color:rgb(0,0,0)">&nbsp0.0&nbsp</span>'
    # Do positive
    if(max_value > positive_threshold):
        for i in range(1, num_colors+1):
            ratio = i / (num_colors)
            value = round((max_value*ratio),1)
            text_color = "255,255,255" if ratio > 0.5 else "0,0,0"
            colorbar += f'<span style="background-color:rgba({int(red_blue_ness-(red_blue_ness*ratio))},{int(red_blue_ness-(red_blue_ness*ratio))},255,1);color:rgb({text_color})">&nbsp{value}&nbsp</span>'
    return colorbar

def value_to_color(activation, max_value, min_value, white = 255, red_blue_ness = 250, positive_threshold = 0.01, negative_threshold = 0.01):
    if activation > positive_threshold:
        ratio = activation/max_value
        text_color = "0,0,0" if ratio <= 0.5 else "255,255,255"  
        background_color = f'rgba({int(red_blue_ness-(red_blue_ness*ratio))},{int(red_blue_ness-(red_blue_ness*ratio))},255,1)'
    elif activation < -negative_threshold:
        ratio = activation/min_value
        text_color = "0,0,0" if ratio <= 0.5 else "255,255,255"  
        background_color = f'rgba(255, {int(red_blue_ness-(red_blue_ness*ratio))},{int(red_blue_ness-(red_blue_ness*ratio))},1)'
    else:
        text_color = "0,0,0"
        background_color = f'rgba({white},{white},{white},1)'
    return text_color, background_color

def convert_token_array_to_list(array):
    if isinstance(array, torch.Tensor):
        if array.dim() == 1:
            array = [array.tolist()]
        elif array.dim()==2:
            array = array.tolist()
        else: 
            raise NotImplementedError("tokens must be 1 or 2 dimensional")
    elif isinstance(array, list):
        # ensure it's a list of lists
        if isinstance(array[0], int):
            array = [array]
    return array

def tokens_and_activations_to_html(toks, activations, tokenizer, logit_diffs=None, model_type="causal", text_above_each_act=None):
    # text_spacing = "0.07em"
    text_spacing = "0.00em"
    toks = convert_token_array_to_list(toks)
    activations = convert_token_array_to_list(activations)
    # toks = [[tokenizer.decode(t).replace('Ġ', '&nbsp').replace('\n', '↵') for t in tok] for tok in toks]
    toks = [[tokenizer.decode(t).replace('Ġ', '&nbsp').replace('\n', '\\n') for t in tok] for tok in toks]
    highlighted_text = []
    # Make background black
    # highlighted_text.append('<body style="background-color:black; color: white;">')
    highlighted_text.append("""
<body style="background-color: black; color: white;">
""")
    max_value = max([max(activ) for activ in activations])
    min_value = min([min(activ) for activ in activations])
    if(logit_diffs is not None and model_type != "reward_model"):
        logit_max_value = max([max(activ) for activ in logit_diffs])
        logit_min_value = min([min(activ) for activ in logit_diffs])

    # Add color bar
    highlighted_text.append("Token Activations: " + make_colorbar(min_value, max_value))
    if(logit_diffs is not None and model_type != "reward_model"):
        highlighted_text.append('<div style="margin-top: 0.1em;"></div>')
        highlighted_text.append("Logit Diff: " + make_colorbar(logit_min_value, logit_max_value))
    
    highlighted_text.append('<div style="margin-top: 0.5em;"></div>')
    for seq_ind, (act, tok) in enumerate(zip(activations, toks)):
        if(text_above_each_act is not None):
            highlighted_text.append(f'<span>{text_above_each_act[seq_ind]}</span>')
        for act_ind, (a, t) in enumerate(zip(act, tok)):
            if(logit_diffs is not None and model_type != "reward_model"):
                highlighted_text.append('<div style="display: inline-block;">')
            text_color, background_color = value_to_color(a, max_value, min_value)
            highlighted_text.append(f'<span style="background-color:{background_color};margin-right: {text_spacing}; color:rgb({text_color})">{t.replace(" ", "&nbsp")}</span>')
            if(logit_diffs is not None and model_type != "reward_model"):
                logit_diffs_act = logit_diffs[seq_ind][act_ind]
                _, logit_background_color = value_to_color(logit_diffs_act, logit_max_value, logit_min_value)
                highlighted_text.append(f'<div style="display: block; margin-right: {text_spacing}; height: 10px; background-color:{logit_background_color}; text-align: center;"></div></div>')
        if(logit_diffs is not None and model_type=="reward_model"):
            reward_change = logit_diffs[seq_ind].item()
            text_color, background_color = value_to_color(reward_change, 10, -10)
            highlighted_text.append(f'<br><span>Reward: </span><span style="background-color:{background_color};margin-right: {text_spacing}; color:rgb({text_color})">{reward_change:.2f}</span>')
        highlighted_text.append('<div style="margin-top: 0.2em;"></div>')
        # highlighted_text.append('<br><br>')
    # highlighted_text.append('</body>')
    highlighted_text = ''.join(highlighted_text)
    return highlighted_text
def save_token_display(tokens, activations, tokenizer, path, save=True, logit_diffs=None, show=False, model_type="causal"):
    html = tokens_and_activations_to_html(tokens, activations, tokenizer, logit_diffs, model_type=model_type)
    # if(save):
    #     imgkit.from_string(html, path)
    # if(show):
    return display(HTML(html))

def get_feature_indices(feature_activations, k=10, setting="max"):
    # Sort the features by activation, get the indices
    batch_size, seq_len = feature_activations.shape
    feature_activations = rearrange(feature_activations, 'b s -> (b s)')
    if setting=="max":
        found_indices = torch.argsort(feature_activations, descending=True)[:k]
    elif setting=="uniform":
        # min_value = torch.min(feature_activations)
        min_value = torch.min(feature_activations)
        max_value = torch.max(feature_activations)

        # Define the number of bins
        num_bins = k

        # Calculate the bin boundaries as linear interpolation between min and max
        bin_boundaries = torch.linspace(min_value, max_value, num_bins + 1)

        # Assign each activation to its respective bin
        bins = torch.bucketize(feature_activations, bin_boundaries)

        # Initialize a list to store the sampled indices
        sampled_indices = []

        # Sample from each bin
        for bin_idx in torch.unique(bins):
            if(bin_idx==0): # Skip the first one. This is below the median
                continue
            # Get the indices corresponding to the current bin
            bin_indices = torch.nonzero(bins == bin_idx, as_tuple=False).squeeze(dim=1)
            
            # Randomly sample from the current bin
            sampled_indices.extend(np.random.choice(bin_indices, size=1, replace=False))

        # Convert the sampled indices to a PyTorch tensor & reverse order
        found_indices = torch.tensor(sampled_indices).long().flip(dims=[0])
    else: # random
        # get nonzero indices
        nonzero_indices = torch.nonzero(feature_activations)[:, 0]
        # shuffle
        shuffled_indices = nonzero_indices[torch.randperm(nonzero_indices.shape[0])]
        found_indices = shuffled_indices[:k]
    d_indices = found_indices // seq_len
    s_indices = found_indices % seq_len
    return d_indices, s_indices

def get_feature_datapoints(d_idx, seq_pos_idx, all_activations, all_tokens, tokenizer, append=0):
    full_activations = []
    partial_activations = []
    text_list = []
    full_text = []
    token_list = []
    full_token_list = []
    for md, s_ind in zip(d_idx, seq_pos_idx):
        md = int(md)
        s_ind = int(s_ind)
        # full_tok = torch.tensor(dataset[md]["input_ids"])
        
        full_tok = all_tokens[md]
        # [tokenizer.decode(t) for t in tokens[0]]

        full_text.append(tokenizer.decode(full_tok))
        # we want to add append more tokens, but only 
        tok = full_tok[:s_ind+1+append]
        # tok = dataset[md]["input_ids"][:s_ind+1]
        full_activations.append(all_activations[md].tolist())
        partial_activations.append(all_activations[md][:s_ind+1+append].tolist())
        text = tokenizer.decode(tok)
        text_list.append(text)
        token_list.append(tok)
        full_token_list.append(full_tok)
    return text_list, full_text, token_list, full_token_list, partial_activations, full_activations
