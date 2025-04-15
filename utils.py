from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from baukit import Trace
from tqdm import tqdm
import einops
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

    def __init__(self, activation_dim: int, dict_size: int, k: int, data_mean = None, tokens_to_combine=None):
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

    def from_pretrained(path, k: int, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = torch.load(path, weights_only=True, map_location='cpu')
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        autoencoder = AutoEncoderTopK(activation_dim, dict_size, k)
        autoencoder.load_state_dict(state_dict)
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

def prepare_streaming_dataset(tokenizer, dataset_name, max_length, batch_size, num_datapoints=None, num_cpu_cores=6):
    """Create a generator that streams batches from the dataset"""
    split = "train"
    split_text = f"{split}[:{num_datapoints}]" if num_datapoints else split
    
    # Load the dataset
    # dataset = load_dataset(dataset_name, split=split_text, streaming=True)
    dataset = load_dataset(dataset_name, split=split_text)
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