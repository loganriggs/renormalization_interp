# %%
import torch
from utils import *
# Example usage:
from transformers import AutoModelForCausalLM, AutoTokenizer
# model_name = "gpt2"
# model_name = "HuggingFaceTB/SmolLM-360M"
model_name = "HuggingFaceTB/SmolLM-135M"
m_name_save= model_name.replace("/", "_")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)


tokenizer.add_bos_token = True
# batch_size = 512
# batch_size = 32
batch_size = 128
max_length = 128
learning_rate = 1e-3
if(model_name == "gpt2"):
    target_layer = 'transformer.h.5'
    d_name = None
else: 
    target_layer = "model.layers.18"
    d_name = "cosmopedia-v2"
# dataset_name = "prithivMLmods/OpenWeb888K"
# def redo_data(num_datapoints=None, batch_size=128, max_length=128, dataset_specific_name=None):
#     data_generator = prepare_streaming_dataset(
#         tokenizer=tokenizer,
#         dataset_name=dataset_name,
#         max_length=max_length,
#         batch_size=batch_size,
#         num_datapoints=num_datapoints,  # Optional: limit number of datapoints
#         name = dataset_specific_name,
#     )
#     return data_generator

debug = False
if(debug):
    if(model_name == "gpt2"):
        dataset_name = "Elriggs/openwebtext-100k"
    else: 
        dataset_name = "HuggingFaceTB/smollm-corpus"
    # num_datapoints = 100_000
    num_datapoints = 1_000
    total_batches = num_datapoints // batch_size
    print(f"total amount of tokens in dataset: {num_datapoints * max_length / 1e6}M")
else:    
    if(model_name == "gpt2"):
        dataset_name = "prithivMLmods/OpenWeb888K"
        num_datapoints = None # 880_000
        total_batches = 888_000 // batch_size
    else: 
        dataset_name = "HuggingFaceTB/smollm-corpus"
        num_datapoints = 2_000_000
        total_batches = num_datapoints // batch_size
        print(f"total amount of tokens in dataset: {num_datapoints * max_length / 1e6}M")

# data_generator = redo_data(num_datapoints=num_datapoints, batch_size=batch_size, dataset_specific_name=d_name)

data_generator = TokenizedDataset(dataset_name, tokenizer, d_name, batch_size=batch_size, max_length=max_length, total_batches=total_batches)
# %%
import wandb
from dataclasses import dataclass
from typing import List, Optional
from einops import repeat
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json

@dataclass
class BiasConfig:
    use_positional: bool = False
    use_embedding: bool = False
    use_shifted_embedding: bool = False
    subtract_biases: bool = True
import bitsandbytes as bnb
from torch.optim.lr_scheduler import ReduceLROnPlateau

class MLP(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_hidden, bias=False)
        self.linear2 = nn.Linear(d_hidden, d_model, bias=False)
        # Initialize weights to xavier normal
        torch.nn.init.xavier_normal_(self.linear1.weight)
        torch.nn.init.xavier_normal_(self.linear2.weight)
        # increase the scale of the encoder weights by 10x
        # self.linear1.weight.data *= 10

    def forward(self, x):
        # x = torch.relu(self.linear1(x))
        # Do GeLU
        x = F.gelu(self.linear1(x))
        # x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

def train_with_biases(
    model,
    sae,
    cfg,
    data_generator,
    # positional_bias,
    # shifted_embedding_bias,
    device,
    target_layer,
    learning_rate: float = 1e-4,
    total_batches: Optional[int] = None,
    log_every: int = 10,
    print_batch_every: int = 10,
    mlp = None, 
):
    dead_feature_count = torch.zeros(sae.decoder.weight.shape[1], device=device)
    # Initialize optimizers
    bias_params = []
    # if bias_config.use_positional:
    #     bias_params.extend(positional_bias.parameters())
    per_token_bias_name = ['per_token_bias.bias']
    per_token_params = list(filter(lambda kv: kv[0] in per_token_bias_name, sae.named_parameters()))
    base_params = list(filter(lambda kv: kv[0] not in per_token_bias_name, sae.named_parameters()))
    lr_params = [{'params': base_params}]
    if cfg.use_embedding:
        # bias_params.extend(embedding_bias.parameters())
        lr_params.append({'params': per_token_params, 'lr': learning_rate*10})
    # if bias_config.use_shifted_embedding:
    #     bias_params.extend(shifted_per_token_params)
    if(cfg.optimizer == "Adam"):
        # So SAE will use LR & embedding bias will use 10x LR
        opt = torch.optim.Adam(lr_params, lr=learning_rate)
    elif(cfg.optimizer == "Adam8bit"):
        opt = bnb.optim.Adam8bit(lr_params, lr=learning_rate)
    elif(cfg.optimizer == "Adafactor"):
        opt = torch.optim.Adafactor([{'params': base_params}, {'params': per_token_params}])
    # batches_patience = 50
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=batches_patience, min_lr=1e-6) if bias_params else None
    # scheduler_sae = ReduceLROnPlateau(optimizer_sae, mode='min', factor=0.5, patience=batches_patience, min_lr=1e-6)
        # optimizer = torch.optim.Adafactor(bias_params, lr=learning_rate*10) if bias_params else None
        # optimizer_sae = torch.optim.Adafactor(sae.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adafactor(bias_params, lr=learning_rate*10) if bias_params else None
    # optimizer_sae = torch.optim.Adafactor(sae.parameters(), lr=learning_rate)
    # all_schedulers = [sched for sched in [scheduler, scheduler_sae] if sched is not None]
    # Define after your optimizers
    
    metrics_history = []
    if total_batches is None:
        total_batches = 10_000
        print("tqdm total batches not provided, using made-up value of 10k")

    total_tokens_processed = 0
    for batch_idx in tqdm(range(data_generator.total_batches)):
    # for batch_idx, batch in enumerate(tqdm(data_generator, total=total_batches)):
        input_ids = data_generator.next().to(device)
        
        # Get original outputs
        with torch.no_grad():
            with Trace(model, target_layer) as original_trace:
                _ = model(input_ids).logits
                x = original_trace.output[0] if isinstance(original_trace.output, tuple) else original_trace.output
            del original_trace
            torch.cuda.empty_cache()  # Call occasionally, not every batch

            normalize_input = True
            if normalize_input:
                # Normalize the input
                norm_of_activation = x.norm(dim=-1).mean().item()
                x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-8)

        # Calculate all biases
        all_biases = torch.zeros_like(x)
        # if bias_config.use_positional:
        #     positions = repeat(torch.arange(max_length), 'l -> b l', b=input_ids.shape[0]).to(device)
        #     all_biases = all_biases + positional_bias(positions)
        
        if sae.per_token_bias:
            all_biases = all_biases + sae.per_token_bias(input_ids)
            
            
        # if bias_config.use_shifted_embedding:
        #     # Shift input_ids right by 1, pad with zeros
        #     shifted_ids = torch.nn.functional.pad(input_ids[:, 1:], (1, 0), value=0)
        #     shifted_bias = shifted_embedding_bias(shifted_ids)
        #     all_biases = all_biases + shifted_bias

        comb_seq_length = cfg.tokens_to_combine if cfg.shift_window else 1
        # all_fvus = []
        # #TODO: don't for loop, but just shift by mod of batch_idx
        # for comb_seq in range(comb_seq_length):
        comb_seq = batch_idx % comb_seq_length if cfg.shift_window else 0

        if(comb_seq > 0):
            with torch.no_grad():
                end_slice = -1 * (cfg.tokens_to_combine - comb_seq)
                x_int = x[:, comb_seq:end_slice]
                all_biases_int = all_biases[:, comb_seq:end_slice]
                
        else:
            x_int = x
            all_biases_int = all_biases
        opt.zero_grad()

        if(cfg.norm_decoder):
            sae.set_decoder_norm_to_unit_norm()
        
        x_rearranged = einops.rearrange(
            x_int,
            'b (new_seq tokens_to_combine) d -> b new_seq (tokens_to_combine d)',
            tokens_to_combine=cfg.tokens_to_combine,
        )
        all_biases_rearranged = einops.rearrange(
            all_biases_int,
            'b (new_seq tokens_to_combine) d -> b new_seq (tokens_to_combine d)',
            tokens_to_combine=cfg.tokens_to_combine,
        )
        # Apply SAE with or without bias subtraction
        if cfg.subtract_bias:
            if(cfg.rand_mlp):
                x_hat = sae(mlp(x_rearranged - all_biases_rearranged)) + all_biases_rearranged
            else:
                x_hat = sae(x_rearranged - all_biases_rearranged) + all_biases_rearranged
        else:
            x_hat = sae(x_rearranged) + all_biases_rearranged

        # Forward with modified activations
        # def modify_activation(act):
        #     if(cfg.tokens_to_combine > 1):
        #         # Undo the concatenation
        #         x_hat = einops.rearrange(
        #             x_hat,
        #             'b new_seq (tokens_to_combine d) -> b (new_seq tokens_to_combine) d',
        #             tokens_to_combine=cfg.tokens_to_combine,
        #         )
        #     if isinstance(act, tuple):
        #         return (x_hat, act[1])
        #     return x_hat
        
        # Calculate losses
        mse_loss = (x_hat - x_rearranged).pow(2).mean()
        loss = mse_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=sae.parameters(), max_norm=1.0)
        if(cfg.use_embedding):
            torch.nn.utils.clip_grad_norm_(parameters=bias_params, max_norm=1.0)


        if(cfg.norm_decoder):
            sae.remove_gradient_parallel_to_decoder_directions()

        opt.step()
        with torch.no_grad():
            total_tokens_processed += input_ids.numel()
            # fvu = calculate_fvu(x_rearranged.detach(), x_hat.detach())
            # all_fvus.append(fvu.item())
        
        if total_tokens_processed > 60_000_000:
            cfg.norm_decoder = False

        with torch.no_grad():            
            if batch_idx % log_every == 0:
                # with Trace(model, target_layer, edit_output=modify_activation) as modified_trace:
                #     modified_outputs = model(input_ids).logits
                fvu = calculate_fvu(x_rearranged.detach(), x_hat.detach())
                # fvu = torch.tensor(all_fvus).mean()
                # Log to wandb
                grad_norm = torch.norm(torch.stack([p.grad.norm() for p in sae.parameters() if p.grad is not None]), p=2).item()
                grad_norm_bias = torch.norm(torch.stack([p.grad.norm() for p in bias_params if p.grad is not None]), p=2).item() if bias_params else 0
                wandb.log({
                    'mse': mse_loss.item(),
                    'fvu': fvu.item(),
                    # 'norm_decoder': sae.decoder.weight.norm(dim=0).mean().item(),
                    "act_norm": norm_of_activation,
                    'grad_norm': grad_norm,
                    'grad_norm_bias': grad_norm_bias,
                },step =total_tokens_processed)
                # ce_loss = F.cross_entropy(modified_outputs[..., :-1, :].reshape(-1, modified_outputs.size(-1)),
                #                         input_ids[..., 1:].reshape(-1))
                # ce_loss_original = F.cross_entropy(original_outputs[..., :-1, :].reshape(-1, original_outputs.size(-1)),
                #                         input_ids[..., 1:].reshape(-1))
                # ce_diff = ce_loss_original - ce_loss
                metrics_history.append({
                    'mse': mse_loss.item(),
                    'fvu': fvu.item(),
                    # 'ce': ce_loss.item(),
                    # 'ce_diff': ce_diff.item(),
                })
                if batch_idx % print_batch_every == 0:
                    print(f"Batch {batch_idx} - MSE: {mse_loss.item():.4f}, "
                        #   f"FVU: {fvu.item():.4f}, CE: {ce_loss.item():.4f}, CE Diff: {ce_diff.item():.4f}")
                          f"FVU: {fvu.item():.4f}")
                    # print GPU usage by GB
                    gpu_usage = torch.cuda.memory_allocated(device) / (1024 ** 3)
                    print(f"GPU usage: {gpu_usage:.2f} GB")
    
    return metrics_history

# %%
# Example usage:
def run_all_configurations(
    model,
    data_generator,
    d_model,
    dict_scalar,
    k,
    device,
    target_layer,
    learning_rate=1e-3,
    total_batches=None,
    tokens_to_combine=1,
):

    # Initialize wandb at the beginning of your main function, before the config loop
    wandb_project_name = "sae-token-combine"  # Choose a meaningful project name
    # wandb.login(
    #     key="b3d846c89a6848d50969d23df6a4026b4980f95d"
    # )  # Replace with your actual API key

    

    # make a dot dict
    all_configs = []
    # just vary tokens_to_combine
    # for use_embedding in [True, False]:
    optimizer = "Adam8bit"
    for use_embedding in [True]:
        # for tokens_to_combine in [2, 4, 8, 16, 32, 64, 128]:
        # for tokens_to_combine in [16, 4, 2, 8, 32, 64]:
        for tokens_to_combine in [1]:
            # for optimizer in ["Adam8bit"]:
            cfg = DotDict({
                'd_model': d_model, 
                'dict_size': d_model * dict_scalar,
                'k': k,
                'tokens_to_combine': tokens_to_combine,
                "lr": learning_rate,
                "use_embedding": use_embedding,
                "subtract_bias": True,
                "norm_decoder": True,
                "max_length": max_length,
                "optimizer": optimizer,
                "batch_size": batch_size,
                "shift_window": False,
                "rand_mlp": False,
                "scale_num_features": False,

            })
            if(cfg.scale_num_features):
                cfg.k = int(cfg.k * cfg.tokens_to_combine)
            all_configs.append(cfg)


    all_results = {}
    for cfg in all_configs:
        model_save_name = f"sae_k={cfg.k}_tokBias={cfg.use_embedding}" 
        # model_save_name = f"rand_MLP_tokBias={cfg.use_embedding}" 
        # model_save_name = f"seqComb={cfg.tokens_to_combine}_tokBias={cfg.use_embedding}_shiftWindow={cfg.shift_window}BatchStyle_k={cfg.k}"
        # model_save_name = f"seqComb={cfg.tokens_to_combine}_tokBias={cfg.use_embedding}_shiftWindow={cfg.shift_window}BatchStyle_k={cfg.k}_undecode"
        wandb.init(
            project=wandb_project_name,
            name=model_save_name,
            config=dict(cfg),  # Log the configuration
            reinit=True  # Allow multiple runs in the same process
        )
 
        print("batch_size", batch_size)
        
        # Log the batch size
        wandb.config.update({"batch_size": batch_size})

        # data_generator = redo_data(num_datapoints=num_datapoints, batch_size=batch_size, max_length=max_length, dataset_specific_name=d_name)
        data_generator.reset_iterator()
        print(f"\nTraining with config: {cfg}")
        if(model_name == "gpt2"):
            embedding_bias = EmbeddingBias(model.transformer.wte).to(device)
        else:
            embedding_bias = EmbeddingBias(model.model.embed_tokens).to(device)
        sae = AutoEncoderTopK(activation_dim=d_model*cfg.tokens_to_combine, dict_size=d_model*dict_scalar, k=cfg.k).to(device)
        sae.per_token_bias = embedding_bias
        print(sae.encoder.weight.shape)
        print(embedding_bias.bias.shape)
        print('---------------------')
            
        if(cfg.rand_mlp):
            mlp = MLP(d_model=cfg.d_model, d_hidden=cfg.d_model*4).to(device)
        else:
            mlp = None
        # positional_bias = EmbeddingBias(model.transformer.wpe).to(device)
        # shifted_embedding_bias = EmbeddingBias(model.transformer.wte).to(device)
        
        results = train_with_biases(
            model=model,
            sae=sae,
            cfg = cfg,
            data_generator=data_generator,
            # positional_bias=positional_bias,
            # shifted_embedding_bias=shifted_embedding_bias,
            device=device,
            target_layer=target_layer,
            learning_rate=learning_rate,
            total_batches=total_batches,
            log_every=5,
            print_batch_every=50,
            mlp=mlp,
        )
                # Log final metrics
        wandb.log({"final_mse": results[-1]["mse"], "final_fvu": results[-1]["fvu"]})
        wandb.finish()
        
        all_results[model_save_name] = results
        import os
        save_dir = "models"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Save the model
        model_save_path = f"{model_save_name}.pt"
        cfg_save_path = f"{model_save_name}_cfg.json"
        token_bias_save_path = f"{model_save_name}_token_bias.pt" if cfg.use_embedding else None
        results_save_path = f"{model_save_name}_results.pt"
        torch.save(sae.state_dict(), os.path.join(save_dir, model_save_path))
        with open(os.path.join(save_dir, cfg_save_path), 'w') as f:
            json.dump(dict(cfg), f, indent=2)
        torch.save(results, os.path.join(save_dir, results_save_path))
        print(f"SAE saved to {model_save_path}")

        # upload sae to huggingface
        push_sae_to_huggingface(
            save_dir=save_dir,
            model_save_path=model_save_path,
            cfg_save_path=cfg_save_path,
            hf_repo_id=f"Elriggs/seq_concat_{m_name_save}_{target_layer}"  # Replace with your actual repo
        )

        # Save the rand_mlp weights
        if cfg.rand_mlp:
            mlp_save_path = f"{model_save_name}_rand_mlp.pt"
            torch.save(mlp.state_dict(), os.path.join(save_dir, mlp_save_path))
            print(f"MLP saved to {mlp_save_path}")
        

    return all_results


# %%
if(model_name == "gpt2"):
    d_model = 768
else:
    print(model.model.embed_tokens.weight.shape)
    d_model = model.model.embed_tokens.weight.shape[-1]
results = run_all_configurations(
    model=model,
    data_generator=data_generator,
    d_model=d_model,
    dict_scalar=16,
    k=30,
    device=device,
    target_layer=target_layer,
    total_batches=total_batches,
)