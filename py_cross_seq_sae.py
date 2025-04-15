# %%
import torch
from utils import *
# Example usage:
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "gpt2"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_bos_token = True
batch_size = 128
max_length = 128
learning_rate = 1e-3
target_layer = 'transformer.h.5'

dataset_name = "Elriggs/openwebtext-100k"
# dataset_name = "prithivMLmods/OpenWeb888K"
def redo_data(num_datapoints=None, batch_size=128):
    data_generator = prepare_streaming_dataset(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        max_length=max_length,
        batch_size=batch_size,
        num_datapoints=num_datapoints,  # Optional: limit number of datapoints
    )
    return data_generator

debug = True
if(debug):
    dataset_name = "Elriggs/openwebtext-100k"
    num_datapoints = 10_000
    total_batches = num_datapoints // batch_size
    print(f"total amount of tokens in dataset: {num_datapoints * 128}")
else:
    dataset_name = "prithivMLmods/OpenWeb888K"
    num_datapoints = None # 880_000
    total_batches = 888_000 // batch_size
    print(f"total amount of tokens in dataset: {880_000 * 128}")

data_generator = redo_data(num_datapoints=num_datapoints, batch_size=batch_size)


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

def train_with_biases(
    model,
    sae,
    cfg,
    data_generator,
    embedding_bias,
    # positional_bias,
    # shifted_embedding_bias,
    device,
    target_layer,
    learning_rate: float = 1e-4,
    total_batches: Optional[int] = None,
    log_every: int = 10,
    print_batch_every: int = 10,
):
    # Initialize optimizers
    bias_params = []
    # if bias_config.use_positional:
    #     bias_params.extend(positional_bias.parameters())
    if cfg.use_embedding:
        bias_params.extend(embedding_bias.parameters())
    # if bias_config.use_shifted_embedding:
    #     bias_params.extend(shifted_embedding_bias.parameters())
    
    optimizer = torch.optim.Adam(bias_params, lr=learning_rate*10) if bias_params else None
    optimizer_sae = torch.optim.Adam(sae.parameters(), lr=learning_rate)
    all_optimizers = [opt for opt in [optimizer, optimizer_sae] if opt is not None]
    
    metrics_history = []
    if total_batches is None:
        total_batches = 10_000
        print("tqdm total batches not provided, using made-up value of 10k")

    total_tokens_processed = 0
    for batch_idx, batch in enumerate(tqdm(data_generator, total=total_batches)):
        for opt in all_optimizers:
            opt.zero_grad()
        if(cfg.norm_decoder):
            sae.set_decoder_norm_to_unit_norm()
        
        input_ids = batch.to(device)
        
        # Get original outputs
        with torch.no_grad():
            with Trace(model, target_layer) as original_trace:
                _ = model(input_ids).logits
                x = original_trace.output[0] if isinstance(original_trace.output, tuple) else original_trace.output
        
        # concatenate x by every cfg.tokens_to_combine tokens

        # Calculate all biases
        all_biases = torch.zeros_like(x)
        # if bias_config.use_positional:
        #     positions = repeat(torch.arange(max_length), 'l -> b l', b=input_ids.shape[0]).to(device)
        #     all_biases = all_biases + positional_bias(positions)
        
        if cfg.use_embedding:
            all_biases = all_biases + embedding_bias(input_ids)
            
        # if bias_config.use_shifted_embedding:
        #     # Shift input_ids right by 1, pad with zeros
        #     shifted_ids = torch.nn.functional.pad(input_ids[:, 1:], (1, 0), value=0)
        #     shifted_bias = shifted_embedding_bias(shifted_ids)
        #     all_biases = all_biases + shifted_bias

        if cfg.tokens_to_combine > 1:
            x = einops.rearrange(
                x,
                'b (new_seq tokens_to_combine) d -> b new_seq (tokens_to_combine d)',
                tokens_to_combine=cfg.tokens_to_combine,
            )
            all_biases = einops.rearrange(
                all_biases,
                'b (new_seq tokens_to_combine) d -> b new_seq (tokens_to_combine d)',
                tokens_to_combine=cfg.tokens_to_combine,
            )
        # Apply SAE with or without bias subtraction
        if cfg.subtract_bias:
            x_hat = sae(x - all_biases) + all_biases
        else:
            x_hat = sae(x) + all_biases

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
        mse_loss = (x_hat - x).pow(2).mean()
        loss = mse_loss
        loss.backward()

        if(cfg.norm_decoder):
            sae.remove_gradient_parallel_to_decoder_directions()

        for opt in all_optimizers:
            opt.step()
            
        # Logging
        total_tokens_processed += input_ids.numel()

        with torch.no_grad():            
            if batch_idx % log_every == 0:
                # with Trace(model, target_layer, edit_output=modify_activation) as modified_trace:
                #     modified_outputs = model(input_ids).logits
                fvu = calculate_fvu(x, x_hat)
                # Log to wandb
                wandb.log({
                    'mse': mse_loss.item(),
                    'fvu': fvu.item(),
                    'batch': batch_idx,
                    
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

    
    # # Define all configurations to test
    # configs = [
    #     BiasConfig(),  # No biases
    #     # BiasConfig(use_positional=True),  # Positional only
    #     # BiasConfig(use_embedding=True),  # Embedding only
    #     # BiasConfig(use_shifted_embedding=True),  # Shifted embedding only
    #     # BiasConfig(use_positional=True, use_embedding=True),  # Positional + Embedding
    #     # BiasConfig(use_positional=True, use_shifted_embedding=True),  # Positional + Shifted
    #     # BiasConfig(use_embedding=True, use_shifted_embedding=True),  # Both embeddings
    #     # BiasConfig(use_positional=True, use_embedding=True, use_shifted_embedding=True),  # All biases
    # ]
    # make a dot dict
    all_configs = []
    # just vary tokens_to_combine
    for use_embedding in [True, False]:
    # for norm_decoder in [False]:
        for tokens_to_combine in [1, 2, 4, 8, 16, 32, 64]:
        # for tokens_to_combine in [128, 256, 512, 1024]:
            cfg = DotDict({
                'd_model': d_model, 
                'dict_size': d_model * dict_scalar,
                'k': k,
                'tokens_to_combine': tokens_to_combine,
                "lr": learning_rate,
                "use_embedding": True,
                "subtract_bias": True,
                "norm_decoder": False,
            })
            # cfg.tokens_to_combine = tokens_to_combine
            all_configs.append(cfg)


    all_results = {}
    for cfg in all_configs:
        model_save_name = f"seqComb={cfg.tokens_to_combine}_tokBias={cfg.use_embedding}_normDec={cfg.norm_decoder}"
        wandb.init(
            project=wandb_project_name,
            name=model_save_name,
            config=dict(cfg),  # Log the configuration
            reinit=True  # Allow multiple runs in the same process
        )
        standard_batch_size = 128
        if(cfg.tokens_to_combine ==16):
            batch_size = standard_batch_size//2
        elif(cfg.tokens_to_combine == 32):
            print("Using 1/4 batch size")
            batch_size = standard_batch_size//4
        elif(cfg.tokens_to_combine == 64):
            print("Using 1/8 batch size")
            batch_size = standard_batch_size//4
        elif(cfg.tokens_to_combine == 128):
            print("Using 1/8 batch size")
            batch_size = standard_batch_size//32
        elif(cfg.tokens_to_combine == 256):
            print("Using 1/8 batch size")
            batch_size = standard_batch_size//32
        elif(cfg.tokens_to_combine == 512):
            print("Using 1/8 batch size")
            batch_size = standard_batch_size//32
        elif(cfg.tokens_to_combine == 1024):
            print("Using 1/8 batch size")
            batch_size = standard_batch_size//64
            
        
        else:
            batch_size = standard_batch_size
            # batch_size = 128
        print("batch_size", batch_size)
        
        # Log the batch size
        wandb.config.update({"batch_size": batch_size})

        data_generator = redo_data(num_datapoints=num_datapoints, batch_size=batch_size)
        print(f"\nTraining with config: {cfg}")
        sae = AutoEncoderTopK(activation_dim=d_model*cfg.tokens_to_combine, dict_size=d_model*dict_scalar, k=k).to(device)
        embedding_bias = EmbeddingBias(model.transformer.wte).to(device)
        # positional_bias = EmbeddingBias(model.transformer.wpe).to(device)
        # shifted_embedding_bias = EmbeddingBias(model.transformer.wte).to(device)
        
        results = train_with_biases(
            model=model,
            sae=sae,
            cfg = cfg,
            data_generator=data_generator,
            embedding_bias=embedding_bias,
            # positional_bias=positional_bias,
            # shifted_embedding_bias=shifted_embedding_bias,
            device=device,
            target_layer=target_layer,
            learning_rate=learning_rate,
            total_batches=total_batches,
            log_every=5,
            print_batch_every=50,
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
            hf_repo_id=f"Elriggs/seq_concat_{model_name}_{target_layer}"  # Replace with your actual repo
        )
        

    return all_results


# %%
results = run_all_configurations(
    model=model,
    data_generator=data_generator,
    d_model=768,
    dict_scalar=8,
    k=30,
    device=device,
    target_layer=target_layer,
    total_batches=total_batches,
)