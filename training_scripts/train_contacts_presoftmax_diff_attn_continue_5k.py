import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torch.amp import autocast

import os
import pickle
import numpy as np
import re
import gc
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from typing import Optional, Dict


import wandb
import logging
from datetime import datetime
from tqdm.contrib.logging import logging_redirect_tqdm

import sys
from MSAPairformer.modules import MSAPairformerContactModel
from MSAPairformer.training_utils import init_msa_confind_dataloaders, init_trRosetta_contact_dataloaders, set_seed
from MSAPairformer.utils import evaluate_contact_prediction

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set seed
set_seed(42)

def load_model(pairwise_repr_layer_idx):
    # Load/initialize model
    model_kwargs = dict(
        dim_msa_input = 28,
        dim_pairwise = 256,
        dim_msa = 464,
        dim_logits = 26,
        msa_module_kwargs = dict(
            depth = 22,
            opm_kwargs = dict(
                dim_opm_hidden = 16,
                outer_product_flavor = "presoftmax_differential_attention",
                seq_attn = True,
                dim_qk = 128,
                chunk_size = None,
                return_seq_weights = True,
                return_attn_logits = False,
                lambda_init = None,
                eps = 1e-32,
            ),
            pwa_kwargs = dict(
                heads = 8,
                dim_head = 32,
                dropout = 0.1,
                dropout_type = "row",
            ),
            pairwise_block_kwargs = dict(
                tri_mult_dim_hidden = None,
                use_triangle_attn = False,
                use_triangle_updates = True
            ),
            return_after_layer_idx = pairwise_repr_layer_idx,
        ),
        relative_position_encoding_kwargs = dict(
            r_max = 32,
            s_max = 2,
        ),
        return_msa_repr = False,
        return_pairwise_repr = True,
        query_only = True
    )
    # Load query-biased attention model
    checkpoint_path = "/home/ubuntu/MSA_Pairformer/model_checkpoints/train_presoftmax_diff_attn_continue_5k/model_final.pt"
    model = MSAContactModel(
        pretrained_weights_path=checkpoint_path,
        pretrained_is_final=True,
        msa_model_kwargs=model_kwargs,
        compiled_checkpoint=True,
        logreg_contact_head=True,
        dim_contact_input=256,
    )
    model.to(device)
    return model

# Training set (same set as MSA Transformer)
training_set_file = "/home/ubuntu/data/trRosetta/MSA_Transformer_training_set.txt"
with open(training_set_file, "r") as f:
    training_set = f.readlines()
training_set = [os.path.join("/home/ubuntu/data/trRosetta/training_set/a3m", line.strip() + ".a3m") for line in training_set]
training_paired_paths_l = [(p, p.replace("a3m", "npz")) for p in training_set]
assert all([os.path.exists(p[0]) for p in training_paired_paths_l]) and all([os.path.exists(p[1]) for p in training_paired_paths_l])


# Select random 100 proteins for validation
n_val = 100
tr_full_set = glob("/home/ubuntu/data/trRosetta/training_set/a3m/*.a3m")
tr_full_set = [os.path.basename(p).split(".a3m")[0] for p in tr_full_set if os.path.basename(p) not in training_set]
validation_set = np.random.choice(tr_full_set, size=n_val, replace=False)
validation_set = [os.path.join("/home/ubuntu/data/trRosetta/training_set/npz", line.strip() + ".npz") for line in validation_set]
validation_paired_paths_l = [(p.replace("npz", "a3m"), p) for p in validation_set]
assert all([os.path.exists(p[0]) for p in validation_paired_paths_l]) and all([os.path.exists(p[1]) for p in validation_paired_paths_l])

# Initialize dataloaders
train_dataloader, val_dataloader = init_trRosetta_contact_dataloaders(
    training_paired_paths_l,
    validation_paired_paths_l,
    max_seq_length_train=512,
    max_seq_length_val_test=512,
    max_msa_depth_train=1024,
    max_msa_depth_val_test=1024,
    min_msa_depth_train=32,
    min_msa_depth_val_test=32,
    max_tokens_train=2**17,
    max_tokens_val_test=2**17,
    batch_size_train=1,
    batch_size_val_test=2,
    num_workers=16,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
    data_random_seed=42
)

def main():
    # Training hyperparameters
    total_steps = 5000
    weight_decay = 0.1
    lr = 0.01
    lr_warmup_steps = 50
    lr_min = lr * 0.1
    lr_decay_steps = total_steps - lr_warmup_steps
    adamw_kwargs = dict(
        betas = (0.9, 0.999),
        eps = 1e-8,
    )
    def lr_lambda(step):
        if step < lr_warmup_steps:
            return float(step) / float(max(1, lr_warmup_steps))
        # Linear decay
        progress = float(step - lr_warmup_steps) / float(max(1, total_steps - lr_warmup_steps))
        return max(0.0, 1.0 - progress) * (1.0 - lr_min/lr) + lr_min/lr
    # Loss and metrics
    criterion = torch.nn.BCELoss(reduction='mean')
    dist_type_l = ["local", "short", "medium", "long", "morethansix"]
    precision_categories_l = [f"{l}_AUC" for l in dist_type_l] + [f"{l}_P@L5" for l in dist_type_l] + [f"{l}_P@L2" for l in dist_type_l] + [f"{l}_P@L" for l in dist_type_l]

    nLayers = 22
    model_out_dir = "/home/ubuntu/MSA_Pairformer/model_checkpoints/train_contacts_presoftmax_diff_attn_continue_5k/"
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)
    log_every_n_steps = 100
    wandb_config = dict(
        lr = lr,
        weight_decay = weight_decay,
        adamw_kwargs = adamw_kwargs,
        max_seq_length_train = 512,
        max_msa_depth_train = 1024,
        min_msa_depth_train = 128,
        max_tokens_train = 2**17,
        max_seq_length_val_test = 512,
        max_msa_depth_val_test = 1024,
        min_msa_depth_val_test = 128,
        max_tokens_val_test = 2**17,
        lr_warmup_steps = lr_warmup_steps,
        lr_min = lr_min,
        total_steps = total_steps,
        data_random_seed = 42
    )

    # for layer_idx in range(nLayers):
    for layer_idx in range(15, nLayers):
        model = load_model(layer_idx)
        model.eval()
        # Compute pairwise representations
        pairwise_repr_d = {}
        true_contacts_d = {}
        pairwise_mask_d = {}
        for batch in tqdm(train_dataloader):
            if batch is None:
                continue
            with torch.no_grad():
                pairwise_repr = model.compute_pairwise_repr(
                    additional_molecule_feats = batch['additional_molecule_feats'].to(device),
                    msa = batch['msas_onehot'].float().to(device),
                    mask = batch['mask'].to(device),
                    msa_mask = batch['msa_mask'].to(device),
                    pairwise_mask = batch['pairwise_mask'].to(device),
                    full_mask = batch['full_mask'].to(device)
                )
            for i in range(len(batch['msa_file_path'])):
                base_id = os.path.basename(batch['msa_file_path'][i]).split('.')[0]
                pairwise_repr_d[base_id] = pairwise_repr[i].cpu()
                true_contacts_d[base_id] = batch['contact_map'][i]
                pairwise_mask_d[base_id] = batch['pairwise_mask'][i]
        # Freeze all but contact prediction head
        model.train()
        for param in model.parameters():
            param.requires_grad = False
        for param in model.contact_head.parameters():
            param.requires_grad = True
            
        # Split params into two groups: with and without weight decay
        weight_decay_params = []
        no_weight_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name:
                no_weight_decay_params.append(param)
            elif 'norm' in name:
                no_weight_decay_params.append(param)
            else:
                weight_decay_params.append(param)

        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            [{'params': no_weight_decay_params, 'weight_decay': 0.0}, {'params': weight_decay_params, 'weight_decay': weight_decay}],
            lr=lr,
            **adamw_kwargs
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Initialize wandb logger
        experiment_name = f"presoftmax_diff_attn_continue_5k_trRosetta_layer_{layer_idx}"
        wandb.init(
            project = "MSA-Pairformer",
            name = experiment_name,
            config = {**wandb_config, "layer_idx": layer_idx, "model_kwargs": model.msa_model_kwargs}
        )
        # Prepare data
        true_contacts_t = torch.stack(list(true_contacts_d.values())).to(device)
        pairwise_repr_t = torch.stack(list(pairwise_repr_d.values())).to(device)
        # Prepare contact mask (only consider contacts at least 6 residues apart)
        contact_mask = true_contacts_t != -1
        batch_size, seq_len, _ = true_contacts_t.shape
        indices = torch.arange(seq_len, device=device)
        seq_separation = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))
        seq_sep_mask = seq_separation >= 6
        seq_sep_mask = seq_sep_mask.unsqueeze(0).expand(batch_size, -1, -1)
        final_contact_mask = (contact_mask & seq_sep_mask).to(device)

        for curr_step in tqdm(range(total_steps)):
            contacts_t = model.compute_contacts_from_pairwise_repr(pairwise_repr_t)
            loss = criterion(contacts_t[final_contact_mask].float(), true_contacts_t[final_contact_mask].float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            # Log metrics
            if curr_step % log_every_n_steps == 0:
                log_d = {"loss": loss.item()}
                counts_d = {}
                for cat in precision_categories_l:
                    log_d[cat] = 0
                    counts_d[cat] = 0
                with torch.no_grad():
                    for j in range(contacts_t.shape[0]):
                        ub_idx = contact_mask[i].sum(dim=-1)[0].item()
                        ex_pred_contacts_t = contacts_t[j][:ub_idx, :ub_idx].detach().cpu()
                        ex_true_contacts_t = true_contacts_t[j][:ub_idx, :ub_idx].detach().cpu()
                        precision_d = evaluate_contact_prediction(ex_pred_contacts_t, ex_true_contacts_t)
                        for key, val in precision_d.items():
                            log_d[key] += val
                            counts_d[key] += 1
                for key in precision_categories_l:
                    log_d[key] /= counts_d[key]
                wandb.log(log_d)
        # Save model weights
        torch.save(
            model.contact_head.state_dict(),
            os.path.join(model_out_dir,
            f"presoftmax_diff_attn_continue_5k_trRosetta_layer_{layer_idx}.pt")
        )

        # Eval
        model.eval()
        del pairwise_repr_d, true_contacts_d, pairwise_mask_d
        gc.collect()
        torch.cuda.empty_cache()
        log_d = {}
        counts_d = {}
        for cat in precision_categories_l:
            log_d[f"val_{cat}"] = 0
            counts_d[f"val_{cat}"] = 0
        with torch.no_grad():
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                for batch in tqdm(val_dataloader):
                    if batch is None:
                        continue
                    res = model(
                        additional_molecule_feats = batch['additional_molecule_feats'].to(device),
                        msa = batch['msas_onehot'].bfloat16().to(device),
                        mask = batch['mask'].to(device),
                        msa_mask = batch['msa_mask'].to(device),
                        pairwise_mask = batch['pairwise_mask'].to(device),
                        full_mask = batch['full_mask'].to(device)
                    )
                true_contacts_t = batch['contact_map']
                predicted_contacts_t = res['predicted_contact_map'].float()
                for j in range(predicted_contacts_t.shape[0]):
                    ub_idx = contact_mask[j].sum(dim=-1)[0].item()
                    ex_pred_contacts_t = predicted_contacts_t[j][:ub_idx, :ub_idx].detach().cpu()
                    ex_true_contacts_t = true_contacts_t[j][:ub_idx, :ub_idx].detach().cpu()
                    precision_d = evaluate_contact_prediction(ex_pred_contacts_t, ex_true_contacts_t)
                    for key, val in precision_d.items():
                        log_d[f"val_{key}"] += np.sum(val)
                        counts_d[f"val_{key}"] += 1
        for cat in precision_categories_l:
            if counts_d[f"val_{cat}"] > 0:
                log_d[f"val_{cat}"] /= counts_d[f"val_{cat}"]
        wandb.log(log_d)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        wandb.finish()
if __name__ == "__main__":
    main()