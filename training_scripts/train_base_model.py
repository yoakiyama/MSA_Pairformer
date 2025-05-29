import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torch.nn import Module, Linear, LayerNorm
from torch.amp import autocast

import pandas as pd
import numpy as np
import os
import math
from einops import einsum
from copy import deepcopy
from tqdm import tqdm

import wandb
import logging
from datetime import datetime
from tqdm.contrib.logging import logging_redirect_tqdm

import sys
from MSAPairformer.modules import MSAPairformer
from MSAPairformer.training_utils import init_dataloaders, evaluate_prediction, LossTracker, set_seed, GradAccumStatTracker
from MSAPairformer.dataset import  aa2tok_d

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Training experiment name
experiment_name = "train_base_model"

def main():
    # Set seed
    set_seed(42)
    # Setup standard output logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/{experiment_name}_{timestamp}.log"
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(levelname)s - %(message)s",
        handlers = [
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting training...")

    # Initialize data
    num_workers = 24
    train_prop = 0.8
    val_size = 100
    max_seq_length = 312
    max_msa_depth = 256
    min_msa_depth = 8
    max_tokens = 98304
    batch_size = 1
    val_batch_size = 16
    test_batch_size = 16
    max_val_seq_length = 512
    max_test_seq_length = 512
    pin_memory = True
    query_only = False
    random_query = False
    min_query_coverage = 1.0
    n_append_random = 0
    data_random_seed = 42
    mutate_pssm = True
    valid_a3m_list_path = "/home/ubuntu/data/valid_msas.crop_length_256.max_depth_256.min_depth_8.lambdalabs.txt"
    with open(valid_a3m_list_path, "r") as oFile:
        msa_file_paths = oFile.readlines()
    msa_file_paths = [p.strip() for p in msa_file_paths]
    msa_dataloader_train, msa_dataloader_val, msa_dataloader_test = init_dataloaders(
        msa_file_paths=msa_file_paths, max_seq_length=max_seq_length, max_val_seq_length=max_val_seq_length, max_test_seq_length=max_test_seq_length, max_msa_depth=max_msa_depth, min_msa_depth=min_msa_depth,
        max_tokens=max_tokens, batch_size=batch_size, val_batch_size=val_batch_size, test_batch_size=test_batch_size,
        num_workers=num_workers, train_prop=train_prop, val_size=val_size, pin_memory=pin_memory, query_only=query_only, random_query=random_query, min_query_coverage=min_query_coverage,
        n_append_random=n_append_random, data_random_seed=data_random_seed, mutate_pssm=mutate_pssm
    )

    # Initialize model
    dim_msa_input = len(aa2tok_d)
    msa_module_kwargs = dict(
        depth = 22,
        dim_msa = 464,
        outer_product_mean_dim_hidden = 16,
        msa_pwa_dropout_row_prob = 0.1,
        msa_pwa_heads = 8,
        msa_pwa_dim_head = 32,
        opm_chunk_size = 48,
        use_triangle_attn = False,
        use_triangle_updates = True,
        gated_opm = False,
        gated_single_op = False,
        drop_last_msa_update = False,
        seq_attn = False,
        differential_opm = False
    )
    pairwise_block_kwargs = dict(
        dropout_row_prob = 0.1,
        dropout_col_prob = 0.1
    )
    dim_pairwise = 256
    dim_logits = 26
    model = MSAPairformer(
        dim_msa_input = dim_msa_input,
        dim_logits = dim_logits,
        msa_module_kwargs = msa_module_kwargs,
        dim_pairwise = dim_pairwise,
        return_msa_repr = False,
        return_pairwise_repr = False
    )
    model = model.to(device).to(torch.bfloat16)
    model = torch.compile(model, dynamic=True)

    # Optimizer and LR scheduler
    total_steps = 50000
    weight_decay = 0.1
    lr = 8e-4
    lr_warmup_steps = 1000
    lr_min = lr * 0.1
    lr_decay_steps = total_steps - lr_warmup_steps
    adamw_kwargs = dict(
        betas = (0.95, 0.99),
        eps = 1e-8,
        weight_decay = weight_decay
    )
    # Split params into two groups: with and without weight decay
    weight_decay_params = []
    no_weight_decay_params = []
    for name, param in model.named_parameters():
        if 'bias' in name:
            no_weight_decay_params.append(param)
        elif 'norm' in name:
            no_weight_decay_params.append(param)
        elif name == "relative_position_encoding.out_embedder.weight":
            no_weight_decay_params.append(param)
        else:
            weight_decay_params.append(param)
    optimizer = torch.optim.AdamW(
        [{'params': no_weight_decay_params, 'weight_decay': 0.0}, {'params': weight_decay_params, 'weight_decay': weight_decay}],
        lr=lr,
        **adamw_kwargs
    )
    def lr_lambda(step):
        if step < lr_warmup_steps:
            return float(step) / float(max(1, lr_warmup_steps))
        # Linear decay
        progress = float(step - lr_warmup_steps) / float(max(1, total_steps - lr_warmup_steps))
        return max(0.0, 1.0 - progress) * (1.0 - lr_min/lr) + lr_min/lr
        # # No decay for now
        # return 1.0
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # Gradient clipping
    max_grad_norm = 1

    # Loss
    criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)

    # Other training parameters
    # Gradient accumulation
    grad_accum_every = 12
    effective_batch_size = grad_accum_every * batch_size
    # Number of gradient steps before validation
    validate_every = 64

    # Initialize wandb
    wandb.init(
        project = "MSA-Pairformer",
        name = experiment_name,
        config = dict(
            lr = lr,
            adamw_kwargs = adamw_kwargs,
            max_seq_length = max_seq_length,
            max_msa_depth = max_msa_depth,
            min_msa_depth = min_msa_depth,
            effective_batch_size = effective_batch_size,
            gradient_accumulation_every = grad_accum_every,
            **msa_module_kwargs,
            dim_pairwise = dim_pairwise,
            dim_logits = dim_logits,
            validate_every = validate_every,
            query_only = query_only,
            total_steps = total_steps,
            warmup_steps = lr_warmup_steps,
            min_lr = lr_min,
            max_grad_norm = max_grad_norm,
            mutate_pssm = mutate_pssm
        )
    )
    log_every_n_steps = 32
    checkpoint_every = 1000
    checkpoint_dir = f"/home/ubuntu/AF_inv_covariance/model_checkpoints/{experiment_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize loss tracker for logging exponential moving averages
    training_loss_categories_l = ['mlm_loss', 'accuracy', 'perplexity']
    training_loss_categories_l += ["val_" + category for category in training_loss_categories_l]
    loss_tracker = LossTracker(categories_l = training_loss_categories_l)

    # Track total number of training steps and current accumulation step
    track_stats_l = ["mlm_loss", "accuracy", "perplexity"]
    train_stat_tracker = GradAccumStatTracker(stats_l = track_stats_l)
    val_stat_tracker = GradAccumStatTracker(stats_l = [f"val_{stat}" for stat in track_stats_l])
    test_stat_tracker = GradAccumStatTracker(stats_l = [f"test_{stat}" for stat in track_stats_l])
    step = 0
    with logging_redirect_tqdm():
        while step < total_steps:
            for batch in tqdm(msa_dataloader_train, position = 0, leave = True):
                # Zero gradient and set model to training mode
                if train_stat_tracker.nSteps == 0:
                    optimizer.zero_grad()
                    model.train()
                # Skip if MSA depth is too low (will appear as batch is None)
                if batch is None:
                    continue
                # Move tensor inputs to device (skip unmasked_msas_onehot)
                for k, v in batch.items():
                    if k == "unmasked_msas_onehot":
                        continue
                    if isinstance(v, torch.Tensor):
                        if k in ["msas_onehot", "additional_molecule_feats"]:
                            batch[k] = v.bfloat16().to(device)
                        else:
                            batch[k] = v.to(device)
                # Forward pass
                with autocast(dtype=torch.bfloat16, device_type = device):
                    result = model(
                        msa = batch['msas_onehot'],
                        mask = batch['mask'],
                        msa_mask = batch['msa_mask'],
                        full_mask = batch['full_mask'],
                        pairwise_mask = batch['pairwise_mask'],
                        additional_molecule_feats = batch['additional_molecule_feats']
                    )
                    logits = result['logits']
                    # Compute loss
                    loss_d = evaluate_prediction(logits=logits, batch=batch, device=device, criterion=criterion)
                # Backpropagate (scale by number of gradient accumulation steps)
                total_loss = loss_d["mlm_loss"] / grad_accum_every
                total_loss.backward()
                # Update stats
                for metric_name, metric in loss_d.items():
                    if isinstance(metric, torch.Tensor):
                        metric = metric.detach().cpu()
                    train_stat_tracker.update_single(metric_name, metric)
                train_stat_tracker.increment_step()
                # Update model weights if gradient accumulation is complete
                if train_stat_tracker.nSteps == grad_accum_every:
                    grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    update_stat_d = train_stat_tracker.get_all()
                    loss_tracker.update_multiple(update_stat_d)
                    # Log stats
                    if step % log_every_n_steps == 0:
                        log_d = {**{k: v for k, v in loss_tracker.get_all().items() if 'val_' not in k}, 
                                "learning_rate": lr_scheduler.get_last_lr()[0], "step": step,
                                "grad_norm": grad_norm}
                        wandb.log(log_d)
                    # Reset stats and accumulation step, increment gradient step
                    train_stat_tracker.reset()
                    step += 1
                # Validation step
                if (step % validate_every == 0) and (train_stat_tracker.nSteps == 0):
                    # Set model to evaluation mode
                    model.eval()
                    for batch in tqdm(msa_dataloader_val, position = 0, leave = True):
                        # Skip if MSA depth is too low (will appear as batch is None)
                        if batch is None:
                            # print("Skipped batch due to insufficient MSA depth")
                            continue
                        # Move tensor inputs to device
                        for k, v in batch.items():
                            if k == "unmasked_msas_onehot":
                                continue
                            if isinstance(v, torch.Tensor):
                                if k in ["msas_onehot", "additional_molecule_feats"]:
                                    batch[k] = v.bfloat16().to(device)
                                else:
                                    batch[k] = v.to(device)
                        # Validation forward pass
                        with torch.no_grad():
                            with autocast(dtype=torch.bfloat16, device_type = device):
                                result = model(msa = batch['msas_onehot'], mask = batch['mask'], msa_mask = batch['msa_mask'], full_mask = batch['full_mask'], 
                                               pairwise_mask = batch['pairwise_mask'], additional_molecule_feats = batch['additional_molecule_feats'])
                                logits = result['logits']
                                # Compute loss
                                loss_d = evaluate_prediction(logits=logits, batch=batch, device=device, criterion=criterion)
                        # Update stats
                        for metric_name, metric in loss_d.items():
                            if isinstance(metric, torch.Tensor):
                                metric = metric.detach().cpu()
                            val_stat_tracker.update_single(f"val_{metric_name}", metric)
                        val_stat_tracker.increment_step()
                    # Update moving average
                    val_update_stat_d = val_stat_tracker.get_all()
                    loss_tracker.update_multiple(val_update_stat_d)
                    # Log validation loss, accuracy, and perplexity
                    log_d = {**{k: v for k, v in loss_tracker.get_all().items() if 'val_' in k}}
                    wandb.log(log_d)
                    val_stat_tracker.reset()
                # Save checkpoint
                if (step % checkpoint_every == 0) and (train_stat_tracker.nSteps == 0):
                    checkpoint_path = os.path.join(checkpoint_dir, f"model_step_{step}.pt")
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'step': step,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                        'wandb_id': wandb.run.id
                    }
                    torch.save(checkpoint, checkpoint_path)
                if step > total_steps:
                    break

        # Testing loop
        logger.info("Starting evaluation...")
        with logging_redirect_tqdm():
            # set_model_mode(model, training=False)
            model.eval()
            with torch.no_grad():
                for batch in tqdm(msa_dataloader_test, position = 0, leave = True):
                    # Skip if MSA depth is too low (will appear as batch is None)
                    if batch is None:
                        # print("Skipped batch due to insufficient MSA depth")
                        continue
                    # Move tensor inputs to device (skip unmasked_msas_onehot)
                    for k, v in batch.items():
                        if k == "unmasked_msas_onehot":
                            continue
                        if isinstance(v, torch.Tensor):
                            if k in ["msas_onehot", "additional_molecule_feats"]:
                                batch[k] = v.bfloat16().to(device)
                            else:
                                batch[k] = v.to(device)
                    # Forward pass
                    with autocast(dtype=torch.bfloat16, device_type = device):
                        result = model(msa = batch['msas_onehot'], mask = batch['mask'], msa_mask = batch['msa_mask'], full_mask = batch['full_mask'], 
                                    pairwise_mask = batch['pairwise_mask'], additional_molecule_feats = batch['additional_molecule_feats'])
                        logits = result['logits']
                        # Compute loss
                        loss_d = evaluate_prediction(logits=logits, batch=batch, device=device, criterion=criterion)
                    # Update stats
                    for metric_name, metric in loss_d.items():
                        if isinstance(metric, torch.Tensor):
                            metric = metric.detach().cpu()
                        test_stat_tracker.update_single(f"test_{metric_name}", metric)
                    test_stat_tracker.increment_step()
                # Log test loss, accuracy, and perplexity
                test_update_stat_d = test_stat_tracker.get_all()
                wandb.log(test_update_stat_d)
        wandb.finish()
        # Save model
        model_path = os.path.join(checkpoint_dir, "model_final.pt")
        torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    main()
