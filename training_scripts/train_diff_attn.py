import torch
from torch.amp import autocast
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import os
import re
import wandb
import logging
from datetime import datetime
from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm import tqdm

import sys
from MSAPairformer.modules import MSAPairformer
from MSAPairformer.training_utils import init_dataloaders, evaluate_prediction, LossTracker, GradAccumStatTracker, set_seed
from MSAPairformer.dataset import  aa2tok_d
from einops import rearrange

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Training experiment name
experiment_name = "train_diff_attn"

def main():
    # Set seed
    random_seed = 42
    set_seed(random_seed)
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

    # Load pretrained model and set up for fine-tuning of auto-relevance detection
    model_kwargs = dict(
        dim_msa_input = 28,
        dim_pairwise = 256,
        dim_msa = 464,
        dim_logits = 26,
        msa_module_kwargs = dict(
            depth = 22,
            opm_kwargs = dict(
                dim_opm_hidden = 16,
                outer_product_flavor = "differential_attention",
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
                dropout = 0.0,
                dropout_type = "row",
            ),
            pairwise_block_kwargs = dict(
                tri_mult_dim_hidden = None,
                use_triangle_attn = False,
                use_triangle_updates = True
            )
        ),
        relative_position_encoding_kwargs = dict(
            r_max = 32,
            s_max = 2,
        ),
        return_msa_repr = False,
        return_pairwise_repr = False,
        query_only = True
    )
    model = MSAPairformer(**model_kwargs)
    checkpoint_path = "/home/ubuntu/MSA_Pairformer/model_checkpoints/train_base_model/model_final.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    checkpoint = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()} 
    checkpoint = {
        k if len(re.findall("msa_module.layers.\d+.2.", k)) == 0 else
        '.'.join(k.split('.')[:-2]) + '.opm.' + '.'.join(k.split('.')[-2:]): v
        for k, v in checkpoint.items()
    }
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    print("Missing keys: ", np.unique(['.'.join(k.split('.')[-2:]) for k in missing_keys]))
    print("Unexpected keys: ", unexpected_keys)
    # Load model to GPU and use bfloat16 with compile
    model = model.to(device).to(torch.bfloat16)
    model = torch.compile(model, dynamic=True)

    # Optimizer and LR scheduler
    total_steps = 12500
    weight_decay = 1e-3
    weight_decay_base = 0.1
    lr = 5e-5
    lr_min = lr * 1e-5
    lr_warmup_steps = 500
    lr_decay_steps = total_steps - lr_warmup_steps
    higher_lr_param_l = ["q_proj", "k_proj", "q_norm", "k_norm", "lambda_q1", "lambda_k1", "lambda_q2", "lambda_k2"]
    params = [
        {'params': [p for n, p in model.named_parameters() if any(param_name in n for param_name in higher_lr_param_l)], 'lr': lr*10, 'weight_decay': weight_decay},  # Higher learning rate for attention params
        {'params': [p for n, p in model.named_parameters() if not any(param_name in n for param_name in higher_lr_param_l)], 'lr': lr, 'weight_decay': weight_decay_base}  # Default learning rate for other params
    ]
    optimizer = torch.optim.AdamW(params)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[lr*10, lr],
        total_steps=total_steps,
        pct_start=lr_warmup_steps/total_steps,
        final_div_factor=1e5,
        anneal_strategy='cos',
    )
    logger.info("Scheduler initialized with linear warmup and cosine decay")

    # Gradient clipping
    max_grad_norm = 5

    # Retrieve valid MSA paths
    # valid_a3m_list_path = "/home/ubuntu/data/valid_msas.crop_length_256.max_depth_256.min_depth_8.lambdalabs.txt"
    a3m_list_path = "/home/ubuntu/data/all_uniclust30_msas.txt"
    with open(a3m_list_path, "r") as oFile:
        msa_file_paths = oFile.readlines()
    msa_file_paths = np.array([p.strip() for p in msa_file_paths])

    # Data parameters
    data_params = dict(
        msa_file_paths = msa_file_paths,
        max_seq_length = 320,
        max_seq_length_val_test = 512,
        max_msa_depth = 320,
        max_msa_depth_val_test = 512,
        min_msa_depth = 128,
        min_msa_depth_val_test = 256,
        max_tokens = 196608,
        max_tokens_val_test = 196608,
        batch_size = 1,
        batch_size_val_test = 8,
        num_workers = 18,
        train_prop = 0.8,
        val_size = 100,
        pin_memory = True,
        prefetch_factor = 2,
        query_only = True,
        random_query = True,
        min_query_coverage = 0.9,
        n_append_random = 0,
        n_append_random_val_test = 0,
        shifted_random = False,
        data_random_seed = 42,
        mutate_pssm = False,
        noising_rate = 0.15,
    )
    msa_dataloader_train, msa_dataloader_val, msa_dataloader_test = init_dataloaders(**data_params)

    # Gradient accumulation (samples not batches)
    grad_accum_every = 32
    effective_batch_size = grad_accum_every

    # Number of gradient steps before validation
    validate_every = 32

    # Total number of test batches
    test_batches = 100

    # Loss and adversarial loss scaling (with warmup and exponential decay)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)

    # Initialize wandb
    wandb.init(
        project = "MSA-Pairformer",
        name = experiment_name,
        config = dict(
            max_seq_length = data_params['max_seq_length'],
            max_msa_depth = data_params['max_msa_depth'],
            min_msa_depth = data_params['min_msa_depth'],
            effective_batch_size = effective_batch_size,
            gradient_accumulation_every = grad_accum_every,
            **model_kwargs,
            n_append_random_seqs = data_params['n_append_random'],
            n_append_random_seqs_val_test = data_params['n_append_random_val_test'],
            qk_norm = False,
            weight_decay = weight_decay,
            lr = lr,
            lr_warmup_steps = lr_warmup_steps,
            lr_min = lr_min,
            lr_decay_steps = lr_decay_steps,
            betas = (0.9, 0.999)
        )
    )
    log_every_n_steps = 4
    checkpoint_every = 1000
    checkpoint_dir = f"/home/ubuntu/MSA_Pairformer/model_checkpoints/{experiment_name}/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Initialize loss tracker for logging exponential moving averages
    training_loss_categories_l = ["mlm_loss", "accuracy", "perplexity", "avg_seq_weight", "min_seq_weight", "max_seq_weight"]
    training_loss_categories_l += ["val_" + category for category in training_loss_categories_l]
    loss_tracker = LossTracker(categories_l = training_loss_categories_l)

    # Track total number of training steps and current accumulation step
    val_track_stats_l = ["mlm_loss", "accuracy", "perplexity", "avg_seq_weight", "min_seq_weight", "max_seq_weight"]
    track_stats_l = ["mlm_loss", "accuracy", "perplexity"]
    train_stat_tracker = GradAccumStatTracker(stats_l = track_stats_l)
    val_stat_tracker = GradAccumStatTracker(stats_l = [f"val_{stat}" for stat in val_track_stats_l])
    test_stat_tracker = GradAccumStatTracker(stats_l = [f"test_{stat}" for stat in val_track_stats_l])
    step = 0
    print("STARTING TRAINING")
    with logging_redirect_tqdm():
        while step < total_steps:
            for batch in tqdm(msa_dataloader_train, position = 0, leave = True):
                # Zero gradient and set model to training mode
                if train_stat_tracker.nSteps == 0:
                    model.train()
                # Skip if MSA depth is too low (will appear as batch is None)
                if (batch is None) or (batch['msas_onehot'].shape[0] < data_params['batch_size']):
                    # print("Skipped batch due to insufficient MSA depth")
                    continue
                # Move tensor inputs to device (skip unmasked_msas_onehot)
                n_seqs_msa_1 = batch['msa_mask'].sum(dim=-1)[0]
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
                    result = model(msa = batch['msas_onehot'], msa_mask = batch['msa_mask'], mask = batch['mask'], full_mask = batch['full_mask'], 
                                pairwise_mask = batch['pairwise_mask'], additional_molecule_feats = batch['additional_molecule_feats'])
                    logits = result['logits']
                    # Compute loss
                    loss_d = evaluate_prediction(logits=logits, batch=batch, device=device, criterion=criterion, query_only=True)
                total_loss = loss_d['mlm_loss'] / grad_accum_every
                total_loss.backward()
                # Update stats
                for metric_name, metric in loss_d.items():
                    if isinstance(metric, torch.Tensor):
                        metric = metric.detach().cpu()
                    train_stat_tracker.update_single(metric_name, metric)
                train_stat_tracker.nSteps += batch['msas_onehot'].shape[0]
                # Update model weights if gradient accumulation is complete
                if train_stat_tracker.nSteps >= grad_accum_every:
                    grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    update_stat_d = train_stat_tracker.get_all()
                    loss_tracker.update_multiple(update_stat_d)
                    # Log stats
                    if step % log_every_n_steps == 0:
                        log_d = {**{k: v for k, v in loss_tracker.get_all().items() if 'val_' not in k}, 
                                "learning_rate": scheduler.get_last_lr()[0], "step": step,
                                "grad_norm": grad_norm}
                        # Summarize sequence weights
                        with torch.no_grad():
                            seq_weights = torch.stack([result['seq_weights'][layer] for layer in result['seq_weights']], dim=1) # [b layers s]
                            avg_seq_weights = seq_weights.mean(dim=1)
                            avg_seq_weights = avg_seq_weights[batch['msa_mask']]
                            log_d['avg_seq_weight'] = avg_seq_weights.mean().item()
                            log_d['min_seq_weight'] = avg_seq_weights.min().item()
                            log_d['max_seq_weight'] = avg_seq_weights.max().item()
                            del seq_weights, avg_seq_weights
                            for k in ['avg_seq_weight', 'min_seq_weight', 'max_seq_weight']:
                                loss_tracker.update_single(k, log_d[k])
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
                        # Move tensor inputs to device (skip unmasked_msas_onehot)
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
                                result = model(
                                    msa = batch['msas_onehot'], mask = batch['mask'], msa_mask = batch['msa_mask'], full_mask = batch['full_mask'], 
                                    pairwise_mask = batch['pairwise_mask'], additional_molecule_feats = batch['additional_molecule_feats']
                                )
                                logits = result['logits']
                                loss_d = evaluate_prediction(logits=logits, batch=batch, device=device, criterion=criterion, query_only=True)
                                with torch.no_grad():
                                    # Summarize sequence weights
                                    seq_weights = torch.stack([result['seq_weights'][layer] for layer in result['seq_weights']], dim=1) # [b layers s]
                                    avg_seq_weights = seq_weights.mean(dim=1)
                                    avg_seq_weights = avg_seq_weights[batch['msa_mask']]
                                    loss_d['avg_seq_weight'] = avg_seq_weights.mean()
                                    loss_d['min_seq_weight'] = avg_seq_weights.min()
                                    loss_d['max_seq_weight'] = avg_seq_weights.max()
                                    del seq_weights, avg_seq_weights
                        # Update stats
                        for metric_name, metric in loss_d.items():
                            if isinstance(metric, torch.Tensor):
                                metric = metric.detach().cpu()
                            val_stat_tracker.update_single(f"val_{metric_name}", metric)
                        val_stat_tracker.increment_step()
                    # # Update moving average
                    val_update_stat_d = val_stat_tracker.get_all()
                    loss_tracker.update_multiple(val_update_stat_d)
                    # Compute lambda (differential scaling) for each layer
                    lambda_d = {}
                    for idx, layer in enumerate(model.msa_module.layers):
                        lambda_1 = torch.exp(torch.sum(layer[2].opm.lambda_q1 * layer[2].opm.lambda_k1, dim=-1)).to(torch.float32)
                        lambda_2 = torch.exp(torch.sum(layer[2].opm.lambda_q2 * layer[2].opm.lambda_k2, dim=-1)).to(torch.float32)
                        lambda_full = lambda_1 - lambda_2 + layer[2].opm.lambda_init
                        lambda_d[f"layer_{idx}_lambda"] = lambda_full
                    # Log validation loss, accuracy, and perplexity
                    log_d = {**{k: v for k, v in loss_tracker.get_all().items() if 'val_' in k}, **lambda_d}
                    wandb.log(log_d)
                    val_stat_tracker.reset()
                if (step % checkpoint_every == 0) and (train_stat_tracker.nSteps == 0):
                    checkpoint_path = os.path.join(checkpoint_dir, f"model_step_{step}.pt")
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'step': step,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': scheduler.state_dict(),
                        'wandb_id': wandb.run.id
                    }
                    torch.save(checkpoint, checkpoint_path)
                if step >= total_steps:
                    break
    # Testing loop
    logger.info("Starting evaluation...")
    with logging_redirect_tqdm():
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
                    loss_d = evaluate_prediction(logits=logits, batch=batch, device=device, criterion=criterion, query_only=True)
                    with torch.no_grad():
                        # Summarize sequence weights
                        seq_weights = torch.stack([result['seq_weights'][layer] for layer in result['seq_weights']], dim=1) # [b layers s]
                        avg_seq_weights = seq_weights.mean(dim=1)
                        avg_seq_weights = avg_seq_weights[batch['msa_mask']]
                        loss_d['avg_seq_weight'] = avg_seq_weights.mean()
                        loss_d['min_seq_weight'] = avg_seq_weights.min()
                        loss_d['max_seq_weight'] = avg_seq_weights.max()
                        del seq_weights, avg_seq_weights
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