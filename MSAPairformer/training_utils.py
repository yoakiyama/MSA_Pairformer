import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, random_split, Dataset
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

import sys
from MSAPairformer.dataset import MSADataset, CollateBatch, MSA, aa2tok_d, tok2aa_d, nTokenTypes, trRosettaContactMSADataset, CollatetrRosettaContactMSABatch

def AF3LRScheduler(steps):
    # Linearly increases from 0 for the first 1000 steps
    if steps < 1000:
        return steps / 1000
    # Decrease by a factor of 0.95 every 5e4 steps
    steps -= 1000
    return 0.95 ** (steps / 5e4)

def InverseSquareRootLRScheduler(steps):
    # Increase from 0 over the first 1K steps then decrease by a factor of 0.95 every 5*10^4 steps
    return 0

# Dataloader cycler/iteratoor
def cycle(dataloader: DataLoader):
    while True:
        for batch in dataloader:
            yield batch

# Compute accuracy
def compute_accuracy(logits, tgts):
    return (logits.argmax(dim=1) == tgts).sum().item() / tgts.shape[0]

def get_msa_column_flat_idx(msa_t, flat_idx):
    i, j, k = torch.unravel_index(torch.tensor(flat_idx), msa_t.shape)
    return msa_t[i, :, k]

def get_msa_pssm(msa_t, flat_idx):
    # Get MSA column
    msa_col = get_msa_column_flat_idx(msa_t, flat_idx).cpu()
    # Compute PSSM
    unique_vals_t, counts_t = msa_col.unique(return_counts=True)
    pssm_s = pd.Series(counts_t, index=unique_vals_t.numpy())
    pssm_s = pssm_s / pssm_s.sum()
    pssm_s = pssm_s.reindex(range(0, nTokenTypes-2)).fillna(0)
    return pssm_s

def get_predicted_posterior(pred_logits, flat_idx):
    post_t = torch.nn.functional.softmax(pred_logits[flat_idx].detach(), dim=0).cpu()
    post_s = pd.Series(post_t, index=range(0, nTokenTypes-2))
    return post_s

def prep_posterior_pssm_plot(msa_t, flat_idx, pred_logits):
    # Get true AA label
    true_aa = msa_t.view(-1)[flat_idx].detach().cpu().item()
    # MSA PSSM
    pssm_s = get_msa_pssm(msa_t, flat_idx)
    # Predicted posterior
    post_s = get_predicted_posterior(pred_logits, flat_idx)
    return (true_aa, pssm_s, post_s)

def plot_posterior_pssm(pssm_s, post_s, true_aa, figsize=(5,3)):
    f, ax = plt.subplots(1, 1, figsize=figsize)
    ax.bar(x=post_s.index, height=post_s, alpha=0.5, label="Predicted probabilities")
    ax.bar(x=pssm_s.index, height=pssm_s, alpha=0.5, label="MSA PSSM")
    ax.set_xticks(range(0, len(pssm_s)))
    ax.set_xticklabels(post_s.index.map(tok2aa_d))
    ax.get_xticklabels()[true_aa].set_color('red')
    ax.legend()
    return f, ax

def plot_posterior_pssm_best_worst(pssm_d, post_d, true_aa_d, steps, figsize=(10,3)):
    f, ax = plt.subplots(1, 2, figsize=figsize)
    # Max loss
    max_pssm_s, max_post_s, max_true_aa = pssm_d['max'], post_d['max'], true_aa_d['max']
    ax[0].bar(x=max_post_s.index, height=max_post_s, alpha=0.5, label="Predicted probabilities")
    ax[0].bar(x=max_pssm_s.index, height=max_pssm_s, alpha=0.5, label="MSA PSSM")
    ax[0].set_xticks(range(0, len(max_pssm_s)))
    ax[0].set_xticklabels(max_post_s.index.map(tok2aa_d))
    ax[0].get_xticklabels()[max_true_aa].set_color('red')
    ax[0].set_title(f"Max loss predicted posteriors vs. MSA PSSM\n(step: {steps})", size=12)
    # Min loss
    min_pssm_s, min_post_s, min_true_aa = pssm_d['min'], post_d['min'], true_aa_d['min']
    ax[1].bar(x=min_post_s.index, height=min_post_s, alpha=0.5, label="Predicted probabilities")
    ax[1].bar(x=min_pssm_s.index, height=min_pssm_s, alpha=0.5, label="MSA PSSM")
    ax[1].set_xticks(range(0, len(min_pssm_s)))
    ax[1].set_xticklabels(min_post_s.index.map(tok2aa_d))
    ax[1].get_xticklabels()[min_true_aa].set_color('red')
    ax[1].legend()
    ax[1].set_title(f"Min loss predicted posteriors vs. MSA PSSM\n(step: {steps})", size=12)
    return f, ax

def init_dataloaders(
    msa_file_paths,
    max_seq_length,
    max_msa_depth,
    min_msa_depth,
    max_tokens,
    max_seq_length_val_test,
    max_msa_depth_val_test,
    max_tokens_val_test,
    min_msa_depth_val_test,
    batch_size,
    batch_size_val_test,
    num_workers,
    train_prop = 0.8,
    val_size = 100,
    pin_memory = True,
    query_only = False,
    random_query = False,
    min_query_coverage = 0.8,
    n_append_random = 0,
    n_append_random_val_test = 0,
    shifted_random = False,
    data_random_seed = 42,
    noising_rate = 0.15,
    mask_ratio = 0.8,
    mutate_ratio = 0.1,
    keep_ratio = 0.1,
    mutate_pssm = False,
    persistent_workers=True,
    prefetch_factor=2,
    scramble_seq_perc_train = None,
    scramble_col_perc_train = None,
    scramble_seq_perc_val_test = None,
    scramble_col_perc_val_test = None,
    random_depth = False
):
    # Set random seed
    torch.manual_seed(data_random_seed)
    np.random.seed(data_random_seed)
    # Randomly subset sequences for train, val, and test
    nTrain = int(train_prop * len(msa_file_paths))
    nVal = val_size
    nTest = len(msa_file_paths) - nTrain - nVal
    remaining_indices = np.arange(len(msa_file_paths))
    train_seq_indices = np.random.choice(remaining_indices, size=nTrain, replace=False)
    remaining_indices = np.setdiff1d(remaining_indices, train_seq_indices)
    val_seq_indices = np.random.choice(remaining_indices, size=nVal, replace=False)
    remaining_indices = np.setdiff1d(remaining_indices, val_seq_indices)
    test_seq_indices = np.random.choice(remaining_indices, size=nTest, replace=False)
    # Initialize datasets
    train_msa_dataset = MSADataset(
        msa_dir = None,
        msa_paths = msa_file_paths[train_seq_indices],
        max_seq_length = max_seq_length,
        max_msa_depth = max_msa_depth,
        max_tokens = max_tokens,
        random_query = random_query,
        min_query_coverage = min_query_coverage,
        n_append_random = n_append_random,
        shifted_random = shifted_random,
        scramble_seq_perc = scramble_seq_perc_train,
        scramble_col_perc = scramble_col_perc_train
    )
    val_msa_dataset = MSADataset(
        msa_dir = None,
        msa_paths = msa_file_paths[val_seq_indices],
        max_seq_length = max_seq_length_val_test,
        max_msa_depth = max_msa_depth_val_test,
        max_tokens = max_tokens_val_test,
        random_query = random_query,
        min_query_coverage = min_query_coverage,
        n_append_random = n_append_random_val_test,
        shifted_random = shifted_random,
        scramble_seq_perc = scramble_seq_perc_val_test,
        scramble_col_perc = scramble_col_perc_val_test
    )
    test_msa_dataset = MSADataset(
        msa_dir = None,
        msa_paths = msa_file_paths[test_seq_indices],
        max_seq_length = max_seq_length_val_test,
        max_msa_depth = max_msa_depth_val_test,
        max_tokens = max_tokens_val_test,
        random_query = random_query,
        min_query_coverage = min_query_coverage,
        n_append_random = n_append_random_val_test,
        shifted_random = shifted_random,
        scramble_seq_perc = scramble_seq_perc_val_test,
        scramble_col_perc = scramble_col_perc_val_test
    )

    # # Adjust maximum sequence lengths for validation and test datasets
    collate_fn_train = CollateBatch(
        max_seq_depth = max_msa_depth + n_append_random,
        max_seq_length = max_seq_length,
        min_seq_depth = min_msa_depth,
        query_only = query_only,
        mask_prob = noising_rate,
        mask_ratio = mask_ratio,
        mutate_ratio = mutate_ratio,
        keep_ratio = keep_ratio,
        mutate_pssm = mutate_pssm,
        n_append_random = n_append_random
    )
    collate_fn_val_test = CollateBatch(
        max_seq_depth = max_msa_depth_val_test + n_append_random_val_test,
        max_seq_length = max_seq_length_val_test,
        min_seq_depth = min_msa_depth_val_test,
        query_only = query_only,
        mask_prob = noising_rate,
        mask_ratio = mask_ratio,
        mutate_ratio = mutate_ratio,
        keep_ratio = keep_ratio,
        mutate_pssm = mutate_pssm,
        n_append_random = n_append_random_val_test
    )
    
    # Initialize dataloaders
    msa_dataloader_train = torch.utils.data.DataLoader(
        train_msa_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn_train, pin_memory=pin_memory, 
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )
    msa_dataloader_val = torch.utils.data.DataLoader(
        val_msa_dataset, batch_size=batch_size_val_test, shuffle=False, num_workers=num_workers, collate_fn=collate_fn_val_test, pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )
    msa_dataloader_test = torch.utils.data.DataLoader(
      test_msa_dataset, batch_size=batch_size_val_test, shuffle=False, num_workers=num_workers, collate_fn=collate_fn_val_test, pin_memory=pin_memory,
      persistent_workers=persistent_workers,
      prefetch_factor=prefetch_factor
    )
    return msa_dataloader_train, msa_dataloader_val, msa_dataloader_test

def init_trRosetta_contact_dataloaders(
    train_paired_paths_l,
    val_paired_paths_l,
    max_seq_length_train,
    max_seq_length_val_test,
    max_msa_depth_train,
    max_msa_depth_val_test,
    min_msa_depth_train,
    min_msa_depth_val_test,
    max_tokens_train,
    max_tokens_val_test,
    batch_size_train,
    batch_size_val_test,
    num_workers,
    pin_memory = True,
    prefetch_factor = 2,
    persistent_workers = True,
    data_random_seed = 42
):
    # Set random seed
    torch.manual_seed(data_random_seed)
    np.random.seed(data_random_seed)
    random.seed(data_random_seed)
    # Initialize datasets
    train_dataset = trRosettaContactMSADataset(
        paired_paths_l = train_paired_paths_l,
        max_seq_length = max_seq_length_train,
        max_msa_depth = max_msa_depth_train,
        min_msa_depth = min_msa_depth_train,
        max_tokens = max_tokens_train,
        random_query = False,
        n_append_random = 0,
        shifted_random = False
    )
    val_dataset = trRosettaContactMSADataset(
        paired_paths_l = val_paired_paths_l,
        max_seq_length = max_seq_length_val_test,
        max_msa_depth = max_msa_depth_val_test,
        min_msa_depth = min_msa_depth_val_test,
        max_tokens = max_tokens_val_test,
        random_query = False,
        n_append_random = 0,
        shifted_random = False
    )

    # Adjust maximum sequence lengths for validation and test datasets
    collate_fn_train = CollatetrRosettaContactMSABatch(
        max_seq_depth = max_msa_depth_train,
        max_seq_length = max_seq_length_train,
        min_seq_depth = min_msa_depth_train,
        pad_tok = aa2tok_d['PAD']
    )
    collate_fn_val = CollatetrRosettaContactMSABatch(
        max_seq_depth = max_msa_depth_val_test,
        max_seq_length = max_seq_length_val_test,
        min_seq_depth = min_msa_depth_val_test,
        pad_tok = aa2tok_d['PAD']
    )

    # Initialize dataloaders
    dataloader_train = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=num_workers, collate_fn=collate_fn_train, pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )
    dataloader_val = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size_val_test, shuffle=False, num_workers=num_workers, collate_fn=collate_fn_val, pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )
    return dataloader_train, dataloader_val


def evaluate_prediction(logits, batch, device, criterion, query_only = False, loss_weights = None, mean_reduction = False):
    dim_logits = logits.shape[-1]
    pred_logits = logits.view(-1, dim_logits)[batch['masked_idx']]
    if query_only:
        mlm_tgt = batch['msas'][:, 0, :].flatten()[batch['masked_idx']]
    else:
        mlm_tgt = batch['msas'].view(-1)[batch['masked_idx']]
    mlm_loss = criterion(pred_logits.float(), mlm_tgt)
    if loss_weights is not None:
        assert loss_weights.shape == mlm_loss.shape
        mlm_loss *= loss_weights
        if mean_reduction:
            mlm_loss = mlm_loss.mean()
    perplexity = torch.exp(mlm_loss)
    accuracy = (pred_logits.argmax(dim = -1) == mlm_tgt).float().mean()
    loss_d = {"mlm_loss": mlm_loss, "accuracy": accuracy, "perplexity": perplexity}
    return loss_d

def evaluate_prediction_no_reduction(logits, batch, device, criterion, query_only = False):
    dim_logits = logits.shape[-1]
    pred_logits = logits.view(-1, dim_logits)[batch['masked_idx']]
    if query_only:
        mlm_tgt = batch['msas'][:, 0, :].flatten()[batch['masked_idx']]
    else:
        mlm_tgt = batch['msas'].view(-1)[batch['masked_idx']]
    mlm_loss = criterion(pred_logits.float(), mlm_tgt)
    accuracy = (pred_logits.argmax(dim = -1) == mlm_tgt).float().sum()
    loss_d = {"mlm_loss": mlm_loss, "accuracy": accuracy}
    return loss_d

def evaluate_prediction_adversarial_attention(logits, adversarial_pssm_attention_weights, adversarial_random_attention_weights, batch, device, criterion, adversarial_pssm_scale = 1, adversarial_random_scale = 1):
    dim_logits = logits.shape[-1]
    pred_logits = logits.view(-1, dim_logits)[batch['masked_idx']]
    mlm_tgt = batch['msas'].view(-1)[batch['masked_idx']].to(device)
    if torch.isnan(pred_logits).any():
        print(f"NaN in pred_logits. Range: {pred_logits.min().item()} to {pred_logits.max().item()}")
        print(f"Actual labels - min: {mlm_tgt.min().item():.4f}, max: {mlm_tgt.max().item():.4f}")
        print(batch['file_path'])
    mlm_loss = criterion(pred_logits.float(), mlm_tgt.to(device))
    perplexity = torch.exp(mlm_loss)
    accuracy = (pred_logits.argmax(dim = -1) == mlm_tgt).float().mean()
    # Adversarial attention loss
    mean_adversarial_pssm_attention_weights = adversarial_pssm_attention_weights.mean()
    mean_adversarial_random_attention_weights = adversarial_random_attention_weights.mean()
    adversarial_pssm_loss = mean_adversarial_pssm_attention_weights * adversarial_pssm_scale
    adversarial_random_loss = mean_adversarial_random_attention_weights * adversarial_random_scale
    total_loss = mlm_loss + adversarial_pssm_loss + adversarial_random_loss
    loss_d = {"loss": total_loss, 
              "mlm_loss": mlm_loss.item(),
              "accuracy": accuracy.item(), 
              "perplexity": perplexity.item(),
              "adversarial_pssm_loss": adversarial_pssm_loss.item(),
              "adversarial_random_loss": adversarial_random_loss.item(),
              "adversarial_attention_loss": adversarial_pssm_loss.item() + adversarial_random_loss.item(),
              "mean_adversarial_attention_pssm": mean_adversarial_pssm_attention_weights.item(),
              "mean_adversarial_attention_random": mean_adversarial_random_attention_weights.item()
              }
    return loss_d

def evaluate_prediction_l1l2_sparsify(logits, attention_weights, batch, device, criterion, nSeqs, 
                                      sparsity_sigmoid = True, sparsity_sigmoid_scale = 1.0, sparsity_sigmoid_offset = 0.5, 
                                      sparsity_loss_scale = 1.0):
    # Evaluate MLM loss
    dim_logits = logits.shape[-1]
    pred_logits = logits.view(-1, dim_logits)[batch['masked_idx']]
    mlm_tgt = batch['msas'].view(-1)[batch['masked_idx']].to(device)
    if torch.isnan(pred_logits).any():
        print(f"NaN in pred_logits. Range: {pred_logits.min().item()} to {pred_logits.max().item()}")
        print(f"Actual labels - min: {mlm_tgt.min().item():.4f}, max: {mlm_tgt.max().item():.4f}")
        print(batch['file_path'])
    mlm_loss = criterion(pred_logits.float(), mlm_tgt.to(device))
    perplexity = torch.exp(mlm_loss)
    accuracy = (pred_logits.argmax(dim = -1) == mlm_tgt).float().mean()
    # Sparsification loss of attention weights
    mean_ratio_l1_l2 = compute_average_l1_l2_ratio(attention_weights, nSeqs)
    if sparsity_sigmoid:
        sparsity_loss = torch.sigmoid(sparsity_sigmoid_scale * (mean_ratio_l1_l2 - sparsity_sigmoid_offset)) * sparsity_loss_scale
    # sparsity_loss = mean_ratio_l1_l2 * sparsity_scaler
    total_loss = mlm_loss + sparsity_loss
    loss_d = {"loss": total_loss, 
              "mlm_loss": mlm_loss.item(),
              "sparsity_loss": sparsity_loss.item(),
              "accuracy": accuracy.item(), 
              "perplexity": perplexity.item(),
              "mean_ratio_l1_l2": mean_ratio_l1_l2.item(),
              "sparsity_loss": sparsity_loss.item()
              }
    return loss_d

def compute_average_l1_l2_ratio(attention_weights, nSeqs):
    max_ratios = nSeqs.sqrt().unsqueeze(-1)
    ratio_l1_l2 = (1 / attention_weights.norm(p=2, dim=-1)) / max_ratios
    return ratio_l1_l2.mean()

# def evaluate_prediction_adversarial_attention(logits, adversarial_attention_weights, batch, device, criterion, adversarial_scale = 1):
#     dim_logits = logits.shape[-1]
#     pred_logits = logits.view(-1, dim_logits)[batch['masked_idx']]
#     mlm_tgt = batch['msas'].view(-1)[batch['masked_idx']].to(device)
#     mlm_loss = criterion(pred_logits.float(), mlm_tgt.to(device))
#     perplexity = torch.exp(mlm_loss)
#     accuracy = (pred_logits.argmax(dim = -1) == mlm_tgt).float().mean()
#     adversarial_attention_loss = adversarial_attention_weights.sum() * adversarial_scale
#     total_loss = mlm_loss + adversarial_attention_loss
#     loss_d = {"total_loss": total_loss, 
#               "mlm_loss": mlm_loss,
#               "accuracy": accuracy, 
#               "perplexity": perplexity,
#               "adversarial_attention_loss": adversarial_attention_loss
#               }
#     return loss_d

# Tracking moving averages
class LossTracker:
    def __init__(self, categories_l, ema_alpha = 0.05):
        self.categories_l = categories_l
        self.ema_alpha = ema_alpha
        self.ema_d = {}
        for category in self.categories_l:
            self.ema_d[category] = None

    def update_single(self, category_name, value):
        if category_name not in self.categories_l:
            raise ValueError(f"Category {category_name} not found in categories list")
        value = float(value)
        if self.ema_d[category_name] is None:
            self.ema_d[category_name] = value
        else:
            self.ema_d[category_name] = self.ema_alpha * value + (1 - self.ema_alpha) * self.ema_d[category_name]

    def get_single(self, category_name):
        if category_name not in self.categories_l:
            raise ValueError(f"Category {category_name} not found in categories list")
        if self.ema_d[category_name] is None:
            raise ValueError(f"No values recorded yet for category {category_name}")
        return self.ema_d[category_name]

    def update_multiple(self, loss_d):
        for category_name, value in loss_d.items():
            if category_name not in self.categories_l:
                continue  # Skip unknown categories silently
            self.update_single(category_name, value)

    def get_all(self):
        # Filter out None values
        return {k: v for k, v in self.ema_d.items() if v is not None}

    def reset(self):
        for category_name in self.categories_l:
            self.ema_d[category_name] = None

# Tracks average values over gradient accumulation steps
class GradAccumStatTracker:
    def __init__(self, stats_l):
        self.stats_l = stats_l
        self.nSteps = 0
        self.stat_d = {}
        for stat in self.stats_l:
            self.stat_d[stat] = 0

    def increment_step(self):
        self.nSteps += 1

    def update_single(self, stat_name, value):
        value = float(value)
        self.stat_d[stat_name] += value

    def add_category(self, stat_name):
        self.stats_l.append(stat_name)
        self.stat_d[stat_name] = 0

    def get_single(self, stat_name):
        return self.stat_d[stat_name] / self.nSteps

    def update_multiple(self, update_stat_d):
        for stat_name, value in update_stat_d.items():
            self.update_single(stat_name, value)

    def get_all(self):
        return {k: v / self.nSteps for k, v in self.stat_d.items()}

    def reset(self):
        for stat_name in self.stats_l:
            self.stat_d[stat_name] = 0
        self.nSteps = 0


class GradAccumLossTracker:
    def __init__(self):
        self.total_loss = 0
        self.total_tokens = 0
        self.total_correct = 0
        self.nSteps = 0
    
    def update_total_loss(self, loss):
        self.total_loss += loss
    
    def update_total_tokens(self, tokens):
        self.total_tokens += tokens

    def update_total_correct(self, correct):
        self.total_correct += correct
    
    def get_final_loss(self):
        return self.total_loss / self.total_tokens
    
    def get_final_accuracy(self):
        return self.total_correct / self.total_tokens

    def get_final_perplexity(self):
        with torch.no_grad():
            return torch.exp(self.get_final_loss())

    def increment_step(self):
        self.nSteps += 1

    def reset(self):
        self.total_loss = 0
        self.total_tokens = 0
        self.total_correct = 0
        self.nSteps = 0

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
