# Adapted from https://github.com/Bitbol-Lab/DiffPALM/blob/main/diffpalm/gumbel_sinkhorn_utils.py
# Original author: Umberto Lupo et al. (2024), Pairing interacting protein sequences using masked language modeling
# Modified from: https://github.com/perrying/gumbel-sinkhorn

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from typing import Tuple


def sample_uniform(log_alpha_size: torch.Size):
    return torch.rand(log_alpha_size)


def sinkhorn_norm(alpha: torch.Tensor, n_iter: int = 20) -> Tuple[torch.Tensor,]:
    for _ in range(n_iter):
        alpha = alpha / alpha.sum(-1, keepdim=True)
        alpha = alpha / alpha.sum(-2, keepdim=True)
    return alpha


def log_sinkhorn_norm(
    log_alpha: torch.Tensor, n_iter: int = 20
) -> Tuple[torch.Tensor,]:
    for _ in range(n_iter):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -2, keepdim=True)
    return log_alpha.exp()


def gumbel_sinkhorn(
    log_alpha: torch.Tensor,
    noise_mat: torch.Tensor = 0,
    tau: float = 1.0,
    n_iter: int = 20,
    noise: bool = True,
    noise_factor: float = 1.0,
    noise_std: bool = False,
    rand_perm=None,
    cost_bias=None,
) -> Tuple[torch.Tensor,]:
    if noise:
        if noise_std:
            noise_factor = noise_factor * torch.std(log_alpha)
        gumbel_noise = -torch.log(-torch.log(noise_mat + 1e-20) + 1e-20)
        log_alpha = log_alpha + gumbel_noise * noise_factor
    log_alpha = log_alpha / tau
    sampled_perm_mat = log_sinkhorn_norm(log_alpha, n_iter)
    return sampled_perm_mat


def gen_assignment(cost_matrix, cost_bias=None):
    if cost_bias is not None:
        cost_matrix = cost_matrix + cost_bias
    row, col = linear_sum_assignment(cost_matrix, maximize=True)
    np_assignment_matrix = np.zeros_like(cost_matrix)
    np_assignment_matrix[row, col] = 1
    return np_assignment_matrix


def gumbel_matching(
    log_alpha: torch.Tensor,
    noise_mat: torch.Tensor = 0,
    noise: bool = True,
    noise_factor: float = 1.0,
    noise_std: bool = False,
    rand_perm=None,
    cost_bias=None,
) -> Tuple[torch.Tensor,]:
    if noise:
        if noise_std:
            noise_factor = noise_factor * torch.std(log_alpha)
        gumbel_noise = -torch.log(-torch.log(noise_mat + 1e-20) + 1e-20)
        log_alpha = log_alpha + gumbel_noise * noise_factor
    if rand_perm is not None:
        log_alpha = rand_perm[0] @ log_alpha @ rand_perm[1].T
    np_log_alpha = log_alpha.detach().to("cpu").numpy()
    np_assignment_mat = gen_assignment(np_log_alpha, cost_bias)
    assignment_mat = torch.from_numpy(np_assignment_mat).float().to(log_alpha.device)
    if rand_perm is not None:
        assignment_mat = rand_perm[0].T @ assignment_mat @ rand_perm[1]
    return assignment_mat


def no_noise_matching(
    log_alpha: torch.Tensor
) -> Tuple[torch.Tensor,]:
    np_log_alpha = log_alpha.detach().to("cpu").numpy()
    np_assignment_mat = gen_assignment(np_log_alpha)
    assignment_mat = torch.from_numpy(np_assignment_mat).float().to(log_alpha.device)
    return assignment_mat


def MSA_inverse_permutation(X, permutation_matrix):
    return torch.einsum("pq,bprs->bqrs", (permutation_matrix, X))


def MSA_inverse_permutation_batch(X, permutation_matrices):
    return torch.einsum("bpq,prs->bqrs", (permutation_matrices, X))


def inverse_permutation(X, permutation_matrix):
    return torch.einsum("pq,pr->qr", (permutation_matrix, X))
