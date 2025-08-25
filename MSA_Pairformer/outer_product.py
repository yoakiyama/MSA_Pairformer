import torch
from functools import partial
from einops import rearrange
from torch import nn
from torch.nn import Module, LayerNorm, Linear
from typing import Literal, Optional
from .core import to_pairwise_mask, LinearNoBias, SwiGLU, exists, RMSNorm
from .custom_typing import (
    Float,
    Bool
)
import sys
from .chunk_layer import chunk_layer

class OuterProductMean(Module):
    def __init__(
        self,
        dim_hidden = 32,
        dim_msa = 64,
        dim_pairwise = 128,
        eps = 1e-18,
        seq_attn: bool = False,
        dim_qk: int = 64,
        return_seq_weights: bool = False,
        chunk_size: int | None = None,
    ):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.dim_msa = dim_msa
        self.dim_pairwise = dim_pairwise
        self.eps = eps
        self.chunk_size = chunk_size
        self.norm = LayerNorm(dim_msa)
        self.to_left_hidden = LinearNoBias(dim_msa, dim_hidden)
        self.to_right_hidden = LinearNoBias(dim_msa, dim_hidden)
        self.to_pairwise_repr = Linear(dim_hidden ** 2, dim_pairwise * 2)
        self.activation = SwiGLU()

        # Unsupervised sequence weight learning
        self.seq_attn = seq_attn
        if self.seq_attn:
            self.dim_qk = dim_qk
            self.scaling = self.dim_qk ** -0.5
            self.q_proj = LinearNoBias(dim_msa, dim_qk)
            self.k_proj = LinearNoBias(dim_msa, dim_qk)
            k_weights = self.q_proj.weight.clone() + torch.randn_like(self.q_proj.weight) * 0.1
            self.k_proj.weight = nn.Parameter(k_weights)
            self.q_norm = LayerNorm(dim_qk)
            self.k_norm = LayerNorm(dim_qk)
        self.return_seq_weights = return_seq_weights
    def _opm(self, a, b):
        outer = torch.einsum("...bac,...dae->...bdce", a, b)
        outer = outer.reshape(outer.shape[:-2] + (-1,))
        outer = self.to_pairwise_repr(outer)
        outer = self.activation(outer)
        return outer

    @torch.jit.ignore
    def _chunk(
        self,
        a,
        b,
        chunk_size
    ):
        a_reshape = a.reshape((-1,) + a.shape[-3:])
        b_reshape = b.reshape((-1,) + b.shape[-3:])
        out = []
        for a_prime, b_prime in zip(a_reshape, b_reshape):
            outer = chunk_layer(
                partial(self._opm, b=b_prime),
                {"a": a_prime},
                chunk_size = chunk_size,
                no_batch_dims=1
            )
            out.append(outer)

        if(len(out) == 1):
            outer = out[0].unsqueeze(0)
        else:
            outer = torch.stack(out, dim=0)

        outer = outer.reshape(a.shape[:-3] + outer.shape[1:])
        return outer

    def forward(
        self,
        msa: Float['b s n d'],
        mask: Bool['b n'] | None = None,
        msa_mask: Bool['b s'] | None = None,
        full_mask: Bool['b s n'] | None = None,
        pairwise_mask: Bool['b n n'] | None = None,
        seq_weights: Float["b s"] | None = None
    ) -> Float['b n n dp']:
        # Default to full mask if not provided
        if not exists(full_mask):
            full_mask = msa.new_ones(msa.shape[:-1])

        # Normalize MSA representation
        norm_msa = self.norm(msa)

        # Default to uniform sequence weights
        if self.seq_attn:
            # Project to multi-headed queries, keys and values
            q = self.q_proj(norm_msa[:, 0])  # [b n d] - only first sequence (query)
            k = self.k_proj(norm_msa)        # [b s n d] - all sequences (keys)
            q = self.q_norm(q)
            k = self.k_norm(k)
            # Compute attention scores
            attn_logits = torch.einsum("...id,...sid->...si", q, k) * self.scaling  # [b s n]
            # Average over all positions for each sequence (Apply mask to seq weights before averaging)
            norm_factor = (full_mask.sum(dim=-1) + self.eps) # [b, n]
            attn_logits = attn_logits.masked_fill(~full_mask, 0)
            attn_logits = attn_logits.sum(dim=-1) / norm_factor # [b s]
            # Apply attention mask and compute softmax attention weights
            attn_logits = attn_logits.masked_fill(~msa_mask, -1e9)
            # Compute softmax attention weights
            seq_weights = attn_logits.softmax(dim=-1)
            del q, k
        else:
            if not exists(seq_weights):
                # Default to uniform sequence weights if not provided
                attn_logits = msa_mask.new_ones(msa_mask.shape)
                seq_weights = msa_mask.new_ones(msa_mask.shape) / msa_mask.sum(dim=-1).unsqueeze(-1)

        # Create left and right hidden representations and apply mask to padded positions
        expanded_full_mask = full_mask.unsqueeze(-1)
        a = self.to_left_hidden(norm_msa) * expanded_full_mask # [b s n c]
        b = self.to_right_hidden(norm_msa) * expanded_full_mask # [b s n c]
        del norm_msa, expanded_full_mask
        # Transpose for efficient matrix multiplication
        a = a.transpose(-2, -3) # [b n s c]
        b = b.transpose(-2, -3) # [b n s c]
        # Scale a and b by the square root of the sequence weights
        scaled_seq_weights = (seq_weights + self.eps).sqrt()
        a = torch.einsum("...s,...nsc->...nsc", scaled_seq_weights, a) # [b n s c]
        b = torch.einsum("...s,...nsc->...nsc", scaled_seq_weights, b) # [b n s c]
        if self.chunk_size is not None:
            outer = self._chunk(a, b, self.chunk_size)
        else:
            outer = self._opm(a, b)
        # Mask invalid pairwise positions
        if not exists(pairwise_mask):
            pairwise_mask = to_pairwise_mask(mask)
        outer = torch.einsum("... i j d, ... i j -> ... i j d", outer, pairwise_mask)
        if not self.return_seq_weights:
            del seq_weights
            seq_weights = None
        return outer, seq_weights

class PresoftmaxDifferentialOuterProductMean(Module):
    def __init__(
        self,
        dim_hidden = 32,
        dim_msa = 64,
        dim_pairwise = 128,
        eps = 1e-32,
        chunk_size: int | None = None,
        seq_attn: bool = False,
        dim_qk: int = 64,
        return_seq_weights: bool = False,
        lambda_init: float = 1.0
    ):
        super().__init__()
        # Store hyperparameters
        self.dim_hidden = dim_hidden
        self.dim_msa = dim_msa
        self.dim_pairwise = dim_pairwise
        self.eps = eps
        self.chunk_size = chunk_size
        # Initialize LayerNorm to normalize input MSA representation
        self.norm = LayerNorm(dim_msa)
        # Initialize linear layers to project MSA representation into hidden representations for outer product
        self.to_left_hidden = LinearNoBias(dim_msa, dim_hidden)
        self.to_right_hidden = LinearNoBias(dim_msa, dim_hidden)
        # Initialize linear layer to project outer product representation into the pairwise representation
        self.to_pairwise_repr = Linear(dim_hidden ** 2, dim_pairwise * 2)
        self.activation = SwiGLU()

        # Unsupervised sequence weight learning
        self.seq_attn = seq_attn
        if self.seq_attn:
            self.dim_qk = dim_qk
            self.scaling = self.dim_qk ** -0.5
            self.q_proj = LinearNoBias(dim_msa, dim_qk * 2)
            self.k_proj = LinearNoBias(dim_msa, dim_qk * 2)
            k_weights = self.q_proj.weight.clone() + torch.randn_like(self.q_proj.weight) * 0.1
            self.k_proj.weight = nn.Parameter(k_weights)
            self.q_norm = LayerNorm(dim_qk)
            self.k_norm = LayerNorm(dim_qk)
            self.lambda_init = lambda_init
            self.lambda_q1 = nn.Parameter(torch.zeros(self.dim_qk, dtype=torch.float32).normal_(mean=0,std=0.1))
            self.lambda_k1 = nn.Parameter(torch.zeros(self.dim_qk, dtype=torch.float32).normal_(mean=0,std=0.1))
            self.lambda_q2 = nn.Parameter(torch.zeros(self.dim_qk, dtype=torch.float32).normal_(mean=0,std=0.1))
            self.lambda_k2 = nn.Parameter(torch.zeros(self.dim_qk, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.return_seq_weights = return_seq_weights

    def _opm(self, a, b):
        outer = torch.einsum("...bac,...dae->...bdce", a, b)
        outer = outer.reshape(outer.shape[:-2] + (-1,))
        outer = self.to_pairwise_repr(outer)
        outer = self.activation(outer)
        return outer

    @torch.jit.ignore
    def _chunk(
        self,
        a,
        b,
        chunk_size
    ):
        a_reshape = a.reshape((-1,) + a.shape[-3:])
        b_reshape = b.reshape((-1,) + b.shape[-3:])
        out = []
        for a_prime, b_prime in zip(a_reshape, b_reshape):
            outer = chunk_layer(
                partial(self._opm, b=b_prime),
                {"a": a_prime},
                chunk_size = chunk_size,
                no_batch_dims=1
            )
            out.append(outer)

        if(len(out) == 1):
            outer = out[0].unsqueeze(0)
        else:
            outer = torch.stack(out, dim=0)

        outer = outer.reshape(a.shape[:-3] + outer.shape[1:])
        return outer

    def forward(
        self,
        msa: Float['b s n d'],
        mask: Bool['b n'] | None = None,
        msa_mask: Bool['b s'] | None = None,
        seq_weights: Float["b s"] | None = None,
        full_mask: Bool['b s n'] | None = None,
        pairwise_mask: Bool['b n n'] | None = None
    ) -> Float['b n n dp']:
        # Default to full mask if not provided
        if not exists(full_mask):
            full_mask = msa.new_ones(msa.shape[:-1]).to(bool).to(msa.device) # [b, s, n]
        if not exists(msa_mask):
            msa_mask = msa.new_ones(msa.shape[:-2]).to(bool).to(msa.device) # [b, s]
        if not exists(mask):
            mask = msa.new_ones((msa.shape[0],msa.shape[2])).to(bool).to(msa.device)
        # Normalize MSA representation
        norm_msa = self.norm(msa)
        # Unsupervised sequence weight learning
        if self.seq_attn:
            # Compute Q, K (Both q1/q2 and k1/k2 are computed from the same projection)
            q = self.q_proj(norm_msa[:, 0]) # [b n (2d)]
            k = self.k_proj(norm_msa)        # [b s n (2d)]
            q_type = q.dtype
            # Split last dimension in half to create q1/q2 and k1/k2
            q = rearrange(q, '... n (two d) -> ... two n d', two=2)
            k = rearrange(k, '... s n (two d) -> ... two s n d', two=2)
            # Normalize q and k
            q = self.q_norm(q)
            k = self.k_norm(k)
            # Compute attention scores
            seq_weights = torch.einsum("... t n d, ... t s n d-> ... t s n", q, k) * self.scaling  # [b 2 s n]
            # Compute lambda
            lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1)).to(q_type)
            lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1)).to(q_type)
            lambda_full = lambda_1 - lambda_2 + self.lambda_init
            # Average pooling
            norm_factor = (full_mask.sum(dim=-1) + self.eps).unsqueeze(1).expand(-1, 2, -1) # [b, 2, s]
            seq_weights = seq_weights.masked_fill(~full_mask.unsqueeze(1).expand(-1, 2, -1, -1), 0)
            seq_weights = seq_weights.sum(dim=-1) / norm_factor # [b 2 s]
            # Compute differential ``
            seq_weights = seq_weights[:, 0, :] - (lambda_full * seq_weights[:, 1, :]) # [b s]
            seq_weights = seq_weights.masked_fill(~msa_mask, -1e9)
            seq_weights = seq_weights.softmax(dim=-1)
            del lambda_1, lambda_2, lambda_full, q, k
        else:
            # Default to uniform sequence weights
            if not exists(seq_weights):
                seq_weights = msa_mask / msa_mask.sum(dim=-1, keepdim=True) # [b s]
        # Create left and right hidden representations and apply mask to padded positions
        expanded_full_mask = full_mask.unsqueeze(-1)
        a = self.to_left_hidden(norm_msa) * expanded_full_mask # [b s n c]
        b = self.to_right_hidden(norm_msa) * expanded_full_mask # [b s n c]
        del norm_msa, expanded_full_mask
        # Transpose for efficient matrix multiplication
        a = a.transpose(-2, -3) # [b n s c]
        b = b.transpose(-2, -3) # [b n s c]
        # Scale a and b by the square root of the sequence weights
        scaled_seq_weights = (seq_weights + self.eps).sqrt()
        a = torch.einsum("...s,...nsc->...nsc", scaled_seq_weights, a) # [b n s c]
        b = torch.einsum("...s,...nsc->...nsc", scaled_seq_weights, b) # [b n s c]
        if self.chunk_size is not None:
            outer = self._chunk(a, b, self.chunk_size)
        else:
            outer = self._opm(a, b)
        # Mask invalid pairwise positions
        if not exists(pairwise_mask):
            pairwise_mask = to_pairwise_mask(mask)
        outer = torch.einsum("... i j d, ... i j -> ... i j d", outer, pairwise_mask)
        if not self.return_seq_weights:
            del seq_weights
            seq_weights = None
        return outer, seq_weights


class OuterProduct(Module):
    def __init__(
        self,
        dim_msa: int,
        dim_pairwise: int,
        dim_opm_hidden: int,
        outer_product_flavor: Literal["vanilla", "vanilla_attention", "presoftmax_differential_attention"],
        seq_attn: bool = False,
        dim_qk: Optional[int] = None,
        chunk_size: Optional[int] = None,
        return_seq_weights: Optional[bool] = False,
        lambda_init: Optional[float] = None,
        eps: float = 1e-32,
    ):
        super().__init__()
        assert outer_product_flavor in ["vanilla", "vanilla_attention", "presoftmax_differential_attention"]
        if outer_product_flavor in ["vanilla", "vanilla_attention"]:
            print("Using vanilla attention / OPM")
            self.opm = OuterProductMean(
                dim_msa = dim_msa,
                dim_pairwise = dim_pairwise,
                dim_hidden = dim_opm_hidden,
                chunk_size = chunk_size,
                seq_attn = seq_attn,
                dim_qk = dim_qk,
                return_seq_weights = return_seq_weights,
            )
        elif outer_product_flavor == "presoftmax_differential_attention":
            assert (dim_qk is not None) and (lambda_init is not None), "dim_qk and lambda_init must be provided for presoftmax_differential_attention"
            self.opm = PresoftmaxDifferentialOuterProductMean(
                dim_hidden = dim_opm_hidden,
                dim_msa = dim_msa,
                dim_pairwise = dim_pairwise,
                chunk_size = chunk_size,
                seq_attn = seq_attn,
                dim_qk = dim_qk,
                return_seq_weights = return_seq_weights,
                lambda_init = lambda_init,
                eps = eps
            )
        else:
            raise NotImplementedError(f"OuterProduct flavor not implemented: {outer_product_flavor}")

    def forward(
        self,
        msa: Float['b s n d'],
        mask: Bool['b n'] | None = None,
        msa_mask: Bool['b s'] | None = None,
        seq_weights: Float["b s"] | None = None,
        full_mask: Bool['b s n'] | None = None,
        pairwise_mask: Bool['b n n'] | None = None
    ) -> Float['b n n dp']:
        return self.opm(
            msa = msa,
            mask = mask,
            msa_mask = msa_mask,
            seq_weights = seq_weights,
            full_mask = full_mask,
            pairwise_mask = pairwise_mask
        )
