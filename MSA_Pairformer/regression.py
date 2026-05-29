import torch
import torch.nn.functional as F
from torch.nn import Module, LayerNorm, Linear, Parameter, Sigmoid, ModuleList, GELU, Sequential
from einops import rearrange
from .core import SwiGLU, RMSNorm
from .custom_typing import (
    Float,
    Bool,
    typecheck
)

##########################
# Language modeling head #
##########################
class LMHead(Module):
    def __init__(
        self,
        dim_msa,
        dim_output,
    ):
        super().__init__()
        self.init_ln = LayerNorm(dim_msa)
        self.dense = Linear(dim_msa, dim_msa*2)
        self.dense_activation = SwiGLU()
        self.pre_logit_norm = LayerNorm(dim_msa)
        self.bias = Parameter(torch.zeros(dim_output))
        self.weight = Linear(dim_msa, dim_output, bias=False).weight

    def forward(
        self,
        msa_repr: Float["b s n d"]
    ) -> Float["b s n *"]:
        x = self.init_ln(msa_repr)
        x = self.dense(x)
        x = self.dense_activation(x)
        x = self.pre_logit_norm(x)
        x = F.linear(x, self.weight) + self.bias
        return x

################
# Contact head #
################
class LogisticRegressionContactHead(Module):
    def __init__(
        self,
        dim_pairwise,
    ):
        super().__init__()
        self.init_ln = LayerNorm(dim_pairwise)
        self.bias = Parameter(torch.zeros(1))
        self.weight = Linear(dim_pairwise, 1, bias=False).weight
        self.sigmoid = Sigmoid()

    def forward(
        self,
        pair_repr: Float["b n n d"]
    ) -> Float["b s n *"]:
        x = self.init_ln(pair_repr)
        x = F.linear(x, self.weight) + self.bias
        x = self.sigmoid(x)
        # Symmetrize the output matrix
        x = x.squeeze(-1)
        x = (0.5 * (x + x.transpose(-1, -2)))
        return x

############
# MRF head #
############
"""
Takes in the pairwise representation (b n n dp) and projects to a (b n a n a) tensor.
Diagonal extracted as the b tensor (b n a) and the off-diagonal as the W tensor (b n a n a)
"""
class MRFHead(Module):
    def __init__(
        self,
        dim_pairwise,
        dim_alphabet
    ):
        super().__init__()
        self.dim_alphabet = dim_alphabet
        self.init_ln = RMSNorm(dim_pairwise)
        self.w_dense = Sequential(
            Linear(dim_pairwise, dim_pairwise, bias=False),
            GELU(),
            RMSNorm(dim_pairwise),
            Linear(dim_pairwise, dim_alphabet**2, bias=False)
        )
        self.b_dense = Sequential(
            Linear(dim_pairwise, dim_pairwise, bias=False),
            GELU(),
            RMSNorm(dim_pairwise),
            Linear(dim_pairwise, dim_alphabet, bias=True)
        )
        # Initialize weights and biases
        torch.nn.init.normal_(self.w_dense[-1].weight, mean=0.0, std=1e-3)
        torch.nn.init.normal_(self.b_dense[-1].weight, mean=0.0, std=1e-3)
        torch.nn.init.zeros_(self.b_dense[-1].bias)

    def forward(
        self,
        pairwise_repr: Float["b n n dp"],
        pairwise_mask: Bool["b n n"],
        mask: Bool["b n"]
    ):
        # Normalize pairwise representation
        w = self.init_ln(pairwise_repr)

        # Get diagonal elements of pairwise representation
        b = torch.diagonal(w, dim1=1, dim2=2).transpose(-1, -2) # (b n dp)
        b = self.b_dense(b)
        b = b * mask.unsqueeze(-1)
        
        # Project to W
        w = self.w_dense(w) # (b n n a**2)

        # Zero out diagonal of W, rearrange to (b n a n a), then symmetrize
        w = w * (~torch.eye(w.shape[1], device=w.device, dtype=bool, requires_grad=False)[None, :, :, None])
        w = rearrange(w, '... n1 n2 (d1 d2) -> ... n1 d1 n2 d2', d1=self.dim_alphabet, d2=self.dim_alphabet)
        w = (0.5 * (w + rearrange(w, "... n1 a1 n2 a2 -> ... n2 a2 n1 a1")))
        # Apply pairwise mask to w
        w = w * pairwise_mask.unsqueeze(2).unsqueeze(-1)
        
        return w, b