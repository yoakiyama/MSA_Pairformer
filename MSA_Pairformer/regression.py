import torch
import torch.nn.functional as F
from torch.nn import Module, LayerNorm, Linear, Parameter, Sigmoid, ModuleList
from einops import rearrange
from .core import SwiGLU
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