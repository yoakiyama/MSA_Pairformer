import torch
import torch.nn.functional as F
from torch.nn import Module, Linear, Parameter, Sigmoid, ModuleList, RMSNorm
from einops import rearrange
from MSAPairformer.core import SwiGLU
from MSAPairformer.custom_typing import (
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
        self.init_ln = RMSNorm(dim_msa)
        self.dense = Linear(dim_msa, dim_msa*2)
        self.dense_activation = SwiGLU()
        self.pre_logit_norm = RMSNorm(dim_msa)
        self.regression = Linear(dim_msa, dim_output) 
        torch.nn.init.xavier_uniform_(self.regression.weight)
        torch.nn.init.constant_(self.regression.bias, 0)

    def forward(
        self,
        msa_repr: Float["b s n d"]
    ) -> Float["b s n *"]:
        x = self.init_ln(msa_repr)
        x = self.dense(x)
        x = self.dense_activation(x)
        x = self.pre_logit_norm(x)
        x = self.regression(x)
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
        self.init_ln = RMSNorm(dim_pairwise)
        self.regression = Linear(dim_pairwise, 1)
        torch.nn.init.xavier_uniform_(self.regression.weight)
        torch.nn.init.constant_(self.regression.bias, 0)
        self.sigmoid = Sigmoid()

    def forward(
        self,
        pair_repr: Float["b n n d"]
    ) -> Float["b s n *"]:
        x = self.init_ln(pair_repr)
        x = self.regression(x)
        x = self.sigmoid(x)
        # Symmetrize the output matrix
        x = x.squeeze(-1)
        x = (0.5 * (x + x.transpose(-1, -2)))
        return x