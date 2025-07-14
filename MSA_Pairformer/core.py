import torch
import torch.nn.functional as F
from torch.nn import Module
from torch import nn
from torch.nn import Linear, Sequential, LayerNorm
from torch import Tensor
from functools import partial
from typing import Literal
from einops import pack, unpack
import einx
from .custom_typing import (
    Float,
    Bool,
    typecheck
)

# Linear layer without bias term (simple linear transformation)
LinearNoBias = partial(Linear, bias = False)

class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SwiGLU(Module):
    """
    Input is a tensor representing the projected output of previous layer and divides it 
    into two equal dimensionality tensors. One half is passed through Swish activation 
    and the output of this and the other half is multiplied element-wise
    """
    @typecheck
    def forward(
        self,
        x: Float['... d']
    ) -> Float['... d//2']:
        x, gates = x.chunk(2, dim=-1)
        return F.silu(gates) * x
        
class Transition(Module):
    """
    Algorithm 11
    x <- LayerNorm(x); x in R^{c}
    a = LinearNoBias(x); a in R^{n*c}
    b = LinearNoBias(x); b in R^{n*c}
    x <- LinearNoBias(swish(a) (*)  b); x in R^{c}
    """
    def __init__(
        self,
        dim,
        expansion_factor = 4
    ):
        super().__init__()
        
        dim_inner = int(dim * expansion_factor)
        # We apply PreLayerNorm to these transition layers so it's okay not to do it here
        self.ff = Sequential(
            LinearNoBias(dim, dim_inner * 2),
            SwiGLU(),
            LinearNoBias(dim_inner, dim)
        )

    @typecheck
    def forward(
        self,
        x: Float['... d'],
    ) -> Float['... d']:
        return self.ff(x)

class Dropout(Module):
    @typecheck
    def __init__(
        self,
        prob: float,
        dropout_type: Literal['row','col'] | None = None
    ):
        super().__init__()
        self.dropout = nn.Dropout(prob)
        self.dropout_type = dropout_type

    def forward(
        self,
        t: Tensor
    ) -> Tensor:
        # Check tensor dimensionality
        if self.dropout_type in {'row', 'col'}:
            assert t.ndim == 4, 'Tensor must be 4 dimensions for row / col structured dropout'
        # If dropout type is not specified, use standard dropout
        if not exists(self.dropout_type):
            return self.dropout(t)

        # Row (sequence) dropout
        if self.dropout_type == 'row':
            batch, row, _, _ = t.shape
            ones_shape = (batch, row, 1, 1)
        # Column (residue) dropout
        elif self.dropout_type == 'col':
            batch, _, col, _ = t.shape
            ones_shape = (batch, 1, col, 1)

        ones = t.new_ones(ones_shape)
        dropped = self.dropout(ones)
        return t * dropped

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def pack_one(t, pattern):
    packed, ps = pack([t], pattern)

    def unpack_one(to_unpack, unpack_pattern = None):
        unpacked, = unpack(to_unpack, ps, default(unpack_pattern, pattern))
        return unpacked

    return packed, unpack_one

def to_pairwise_mask(
    mask_i: Bool['... n'],
    mask_j: Bool['... n'] | None = None
) -> Bool['... n n']:

    mask_j = default(mask_j, mask_i)
    assert mask_i.shape == mask_j.shape
    return einx.logical_and('... i, ... j -> ... i j', mask_i, mask_j)

def max_neg_value(t: Tensor):
    return -torch.finfo(t.dtype).max

class PreRMSNorm(Module):
    @typecheck
    def __init__(
        self,
        fn: Module,
        dim: int
    ):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)
    
    @typecheck
    def forward(
        self,
        x: Float['... n d'],
        **kwargs
    ) -> Float['... n d']:
        x = self.norm(x)
        return self.fn(x, **kwargs)

class PreLayerNorm(Module):
    @typecheck
    def __init__(
        self,
        fn: Module,
        dim: int
    ):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    @typecheck
    def forward(
        self,
        x: Float['... n d'],
        **kwargs
    ) -> Float['... n d']:
        # LayerNorm the input tensor
        x = self.norm(x)
        # Forward pass through specified function
        return self.fn(x, **kwargs)