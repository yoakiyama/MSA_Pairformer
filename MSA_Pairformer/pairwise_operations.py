import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.nn import Module, Sequential, LayerNorm, Linear
from einops import rearrange, einsum, repeat
from einops.layers.torch import Rearrange
from typing import Literal
from functools import partial
from .core import LinearNoBias, to_pairwise_mask, max_neg_value, exists, pack_one, PreLayerNorm, PreRMSNorm, Dropout, default, Transition
from .custom_typing import (
    Float,
    Bool,
    typecheck
)

# Load environment variables
from environs import Env
_env = Env()
_env.read_env()

try:
    if torch.cuda.is_available():
        import cuequivariance_torch as cuex
        CUEQUIVARIANCE_AVAILABLE = True
        os.environ["CUEQ_DISABLE_AOT_TUNING"] = _env.str(
            "CUEQ_DISABLE_AOT_TUNING", default="1"
        )
        os.environ["CUEQ_DEFAULT_CONFIG"] = _env.str("CUEQ_DEFAULT_CONFIG", default="1")
except ImportError:
    CUEQUIVARIANCE_AVAILABLE = False

def cuex_is_available():
    return CUEQUIVARIANCE_AVAILABLE


class TriangleMultiplication(Module):
    def __init__(
        self,
        dim_pairwise: int,
        dim_hidden: int,
        direction: Literal["outgoing", "incoming"],
        use_in_bias: bool = False,
        use_out_bias: bool = False,
        eps: float = 1e-5,
        use_cuequivariance: bool = CUEQUIVARIANCE_AVAILABLE
    ):
        """
        Triangle multiplicative update class. Defaults to using cuEquivariance, but falls back to standard PyTorch implementation if unavailable.
        Parameter names thus match exactly to those used in cuequivariance_torch.triangle_multiplicative_update.
        Model weight and bias initialization is also identical to cuequivariance_torch.triangle_multiplicative_update.
        cuEquivariance allows for no bias terms in the input and output hidden and gating projections (as in AlphaFold3).
        To experiment with including bias terms, set use_in_bias and/or use_out_bias to True when training.
        """
        super().__init__()
        self.dim_pairwise = dim_pairwise
        self.dim_hidden = dim_hidden
        self.direction = direction
        self.eps = eps
        self.use_cuequivariance = use_cuequivariance
        if self.use_cuequivariance:
            assert dim_pairwise % 32 == 0, "dim_pairwise must be divisible by 32 when using cuequivariance"

        # Triangle multiplicative update einsum operation
        assert direction in ["outgoing", "incoming"], "direction must be either outgoing or incoming"
        if direction == "outgoing":
            self.tri_eq = "... i k d, ... j k d -> ... i j d"
        elif direction == "incoming":
            self.tri_eq = "... k i d, ... k j d -> ... i j d"

        # Normalize input
        self.norm_in = nn.LayerNorm(dim_pairwise)

        # Input projection and gate (left and right combined)
        self.p_in = nn.Linear(dim_pairwise, dim_hidden * 2, bias=use_in_bias)
        self.g_in = nn.Linear(dim_pairwise, dim_hidden * 2, bias=use_in_bias)

        # Output projection and gate
        self.p_out = nn.Linear(dim_pairwise, dim_pairwise, bias=use_out_bias)
        self.g_out = nn.Linear(dim_pairwise, dim_pairwise, bias=use_out_bias)

        # Normalize output 
        self.norm_out = nn.LayerNorm(dim_pairwise)

    def forward(
        self,
        pair_rep,
        pairwise_mask=None
    ):
        if self.use_cuequivariance:
            return self._cuex_forward(
                pair_rep,
                pairwise_mask
            )
        else:
            return self._vanilla_forward(
                pair_rep,
                pairwise_mask
            )

    def _cuex_forward(
        self,
        pair_rep,
        pairwise_mask=None
    ):
        return cuex.triangle_multiplicative_update(
            x = pair_rep,
            direction=self.direction,
            mask=pairwise_mask,
            norm_in_weight=self.norm_in.weight,
            norm_in_bias=self.norm_in.bias,
            p_in_weight=self.p_in.weight,
            p_in_bias=self.p_in.bias,
            g_in_weight=self.g_in.weight,
            g_in_bias=self.g_in.bias,
            norm_out_weight=self.norm_out.weight,
            norm_out_bias=self.norm_out.bias,
            p_out_weight=self.p_out.weight,
            p_out_bias=self.p_out.bias,
            g_out_weight=self.g_out.weight,
            g_out_bias=self.g_out.bias,
            eps=self.eps,
            precision=None, # Triton dot's default
        )

    def _vanilla_forward(
        self,
        pair_rep,
        pairwise_mask=None
    ):
        """
        Standard PyTorch implementation of the triangle multiplicative update.
        Must align with implementation in cuequivariance_torch.triangle_multiplicative_update
        in order to serve as a fallback
        """
        # Normalize input
        pair_rep = torch.nn.functional.layer_norm(
            pair_rep.float(),
            pair_rep.shape[-1:],
            weight = self.norm_in.weight.float(),
            bias = self.norm_in.bias.float(),
        ).to(pair_rep.dtype)

        # Projection to a and b hidden representations, element-wise gating, and masking
        left_right = torch.sigmoid( self.g_in(pair_rep) ) * self.p_in(pair_rep)
        if pairwise_mask is not None:
            left_right = left_right * pairwise_mask.unsqueeze(-1)
        a, b = torch.chunk(left_right, 2, dim=-1)

        # Triangle multiplications and normalization
        out_rep = einsum(a.contiguous(), b.contiguous(), self.tri_eq)
        out_rep = torch.nn.functional.layer_norm(
            out_rep.float(),
            out_rep.shape[-1:],
            weight = self.norm_out.weight.float(),
            bias = self.norm_out.bias.float(),
        ).to(out_rep.dtype)

        # Output projection and element-wise gating
        out_rep = torch.sigmoid(self.g_out(pair_rep)) * self.p_out(out_rep)
        return out_rep

####################
# Triangle updates #
####################
# class TriangleMultiplication(Module):
#     """
#     Combines incoming and outgoing edges algorithms (Algos 12 and 13) into a single module
#     Only difference between the two is in the indices/elements used in the 
#     multiplicative update on line 4 of the pseudo-code
#     sum_{k}( a_{ik} (*) b_{jk} ) vs sum_{k}( a_{ki} (*) b_{kj} )
#     for outgoing and incoming, respectively
#     """
#     @typecheck
#     def __init__(
#         self,
#         dim,
#         dim_hidden = None,
#         mix: Literal["incoming", "outgoing"] = "incoming",
#         dropout = 0.,
#         dropout_type: Literal["row", "col"] | None = None
#     ):
#         super().__init__()

#         # Hidden dimension defaults to input dimension if not specified
#         dim_hidden = default(dim_hidden, dim)

#         # Linear projection into higher dimensionality and apply GLU
#         # aij, bij = sigmoid(LinearNoBias(zij) (*) LinearNoBias(zij)
#         # aib, bij in R^{c}
#         # Projects to R^{c*4} and will chunk to combine the 4 (a, b, and two gating vectors)
#         self.left_right_proj = Sequential(
#             LinearNoBias(dim, dim_hidden * 4),
#             nn.GLU(dim=-1)
#         )

#         # zij = gij (*) LinearNoBias(LayerNorm(sum(a_{..} (*) b_{..})))
#         # gij = sigmoid(LinearNoBias(zij))
#         self.out_gate = LinearNoBias(dim, dim_hidden)

#         # Incoming vs outgoing edges
#         if mix == "outgoing":
#             # sum_{k}( a_{ik} (*) b_{jk} )
#             self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
#         elif mix == "incoming":
#             # sum_{k}( a_{ki} (*) b_{kj} )
#             self.mix_einsum_eq = "... k j d, ... k i d -> ... i j d"

#         # LayerNorm representation before projecting to output dimension
#         self.to_out_norm = LayerNorm(dim_hidden)

#         # Project back to input representation dimensionality
#         self.to_out = Sequential(
#             LinearNoBias(dim_hidden, dim),
#             Dropout(dropout, dropout_type = dropout_type)
#         )

#     @typecheck
#     def forward(
#         self,
#         x: Float["b n n d"],
#         mask: Bool["b n"] | None = None,
#     ) -> Float["b n n d"]:

#         # Get mask
#         if exists(mask):
#             mask = to_pairwise_mask(mask)
#             mask = rearrange(mask, '... -> ... 1')

#         # Compute a and b (line 2)
#         left, right = self.left_right_proj(x).chunk(2, dim = -1)
#         if exists(mask):
#             left = left * mask
#             right = right * mask

#         # Triangular update (line 3 + 4) and LayerNorm
#         out = einsum(left, right, self.mix_einsum_eq)
#         out = self.to_out_norm(out)

#         # Compute output gate (line 3) and gate (line 4)
#         out_gate = self.out_gate(x).sigmoid()

#         # Project back to original dimensionality
#         return self.to_out(out) * out_gate

#####################################
# Pair updates (for ablation study) #
#####################################
class PairMultiplication(Module):
    """
    Combines incoming and outgoing edges algorithms (Algos 12 and 13) into a single module
    Only difference between the two is in the indices/elements used in the 
    multiplicative update on line 4 of the pseudo-code
    sum_{k}( a_{ik} (*) b_{jk} ) vs sum_{k}( a_{ki} (*) b_{kj} )
    for outgoing and incoming, respectively
    """
    @typecheck
    def __init__(
        self,
        dim,
        dim_hidden = None,
        mix: Literal["incoming", "outgoing"] = "incoming",
        dropout = 0.,
        dropout_type: Literal["row", "col"] | None = None
    ):
        super().__init__()

        # Hidden dimension defaults to input dimension if not specified
        dim_hidden = default(dim_hidden, dim)
        
        # # LayerNorm of input tensor
        # # zij <- LayerNorm(zij) 
        # We apply prelayernorm so no need for normalization here

        # Linear projection into higher dimensionality and apply GLU
        # aij, bij = sigmoid(LinearNoBias(zij) (*) LinearNoBias(zij)
        # aib, bij in R^{c}
        # LinearNoBias projects to R^{c*4} and will chunk to combine the 4
        # LinearNoBias operations into a single operation
        self.left_right_proj = Sequential(
            LinearNoBias(dim, dim_hidden * 4),
            nn.GLU(dim=-1)
        )

        # Line 4: zij = gij (*) LinearNoBias(LayerNorm(sum(a_{..} (*) b_{..})))
        # Gating operations
        # gij = sigmoid(LinearNoBias(zij))
        # Will apply sigmoid after LinearNoBias
        self.out_gate = LinearNoBias(dim, dim_hidden)

        # # Incoming vs outgoing edges
        # if mix == "outgoing":
        #     # sum_{k}( a_{ik} (*) b_{jk} )
        #     self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
        # elif mix == "incoming":
        #     # sum_{k}( a_{ki} (*) b_{kj} )
        #     self.mix_einsum_eq = "... k j d, ... k i d -> ... i j d"

        # LayerNorm representation before projecting to output dimension
        self.to_out_norm = LayerNorm(dim_hidden)
        # self.to_out_norm = RMSNorm(dim_hidden)

        # Project back to input representation dimensionality
        self.to_out = Sequential(
            LinearNoBias(dim_hidden, dim),
            Dropout(dropout, dropout_type = dropout_type)
        )

    @typecheck
    def forward(
        self,
        x: Float["b n n d"],
        pairwise_mask: Bool["b n"] | None = None,
    ) -> Float["b n n d"]:

        # Get mask
        pairwise_mask = rearrange(pairwise_mask, '... -> ... 1')

        # Compute a and b (line 2)
        left, right = self.left_right_proj(x).chunk(2, dim = -1) # b n n c
        if exists(pairwise_mask):
            left = left * pairwise_mask
            right = right * pairwise_mask

        # Pair update just updates node ij by combining information from ij and ji and LayerNorm
        out = left * right.transpose(1, 2)
        out = self.to_out_norm(out)

        # Compute output gate (line 3) and gate (line 4)
        out_gate = self.out_gate(x).sigmoid()

        # Project back to original dimensionality
        return self.to_out(out) * out_gate
        
class MSAPairWeightedAveraging(Module):
    """ Algorithm 10 """
    def __init__(
        self,
        *,
        dim_msa = 64,
        dim_pairwise = 128,
        dim_head = 32,
        heads = 8,
        dropout = 0.1,
        dropout_type: Literal['row', 'col'] | None = None,
        return_attn_weights = False
    ):
        super().__init__()
        self.return_attn_weights = return_attn_weights
        dim_inner = dim_head * heads

        self.msa_to_values_and_gates = Sequential(
            LayerNorm(dim_msa), # m_si <- LayerNorm(m_si)
            LinearNoBias(dim_msa, dim_inner * 2), # v^h_si, g^h_si <- LinearNoBias(m_si), sigmoid(LinearNoBias(m_si))
            Rearrange('b s n (gv h d) -> gv b h s n d', gv = 2, h = heads)
        )

        self.pairwise_repr_to_attn = Sequential( # b^h_ij <- LinearNoBias(LayerNorm(z_ij))
            LayerNorm(dim_pairwise),
            LinearNoBias(dim_pairwise, heads),
            Rearrange('b i j h -> b h i j')
        )

        self.to_out = Sequential(
            Rearrange('b h s n d -> b s n (h d)'),
            LinearNoBias(dim_inner, dim_msa),
            Dropout(dropout, dropout_type = dropout_type)
        )

    @typecheck
    def forward(
        self,
        *,
        msa: Float['b s n d'],
        pairwise_repr: Float['b n n dp'],
        mask: Bool['b n'] | None = None
    ) -> Float['b s n d']:

        values, gates = self.msa_to_values_and_gates(msa)
        gates = gates.sigmoid()

        # Project pairwise representation to attention weights
        b = self.pairwise_repr_to_attn(pairwise_repr) # b h i j
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            b = b.masked_fill(~mask, max_neg_value(b))
        weights = b.softmax(dim = -1)

        # Value vector weighted average
        out = einsum(weights, values, 'b h i j, b h s j d -> b h s i d')
        # Apply gates
        out = out * gates

        # Combine heads and project to output dimension
        return self.to_out(out)

class PairwiseBlock(Module):
    """
    Full PairwiseBlock from AlphaFold3: uses triangle modules
    """
    def __init__(
        self,
        dim_pairwise = 128,
        tri_mult_dim_hidden = None,
        dropout_row_prob = 0,
        dropout_col_prob = 0,
        use_triangle_updates = True,
        use_pair_updates = False
    ):
        super().__init__()

        tri_mult_dim_hidden = default(tri_mult_dim_hidden, dim_pairwise)

        # LayerNorm inputs to layers
        pre_ln = partial(PreLayerNorm, dim = dim_pairwise)

        # Triangle multiplication parameters
        tri_mult_kwargs = dict(
            dim_pairwise = dim_pairwise,
            dim_hidden = tri_mult_dim_hidden,
            use_in_bias = False,
            use_out_bias = False
        )
        pair_mult_kwargs = dict(
            dim = dim_pairwise,
            dim_hidden = tri_mult_dim_hidden
        )

        # Define incoming/outgoing triangle multiplication and transition layer
        self.use_triangle_updates = use_triangle_updates
        self.use_pair_updates = use_pair_updates
        if self.use_triangle_updates:
            self.tri_mult_outgoing = TriangleMultiplication(direction = "outgoing", use_cuequivariance = CUEQUIVARIANCE_AVAILABLE, **tri_mult_kwargs)
            self.tri_mult_incoming = TriangleMultiplication(direction = "incoming", use_cuequivariance = CUEQUIVARIANCE_AVAILABLE, **tri_mult_kwargs)
        if self.use_pair_updates:
            self.pair_mult_first = pre_ln(PairMultiplication(dropout = dropout_row_prob, dropout_type = "row", **pair_mult_kwargs))
            self.pair_mult_second = pre_ln(PairMultiplication(dropout = dropout_col_prob, dropout_type = "row", **pair_mult_kwargs))
        self.pairwise_transition = pre_ln(Transition(dim = dim_pairwise))

    @typecheck
    def forward(
        self,
        pairwise_repr: Float["b n n d"],
        pairwise_mask: Bool["b n"] | None = None
    ) -> Float["b n n d"]:
        if self.use_triangle_updates:
            pairwise_repr = self.tri_mult_outgoing(pairwise_repr, pairwise_mask = pairwise_mask) + pairwise_repr
            pairwise_repr = self.tri_mult_incoming(pairwise_repr, pairwise_mask = pairwise_mask) + pairwise_repr
        if self.use_pair_updates:
            pairwise_repr = self.pair_mult_first(pairwise_repr, pairwise_mask = pairwise_mask) + pairwise_repr
            pairwise_repr = self.pair_mult_second(pairwise_repr, pairwise_mask = pairwise_mask) + pairwise_repr
        pairwise_repr = self.pairwise_transition(pairwise_repr) + pairwise_repr
        return pairwise_repr