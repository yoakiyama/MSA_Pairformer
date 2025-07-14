import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Sequential, LayerNorm, Linear
from einops import rearrange, einsum, repeat
from einops.layers.torch import Rearrange
from typing import Literal
from .core import LinearNoBias, to_pairwise_mask, max_neg_value, exists, pack_one, PreLayerNorm, PreRMSNorm, Dropout, default, Transition
from functools import partial
from .custom_typing import (
    Float,
    Bool,
    typecheck
)

####################
# Triangle updates #
####################
class TriangleMultiplication(Module):
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

        # Linear projection into higher dimensionality and apply GLU
        # aij, bij = sigmoid(LinearNoBias(zij) (*) LinearNoBias(zij)
        # aib, bij in R^{c}
        # Projects to R^{c*4} and will chunk to combine the 4 (a, b, and two gating vectors)
        self.left_right_proj = Sequential(
            LinearNoBias(dim, dim_hidden * 4),
            nn.GLU(dim=-1)
        )

        # zij = gij (*) LinearNoBias(LayerNorm(sum(a_{..} (*) b_{..})))
        # gij = sigmoid(LinearNoBias(zij))
        self.out_gate = LinearNoBias(dim, dim_hidden)

        # Incoming vs outgoing edges
        if mix == "outgoing":
            # sum_{k}( a_{ik} (*) b_{jk} )
            self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
        elif mix == "incoming":
            # sum_{k}( a_{ki} (*) b_{kj} )
            self.mix_einsum_eq = "... k j d, ... k i d -> ... i j d"

        # LayerNorm representation before projecting to output dimension
        self.to_out_norm = LayerNorm(dim_hidden)

        # Project back to input representation dimensionality
        self.to_out = Sequential(
            LinearNoBias(dim_hidden, dim),
            Dropout(dropout, dropout_type = dropout_type)
        )

    @typecheck
    def forward(
        self,
        x: Float["b n n d"],
        mask: Bool["b n"] | None = None,
    ) -> Float["b n n d"]:

        # Get mask
        if exists(mask):
            mask = to_pairwise_mask(mask)
            mask = rearrange(mask, '... -> ... 1')

        # Compute a and b (line 2)
        left, right = self.left_right_proj(x).chunk(2, dim = -1)
        if exists(mask):
            left = left * mask
            right = right * mask

        # Triangular update (line 3 + 4) and LayerNorm
        out = einsum(left, right, self.mix_einsum_eq)
        out = self.to_out_norm(out)

        # Compute output gate (line 3) and gate (line 4)
        out_gate = self.out_gate(x).sigmoid()

        # Project back to original dimensionality
        return self.to_out(out) * out_gate

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
        mask: Bool["b n"] | None = None,
    ) -> Float["b n n d"]:

        # Get mask
        if exists(mask):
            mask = to_pairwise_mask(mask)
            mask = rearrange(mask, '... -> ... 1')

        # Compute a and b (line 2)
        left, right = self.left_right_proj(x).chunk(2, dim = -1) # b n n c
        if exists(mask):
            left = left * mask
            right = right * mask

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

        # LayerNorm inputs to layers
        pre_ln = partial(PreLayerNorm, dim = dim_pairwise)

        # Triangle multiplication parameters
        tri_mult_kwargs = dict(
            dim = dim_pairwise,
            dim_hidden = tri_mult_dim_hidden
        )
        pair_mult_kwargs = dict(
            dim = dim_pairwise,
            dim_hidden = tri_mult_dim_hidden
        )

        # Define incoming/outgoing triangle multiplication and transition layer
        self.use_triangle_updates = use_triangle_updates
        self.use_pair_updates = use_pair_updates
        if self.use_triangle_updates:
            self.tri_mult_outgoing = pre_ln(TriangleMultiplication(mix = "outgoing", dropout = dropout_row_prob, dropout_type = "row", **tri_mult_kwargs))
            self.tri_mult_incoming = pre_ln(TriangleMultiplication(mix = "incoming", dropout = dropout_row_prob, dropout_type = "row", **tri_mult_kwargs))
        if self.use_pair_updates:
            self.pair_mult_first = pre_ln(PairMultiplication(dropout = dropout_row_prob, dropout_type = "row", **pair_mult_kwargs))
            self.pair_mult_second = pre_ln(PairMultiplication(dropout = dropout_col_prob, dropout_type = "row", **pair_mult_kwargs))
        self.pairwise_transition = pre_ln(Transition(dim = dim_pairwise))

    @typecheck
    def forward(
        self,
        pairwise_repr: Float["b n n d"],
        mask: Bool["b n"] | None = None
    ) -> Float["b n n d"]:
        if self.use_triangle_updates:
            pairwise_repr = self.tri_mult_outgoing(pairwise_repr, mask = mask) + pairwise_repr
            pairwise_repr = self.tri_mult_incoming(pairwise_repr, mask = mask) + pairwise_repr
        if self.use_pair_updates:
            pairwise_repr = self.pair_mult_first(pairwise_repr, mask = mask) + pairwise_repr
            pairwise_repr = self.pair_mult_second(pairwise_repr, mask = mask) + pairwise_repr
        pairwise_repr = self.pairwise_transition(pairwise_repr) + pairwise_repr
        return pairwise_repr