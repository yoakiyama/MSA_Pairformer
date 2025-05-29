import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Sequential, LayerNorm, RMSNorm, Linear
from einops import rearrange, einsum, repeat
from einops.layers.torch import Rearrange
from typing import Literal
from functools import partial

from MSAPairformer.custom_typing import (
    Float,
    Bool,
    typecheck
)
from MSAPairformer.attention import Attention
from MSAPairformer.core import LinearNoBias, to_pairwise_mask, max_neg_value, exists, pack_one, PreRMSNorm, Dropout, default, Transition

##############################
# Triangle updates/attention #
##############################
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

        
        self.left_right_proj = Sequential(
            LinearNoBias(dim, dim_hidden * 4),
            nn.GLU(dim=-1)
        )

        # Line 4: zij = gij (*) LinearNoBias(LayerNorm(sum(a_{..} (*) b_{..})))
        # Gating operations
        # gij = sigmoid(LinearNoBias(zij))
        # Will apply sigmoid after LinearNoBias
        self.out_gate = LinearNoBias(dim, dim_hidden)

        # Incoming vs outgoing edges
        if mix == "outgoing":
            # sum_{k}( a_{ik} (*) b_{jk} )
            self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
        elif mix == "incoming":
            # sum_{k}( a_{ki} (*) b_{kj} )
            self.mix_einsum_eq = "... k j d, ... k i d -> ... i j d"

        # LayerNorm representation before projecting to output dimension
        self.to_out_norm = RMSNorm(dim_hidden)

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
        self.to_out_norm = RMSNorm(dim_hidden)

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


class TriangleAttention(Module):
    """
    Single module that implements Alogirthms 14 and 15
    Trianglular gated self-attention around starting/ending node, respectively
    Only difference is in the incides of the key, bias, and value terms in the attention
    calculation (similar to TriangleMultiplication)
    LayerNorm (doesn't show up in public re-implementation lucidrains repo. Applies
    PreLayerNorm to these modules so that input is already normalized
    """
    @typecheck
    def __init__(
        self,
        dim,    
        heads,
        node_type: Literal["starting", "ending"],
        dropout = 0.,
        dropout_type: Literal["row", "col"] | None = None,
        **attn_kwargs
    ):
        super().__init__()
        
        self.need_transpose = node_type == "ending"

        self.attn = Attention(dim = dim, heads = heads, **attn_kwargs)

        self.dropout = Dropout(dropout, dropout_type = dropout_type)

        self.to_attn_bias = Sequential(
            LinearNoBias(dim, heads),
            Rearrange('... i j h -> ... h i j')
        )

    @typecheck
    def forward(
        self,
        pairwise_repr: Float["b n n d"],
        mask: Bool["b n"] | None = None,
        **kwargs
    ) -> Float["b n n d"]:

        # If doing triangle attention around ending node, rearrange pair-rep so that
        # operations use appropriate indices
        if self.need_transpose:
            pairwise_repr = rearrange(pairwise_repr, 'b i j d -> b j i d')

        # Compute attention bias terms (line 3)
        # b_{ij}^{h} = LinearNoBias(zij) 
        # b h n n 
        attn_bias = self.to_attn_bias(pairwise_repr)

        # For ease of indexing, we will flatten the pairwise_repr tensor
        # pairwise_repr is a (b, n, n, d) tensor
        # We combine/flatten the first two dimensions such that it is a (b * n, n, d) tensor
        # That is, the pairwise_repr is flattened by 1 dimension so that it stacks the batches
        # (batches no longer separated by its own dimension)
        batch_repeat = pairwise_repr.shape[1]
        attn_bias = repeat(attn_bias, 'b ... -> (b repeat) ...', repeat = batch_repeat) # b*n h n n
        if exists(mask):
            mask = repeat(mask, 'b ... -> (b repeat) ...', repeat = batch_repeat)
        pairwise_repr, unpack_one = pack_one(pairwise_repr, '* n d') # b*n, n, d

        # Compute self-attention
        out = self.attn(
            pairwise_repr, # b*n, n, d
            mask = mask, 
            attn_bias = attn_bias, # b*n, h, n, n
            **kwargs
        )
        # Re-expand from (b*n, n d) to (b, n, n, d)
        out = unpack_one(out)
        # Transpose back to (b i j d) if triangle attention around ending node
        if self.need_transpose:
            out = rearrange(out, 'b j i d -> b i j d')

        # Apply dropout to transformed pairwise representation
        return self.dropout(out)

        
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
            RMSNorm(dim_msa), # m_si <- RMSNorm(m_si)
            LinearNoBias(dim_msa, dim_inner * 2), # v^h_si, g^h_si <- LinearNoBias(m_si), sigmoid(LinearNoBias(m_si))
            Rearrange('b s n (gv h d) -> gv b h s n d', gv = 2, h = heads)
        )

        self.pairwise_repr_to_attn = Sequential( # b^h_ij <- LinearNoBias(RMSNorm(z_ij))
            RMSNorm(dim_pairwise),
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
        attn_weights: Float['b s n'] | None = None,
        mask: Bool['b n'] | None = None
    ) -> Float['b s n d']:

        values, gates = self.msa_to_values_and_gates(msa)
        gates = gates.sigmoid()
        if exists(attn_weights):
            attn_gates = attn_weights.sigmoid()
            gates = gates * attn_gates.unsqueeze(1).unsqueeze(-1)
        else:
            attn_gates = None

        # line 3
        b = self.pairwise_repr_to_attn(pairwise_repr) # b h i j

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            b = b.masked_fill(~mask, max_neg_value(b))

        # line 5
        weights = b.softmax(dim = -1)

        # line 6
        out = einsum(weights, values, 'b h i j, b h s j d -> b h s i d')
        
        out = out * gates

        # combine heads
        if not self.return_attn_weights:
            weights = None
        return self.to_out(out), weights, attn_gates

class PairwiseBlock(Module):
    def __init__(
        self,
        dim_pairwise = 128,
        tri_mult_dim_hidden = None,
        tri_attn_dim_head = 32,
        tri_attn_heads = 4,
        dropout_row_prob = 0,
        dropout_col_prob = 0,
        use_triangle_attn = True,
        use_triangle_updates = True,
        use_pair_updates = False
    ):
        super().__init__()

        # LayerNorm inputs to layers
        pre_ln = partial(PreRMSNorm, dim = dim_pairwise)

        # Triangle multiplication and attention parameters
        tri_mult_kwargs = dict(
            dim = dim_pairwise,
            dim_hidden = tri_mult_dim_hidden
        )
        tri_attn_kwargs = dict(
            dim = dim_pairwise,
            heads = tri_attn_heads,
            dim_head = tri_attn_dim_head
        )
        pair_mult_kwargs = dict(
            dim = dim_pairwise,
            dim_hidden = tri_mult_dim_hidden
        )

        # Define incoming/outgoing triangle multiplication and starting/ending attention blocks and transition layer
        self.use_triangle_attn = use_triangle_attn
        self.use_triangle_updates = use_triangle_updates
        self.use_pair_updates = use_pair_updates
        if self.use_triangle_updates:
            self.tri_mult_outgoing = pre_ln(TriangleMultiplication(mix = "outgoing", dropout = dropout_row_prob, dropout_type = "row", **tri_mult_kwargs))
            self.tri_mult_incoming = pre_ln(TriangleMultiplication(mix = "incoming", dropout = dropout_row_prob, dropout_type = "row", **tri_mult_kwargs))
        if self.use_pair_updates:
            self.pair_mult_first = pre_ln(PairMultiplication(dropout = dropout_row_prob, dropout_type = "row", **pair_mult_kwargs))
            self.pair_mult_second = pre_ln(PairMultiplication(dropout = dropout_col_prob, dropout_type = "row", **pair_mult_kwargs))
        if self.use_triangle_attn:
            self.tri_attn_starting = pre_ln(TriangleAttention(node_type = "starting", dropout = dropout_row_prob, dropout_type = "row", **tri_attn_kwargs))
            self.tri_attn_ending = pre_ln(TriangleAttention(node_type = "ending", dropout = dropout_col_prob, dropout_type = "col", **tri_attn_kwargs))
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
        if self.use_triangle_attn:
            pairwise_repr = self.tri_attn_starting(pairwise_repr, mask = mask) + pairwise_repr
            pairwise_repr = self.tri_attn_ending(pairwise_repr, mask = mask) + pairwise_repr
        pairwise_repr = self.pairwise_transition(pairwise_repr) + pairwise_repr
        return pairwise_repr