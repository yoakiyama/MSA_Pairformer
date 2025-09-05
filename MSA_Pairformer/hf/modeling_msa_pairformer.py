r"""PyTorch MSA Pairformer model."""
import logging
from collections import OrderedDict
from typing import Literal

# TODO For "transformers": Need to replace einx and einops operations
import einops
import einx
import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from torch.nn import ModuleList
from transformers import AutoTokenizer, PreTrainedModel

# TODO For "transformers": Replace with BoolTensor, FloatTensor because transformers does not support jaxtyping
from MSA_Pairformer.custom_typing import Bool, Float, typecheck
from MSA_Pairformer.hf.configuration_msa_pairformer import MsaPairformerConfig

logger = logging.getLogger(__name__)


# core.Dropout
class Dropout(nn.Module):
    @typecheck
    def __init__(
        self,
        prob: float,
        dropout_type: Literal['row', 'col'] | None = None
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
        if self.dropout_type is None:
            return self.dropout(t)

        # Row (sequence) dropout
        if self.dropout_type == 'row':
            batch, row, _, _ = t.shape
            ones = t.new_ones((batch, row, 1, 1))
        # Column (residue) dropout
        else:  # self.dropout_type == 'col':
            batch, _, col, _ = t.shape
            ones = t.new_ones((batch, 1, col, 1))

        dropped = self.dropout(ones)
        return t * dropped


# core.SwiGLU
class SwiGLU(nn.Module):
    r"""
    Divides input tensor into two tensors of equal dimensionality.
    One half is passed through the Swish activation and multiplied element-wise with the other half.

    References:
        "GLU Variants Improve Transformer" - https://arxiv.org/abs/2002.05202
    """

    @typecheck
    def forward(
        self,
        x: Float['... d']
    ) -> Float['... d//2']:
        x, gates = x.chunk(2, dim=-1)
        return F.silu(gates) * x


# positional_encoding.RelativePositionEncoding
class RelativePositionEncoding(nn.Module):
    """
    Clipped relative positional encoding to initialize the pair representation.
    Some of this is actually a bit unnecessary, but was initially framed to support multi-chain MSAs.
    The current MSA Pairformer release does not actually support multi-chain MSAs, but this is left here for future use.
    """

    def __init__(self, config: MsaPairformerConfig):
        super().__init__()
        self.config = config

        dim_input = (2 * config.r_max + 2) + (2 * config.r_max + 2) + 1 + (2 * config.s_max + 2)
        self.pairwise_init_proj = nn.Linear(dim_input, config.dim_pairwise, bias=False)

    def forward(
        self,
        msa: Float['b s n'],
    ) -> Float['b n n dp']:
        batch_size, _, seq_len = msa.shape

        # One-hot encode distances
        # bins will be single-offset distances
        def onehot(x, bins):
            dist_from_bins = einx.subtract('... i, j -> ... i j', x, bins)
            indices = dist_from_bins.abs().min(dim=-1, keepdim=True).indices
            one_hots = F.one_hot(indices.long(), num_classes=len(bins))
            return one_hots

        # Initialize token indices
        token_idx = torch.arange(seq_len, device=msa.device).unsqueeze(0).expand(batch_size, seq_len)
        # Compute residue, token, and chain distances
        diff_token_idx = einx.subtract('b i, b j -> b i j', token_idx, token_idx)
        # Compute clipped distances
        d_token = torch.clip(diff_token_idx + self.config.r_max, 0, 2 * self.config.r_max)
        d_res = torch.full((batch_size, seq_len, seq_len), self.config.r_max, device=msa.device)
        d_chain = torch.full((batch_size, seq_len, seq_len), 2 * self.config.s_max + 1, device=msa.device)
        # Define bins
        r_arange = torch.arange(2 * self.config.r_max + 2, device=msa.device)
        s_arange = torch.arange(2 * self.config.s_max + 2, device=msa.device)
        # Assign 1-hot encoding of distances
        a_rel_pos = onehot(d_res, r_arange)
        a_rel_token = onehot(d_token, r_arange)
        a_rel_chain = onehot(d_chain, s_arange)
        # Mask for same residue, chain, and entity
        mask_same_entity = torch.ones((batch_size, seq_len, seq_len, 1), device=msa.device)
        # Concatenate tensors and project
        out, _ = einops.pack((a_rel_pos, a_rel_token, mask_same_entity, a_rel_chain), 'b i j *')

        return self.pairwise_init_proj(out)


class MsaPairformerEmbeddings(nn.Module):
    r"""
    Initial embedding layer for the MSA and pairwise representation.

    References:
        Figure 1A, left side (MSA Pairformer).
    """

    def __init__(self, config: MsaPairformerConfig):
        super().__init__()
        self.config = config

        if config.vocab_size is not None:
            self.msa_init_proj = nn.Linear(config.vocab_size, config.dim_msa, bias=False)
        else:
            self.msa_init_proj = nn.Identity()

        self.relative_position_encoding = RelativePositionEncoding(config)
        self.rearrange = Rearrange('... -> ... 1')
        self.token_bond_to_pairwise_feat = nn.Linear(1, config.dim_pairwise, bias=False)

    def forward(
        self,
        msa: Float['b s n'],
    ) -> tuple[Float['b s n dm'], Float['b n n dp']]:
        batch_size, num_seqs, seq_len = msa.shape
        weight_dtype = self.token_bond_to_pairwise_feat.weight.dtype

        # Initialize pair representation
        # Do not support unused "complex_chain_break_indices" argument
        pairwise_repr = self.relative_position_encoding(msa)
        seq_arange = torch.arange(seq_len, device=msa.device)
        token_bonds = einx.subtract('i, j -> i j', seq_arange, seq_arange).abs() == 1
        token_bonds = token_bonds.to(dtype=weight_dtype)
        token_bonds = self.rearrange(token_bonds)
        token_bonds_features = self.token_bond_to_pairwise_feat(token_bonds)
        pairwise_repr = pairwise_repr + token_bonds_features

        # Initialize MSA representation
        msa_onehot = F.one_hot(msa, num_classes=self.config.vocab_size).to(dtype=weight_dtype)
        msa_repr = self.msa_init_proj(msa_onehot)

        return msa_repr, pairwise_repr


# pairwise_operations.MSAPairWeightedAveraging
class MsaPairWeightedAveraging(nn.Module):
    r"""
    References:
        Algorithm 10 (AlphaFold 3 SOM, https://www.nature.com/articles/s41586-024-07487-w).
        Figure 1A, gray box, "Pair-weighted averaging" block on the left side of the MSA track (MSA Pairformer).
    """

    def __init__(self, config: MsaPairformerConfig):
        super().__init__()
        dim_inner = config.dim_head * config.heads

        # Regular dict still throws "TypeError: dict is not a Module subclass"
        self.msa_repr_to_values_and_gates = nn.Sequential(OrderedDict(
            # m_si <- LayerNorm(m_si)
            pre_norm=nn.LayerNorm(config.dim_msa),
            # v^h_si, g^h_si <- LinearNoBias(m_si), sigmoid(LinearNoBias(m_si))
            linear_no_bias=nn.Linear(config.dim_msa, dim_inner * 2, bias=False),
            rearrange=Rearrange('b s n (gv h d) -> gv b h s n d', gv=2, h=config.heads),
        ))

        # b^h_ij <- LinearNoBias(LayerNorm(z_ij))
        self.pairwise_repr_to_attn = nn.Sequential(OrderedDict(
            pre_norm=nn.LayerNorm(config.dim_pairwise),
            linear_no_bias=nn.Linear(config.dim_pairwise, config.heads, bias=False),
            rearrange=Rearrange('b i j h -> b h i j')
        ))

        self.out_proj = nn.Sequential(OrderedDict(
            rearrange=Rearrange('b h s n d -> b s n (h d)'),
            linear_no_bias=nn.Linear(dim_inner, config.dim_msa, bias=False),
            dropout=Dropout(config.dropout, dropout_type=config.dropout_type)
        ))

    @typecheck
    def forward(
        self,
        msa_repr: Float['b s n d'],
        pairwise_repr: Float['b n n dp'],
        residue_mask: Bool['b n']
    ) -> Float['b s n d']:
        # Project MSA representation to values and gates
        values, gates = self.msa_repr_to_values_and_gates(msa_repr)
        gates = gates.sigmoid()

        # Project pairwise representation to attention weights
        b = self.pairwise_repr_to_attn(pairwise_repr)  # b h i j
        residue_mask = einops.rearrange(residue_mask, 'b n -> b 1 1 n')
        max_neg_value = -torch.finfo(b.dtype).max
        b = b.masked_fill(~residue_mask, max_neg_value)
        weights = b.softmax(dim=-1)

        # Value vector weighted average
        out = einops.einsum(weights, values, 'b h i n, b h s n d -> b h s i d')
        # Apply gates
        out = out * gates

        # Combine heads and project to output dimension
        return self.out_proj(out)


# core.Transition
class Transition(nn.Module):
    r"""
    x <- LayerNorm(x); x in R^{c}
    a = LinearNoBias(x); a in R^{n*c}
    b = LinearNoBias(x); b in R^{n*c}
    x <- LinearNoBias(swish(a) (*)  b); x in R^{c}

    References:
        Algorithm 11 (AlphaFold 3 SOM, https://www.nature.com/articles/s41586-024-07487-w).
        Figure 1A, gray box, right-most "Transition" block of both the MSA and pairwise track (MSA Pairformer).
    """

    def __init__(
        self,
        config: MsaPairformerConfig,
        dim: int
    ):
        super().__init__()
        dim_inner = dim * config.transition_expansion_factor

        self.pre_norm = nn.LayerNorm(dim)
        self.linear_no_bias = nn.Linear(dim, dim_inner * 2, bias=False)
        self.swiglu = SwiGLU()
        self.out_proj = nn.Linear(dim_inner, dim, bias=False)

    @typecheck
    def forward(
        self,
        x: Float['... d'],
    ) -> Float['... d']:
        x = self.pre_norm(x)
        x = self.linear_no_bias(x)
        x = self.swiglu(x)
        x = self.out_proj(x)
        return x


class MsaBlock(nn.Module):
    r"""
    MSA pair weighted averaging with gating followed by MSA transition.

    References:
         Figure 1 A, gray box, upper MSA track (MSA Pairformer).
    """

    def __init__(self, config: MsaPairformerConfig):
        super().__init__()
        self.msa_pwa = MsaPairWeightedAveraging(config)
        self.msa_transition = Transition(config, dim=config.dim_msa)

    def forward(
        self,
        msa_repr: Float['b s n dm'],
        pairwise_repr: Float['b n n dp'],
        residue_mask: Bool['b n'],
    ) -> Float['b s n dm']:
        msa_residual = self.msa_pwa(
            msa_repr=msa_repr,
            pairwise_repr=pairwise_repr,
            residue_mask=residue_mask
        )
        msa_repr = msa_repr + msa_residual
        msa_repr = msa_repr + self.msa_transition(msa_repr)
        return msa_repr


# outer_product.PresoftmaxDifferentialOuterProductMean.forward (if self.seq_attn:)
class PreSoftmaxDifferentialAttention(nn.Module):
    r"""
    References:
        Figure 2B, upper middle and Equation 2 (MSA Pairformer).
    """

    def __init__(
        self,
        config: MsaPairformerConfig,
        layer_idx: int
    ):
        super().__init__()
        self.config = config

        self.q_proj = nn.Linear(config.dim_msa, config.dim_qk * 2, bias=False)
        self.k_proj = nn.Linear(config.dim_msa, config.dim_qk * 2, bias=False)
        k_weights = self.q_proj.weight.clone() + torch.randn_like(self.q_proj.weight) * 0.1
        self.k_proj.weight = nn.Parameter(k_weights)
        self.q_norm = nn.LayerNorm(config.dim_qk)
        self.k_norm = nn.LayerNorm(config.dim_qk)

        self.lambda_init = torch.tensor(config.differential_attention_lambda(layer_idx=layer_idx), dtype=torch.bfloat16)
        self.lambda_q1 = nn.Parameter(torch.zeros(config.dim_qk, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(config.dim_qk, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(config.dim_qk, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(config.dim_qk, dtype=torch.float32).normal_(mean=0, std=0.1))

    def forward(
        self,
        msa_repr: Float['b s n d'],
        sequence_mask: Bool['b s'],
        full_mask: Bool['b s n'],
    ) -> Float['b n n dp']:
        # Convention: The first sequence is the "query"
        query_repr = msa_repr[:, 0]

        # Compute Q, K (Both q1/q2 and k1/k2 are computed from the same projection)
        q = self.q_proj(query_repr)  # [b n (2d)]
        k = self.k_proj(msa_repr)  # [b s n (2d)]
        # Split last dimension in half to create q1/q2 and k1/k2
        q = einops.rearrange(q, '... n (two d) -> ... two n d', two=2)
        k = einops.rearrange(k, '... s n (two d) -> ... two s n d', two=2)
        # Normalize q and k
        q = self.q_norm(q)
        k = self.k_norm(k)
        # Compute scaled attention scores (Equation 2)
        scaling = self.config.dim_qk ** -0.5
        seq_weights = torch.einsum("... t n d, ... t s n d-> ... t s n", q, k) * scaling  # [b 2 s n]
        # Compute lambda
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1)).to(q.dtype)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1)).to(q.dtype)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init.to(q.dtype)
        # Average pooling
        norm_factor = (full_mask.sum(dim=-1) + self.config.eps).unsqueeze(1).expand(-1, 2, -1)  # [b, 2, s]
        seq_weights = seq_weights.masked_fill(~full_mask.unsqueeze(1).expand(-1, 2, -1, -1), 0)
        seq_weights = seq_weights.sum(dim=-1) / norm_factor  # [b 2 s]
        # Compute differential
        seq_weights = seq_weights[:, 0, :] - (lambda_full * seq_weights[:, 1, :])  # [b s]
        seq_weights = seq_weights.masked_fill(~sequence_mask, -1e9)
        seq_weights = seq_weights.softmax(dim=-1)
        return seq_weights


# outer_product.PresoftmaxDifferentialOuterProductMean._opm
class OuterProductMean(nn.Module):
    r"""
    References:
        Algorithm 9 (AlphaFold 3 SOM, https://www.nature.com/articles/s41586-024-07487-w).
        Figure 2B, lower middle and Equation 1 (MSA Pairformer).
    """

    def __init__(self, config: MsaPairformerConfig):
        super().__init__()
        # Initialize linear layer to project outer product representation into the pairwise representation
        self.to_pairwise_repr = nn.Linear(config.dim_opm_hidden ** 2, config.dim_pairwise * 2)
        self.swiglu = SwiGLU()

    def forward(
        self,
        a: Float['b n s c'],
        b: Float['b n s c'],
    ) -> Float['b n n dp']:
        outer = torch.einsum("...bac,...dae->...bdce", a, b)
        outer = outer.reshape(outer.shape[:-2] + (-1,))
        outer = self.to_pairwise_repr(outer)
        outer = self.swiglu(outer)
        return outer


# outer_product.PresoftmaxDifferentialOuterProductMean
class QueryBiasedOuterProduct(nn.Module):
    r"""
    The "query" in this context is the first sequence in the MSA.

    References:
        Figure 1A, gray box, "Query-biased outer product" block in the middle (MSA Pairformer).
        Figure 2B and Equation 1 (MSA Pairformer).
    """

    def __init__(
        self,
        config: MsaPairformerConfig,
        layer_idx: int
    ):
        super().__init__()
        # Store config in order to reflect dynamic changes to "use_query_biasing"
        self.config = config
        self.layer_idx = layer_idx

        # Initialize LayerNorm to normalize input MSA representation
        self.pre_norm = nn.LayerNorm(config.dim_msa)
        # Initialize linear layers to project MSA representation into hidden representations for outer product
        self.to_left_hidden = nn.Linear(config.dim_msa, config.dim_opm_hidden, bias=False)
        self.to_right_hidden = nn.Linear(config.dim_msa, config.dim_opm_hidden, bias=False)

        self.pre_softmax_differential_attention = None
        if self.config.use_query_biasing:
            self.pre_softmax_differential_attention = PreSoftmaxDifferentialAttention(config, layer_idx)

        self.outer_product_mean = OuterProductMean(config)

    def forward(
        self,
        msa_repr: Float['b s n d'],
        sequence_mask: Bool['b s'],
        full_mask: Bool['b s n'],
        pairwise_mask: Bool['b n n'],
        seq_weights: Float['b s'] | None = None,
    ) -> Float['b n n dp']:
        normalized_msa_repr = self.pre_norm(msa_repr)

        if self.config.use_query_biasing:
            # Use unsupervised sequence weight learning
            if self.pre_softmax_differential_attention is None:
                self.pre_softmax_differential_attention = PreSoftmaxDifferentialAttention(self.config, self.layer_idx)
            seq_weights = self.pre_softmax_differential_attention(
                msa_repr=normalized_msa_repr,
                sequence_mask=sequence_mask,
                full_mask=full_mask
            )
        elif seq_weights is not None:
            # Use passed sequence weights
            pass
        else:
            # Default to uniform sequence weights
            seq_weights = sequence_mask / sequence_mask.sum(dim=-1, keepdim=True)  # [b, s]

        # Create left and right hidden representations and apply mask to padded positions
        expanded_full_mask = full_mask.unsqueeze(-1)
        a = self.to_left_hidden(normalized_msa_repr) * expanded_full_mask  # [b s n c]
        b = self.to_right_hidden(normalized_msa_repr) * expanded_full_mask  # [b s n c]
        # Transpose for efficient matrix multiplication
        a = a.transpose(-2, -3)  # [b n s c]
        b = b.transpose(-2, -3)  # [b n s c]
        # Scale a and b by the square root of the sequence weights
        scaled_seq_weights = (seq_weights + self.config.eps).sqrt()
        a = torch.einsum("...s,...nsc->...nsc", scaled_seq_weights, a)  # [b n s c]
        b = torch.einsum("...s,...nsc->...nsc", scaled_seq_weights, b)  # [b n s c]
        # Outer product
        outer = self.outer_product_mean(a, b)
        # Mask invalid pairwise positions
        outer = torch.einsum("... i j d, ... i j -> ... i j d", outer, pairwise_mask)
        return outer, seq_weights


# pairwise_operations.TriangleMultiplication
class TriangleMultiplication(nn.Module):
    r"""
    Combines incoming and outgoing edges algorithms (12 and 13) into a single module.
    Only difference between the two is in the indices/elements used in the multiplicative update on line 4 of the
    pseudocode sum_{k}( a_{ik} (*) b_{jk} ) vs sum_{k}( a_{ki} (*) b_{kj} ) for outgoing and incoming, respectively.

    References:
        Algorithm 12 and 13 (AlphaFold 3 SOM, https://www.nature.com/articles/s41586-024-07487-w).
        Figure 1A, gray box, "Triangle updates (outgoing)" and "Triangle updates (incoming)" blocks in the middle of
        the pairwise track (MSA Pairformer).
    """

    @typecheck
    def __init__(
        self,
        config: MsaPairformerConfig,
        mix: Literal['incoming', 'outgoing'] = 'incoming',
    ):
        super().__init__()
        assert mix in ['incoming', 'outgoing'], "mix must be either 'incoming' or 'outgoing'"
        self.mix = mix

        self.pre_norm = nn.LayerNorm(config.dim_pairwise)
        # Linear projection into higher dimensionality and apply GLU
        # aij, bij = sigmoid(LinearNoBias(zij) (*) LinearNoBias(zij))
        # aib, bij in R^{c}
        # Projects to R^{c*4} and will chunk to combine the 4 (a, b, and two gating vectors)
        self.left_right_proj = nn.Sequential(OrderedDict(
            linear_no_bias=nn.Linear(config.dim_pairwise, config.dim_triangle_multiplication * 4, bias=False),
            glu=nn.GLU(dim=-1)
        ))

        # LayerNorm representation before projecting to output dimension
        self.out_norm = nn.LayerNorm(config.dim_triangle_multiplication)

        # zij = gij (*) LinearNoBias(LayerNorm(sum(a_{..} (*) b_{..})))
        # gij = sigmoid(LinearNoBias(zij))
        self.out_gate = nn.Sequential(OrderedDict(
            linear_no_bias=nn.Linear(config.dim_pairwise, config.dim_triangle_multiplication, bias=False),
            sigmoid=nn.Sigmoid()
        ))

        # Project back to input representation dimensionality
        self.out_proj = nn.Sequential(OrderedDict(
            linear_no_bias=nn.Linear(config.dim_triangle_multiplication, config.dim_pairwise, bias=False),
            dropout=Dropout(prob=config.dropout_row_prob, dropout_type='row')
        ))

    @typecheck
    def forward(
        self,
        pairwise_repr: Float['b n n d'],
        pairwise_mask: Bool['b n n'],
    ) -> Float['b n n d']:
        # Both left_right_proj and out_gate need pre-normalized pairwise_repr
        pairwise_repr = self.pre_norm(pairwise_repr)
        left, right = self.left_right_proj(pairwise_repr).chunk(2, dim=-1)  # [b, n, n, d], [b, n, n, d]

        # Compute a and b (line 2)
        pairwise_mask = einops.rearrange(pairwise_mask, '... -> ... 1')  # [b, n, n, 1]
        left = left * pairwise_mask
        right = right * pairwise_mask

        # Triangular update (line 3 + 4) -> LayerNorm
        if self.mix == 'outgoing':
            # sum_{k}( a_{ik} (*) b_{jk} )
            out = einops.einsum(left, right, '... i k d, ... j k d -> ... i j d')
        else:  # "incoming"
            # sum_{k}( a_{ki} (*) b_{kj} )
            out = einops.einsum(left, right, '... k j d, ... k i d -> ... i j d')
        out = self.out_norm(out)  # [b, n, n, d]

        # Compute output gate (line 3) and gate (line 4)
        out_gate = self.out_gate(pairwise_repr)

        # Project back to original dimensionality
        return self.out_proj(out) * out_gate


# pairwise_operations.PairwiseBlock
class PairwiseBlock(nn.Module):
    r"""
    Full PairwiseBlock from AlphaFold3 (uses triangle modules).

    References:
        Figure 1 A, gray box, lower pairwise track (MSA Pairformer).
    """

    def __init__(self, config: MsaPairformerConfig):
        super().__init__()
        # Define incoming/outgoing triangle multiplication and transition layer
        self.triangle_multiplication_outgoing = TriangleMultiplication(config, mix='outgoing')
        self.triangle_multiplication_incoming = TriangleMultiplication(config, mix='incoming')
        self.pairwise_transition = Transition(config, dim=config.dim_pairwise)

    @typecheck
    def forward(
        self,
        pairwise_repr: Float['b n n d'],
        pairwise_mask: Bool['b n n'],
    ) -> Float['b n n d']:
        pairwise_repr = self.triangle_multiplication_outgoing(pairwise_repr, pairwise_mask) + pairwise_repr
        pairwise_repr = self.triangle_multiplication_incoming(pairwise_repr, pairwise_mask) + pairwise_repr
        pairwise_repr = self.pairwise_transition(pairwise_repr) + pairwise_repr
        return pairwise_repr


class MsaPairformerLayer(nn.Module):
    r"""
    Core layer of the MSA Pairformer.

    References:
        Figure 1A, gray box (MSA Pairformer).
    """

    def __init__(
        self,
        config: MsaPairformerConfig,
        layer_idx: int
    ):
        super().__init__()
        # MSA pair weighted averaging with gating -> transition
        self.msa_block = MsaBlock(config)

        # Pairwise representation query-biased outer product -> triangle updates
        self.outer_product = QueryBiasedOuterProduct(config, layer_idx)
        self.pairwise_block = PairwiseBlock(config)

    @typecheck
    def forward(
        self,
        msa_repr: Float['b s n dm'],
        pairwise_repr: Float['b n n dp'],
        residue_mask: Bool['b n'],
        sequence_mask: Bool['b s'],
        full_mask: Bool['b s n'],
        pairwise_mask: Bool['b n n'],
        seq_weights: Float['b s'] | None = None,
    ) -> tuple[Float['b s n dm'], Float['b n n dp'], Float['b s']]:
        # Update MSA representation through pair weighted averaging (with residual connection)
        msa_repr = self.msa_block(
            msa_repr=msa_repr,
            pairwise_repr=pairwise_repr,
            residue_mask=residue_mask
        )

        # Compute outer product mean (with residual connection)
        updated_pairwise_repr, normalized_sequence_weights = self.outer_product(
            msa_repr=msa_repr,
            sequence_mask=sequence_mask,
            full_mask=full_mask,
            pairwise_mask=pairwise_mask,
            seq_weights=seq_weights
        )
        # Update pairwise representation
        pairwise_repr = pairwise_repr + updated_pairwise_repr
        pairwise_repr = self.pairwise_block(pairwise_repr=pairwise_repr, pairwise_mask=pairwise_mask)

        return msa_repr, pairwise_repr, normalized_sequence_weights


# model.CoreModule
class MsaPairformerEncoder(nn.Module):
    r"""
    Core module for MSA Pairformer which includes stacked layers of:

    1) MSA pair weighted averaging (updates MSA representation using pairwise relationships from the pair representation)
    2) Query-biased outer product (updates pair representation using MSA representation)
    3) Triangle updates (updates pair representation using triplet information)
    """

    def __init__(self, config: MsaPairformerConfig):
        super().__init__()
        self.config = config

        self.layers = ModuleList([])
        for layer_idx in range(config.depth):
            msa_pairformer_layer = MsaPairformerLayer(config, layer_idx)
            self.layers.append(msa_pairformer_layer)

        # If we want to do a final MSA update
        self.final_msa_block = MsaBlock(config) if config.do_last_msa_update else None

    def forward(
        self,
        msa_repr: Float['b s n dm'],
        pairwise_repr: Float['b n n dp'],
        residue_mask: Bool['b n'],  # Column mask (masks out specific residue positions)
        sequence_mask: Bool['b s'],  # Row mask (masks out specific sequences)
        full_mask: Bool['b s n'],
        pairwise_mask: Bool['b n n'],
        seq_weights: Float['b s'] | None = None,
        return_seq_weights: bool = False,
        return_query_only: bool = True,
        return_msa_repr_layer_idx: list[int] | None = None,
        return_pairwise_repr_layer_idx: list[int] | None = None,
    ) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        # Track seq weights for all layers
        seq_weights_list_d = {}
        # Track MSA and pairwise representations for specified layers
        msa_repr_d = {}
        pairwise_repr_d = {}

        return_msa_repr_layer_idx = return_msa_repr_layer_idx or []
        return_pairwise_repr_layer_idx = return_pairwise_repr_layer_idx or []

        # Pass MSA through each layer of the core module stack
        for layer_idx, msa_pairformer_layer in enumerate(self.layers):
            msa_repr, pairwise_repr, normalized_sequence_weights = msa_pairformer_layer(
                msa_repr=msa_repr,
                pairwise_repr=pairwise_repr,
                residue_mask=residue_mask,
                sequence_mask=sequence_mask,
                full_mask=full_mask,
                pairwise_mask=pairwise_mask,
                seq_weights=seq_weights
            )

            if return_seq_weights:
                seq_weights_list_d[f"layer_{layer_idx}"] = normalized_sequence_weights
            if layer_idx in return_msa_repr_layer_idx:
                msa_repr_d[f"layer_{layer_idx}"] = msa_repr[:, :1, :, :].cpu() if return_query_only else msa_repr.cpu()
            if layer_idx in return_pairwise_repr_layer_idx:
                pairwise_repr_d[f"layer_{layer_idx}"] = pairwise_repr.cpu()

        # Final MSA update
        if self.final_msa_block is not None:
            msa_repr = self.final_msa_block(
                msa_repr=msa_repr,
                pairwise_repr=pairwise_repr,
                residue_mask=residue_mask
            )
            final_idx = self.config.depth + 1
            if final_idx in return_msa_repr_layer_idx:
                msa_repr_d[f"layer_{final_idx}"] = msa_repr[:, :1, :, :].cpu() if return_query_only else msa_repr.cpu()

        # Organize results
        results = {}
        results['final_msa_repr'] = msa_repr[:, :1, :, :] if return_query_only else msa_repr
        results['final_pairwise_repr'] = pairwise_repr
        results['seq_weights_list_d'] = seq_weights_list_d
        results['pairwise_repr_d'] = pairwise_repr_d
        results['msa_repr_d'] = msa_repr_d
        return results


# regression.LogisticRegressionContactHead
class MsaPairformerContactHead(nn.Module):
    def __init__(self, config: MsaPairformerConfig):
        super().__init__()
        self.init_ln = nn.LayerNorm(config.dim_pairwise)
        self.dense = nn.Linear(config.dim_pairwise, 1, bias=True)
        # Preserve zero initialization of bias from regression.LogisticRegressionContactHead
        self.dense.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        pairwise_repr: Float['b n n d']
    ) -> Float['b s n *']:
        x = self.init_ln(pairwise_repr)
        x = self.dense(x)
        x = F.sigmoid(x)
        # Symmetrize the output matrix
        x = x.squeeze(-1)
        x = (0.5 * (x + x.transpose(-1, -2)))
        return x


# regression.LMHead
class MsaPairformerLmHead(nn.Module):
    def __init__(self, config: MsaPairformerConfig):
        super().__init__()
        self.init_ln = nn.LayerNorm(config.dim_msa)
        self.dense = nn.Linear(config.dim_msa, config.dim_msa * 2)
        self.dense_activation = SwiGLU()
        self.pre_logit_norm = nn.LayerNorm(config.dim_msa)
        self.output = nn.Linear(config.dim_msa, config.dim_logits, bias=True)
        # Preserve zero initialization of bias from regression.LMHead
        self.output.bias = nn.Parameter(torch.zeros(config.dim_logits))

    def forward(
        self,
        msa_repr: Float['b s n d']
    ) -> Float['b s n *']:
        x = self.init_ln(msa_repr)
        x = self.dense(x)
        x = self.dense_activation(x)
        x = self.pre_logit_norm(x)
        x = self.output(x)
        return x


# @auto_docstring
class MsaPairformerPreTrainedModel(PreTrainedModel):
    config_class = MsaPairformerConfig
    config: MsaPairformerConfig
    base_model_prefix = "msa_pairformer"
    supports_gradient_checkpointing = True
    _no_split_modules = []
    _keys_to_ignore_on_load_unexpected = []
    _supports_flash_attn = False

    def _init_weights(self, module):
        return
        """Initialize the weights"""
        # TODO Initialize right modules with right initializations
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, MsaPairformerLmHead):
            module.bias.data.zero_()

    def get_output_embeddings(self):
        # TODO Check if necessary
        # NOTE: get_output_embeddings() must return None to prevent accidental weight tying.
        # See e.g. https://github.com/huggingface/transformers/pull/39339#discussion_r2219126400
        return None


# model.MSAPairformer
# @auto_docstring
class MsaPairformer(MsaPairformerPreTrainedModel):
    r"""
    4 main components:

    1) MsaPairformerEmbeddings
        a) Takes as input the tokenized MSA
        b) Creates the initial pair representation
        c) Creates the initial MSA representation (embedding) from the tokenized MSA
        d) Outputs the MSA and pair representation
    2) MsaPairformerEncoder
        a) Takes as input i) MSA representation; ii) Pair representation
        b) Iteratively updates MSA representation and pair representation bidirectionally
        c) Outputs refined MSA and pair representations
    3) MSA language model head
        b) Takes as input the final MSA representation
        c) Outputs the logits for the MSA language model
    4) Contact head
        a) Takes as input a pairwise representation (in the final released model, this is the representation at layer 15)
        b) Outputs a contact prediction between 0 and 1
    """

    def __init__(self, config: MsaPairformerConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = MsaPairformerEmbeddings(config)
        self.encoder = MsaPairformerEncoder(config)

        self.lm_head = MsaPairformerLmHead(config)
        self.contact_head = MsaPairformerContactHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    # @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,  # [b, s, n]
        attention_mask: torch.BoolTensor | None = None,  # [b, s, n]
        seq_weights: torch.FloatTensor | None = None,  # [b, s]
        return_query_only: bool | None = None,
        return_contacts: bool | None = None,
        return_seq_weights: bool | None = None,  # output_attentions
        return_msa_repr_layer_idx: list[int] | int | None = None,  # output_hidden_states
        return_pairwise_repr_layer_idx: list[int] | int | None = None,  # output_hidden_states
    ) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, number_of_sequences, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.LongTensor` of shape `(batch_size, number_of_sequences, sequence_length)`):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        seq_weights (`torch.FloatTensor` of shape `(batch_size, number_of_sequences)`  *optional*):
            Optional weights for each sequence in the MSA.
        return_query_only (`bool`, *optional*):
            Whether to return only the representation of the query sequence (first sequence in the MSA).
        return_contacts (`bool`, *optional*):
            Whether to return the contact maps predicted from the final pair representation.
        return_seq_weights (`bool`, *optional*):
            Whether to return the sequence attention? weights.
        return_msa_repr_layer_idx (`list[int] or int`, *optional*):
            Optionally return the MSA representation of the specified layers. Respects `return_query_only`.
        return_pairwise_repr_layer_idx (`list[int] or int`, *optional*):
            Optionally return the pairwise representation of the specified layers.
        """
        # Prepare masks
        residue_mask = attention_mask.any(dim=1)  # [b, n]
        sequence_mask = attention_mask.any(dim=2)  # [b, s]
        full_mask = attention_mask.bool()  # [b, s, n]
        pairwise_mask = einx.logical_and('... i, ... j -> ... i j', residue_mask, residue_mask)  # [b, n, n]

        # Initialize representations
        msa_repr, pairwise_repr = self.embeddings(msa=input_ids)

        def default(value, default_value):
            return value if value is not None else default_value

        return_query_only = default(return_query_only, self.config.return_query_only)
        return_contacts = default(return_contacts, self.config.return_contacts)
        return_seq_weights = default(return_seq_weights, self.config.return_seq_weights)

        return_msa_repr_layer_idx = default(return_msa_repr_layer_idx, self.config.return_msa_repr_layer_idx)
        return_msa_repr_layer_idx = default(return_msa_repr_layer_idx, [])
        if isinstance(return_msa_repr_layer_idx, int):
            return_msa_repr_layer_idx = [return_msa_repr_layer_idx]

        return_pairwise_repr_layer_idx = default(return_pairwise_repr_layer_idx,
                                                 self.config.return_pairwise_repr_layer_idx)
        return_pairwise_repr_layer_idx = default(return_pairwise_repr_layer_idx, [])
        if isinstance(return_pairwise_repr_layer_idx, int):
            return_pairwise_repr_layer_idx = [return_pairwise_repr_layer_idx]

        # Ensure that contact layer is in return_pairwise_repr_layer_idx if returning contacts
        if return_contacts and self.config.contact_layer not in return_pairwise_repr_layer_idx:
            return_pairwise_repr_layer_idx.append(self.config.contact_layer)

        # Pass through layers
        results = self.encoder.forward(
            msa_repr=msa_repr,
            pairwise_repr=pairwise_repr,
            residue_mask=residue_mask,
            sequence_mask=sequence_mask,
            full_mask=full_mask,
            pairwise_mask=pairwise_mask,
            seq_weights=seq_weights,
            return_query_only=return_query_only,
            return_seq_weights=return_seq_weights,
            return_msa_repr_layer_idx=return_msa_repr_layer_idx,
            return_pairwise_repr_layer_idx=return_pairwise_repr_layer_idx,
        )

        logits = self.lm_head(results['final_msa_repr'])
        results['logits'] = logits

        if return_contacts:
            pairwise_repr_d = results['pairwise_repr_d'][f'layer_{self.config.contact_layer}']
            contacts = self.contact_head(pairwise_repr_d)
            results['predicted_contacts'] = contacts

        return results

    def predict_contacts(
        self,
        input_ids: Float['b s n d'],
        attention_mask: Bool['b n'] | None = None,
        seq_weights: Float['b s'] | None = None,
    ) -> Float['b s n *']:
        r"""
        Predicts contacts for the query sequence (first sequence) in the MSA (`input_ids`).
        Follows more or less the same interface as `EsmModel.predict_contacts`.

        Examples:

        ```python
        >>> from transformers import EsmModel
        >>> from MSA_Pairformer.hf.modeling_msa_pairformer import MsaPairformer

        >>> esm = EsmModel.from_pretrained('facebook/esm2_t33_650M_UR50D')
        >>> msa_pairformer = MsaPairformer.from_pretrained('yoakiyama/MSA-Pairformer', revision='refs/pr/1')

        >>> msa: Float['b s n']  # MSA (batch size, number of sequences, sequence length)
        >>> query_sequence = msa[:, 0, :]

        >>> esm_contacts = esm.predict_contacts(query_sequence, torch.ones_like(query_sequence))
        >>> msa_pairformer_contacts = msa_pairformer.predict_contacts(msa, torch.ones_like(msa))
        >>> assert esm_contacts.shape == msa_pairformer_contacts.shape
        ```
        """
        results = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            seq_weights=seq_weights,
            return_query_only=True,
            return_contacts=True,
            return_seq_weights=False,
            return_msa_repr_layer_idx=None,
            return_pairwise_repr_layer_idx=[self.config.contact_layer],
        )
        return results['predicted_contacts']


__all__ = [
    'MsaPairformerPreTrainedModel',
    'MsaPairformer'
]

if __name__ == '__main__':
    repo_id = 'yoakiyama/MSA-Pairformer'
    msa_pairformer_tokenizer = AutoTokenizer.from_pretrained(repo_id, revision='refs/pr/1')
    msa_pairformer = MsaPairformer.from_pretrained(repo_id, revision='refs/pr/1')

    from MSA_Pairformer.model import MSAPairformer
    from MSA_Pairformer.dataset import prepare_msa_masks

    device = torch.device('cpu')
    msa_pairformer = msa_pairformer.to(device=device)
    og_msa_pairformer: MSAPairformer = MSAPairformer.from_pretrained(device=device)

    example_msa = [
        'SEQV---ENCE',
        'SEQW-----CE',
        'SEQVEEEENCE'
    ]

    inputs_ = msa_pairformer_tokenizer(example_msa, return_tensors='pt', padding='longest')
    # Add batch dimension
    input_ids_ = inputs_['input_ids'][None].to(device=device)
    attention_mask_ = inputs_['attention_mask'][None].to(device=device)

    results_ = msa_pairformer(input_ids_, attention_mask_, return_query_only=False)

    if device.type != 'cpu':
        print('Skipping sanity check because of GPU non-determinism (results differ after third decimal place or so')
        exit(0)

    # Small sanity check
    dtype = og_msa_pairformer.lm_head.weight.dtype
    msa_onehot_ = F.one_hot(input_ids_, num_classes=msa_pairformer_tokenizer.vocab_size).to(dtype=dtype)
    mask_, msa_mask_, full_mask_, pairwise_mask_ = prepare_msa_masks(input_ids_, device=device)
    layer_indices = list(range(og_msa_pairformer.core_stack.depth))
    og_results_ = og_msa_pairformer.forward(
        msa=msa_onehot_,
        mask=mask_,
        msa_mask=msa_mask_,
        full_mask=full_mask_,
        pairwise_mask=pairwise_mask_,
        return_contacts=True,
        return_seq_weights=True,
        query_only=False,
        return_pairwise_repr_layer_idx=layer_indices,
        return_msa_repr_layer_idx=layer_indices,
    )

    for key_, og_tensor_or_dict in og_results_.items():
        assert key_ in results_, f'MsaPairformer (transformers PreTrainedModel) does not return {key_}'
        tensor_or_dict = results_[key_]

        if isinstance(og_tensor_or_dict, torch.Tensor):
            assert og_tensor_or_dict.dtype == tensor_or_dict.dtype
            assert og_tensor_or_dict.shape == tensor_or_dict.shape
            if key_ == 'predicted_contacts':
                # Don't ask me where this slight difference comes from, the parameters are exactly the same
                assert torch.allclose(og_tensor_or_dict, tensor_or_dict)
                continue

            assert (og_tensor_or_dict == tensor_or_dict).all(), (f'Tensors differ for {key_}: {og_tensor_or_dict} != '
                                                                 f'{tensor_or_dict}')
            continue

        for layer_, og_tensor_ in og_tensor_or_dict.items():
            assert layer_ in tensor_or_dict, f'Missing tensor for {key_}, {layer_}'
            tensor_ = tensor_or_dict[layer_]
            assert og_tensor_.dtype == tensor_.dtype
            assert og_tensor_.shape == tensor_.shape
            assert (og_tensor_ == tensor_).all(), f'Tensors differ for {key_}, {layer_}: {og_tensor_} != {tensor_}'
