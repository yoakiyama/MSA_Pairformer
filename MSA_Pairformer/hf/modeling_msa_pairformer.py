r"""PyTorch MSA Pairformer model."""
import logging
import math
from functools import partial

# TODO For "transformers": Need to replace einx and einops operations
import einx
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn import ModuleList, Sigmoid
from transformers import AutoTokenizer, PreTrainedModel

# TODO For "transformers": Add the following PyTorch modules to this file
from MSA_Pairformer.core import PreLayerNorm, SwiGLU, Transition
from MSA_Pairformer.custom_typing import Bool, Float
from MSA_Pairformer.dataset import prepare_msa_masks
from MSA_Pairformer.hf.configuration_msa_pairformer import MsaPairformerConfig
from MSA_Pairformer.model import MSAPairformer
from MSA_Pairformer.outer_product import OuterProduct
from MSA_Pairformer.pairwise_operations import MSAPairWeightedAveraging, PairwiseBlock
from MSA_Pairformer.positional_encoding import RelativePositionEncoding

logger = logging.getLogger(__name__)


class MsaPairformerEmbeddings(nn.Module):
    r"""
    Initial embedding layer for pair and MSA representation.
    """

    def __init__(self, config: MsaPairformerConfig):
        super().__init__()
        self.config = config

        self.relative_position_encoding = RelativePositionEncoding(
            dim_out=config.dim_pairwise,
            r_max=config.r_max,
            s_max=config.s_max
        )

        self.rearrange = Rearrange('... -> ... 1')
        self.token_bond_to_pairwise_feat = nn.Linear(1, config.dim_pairwise, bias=False)

        if config.vocab_size is not None:
            self.msa_init_proj = nn.Linear(config.vocab_size, config.dim_msa, bias=False)
        else:
            self.msa_init_proj = nn.Identity()

    def forward(
        self,
        msa: Float['b s n'],
    ) -> tuple[Float['b s n dm'], Float['b n n dp']]:
        batch_size, num_seqs, seq_len = msa.shape
        weight_dtype = self.token_bond_to_pairwise_feat.weight.dtype

        # Initialize pair representation
        # Do not support unused "complex_chain_break_indices" argument
        pairwise_repr = self.relative_position_encoding(
            batch_size=batch_size,
            seq_len=seq_len,
            device=msa.device,
        )
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


# model.CoreModule
class MsaPairformerEncoder(nn.Module):
    r"""
    Core module for MSA Pairformer which includes stacked layers of:

    1) MSA pair weighted averaging (updates MSA representation using pairwise relationships from the pair representation)
    2) Query-biased outer product (updates pair representation using MSA representation)
    3) Triangle updates (updates pair representation using triplet information)
    """

    def __init__(
        self,
        config: MsaPairformerConfig
    ):
        super().__init__()
        self.config = config

        # Automatically assign lambda init if not provided (for presoftmax differential attention)
        if config.lambda_init is None:
            auto_lambda_init = True
        else:
            auto_lambda_init = False

        # Initialize module stack
        self.layers = ModuleList([])
        for layer_idx in range(config.depth):
            layer_modules = ModuleList()

            # TODO For "transformers": Pass config to modules
            # MSA pair weighted averaging with gating -> transition
            msa_pair_weighted_avg = MSAPairWeightedAveraging(
                dim_msa=config.dim_msa,
                dim_pairwise=config.dim_pairwise,
                dim_head=config.dim_head,
                heads=config.heads,
                dropout=config.dropout,
                dropout_type=config.dropout_type,
                return_attn_weights=config.return_attn_weights
            )
            msa_pre_ln = partial(PreLayerNorm, dim=config.dim_msa)
            layer_modules.append(msa_pair_weighted_avg)
            layer_modules.append(msa_pre_ln(Transition(dim=config.dim_msa)))

            # Outer product
            if auto_lambda_init and ("differential" in config.outer_product_flavor):
                lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_idx)
                lambda_init = torch.tensor(lambda_init, dtype=torch.bfloat16)
            else:
                lambda_init = None
            opm = OuterProduct(
                dim_msa=config.dim_msa,
                dim_pairwise=config.dim_pairwise,
                dim_opm_hidden=config.dim_opm_hidden,
                outer_product_flavor=config.outer_product_flavor,
                seq_attn=config.seq_attn,
                dim_qk=config.dim_qk,
                chunk_size=config.return_seq_weights,
                lambda_init=lambda_init,
                eps=config.eps
            )
            layer_modules.append(opm)

            # Pairwise representation update block
            pairwise_block = PairwiseBlock(
                dim_pairwise=config.dim_pairwise,
                tri_mult_dim_hidden=config.tri_mult_dim_hidden,
                dropout_row_prob=config.dropout_row_prob,
                dropout_col_prob=config.dropout_col_prob,
                use_triangle_updates=config.use_triangle_updates,
                use_pair_updates=config.use_pair_updates
            )
            layer_modules.append(pairwise_block)

            # Append all blocks of current layer to module list
            self.layers.append(layer_modules)

        # If we want to do a final MSA update
        self.final_msa_pwa = None
        self.final_msa_transition = None
        if not config.drop_last_msa_update:
            # MSA pair weighted averaging with gating
            self.final_msa_pwa = MSAPairWeightedAveraging(
                dim_msa=config.dim_msa,
                dim_pairwise=config.dim_pairwise,
                dim_head=config.dim_head,
                heads=config.heads,
                dropout=config.dropout,
                dropout_type=config.dropout_type,
                return_attn_weights=config.return_attn_weights
            )
            # Transition module
            msa_pre_ln = partial(PreLayerNorm, dim=config.dim_msa)
            self.final_msa_transition = msa_pre_ln(Transition(dim=config.dim_msa))

        # Other parameters
        self.register_buffer('zero', torch.tensor(0.), persistent=False)

    def turn_off_seq_attn(self):
        for layer in self.layers:
            for module in layer:
                if isinstance(module, OuterProduct):
                    module.opm.seq_attn = False

    def turn_on_seq_attn(self):
        for layer in self.layers:
            for module in layer:
                if isinstance(module, OuterProduct):
                    module.opm.seq_attn = True

    def forward(
        self,
        msa: Float['b s n dm'],
        pairwise_repr: Float['b n n dp'],
        residue_mask: Bool['b n'] | None = None,  # Column mask (masks out specific residue positions)
        sequence_mask: Bool['b s'] | None = None,  # Row mask (masks out specific sequences)
        full_mask: Bool['b s n'] | None = None,
        pairwise_mask: Bool['b n n'] | None = None,
        seq_weights: Float['b s'] | None = None,
        return_seq_weights: bool = False,
        return_query_only: bool = True,
        return_msa_repr_layer_idx: list[int] | int | None = None,
        return_pairwise_repr_layer_idx: list[int] | int | None = None,
        return_repr_after_layer_idx: int | None = None,
    ) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        # Track seq weights, pairwise representations, and MSA representations for specified layers
        # seq weights are tracked throughout all layers
        seq_weights_list_d = {}
        pairwise_repr_d = {}
        msa_repr_d = {}

        # Turn return layer indices into lists
        if isinstance(return_msa_repr_layer_idx, int):
            return_msa_repr_layer_idx = [return_msa_repr_layer_idx]
        if isinstance(return_pairwise_repr_layer_idx, int):
            return_pairwise_repr_layer_idx = [return_pairwise_repr_layer_idx]

        # Pass MSA through each layer of the core module stack
        for layer_idx, (msa_pair_weighted_avg, msa_transition, outer_product, pairwise_block) in enumerate(self.layers):
            # Pair weighted averaging (with residual connection)
            msa_residual = msa_pair_weighted_avg(msa=msa, pairwise_repr=pairwise_repr, mask=residue_mask)
            msa = msa + msa_residual
            msa = msa + msa_transition(msa)
            if return_msa_repr_layer_idx is None or layer_idx in return_msa_repr_layer_idx:
                msa_repr_d[f"layer_{layer_idx}"] = msa[:, :1, :, :] if return_query_only else msa
            del msa_residual

            # Compute outer product mean (with residual connection)
            update_pairwise_repr, norm_weights = outer_product(
                msa=msa,
                mask=residue_mask,
                msa_mask=sequence_mask,
                full_mask=full_mask,
                pairwise_mask=pairwise_mask,
                seq_weights=seq_weights
            )
            pairwise_repr = pairwise_repr + update_pairwise_repr
            del update_pairwise_repr
            if return_seq_weights:
                seq_weights_list_d[f"layer_{layer_idx}"] = norm_weights

            # Pairwise representation block
            pairwise_repr = pairwise_block(pairwise_repr=pairwise_repr, mask=residue_mask)
            if (return_pairwise_repr_layer_idx is None) or (layer_idx in return_pairwise_repr_layer_idx):
                pairwise_repr_d[f"layer_{layer_idx}"] = pairwise_repr

            # Break out of loop early if we've reached the layer from which we want to compute the representations
            if return_repr_after_layer_idx is not None and layer_idx == return_repr_after_layer_idx:
                break

        # Final MSA update
        if self.final_msa_pwa is not None and return_repr_after_layer_idx is None:
            msa_residual = self.final_msa_pwa(
                msa=msa,
                pairwise_repr=pairwise_repr,
                mask=residue_mask
            )
            msa = msa + msa_residual
            del msa_residual
            msa = msa + self.final_msa_transition(msa)
            if return_msa_repr_layer_idx is not None and layer_idx + 1 in return_msa_repr_layer_idx:
                msa_repr_d[f"layer_{layer_idx + 1}"] = msa[:, :1, :, :] if return_query_only else msa

        # Organize results
        results = {}
        results['final_msa_repr'] = msa[:, :1, :, :] if return_query_only else msa
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
        self.sigmoid = Sigmoid()

    def forward(
        self,
        pairwise_repr: Float["b n n d"]
    ) -> Float["b s n *"]:
        x = self.init_ln(pairwise_repr)
        x = self.dense(x)
        x = self.sigmoid(x)
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
        msa_repr: Float["b s n d"]
    ) -> Float["b s n *"]:
        x = self.init_ln(msa_repr)
        x = self.dense(x)
        x = self.dense_activation(x)
        x = self.pre_logit_norm(x)
        x = self.output(x)
        return x


# @auto_docstring
class MsaPairformerPreTrainedModel(PreTrainedModel):
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

    # Toggle query-biased attention
    def turn_off_seq_attn(self):
        self.encoder.turn_off_seq_attn()

    def turn_on_seq_attn(self):
        self.encoder.turn_on_seq_attn()

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
        return_repr_after_layer_idx: int | None = None,  # output_hidden_states
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
        return_repr_after_layer_idx (`int`, *optional*):
            Optionally stop early and return the representations of the specified layer. Respects `return_query_only`.
        """
        # Prepare masks
        residue_mask = attention_mask.any(dim=1)  # [b, n]
        sequence_mask = attention_mask.any(dim=2)  # [b, s]
        full_mask = attention_mask.bool()  # [b, s, n]
        pairwise_mask = einx.logical_and('... i, ... j -> ... i j', residue_mask, residue_mask)  # [b, n, n]

        # Initialize representations
        msa_repr, pairwise_repr = self.embeddings(msa=input_ids)

        return_query_only = return_query_only if return_query_only is not None else self.config.return_query_only
        return_contacts = return_contacts if return_contacts is not None else self.config.return_contacts
        return_seq_weights = return_seq_weights if return_seq_weights is not None else self.config.return_seq_weights

        return_pairwise_repr_layer_idx = return_pairwise_repr_layer_idx or self.config.return_pairwise_repr_layer_idx

        # Ensure that contact layer is in return_pairwise_repr_layer_idx if returning contacts
        if return_contacts:
            if return_pairwise_repr_layer_idx is None:
                return_pairwise_repr_layer_idx = list(range(self.config.depth))
            elif isinstance(return_pairwise_repr_layer_idx, int):
                return_pairwise_repr_layer_idx = [return_pairwise_repr_layer_idx, self.config.contact_layer]
            elif self.config.contact_layer not in return_pairwise_repr_layer_idx:
                return_pairwise_repr_layer_idx.append(self.config.contact_layer)

        # Pass through layers
        results = self.encoder.forward(
            msa=msa_repr,
            pairwise_repr=pairwise_repr,
            residue_mask=residue_mask,
            sequence_mask=sequence_mask,
            full_mask=full_mask,
            pairwise_mask=pairwise_mask,
            seq_weights=seq_weights,
            return_query_only=return_query_only or self.config.return_query_only,
            return_seq_weights=return_seq_weights or self.config.return_seq_weights,
            return_msa_repr_layer_idx=return_msa_repr_layer_idx or self.config.return_msa_repr_layer_idx,
            return_pairwise_repr_layer_idx=return_pairwise_repr_layer_idx,
            return_repr_after_layer_idx=return_repr_after_layer_idx or self.config.return_repr_after_layer_idx
        )

        logits = self.lm_head(results['final_msa_repr'])
        results['logits'] = logits

        if return_contacts:
            pairwise_repr_d = results['pairwise_repr_d'][f'layer_{self.config.contact_layer}']
            contacts = self.contact_head(pairwise_repr_d)
            results['contacts'] = contacts

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
        >>> msa_pairformer = MsaPairformer.from_pretrained('mauricebrenner/msa_pairformer')

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
            return_repr_after_layer_idx=self.config.contact_layer,
        )
        return results["contacts"]


__all__ = [
    'MsaPairformerPreTrainedModel',
    'MsaPairformer'
]

if __name__ == '__main__':
    repo_id = 'yoakiyama/MSA-Pairformer'
    msa_pairformer_tokenizer = AutoTokenizer.from_pretrained(repo_id)
    msa_pairformer = MsaPairformer.from_pretrained(repo_id)

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

    results_ = msa_pairformer(input_ids_, attention_mask_)

    if device.type != 'cpu':
        print('Skipping sanity check because of GPU non-determinism (results differ after third decimal place or so')
        exit(0)

    # Small sanity check
    dtype = og_msa_pairformer.lm_head.weight.dtype
    msa_onehot_ = F.one_hot(input_ids_, num_classes=msa_pairformer_tokenizer.vocab_size).to(dtype=dtype)
    mask_, msa_mask_, full_mask_, pairwise_mask_ = prepare_msa_masks(input_ids_, device=device)
    og_results_ = og_msa_pairformer.forward(
        msa=msa_onehot_,
        mask=mask_,
        msa_mask=msa_mask_,
        full_mask=full_mask_,
        pairwise_mask=pairwise_mask_
    )

    for key_, og_tensor_or_dict in og_results_.items():
        assert key_ in results_, f'MsaPairformer (transformers PreTrainedModel) does not return {key_}'
        tensor_or_dict = results_[key_]

        if isinstance(og_tensor_or_dict, torch.Tensor):
            assert og_tensor_or_dict.dtype == tensor_or_dict.dtype
            assert og_tensor_or_dict.shape == tensor_or_dict.shape
            if key_ == 'contacts':
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
