#### Adapted from alphafold3-pytorch repository (https://github.com/lucidrains/alphafold3-pytorch/)
import torch
import torch.nn.functional as F
from torch import nn, sigmoid
from torch import Tensor
from torch.nn import Module, ModuleList, Linear, Sequential, ModuleDict, LayerNorm
from torch.utils.checkpoint import checkpoint

import einx
import re
import numpy as np
from math import pi, sqrt, exp
from functools import partial, wraps
from einops import rearrange, einsum, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange
from typing import List, Literal, Tuple, Optional

from .core import LinearNoBias, PreLayerNorm, Transition, exists, RMSNorm
from .outer_product import OuterProduct
from .regression import ContactHead, LMHead, LogisticRegressionContactHead, MultiLayerLogisticRegressionContactHead
from .pairwise_operations import MSAPairWeightedAveraging, PairwiseBlock
from .positional_encoding import RelativePositionEncoding
from .alphafold3_attention import Attention
from .alphafold3_custom_typing import (
    Float,
    Bool,
    typecheck
)

###############################################################
# Partially ablated MSA modules for MLM or inv cov. objective #
###############################################################
class CoreModule(Module):
    """
    Removes pair representation input and operations, but maintains an outer product mean
    tensor and updates in each layer using a residual connection.
    Also performs pair-weighted averaging + transition after
    pair representation updates (AF3 performs this before pair rep updates)
    """
    def __init__(
        self,
        *,
        depth: int = 4, # Number of blocks in stack
        dim_pairwise: int = 128,
        dim_msa: int = 64,
        opm_kwargs: dict = dict(
            dim_opm_hidden = 32,
            outer_product_flavor = "vanilla",
            seq_attn = False,
            dim_qk = None,
            chunk_size = None,
            return_seq_weights = False,
            return_attn_logits = False,
            lambda_init = None,
            eps = 1e-32,
            nSeqs = None,
            nResidues = None
        ),
        pwa_kwargs: dict = dict(
            heads = 8,
            dim_head = 32,
            dropout = 0.15,
            dropout_type = "row",
            return_attn_weights = False
        ),
        pairwise_block_kwargs: dict = dict(
            tri_mult_dim_hidden = None,
            tri_attn_dim_head = 32,
            tri_attn_heads = 4,
            dropout_row_prob = 0,
            dropout_col_prob = 0,
            use_triangle_updates = True,
            use_pair_updates = False
        ),
        dim_logits = 26,
        drop_last_msa_update = False,
        return_after_layer_idx = None,
        return_all_pairwise_repr = False
    ):
        super().__init__()

        # Store parameters
        self.dim_pairwise = dim_pairwise
        self.return_seq_weights = opm_kwargs['return_seq_weights'] if 'return_seq_weights' in opm_kwargs else False
        self.return_attn_weights = pwa_kwargs['return_attn_weights'] if 'return_attn_weights' in pwa_kwargs else False
        self.return_after_layer_idx = return_after_layer_idx
        self.return_all_pairwise_repr = return_all_pairwise_repr

        # Automatically assign lambda init if not provided
        if ('lambda_init' not in opm_kwargs) or (opm_kwargs['lambda_init'] is None):
            auto_lambda_init = True
        else:
            auto_lambda_init = False
            lambda_init = opm_kwargs['lambda_init']
        if 'lambda_init' in opm_kwargs:
            opm_kwargs.pop("lambda_init")

        # Initialize module stack
        layers = ModuleList([])
        for layer_idx in range(depth):
            curr_module_list = ModuleList()

            # MSA pair weighted averaging with gating -> transition
            msa_pair_weighted_avg = MSAPairWeightedAveraging(
                dim_msa = dim_msa, 
                dim_pairwise = dim_pairwise,
                **pwa_kwargs
            )
            msa_pre_ln = partial(PreLayerNorm, dim = dim_msa)
            curr_module_list.append(msa_pair_weighted_avg)
            curr_module_list.append(msa_pre_ln(Transition(dim = dim_msa)))

            # Outer product
            if auto_lambda_init and ('differential' in opm_kwargs['outer_product_flavor']):
                lambda_init = 0.8 - 0.6 * exp(-0.3 * layer_idx)
                lambda_init = torch.tensor(lambda_init, dtype=torch.bfloat16)
                # print(f"Layer {layer_idx} lambda init: {lambda_init}")
            else:
                lambda_init = None
            opm = OuterProduct(
                dim_msa = dim_msa,
                dim_pairwise = dim_pairwise,
                lambda_init = lambda_init,
                **opm_kwargs
            )
            curr_module_list.append(opm)

            # Pairwise representation update block with triangle attention
            pairwise_block = PairwiseBlock(
                dim_pairwise = dim_pairwise,
                **pairwise_block_kwargs
            )
            curr_module_list.append(pairwise_block)

            # Append all blocks of current layer to module list
            layers.append(curr_module_list)

        # Store layers in class object
        self.layers = layers

        # If we want to do a final MSA update
        self.final_msa_pwa = None
        self.final_msa_transition = None
        if not drop_last_msa_update:
            # MSA pair weighted averaging with gating
            self.final_msa_pwa = MSAPairWeightedAveraging(
                dim_msa = dim_msa, 
                dim_pairwise = dim_pairwise,
                **pwa_kwargs
            )
            # Transition module
            msa_pre_ln = partial(PreLayerNorm, dim = dim_msa)
            self.final_msa_transition = msa_pre_ln(Transition(dim = dim_msa))

        # Other parameters
        self.register_buffer('zero', torch.tensor(0.), persistent = False)

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
        mask: Bool['b n'] | None = None, # Column mask
        msa_mask: Bool['b s'] | None = None, # Row mask,
        full_mask: Bool['b s n'] | None = None,
        pairwise_mask: Bool['b n n'] | None = None,
        seq_weights: Float['b s'] | None = None
    ) -> Float['b s n dm']:
        # Track seq weights
        seq_weights_list_d = {}
        attn_gates_list_d = {}
        attn_weights_list_d = {}
        pairwise_repr_l = []
        layer_idx = 0
        attn_weights = None
        # Pass MSA through each layer of the MSA module stack
        for (
            msa_pair_weighted_avg,
            msa_transition,
            outer_product,
            pairwise_block
        ) in self.layers:
            # Pair weighted averaging (with residual connection)
            msa_residual, attn_weights, attn_gates = msa_pair_weighted_avg(msa = msa, pairwise_repr = pairwise_repr, attn_weights = attn_weights, mask = mask)
            msa = msa + msa_residual
            if self.return_seq_weights and exists(attn_gates):
                attn_gates_list_d[f"layer_{layer_idx}"] = attn_gates
            if self.return_attn_weights and exists(attn_weights):
                attn_weights_list_d[f"layer_{layer_idx}"] = attn_weights
            del attn_gates, msa_residual
            msa = msa + msa_transition(msa)

            # Compute outer product mean (with residual connection)
            update_pairwise_repr, norm_weights, attn_weights = outer_product(msa, mask = mask, msa_mask = msa_mask, full_mask = full_mask, pairwise_mask = pairwise_mask, seq_weights = seq_weights)
            if self.return_seq_weights:
                seq_weights_list_d[f"layer_{layer_idx}"] = norm_weights
            pairwise_repr = pairwise_repr + update_pairwise_repr
            del update_pairwise_repr

            # Pairwise representation block (# Debug: make sure we haven't already normalized the pairwise repr)
            pairwise_repr = pairwise_block(pairwise_repr = pairwise_repr, mask = mask)
            layer_idx += 1
            if self.return_after_layer_idx is not None and layer_idx > self.return_after_layer_idx:
                break
            if self.return_all_pairwise_repr:
                pairwise_repr_l.append(pairwise_repr)

        if exists(self.final_msa_pwa) and (self.return_after_layer_idx is None):
            msa_residual, attn_weights, attn_gates = self.final_msa_pwa(msa = msa, pairwise_repr = pairwise_repr, mask = mask, attn_weights = attn_weights)
            msa = msa + msa_residual
            if self.return_seq_weights:
                attn_gates_list_d[f"layer_{layer_idx+1}"] = attn_gates
            del attn_gates, msa_residual
            msa = msa + self.final_msa_transition(msa)
        if not self.return_seq_weights:
            seq_weights_list_d, attn_weights_list_d = None, None
        if not self.return_attn_weights:
            attn_weights_list_d = None
        if self.return_all_pairwise_repr:
            pairwise_repr = torch.stack(pairwise_repr_l, dim=1)
        return msa, pairwise_repr, seq_weights_list_d, attn_gates_list_d, attn_weights_list_d

class MSAPairformer(Module):
    """
    3 main components:
        1) Core module
            a) Takes as input i) MSA representation; ii) Pair representation
            b) Iteratively updates MSA representation and pair representation
            c) Outputs refined MSA and pair representations
        2) MSA language model head
            b) Takes as input the final MSA representation
            c) Outputs the logits for the MSA language model
        3) Pairformer stack (optional)
    Trained using a variety of potential objectives:
        1) MSA language model (corruption via masking + mutation -> original MSA)
        2) MRF cross-entropy loss
        3) Alignment (Unalign MSA --> Realign MSA)
        4) Combination of multiple
    """
    def __init__(
        self,
        *,
        dim_msa_input: int = 28,
        dim_pairwise = 128,
        dim_msa = 64,
        dim_logits = 26,
        msa_module_kwargs: dict = dict(
            depth = 4, # Number of blocks in stack
            opm_kwargs = dict(
                dim_opm_hidden = 32,
                outer_product_flavor = "vanilla",
                seq_attn = False,
                dim_qk = None,
                chunk_size = None,
                return_seq_weights = False,
                lambda_init = None,
                eps = 1e-32
            ),
            pwa_kwargs = dict(
                heads = 8,
                dim_head = 32,
                dropout = 0.15,
                dropout_type = "row",
                return_attn_weights = False
            ),
            pairwise_block_kwargs = dict(
                tri_mult_dim_hidden = None,
                tri_attn_dim_head = 32,
                tri_attn_heads = 4,
                dropout_row_prob = 0,
                dropout_col_prob = 0,
                use_triangle_updates = True,
                use_pair_updates = False
            ),
            return_after_layer_idx = None,
            return_all_pairwise_repr = False
        ),
        basic_relative_position_encoding_kwargs: dict = dict(
            r_max = 32, # Maximum relative distance between two residues
        ),
        relative_position_encoding_kwargs: dict = dict(
            r_max = 32, # Maximum relative distance between two residues
            s_max = 2, # Maximum relative "chain" distance between two chains
        ),
        use_basic_relative_position_encoding: bool = False,
        return_msa_repr: bool = False,
        return_pairwise_repr: bool = False,
        query_only: bool = False
    ):
        super().__init__()
        self.dim_pairwise = dim_pairwise
        self.dim_msa = dim_msa
        # Initializing pairwise representation
        # Relative position encoding
        if use_basic_relative_position_encoding:
            self.relative_position_encoding = BasicRelativePositionEncoding(
                dim_out = dim_pairwise,
                **basic_relative_position_encoding_kwargs
            )
        else:
            self.relative_position_encoding = RelativePositionEncoding(
                dim_out = dim_pairwise,
                **relative_position_encoding_kwargs
            )
        # Token bonds for flanking residues (Algorithm 1 line 5)
        # z^{init}_{ij} += LinearNoBias(f^{token_bonds}_{ij})
        self.token_bond_to_pairwise_feat = Sequential(
            Rearrange('... -> ... 1'),
            LinearNoBias(1, dim_pairwise)
        )

        # Initial MSA projection
        self.msa_init_proj = LinearNoBias(dim_msa_input, dim_msa) if exists(dim_msa_input) else nn.Identity()
        
        # MSA module
        self.msa_module = CoreModule(
            dim_msa = dim_msa,
            dim_pairwise = dim_pairwise,
            **msa_module_kwargs
        )

        # Projection layer for logits
        self.lm_head = LMHead(
            dim_msa,
            dim_logits
        )
        # Initialize so that predictions are close to uniform across amino acids
        torch.nn.init.xavier_uniform_(self.lm_head.weight)
        torch.nn.init.constant_(self.lm_head.bias, 0)

        # Flag whether to return MSA and pairwise representations
        self.use_basic_relative_position_encoding = use_basic_relative_position_encoding
        self.return_msa_repr = return_msa_repr
        self.return_pairwise_repr = return_pairwise_repr
        self.return_after_layer_idx = msa_module_kwargs['return_after_layer_idx'] if 'return_after_layer_idx' in msa_module_kwargs else None
        self.query_only = query_only
        self.return_seq_weights = msa_module_kwargs['opm_kwargs']['return_seq_weights'] if 'return_seq_weights' in msa_module_kwargs['opm_kwargs'] else False
        self.return_attn_weights = msa_module_kwargs['pwa_kwargs']['return_attn_weights'] if 'return_attn_weights' in msa_module_kwargs['pwa_kwargs'] else False

    @property
    def device(self):
        """Device of the model."""
        return self.zero.device

    def turn_off_seq_attn(self):
        self.msa_module.turn_off_seq_attn()

    def turn_on_seq_attn(self):
        self.msa_module.turn_on_seq_attn()

    def forward(
        self,
        additional_molecule_feats,
        msa: Float['b s n d'], 
        prev_msa_repr: Float['b s n d'] | None = None,
        prev_pairwise_repr: Float['b n n d'] | None = None,
        recycled_input: bool = False,
        mask: Bool['b n'] | None = None,
        msa_mask: Bool['b s'] | None = None,
        full_mask: Bool['b s n'] | None = None,
        pairwise_mask: Bool['b n n'] | None = None,
        seq_weights: Float['b s'] | None = None
    ):      
        ########## Prepare pairwise representation ##########
        # Initialize pair representation (INIT WITH POSITIONAL EMBEDDING)
        pairwise_repr = self.relative_position_encoding(additional_molecule_feats = additional_molecule_feats)
        seq_len = msa.shape[2]
        seq_arange = torch.arange(seq_len, device = msa.device)
        token_bonds = einx.subtract('i, j -> i j', seq_arange, seq_arange).abs() == 1
        token_bonds_feats = self.token_bond_to_pairwise_feat(token_bonds.float())
        pairwise_repr = pairwise_repr + token_bonds_feats
        
        ########## Prepare MSA ##########
        # Initial MSA projection to representation dimensionality
        msa = self.msa_init_proj(msa)

        ########## Add previous iteration's MSA and pairwise representations ##########
        if recycled_input:
            assert prev_msa_repr is not None and prev_pairwise_repr is not None, "prev_msa_repr and prev_pairwise_repr must be provided if recycled_input is True"
            # Normalize the MSA and pairwise representation
            msa += self.prep_recycle_msa(prev_msa_repr)
            pairwise_repr += self.prep_recycle_pairwise(prev_pairwise_repr)
        
        ########## Pass through MSA module ##########
        # Updates the MSA and pairwise representations
        msa, pairwise_repr, seq_weights_list, attn_gates_list, attn_weights_list = self.msa_module(
            pairwise_repr = pairwise_repr,
            msa = msa,
            mask = mask,
            msa_mask = msa_mask,
            full_mask = full_mask,
            pairwise_mask = pairwise_mask,
            seq_weights = seq_weights
        )
        # Initialize output dictionary
        res = {}

        ########## Predict amino acid identities at each position in the MSA ##########
        # Get logits via language modeling head
        if self.query_only:
            logits = self.lm_head(msa[:, 0:1, :, :])
        else:
            logits = self.lm_head(msa)
        res['logits'] = logits

        ########## Return representations ##########
        if self.return_msa_repr:
            res["msa_repr"] = msa
        if self.return_pairwise_repr:
            res["pairwise_repr"] = pairwise_repr
        if self.return_seq_weights:
            res["seq_weights"] = seq_weights_list
            res["attn_gates"] = attn_gates_list
        if self.return_attn_weights:
            res["attn_weights"] = attn_weights_list
        return res

class MSAContactModel(Module):
    """
    Loads a pre-trained MSA language model and predicts contact maps from MSAs
    """
    def __init__(
        self,
        pretrained_weights_path: str,
        pretrained_is_final: bool,
        msa_model_kwargs: dict,
        dim_contact_hidden: int = None,
        dim_contact_input: int = None,
        compiled_checkpoint: bool = False,
        logits_grad: bool = False,
        logreg_contact_head: bool = False,
        multi_layer_logreg_contact_head: bool = False,
        multi_layer_logreg_contact_head_num_layers: int = None,
        apply_apc: bool = False
    ):
        super().__init__()
        self.msa_model_kwargs = msa_model_kwargs
        self.logits_grad = logits_grad
        if 'return_pairwise_repr' in msa_model_kwargs:
            assert msa_model_kwargs['return_pairwise_repr'] == True, "return_pairwise_repr must be True"
        self.pretrained_model = AF3MSAMLM(
            **msa_model_kwargs
        )
        checkpoint = torch.load(pretrained_weights_path, weights_only=True)
        if not pretrained_is_final:
            checkpoint = checkpoint['model_state_dict']
        if compiled_checkpoint:
            checkpoint = remove_orig_mod_prefix(checkpoint)

        missing_keys, unexpected_keys = self.pretrained_model.load_state_dict(checkpoint, strict=False)
        print("Missing keys: ", np.unique(['.'.join(k.split('.')[-2:]) for k in missing_keys]))
        print("Unexpected keys: ", unexpected_keys)
        self.pretrained_model.eval()
        # Initialize contact prediction head
        dim_input = dim_contact_input if dim_contact_input is not None else msa_model_kwargs['dim_pairwise']
        if logreg_contact_head:
            self.contact_head = LogisticRegressionContactHead(
                dim_pairwise = dim_contact_input,
            )
        elif multi_layer_logreg_contact_head:
            assert multi_layer_logreg_contact_head_num_layers is not None, "multi_layer_logreg_contact_head_num_layers must be provided if multi_layer_logreg_contact_head is True"
            self.contact_head = MultiLayerLogisticRegressionContactHead(
                dim_pairwise = dim_input,
                num_layers = multi_layer_logreg_contact_head_num_layers
            )
        else:
            assert dim_contact_hidden is not None, "dim_contact_hidden must be provided if logreg_contact_head is False"
            self.contact_head = ContactHead(
                dim_pairwise = dim_input,
                dim_dense_hidden = dim_contact_hidden
            )

    def turn_off_seq_attn(self):
        self.pretrained_model.turn_off_seq_attn()

    def turn_on_seq_attn(self):
        self.pretrained_model.turn_on_seq_attn()

    def forward(
        self,
        additional_molecule_feats,
        msa: Float['b s n d'], 
        mask: Bool['b n'] | None = None,
        msa_mask: Bool['b s'] | None = None,
        full_mask: Bool['b s n'] | None = None,
        pairwise_mask: Bool['b n n'] | None = None,
        seq_weights: Float['b s'] | None = None,
        use_layer_l: List[int] | None = None
    ):
        if not self.logits_grad:
            with torch.no_grad():
                res = self.pretrained_model(
                    additional_molecule_feats = additional_molecule_feats,
                    msa = msa,
                    mask = mask,
                    msa_mask = msa_mask,
                    full_mask = full_mask,
                    pairwise_mask = pairwise_mask,
                    seq_weights = seq_weights
                )
        else:
            res = self.pretrained_model(
                additional_molecule_feats = additional_molecule_feats,
                msa = msa,
                mask = mask,
                msa_mask = msa_mask,
                full_mask = full_mask,
                pairwise_mask = pairwise_mask,
                seq_weights = seq_weights
            )
        pairwise_repr = res['pairwise_repr']
        if use_layer_l is not None:
            assert pairwise_repr.dim() == 5, "pairwise_repr must be 5D if use_layer_l is provided (b, l, n, n, d)"
            b, l, n, _, d = pairwise_repr.shape
            use_nlayers = len(use_layer_l)
            # pairwise_repr = pairwise_repr[:, use_layer_l, :, :, :].view(b, n, n, d * use_nlayers)
            pairwise_repr = pairwise_repr[:, use_layer_l, :, :, :]
        predicted_contact_map = self.contact_head(pairwise_repr)
        res['predicted_contact_map'] = predicted_contact_map
        return res

    def compute_pairwise_repr(
        self,
        additional_molecule_feats,
        msa: Float['b s n d'], 
        mask: Bool['b n'] | None = None,
        msa_mask: Bool['b s'] | None = None,
        full_mask: Bool['b s n'] | None = None,
        pairwise_mask: Bool['b n n'] | None = None,
        seq_weights: Float['b s'] | None = None
    ):
        with torch.no_grad():
            res = self.pretrained_model(
                additional_molecule_feats = additional_molecule_feats,
                msa = msa,
                mask = mask,
                msa_mask = msa_mask,
                full_mask = full_mask,
                pairwise_mask = pairwise_mask,
                seq_weights = seq_weights
            )
        return res['pairwise_repr']

    def compute_contacts_from_pairwise_repr(
        self,
        pairwise_repr: Float['b n n d']
    ):
        return self.contact_head(pairwise_repr)

def remove_orig_mod_prefix(state_dict):
    return {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}