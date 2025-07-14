import torch
from torch.nn import Module, ModuleList, Sequential
import numpy as np
import einx
from math import exp
from functools import partial
from einops.layers.torch import Rearrange
from huggingface_hub import snapshot_download
from pathlib import Path
from typing import List

from MSA_Pairformer.core import LinearNoBias, PreLayerNorm, Transition, exists
from MSA_Pairformer.outer_product import OuterProduct
from MSA_Pairformer.regression import LMHead, LogisticRegressionContactHead
from MSA_Pairformer.pairwise_operations import MSAPairWeightedAveraging, PairwiseBlock
from MSA_Pairformer.positional_encoding import RelativePositionEncoding
from MSA_Pairformer.custom_typing import Float, Bool

class CoreModule(Module):
    """
    Core module for MSA Pairformer which includes stacked layers of:
    1) MSA pair weighted averaging (updates MSA representation using pairwise relationships from the pair representation)
    2) Query-biased outer product (updates pair representation using MSA representation)
    3) Triangle updates (updates pair representation using triplet information)
    """
    def __init__(
        self,
        *,
        depth: int = 22,
        dim_pairwise: int = 256,
        dim_msa: int = 464,
        opm_kwargs: dict = dict(
            dim_opm_hidden = 16,
            outer_product_flavor = "presoftmax_differential_attention",
            seq_attn = True,
            dim_qk = 128,
            chunk_size = None,
            return_seq_weights = True,
            lambda_init = None,
            eps = 1e-32,
        ),
        pwa_kwargs: dict = dict(
            heads = 8,
            dim_head = 32,
            dropout = 0.0,
            dropout_type = "row",
        ),
        pairwise_block_kwargs: dict = dict(
            dropout_row_prob = 0,
            dropout_col_prob = 0,
            tri_mult_dim_hidden = None, # Defaults to pair representation dimension
            use_triangle_updates = True,
            use_pair_updates = False # For ablation study
        ),
        drop_last_msa_update = False,
        # return_all_pairwise_repr = False,
        # return_pairwise_repr_layer_idx = 15,
        # return_all_msa_repr = False,
        # return_msa_repr_layer_idx = 22,
        # return_repr_after_layer_idx = None,
    ):
        super().__init__()

        # Store parameters
        self.dim_pairwise = dim_pairwise
        self.depth = depth

        # # Store return flags
        # self.return_seq_weights = opm_kwargs['return_seq_weights'] if 'return_seq_weights' in opm_kwargs else False
        # self.return_all_pair_repr = return_all_pairwise_repr
        # self.return_pairwise_repr_layer_idx = return_pairwise_repr_layer_idx
        # self.return_all_msa_repr = return_all_msa_repr
        # self.return_msa_repr_layer_idx = return_msa_repr_layer_idx
        # self.return_repr_after_layer_idx = return_repr_after_layer_idx

        # Automatically assign lambda init if not provided (for presoftmax differential attention)
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
            else:
                lambda_init = None
            opm = OuterProduct(
                dim_msa = dim_msa,
                dim_pairwise = dim_pairwise,
                lambda_init = lambda_init,
                **opm_kwargs
            )
            curr_module_list.append(opm)

            # Pairwise representation update block
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

    # def set_return_all_pairwise_repr(self, return_all_pairwise_repr: bool):
    #     self.return_all_pairwise_repr = return_all_pairwise_repr

    # def set_return_pairwise_repr_layer_idx(self, return_pairwise_repr_layer_idx: int):
    #     self.return_pairwise_repr_layer_idx = return_pairwise_repr_layer_idx

    # def set_return_all_msa_repr(self, return_all_msa_repr: bool):
    #     self.return_all_msa_repr = return_all_msa_repr

    # def set_return_msa_repr_layer_idx(self, return_msa_repr_layer_idx: int):
    #     self.return_msa_repr_layer_idx = return_msa_repr_layer_idx
    
    # def set_return_repr_after_layer_idx(self, return_repr_after_layer_idx: int):
    #     self.return_repr_after_layer_idx = return_repr_after_layer_idx

    def forward(
        self,
        msa: Float['b s n dm'],
        pairwise_repr: Float['b n n dp'],
        mask: Bool['b n'] | None = None, # Column mask
        msa_mask: Bool['b s'] | None = None, # Row mask,
        full_mask: Bool['b s n'] | None = None,
        pairwise_mask: Bool['b n n'] | None = None,
        seq_weights: Float['b s'] | None = None,
        query_only: bool = True,
        return_msa_repr_layer_idx: List[int] | int | None = None,
        return_pairwise_repr_layer_idx: List[int] | int | None = None,
        return_repr_after_layer_idx: int | None = None,
        return_seq_weights: bool = False,
    ) -> Float['b s n dm']:
        # Track seq weights
        seq_weights_list_d = {}
        pairwise_repr_d = {}
        msa_repr_d = {}
        # Turn return layer indices into lists
        if isinstance(return_msa_repr_layer_idx, int):
            return_msa_repr_layer_idx = [return_msa_repr_layer_idx]
        if isinstance(return_pairwise_repr_layer_idx, int):
            return_pairwise_repr_layer_idx = [return_pairwise_repr_layer_idx]
        # Pass MSA through each layer of the core module stack
        for layer_idx, (
            msa_pair_weighted_avg,
            msa_transition,
            outer_product,
            pairwise_block
        ) in enumerate(self.layers):
            # Pair weighted averaging (with residual connection)
            msa_residual = msa_pair_weighted_avg(msa = msa, pairwise_repr = pairwise_repr, mask = mask)
            msa = msa + msa_residual
            msa = msa + msa_transition(msa)
            if (return_msa_repr_layer_idx is None) or (layer_idx in return_msa_repr_layer_idx):
                msa_repr_d[f"layer_{layer_idx}"] = msa[:, :1, :, :] if query_only else msa
            del msa_residual

            # Compute outer product mean (with residual connection)
            update_pairwise_repr, norm_weights = outer_product(
                msa = msa,
                mask = mask,
                msa_mask = msa_mask,
                full_mask = full_mask,
                pairwise_mask = pairwise_mask,
                seq_weights = seq_weights
            )
            pairwise_repr = pairwise_repr + update_pairwise_repr
            del update_pairwise_repr
            if return_seq_weights:
                seq_weights_list_d[f"layer_{layer_idx}"] = norm_weights

            # Pairwise representation block
            pairwise_repr = pairwise_block(pairwise_repr = pairwise_repr, mask = mask)
            if (return_pairwise_repr_layer_idx is None) or (layer_idx in return_pairwise_repr_layer_idx):
                pairwise_repr_d[f"layer_{layer_idx}"] = pairwise_repr

            # Break out of loop early if we've reached the layer from which we want to compute the representations
            if (return_repr_after_layer_idx is not None) and (layer_idx == return_repr_after_layer_idx):
                break

        # Final MSA update
        if exists(self.final_msa_pwa) and (return_repr_after_layer_idx is None):
            msa_residual = self.final_msa_pwa(
                msa = msa,
                pairwise_repr = pairwise_repr,
                mask = mask
            )
            msa = msa + msa_residual
            del msa_residual
            msa = msa + self.final_msa_transition(msa)
            if (return_msa_repr_layer_idx is not None) and (layer_idx+1 in return_msa_repr_layer_idx):
                msa_repr_d[f"layer_{layer_idx+1}"] = msa[:, :1, :, :] if query_only else msa
        
        # Organize results
        res = {}
        res['msa_repr'] = msa[:, :1, :, :] if query_only else msa
        res['pairwise_repr'] = pairwise_repr
        res['msa_repr_d'] = msa_repr_d
        res['pairwise_repr_d'] = pairwise_repr_d
        res['seq_weights_list_d'] = seq_weights_list_d
        return res

class MSAPairformer(Module):
    """
    3 main components:
        1) Core module
            a) Takes as input i) MSA representation; ii) Pair representation
            b) Iteratively updates MSA representation and pair representation bidirectionally
            c) Outputs refined MSA and pair representations
        2) MSA language model head
            b) Takes as input the final MSA representation
            c) Outputs the logits for the MSA language model
        3) Contact head
            a) takes as input a pairwise representation (in the final released model, this is the representation at layer 15)
            b) outputs a contact prediction between 0 and 1
    """
    def __init__(
        self,
        *,
        dim_msa_input: int = 28,
        dim_pairwise = 256,
        dim_msa = 464,
        dim_logits = 26,
        core_module_kwargs: dict = dict(
            depth = 22,
            opm_kwargs = dict(
                dim_opm_hidden = 16,
                outer_product_flavor = "presoftmax_differential_attention",
                seq_attn = True,
                dim_qk = 128,
                chunk_size = None,
                return_seq_weights = True,
                lambda_init = None,
                eps = 1e-32
            ),
            pwa_kwargs = dict(
                heads = 8,
                dim_head = 32,
                dropout = 0.0,
                dropout_type = "row",
            ),
            pairwise_block_kwargs = dict(
                dropout_row_prob = 0,
                dropout_col_prob = 0,
                tri_mult_dim_hidden = None,
                use_triangle_updates = True,
                use_pair_updates = False
            ),
            # return_all_pairwise_repr = False,
            # return_all_msa_repr = False,
            # return_msa_repr_layer_idx = 22,
            # return_repr_after_layer_idx = None
        ),
        relative_position_encoding_kwargs: dict = dict(
            r_max = 32, 
            s_max = 2,
        ),
        contact_layer: int = 15,
    ):
        super().__init__()
        self.dim_pairwise = dim_pairwise
        self.dim_msa = dim_msa
        self.contact_layer = contact_layer

        # Relative position encoding
        self.relative_position_encoding = RelativePositionEncoding(
            dim_out = dim_pairwise,
            **relative_position_encoding_kwargs
        )
        self.token_bond_to_pairwise_feat = Sequential(
            Rearrange('... -> ... 1'),
            LinearNoBias(1, dim_pairwise)
        )

        # Initial MSA projection
        self.msa_init_proj = LinearNoBias(dim_msa_input, dim_msa) if exists(dim_msa_input) else torch.nn.Identity()
        
        # Core module
        self.core_stack = CoreModule(
            dim_msa = dim_msa,
            dim_pairwise = dim_pairwise,
            **core_module_kwargs
        )

        # Projection layer for logits
        self.lm_head = LMHead(
            dim_msa,
            dim_logits
        )

        self.contact_head = LogisticRegressionContactHead(
            dim_pairwise = dim_pairwise,
        )

    @property
    def device(self):
        """Device of the model."""
        return self.zero.device

    ###### Load model ######
    @classmethod
    def from_pretrained(
        cls,
        device: torch.device | None = None,
        weights_dir: str | None = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = cls()
        path = Path(snapshot_download(repo_id="yakiyama/MSA-Pairformer", cache_dir=weights_dir))
        checkpoint = torch.load(path / "model.bin", weights_only=True)
        checkpoint = {k.replace("core_module", "core_stack"): v for k, v in checkpoint.items()}
        contact_checkpoint = torch.load(path / "contact.bin", weights_only=True)
        contact_checkpoint = {f"contact_head.{k}": v for k, v in contact_checkpoint.items()}
        checkpoint.update(contact_checkpoint)
        model.load_state_dict(checkpoint)
        model.to(device)
        return model

    ###### Embed sequences ######
    def init_representations(
        self,
        msa: Float['b s n d'],
        complex_chain_break_indices: List[int] | None = None,
    ):
        # Initialize pair representation
        batch_size, _, seq_len, _ = msa.shape
        pairwise_repr = self.relative_position_encoding(
            batch_size = batch_size,
            seq_len = seq_len,
            device = msa.device,
            complex_chain_break_indices = complex_chain_break_indices
        )
        seq_arange = torch.arange(seq_len, device = msa.device)
        token_bonds = einx.subtract('i, j -> i j', seq_arange, seq_arange).abs() == 1
        token_bonds_feats = self.token_bond_to_pairwise_feat(token_bonds.float())
        pairwise_repr = pairwise_repr + token_bonds_feats

        # Initialize MSA representation
        msa = self.msa_init_proj(msa)

        return msa, pairwise_repr

    ###### Make predictions / embeddings ######
    def forward(
        self,
        msa: Float['b s n d'],
        mask: Bool['b n'] | None = None,
        msa_mask: Bool['b s'] | None = None,
        full_mask: Bool['b s n'] | None = None,
        pairwise_mask: Bool['b n n'] | None = None,
        seq_weights: Float['b s'] | None = None,
        return_contacts: bool = True,
        query_only: bool = True,
        return_pairwise_repr_layer_idx: List[int] | int | None = None,
        return_msa_repr_layer_idx: List[int] | int | None = None,
        complex_chain_break_indices: List[int] | None = None,
    ):
        # Ensure that contact layer is in return_pairwise_repr_layer_idx if returning contacts
        if return_contacts:
            if return_pairwise_repr_layer_idx is None:
                return_pairwise_repr_layer_idx = list(np.arange(self.core_stack.depth))
            elif isinstance(return_pairwise_repr_layer_idx, int):
                return_pairwise_repr_layer_idx = [return_pairwise_repr_layer_idx, self.contact_layer]
            elif self.contact_layer not in return_pairwise_repr_layer_idx:
                return_pairwise_repr_layer_idx.append(self.contact_layer)

        # Initialize representations
        msa, pairwise_repr = self.init_representations(msa, complex_chain_break_indices)
        # Pass through layers
        results = self.core_stack(
            pairwise_repr = pairwise_repr,
            msa = msa,
            mask = mask,
            msa_mask = msa_mask,
            full_mask = full_mask,
            pairwise_mask = pairwise_mask,
            seq_weights = seq_weights,
            query_only = query_only,
            return_msa_repr_layer_idx = return_msa_repr_layer_idx,
            return_pairwise_repr_layer_idx = return_pairwise_repr_layer_idx
        )

        # Get logits via language modeling head
        if query_only:
            logits = self.lm_head(results['msa_repr'])
        else:
            logits = self.lm_head(results['msa_repr'])
        results['logits'] = logits

        # Predict contacts
        if return_contacts:
            contacts = self.contact_head(results['pairwise_repr_d'][f'layer_{self.contact_layer}'])
            results['contacts'] = contacts
        return results

    ###### Contact prediction ######
    def predict_contacts(
        self,
        msa: Float['b s n d'],
        mask: Bool['b n'] | None = None,
        msa_mask: Bool['b s'] | None = None,
        full_mask: Bool['b s n'] | None = None,
        pairwise_mask: Bool['b n n'] | None = None,
        seq_weights: Float['b s'] | None = None,
        complex_chain_break_indices: List[int] | None = None,
    ):
        # Initialize representations
        msa, pairwise_repr = self.init_representations(msa, complex_chain_break_indices)
        # Pass through layers
        results = self.core_stack(
            pairwise_repr = pairwise_repr,
            msa = msa,
            mask = mask,
            msa_mask = msa_mask,
            full_mask = full_mask,
            pairwise_mask = pairwise_mask,
            seq_weights = seq_weights,
            query_only = True,
            return_repr_after_layer_idx = self.contact_layer,
            return_pairwise_repr_layer_idx = [self.contact_layer]
        )

        # Predict contacts
        contacts = self.contact_head(results['pairwise_repr_d'][f'layer_{self.contact_layer}'])
        return contacts