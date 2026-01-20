import torch
from torch.nn import Module, ModuleList, Sequential
import numpy as np
import einx
import os
from math import exp
from functools import partial
from einops.layers.torch import Rearrange
from huggingface_hub import snapshot_download
from pathlib import Path
from typing import List
from glob import glob

from MSA_Pairformer.core import LinearNoBias, PreLayerNorm, Transition, exists
from MSA_Pairformer.outer_product import OuterProduct
from MSA_Pairformer.regression import LMHead, LogisticRegressionContactHead
from MSA_Pairformer.pairwise_operations import MSAPairWeightedAveraging, PairwiseBlock, cuex_is_available
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
        )
    ):
        super().__init__()

        # Store parameters
        self.dim_pairwise = dim_pairwise
        self.depth = depth

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

        # Final MSA update using MSA pair weighted averaging with gating
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

    def turn_off_query_biasing(self):
        for layer in self.layers:
            for module in layer:
                if isinstance(module, OuterProduct):
                    module.opm.seq_attn = False

    def turn_on_query_biasing(self):
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
        seq_weights: Float['b s'] | None = None,
        seq_weights_dict: dict = None,
        query_only: bool = True,
        return_msa_repr_layer_idx: List[int] | int | None = None,
        return_pairwise_repr_layer_idx: List[int] | int | None = None,
        return_repr_after_layer_idx: int | None = None,
        return_seq_weights: bool = False,
        store_pairwise_repr_cpu: bool = True,
        store_msa_repr_cpu: bool = True,
        return_pairwise_repr_only: bool = False,
    ):
        # Track seq weights, pairwise representations, and MSA representations for specified layers
        # seq weights are tracked throughout all layers
        seq_weights_list_d = {}
        pairwise_repr_d = {}
        msa_repr_d = {}
        if isinstance(return_msa_repr_layer_idx, int):
            return_msa_repr_layer_idx = [return_msa_repr_layer_idx]
        elif return_msa_repr_layer_idx is None:
            return_msa_repr_layer_idx = []
        if isinstance(return_pairwise_repr_layer_idx, int):
            return_pairwise_repr_layer_idx = [return_pairwise_repr_layer_idx]
        elif return_pairwise_repr_layer_idx is None:
            return_pairwise_repr_layer_idx = []
            
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
            del msa_residual
            msa = msa + msa_transition(msa)
            if layer_idx in return_msa_repr_layer_idx:
                if store_msa_repr_cpu:
                    msa_repr_d[f"layer_{layer_idx}"] = msa[:, :1, :, :].cpu() if query_only else msa.cpu()
                else:
                    msa_repr_d[f"layer_{layer_idx}"] = msa[:, :1, :, :] if query_only else msa

            # Compute outer product mean (with residual connection)
            curr_seq_weights = None
            if seq_weights_dict is not None:
                if f"layer_{layer_idx}" in seq_weights_dict:
                    curr_seq_weights = seq_weights_dict[f"layer_{layer_idx}"].to(msa.device)
            elif seq_weights is not None:
                curr_seq_weights = seq_weights

            update_pairwise_repr, norm_weights = outer_product(
                msa = msa,
                mask = mask,
                msa_mask = msa_mask,
                full_mask = full_mask,
                pairwise_mask = pairwise_mask,
                seq_weights = curr_seq_weights
            )
            pairwise_repr = pairwise_repr + update_pairwise_repr
            del update_pairwise_repr
            if return_seq_weights:
                seq_weights_list_d[f"layer_{layer_idx}"] = norm_weights.cpu()

            # Pairwise representation block
            pairwise_repr = pairwise_block(pairwise_repr = pairwise_repr, pairwise_mask = pairwise_mask)
            if layer_idx in return_pairwise_repr_layer_idx:
                if store_pairwise_repr_cpu:
                    pairwise_repr_d[f"layer_{layer_idx}"] = pairwise_repr.cpu()
                else:
                    pairwise_repr_d[f"layer_{layer_idx}"] = pairwise_repr

            # Break out of loop early if we've reached the layer from which we want to compute the representations
            if (return_repr_after_layer_idx is not None) and (layer_idx == len(self.layers)):
                break
        layer_idx += 1

        # Final MSA update
        if (return_repr_after_layer_idx is None) or (layer_idx == len(self.layers)):
            msa_residual = self.final_msa_pwa(
                msa = msa,
                pairwise_repr = pairwise_repr,
                mask = mask
            )
            msa = msa + msa_residual
            del msa_residual
            msa = msa + self.final_msa_transition(msa)
            if layer_idx in return_msa_repr_layer_idx:
                if store_msa_repr_cpu:
                    msa_repr_d[f"layer_{layer_idx}"] = msa[:, :1, :, :].cpu() if query_only else msa.cpu()
                else:
                    msa_repr_d[f"layer_{layer_idx}"] = msa[:, :1, :, :] if query_only else msa
        
        # Organize results
        res = {}
        # Return only pairwise representation
        if return_pairwise_repr_only:
            res['final_pairwise_repr'] = pairwise_repr if not store_pairwise_repr_cpu else pairwise_repr.cpu()
            return res
        # Return all
        if store_msa_repr_cpu:
            res['final_msa_repr'] = msa[:, :1, :, :] if query_only else msa.cpu()
        else:
            res['final_msa_repr'] = msa[:, :1, :, :] if query_only else msa
        res['final_pairwise_repr'] = pairwise_repr if not store_pairwise_repr_cpu else pairwise_repr.cpu()
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
        ),
        relative_position_encoding_kwargs: dict = dict(
            r_max = 32, 
            s_max = 2,
        ),
        contact_layer: int = 15,
        confind_contact_layer: int = 18
    ):
        super().__init__()
        self.dim_pairwise = dim_pairwise
        self.dim_msa = dim_msa
        self.contact_layer = contact_layer
        self.confind_contact_layer = confind_contact_layer

        # Check if cuEquivariance is available
        if cuex_is_available():
            print("Using cuEquivariance for triangle multiplicative update")
        else:
            print("cuEquivariance is not available. Using standard PyTorch implementation for triangle multiplicative update")

        # Relative position encoding
        self.relative_position_encoding = RelativePositionEncoding(
            dim_out = dim_pairwise,
            **relative_position_encoding_kwargs,
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
        self.confind_contact_head = LogisticRegressionContactHead(
            dim_pairwise = dim_pairwise
        )

    @property
    def device(self):
        """Device of the model."""
        return self.core_stack.zero.device

    ###### Load model ######
    @classmethod
    def from_pretrained(
        cls,
        device: torch.device | None = None,
        weights_dir: str | None = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loaded = False
        if weights_dir is not None:
            # If weights have already been saved
            weights_files_l = glob(os.path.join(weights_dir, "*/snapshots/*/*"))
            if (
                any([os.path.basename(p) == "model_cuex.bin" for p in weights_files_l]) and 
                any([os.path.basename(p) == "confind_contact.bin" for p in weights_files_l]) and
                any([os.path.basename(p) == "contact.bin" for p in weights_files_l])
            ):
                main_weights_path = [p for p in weights_files_l if os.path.basename(p) == 'model_cuex.bin'][0]
                checkpoint = torch.load(main_weights_path, weights_only=True, map_location=device)
                confind_contact_path = [p for p in weights_files_l if os.path.basename(p) == 'confind_contact.bin'][0]
                confind_contact_checkpoint = torch.load(confind_contact_path, weights_only=True, map_location=device)
                cb_contact_path = [p for p in weights_files_l  if os.path.basename(p) == 'contact.bin'][0]
                cb_contact_checkpoint = torch.load(cb_contact_path, weights_only=True, map_location=device)
                loaded = True
        if not loaded:
            path = Path(snapshot_download(repo_id="yakiyama/MSA-Pairformer", cache_dir=weights_dir))
            checkpoint = torch.load(path / "model_cuex.bin", weights_only=True, map_location=device)
            confind_contact_checkpoint = torch.load(path / "confind_contact.bin", weights_only=True, map_location=device)
            cb_contact_checkpoint = torch.load(path / "contact.bin", weights_only=True, map_location=device)
        checkpoint.update(confind_contact_checkpoint)
        checkpoint.update(cb_contact_checkpoint)
        model = cls()
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

    # Toggle query-biased attention
    def turn_off_query_biasing(self):
        self.core_stack.turn_off_query_biasing()

    def turn_on_query_biasing(self):
        self.core_stack.turn_on_query_biasing()

    ###### Make predictions / embeddings ######
    def forward(
        self,
        msa: Float['b s n d'],
        mask: Bool['b n'] | None = None,
        msa_mask: Bool['b s'] | None = None,
        full_mask: Bool['b s n'] | None = None,
        pairwise_mask: Bool['b n n'] | None = None,
        seq_weights: Float['b s'] | None = None,
        seq_weights_dict: dict = {},
        return_cb_contacts: bool = True,
        return_confind_contacts: bool = True,
        return_seq_weights: bool = False,
        query_only: bool = True,
        return_pairwise_repr_layer_idx: List[int] | int | None = None,
        return_msa_repr_layer_idx: List[int] | int | None = None,
        complex_chain_break_indices: List[int] | None = None,
        return_repr_after_layer_idx: int | None = None,
        store_msa_repr_cpu: bool = True,
        store_pairwise_repr_cpu: bool = True
    ):
        # Ensure that contact layer is in return_pairwise_repr_layer_idx if returning contacts
        if return_cb_contacts or return_confind_contacts:
            if return_pairwise_repr_layer_idx is None:
                return_pairwise_repr_layer_idx = []
                if return_cb_contacts:
                    return_pairwise_repr_layer_idx.append(self.contact_layer)
                if return_confind_contacts:
                    return_pairwise_repr_layer_idx.append(self.confind_contact_layer)
            elif isinstance(return_pairwise_repr_layer_idx, int):
                return_pairwise_repr_layer_idx = [return_pairwise_repr_layer_idx]
                if return_cb_contacts:
                    return_pairwise_repr_layer_idx.append(self.contact_layer)
                if return_confind_contacts:
                    return_pairwise_repr_layer_idx.append(self.confind_contact_layer)
            elif isinstance(return_pairwise_repr_layer_idx, list):
                if return_cb_contacts and (self.contact_layer not in return_pairwise_repr_layer_idx):
                    return_pairwise_repr_layer_idx.append(self.contact_layer)
                if return_confind_contacts and (self.confind_contact_layer not in return_pairwise_repr_layer_idx):
                    return_pairwise_repr_layer_idx.append(self.confind_contact_layer)

        # Initialize representations
        msa, pairwise_repr = self.init_representations(msa, complex_chain_break_indices)
        # Pass through layers
        if return_cb_contacts or return_confind_contacts:
            store_pairwise_repr_cpu = False
        results = self.core_stack(
            pairwise_repr = pairwise_repr,
            msa = msa,
            mask = mask,
            msa_mask = msa_mask,
            full_mask = full_mask,
            pairwise_mask = pairwise_mask,
            seq_weights = seq_weights,
            seq_weights_dict = seq_weights_dict,
            query_only = query_only,
            return_seq_weights = return_seq_weights,
            return_msa_repr_layer_idx = return_msa_repr_layer_idx,
            return_pairwise_repr_layer_idx = return_pairwise_repr_layer_idx,
            return_repr_after_layer_idx = return_repr_after_layer_idx,
            store_pairwise_repr_cpu = store_pairwise_repr_cpu,
            store_msa_repr_cpu = store_msa_repr_cpu,
        )

        # Get logits via language modeling head
        if store_msa_repr_cpu:
            logits = self.lm_head(results['final_msa_repr'].to(self.device))
        else:
            logits = self.lm_head(results['final_msa_repr'])
        results['logits'] = logits

        # Predict Cb-Cb contacts
        if return_cb_contacts:
            if store_pairwise_repr_cpu:
                contacts = self.contact_head(results['pairwise_repr_d'][f'layer_{self.contact_layer}'].to(self.device))
            else:
                contacts = self.contact_head(results['pairwise_repr_d'][f'layer_{self.contact_layer}'])
            results['predicted_cb_contacts'] = contacts
        # Predict ConFind contacts
        if return_confind_contacts:
            if store_pairwise_repr_cpu:
                confind_contacts = self.confind_contact_head(results['pairwise_repr_d'][f'layer_{self.confind_contact_layer}'].to(self.device))
            else:
                confind_contacts = self.confind_contact_head(results['pairwise_repr_d'][f'layer_{self.confind_contact_layer}'])
            results['predicted_confind_contacts'] = confind_contacts
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
        seq_weights_dict: dict = {},
        complex_chain_break_indices: List[int] | None = None,
        return_seq_weights: bool = False,
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
            seq_weights_dict = seq_weights_dict,
            query_only = True,
            return_repr_after_layer_idx = None,
            return_pairwise_repr_layer_idx = [self.contact_layer, self.confind_contact_layer],
            return_seq_weights = return_seq_weights,
        )

        # Predict contacts
        cb_contacts = self.contact_head(results['pairwise_repr_d'][f'layer_{self.contact_layer}'].to(self.device))
        confind_contacts = self.confind_contact_head(results['pairwise_repr_d'][f'layer_{self.confind_contact_layer}'].to(self.device))
        res = {}
        res['predicted_cb_contacts'] = cb_contacts
        res['predicted_confind_contacts'] = confind_contacts
        if return_seq_weights:
            res['seq_weights_list_d'] = results['seq_weights_list_d']
        return res

    def predict_cb_contacts(
        self,
        msa: Float['b s n d'],
        mask: Bool['b n'] | None = None,
        msa_mask: Bool['b s'] | None = None,
        full_mask: Bool['b s n'] | None = None,
        pairwise_mask: Bool['b n n'] | None = None,
        seq_weights: Float['b s'] | None = None,
        seq_weights_dict: dict = {},
        complex_chain_break_indices: List[int] | None = None,
        return_seq_weights: bool = False,
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
            seq_weights_dict = seq_weights_dict,
            query_only = True,
            return_repr_after_layer_idx = self.contact_layer,
            return_pairwise_repr_layer_idx = [self.contact_layer],
            return_seq_weights = return_seq_weights,
        )

        # Predict contacts
        contacts = self.contact_head(results['pairwise_repr_d'][f'layer_{self.contact_layer}'].to(self.device))
        res = {}
        res['predicted_cb_contacts'] = contacts
        if return_seq_weights:
            res['seq_weights_list_d'] = results['seq_weights_list_d']

        return res

    def predict_confind_contacts(
        self,
        msa: Float['b s n d'],
        mask: Bool['b n'] | None = None,
        msa_mask: Bool['b s'] | None = None,
        full_mask: Bool['b s n'] | None = None,
        pairwise_mask: Bool['b n n'] | None = None,
        seq_weights: Float['b s'] | None = None,
        seq_weights_dict: dict = {},
        complex_chain_break_indices: List[int] | None = None,
        return_seq_weights: bool = False,
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
            seq_weights_dict = seq_weights_dict,
            query_only = True,
            return_repr_after_layer_idx = self.confind_contact_layer,
            return_pairwise_repr_layer_idx = [self.confind_contact_layer],
            return_seq_weights = return_seq_weights,
        )

        # Predict contacts
        contacts = self.confind_contact_head(results['pairwise_repr_d'][f'layer_{self.confind_contact_layer}'].to(self.device))
        res = {}
        res['predicted_confind_contacts'] = contacts
        if return_seq_weights:
            res['seq_weights_list_d'] = results['seq_weights_list_d']

        return res

    def get_pairwise_repr_at_layer(
        self,
        msa: Float['b s n d'],
        return_repr_after_layer_idx: int,
        mask: Bool['b n'] | None = None,
        msa_mask: Bool['b s'] | None = None,
        full_mask: Bool['b s n'] | None = None,
        pairwise_mask: Bool['b n n'] | None = None,
        seq_weights: Float['b s'] | None = None,
        seq_weights_dict: dict = {},
        complex_chain_break_indices: List[int] | None = None,
        store_pairwise_repr_cpu: bool = False
    ):
        # Initialize representations
        msa, pairwise_repr = self.init_representations(msa, complex_chain_break_indices)
        # Pass through layers
        results = self.core_stack(
            msa = msa,
            pairwise_repr = pairwise_repr,
            mask = mask,
            msa_mask = msa_mask,
            full_mask = full_mask,
            pairwise_mask = pairwise_mask,
            seq_weights = seq_weights,
            seq_weights_dict = seq_weights_dict,
            query_only = True,
            return_seq_weights = False,
            return_msa_repr_layer_idx = None,
            return_pairwise_repr_layer_idx = None,
            return_repr_after_layer_idx = return_repr_after_layer_idx,
            store_pairwise_repr_cpu = store_pairwise_repr_cpu,
            store_msa_repr_cpu = False,
            return_pairwise_repr_only = True
        )
        return results['final_pairwise_repr']