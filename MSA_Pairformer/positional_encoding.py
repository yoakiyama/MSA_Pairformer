import torch
import torch.nn.functional as F
from torch.nn import Module
from typing import List
import einx
from einops import pack
from .core import LinearNoBias
from .custom_typing import (
    Float
)

# Molecule features
"""
molecule_feats: [*, 5]:
0: molecule_index
1: token_index
2: asym_id
3: entity_id
4: sym_id
"""
MOLECULE_FEATS = 5
    
# class RelativePositionEncoding(Module):
#     def __init__(
#         self,
#         r_max = 32,
#         s_max = 2,
#         dim_out = 128
#     ):
#         super().__init__()
#         self.r_max = r_max
#         self.s_max = s_max

#         dim_input = (2*r_max + 2) + (2*r_max + 2) + 1 + (2*s_max + 2)
#         self.out_embedder = LinearNoBias(dim_input, dim_out)

#     def forward(
#         self,
#         molecule_feats: Float[f"b n {MOLECULE_FEATS}"]
#     ) -> Float['b n n dp']:
#         device = molecule_feats.device

#         # Get relevant IDs for each residue stored in the first 5 fields
#         res_idx, token_idx, asym_id, entity_id, sym_id = molecule_feats.unbind(dim = -1)
        
#         # Compute residue, token, and chain distances
#         diff_res_idx = einx.subtract('b i, b j -> b i j', res_idx, res_idx)
#         diff_token_idx = einx.subtract('b i, b j -> b i j', token_idx, token_idx)
#         diff_sym_id = einx.subtract('b i, b j -> b i j', sym_id, sym_id)
#         # Mask for same residue, chain, and entity
#         mask_same_chain = einx.subtract('b i, b j -> b i j', asym_id, asym_id) == 0
#         mask_same_entity = einx.subtract('b i, b j -> b i j 1', entity_id, entity_id) == 0
#         # Compute clipped distances
#         d_res = torch.clip(diff_res_idx + self.r_max, 0, 2 * self.r_max)
#         d_token = torch.clip(diff_token_idx + self.r_max, 0, 2 * self.r_max)
#         d_chain = torch.where(
#             ~mask_same_chain,
#             torch.clip(diff_sym_id + self.s_max, 0, 2 * self.s_max),
#             2 * self.s_max + 1
#         )

#         # One-hot encode distances
#         # bins will be single-offset distances
#         def onehot(x, bins):
#             dist_from_bins = einx.subtract('... i, j -> ... i j', x, bins)
#             indices = dist_from_bins.abs().min(dim = -1, keepdim=True).indices
#             one_hots = F.one_hot(indices.long(), num_classes = len(bins))
#             return one_hots
        
#         # Define bins
#         r_arange = torch.arange(2*self.r_max + 2, device=device)
#         s_arange = torch.arange(2*self.s_max + 2, device=device)
#         # Assign 1-hot encoding of distances
#         mf_dtype = molecule_feats.dtype
#         a_rel_pos = onehot(d_res, r_arange).type(mf_dtype)
#         a_rel_token = onehot(d_token, r_arange).type(mf_dtype)
#         a_rel_chain = onehot(d_chain, s_arange).type(mf_dtype)
#         # Concatenate tensors and project
#         out, _ = pack((a_rel_pos, a_rel_token, mask_same_entity, a_rel_chain), 'b i j *')
#         return self.out_embedder(out)

class RelativePositionEncoding(Module):
    """
    Clipped relative positional encoding to initialize the pair representation
    Some of this is actually a bit unnecessary, but was initially framed to support multi-chain MSAs
    The current MSA Pairformer release does not actually support multi-chain MSAs, but this is left here for future use
    """
    def __init__(
        self,
        r_max = 32,
        s_max = 2,
        dim_out = 128
    ):
        super().__init__()
        self.r_max = r_max
        self.s_max = s_max

        dim_input = (2*r_max + 2) + (2*r_max + 2) + 1 + (2*s_max + 2)
        self.out_embedder = LinearNoBias(dim_input, dim_out)

    def forward(
        self,
        batch_size: int, 
        seq_len: int,
        device: torch.device,
        complex_chain_break_indices: List[int] | None = None
    ) -> Float['b n n dp']:

        # One-hot encode distances
        # bins will be single-offset distances
        def onehot(x, bins):
            dist_from_bins = einx.subtract('... i, j -> ... i j', x, bins)
            indices = dist_from_bins.abs().min(dim = -1, keepdim=True).indices
            one_hots = F.one_hot(indices.long(), num_classes = len(bins))
            return one_hots

        # Initialize token indices
        token_idx = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)
        if complex_chain_break_indices is not None:
            for batch_idx, chain_break_indices in enumerate(complex_chain_break_indices):
                if len(chain_break_indices) > 0:
                    for chain_break_idx in chain_break_indices:
                        token_idx[batch_idx, chain_break_idx:] += 100
        # Compute residue, token, and chain distances
        diff_token_idx = einx.subtract('b i, b j -> b i j', token_idx, token_idx)
        # Mask for same residue, chain, and entity
        mask_same_entity = torch.ones((batch_size, seq_len, seq_len, 1)).to(device)
        # Compute clipped distances
        d_token = torch.clip(diff_token_idx + self.r_max, 0, 2 * self.r_max).to(device)
        d_res = torch.full((batch_size, seq_len, seq_len), self.r_max, device=device)
        d_chain = torch.full((batch_size, seq_len, seq_len), 2*self.s_max+1, device=device)
        # Define bins
        r_arange = torch.arange(2*self.r_max + 2, device=device)
        s_arange = torch.arange(2*self.s_max + 2, device=device)
        # Assign 1-hot encoding of distances
        a_rel_pos = onehot(d_res, r_arange)
        a_rel_token = onehot(d_token, r_arange)
        a_rel_chain = onehot(d_chain, s_arange)
        # Concatenate tensors and project
        out, _ = pack((a_rel_pos, a_rel_token, mask_same_entity, a_rel_chain), 'b i j *')
        
        return self.out_embedder(out)