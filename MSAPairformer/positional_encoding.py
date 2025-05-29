import torch
import torch.nn.functional as F
from torch.nn import Module, Linear, Embedding

import einx
from einops import rearrange, pack
from MSAPairformer.core import LinearNoBias
from MSAPairformer.custom_typing import (
    Float
)

# Molecule features
"""
additional_molecule_feats: [*, 9]:
0: molecule_index
1: token_index
2: asym_id
3: entity_id
4: sym_id
5: is_protein
6: is_rna
7: is_dna
8: is_ligand
"""
ADDITIONAL_MOLECULE_FEATS = 9

class FourierEmbedding(Module):
    """
    Fourier embedding
    Randomly genereate weight/bias from Gaussian once before training
    w, b ~ N(0, Ic); w, b in Rc
    w, b are not learned! 
    """
    def __init__(self, dim):
        super().__init__()
        self.proj = Linear(1, dim)
        self.proj.requires_grad_(False)
        
    def forward(
        self, 
        times #: Float['b']
    ) -> Float['b d']:
        # Reshape times tensor from (b, ) to (b, 1)
        # Returns (b, d) tensor
        times = rearrange(times, 'b -> b 1')
        random_proj = self.proj(times)
        return torch.cos(2 * pi * random_proj)
    
class RelativePositionEncoding(Module):
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
        additional_molecule_feats: Float[f"b n {ADDITIONAL_MOLECULE_FEATS}"]
    ) -> Float['b n n dp']:
        device = additional_molecule_feats.device

        # Get relevant IDs for each residue stored in the first 5 fields
        res_idx, token_idx, asym_id, entity_id, sym_id = additional_molecule_feats.unbind(dim = -1)
        
        # Compute residue, token, and chain distances
        # Part of lines 4, 6, 8 in Algorithm 3
        diff_res_idx = einx.subtract('b i, b j -> b i j', res_idx, res_idx)
        diff_token_idx = einx.subtract('b i, b j -> b i j', token_idx, token_idx)
        diff_sym_id = einx.subtract('b i, b j -> b i j', sym_id, sym_id)
        # Mask for same residue, chain, and entity
        # Lines 1-3 in Algorithm 3
        mask_same_chain = einx.subtract('b i, b j -> b i j', asym_id, asym_id) == 0
        mask_same_res = diff_res_idx == 0
        mask_same_entity = einx.subtract('b i, b j -> b i j 1', entity_id, entity_id) == 0
        # Compute clipped distances
        # Part of lines 4, 6, and 8 from Algorithm 3
        d_res = torch.where(
            mask_same_chain,
            torch.clip(diff_res_idx + self.r_max, 0, 2 * self.r_max),
            2 * self.r_max + 1
        )
        d_token = torch.where(
            mask_same_chain * mask_same_res,
            torch.clip(diff_token_idx + self.r_max, 0, 2 * self.r_max),
            2 * self.r_max + 1
        )
        d_chain = torch.where(
            ~mask_same_chain,
            torch.clip(diff_sym_id + self.s_max, 0, 2 * self.s_max),
            2 * self.s_max + 1
        )

        # One-hot encode distances
        # bins will be single-offset distances
        def onehot(x, bins):
            dist_from_bins = einx.subtract('... i, j -> ... i j', x, bins)
            indices = dist_from_bins.abs().min(dim = -1, keepdim=True).indices
            one_hots = F.one_hot(indices.long(), num_classes = len(bins))
            return one_hots
        # Define bins
        r_arange = torch.arange(2*self.r_max + 2, device=device)
        s_arange = torch.arange(2*self.s_max + 2, device=device)
        # Assign 1-hot encoding of distances (lines 5, 7, and 9 of Algorithm 1)
        amf_dtype = additional_molecule_feats.dtype
        a_rel_pos = onehot(d_res, r_arange).type(amf_dtype)
        a_rel_token = onehot(d_token, r_arange).type(amf_dtype)
        a_rel_chain = onehot(d_chain, s_arange).type(amf_dtype)
        # Create position encoding (Line p
        # Concatenate tensors and pass through LinearNoBias
        out, _ = pack((a_rel_pos, a_rel_token, mask_same_entity, a_rel_chain), 'b i j *')
        return self.out_embedder(out)

class BasicRelativePositionEncoding(Module):
    def __init__(
        self,
        r_max = 32,
        dim_out = 128
    ):
        super().__init__()
        self.r_max = r_max

        # dim_input = 2*r_max + 1
        # self.out_embedder = LinearNoBias(dim_input, dim_out)
        dim_input = 2 * r_max + 1
        self.out_embedder = Embedding(dim_input, dim_out)

    def forward(
        self,
        token_indices: Float['b n']
    ) -> Float['b n n dp']:
        device = token_indices.device
        
        # Compute token distances
        diff_token_idx = einx.subtract('b i, b j -> b i j', token_indices, token_indices)
        # Compute clipped distances
        # Shift to get values from 0 to 2*r_max
        d_token = torch.clip(diff_token_idx + self.r_max, 0, 2*self.r_max)
        # One-hot encode distances
        # token_rel_pos = F.one_hot(d_token.long(), num_classes=2*self.r_max+1)
        return self.out_embedder(d_token.long())