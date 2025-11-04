import re
import os
import numpy as np
import tempfile
import subprocess
import pickle
import einx

from glob import glob
from typing import List, Any, Union, Tuple
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq

from scipy.spatial.distance import cdist

import torch
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
from copy import deepcopy

# Amino acid code to character
code2aa_d = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLU": "E",
    "GLN": "Q",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V"
}

# Tokenize amino acids
aa2tok_d = {
    "A": 0, # ALA
    "R": 1, # ARG
    "N": 2, # ASN
    "D": 3, # ASP
    "C": 4, # CYS
    "E": 5, # GLU
    "Q": 6, # GLN
    "G": 7, # GLY
    "H": 8, # HIS
    "I": 9, # ILE
    "L": 10, # LEU
    "K": 11, # LYS
    "M": 12, # MET
    "F": 13, # PHE
    "P": 14, # PRO
    "S": 15, # SER
    "T": 16, # THR
    "W": 17, # TRP
    "Y": 18, # TYR
    "V": 19, # VAL
    "X": 20, # UNK
    "B": 21, # ASP or ASN
    "Z": 22, # GLU or GLN
    "U": 23, # SEC
    "O": 24, # PYL
    "-": 25, # GAP
    "<pad>": 26, # Padded positions
    "<mask>": 27, # Mask
}
tok2aa_d = {aa2tok_d[k]:k for k in aa2tok_d}
nTokenTypes = len(np.unique(list(aa2tok_d.values())))

# ESM token to amino acid
ESM_SEQUENCE_VOCAB = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K",
    "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z",
    "O", ".", "-", "|",
    "<mask>"
]
esm_tok2aa_d = {ind: alph for ind, alph in enumerate(ESM_SEQUENCE_VOCAB)}
esm_aa2tok_d = {alph: ind for ind, alph in enumerate(ESM_SEQUENCE_VOCAB)}
esmtok_to_pairformertok_d = {esm_aa2tok_d[aa]: aa2tok_d[aa] if aa in aa2tok_d else -1 for aa in ESM_SEQUENCE_VOCAB}
esmtok_to_pairformertok_d[esm_aa2tok_d['<unk>']] = aa2tok_d['X'] # Handle unknown tokens with X
# MSA Pairformer was trained without insertions ".", so they are not part of the vocabulary
# Full list of tokens that do not appear in the MSA Pairformer are <cls>, <eos>, <unk>, <null_1>, <.>
# All are mapped to -1 using esmtok_to_pairformertok_d

def convert_tokens_esm2pairformer(batch_tokens: torch.Tensor, device: torch.device = torch.device('cpu')):
    # MSA Transformer uses a CLS token, but MSA Pairformer does not
    # We'll remove the first token to account for this disparity
    return torch.from_numpy(np.vectorize(esmtok_to_pairformertok_d.get)(batch_tokens.cpu().numpy()))[:, :, 1:].to(device)


class MSA:
    def __init__(
        self,
        msa_file_path: Union[str, Path], # Path to MSA file
        max_length: int = 1024, # Maximum length of the MSA (default is 1024)
        max_tokens: int = 1048576, # Maximum number of tokens in the MSA (default is 1024 * 1024)
        max_seqs: int = 1024, # Maximum number of sequences in the MSA (default is 1024)
        diverse_select_method = "hhfilter", # Method to select diverse sequences (default is "hhfilter")
        secondary_filter_method: str = "greedy",  # Options: "greedy" or "random" (default is "greedy")
        random_query: bool = False, # Whether to select a random query sequence (default is to use the first sequence as the query)
        min_query_coverage: float = 0.8, # Minimum query coverage if selecting a random query sequence
        parser_kwargs: dict = {
            "keep_insertions": False,
            "to_upper": False,
            "remove_lowercase_cols": False
        },
        hhfilter_kwargs: dict = {}
    ):
        self.msa_file_path = msa_file_path
        self.max_length = max_length
        self.max_tokens = max_tokens
        self.max_seqs = max_seqs
        self.diverse_select_method = diverse_select_method
        self.secondary_filter_method = secondary_filter_method
        self.random_query = random_query

        assert self.diverse_select_method in ["greedy", "hhfilter", "none"], "Diverse select method must be either 'greedy' or 'hhfilter', or 'none' if no diverse selection is desired"
        assert self.secondary_filter_method in ["greedy", "random", "none"], "Secondary filter method must be either 'greedy' or 'random', or 'none' if no secondary filter is desired"

        # Parse MSA file
        self.seq_l, self.ids_l = self.parse_a3m_file(**parser_kwargs)
        self.seq_a = self.seq_list_to_arr(self.seq_l)
        if self.random_query:
            # Select random sequence index and swap with query sequence
            # Ensure that randomly selected sequence has at least min_query_coverage (e.g. 80%) coverage
            valid_indices = np.where((self.seq_a != '-').sum(axis=-1) >= (self.seq_a.shape[-1] * min_query_coverage))[0]
            if len(valid_indices) > 0:
                random_idx = np.random.choice(valid_indices)
                self.seq_a[[0, random_idx]] = self.seq_a[[random_idx, 0]]
                # Swap ids
                tmp_id = self.ids_l[0]
                self.ids_l[0] = self.ids_l[random_idx]
                self.ids_l[random_idx] = tmp_id

        # Get random crop based on max length
        self.random_crop, self.random_crop_min, self.random_crop_max = self.get_random_crop(self.seq_a, max_length=self.max_length)
        # Select sequences based on max depth (greedy selection or hhfilter)
        # Determine maximum depth based on sequence length and max tokens
        # Currently capping the MSA to some number (max_seqs) of sequences
        # Could instead use min(full_msa_depth, max_tokens // sequence_length)
        sequence_length = self.random_crop.shape[1]
        self.subset_depth = min(self.max_tokens // sequence_length, self.max_seqs)
        if self.diverse_select_method == "none":
            self.select_diverse_msa = self.random_crop
            self.select_diverse_indices = np.arange(self.random_crop.shape[0])
        else:
            self.select_diverse_msa, self.select_diverse_indices = self.select_diverse(
                msa_a=self.random_crop,
                num_seqs=self.max_seqs,
                method=diverse_select_method,
                hhfilter_kwargs=hhfilter_kwargs,
                secondary_filter_method=self.secondary_filter_method
            )
        # Tokenize diverse MSA
        self.diverse_tokenized_msa = self.tokenize_msa(self.select_diverse_msa)
        self.n_diverse_seqs = self.diverse_tokenized_msa.shape[0]

    def compute_pssm(self, msa_a):
        # Get counts excluding padding token (26) and create probability distribution
        nSeqs, nPos = msa_a.shape
        probs = torch.zeros(nPos, aa2tok_d['<pad>'])  # Initialize output tensor [nPos, 26]

        # For each position in sequence
        for i in range(nPos):
            pos_tensor = msa_a[:, i]  # Get all sequences at this position
            mask = pos_tensor != aa2tok_d['<pad>']  # Mask padding tokens
            counts = torch.bincount(pos_tensor[mask], minlength=aa2tok_d['<pad>'])  # Count tokens at this position
            probs[i] = counts.float() / counts.sum()  # Convert to probabilities
        return probs

    def generate_random_sequences(self, probs, nSeqs):
        # Sample k sequences from probability distribution
        sampled_seqs = torch.zeros(nSeqs, probs.shape[0], dtype=torch.long)
        for i in range(probs.shape[0]):
            sampled_seqs[:, i] = torch.multinomial(probs[i], nSeqs, replacement=True)
        return sampled_seqs

    def parse_a3m_file(
        self,
        keep_insertions: bool = False,
        to_upper: bool = False,
        remove_lowercase_cols: bool = False, # Any lowercase columns in the query sequence
        **kwargs
    ):
        """
        Parse sequences from a3m file. 
        Returns list of full length sequences aligned to query sequence (top row)
        keep_insertions determines whether to keep insertions in sequences
        to_upper determines whether to convert sequences to uppercase (unnecessary if removing insertions)
        """
        seq_l = []
        ids_l = []
        valid_indices = None
        for record in SeqIO.parse(self.msa_file_path, "fasta"):
            sequence = str(record.seq)
            if remove_lowercase_cols:
                if valid_indices is None:
                    valid_indices = [i for i, aa in enumerate(sequence) if aa.isupper()]
                sequence = "".join([sequence[i] for i in valid_indices])
            if not keep_insertions:
                sequence = re.sub(r"[a-z]|\.|\*", "", sequence)
            if to_upper:
                sequence = sequence.upper()
            seq_l.append(sequence)
            ids_l.append(record.name)
        return seq_l, ids_l

    @property
    def inverse_covariance(self):
        if not hasattr(self, "_inverse_covariance"):
            # One hot encode
            msa_onehot = one_hot(self.diverse_tokenized_msa, num_classes=nTokenTypes)
            # Get shape
            n, l, a = msa_onehot.shape
            flat_msa = msa_onehot.reshape(n, -1)
            # Compute covariance
            c = torch.cov(flat_msa.T)
            # Inverse covariance
            shrink = 4.5 / np.sqrt(n) * torch.eye(c.shape[0])
            ic = torch.linalg.inv(c + shrink)
            # Sum across amino acid inverse covariances (1-21 in our case)
            ic = ic.reshape(l, a, l, a)[:, 1:21, :, 1:21].sum((1, 3))
            self._inverse_covariance = ic
        return self._inverse_covariance

    def tokenize_msa(self, msa_a):
        return torch.from_numpy(np.vectorize(aa2tok_d.get)(msa_a))

    def seq_list_to_arr(self, seq_l):
        return np.array([list(seq) for seq in seq_l])

    def get_random_crop(self, msa_a, max_length: int = 1024):
        seq_len = msa_a.shape[1]
        if seq_len <= max_length:
            return msa_a, 0, seq_len-1
        start = np.random.randint(0, seq_len - max_length)
        return self.seq_a[:, start:start+max_length], start, start+max_length-1

    def select_diverse(self, msa_a, num_seqs: int, method: str = "hhfilter", hhfilter_kwargs: dict = {}, secondary_filter_method: str = "greedy"):
        assert method in ['greedy', 'hhfilter'], "Method must be either 'greedy' or 'hhfilter'"
        if method == 'greedy':
            return self.greedy_select(msa_a, num_seqs)
        elif method == 'hhfilter':
            hhfilter_kwargs = {**hhfilter_kwargs, "diff": num_seqs}
            filtered_msa, kept_indices = self.hhfilter_select(msa_a, **hhfilter_kwargs)
            # If hhfilter returns more sequences than maximum depth,
            # maximize diversity with maximum depth
            if num_seqs < filtered_msa.shape[0]:
                if secondary_filter_method == "greedy":
                    # Greedily select for maximum MSA diversity
                    filtered_msa, greedy_kept_indices = self.greedy_select(filtered_msa, num_seqs)
                    kept_indices = [kept_indices[i] for i in greedy_kept_indices]
                elif secondary_filter_method == "random":
                    # Randomly select num_seqs sequences (always include query sequence)
                    random_indices = np.concatenate([[0], np.random.choice(np.arange(1, filtered_msa.shape[0]), num_seqs-1, replace=False)])
                    filtered_msa = filtered_msa[random_indices]
                    kept_indices = [kept_indices[i] for i in random_indices]
                elif secondary_filter_method == "none":
                    pass
                else:
                    raise ValueError(f"Secondary filter method must be either 'greedy' or 'random', got {secondary_filter_method}")
            return filtered_msa, kept_indices
        else:
            raise ValueError("Method must be either 'greedy' or 'hhfilter'")

    def greedy_select(self, msa_a, num_seqs: int):
        tokenized_msa = self.tokenize_msa(msa_a)
        # Already below depth threshold
        curr_depth = msa_a.shape[0]
        if curr_depth <= num_seqs:
            return msa_a, np.arange(curr_depth)
        # Greedily maximize diversity
        all_indices = np.arange(curr_depth)
        indices = [0]
        pairwise_distances = np.zeros((0, curr_depth))
        for _ in range(num_seqs - 1):
            dist = cdist(tokenized_msa[indices[-1:]], tokenized_msa, "hamming")
            pairwise_distances = np.concatenate([pairwise_distances, dist])
            shifted_distance = np.delete(pairwise_distances, indices, axis=1).mean(0)
            shifted_index = np.argmax(shifted_distance)
            index = np.delete(all_indices, indices)[shifted_index]
            indices.append(index)
        indices = sorted(indices)
        return msa_a[indices], indices

    def hhfilter_select(
        self,
        msa_a,
        M = "a3m",
        seq_id: int=90,
        diff: int=0, # Number of sequences
        cov: int=70,
        qid: int=30,
        qsc: float=-20.0,
        maxseq: int=None,
        binary="hhfilter",
    ):
        # Get tmp directory from environment
        tmpdir_base = os.environ.get("TMPDIR", "/tmp/")
        with tempfile.TemporaryDirectory(dir=tmpdir_base) as tmpdirname:
            tmpdir = Path(tmpdirname)
            random_prefix = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz0123456789'), size=8))
            fasta_file = tmpdir / f"{random_prefix}.input.fasta"
            fasta_file.write_text(
                "\n".join([f">{i}\n{''.join(seq)}" for i, seq in enumerate(msa_a)])
            )
            output_file = tmpdir / f"{random_prefix}.output.fasta"
            command = " ".join(
                [
                    f"{binary}",
                    f"-i {fasta_file}",
                    f"-M {M}",
                    f"-o {output_file}",
                    f"-id {seq_id}",
                    f"-diff {diff}",
                    f"-cov {cov}",
                    f"-qid {qid}",
                    f"-qsc {qsc}",
                ]
            ).split(" ")
            if maxseq is not None:
                command.append(f"-maxseq {maxseq}")
            result = subprocess.run(command, capture_output=True)
            result.check_returncode()
            with output_file.open() as f:
                indices = [int(line[1:].strip()) for line in f if line.startswith(">")]
            return msa_a[indices], indices
            

class MSADataset(Dataset):
    def __init__(
        self,
        msa_dir=None,
        msa_paths = None,
        max_seq_length = 1024,
        max_msa_depth = 1024,
        max_tokens = 2**17,
        min_depth = 4,
        transform = None,
        random_query: bool = False,
        min_query_coverage: float = 0.8,
        hhfilter_kwargs: dict = {},
        secondary_filter_method: str = "greedy"
    ):
        """
        data_dir is the parent directory that stores all of the MSA .a3m files
        """
        assert (msa_dir is not None) or (msa_paths is not None), "Must provide MSA paths or directory path"
        super().__init__()
        self.msa_dir = msa_dir
        self.transform = transform
        self.max_seq_length = max_seq_length
        self.max_msa_depth = max_msa_depth
        self.max_tokens = max_tokens
        self.min_depth = min_depth
        self.random_query = random_query
        self.min_query_coverage = min_query_coverage
        self.hhfilter_kwargs = hhfilter_kwargs
        self.secondary_filter_method = secondary_filter_method
        if msa_paths is None:
            self.msa_paths = glob(os.path.join(msa_dir, "*/a3m/*.a3m"))
        else:
            self.msa_paths = msa_paths
        
    def __len__(self):
        return len(self.msa_paths)

    def __getitem__(self, idx):
        res_d = {}
        # Get MSA file path
        msa_path = self.msa_paths[idx]
        # Create MSA object
        msa = MSA(
            msa_path,
            max_length = self.max_seq_length,
            max_seqs = self.max_msa_depth,
            max_tokens = self.max_tokens,
            diverse_select_method = "hhfilter",
            random_query = self.random_query,
            min_query_coverage = self.min_query_coverage,
            hhfilter_kwargs = self.hhfilter_kwargs,
            secondary_filter_method = self.secondary_filter_method
        )
        # Get tokenized MSA
        res_d['tokenized_msa'] = msa.diverse_tokenized_msa
        # Add MSA depth
        res_d['n_diverse_seqs'] = msa.n_diverse_seqs
        # Add file path
        res_d['file_path'] = msa_path
        # Add sequence indices
        res_d['seq_indices'] = msa.select_diverse_indices
        return res_d     

def msa_mlm(
    msa_t: torch.tensor, 
    mask_tok: int = aa2tok_d['<mask>'],
    pad_tok: int = aa2tok_d['<pad>'],
    mask_prob: float = 0.15,
    mask_ratio: float = 0.8,
    mutate_ratio: float = 0.1,
    keep_ratio: float = 0.1,
    mutate_pssm: bool = True,
    mutate_tok_low: int = 0,
    mutate_tok_high: int = 19,
    query_only: bool = False
):
    # Create masked input (don't mask padding tokens)
    masked_msas = msa_t.clone()
    flat_msas = masked_msas.view(-1)
    if query_only:
        nMSAs, depth, length = masked_msas.shape
        intervals_t = torch.arange(nMSAs).reshape(nMSAs, 1) * depth * length
        non_pad_indices = (torch.arange(length).unsqueeze(0).repeat(nMSAs, 1) + intervals_t).flatten()
        non_pad_indices = non_pad_indices[flat_msas[non_pad_indices] != pad_tok]
    else:
        non_pad_indices = torch.nonzero(flat_msas != pad_tok, as_tuple=False).view(-1)
    
    # Calculate the number of positions to mask
    num_mask = int(len(non_pad_indices) * mask_prob)

    # Generate mask indices
    mlm_indices = np.random.choice(non_pad_indices.numpy(), num_mask, replace=False)
    mask_ub = int(mask_ratio * num_mask)
    mask_indices = mlm_indices[:mask_ub]
    mutate_ub = mask_ub + int(mutate_ratio * num_mask)
    mutate_indices = mlm_indices[mask_ub : mutate_ub]
    keep_indices = mlm_indices[mutate_ub:]

    # Apply MLM (masking and mutating)
    masked_msas.view(-1)[mask_indices] = torch.tensor(mask_tok)
    if mutate_pssm:
        # Get column indices of mutate_indices
        batch_indices, seq_indices, pos_indices = torch.unravel_index(torch.tensor(mutate_indices), msa_t.shape)
        # Compute PSSM of mutate_indices (exclude padding tokens)
        counts_t = torch.nn.functional.one_hot(msa_t, num_classes=msa_t.shape[-1]).sum(dim=1)[:, :, :26]
        pssm = counts_t / counts_t.sum(dim=-1, keepdim=True)
        probs_t = pssm[batch_indices, pos_indices]
        probs_t = probs_t[mutate_indices]
        new_toks_t = torch.multinomial(probs_t, num_samples=1)
        # new_toks_t = torch.stack([torch.multinomial(probs_t[i], num_samples=1) for i in range(len(mutate_indices))]).squeeze()
        # Apply mutations
        masked_msas.view(-1)[mutate_indices] = new_toks_t
    else:
        masked_msas.view(-1)[mutate_indices] = torch.randint_like(input = masked_msas.view(-1)[mutate_indices], low=mutate_tok_low, high=mutate_tok_high+1)

    # Return masked MSA and indices of tokens to predict
    return masked_msas, mlm_indices

class CollateAFBatch():
    def __init__(
        self,
        max_seq_length,
        max_seq_depth,
        min_seq_depth,
        pad_tok=aa2tok_d['<pad>'],
        mask_tok=aa2tok_d['<mask>'], 
        mask_prob=0.15,
        mask_ratio=0.8,
        mutate_ratio=0.1,
        keep_ratio=0.1,
        tok_low=0,
        tok_high=25,
        query_only=False,
        mutate_pssm=False,
    ):
        self.max_seq_depth = max_seq_depth
        self.max_seq_length = max_seq_length
        self.min_seq_depth = min_seq_depth
        self.pad_tok = pad_tok
        self.mask_tok = mask_tok
        self.mask_prob = mask_prob
        self.mask_ratio = mask_ratio
        self.mutate_ratio = mutate_ratio
        self.keep_ratio = keep_ratio
        self.tok_low = tok_low
        self.tok_high = tok_high
        self.mutate_pssm = mutate_pssm
        self.query_only = query_only

    def __call__(self, batch):
        # Initialize output dictionary
        output_dict = {}

        # Skip MSAs with too few sequences
        valid_msa = [b['n_diverse_seqs'] >= self.min_seq_depth for b in batch]
        if not any(valid_msa):
            return None
        # Get batch size
        og_batch_size = len(batch)
        effective_batch_size = sum(valid_msa)
        
        # Match sequence lengths and stack tensors
        tokenized_msa_l = [batch[i]['tokenized_msa'] for i in range(og_batch_size) if valid_msa[i]]
        # shape = [batch_size] + np.max([seq.shape for seq in tokenized_msa_l], 0).tolist()
        shape = [effective_batch_size, self.max_seq_depth, self.max_seq_length]
        dtype = tokenized_msa_l[0].dtype
        if isinstance(tokenized_msa_l[0], np.ndarray):
            msas = np.full(shape, self.pad_tok, dtype=dtype)
        elif isinstance(tokenized_msa_l[0], torch.Tensor):
            msas = torch.full(shape, self.pad_tok, dtype=dtype)
        for msa, seq in zip(msas, tokenized_msa_l):
            msaslice = tuple(slice(dim) for dim in seq.shape)
            msa[msaslice] = seq
        output_dict['msas'] = msas
    
        # Create masked input
        if self.mask_prob > 0:
            masked_msas, mlm_indices = msa_mlm(
                msas, mask_tok=self.mask_tok, pad_tok=self.pad_tok, mask_prob=self.mask_prob, mask_ratio=self.mask_ratio, 
                mutate_ratio=self.mutate_ratio, keep_ratio=self.keep_ratio, mutate_tok_low=self.tok_low, 
                mutate_tok_high=self.tok_high, query_only=self.query_only, mutate_pssm=self.mutate_pssm
            )
        else:
            masked_msas = deepcopy(msas)
            mlm_indices = None
        # One-hot encode masked/mutated MSA
        masked_msas_onehot = one_hot(masked_msas, num_classes = nTokenTypes)
        output_dict['msas_onehot'] = masked_msas_onehot
        if self.query_only:
            if mlm_indices is None:
                output_dict['masked_idx'] = None
            else:
                _, _, n_pos = msas.shape
                batch_idx, seq_idx, pos_idx = np.unravel_index(mlm_indices, msas.shape)
                query_only_pos_idx = pos_idx + batch_idx * n_pos
                output_dict['masked_idx'] = query_only_pos_idx
        else:
            output_dict['masked_idx'] = mlm_indices
        output_dict['unmasked_msas_onehot'] = one_hot(msas, num_classes = nTokenTypes)
    
        # Initialize additional molecule features
        molecule_feats = prep_molecule_feats(output_dict['msas_onehot'])
        output_dict['molecule_feats'] = molecule_feats
        # Store file path
        output_dict['file_path'] = [batch[i]['file_path'] for i in range(og_batch_size) if valid_msa[i]]
        # Store sequence indices
        output_dict['seq_indices'] = [batch[i]['seq_indices'] for i in range(og_batch_size) if valid_msa[i]]
        # Store MSA depth
        output_dict['msa_depths'] = torch.tensor([batch[i]['n_diverse_seqs'] for i in range(og_batch_size) if valid_msa[i]])
        # Prepare masks
        mask, msa_mask, full_mask, pairwise_mask = prepare_msa_masks(output_dict['msas'])
        output_dict['mask'] = mask
        output_dict['msa_mask'] = msa_mask
        output_dict['full_mask'] = full_mask
        output_dict['pairwise_mask'] = pairwise_mask
        return output_dict

def prep_molecule_feats(msa_input):
    batch_size = msa_input.shape[0]
    seq_len = msa_input.shape[2]
    molecule_feats = torch.stack([
        torch.ones(size=(batch_size, seq_len)), # molecule_idx
        torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1), # token_index
        torch.zeros(size=(batch_size, seq_len)), # molecule_idx
        torch.zeros(size=(batch_size, seq_len)),# entity_id,
        torch.zeros(size=(batch_size, seq_len)), # sym_id
    ], dim=-1).reshape((batch_size, seq_len, 5))
    return molecule_feats

def get_relative_positions(msa_input):
    batch_size = msa_input.shape[0]
    seq_len = msa_input.shape[2]
    token_indices = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    return token_indices

def onehot_msa(msa_input, device=torch.device('cpu')):
    return torch.nn.functional.one_hot(msa_input, num_classes=len(aa2tok_d)).to(device)

def prepare_msa_masks(msa_input, device=torch.device('cpu')):
    # msa_input is of shape [b, s, n]
    mask = (msa_input != aa2tok_d['<pad>']).any(dim=1) # [b, n]
    msa_mask = (msa_input != aa2tok_d['<pad>']).any(dim=2) # [b, s]
    full_mask = (msa_input != aa2tok_d['<pad>']) # [b, s, n]
    pairwise_mask = einx.logical_and("... i, ... j -> ... i j", mask, mask) # [b, n, n]
    return mask.to(device), msa_mask.to(device), full_mask.to(device), pairwise_mask.to(device)

def prepare_inputs(batch, device):
    msa_repr = batch['msas_onehot'].float().to(device)
    mask, msa_mask, full_mask, pairwise_mask = prepare_msa_masks(batch['msas'])
    msa_mask = msa_mask.to(device)
    mask = mask.to(device)
    full_mask = full_mask.to(device)
    pairwise_mask = pairwise_mask.to(device)
    molecule_feats = batch['molecule_feats'].to(device)
    return msa_repr, mask, msa_mask, full_mask, pairwise_mask, molecule_feats

def prepare_inputs_bf16(batch, device):
    msa_repr = batch['msas_onehot'].bfloat16().to(device)
    mask, msa_mask, full_mask, pairwise_mask = prepare_msa_masks(batch['msas'])
    msa_mask = msa_mask.to(device)
    mask = mask.to(device)
    full_mask = full_mask.to(device)
    pairwise_mask = pairwise_mask.to(device)
    molecule_feats = batch['molecule_feats'].bfloat16().to(device)
    return msa_repr, mask, msa_mask, full_mask, pairwise_mask, molecule_feats

def create_msa_subset_mask(subset_msa_idx_l, L):
    b = len(subset_msa_idx_l)
    mask = torch.zeros(b, L, L)
    for i in range(b):
        indices = torch.tensor(subset_msa_idx_l[i])
        row_mask = torch.zeros(L, dtype=torch.bool)
        row_mask[indices] = True
        mask[i] = row_mask.unsqueeze(0) & row_mask.unsqueeze(1)
    return mask > 0


class trRosettaContactMSADataset(Dataset):
    def __init__(
        self,
        paired_paths_l: List[Tuple[Union[str, Path], Union[str, Path]]],
        max_seq_length: int = 1024,
        max_msa_depth: int = 1024,
        min_msa_depth: int = 8,
        max_tokens: int = 2**17,
        random_query: bool = False,
        min_query_coverage: float = 0.9,
        hhfilter_kwargs: dict = {},
        secondary_filter_method: str = "greedy"
    ):
        self.paired_paths_l = paired_paths_l
        self.max_seq_length = max_seq_length
        self.max_msa_depth = max_msa_depth
        self.max_tokens = max_tokens
        self.random_query = random_query
        self.min_query_coverage = min_query_coverage
        self.hhfilter_kwargs = hhfilter_kwargs
        self.secondary_filter_method = secondary_filter_method

    def __len__(self):
        return len(self.paired_paths_l)

    def __getitem__(self, idx):
        # Get MSA file path
        msa_path, npz_file_path = self.paired_paths_l[idx]
        # Create MSA mobject
        msa = MSA(
            msa_path,
            max_length = self.max_seq_length,
            max_seqs = self.max_msa_depth,
            max_tokens = self.max_tokens,
            diverse_select_method = "hhfilter",
            random_query = self.random_query,
            min_query_coverage = self.min_query_coverage
        )
        res_d = {}
        # Get tokenized MSA
        res_d['tokenized_msa'] = msa.diverse_tokenized_msa
        res_d['n_diverse_seqs'] = msa.n_diverse_seqs
        # Get contact map
        npz_obj = np.load(npz_file_path)
        res_d['contact_map'] = torch.tensor((npz_obj['dist6d'] > 0) & (npz_obj['dist6d'] < 8))
        # Add file path
        res_d['msa_file_path'] = msa_path
        res_d['npz_file_path'] = npz_file_path
        # Get random crop bounds
        res_d['msa_crop_bounds'] = (msa.random_crop_min, msa.random_crop_max)
        return res_d

class CollatetrRosettaContactMSABatch():
    def __init__(
        self, 
        max_seq_length: int, 
        max_seq_depth: int,
        min_seq_depth: int,
        pad_tok: int = aa2tok_d['<pad>']
    ):
        self.max_seq_depth = max_seq_depth
        self.max_seq_length = max_seq_length
        self.min_seq_depth = min_seq_depth
        self.pad_tok = pad_tok

    def __call__(self, batch):
        # Initialize output dictionary
        output_dict = {}

        # Skip MSAs with too few sequences
        valid_msa = [b['n_diverse_seqs'] >= self.min_seq_depth for b in batch]
        if not any(valid_msa):
            return None
    
        # Get batch size
        og_batch_size = len(batch)
        effective_batch_size = sum(valid_msa)
        
        # Match sequence lengths and stack tensors
        tokenized_msa_l = [batch[i]['tokenized_msa'] for i in range(og_batch_size) if valid_msa[i]]
        shape = [effective_batch_size, self.max_seq_depth, self.max_seq_length]
        dtype = tokenized_msa_l[0].dtype
        if isinstance(tokenized_msa_l[0], np.ndarray):
            msas = np.full(shape, self.pad_tok, dtype=dtype)
        elif isinstance(tokenized_msa_l[0], torch.Tensor):
            msas = torch.full(shape, self.pad_tok, dtype=dtype)
        for msa, seq in zip(msas, tokenized_msa_l):
            msaslice = tuple(slice(dim) for dim in seq.shape)
            msa[msaslice] = seq
        output_dict['msas'] = msas
        # Match lengths of contact maps
        contact_map_l = [batch[i]['contact_map'].float() for i in range(og_batch_size) if valid_msa[i]]
        msa_crop_bounds_l = [batch[i]['msa_crop_bounds'] for i in range(og_batch_size) if valid_msa[i]]
        contact_map_l = [contact_map_l[i][msa_crop_bounds_l[i][0]:msa_crop_bounds_l[i][1], msa_crop_bounds_l[i][0]:msa_crop_bounds_l[i][1]] for i in range(len(contact_map_l))]
        shape = [effective_batch_size, shape[-1], shape[-1]]
        dtype = contact_map_l[0].dtype
        if isinstance(contact_map_l[0], np.ndarray):
            contacts = np.full(shape, -1, dtype=dtype)
        elif isinstance(contact_map_l[0], torch.Tensor):
            contacts = torch.full(shape, -1, dtype=dtype)
        for i, contact_map in enumerate(contact_map_l):
            contacts[i, :contact_map.shape[0], :contact_map.shape[1]] = contact_map
        
        output_dict['contact_map'] = contacts

        # One-hot encode MSA
        msas_onehot = one_hot(msas, num_classes = nTokenTypes)
        output_dict['msas_onehot'] = msas_onehot
        # Initialize additional molecule features
        molecule_feats = prep_molecule_feats(output_dict['msas_onehot'])
        output_dict['molecule_feats'] = molecule_feats
        # Store file path
        output_dict['msa_file_path'] = [batch[i]['msa_file_path'] for i in range(og_batch_size) if valid_msa[i]]
        output_dict['npz_file_path'] = [batch[i]['npz_file_path'] for i in range(og_batch_size) if valid_msa[i]]
        # Prepare masks
        mask, msa_mask, full_mask, pairwise_mask = prepare_msa_masks(output_dict['msas'])
        output_dict['mask'] = mask
        output_dict['msa_mask'] = msa_mask
        output_dict['full_mask'] = full_mask
        output_dict['pairwise_mask'] = pairwise_mask
        return output_dict