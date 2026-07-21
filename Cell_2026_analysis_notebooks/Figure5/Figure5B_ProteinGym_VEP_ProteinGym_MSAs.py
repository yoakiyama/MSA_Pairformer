# Results may differ slightly depending on 
import os, ctypes
from glob import glob
append_ld_library_path = "~/.local/lib/python3.10/site-packages/nvidia/cublas/lib"
LD_LIBRARY_PATH = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = f"{append_ld_library_path}:{LD_LIBRARY_PATH}" if LD_LIBRARY_PATH else append_ld_library_path

cublas_dir = "/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cublas/lib"
for pat in ["libcublas.so*", "libcublasLt.so*", "libcudart.so*"]:
    for lib in sorted(glob(os.path.join(cublas_dir, pat))):
        try:
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            print(f"Warning: could not load {lib}: {e}")

import os
import torch
from torch.amp import autocast

import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from scipy.stats import spearmanr

from MSA_Pairformer.model import MSAPairformer
from MSA_Pairformer.dataset import aa2tok_d
import sys
sys.path.append("utils")
from compute_fitness_utils import * # prepare_msa_inputs, sample_msa, process_msa

# Initialize device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load model
model = MSAPairformer.from_pretrained().to(torch.bfloat16).to(device)
model.eval()
del model.contact_head
del model.confind_contact_head
model = torch.compile(model, dynamic=True)

# File paths
mapping_file_path = "../../data/Figure5_CASP_ProteinGym/ProteinGym/DMS_substitutions.mmseqs_filtered.reannotated.csv"
dms_dir = "../../data/Figure5_CASP_ProteinGym/ProteinGym/DMS_ProteinGym_Substitutions/"
# msa_dir = "../../data/Figure5_CASP_ProteinGym/ProteinGym/logan_proteingym_combined_msas/" # or "../../data/Figure5_CASP_ProteinGym/uniref_proteingym_msas/"
# weights_dir = "../../data/Figure5_CASP_ProteinGym/ProteinGym/logan_proteingym_combined_weights/" # or "../../data/Figure5_CASP_ProteinGym/ProteinGym/uniref_proteingym_weights/"
# msa_path_colname = "Logan_ProteinGym_MSA_filename" # or ProteinGym_MSA_filename
# results_dir = "results/proteingym_logan" # or "results/proteingym_uniref"
msa_dir = "../../data/Figure5_CASP_ProteinGym/ProteinGym/uniref_proteingym_msas/" 
weights_dir = "../../data/Figure5_CASP_ProteinGym/ProteinGym/uniref_proteingym_weights/" # or "../../data/Figure5_CASP_ProteinGym/ProteinGym/uniref_proteingym_weights/"
msa_path_colname = "ProteinGym_MSA_filename" # or ProteinGym_MSA_filename
results_dir = "results/proteingym_uniref" # or "results/proteingym_uniref"


def get_mutated_positions(mutant):
    mut_split = mutant.split(":")
    mutated_pos_l = []
    for mut in mut_split:
        mut_pos = int(mut[1:-1])
        mutated_pos_l.append(mut_pos)
    return mutated_pos_l
def get_alt_indices(mutant):
    mut_split = mutant.split(":")
    alt_indices_l = []
    for mut in mut_split:
        mut_aa = mut[-1]
        alt_indices_l.append(aa2tok_d[mut_aa])
    return alt_indices_l
def get_wt_indices(mutant):
    mut_split = mutant.split(":")
    wt_indices_l = []
    for mut in mut_split:
        wt_aa = mut[0]
        wt_indices_l.append(aa2tok_d[wt_aa])
    return wt_indices_l
def score_mutations(mut_pos_l, alt_idx_l, wt_idx_l, log_probs_t):
    zero_idx_mut_pos_l = [i-1 for i in mut_pos_l]
    return torch.sum(log_probs_t[zero_idx_mut_pos_l, alt_idx_l] - log_probs_t[zero_idx_mut_pos_l, wt_idx_l]).item()

def main():
    for dms_index in range(217):
        # Load metadata file
        mapping_protein_seq_DMS = pd.read_csv(mapping_file_path)
        # Get DMS ID and relevant file paths
        DMS_id = mapping_protein_seq_DMS["DMS_id"][dms_index]
        print(f"DMS ID: {DMS_id}")
        row = mapping_protein_seq_DMS[mapping_protein_seq_DMS["DMS_id"]==DMS_id]
        row = row.iloc[0]
        row = row.replace(np.nan, "")

        # Retrieve sequence and DMS data file path
        wt_sequence = row["target_seq"].upper()
        dms_input = os.path.join(dms_dir, row["DMS_filename"])
        # Retrieve mutation column
        mutant_col = row["DMS_mutant_column"] if "DMS_mutant_column" in mapping_protein_seq_DMS.columns else "mutant"

        # Get MSA file paths and indices
        msa_filename = row[msa_path_colname]
        if msa_filename == "":
            raise ValueError("No MSA found for DMS: "+str(DMS_id))
        msa_path = os.path.join(msa_dir, msa_filename)
        with open(msa_path, "r") as f:
            f.readline()
            query_seq = ""
            while True:
                line = f.readline()
                if line.startswith(">"):
                    break
                query_seq += line.strip()
        query_seq = query_seq.upper()
        start_pos_offset = wt_sequence.index(query_seq)

        # Read DMS data
        df = pd.read_csv(dms_input)
        # Check if the dataframe is empty
        if len(df) == 0:
            raise ValueError("No rows found in the dataframe")
        print(f"df shape: {df.shape}", flush=True)

        # Get all variant positions
        dms_positions = set(df[mutant_col].map(lambda x: re.findall(r'[A-Z](\d+)[A-Z]', x)).explode().astype(int).tolist())
        print(f"Number of DMS positions: {len(dms_positions)}", flush=True)
        print("WT sequence length: ", len(wt_sequence))

        # Skip if already analyzed
        output_file_base = f"{DMS_id}.logan.csv"
        output_file_path = os.path.join(results_dir, output_file_base)
        if os.path.exists(output_file_path):
            print(f"Skipping {DMS_id} because it has already been analyzed")
            continue

        # Compute fitness scores
        # Load MSA
        np.random.seed(42)
        weight_filename = os.path.join(weights_dir, row["weight_file_name"])
        processed_msa = process_msa(
            filename=msa_path,
            weight_filename=weight_filename,
            filter_msa=False,
            hhfilter_min_cov=None,
            hhfilter_max_seq_id=None,
            hhfilter_min_seq_id=None,
            path_to_hhfilter="hhfilter",
            num_cpus=60
        )
        seed = 0
        nSeqs = 4096
        data = [
            sample_msa(
                sampling_strategy="sequence-reweighting",
                filename=msa_path,
                nseq=nSeqs,
                weight_filename=weight_filename,
                processed_msa=processed_msa,
                random_seed=seed,
                num_cpus=62
            )
        ]
        nTokenTypes = len(np.unique(list(aa2tok_d.values())))
        msa_onehot_t, msa_tokenized_t, mask, msa_mask, full_mask, pairwise_mask = prepare_msa_inputs(data)
        (
            msa_onehot_t,
            mask, 
            msa_mask, 
            full_mask, 
            pairwise_mask, 
        ) = (msa_onehot_t.to(device).unsqueeze(0), mask.to(device), msa_mask.to(device), full_mask.to(device), pairwise_mask.to(device))
        print(f"\tMSA depth: {msa_tokenized_t.shape[0]}")

        # For each position, mask the WT residue exactly once and store the log probabilities
        df['mutated_positions'] = df['mutant'].map(get_mutated_positions)
        df['alt_indices'] = df['mutant'].map(get_alt_indices)
        df['wt_indices'] = df['mutant'].map(get_wt_indices)

        all_mut_pos_l = df['mutated_positions'].explode().unique()
        # Tensor to store log probabilities
        log_probs_t = torch.zeros(len(wt_sequence), len(aa2tok_d)-2)
        for mut_pos in tqdm(all_mut_pos_l):
            msa_mut_pos = mut_pos - start_pos_offset
            # Mask mutated position
            wt_idx = msa_onehot_t[0, 0, msa_mut_pos-1, :].argmax()
            msa_onehot_t[0, 0, msa_mut_pos-1, :] = 0
            msa_onehot_t[0, 0, msa_mut_pos-1, aa2tok_d['<mask>']] = 1
            if (msa_onehot_t.shape[2] > 1024) and (msa_onehot_t.shape[1] > 2048): 
                # Longer than 1024 residues and more than 2048 sequences --> Subset to 1024 residues
                msa_lb = max(0, msa_mut_pos-1-512)
                msa_ub = min(msa_onehot_t.shape[2], msa_lb+1024)
            else:
                msa_lb = 0
                msa_ub = msa_onehot_t.shape[2]
            new_mut_idx = msa_mut_pos-1-msa_lb
            with torch.no_grad():
                with autocast(dtype=torch.bfloat16, device_type = device.type):
                    result = model(
                        msa=msa_onehot_t[:, :, msa_lb:msa_ub, :].to(torch.bfloat16),
                        mask=mask[:, msa_lb:msa_ub],
                        msa_mask=msa_mask,
                        full_mask=full_mask[:, :, msa_lb:msa_ub],
                        pairwise_mask=pairwise_mask[:, msa_lb:msa_ub, msa_lb:msa_ub],
                        complex_chain_break_indices=None,
                        return_seq_weights=False,
                        return_pairwise_repr_layer_idx=None,
                        return_msa_repr_layer_idx=None,
                        return_cb_contacts=False,
                        return_confind_contacts=False,
                        query_only=True,
                        store_pairwise_repr_cpu=False,
                        store_msa_repr_cpu=False,
                    )
            log_probs_t[mut_pos-1, :] = result['logits'][0, 0, new_mut_idx, :].log_softmax(dim=-1).float().cpu()
            # Mutate back to WT
            msa_onehot_t[0, 0, msa_mut_pos-1, :] = 0
            msa_onehot_t[0, 0, msa_mut_pos-1, wt_idx] = 1

        df['mutation_score'] = df.apply(lambda x: score_mutations(x['mutated_positions'], x['alt_indices'], x['wt_indices'], log_probs_t), axis=1)
        df.to_csv(output_file_path)
        # Compute Spearman correlation
        dms_score_col = row["DMS_score_column"] if "DMS_score_column" in mapping_protein_seq_DMS.columns else "DMS_score"
        spearman_corr, p_value = spearmanr(df[dms_score_col], df['mutation_score'])
        print(f"DMS ID: {DMS_id}, Spearman correlation: {spearman_corr:.4f} (p-value: {p_value:.4e})")

if __name__ == "__main__":
    main()