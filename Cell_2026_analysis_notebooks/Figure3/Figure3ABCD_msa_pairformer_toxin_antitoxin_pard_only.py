# Set LD_LIBRARY_PATH for cuequivariance_ops
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

import torch

import pandas as pd
import numpy as np

import os
from tqdm import tqdm
from glob import glob
from copy import deepcopy

from MSA_Pairformer.model import MSAPairformer
from MSA_Pairformer.dataset import aa2tok_d, prepare_msa_masks

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('high')
print(f"Using device: {torch.cuda.get_device_name(device)}")

# File paths
parde_alignment_path = "../../data/Figure3_toxin_antitoxin/ParD_only_msa.a3m"
fitness_repA_path = "../../data/Figure3_toxin_antitoxin/Library_fitness_vs_parE3_replicate_A.csv"
fitness_repB_path = "../../data/Figure3_toxin_antitoxin/Library_fitness_vs_parE3_replicate_B.csv"

# Load model
model = MSAPairformer.from_pretrained(device=device)
model.eval()
model = torch.compile(model, dynamic=True)


def main():
    # Load fitness data
    fitness_repA = pd.read_csv(fitness_repA_path, index_col=None, header=None)
    fitness_repB = pd.read_csv(fitness_repB_path, index_col=None, header=None)

    # Load WT sequence and chain break index
    with open(parde_alignment_path, "r") as oFile:
        query_seq = oFile.readlines()[1].strip()
    print("Query sequence:", query_seq)
    pard_seq = "MANVEKMSVAVTPQQAAVMREAVEAGEYATASEIVREAVRDWLAKRELRHDDIRRLRQLWDEGKASGRPEPVDFDALRKEARQKLTEVPPNGR"
    pare_seq = "MAVRLVWSPTAKADLIDIYVMIGSENIRAADRYYDQLEARALQLADQPRMGVRRPDIRPSARMLVEAPFVLLYETVPDTDDGPVEWVEIVRVVDGRRDLNRLF"
    # chain_break_idx = query_seq.index(pare_seq)
    # print(f"ParD sequence: {pard_seq}")
    # print(f"ParE sequence: {pare_seq}")
    # print(f"Chain break index: {chain_break_idx}")
    # assert query_seq[chain_break_idx:] == pare_seq

    mutated_indices_l = [58, 59, 60, 63]
    print(f"Mutated indices: {mutated_indices_l}")
    print(f"Amino acids at mutated indices:")
    for mut_idx in mutated_indices_l:
        print(f"\tPosition {mut_idx}: {query_seq[mut_idx]}")

    # Load MSA and tokenize
    with open(parde_alignment_path, "r") as oFile:
        msa_str_l = oFile.readlines()[1::2]
    msa_tokenized_t = np.array([list(seq.strip()) for seq in msa_str_l])
    msa_tokenized_t = np.vectorize(aa2tok_d.get)(msa_tokenized_t)
    msa_tokenized_t = torch.from_numpy(msa_tokenized_t)

    # Prepare input and masks
    msa_onehot_t = torch.nn.functional.one_hot(msa_tokenized_t, num_classes=len(aa2tok_d)).unsqueeze(0)
    mask, msa_mask, full_mask, pairwise_mask = prepare_msa_masks(msa_tokenized_t.unsqueeze(0))
    mask, msa_mask, full_mask, pairwise_mask = mask.to(device), msa_mask.to(device), full_mask.to(device), pairwise_mask.to(device)
    # Expand masks to match the expanded batch dimension (from [1,...] to [4,...])
    mask = mask.expand(4, *mask.shape[1:])
    msa_mask = msa_mask.expand(4, *msa_mask.shape[1:])
    full_mask = full_mask.expand(4, *full_mask.shape[1:])
    pairwise_mask = pairwise_mask.expand(4, *pairwise_mask.shape[1:])

    pseudolikelihood_l = []
    avg_fitness_l = []
    for (mut_a, rep_a), (mut_b, rep_b) in tqdm(zip(fitness_repA.values, fitness_repB.values), total=fitness_repA.shape[0]):
        mut_pos_msa_t = []
        for i in range(4):
            mask_pos = mutated_indices_l[i]
            tmp_msa_t = deepcopy(msa_tokenized_t)
            tmp_msa_t[0, mask_pos] = aa2tok_d['<mask>']
            for j in range(4):
                if j != i:
                    tmp_msa_t[0, mutated_indices_l[j]] = aa2tok_d[mut_a[j]]
            mut_pos_msa_t.append(tmp_msa_t)
        mut_pos_msa_t = torch.stack(mut_pos_msa_t)
        # One-hot encode the MSAs
        msa_onehot_t = torch.nn.functional.one_hot(mut_pos_msa_t, num_classes=len(aa2tok_d)).bfloat16().to(device)
        with torch.amp.autocast(dtype=torch.bfloat16, device_type='cuda'):
            with torch.no_grad():
                res = model(
                    msa = msa_onehot_t,
                    mask = mask,
                    msa_mask = msa_mask,
                    full_mask = full_mask,
                    pairwise_mask = pairwise_mask,
                    complex_chain_break_indices = None,
                    return_seq_weights = False,
                    return_cb_contacts = False,
                    return_confind_contacts = False
                )
        pseudolikelihood = np.sum(
            [torch.log_softmax(res['logits'][mut_idx, 0, mutated_indices_l[mut_idx]], dim=0)[aa2tok_d[mut_a[mut_idx]]].item() for mut_idx in range(4)]
        )
        pseudolikelihood_l.append(pseudolikelihood)
        avg_fitness_l.append(np.mean(np.array([rep_a, rep_b])))
        np.save("results/pseudolikelihood_l.msa_pairformer.qid30_id90.1024.pard_only.npy", np.array(pseudolikelihood_l))

if __name__ == "__main__":
    main()


