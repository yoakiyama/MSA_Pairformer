import torch

import pandas as pd
import numpy as np
from tqdm import tqdm
from copy import deepcopy

import esm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('high')
print(f"Using device: {torch.cuda.get_device_name(device)}")

# Load model
model_path = "/home/ubuntu/esm_models/esm_msa1b_t12_100M_UR50S.pt"
msa_transformer, msa_transformer_alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
msa_transformer = msa_transformer.eval().to(torch.float16).cuda()
msa_transformer = torch.compile(msa_transformer, dynamic=True)
msa_transformer_batch_converter = msa_transformer_alphabet.get_batch_converter()
total_params = sum(p.numel() for p in msa_transformer.parameters())
print(f"Total number of parameters: {total_params:,}")

# File paths
parde_alignment_path = "../../data/Figure3_toxin_antitoxin/ParDE_hhfilter.a3m"
fitness_repA_path = "../../data/Figure3_toxin_antitoxin/Library_fitness_vs_parE3_replicate_A.csv"
fitness_repB_path = "../../data/Figure3_toxin_antitoxin/Library_fitness_vs_parE3_replicate_B.csv"

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
    chain_break_idx = query_seq.index(pare_seq)
    print(f"ParD sequence: {pard_seq}")
    print(f"ParE sequence: {pare_seq}")
    print(f"Chain break index: {chain_break_idx}")
    assert query_seq[chain_break_idx:] == pare_seq

    mutated_indices_l = [58, 59, 60, 63]
    print(f"Mutated indices: {mutated_indices_l}")
    print(f"Amino acids at mutated indices:")
    for mut_idx in mutated_indices_l:
        print(f"\tPosition {mut_idx}: {query_seq[mut_idx]}")

    # Load MSA
    with open(parde_alignment_path, "r") as oFile:
            msa_str_l = oFile.readlines()[1::2]
    msa_str_l = [(seq_idx, seq.strip()) for seq_idx, seq in enumerate(msa_str_l)]
    # Prep MSA Transformer inputs
    batch_labels, batch_strs, batch_tokens = msa_transformer_batch_converter([msa_str_l])
    wt_tokens = deepcopy(batch_tokens[0, 0])

    # Score variants
    pseudolikelihood_l = []
    avg_fitness_l = []
    msa_transformer_alphabet_d = msa_transformer_alphabet.to_dict()
    mask_tok = msa_transformer_alphabet_d['<mask>']
    for (mut_a, rep_a), (mut_b, rep_b) in tqdm(zip(fitness_repA.values, fitness_repB.values), total=fitness_repA.shape[0]):
        mut_tokens = deepcopy(wt_tokens)
        batch_tokens_l = []
        for i in range(4):
            mask_pos = mutated_indices_l[i] + 1 # +1 to account for CLS token prepended to sequence
            mut_tokens[mask_pos] = mask_tok
            for j in range(4):
                if j != i:
                    mutated_idx = mutated_indices_l[j] + 1
                    mut_tokens[mutated_idx] = msa_transformer_alphabet_d[mut_a[j]]
            curr_batch_tokens = deepcopy(batch_tokens)
            curr_batch_tokens[0, 0] = mut_tokens
            batch_tokens_l.append(curr_batch_tokens)
        all_batch_tokens = torch.concatenate(batch_tokens_l, dim=0).to(device)
        with torch.no_grad():
            res = msa_transformer(all_batch_tokens)
        # Compute pseudolikelihood
        pseudolikelihood = np.sum(
            [torch.log_softmax(res['logits'][mut_idx, 0, mutated_indices_l[mut_idx] + 1], dim=0)[msa_transformer_alphabet_d[mut_a[mut_idx]]].item() for mut_idx in range(4)]
        )
        pseudolikelihood_l.append(pseudolikelihood)
        avg_fitness_l.append(np.mean(np.array([rep_a, rep_b])))
    np.save("results/pseudolikelihood_l.msa_transformer.qid30_id90.npy", np.array(pseudolikelihood_l))

if __name__ == "__main__":
    main()