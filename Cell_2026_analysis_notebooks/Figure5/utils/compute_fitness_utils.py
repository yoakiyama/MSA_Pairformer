# Adapted from the ProteinGym codebase https://github.com/OATML-Markslab/ProteinGym/blob/main/proteingym/baselines/esm/compute_fitness.py
import argparse
import pathlib
import os
import random
import numpy as np
import itertools
from Bio import SeqIO
from typing import List, Tuple
import torch
from torch.nn.functional import one_hot

from MSA_Pairformer.dataset import aa2tok_d, prepare_msa_masks

import msa_utils

def sample_msa(filename: str, nseq: int, sampling_strategy: str, random_seed: int, weight_filename=None, processed_msa=None, num_cpus=1):
    """Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    print("Sampling sequences from MSA with strategy: "+str(sampling_strategy))
    random.seed(random_seed)
    if sampling_strategy=='first_x_rows':
        msa = [
            (record.description, str(record.seq))
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)
        ]
    elif sampling_strategy=='random':
        msa = [
            (record.description, str(record.seq)) for record in SeqIO.parse(filename, "fasta")
        ]
        nseq = min(len(msa),nseq)
        msa = random.sample(msa, nseq)
    elif sampling_strategy=='sequence-reweighting':
        # If MSA has already been processed, just use it here
        if processed_msa is None:
            if weight_filename is None:
                print("Need weight filename if using sequence-reweighting sample strategy")
            MSA = msa_utils.MSA_processing(
                MSA_location=filename,
                use_weights=True,
                weights_location=weight_filename,
                num_cpus=num_cpus
            )
            print("Name of focus_seq: "+str(MSA.focus_seq_name))
        else:
            MSA = processed_msa

        # Make sure we always keep the WT in the subsampled MSA
        msa = [(MSA.focus_seq_name,MSA.raw_seq_name_to_sequence[MSA.focus_seq_name])]

        non_wt_weights = np.array([w for k, w in MSA.seq_name_to_weight.items() if k != MSA.focus_seq_name])
        non_wt_sequences = [(k, s) for k, s in MSA.seq_name_to_sequence.items() if k != MSA.focus_seq_name]
        non_wt_weights = non_wt_weights / non_wt_weights.sum() # Renormalize weights

        # Sample the rest of the MSA according to their weights
        if len(non_wt_sequences) > 0:
            msa.extend(random.choices(non_wt_sequences, weights=non_wt_weights, k=nseq-1))

        print("Check sum weights MSA: "+str(non_wt_weights.sum()))

    msa = [(desc, ''.join(seq) if isinstance(seq, list) else seq) for desc, seq in msa]
    msa = [(desc, seq.upper()) for desc, seq in msa]
    # print("First 10 elements of sampled MSA: ")
    # print(msa[:10])
    return msa


def process_msa(filename: str, weight_filename: str, filter_msa: bool, path_to_hhfilter: str, hhfilter_min_cov=75, hhfilter_max_seq_id=100, hhfilter_min_seq_id=0, num_cpus=1) -> List[Tuple[str, str]]:
    if filter_msa:
        input_folder = '/'.join(filename.split('/')[:-1])
        msa_name = filename.split('/')[-1].split('.')[0]
        if not os.path.isdir(input_folder+os.sep+'preprocessed'):
            os.mkdir(input_folder+os.sep+'preprocessed')
        if not os.path.isdir(input_folder+os.sep+'hhfiltered'):
            os.mkdir(input_folder+os.sep+'hhfiltered')
        preprocessed_filename = input_folder+os.sep+'preprocessed'+os.sep+msa_name
        os.system('cat '+filename+' | tr  "."  "-" >> '+preprocessed_filename+'.a2m')
        os.system('dd if='+preprocessed_filename+'.a2m of='+preprocessed_filename+'_UC.a2m conv=ucase')
        output_filename = input_folder+os.sep+'hhfiltered'+os.sep+msa_name+'_hhfiltered_cov_'+str(hhfilter_min_cov)+'_maxid_'+str(hhfilter_max_seq_id)+'_minid_'+str(hhfilter_min_seq_id)+'.a2m'
        os.system(path_to_hhfilter+os.sep+'bin/hhfilter -cov '+str(hhfilter_min_cov)+' -id '+str(hhfilter_max_seq_id)+' -qid '+str(hhfilter_min_seq_id)+' -i '+preprocessed_filename+'_UC.a2m -o '+output_filename+' -maxseq 10000000')
        filename = output_filename

    MSA = msa_utils.MSA_processing(
        MSA_location=filename,
        use_weights=True,
        weights_location=weight_filename,
        num_cpus=num_cpus
    )
    print("Name of focus_seq: "+str(MSA.focus_seq_name))
    return MSA


def create_parser():
    parser = argparse.ArgumentParser(
        description="Label a deep mutational scan with predictions from MSA model."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the MSA model .pt file",
        required=True,
    )
    parser.add_argument(
        "--sequence",
        type=str,
        help="Base sequence to which mutations were applied",
    )
    parser.add_argument(
        "--dms-input",
        type=pathlib.Path,
        help="CSV file containing the deep mutational scan",
    )
    parser.add_argument(
        "--dms-index",
        type=int,
        help="Index of DMS in mapping file",
    )
    parser.add_argument(
        "--dms_mapping",
        type=str,
        help="Location of DMS_mapping",
    )
    parser.add_argument(
        "--mutation-col",
        type=str,
        default="mutant",
        help="column in the deep mutational scan labeling the mutation as 'AiB'"
    )
    parser.add_argument(
        "--dms-output",
        type=pathlib.Path,
        help="Output file containing the deep mutational scan along with predictions",
    )
    parser.add_argument(
        "--offset-idx",
        type=int,
        default=1,
        help="Offset of the mutation positions in `--mutation-col`"
    )
    parser.add_argument(
        "--scoring-strategy",
        type=str,
        default="wt-marginals",
        choices=["wt-marginals", "pseudo-ppl", "masked-marginals"],
        help=""
    )
    parser.add_argument(
        "--msa-path",
        type=pathlib.Path,
        help="path to MSA (required for MSA Transformer)"
    )
    parser.add_argument(
        "--msa-sampling-strategy",
        type=str,
        default='sequence-reweighting',
        help="Strategy to sample sequences from MSA [sequence-reweighting|random|first_x_rows]"
    )
    parser.add_argument(
        "--msa-samples",
        type=int,
        default=400,
        help="number of sequences to randomly sample from the MSA"
    )
    parser.add_argument(
        "--msa-weights-folder",
        type=str,
        default=None,
        help="Folder with weights to sample MSA sequences in 'sequence-reweighting' scheme"
    )
    parser.add_argument(
        '--filter-msa',
        action='store_true',
        help='Whether to use hhfilter to filter input MSA before sampling'
    )
    parser.add_argument(
        '--hhfilter-min-cov',
        type=int,
        default=75, 
        help='minimum coverage with query (%)'
    )
    parser.add_argument(
        '--hhfilter-max-seq-id',
        type=int,
        default=90, 
        help='maximum pairwise identity (%)'
    )
    parser.add_argument(
        '--hhfilter-min-seq-id',
        type=int,
        default=0, 
        help='minimum sequence identity with query (%)'
    )
    parser.add_argument(
        '--path-to-hhfilter',
        type=str,
        default='/home/ubuntu/tools/hhfilter/bin', 
        help='Path to hhfilter binaries'
    )
    parser.add_argument(
        '--scoring-window',
        type=str,
        default='optimal', 
        help='Approach to handle long sequences [optimal|overlapping]'
    )
    parser.add_argument(
        '--overwrite-prior-scores',
        action='store_true',
        help='Whether to overwrite prior scores in the dataframe'
    )
    parser.add_argument(
        '--seeds',
        type=int,
        default=0,
        nargs='+',
        help='List of seeds to use for sampling'
    )
    parser.add_argument(
        '--num-cpus',
        type=int,
        default=1,
        help='Number of CPUs to use'
    )
    parser.add_argument(
        '--use-bfloat16',
        action='store_true',
        help='Whether to use bfloat16'
    )
    #No ref file provided
    parser.add_argument('--target_seq', default=None, type=str, help='WT sequence mutated in the assay')
    parser.add_argument('--weight_file_name', default=None, type=str, help='Wild type sequence mutated in the assay (to be provided if not using a reference file)')
    parser.add_argument('--MSA_start', default=None, type=int, help='Index of first AA covered by the MSA relative to target_seq coordinates (1-indexing)')
    parser.add_argument('--MSA_end', default=None, type=int, help='Index of last AA covered by the MSA relative to target_seq coordinates (1-indexing)')
    
    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    return parser

def label_row(row, sequence, token_probs, aa2tok_d, offset_idx):
    score=0
    for mutation in row.split(":"):
        wt, idx, mt = mutation[0], int(mutation[1:-1]) - offset_idx, mutation[-1]
        assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

        wt_encoded, mt_encoded = aa2tok_d[wt], aa2tok_d[mt]

        score += (token_probs[0, idx, mt_encoded] - token_probs[0, idx, wt_encoded]).item()
    return score

def get_mutated_sequence(row, wt_sequence, offset_idx):
    wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
    assert wt_sequence[idx] == wt, "The listed wildtype does not match the provided sequence"
    # modify the sequence
    sequence = wt_sequence[:idx] + mt + wt_sequence[(idx + 1) :]
    return sequence


def compute_pppl(sequence, model, alphabet, MSA_data = None, mode = "ESM1v"):
    # encode the sequence
    data = [
        ("protein1", sequence),
    ]
    if mode == "MSA_Transformer":
        data = [data + MSA_data[0]]
    batch_converter = alphabet.get_batch_converter()

    _, _, batch_tokens = batch_converter(data)
    # compute probabilities at each position
    log_probs = []
    for i in range(1, len(sequence) - 1):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, i] = alphabet.mask_idx
        with torch.no_grad():
            token_probs = torch.log_softmax(model(batch_tokens_masked.cuda())["logits"], dim=-1)  # This might OOM because the MSA batch is large (400 by default)
        if mode == "ESM1v":
            log_probs.append(token_probs[0, i, alphabet.get_idx(sequence[i])].item())  # vocab size
        elif mode == "MSA_Transformer":
            log_probs.append(token_probs[0, 0, i, alphabet.get_idx(sequence[i])].item())  # vocab size
    return sum(log_probs)

def prepare_msa_inputs(data):
    # Get all sequences from the MSA
    seq_l = [data[0][i][1] for i in range(len(data[0]))]
    # Convert sequences to numpy array
    seq_a = np.array([list(seq) for seq in seq_l])
    # Tokenize MSA
    tokenized_msa_a = torch.from_numpy(np.vectorize(aa2tok_d.get)(seq_a))
    # Prepare MSA masks
    mask, msa_mask, full_mask, pairwise_mask = prepare_msa_masks(tokenized_msa_a.unsqueeze(0))
    # One-hot encode MSA
    nTokenTypes = len(np.unique(list(aa2tok_d.values())))
    msa_onehot_t = one_hot(tokenized_msa_a, num_classes=nTokenTypes).float()
    return msa_onehot_t, tokenized_msa_a, mask, msa_mask, full_mask, pairwise_mask
