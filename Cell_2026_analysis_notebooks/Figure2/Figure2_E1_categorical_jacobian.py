import os
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob

torch._dynamo.config.capture_scalar_outputs = True

from E1.batch_preparer import E1BatchPreparer
from E1.modeling import E1ForMaskedLM
from E1.dynamic_cache import KVCache
from E1.msa_sampling import sample_context

def jac_to_con(jac, center=True, diag="remove", apc=True,
               symm_before=True, symm_after=False):
    """Convert Jacobian to contact map"""
    X = jac.copy()
    Lx, Ax, Ly, Ay = X.shape

    if symm_before:
        X = X + X.transpose(2, 3, 0, 1)

    if center:
        for i in range(4):
            if X.shape[i] > 1:
                X -= X.mean(i, keepdims=True)

    contacts = np.sqrt(np.square(X).sum((1, 3)))

    if symm_after:
        contacts = contacts + contacts.T

    if diag == "remove":
        np.fill_diagonal(contacts, 0)

    if diag == "normalize":
        contacts_diag = np.diag(contacts)
        contacts = contacts / np.sqrt(contacts_diag[:, None] * contacts_diag[None, :])

    if apc:
        ap = contacts.sum(0, keepdims=True) * contacts.sum(1, keepdims=True) / contacts.sum()
        contacts = contacts - ap

    if diag == "remove":
        np.fill_diagonal(contacts, 0)

    return contacts

def get_logits(
    batch,
    model,
    query_idx_t,
    aa_indices_t
):
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
      with torch.autocast(device_type, torch.bfloat16):
          output = model(
              input_ids=batch["input_ids"],
              within_seq_position_ids=batch["within_seq_position_ids"],
              global_position_ids=batch["global_position_ids"],
              sequence_ids=batch["sequence_ids"],
              past_key_values=batch.get("past_key_values", None),
              use_cache=batch.get("use_cache", False),
              output_attentions=False,
              output_hidden_states=False,
          )
    return output['logits'][0, query_idx_t[2:-2]][:, aa_indices_t].float().cpu().numpy()

def get_categorical_jacobian(
    batch,
    model,
    query_idx_t,
    aa_indices_t,
    mutation_subset=None,
    show_progress=True
):
    input_ids = batch['input_ids']
    tokenizes_seqs = input_ids.clone()
    device = next(model.parameters()).device

    # Compute baseline logits for WT query sequence
    wt_tokens = input_ids[0, query_idx_t]
    with torch.no_grad():
        with torch.autocast("cuda", torch.bfloat16):
            output = model(
                input_ids=batch["input_ids"],
                within_seq_position_ids=batch["within_seq_position_ids"],
                global_position_ids=batch["global_position_ids"],
                sequence_ids=batch["sequence_ids"],
                past_key_values=batch.get("past_key_values", None),
                use_cache=batch.get("use_cache", False),
                output_attentions=False,
                output_hidden_states=False,
            )
    fx = output['logits'][0, query_idx_t[2:-2]][:, aa_indices_t].float().cpu().numpy()

    # Parse mutation subset if available
    if mutation_subset is None:
        mutation_indices = list(aa_indices_t.numpy())
    else:
        mutation_indices = mutation_subset

    # Initialize Jacobian matrix
    seq_length = query_idx_t.shape[0] - 4
    numAAs = aa_indices_t.shape[0]
    fx_h = np.zeros((seq_length, numAAs, seq_length, numAAs))

    # Progress bar (optional)
    if show_progress:
        try:
            iterator = tqdm(range(seq_length), desc='Computing categorical Jacobian')
        except ImportError:
            print("Computing categorical Jacobian")
            iterator = range(seq_length)
    else:
        iterator = range(seq_length)

    # For each residue position
    for n in iterator:
        # Get WT AA at this position
        wt_aa = wt_tokens[n+2] # Skip 2 positions since prepende with <bos>1
        # Compute logits after mutating to all specified mutations
        for idx, mutation_aa in enumerate(mutation_indices):
            # Skip computation for WT
            if mutation_aa == wt_aa:
                fx_h[n, idx] = fx.copy()
            else:
                # Compute mutation effect
                mut_tokens = input_ids.clone()
                mut_tokens[0, query_idx_t[n+2]] = mutation_aa
                batch['input_ids'] = mut_tokens
                with torch.no_grad():
                    with torch.autocast("cuda", torch.bfloat16):
                        output = model(
                            input_ids=batch["input_ids"],
                            within_seq_position_ids=batch["within_seq_position_ids"],
                            global_position_ids=batch["global_position_ids"],
                            sequence_ids=batch["sequence_ids"],
                            past_key_values=batch.get("past_key_values", None),
                            use_cache=batch.get("use_cache", False),
                            output_attentions=False,
                            output_hidden_states=False,
                        )
                fx_h[n, idx] = output['logits'][0, query_idx_t[2:-2]][:, aa_indices_t].float().cpu().numpy()

    # Compute Jacobian
    result = fx - fx_h
    return result

def main():
    # Load E1 600M model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = E1ForMaskedLM.from_pretrained("Profluent-Bio/E1-600m").to(device).eval()
    model = torch.compile(model, dynamic=True)
    batch_preparer = E1BatchPreparer()

    id_to_tok_d = {v: k for k,v in batch_preparer.tokenizer.get_vocab().items()}
    aa_indices_t = torch.tensor([8,25,21,11,10,12,24,14,15,16,19,18,20,13,23,26,27,30,32,29])
    aa_l = ['A','R','N','D','C','E','Q','G','H','I','L','K','M','F','P','S','T','W','Y','V']
    assert(len(aa_l) == aa_indices_t.shape[0])

    # MSA files
    target_data_file = "../../data/Figure2_heterooligomer_contact_prediction/target_data.pkl"
    with open(target_data_file, "rb") as oFile:
        target_data_d = pickle.load(oFile)

    # max_token_length = 262144 # 256 sequences for 1024 residue pairs
    max_token_length = 14784
    max_depth = 511
    max_query_similarity = 0.9
    min_query_similarity = 0.3
    neighbor_similarity_lower_bound = 0.8
    seed = 0
    res_dir = "results/e1/"
    os.makedirs(res_dir, exist_ok=True)

    for target_id in tqdm(target_data_d):
        msa_file = target_data_d[target_id]['msa_file']
        out_path = os.path.join(res_dir, f"{target_id}.txt")
        # Get query sequence
        with open(msa_file, "r") as oFile:
            query_seq = oFile.readlines()[1].strip()
        # Parse MSA for context
        print(f"Parsing MSA for {target_id}...")
        try:
            context, _ = sample_context(
                msa_path=msa_file,
                # Maximum number of sequences that can be in context (hard limit of 511)
                max_num_samples=max_depth,
                # Total number of residues in the context
                max_token_length=max_token_length,
                # Maximum similarity of any context sequence to the query sequence
                max_query_similarity=max_query_similarity,
                # Minimum similarity of any context sequence to the query sequence
                min_query_similarity=min_query_similarity,
                # Minimum similarity between any two context sequences for them to be considered neighbors
                neighbor_similarity_lower_bound=neighbor_similarity_lower_bound,
                seed=seed,
            )
            full_prompt = [context + ',' + query_seq]
        except Exception as e:
            print(f"No sequences found in MSA for {target_id}...")
            full_prompt = [query_seq]

        batch = batch_preparer.get_batch_kwargs(full_prompt, device=device)

        # Get token indices of query sequence
        query_idx_t = torch.where(batch['sequence_ids'][0] == batch['sequence_ids'][0].max())[0]

        # Get logits
        catjac_result = get_categorical_jacobian(batch, model, query_idx_t, aa_indices_t)
        catjac_contacts = jac_to_con(catjac_result)
        # Save contacts
        np.savetxt(out_path, catjac_contacts)
    
if __name__ == "__main__":
    main()