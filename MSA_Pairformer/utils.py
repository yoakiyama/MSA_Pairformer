import subprocess
import os
import torch
import multiprocessing as mp
from tqdm import tqdm
from subprocess import Popen, PIPE
from Bio.PDB import *
from typing import Optional, Dict
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import numpy as np
import torch.nn.functional as F

def _run(x):
    "Generic run."
    if isinstance(x, str):
        res = subprocess.run(x.split(' '), stdout=PIPE, stderr=PIPE, universal_newlines=True)
    elif isinstance(x, list):
        res = subprocess.run(x, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    else:
        print("Must pass a string or a list of strings to _run()")
        return -1
    res.stdout = res.stdout.strip().split('\n')
    res.stderr = res.stderr.strip().split('\n')
    return res

def write_log(fout, res):
    """Write output logs."""
    with open(fout+'.out', "a") as f:
        for l in res.stdout:
            f.write(l+'\n')

    with open(fout+'.err', "a") as f:
        for l in res.stderr:
            f.write(l+'\n')

#############
# PDB utils #
#############
def write_chain_from_pdb(structure_file_path, chain_ids, out_file_path: None, ignore_hetatm = True):
    """Write chains from a PDB file."""
    atom_counter = 1
    atom_label_l = ['ATOM  ', 'HETATM'] if ignore_hetatm else ['ATOM  ']
    if out_file_path is None:
        chain_id_str = '_'.join(chain_ids)
        out_file_path = structure_file_path.replace('.pdb', f'_{chain_id_str}.pdb')
    with open(structure_file_path, "r") as inFile, open(out_file_path, "w") as outFile:
        for line in inFile.readlines():
            if line[:6] in ['HEADER', 'TITLE ']:
                outFile.write(line)
                continue
            line_chain = line[21]
            if line_chain not in chain_ids:
                continue
            if line[:6] in atom_label_l:
                new_line = (line[:6] + str(atom_counter).rjust(5) + line[11:])
                outFile.write(new_line)
                atom_counter += 1
            if line.startswith('TER'):
                new_ter = ("TER   " + str(atom_counter).rjust(5) + line[11:])
                outFile.write(new_ter)
            if line.startswith('END'):
                outFile.write(line)
    return out_file_path

def write_chain_from_cif(structure_file_path, chain_ids, out_file_path):
    # Create MMCIFParser
    parser = MMCIFParser(QUIET=True)
    # Parse structure from CIF file
    structure = parser.get_structure('protein_structure', structure_file_path)
    # Create subclass of Select to filter out chains
    class ChainSelect(Select):
        def __init__(self, chain_ids):
            self.chain_ids = chain_ids
        def accept_chain(self, chain):
            if chain.id in self.chain_ids:
                return True
            else:
                return False
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(out_file_path, ChainSelect(chain_ids))

def convert_cif_to_pdb(input_cif_path, output_pdb_path):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('protein_structure', input_cif_path)
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb_path)


def get_distance_matrix(structure_file_path, chain_id):
    # Load coordinates
    with open(structure_file_path, "r") as oFile:
        cif_lines_l = [l for l in oFile.readlines() if 
                       (l.startswith('ATOM') and (l.split()[6] == chain_id) and th
                        (((l.split()[3] == 'CA') and (l.split()[5]=='GLY')) or 
                         ((l.split()[3]=='CB') and (l.split()[5] != 'GLY'))))]
    coords_a = np.array([[float(xyz) for xyz in l.split()[10:13]] for l in cif_lines_l])
    # Compute distances
    dist_a = np.linalg.norm(coords_a[:, None] - coords_a, axis=-1)
    return dist_a


####################
# Contact analysis #
####################
def get_contacts(structure_file_path, chain_id, max_dist = 8):
    # Get distance matrix
    dist_a = get_distance_matrix(structure_file_path, chain_id)
    # Threshold distances to determine contacts
    contacts_a = dist_a.copy() <= max_dist
    return contacts_a

def compute_precision(predictions_a, targets_a, min_seq_sep = 6):
    assert predictions_a.shape == targets_a.shape, "Predictions and targets must have the same shape"
    # Get valid indices
    seqlen = predictions_a.shape[1]
    seqlen_range = torch.arange(seqlen)
    valid_mask = seqlen_range[None, :] - seqlen_range[:, None] >= min_seq_sep
    # Some contact maps have -1 for invalid pairs
    valid_mask = valid_mask & (targets_a >= 0)

def run_confind(
    structure_file_path, 
    output_contact_file_path, 
    bin_path = "/home/ubuntu/tools/confind/confind",
    rot_lib_path = "/home/ubuntu/tools/confind/rotlibs",
    log = None
):
    """Run confind."""
    cmd = f"{bin_path} --p {structure_file_path} --o {output_contact_file_path} --rLib {rot_lib_path}"
    res = _run(cmd)
    if log is not None:
        assert isinstance(log, str), "Log must be a string"
        write_log(log, res)
    return res

def run_confind_mp(param_d):
    """Run confind in parallel."""
    structure_file_path = param_d["structure_file_path"]
    output_contact_file_path = param_d["output_contact_file_path"]
    bin_path = param_d["bin_path"]
    rot_lib_path = param_d["rot_lib_path"]  
    run_confind(structure_file_path, output_contact_file_path, bin_path, rot_lib_path)

def run_batch_confind(
    structure_file_paths_l: list, 
    output_contact_file_paths_l: list, 
    bin_path = "/home/ubuntu/tools/confind/confind", 
    rot_lib_path = "/home/ubuntu/tools/confind/rotlibs",
    nproc = None,
    cpu_buffer: int = 4
):
    """Run confind in batch."""
    if nproc is None:
        nproc = mp.cpu_count() - cpu_buffer
    zipped_paths = zip(structure_file_paths_l, output_contact_file_paths_l)
    param_d_l = [{'structure_file_path': structure_file_path, 'output_contact_file_path': output_contact_file_path, 'bin_path': bin_path, 'rot_lib_path': rot_lib_path} for 
                 structure_file_path, output_contact_file_path in zipped_paths]
    with mp.Pool(processes=nproc) as pool:
        list(tqdm(pool.imap(run_confind_mp, param_d_l), total=len(param_d_l), leave=True))
        pool.close()


# ######################
# # Contact evaluation #
# ######################
# def calculate_precision_batch(pred_contacts, true_contacts, minsep=6, maxsep=None):
#     if pred_contacts.shape != true_contacts.shape:
#         raise ValueError("Predicted and true contact matrices must have the same shape")
#     if pred_contacts.dim() == 2:
#         pred_contacts = pred_contacts.unsqueeze(0)
#     if true_contacts.dim() == 2:
#         true_contacts = true_contacts.unsqueeze(0)
#     # Get batch size and longest sequence length
#     B, L, _ = pred_contacts.shape
#     # Get device
#     device = pred_contacts.device
#     if true_contacts.device != device:
#         true_contacts = true_contacts.to(device)
#     # Create valid mask (only consider upper triangular matrix)
#     # Padded regions have -1 in true_contacts
#     # Valid mask for both predicted and true contact matrices
#     seqlen_range = torch.arange(L, device=device)
#     sep = seqlen_range.unsqueeze(0) - seqlen_range.unsqueeze(1)
#     sep = sep.unsqueeze(0)
#     valid_mask = sep >= minsep
#     if maxsep is not None:
#         valid_mask &= sep < maxsep
#     valid_mask = valid_mask & (true_contacts >= 0)
    
#     # Fill prediction matrix with -inf if it's not valid
#     pred_contacts = pred_contacts.masked_fill(~valid_mask, float('-inf'))

#     # Get upper triangular predictions and true contacts
#     # Predictions have been masked with -inf if they are not valid, so when we sort and take the topk, we are only taking valid predictions
#     x_ind, y_ind = np.triu_indices(L, minsep)
#     predictions_upper = pred_contacts[:, x_ind, y_ind]
#     true_upper = true_contacts[:, x_ind, y_ind]
    
#     # Determine number of true contacts in valid region
#     K = true_contacts.masked_fill(~valid_mask, 0).sum(dim=-1).sum(dim=-1)
    
#     # Get Top-K predictions and pad if not enough predictions
#     max_k = int(K.max().item())
#     if max_k == 0:
#         return None
#     indices = predictions_upper.argsort(dim=-1, descending=True)[:, :max_k]
#     topk_targets = true_upper[torch.arange(B).unsqueeze(1), indices]
#     if topk_targets.size(1) < max_k:
#         topk_targets = torch.nn.functional.pad(topk_targets, [0, max_k - topk_targets.size(1)])
#     # Get cumulative sum of true positives
#     cumulative_dist = topk_targets.type_as(pred_contacts).cumsum(dim=-1)
#     # Compute precision based on true number of contacts
#     gather_lengths = K.unsqueeze(1)
#     gather_indices = (
#         torch.arange(0.1, 1.1, 0.1, device=device).unsqueeze(0) * gather_lengths
#     ).type(torch.long) - 1
#     gather_indices = gather_indices.clamp_min(0)
#     # Bin cumulative sum of true positives with intervals of sequence length
#     binned_cumulative_dist = cumulative_dist.gather(1, gather_indices)
#     binned_precisions = binned_cumulative_dist / (gather_indices + 1).type_as(binned_cumulative_dist)
#     # Get precisions 
#     pl5 = binned_precisions[:, 1]
#     pl2 = binned_precisions[:, 4]
#     pl = binned_precisions[:, 9]
#     auc = binned_precisions.mean(dim=-1)

#     return {"AUC": auc, "P@L5": pl5, "P@L2": pl2, "P@L": pl}

# def evaluate_contact_prediction(
#     predictions: torch.Tensor,
#     targets: torch.Tensor,
# ) -> Dict[str, float]:
#     if isinstance(targets, np.ndarray):
#         targets = torch.from_numpy(targets)
#     if isinstance(predictions, np.ndarray):
#         predictions = torch.from_numpy(predictions)
#     contact_ranges = [
#         ("local", 3, 6),
#         ("short", 6, 12),
#         ("medium", 12, 24),
#         ("long", 24, None),
#     ]
#     metrics = {}
#     targets = targets.to(predictions.device)
#     for name, minsep, maxsep in contact_ranges:
#         rangemetrics = calculate_precision_batch(
#             predictions,
#             targets,
#             minsep=minsep,
#             maxsep=maxsep,
#         )
#         if rangemetrics is not None:
#             for key, val in rangemetrics.items():
#                 metrics[f"{name}_{key}"] = list(val.float().cpu().numpy())
#     return metrics

def compute_precisions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    src_lengths: Optional[torch.Tensor] = None,
    minsep: int = 6,
    maxsep: Optional[int] = None,
    override_length: Optional[int] = None,  # for casp
):
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(0)
    if targets.dim() == 2:
        targets = targets.unsqueeze(0)
    override_length = (targets[0, 0] >= 0).sum()

    # Check sizes
    if predictions.size() != targets.size():
        raise ValueError(
            f"Size mismatch. Received predictions of size {predictions.size()}, "
            f"targets of size {targets.size()}"
        )
    device = predictions.device

    batch_size, seqlen, _ = predictions.size()
    seqlen_range = torch.arange(seqlen, device=device)

    sep = seqlen_range.unsqueeze(0) - seqlen_range.unsqueeze(1)
    sep = sep.unsqueeze(0)
    valid_mask = sep >= minsep
    valid_mask = valid_mask & (targets >= 0)  # negative targets are invalid

    if maxsep is not None:
        valid_mask &= sep < maxsep

    if src_lengths is not None:
        valid = seqlen_range.unsqueeze(0) < src_lengths.unsqueeze(1)
        valid_mask &= valid.unsqueeze(1) & valid.unsqueeze(2)
    else:
        src_lengths = torch.full([batch_size], seqlen, device=device, dtype=torch.long)

    predictions = predictions.masked_fill(~valid_mask, float("-inf"))

    x_ind, y_ind = np.triu_indices(seqlen, minsep)
    predictions_upper = predictions[:, x_ind, y_ind]
    targets_upper = targets[:, x_ind, y_ind]

    topk = seqlen if override_length is None else max(seqlen, override_length)
    indices = predictions_upper.argsort(dim=-1, descending=True)[:, :topk]
    topk_targets = targets_upper[torch.arange(batch_size).unsqueeze(1), indices]
    if topk_targets.size(1) < topk:
        topk_targets = F.pad(topk_targets, [0, topk - topk_targets.size(1)])

    cumulative_dist = topk_targets.type_as(predictions).cumsum(-1)

    gather_lengths = src_lengths.unsqueeze(1)
    if override_length is not None:
        gather_lengths = override_length * torch.ones_like(
            gather_lengths, device=device
        )

    gather_indices = (
        torch.arange(0.1, 1.1, 0.1, device=device).unsqueeze(0) * gather_lengths
    ).type(torch.long) - 1

    binned_cumulative_dist = cumulative_dist.gather(1, gather_indices)
    binned_precisions = binned_cumulative_dist / (gather_indices + 1).type_as(
        binned_cumulative_dist
    )

    pl5 = binned_precisions[:, 1]
    pl2 = binned_precisions[:, 4]
    pl = binned_precisions[:, 9]
    auc = binned_precisions.mean(-1)

    return {"AUC": auc, "P@L": pl, "P@L2": pl2, "P@L5": pl5}


def evaluate_contact_prediction(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    contact_ranges = [
        ("local", 3, 6),
        ("short", 6, 12),
        ("medium", 12, 24),
        ("long", 24, None),
        ("morethansix", 6, None)
    ]
    metrics = {}
    targets = targets.to(predictions.device)
    for name, minsep, maxsep in contact_ranges:
        rangemetrics = compute_precisions(
            predictions,
            targets,
            minsep=minsep,
            maxsep=maxsep,
        )
        for key, val in rangemetrics.items():
            metrics[f"{name}_{key}"] = val.item()
    return metrics

def extract_confind_contacts(confind_file_path):
    with open(confind_file_path, "r") as oFile:
        lines = oFile.readlines()
        # Get protein length
        max_res_idx = int(lines[-2].split()[1].split(',')[1])
        length = len(lines[-1].split()[1:])
        contact_a = np.zeros((max_res_idx, max_res_idx))
        for line in lines:
            if line.startswith('contact'):
                split_line = line.split()
                res_i = int(split_line[1].split(',')[1]) - 1
                res_j = int(split_line[2].split(',')[1]) - 1
                c = float(split_line[3])
                if c > contact_a[res_i, res_j]:
                    contact_a[res_i, res_j] = c
                    contact_a[res_j, res_i] = c
    return contact_a

def extract_homooligomeric_confind_contacts(confind_file_path):
    with open(confind_file_path, "r") as oFile:
        lines = oFile.readlines()
        # Get protein length
        observed_chains_d = {}
        for line in lines:
            if line.startswith("freedom"):
                chain_id = line.split()[1].split(',')[0]
                res_idx = int(line.split()[1].split(',')[1])
                if chain_id not in observed_chains_d:
                    observed_chains_d[chain_id] = res_idx
                else:
                    if res_idx > observed_chains_d[chain_id]:
                        observed_chains_d[chain_id] = res_idx
        max_res_idx = max(list(observed_chains_d.values()))
        contact_a = np.zeros((max_res_idx, max_res_idx))
        # Get contacts
        for line in lines:
            if line.startswith('contact'):
                split_line = line.split()
                res_i = int(split_line[1].split(',')[1]) - 1
                res_j = int(split_line[2].split(',')[1]) - 1
                c = float(split_line[3])
                if c > contact_a[res_i, res_j]:
                    contact_a[res_i, res_j] = c
                    contact_a[res_j, res_i] = c
    return contact_a

def fit_seq_weight_mixture_model(
    seq_weights_a: np.array,
    n_components: int = 2,
    random_state: int = 42,
    max_iter: int = 5000,
    n_init: int = 100,
    tol: float = 1e-6,
    scaling_factor = 1e5,
    plot = True,
    figsize = (6, 4),
    return_gmm = False
):
    # Fit a mixture model to the sequence weights
    # Prepare the data
    X = seq_weights_a.reshape(-1, 1)
    # Initialize and fit GMM to scaled data
    gmm = GaussianMixture(n_components=n_components, random_state=random_state, max_iter=max_iter, n_init=n_init, tol=tol)
    gmm.fit(X * scaling_factor)
    average_log_likelihood = gmm.score(X * scaling_factor)
    n_samples = X.shape[0]
    total_log_likelihood = average_log_likelihood * n_samples
    # Generate x-values in original scale
    x = np.linspace(seq_weights_a.min(), seq_weights_a.max(), 1000).reshape(-1, 1)
    # Compute PDFs in scaled space
    x_scaled = x * scaling_factor
    log_prob_scaled = gmm.score_samples(x_scaled)
    pdf_scaled = np.exp(log_prob_scaled)
    responsibilities = gmm.predict_proba(x_scaled)
    pdf_individual_scaled = responsibilities * pdf_scaled[:, np.newaxis]
    
    if plot:
        f, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Plot histogram of original data with density=False to show counts
        hist_counts, hist_bins, _ = ax.hist(seq_weights_a, bins=100, density=False, 
                                          alpha=0.5, color="grey", 
                                          label="Avg. sequence attention weights")
        
        # Calculate bin width for scaling PDFs to match count data
        bin_width = hist_bins[1] - hist_bins[0]
        
        # Scale factor to convert density to counts
        scale_to_counts = bin_width * len(seq_weights_a)
        
        # Plot the PDFs scaled to match counts
        for component_idx in range(n_components):
            # Transform PDFs to count scale
            pdf_component = pdf_individual_scaled[:, component_idx] * scaling_factor * scale_to_counts
            ax.plot(x, pdf_component, '--', color=f'C{component_idx}', label=f"Component {component_idx}")
        
        # Set labels and title
        ax.set_xlabel("Sequence weight", size=12)
        ax.set_ylabel("Count", size=12)  # Changed from "Density" to "Count"
        ax.set_title("OmpR sequence weight distribution", size=12)  # Updated title to match your image
        
        # Add vertical line at x = 1/num_seqs
        ax.axvline(x=1 / seq_weights_a.shape[0], color='grey', linestyle='--', label="Uniform weighting")
        ax.legend(fontsize=10)
        
        return gmm.means_ / scaling_factor, gmm.covariances_ / scaling_factor**2, total_log_likelihood, f, ax
    
    return gmm.means_ / scaling_factor, gmm.covariances_ / scaling_factor**2, total_log_likelihood


# Get top L pairs
def get_top_L_pairs(predicted_contacts_a, minsep=24):
    L = predicted_contacts_a.shape[0]
    # Only consider upper triangular part
    triu_idx = np.triu_indices_from(predicted_contacts_a, 1)
    # Mask out pairs < 24 residues apart
    mask = np.abs(triu_idx[0] - triu_idx[1]) >= minsep
    filtered_i = triu_idx[0][mask]
    filtered_j = triu_idx[1][mask]
    # Get values for these filtered indices
    vals = predicted_contacts_a[filtered_i, filtered_j]
    # Get top L
    L = min(L, len(vals))
    cutoff = np.sort(vals)[::-1][L-1]
    vals_sort_idx = vals >= cutoff
    return filtered_i[vals_sort_idx], filtered_j[vals_sort_idx]

def get_p_at_k(gt_contacts_a, pred_contacts_a, minsep=24, upper_triangle=True):
    assert gt_contacts_a.shape == pred_contacts_a.shape, f"Size mismatch. Received predictions of size {pred_contacts_a.shape}, targets of size {gt_contacts_a.shape}"
    # Subset for upper triangle if necessary (for monomer but not for hetero-oligomer)
    if upper_triangle:
        triu_idx = np.triu_indices_from(gt_contacts_a, 1)
        mask = np.abs(triu_idx[0] - triu_idx[1]) >= minsep
        filtered_i, filtered_j = triu_idx[0][mask], triu_idx[1][mask]
    else:
        filtered_i, filtered_j = np.indices(gt_contacts_a.shape)
        filtered_i, filtered_j = filtered_i.flatten(), filtered_j.flatten()
    gt_contact_labels = gt_contacts_a[filtered_i, filtered_j]
    pred_contact_vals = pred_contacts_a[filtered_i, filtered_j]
    
    # Get total number of ground truth contacts
    k = gt_contact_labels.sum()

    # Sort predictions by descending order and take the top k
    top_k_indices = np.argpartition(pred_contact_vals, -k)[-k:]
    
    p_at_k = (gt_contact_labels[top_k_indices] == 1).sum() / k
    return p_at_k

def get_p_at_l(gt_contacts_a, pred_contacts_a, minsep=24, upper_triangle=True):
    assert gt_contacts_a.shape == pred_contacts_a.shape, f"Size mismatch. Received predictions of size {pred_contacts_a.shape}, targets of size {gt_contacts_a.shape}"
    # Subset for upper triangle if necessary (for monomer but not for hetero-oligomer)
    if upper_triangle:
        triu_idx = np.triu_indices_from(gt_contacts_a, 1)
        mask = np.abs(triu_idx[0] - triu_idx[1]) >= minsep
        filtered_i, filtered_j = triu_idx[0][mask], triu_idx[1][mask]
    else:
        filtered_i, filtered_j = np.indices(gt_contacts_a.shape)
        filtered_i, filtered_j = filtered_i.flatten(), filtered_j.flatten()
    gt_contact_labels = gt_contacts_a[filtered_i, filtered_j]
    pred_contact_vals = pred_contacts_a[filtered_i, filtered_j]
    
    # Get total number of ground truth contacts
    k = gt_contacts_a.shape[0]

    # Sort predictions by descending order and take the top k
    top_k_indices = np.argpartition(pred_contact_vals, -k)[-k:]
    
    p_at_l = (gt_contact_labels[top_k_indices] == 1).sum() / k
    return p_at_l


def get_coords(structure_file_path, chain_id):
    # Load coordinates
    with open(structure_file_path, "r") as oFile:
        cif_lines_l = [l for l in oFile.readlines() if 
                       (l.startswith('ATOM') and (l.split()[4] == chain_id) and
                        (((l.split()[2] == 'CA') and (l.split()[3]=='GLY')) or 
                         ((l.split()[2]=='CB') and (l.split()[3] != 'GLY'))))]
    coords_a = np.array([[float(xyz) for xyz in l.split()[6:9]] for l in cif_lines_l])
    return coords_a

def get_coords_cif(structure_file_path, chain_id):
    """
    Extract CA (GLY) / CB (all others) coordinates for a single chain from an mmCIF file.

    Returns:
        coords_a    : np.ndarray of shape (N, 3) — coordinates of CA/CB atoms
        sequence    : str — full one-letter sequence of the chain (in residue order)
        coord_indices: list[int] — indices into `sequence` for each row of coords_a
    """
    FIELDS = {
        "_atom_site.group_PDB",
        "_atom_site.label_atom_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_seq_id",   # residue sequence number
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
    }

    THREE_TO_ONE = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        "MSE": "M"
    }

    with open(structure_file_path, "r") as f:
        lines = f.readlines()

    # --- Parse _atom_site loop header ---
    col_index = {}
    in_atom_site_loop = False
    col_counter = 0
    atom_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped == "loop_":
            in_atom_site_loop = False
            col_counter = 0
            col_index = {}
            continue
        if stripped.startswith("_atom_site."):
            in_atom_site_loop = True
            if stripped in FIELDS:
                col_index[stripped] = col_counter
            col_counter += 1
            continue
        if in_atom_site_loop and stripped and not stripped.startswith("_") and not stripped.startswith("#"):
            atom_lines.append(stripped)
        if stripped == "#" and atom_lines:
            break

    missing = FIELDS - set(col_index.keys())
    if missing:
        raise ValueError(f"CIF file missing expected _atom_site fields: {missing}")

    i_group = col_index["_atom_site.group_PDB"]
    i_atom  = col_index["_atom_site.label_atom_id"]
    i_resn  = col_index["_atom_site.label_comp_id"]
    i_chain = col_index["_atom_site.label_asym_id"]
    i_seqid = col_index["_atom_site.label_seq_id"]
    i_x     = col_index["_atom_site.Cartn_x"]
    i_y     = col_index["_atom_site.Cartn_y"]
    i_z     = col_index["_atom_site.Cartn_z"]

    # --- First pass: collect all residues in order to build the full sequence ---
    # Use an ordered dict keyed by seq_id to deduplicate (many atoms per residue)
    residue_map = {}   # seq_id (int) -> three-letter code
    for line in atom_lines:
        cols = line.split()
        if cols[i_group] != "ATOM" or cols[i_chain] != chain_id:
            continue
        seq_id = int(cols[i_seqid])
        if seq_id not in residue_map:
            residue_map[seq_id] = cols[i_resn]

    # Sort by seq_id to ensure correct order, then build sequence + a seq_id -> position index
    sorted_seq_ids = sorted(residue_map.keys())
    sequence = "".join(THREE_TO_ONE.get(residue_map[s], "X") for s in sorted_seq_ids)
    seqid_to_pos = {seq_id: pos for pos, seq_id in enumerate(sorted_seq_ids)}

    # --- Second pass: extract CA/CB coordinates and their positions in the sequence ---
    coords = []
    coord_indices = []
    seen_seqids = set()   # one coordinate per residue

    for line in atom_lines:
        cols = line.split()
        if cols[i_group] != "ATOM" or cols[i_chain] != chain_id:
            continue
        atom_name = cols[i_atom]
        resn      = cols[i_resn]
        seq_id    = int(cols[i_seqid])
        if not ((atom_name == "CA" and resn == "GLY") or
                (atom_name == "CB" and resn != "GLY")):
            continue
        if seq_id in seen_seqids:   # guard against duplicate ATOM records
            continue
        seen_seqids.add(seq_id)
        coords.append([float(cols[i_x]), float(cols[i_y]), float(cols[i_z])])
        coord_indices.append(seqid_to_pos[seq_id])

    return np.array(coords), sequence, coord_indices