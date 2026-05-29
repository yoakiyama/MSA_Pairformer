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

import numpy as np
import pickle
import torch
import time
from tqdm import tqdm
from huggingface_hub import snapshot_download
from pathlib import Path
from scipy.optimize import linear_sum_assignment

from MSA_Pairformer.dataset import esmtok_to_pairformertok_d, aa2tok_d
from MSA_Pairformer.model import MSAPairformer
from MSA_Pairformer.regression import MRFHead

from MSA_Pairformer.pairing_optimization.mp_pdp import MP_PDP
from MSA_Pairformer.pairing_optimization.msa_parsing import read_msa
from MSA_Pairformer.pairing_optimization.datasets import generate_dataset, dataset_tokenizer

def save_parameters(parameters_all, filepath):
    """Saves the parameters dictionary"""
    for name, parameters in parameters_all.items():
        with open(filepath / f"{name}.pkl", "wb") as f:
            pickle.dump(parameters, f)
        with open(filepath / f"{name}.csv", "w") as f:
            for key in parameters.keys():
                f.write("%s, %s\n" % (key, parameters[key]))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

# Reset GPU memory stats
torch.cuda.empty_cache()
torch.cuda.synchronize()
baseline_alloc = torch.cuda.memory_allocated()
baseline_resv  = torch.cuda.memory_reserved()
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

# Set device and load MSA Pairformer to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
msa_pairformer = MSAPairformer.from_pretrained(
    device=device
)
msa_pairformer.eval()
msa_pairformer = torch.compile(msa_pairformer, dynamic=True)
mrf_head = MRFHead(
    dim_pairwise = 256,
    dim_alphabet = 26
)
path = Path(snapshot_download(repo_id="yakiyama/MSA-Pairformer"))
mrf_head_checkpoint = torch.load(path / "mrf_head.bin", weights_only=True, map_location=device)
# Remove "mrf_head." prefix from every key in mrf_head_checkpoint, if it exists
mrf_head_checkpoint = {k.replace("mrf_head.", ""): v for k, v in mrf_head_checkpoint.items()}
mrf_head.load_state_dict(mrf_head_checkpoint)
mrf_head.to(device)


for p in mrf_head.parameters():
    p.requires_grad = False
for p in msa_pairformer.parameters():
    p.requires_grad = False

def main():
    warm_epochs = 20
    main_epochs = 400
    nSeqs = 384
    RESULTS_DIR = f"results/pairing/hk_rr_{nSeqs}_warm"
    # RESULTS_DIR = "test"
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    for seed in range(5):
        # Load data 
        msa_data = [read_msa("../../data/Figure3_toxin_antitoxin/HK_in_Concat_nnn.fasta", -1),
                    read_msa("../../data/Figure3_toxin_antitoxin/RR_in_Concat_nnn.fasta", -1)]
        get_species_name = (lambda strn: strn.split("|")[1])
        parameters_dataset = {
            "N": nSeqs,  # Average number of sequences in the input
            "pos": 0,  # Size of the context pairs to use as positive example 
            "max_size": 100,  # Max size of species MSAs (if same as N there is no limit on size)
            "NUMPY_SEED": seed,
            "NUMPY_SEED_OTHER": seed,
        }
        dataset, species_sizes = generate_dataset(
            parameters_dataset, msa_data, get_species_name=get_species_name
        )
        all_seq_set = set()
        filtered_dataset = {'msa': {'left': [], 'right': []}, 'positive_examples': None}
        for i in range(len(dataset['msa']['left'])):
            curr_seq = dataset['msa']['left'][i][1] + dataset['msa']['right'][i][1]
            if curr_seq not in all_seq_set:
                filtered_dataset['msa']['left'].append(dataset['msa']['left'][i])
                filtered_dataset['msa']['right'].append(dataset['msa']['right'][i])
                all_seq_set.add(curr_seq)
        curr_taxid = get_species_name(dataset['msa']['left'][0][0])
        curr_taxid_size = 1
        filtered_species_sizes = []
        solo_seq_idx_l = []
        for i in range(1, len(filtered_dataset['msa']['left'])):
            if get_species_name(filtered_dataset['msa']['left'][i][0]) == curr_taxid:
                curr_taxid_size += 1
            else:
                if curr_taxid_size == 1:
                    solo_seq_idx_l.append(i)
                filtered_species_sizes.append(curr_taxid_size)
                curr_taxid = get_species_name(filtered_dataset['msa']['left'][i][0])
                curr_taxid_size = 1
        filtered_species_sizes.append(curr_taxid_size)
        tokenized_dataset = dataset_tokenizer(filtered_dataset, device=device)
        # Get ESM tokens and convert to MSA Pairformer tokens
        esm_tokens_left = tokenized_dataset["msa"]["left"].argmax(dim=-1)
        esm_tokens_right = tokenized_dataset["msa"]["right"].argmax(dim=-1)
        # Remove singletons
        nonsingle_mask_t = torch.ones(esm_tokens_left.shape[1], dtype=torch.bool)
        nonsingle_mask_t[solo_seq_idx_l] = False
        esm_tokens_left = esm_tokens_left[:, nonsingle_mask_t, :]
        esm_tokens_right = esm_tokens_right[:, nonsingle_mask_t, :]
        filtered_species_sizes = np.array([s for s in filtered_species_sizes if s > 1])

        # Map ESM tokens to MSA Pairformer tokens and remove CLS token
        mp_tokenized_msa_left_t = torch.from_numpy(
            np.vectorize(esmtok_to_pairformertok_d.get)(esm_tokens_left.cpu().numpy())
        )[:, :, 1:].cpu()
        mp_tokenized_msa_right_t = torch.from_numpy(
            np.vectorize(esmtok_to_pairformertok_d.get)(esm_tokens_right.cpu().numpy())
        )[:, :, 1:].cpu()
        # One hot encode the tokens
        mp_ohe_msa_left_t = torch.nn.functional.one_hot(mp_tokenized_msa_left_t, num_classes=len(aa2tok_d)).to(torch.bfloat16).to(device)
        mp_ohe_msa_right_t = torch.nn.functional.one_hot(mp_tokenized_msa_right_t, num_classes=len(aa2tok_d)).to(torch.bfloat16).to(device)
        tokenized_dataset["msa"]["left"] = mp_ohe_msa_left_t # (1, N, L_left, 28)
        tokenized_dataset["msa"]["right"] = mp_ohe_msa_right_t # (1, N, L_right, 28)

        left_msa, right_msa = tokenized_dataset["msa"]["left"], tokenized_dataset["msa"]["right"]
        positive_examples = tokenized_dataset["positive_examples"] # empty

        # Designate the first sequence in the MSA as the query sequence
        # This will be a positive example and will be unchanged throughout optimization
        num_positive = 1
        positive_examples = torch.cat(
            (left_msa[:, :num_positive, :, :], right_msa[:, :num_positive, :, :]),
            dim=2
        ).to(device)
        left_msa_train = left_msa[:, num_positive:, :, :]
        right_msa_train = right_msa[:, num_positive:, :, :]
        filtered_species_sizes[0] = filtered_species_sizes[0] - 1
        # Make sure the first species has at least 2 sequences
        if filtered_species_sizes[0] < 2:
            filtered_species_sizes = filtered_species_sizes[1:]
            left_msa_train = left_msa_train[:, 1:, :, :]
            right_msa_train = right_msa_train[:, 1:, :, :]

            
        # Get target loss
        parameters_target_loss = {
            "batch_size": 20
        }

        # Run 20 times with 20 different seeds for 20 epochs each
        final_log_alpha_l = []
        n_runs = 20
        n_epochs = 20
        parameters_train = {
            "std_init": 0.1,
            "scheduler_name": "ReduceLROnPlateau",
            "scheduler_kwargs": {"mode": "min", "factor": 0.8, "patience": 20},
            "optimizer_name": "Adadelta",
            "optimizer_kwargs": {"lr": 9, "weight_decay": 1e-1},
            "tau": 1.,
            "n_sink_iter": 10,
            "batch_size": 1,
            "epochs": n_epochs,
            "noise": True,
            "noise_factor": 0.1,  # If noise_std is False, this is just the std of the noise
            "noise_scheduler": True,
            "noise_std": True,
            "use_rand_perm": True,
        }
        parameters_train = {
            "std_init": 0.,
            "scheduler_name": "ReduceLROnPlateau",
            "scheduler_kwargs": {"mode": "min", "factor": 0.8, "patience": 20, "min_lr": 1},
            "optimizer_name": "Adadelta",
            "optimizer_kwargs": {"lr": 9, "weight_decay": 1e-1},
            "tau": 1.,
            "n_sink_iter": 10,
            "batch_size": 1,#3,
            "epochs": warm_epochs,
            "noise": True,
            "noise_factor": 0.5,  # If noise_std is False, this is just the std of the noise
            "min_noise_factor": 0.1,
            "noise_scheduler": True,
            "noise_std": True,
            "use_rand_perm": True,
            "optimize_first_for_n_epochs": 0,
            "p_mask": 0.3
        }

        # Start timer
        start_time = time.time()
        for warm_seed in tqdm(range(n_runs)):
            # Initialization parameters
            parameters_init = {
                "model": msa_pairformer,
                "mrf_head": mrf_head,
                "species_sizes": filtered_species_sizes,
                "device": device,
                "random_seed": warm_seed,
                "compile": False,
                "query_biasing": True,
                "potts_layer_idx": 15,
                "update_potts_model_every_n_epochs": 1
            }
            pairing_optimizer = MP_PDP(**parameters_init)
            if warm_seed == 0:
                tar_loss = pairing_optimizer.target_loss(
                    left_msa_train,
                    right_msa_train,
                    positive_examples=positive_examples,
                    **parameters_target_loss
                )
            if not os.path.exists(f"{RESULTS_DIR}/mp_pdp_warm_seed_{warm_seed}"):
                os.makedirs(f"{RESULTS_DIR}/mp_pdp_warm_seed_{warm_seed}")
            (losses, list_scheduler, shuffled_indexes, mat_perm, mat_gs, list_log_alpha) = pairing_optimizer.train(
                left_msa_train,
                right_msa_train,
                positive_examples=positive_examples,
                tar_loss=np.mean(tar_loss),
                output_dir=Path(f"{RESULTS_DIR}/mp_pdp_warm_seed_{warm_seed}"),
                **parameters_train
            )
            final_log_alpha_l.append(list_log_alpha[-1])

        avg_log_alpha = torch.from_numpy(np.stack(final_log_alpha_l, axis=0).mean(axis=0))
        init_log_alpha_t = []
        for species_size in filtered_species_sizes:
            init_log_alpha_t.append(avg_log_alpha[:species_size, :species_size])
            avg_log_alpha = avg_log_alpha[species_size:, species_size:]
        torch.save(init_log_alpha_t, f"{RESULTS_DIR}/mp_pdp_init_log_alpha_t.seed_{seed}.pt")

        # Final run for 400 epochs initializing with the average of 20 runs
        parameters_train = {
            "std_init": 0.,
            "scheduler_name": "ReduceLROnPlateau",
            "scheduler_kwargs": {"mode": "min", "factor": 0.8, "patience": 20},
            "optimizer_name": "Adadelta",
            "optimizer_kwargs": {"lr": 9, "weight_decay": 1e-1},
            "tau": 1.,
            "n_sink_iter": 10,
            "batch_size": 1,
            "epochs": main_epochs,
            "noise": True,
            "noise_factor": 0.1,  # If noise_std is False, this is just the std of the noise
            "noise_scheduler": True,
            "noise_std": True,
            "use_rand_perm": True,
            "starting_log_alpha": init_log_alpha_t,
            "init_log_alpha": False
        }
        TORCH_SEED = 6
        parameters_init = {
            "model": msa_pairformer,
            "mrf_head": mrf_head,
            "species_sizes": filtered_species_sizes,
            "device": device,
            "random_seed": TORCH_SEED,
            "compile": False,
            "query_biasing": True,
            "potts_layer_idx": 15,
            "update_potts_model_every_n_epochs": 1
        }
        pairing_optimizer = MP_PDP(**parameters_init)
        (losses, list_scheduler, shuffled_indexes, mat_perm, mat_gs, list_log_alpha) = pairing_optimizer.train(
            left_msa_train,
            right_msa_train,
            positive_examples=positive_examples,
            tar_loss=np.mean(tar_loss),
            output_dir=Path(f"{RESULTS_DIR}/"),
            **parameters_train
        )
        # End timer
        end_time = time.time()
        res_d = {
            'losses': losses,
            'list_scheduler': list_scheduler,
            'shuffled_indexes': shuffled_indexes,
            'mat_perm': mat_perm,
            'mat_gs': mat_gs,
            'list_log_alpha': list_log_alpha,
            'total_runtime': end_time - start_time
        }
        with open(Path(f"{RESULTS_DIR}/mp_pdp_results.seed_{seed}.pkl"), "wb") as f:
            pickle.dump(res_d, f)
        
        # Count values on the diagonal (where prediction matches ground truth)
        diagonal_values = np.trace(mat_perm[-1])
        print("Number of correct pairs: {}/{} ({}%)".format(int(diagonal_values), mat_perm[-1].shape[0], round(diagonal_values/mat_perm[-1].shape[0]*100, 1)))
        # Count values on the diagonal (where prediction matches ground truth)
        q = 400
        q_lowest_losses = np.argsort(losses)[:q]
        avg_perm_mat = torch.stack([torch.from_numpy(mat_perm[i]) for i in q_lowest_losses]).mean(dim=0)
        row_ind, col_ind = linear_sum_assignment(-avg_perm_mat)
        final_perm_mat = torch.zeros_like(avg_perm_mat)
        final_perm_mat[row_ind, col_ind] = 1
        print("\tNumber of correct pairs: {}".format(np.trace(final_perm_mat) / np.sum(final_perm_mat.shape[0])))

    peak_alloc = torch.cuda.max_memory_allocated()
    peak_resv  = torch.cuda.max_memory_reserved()
    memory_stats_d = {
        "peak_alloc_GB": peak_alloc / 1024**3,
        "peak_resv_GB":  peak_resv  / 1024**3,
        "incr_peak_alloc_GB": (peak_alloc - baseline_alloc) / 1024**3,
        "incr_peak_resv_GB":  (peak_resv  - baseline_resv)  / 1024**3,
    }
    print(f"Peak GPU VRAM allocated: {peak_alloc / 1024**3:.2f} GB")
    print(f"Peak GPU VRAM reserved: {peak_resv / 1024**3:.2f} GB")
    print(f"Peak GPU VRAM allocated - baseline: {(peak_alloc - baseline_alloc) / 1024**3:.2f} GB")
    print(f"Peak GPU VRAM reserved - baseline: {(peak_resv - baseline_resv) / 1024**3:.2f} GB")
    with open(Path(f"{RESULTS_DIR}/mp_pdp_memory_usage.pkl"), "wb") as f:
        pickle.dump(memory_stats_d, f)

if __name__ == "__main__":
    main()