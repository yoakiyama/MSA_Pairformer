# Adapted from https://github.com/Bitbol-Lab/DiffPALM/blob/main/diffpalm/msa_parsing.py
# Original author: Umberto Lupo et al. (2024), Pairing interacting protein sequences using masked language modeling

# Progress bars
from tqdm import tqdm
from einops import einsum

# NumPy
import numpy as np

# Torch
import torch
from pathlib import Path
import einx
from scipy.optimize import linear_sum_assignment

from huggingface_hub import snapshot_download

# DiffPALM imports
from .gumbel_sinkhorn_utils import MSA_inverse_permutation

# MSA Pairformer imports
from MSA_Pairformer.model import MSAPairformer
from MSA_Pairformer.regression import MRFHead
from MSA_Pairformer.dataset import aa2tok_d

def DCN(x):
    return x.detach().clone().cpu().numpy()



class MP_EM(torch.nn.Module):
    """
    Permute all the pairs between two concatenated MSAs (for each species), randomly mask the left MSA
    and compute the MLM loss. Backpropagate the loss on the permutation matrix and iterate the process
    to get the correct permutation of interacting pairs.
    `species_sizes`: list of species sizes for the paired MSA
    `random_seed`: random seed
    `device`: device to use for computations
    `weights_dir`: directory to save/load model weights
    `query_biasing`: if True, use query-biased attention in MSA Pairformer
    `init_potts_model`
    """

    def __init__(
        self,
        species_sizes,
        model=None,
        mrf_head=None,
        random_seed=42,
        device="cuda",
        weights_dir=None,
        query_biasing=True,
        compile=False,
        potts_layer_idx=15,
        init_potts_model=None
    ):
        super().__init__()

        # List of species sizes for the paired MSA
        self.species_sizes = species_sizes

        # Loss function
        self.loss = torch.nn.CrossEntropyLoss()

        # Set random seed
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        # Set device and load MSA Pairformer to device
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        if model is None:
            self.msa_pairformer = MSAPairformer.from_pretrained(
                weights_dir=weights_dir,
                device=self.device
            )
        else:
            self.msa_pairformer = model
        self.msa_pairformer.eval()

        if mrf_head is None:
            self.mrf_head = MRFHead(
                dim_pairwise = 256,
                dim_alphabet = 26
            )
            path = Path(snapshot_download(repo_id="yakiyama/MSA-Pairformer"))
            mrf_head_checkpoint = torch.load(path / "mrf_head.bin", weights_only=True, map_location=self.device)
            # Remove "mrf_head." prefix from every key in mrf_head_checkpoint, if it exists
            mrf_head_checkpoint = {k.replace("mrf_head.", ""): v for k, v in mrf_head_checkpoint.items()}
            self.mrf_head.load_state_dict(mrf_head_checkpoint)
            self.mrf_head.to(device)
        else:
            self.mrf_head = mrf_head

        self.potts_model = init_potts_model
        self.potts_layer_idx = potts_layer_idx

        # Turn off query biasing unless specified
        if not query_biasing:
            self.msa_pairformer.turn_off_query_biasing()
        else:
            self.msa_pairformer.turn_on_query_biasing()

        if compile:
            self.msa_pairformer = torch.compile(self.msa_pairformer, dynamic=True)

    def prepare_msa_masks(self, msa_input, chain_break_idx, coverage_threshold=0.7):
        """
        Prepare masks for MSA Pairformer.
        `msa_input`: one-hot encoded MSA tensor (B, D, L, 28)
        """
        token_ids = msa_input.argmax(dim=-1)  # [b, s, n]
        left_token_ids = token_ids[:, :, :chain_break_idx]
        right_token_ids = token_ids[:, :, chain_break_idx:]

        res_mask = (token_ids != aa2tok_d['<pad>']).any(dim=1)  # [b, n]
        msa_mask = (left_token_ids != aa2tok_d['<pad>']).any(dim=2) | (right_token_ids != aa2tok_d['<pad>']).any(dim=2)  # [b, s]
        full_mask = (token_ids != aa2tok_d['<pad>'])  # [b, s, n]
        pairwise_mask = einx.logical_and('... i, ... j -> ... i j', res_mask, res_mask)  # [b, n, n]

        # Create a column-wise mask based on coverage. Only include columns with coverage >= min_coverage
        coverage_t = torch.sum((token_ids != aa2tok_d['-']) & (token_ids != aa2tok_d['<pad>']), dim=1) / torch.sum((token_ids != aa2tok_d['<pad>']), dim=1)
        coverage_mask = coverage_t >= coverage_threshold # [b, n]

        return res_mask, msa_mask, full_mask, pairwise_mask, coverage_mask

    def update_potts_model(self, input_ord, input_right, positive_examples=None, coverage_threshold=0.7, p_mask=0.0, w_mask=None):
        with torch.no_grad():
            # Concatenate input MSAs and positive examples
            input_msa = torch.cat((input_ord, input_right), dim=2) # (1, N, L1 + L2, 28)
            if positive_examples is not None:
                input_msa = torch.cat((positive_examples, input_msa), dim=1) # (1, M+N, L1 + L2, 28)
            if p_mask > 0:
                input_msa = self.mask_msa(input_msa, p_mask)
            res_mask, msa_mask, full_mask, pairwise_mask, coverage_mask = self.prepare_msa_masks(input_msa, chain_break_idx=input_ord.shape[2], coverage_threshold=coverage_threshold)
            either_paired = ~torch.all(full_mask == 0, dim=-1).squeeze(0) # (n)
            full_mask = full_mask[:, either_paired, :]
            msa_mask = msa_mask[:, either_paired]

            chain_break_idx = input_ord.shape[2]
            batch_size = input_msa.shape[0]
            complex_chain_break_indices = [[chain_break_idx]] * batch_size
            with torch.amp.autocast(dtype=torch.bfloat16, device_type='cuda'):
                # w, b = self.msa_pairformer.get_potts_model(
                #     msa=input_msa.float()[:, either_paired, :, :],
                #     mask=res_mask,
                #     msa_mask=msa_mask,
                #     full_mask=full_mask,
                #     pairwise_mask=pairwise_mask,
                #     complex_chain_break_indices=complex_chain_break_indices,
                # )
                results = self.msa_pairformer(
                    msa=input_msa.float()[:, either_paired, :, :],
                    mask=res_mask,
                    msa_mask=msa_mask,
                    full_mask=full_mask,
                    pairwise_mask=pairwise_mask,
                    return_seq_weights=False,
                    return_pairwise_repr_layer_idx=[self.potts_layer_idx],
                    return_msa_repr_layer_idx=[self.potts_layer_idx],
                    return_cb_contacts=False,
                    return_confind_contacts=False,
                    query_only=False,
                    store_msa_repr_cpu=False,
                    store_pairwise_repr_cpu=False,
                    use_checkpointing_triangles=False,
                    complex_chain_break_indices=complex_chain_break_indices
                )
                w, b = self.mrf_head(
                    results['pairwise_repr_d'][f'layer_{self.potts_layer_idx}'],
                    pairwise_mask,
                    res_mask
                )
                # Zero out intra-chain couplings
                w[:, :chain_break_idx, :, :chain_break_idx, :] = 0
                w[:, chain_break_idx:, :, chain_break_idx:, :] = 0
                if w_mask is not None:
                    # w: (b, n, a, n, a); w_mask: (b, n, a, n, a)
                    # w_mask: (b, n, n)
                    w = w * w_mask
                # Apply coverage mask to w - zero out rows and columns where coverage < threshold
                # w: (b, n, a, n, a); coverage_mask: (b, n) where True means coverage >= threshold
                # We want to zero out positions where coverage is below threshold (i.e., where coverage_mask is False)
                coverage_5d = coverage_mask[:, :, None, None, None] & coverage_mask[:, None, None, :, None]
                w = w * coverage_5d

        # self.potts_model = (w, b)
        return w, b

    def mask_msa(self, input_msa, p_mask):
        vec_mask = torch.zeros(input_msa.shape[-1]).to(input_msa.device)
        vec_mask[aa2tok_d['<mask>']] = 1
        vec_mask = vec_mask.to(input_msa.dtype)
        b, n, l, a = input_msa.shape
        mask = (torch.rand(b, n, l, device=input_msa.device) < p_mask).bool()
        # No mask for query sequence
        mask[:, 0, :] = False
        input_msa[mask] = vec_mask[None, None, None, :]
        return input_msa
        

    def forward(self, input_ord, input_right):
        """
        Mask input MSA and concatenate with fixed MSA. Then compute output logits
        for the masked positions using MSA Pairformer.

        `input_ord`: variable input at each iteration (to be masked)  --> (B, N, L1, 28)
        `input_right`: fixed input (no masking)                       --> (B, N, L2, 28)
        `positive_examples`: if not None it's a concatenation of correct pairs to use
                             as context (not masked)                  --> (B, N, L1 + L2, 28)
        """
        L1 = input_ord.shape[2]

        # Construct MSA containing all pairs of sequences between input_ord and input_right
        # input_ord and input_right are (B, N, L1, 28) and (B, N, L2, 28) respectively
        # constructed MSA is (B, N^2, L1 + L2, 28)
        batch, n, l1, a = input_ord.shape
        l2 = input_right.shape[2]

        # Potts model: zero left-left and right-right couplings
        # Clone w to avoid in-place modification
        w, bias = self.potts_model
        species_costs = []

        offset = 0
        for s_k in self.species_sizes:
            if s_k == 1:
                cost_t = torch.zeros((s_k, s_k)).float().numpy()
                species_costs.append(cost_t)
                offset += s_k
                continue
            left_sp = input_ord[:, offset:offset + s_k]   # (b, s_k, l1, a)
            right_sp = input_right[:, offset:offset + s_k] # (b, s_k, l2, a)

            left_expanded = left_sp.unsqueeze(2).expand(batch, s_k, s_k, l1, a)
            right_expanded = right_sp.unsqueeze(1).expand(batch, s_k, s_k, l2, a)
            input_msa = torch.cat([left_expanded, right_expanded], dim=3) # (b, s_k, s_k, l1+l2, a)
            input_msa = input_msa.reshape(batch, s_k * s_k, l1 + l2, a) # (batch, s_k^2, l1+l2, a)

            both_paired = ~torch.any(input_msa[..., aa2tok_d['<pad>']], dim=-1).squeeze(0) # (s_k^2)
            left_is_real = ~torch.any(input_msa[:, :, :l1, aa2tok_d['<pad>']], dim=-1).squeeze(0) # (s_k^2)

            mrf_logits = bias.unsqueeze(1) + einsum(
                input_msa[:, :, :, :w.shape[-1]], w,
                'b n r s, b r s l k -> b n l k'
            )
            bias_only_logits = bias.unsqueeze(1) + torch.zeros_like(mrf_logits)

            targets = input_msa.argmax(dim=-1)[..., None] # (b, s_k^2, L1+L2, 1)
            targets_masked = targets[:, both_paired, :, :] # (b, s_k^2, L1+L2, 1)
            mrf_logits_masked = mrf_logits[:, both_paired, :, :] # (b, s_k^2, L1+L2, 26)
            bias_only_masked = bias_only_logits[:, both_paired, :, :]

            mrf_pll = mrf_logits_masked.log_softmax(dim=-1).gather(
                dim=-1, index=targets_masked
            ).squeeze(-1).mean(dim=-1)[0] # (s_k^2)
            bias_pll = bias_only_masked.log_softmax(dim=-1).gather(
                dim=-1, index=targets_masked
            ).squeeze(-1).mean(dim=-1)[0] # (s_k^2)

            cost_t = torch.zeros(s_k * s_k, device=input_ord.device, dtype=torch.bfloat16)
            cost_t[both_paired] = bias_pll - mrf_pll # (s_k^2)
            cost_t = cost_t.reshape(s_k, s_k).float().detach().cpu().numpy() # (s_k, s_k)
            species_costs.append(cost_t)
            offset += s_k
        return species_costs
    
    def compute_curr_pseudolikelihood(self, input_left, input_right):
        with torch.no_grad():
            L1 = input_left.shape[2]
            # Identify real vs dummy rows (dummy rows are all pad tokens)
            left_is_dummy = torch.all(input_left[..., aa2tok_d['<pad>']], dim=-1)
            right_is_dummy = torch.all(input_right[..., aa2tok_d['<pad>']], dim=-1)
            left_is_real = ~left_is_dummy.squeeze(0)
            right_is_real = ~right_is_dummy.squeeze(0)
            both_paired = left_is_real & right_is_real # real left and real right
            input_msa = torch.cat((input_left, input_right), dim=2)
            w, b = self.potts_model
            mrf_logits = b.unsqueeze(1) + einsum(input_msa[:, :, :, :w.shape[-1]], w, 'b n r s, b r s l k -> b n l k')
            mrf_logits = mrf_logits[:, both_paired, :, :] # (1, N, L1+L2, 26)
            targets = input_msa.argmax(dim=-1)[...,None] # (1, N, L1+L2, 1)
            targets = targets[:, both_paired, :, :] # (1, N, L1+L2, 1)
            pll = mrf_logits.log_softmax(dim=-1).gather(dim=-1, index=targets).squeeze(-1).sum(dim=-1) # (1, N)
            return -pll.mean().item()

    def train(
        self,
        input_left,
        input_right,
        positive_examples=None,
        epochs=1,
        output_dir=None,
        coverage_threshold_potts_model=0.7,
        random_init=False,
        noise_scheduler="linear",
        max_noise=0.8,
        min_noise=0.0,
        potts_model_ema_alpha=0.5,
        n_perm_thresh=5,
        n_consensus_steps=200,
        p_mask=0.0,
        cost_bias=None,
        w_mask=None,
        init_perm_mat=None
    ):
        """
        Train the model using the input MSAs (`input_left`, `input_right`) and the fixed pairings.
        `input_left`: input left MSA
        `input_right`: input right MSA
        `positive_examples`: positive examples to use
        `coverage_threshold_potts_model`: coverage threshold for the Potts model; exclude columns with coverage < threshold
        `random_init`: if True, randomly initialize the pairings and Potts model 
            (in practice, you may want to set this to False if using paired MSAs from something like AlphaFold-Multimer)
        `noise_scheduler`: scheduler for the noise
        `max_noise`: maximum noise
        `min_noise`: minimum noise
        `potts_model_ema_alpha`: exponential moving average alpha for the Potts model
        `n_perm_thresh`: stop if permutation matrices have not changed in this many steps
        `n_consensus_steps`: number of steps to run the consensus algorithm
        `epochs`: number of epochs of the training
        `output_dir`: if not None save the plots in this directory
        `n_perm_thresh`: stop if permutation matrices have not changed in this many steps
        `n_consensus_steps`: number of steps to run the consensus algorithm
        `p_mask`: probability of masking a sequence
        `cost_bias`: bias to add to the cost matrix
        `w_mask`: mask to apply to the Potts model
        `init_perm_mat`: initial permutation matrix to use
        Outputs:
        `losses`: list of loss values for each iteration (`batch_size`*`epochs`)
        `list_lr`: list of the learning rate used at each epoch
        `list_idx`: list of the indexes of the predicted pairs at each iteration (`batch_size`*`epochs`)
        `mats`: list of the permutation matrices at each epoch (hard permutation)
        `mats_gs`: list of the soft-permutation matrices at each epoch
        `list_log_alpha`: list of the log_alpha matrices at each epoch
        """
        # ------------------------------------------------------------------------------------------
        ## Input MSAs and initial variables
        # ------------------------------------------------------------------------------------------
        input_left = input_left.to(self.device)
        input_right = input_right.to(self.device)

        # Lists of parameters
        losses = []
        perm_mats = []
        cost_ts = []
        unchanged_count = 0  # Track consecutive unchanged permutations
        # ------------------------------------------------------------------------------------------
        ## Initializations
        # ------------------------------------------------------------------------------------------

        if output_dir is not None:
            (output_dir / "Iterations").mkdir(exist_ok=True)

        # Initialize Potts model
        if random_init:
            per_species_mats = []
            permuted_slices = []
            offset = 0
            for i, s_k in enumerate(self.species_sizes):
                cost = torch.randn(s_k, s_k)
                if cost_bias is not None:
                    curr_bias = cost_bias[i]
                    cost += curr_bias
                row_idx, col_idx = linear_sum_assignment(cost)
                mat = torch.zeros(s_k, s_k, device=input_left.device)
                mat[row_idx, col_idx] = 1.0
                per_species_mats.append(mat)
                left_sp = input_left[:, offset:offset + s_k]
                permuted_slices.append(MSA_inverse_permutation(left_sp, mat.to(input_left.dtype)))
                offset += s_k
            perm_mat = torch.zeros(sum(self.species_sizes), sum(self.species_sizes))
            offset = 0
            for species_idx, s_k in enumerate(self.species_sizes):
                perm_mat[offset:offset + s_k, offset:offset + s_k] = per_species_mats[species_idx]
                offset += s_k
            perm_mats.append(perm_mat)#.to(torch.int8))
            # Cast permutation matrices to match input dtype for einsum operation
            input_left_ord_hard = torch.cat(permuted_slices, dim=1)
        else:
            assert init_perm_mat is not None
            input_left_ord_hard = MSA_inverse_permutation(input_left, init_perm_mat.to(input_left.dtype).to(input_left.device))
            perm_mats.append(init_perm_mat)
        self.potts_model = self.update_potts_model(input_left_ord_hard, input_right, positive_examples=positive_examples, coverage_threshold=coverage_threshold_potts_model, p_mask=p_mask, w_mask=w_mask)
        curr_pll = self.compute_curr_pseudolikelihood(input_left_ord_hard, input_right)
        losses.append(curr_pll)

        # ------------------------------------------------------------------------------------------
        ## Start training
        # ------------------------------------------------------------------------------------------
        iterations = epochs
        for i in tqdm(range(iterations)):
            # Compute costs of all pairs of sequences given current Potts model
            with torch.no_grad():
                species_costs = self(input_left, input_right)
            # Compute next hard permutation matrix
            permuted_slices = []
            per_species_mats = []
            offset = 0
            if noise_scheduler == "linear":
                # Linear decay of noise from 1 to 0 over iterations
                noise_scale = 1.0 - (i / iterations) if iterations > 0 else 0.0
            elif noise_scheduler == "constant":
                noise_scale = 1
            elif noise_scheduler == "cosine":
                noise_scale = min_noise + (max_noise - min_noise) * 0.5 * (1 + np.cos(np.pi * i / iterations)) if iterations > 0 else min_noise
            else:
                noise_scale = 0
            for species_idx, s_k in enumerate(self.species_sizes):
                cost = species_costs[species_idx]
                # Noise cost matrix with linear decay
                noise_cost = np.random.randn(s_k, s_k) * cost.std() * noise_scale
                cost = cost + noise_cost
                if cost_bias is not None:
                    curr_bias = cost_bias[species_idx]
                    cost += curr_bias
                row_idx, col_idx = linear_sum_assignment(cost)
                mat = torch.zeros(s_k, s_k, device=input_left.device)
                mat[row_idx, col_idx] = 1.0
                per_species_mats.append(mat)
                left_sp = input_left[:, offset:offset + s_k]
                permuted_slices.append(MSA_inverse_permutation(left_sp, mat.to(input_left.dtype)))
                offset += s_k
                # Replace species cost with updated cost
                species_costs[species_idx] = cost
            # per_species_mats is a list of permutation matrices for each species
            # Create a single permutation matrix for all species
            perm_mat = torch.zeros(sum(self.species_sizes), sum(self.species_sizes))
            offset = 0
            for species_idx, s_k in enumerate(self.species_sizes):
                perm_mat[offset:offset + s_k, offset:offset + s_k] = per_species_mats[species_idx]
                offset += s_k
            
            # Check if permutation matrices have changed
            perm_mat = perm_mat
            if len(perm_mats) > 0:
                prev_mats = perm_mats[-1]
                mats_unchanged = torch.allclose(prev_mats, perm_mat)
                if mats_unchanged:
                    unchanged_count += 1
                else:
                    unchanged_count = 0
            perm_mats.append(perm_mat)

            # Update Potts model given current permutation matrix
            input_left_ord_hard = torch.cat(permuted_slices, dim=1)
            new_w, new_b = self.update_potts_model(input_left_ord_hard, input_right, positive_examples=positive_examples, coverage_threshold=coverage_threshold_potts_model, p_mask=p_mask, w_mask=w_mask)
            self.potts_model = (
                self.potts_model[0] * potts_model_ema_alpha + new_w * (1 - potts_model_ema_alpha),
                self.potts_model[1] * potts_model_ema_alpha + new_b * (1 - potts_model_ema_alpha)
            )
            # Compute new pseudolikelihood
            curr_pll = self.compute_curr_pseudolikelihood(input_left_ord_hard, input_right)
            losses.append(curr_pll)
            cost_ts.append(species_costs)
            
            # Early stopping if permutations unchanged for n_perm_thresh steps
            if unchanged_count >= n_perm_thresh:
                break
        return (
            losses,
            perm_mats,
            cost_ts
        )