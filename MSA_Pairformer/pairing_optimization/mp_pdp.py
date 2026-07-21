# Adapted from https://github.com/Bitbol-Lab/DiffPALM/blob/main/diffpalm/msa_parsing.py
# Original author: Umberto Lupo et al. (2024), Pairing interacting protein sequences using masked language modeling

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from copy import deepcopy
from huggingface_hub import snapshot_download
from tqdm import tqdm
from einops import einsum
from matplotlib.colors import CenteredNorm

# Torch
import torch
import einx

# DiffPALM imports
from MSA_Pairformer.pairing_optimization.gumbel_sinkhorn_utils import (
    gumbel_sinkhorn,
    gumbel_matching,
    MSA_inverse_permutation,
    sample_uniform,
    no_noise_matching,
)

from MSA_Pairformer.model import MSAPairformer
from MSA_Pairformer.dataset import aa2tok_d
from MSA_Pairformer.regression import MRFHead

def DCN(x):
    return x.detach().clone().cpu().numpy()

class PermutationsMixin:
    """Mixin class for validating input and plotting the results of the optimization."""

    mask_idx = aa2tok_d['<mask>']

    def _init_log_alpha(self, skip=False):
        """Intialize log_alpha as a list of matrices of shape (d, d) where d is the
        depth of the species MSA. The matrices are initialized with standard normal entries.
            - log_alpha will eventually become the permutation matrix that sorts the sequences in the left MSA
        """
        if not skip:
            self.log_alpha = []
            for d in self._effective_depth_not_fixed:
                la = self.std_init * torch.randn(d, d, device=self.device)
                # la += torch.eye(d, device=self.device) # Not for evaluation, but for practical use, rank-order is likely useful (if initial pairing is somewhat correct). 
                # For evaluations in the paper, this would give us 100% correct pairs, so we turn it off
                self.log_alpha.append(la.requires_grad_(True))


    def _validator(self, input_left, input_right, fixed_pairings=None):
        """
        Validate input MSAs and check fixed pairings.
            - Positive examples (known interacting pairs) remain fixed
            - _effective_mask_not_fixed is a boolean mask that is True for the positions that are not fixed
            - plot_real_time plots the permutation matrices (soft and hard) and the loss curve during training
        """
        # Validate input MSAs
        depth_left, length_left, alphabet_size_left = input_left.shape[1:]
        depth_right, length_right, alphabet_size_right = input_right.shape[1:]
        if depth_left != depth_right:
            raise ValueError(
                f"Depth mismatch between left MSA ({depth_left}) and right MSA "
                f"({depth_right})"
            )
        if alphabet_size_left != alphabet_size_right:
            raise ValueError("Input MSAs must have the same alphabet size/")
        self._alphabet_size = alphabet_size_left

        # Define oh vector for mask token
        self._vec_mask = torch.zeros(self._alphabet_size, device=self.device)
        self._vec_mask[self.mask_idx] = 1

        # Validate depth attribute
        self._depth_total = sum(self.species_sizes)
        if depth_left != self._depth_total:
            raise ValueError(
                f"Input MSAs have depth {depth_left} but model expects a total "
                f"depth of {self._depth_total}"
            )
        self._length_left, self._length_right = length_left, length_right
        self._length = length_left + length_right

        self._effective_mask_not_fixed = torch.ones(
            self._depth_total, self._depth_total, dtype=torch.bool, device=self.device
        )

        # Create masking array for non-fixed partial rows in concatenated MSA
        self._effective_mask_not_fixed_cat = torch.ones(
            1, self._depth_total, self._length, dtype=torch.bool, device=self.device
        )
        if fixed_pairings is not None:
            if len(fixed_pairings) != len(self.species_sizes):
                raise ValueError(
                    f"`fixed_pairings` has length {len(fixed_pairings)} but "
                    f"there are {self.species_sizes} species."
                )
            _fixed_pairings = fixed_pairings

            start = 0
            self._effective_depth_not_fixed = []
            self._effective_fixed_pairings_zip = []
            for species_idx, (species_size, species_fixed_pairings) in enumerate(
                zip(self.species_sizes, _fixed_pairings)
            ):
                # Check uniqueness of pairs (i, j)
                n_fixed = len(set(species_fixed_pairings))
                if len(species_fixed_pairings) > n_fixed:
                    raise ValueError(
                        "Repeated indices for fixed pairings at species "
                        f"{species_idx}: {species_fixed_pairings}"
                    )
                fixed_pairings_arr = np.zeros((species_size, species_size), dtype=int)
                if species_fixed_pairings:
                    species_fixed_pairings_zip = tuple(zip(*species_fixed_pairings))
                else:
                    # species_fixed_pairings is an empty list
                    species_fixed_pairings_zip = (tuple(), tuple())
                try:
                    fixed_pairings_arr[species_fixed_pairings_zip] = 1
                except IndexError:
                    raise ValueError(
                        f"Fixed pairings indices out of bounds: passed {species_fixed_pairings} "
                        f"for species {species_idx} with size {species_size}."
                    )
                partial_sum_0 = fixed_pairings_arr.sum(axis=0)
                partial_sum_1 = fixed_pairings_arr.sum(axis=1)
                if (partial_sum_0 > 1).any() or (partial_sum_1 > 1).any():
                    raise ValueError(
                        f"Passed fixed pairings for species {species_idx} are either not one-one "
                        "or a multiply-defined mapping from row to column indices: "
                        f"{species_fixed_pairings}"
                    )
                for i, j in species_fixed_pairings:
                    self._effective_mask_not_fixed[start + i, :] = False
                    self._effective_mask_not_fixed[:, start + j] = False
                total_minus_fixed = species_size - n_fixed
                # If species_size - n_fixed <= 1 then actually everything is fixed
                self._effective_depth_not_fixed.append(
                    int(total_minus_fixed > 1) * total_minus_fixed
                )
                if total_minus_fixed == 1:
                    # Deduce implicitly fixed pair
                    i_implicit, j_implicit = np.argmin(partial_sum_1), np.argmin(
                        partial_sum_0
                    )
                    self._effective_mask_not_fixed[start + i_implicit, :] = False
                    self._effective_mask_not_fixed[:, start + j_implicit] = False
                    species_fixed_pairings_zip = (
                        species_fixed_pairings_zip[0] + (i_implicit,),
                        species_fixed_pairings_zip[1] + (j_implicit,),
                    )
                self._effective_fixed_pairings_zip.append(species_fixed_pairings_zip)
                start += species_size
            start = 0
            for species_size, (rows_fixed, cols_fixed) in zip(
                self.species_sizes, self._effective_fixed_pairings_zip
            ):
                self._effective_mask_not_fixed_cat[:, start:, ...][
                    :, rows_fixed, :length_left
                ] = False
                self._effective_mask_not_fixed_cat[:, start:, ...][
                    :, cols_fixed, length_left:
                ] = False
                start += species_size
        else:
            self._effective_depth_not_fixed = self.species_sizes
            self._effective_fixed_pairings_zip = None

        self._default_target_idx = torch.arange(
            self._depth_total, dtype=torch.int64, device=self.device
        )

    def plot_real_time(
        self,
        it,
        gs_matching_mat_np,
        gs_mat_np,
        list_idx,
        target_idx,
        list_log_alpha,
        losses,
        batch_size,
        epochs,
        lr,
        tar_loss,
        new_noise_factor,
        output_dir,
        only_loss_plot,
    ):
        """Plot the results of the optimization in real time."""
        n_correct = [sum(idx == target_idx) for idx in list_idx[::batch_size]]

        cmap = cm.get_cmap("Blues")
        normalizer = None
        fig, axes = plt.subplots(figsize=(30, 5), ncols=5)

        null_model = 1
        null_model = len(self.species_sizes)
        _depth = [0] + list(np.cumsum(self.species_sizes))
        for k in range(1, len(_depth)):
            for ii in range(2):
                elem, elem1 = _depth[k - 1], _depth[k]
                axes[ii].plot(
                    [elem - 0.5, elem1 - 0.5, elem1 - 0.5, elem - 0.5],
                    [elem - 0.5, elem - 0.5, elem1 - 0.5, elem1 - 0.5],
                    color="r",
                )
                axes[ii].plot(
                    [elem - 0.5, elem - 0.5, elem1 - 0.5, elem1 - 0.5],
                    [elem - 0.5, elem1 - 0.5, elem1 - 0.5, elem - 0.5],
                    color="r",
                )

        ims_soft = axes[0].imshow(gs_mat_np, cmap=cmap, norm=normalizer)
        axes[0].set_title(f"Soft {it // batch_size}")
        axes[1].imshow(gs_matching_mat_np, cmap=cmap, norm=normalizer)
        axes[1].set_title("Hard")

        curr_log_alpha = list_log_alpha[-1]
        ims_log_alpha = axes[2].imshow(curr_log_alpha, norm=CenteredNorm(), cmap=cm.bwr)
        axes[2].set_title("Log-alpha")

        prev_log_alpha = (
            list_log_alpha[-2] if len(list_log_alpha) > 1 else curr_log_alpha
        )
        diff_log_alpha = curr_log_alpha - prev_log_alpha
        if np.nansum(np.abs(diff_log_alpha)):
            ims_log_alpha_diff = axes[3].imshow(
                diff_log_alpha, norm=CenteredNorm(), cmap=cm.bwr
            )
            cb3 = fig.colorbar(ims_log_alpha_diff, ax=axes[3], shrink=0.8)
        else:
            ims_log_alpha_diff = axes[3].imshow(
                np.zeros_like(diff_log_alpha), cmap=cm.bwr
            )
        axes[3].set_title("Log-alpha diff")

        avg_loss = np.mean(np.array(losses).reshape(-1, batch_size), axis=1)
        axes[4].plot(avg_loss, color="b", linewidth=1)
        ax3_2 = None
        if not only_loss_plot:
            if tar_loss is not None:
                axes[4].axhline(tar_loss, color="b", linewidth=2, linestyle="--", label="Target loss")
            diff = avg_loss[0] - tar_loss
            axes[4].set_ylim([tar_loss - 0.6 * diff, avg_loss[0] + 0.5 * diff])
            ax3_2 = axes[4].twinx()
            ax3_2.set_ylabel("No. of correct pairs", color="red")
            ax3_2.plot(n_correct, color="red", linewidth=1)
            ax3_2.axhline(null_model, color="red", linewidth=2, linestyle="--", label="Null model")
            ax3_2.tick_params(axis="y", labelcolor="red")
            
        axes[4].set_ylabel("Loss")
        axes[4].set_xlim([0, epochs])
        axes[4].set_title(f"lr: {lr:.3g}, noise:{new_noise_factor:.3g}")
        fig.colorbar(ims_soft, ax=axes[0], shrink=0.8)
        fig.colorbar(ims_log_alpha, ax=axes[2], shrink=0.8)

        if output_dir is not None:
            fig.savefig(output_dir / "Iterations" / f"Epoch={it // batch_size}.svg", bbox_inches='tight')
        plt.show()
        plt.close(fig)

class MP_PDP(torch.nn.Module, PermutationsMixin):
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
        *,
        model=None,
        mrf_head=None,
        random_seed=42,
        device="cuda",
        weights_dir=None,
        query_biasing=True,
        compile=False,
        potts_layer_idx=15,
        update_potts_model_every_n_epochs=10,
        init_potts_model=None,
    ):
        super().__init__()

        # List of species sizes for the paired MSA
        self.species_sizes = species_sizes

        # Loss function
        self.loss = torch.nn.CrossEntropyLoss()

        # Set random seed
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)

        # Update Potts model every n epochs
        self.update_potts_model_every_n_epochs = update_potts_model_every_n_epochs

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

        self.potts_model = init_potts_model
        self.potts_layer_idx = potts_layer_idx

        if mrf_head is None:
            mrf_head = MRFHead(
                dim_pairwise = 256,
                dim_alphabet = 26
            )
            path = Path(snapshot_download(repo_id="yakiyama/MSA-Pairformer"))
            mrf_head_checkpoint = torch.load(path / "mrf_head.bin", weights_only=True, map_location=self.device)
            # Remove "mrf_head." prefix from every key in mrf_head_checkpoint, if it exists
            mrf_head_checkpoint = {k.replace("mrf_head.", ""): v for k, v in mrf_head_checkpoint.items()}
            mrf_head.load_state_dict(mrf_head_checkpoint)
            mrf_head.to(self.device)
            self.mrf_head = mrf_head
        else: 
            self.mrf_head = mrf_head

        # Turn off query biasing unless specified
        if not query_biasing:
            self.msa_pairformer.turn_off_query_biasing()
        else:
            self.msa_pairformer.turn_on_query_biasing()

        if compile:
            self.msa_pairformer = torch.compile(self.msa_pairformer, dynamic=True)

    def prepare_msa_masks(self, msa_input, chain_break_idx, coverage_threshold=0.75):
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

        res_mask = (token_ids != aa2tok_d['-']).any(dim=1)  # [b, n]
        msa_mask = (left_token_ids != aa2tok_d['-']).any(dim=2) | (right_token_ids != aa2tok_d['-']).any(dim=2)  # [b, s]
        # msa_mask = (token_ids != aa2tok_d['<pad>']).any(dim=2)  # [b, s]
        full_mask = (token_ids != aa2tok_d['<pad>'])  # [b, s, n]
        pairwise_mask = einx.logical_and('... i, ... j -> ... i j', res_mask, res_mask)  # [b, n, n]

        # Create a column-wise mask based on coverage. Only include columns with coverage >= min_coverage
        coverage_t = torch.sum((token_ids != aa2tok_d['-']) & (token_ids != aa2tok_d['<pad>']), dim=1) / torch.sum((token_ids != aa2tok_d['<pad>']), dim=1)
        coverage_mask = coverage_t >= coverage_threshold # [b, n]

        return res_mask, msa_mask, full_mask, pairwise_mask, coverage_mask

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


    def update_potts_model(self, input_ord, input_right, positive_examples=None, coverage_threshold=0.75, p_mask=0.0):
        with torch.no_grad():
            # Concatenate input MSAs and positive examples
            input_msa = torch.cat((input_ord, input_right), dim=2) # (1, N, L1 + L2, 28)
            if positive_examples is not None:
                input_msa = torch.cat((positive_examples, input_msa), dim=1) # (1, M+N, L1 + L2, 28)
            if p_mask > 0:
                input_msa = self.mask_msa(input_msa, p_mask)
            res_mask, msa_mask, full_mask, pairwise_mask, coverage_mask = self.prepare_msa_masks(input_msa, chain_break_idx=input_ord.shape[2], coverage_threshold=coverage_threshold)

            chain_break_idx = input_ord.shape[2]
            batch_size = input_msa.shape[0]
            complex_chain_break_indices = [[chain_break_idx]] * batch_size
            with torch.amp.autocast(dtype=torch.bfloat16, device_type='cuda'):
                results = self.msa_pairformer(
                    msa=input_msa.float(),
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
                # Apply coverage mask to w - zero out rows and columns where coverage < threshold
                # w: (b, n, a, n, a); coverage_mask: (b, n) where True means coverage >= threshold
                # We want to zero out positions where coverage is below threshold (i.e., where coverage_mask is False)
                coverage_5d = coverage_mask[:, :, None, None, None] & coverage_mask[:, None, None, :, None]
                w = w * coverage_5d

        self.potts_model = (w, b)
        return 0

    def forward(self, input_ord, input_right, positive_examples=None, use_checkpointing_triangles=True):
        """
        Mask input MSA and concatenate with fixed MSA. Then compute output logits
        for the masked positions using MSA Pairformer.

        `input_ord`: variable input at each iteration (to be masked)  --> (B, D, L1, 28)
        `input_right`: fixed input (no masking)                       --> (B, D, L2, 28)
        `positive_examples`: if not None it's a concatenation of correct pairs to use
                             as context (not masked)                  --> (B, D, L1 + L2, 28)
        """
        n_pos = positive_examples.shape[1] if positive_examples is not None else 0
        pad_idx = aa2tok_d['<pad>']
        gap_idx = aa2tok_d['-']

        # Build input MSA and prepend positive examples
        input_msa = torch.cat((input_ord, input_right), dim=2)
        if positive_examples is not None:
            input_msa = torch.cat((positive_examples, input_msa), dim=1)
        
        # Potts model: zero left-left and right-right couplings
        # Clone w to avoid in-place modification
        w, b = self.potts_model

        # Compute logits using Potts model
        mrf_logits = b + einsum(
            input_msa[:, :, :, :w.shape[-1]], w,
            'b n r s, b r s l k -> b n l k'
        )
        mrf_logits = mrf_logits[:, n_pos:, :, :]
        targets = input_msa[:, n_pos:, :, :].argmax(dim=-1)
        targets = torch.where(targets == pad_idx, gap_idx, targets)
        pll = mrf_logits.log_softmax(dim=-1).gather(
            dim=-1, index=targets.unsqueeze(-1)
        ).squeeze(-1).sum(dim=-1)
        return -pll.mean()


    def train(
        self,
        input_left,
        input_right,
        starting_log_alpha=None, # If not None, use this log_alpha as starting point for the training
        fixed_pairings=None,  # Format: list of lists of pairs of paired indices relative to each species [[(i, j), ...], ...]
        positive_examples=None,
        init_log_alpha=True,
        std_init=0.1,
        epochs=1,
        optimizer_name="Adadelta",
        optimizer_kwargs=None,
        tau=1.0,
        n_sink_iter=10,
        noise=True,
        noise_std=False,
        noise_factor=0.1,
        min_noise_factor=0.1,
        noise_scheduler=False,
        scheduler_name="ReduceLROnPlateau",
        scheduler_kwargs=None,
        batch_size=1,
        use_rand_perm=False,
        mean_centering=True,
        tar_loss=None,
        output_dir=None,
        save_all_figs=False,
        only_loss_plot=False,
        use_checkpointing_triangles=True,
        init_potts_starting_point=False,
        optimize_first_for_n_epochs=0,
        optimize_for_n_epochs_end=0,
        coverage_threshold_potts_model=0.75,
        p_mask=0.0,
        cost_bias=None
    ):
        """
        Train the model using the input MSAs (`input_left`, `input_right`) and the fixed pairings.

        `fixed_pairings`: list of lists of pairs of paired indices relative to each species
                            Format: [[(i, j), ...], ...]
        `init_log_alpha`: if True initialize log_alpha with random values
        `std_init`: standard deviation of the normal distribution used to initialize log_alpha
        `epochs`: number of epochs of the training
        `optimizer_name`: name of the optimizer to use
        `optimizer_kwargs`: kwargs of the optimizer
        `tau`: temperature parameter for the Sinkhorn operator
        `n_sink_iter`: number of Sinkhorn iterations
        `noise`: if True add noise to the Gumbel-Matching algorithm
        `noise_std`: if True use a fixed noise_std for the noise matrices
        `noise_factor`: noise correction factor
        `min_noise_factor`: minimum value for the noise factor
        `noise_scheduler`: if True use the optimizer learning rate to scale the noise_factor
        `scheduler_name`: name of the learning rate scheduler to use
        `scheduler_kwargs`: kwargs of the scheduler
        `batch_size`: batch size for the training (number of different masks to use at each epoch)
        `use_rand_perm`: if True use random permutations on the input MSAs to change the order of
                         the sequences at each epoch
        `mean_centering`: if True mean-center log_alphas at each epoch
        `tar_loss`: if not None use this value as target loss for the training
        `output_dir`: if not None save the plots in this directory
        `save_all_figs`: if True save all the plots at each batch_size
        `only_loss_plot`: if True save only the loss plot at each batch_size
        `init_potts_starting_point`: if True initialize the Potts model using the initial pairing of the input MSAs
        `p_mask`: token masking probability for left MSA
        `cost_bias`: cost bias for the training
        Outputs:
        `losses`: list of loss values for each iteration (`batch_size`*`epochs`)
        `list_lr`: list of the learning rate used at each epoch
        `list_idx`: list of the indexes of the predicted pairs at each iteration (`batch_size`*`epochs`)
        `mats`: list of the permutation matrices at each epoch (hard permutation)
        `mats_gs`: list of the soft-permutation matrices at each epoch
        `list_log_alpha`: list of the log_alpha matrices at each epoch
        """
        self._validator(input_left, input_right, fixed_pairings=fixed_pairings)
        if not sum(self._effective_depth_not_fixed):
            print(
                "No parameters available to optimize, pairings are fixed by the input."
            )
            return None
        self.std_init = std_init

        base_params = {"noise": noise, "noise_std": noise_std}
        sinkhorn_params = {"tau": tau, "n_iter": n_sink_iter}

        # Initialize log_alpha given fixed pairings
        if starting_log_alpha is not None:
            assert init_log_alpha is False, "Cannot initialize log_alpha and use starting_log_alpha"
            self.log_alpha = [la.clone().detach().to(self.device).requires_grad_(True) for la in starting_log_alpha]
        elif init_log_alpha:
            self._init_log_alpha()

        # ------------------------------------------------------------------------------------------
        ## Useful functions
        # ------------------------------------------------------------------------------------------
        def _apply_species_wise(func):
            # Apply `func` to the blocks for permutations restricted to species
            def _impl(log_alpha, **params):
                # Block matrix for permutations within species
                noise_mat = params.pop("noise_mat")  # List of noise matrices
                rand_perm = params.pop("rand_perm")  # List of random permutations
                return torch.block_diag(
                    *[
                        func(la, noise_mat=nm, rand_perm=rp, cost_bias=None, **params)
                        if la.size(0)
                        else la
                        for la, nm, rp in zip(log_alpha, noise_mat, rand_perm)
                    ]
                )

            return _impl

        def _apply_species_wise_no_noise(func):
            def _impl(log_alpha):
                return torch.block_diag(
                    *[
                        func(la) if la.size(0) else la
                        for la in log_alpha
                    ]
                )

            return _impl

        def _noise_mat():
            if noise:
                return [
                    sample_uniform(la.size()).to(self.device) for la in self.log_alpha
                ]

            return [None for la in self.log_alpha]

        def _rand_perm():
            if use_rand_perm:
                rand_perm = []
                for la in self.log_alpha:
                    n = la.shape[0]
                    rp = []
                    for _ in range(2):
                        rp_i = torch.zeros_like(la, device=self.device)
                        rp_i[torch.arange(n), torch.randperm(n)] = 1
                        rp.append(rp_i)
                    rand_perm.append(rp)
            else:
                rand_perm = [None] * len(self.log_alpha)

            return rand_perm

        gumbel_matching_species_wise = _apply_species_wise(gumbel_matching)
        gumbel_sinkhorn_species_wise = _apply_species_wise(gumbel_sinkhorn)

        # ------------------------------------------------------------------------------------------
        ## Input MSAs and initial variables
        # ------------------------------------------------------------------------------------------
        input_left = input_left.to(self.device)
        input_right = input_right.to(self.device)

        # Lists of parameters
        losses = []
        mats, mats_gs = [], []
        list_idx = []
        list_log_alpha = []
        list_lr = []
        gs_matching_mat = None
        target_idx = torch.arange(
            self._depth_total, dtype=torch.float, device=self.device
        )
        target_idx_np = DCN(target_idx)
        # ------------------------------------------------------------------------------------------
        ## Initializations
        # ------------------------------------------------------------------------------------------
        # Optimizer
        optimizer_params = [{"params": la} for la in self.log_alpha]
        optimizer_kwargs_ = (
            {} if optimizer_kwargs is None else deepcopy(optimizer_kwargs)
        )
        optimizer_cls = getattr(torch.optim, optimizer_name, torch.optim.SGD)
        optimizer = optimizer_cls(optimizer_params, **optimizer_kwargs_)
        # Scheduler
        if scheduler_name is not None:
            scheduler_cls = getattr(
                torch.optim.lr_scheduler,
                scheduler_name,
                torch.optim.lr_scheduler.ReduceLROnPlateau,
            )
            scheduler_kwargs = {} if scheduler_kwargs is None else deepcopy(scheduler_kwargs)
            scheduler = scheduler_cls(optimizer, **scheduler_kwargs)

        if output_dir is not None:
            (output_dir / "Iterations").mkdir(exist_ok=True)

        if init_potts_starting_point:
            self.update_potts_model(input_left, input_right, positive_examples=positive_examples, coverage_threshold=coverage_threshold_potts_model, p_mask=p_mask)

        # ------------------------------------------------------------------------------------------
        ## Start training
        # ------------------------------------------------------------------------------------------
        iterations = epochs * batch_size + optimize_for_n_epochs_end
        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            for i in tqdm(range(iterations)):
                # ----------------------------------------------------------------------------------
                ## Noise Matrices for permutations
                # ----------------------------------------------------------------------------------
                if i % batch_size == 0:
                    # Save log_alpha
                    _log_alpha = torch.full(
                        (self._depth_total, self._depth_total),
                        torch.nan,
                        dtype=torch.float,
                        device=self.device,
                    )
                    _log_alpha.masked_scatter_(
                        self._effective_mask_not_fixed,
                        torch.block_diag(*self.log_alpha),
                    )
                    list_log_alpha.append(DCN(_log_alpha))
                    # Create new noise matrices and random shufflings only every `batch_size` iterations
                    noise_mat = _noise_mat()
                    rand_perm = _rand_perm()
                # Set value of noise correction
                new_noise_factor = 0
                if noise:
                    new_noise_factor = noise_factor
                    if noise_scheduler:
                        if min_noise_factor is not None:
                            lr_range = optimizer_kwargs["lr"] - scheduler_kwargs.get("min_lr", 0.0)
                            current_lr_adjusted = optimizer.param_groups[0]["lr"] - scheduler_kwargs.get("min_lr", 0.0)
                            new_noise_factor = max(min_noise_factor, noise_factor * current_lr_adjusted / lr_range)
                        else:
                            new_noise_factor = noise_factor * optimizer.param_groups[0]["lr"] / optimizer_kwargs["lr"]

                # Mean-center log-alphas
                if mean_centering:
                    with torch.no_grad():
                        for la in self.log_alpha:
                            la[...] -= la.mean()

                # ----------------------------------------------------------------------------------
                ## Compute permutation matrices
                # ----------------------------------------------------------------------------------
                params = {
                    **base_params,
                    **{
                        "noise_mat": noise_mat,
                        "noise_factor": new_noise_factor,
                        "rand_perm": rand_perm,
                    },
                }
                gs_matching_mat_not_fixed = gumbel_matching_species_wise(
                    self.log_alpha, **params
                )
                params.update(sinkhorn_params)
                gs_mat_not_fixed = gumbel_sinkhorn_species_wise(
                    self.log_alpha, **params
                )
                if fixed_pairings is not None:
                    gs_matching_mat = torch.zeros(
                        self._depth_total,
                        self._depth_total,
                        dtype=torch.float,
                        device=self.device,
                    )
                    gs_mat = torch.zeros(
                        self._depth_total,
                        self._depth_total,
                        dtype=torch.float,
                        device=self.device,
                    )
                    start = 0
                    for species_size, species_fixed_pairings in zip(
                        self.species_sizes, self._effective_fixed_pairings_zip
                    ):
                        gs_matching_mat[start:, start:][species_fixed_pairings] = 1.0
                        gs_mat[start:, start:][species_fixed_pairings] = 1.0
                        start += species_size
                    gs_mat.masked_scatter_(
                        self._effective_mask_not_fixed, gs_mat_not_fixed
                    )
                    gs_matching_mat.masked_scatter_(
                        self._effective_mask_not_fixed, gs_matching_mat_not_fixed
                    )
                else:
                    gs_matching_mat = gs_matching_mat_not_fixed
                    gs_mat = gs_mat_not_fixed
                # Save hard or soft permutation matrix
                if i % batch_size == 0:
                    mats.append(DCN(gs_matching_mat))
                    mats_gs.append(DCN(gs_mat))
                # Save permuted indexes
                list_idx.append(
                    DCN(torch.einsum("pq,p->q", (gs_matching_mat.detach(), target_idx.detach())))
                )
                # ----------------------------------------------------------------------------------
                ## Permute sequences of input_left using detach trick to backprop only on soft perm.
                # ----------------------------------------------------------------------------------
                # Cast permutation matrices to match input dtype for einsum operation
                input_left_ord = MSA_inverse_permutation(input_left, gs_mat.to(input_left.dtype))
                input_left_ord_hard = MSA_inverse_permutation(
                    input_left, gs_matching_mat.to(input_left.dtype)
                )
                # Detach trick to backprop only on soft perm.
                input_left_ord = (
                    input_left_ord_hard - input_left_ord
                ).detach() + input_left_ord
                # ----------------------------------------------------------------------------------
                ## Update Potts model every n epochs
                # ----------------------------------------------------------------------------------
                if (i % self.update_potts_model_every_n_epochs == 0) and (i >= optimize_first_for_n_epochs*batch_size) and (i < iterations - optimize_for_n_epochs_end) and (i % batch_size == 0):
                    self.update_potts_model(input_left_ord.detach(), input_right.detach(), positive_examples=positive_examples, coverage_threshold=coverage_threshold_potts_model, p_mask=p_mask)

                delta_pll = self(
                    input_left_ord,
                    input_right,
                    positive_examples=positive_examples,
                )
                loss = delta_pll
                
                loss = loss / batch_size
                pure_loss = loss.item()
                loss.backward()
                # Save loss values
                losses.append(pure_loss * batch_size)
                #      plot and save at every batch_size     or       no plots and save at last iteration
                if (((i + 1) % batch_size == 0) and save_all_figs) or (
                    (i == iterations - 1) and not save_all_figs
                ):
                    self.plot_real_time(
                        i,
                        DCN(gs_matching_mat),
                        DCN(gs_mat),
                        list_idx,
                        target_idx_np,
                        list_log_alpha,
                        losses,
                        batch_size,
                        epochs + optimize_for_n_epochs_end,
                        optimizer.param_groups[0]["lr"],
                        tar_loss,
                        new_noise_factor,
                        output_dir,
                        only_loss_plot,
                    )
                # ----------------------------------------------------------------------------------
                ## Optimizer and Scheduler step (with gradient accumulation in batches)
                # ----------------------------------------------------------------------------------
                # Compute this every time with exception of last iteration
                if i < iterations:
                    # Gradient Accumulation
                    if ((i + 1) % batch_size == 0) or ((i + 1) == iterations):
                        optimizer.step()
                        optimizer.zero_grad()
                        # Update scheduler
                        if scheduler_name is not None:
                            if scheduler_name == "ReduceLROnPlateau":
                                scheduler.step(sum(losses[-batch_size:]))
                            else:
                                scheduler.step()

                        list_lr.append(optimizer.param_groups[0]["lr"])

        return (
            losses,
            list_lr,
            list_idx,
            mats,
            mats_gs,
            list_log_alpha,
        )

    def target_loss(
        self,
        input_left,
        input_right,
        fixed_pairings=None,
        positive_examples=None,
        batch_size=1,
        coverage_threshold_potts_model=0.75,
        p_mask=0.0,
    ):
        """
        Function that computes the target value of the loss function using the pairs of `input_left`
        and `input_right` ordered as in the input. The loss is computed using MSA Pairformer.
        `fixed_pairings`: list of lists of pairs of paired indices relative to each species
        `positive_examples`: if not None it's a concatenation of correct pairs to use
                             as context (not masked)
        `batch_size`: batch size for the target loss (number of different masks to use at each epoch)

        Output: list of target loss values for each masking iteration (`batch_size`)
        """
        self._validator(input_left, input_right, fixed_pairings=fixed_pairings)
        pbar = tqdm(range(batch_size))
        pbar.set_description("Computing target loss")

        # Input MSAs
        input_left = input_left.to(self.device)
        input_right = input_right.to(self.device)

        with torch.no_grad():
            if self.potts_model is None:
                self.update_potts_model(input_left, input_right, positive_examples=positive_examples, coverage_threshold=coverage_threshold_potts_model, p_mask=p_mask)
            target_loss_val = []
            torch.cuda.empty_cache()
            for i in pbar:
                delta_pll = self(
                    input_left,
                    input_right,
                    positive_examples=positive_examples,
                )
                target_loss_val.append(delta_pll.item())
        return target_loss_val
