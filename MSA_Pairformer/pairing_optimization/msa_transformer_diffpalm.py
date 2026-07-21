# Adapted from https://github.com/Bitbol-Lab/DiffPALM/blob/main/diffpalm/msa_parsing.py
# Original author: Umberto Lupo et al. (2024), Pairing interacting protein sequences using masked language modeling

from pathlib import Path
from copy import deepcopy

# Progress bars
from tqdm import tqdm

# NumPy
import numpy as np

# Torch and ESM
import torch
import esm

from MSA_Pairformer.pairing_optimization.gumbel_sinkhorn_utils import (
    gumbel_sinkhorn,
    gumbel_matching,
    MSA_inverse_permutation,
    sample_uniform,
)

# PLOTTING
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
import matplotlib.cm as cm

msa_transformer, _ = esm.pretrained.esm_msa1b_t12_100M_UR50S()
msa_transformer = msa_transformer.eval()


def DCN(x):
    return x.detach().clone().cpu().numpy()

class PermutationsMixin:
    """Mixin class for validating input and plotting the results of the optimization."""

    mask_idx = 32

    def _init_log_alpha(self, skip=False):
        """Intialize log_alpha as a list of matrices of shape (d, d) where d is the
        depth of the species MSA. The matrices are initialized with standard normal entries.
        """
        if not skip:
            # Permutations restricted to species
            self.log_alpha = [
                (self.std_init * torch.randn(d, d, device=self.device)).requires_grad_(
                    True
                )
                for d in self._effective_depth_not_fixed
            ]

    def _validator(self, input_left, input_right, fixed_pairings=None):
        """Validate input MSAs and check fixed pairings."""
        # Validate input MSAs
        depth_left, length_left, alphabet_size_left = input_left.shape[1:]
        depth_right, length_right, alphabet_size_right = input_right.shape[1:]
        length_left -= 1
        length_right -= 1
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
        fig, axes = plt.subplots(figsize=(30, 5), ncols=5, constrained_layout=True)

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
                axes[4].axhline(tar_loss, color="b", linewidth=2)
            diff = avg_loss[0] - tar_loss
            axes[4].set_ylim([tar_loss - 0.6 * diff, avg_loss[0] + 0.5 * diff])
            ax3_2 = axes[4].twinx()
            ax3_2.set_ylabel("No. of correct pairs", color="red")
            ax3_2.plot(n_correct, color="red", linewidth=1)
            ax3_2.axhline(null_model, color="red", linewidth=2)
            ax3_2.tick_params(axis="y", labelcolor="red")
        axes[4].set_ylabel("Loss")
        axes[4].set_xlim([0, epochs])
        axes[4].set_title(f"lr: {lr:.3g}, noise:{new_noise_factor:.3g}")
        fig.colorbar(ims_soft, ax=axes[0], shrink=0.8)
        fig.colorbar(ims_log_alpha, ax=axes[2], shrink=0.8)

        plt.show()

class DiffPALM(torch.nn.Module, PermutationsMixin):
    """
    Permute all the pairs between two concatenated MSAs (for each species), randomly mask the left MSA
    and compute the MLM loss. Backpropagate the loss on the permutation matrix and iterate the process
    to get the correct permutation of interacting pairs.
    `species_sizes`: list of species sizes for the paired MSA
    `p_mask`: token masking probability for left MSA
    `random_seed`: random seed
    `device`: device to use for computations
    """

    def __init__(self, species_sizes, *, p_mask=0.7, random_seed=42, device="cuda"):
        super().__init__()

        # List of species sizes for the paired MSA
        self.species_sizes = species_sizes

        # Token masking probability for left MSA
        self.p_mask = p_mask

        # Loss function
        self.loss = torch.nn.CrossEntropyLoss()

        # Set random seed
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)

        # Set device and load MSA-Transformer to device
        self.device = device
        self.msa_tr = msa_transformer.to(self.device)

    def forward(self, input_ord, input_right, positive_examples=None):
        """
        Mask input MSA and concatenate with fixed MSA. Then compute output logits
        for the masked positions using MSA-Transformer.

        `input_ord`: variable input at each iteration (masked)  --> (B, D, L1 + 1, 33)
        `input_right`: fixed input (no masking)                 --> (B, D, L2 + 1, 33)
        `positive_examples`: if not None it's a concatenation of correct pairs to use
                             as context (not masked)            --> (B, D, L1 + L2 + 1, 33)
        """
        # One-hot vector to use as mask in the MSA
        vec_mask = torch.zeros(input_ord.shape[-1]).to(self.device)
        vec_mask[self.mask_idx] = 1
        # Create masking array # --> (B,D,L1)

        # Mask input MSA in the positions specified by `mask` matrix (using`vec_mask`)
        input_mask = input_ord.clone()
        mask = (torch.rand(input_ord[..., 1:, 0].shape) < self.p_mask).to(self.device)

        if self._effective_fixed_pairings_zip is not None:
            start = 0
            for species_size, (_, cols_fixed) in zip(
                self.species_sizes, self._effective_fixed_pairings_zip
            ):
                mask[:, start:, ...][:, cols_fixed, ...] = False
                start += species_size

        input_mask[..., 1:, :][mask] = vec_mask
        # Concatenate masked MSA with fixed MSA (not masked)
        input_mask = torch.cat((input_mask, input_right[..., 1:, :]), dim=2)

        # Add positive examples (correct pairs) on top of input MSA
        if positive_examples is not None:
            input_mask = torch.cat((positive_examples, input_mask), dim=1)

        # Compute output logits
        input_mask = input_mask.argmax(dim=-1)
        results = self.msa_tr(input_mask, repr_layers=[12])
        logits = results["logits"]
        # Restrict to logits of masked (left) MSA, excluding positive examples
        logits = logits[:, :, 1 : input_ord.shape[2], :]
        if positive_examples is not None:
            logits = logits[:, positive_examples.shape[1] :, :, :]

        return logits, mask

    def train(
        self,
        input_left,
        input_right,
        fixed_pairings=None,  # Format: list of lists of pairs of paired indices relative to each species [[(i, j), ...], ...]
        positive_examples=None,
        init_log_alpha=True,
        starting_log_alpha=None,
        std_init=0.1,
        epochs=1,
        optimizer_name="Adadelta",
        optimizer_kwargs=None,
        tau=1.0,
        n_sink_iter=10,
        noise=True,
        noise_std=False,
        noise_factor=0.1,
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
    ):
        """
        Train the model using the input MSAs (`input_left`, `input_right`) and the fixed pairings.

        `fixed_pairings`: list of lists of pairs of paired indices relative to each species
                            Format: [[(i, j), ...], ...]
        `init_log_alpha`: if True initialize log_alpha with random values
        `starting_log_alpha`: if not None, use this log_alpha as starting point for the training
        `std_init`: standard deviation of the normal distribution used to initialize log_alpha
        `epochs`: number of epochs of the training
        `optimizer_name`: name of the optimizer to use
        `optimizer_kwargs`: kwargs of the optimizer
        `tau`: temperature parameter for the Sinkhorn operator
        `n_sink_iter`: number of Sinkhorn iterations
        `noise`: if True add noise to the Gumbel-Matching algorithm
        `noise_std`: if True use a fixed noise_std for the noise matrices
        `noise_factor`: noise correction factor
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
        if init_log_alpha:
            self._init_log_alpha()
        else:
            assert starting_log_alpha is not None, "Cannot initialize log_alpha and use starting_log_alpha"
            self.log_alpha = [la.clone().detach().to(self.device).requires_grad_(True) for la in starting_log_alpha]

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
                        func(la, noise_mat=nm, rand_perm=rp, **params)
                        if la.size(0)
                        else la
                        for la, nm, rp in zip(log_alpha, noise_mat, rand_perm)
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
        input_left = input_left.to(self.device).requires_grad_(True)
        input_right = input_right.to(self.device).requires_grad_(True)

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

        # ------------------------------------------------------------------------------------------
        ## Start training
        # ------------------------------------------------------------------------------------------
        iterations = epochs * batch_size
        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            for i in tqdm(range(iterations + batch_size)):
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
                        new_noise_factor = (
                            noise_factor
                            * optimizer.param_groups[0]["lr"]
                            / optimizer_kwargs["lr"]
                        )

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
                    _gs_mat = torch.zeros_like(
                        gs_matching_mat, device=self.device, requires_grad=True
                    )
                    gs_mat = _gs_mat.clone()
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
                    DCN(torch.einsum("pq,p->q", (gs_matching_mat, target_idx)))
                )
                # ----------------------------------------------------------------------------------
                ## Permute sequences of input_left using detach trick to backprop only on soft perm.
                # ----------------------------------------------------------------------------------
                input_left_ord = MSA_inverse_permutation(input_left, gs_mat)
                input_left_ord_hard = MSA_inverse_permutation(
                    input_left, gs_matching_mat
                )
                input_left_ord = (
                    input_left_ord_hard - input_left_ord
                ).detach() + input_left_ord
                # ----------------------------------------------------------------------------------
                ## Get output logits of MSA Transformer for the permuted, masked input
                # ----------------------------------------------------------------------------------
                loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                logits, mask = self(
                    input_left_ord, input_right, positive_examples=positive_examples
                )
                loss = loss + self.loss(logits[mask], input_left_ord[..., 1:, :][mask])
                loss = loss / batch_size
                # ----------------------------------------------------------------------------------
                ## Compute pseudoLikelihood loss only on masked positions and save gradients
                # ----------------------------------------------------------------------------------
                pure_loss = loss.item()
                # Compute gradients
                loss.backward()
                # Save loss values
                losses.append(pure_loss * batch_size)
                #      plot and save at every batch_size     or       no plots and save at last iteration
                if (((i + 1) % batch_size == 0) and save_all_figs) or (
                    (i == iterations + batch_size - 1) and not save_all_figs
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
                        epochs,
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
    ):
        """
        Function that computes the target value of the loss function using the pairs of `input_left`
        and `input_right` ordered as in the input. The loss is computed using the MSA-Transformer.
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

        with torch.set_grad_enabled(False):
            target_loss_val = []
            torch.cuda.empty_cache()
            for i in pbar:
                logits, mask = self(
                    input_left,
                    input_right,
                    positive_examples=positive_examples,
                )
                loss = self.loss(logits[mask], input_left[..., 1:, :][mask]).item()
                target_loss_val.append(loss)

        return target_loss_val
