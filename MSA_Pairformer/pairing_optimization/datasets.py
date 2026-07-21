import numpy as np
from numpy.random import default_rng

import torch

import esm


_, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
msa_batch_converter = msa_alphabet.get_batch_converter()


def generate_dataset(parameters, msa_data, get_species_name, return_species=False, verbose=True):
    """
    Function that given the two full paired MSAs of interacting sequences (seen as a list of tuples)
    creates the dataset (dictionary of MSAs, both "left" and "right" ones), made of:

    - "msa":   the MSA used to start the training of the permutation matrix.
    - "positive_examples":  MSA of correct pairs to use as context during the training. It can be None
                            if we don't want any context.

    We have to specify if we either want the list of blocks or the positive examples by setting the value of
    `generate_blocks` to True or False.

    We can also limit the depth of the MSA by changing `limit_depth`.
    Keep in mind that the maximum limit of sequences depends on the GPU memory.
    """
    assert len(msa_data[0]) == len(msa_data[1])
    dataset = {}

    # Set random generators
    rng = default_rng(seed=parameters["NUMPY_SEED"])
    rng_other = default_rng(seed=parameters["NUMPY_SEED_OTHER"])
    # Parameters of msa
    N_init = parameters["N"]
    max_size_init = parameters["max_size"]
    # Count species in full MSA
    species_l, inverse_l, counts_l = np.unique(
        [get_species_name(rec[0]) for rec in msa_data[0]],
        return_inverse=True,
        return_counts=True,
    )
    species_r, inverse_r, counts_r = np.unique(
        [get_species_name(rec[0]) for rec in msa_data[1]],
        return_inverse=True,
        return_counts=True,
    )
    assert set(species_l) == set(
        species_r
    ), "Species must be the same in the left and right MSA."
    # ----------------------------------------------------------------------------------------------
    # MAIN MSA
    # ----------------------------------------------------------------------------------------------
    # Set positive_examples to None
    dataset["positive_examples"] = None
    while True:
        # Iterate until we find a collection of sequences with total depth
        # 0.9 * N <= D <= 1.1 * N
        idxs_shuffled = np.arange(len(species_l))
        rng.shuffle(idxs_shuffled)
        cumsum_counts_shuffled = np.cumsum(counts_l[idxs_shuffled])
        idxs_in_range = np.flatnonzero(
            np.abs(cumsum_counts_shuffled - N_init) <= N_init * 0.1
        )
        if len(idxs_in_range):
            num_species = rng.choice(idxs_in_range) + 1
            rand_species = np.sort(idxs_shuffled[:num_species])
            counts_in_sample = counts_l[rand_species]
            if np.all(counts_in_sample > 1) and np.all(
                counts_in_sample <= max_size_init
            ):
                break
    # Create msa by concatenating the selected sequences
    rand_idxs_l = []
    rand_idxs_r = []
    for unique_species_idx in rand_species:
        rand_idxs_l += [
            i for i, label in enumerate(inverse_l) if label == unique_species_idx
        ]
        rand_idxs_r += [
            i for i, label in enumerate(inverse_r) if label == unique_species_idx
        ]
    dataset["msa"] = {
        "left": [msa_data[0][i] for i in rand_idxs_l],
        "right": [msa_data[1][i] for i in rand_idxs_r],
    }
    # Print data
    if verbose:
        print("Generated initial MSA")
        print("\tSpecies selected, total number of species selected:")
        print(species_l[rand_species])
        print(rand_species, ",", len(rand_species))
        print("\tPairs per species, total number of pairs:")
        print(counts_in_sample, ",", sum(counts_in_sample))
    # ----------------------------------------------------------------------------------------------
    # POSITIVE EXAMPLES
    # ----------------------------------------------------------------------------------------------
    if parameters["pos"]:
        while True:
            # Indices of species not used in msa
            unused_species_idxs = idxs_shuffled[num_species:].copy()
            rng_other.shuffle(unused_species_idxs)
            cumsum_counts_shuffled = np.cumsum(counts_l[unused_species_idxs])
            # Iterate until we find a collection of sequences with total depth
            # 0.9 * pos <= D <= 1.1 * pos
            idxs_in_range_pos = np.flatnonzero(
                np.abs(cumsum_counts_shuffled - parameters["pos"])
                <= parameters["pos"] * 0.1
            )
            if len(idxs_in_range_pos):
                num_species_pos = rng.choice(idxs_in_range_pos) + 1
                rand_species_pos = np.sort(unused_species_idxs[:num_species_pos])
                counts_in_sample_pos = counts_l[rand_species_pos]
                if np.all(counts_in_sample_pos > 1):
                    break
        # Create msa of positive examples by concatenating the selected sequences
        rand_idxs_pos_l = []
        rand_idxs_pos_r = []
        for unique_species_idx in rand_species_pos:
            rand_idxs_pos_l += [
                i for i, label in enumerate(inverse_l) if label == unique_species_idx
            ]
            rand_idxs_pos_r += [
                i for i, label in enumerate(inverse_r) if label == unique_species_idx
            ]
        dataset["positive_examples"] = {
            "left": [msa_data[0][i] for i in rand_idxs_pos_l],
            "right": [msa_data[1][i] for i in rand_idxs_rand_idxs_pos_rpos],
        }
        # Print data
        if verbose:
            print("\n\nGenerated positive examples")
            print("\tSpecies selected, total number of species selected:")
            print(species_l[rand_species_pos])
            print(rand_species_pos, ",", len(rand_species_pos))
            print("\tPairs per species, total number of pairs:")
            print(counts_in_sample_pos, ",", sum(counts_in_sample_pos))
    else:
        dataset["positive_examples"] = None

    if return_species:
        return dataset, counts_in_sample, species_l[rand_species]
    else:
        return dataset, counts_in_sample


def tokenizer(list_msa, device="cuda"):
    """
    Function that given an MSA (seen as a list of tuples) tokenizes it using the MSA Transformer
    tokenizer and transform the tokens into one-hot encodings.
    """
    msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(list_msa)
    tokens = (
        torch.nn.functional.one_hot(msa_batch_tokens, len(msa_alphabet.all_toks))
        .type(torch.FloatTensor)
        .to(device)
    )
    return tokens


def dataset_tokenizer(dataset, device="cuda"):
    """
    Function that given a dictionary `dataset` of MSAs (initial MSA, blocks, positive examples) tokenizes
    each MSA and return them in a dictionary with the same keys.
    """
    dataset_tokens = {}

    with torch.set_grad_enabled(False):
        # Tokenize initial MSA
        dataset_tokens["msa"] = {
            key: tokenizer(dataset["msa"][key], device=device)
            for key in dataset["msa"].keys()
        }
        # Tokenize MSAs of positive examples and concatenate together the correct pairs, returns None
        # if there are no positive examples
        if dataset["positive_examples"] is None:
            dataset_tokens["positive_examples"] = None
        else:
            tmp_pos_examples = {
                key: tokenizer(dataset["positive_examples"][key], device=device)
                for key in dataset["positive_examples"].keys()
            }
            dataset_tokens["positive_examples"] = torch.cat(
                (tmp_pos_examples["left"], tmp_pos_examples["right"][..., 1:, :]), dim=2
            )

        return dataset_tokens
