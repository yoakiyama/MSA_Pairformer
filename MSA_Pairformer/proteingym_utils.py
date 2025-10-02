# All code below is originally from the ProteinGym codebase to standardize analyses: https://github.com/OATML-Markslab/ProteinGym
# Please follow their citation instructions when using this code https://github.com/OATML-Markslab/ProteinGym?tab=readme-ov-file#reference
import os
import random
import numpy as np
import pandas as pd
import numba
from numba import prange
from collections import defaultdict
import itertools
from Bio import SeqIO
from typing import List, Tuple

GAP = "-"
MATCH_GAP = GAP
INSERT_GAP = "."

ALPHABET_PROTEIN_NOGAP = "ACDEFGHIKLMNPQRSTVWY"
ALPHABET_PROTEIN_GAP = GAP + ALPHABET_PROTEIN_NOGAP

def calc_weights_fast(matrix_mapped, identity_threshold, empty_value, num_cpus=1):
    """
        Modified from EVCouplings: https://github.com/debbiemarkslab/EVcouplings
        
        Note: Numba by default uses `multiprocessing.cpu_count()` threads. 
        On a cluster where a process might only have access to a subset of CPUs, this may be less than the number of CPUs available.
        The caller should ideally use len(os.sched_getaffinity(0)) to get the number of CPUs available to the process.
        
        Calculate weights for sequences in alignment by
        clustering all sequences with sequence identity
        greater or equal to the given threshold.
        Parameters
        ----------
        identity_threshold : float
            Sequence identity threshold
        """
    empty_idx = is_empty_sequence_matrix(matrix_mapped, empty_value=empty_value)  # e.g. sequences with just gaps or lowercase, no valid AAs
    N = matrix_mapped.shape[0]

    # Original EVCouplings code structure, plus gap handling
    if num_cpus != 1:
        # print("Calculating weights using Numba parallel (experimental) since num_cpus > 1. If you want to disable multiprocessing set num_cpus=1.")
        # print("Default number of threads for Numba:", numba.config.NUMBA_NUM_THREADS)
        
        # num_cpus > numba.config.NUMBA_NUM_THREADS will give an error.
        # But we'll leave it so that the user has to be explicit.
        numba.set_num_threads(num_cpus)
        print("Set number of threads to:", numba.get_num_threads())  # Sometimes Numba uses all the CPUs anyway
        
        num_cluster_members = calc_num_cluster_members_nogaps_parallel(matrix_mapped[~empty_idx], identity_threshold,
                                                                       invalid_value=empty_value)
        
    else:
        # Use the serial version
        num_cluster_members = calc_num_cluster_members_nogaps(matrix_mapped[~empty_idx], identity_threshold,
                                                              invalid_value=empty_value)

    # Empty sequences: weight 0
    weights = np.zeros((N))
    weights[~empty_idx] = 1.0 / num_cluster_members
    return weights

# Below are util functions copied from EVCouplings
def is_empty_sequence_matrix(matrix, empty_value):
    assert len(matrix.shape) == 2, f"Matrix must be 2D; shape={matrix.shape}"
    assert isinstance(empty_value, (int, float)), f"empty_value must be a number; type={type(empty_value)}"
    # Check for each sequence if all positions are equal to empty_value
    empty_idx = np.all((matrix == empty_value), axis=1)
    return empty_idx


def map_from_alphabet(alphabet, default):
    """
    Creates a mapping dictionary from a given alphabet.
    Parameters
    ----------
    alphabet : str
        Alphabet for remapping. Elements will
        be remapped according to alphabet starting
        from 0
    default : Elements in matrix that are not
        contained in alphabet will be treated as
        this character
    Raises
    ------
    ValueError
        For invalid default character
    """
    map_ = {
        c: i for i, c in enumerate(alphabet)
    }

    try:
        default = map_[default]
    except KeyError:
        raise ValueError(
            "Default {} is not in alphabet {}".format(default, alphabet)
        )

    return defaultdict(lambda: default, map_)



def map_matrix(matrix, map_):
    """
    Map elements in a numpy array using alphabet
    Parameters
    ----------
    matrix : np.array
        Matrix that should be remapped
    map_ : defaultdict
        Map that will be applied to matrix elements
    Returns
    -------
    np.array
        Remapped matrix
    """
    return np.vectorize(map_.__getitem__)(matrix)


# Fastmath should be safe here, as we can assume that there are no NaNs in the input etc.
@numba.jit(nopython=True, fastmath=True)  #parallel=True
def calc_num_cluster_members_nogaps(matrix, identity_threshold, invalid_value):
    """
    From EVCouplings: https://github.com/debbiemarkslab/EVcouplings/blob/develop/evcouplings/align/alignment.py#L1172.
    Modified to use non-gapped length and not counting gaps as sequence similarity matches.
    
    Calculate number of sequences in alignment
    within given identity_threshold of each other
    Parameters
    ----------
    matrix : np.array
        N x L matrix containing N sequences of length L.
        Matrix must be mapped to range(0, num_symbols) using
        map_matrix function
    identity_threshold : float
        Sequences with at least this pairwise identity will be
        grouped in the same cluster.
    Returns
    -------
    np.array
        Vector of length N containing number of cluster
        members for each sequence (inverse of sequence
        weight)
    """
    N, L = matrix.shape
    L = 1.0 * L

    # Empty sequences are filtered out before this function and are ignored
    # minimal cluster size is 1 (self)
    num_neighbors = np.ones((N))
    L_non_gaps = L - np.sum(matrix == invalid_value, axis=1)  # Edit: From EVE, use the non-gapped length
    # compare all pairs of sequences
    for i in range(N - 1):
        for j in range(i + 1, N):
            pair_matches = 0
            for k in range(L):
                # Edit(Lood): Don't count gaps as matches
                if matrix[i, k] == matrix[j, k] and matrix[i, k] != invalid_value:
                    pair_matches += 1

            # Edit(Lood): Calculate identity as fraction of non-gapped positions (so asymmetric)
            # Note: Changed >= to > to match EVE / DeepSequence code
            if pair_matches / L_non_gaps[i] > identity_threshold:
                num_neighbors[i] += 1
            if pair_matches / L_non_gaps[j] > identity_threshold:
                num_neighbors[j] += 1

    return num_neighbors


@numba.jit(nopython=True, fastmath=True, parallel=True)
def calc_num_cluster_members_nogaps_parallel(matrix, identity_threshold, invalid_value):
    """
    Parallel implementation of calc_num_cluster_members_nogaps above.
    
    Calculate number of sequences in alignment
    within given identity_threshold of each other
    Parameters
    ----------
    matrix : np.array
        N x L matrix containing N sequences of length L.
        Matrix must be mapped to range(0, num_symbols) using
        map_matrix function
    identity_threshold : float
        Sequences with at least this pairwise identity will be
        grouped in the same cluster.
    invalid_value : int
        Value in matrix that is considered invalid, e.g. gap or lowercase character.
    Returns
    -------
    np.array
        Vector of length N containing number of cluster
        members for each sequence (inverse of sequence
        weight)
    """
    N, L = matrix.shape
    L = 1.0 * L

    # Empty sequences are filtered out before this function and are ignored
    # minimal cluster size is 1 (self)
    num_neighbors = np.ones((N))
    L_non_gaps = L - np.sum(matrix == invalid_value, axis=1)  # Edit: From EVE, use the non-gapped length
    # compare all pairs of sequences
    # Edit: Rewrote loop without any dependencies between inner and outer loops, so that it can be parallelized
    for i in prange(N):
        num_neighbors_i = 1
        for j in range(N):
            if i == j:
                continue
            pair_matches = 0
            for k in range(L):  # This should hopefully be vectorised by numba
                # Edit(Lood): Don't count gaps as matches
                if matrix[i, k] == matrix[j, k] and matrix[i, k] != invalid_value:
                    pair_matches += 1

            # Edit(Lood): Calculate identity as fraction of non-gapped positions (so this similarity is asymmetric)
            # Note: Changed >= to > to match EVE / DeepSequence code
            if pair_matches / L_non_gaps[i] > identity_threshold:
                num_neighbors_i += 1

        num_neighbors[i] = num_neighbors_i

    return num_neighbors

@numba.jit(nopython=True, fastmath=True, parallel=True)
def calc_num_cluster_members_nogaps_parallel_print(matrix, identity_threshold, invalid_value, progress_proxy=None, update_frequency=1000):
    """
    Modified calc_num_cluster_members_nogaps_parallel to add tqdm progress bar - useful for multi-hour weights calc.
    
    progress_proxy : numba_progress.ProgressBar
        A handle on the progress bar to update
    update_frequency : int
        Similar to miniters in tqdm, how many iterations between updating the progress bar (which then will only print every `update_interval` seconds)
    """
    
    N, L = matrix.shape
    L = 1.0 * L

    # Empty sequences are filtered out before this function and are ignored
    # minimal cluster size is 1 (self)
    num_neighbors = np.ones((N))
    L_non_gaps = L - np.sum(matrix == invalid_value, axis=1)  # Edit: From EVE, use the non-gapped length
    # compare all pairs of sequences
    # Edit: Rewrote loop without any dependencies between inner and outer loops, so that it can be parallelized
    for i in prange(N):
        num_neighbors_i = 1
        for j in range(N):
            if i == j:
                continue
            pair_matches = 0
            for k in range(L):  # This should hopefully be vectorised by numba
                # Edit(Lood): Don't count gaps as matches
                if matrix[i, k] == matrix[j, k] and matrix[i, k] != invalid_value:
                    pair_matches += 1
            # Edit(Lood): Calculate identity as fraction of non-gapped positions (so this similarity is asymmetric)
            # Note: Changed >= to > to match EVE / DeepSequence code
            if pair_matches / L_non_gaps[i] > identity_threshold:
                num_neighbors_i += 1

        num_neighbors[i] = num_neighbors_i
        if progress_proxy is not None and i % update_frequency == 0:
            progress_proxy.update(update_frequency)

    return num_neighbors


class MSA_processing:
    def __init__(self,
        MSA_location="",
        theta=0.2,
        use_weights=True,
        weights_location="./data/weights",
        preprocess_MSA=True,
        threshold_sequence_frac_gaps=0.5,
        threshold_focus_cols_frac_gaps=1.0,
        remove_sequences_with_indeterminate_AA_in_focus_cols=True,
        weights_calc_method="eve",
        num_cpus=1,
        skip_one_hot_encodings=False,
        ):
        
        """
        This class was borrowed from our EVE codebase: https://github.com/OATML-Markslab/EVE
        Parameters:
        - msa_location: (path) Location of the MSA data. Constraints on input MSA format: 
            - focus_sequence is the first one in the MSA data
            - first line is structured as follows: ">focus_seq_name/start_pos-end_pos" (e.g., >SPIKE_SARS2/310-550)
            - corespondding sequence data located on following line(s)
            - then all other sequences follow with ">name" on first line, corresponding data on subsequent lines
        - theta: (float) Sequence weighting hyperparameter. Generally: Prokaryotic and eukaryotic families =  0.2; Viruses = 0.01
        - use_weights: (bool) If False, sets all sequence weights to 1. If True, checks weights_location -- if non empty uses that; 
            otherwise compute weights from scratch and store them at weights_location
        - weights_location: (path) Location to load from/save to the sequence weights
        - preprocess_MSA: (bool) performs pre-processing of MSA to remove short fragments and positions that are not well covered.
        - threshold_sequence_frac_gaps: (float, between 0 and 1) Threshold value to define fragments
            - sequences with a fraction of gap characters above threshold_sequence_frac_gaps are removed
            - default is set to 0.5 (i.e., fragments with 50% or more gaps are removed)
        - threshold_focus_cols_frac_gaps: (float, between 0 and 1) Threshold value to define focus columns
            - positions with a fraction of gap characters above threshold_focus_cols_pct_gaps will be set to lower case (and not included in the focus_cols)
            - default is set to 0.3 (i.e., focus positions are the ones with 30% of gaps or less, i.e., 70% or more residue occupancy)
        - remove_sequences_with_indeterminate_AA_in_focus_cols: (bool) Remove all sequences that have indeterminate AA (e.g., B, J, X, Z) at focus positions of the wild type
        - weights_calc_method: (str) Method to use for calculating sequence weights. Options: "eve" or "identity". (default "eve")
        - num_cpus: (int) Number of CPUs to use for parallel weights calculation processing. If set to -1, all available CPUs are used. If set to 1, weights are computed in serial.
        - skip_one_hot_encodings: (bool) If True, only use this class to calculate weights. Skip the one-hot encodings (which can be very memory/compute intensive)
            and don't calculate all singles.
        """
        np.random.seed(2021)
        self.MSA_location = MSA_location
        self.weights_location = weights_location
        self.theta = theta
        self.alphabet = ALPHABET_PROTEIN_NOGAP
        self.use_weights = use_weights
        self.preprocess_MSA = preprocess_MSA
        self.threshold_sequence_frac_gaps = threshold_sequence_frac_gaps
        self.threshold_focus_cols_frac_gaps = threshold_focus_cols_frac_gaps
        self.remove_sequences_with_indeterminate_AA_in_focus_cols = remove_sequences_with_indeterminate_AA_in_focus_cols
        self.weights_calc_method = weights_calc_method
        self.skip_one_hot_encodings = skip_one_hot_encodings

        # Defined by gen_alignment
        self.aa_dict = {}
        self.focus_seq_name = ""
        self.seq_name_to_sequence = defaultdict(str)
        self.focus_seq, self.focus_cols, self.focus_seq_trimmed, self.seq_len, self.alphabet_size = [None] * 5
        self.focus_start_loc, self.focus_stop_loc = None, None
        self.uniprot_focus_col_to_wt_aa_dict, self.uniprot_focus_col_to_focus_idx = None, None
        self.one_hot_encoding, self.weights, self.Neff, self.num_sequences = [None] * 4

        # Fill in the instance variables
        self.gen_alignment()

        # Note: One-hot encodings might take up huge amounts of memory, and this could be skipped in many use cases
        if not self.skip_one_hot_encodings:
            #print("One-hot encoding sequences")
            self.one_hot_encoding = one_hot_3D(
                seq_keys=self.seq_name_to_sequence.keys(),  # Note: Dicts are unordered for python < 3.6
                seq_name_to_sequence=self.seq_name_to_sequence,
                alphabet=self.alphabet,
                seq_length=self.seq_len,
            )
            print ("Data Shape =", self.one_hot_encoding.shape)
            
        self.calc_weights(num_cpus=num_cpus, method=weights_calc_method)

    def gen_alignment(self):
        """ Read training alignment and store basics in class instance """
        self.aa_dict = {}
        for i, aa in enumerate(self.alphabet):
            self.aa_dict[aa] = i

        self.seq_name_to_sequence = defaultdict(str)
        name = ""
        with open(self.MSA_location, "r") as msa_data:
            for i, line in enumerate(msa_data):
                line = line.rstrip()
                if line.startswith(">"):
                    name = line
                    if i == 0:
                        self.focus_seq_name = name
                else:
                    self.seq_name_to_sequence[name] += line
        print("Number of sequences in MSA (before preprocessing):", len(self.seq_name_to_sequence))

        ## MSA pre-processing to remove inadequate columns and sequences
        if self.preprocess_MSA:
            # Overwrite self.seq_name_to_sequence
            self.seq_name_to_sequence = self.preprocess_msa(
                seq_name_to_sequence=self.seq_name_to_sequence,
                focus_seq_name=self.focus_seq_name,
                threshold_sequence_frac_gaps=self.threshold_sequence_frac_gaps,
                threshold_focus_cols_frac_gaps=self.threshold_focus_cols_frac_gaps
            )

        self.focus_seq = self.seq_name_to_sequence[self.focus_seq_name]
        self.focus_cols = [ix for ix, s in enumerate(self.focus_seq) if s == s.upper() and s != '-']
        self.focus_seq_trimmed = "".join([self.focus_seq[ix] for ix in self.focus_cols])
        self.seq_len = len(self.focus_cols)
        self.alphabet_size = len(self.alphabet)

        # Move all letters to CAPS; keeps focus columns only
        self.raw_seq_name_to_sequence = self.seq_name_to_sequence.copy()
        for seq_name, sequence in self.seq_name_to_sequence.items():
            sequence = sequence.replace(".", "-")
            self.seq_name_to_sequence[seq_name] = "".join(
                [sequence[ix].upper() for ix in self.focus_cols])  # Makes a List[str] instead of str

        # Remove sequences that have indeterminate AA (e.g., B, J, X, Z) in the focus columns
        if self.remove_sequences_with_indeterminate_AA_in_focus_cols:
            num_sequences_removed_due_to_indeterminate_AAs = 0
            num_sequences_before_indeterminate_AA_drop = len(self.seq_name_to_sequence)
            alphabet_set = set(list(self.alphabet))
            seq_names_to_remove = []
            for seq_name, sequence in self.seq_name_to_sequence.items():
                for letter in sequence:
                    if letter not in alphabet_set and letter != "-":
                        seq_names_to_remove.append(seq_name)
                        continue
            seq_names_to_remove = list(set(seq_names_to_remove))
            for seq_name in seq_names_to_remove:
                num_sequences_removed_due_to_indeterminate_AAs+=1
                del self.seq_name_to_sequence[seq_name]
            print("Proportion of sequences dropped due to indeterminate AAs: {}%".format(round(float(num_sequences_removed_due_to_indeterminate_AAs/num_sequences_before_indeterminate_AA_drop*100),2)))
        
        print("Number of sequences after preprocessing:", len(self.seq_name_to_sequence))
        self.num_sequences = len(self.seq_name_to_sequence.keys())

    # Using staticmethod to keep this under the MSAProcessing namespace, but this is apparently not best practice
    @staticmethod
    def preprocess_msa(seq_name_to_sequence, focus_seq_name, threshold_sequence_frac_gaps, threshold_focus_cols_frac_gaps):
        """Remove inadequate columns and sequences from MSA, overwrite self.seq_name_to_sequence."""
        msa_df = pd.DataFrame.from_dict(seq_name_to_sequence, orient='index', columns=['sequence'])
        # Data clean up
        msa_df.sequence = msa_df.sequence.apply(lambda x: x.replace(".", "-")).apply(
            lambda x: ''.join([aa.upper() for aa in x]))
        # Remove columns that would be gaps in the wild type
        non_gap_wt_cols = [aa != '-' for aa in msa_df.sequence[focus_seq_name]]
        msa_df['sequence'] = msa_df['sequence'].apply(
            lambda x: ''.join([aa for aa, non_gap_ind in zip(x, non_gap_wt_cols) if non_gap_ind]))
        assert 0.0 <= threshold_sequence_frac_gaps <= 1.0, "Invalid fragment filtering parameter"
        assert 0.0 <= threshold_focus_cols_frac_gaps <= 1.0, "Invalid focus position filtering parameter"
        print("Calculating proportion of gaps")
        msa_array = np.array([list(seq) for seq in msa_df.sequence])
        gaps_array = np.array(list(map(lambda seq: [aa == '-' for aa in seq], msa_array)))
        # Identify fragments with too many gaps
        seq_gaps_frac = gaps_array.mean(axis=1)
        seq_below_threshold = seq_gaps_frac <= threshold_sequence_frac_gaps
        print("Proportion of sequences dropped due to fraction of gaps: " + str(
            round(float(1 - seq_below_threshold.sum() / seq_below_threshold.shape) * 100, 2)) + "%")
        # Identify focus columns
        columns_gaps_frac = gaps_array[seq_below_threshold].mean(axis=0)
        index_cols_below_threshold = columns_gaps_frac <= threshold_focus_cols_frac_gaps
        print("Proportion of non-focus columns removed: " + str(
            round(float(1 - index_cols_below_threshold.sum() / index_cols_below_threshold.shape) * 100, 2)) + "%")
        # Lower case non focus cols and filter fragment sequences
        def _lower_case_and_filter_fragments(seq):
            return ''.join([aa.lower() if aa_ix in index_cols_below_threshold else aa for aa_ix, aa in enumerate(seq)])
        msa_df['sequence'] = msa_df['sequence'].apply(
            lambda seq: ''.join([aa.upper() if upper_case_ind else aa.lower() for aa, upper_case_ind in
             zip(seq, index_cols_below_threshold)]))
        msa_df = msa_df[seq_below_threshold]
        # Overwrite seq_name_to_sequence with clean version
        seq_name_to_sequence = defaultdict(str)
        # Create a dictionary from msa_df.index to msa_df.sequence
        seq_name_to_sequence = dict(zip(msa_df.index, msa_df.sequence))
        # for seq_idx in range(len(msa_df['sequence'])):
        #     seq_name_to_sequence[msa_df.index[seq_idx]] = msa_df.sequence[seq_idx]

        return seq_name_to_sequence

    def calc_weights(self, num_cpus=1, method="eve"):
        """
        From the EVE repo, but modified to skip printing out progress bar / time taken 
        (because for ProteinNPT embeddings, weights will usually be computed on the fly for small subsamples of MSA).
        
        If num_cpus == 1, weights are computed in serial.
        If num_cpus == -1, weights are computed in parallel using all available cores.
        Note: This will use multiprocessing.cpu_count() to get the number of available cores, which on clusters may
        return all cores, not just the number of cores available to the user.
        """
        # Refactored into its own function so that we can call it separately
        if self.use_weights:
            if os.path.isfile(self.weights_location):
                print("Loading sequence weights from disk: {}".format(self.weights_location))
                self.weights = np.load(file=self.weights_location)
            else:
                print("Computing sequence weights")
                if num_cpus == -1:
                    num_cpus = get_num_cpus()

                if method == "eve":
                    alphabet_mapper = map_from_alphabet(ALPHABET_PROTEIN_GAP, default=GAP)
                    arrays = []
                    for seq in self.seq_name_to_sequence.values():
                        arrays.append(np.array(list(seq)))
                    sequences = np.vstack(arrays)
                    sequences_mapped = map_matrix(sequences, alphabet_mapper)
                    self.weights = calc_weights_fast(sequences_mapped, identity_threshold=1 - self.theta,
                                                            empty_value=0, num_cpus=num_cpus)  # GAP = 0
                elif method == "identity":
                    self.weights = np.ones(self.one_hot_encoding.shape[0])
                else:
                    raise ValueError(f"Unknown method: {method}. Must be either 'eve' or 'identity'.")
                print("Saving sequence weights to disk")
                np.save(file=self.weights_location, arr=self.weights)
        else:
            # If not using weights, use an isotropic weight matrix
            print("Not weighting sequence data")
            self.weights = np.ones(self.one_hot_encoding.shape[0])

        self.Neff = np.sum(self.weights)
        print("Neff =", str(self.Neff))
        print("Number of sequences: ", self.num_sequences)
        assert self.weights.shape[0] == self.num_sequences, f"Expected {self.num_sequences} sequences, loaded weights have {self.weights.shape[0]}"
        self.seq_name_to_weight={}  # For later, if we want to remove certain sequences and associated weights
        for i,seq_name in enumerate(self.seq_name_to_sequence.keys()):
            self.seq_name_to_weight[seq_name]=self.weights[i]
        
        return self.weights

# One-hot encoding of sequences
def one_hot_3D(seq_keys, seq_name_to_sequence, alphabet, seq_length):
    """
    Take in a list of sequence names/keys and corresponding sequences, and generate a one-hot array according to an alphabet.
    """
    aa_dict = {letter: i for (i, letter) in enumerate(alphabet)}

    one_hot_out = np.zeros((len(seq_keys), seq_length, len(alphabet)))
    for i, seq_key in enumerate(seq_keys):
        sequence = seq_name_to_sequence[seq_key]
        for j, letter in enumerate(sequence):
            if letter in aa_dict:
                k = aa_dict[letter]
                one_hot_out[i, j, k] = 1.0
    # one_hot_out = torch.tensor(one_hot_out)
    return one_hot_out

def get_num_cpus():
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        num_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
        print("SLURM_CPUS_PER_TASK:", os.environ['SLURM_CPUS_PER_TASK'])
        print("Using all available cores (calculated using SLURM_CPUS_PER_TASK):", num_cpus)
    else:
        num_cpus = len(os.sched_getaffinity(0)) 
        print("Using all available cores (calculated using len(os.sched_getaffinity(0))):", num_cpus)
    return num_cpus

def process_msa(
    filename: str,
    weight_filename: str,
    filter_msa: bool,
    path_to_hhfilter: str,
    hhfilter_min_cov=75,
    hhfilter_max_seq_id=100,
    hhfilter_min_seq_id=0,
    num_cpus=1
) -> List[Tuple[str, str]]:
    if filter_msa:
        input_folder = '/'.join(filename.split('/')[:-1])
        msa_name = filename.split('/')[-1].split('.')[0]
        # Create preprocessed directory
        preprocessed_dir = os.path.join(input_folder, "preprocessed")
        if not os.path.isdir(preprocessed_dir):
            os.mkdir(preprocessed_dir)
        # Create hhfiltered directory
        hhfiltered_dir = os.path.join(input_folder, "hhfiltered")
        if not os.path.isdir(hhfiltered_dir):
            os.mkdir(hhfiltered_dir)
        # Create preprocessed filename
        preprocessed_filename_prefix = os.path.join(input_folder, 'preprocessed', msa_name)
        os.system('cat '+filename+' | tr  "."  "-" >> '+preprocessed_filename_prefix+'.a2m')
        os.system('dd if='+preprocessed_filename_prefix+'.a2m of='+preprocessed_filename_prefix+'_UC.a2m conv=ucase')
        output_filename = os.path.join(
            input_folder,
            'hhfiltered',
            f"{msa_name}_hhfiltered_cov_{str(hhfilter_min_cov)}_maxid_{str(hhfilter_max_seq_id)}_minid_{str(hhfilter_min_seq_id)}.a2m"
        )
        bin_file = os.path.join(path_to_hhfilter, 'bin', 'hhfilter')
        os.system(f"{bin_file} -cov {str(hhfilter_min_cov)} -id {str(hhfilter_max_seq_id)} -qid {str(hhfilter_min_seq_id)} -i {preprocessed_filename_prefix}_UC.a2m -o {output_filename} -maxseq 10000000")
        filename = output_filename

    MSA = MSA_processing(
        MSA_location=filename,
        use_weights=True,
        weights_location=weight_filename,
        num_cpus=num_cpus
    )
    print("Name of focus_seq: "+str(MSA.focus_seq_name))
    return MSA

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
            MSA = MSA_processing(
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
    return msa

def label_row(row, sequence, token_probs, aa2tok_d, offset_idx):
    score=0
    for mutation in row.split(":"):
        wt, idx, mt = mutation[0], int(mutation[1:-1]) - offset_idx, mutation[-1]
        assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

        wt_encoded, mt_encoded = aa2tok_d[wt], aa2tok_d[mt]

        score += (token_probs[0, idx, mt_encoded] - token_probs[0, idx, wt_encoded]).item()
    return score