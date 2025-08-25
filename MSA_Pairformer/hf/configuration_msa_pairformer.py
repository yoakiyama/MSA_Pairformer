r"""MSA Pairformer model configuration"""
import math
from typing import Callable, Literal

from transformers import PretrainedConfig


class MsaPairformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MsaPairformer`]. It is used to instantiate a
    MSA Pairformer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield the configuration of the paper.

    References:
        "Scaling down protein language modeling with MSA Pairformer" -
        https://www.biorxiv.org/content/10.1101/2025.08.02.668173v1.abstract

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 28):
            Vocabulary size of the MSA Pairformer model. Defines the number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`MsaPairformer`].
        mask_token_id (`int`, *optional*, defaults to 27):
            The index of the mask token in the vocabulary. This must be included in the config because of the
            "mask-dropout" scaling trick, which will scale the inputs depending on the number of masked tokens.
        pad_token_id (`int`, *optional*, defaults to 26):
            The index of the padding token in the vocabulary.
        depth (`int`, *optional*, defaults to 22):
            Number of MSA Pairformer layers.
        dim_pairwise (`int`, *optional*, defaults to 256):
            Dimensionality of the pairwise representation.
        dim_msa (`int`, *optional*, defaults to 464):
            Dimensionality of the MSA representation.

        return_query_only (`bool`, *optional*, defaults to `True`):
            Whether to return only the representation of the query sequence (first sequence in the MSA).
        return_contacts (`bool`, *optional*, defaults to `True`):
            Whether to return the contact maps predicted from the final pair representation.
        return_seq_weights (`bool`, *optional*, defaults to `True`):
            Whether to return the sequence attention? weights.
        return_msa_repr_layer_idx (`list[int]` or `int` or `None`, *optional*):
            Optionally return the MSA representation of the specified layers. Respects `return_query_only`.
        return_pairwise_repr_layer_idx (`list[int]` or `int` or `None`, *optional*):
            Optionally return the pairwise representation of the specified layers.

        dim_opm_hidden (`int`, *optional*, defaults to 16):
            Dimensionality of the hidden projection in the QueryBiasedOuterProduct module.
        dim_qk (`int`, *optional*, defaults to 128):
            Dimensionality of query and key vectors in the PreSoftmaxDifferentialAttention module.
        use_query_biasing (`bool`, *optional*, defaults to `True`):
            Whether to use sequence attention in the QueryBiasedOuterProduct module. This can be dynamically toggled.
        initial_lambda (`float` or `Callable[[int], float]` or `None`, *optional*):
            Initialization value for the λ parameter in the PreSoftmaxDifferentialAttention module.
            If `None`, defaults to paper settings.
        eps (`float`, *optional*, defaults to 1e-32):
            Small epsilon for numerical stability of the QueryBiasedOuterProduct module.

        heads (`int`, *optional*, defaults to 8):
            Number of attention heads in the MsaPairWeightedAveraging module.
        dim_head (`int`, *optional*, defaults to 32):
            Dimensionality of each attention head in the MsaPairWeightedAveraging module.
        dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability applied in the MsaPairWeightedAveraging module.
        dropout_type (`str` or `None`, *optional*, defaults to `"row"`):
            Dropout type in MsaPairWeightedAveraging, one of `"row"` or `"col"`. If `None`, disables structured dropout.

        dropout_row_prob (`float`, *optional*, defaults to 0):
            Probability of dropping rows in the TriangleMultiplication module.
        dropout_col_prob (`float`, *optional*, defaults to 0):
            Probability of dropping columns in the TriangleMultiplication module.
        dim_triangle_multiplication (`int` or `None`, *optional*):
            Hidden dimension of the triangular multiplicative update in the PairwiseBlock. Defaults to `dim_pairwise`.

        r_max (`int`, *optional*, defaults to 32):
            Maximum relative distance for relative position encoding.
        s_max (`int`, *optional*, defaults to 2):
            Number of scales for relative position encoding.
        contact_layer (`int`, *optional*, defaults to 15):
            Index of the layer from which to extract representations for contact prediction.
        dim_logits (`int`, *optional*, defaults to 26):
            Dimensionality of the output logits for masked language modeling.
        do_last_msa_update (`bool`, *optional*, defaults to `False`):
            Whether to drop the final MSA update step.

    Examples:

    ```python
    >>> from MSA_Pairformer.hf.configuration_msa_pairformer import MsaPairformerConfig
    >>> from MSA_Pairformer.hf.modeling_msa_pairformer import MsaPairformer

    >>> # Make it twice as deep
    >>> configuration = MsaPairformerConfig(depth=44)

    >>> # Initializing a model from the configuration
    >>> model = MsaPairformer(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = 'msa_pairformer'

    # TODO For "transformers": Rename arguments that already exist in PretrainedConfig
    def __init__(
        self,
        # Core dimensions
        vocab_size: int = 28,  # dim_msa_input
        mask_token_id: int = 27,
        pad_token_id: int = 26,
        depth: int = 22,
        dim_pairwise: int = 256,
        dim_msa: int = 464,  # hidden_size
        # Return flags of MsaPairformer.forward
        return_query_only: bool = True,
        return_contacts: bool = True,
        return_seq_weights: bool = True,  # PreSoftmaxDifferentialAttention
        return_msa_repr_layer_idx: list[int] | int | None = None,
        return_pairwise_repr_layer_idx: list[int] | int | None = None,
        # QueryBiasedOuterProduct and PreSoftmaxDifferentialAttention
        dim_opm_hidden: int = 16,
        dim_qk: int = 128,
        use_query_biasing: bool = True,
        initial_lambda: float | Callable[[int], float] | None = None,
        eps: float = 1e-32,
        # MsaPairWeightedAveraging
        heads: int = 8,
        dim_head: int = 32,
        dropout: float = 0.0,
        dropout_type: Literal['row', 'col'] | None = 'row',
        # Transition
        transition_expansion_factor: int = 4,
        # PairwiseBlock
        dropout_row_prob: float = 0,
        dropout_col_prob: float = 0,
        dim_triangle_multiplication: int | None = None,
        # RelativePositionEncoding
        r_max: int = 32,
        s_max: int = 2,
        # MsaPairformerContactHead
        contact_layer: int = 15,
        # MsaPairformerLMHead
        dim_logits: int = 26,
        do_last_msa_update: bool = True,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            mask_token_id=mask_token_id,
            **kwargs
        )
        self.depth = depth
        self.dim_pairwise = dim_pairwise
        self.dim_msa = dim_msa
        self.hidden_size = dim_msa

        # Return flags
        self.return_contacts = return_contacts
        self.return_query_only = return_query_only
        self.return_seq_weights = return_seq_weights
        self.return_pairwise_repr_layer_idx = return_pairwise_repr_layer_idx
        self.return_msa_repr_layer_idx = return_msa_repr_layer_idx

        # QueryBiasedOuterProduct
        self.dim_opm_hidden = dim_opm_hidden
        self.dim_qk = dim_qk
        self.use_query_biasing = use_query_biasing
        self.initial_lambda = initial_lambda
        self.eps = eps

        # Transition
        self.transition_expansion_factor = transition_expansion_factor

        # MSAPairWeightedAveraging
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.dropout_type = dropout_type

        # PairwiseBlock
        self.dropout_row_prob = dropout_row_prob
        self.dropout_col_prob = dropout_col_prob  # TODO Unused?
        self.dim_triangle_multiplication = dim_triangle_multiplication or dim_pairwise

        # RelativePositionEncoding
        self.r_max = r_max
        self.s_max = s_max

        # MsaPairformerContactHead
        self.contact_layer = contact_layer

        # MsaPairformerLMHead
        self.dim_logits = dim_logits

        self.do_last_msa_update = do_last_msa_update

    def differential_attention_lambda(self, layer_idx: int) -> float:
        if self.initial_lambda is not None:
            if callable(self.initial_lambda):
                return self.initial_lambda(layer_idx)
            return self.initial_lambda

        return 0.8 - 0.6 * math.exp(-0.3 * layer_idx)


__all__ = ['MsaPairformerConfig']
