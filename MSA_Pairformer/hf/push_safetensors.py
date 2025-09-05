r"""Push safetensors weights of MsaPairformer to HuggingFace."""
from copy import deepcopy

import torch

import MSA_Pairformer.hf.modeling_msa_pairformer as hf
from MSA_Pairformer.core import PreLayerNorm
from MSA_Pairformer.hf.configuration_msa_pairformer import MsaPairformerConfig
from MSA_Pairformer.model import MSAPairformer
from MSA_Pairformer.outer_product import PresoftmaxDifferentialOuterProductMean
from MSA_Pairformer.pairwise_operations import MSAPairWeightedAveraging, PairwiseBlock


def copy_msa_block(
    og_msa_pwa: MSAPairWeightedAveraging,
    og_msa_transition_norm: PreLayerNorm,
    msa_block: hf.MsaBlock
) -> None:
    # Deepcopy to prevent RuntimeError: The weights trying to be saved contained shared tensors
    new_msa_pwa: hf.MsaPairWeightedAveraging = msa_block.msa_pwa

    new_msa_pwa.msa_repr_to_values_and_gates.pre_norm = deepcopy(og_msa_pwa.msa_to_values_and_gates[0])
    new_msa_pwa.msa_repr_to_values_and_gates.linear_no_bias = deepcopy(og_msa_pwa.msa_to_values_and_gates[1])
    new_msa_pwa.msa_repr_to_values_and_gates.rearrange = deepcopy(og_msa_pwa.msa_to_values_and_gates[2])

    new_msa_pwa.pairwise_repr_to_attn.pre_norm = deepcopy(og_msa_pwa.pairwise_repr_to_attn[0])
    new_msa_pwa.pairwise_repr_to_attn.linear_no_bias = deepcopy(og_msa_pwa.pairwise_repr_to_attn[1])
    new_msa_pwa.pairwise_repr_to_attn.rearrange = deepcopy(og_msa_pwa.pairwise_repr_to_attn[2])

    new_msa_pwa.out_proj.rearrange = deepcopy(og_msa_pwa.to_out[0])
    new_msa_pwa.out_proj.linear_no_bias = deepcopy(og_msa_pwa.to_out[1])
    # new_mpwa.out_proj.dropout = deepcopy(og_mpwa.to_out[2])

    new_msa_transition: hf.Transition = msa_block.msa_transition
    new_msa_transition.pre_norm = deepcopy(og_msa_transition_norm.norm)
    new_msa_transition.linear_no_bias = deepcopy(og_msa_transition_norm.fn.ff[0])
    # new_msa_transition.swiglu = deepcopy(og_msa_transition.fn.ff[1])
    new_msa_transition.out_proj = deepcopy(og_msa_transition_norm.fn.ff[2])


def copy_weights_triangle_multiplication(
    og_triangle_multiplication_norm: PreLayerNorm,
    new_triangle_multiplication: hf.TriangleMultiplication
) -> None:
    og_triangle_multiplication = og_triangle_multiplication_norm.fn
    new_triangle_multiplication.pre_norm = deepcopy(og_triangle_multiplication_norm.norm)
    new_triangle_multiplication.left_right_proj.linear_no_bias = deepcopy(og_triangle_multiplication.left_right_proj[0])
    # new_triangle_multiplication.left_right_proj.glu = deepcopy(og_triangle_multiplication.left_right_proj[1])
    new_triangle_multiplication.out_norm = deepcopy(og_triangle_multiplication.to_out_norm)
    new_triangle_multiplication.out_gate.linear_no_bias = deepcopy(og_triangle_multiplication.out_gate)
    new_triangle_multiplication.out_proj.linear_no_bias = deepcopy(og_triangle_multiplication.to_out[0])
    # new_triangle_multiplication.out_proj.dropout = deepcopy(og_triangle_multiplication.to_out[1])


if __name__ == '__main__':
    og_msa_pairformer: MSAPairformer = MSAPairformer.from_pretrained(device=torch.device('cpu'))

    # New HuggingFace PreTrainedModel
    config = MsaPairformerConfig()
    pretrained_model_msa_pairformer: hf.MsaPairformer = hf.MsaPairformer(config)

    new_embeddings = pretrained_model_msa_pairformer.embeddings
    new_encoder = pretrained_model_msa_pairformer.encoder
    new_contact_head = pretrained_model_msa_pairformer.contact_head
    new_lm_head = pretrained_model_msa_pairformer.lm_head

    # Copy weights from pickled checkpoint
    new_embeddings.relative_position_encoding.pairwise_init_proj = og_msa_pairformer.relative_position_encoding.out_embedder
    new_embeddings.token_bond_to_pairwise_feat = og_msa_pairformer.token_bond_to_pairwise_feat[1]
    new_embeddings.msa_init_proj = og_msa_pairformer.msa_init_proj

    for new_layer, og_layer in zip(new_encoder.layers, og_msa_pairformer.core_stack.layers):
        new_layer: hf.MsaPairformerLayer

        msa_block_: hf.MsaBlock = new_layer.msa_block
        og_msa_pwa_: MSAPairWeightedAveraging = og_layer[0]
        og_msa_transition_norm_: PreLayerNorm = og_layer[1]

        copy_msa_block(
            og_msa_pwa=og_msa_pwa_,
            og_msa_transition_norm=og_msa_transition_norm_,
            msa_block=msa_block_
        )

        og_outer_product: PresoftmaxDifferentialOuterProductMean = og_layer[2].opm
        new_outer_product: hf.QueryBiasedOuterProduct = new_layer.outer_product
        new_outer_product.pre_norm = og_outer_product.norm
        new_outer_product.to_left_hidden = og_outer_product.to_left_hidden
        new_outer_product.to_right_hidden = og_outer_product.to_right_hidden
        new_outer_product.pre_softmax_differential_attention.q_proj = og_outer_product.q_proj
        new_outer_product.pre_softmax_differential_attention.k_proj = og_outer_product.k_proj
        new_outer_product.pre_softmax_differential_attention.q_norm = og_outer_product.q_norm
        new_outer_product.pre_softmax_differential_attention.k_norm = og_outer_product.k_norm
        new_outer_product.pre_softmax_differential_attention.lambda_init = og_outer_product.lambda_init
        new_outer_product.pre_softmax_differential_attention.lambda_q1 = og_outer_product.lambda_q1
        new_outer_product.pre_softmax_differential_attention.lambda_q2 = og_outer_product.lambda_q2
        new_outer_product.pre_softmax_differential_attention.lambda_k1 = og_outer_product.lambda_k1
        new_outer_product.pre_softmax_differential_attention.lambda_k2 = og_outer_product.lambda_k2
        new_outer_product.outer_product_mean.to_pairwise_repr = og_outer_product.to_pairwise_repr
        # new_outer_product.outer_product_mean.swiglu = og_outer_product.activation

        og_pairwise_block: PairwiseBlock = og_layer[3]
        new_pairwise_block: hf.PairwiseBlock = new_layer.pairwise_block

        copy_weights_triangle_multiplication(
            og_triangle_multiplication_norm=og_pairwise_block.tri_mult_outgoing,
            new_triangle_multiplication=new_pairwise_block.triangle_multiplication_outgoing
        )
        copy_weights_triangle_multiplication(
            og_triangle_multiplication_norm=og_pairwise_block.tri_mult_incoming,
            new_triangle_multiplication=new_pairwise_block.triangle_multiplication_incoming
        )
        new_pairwise_block.pairwise_transition.pre_norm = og_pairwise_block.pairwise_transition.norm
        new_pairwise_block.pairwise_transition.linear_no_bias = og_pairwise_block.pairwise_transition.fn.ff[0]
        # new_pairwise_block.pairwise_transition.swiglu = og_pairwise_block.pairwise_transition.fn.ff[1]
        new_pairwise_block.pairwise_transition.out_proj = og_pairwise_block.pairwise_transition.fn.ff[2]

    copy_msa_block(
        og_msa_pwa=og_msa_pairformer.core_stack.final_msa_pwa,
        og_msa_transition_norm=og_msa_pairformer.core_stack.final_msa_transition,
        msa_block=new_encoder.final_msa_block
    )

    new_contact_head.init_ln = og_msa_pairformer.contact_head.init_ln
    new_contact_head.dense.weight = og_msa_pairformer.contact_head.weight
    new_contact_head.dense.bias = og_msa_pairformer.contact_head.bias

    new_lm_head.init_ln = og_msa_pairformer.lm_head.init_ln
    new_lm_head.dense.weight = og_msa_pairformer.lm_head.dense.weight
    new_lm_head.dense.bias = og_msa_pairformer.lm_head.dense.bias
    new_lm_head.pre_logit_norm = og_msa_pairformer.lm_head.pre_logit_norm
    new_lm_head.output.weight = og_msa_pairformer.lm_head.weight
    new_lm_head.output.bias = og_msa_pairformer.lm_head.bias

    pretrained_model_msa_pairformer.push_to_hub(
        repo_id='yoakiyama/MSA-Pairformer',
        commit_message='Rename msa_pair_weighted_averaging to msa_pwa (https://github.com/huggingface/peft/issues/2772)',
        revision='refs/pr/1'
    )
