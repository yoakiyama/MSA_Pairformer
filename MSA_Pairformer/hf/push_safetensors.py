r"""Push safetensors weights of MsaPairformer to HuggingFace."""
import torch

from MSA_Pairformer.hf.configuration_msa_pairformer import MsaPairformerConfig
from MSA_Pairformer.model import MSAPairformer
from MSA_Pairformer.hf.modeling_msa_pairformer import MsaPairformer

if __name__ == '__main__':
    og_msa_pairformer: MSAPairformer = MSAPairformer.from_pretrained(device=torch.device('cpu'))

    # New HuggingFace PreTrainedModel
    config = MsaPairformerConfig()
    pretrained_model_msa_pairformer: MsaPairformer = MsaPairformer(config)

    new_embeddings = pretrained_model_msa_pairformer.embeddings
    new_encoder = pretrained_model_msa_pairformer.encoder
    new_contact_head = pretrained_model_msa_pairformer.contact_head
    new_lm_head = pretrained_model_msa_pairformer.lm_head

    # Copy weights from pickled checkpoint
    new_embeddings.relative_position_encoding = og_msa_pairformer.relative_position_encoding
    new_embeddings.token_bond_to_pairwise_feat = og_msa_pairformer.token_bond_to_pairwise_feat[1]
    new_embeddings.msa_init_proj = og_msa_pairformer.msa_init_proj

    new_encoder.layers = og_msa_pairformer.core_stack.layers
    new_encoder.final_msa_pwa = og_msa_pairformer.core_stack.final_msa_pwa
    new_encoder.final_msa_transition = og_msa_pairformer.core_stack.final_msa_transition

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
        commit_message='Converting commit 420a3b02486be6c04906267b89b6763214d10ce6 to safetensors for use with '
                       'HuggingFace PreTrainedModel',
        revision='refs/pr/1'
    )
