from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from MSA_Pairformer.dataset import aa2tok_d


# Option 1 (available offline through class)
class MsaPairformerTokenizer(PreTrainedTokenizerFast):
    r"""
    "Fast" tokenizer for MsaPairformer.
    Expects lists of sequences from MSA (typically same length) and returns tokenized sequences and full attention mask.

    References:
        https://huggingface.co/docs/transformers/main_classes/tokenizer
    """
    vocab_files_names = {'tokenizer_file': 'tokenizer.json'}

    def __init__(self, **kwargs):
        # Build a WordLevel tokenizer for the MsaPairformer vocabulary
        tokenizer_object = Tokenizer(model=models.WordLevel(vocab=aa2tok_d, unk_token='X'))
        tokenizer_object.pre_tokenizer = pre_tokenizers.Split('', behavior='isolated')
        tokenizer_object.decoder = decoders.Sequence([decoders.Replace('', '')])

        super().__init__(
            name_or_path='yoakiyama/MSA-Pairformer',
            tokenizer_object=tokenizer_object,
            # SpecialTokensMixin args
            unk_token='X',
            pad_token='<pad>',
            mask_token='<mask>',
            **kwargs
        )


if __name__ == '__main__':
    # Option 2 (available online or as tokenizer.json file through AutoTokenizer)
    # Build a WordLevel tokenizer for the MsaPairformer vocabulary
    tokenizer_backend = Tokenizer(model=models.WordLevel(vocab=aa2tok_d, unk_token='X'))
    tokenizer_backend.pre_tokenizer = pre_tokenizers.Split('', behavior='isolated')
    tokenizer_backend.decoder = decoders.Sequence([decoders.Replace('', '')])

    # Wrap with HF PreTrainedTokenizerFast
    msa_pairformer_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        unk_token='X',
        pad_token='<pad>',
        mask_token='<mask>'
    )
    msa_pairformer_tokenizer.push_to_hub(
        repo_id='yoakiyama/MSA-Pairformer',
        commit_message='Updating padding and mask token of fast tokenizer',
        revision="refs/pr/1"
    )
