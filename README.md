# MSA_Pairformer

[![Contact prediction](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

- [MSA Pairformer](#MSA-Pairformer)
  - [Installation ](#installation-)
  - [ESM 3  ](#esm-3--)
    - [Quickstart for ESM3-open ](#quickstart-for-esm3-open-)
    - [ESM3 Example Usage](#esm3-example-usage)
  - [Licenses  ](#licenses--)
  - [Citations  ](#citations--)

This repository contains the latest release of MSA Pairformer and Google Colab notebooks for relevant analyses. Here, you will find how to use MSA Pairformer to embed protein sequences, predict residue contacts, and perform zero-shot variant effect prediction.

## Installation <a name="installation"></a>

To get started with MSA Pairformer, install the python library using pip:

```bash
pip install esm
```

or download this Github repository and install manually
```bash
git clone git@github.com:yoakiyama/MSA_Pairformer.git
pip install -e .
```

## MSA Pairformer <a name="MSA-Pairformer"></a>

[MSA Pairformer](https://arxiv.org/) extracts evolutionary signals most relevant to a query sequence from a set of aligned homologous sequences. Using only 111M parameters, it can easily run on consumer-grade hardware (e.g. NVIDIA RTX 4090) and achieve state-of-the-art performance. In this repository, we provide training code and Google Colab notebooks to reproduce the results in the pre-print. We are excited to deliver this tool to the research community and to see all of its applications to real-world biological challenges.

### Getting started with MSA Pairformer <a name="getting-started"></a>
The model's weights can be downloaded from Huggingface under [HuggingFace/yakiyama/MSA-Pairformer](https://huggingface.co/yakiyama/MSA-Pairformer/).
```py
from huggingface_hub import login
from MSAPairformer.model import MSAPairformer

# Use the GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {torch.cuda.get_device_name(device)}")

# This function will allow you to login to huggingface via an API key
login()

# Download model weights and load model
# As long as the cache doesn't get cleared, you won't need to re-download the weights whenever you re-run this
model = MSAPairformer.from_pretrained().to(device)

# You can also save the downloaded weights to a specified directory in your filesystem.
# Saving the model weights like so will allow you to load the model without re-downloading if your cache gets cleared.
# Once you run this code once, you can re-run and it will automatically load the weights
save_model_dir = "model_weights"
model = MSAPairformer.from_pretrained(weights_dir=save_model_dir)

# Pre-process data
batch_tokens

# Predict contacts and embed query sequence
results_dict = model.get_embeddings_and_contacts()

# Only generate embeddings
embeddings_dict = model.embed_sequences()

# Predict just contacts
contacts_dict = model.predict_contacts()

# Zero-shot variant effect prediction using log probability ratio
# Just replace the residue of interest with a mask token

```

That's it -- you've generated embeddings, predicted contacts, and predicted variant effects using MSA Pairformer!

## Licenses <a name="licenses"></a>

MSA Pairformer code and model weights are released under a permissive, slightly modified 🍕, MIT license. It can be freely used for both academic and commercial purposes.

## Citation <a name="citation"></a>
If you use MSA Pairformer in your work, please use the following citation
```
@article{
  msapairformer,
  author = {Akiyama, Yo and Zhang, Zhidian and Ovchinnikov, Sergey},
  title = {Scaling back protein language models with MSA Pairformer},
  year = {2025},
  journal = {bioRiv}
}
```