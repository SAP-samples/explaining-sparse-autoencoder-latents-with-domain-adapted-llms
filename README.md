[![REUSE status](https://api.reuse.software/badge/github.com/SAP-samples/explaining-sparse-autoencoder-latents-with-domain-adapted-llms)](https://api.reuse.software/info/github.com/SAP-samples/explaining-sparse-autoencoder-latents-with-domain-adapted-llms)

# Explaining Sparse Autoencoder Latents with Domain-Adapted LLMs

## Description

This repository contains the code and experimental results for our work on **explaining sparse
autoencoder (SAE) latents using domain-adapted large language models**.

We investigate whether continued-pretrained (CPT), domain-adapted LLMs produce better
natural-language explanations of SAE and CrossCoder features than their base counterparts, across
two domains — SAP (using Gemma-3-12B) and a finance/Salesforce domain (using Llama-3-8B). Features
are extracted with JumpReLU SAEs and CrossCoders, explained under several conditions (BASE / CPT /
SOTA explainers, with and without retrieval-augmented generation), and scored with an
LLM-as-a-Judge protocol with bootstrap confidence intervals.

The repository has three parts:

| Folder | Contents |
|---|---|
| [`Paper-Code/`](Paper-Code/) | The end-to-end evaluation pipeline: dataset generation → activation collection → SAE training → explanation + LLM-judge scoring → bootstrap confidence intervals. See [`Paper-Code/PIPELINE.md`](Paper-Code/PIPELINE.md). |
| [`Paper-Results/`](Paper-Results/) | Raw per-feature result files (JSON) and MMLU results produced by the pipeline, used to compute the figures and tables in the paper. |
| [`Interpretability-App/`](Interpretability-App/) | An interactive CrossCoder interpretability web app that trains CrossCoders on two models and serves a feature browser with LLM-as-a-Judge labels. See [`Interpretability-App/README.md`](Interpretability-App/README.md). |

## Requirements

- Python 3.10+
- A CUDA-capable GPU (experiments were run on an SLURM cluster with NVIDIA GPUs)
- [PyTorch](https://pytorch.org) built for your CUDA version (e.g. `torch==2.6.0` with CUDA 12.4)
- Access to an LLM inference backend for the LLM-as-a-Judge steps. The code targets
  [SAP generative-AI-hub / AI Core](https://help.sap.com/docs/sap-ai-core) via the `gen-ai-hub`
  SDK, configured entirely through environment variables (see **Download and Installation**).
- For the RAG baseline: a [Qdrant](https://qdrant.tech/) instance and the BGE-M3 embedding model.

Python dependencies for the web app are listed in
[`Interpretability-App/requirements.txt`](Interpretability-App/requirements.txt).

## Download and Installation

```bash
git clone https://github.com/SAP-samples/explaining-sparse-autoencoder-latents-with-domain-adapted-llms.git
cd explaining-sparse-autoencoder-latents-with-domain-adapted-llms

# Interpretability web app
cd Interpretability-App
pip install -r requirements.txt
```

The LLM-as-a-Judge and explainer steps read all credentials from environment variables — nothing is
hardcoded. Set these before running those steps (e.g. in a `.env` file):

```bash
AICORE_AUTH_URL=...        # SAP AI Core OAuth token endpoint
AICORE_CLIENT_ID=...        # SAP AI Core client id
AICORE_CLIENT_SECRET=...    # SAP AI Core client secret
AICORE_BASE_URL=...         # SAP AI Core base URL (/v2)
AICORE_RESOURCE_GROUP=...   # SAP AI Core resource group
```

To reproduce the paper pipeline, follow the step-by-step instructions in
[`Paper-Code/PIPELINE.md`](Paper-Code/PIPELINE.md). The pipeline scripts are written for an SLURM
cluster and contain cluster-specific paths and model names that must be adapted to your environment.

## Known Issues

- The pipeline scripts under [`Paper-Code/`](Paper-Code/) contain absolute cluster paths, SLURM
  directives, and internal model checkpoint names from the original research environment. They are
  provided for transparency and reproducibility and are expected to be adapted before use elsewhere.
- Some result files under [`Paper-Results/`](Paper-Results/) are large (up to ~64 MB).

## How to obtain support

[Create an issue](https://github.com/SAP-samples/explaining-sparse-autoencoder-latents-with-domain-adapted-llms/issues)
in this repository if you find a bug or have questions about the content.

For additional support, [ask a question in SAP Community](https://answers.sap.com/questions/ask.html).

## Contributing

If you wish to contribute code, offer fixes or improvements, please send a pull request. Due to
legal reasons, contributors will be asked to accept a DCO when they create the first pull request to
this project. This happens in an automated fashion during the submission process. SAP uses
[the standard DCO text of the Linux Foundation](https://developercertificate.org/). See
[CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

Copyright 2026 SAP SE or an SAP affiliate company and explaining-sparse-autoencoder-latents-with-domain-adapted-llms contributors. Please see our [LICENSE](LICENSE) for copyright and license information. Detailed information including third-party components and their licensing/copyright information is available [via the REUSE tool](https://api.reuse.software/info/github.com/SAP-samples/explaining-sparse-autoencoder-latents-with-domain-adapted-llms).
