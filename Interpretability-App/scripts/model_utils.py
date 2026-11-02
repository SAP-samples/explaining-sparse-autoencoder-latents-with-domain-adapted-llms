"""
model_utils.py — shared helpers for activation collection scripts.

Provides architecture-agnostic layer access for any HuggingFace causal LM.
"""

import sys


def get_layer_module(model, layer_idx: int):
    """
    Return the transformer block at layer_idx for any common architecture.

    Tries these attribute paths in order:
      model.model.layers[i]                       — Gemma-2, LLaMA-2/3, Mistral, Qwen2, Yi, DeepSeek, Phi-3
      model.language_model.model.layers[i]        — Gemma-3/4 ForConditionalGeneration (VLM wrapper)
      model.model.decoder.layers[i]               — OPT
      model.transformer.h[i]                      — GPT-2, Falcon, BLOOM-style
      model.gpt_neox.layers[i]                    — GPT-NeoX, Pythia
      model.model.blocks[i]                       — some MoE / custom models

    Raises ValueError with a diagnostic if none match.
    """
    PATHS = [
        lambda m: m.model.layers[layer_idx],
        lambda m: m.language_model.model.layers[layer_idx],  # Gemma3/4 VLM
        lambda m: m.model.decoder.layers[layer_idx],
        lambda m: m.transformer.h[layer_idx],
        lambda m: m.gpt_neox.layers[layer_idx],
        lambda m: m.model.blocks[layer_idx],
    ]
    for path_fn in PATHS:
        try:
            module = path_fn(model)
            if module is not None:
                return module
        except (AttributeError, IndexError, TypeError):
            continue

    top_attrs = [a for a in dir(model) if not a.startswith("_")]
    raise ValueError(
        f"Could not find layer {layer_idx} on model of type {type(model).__name__}.\n"
        f"Top-level attributes: {top_attrs[:30]}\n"
        f"Hint: inspect model.named_modules() to find the correct layer path."
    )


def count_layers(model) -> int:
    """Return the total number of transformer blocks in the model."""
    PATHS = [
        lambda m: len(m.model.layers),
        lambda m: len(m.language_model.model.layers),  # Gemma3/4 VLM
        lambda m: len(m.model.decoder.layers),
        lambda m: len(m.transformer.h),
        lambda m: len(m.gpt_neox.layers),
        lambda m: len(m.model.blocks),
    ]
    for path_fn in PATHS:
        try:
            n = path_fn(model)
            if n is not None and n > 0:
                return n
        except (AttributeError, TypeError):
            continue
    return -1
