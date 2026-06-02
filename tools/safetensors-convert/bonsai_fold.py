"""Bonsai (QLlama) -> stock Llama adapter.

Bonsai stores ternary {-1,0,+1} weights in BF16 plus a per-output-channel ``.scales`` tensor
that ``QLinear`` applies *after* the matmul:

    y = (x @ W^T) * scales        # scales has shape [out_features]

Because the scale is per output row, it folds into the weight exactly:

    y = x @ (W * scales[:, None])^T

so ``W_eff = W * scales[:, None]`` turns each ``QLinear`` back into an ordinary ``nn.Linear``.
After folding (and rewriting ``config.json`` so the architecture is plain ``LlamaForCausalLM``)
the checkpoint is a stock Llama model that the upstream ``convert_hf_to_gguf.py`` can consume.

This module is backend-agnostic: it works with numpy arrays or torch tensors (anything that
supports ``*``, ``.reshape`` and ``.ndim``), so the math can be unit-tested with numpy alone.
"""

from __future__ import annotations

SCALE_SUFFIX = ".scales"
WEIGHT_SUFFIX = ".weight"


def _row_scale(weight, scale):
    """Multiply each output row ``o`` of ``weight`` by ``scale[o]``.

    ``weight`` has shape ``[out, in]`` (or ``[out, ...]``); ``scale`` has shape ``[out]``.
    """
    if scale.ndim != 1:
        raise ValueError(f"scale must be 1-D [out_features], got shape {tuple(scale.shape)}")
    if weight.shape[0] != scale.shape[0]:
        raise ValueError(
            f"out_features mismatch: weight {tuple(weight.shape)} vs scale {tuple(scale.shape)}"
        )
    broadcast = (-1,) + (1,) * (weight.ndim - 1)
    return weight * scale.reshape(broadcast)


def fold_scales(tensors):
    """Return a new tensor dict with every ``X.scales`` folded into ``X.weight`` and dropped.

    Tensors without a paired ``.scales`` (embeddings, lm_head, layernorms) pass through unchanged.
    Raises if a ``.scales`` has no matching ``.weight``.
    """
    scale_keys = [k for k in tensors if k.endswith(SCALE_SUFFIX)]
    consumed = set()
    out = {}
    for sk in scale_keys:
        wk = sk[: -len(SCALE_SUFFIX)] + WEIGHT_SUFFIX
        if wk not in tensors:
            raise KeyError(f"scale tensor {sk!r} has no matching weight {wk!r}")
        out[wk] = _row_scale(tensors[wk], tensors[sk])
        consumed.add(wk)
        consumed.add(sk)
    for k, v in tensors.items():
        if k in consumed:
            continue
        out[k] = v
    return out


def num_folded(tensors) -> int:
    """How many ``.scales`` tensors would be folded (for logging / sanity checks)."""
    return sum(1 for k in tensors if k.endswith(SCALE_SUFFIX))


def rewrite_config(config: dict) -> dict:
    """Rewrite a Bonsai ``config.json`` dict into a stock Llama one.

    Drops the custom-modeling hooks so transformers / the GGUF converter treat it as plain Llama.
    """
    c = dict(config)
    c["architectures"] = ["LlamaForCausalLM"]
    c["model_type"] = "llama"
    for key in ("auto_map", "quantization_config"):
        c.pop(key, None)
    return c
