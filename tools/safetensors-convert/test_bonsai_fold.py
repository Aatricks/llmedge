"""Unit tests for the Bonsai scale-fold math. Runnable with numpy alone (no torch/model download).

    python -m pytest tools/safetensors-convert/test_bonsai_fold.py
    # or, without pytest:
    python tools/safetensors-convert/test_bonsai_fold.py
"""

import numpy as np

from bonsai_fold import fold_scales, num_folded, rewrite_config


def test_row_scale_matches_post_matmul_scaling():
    rng = np.random.default_rng(0)
    # Ternary weight {-1,0,+1}, shape [out=4, in=3]; per-output scale [4].
    w = rng.integers(-1, 2, size=(4, 3)).astype(np.float32)
    scale = rng.standard_normal(4).astype(np.float32)
    x = rng.standard_normal((5, 3)).astype(np.float32)

    tensors = {"layer.weight": w.copy(), "layer.scales": scale.copy()}
    folded = fold_scales(tensors)

    # QLinear semantics: y = (x @ W^T) * scale  must equal  x @ W_eff^T
    expected = (x @ w.T) * scale
    got = x @ folded["layer.weight"].T
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_scales_are_dropped_and_others_pass_through():
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": np.ones((6, 4), np.float32),
        "model.layers.0.self_attn.q_proj.scales": np.full((6,), 2.0, np.float32),
        "model.embed_tokens.weight": np.ones((8, 4), np.float32),  # no scale -> untouched
        "model.layers.0.input_layernorm.weight": np.ones((4,), np.float32),
    }
    assert num_folded(tensors) == 1
    out = fold_scales(tensors)

    assert not any(k.endswith(".scales") for k in out), "scales must be removed"
    assert "model.embed_tokens.weight" in out
    np.testing.assert_array_equal(out["model.embed_tokens.weight"], np.ones((8, 4), np.float32))
    # q_proj weight was all ones, scale all twos -> folded all twos
    np.testing.assert_array_equal(
        out["model.layers.0.self_attn.q_proj.weight"], np.full((6, 4), 2.0, np.float32)
    )


def test_missing_weight_raises():
    try:
        fold_scales({"x.scales": np.ones((3,), np.float32)})
    except KeyError:
        return
    raise AssertionError("expected KeyError for scale without matching weight")


def test_shape_mismatch_raises():
    try:
        fold_scales(
            {"x.weight": np.ones((4, 3), np.float32), "x.scales": np.ones((5,), np.float32)}
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for out_features mismatch")


def test_rewrite_config_strips_custom_modeling():
    cfg = {
        "architectures": ["QLlamaForCausalLM"],
        "model_type": "llama",
        "auto_map": {"AutoModelForCausalLM": "modeling_qllama.QLlamaForCausalLM"},
        "hidden_size": 1536,
    }
    out = rewrite_config(cfg)
    assert out["architectures"] == ["LlamaForCausalLM"]
    assert out["model_type"] == "llama"
    assert "auto_map" not in out
    assert out["hidden_size"] == 1536  # untouched
    assert cfg["architectures"] == ["QLlamaForCausalLM"]  # input not mutated


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
