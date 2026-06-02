# safetensors → GGUF converter (Track B / Phase B1)

Host-side tool that turns a Hugging Face safetensors model into a GGUF the llmedge runtime
(`ik_llama.cpp`) can load. One converter, a `--precision` parameter:

| `--precision` | meaning | needs |
|---|---|---|
| `f16` | "direct" load, no precision loss | upstream converter only |
| `q8_0` | light, near-lossless | upstream converter only |
| `q4_k_m` / `q5_k_m` | "lossy" k-quant | a built `llama-quantize` |
| `iq2_bn` / `iq2_bn_r4` | ik_llama ternary (ternary models only) | a built `llama-quantize` |

Runs on a dev box / CI — **never on-device** (that's Phase B2). Converting a model loads it in full
precision, which is desktop-class work; do it once and ship the GGUF.

## Setup

```bash
git submodule update --init llama.cpp          # provides convert_hf_to_gguf.py + gguf-py
python3 -m venv .venv && source .venv/bin/activate
pip install -r tools/safetensors-convert/requirements.txt
```

## Use

```bash
cd tools/safetensors-convert

# Stock Llama/Mistral/Qwen… -> F16 (arbitrary archs inherit upstream converter coverage)
python convert.py --source unsloth/Qwen3-0.6B --precision f16 --out qwen3-0.6b-f16.gguf

# Bonsai 0.5B: fold the QLinear per-output scales, then quantize
python convert.py --source deepgrove/Bonsai --adapter bonsai-qlinear \
    --precision q8_0 --out bonsai-0.5b-q8_0.gguf
```

For `q4_k_m` / `iq2_bn`, build `llama-quantize` from the submodule (host build, **no Android NDK**):

```bash
cmake -S llama.cpp -B build-host -DGGML_IQK_FLASH_ATTENTION=ON && cmake --build build-host -t llama-quantize
python convert.py --source deepgrove/Bonsai --adapter bonsai-qlinear \
    --precision iq2_bn --out bonsai-0.5b-iq2_bn.gguf --quantize-bin build-host/bin/llama-quantize
```

Then load it from the app via `ModelSpec.localFile("…/bonsai-0.5b-q8_0.gguf")` (sideload) or upload to
Hugging Face and use `ModelSpec.huggingFace(...)` / `ModelSpec.safetensors(...)`.

## Why Bonsai needs the adapter

Bonsai is `QLlamaForCausalLM`: ternary weights in BF16 **plus** a per-output `.scales` applied after the
matmul. Stock `convert_hf_to_gguf.py` can't load it. Since `y = (x·Wᵀ)·scale = x·(W·scale[:,None])ᵀ`, the
adapter folds the scale into the weight (`bonsai_fold.py`) and rewrites the config to plain Llama. This is
**lossy vs. the native 1.58-bit format** (lossless ternary only exists in PrismML's `Q2_0` fork, which
ik_llama lacks), but yields a small, working GGUF.

## Tests

The fold math is unit-tested with numpy alone (no torch / model download):

```bash
pip install numpy
python tools/safetensors-convert/test_bonsai_fold.py
```
