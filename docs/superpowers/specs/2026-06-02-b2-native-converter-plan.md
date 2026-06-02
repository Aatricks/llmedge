# B2 — on-device native safetensors → GGUF converter (implementation plan)

**Date:** 2026-06-02
**Status:** In progress (multi-session). Branch `feat/low-end-models`.
**Parent spec:** `2026-06-01-safetensors-conversion-design.md`

## Decision: reimplement in C++ (reuse ruled out)

`llama.cpp/convert_hf_to_gguf.py` has a hard top-level `import torch` (254 uses, `LazyTorchTensor`).
Embedding it on-device (Chaquopy/CPython) would require full PyTorch in the APK — impractical. Confirmed,
not assumed. So B2 is a native C++/JNI reimplementation.

## Honest scope (v1)

**One vertical slice: Llama architecture + SentencePiece tokenizer** (covers Bonsai/Llama). NOT "arbitrary
architectures" — `convert_hf_to_gguf` is multi-month of per-arch/per-tokenizer logic. Coverage grows
incrementally; anything unsupported **fails loudly** (clear error), never emits a broken GGUF. The
`ModelSpec.safetensors` API stays arbitrary-looking, but B2's engine is explicitly single-arch in v1.

## The oracle is non-negotiable and comes first

"Generates non-empty / looks coherent" already hid pure garbage this session. A converter that flips a
tensor layout or a tokenizer score produces *plausible* garbage. So the verification oracle is a
**ground-truth diff against an official GGUF**:

- Reference model: **SmolLM-135M** (Llama-arch, tiny, ships both safetensors and an official GGUF).
- Convert safetensors → GGUF at **F16**, compare against an **F16/Q8 reference at the same precision**
  (never vs a Q4 — quant noise masks bugs).
- Assertion: **identical greedy tokens** (or matching first-token logits within tolerance) on fixed
  prompts, run through the host `libsmollm`.

Build the oracle harness before trusting any conversion output.

## Isolate the two halves (verify independently)

1. **Tensors + hparams** — convert tensors + config.json hparams, but **borrow the `tokenizer.ggml.*` KV
   from the official GGUF**. Validates the tensor/hparam path alone via the logit oracle.
2. **Tokenizer baking** — emit `tokenizer.ggml.*` from `tokenizer.model`/`tokenizer.json`, validated by
   **diffing the emitted KV arrays against the official GGUF's arrays** (tokens, scores, token_type,
   merges, special-token ids). Do not combine with (1) until both pass independently.

## Components (`llmedge/src/main/cpp/convert/`)

- `safetensors_reader.{h,cpp}` — parse the 8-byte header-len + JSON header; mmap tensor data; yield
  `{name, dtype, shape, offset, nbytes}`. Backend-agnostic, host-testable. **(Layer 1 — start here.)**
- `gguf_writer_support` — thin use of ggml `gguf_init_empty` / `gguf_set_val_*` / `gguf_add_tensor` /
  `gguf_write_to_file`, with bf16/f16→f16 casting via ggml.
- `hf_to_gguf.{h,cpp}` — orchestrator: read config.json → arch/hparams → HF→llama tensor-name map (Llama)
  → set GGUF KV → (bake or borrow) tokenizer → write tensors.
- `tokenizer_bake.{h,cpp}` — SentencePiece (`tokenizer.model`, protobuf) → `tokenizer.ggml.*`. BPE later.
- `convert_jni.cpp` — `nativeConvertSafetensors(srcDir, outPath, quantType)`; quantize via the
  already-built `llama_model_quantize`.
- Kotlin `SafetensorsConverter` + wire into `DefaultModelRepository.resolve` (replace the "throw
  instructions" branch with actual conversion when the native converter is available).

## Build verification ladder (each layer independently verified)

1. **Reader** → parse a known safetensors; assert tensor count/names/shapes/dtypes (ground truth:
   Bonsai = 259 BF16 tensors, known shapes). Host unit test.
2. **GGUF writer** → write + read back (round-trip) a GGUF; assert KV/tensors survive.
3. **Tensors+hparams** (borrowed tokenizer) → logit/greedy-token oracle vs official SmolLM-135M GGUF.
4. **Tokenizer baking** → KV diff vs official GGUF.
5. **Full path** → end-to-end safetensors → GGUF → generate, oracle-checked; then quantize (IQ2_BN) +
   the Bonsai QLinear fold adapter (reuse B1's fold logic, ported to C++).

## Out of scope v1 (fail loudly)

Non-Llama archs, BPE/WordPiece/Unigram-only tokenizers, sharded > simple cases, models needing custom
remote code beyond the Bonsai fold. Each emits a clear "unsupported … convert on host with
tools/safetensors-convert" error.
