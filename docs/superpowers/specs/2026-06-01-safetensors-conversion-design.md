# Safetensors → GGUF conversion as a library option — design (Track B)

**Date:** 2026-06-01
**Status:** Design for approval
**Branch:** `feat/low-end-models`
**Depends on:** Track A presets spec (independent; can land before or after)

## What the user asked

> implement safetensors direct loading **or** lossy precision safetensors loading **or** conversion
> as an option in the library

These three collapse to **one converter with a precision parameter**:

- **"direct loading"** = convert to **F16** GGUF (no precision loss) and load it — functionally
  "load the safetensors as-is" from the caller's view, with a transparent, cached GGUF intermediate.
- **"lossy loading"** = convert to a **quantized** GGUF (`Q8_0` / `Q4_K_M` / `IQ2_BN`).
- **"conversion as an option"** = the converter itself, exposed through the library.

We build **one** thing: a safetensors→GGUF pipeline with `precision ∈ {F16, Q8_0, Q4_K_M, …}`.

## Why GGUF stays in the middle (no true "direct" runtime load)

The LLM backend is `ik_llama.cpp` (GGML). Its model loader (`llama_model_load_from_file`) consumes
**GGUF only**; there is no safetensors code path for LLMs, and the SD safetensors reader (`sd.cpp`) is a
separate codebase with no tokenizer/llama-name-mapping logic. A real "load safetensors directly into the
LLM runtime" means reimplementing the GGUF loader against safetensors — large, per-architecture, and it
buys nothing a cached F16 GGUF doesn't. So **every option is "convert → load GGUF"**, differing only by
precision and *where* the conversion runs.

## The hard part is the tokenizer, not the tensors

Reading safetensors is trivial (8-byte header length + JSON header + raw tensor blobs, per-tensor offsets
→ streamable one tensor at a time, so RAM is bounded). The genuinely hard, fragile work is **baking the HF
tokenizer into GGUF metadata** (vocab, merges, scores, special tokens, pre-tokenizer regex) — upstream
`convert_hf_to_gguf.py` spends thousands of lines on tokenizer variants. Any native reimplementation only
supports the few tokenizers it explicitly handles.

## Verified building blocks (already in-repo)

- `llama.cpp/convert_hf_to_gguf.py` + `gguf-py/` — the full HF→GGUF converter, **ships in the submodule**.
- `gguf_init_empty` / `gguf_set_val_*` / `gguf_add_tensor` / `gguf_write_to_file` (`ggml.h`) — native GGUF writer.
- `llama_model_quantize` (`llama.h:663`) — native GGUF→GGUF quantize.
- ik_llama quant types available as targets: `F16`, `Q8_0`, `Q4_K_M`, … and `IQ2_BN` (ternary).

## Bonsai is the motivating model — and needs a model-specific pre-step

`deepgrove/Bonsai` (0.5B) is `QLlamaForCausalLM` (`model_type: llama`) but:

- weights are **BF16 unpacked ternary** {-1,0,+1} **plus per-output-channel `scales`** applied
  **post-matmul** (`QLinear`); requires `trust_remote_code` / `modeling_qllama.py`.
- **stock `convert_hf_to_gguf.py` fails** (can't instantiate QLlama; post-matmul channel scales aren't
  GGUF-native).

**Escape hatch (math):** `y = (x·Wᵀ)·scales = x·(W·scales[:,None])ᵀ`. Fold the per-row scale into the
weight → `W_eff = W_ternary × scale` → a normal real-valued Llama weight → stock-convertible → quantize.
This is **lossy vs. the native 1.58-bit format** (lossless ternary only exists in PrismML's `Q2_0` fork,
which ik_llama lacks), but yields a working low-end GGUF (0.5B @ `Q4_K_M` ≈ 350 MB).

Consequence: the converter cannot be fully generic — **Bonsai needs a small `QLinear` scale-fold adapter**
before the stock convert path. Plain `LlamaForCausalLM` safetensors need no adapter.

## Design: phased, recommendation-first

### Phase B1 — host/build-time converter + library consumption  (RECOMMENDED, ship first)

**Where it runs: a dev box / CI, not the phone.** "Low-end devices" is the North Star — converting a
~1 GB fp model is desktop-class work we do **once**, never on the weakest device.

1. **Repo tool** `tools/safetensors-convert/` (thin wrapper + Gradle task over the in-repo
   `convert_hf_to_gguf.py` + a quantize step):
   - `--source <hf-repo|local-dir>`  `--precision {f16|q8_0|q4_k_m|iq2_bn}`  `--out <file.gguf>`
   - `--adapter bonsai-qlinear` → numpy/safetensors pre-pass: read `scales`, compute `W_eff = W×scale`,
     drop scale tensors, rewrite as stock Llama, then run the normal converter.
   - generic stock-Llama/Mistral models work with no adapter.
2. **Library consumption API** (the "option in the library"):
   - Existing `ModelSpec.huggingFace(...)` / `ModelSpec.localFile(...)` already load any produced GGUF.
   - Add `ModelSpec.safetensors(source, precision)` that **resolves** to a converted GGUF: looks for a
     cached/companion converted file (cache key includes source + precision); if present, loads it; if
     absent, fails fast with an actionable error naming the `tools/safetensors-convert` command (and, if
     B2 is built, transparently invokes the on-device converter instead).
   - `precision` enum on the spec → F16 ("direct") vs quantized ("lossy").
3. **Bonsai** becomes a documented, supported model end-to-end via the tool → sideload/host the GGUF →
   `ModelSpec.localFile(...)` / `huggingFace(...)`. Folds into Track A's appendix.

Deliverable surface: one Python adapter + Gradle task + one new `ModelSpec` variant + resolver hook + docs.
Effort: days. Risk: low (reuses upstream converter).

### Phase B2 — on-device native converter, stock-Llama only  (OPTIONAL, larger, later)

Only if a concrete need to convert *on the device* emerges (rare for low-end).

- JNI converter: hand-rolled native safetensors reader (streaming, per-tensor) → HF→llama tensor-name map
  for Llama-family → **bake Llama/SPM tokenizer from `tokenizer.json`** into GGUF (the fragile core) →
  `gguf_*` write → optional `llama_model_quantize`.
- Same `precision` param (F16 = "direct", quantized = "lossy").
- **Explicitly excludes custom-modeling models** (Bonsai's QLlama — needs the host fold step) and any
  tokenizer the native baker doesn't implement.
- RAM bounded by streaming, but conversion CPU/time is still heavy on low-end — document and gate (e.g.
  refuse below a RAM threshold).
- Effort: weeks. Risk: high (tokenizer fidelity). Recommend deferring until B1 is in use.

## Recommendation

Build **B1 now**. It satisfies "conversion as a library option" with a real `ModelSpec.safetensors(…,
precision)` API and unblocks Bonsai, reusing the converter that already ships in-repo — without a
weeks-long, fragile native tokenizer reimplementation that runs desktop-class work on the weakest phones.
Treat **B2 as a separate, opt-in phase** with its own plan, pursued only if on-device conversion is truly
required.

## Open question for the user

Should v1 target **only Bonsai + stock Llama/Mistral**, or aim for **arbitrary HF architectures**?
Recommendation: Llama/Mistral + the Bonsai adapter for v1 — arbitrary-arch support is mostly upstream
converter coverage and can grow incrementally.

## Risks

1. **Tokenizer fidelity** (B2 especially) — wrong vocab/merges = subtly broken generation. B1 sidesteps
   this by reusing the upstream Python tokenizer baker.
2. **Bonsai fold correctness** — verify `W_eff` produces matching logits vs. the `trust_remote_code`
   reference on a few prompts before trusting the GGUF.
3. **Quantizing folded ternary** — `Q4_K_M`/`IQ2_BN` of folded weights drifts from native ternary; measure
   perplexity vs. the reference and document expected quality.
4. **`ModelSpec.safetensors` UX** — must fail loudly and actionably when no converted GGUF exists, never
   silently download a 1 GB safetensors the runtime can't load.

## Testing

- **B1:** unit-test the Bonsai fold (`W_eff` equals `W×scale`); golden-file the converted GGUF metadata;
  smoke-test loading a small converted model. Verify the resolver's cache-hit / missing-GGUF error paths.
- **B2 (if built):** tokenizer round-trip tests (encode/decode parity vs. HF), per-arch load tests,
  RAM-gate behavior.
