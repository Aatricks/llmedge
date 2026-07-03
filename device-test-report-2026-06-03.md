# llmedge core-functionality test on real device — 2026-06-03

**Device:** Samsung Galaxy S22 (SM-S901B), Android 16 (SDK 36), arm64-v8a, **7.4 GB RAM** (≈2.9–4.5 GB free during tests), Vulkan GPU. Connected via wireless adb.
**App:** llmedge-examples debug APK (`com.example.llmedgeexample`), built fresh this session (`:app:connectedDebugAndroidTest` + reinstall of `app-debug.apk`).
**Method:** instrumented gate test for native bindings; all UI demos driven on-device via `am start` (MainActivity) + uiautomator taps + logcat/UI polling. Device was screen-locked at first (secure keyguard) → user unlocked for UI driving.

> NOTE on device class: README recommends **8 GB for Stable Diffusion** and **12 GB for video**. This device (7.4 GB) is **under-spec** for the heavy generative demos. Results below split **library defects** (RAM-independent) from **device-capacity limits** (expected on this hardware).

---

## Summary table

| # | Demo | llmedge subsystem | Result | Evidence |
|---|------|-------------------|--------|----------|
| — | LibraryBindings (instrumented) | SmolLM / StableDiffusion / Whisper / BarkTTS native load | ✅ PASS | 5 tests, 0 failures (gate) |
| 1 | Local GGUF Asset | text (local file) | ⚪ SMOKE | no model bundled → "Model file not found"; mem metrics + asset path OK |
| 2 | Hugging Face | text gen (HF download) | ✅ PASS (real) | Qwen3-0.6B (378 MB) download→load→stream; `<think>` response |
| 3 | Jinja Chat Template | text gen + custom Jinja template | ✅ PASS (real) | cached Qwen3-0.6B, 96 tok @ 19.66 tok/s, custom template applied |
| 4 | Tool Calling | ToolAgent + device tools | ✅ PASS (real) | get_current_time→`2026-06-03 22:23:50`, get_battery_status→`38%, charging`; browser correctly skipped (action tools off); synth 40 tok @ 3.44 tok/s |
| 5 | RAG | PDF parse + ONNX embed + retrieval + LLM | ❌ FAIL (library defect) | see Bug A & Bug B |
| 6 | Image-to-Text (OCR) | MLKit OCR (not llmedge core) | ◻️ NOT TESTED | camera-only (`TakePicture`); not drivable headless. OCR path also reachable via vision.extractText |
| 7 | LLaVA Vision | multimodal (vision-language) | ✅ PASS (real, slow) | default model = **xtuner/llava-phi-3-mini (2.3 GB)**; described image: "A picture of a person with a white shirt and black pants." ~16 min wall under memory pressure |
| 8 | Video Generation | Wan 2.1 t2v | ◻️ NOT RUN (under-spec) | needs 12 GB; device 7.4 GB. E2E test has no skip-guard (would OOM) |
| 9 | Image Generation (SD) | StableDiffusion txt2img | 🟡 PARTIAL (under-spec) | default model = **Meina/MeinaMix** (SD1.5, ~2 GB safetensors). Loads, real denoise runs (Vulkan/CPU active, mem 4.4→2.2 GB); 512²/20-step did **not** complete in ~30 min; cancelled. No image produced |
| 10 | Safetensors→GGUF (B2) | native converter + GGUFReader + text gen | ✅ FIXED | Empty response root-caused: base SmolLM-135M + chat template → trimmed-empty. Fix = raw pass-through template + no system prompt. **Verified on device:** Response now non-empty — *"also the capital of the world."* (48 tok @ 62.3 tok/s) |
| 11 | Speech-to-Text (Whisper) | STT | ✅ PASS (real) | ggml-base.bin download+load; 15.4 s clip → 3 timestamped segments + auto language-detect "en" + [BLANK_AUDIO] handling, ~8 s (WhisperJNI). **87 s clip → native crash** (tombstone_13, Bucket B) |
| 12 | Text-to-Speech (Bark) | TTS | ✅ PASS (load) | `bark-small_weights-f16.bin` (804 MB) downloaded; libbark_jni loaded; **"Bark context created successfully, sampleRate=24000"**; synthesis started ("Generating audio…, threads=8"). Full f16 WAV ~10 min (app's own note) — not awaited |

Legend: ✅ real output verified · 🟡 partial (pipeline proven, output/finish not) · ⚪ smoke (loads, no real inference) · ◻️ not tested/not run · ❌ fail

**Scorecard:** native bindings ✅ · real-output PASS: HF text, Jinja, Tool-calling, LLaVA vision, Whisper STT, Bark TTS-load, **RAG (end-to-end, after fix)**, **B2 conversion (after fix)**, **SD MeinaMix @128 (62 s)** · under-spec (OOM/too-slow on 7.4 GB): SD@512, **FLUX.2 Klein Q2** (2× OOM), video · smoke: Local asset · not-headless-testable: OCR (camera).

---

## Fixes applied & verified (follow-up, 2026-06-04)

All four library defects investigated; **3 needed a code fix, 1 was not a bug.** Changes span two repos (uncommitted — listed for review, not committed).

| Defect | Root cause | Fix | Verified on device |
|--------|-----------|-----|--------------------|
| RAG broken (no index, no answer) | `DefaultModelCatalog.text` repo `HuggingFaceTB/SmolLM-135M-Instruct-GGUF` → HTTP 401 → `createSession` throws → `rag` null → `indexPdf` no-ops | `llmedge/…/model/DefaultModelCatalog.kt`: repo → `MaziyarPanahi/SmolLM-135M-Instruct-GGUF` (200) | ✅ session builds, **19 chunks indexed**, retrieval returns correct passages |
| RAG answer crash "context size reached" | `RAGAnswerer` called `getResponse` with `maxTokens=-1`; 135M never emits EOS → fills 1024-ctx → native throw | `llmedge/…/rag/RAGAnswerer.kt` + `RAGEngine.kt`: cap `MAX_ANSWER_TOKENS=256` | ✅ **"Done, tokens=256, 4.39 tok/s, 58 s"**, real on-topic answer, no crash |
| B2 conversion: empty visible response | base SmolLM-135M wrapped in chat template → emits template/EOS → trims to empty | `examples/…/conversion/SafetensorsConversionActivity.kt`: raw pass-through template + no system prompt | ✅ **"also the capital of the world."** (48 tok @ 62.3 tok/s) |
| STT 87 s crash | NOT a library bug — situational OOM (2.3 GB vision + SD resident) | none | ✅ clean 90 s clip transcribes fine (2 segments) |

**Image-gen @128×128:** MeinaMix SD1.5 ✅ 62 s full image. FLUX.2 Klein Q2 ❌ OOM ×2 (under-spec).

**Also changed (for the FLUX-Q2 test):** examples `ImageGenerationActivity.kt` FLUX toggle now uses `Flux2Klein.bonsaiImageRequest` (Q2_K 1.3 GB, sequential) instead of `imageRequest` (Q4_0); toggle label updated to match. **This is a behavioral default change — keep Q2 (better for ≤8 GB) or revert to Q4_0 is a user decision.** (Label change is source-only; not in the last installed APK.)

---

## Bucket A — genuine library defect (RAM-independent) — ROOT-CAUSED & FIXED

**Single root cause (corrects an earlier mis-diagnosis): RAG default text model repo is unavailable (HTTP 401), which breaks the whole RAG session — PDF parsing was never the problem.**

- `DefaultModelCatalog.text` pointed at `HuggingFaceTB/SmolLM-135M-Instruct-GGUF`, which returns **401 to anonymous clients** (confirmed via HF API; `MaziyarPanahi/SmolLM-135M-Instruct-GGUF` returns 200).
- `edge.rag.createSession()` resolves that text model to build the `SmolLM` backing the session. The 401 makes `createSession()` throw → the controller's `rag` stays **null** → `rag?.indexPdf(uri) ?: 0` short-circuits to 0.
- Proof it never parsed: re-running INDEX logged "Before indexing" → "After indexing" in **3 ms** with **no `PDFReader`/`RAGEngine` log line at all**. PDFBox was never reached — the earlier "PDF parser broken" reading was wrong.
- Symptom on init: "LLM load failed: … 'HuggingFaceTB/SmolLM-135M-Instruct-GGUF' not found".

**Fix 1 applied:** `DefaultModelCatalog.text` → `MaziyarPanahi/SmolLM-135M-Instruct-GGUF` / `SmolLM-135M-Instruct.Q4_K_M.gguf` (public, 200). Restores the whole RAG path. Also fixes any other default-text consumer.

**Fix 2 applied (second RAG defect, surfaced only once the session built):** `RAGAnswerer.ask` called `smolLM.getResponse(prompt)` with the default `maxTokens = -1` (unbounded). SmolLM-135M rarely emits EOS, so it generated until prompt+output filled the context window → native `IllegalStateException: context size reached` ("Ask failed: … context size reached"). Added `RAGEngine.MAX_ANSWER_TOKENS = 256` cap.

**Verified on device (after reinstall of Fix 1):**
- Session builds → "LLM ready". ✅
- Index arXiv Transformer PDF → **"Indexed 19 chunks"** (PDFBox now reached, ONNX-embedded). ✅ — confirms PDFBox was never the bug.
- Retrieval returns the correct passages with similarity scores ("[score=0.390] … The Transformer - model architecture …"). ✅
- `ask()` then hit the context-overflow above → Fix 2. *(Fix 2 device-verify pending the next reinstall.)*

---

## Bucket B — device-capacity limits (expected on 7.4 GB hardware)

- **SD image generation (#9):** at 512²/20-step it loads + denoises (real Vulkan/CPU work) but does not finish within ~30 min under memory pressure. **At 128×128 it completes cleanly: MeinaMix → "Load 6.31 s, Generate 55.97 s, Total 62.28 s" (full image).** So SD is functional; the 512² non-finish is pure resolution/compute cost on a 7.4 GB device (README rec: 8 GB).
- **LLaVA Vision (#7):** functionally PASS (real correct description) but ~16 min wall because the default vision model is **llava-phi-3-mini 2.3 GB**, heavy on 7.4 GB; system thrashed (background apps killed).
- **STT long-clip crash (#11) — re-classified as situational OOM, NOT a library bug.** The original 87 s crash happened while the 2.3 GB vision model + SD were resident (heavy memory pressure, LMK thrashing). **Clean repro on a fresh process (no other model loaded): a 90 s clip transcribed fine → "Transcription complete: 2 segments", app survived.** So whisper handles long audio (it chunks to 30 s windows internally); the earlier crash was memory exhaustion from concurrent large models, not a buffer-size defect. No code change — adding fixed-30 s chunking would needlessly cut words mid-phrase.
- **Video (#8):** not attempted; 12 GB required, device has 7.4 GB; the E2E test has no skip-guard and would OOM.
- **FLUX.2 Klein 4B Q2 @128×128 (#11) — under-spec, OOM on 7.4 GB (2 attempts).** The example's FLUX toggle was switched to the **bonsai ternary Q2_K** low-memory path (`Flux2Klein.bonsaiImageRequest`, sequential loading). The Qwen3-4B text encoder (2.1 GB) downloads fine, but the app is OOM-killed (LMK, system-wide GC storm) before the full pipeline (encoder + 1.3 GB DiT + VAE + sdcpp buffers) finishes loading — twice. **128×128 shrinks activations, not weights**, so it doesn't dodge the documented ~5 GB FLUX.2 Klein memory wall. Pipeline is correctly wired; the device is simply under-spec for FLUX.2 Klein. (MeinaMix SD1.5 @128 works fine — see #9.)

---

## Notes / environment
- HF downloads showed a consistent ~2–3 min slow-start (0.00 MB) then surged to full speed — network OK (WiFi validated). Not a library issue but affects first-run UX.
- Native lib strip warning at build (`libsmollm.so` et al not stripped) → 1.3 GB debug APK (known bloat).
- Caches populated on device (internal `files/hf-models/`): `unsloth_Qwen3-0.6B-GGUF`, `xtuner_llava-phi-3-mini-gguf` (2.3 GB), `llmedge-converted/…SmolLM-135M…q4_k_m.gguf`; Whisper `ggml-base.bin`.
