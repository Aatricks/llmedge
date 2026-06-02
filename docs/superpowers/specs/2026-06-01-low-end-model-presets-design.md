# Low-end model presets for llmedge — design

**Date:** 2026-06-01
**Status:** Implemented + unit-tested (`ModelPresets.bitnet`, `ModelPresets.smolVlm2`)
**Branch:** `feat/low-end-models`

## Summary

Add ready-to-use model presets to llmedge for two low-end-device-friendly models that the
bundled runtime (`ikawrakow/ik_llama.cpp`) already supports, plus an optional appendix for Bonsai:

| Model | Type | Quant | Size | Verdict |
|---|---|---|---|---|
| **Microsoft BitNet b1.58 2B4T** | text LLM (1-bit) | `IQ2_BN_R4` | ~988 MB | ship |
| **SmolVLM2-256M-Video-Instruct** | vision (VLM) | `Q8_0` + mmproj | ~280 MB | ship |
| **Bonsai** (ternary) | text LLM | n/a on this runtime | — | appendix only |

This is **Track A**. A separate spec (`…-safetensors-conversion-design.md`, Track B) covers the
in-library safetensors→GGUF converter the user also requested; it is intentionally **out of scope here**
so these presets can land without blocking on a much larger subsystem.

## Context (verified)

- The `llama.cpp` submodule is actually **`ikawrakow/ik_llama.cpp`** pinned at `78977c0c` (2026-04-24).
  Its `ggml.h` defines `GGML_TYPE_IQ1_BN` / `IQ2_BN` / `IQ2_BN_R4` (ternary/BitNet), compiled into the
  arm64 JNI targets via the IQK kernels. It does **not** define `Q2_0` / `TQ1_0` / `TQ2_0`.
- The multimodal path uses `mtmd` + `clip` (LLaVA-style: base GGUF + mmproj GGUF). `clip` supports
  ~20 vision architectures including `smolvlm`/`idefics3` — **not** any Microsoft-native VLM
  (Florence-2, Phi-3/3.5-vision).
- Models are declared as `ModelSpec` presets. `DefaultModelCatalog` (internal) holds the built-in
  specs; `ModelRegistry` wires them as `config.models.*` defaults. The example app has no hardcoded
  picker — apps select models by passing a `ModelSpec` or overriding `LLMEdgeConfig.models`.
- **Chat template flow:** `LLMInference.loadModel(path, chatTemplate, …)`. When `chatTemplate` is empty,
  native falls back to the GGUF's embedded template. The Kotlin side sets the template at the
  **`TextClient` options** level (`options.chatTemplate`), **not** on `ModelSpec`. `ModelHints` currently
  has only `{artifactKind, capabilities}` — no template field.

## Problem this solves

1. Users have to hand-type repo/filename for good low-end models. Presets make BitNet + SmolVLM2
   first-class.
2. **BitNet's embedded chat template is wrong** (per the GGUF uploader). A naive preset would load the
   wrong template and produce garbled output — a silently-broken preset. The preset must carry and apply
   the correct template itself.

## Design

### 1. `ModelHints` carries an optional chat template

Add a field so a text preset is self-contained:

```kotlin
data class ModelHints(
    val artifactKind: ModelArtifactKind = ModelArtifactKind.AUTO,
    val capabilities: Set<ModelCapability> = emptySet(),
    val chatTemplate: String? = null,   // NEW: applied at load when caller didn't override
)
```

Wiring rule (text load path): the effective template is
`callerOptions.chatTemplate ?: resolvedSpec.hints.chatTemplate` (then native's GGUF-embedded fallback).
Caller override always wins; otherwise the preset's template is used. Exact thread-through point
(`TextRuntimeSupport` / `TextRuntimeSession` where the resolved `ModelSpec` + options meet `loadModel`)
is nailed in the plan. Include the spec's `hints.chatTemplate` in the runtime cache key (alongside the
existing `chatTemplate` token) so two specs differing only by template don't collide.

This keeps `ModelSpec` unchanged (it already carries `hints`) and makes BitNet correct out of the box.

### 2. Catalog entries (internal) + a public presets surface

`DefaultModelCatalog` is `internal`, so end users cannot reference its entries. Add the new specs there
**and** expose a **public** presets object so apps can use them directly:

```kotlin
// internal DefaultModelCatalog additions
val bitnetText = huggingFaceSpec(
    repoId = "tdh111/bitnet-b1.58-2B-4T-GGUF",
    filename = "bitnet1582b4t-iq2_bn_r4.gguf",
    preferredQuantizations = emptyList(),
    artifactKind = ModelArtifactKind.GGUF_MODEL,
    capabilities = setOf(ModelCapability.TEXT),
    chatTemplate = BitNetTemplate.CHAT,   // pinned, see Risks
)

val smolVlm2Model = huggingFaceSpec(
    repoId = "ggml-org/SmolVLM2-256M-Video-Instruct-GGUF",
    filename = "SmolVLM2-256M-Video-Instruct-Q8_0.gguf",
    preferredQuantizations = emptyList(),
    artifactKind = ModelArtifactKind.GGUF_MODEL,
    capabilities = setOf(ModelCapability.TEXT, ModelCapability.VISION),
)
val smolVlm2Projector = huggingFaceSpec(
    repoId = "ggml-org/SmolVLM2-256M-Video-Instruct-GGUF",
    filename = "mmproj-SmolVLM2-256M-Video-Instruct-Q8_0.gguf",
    preferredQuantizations = emptyList(),
    artifactKind = ModelArtifactKind.PROJECTOR,
    capabilities = setOf(ModelCapability.PROJECTOR),
)
```

```kotlin
// NEW public surface — io.aatricks.llmedge.model.ModelPresets
object ModelPresets {
    /** Microsoft BitNet b1.58 2B4T, 1-bit (IQ2_BN_R4). ~988 MB. Self-contained chat template. */
    val bitnet: ModelSpec = DefaultModelCatalog.bitnetText
    /** SmolVLM2-256M vision base + projector. ~280 MB total. */
    val smolVlm2: VisionModels = VisionModels(
        model = DefaultModelCatalog.smolVlm2Model,
        projector = DefaultModelCatalog.smolVlm2Projector,
    )
}
```

`huggingFaceSpec(...)` gets a `chatTemplate` parameter passed into `ModelHints`.

Defaults in `ModelRegistry` are **unchanged** (text stays SmolLM-135M, vision stays llava-phi-3) so we
don't force large downloads on existing users. The new models are opt-in via `ModelPresets`.

### 3. `BitNetTemplate` constant

Add an internal `BitNetTemplate.CHAT` string holding the canonical BitNet b1.58 2B4T chat template
(pinned from the Microsoft model card — see Risks). Used by the catalog entry.

### 4. Docs

- `docs/usage.md` — under "Downloading Models" / vision: show `ModelPresets.bitnet` and
  `ModelPresets.smolVlm2` usage; note BitNet works without manually setting a template.
- `README.md` — add the two models to the feature/model notes; flag BitNet ~988 MB.

## Usage (target API)

```kotlin
val edge = LLMEdge.create(context, scope)

// BitNet — template applied automatically from the preset
val reply = edge.text.complete(prompt = "Hi", model = ModelPresets.bitnet)

// SmolVLM2 vision
val desc = edge.vision.analyze(
    image = bitmap,
    model = ModelPresets.smolVlm2.model,
    projector = ModelPresets.smolVlm2.projector,
)
```

## Data flow (existing plumbing, unchanged)

- **Text:** `ModelPresets.bitnet` → `ModelRepository.prefetch` (HF download → cache) →
  `TextClient.complete(model=…)` → effective template resolved (preset's `hints.chatTemplate`) →
  `LLMInference.loadModel(gguf, template)`.
- **Vision:** base+projector specs → `VisionClient.analyze` → `mtmd` projector (`clip` smolvlm arch) +
  base model.

No native/CMake/submodule changes — both quants and the smolvlm clip arch are already compiled in.

## Risks & how the plan resolves them

1. **BitNet template correctness (highest).** Wrong template = garbled output. The plan must pin the
   canonical template from `microsoft/bitnet-b1.58-2B-4T` `tokenizer_config.json` (chat_template) and
   verify a rendered sample matches the documented BitNet format
   (`<|begin_of_text|>` … `System:`/`User:`/`Assistant:` … `<|eot_id|>`). Do not trust the GGUF's embedded one.
2. **Exact HF filenames.** BitNet `bitnet1582b4t-iq2_bn_r4.gguf` verified on `tdh111/...`. Re-confirm the
   SmolVLM2 base + `mmproj-` filenames on `ggml-org/SmolVLM2-256M-Video-Instruct-GGUF` at implementation
   (use the `HFFileSelectionSupport` listing).
3. **SmolVLM2 end-to-end** through the existing `VisionPipeline`/`mtmd` on the 256M build — confirm
   projector init + an analyze call returns sane output (the `<__media__>` marker + clip smolvlm path).
4. **Template threading cache-key collision** — include `hints.chatTemplate` in the runtime cache token.
5. **Size** — BitNet 988 MB is the heavy one; document it. SmolVLM2 ~280 MB is fine.

## Testing

- **Unit:** catalog/registry/`ModelPresets` wiring; `ModelHints.chatTemplate` precedence
  (caller override > preset > embedded); `BitNetTemplate.CHAT` non-empty + renders.
- **Integration (`androidTest`, emulator/device):** download+load `ModelPresets.bitnet`, generate with
  the auto-applied template; load SmolVLM2 base+projector and run one `analyze`. Gate behind the existing
  network/model-download test conventions.

## Out of scope (→ Track B spec)

- In-library safetensors **direct loading / lossy loading / conversion**. Tracked separately as the
  safetensors→GGUF converter (one converter, precision param: F16 = "direct", quantized = "lossy"),
  v1 scoped to Llama/Mistral. Gated on confirming Bonsai's ternary weight storage.

## Appendix: Bonsai (optional, not default)

Bonsai has no GGUF loadable by this runtime (its only GGUF is `Q2_0`, exclusive to PrismML's fork; the
0.5B is safetensors-only). The only path onto ik_llama is an **offline** convert
(`convert_hf_to_gguf.py` → `llama-quantize --outtype iq2_bn`), then sideload the resulting GGUF or host it
and reference via `ModelSpec.huggingFace(...)` / `ModelSpec.localFile(...)`. Documented as a recipe; not a
built-in preset. (Becomes cleaner once Track B's converter exists.)
