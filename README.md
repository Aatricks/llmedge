# llmedge

**llmedge** is a lightweight Android library for running GGUF language models fully on-device, powered by [llama.cpp](https://github.com/ggerganov/llama.cpp).

See the [examples repository](https://github.com/Aatricks/llmedge-examples) for sample usage.

Acknowledgments to Shubham Panchal and upstream projects are listed in [`CREDITS.md`](./CREDITS.md).

> [!NOTE]
> This library is in early development and may change significantly.

> [!IMPORTANT]
> API maturity is uneven by feature area. `LLMEdge`, text inference, speech inference, and model management are the most stable entry points today. OCR via `edge.vision.extractText(...)` is also reliable. Vision/VLM analysis, RAG, and some image/video-generation flows are available and tested, but should still be treated as evolving APIs.

---

## Features

- **LLM Inference**: Run GGUF models directly on Android using llama.cpp (JNI)
- **Model Downloads**: Download and cache models from Hugging Face Hub
- **Low-end Presets**: `ModelPresets` ready-to-use specs (Microsoft BitNet b1.58 2B4T, SmolVLM2-256M) tuned for low-end devices
- **On-device Safetensors → GGUF**: Convert Hugging Face safetensors models on-device (Llama arch + GPT2-BPE tokenizer), with optional quantization (Q8_0 / Q4_K_M / IQ2_BN), via `ModelSpec.safetensors(...)`
- **Optimized Inference**: Native KV cache reuse for compact chats, default batched blocking and streaming text generation, separate prompt vs generation thread tuning, and Kotlin-managed `ChatSession` replay for reasoning-heavy models
- **Speech-to-Text (STT)**: Whisper.cpp integration with timestamp support, language detection, streaming transcription, and SRT generation
- **Text-to-Speech (TTS)**: Bark.cpp integration with ARM optimizations
- **Image Generation**: Stable Diffusion with EasyCache and LoRA support, plus FLUX.2 Klein 4B (distilled DiT, the architecture behind PrismML's binary/ternary Bonsai Image); per-step progress streaming and ESRGAN upscaling (Remacri)
- **Video Generation**: Wan 2.1 models (4-64 frames) with sequential loading
- **On-device RAG**: PDF indexing, embeddings, vector search, Q&A
- **OCR**: Google ML Kit text extraction
- **Memory Metrics**: Built-in RAM usage monitoring
- **Vision Models**: Architecture prepared for LLaVA-style models (requires specific model formats)
- **GPU Acceleration**: Optional Android GPU backends for text, Whisper, and image/video with experimental OpenCL preferred first, Vulkan fallback second, and CPU fallback last

---

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
   - [Downloading Models](#downloading-models)
   - [Reasoning Controls](#reasoning-controls)
   - [Managed Chat Sessions](#managed-chat-sessions)
   - [Tool Calling](#tool-calling)
   - [Image Text Extraction (OCR)](#image-text-extraction-ocr)
   - [Vision Models](#vision-models)
   - [Speech-to-Text (Whisper)](#speech-to-text-whisper)
   - [Text-to-Speech (Bark)](#text-to-speech-bark)
   - [Speech Performance Status](#speech-performance-status)
   - [Stable Diffusion (image generation)](#stable-diffusion-image-generation)
   - [Video Generation](#video-generation)
   - [On-device RAG](#on-device-rag)
   - [Expert APIs](#expert-apis)
3. [Building](#building)
4. [Architecture](#architecture)
5. [Technologies](#technologies)
6. [Memory Metrics](#memory-metrics)
7. [Notes](#notes)
8. [Testing](#testing)

---

## Installation

> [!WARNING]
> For development, Linux is strongly recommended for GPU-enabled builds. The Vulkan shader-generation path used by Stable Diffusion is still unreliable on Windows cross-builds.

Clone the repository along with the `llama.cpp` and `stable-diffusion.cpp` submodule:

```bash
git clone --depth=1 https://github.com/Aatricks/llmedge
cd llmedge
git submodule update --init --recursive
```

Open the project in Android Studio. If it does not build automatically, use ***Build > Rebuild Project.***

### Consume as a dependency

For Maven Central:

```kotlin
repositories {
    google()
    mavenCentral()
}

dependencies {
    implementation("io.github.aatricks:llmedge:0.3.9")
}
```

For GitHub Packages:

```kotlin
repositories {
    google()
    mavenCentral()
    maven {
        url = uri("https://maven.pkg.github.com/Aatricks/llmedge")
        credentials {
            username = providers.gradleProperty("gpr.user").orNull ?: System.getenv("GITHUB_ACTOR")
            password = providers.gradleProperty("gpr.key").orNull ?: System.getenv("GITHUB_TOKEN")
        }
    }
}

dependencies {
    implementation("io.github.aatricks:llmedge:0.3.9")
}
```

## Usage

### Quick Start

The recommended entry point is the instance-based `LLMEdge` facade. It exposes domain clients for text, speech, image generation, vision, and RAG while keeping model resolution and resource ownership explicit.

```kotlin
val edge = LLMEdge.create(
    context = context,
    scope = viewModelScope,
)

viewModelScope.launch {
    val reply = edge.text.generate(
        prompt = "Summarize on-device LLMs in one sentence.",
    )
    outputView.text = reply
}
```

Low-level wrappers like `SmolLM`, `StableDiffusion`, `Whisper`, and `BarkTTS` remain available for expert workflows, but new code should prefer `LLMEdge`.

The intended acquisition path for application code is:

- `edge.models.prefetch(...)` when you want explicit downloads
- feature clients like `edge.text`, `edge.speech`, `edge.image`, and `edge.vision` when you want inference

Direct `HuggingFaceHub` calls and expert runtime `loadFromHuggingFace(...)` helpers are still supported, but they are advanced APIs for callers that need artifact-level control.

By default, `edge.text.generate(...)` uses batched native decoding for lower JNI overhead, while
`edge.text.stream(...)` uses smaller batched chunks so UI updates stay responsive without paying a
JNI crossing per token.

### Downloading Models

llmedge can resolve and cache model weights independently of inference:

```kotlin
val edge = LLMEdge.create(context, viewModelScope)

val modelFile = edge.models.prefetch(
    ModelSpec.huggingFace(
        repoId = "unsloth/Qwen3-0.6B-GGUF",
        filename = "Qwen3-0.6B-Q4_K_M.gguf",
    ),
)

Log.d("llmedge", "Cached ${modelFile.name} at ${modelFile.parent}")
```

#### Key points:

- `edge.models.prefetch(...)` and `BoundModelRepository.resolve(...)` keep model acquisition separate from any one inference client.

- Supports progress callbacks and private repositories via token through `ModelSpec.huggingFace(...)`.

- Requests to old mirrors automatically resolve to up-to-date Hugging Face repos.

- Automatically uses the model's declared context window (minimum 1K tokens) and caps it to a heap-aware limit (2K–8K). Override with `InferenceParams(contextSize = …)` if needed.

- Large downloads use Android's DownloadManager when `preferSystemDownloader = true` to keep transfers out of the Dalvik heap.

- Direct `HuggingFaceHub` downloads remain available for expert workflows, but most app code should stay on the facade/model-repository path.

#### Built-in low-end presets

`ModelPresets` exposes ready-to-use specs tuned for low-end devices, already supported by the bundled
ik_llama.cpp runtime — no need to hand-type repo/filenames:

```kotlin
// Microsoft BitNet b1.58 2B4T — native 1-bit LLM (IQ2_BN, ~988 MB).
// The correct chat template ships on the preset, so generation is well-formed out of the box.
val reply = edge.text.generate(prompt = "Hi", model = ModelPresets.bitnet)

// SmolVLM2-256M — tiny vision model (~280 MB total: base + projector).
val caption = edge.vision.analyze(
    image = bitmap,
    prompt = "Describe this image.",
    model = ModelPresets.smolVlm2.model,
    projector = ModelPresets.smolVlm2.projector,
)
```

> [!NOTE]
> BitNet's GGUF metadata carries an incorrect chat template, so llmedge supplies the canonical one via
> `ModelHints.chatTemplate`. A template you pass through `TextModelOptions.chatTemplate` always overrides it.
> Bonsai and other ternary models distributed only as PrismML `Q2_0` GGUFs are **not** loadable by this
> runtime (it uses `IQ2_BN`); convert the safetensors instead — see
> [Converting safetensors models](docs/usage.md#converting-safetensors-models).

### Reasoning Controls

Reasoning-aware models can be controlled from the facade through `TextModelOptions`. The default configuration keeps thinking enabled (`ThinkingMode.DEFAULT`, reasoning budget `-1`). To disable thinking for a request or session, pass the options explicitly:

```kotlin
val edge = LLMEdge.create(context, viewModelScope)

val reply = edge.text.generate(
    prompt = "Solve this step by step, then give only the final answer.",
    options = TextModelOptions(
        thinkingMode = SmolLM.ThinkingMode.DISABLED,
        reasoningBudget = 0,
    ),
)
```

The same options work with `edge.text.session(...)` and `edge.text.toolAgent(...)`.

Setting the budget to `0` always disables thinking, while `-1` leaves it unrestricted. If you omit `reasoningBudget`, the library chooses `0` when the mode is `DISABLED` and `-1` otherwise. The API also injects the `/no_think` tag automatically when thinking is disabled, so you do not need to modify prompts manually. If you need to flip reasoning state on a live expert runtime without reloading, see [Expert APIs](#expert-apis).

### Managed Chat Sessions

Use `edge.text.session(...)` when you want bounded multi-turn chat without exposing native `storeChats` state to application code.

```kotlin
val edge = LLMEdge.create(context, viewModelScope)

val session = edge.text.session(
    memory = ConversationWindow(
        maxTurns = 6,
        maxTokens = 4096,
        stripThinkTags = true,
    ),
    systemPrompt = "You are a concise assistant.",
)

viewModelScope.launch {
    session.prepare()
    val reply = session.reply("Explain why context windows fill up.")
    session.stream("Now summarize that in 3 bullets.").collect { event ->
        when (event) {
            is TextStreamEvent.Chunk -> print(event.value)
            is TextStreamEvent.Completed -> println(event.fullText)
            else -> Unit
        }
    }
}
```

The new session API keeps transcript state in Kotlin, applies sliding-window trimming, and strips replayed `<think>...</think>` blocks by default so reasoning-heavy models do not exhaust the context window as quickly.

### Tool Calling

Use `edge.text.toolAgent(...)` when you want the model to call app-defined tools. Read-only tools execute automatically; action tools require an explicit policy decision.

```kotlin
val edge = LLMEdge.create(context, viewModelScope)
val factory = DeviceToolFactory(context)

val agent = edge.text.toolAgent(
    tools = factory.createDefaultTools(),
    systemPrompt = "Be concise and only use tools when needed.",
    policy = ToolPolicies.ALLOW_ALL, // or keep the default to deny action tools
)

viewModelScope.launch {
    val result = agent.reply("What time is it and how much battery is left?")
    println(result.text)

    agent.stream("Open https://example.com").collect { event ->
        when (event) {
            is ToolAgentEvent.ToolCallRequested -> println("Tool: ${event.call.tool}")
            is ToolAgentEvent.TextChunk -> print(event.value)
            is ToolAgentEvent.Completed -> println("\nDone: ${event.result.finishReason}")
            else -> Unit
        }
    }
}
```

Tool calls use a structured JSON envelope internally: `{"tool":"name","arguments":{...}}`. The parser also accepts the legacy `tool_name` field for robustness, but new prompts only emit the `tool` shape.

For JVM or desktop hosts where `bash` is available, you can also opt into a shell-execution tool:

```kotlin
val edge = LLMEdge.create(context, viewModelScope)
val bashTool =
    BashToolFactory(
        BashToolOptions(
            allowRawShell = true, // raw `command` strings are disabled unless you opt in
            defaultWorkingDirectory = context.filesDir.absolutePath,
        ),
    ).createBashTool()

val agent = edge.text.toolAgent(
    tools = listOf(bashTool),
    systemPrompt = "Use shell commands only when necessary.",
    policy = ToolPolicies.ALLOW_ALL, // required because run_bash_command is an action tool
)
```

The bash tool accepts either `argv` for structured commands or `command` for raw shell strings. If `bash` is unavailable or the command fails, the tool returns a structured error result instead of claiming success.

### Speech Request Objects

Speech APIs now support request-first calls in addition to the existing convenience overloads:

```kotlin
val result = edge.speech.transcribe(
    SpeechToTextRequest(
        audioSamples = samples,
        model = edge.config.models.speechToText,
        params = Whisper.TranscribeParams(language = "en"),
        runtime = WhisperRuntimeRequest(gpuEnabled = false, flashAttention = true),
    ),
)
```

This keeps new speech entrypoints aligned with the request-first style already used by text and image generation, while preserving the older parameter-list overloads for compatibility.

### Text Generation Performance Tuning

The text stack now separates prompt/batch processing from single-token generation so you can tune
the two phases independently:

```kotlin
val edge = LLMEdge.create(
    context = context,
    scope = viewModelScope,
    config = LLMEdgeConfig(
        text = TextRuntimeConfig(
            promptThreads = 6,            // prompt/batch phase
            generationThreads = 2,       // token-by-token phase
            batchSize = 8,
            streamBatchSize = 4,
            cache = RuntimeCacheConfig(maxEntries = 2, maxMemoryMb = 1536),
        ),
    ),
)

val reply = edge.text.generate(
    prompt = "Explain speculative decoding.",
    options = TextModelOptions(numThreads = 8, generationThreads = 3),
    batchSize = 12,
)
```

Practical defaults:

- `text.promptThreads`: prompt/batch decode threads
- `text.generationThreads`: single-token generation threads
- `text.batchSize`: blocking text batch size (default `8`)
- `text.streamBatchSize`: streaming batch size (default `4`)
- `text.cache.maxMemoryMb`: upper bound for text-model cache accounting; the cache now refreshes against
  native model/state footprint instead of only the GGUF file size

Batch-size guidance:

- `1`: lowest latency per chunk, highest JNI overhead
- `4`: good default for streaming UI updates
- `8`: good default for blocking text responses
- `12+`: better throughput for longer offline generations, but can delay intermediate updates

### Image Text Extraction (OCR)

llmedge uses Google ML Kit Text Recognition for extracting text from images.

#### Quick Start

```kotlin
val edge = LLMEdge.create(context, viewModelScope)
val text = edge.vision.extractText(bitmap)
println("Extracted text: $text")
```

#### OCR Engines

**Google ML Kit Text Recognition**
- Fast and lightweight
- No additional data files needed
- Good for Latin scripts
- Add dependency: `implementation("com.google.mlkit:text-recognition:16.0.0")`

OCR is exposed directly through `edge.vision.extractText(...)`. The older `VisionMode` convenience
wrapper is gone; callers now choose explicitly between OCR and VLM analysis instead of routing both
through a second abstraction layer.

### Vision Models

Analyze images using Vision Language Models (like LLaVA or Phi-3 Vision) via `edge.vision`.

> [!WARNING]
> The VLM path is experimental. It requires a vision-capable GGUF and a matching mmproj/projector file. When those components are unavailable or incompatible, `edge.vision.analyze(...)` now fails fast with a clear error instead of silently falling back to text-only prompting. OCR remains available through `edge.vision.extractText(...)`.

```kotlin
val edge = LLMEdge.create(context, viewModelScope)

val description = edge.vision.analyze(
    image = bitmap,
    prompt = "Describe this image in detail.",
    numThreads = 4,
    generationThreads = 2,
) { status ->
    Log.d("Vision", "Status: $status")
}
```

The current high-level vision path creates a fresh `SmolLM` runtime per request, so it favors
isolation and predictable cleanup over pooled high-throughput reuse.

The manager handles the complex pipeline of:
1. Preprocessing the image
2. Loading the vision projector and model
3. Encoding the image to embeddings
4. Generating the textual response

Vision model support is currently experimental and requires specific model architectures (like LLaVA-Phi-3).

### Speech-to-Text (Whisper)

Transcribe audio using the new `edge.speech` client:

```kotlin
val edge = LLMEdge.create(context, viewModelScope)

val text = edge.speech.transcribeToText(audioSamples)

val segments = edge.speech.transcribe(
    audioSamples = audioSamples,
    params = Whisper.TranscribeParams(language = "en"),
)
segments.forEach { segment ->
    println("[${segment.startTimeMs}ms] ${segment.text}")
}

val lang = edge.speech.detectLanguage(audioSamples)
```

#### Real-time Streaming Transcription

For live captioning, use the streaming transcription API with a sliding window approach:

```kotlin
val edge = LLMEdge.create(context, viewModelScope)

val session = edge.speech.createStreamingSession(
    params = Whisper.StreamingParams(
        stepMs = 3000,
        lengthMs = 10000,
        keepMs = 200,
        language = "en",
        useVad = true,
    ),
)

viewModelScope.launch {
    session.events().collect { segment ->
        updateCaptions(segment.text)
    }
}

audioRecorder.onAudioChunk { samples ->
    viewModelScope.launch { session.feedAudio(samples) }
}

session.stop()
```

**Streaming parameters:**
- `stepMs`: How often transcription runs (default: 3000ms). Lower = faster updates, higher CPU usage.
- `lengthMs`: Audio window size (default: 10000ms). Longer windows improve accuracy.
- `keepMs`: Overlap with previous window (default: 200ms). Helps maintain context.
- `useVad`: Voice Activity Detection - skips silent audio (default: true).

Direct `Whisper` access remains available for expert workflows, but the namespaced speech client is the standard integration path.

**Recommended models:**
- `ggml-tiny.bin` (~75MB) - Fast, lower accuracy
- `ggml-base.bin` (~142MB) - Good balance
- `ggml-small.bin` (~466MB) - Higher accuracy

### Text-to-Speech (Bark)

Generate speech using `edge.speech`:

```kotlin
val edge = LLMEdge.create(context, viewModelScope)

val audio = edge.speech.synthesize("Hello, world!")

viewModelScope.launch {
    edge.speech.synthesizeStream("Hello, world!").collect { event ->
        when (event) {
            is AudioStreamEvent.Progress -> Log.d("Bark", "${event.step.name}: ${event.percent}%")
            is AudioStreamEvent.Result -> saveAudio(event.audio)
            else -> Unit
        }
    }
}
```

Direct `BarkTTS` access remains available for expert workflows, but the namespaced speech client is the standard integration path.

### Stable Diffusion (image generation)

Generate images on-device using the namespaced `edge.image` client:

```kotlin
val edge = LLMEdge.create(context, viewModelScope)

val bitmap = edge.image.generate(
    ImageGenerationRequest(
        prompt = "a cute pastel anime cat, soft colors, high quality <lora:detail_tweaker:1.0>",
        width = 512,
        height = 512,
        steps = 20,
        loraModelDir = "/path/to/loras",
        loraApplyMode = StableDiffusion.LoraApplyMode.AUTO,
    ),
)
imageView.setImageBitmap(bitmap)
```

**Key Optimizations:**
- **EasyCache**: `edge.image` automatically enables EasyCache for supported Diffusion Transformer (DiT) models such as Flux, SD3, Wan, Qwen Image, and Z-Image; it stays disabled for classic UNet pipelines.
- **Flash Attention**: Automatically enabled for compatible image dimensions.
- **LoRA**: Apply fine-tuned weights on the fly without merging models.

For explicit runtime ownership or custom native-load experiments, the `StableDiffusion` class remains available in the expert API layer.

#### Streaming progress

`generate` blocks until the bitmap is ready. To drive a progress bar, use `generateStream`, which emits one `Progress` event per denoising step and a final `Completed` event carrying the image:

```kotlin
edge.image.generateStream(request).collect { event ->
    when (event) {
        is GenerationStreamEvent.Progress ->
            progressBar.progress = event.update.current * 100 / event.update.total
        is GenerationStreamEvent.Completed ->
            imageView.setImageBitmap(event.frames.first())
    }
}
```

Cancelling the collection cancels the generation. Step events only cover the sampling loop; model loading happens before the first event, so keep the bar indeterminate until then. The example app derives a remaining-time estimate from the delay between events (`StepEtaEstimator` in llmedge-examples).

#### Image upscaling (ESRGAN)

`edge.image.upscale` runs an ESRGAN model over a bitmap and returns the enlarged result. Old-architecture ESRGAN checkpoints load directly; 4x_foolhardy_Remacri is the tested reference:

```kotlin
val esrgan = edge.models.resolve(
    ModelSpec.huggingFace(
        repoId = "LyliaEngine/4x_foolhardy_Remacri",
        filename = "4x_foolhardy_Remacri.safetensors",
        hints = ModelHints(
            artifactKind = ModelArtifactKind.REPO_FILE,
            capabilities = setOf(ModelCapability.IMAGE),
        ),
    ),
)

val upscaled = edge.image.upscale(
    UpscaleRequest(input = bitmap, model = ModelSpec.localFile(esrgan)),
) { tile, totalTiles ->
    // fires once per processed tile
}
```

Details worth knowing:

- `factor = 0` (the default) uses the scale baked into the model. Remacri is 4x.
- The upscaler runs on CPU unless the request sets `useVulkan = true`. Large inputs are processed in 128px tiles, and the progress callback reports tile counts, not steps.
- Input is capped at 1024×1024. A 4x pass over a 1024px image already produces a 16 MP bitmap, which is near the practical memory limit on mid-range phones.
- In isolated worker mode the upscale runs in the `:llmedge_sd` process with the same watchdog and crash recovery as image generation.
- The ESRGAN context is created and freed per call, so expect a short model-load pause (the weights are ~64 MB) before the first tile.

#### MiniT2I

The `MiniT2I` helper downloads the standalone MiniT2I B/16 diffusion transformer and its FLAN-T5 Large text encoder, then routes them to stable-diffusion.cpp's split diffusion and T5 model slots:

```kotlin
val edge = LLMEdge.create(context, viewModelScope)
val bitmap = edge.image.generate(
    MiniT2I.imageRequest("a small robot watering flowers"),
)
```

The helper defaults to the model's 512×512, 100-step, CFG 6 configuration. Width, height, steps, CFG, seed, and flash attention remain configurable through `MiniT2I.imageRequest(...)`.

#### FLUX.2 Klein 4B (distilled DiT, split model)

[FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) is a step-distilled diffusion transformer that produces high-quality images in ~4 steps. It is the same architecture PrismML's binary/ternary **Bonsai Image** models are built on — Bonsai's own 1-bit/ternary weights ship only in MLX (Apple) and GemLite (CUDA) packings, which don't load on Android, so this GGUF build is the Android-runnable equivalent at a comparable footprint.

Unlike a classic single-file checkpoint, FLUX.2 loads as three components: the diffusion transformer (GGUF), a Qwen3-4B text encoder, and the FLUX.2 VAE. The `Flux2Klein` helper wires all three plus the distilled defaults (CFG 1.0, 4 steps):

```kotlin
val edge = LLMEdge.create(context, viewModelScope)
val bitmap = edge.image.generate(
    Flux2Klein.imageRequest("a red fox in snow, detailed, 8k"),
)
```

Internally this sets `ImageGenerationRequest.splitDiffusionModel = true`, which routes the transformer to stable-diffusion.cpp's `diffusion_model_path` and the Qwen3 encoder to `llm_path` (instead of the single `model_path` slot), and offloads weights to CPU. Footprint: ~2.5 GB DiT (Q4_0) + ~2.1 GB encoder (Q3_K_M) + ~0.3 GB VAE, so it targets higher-RAM devices.

##### Low-end: Bonsai (QAT) DiT for a smaller transformer

PrismML's **Bonsai Image** models are FLUX.2 Klein 4B fine-tuned with quantization-aware training (QAT) to ternary weights. They ship only in MLX/GemLite packings (not Android-loadable) and in a dense-bf16 `-unpacked` form using the non-standard `Flux2KleinPipeline` diffusers naming, which stable-diffusion.cpp cannot ingest directly.

`scripts/convert_bonsai_flux2_to_bfl.py` converts a Bonsai unpacked transformer into the BFL naming sdcpp expects (renames + fuses the double-block `to_q/k/v` into `*_attn.qkv`). Then quantize with stable-diffusion.cpp:

```bash
python3 scripts/convert_bonsai_flux2_to_bfl.py \
    bonsai-image-ternary-4B-unpacked/transformer/diffusion_pytorch_model.safetensors \
    bonsai-flux2-bfl.safetensors
sd -M convert -m bonsai-flux2-bfl.safetensors --type q2_K -o bonsai-flux2-klein-q2_K.gguf
```

A prebuilt Q2_K of this is published at [`Aatricks/bonsai-image-ternary-4B-FLUX2-klein-GGUF`](https://huggingface.co/Aatricks/bonsai-image-ternary-4B-FLUX2-klein-GGUF) and wired into `Flux2Klein.bonsaiDiffusionModel`. The QAT weights survive `Q2_K` well, giving a **coherent ~1.3 GB DiT** (vs ~2.5 GB for base Q4_0). Note: ggml's literal ternary types (`tq1_0`/`tq2_0`, ~0.8–1.0 GB) load and run on CPU but their per-256-weight scale is too coarse for Bonsai's per-128 trained scales and produce degraded output — `Q2_K`'s finer per-16 sub-block scales are what preserve quality.

#### Sequential loading for ~4 GB-RAM devices

The text encoder (~2 GB) is the dominant memory cost. Sequential mode loads only the Qwen3 encoder to precompute the text conditioning, frees it, then loads only the DiT to generate — so peak RAM is `max(encoder, DiT)` (~2.6 GB) instead of the sum (~4 GB):

```kotlin
val bmp = edge.image.generate(Flux2Klein.bonsaiImageRequest("a red fox in snow, 8k"))
// or for the base FLUX.2 Klein DiT: Flux2Klein.imageRequest(prompt, sequential = true)
```

`ImageGenerationRequest.sequential` drives this; the runtime runs the two phases automatically (encoder-only → precompute → free → DiT-only → generate via a precomputed condition), backed by stable-diffusion.cpp's `sd_precompute_condition` / `sd_generate_image_with_precomputed_condition`.

### Video Generation

Generate short video clips using `edge.image.generateVideo(...)`. The namespaced client surfaces progress as a `Flow` while reusing the existing Wan loading logic internally.

**Hardware Requirements**:
- **12GB+ RAM** recommended for standard loading.
- **8GB+ RAM** supported via `forceSequentialLoad = true` (slower but memory-safe).

```kotlin
val edge = LLMEdge.create(context, viewModelScope)

val params = VideoGenerationRequest(
    prompt = "a cat walking in a garden, high quality",
    videoFrames = 8,
    width = 512,
    height = 512,
    steps = 20,
    cfgScale = 7.0f,
    flowShift = 3.0f,
    forceSequentialLoad = true,
)

viewModelScope.launch {
    edge.image.generateVideo(params).collect { event ->
        when (event) {
            is GenerationStreamEvent.Progress -> Log.d("VideoGen", event.update.message)
            is GenerationStreamEvent.Completed -> previewImageView.setImageBitmap(event.frames.first())
        }
    }
}
```

`edge.image` automatically:
1. Downloads the necessary Wan 2.1 model files (Diffusion, VAE, T5).
2. Sequentially loads components to minimize peak memory usage (if requested).
3. Manages the generation loop and frame conversion.

See `llmedge-examples` for a complete UI implementation.


Running the example app:
1. Build the library (from the repo root):

```bash
./gradlew :llmedge:assembleRelease
```

2. Build and install the example app:

```bash
cd llmedge-examples
../gradlew :app:assembleDebug
../gradlew :app:installDebug
```

3. Open the app on device and pick the "Stable Diffusion" demo from the launcher. The demo downloads any missing files from Hugging Face and runs a quick txt2img generation.

Notes:
- The example explicitly downloads a VAE safetensors file for the `Meina/MeinaMix` demo; many repos include VAE files, but some GGUF model repos bundle everything you need. If the repo lacks a GGUF model file you'll get an obvious IllegalArgumentException — provide a `filename` or choose a different repo in that case.
- Use the system downloader for large safetensors/gguf files to avoid heap pressure on Android.
### On-device RAG

The library includes a minimal on-device RAG pipeline, similar to Android-Doc-QA, built with:
- Sentence embeddings (ONNX)
- Whitespace `TextSplitter`
- In-memory cosine `VectorStore` with JSON persistence
- `SmolLM` for context-aware responses through the facade-managed RAG session

### Setup

1. Download embeddings

   From the Hugging Face repository `sentence-transformers/all-MiniLM-L6-v2`, place:

```
llmedge/src/main/assets/embeddings/all-minilm-l6-v2/model.onnx
llmedge/src/main/assets/embeddings/all-minilm-l6-v2/tokenizer.json
```

2. Build the library

```
./gradlew :llmedge:assembleRelease
```

3. Use in your application

```kotlin
val edge = LLMEdge.create(this, lifecycleScope)
val rag = edge.rag.createSession()

lifecycleScope.launch {
    rag.init()
    val count = rag.indexPdf(pdfUri)
    val answer = rag.ask("What are the key points?")
    // render answer
}
```

Direct `RAGEngine` construction remains available for expert workflows, but new app code should prefer `edge.rag.createSession()` so runtime ownership and teardown stay aligned with the rest of the library.

### Expert APIs

`SmolLM`, `StableDiffusion`, `Whisper`, `BarkTTS`, `RAGEngine`, and direct `HuggingFaceHub` access are still available when you need to hold a native runtime directly or override low-level loading behavior. They are intentionally secondary to the facade APIs.

Examples:

```kotlin
// Direct model download when you need full control over artifact selection.
val download = HuggingFaceHub.ensureModelOnDisk(
    context = context,
    modelId = "unsloth/Qwen3-0.6B-GGUF",
    filename = "Qwen3-0.6B-Q4_K_M.gguf",
)

// Expert text runtime with live reasoning-state control.
val smol = SmolLM()
smol.load(download.file.absolutePath)
smol.setThinkingEnabled(false)

// Expert RAG wiring when you want to own both the runtime and the pipeline yourself.
val ragEngine = RAGEngine(context = context, smolLM = smol)
```

## Building

Building GPU backends on Android
--------------------------------

If you want GPU acceleration for the native inference backends, follow these notes and requirements. On Android, llmedge now prefers `OPENCL -> VULKAN -> CPU` when GPU use is allowed for text, Whisper, and image/video requests. OpenCL support is experimental, Android-only, and currently limited to `arm64-v8a`. Bark remains CPU-only.

Prerequisites
- Android NDK r27 or newer (NDK r27 used in development; the NDK provides the Vulkan C headers). Ensure your NDK matches the version used by your build environment.
- CMake 3.22+ and Ninja (the Android Gradle plugin will pick up CMake when configured).
- Gradle (use the wrapper: `./gradlew`).
- Android API (minSdk) 30 or higher. `llmedge` targets Android 11+ today, and Vulkan support still requires Vulkan 1.2.
- (Optional) `VULKAN_SDK` set in the environment if you build shaders or use Vulkan SDK tools on the host. The build fetches a matching `vulkan.hpp` header when needed.

### Host Setup for Vulkan Builds (Ubuntu/WSL)

To build the library with Vulkan support on a Linux host or WSL2, you must install the Vulkan shader compiler and development headers:

1. **Install Dependencies**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y glslc libvulkan-dev
   ```

2. **Verify glslc**:
   Ensure `glslc` is in your PATH:
   ```bash
   glslc --version
   ```

3. **Android NDK**:
   Ensure you have Android NDK **r27** (specifically `27.2.12479018`) installed via Android Studio or the SDK manager.

Build flags
- On Linux/macOS hosts, the Gradle build enables Vulkan by default. On Windows hosts, it defaults to `OFF` because the upstream shader-generator step is still fragile under the Android cross-build toolchain. Re-enable it explicitly only when your environment supports that path.
- Experimental Android OpenCL is disabled by default. Enable it with `-PllmedgeAndroidOpencl=ON` or the environment variable `LLMEDGE_ANDROID_OPENCL=ON`.
- If you want both OpenCL and Vulkan compiled in explicitly, use:

```bash
./gradlew :llmedge:assembleRelease \
  -PllmedgeAndroidOpencl=ON \
  -Pandroid.injected.build.api=30 \
  -Pandroid.jniCmakeArgs="-DSD_VULKAN=ON -DGGML_VULKAN=ON"
```

Alternatively, set the same flags in your Android Studio CMake configuration. `LLMEDGE_ANDROID_OPENCL` is the library's experimental OpenCL toggle, while `-DSD_VULKAN=ON` and `-DGGML_VULKAN=ON` force Vulkan support for Stable Diffusion and ggml.

Notes about headers and toolchain
- The build fetches `Vulkan-Hpp` (`vulkan.hpp`) and pins it to the NDK's Vulkan headers to avoid API mismatch. If you have a local `VULKAN_SDK` you can point to it, otherwise the project will use the fetched headers.
- When OpenCL is enabled, the build uses repo-managed OpenCL headers and a link-time loader shim. The packaged app still resolves the device's OpenCL implementation at runtime rather than shipping its own platform ICD.
- The repository also builds a small host toolchain to generate SPIR-V shaders at build time; ensure your build host has a working C++ toolchain (clang/gcc) and CMake configured.

Runtime verification
- To verify GPU capability at runtime:
    - Run the app on an Android 11+ device.
    - Use the per-subsystem capability APIs to inspect the engines you care about, for example `LLMEdge.getTextBackendAvailability()`, `LLMEdge.getSpeechBackendAvailability()`, `LLMEdge.getImageBackendAvailability()`, and `LLMEdge.getVisionBackendAvailability()`.
    - Inspect runtime logs for the selected backend and any fallback reason. Example:

```bash
adb logcat -s SmolSD:* | sed -n '1,200p'
```

    Look for messages indicating OpenCL or Vulkan initialization. `LLMEdgeConfig(text = TextRuntimeConfig(useVulkan = true))` means "allow a supported GPU backend", not "force Vulkan".

Troubleshooting
- If you see "Vulkan 1.2 required" or linker errors for Vulkan symbols, confirm `minSdk` is set to 30 or higher in `llmedge/build.gradle.kts` and that your NDK provides the expected Vulkan headers.
- If experimental OpenCL is not available, or if a GPU backend fails to initialize or execute, llmedge falls back to Vulkan or CPU automatically. For text, Whisper, and image/video, a failing backend is blacklisted per subsystem for the rest of the process and the next backend is retried once.
- If your device lacks both usable OpenCL and Vulkan support, the native code falls back to the CPU backend.
- If the Vulkan driver initializes but the first generate hangs forever at the first compute dispatch (observed on PowerVR DXT-48 / Pixel 10 Tensor G5), the automatic fallback cannot trigger — load succeeds, so no failure is observed. Force the CPU backend for image/video generation with `LLMEdgeConfig(image = ImageRuntimeConfig(useVulkan = false))`, or enable process isolation (below) so the library detects and recovers from the hang automatically.

Process isolation for image/video generation (opt-in)

`LLMEdgeConfig(image = ImageRuntimeConfig(workerMode = DiffusionWorkerMode.ISOLATED_PROCESS))` runs the diffusion stack in a library-owned `:llmedge_sd` worker process:

- A native crash in the diffusion stack surfaces as a typed `WorkerCrashedException` (with one automatic CPU retry) instead of killing the app.
- A GPU driver that hangs at dispatch is detected by a watchdog — no progress heartbeat while the worker's CPU time stays flat (a legitimate cold shader compile pegs a core and is never killed) — the worker is killed, and per `hangRecoveryPolicy` the request is transparently retried on CPU (default) or failed with `GenerationHangException`.
- Hang verdicts persist across sessions (keyed to the OS build fingerprint, so a system/driver update re-enables the GPU automatically). `ImageClient.resetBackendVerdicts(context)` clears them manually.
- The public `ImageClient` API is unchanged; `DiffusionWorkerMode.IN_PROCESS` remains the default for now. Apps that already host llmedge in their own service process should stay in-process to avoid a redundant extra process.

#### Notes:

- Uses `com.tom-roush:pdfbox-android` for PDF parsing.
- Embeddings library: `io.gitlab.shubham0204:sentence-embeddings:v6`.
- Scanned PDFs require OCR (e.g., ML Kit or Tesseract) before indexing.
- ONNX `token_type_ids` errors are automatically handled; override via `EmbeddingConfig` if required.

## Architecture

The Kotlin side is now organized around a few explicit layers instead of one eager facade:

1. `LLMEdge` is a thin convenience shell that lazy-creates domain clients (`text`, `speech`, `image`, `vision`, `rag`) on first access.
2. `ModelRepository` owns model acquisition and validation for local files and Hugging Face downloads.
3. `RuntimePool` and `RuntimeCoordinator` provide shared runtime caching, backend selection, and failure blacklisting.
4. `RuntimePoolProfile` lets each domain describe cache sizing, keying, loading, and backend policy without duplicating pool boilerplate.
5. `TextClient`, `SpeechClient`, `ImageClient`, `VisionClient`, and `RAGClient` remain independently constructible for advanced use, but `LLMEdge` is the canonical public entrypoint.
6. `ConversationSessionSupport` centralizes transcript state and runtime access for chat sessions and tool agents.
7. `VisionInputPreparer` and `VisionRuntimeExecutor` split image preprocessing/embedding from generation execution.
8. `RAGIndexer`, `RAGRetriever`, and `RAGAnswerer` separate document ingestion, retrieval, and answer generation.
9. Native libraries remain in the same Android module, but native loading is now explicit and overridable for JVM tests instead of relying on static side effects.

On the native side, the project still builds llama.cpp, stable-diffusion.cpp, whisper.cpp, bark.cpp, and the JNI bridge sources through the Android NDK.

## Technologies

- [llama.cpp](https://github.com/ggml-org/llama.cpp) — Core LLM backend
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) — Image/video generation backend
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) — Speech-to-text backend
- [bark.cpp](https://github.com/PABannier/bark.cpp) — Text-to-speech backend
- GGUF / GGML — Model formats
- Android NDK / JNI — Native bindings
- ONNX Runtime — Sentence embeddings
- Android DownloadManager — Large file downloads

## Memory Metrics

You can measure RAM usage at runtime:

```kotlin
val snapshot = MemoryMetrics.snapshot(context)
Log.d("Memory", snapshot.toPretty(context))
```

Typical measurement points:

- Before model load
- After model load
- After blocking prompt
- After streaming prompt

#### Key fields:

- `totalPssKb`: Total proportional RAM usage. Best for overall tracking.
- `dalvikPssKb`: JVM-managed heap and runtime.
- `nativePssKb`: Native heap (llama.cpp, ONNX, tensors, KV cache).
- `otherPssKb`: Miscellaneous memory.

Monitor `nativePssKb` closely during model loading and inference to understand LLM memory footprint.
Expert runtimes such as `SmolLM` also expose native/state-specific memory estimates when you need lower-level instrumentation.

## Notes

- `VULKAN_SDK` may still be required when you are building the Vulkan path on the host.
- Check Android GPU capability with the explicit per-subsystem helpers such as `LLMEdge.getTextBackendAvailability()` and `LLMEdge.getImageBackendAvailability()`.

### ProGuard/R8 Configuration

The library includes consumer ProGuard rules. If you need to add custom rules:

```proguard
# Keep OCR engines
-keep class io.aatricks.llmedge.vision.** { *; }
-keep class org.bytedeco.** { *; }
-keep class com.google.mlkit.** { *; }

# Suppress warnings for optional dependencies
-dontwarn org.bytedeco.**
-dontwarn com.google.mlkit.**
```

### Licenses

- **llmedge**: Apache 2.0
- **llama.cpp**: MIT
- **stable-diffusion.cpp**: MIT
- **whisper.cpp**: MIT
- **bark.cpp**: MIT
- **Leptonica**: Custom (BSD-like)
- **Google ML Kit**: Proprietary (see ML Kit terms)
- **JavaCPP**: Apache 2.0

## License and Credits

This project builds upon work by [Shubham Panchal](https://github.com/shubham0204), [ggerganov](https://github.com/ggerganov), and [PABannier](https://github.com/PABannier).
See [CREDITS.md](CREDITS.md) for full details.

## Testing

Looking to run unit and instrumentation tests locally, including optional native txt2img E2E checks? See the step-by-step guide in [docs/testing.md](docs/testing.md).
