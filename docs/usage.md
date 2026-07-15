# Usage & API

This page explains how to use `llmedge`'s Kotlin API. The library offers two layers of abstraction:

1.  **High-Level API (`LLMEdge`)**: Recommended for most use cases. It exposes instance-scoped clients for text, speech, image, vision, and RAG while keeping model resolution and cleanup explicit.
2.  **Low-Level API (`SmolLM`, `StableDiffusion`)**: For advanced users who need fine-grained control over model lifecycle and parameters.

For application code, prefer:

- `LLMEdge.create(...)` for lifecycle-managed access
- `edge.models.prefetch(...)` / `edge.models.resolve(...)` for downloads
- `edge.text`, `edge.speech`, `edge.image`, `edge.vision`, and `edge.rag` for inference

Direct `HuggingFaceHub` calls and `*.loadFromHuggingFace(...)` helpers are still supported, but they are expert APIs.

Examples reference the `llmedge-examples` [repo](https://github.com/aatricks/llmedge-examples).

---

## GPU Backends on Android

`TextModelOptions.useVulkan`, `LLMEdgeConfig.textUseVulkan`, and `SmolLM(useVulkan = true)` keep their historical names for source compatibility. On Android, `true` now means "allow GPU acceleration": llmedge prefers OpenCL first, then Vulkan, then CPU. `WhisperLoadOptions.useGpu` follows the same rule. Bark remains CPU-only.

## High-Level API (LLMEdge)

Create an `LLMEdge` instance from an Android-aware coroutine scope, then use the domain clients exposed by the facade.

### Text Generation

```kotlin
val edge = LLMEdge.create(context, viewModelScope)

val response = edge.text.generate(
    prompt = "Write a haiku about Kotlin.",
    model = ModelSpec.huggingFace(
        repoId = "HuggingFaceTB/SmolLM-135M-Instruct-GGUF",
        filename = "smollm-135m-instruct.q4_k_m.gguf",
    ),
)
```

The high-level text client defaults to **batched blocking generation** to reduce JNI overhead.
Override it when needed:

```kotlin
val response = edge.text.generate(
    prompt = "Summarize the latest release notes.",
    maxTokens = 256,
    batchSize = 12,
    options = TextModelOptions(
        numThreads = 8,          // prompt/batch processing
        generationThreads = 3,   // single-token generation
    ),
)
```

Streaming uses smaller batched native chunks by default. This keeps UI updates responsive without
crossing JNI once per token:

```kotlin
edge.text.stream(
    prompt = "List the key takeaways.",
    batchSize = 6,
    options = TextModelOptions(
        numThreads = 6,
        generationThreads = 2,
    ),
).collect { event ->
    if (event is TextStreamEvent.Chunk) {
        appendToUi(event.value)
    }
}
```

Default batch sizes are currently `8` for blocking generation and `4` for streaming. Passing
`batchSize = 0` uses the configured default for the relevant path.

### Batch Size Tuning

| Workload | Suggested batch size | Why |
|---|---:|---|
| Token-by-token UI streaming | `1-4` | keeps updates frequent and reduces perceived latency |
| General chat replies | `4-8` | good balance between JNI overhead and responsiveness |
| Longer offline generation | `8-16` | better throughput when intermediate updates matter less |

If you are tuning on big.LITTLE devices, adjust `batchSize` together with `numThreads` and
`generationThreads` rather than treating them in isolation.

### Tool Calling

Use `edge.text.toolAgent(...)` when the model should call app-defined tools instead of only returning text.
Read-only tools run automatically. Action tools, including the bash tool below, still require an
explicit policy approval.

```kotlin
val edge = LLMEdge.create(context, viewModelScope)
val agent =
    edge.text.toolAgent(
        tools =
            listOf(
                BashToolFactory(
                    BashToolOptions(
                        allowRawShell = true,
                        defaultWorkingDirectory = context.filesDir.absolutePath,
                    ),
                ).createBashTool(),
            ),
        systemPrompt = "Use shell commands only when they materially help answer the user.",
        policy = ToolPolicies.ALLOW_ALL,
    )
```

`run_bash_command` supports two argument shapes:

- `argv`: a structured array such as `["echo", "hello"]`
- `command`: a raw shell string such as `"pwd"`; this stays disabled unless `allowRawShell = true`

The tool captures `stdout`, `stderr`, exit code, timeout state, truncation state, and the executed
command details in the returned `ToolResult.data`. If `bash` is unavailable at runtime or process
startup fails, the tool reports that as a structured tool error.

### Image Generation

Handles model resolution and memory-safe loading through the `edge.image` client.

```kotlin
val edge = LLMEdge.create(context, viewModelScope)

val bitmap = edge.image.generate(
    ImageGenerationRequest(
        prompt = "A cyberpunk city street at night, neon lights <lora:detail_tweaker_lora_sd15:1.0>",
        width = 512,
        height = 512,
        steps = 20,
        loraModelDir = getExternalFilesDir("loras")?.absolutePath + "/detail-tweaker-lora-sd15",
        loraApplyMode = StableDiffusion.LoraApplyMode.AUTO
    ),
)
```

**Key Optimizations for Image Generation:**

- **EasyCache**: Automatically enabled by `edge.image` for supported Diffusion Transformer (DiT) models such as Flux, SD3, Wan, Qwen Image, and Z-Image. It remains disabled for classic UNet-based pipelines such as SD 1.5/SDXL.
- **LoRA Support**: `ImageGenerationRequest` accepts `loraModelDir` and `loraApplyMode` for on-the-fly fine-tuning.
- **Flash Attention**: Automatically enabled for compatible image dimensions.

#### Streaming progress

`generate` blocks until the bitmap is ready. `generateStream` emits a `Progress` event per denoising step and finishes with `Completed`:

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

Cancelling the collection cancels the generation. The step events cover the sampling loop only; model loading happens before the first event arrives. The example app turns the inter-event timing into a remaining-time label (`StepEtaEstimator`).

#### Image upscaling (ESRGAN)

`edge.image.upscale` enlarges a bitmap with an ESRGAN model. 4x_foolhardy_Remacri is the tested reference; other old-architecture ESRGAN checkpoints load the same way.

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

`factor = 0` uses the model's native scale (4x for Remacri). The upscaler defaults to CPU; set `useVulkan = true` to opt into the GPU path. Inputs above 1024×1024 are rejected because the 4x output would not fit comfortably in memory on most devices. In isolated worker mode the call runs inside the `:llmedge_sd` process, so a native crash is contained and retried on CPU like any other diffusion request. The ESRGAN context is created and freed per call.

The example app has two entry points: the image generation screen upscales its last result, and the standalone upscaler screen takes any picture from the device.

### Video Generation (Wan 2.1)

Handles the complex multi-model loading (Diffusion, VAE, T5) and sequential processing required for video generation on mobile.

```kotlin
val edge = LLMEdge.create(context, viewModelScope)

val request =
    VideoGenerationRequest(
        prompt = "A robot dancing in the rain",
        videoFrames = 16,
        width = 512,
        height = 512,
        steps = 20,
        cfgScale = 7.0f,
        flowShift = 3.0f,
        forceSequentialLoad = true,
    )

viewModelScope.launch {
    edge.image.generateVideo(request).collect { event ->
        when (event) {
            is GenerationStreamEvent.Progress -> Log.d("Video", event.update.message)
            is GenerationStreamEvent.Completed -> previewImageView.setImageBitmap(event.frames.first())
        }
    }
}
```

### Vision Analysis

Analyze images using a Vision Language Model (VLM).

```kotlin
val edge = LLMEdge.create(context, viewModelScope)
val description =
    edge.vision.analyze(
        VisionRequest(
            image = bitmap,
            prompt = "What is in this image?",
            model = edge.config.models.vision.model,
            projector = edge.config.models.vision.projector,
        ),
    )
```

Vision analysis also exposes separate prompt and generation thread counts for the underlying
SmolLM runtime:

```kotlin
val description =
    edge.vision.analyze(
        VisionRequest(
            image = bitmap,
            prompt = "What is in this image?",
            model = edge.config.models.vision.model,
            projector = edge.config.models.vision.projector,
            numThreads = 4,
            generationThreads = 2,
        ),
    )
```

```kotlin
edge.vision.prepare(
    VisionPrepareRequest(
        model = edge.config.models.vision.model,
        projector = edge.config.models.vision.projector,
        promptThreads = 4,
        generationThreads = 2,
    ),
)
```

The current high-level vision pipeline prioritizes isolation and predictable cleanup over manual runtime ownership.

### OCR (Text Extraction)

Extract text using ML Kit.

```kotlin
val edge = LLMEdge.create(context, viewModelScope)
val text = edge.vision.extractText(bitmap)
```

### Speech-to-Text (Whisper)

Transcribe audio using the high-level API:

```kotlin
val edge = LLMEdge.create(context, viewModelScope)

val text =
    edge.speech.transcribeToText(
        SpeechToTextRequest(
            audioSamples = audioSamples,
            model = edge.config.models.speechToText,
        ),
    )

val segments =
    edge.speech.transcribe(
        SpeechToTextRequest(
            audioSamples = audioSamples,
            model = edge.config.models.speechToText,
            params = Whisper.TranscribeParams(language = "en", translate = false),
            runtime = WhisperRuntimeRequest(gpuEnabled = false),
        ),
    )

val lang =
    edge.speech.detectLanguage(
        SpeechLanguageDetectionRequest(
            audioSamples = audioSamples,
            model = edge.config.models.speechToText,
        ),
    )
```

New code should prefer the request objects so speech usage matches the request-first shape used elsewhere in the facade. The older parameter-list overloads remain supported.

### Streaming Transcription (Real-time Captioning)

For live transcription from a microphone or audio stream, use the streaming API:

```kotlin
import kotlinx.coroutines.launch

val edge = LLMEdge.create(context, lifecycleScope)
val transcriber = edge.speech.createStreamingSession(
    StreamingTranscriptionRequest(
        model = edge.config.models.speechToText,
        params = Whisper.StreamingParams(
            stepMs = 3000,      // Run transcription every 3 seconds
            lengthMs = 10000,   // Use 10-second audio windows
            keepMs = 200,       // Keep 200ms overlap for context
            language = "en",    // null for auto-detect
            useVad = true       // Skip silent audio
        ),
    )
)

// Collect real-time transcription results
launch {
    transcriber.events().collect { segment ->
        runOnUiThread {
            textView.append("${segment.text}\n")
        }
    }
}

// Feed audio samples from microphone (16kHz mono PCM float32)
audioRecorder.setOnAudioDataListener { samples ->
    lifecycleScope.launch {
        transcriber.feedAudio(samples)
    }
}

// Stop when done
fun stopTranscription() {
    transcriber.stop()
}
```

**Streaming Parameters Explained:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stepMs` | 3000 | How often transcription runs (lower = faster updates) |
| `lengthMs` | 10000 | Audio window size (longer = more accurate) |
| `keepMs` | 200 | Overlap with previous window for context |
| `vadThreshold` | 0.6 | Voice activity threshold (0.0-1.0) |
| `useVad` | true | Skip transcription during silence |

**Preset Configurations:**

- **Fast captioning:** `stepMs=1000, lengthMs=5000` - Quick updates, lower accuracy
- **Balanced (default):** `stepMs=3000, lengthMs=10000` - Good tradeoff
- **High accuracy:** `stepMs=5000, lengthMs=15000` - Better accuracy, more delay

### Text-to-Speech (Bark)

Generate speech using the high-level API:

```kotlin
val edge = LLMEdge.create(context, viewModelScope)

val audio =
    edge.speech.synthesize(
        SpeechSynthesisRequest(
            text = "Hello, world!",
            model = edge.config.models.textToSpeech,
        ),
    )
audioPlayer.play(audio.samples, audio.sampleRate)
```

---

## Low-Level API

Direct usage of `SmolLM` and `StableDiffusion` classes. Use this if you need to manage the model lifecycle manually (e.g., keeping a model loaded across multiple disparate activities) or require configuration not exposed by `LLMEdge`.

### Core components

- `SmolLM` — Kotlin front-end class that wraps native inference calls.
- `GGUFReader` — C++/JNI reader for GGUF model files.
- `Whisper` — Speech-to-text via whisper.cpp (JNI bindings).
- `BarkTTS` — Text-to-speech via bark.cpp (JNI bindings).
- Vision helpers — `ImageUnderstanding`, `OcrEngine` (with `MlKitOcrEngine` implementation).
- RAG helpers — `RAGEngine`, `VectorStore`, `PDFReader`, `EmbeddingProvider`.

### Basic LLM Inference

Load a GGUF model and run inference:

```kotlin
val smol = SmolLM()
smol.load(modelPath, InferenceParams(numThreads = 4, contextSize = 4096L))
val reply = smol.getResponse("Your prompt here")
smol.close()  // Free native memory when done
```

### Managed chat history with `edge.text.session(...)`

Use a Kotlin-managed chat session when you want bounded multi-turn history without relying on the
native KV cache to retain earlier turns:

```kotlin
runBlocking {
    val edge = LLMEdge.create(context, this)

    val session =
        edge.text.session(
            model = ModelSpec.localFile(modelPath),
            memory = ConversationWindow(maxTurns = 6, maxTokens = 4096, stripThinkTags = true),
            systemPrompt = "You are a concise assistant.",
        )

    session.prepare()
    val firstReply = session.reply("Explain KV cache in one paragraph.")
    session.stream("Now summarize that in 3 bullets.").collect { event ->
        if (event is TextStreamEvent.Chunk) {
            print(event.value)
        }
    }

    edge.close()
}
```

`edge.text.session(...)` keeps the transcript in Kotlin memory, replays only the active sliding
window, and strips older `<think>...</think>` traces before replaying assistant messages.

Use it when:

- reasoning-enabled models emit large `<think>...</think>` blocks that would otherwise bloat native chat history
- you need a bounded sliding window (`ConversationWindow`) for long-running chats
- you want streaming via `stream()` while still persisting the completed assistant reply in Kotlin memory

Prefer plain `SmolLM` with `storeChats = true` only for tightly scoped native-KV-cache flows where
you explicitly want the model runtime to own all chat history.

See [Examples](examples.md#chatsession-pattern) for a focused session snippet, or [LocalAssetDemoActivity](examples.md#localassetdemoactivity) for a complete app-level example.

### Built-in low-end model presets

`ModelPresets` provides ready-to-use specs for models that run well on low-end devices and are supported
by the bundled ik_llama.cpp runtime:

```kotlin
// Microsoft BitNet b1.58 2B4T — native 1-bit LLM (IQ2_BN, ~988 MB).
// The canonical chat template ships on the preset (BitNet's GGUF metadata one is wrong),
// so this is well-formed without setting TextModelOptions.chatTemplate.
val reply = edge.text.generate(prompt = "Hi", model = ModelPresets.bitnet)

// SmolVLM2-256M — tiny vision model (~280 MB total: base + projector).
val caption = edge.vision.analyze(
    image = bitmap,
    prompt = "Describe this image.",
    model = ModelPresets.smolVlm2.model,
    projector = ModelPresets.smolVlm2.projector,
)
```

Presets are plain `ModelSpec`s, so they compose with everything else (`edge.models.prefetch(...)`,
`ModelRegistry`, per-call `model =` overrides). A template passed via `TextModelOptions.chatTemplate`
always overrides a preset's `ModelHints.chatTemplate`.

### Converting safetensors models

The runtime loads GGUF, not safetensors. `ModelSpec.safetensors(...)` declares a safetensors source plus
a target precision; resolution returns a converted GGUF from the app cache, and if none exists yet it
**converts on-device**: it downloads the model directory (config + safetensors + tokenizer files), runs
the native converter, optionally quantizes, caches the result, then loads it. `safetensorsLocal(path,…)`
does the same from a local model directory (no download).

```kotlin
// "direct" = F16 (no precision loss); or Q8_0 / Q4_K_M / IQ2_BN for smaller, lossy output.
val smol = ModelSpec.safetensors(
    repoId = "HuggingFaceTB/SmolLM-135M-Instruct",
    precision = ConversionPrecision.Q4_K_M,
    tokenizerPre = "smollm", // REQUIRED: the tokenizer.ggml.pre id to bake (see below)
)
val reply = edge.text.generate(prompt = "Hi", model = smol)
```

`tokenizerPre` is mandatory for on-device conversion of a **GPT2-BPE** text model: a GGUF without a baked
tokenizer is not loadable, and the BPE pre-tokenizer id cannot be derived safely on-device, so the caller
declares it (it comes straight from upstream's `convert_hf_to_gguf.py` table — e.g. `"smollm"`,
`"llama-bpe"`, `"gpt-2"`). Omit it and resolution fails fast, before downloading anything.

**Bonsai (ternary QLlama)** converts on-device too — pass the adapter, no `tokenizerPre` needed (it folds
the per-output `.scales` into the weights and bakes a self-contained Llama-style tokenizer):

```kotlin
val bonsai = ModelSpec.safetensors(
    repoId = "deepgrove/Bonsai",
    precision = ConversionPrecision.Q4_K_M,
    adapter = ConversionAdapter.BONSAI_QLINEAR,
)
```

**Scope (v1):** the on-device converter handles **Llama-architecture models** with either a GPT2-BPE
tokenizer (needs `tokenizerPre`) or the Bonsai QLlama adapter, and a single-file `model.safetensors`.
Anything else — other architectures, other tokenizer families, sharded safetensors — fails loud with
instructions; for those, produce the GGUF once on a dev box / CI with
[`tools/safetensors-convert`](../tools/safetensors-convert/README.md) and drop it where the error message
points (the app cache), or load it directly via `ModelSpec.localFile("…/model.gguf")`.

Verified end-to-end on a real arm64 device (SmolLM, HF → convert → quantize → generate) and against the
host tool on `deepgrove/Bonsai` (147/147 tensors + 12/12 tokenizer KVs match; both emit the same text).

### Downloading Models from Hugging Face

For app code, prefer the facade-managed model repository:

```kotlin
val edge = LLMEdge.create(context, viewModelScope)

val modelFile = edge.models.prefetch(
    ModelSpec.huggingFace(
        repoId = "unsloth/Qwen3-0.6B-GGUF",
        filename = "Qwen3-0.6B-Q4_K_M.gguf",
        preferSystemDownloader = true,
    ),
    onProgress = { progress -> /* update UI */ },
)
```

Use direct runtime download helpers only when you intentionally want to own the expert runtime:

```kotlin
val download = smol.loadFromHuggingFace(
    context = context,
    modelId = "unsloth/Qwen3-0.6B-GGUF",
    filename = "Qwen3-0.6B-Q4_K_M.gguf",
    params = InferenceParams(contextSize = 4096L),
    preferSystemDownloader = true,
    onProgress = { downloaded, total -> /* update UI */ }
)
```

For Wan video models, prefer `edge.image.generateVideo(...)`. If you need manual multi-asset runtime ownership, use:

```kotlin
val sdWan = StableDiffusion.loadFromHuggingFace(
    context = context,
    modelId = "wan/Wan2.1-T2V-1.3B",
    preferSystemDownloader = true,
    onProgress = { name, downloaded, total -> /* update progress */ }
)
```

**Key features:**

- Downloads are cached automatically
- Supports private repositories with `token` parameter
- Uses Android DownloadManager for large files to avoid heap pressure
- Auto-resolves model aliases and mirrors
- Context size auto-caps based on device heap (override via `InferenceParams`)

See [HuggingFaceDemoActivity example](examples.md#huggingfacedemoactivity) for a complete implementation with progress updates and error handling.

### Reasoning Controls

Control "thinking" traces in reasoning-aware models:

```kotlin
// Disable thinking at load time
val params = InferenceParams(
    thinkingMode = ThinkingMode.DISABLED,
    reasoningBudget = 0
)
smol.load(modelPath, params)

// Toggle at runtime
smol.setThinkingEnabled(false)  // disable
smol.setReasoningBudget(-1)     // unrestricted
```

- `reasoningBudget = 0`: thinking disabled
- `reasoningBudget = -1`: unrestricted (default)
- The library auto-injects `/no_think` tags when disabled

### Image Text Extraction (OCR)

Extract text from images using Google ML Kit:

```kotlin
val mlKitEngine = MlKitOcrEngine(context)
val result = mlKitEngine.extractText(ImageSource.FileSource(imageFile))
println("Extracted: ${result.text}")
```

**Vision modes:**

- `AUTO_PREFER_OCR`: Try OCR first, fall back to vision
- `AUTO_PREFER_VISION`: Try vision first, fall back to OCR
- `FORCE_MLKIT`: ML Kit only
- `FORCE_VISION`: Vision model only

Use `ImageUnderstanding` to orchestrate between OCR and vision models with automatic fallback.

See [ImageToTextActivity example](examples.md#imagetotextactivity) for complete implementation including camera capture.

### Vision Models (Low-Level)

The library has interfaces for vision-capable LLMs (LLaVA-style models):

```kotlin
interface VisionModelAnalyzer {
    suspend fun analyze(image: ImageSource, prompt: String): VisionResult
    fun hasVisionCapabilities(): Boolean
}
```

**Status:** Architecture is prepared, but native vision support from llama.cpp is still being integrated for Android. Currently use OCR for text extraction. See [LlavaVisionActivity example](examples.md#llavavisionactivity) for the prepared integration pattern.


### Speech-to-Text (Whisper Low-Level)

Use `Whisper` directly for fine-grained control:

```kotlin
import io.aatricks.llmedge.Whisper

// Load model with options
val whisper = Whisper.load(
    modelPath = "/path/to/ggml-base.bin",
    useGpu = true // allow OpenCL/Vulkan when available
)

// Configure transcription parameters
val params = Whisper.TranscribeParams(
    language = "en",           // null for auto-detect
    translate = false,         // translate to English
    tokenTimestamps = true,
    beamSize = 1,
)

// Transcribe (16kHz mono PCM float32)
val segments = whisper.transcribe(audioSamples, params)
segments.forEach { segment ->
    println("[${segment.startTimeMs}-${segment.endTimeMs}ms] ${segment.text}")
}

// Utility functions
val srt = whisper.generateSrt(segments)
val lang = whisper.detectLanguage(audioSamples)
val isMultilingual = whisper.isMultilingual()
val modelType = whisper.getModelType()

whisper.close()
```

Set `useGpu = false` to force CPU. At runtime, use `LLMEdge.isOpenClAvailable()` and `LLMEdge.isVulkanAvailable()` to inspect device GPU capability.

**Model sources:**

- HuggingFace: `ggerganov/whisper.cpp` (ggml-tiny.bin, ggml-base.bin, ggml-small.bin)
- Sizes: tiny (~75MB), base (~142MB), small (~466MB)

### Text-to-Speech (Bark Low-Level)

Use `BarkTTS` directly:

```kotlin
import io.aatricks.llmedge.BarkTTS

// Load model
val tts = BarkTTS.load(
    modelPath = "/path/to/bark-small_weights-f16.bin",
    temperature = 0.7f,
    fineTemperature = 0.5f,
)

tts.setProgressCallback { step, progress ->
    Log.d("Bark", "${step.name}: $progress%")
}

val audio = tts.generate("Hello, world!", BarkTTS.GenerateParams(nThreads = 4))

// AudioResult contains:
// - samples: FloatArray (32-bit PCM)
// - sampleRate: Int (typically 24000)
// - durationSeconds: Float

// Save as WAV
tts.saveAsWav(audio, File("/path/to/output.wav"))

tts.close()
```

**Model sources:**

- HuggingFace: `Green-Sky/bark-ggml` (bark-small_weights-f16.bin, bark_weights-f16.bin)
- Sizes: small (~843MB), full (~2.2GB)


### Stable Diffusion (Image & Video Generation)

Generate images and video on-device using Stable Diffusion and Wan models:

**Image Generation:**

```kotlin
val sd = StableDiffusion.load(
    context = context,
    modelId = "Meina/MeinaMix",
    offloadToCpu = true,
    keepClipOnCpu = true,
    // Optional: Load with LoRA
    loraModelDir = "/path/to/your/lora/files", // Directory containing .safetensors
    loraApplyMode = StableDiffusion.LoraApplyMode.AUTO
)

val bitmap = sd.txt2img(
    GenerateParams(
        prompt = "a cute cat <lora:your_lora_name:1.0>", // LoRA tag in prompt
        width = 256, height = 256,
        steps = 20, cfgScale = 7.0f,
        // Optional: EasyCache parameters
        easyCacheParams = StableDiffusion.EasyCacheParams(enabled = true, reuseThreshold = 0.2f)
    )
)
sd.close()
```

**Video Generation (Wan 2.1):**

```kotlin
// Load Wan model (loads diffusion, VAE, and T5 encoder)
val sd = StableDiffusion.loadFromHuggingFace(
    context = context,
    modelId = "wan/Wan2.1-T2V-1.3B",
    preferSystemDownloader = true
)

val frames = sd.txt2vid(
    VideoGenerateParams(
        prompt = "A cinematic shot of a robot walking",
        width = 480, height = 480,
        videoFrames = 16,
        steps = 20
    )
)
sd.close()
```

**Memory management:**

- Use small resolutions (128x128 or 256x256) on constrained devices
- Enable CPU offloading flags to reduce native memory pressure
- Always use `preferSystemDownloader = true` for model downloads
- Monitor with `MemoryMetrics` to avoid OOM

See [StableDiffusionActivity example](examples.md#stablediffusionactivity) for complete implementation with error recovery and adaptive resolution.

### Best Practices

**Threading:**

- Route blocking JNI/native work through `Dispatchers.IO` (or the library inference dispatcher used by `LLMEdge`).
- Reserve `Dispatchers.Default` for pure Kotlin/Java CPU work such as post-processing that does not block on JNI calls.
- Update UI only via `withContext(Dispatchers.Main)`.
- Call `.close()` in `onDestroy()` to free native memory.

**Optimization Strategies**:

- Use quantized models (Q4_K_M) for lower memory footprint
- Enable CPU offloading for large models
- Close model instances when not in use
- Process images/video in batches with intermediate cleanup
- Prefer batched text generation (`batchSize > 1`) for blocking calls that do not need token-level UI updates
- Use different thread counts for prompt/batch work and single-token generation when tuning for big.LITTLE devices
- Text-model cache sizing is now refreshed from the native model/state footprint, so `textCacheMemoryMb` is a meaningful guardrail instead of just a file-size hint
- **LLM chat memory**:
    - `storeChats` is deprecated but still available for tightly scoped low-level compatibility flows that intentionally keep chat state inside one native runtime.
  - Use `edge.text.session(...)` when you need bounded history replay or want to strip older reasoning traces before replay.

**See also:**

- [Architecture](architecture.md) for system design and flow diagrams
- [Quirks & Troubleshooting](quirks.md) for detailed JNI notes and debugging
- [Examples](examples.md) for complete working code

### API reference

Key methods:

- `LLMEdge.create(...)` — creates the instance-based high-level facade
- `edge.text.generate(...)` — high-level text generation
- `edge.text.stream(...)` — high-level text streaming
- `edge.text.session(...)` — creates a Kotlin-managed multi-turn chat session
- `TextGenerationRequest.batchSize` — blocking generation batch size (`0` = use configured default)
- `edge.text.stream(..., batchSize = ...)` / `text.ChatSession.stream(..., batchSize = ...)` — streaming batch size override (`0` = use configured default)
- `TextModelOptions.numThreads` / `generationThreads` — prompt/batch vs single-token thread counts
- `edge.image.generate(...)` — high-level image generation
- `edge.image.generateVideo(...)` — high-level video generation
- `edge.speech.transcribe(...)` — high-level speech-to-text
- `edge.speech.synthesize(...)` — high-level text-to-speech
- `SpeechToTextRequest`, `SpeechLanguageDetectionRequest`, `StreamingTranscriptionRequest`, and `SpeechSynthesisRequest` — preferred request-first speech API shapes
- `VisionRequest` and `VisionPrepareRequest` — preferred request-first vision API shapes
- `SmolLM.load(modelPath: String, params: InferenceParams)` — loads a GGUF model from a path
- `SmolLM.loadFromHuggingFace(...)` — downloads and loads a model from Hugging Face
- `SmolLM.getResponse(query: String): String` — runs blocking generation and returns complete text
- `SmolLM.getResponseAsFlow(query: String): Flow<String>` — runs streaming generation
- `SmolLM.getEstimatedNativeMemoryBytes()` / `getEstimatedStateMemoryBytes()` — expose native model/state memory estimates
- `SmolLM.addSystemPrompt(prompt: String)` — adds system prompt to chat history
- `SmolLM.addUserMessage(message: String)` — adds user message to chat history
- `text.ChatSession.reply(message: String): String` — runs bounded multi-turn chat with Kotlin-managed history
- `text.ChatSession.stream(message: String): Flow<TextStreamEvent>` — streams a bounded reply while persisting the final assistant turn
- `ConversationWindow(...)` — configures sliding-window size, token budget, and reasoning stripping
- `SmolLM.close()` — releases native resources

**High-Level Speech API (via `LLMEdge`):**
- `edge.speech.transcribeToText(audioSamples, model?, params?, loadOptions?)` — simple audio transcription
- `edge.speech.transcribe(audioSamples, model?, params?, loadOptions?)` — full transcription with segments
- `edge.speech.detectLanguage(audioSamples, model?, loadOptions?)` — detect spoken language
- `edge.speech.createStreamingSession(model?, params?, loadOptions?)` — create a reusable streaming transcriber
- `edge.speech.synthesize(text, model?, params?, loadOptions?)` — generate speech from text
- `edge.speech.synthesizeStream(text, model?, params?, loadOptions?)` — stream speech generation events
- Request-first overloads are preferred for new code; parameter-list overloads remain for compatibility
- `LLMEdge.isOpenClAvailable()` / `LLMEdge.isVulkanAvailable()` — query Android GPU backend capability

**Low-Level Speech API:**
- `Whisper.load(modelPath: String, useGpu: Boolean, flashAttn: Boolean = true, gpuDevice: Int = 0)` — loads a Whisper model; on Android, `useGpu = true` allows OpenCL first, then Vulkan, then CPU fallback
- `Whisper.loadFromHuggingFace(...)` — downloads and loads Whisper from HuggingFace
- `Whisper.transcribe(samples: FloatArray, params: TranscribeParams)` — transcribes audio
- `Whisper.detectLanguage(samples: FloatArray)` — detects spoken language
- `Whisper.close()` — releases native resources
- `BarkTTS.load(modelPath: String, ...)` — loads a Bark TTS model
- `BarkTTS.loadFromHuggingFace(...)` — downloads and loads Bark from HuggingFace
- `BarkTTS.generate(text: String, params: GenerateParams)` — generates audio from text
- `BarkTTS.saveAsWav(audio: AudioResult, filePath: String)` — saves audio to WAV file
- `BarkTTS.close()` — releases native resources

**Vision & OCR:**
- `OcrEngine.extractText(image: ImageSource, params: OcrParams): OcrResult` — extracts text from image
- `ImageUnderstanding.process(image: ImageSource, mode: VisionMode, prompt: String?)` — processes image with vision/OCR

**Image & Video:**
- `StableDiffusion.txt2img(params: GenerateParams): Bitmap` — generates an image
- `StableDiffusion.txt2vid(params: VideoGenerateParams): List<Bitmap>` — generates video frames

Refer to the `llmedge-examples` activities for complete, working code samples.
