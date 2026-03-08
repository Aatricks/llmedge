# Usage & API

This page explains how to use `llmedge`'s Kotlin API. The library offers two layers of abstraction:

1.  **High-Level API (`LLMEdge`)**: Recommended for most use cases. It exposes instance-scoped clients for text, speech, image, vision, and RAG while keeping model resolution and cleanup explicit.
2.  **Low-Level API (`SmolLM`, `StableDiffusion`)**: For advanced users who need fine-grained control over model lifecycle and parameters.

Examples reference the `llmedge-examples` [repo](https://github.com/aatricks/llmedge-examples).

---

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
val description = edge.vision.analyze(bitmap, "What is in this image?")
```

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

val text = edge.speech.transcribeToText(audioSamples)

val segments =
    edge.speech.transcribe(
        audioSamples = audioSamples,
        params = Whisper.TranscribeParams(language = "en", translate = false),
    )

val lang = edge.speech.detectLanguage(audioSamples)
```

### Streaming Transcription (Real-time Captioning)

For live transcription from a microphone or audio stream, use the streaming API:

```kotlin
import kotlinx.coroutines.launch

val edge = LLMEdge.create(context, lifecycleScope)
val transcriber = edge.speech.createStreamingSession(
    params = Whisper.StreamingParams(
        stepMs = 3000,      // Run transcription every 3 seconds
        lengthMs = 10000,   // Use 10-second audio windows
        keepMs = 200,       // Keep 200ms overlap for context
        language = "en",    // null for auto-detect
        useVad = true       // Skip silent audio
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

val audio = edge.speech.synthesize("Hello, world!")
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

### Downloading Models from Hugging Face

Download and load models directly from Hugging Face Hub:

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

For Wan video models (multi-asset: diffusion, VAE and encoder), use:

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
    useGpu = false
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
- `edge.image.generate(...)` — high-level image generation
- `edge.image.generateVideo(...)` — high-level video generation
- `edge.speech.transcribe(...)` — high-level speech-to-text
- `edge.speech.synthesize(...)` — high-level text-to-speech
- `SmolLM.load(modelPath: String, params: InferenceParams)` — loads a GGUF model from a path
- `SmolLM.loadFromHuggingFace(...)` — downloads and loads a model from Hugging Face
- `SmolLM.getResponse(query: String): String` — runs blocking generation and returns complete text
- `SmolLM.getResponseAsFlow(query: String): Flow<String>` — runs streaming generation
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

**Low-Level Speech API:**
- `Whisper.load(modelPath: String, useGpu: Boolean, flashAttn: Boolean = true, gpuDevice: Int = 0)` — loads a Whisper model
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
