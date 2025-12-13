# Usage & API

This page explains how to use `llmedge`'s Kotlin API. The library offers two layers of abstraction:

1.  **High-Level API (`LLMEdgeManager`)**: Recommended for most use cases. Handles loading, caching, threading, and memory management automatically.
2.  **Low-Level API (`SmolLM`, `StableDiffusion`)**: For advanced users who need fine-grained control over model lifecycle and parameters.

Examples reference the `llmedge-examples` [repo](https://github.com/aatricks/llmedge-examples).

---

## High-Level API (LLMEdgeManager)

The `LLMEdgeManager` singleton simplifies interactions by managing model instances and resources for you.

### Text Generation

```kotlin
val response = LLMEdgeManager.generateText(
    context = context,
    params = LLMEdgeManager.TextGenerationParams(
        prompt = "Write a haiku about Kotlin.",
        modelId = "HuggingFaceTB/SmolLM-135M-Instruct-GGUF", // Optional: defaults to this model
        modelFilename = "smollm-135m-instruct.q4_k_m.gguf"
    )
)
```

### Image Generation

Automatically handles model downloading (MeinaMix), caching, and memory-safe loading.

```kotlin
val bitmap = LLMEdgeManager.generateImage(
    context = context,
    params = LLMEdgeManager.ImageGenerationParams(
        prompt = "A cyberpunk city street at night, neon lights <lora:detail_tweaker_lora_sd15:1.0>",
        width = 512,
        height = 512,
        steps = 20,
        // Optional: Specify LoRA model details.
        // The example app automatically downloads 'imagepipeline/Detail-Tweaker-LoRA-SD1.5'
        // and appends the appropriate LoRA tag to the prompt when the toggle is enabled.
        loraModelDir = getExternalFilesDir("loras")?.absolutePath + "/detail-tweaker-lora-sd15",
        loraApplyMode = StableDiffusion.LoraApplyMode.AUTO
    )
)
```

**Key Optimizations for Image Generation:**

- **EasyCache**: Automatically detected and enabled by `LLMEdgeManager` for supported Diffusion Transformer (DiT) models (e.g., Flux, Wan), significantly accelerating generation by reusing intermediate diffusion states. It is disabled for UNet-based models (like SD 1.5).
- **LoRA Support**: `LLMEdgeManager` can be configured with `loraModelDir` and `loraApplyMode` to use Low-Rank Adaptation models for fine-tuning outputs.
- **Flash Attention**: Automatically enabled for compatible image dimensions.

### Video Generation (Wan 2.1)

Handles the complex multi-model loading (Diffusion, VAE, T5) and sequential processing required for video generation on mobile.

```kotlin
val frames = LLMEdgeManager.generateVideo(
    context = context,
    params = LLMEdgeManager.VideoGenerationParams(
        prompt = "A robot dancing in the rain",
        videoFrames = 16,
        width = 512,
        height = 512,
        steps = 20,
        cfgScale = 7.0f,
        flowShift = 3.0f,
        forceSequentialLoad = true // Recommended for devices with <12GB RAM
    )
) { status, currentFrame, totalFrames ->
    // Optional progress callback
    Log.d("Video", "$status")
}
```

### Vision Analysis

Analyze images using a Vision Language Model (VLM).

```kotlin
val description = LLMEdgeManager.analyzeImage(
    context = context,
    params = LLMEdgeManager.VisionAnalysisParams(
        image = bitmap,
        prompt = "What is in this image?"
    )
)
```

### OCR (Text Extraction)

Extract text using ML Kit.

```kotlin
val text = LLMEdgeManager.extractText(context, bitmap)
```

### Speech-to-Text (Whisper)

Transcribe audio using the high-level API:

```kotlin
import io.aatricks.llmedge.LLMEdgeManager

// Simple transcription to text
val text = LLMEdgeManager.transcribeAudioToText(
    context = context,
    audioSamples = audioSamples  // 16kHz mono PCM float32
)

// Full transcription with segments and timing
val segments = LLMEdgeManager.transcribeAudio(
    context = context,
    params = LLMEdgeManager.TranscriptionParams(
        audioSamples = audioSamples,
        language = "en",  // null for auto-detect
        translate = false  // set true to translate to English
    )
) { progress ->
    Log.d("Whisper", "Progress: $progress%")
}

// Language detection
val lang = LLMEdgeManager.detectLanguage(context, audioSamples)

// Generate SRT subtitles
val srt = LLMEdgeManager.transcribeToSrt(context, audioSamples)
```

### Streaming Transcription (Real-time Captioning)

For live transcription from a microphone or audio stream, use the streaming API:

```kotlin
import io.aatricks.llmedge.LLMEdgeManager
import kotlinx.coroutines.launch

// Create a streaming transcriber with sliding window approach
val transcriber = LLMEdgeManager.createStreamingTranscriber(
    context = context,
    params = LLMEdgeManager.StreamingTranscriptionParams(
        stepMs = 3000,      // Run transcription every 3 seconds
        lengthMs = 10000,   // Use 10-second audio windows
        keepMs = 200,       // Keep 200ms overlap for context
        language = "en",    // null for auto-detect
        useVad = true       // Skip silent audio
    )
)

// Collect real-time transcription results
launch {
    transcriber.start().collect { segment ->
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
    LLMEdgeManager.stopStreamingTranscription()
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
import io.aatricks.llmedge.LLMEdgeManager

// Generate speech (model auto-downloads on first use)
val audio = LLMEdgeManager.synthesizeSpeech(
    context = context,
    params = LLMEdgeManager.SpeechSynthesisParams(
        text = "Hello, world!",
        nThreads = 8  // Use more threads for faster generation
    )
) { step, progress ->
    Log.d("Bark", "${step.name}: $progress%")
}

// Save directly to file
LLMEdgeManager.synthesizeSpeechToFile(
    context = context,
    text = "Hello, world!",
    outputFile = File(cacheDir, "output.wav")
)

// Unload when done to free memory
LLMEdgeManager.unloadSpeechModels()
```

---

## Low-Level API

Direct usage of `SmolLM` and `StableDiffusion` classes. Use this if you need to manage the model lifecycle manually (e.g., keeping a model loaded across multiple disparate activities) or require specific configuration not exposed by the Manager.

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

See [Examples](examples.md#localassetdemoactivity) for complete working code with coroutines and streaming.

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
val params = Whisper.TranscriptionParams(
    language = "en",           // null for auto-detect
    translate = false,         // translate to English
    maxTokens = 0,             // 0 = no limit
    speedUp = false,           // experimental 2x speedup
    audioCtx = 0               // audio context size
)

// Transcribe (16kHz mono PCM float32)
val segments = whisper.transcribe(audioSamples, params)
segments.forEach { segment ->
    println("[${segment.startTimeMs}-${segment.endTimeMs}ms] ${segment.text}")
}

// Utility functions
val srt = whisper.transcribeToSrt(audioSamples)
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
    useGpu = false
)

// Generate audio with progress tracking
val audio = tts.generate(
    text = "Hello, world!",
    onProgress = { step, progress ->
        // step: SEMANTIC, COARSE, or FINE
        // progress: 0.0 to 1.0
        Log.d("Bark", "${step.name}: ${(progress * 100).toInt()}%")
    }
)

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

- For heavy CPU-bound native inference (StableDiffusion CPU generation, large LLM decoding), prefer `Dispatchers.Default` so work schedules onto CPU-optimized thread pool sized to the device cores.
- For blocking calls that wait on I/O (downloads, filesystem access, or JNI calls that wait on network/IO), prefer `Dispatchers.IO`.
- Update UI only via `withContext(Dispatchers.Main)`.
- Call `.close()` in `onDestroy()` to free native memory.

Note: Choosing the correct dispatcher depends on the workload. JNI calls that use CPU-bound native kernels (like `txt2img`) should use Default; calls that perform blocking I/O should use IO.

**Optimization Strategies**:

- Use quantized models (Q4_K_M) for lower memory footprint
- Enable CPU offloading for large models
- Close model instances when not in use
- Process images/video in batches with intermediate cleanup
- **KV Cache**: For LLMs, ensure `storeChats = true` to leverage KV cache for faster multi-turn conversations.

**See also:**

- [Architecture](architecture.md) for system design and flow diagrams
- [Quirks & Troubleshooting](quirks.md) for detailed JNI notes and debugging
- [Examples](examples.md) for complete working code

### API reference

Key methods:

- `LLMEdgeManager.generateText(...)` — High-level text generation
- `LLMEdgeManager.generateImage(...)` — High-level image generation
- `LLMEdgeManager.generateVideo(...)` — High-level video generation
- `SmolLM.load(modelPath: String, params: InferenceParams)` — loads a GGUF model from a path
- `SmolLM.loadFromHuggingFace(...)` — downloads and loads a model from Hugging Face
- `SmolLM.getResponse(query: String): String` — runs blocking generation and returns complete text
- `SmolLM.getResponseAsFlow(query: String): Flow<String>` — runs streaming generation
- `SmolLM.addSystemPrompt(prompt: String)` — adds system prompt to chat history
- `SmolLM.addUserMessage(message: String)` — adds user message to chat history
- `SmolLM.close()` — releases native resources

**High-Level Speech API (via LLMEdgeManager):**
- `LLMEdgeManager.transcribeAudioToText(context, audioSamples, language?)` — simple audio transcription
- `LLMEdgeManager.transcribeAudio(context, params, onProgress?)` — full transcription with segments
- `LLMEdgeManager.detectLanguage(context, audioSamples)` — detect spoken language
- `LLMEdgeManager.transcribeToSrt(context, audioSamples, language?)` — generate SRT subtitles
- `LLMEdgeManager.synthesizeSpeech(context, params, onProgress?)` — generate speech from text
- `LLMEdgeManager.synthesizeSpeechToFile(context, text, outputFile, onProgress?)` — generate and save WAV
- `LLMEdgeManager.unloadSpeechModels()` — unload all speech models

**Low-Level Speech API:**
- `Whisper.load(modelPath: String, useGpu: Boolean)` — loads a Whisper model
- `Whisper.loadFromHuggingFace(...)` — downloads and loads Whisper from HuggingFace
- `Whisper.transcribe(samples: FloatArray, params: TranscriptionParams)` — transcribes audio
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
