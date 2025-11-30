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

Automatically handles model downloading (MeinaMix) and memory-safe loading.

```kotlin
val bitmap = LLMEdgeManager.generateImage(
    context = context,
    params = LLMEdgeManager.ImageGenerationParams(
        prompt = "A cyberpunk city street at night, neon lights",
        width = 512,
        height = 512,
        steps = 20
    )
)
```

### Video Generation (Wan 2.1)

Handles the complex multi-model loading (Diffusion, VAE, T5) and sequential processing required for video generation on mobile.

```kotlin
val frames = LLMEdgeManager.generateVideo(
    context = context,
    params = LLMEdgeManager.VideoGenerationParams(
        prompt = "A robot dancing in the rain",
        videoFrames = 16,
        width = 480,
        height = 480,
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

---

## Low-Level API

Direct usage of `SmolLM` and `StableDiffusion` classes. Use this if you need to manage the model lifecycle manually (e.g., keeping a model loaded across multiple disparate activities) or require specific configuration not exposed by the Manager.

### Core components

- `SmolLM` — Kotlin front-end class that wraps native inference calls.
- `GGUFReader` — C++/JNI reader for GGUF model files.
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


### Stable Diffusion (Image & Video Generation)

Generate images and video on-device using Stable Diffusion and Wan models:

**Image Generation:**

```kotlin
val sd = StableDiffusion.load(
    context = context,
    modelId = "Meina/MeinaMix",
    offloadToCpu = true,
    keepClipOnCpu = true
)

val bitmap = sd.txt2img(
    GenerateParams(
        prompt = "a cute cat",
        width = 256, height = 256,
        steps = 20, cfgScale = 7.0f
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

**Memory optimization:**

- Use quantized models (Q4_K_M, Q5_K_M)
- Reduce `contextSize` (2048-4096 for constrained devices)
- Cap `numThreads` to avoid CPU oversubscription
- Monitor with `MemoryMetrics.snapshot()`

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
- `OcrEngine.extractText(image: ImageSource, params: OcrParams): OcrResult` — extracts text from image
- `ImageUnderstanding.process(image: ImageSource, mode: VisionMode, prompt: String?)` — processes image with vision/OCR
- `StableDiffusion.txt2img(params: GenerateParams): Bitmap` — generates an image
- `StableDiffusion.txt2vid(params: VideoGenerateParams): List<Bitmap>` — generates video frames

Refer to the `llmedge-examples` activities for complete, working code samples.
