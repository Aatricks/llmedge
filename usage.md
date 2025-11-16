# Usage

This page explains how to use `llmedge`'s Kotlin API and native bindings. Examples reference the `llmedge-examples` [repo](https://github.com/aatricks/llmedge-examples).

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

### Vision Models

The library has interfaces for vision-capable LLMs (LLaVA-style models):

```kotlin
interface VisionModelAnalyzer {
    suspend fun analyze(image: ImageSource, prompt: String): VisionResult
    fun hasVisionCapabilities(): Boolean
}
```

**Status:** Architecture is prepared, but native vision support from llama.cpp is still being integrated for Android. Currently use OCR for text extraction. See [LlavaVisionActivity example](examples.md#llavavisionactivity) for the prepared integration pattern.


### Stable Diffusion (Image Generation)

Generate images on-device using Stable Diffusion:

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

**Memory management:**

- Use small resolutions (128x128 or 256x256) on constrained devices
- Enable CPU offloading flags to reduce native memory pressure
- Always use `preferSystemDownloader = true` for model downloads
- Monitor with `MemoryMetrics` to avoid OOM

See [StableDiffusionActivity example](examples.md#stablediffusionactivity) for complete implementation with error recovery and adaptive resolution.

### Best Practices

**Threading:**

- Always use background dispatchers (`Dispatchers.IO` or `Dispatchers.Default`) for model loading and inference
- Update UI only via `withContext(Dispatchers.Main)`
- Call `.close()` in `onDestroy()` to free native memory

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

- `SmolLM.load(modelPath: String, params: InferenceParams)` — loads a GGUF model from a path
- `SmolLM.loadFromHuggingFace(...)` — downloads and loads a model from Hugging Face
- `SmolLM.getResponse(query: String): String` — runs blocking generation and returns complete text
- `SmolLM.getResponseAsFlow(query: String): Flow<String>` — runs streaming generation
- `SmolLM.addSystemPrompt(prompt: String)` — adds system prompt to chat history
- `SmolLM.addUserMessage(message: String)` — adds user message to chat history
- `SmolLM.close()` — releases native resources
- `OcrEngine.extractText(image: ImageSource, params: OcrParams): OcrResult` — extracts text from image
- `ImageUnderstanding.process(image: ImageSource, mode: VisionMode, prompt: String?)` — processes image with vision/OCR

Refer to the `llmedge-examples` activities for complete, working code samples.