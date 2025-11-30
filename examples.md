# Examples

The `llmedge-examples` [repo](https://github.com/aatricks/llmedge-examples) demonstrates multiple real-world uses.

## Example Activities

The `llmedge-examples` repository contains complete working demonstrations:

| Activity | Purpose |
|----------|----------|
| `LocalAssetDemoActivity` | Load bundled GGUF model, blocking & streaming inference |
| `HuggingFaceDemoActivity` | Download from HF Hub, reasoning controls |
| `ImageToTextActivity` | Camera capture, OCR text extraction |
| `LlavaVisionActivity` | Vision model integration (prepared architecture) |
| `StableDiffusionActivity` | On-device image generation |
| `VideoGenerationActivity` | On-device text-to-video generation (Wan 2.1) |
| `RagActivity` | PDF indexing, vector search, Q&A |

---


## LocalAssetDemoActivity

Demonstrates loading a bundled model asset and running inference using `LLMEdgeManager`.

**Key patterns:**

```kotlin
val modelPath = copyAssetIfNeeded("YourModel.gguf")

// Run inference (Manager handles loading & caching)
val response = LLMEdgeManager.generateText(
    context = context,
    params = LLMEdgeManager.TextGenerationParams(
        prompt = "Say 'hello from llmedge'.",
        modelPath = modelPath,
        modelId = "local-model" // Unique ID for caching
    )
)
```



---

## ImageToTextActivity

Demonstrates camera integration and OCR text extraction using `LLMEdgeManager`.

**Key features:**

- Camera permission handling
- High-resolution image capture
- ML Kit OCR integration via Manager

**Code snippet:**

```kotlin
// Process image
val bitmap = ImageUtils.fileToBitmap(file)
ivPreview.setImageBitmap(bitmap)

// Extract text (handles engine lifecycle automatically)
val text = LLMEdgeManager.extractText(context, bitmap)
tvResult.text = text.ifEmpty { "(no text detected)" }
```

---

## HuggingFaceDemoActivity

Shows how to download models from Hugging Face Hub and run inference using `LLMEdgeManager`.

**Key features:**

- Model download with progress callback
- Heap-aware parameter selection
- Thinking mode configuration

**Code snippet:**

```kotlin
// 1. Download with progress
LLMEdgeManager.downloadModel(
    context = context,
    modelId = "unsloth/Qwen3-0.6B-GGUF",
    filename = "Qwen3-0.6B-Q4_K_M.gguf",
    onProgress = { downloaded, total -> /* update UI */ }
)

// 2. Generate text (model is now cached)
val response = LLMEdgeManager.generateText(
    context = context,
    params = LLMEdgeManager.TextGenerationParams(
        prompt = "List two quick facts about running GGUF models on Android.",
        modelId = "unsloth/Qwen3-0.6B-GGUF",
        modelFilename = "Qwen3-0.6B-Q4_K_M.gguf",
        thinkingMode = SmolLM.ThinkingMode.DISABLED // Optional reasoning control
    )
)
```

---

## RagActivity

Complete RAG (Retrieval-Augmented Generation) pipeline with PDF indexing and Q&A.

**Key features:**

- PDF document picker with persistent permissions
- Sentence embeddings (ONNX Runtime)
- Vector store with cosine similarity search
- Context-aware answer generation

**Full workflow:**

```kotlin
// 1. Get shared LLM instance from Manager
// (Ensures efficient resource sharing with other app components)
val llm = LLMEdgeManager.getSmolLM(context)

// 2. Initialize RAG engine with embedding config
val rag = RAGEngine(
    context = context,
    smolLM = llm,
    splitter = TextSplitter(chunkSize = 600, chunkOverlap = 120),
    embeddingConfig = EmbeddingConfig(
        modelAssetPath = "embeddings/all-minilm-l6-v2/model.onnx",
        tokenizerAssetPath = "embeddings/all-minilm-l6-v2/tokenizer.json"
    )
)
rag.init()

// 3. Index a PDF document
val count = rag.indexPdf(pdfUri)
Log.d("RAG", "Indexed $count chunks")

// 4. Ask questions
val answer = rag.ask("What are the key points?", topK = 5)
```

**Important notes:**

- Scanned PDFs require OCR before indexing (PDFBox extracts text-based only)
- Embedding model must be in `assets/` directory
- Vector store persists to JSON in app files directory
- Adjust chunk size/overlap based on document type

---

## StableDiffusionActivity

Demonstrates on-device image generation using `LLMEdgeManager`.

**Key patterns:**

```kotlin
// Generate image (Manager handles downloading, caching, VAE loading, and memory safety)
val bitmap = LLMEdgeManager.generateImage(
    context = context,
    params = LLMEdgeManager.ImageGenerationParams(
        prompt = "a cute cat",
        width = 256, // Start small on mobile
        height = 256,
        steps = 20,
        cfgScale = 7.0f
    )
)

// Display result
imageView.setImageBitmap(bitmap)
```

**Important notes:**

- Start with 128×128 or 256x256 on devices with <4GB RAM
- `LLMEdgeManager` automatically enables CPU offloading if memory is tight
- Generating images is memory-intensive; close other apps for best results

---

## VideoGenerationActivity

Demonstrates on-device text-to-video generation using Wan 2.1.

**Key patterns:**

```kotlin
// 1. Configure params for mobile-friendly generation
val params = LLMEdgeManager.VideoGenerationParams(
    prompt = "A dog running in the park",
    width = 480,      // 480x480 is a good balance
    height = 480,
    videoFrames = 8,  // Start with fewer frames (8-16)
    steps = 15,       // Reduced steps for speed
    forceSequentialLoad = true // Critical for devices with <12GB RAM
)

// 2. Generate video (returns list of bitmaps)
val frames = LLMEdgeManager.generateVideo(
    context = context,
    params = params
) { phase, current, total ->
    // Update progress UI
    updateProgress("$phase ($current/$total)")
}

// 3. Display or save frames
imageView.setImageBitmap(frames.first())
```

**Important notes:**

- **Sequential Load:** Video models are large (Wan 2.1 is ~5GB). `forceSequentialLoad = true` is essential for mobile devices; it loads components (T5 encoder, Diffusion model, VAE) one by one to keep peak memory low.
- **Frame Count:** Generating 8-16 frames takes significant time. Provide progress feedback.
- **LLMEdgeManager:** This activity uses the high-level `LLMEdgeManager` which handles the complex sequentially loading logic automatically.

---

## LlavaVisionActivity

Demonstrates vision-capable LLM integration using `LLMEdgeManager`.

**Key patterns:**

```kotlin
// 1. Preprocess image
val bmp = ImageUtils.imageToBitmap(context, uri)
val scaledBmp = ImageUtils.preprocessImage(bmp, correctOrientation = true, maxDimension = 1024)

// 2. Run OCR (Optional grounding)
val ocrText = LLMEdgeManager.extractText(context, scaledBmp)

// 3. Build Prompt (e.g. ChatML format for Phi-3)
val prompt = """
    <|system|>You are a helpful assistant.<|end|>
    <|user|>
    Context (OCR): $ocrText
    
    Describe this image.<|end|>
    <|assistant|>
""".trimIndent()

// 4. Run Vision Analysis
val result = LLMEdgeManager.analyzeImage(
    context = context,
    params = LLMEdgeManager.VisionAnalysisParams(
        image = scaledBmp,
        prompt = prompt
    )
)
```

**Status**: Uses `LLMEdgeManager` to orchestrate the experimental vision pipeline (loading projector, encoding image, running inference).

---

## Additional Resources

- [Architecture](architecture.md) for flow diagrams (RAG, OCR, JNI loading)
- [Usage](usage.md) for API reference and concepts
- [Quirks](quirks.md) for troubleshooting specific issues

All example activities are intentionally minimal and well-documented. Copy and adapt them into your own app!