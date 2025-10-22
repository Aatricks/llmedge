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
| `RagActivity` | PDF indexing, vector search, Q&A |

---


## LocalAssetDemoActivity

Demonstrates loading a bundled model asset and running both blocking and streaming inference.

**Key patterns:**

```kotlin
val modelPath = copyAssetIfNeeded("YourModel.gguf")
llm.load(
	modelPath,
	InferenceParams(numThreads = Runtime.getRuntime().availableProcessors().coerceAtMost(4), contextSize = 8192L)
)
llm.addSystemPrompt("You are a helpful assistant.")

// Blocking generation
val response = withContext(Dispatchers.Default) { llm.getResponse("Say 'hello from llmedge'.") }

// Streaming generation (collect pieces)
val sb = StringBuilder()
withContext(Dispatchers.Default) {
	llm.getResponseAsFlow("Write a short haiku about Android.")
		.collect { piece -> if (piece != "[EOG]") sb.append(piece) }
}
```



---

## ImageToTextActivity

Demonstrates camera integration and OCR text extraction.

**Key features:**

- Camera permission handling
- High-resolution image capture with FileProvider
- ML Kit OCR integration
- Error handling and lifecycle management

**Code snippet:**

```kotlin
// Initialize OCR engine
val ocrEngine = MlKitOcrEngine(context)

// Process image
val bitmap = ImageUtils.fileToBitmap(file)
ivPreview.setImageBitmap(bitmap)
val result = ocrEngine.extractText(ImageSource.BitmapSource(bitmap))
tvResult.text = result.text.ifEmpty { "(no text detected)" }
```

---

## HuggingFaceDemoActivity

Shows how to download models from Hugging Face Hub with progress tracking and reasoning controls.

**Key features:**

- Model download with progress callback
- Heap-aware parameter selection
- Thinking mode configuration
- Performance metrics display

**Code snippet:**

```kotlin
val safeParams = InferenceParams(
    storeChats = false,
    numThreads = 4,
    contextSize = 4096L
)

val result = llm.loadFromHuggingFace(
    context = this,
    modelId = "unsloth/Qwen3-0.6B-GGUF",
    revision = "main",
    filename = null,
    params = safeParams,
    forceDownload = false,
    onProgress = { downloaded, total -> /* update UI */ }
)

val response = withContext(Dispatchers.Default) { 
    llm.getResponse("List two quick facts about running GGUF models on Android.") 
}
```

---

## RagActivity

Complete RAG (Retrieval-Augmented Generation) pipeline with PDF indexing and Q&A.

**Key features:**

- PDF document picker with persistent permissions
- Sentence embeddings (ONNX Runtime)
- Vector store with cosine similarity search
- Context-aware answer generation
- Retrieval preview for debugging

**Full workflow:**

```kotlin
// 1. Load LLM
val llm = SmolLM()
llm.load(modelPath, InferenceParams(
    contextSize = 8192L,
    temperature = 0.2f,
    storeChats = false
))

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

// 5. Preview retrieval (debugging)
val preview = rag.retrievalPreview(question, topK = 5)
val context = rag.contextFor(question, topK = 5)
```

**Important notes:**

- Scanned PDFs require OCR before indexing (PDFBox extracts text-based only)
- Embedding model must be in `assets/` directory
- Vector store persists to JSON in app files directory
- Adjust chunk size/overlap based on document type

---

## StableDiffusionActivity

Demonstrates on-device image generation with memory-safe downloads and OOM recovery.

**Key patterns:**

```kotlin
// 1. Download VAE with system downloader (avoids heap OOM)
val vaeDownload = HuggingFaceHub.ensureRepoFileOnDisk(
    context = context,
    modelId = "Meina/MeinaMix",
    filename = "MeinaPastel - baked VAE.safetensors",
    preferSystemDownloader = true  // Critical for large files
)

// 2. Load model with CPU offloading for constrained devices
val sd = StableDiffusion.load(
    context = context,
    modelId = "Meina/MeinaMix",
    offloadToCpu = true,      // Reduces native memory pressure
    keepClipOnCpu = true,     // Keep CLIP on CPU
    vaePath = vaeDownload.file.absolutePath
)

// 3. Generate with adaptive resolution fallback
val bitmap = try {
    sd.txt2img(GenerateParams(
        prompt = "a cute cat",
        width = 128, height = 128,
        steps = 20, cfgScale = 7.0f
    ))
} catch (oom: OutOfMemoryError) {
    // Fallback: smaller resolution, fewer steps
    sd.txt2img(GenerateParams(
        prompt = "a cute cat",
        width = 64, height = 64,
        steps = 10, cfgScale = 7.0f
    ))
}

sd.close()  // Always close immediately
```

**Important notes:**

- Start with 128×128 or smaller on devices with <3GB RAM
- Always use `preferSystemDownloader = true` for downloads >50MB
- Enable all CPU offload flags if experiencing OOM
- Don't persist SD instance across generations (reload is acceptable)

---

## LlavaVisionActivity

Demonstrates vision-capable LLM integration with OCR and projector (mmproj) support.

**Key patterns:**

```kotlin
// 1. Load vision model with conservative settings
val smol = SmolLM()
smol.loadFromHuggingFace(
    context, 
    modelId = "xtuner/llava-phi-3-mini-gguf",
    params = InferenceParams(
        temperature = 0.0f,  // Reduce hallucination
        thinkingMode = ThinkingMode.DISABLED
    )
)

// 2. Download and initialize projector for vision embeddings
val mmprojFile = HuggingFaceHub.ensureModelOnDisk(
    context, modelId, 
    filename = "llava-phi-3-mini-mmproj-f16.gguf",
    preferSystemDownloader = true
)

val projector = Projector()
projector.init(mmprojFile.file.absolutePath, smol.getNativeModelPointer())

// 3. Preprocess image and encode to embeddings
val preprocessed = ImageUtils.preprocessImage(
    bitmap, 
    correctOrientation = true, 
    maxDimension = 1600
)
val embeddingsFile = File.createTempFile("vision_embd", ".bin", cacheDir)
projector.encodeImageToFile(imageFile.absolutePath, embeddingsFile.absolutePath)
projector.close()  // Close immediately

// 4. Extract OCR text for grounding
val ocrEngine = MlKitOcrEngine(context)
val ocrText = ocrEngine.extractText(ImageSource.FileSource(imageFile)).text

// 5. Build grounded prompt with OCR context
val prompt = """
    Context: Image contains the following OCR text:
    $ocrText
    
    Question: $userQuestion
    Answer using ONLY the image and OCR context. Be concise.
""".trimIndent()

// 6. Run vision analysis
val adapter = SmolLMVisionAdapter(context, smol)
adapter.loadVisionModel(modelPath)
val result = adapter.analyze(ImageSource.FileSource(embeddingsFile), prompt)

// Cleanup
ocrEngine.close()
adapter.close()
smol.close()
```

**Local description fallback** (no model required):

```kotlin
// Fast OCR + basic analysis without loading LLM
val desc = LocalImageDescriber.describe(context, ImageSource.FileSource(imageFile))
println("Summary: ${desc.summary}")
println("Labels: ${desc.labels.joinToString(", ")}")
```

**Important notes:**

- Always preprocess images: downsample to ≤1600px and correct EXIF orientation
- Include OCR text in prompts to reduce vision model hallucination
- Close projector immediately after encoding (before inference)
- mmproj file may not exist for all model repos (check availability)
- Vision support is experimental; OCR-only mode recommended for production

**Status**: Architecture prepared, full integration pending llama.cpp multimodal support for Android.

---

## Additional Resources

- [Architecture](architecture.md) for flow diagrams (RAG, OCR, JNI loading)
- [Usage](usage.md) for API reference and concepts
- [Quirks](quirks.md) for troubleshooting specific issues

All example activities are intentionally minimal and well-documented. Copy and adapt them into your own app!