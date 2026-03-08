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
| `TTSActivity` | Text-to-speech synthesis (Bark) |
| `STTActivity` | Speech-to-text transcription (Whisper) |

Also see the [ChatSession pattern](#chatsession-pattern) below for bounded multi-turn chat UIs backed by `LLMEdge`.

---


## LocalAssetDemoActivity

Demonstrates loading a bundled model asset and running inference using `LLMEdge`.

**Key patterns:**

```kotlin
val modelPath = copyAssetIfNeeded("YourModel.gguf")
val edge = LLMEdge.create(context, lifecycleScope)

// Run inference through the text client
val response = edge.text.generate(
    prompt = "Say 'hello from llmedge'.",
    model = ModelSpec.localFile(modelPath),
)
```

## ChatSession pattern

Use this pattern when your app owns a chat UI and you want bounded, Kotlin-managed transcript replay
instead of relying on unbounded native chat history:

```kotlin
val edge = LLMEdge.create(context, lifecycleScope)

val session =
    edge.text.session(
        model = ModelSpec.localFile(modelPath),
        memory = ConversationWindow(maxTurns = 6, maxTokens = 4096, stripThinkTags = true),
        systemPrompt = "You are a concise assistant.",
    )

session.prepare()

// Blocking reply
val reply = session.reply("Explain why context windows fill up.")

// Streaming reply
session.stream("Now summarize that in 3 bullets.").collect { event ->
    if (event is TextStreamEvent.Chunk) {
        appendToChatUi(event.value)
    }
}
```

**Why use it:**

- Keeps the full transcript in Kotlin memory
- Replays only the active sliding window back into the model
- Strips older `<think>...</think>` traces before replay when enabled
- Works well for reasoning-enabled models that would otherwise exhaust the native context quickly



---

## ImageToTextActivity

Demonstrates camera integration and OCR text extraction using `LLMEdge`.

**Key features:**

- Camera permission handling
- High-resolution image capture
- ML Kit OCR integration via `edge.vision`

**Code snippet:**

```kotlin
// Process image
val bitmap = ImageUtils.fileToBitmap(file)
ivPreview.setImageBitmap(bitmap)

val edge = LLMEdge.create(context, lifecycleScope)

// Extract text
val text = edge.vision.extractText(bitmap)
tvResult.text = text.ifEmpty { "(no text detected)" }
```

---

## HuggingFaceDemoActivity

Shows how to download models from Hugging Face Hub and run inference using `LLMEdge`.

**Key features:**

- Model download with progress callback
- Heap-aware parameter selection
- Thinking mode configuration

**Code snippet:**

```kotlin
val edge = LLMEdge.create(context, lifecycleScope)

// 1. Download with progress
edge.models.prefetch(
    ModelSpec.huggingFace(
        repoId = "unsloth/Qwen3-0.6B-GGUF",
        filename = "Qwen3-0.6B-Q4_K_M.gguf",
    )
)

// 2. Generate text (model is now cached)
val response = edge.text.generate(
    prompt = "List two quick facts about running GGUF models on Android.",
    model = ModelSpec.huggingFace(
        repoId = "unsloth/Qwen3-0.6B-GGUF",
        filename = "Qwen3-0.6B-Q4_K_M.gguf",
    ),
    params = SmolLM.InferenceParams(thinkingMode = SmolLM.ThinkingMode.DISABLED),
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
val edge = LLMEdge.create(context, lifecycleScope)
val rag = edge.rag.createSession()
rag.init()

// 1. Index a PDF document
val count = rag.indexPdf(pdfUri)
Log.d("RAG", "Indexed $count chunks")

// 2. Ask questions
val answer = rag.ask("What are the key points?", topK = 5)
```

**Important notes:**

- Scanned PDFs require OCR before indexing (PDFBox extracts text-based only)
- Embedding model must be in `assets/` directory
- Vector store persists to JSON in app files directory
- Adjust chunk size/overlap based on document type

---

## StableDiffusionActivity

Demonstrates on-device image generation using `LLMEdge`.

**Key patterns:**

```kotlin
val edge = LLMEdge.create(context, lifecycleScope)

// Generate image
val bitmap = edge.image.generate(
    ImageGenerationRequest(
        prompt = "a cute cat",
        width = 256, // Start small on mobile
        height = 256,
        steps = 20,
        cfgScale = 7.0f
    ),
)

// Display result
imageView.setImageBitmap(bitmap)
```

**Important notes:**

- Start with 128×128 or 256x256 on devices with <4GB RAM
- `edge.image` automatically enables CPU offloading if memory is tight
- Generating images is memory-intensive; close other apps for best results

---

## VideoGenerationActivity

Demonstrates on-device text-to-video generation using Wan 2.1.

**Key patterns:**

```kotlin
val edge = LLMEdge.create(context, lifecycleScope)

// 1. Configure params for mobile-friendly generation
val params = VideoGenerationRequest(
    prompt = "A dog running in the park",
    width = 512,      // Wan models require multiples of 64 between 256-960
    height = 512,
    videoFrames = 8,  // Start with fewer frames (8-16)
    steps = 20,
    cfgScale = 7.0f,
    flowShift = 3.0f, // Use Float.POSITIVE_INFINITY to auto-select
    forceSequentialLoad = true // Critical for devices with <12GB RAM
)

// 2. Generate video
edge.image.generateVideo(params).collect { event ->
    when (event) {
        is GenerationStreamEvent.Progress ->
            updateProgress(event.update.message)
        is GenerationStreamEvent.Completed ->
            imageView.setImageBitmap(event.frames.first())
    }
}
```

**Important notes:**

- **Sequential Load:** Video models are large (Wan 2.1 is ~5GB). `forceSequentialLoad = true` is essential for mobile devices; it loads components (T5 encoder, Diffusion model, VAE) one by one to keep peak memory low.
- **Frame Count:** Generating 8-16 frames takes significant time. Provide progress feedback.
- **LLMEdge:** This activity uses the high-level `LLMEdge` facade, which handles the sequential loading logic automatically.

---

## LlavaVisionActivity

Demonstrates vision-capable LLM integration using `LLMEdge`.

**Key patterns:**

```kotlin
val edge = LLMEdge.create(context, lifecycleScope)

// 1. Preprocess image
val bmp = ImageUtils.imageToBitmap(context, uri)
val scaledBmp = ImageUtils.preprocessImage(bmp, correctOrientation = true, maxDimension = 1024)

// 2. Run OCR (Optional grounding)
val ocrText = edge.vision.extractText(scaledBmp)

// 3. Build Prompt (e.g. ChatML format for Phi-3)
val prompt = """
    <|system|>You are a helpful assistant.<|end|>
    <|user|>
    Context (OCR): $ocrText

    Describe this image.<|end|>
    <|assistant|>
""".trimIndent()

// 4. Run Vision Analysis
val result = edge.vision.analyze(scaledBmp, prompt)
```

**Status**: Uses `LLMEdge` to orchestrate the experimental vision pipeline (loading projector, encoding image, running inference).

---

## TTSActivity

Demonstrates text-to-speech synthesis using Bark via `LLMEdge`.

**Key features:**

- Text input for speech synthesis
- Automatic model download from Hugging Face (843MB)
- Progress tracking during generation (semantic, coarse, fine encoding)
- Audio playback of generated speech
- WAV file saving

**Key patterns:**

```kotlin
val edge = LLMEdge.create(context, lifecycleScope)

// Generate speech (model auto-downloads on first use)
val audio = edge.speech.synthesize("Hello, world!")

// Play the generated audio
val audioTrack = AudioTrack.Builder()
    .setAudioAttributes(AudioAttributes.Builder()
        .setUsage(AudioAttributes.USAGE_MEDIA)
        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
        .build())
    .setAudioFormat(AudioFormat.Builder()
        .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
        .setSampleRate(audio.sampleRate)
        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
        .build())
    .setBufferSizeInBytes(audio.samples.size * 4)
    .build()

audioTrack.write(audio.samples, 0, audio.samples.size, AudioTrack.WRITE_BLOCKING)
audioTrack.play()
```

---

## Additional Resources

- [Architecture](architecture.md) for flow diagrams (RAG, OCR, JNI loading)
- [Usage](usage.md) for API reference and concepts
- [Quirks](quirks.md) for troubleshooting specific issues
