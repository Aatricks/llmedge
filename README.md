# llmedge
Library for using GGUF models on Android devices, powered by llama.cpp.


- Remote example repo (optional): [llmedge-examples](https://github.com/Aatricks/llmedge-examples)

See `CREDITS.md` for acknowledgments to the original author Shubham Panchal and upstream projects.

## Setup
### On-device RAG

This repo includes a minimal on-device RAG pipeline inside the `llmedge` Android library module. It mirrors the flow from Android-Doc-QA using:

- `sentence-embeddings` (ONNX) for embeddings
- A simple whitespace `TextSplitter`
- An in-memory cosine `VectorStore` with JSON persistence
- `SmolLM` (llama.cpp JNI) for answering with retrieved context

Setup:

1) Place embedding assets:
   - `llmedge/src/main/assets/embeddings/all-minilm-l6-v2/model.onnx`
   - `llmedge/src/main/assets/embeddings/all-minilm-l6-v2/tokenizer.json`

   You can download from the HF `sentence-transformers/all-MiniLM-L6-v2` repo.

2) Build the library:

```powershell
.\gradlew :llmedge:assembleRelease
```

3) Usage from an app module (Kotlin):

```kotlin
import io.aatricks.llmedge.SmolLM
import io.aatricks.llmedge.rag.RAGEngine

val smol = SmolLM()
// smol.load("/path/to/model.gguf", SmolLM.InferenceParams()) // ensure your GGUF is available

val rag = RAGEngine(context = this, smolLM = smol)
CoroutineScope(Dispatchers.IO).launch {
   rag.init()
   // Index a PDF from SAF Uri
   val count = rag.indexPdf(pdfUri)
   val answer = rag.ask("What are the key points?")
   withContext(Dispatchers.Main) { /* render answer */ }
}
```

Notes:
- For PDF parsing, we use `com.tom-roush:pdfbox-android`.
- Embeddings artifact: `io.gitlab.shubham0204:sentence-embeddings:v6`.
- If you prefer the full Android-Doc-QA app, see the original repo for UI and ObjectBox persistence.

Troubleshooting:
- Empty or irrelevant answers: If the retrieved context is empty, your PDF may be a scanned/image-only PDF. This demo does not perform OCR. Use a text-based PDF or add OCR (e.g., ML Kit Text Recognition or Tesseract) before indexing.
- ONNX error "Missing Input: token_type_ids": The library auto-falls back to enable `token_type_ids` and `last_hidden_state` when needed (e.g., BGE models). For a fixed configuration, pass an `EmbeddingConfig` with `useTokenTypeIds=true`.


1. Clone the repository with its `llama.cpp` submodule:

```commandline
git clone --depth=1 https://github.com/Aatricks/llmedge
cd llmedge
git submodule update --init --recursive
```

2. Android Studio starts building the project automatically. If not, select **Build > Rebuild Project** to start a project build.

3. After a successful project build, [connect an Android device](https://developer.android.com/studio/run/device) to your system. Once connected, the name of the device must be visible in top menu-bar in Android Studio.

## Working

1. The application uses llama.cpp to load and execute GGUF models. As llama.cpp is written in pure C/C++, it is easy 
   to compile on Android-based targets using the [NDK](https://developer.android.com/ndk). 

2. The `llmedge` module uses a `LLMInference.cpp` class which interacts with llama.cpp's C-style API to execute the 
   GGUF model and a JNI binding `smollm.cpp` (native file name retained for binary compatibility). On the Kotlin side, the `SmolLM` class provides 
   the required methods to interact with the JNI (C++ side) bindings under the package `io.aatricks.llmedge`.

## Technologies

* [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) is a pure C/C++ framework to execute machine learning 
   models on multiple execution backends. It provides a primitive C-style API to interact with LLMs 
    converted to the [GGUF format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) native to [ggml](https://github.com/ggerganov/ggml)/llama.cpp. The app uses JNI bindings to interact with a small class `smollm.cpp` which uses llama.cpp to load and execute GGUF models.

## Notes

- You may need to download the Vulkan SDK and set the `VULKAN_SDK` environment variable for building the project.