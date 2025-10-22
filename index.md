# llmedge

llmedge is a lightweight toolkit for running LLM inference, vision models, and multimodal utilities on-device (Android/native). It bundles JNI/C++ inference bindings powered by llama.cpp and stable-diffusion.cpp, Kotlin APIs for Android, and comprehensive example applications.

## Highlights

**Core Features:**

- Native C++ inference via llama.cpp (GGUF model support)
- Kotlin API for Android with coroutines and Flow support
- Automatic CPU feature detection (FP16, dotprod, SVE, i8mm)
- Optional Vulkan acceleration for compatible devices
- Memory-aware context size capping

**Multimodal Capabilities:**

- OCR: Google ML Kit Text Recognition integration
- Image processing utilities with orientation handling
- Vision model interfaces (prepared for LLaVA-style models)
- Stable Diffusion integration for on-device image generation

**RAG Pipeline:**

- PDF text extraction with PDFBox
- Sentence embeddings via ONNX Runtime
- Text chunking with configurable overlap
- In-memory vector store with JSON persistence
- Context-aware question answering

**Hugging Face Integration:**

- Direct model downloads from HF Hub
- Smart quantization selection
- Private repository support with tokens
- Large file handling via Android DownloadManager
- Automatic caching and mirror resolution

**Developer Experience:**

- Comprehensive example apps demonstrating all features
- Built-in memory metrics and performance monitoring
- Reasoning control API (thinking mode)
- Streaming and blocking generation modes
- Detailed documentation and troubleshooting guides

## Quick links
- [Installation](installation.md) — Setup and build instructions
- [Usage](usage.md) — API guide and code patterns
- [Examples](examples.md) — Sample applications and snippets
- [Architecture](architecture.md) — System design and flow diagrams
- [Quirks & Troubleshooting](quirks.md) — Common issues and solutions
- [FAQ](faq.md) — Frequently asked questions
- [Contributing](contributing.md) — Development guidelines

## Getting Started

Get started by reading the [Installation](installation.md) section, then explore the [Usage](usage.md) guide for API details. Check out [llmedge-examples](https://github.com/Aatricks/llmedge-examples) for complete working applications.