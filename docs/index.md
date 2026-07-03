# llmedge

llmedge is a lightweight toolkit for running LLM inference, vision models, and multimodal utilities on-device (Android/native). It bundles JNI/C++ inference bindings powered by llama.cpp and stable-diffusion.cpp, Kotlin APIs for Android, and comprehensive example applications.

## Highlights

**Core Features:**

- Native C++ inference via [llama.cpp](https://github.com/ggerganov/llama.cpp) (GGUF model support)
- On-device safetensors → GGUF conversion (Llama arch + GPT2-BPE) with optional quantization, plus low-end `ModelPresets` (BitNet b1.58 2B4T, SmolVLM2-256M)
- Kotlin API for Android with coroutines and Flow support
- Automatic CPU feature detection (FP16, dotprod, i8mm)
- Optional Android GPU acceleration with experimental OpenCL preferred first, Vulkan fallback second, and CPU fallback last
- Memory-aware context size capping
- **Optimized Inference**: KV cache reuse for compact multi-turn chats, plus `ChatSession` for bounded Kotlin-managed replay when reasoning traces would otherwise exhaust context.

**Generative AI Capabilities:**

- **Image Generation**: Stable Diffusion integration for on-device image generation with:

    - **EasyCache**: Automatically enabled by `edge.image` for supported DiT models (Flux, SD3, Wan, Qwen Image, Z-Image) to accelerate generation.
    - **LoRA Support**: Apply Low-Rank Adaptation models (e.g., for style transfer) with automatic downloading from Hugging Face.
    - **FLUX.2 Klein 4B**: Distilled diffusion transformer (the architecture behind PrismML's binary/ternary Bonsai Image) via the `Flux2Klein` helper — loads the DiT + Qwen3-4B encoder + VAE as a split model in ~4 steps.

- **Video Generation**: Generate short video clips (4-64 frames) from text using Wan models with sequential loading for lower RAM usage.

**Speech Capabilities:**

- **Speech-to-Text (STT)**: Whisper.cpp integration for audio transcription with:

    - Timestamp support for subtitles
    - Language detection
    - SRT subtitle generation
    - Real-time streaming transcription
    - Works well on mobile with tiny/base models

- **Text-to-Speech (TTS)**: Bark.cpp integration for neural speech synthesis

    - High-quality voice generation

**Multimodal Capabilities:**

- OCR: Google ML Kit Text Recognition integration
- Image processing utilities with orientation handling
- Vision model interfaces (prepared for LLaVA-style models)

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
- Managed chat sessions with sliding-window history replay and `<think>` stripping
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
