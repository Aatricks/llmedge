# FAQ

/// details | Q: Which models work with `llmedge`?
A: Any GGUF model supported by llama.cpp should work. Prefer quantized models (Q4_K_M, Q5_K_M, Q8_0) for on-device use. Models like SmolLM, TinyLlama, Qwen, Phi, and Gemma work well. Check model size against your device's available RAM.
///

/// details | Q: Can I run this on iOS or desktop?
A: The core native code uses llama.cpp (portable C++) but the Kotlin APIs target Android specifically. iOS ports would require Swift/Objective-C bindings. Desktop Java/Kotlin apps could potentially use JNI, but this hasn't been tested.
///

/// details | Q: How do I reduce memory usage?
A: Multiple strategies:

- Use smaller, quantized models (Q4_K_M instead of Q8_0 or FP16)
- Reduce `contextSize` in `InferenceParams` (e.g., 2048 instead of 8192)
- Lower `numThreads` parameter
- Call `SmolLM.close()` when done to free native memory
- For Stable Diffusion, enable `offloadToCpu` and reduce image dimensions
///

/// details | Q: Why is inference slow on my device?
A: Several factors affect speed:

- Mobile CPUs are resource-constrained compared to desktop GPUs
- Use quantized models (Q4_K_M is faster than FP16)
- Ensure you're on arm64-v8a architecture (check with `Build.SUPPORTED_ABIS[0]`)
- The library automatically selects optimized native libs based on CPU features
- Lower `temperature` and `maxTokens` can speed up generation
- Vulkan acceleration may help on supported devices
///

/// details | Q: How do I enable Vulkan acceleration?
A: Create SmolLM with `SmolLM(useVulkan = true)` (default). Check if enabled with `smol.isVulkanEnabled()`. Your device needs Android 11+ and Vulkan 1.2 support. See the Building section in the main README for build configuration.
///

/// details | Q: How do vision models work?
A: The library has interfaces for vision-capable LLMs (`VisionModelAnalyzer`), but full vision support is pending llama.cpp's multimodal integration for Android. Currently, use OCR (`MlKitOcrEngine`) for text extraction from images. The architecture is prepared for LLaVA-style models when available.
///

/// details | Q: Can I use custom embedding models for RAG?
A: Yes. Configure `EmbeddingConfig` when creating `RAGEngine`. Place your ONNX model and tokenizer in assets. The library uses sentence-embeddings library which supports various models from Hugging Face.
///

/// details | Q: How do I contribute?
A: See the [Contributing](contributing.md) page for development and PR guidelines.
///

If your question isn't answered, please [open an issue](https://github.com/Aatricks/llmedge/issues) with logs and steps to reproduce.