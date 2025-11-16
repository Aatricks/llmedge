# Quirks & Troubleshooting

This section documents known quirks, limitations, and troubleshooting steps for `llmedge`.

### Model loading and memory

**Common issues:**

- Large models may fail to load or cause OOM (OutOfMemoryError) on devices with limited RAM
- The library automatically caps context size based on heap size, but you may need to reduce it further
- If model loading hangs, check file permissions and storage location
- Android may restrict reading assets directly via native code; use `copyAssetIfNeeded()` pattern (see examples)

**Solutions:**

- Prefer small, quantized GGUF models (Q4_K_M or Q5_K_M)
- Explicitly set lower `contextSize` in `InferenceParams` (e.g., 2048 for <512MB devices)
- Monitor memory with `MemoryMetrics.snapshot()` before and after loading
- Call `SmolLM.close()` to free resources when switching models

### Native compatibility

**Library selection:**

- The library automatically selects the best native `.so` based on CPU features (FP16, dotprod, SVE, i8mm)
- Logs show which library was loaded (e.g., `libsmollm_v8_4_fp16_dotprod.so`)
- Platform ABI mismatches can cause `UnsatisfiedLinkError`

**Common errors:**

- `UnsatisfiedLinkError`: ABI mismatch (check `Build.SUPPORTED_ABIS[0]` matches your build)
- `dlopen failed`: Missing dependencies or incompatible NDK version
- Build for arm64-v8a for modern devices; armeabi-v7a for older 32-bit devices

### Hugging Face downloads

**Download issues:**

- HF rate-limits: downloads may fail if you exceed rate limits; retry or provide a token
- For private repositories, pass `token` parameter to `loadFromHuggingFace()`
- Large files: always use `preferSystemDownloader = true` to avoid heap pressure

**Troubleshooting:**

- Check network connectivity and HF Hub status
- Verify model ID format: `owner/repo-name` (e.g., `unsloth/Qwen3-0.6B-GGUF`)
- Use `forceDownload = true` to redownload corrupted files
- Files are cached in app's files directory under `models/hub/`

### Image/Camera quirks

**Orientation issues:**

- Different devices rotate images differently based on EXIF data
- `ImageUtils` handles basic orientation correction
- Always check and normalize orientation before processing

**Memory considerations:**

- Camera images can be very large (4K+ on modern phones)
- Scale down before OCR or vision processing
- Use `BitmapFactory.Options.inSampleSize` for efficient downscaling

**OCR-specific:**

- ML Kit works offline but requires Google Play Services
- Scanned PDFs need OCR before RAG indexing (PDFBox extracts text only from text-based PDFs)
- Low-quality images may produce poor OCR results

### RAG performance

**Slow retrieval:**

- Large vector stores slow down cosine similarity search
- Consider limiting indexed chunks or implementing approximate search
- Current implementation uses in-memory brute-force search

**No results:**

- Check if PDF text extraction succeeded (`indexPdf()` returns chunk count)
- Scanned PDFs return 0 chunks (no OCR in PDFReader)
- Try `retrievalPreview()` to see what's actually being retrieved
- Adjust `TextSplitter` parameters (smaller `chunkSize` for granular retrieval)

### Stable Diffusion

**OutOfMemoryError:**

- SD models are memory-intensive
- Reduce image dimensions (start with 128x128 or 256x256)
- Enable all CPU offload flags: `offloadToCpu`, `keepClipOnCpu`, `keepVaeOnCpu`
- Use `preferSystemDownloader = true` for model downloads

**Slow generation:**

- Lower `steps` parameter (20 is reasonable, 50+ is slow)
- Reduce image resolution
- Generation speed depends heavily on device CPU

### Performance tips

**Model loading:**

- First load is slower (native memory allocation)
- Subsequent loads reuse memory pools
- Pre-load models at app start if needed immediately

**Inference:**

- Use smaller context windows when memory is constrained
- Lower `temperature` reduces randomness and can be faster
- Streaming (`getResponseAsFlow`) shows results sooner but same total time
- Monitor token/sec with `getLastGenerationMetrics()`

**Memory management:**

- Always call `.close()` on SmolLM, StableDiffusion, and OcrEngine instances
- Use `MemoryMetrics` to track native heap growth
- `nativePssKb` shows native memory (model + KV cache)
- Consider using a single global SmolLM instance instead of creating/destroying frequently

### Debugging JNI

**Getting native logs:**
```fish
adb logcat -s SmolLM:* SmolSD:* llama:*
```

**Common native errors:**

- "Failed to load model": Check file path and permissions
- "ggml_init_cublas: failed": Vulkan/GPU initialization failed (falls back to CPU)
- Crashes without logs: Use `ndk-stack` with symbolicated stack traces

**Debugging steps:**

1. Check logcat for native messages
2. Verify model file exists and is readable
3. Test with a known-good tiny model first
4. Check available memory before loading
5. Try disabling Vulkan: `SmolLM(useVulkan = false)`

**Stack traces:**
```fish
adb logcat | ndk-stack -sym path/to/obj/local/arm64-v8a/
```

If something isn't covered here, please [open an issue](https://github.com/Aatricks/llmedge/issues) with:

- Device model and Android version
- Logcat output (especially native logs)
- Model name and size
- Minimal reproducible code