# Video Generation API Documentation

Complete reference for on-device video generation using Wan models in llmedge.

## Table of Contents

- [Overview](#overview)
- [Supported Models](#supported-models)
- [API Reference](#api-reference)
- [Parameter Guide](#parameter-guide)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Performance Optimization](#performance-optimization)

---

## Overview

llmedge provides on-device video generation through the `StableDiffusion` class, using Wan models. Generate short video clips (4-64 frames) entirely on Android devices.

**⚠️ Hardware Requirements**:

- **Minimum RAM**: 12GB recommended for Wan 2.1 T2V-1.3B
- **Supported Devices**: Galaxy S23 Ultra (12GB), Pixel 8 Pro (12GB), OnePlus 12 (16GB+)
- **Not Supported**: 8GB RAM devices (Galaxy S22, Pixel 7)

**Why 12GB?** Wan models require loading three components simultaneously:

1. Main diffusion model (fp16): ~2.7GB RAM
2. T5XXL text encoder (Q3_K_S GGUF): ~5.9GB RAM
3. VAE decoder (fp16): ~0.14GB RAM
4. Working memory: ~1GB RAM
**Total**: ~9.7GB minimum, 12GB recommended

**Key Features**:

- Text-to-video (T2V) generation
- Multi-file model loading (main + VAE + T5XXL)
- Memory-aware device compatibility checks
- Progress monitoring and cancellation
- Multiple scheduler options

---

## Supported Models

### Official Model Source (Recommended)

**Wan 2.1 T2V 1.3B from Comfy-Org/Wan_2.1_ComfyUI_repackaged:**

All three components are required and must be explicitly downloaded:

1. **Main Model**: `wan2.1_t2v_1.3B_fp16.safetensors` (~2.6GB file, 2.7GB RAM)
2. **VAE**: `wan_2.1_vae.safetensors` (~160MB file, 0.14GB RAM)
3. **T5XXL Encoder**: `umt5-xxl-encoder-Q3_K_S.gguf` from `city96/umt5-xxl-encoder-gguf` (~2.86GB file, 5.9GB RAM)

**Device Requirements**: 

- **RAM**: 12GB+ (9.7GB minimum + overhead)
- **Storage**: 6GB free space for downloads
- **OS**: Android 11+ recommended (Vulkan acceleration)

**Known Limitations**:

- GGUF quantization of main model blocked by metadata issues
- Sequential loading not supported - all three models load simultaneously
- No disk streaming - models must fit in RAM
- 8GB RAM devices cannot run Wan models (architectural constraint)

---

## API Reference

### Loading Models

#### `StableDiffusion.load()`

Load a video generation model with explicit paths to all three required components.

**⚠️ Important**: The simplified `modelId` + `filename` approach does not work for Wan models. You must explicitly download and provide paths to all three files.

```kotlin
suspend fun load(
    context: Context,
    modelPath: String,
    vaePath: String?,
    t5xxlPath: String?,
    nThreads: Int = Runtime.getRuntime().availableProcessors(),
    offloadToCpu: Boolean = true,
    keepClipOnCpu: Boolean = true,
    keepVaeOnCpu: Boolean = true
): StableDiffusion
```

**Parameters**:

- `context`: Android application context
- `modelPath`: Absolute path to main model file (safetensors)
- `vaePath`: Absolute path to VAE file (safetensors)
- `t5xxlPath`: Absolute path to T5XXL encoder (GGUF)
- `nThreads`: Number of CPU threads (default: all cores)
- `offloadToCpu`: Enable CPU offloading (default: true, recommended)
- `keepClipOnCpu`: Keep CLIP model on CPU (default: true, recommended)
- `keepVaeOnCpu`: Keep VAE on CPU (default: true, recommended)

**Returns**: `StableDiffusion` instance ready for generation

**Throws**:

- `FileNotFoundException`: Model file not found
- `IllegalStateException`: Model loading failed (e.g., insufficient RAM)
- `UnsupportedOperationException`: 14B model rejected (mobile unsupported)

**Example**:

```kotlin
// Download all three model files explicitly
val modelFile = HuggingFaceHub.ensureRepoFileOnDisk(
    context = this,
    modelId = "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
    revision = "main",
    filename = "wan2.1_t2v_1.3B_fp16.safetensors",
    allowedExtensions = listOf(".safetensors"),
    preferSystemDownloader = true
)

val vaeFile = HuggingFaceHub.ensureRepoFileOnDisk(
    context = this,
    modelId = "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
    revision = "main",
    filename = "wan_2.1_vae.safetensors",
    allowedExtensions = listOf(".safetensors"),
    preferSystemDownloader = true
)

val t5xxlFile = HuggingFaceHub.ensureRepoFileOnDisk(
    context = this,
    modelId = "city96/umt5-xxl-encoder-gguf",
    revision = "main",
    filename = "umt5-xxl-encoder-Q3_K_S.gguf",
    allowedExtensions = listOf(".gguf"),
    preferSystemDownloader = true
)

// Load all three models together
val sd = StableDiffusion.load(
    context = this,
    modelPath = modelFile.file.absolutePath,
    vaePath = vaeFile.file.absolutePath,
    t5xxlPath = t5xxlFile.file.absolutePath,
    nThreads = Runtime.getRuntime().availableProcessors(),
    offloadToCpu = true,
    keepClipOnCpu = true,
    keepVaeOnCpu = true
)
```

---

### Video Generation

#### `txt2vid()`

Generate video from text prompt and optional initial image.

```kotlin
suspend fun txt2vid(params: VideoGenerateParams): List<Bitmap>
```

**Parameters**:

- `params`: `VideoGenerateParams` object (see [Parameter Guide](#parameter-guide))

**Returns**: `List<Bitmap>` - Generated video frames

**Throws**:

- `IllegalStateException`: Model not loaded or not a video model
- `IllegalArgumentException`: Invalid parameters (dimensions, frame count, etc.)
- `CancellationException`: Generation cancelled via `cancelGeneration()`

**Example**:

```kotlin
val params = StableDiffusion.VideoGenerateParams(
    prompt = "a cat walking in a garden, high quality",
    videoFrames = 16,
    width = 512,
    height = 512,
    steps = 20,
    cfgScale = 7.0,
    seed = 42
)

val frames = sd.txt2vid(params)
```

---

### VideoGenerateParams

Data class for video generation parameters.

```kotlin
data class VideoGenerateParams(
    val prompt: String,
    val videoFrames: Int = 16,
    val width: Int = 512,
    val height: Int = 512,
    val steps: Int = 20,
    val cfgScale: Double = 7.0,
    val seed: Long = -1,
    val scheduler: Scheduler = Scheduler.EULER_A,
    val strength: Float = 0.8f,
    val initImage: Bitmap? = null
)
```

**Field Validation**:

- `prompt`: Non-empty string
- `videoFrames`: 4-64 (capped to 32 for 5B models)
- `width`: 256-960 (multiple of 64)
- `height`: 256-960 (multiple of 64)
- `steps`: 10-50
- `cfgScale`: 1.0-15.0
- `seed`: Any long (-1 for random)
- `scheduler`: EULER_A, DDIM, DDPM, or LCM
- `strength`: 0.0-1.0 (for I2V/TI2V)
- `initImage`: Optional bitmap for I2V/TI2V

See [Parameter Guide](#parameter-guide) for detailed explanations.

---

### Model Introspection

#### `isVideoModel()`

Check if loaded model is a video generation model.

```kotlin
fun isVideoModel(): Boolean
```

**Returns**: `true` if model supports video generation, `false` otherwise

---

#### `getVideoModelMetadata()`

Get metadata about the loaded video model.

```kotlin
fun getVideoModelMetadata(): VideoModelMetadata?
```

**Returns**: `VideoModelMetadata` object or `null` if not a video model

**VideoModelMetadata fields**:

- `architecture`: Model architecture (e.g., "wan")
- `modelType`: "t2v", "i2v", or "ti2v"
- `parameterCount`: "1.3B", "5B", or "14B"
- `mobileSupported`: Boolean (false for 14B models)
- `tags`: Set of model tags
- `filename`: GGUF filename

---

### Progress Monitoring

#### `setProgressCallback()`

Set callback for generation progress updates.

```kotlin
fun setProgressCallback(callback: VideoProgressCallback?)
```

**VideoProgressCallback**:

```kotlin
fun interface VideoProgressCallback {
    fun onProgress(step: Int, totalSteps: Int)
}
```

**Example**:

```kotlin
sd.setProgressCallback { step, totalSteps ->
    val progress = (step.toFloat() / totalSteps * 100).toInt()
    runOnUiThread {
        progressBar.progress = progress
        statusText.text = "Step $step / $totalSteps"
    }
}
```

---

#### `cancelGeneration()`

Cancel ongoing video generation.

```kotlin
fun cancelGeneration()
```

Cancellation is cooperative - the native layer checks the flag periodically. Generation will stop within 1-2 seconds.

**Example**:

```kotlin
cancelButton.setOnClickListener {
    sd.cancelGeneration()
}
```

---

### Resource Management

#### `close()`

Free native resources and reset model state.

```kotlin
fun close()
```

**Important**: Always call `close()` when done with the model to prevent memory leaks. Use Kotlin's `use` block for automatic cleanup:

```kotlin
StableDiffusion.load(context, modelId, filename).use { sd ->
    val frames = sd.txt2vid(params)
    // sd.close() called automatically
}
```

---

## Parameter Guide

### Core Parameters

#### `prompt`

Text description of the video to generate.

**Best Practices**:

- Be specific and descriptive
- Include quality modifiers: "high quality", "detailed", "cinematic"
- Avoid negations (use positive descriptions)
- Keep under 200 characters for best results

**Examples**:
```kotlin
// Good
"a serene ocean sunset, waves gently rolling, golden hour lighting, cinematic"

// Better
"a professional chef preparing pasta, kitchen environment, natural lighting, detailed hands"

// Avoid
"a cat, not blurry" // Negations don't work well
```

---

#### `videoFrames`

Number of frames to generate (4-64).

**Guidelines**:

- **4-8 frames**: Quick tests, ~5-15 seconds generation
- **16 frames**: Standard short clips, ~20-45 seconds generation
- **32 frames**: Longer animations, ~40-90 seconds generation
- **64 frames**: Maximum quality (1.3B models only), ~80-180 seconds

**Memory Impact**:

- 1.3B models: Up to 64 frames
- 5B models: Automatically capped at 32 frames

```kotlin
// Quick test
videoFrames = 8

// Standard production
videoFrames = 16

// High quality (1.3B only)
videoFrames = 64
```

---

#### `width` and `height`

Output resolution (256-960, must be multiples of 64).

**Common Resolutions**:

- **256x256**: Fastest, lowest quality
- **512x512**: Balanced (recommended)
- **768x768**: High quality, slower
- **960x960**: Maximum quality, very slow

**Performance vs Quality**:
```kotlin
// Fast generation (~2 sec/frame on mid-range)
width = 256, height = 256

// Balanced (recommended)
width = 512, height = 512

// High quality (~8 sec/frame on mid-range)
width = 768, height = 768
```

---

#### `steps`

Number of diffusion steps (10-50).

**Guidelines**:

- **10-15 steps**: Fast, lower quality
- **20 steps**: Recommended default
- **25-30 steps**: Higher quality
- **40-50 steps**: Maximum quality, diminishing returns

```kotlin
// Fast generation
steps = 15

// Production quality
steps = 20

// Maximum quality
steps = 30
```

---

#### `cfgScale`

Classifier-free guidance scale (1.0-15.0). Controls adherence to prompt.

**Guidelines**:

- **1.0-3.0**: Very creative, less prompt adherence
- **7.0**: Default, balanced
- **10.0-12.0**: Strong prompt adherence
- **13.0-15.0**: Very strict, may over-saturate

```kotlin
// Creative freedom
cfgScale = 3.0

// Standard (recommended)
cfgScale = 7.0

// Strict prompt following
cfgScale = 10.0
```

---

#### `seed`

Random seed for reproducibility.

**Guidelines**:

- **-1**: Random seed (different output each time)
- **0+**: Fixed seed (reproducible outputs)

```kotlin
// Random generation
seed = -1

// Reproducible generation
seed = 42

// Generate variations
val seeds = listOf(42, 43, 44, 45)
seeds.forEach { seed ->
    val frames = sd.txt2vid(params.copy(seed = seed))
}
```

---

### Advanced Parameters

#### `scheduler`

Diffusion scheduler algorithm.

**Options**:

- `Scheduler.EULER_A`: Default, good quality and speed
- `Scheduler.DDIM`: More deterministic, slightly slower
- `Scheduler.DDPM`: Higher quality, slower
- `Scheduler.LCM`: Fast inference (requires LCM-fine-tuned model)

```kotlin
// Default
scheduler = Scheduler.EULER_A

// Deterministic
scheduler = Scheduler.DDIM

// Quality-focused
scheduler = Scheduler.DDPM
```

---

#### `strength`

Denoising strength for image-to-video (0.0-1.0).

Only used with `initImage` for I2V/TI2V models.

**Guidelines**:

- **0.0-0.3**: Subtle animation, preserves image
- **0.5-0.7**: Moderate animation
- **0.8-1.0**: Strong transformation

```kotlin
val params = VideoGenerateParams(
    prompt = "animate this scene, add motion",
    initImage = initialFrame,
    strength = 0.7f,  // Moderate animation
    videoFrames = 16
)
```

---

#### `initImage`

Initial frame for image-to-video generation (I2V/TI2V models only).

```kotlin
val initialFrame = BitmapFactory.decodeResource(resources, R.drawable.scene)

val params = VideoGenerateParams(
    prompt = "animate this image, add wind and motion",
    initImage = initialFrame,
    strength = 0.8f,
    videoFrames = 16,
    width = initialFrame.width,
    height = initialFrame.height
)
```

**Note**: Image will be resized to match `width` and `height` if needed.

---

## Advanced Usage

### Device-Aware Model Selection

Check device RAM before attempting to load:

```kotlin
fun checkDeviceCompatibility(context: Context): Boolean {
    val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    val memInfo = ActivityManager.MemoryInfo()
    activityManager.getMemoryInfo(memInfo)
    val totalRamGB = memInfo.totalMem / (1024.0 * 1024.0 * 1024.0)
    
    if (totalRamGB < 12.0) {
        Log.w("VideoGen", "Insufficient RAM: ${String.format("%.1f", totalRamGB)}GB (12GB required)")
        return false
    }
    
    return true
}

// Usage
if (!checkDeviceCompatibility(this)) {
    showErrorDialog(
        "Video generation requires 12GB+ RAM. " +
        "This device has ${String.format("%.1f", totalRamGB)}GB. " +
        "Consider using cloud inference APIs instead."
    )
    return
}

// Proceed with model loading
```

---

### Batch Generation

Generate multiple variations efficiently:

```kotlin
val baseParams = StableDiffusion.VideoGenerateParams(
    prompt = "a cat walking",
    videoFrames = 16,
    width = 512,
    height = 512,
    steps = 20
)

val variations = (0..4).map { i ->
    async(Dispatchers.IO) {
        sd.txt2vid(baseParams.copy(seed = 42 + i))
    }
}.awaitAll()

// variations now contains 5 different video sequences
```

---

### Streaming to Video File

Save frames directly to MP4 using MediaCodec:

```kotlin
fun saveFramesToVideo(frames: List<Bitmap>, outputPath: String, fps: Int = 8) {
    val mediaCodec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
    val mediaFormat = MediaFormat.createVideoFormat(
        MediaFormat.MIMETYPE_VIDEO_AVC,
        frames.first().width,
        frames.first().height
    ).apply {
        setInteger(MediaFormat.KEY_BIT_RATE, 2000000)
        setInteger(MediaFormat.KEY_FRAME_RATE, fps)
        setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
        setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
    }
    
    // Configure MediaMuxer and encode frames
    // See Android MediaCodec documentation for full implementation
}
```

---

### Model Switching

Switch between models efficiently:

```kotlin
// Load first model
var sd = StableDiffusion.load(this, modelId1, filename1)
val frames1 = sd.txt2vid(params)
sd.close()

// Switch to second model
sd = StableDiffusion.load(this, modelId2, filename2)
val frames2 = sd.txt2vid(params)
sd.close()
```

**Note**: Metadata caching reduces GGUF parsing overhead on subsequent loads.

---

## Troubleshooting

### Common Issues

#### OutOfMemoryError

**Symptoms**: App crashes during generation with OOM error

**Solutions**:

1. Reduce resolution: `width = 256, height = 256`
2. Reduce frame count: `videoFrames = 8`
3. Reduce steps: `steps = 15`
4. Use smaller quantization: Q3_K_S instead of Q4_K_M
5. Close other apps to free RAM
6. Enable CPU offloading: `offloadToCpu = true`

```kotlin
// Memory-constrained configuration
val params = VideoGenerateParams(
    prompt = "...",
    videoFrames = 8,
    width = 256,
    height = 256,
    steps = 15,
    cfgScale = 7.0
)
```

---

#### Slow Generation

**Symptoms**: Generation takes >5 seconds per frame

**Solutions**:

1. Use 1.3B model instead of 5B
2. Reduce resolution
3. Reduce steps (15-20 is usually sufficient)
4. Enable Vulkan if on Android 11+
5. Close background apps

```kotlin
// Fast generation configuration
val params = VideoGenerateParams(
    prompt = "...",
    videoFrames = 16,
    width = 512,
    height = 512,
    steps = 15,  // Reduced from 20
    cfgScale = 7.0
)
```

---

#### Model Not Loading

**Symptoms**: `FileNotFoundException` or load failure

**Solutions**:

1. Verify model file exists: `File(context.filesDir, "hf-models/$modelId/$filename").exists()`
2. Check internet connection for downloads
3. Verify Hugging Face model ID is correct
4. Check storage space (5B models need ~5GB free)

```kotlin
// Debug model loading
try {
    val sd = StableDiffusion.load(this, modelId, filename)
    Log.d("VideoGen", "Model loaded: ${sd.getVideoModelMetadata()}")
} catch (e: Exception) {
    Log.e("VideoGen", "Load failed", e)
    // Handle error
}
```

---

#### Poor Quality Output

**Symptoms**: Blurry, artifact-heavy, or incoherent frames

**Solutions**:

1. Increase steps: 20-30
2. Increase resolution: 512x512 or higher
3. Adjust cfgScale: Try 7.0-10.0
4. Use Q4_K_M or higher quantization
5. Improve prompt specificity
6. Try different schedulers (DDPM for quality)

```kotlin
// Quality-focused configuration
val params = VideoGenerateParams(
    prompt = "detailed, high quality, cinematic scene...",
    videoFrames = 16,
    width = 768,
    height = 768,
    steps = 25,
    cfgScale = 8.0,
    scheduler = Scheduler.DDPM
)
```

---

#### Generation Hangs

**Symptoms**: Progress stops, app becomes unresponsive

**Solutions**:

1. Ensure generation runs on `Dispatchers.IO`
2. Set progress callback to monitor
3. Implement timeout mechanism
4. Call `cancelGeneration()` if needed

```kotlin
val job = CoroutineScope(Dispatchers.IO).launch {
    try {
        withTimeout(300_000) { // 5 minute timeout
            val frames = sd.txt2vid(params)
            // Process frames
        }
    } catch (e: TimeoutCancellationException) {
        sd.cancelGeneration()
        Log.e("VideoGen", "Generation timed out")
    }
}
```

---

## Performance Optimization

### Memory Management

**Monitor memory usage**:

```kotlin
val runtime = Runtime.getRuntime()
val usedMemoryMB = (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024)
Log.d("VideoGen", "Memory usage: ${usedMemoryMB}MB")
```

**Memory-efficient generation**:

```kotlin
// Process frames immediately instead of accumulating
sd.setProgressCallback { step, total ->
    // ... update UI
}

val frames = sd.txt2vid(params)
frames.forEachIndexed { index, frame ->
    saveFrameToDisk(frame, index)
    frame.recycle() // Free bitmap immediately
}
```

---

### Batch Processing

**Generate multiple videos efficiently**:

```kotlin
// Reuse model instance
val sd = StableDiffusion.load(this, modelId, filename)

prompts.forEach { prompt ->
    val frames = sd.txt2vid(params.copy(prompt = prompt))
    processFrames(frames)
}

sd.close()
```

---

### Background Processing

**Use WorkManager for long generations**:

```kotlin
class VideoGenerationWorker(context: Context, params: WorkerParameters) 
    : CoroutineWorker(context, params) {
    
    override suspend fun doWork(): Result {
        val sd = StableDiffusion.load(applicationContext, modelId, filename)
        val frames = sd.txt2vid(params)
        saveVideo(frames)
        sd.close()
        return Result.success()
    }
}
```

---

### Vulkan Acceleration

**Enable Vulkan on Android 11+**:

Build library with Vulkan support:

```bash
./gradlew :llmedge:assembleRelease -Pandroid.jniCmakeArgs="-DGGML_VULKAN=ON -DSD_VULKAN=ON"
```

**Verify Vulkan at runtime**:

```kotlin
// Vulkan status is logged during initialization
// Check logcat for: "Vulkan initialized successfully"
```

---

## See Also

- [architecture.md](./architecture.md) - System design