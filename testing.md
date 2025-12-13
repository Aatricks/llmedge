# Testing llmedge

This guide shows how to run unit tests, instrumentation tests on emulators/devices, and the opt-in native end-to-end (E2E) image generation test.

Most tests avoid running native code by default and use a stubbed NativeBridge for speed and stability. You can optionally enable real native inference for txt2img when you have a local GGUF model.

## Prerequisites

- Android SDK and NDK installed (Android Studio or command line SDK tools)
- Gradle (wrapper included)
- Java 17+ (recommended by AGP)

## Project modules under test

- Library module: `llmedge`
- Example app: `llmedge-examples` (not required for tests)

## Unit tests (JVM)

Run the library’s JVM unit tests:

```bash
./gradlew :llmedge:testDebugUnitTest
```

Reports:

- HTML: `llmedge/build/reports/tests/testDebugUnitTest/index.html`

## Instrumentation tests (managed emulator)

We use Gradle Managed Devices to provision an emulator automatically (ATD Pixel 6, API 33). This runs all Android instrumentation tests without a plugged‑in device.

```bash
./gradlew :llmedge:pixel6api33DebugAndroidTest
```

Reports:

- HTML summary: `llmedge/build/reports/androidTests/managedDevice/debug/allDevices/index.html`
- Per‑class pages in the same folder

Notes:

- The first run may download the system image (AOSP ATD x86_64 API 33).
- Some tests are intentionally skipped on x86_64 emulators (e.g., WAN E2E which targets arm64 devices).

## Instrumentation tests (connected device)

If you have a physical or virtual device connected via ADB:

```bash
./gradlew :llmedge:connectedDebugAndroidTest
```

Run a specific test class or method:

```bash
# Single class
./gradlew :llmedge:connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class=io.aatricks.llmedge.VideoGenerationE2ETest

# Single method
./gradlew :llmedge:connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class=io.aatricks.llmedge.VideoGenerationE2ETest#testAllSchedulers
```

## Core coverage summary

What’s covered by default:

- txt2vid API layer (progress callbacks, cancellation, memory tracking, scheduler mapping, parameter validation, metadata handling)
- txt2img API layer (RGB→Bitmap conversion) using deterministic stub bytes for speed

What’s skipped by default:

- SmolLM tests (explicit skip)
- WAN video E2E (requires arm64 hardware and large assets)
- Native txt2img E2E (opt‑in; see next section)

## Opt‑in native E2E: txt2img with your model

The test `ImageGenerationE2ENativeTest` runs true native inference via JNI and validates that the output is a non‑uniform 64×64 bitmap. It is skipped unless explicitly enabled.

Enable and provide model paths either with environment variables or Gradle system properties:

Required inputs:

- GGUF model path (text‑to‑image)
- Optional VAE path (if not baked in your model)

Run with environment variables:

```bash
export LLMEDGE_T2I_MODEL_PATH=/absolute/path/to/your_model.gguf
# optional if your model needs a VAE
export LLMEDGE_T2I_VAE_PATH=/absolute/path/to/your_vae.safetensors

./gradlew :llmedge:pixel6api33DebugAndroidTest -Dllmedge.runNativeImageE2E=true
```

Run with Gradle properties only:

```bash
./gradlew :llmedge:pixel6api33DebugAndroidTest \
  -Dllmedge.runNativeImageE2E=true \
  -Dllmedge.t2i.modelPath=/absolute/path/to/your_model.gguf \
  -Dllmedge.t2i.vaePath=/absolute/path/to/your_vae.safetensors
```

Test source:

- `llmedge/src/androidTest/java/io/aatricks/llmedge/ImageGenerationE2ENativeTest.kt`

What it does:

- Calls `StableDiffusion.load` with your model and optional VAE
- Generates a 64×64 image with 10 steps (seed=42, cfgScale=7.0)
- Verifies a valid, non‑uniform bitmap is produced

## WAN video E2E (arm64 device only)

The class `WanVideoE2ETest` exercises native text‑to‑video generation with WAN models. It requires an arm64 device and downloadable assets.

Run on a connected arm64 device:

```bash
./gradlew :llmedge:connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class=io.aatricks.llmedge.WanVideoE2ETest
```

Tip: WAN assets are large. Ensure sufficient disk and network availability.

## Tips & Troubleshooting

- Accept SDK licenses if prompted:
  ```bash
  yes | "$ANDROID_HOME"/tools/bin/sdkmanager --licenses
  ```

- Emulator boot failures: try cleaning managed devices and rerun
  ```bash
  ./gradlew :llmedge:cleanManagedDevices
  ./gradlew :llmedge:pixel6api33DebugAndroidTest
  ```

- Viewing reports:
  - Unit: `llmedge/build/reports/tests/testDebugUnitTest/index.html`
  - Instrumentation (managed): `llmedge/build/reports/androidTests/managedDevice/debug/allDevices/index.html`

- Out‑of‑memory during downloads/inference:
  - Use smaller resolutions/steps for tests
  - Ensure system downloader is used for large files, and prefer the managed device flow to keep memory usage predictable

- Native JNI is skipped in most tests:
  - Test harness sets `llmedge.disableNativeLoad=true` automatically for non‑E2E tests
  - The opt‑in E2E explicitly runs native inference

## Key test files

- Image (API layer): `llmedge/src/androidTest/java/io/aatricks/llmedge/ImageGenerationTest.kt`
- Native image E2E (opt‑in): `llmedge/src/androidTest/java/io/aatricks/llmedge/ImageGenerationE2ENativeTest.kt`
- Video API & integration tests (progress, cancellation, memory, schedulers, metadata):
  - `VideoGenerationTest.kt`, `VideoGenerationE2ETest.kt`, `VideoProgressCallbackTest.kt`
  - `VideoCancellationTest.kt`, `VideoMemoryRegressionTest.kt`, `VideoReproducibilityTest.kt`
  - `ModelSwitchingTest.kt`, `ModelVariantTest.kt`
- WAN E2E (arm64 only): `WanVideoE2ETest.kt`

## Speech E2E Tests

The library includes end-to-end tests for speech processing (Whisper STT and Bark TTS).

### Bark TTS Android E2E Test

Run on a connected device (arm64 recommended):

```bash
# Push the model to device first
adb push models/bark_ggml_weights.bin /data/local/tmp/

# Run the test
./gradlew :llmedge:connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class=io.aatricks.llmedge.BarkTtsAndroidE2ETest#testBarkTtsPerformance
```

**Test Output:**

- Generated samples: 69,120 (~2.88 seconds of audio)
- Progress updates: 583 callbacks
- Output saved to: `/data/user/0/io.aatricks.llmedge.test/cache/bark_test_output.wav`

**View Results:**
```bash
adb shell run-as io.aatricks.llmedge.test cat /data/user/0/io.aatricks.llmedge.test/cache/bark_tts_results.txt
```

### Bark TTS Desktop E2E Test

For faster development iteration, run Bark tests on Linux desktop:

```bash
export LLMEDGE_BUILD_BARK_LIB_PATH="/path/to/libbark_jni.so"
export LLMEDGE_TEST_BARK_MODEL_PATH="/path/to/bark_ggml_weights.bin"
export LD_LIBRARY_PATH="/path/to/lib:$LD_LIBRARY_PATH"

./gradlew :llmedge:testDebugUnitTest --tests "*BarkLinuxE2ETest*" --no-daemon
```

### Whisper STT Tests

Whisper tests are faster and work well on mobile:

```bash
./gradlew :llmedge:connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class=io.aatricks.llmedge.WhisperAndroidE2ETest
```
