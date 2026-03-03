# Copilot Instructions for llmedge

## Project Overview

llmedge is an Android library (AAR) for running ML models on-device via JNI. It wraps four C/C++ inference engines — llama.cpp (LLM), stable-diffusion.cpp (image/video), whisper.cpp (STT), and bark.cpp (TTS) — behind Kotlin APIs. The single Gradle module is `:llmedge`.

## Build Commands

```bash
# Build the AAR (release, Vulkan enabled by default in CMakeLists.txt)
./gradlew :llmedge:assembleRelease

# Build debug variant
./gradlew :llmedge:assembleDebug

# Clean native build artifacts
rm -rf llmedge/.cxx
```

Host prerequisites: JDK 17, Android NDK r27 (`27.2.12479018`), CMake 3.22+, Ninja, `glslc`, `libvulkan-dev`.

## Test Commands

```bash
# Unit tests (JVM, native JNI stubbed out)
./gradlew :llmedge:testDebugUnitTest

# Skip E2E tests in unit run (CI default)
LLMEDGE_SKIP_E2E_IN_UNIT=true ./gradlew :llmedge:testDebugUnitTest

# Run a single test class
./gradlew :llmedge:testDebugUnitTest --tests "*MyTestClass"

# Instrumentation tests on managed emulator (Pixel 6, API 33, ATD)
./gradlew :llmedge:pixel6api33DebugAndroidTest

# Instrumentation tests on connected device
./gradlew :llmedge:connectedDebugAndroidTest

# Single instrumentation test
./gradlew :llmedge:connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class=io.aatricks.llmedge.SomeTest#testMethod

# Show test stdout/stderr
LLMEDGE_SHOW_TEST_OUTPUT=true ./gradlew :llmedge:testDebugUnitTest
```

E2E tests (native inference) require building desktop JNI libs first via `scripts/build_*.sh` and setting `LLMEDGE_BUILD_NATIVE_LIB_PATH` (and related env vars). See `docs/testing.md`.

## Architecture

### Layered JNI Design

```
Kotlin API layer (SmolLM, StableDiffusion, Whisper, BarkTTS)
  └─ LLMEdgeManager (high-level singleton orchestrator)
      └─ JNI bridge (smollm.cpp, sdcpp_jni.cpp, whisper_jni.cpp, bark_jni.cpp)
          └─ C/C++ engines (git submodules: llama.cpp, stable-diffusion.cpp, whisper.cpp, bark.cpp)
```

Each C++ engine is compiled as a **separate shared library** with its own bundled ggml to avoid symbol conflicts. The SmolLM library is built in multiple ARM variants (v8, v8.2, v8.4 with fp16/dotprod/sve/i8mm) and the best one is selected at runtime based on CPU features.

### Key Source Locations

- **Kotlin APIs**: `llmedge/src/main/java/io/aatricks/llmedge/`
- **JNI C++ glue**: `llmedge/src/main/cpp/` (`smollm.cpp`, `sdcpp_jni.cpp`, `whisper_jni.cpp`, `bark_jni.cpp`)
- **CMake build**: `llmedge/src/main/cpp/CMakeLists.txt` — single CMakeLists that builds all four native libraries
- **C++ engine submodules**: `llama.cpp/`, `stable-diffusion.cpp/`, `whisper.cpp/`, `bark.cpp/`
- **Patched SD sources**: `mods/` directory overlays onto stable-diffusion.cpp at build time (keeps submodule clean)
- **Desktop E2E scripts**: `scripts/`

### Submodule + Mods Pattern

The `mods/` directory contains modified copies of stable-diffusion.cpp source files. During CMake configure, these are copied over the submodule sources into a build-time scratch tree (`patched-sd-src`), keeping the git submodule pristine. This is controlled by `LLMEDGE_SDCPP_USE_MODS` in CMake.

## Key Conventions

### Kotlin

- **Use `Dispatchers.IO` for all native JNI calls** — they are blocking I/O. Never use `Dispatchers.Default` (causes thread starvation).
- JNI-exposed methods use `@JvmStatic`. JNI function names follow `Java_io_aatricks_llmedge_` prefix.
- `LLMEdgeManager` is the high-level entry point; `SmolLM`, `StableDiffusion`, `Whisper`, `BarkTTS` are lower-level APIs.
- Always call `.close()` on model instances to free native memory.
- Package: `io.aatricks.llmedge` (namespace in `build.gradle.kts`).

### C++

- `.clang-format` is GNU-based, 4-space indent, 120-column limit, attach braces.
- Use `android/log.h` (`__android_log_print`) for native logging. Tags: `SmolLM`, `SmolSD`.
- Each native library links only `android` and `log`; ggml is statically compiled in (not shared).
- `GGML_COMMIT` and `GGML_VERSION` are set to empty strings since we bypass llama.cpp's CMake.

### Testing

- Most unit tests stub native code via `llmedge.disableNativeLoad=true` system property.
- E2E tests need native libs built for the host platform (Linux x86_64) using `scripts/build_*.sh`.
- Test framework: JUnit + MockK + Robolectric (unit), AndroidX Test + Espresso (instrumentation).
- Jacoco coverage is configured and excludes vision, OCR, RAG, and HuggingFace packages.

### Build

- `minSdk = 30` (Android 11) — required for Vulkan 1.2.
- `compileSdk = 35`, `jvmTarget = 17`.
- Vulkan is enabled by default in CMakeLists.txt (`SD_VULKAN=ON`, `GGML_VULKAN=ON`).
- Gradle logging is set to `quiet` in `gradle.properties`.

## Documentation

Docs are in `docs/` and built with MkDocs (`mkdocs serve` on port 8080). Key files: `testing.md`, `contributing.md`, `architecture.md`, `quirks.md`.
