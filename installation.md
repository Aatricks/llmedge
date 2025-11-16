# Installation

This guide covers installing and building `llmedge` for local development and integrating the library into an Android project.

### Prerequisites
- JDK 17+ (for Gradle Kotlin DSL)
- Android SDK & NDK (if building Android native parts)
- CMake and a C++ toolchain supporting your target (clang/gcc)
- Python 3.8+ for some auxiliary scripts (optional)
- (Optional) Conda/venv for any Python tooling

### Project layout
- `llmedge/` — main Android library module with JNI/C++ sources
- `llmedge-examples/app/` — Android sample app showing use-cases
- `llama.cpp/` — vendored/third-party inference code used by native layer

For simple usage, you can download prebuilt AARs from the Releases page and include them in your Android project.


For development and building from source, follow these steps:
Clone

```fish
git clone https://github.com/Aatricks/llmedge.git
cd llmedge
git submodule update --init --recursive
```

### Build (Android Studio)
1. Open the `llmedge` root in Android Studio (or the `llmedge-examples` project).
2. Let Gradle sync. Android Studio will download required SDK/NDK and Gradle toolchains per `local.properties` and `gradle.properties`.
3. Build the `:llmedge` library and example app (Build -> Make Project).

### Command-line build

- Build AAR of `llmedge` (from project root):

```fish
./gradlew :llmedge:assembleDebug
```

- Build the example app:

```fish
./gradlew :llmedge-examples:app:assembleDebug
```

### Native build notes
- The native code resides in `llmedge/src/main/cpp` and uses JNI bindings and `llama.cpp`-style readers.
- If you run into issues with the C++ toolchain, verify `ANDROID_NDK_HOME` and `ANDROID_SDK_ROOT` in `local.properties` or environment variables.

#### GGUF/Model files
- `GGUFReader` supports loading GGUF model files. Place model files on device storage or in the app's files directory and point APIs at those paths.

#### Hugging Face integration
- The `io.aatricks.llmedge.huggingface` package can download models from the Hugging Face Hub.
- Use `SmolLM.loadFromHuggingFace()` for automatic download and loading.
- Optionally provide an HF token for private repositories via the `token` parameter.

### Troubleshooting
- Gradle sync failed: delete `.gradle` and try again
- Native build fails: ensure correct NDK version and API level
- Missing model: confirm the model path and file permissions on device

If you prefer a containerized CI build, refer to `llama.cpp/ci` for example scripts that set up required toolchains.