# Contributing

Thanks for your interest in contributing to `llmedge`! This project contains native C++ code (llama.cpp, stable-diffusion.cpp), Kotlin Android libraries, and comprehensive examples. This guide will help you contribute effectively.

## Development Setup

### Prerequisites

- JDK 17+ (for Gradle Kotlin DSL)
- Android Studio (latest stable recommended)
- Android SDK & NDK r27+ (for native builds)
- CMake 3.22+ and Ninja
- Git with submodule support
- (Optional) VULKAN_SDK for Vulkan builds

### Initial Setup

1. **Fork the repository** on GitHub

2. **Clone your fork with submodules:**
   ```fish
   git clone --recursive https://github.com/YOUR_USERNAME/llmedge.git
   cd llmedge
   ```

3. **Open in Android Studio** and let Gradle sync

4. **Build the project:**
   ```fish
   ./gradlew :llmedge:assembleDebug
   ./gradlew :llmedge-examples:app:assembleDebug
   ```

5. **Run examples on a device or emulator** to verify setup

## Development Workflow

### Creating a Feature Branch

```fish
git checkout -b feature/your-feature-name
```

Use descriptive branch names:
- `feature/add-xyz` for new features
- `fix/issue-123` for bug fixes
- `docs/improve-readme` for documentation
- `refactor/cleanup-abc` for refactoring

### Making Changes

1. **Keep changes focused** — one feature or fix per PR
2. **Write tests** for new functionality when applicable
3. **Update documentation** in `docs/` if you change APIs
4. **Test on real devices** — emulators may not catch all issues
5. **Check memory usage** with `MemoryMetrics` for native changes
6. **Profile performance** if your change affects inference or loading

### Before Committing

1. **Format your code:**
   - Kotlin: Follow Android Kotlin style guide
   - C++: Follow existing project style
   - Use Android Studio's auto-formatter (Ctrl+Alt+L)

2. **Build and test:**
   ```fish
   ./gradlew clean build
   ```

3. **Run example apps** to verify functionality

4. **Check for warnings** in build output

## Coding Style

### Kotlin

- Follow [Android Kotlin style guide](https://developer.android.com/kotlin/style-guide)
- Use meaningful variable names
- Prefer `suspend fun` for long-running operations
- **Use `Dispatchers.IO` for native JNI operations** - they are blocking I/O operations
- **Avoid `Dispatchers.Default` for native calls** - it has limited parallelism and causes thread starvation
- Document public APIs with KDoc
- Use `@JvmStatic` for JNI-exposed methods

**Example:**
```kotlin
/**
 * Loads a GGUF model from the specified path.
 *
 * @param modelPath Absolute path to the GGUF file.
 * @param params Inference configuration parameters.
 * @throws FileNotFoundException if the model file doesn't exist.
 */
suspend fun load(
    modelPath: String,
    params: InferenceParams = InferenceParams()
) = withContext(Dispatchers.IO) {
    // Implementation - uses IO because native JNI calls block the thread
}
```

### C++

- Follow existing project style (matches llama.cpp conventions)
- Use RAII for resource management
- Check for null pointers from JNI
- Log errors with descriptive messages
- Use `android/log.h` for native logging
- Prefix JNI functions with `Java_io_aatricks_llmedge_`

**Example:**
```cpp
extern "C" JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_SmolLM_loadModel(
    JNIEnv* env, jobject /* this */,
    jstring modelPath,
    jfloat minP,
    // ... other params
) {
    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    if (!path) {
        __android_log_print(ANDROID_LOG_ERROR, "SmolLM", "Failed to get model path");
        return 0;
    }
    
    // Implementation
    
    env->ReleaseStringUTFChars(modelPath, path);
    return reinterpret_cast<jlong>(model_ptr);
}
```

### Documentation

- Use Markdown for all documentation
- Include code examples for new features
- Add screenshots for UI-related changes
- Update the relevant section in `docs/`
- Keep README.md concise; details go in `docs/`

## Testing

### Manual Testing

1. **Test on real devices** with different Android versions
2. **Try different models** (small and large, quantized and full precision)
3. **Test memory limits** — try on low-memory devices (<2GB RAM)
4. **Verify error handling** — test with invalid inputs
5. **Check background/foreground transitions**

### Performance Testing

- Measure inference speed with `getLastGenerationMetrics()`
- Profile memory with `MemoryMetrics.snapshot()`
- Test with different `numThreads` values
- Compare before/after for performance-affecting changes
- Include performance notes in PR description

### Example Apps

Run all example activities:

- LocalAssetDemoActivity
- HuggingFaceDemoActivity
- ImageToTextActivity
- RagActivity
- StableDiffusionActivity
- LlavaVisionActivity

## Submitting a Pull Request

### PR Checklist

- [ ] Code builds without errors or warnings
- [ ] All example apps run successfully
- [ ] Changes are focused and well-scoped
- [ ] Code follows project style guidelines
- [ ] Public APIs are documented
- [ ] Relevant documentation is updated
- [ ] Commit messages are clear and descriptive
- [ ] No unnecessary files committed (build artifacts, IDE configs)

### PR Description Template

```markdown
## Description
Brief description of what this PR does.

## Changes
- Specific change 1
- Specific change 2

## Testing
- Tested on: [Device model, Android version]
- Test results: [Pass/Fail, performance notes]

## Performance Impact
- Before: [metrics if applicable]
- After: [metrics if applicable]

## Screenshots
[If UI-related]

## Related Issues
Fixes #123
Related to #456
```

### Review Process

1. Maintainer will review within a few days
2. Address review feedback promptly
3. Push updates to the same branch (no force push please)
4. Once approved, maintainer will merge

## Reporting Bugs

### Bug Report Template

When reporting bugs, please include:

1. **Device information:**

   - Device make/model
   - Android version
   - ABI (check `Build.SUPPORTED_ABIS[0]`)

2. **Build information:**

   - NDK version
   - llmedge version/commit

3. **Model information:**

   - Model name and size
   - Quantization type
   - Where obtained (HF Hub, local, etc.)

4. **Reproduction steps:**

   - Minimal code to reproduce
   - Expected behavior
   - Actual behavior

5. **Logs:**
   ```fish
   adb logcat -s SmolLM:* SmolSD:* AndroidRuntime:*
   ```

6. **Memory usage:**

   - Use `MemoryMetrics.snapshot()` if relevant

## Feature Requests

Before requesting a feature:

1. Check if it already exists or is planned
2. Search existing issues
3. Describe your use case clearly
4. Explain why it's useful for the community
5. Consider if it fits the project scope (on-device inference)

## Native Development Notes

### Building Native Code

The project uses CMake via Android Gradle plugin:

```fish
# Clean native builds
rm -rf llmedge/.cxx

# Rebuild with Vulkan
./gradlew :llmedge:assembleRelease -Pandroid.jniCmakeArgs="-DGGML_VULKAN=ON -DSD_VULKAN=ON"
```

### Debugging Native Code

1. **Build debug variant:**
   ```fish
   ./gradlew :llmedge:assembleDebug
   ```

2. **Attach debugger** in Android Studio (Run → Attach to Process)

3. **Use native logging:**
   ```cpp
   #include <android/log.h>
   __android_log_print(ANDROID_LOG_DEBUG, "TAG", "Message: %s", str);
   ```

4. **Symbolicate crashes:**
   ```fish
   adb logcat | ndk-stack -sym llmedge/.cxx/Debug/arm64-v8a/
   ```

### Updating llama.cpp Submodule

If updating the vendored llama.cpp:

```fish
cd llama.cpp
git fetch origin
git checkout [desired-commit]
cd ..
git add llama.cpp
git commit -m "Update llama.cpp to [version]"
```

Test thoroughly after submodule updates!

Note: The project's native CMake build supports both legacy llama.cpp layouts that provide per-model
source files under `src/models/*.cpp`, as well as newer llama.cpp versions that consolidate model
implementations into `llama-model.cpp`. If you update the submodule and encounter CMake errors about
missing source files, ensure the `llmedge/src/main/cpp/CMakeLists.txt` file reflects the current
llama.cpp structure or open a PR with a fix similar to the existing guarded `file(GLOB ...)` approach.

## Documentation

### Building Docs Locally

The project uses MkDocs:

```fish
pip install mkdocs mkdocs-material
mkdocs serve
```

View at http://127.0.0.1:8080

### Documentation Structure

- `docs/index.md` — Overview and highlights
- `docs/installation.md` — Setup instructions
- `docs/usage.md` — API documentation
- `docs/examples.md` — Code examples
- `docs/architecture.md` — System design
- `docs/quirks.md` — Troubleshooting
- `docs/faq.md` — Common questions
- `docs/contributing.md` — This file

## Questions?

If you have questions:

1. Check the [FAQ](faq.md)
2. Search existing [issues](https://github.com/Aatricks/llmedge/issues)
3. Ask in a new issue with the "question" label

## License & Code of Conduct

- This project is licensed under Apache 2.0 (see LICENSE file)
- Contributions must be compatible with this license
- Be respectful and constructive in all interactions
- Follow GitHub's community guidelines

Thank you for contributing to llmedge! 🚀