# AGENTS.md — working rules for AI agents in this repo

llmedge is an Android on-device inference SDK: Kotlin API (`llmedge/src/main/java/io/aatricks/llmedge/`)
over JNI/C++ (`llmedge/src/main/cpp/`) wrapping vendored engines: `llama.cpp/` (ik_llama.cpp fork,
text), `whisper.cpp/` (STT), `stable-diffusion.cpp/` (image), `bark.cpp/` (TTS), plus ONNX Runtime
for embeddings. Example app in `llmedge-examples/`.

## Hard rules

- **NEVER commit.** Leave all changes uncommitted in the working tree; the conductor reviews and commits.
- **Never modify the vendored submodules** (`llama.cpp/`, `whisper.cpp/`, `stable-diffusion.cpp/`, `bark.cpp/`)
  unless the task explicitly says so. Integration fixes belong in `llmedge/src/main/cpp/`.
- **No new `@Suppress` annotations** anywhere. Fix the warning instead.
- **Do not change existing abstract method signatures on any `NativeBridge` contract**
  (e.g. `SmolLMNativeBridgeContract`, Whisper/SD bridge contracts): ~15 test fakes implement them and
  all break. To evolve a bridge, ADD a second default-implemented overload that delegates to the old
  method (see `loadModel` nUbatch overload in `SmolLMNativeBridgeContract.kt` for the pattern).
- Match the existing code style exactly (comment density, naming, no decorative comments).
  Comments only for constraints the code can't express.
- Touch only files the task names or clearly requires. No drive-by refactors.

## Build & verify (macOS host)

- Unit suite: `LLMEDGE_SKIP_E2E_IN_UNIT=true JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home" ./gradlew :llmedge:testDebugUnitTest`
- Host native builds MUST use bash 5 and the full JDK home:
  `JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home /opt/homebrew/bin/bash scripts/build_native_linux.sh smollm`
  (stock bash 3.2 fails SILENTLY — `local -n` errors but exits 0 through a pipe; never judge success
  through `| tail`, check the exit code and the produced library). Same for `whisper`. On macOS the
  link step produces a .dylib and the final .so copy fails (exit 1) — "Built target" lines are the
  compile/link oracle there.
- sdcpp/bark cannot be compiled on this host (missing Vulkan SDK); for C++ changes under
  `llmedge/src/main/cpp/sdcpp_*`/`bark_*`, verify syntax/logic carefully and rely on the
  Android NDK build: `JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home" ./gradlew :llmedge:externalNativeBuildDebug` (slow; run once per batch).
- Kotlin-only changes: the unit suite is the gate. Some tests are Robolectric (sdk 34).

## Conventions

- JNI: model/user text crossing JNI must use byte[] paths (Modified-UTF-8 vs real UTF-8 —
  `NewStringUTF`/`GetStringUTFChars` are unsafe for 4-byte UTF-8). Use `llmedge_new_string_utf8`
  (jni_utils.h) native→Java; byte[] + String(bytes, UTF_8) Java→native.
- Process env mutations go through `llmedge_process_env_mutex()` (jni_utils.h).
- Native handles: Kotlin side owns lifecycle; guard close with CAS/lock; never delete a handle
  while its mutex is held (scope the lock, delete after).
- Coroutines: never swallow `CancellationException` — rethrow before generic catch blocks.
- Kotlin tests live in `llmedge/src/test/java/io/aatricks/llmedge/`; follow the existing fake-bridge
  patterns there when a test needs a native stand-in.
