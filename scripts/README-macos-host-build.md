# Running the desktop (host) native tests on macOS

The `scripts/*_linux.sh` host-build/E2E harness also works on macOS (Apple Silicon) with a few
prerequisites, because stock macOS ships tools that are too old or BSD-flavored.

## Prerequisites

```bash
brew install bash cmake ninja gpatch openjdk@17
```

- **bash 5** — the build scripts use `mapfile` and `local -n` namerefs (bash 4.3+); stock macOS bash is 3.2.
  Run the scripts with Homebrew bash: `/opt/homebrew/bin/bash scripts/build_native_linux.sh smollm`.
- **gpatch (GNU patch)** — the mod-overlay step (`android-vulkan-iqk.patch`) needs GNU `patch`; BSD `patch`
  fails to create its temp files. Put GNU patch on `PATH` as `patch`, e.g.:
  `mkdir -p /tmp/gnubin && ln -sf "$(brew --prefix gpatch)/bin/gpatch" /tmp/gnubin/patch` then prepend
  `/tmp/gnubin` to `PATH`.
- **A full JDK for JNI headers** — the Android Studio JBR ships no JNI headers, so cmake's `find_package(JNI)`
  fails. Point the build at a full JDK: `JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home`.

The build emits both `libsmollm.so` and a `libsmollm.dylib` alias on macOS so the JVM name-based loader
(`System.loadLibrary`) can find it.

## Build + run a text model

```bash
PATCH_BIN=/tmp/gnubin   # dir containing a `patch` -> gpatch symlink (see above)
JDK=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
JBR="/Applications/Android Studio.app/Contents/jbr/Contents/Home"   # any JDK 17+ runs gradle

# 1) Build the host smollm JNI library
JAVA_HOME="$JDK" PATH="$PATCH_BIN:/opt/homebrew/bin:$PATH" \
  /opt/homebrew/bin/bash scripts/build_native_linux.sh smollm

# 2) Run the text E2E against any GGUF (here: BitNet b1.58 IQ2_BN)
export LLMEDGE_BUILD_NATIVE_LIB_PATH="$PWD/llmedge/build/native/linux-aarch64/libsmollm.so"
export LLMEDGE_TEST_TEXT_MODEL_PATH="$PWD/models/bitnet1582b4t-iq2_bn.gguf"
JAVA_HOME="$JBR" ./gradlew :llmedge:testDebugUnitTest \
  --tests "io.aatricks.llmedge.TextInferenceLinuxE2ETest" --no-daemon --rerun-tasks
```

`BitNetTemplateE2ETest` additionally checks that the bundled BitNet chat template produces coherent output
(it answers "What is the capital of France?" with "Paris").
