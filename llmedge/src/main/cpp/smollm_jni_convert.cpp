// JNI bridge for the on-device safetensors -> GGUF converter (Track B / Phase B2, Layer 5).
//
// Exposes io.aatricks.llmedge.text.runtime.SmolLM.nativeConvertSafetensors as a static native method.
// The heavy lifting lives in llmedge::convert (convert/*.cpp), which has no JNI dependency and is
// host-testable on its own; this file is only the thin JNI marshalling layer.
#include "convert/hf_to_gguf.h"
#include "smollm_jni_shared.h"

extern "C" JNIEXPORT void JNICALL Java_io_aatricks_llmedge_text_runtime_SmolLM_nativeConvertSafetensors(
        JNIEnv* env, jclass /*clazz*/, jstring modelDir, jstring outPath, jstring tokenizerPre) {
    ScopedUtfChars dir(env, modelDir);
    ScopedUtfChars out(env, outPath);
    ScopedUtfChars pre(env, tokenizerPre);
    if (!dir.ok() || !out.ok() || !pre.ok()) {
        throwJavaException(env, "java/lang/IllegalArgumentException", "failed to read string arguments");
        return;
    }
    if (!dir.get() || !out.get()) {
        throwJavaException(env, "java/lang/IllegalArgumentException", "modelDir and outPath are required");
        return;
    }
    try {
        llmedge::convert::convert_llama_dir(dir.get(), out.get(), pre.get() ? pre.get() : "");
    } catch (const std::exception& error) {
        throwJavaException(env, "java/lang/IllegalStateException", error.what());
    }
}
