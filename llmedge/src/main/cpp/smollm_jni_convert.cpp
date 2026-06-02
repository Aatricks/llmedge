// JNI bridge for the on-device safetensors -> GGUF converter (Track B / Phase B2, Layers 5-6).
//
// Exposes io.aatricks.llmedge.text.runtime.SmolLM.nativeConvertSafetensors as a static native method.
// The architecture/tensor/tokenizer conversion lives in llmedge::convert (convert/*.cpp), which has no
// ggml/llama dependency and is host-testable on its own. Quantization (Layer 6) is layered on here in
// the JNI wrapper, where the linked llama runtime's llama_model_quantize is available: F16 conversion
// happens first, then the result is requantized to the requested precision.
#include <cstdio>
#include <string>

#include "convert/hf_to_gguf.h"
#include "llama.h"
#include "smollm_jni_shared.h"

namespace {

// Map a ConversionPrecision.ggufLabel to a llama_ftype. Returns false for "f16" / unknown labels;
// the caller treats "f16" as "no quantization" and any other unknown label as an error.
bool ftypeForLabel(const std::string& label, llama_ftype& out) {
    if (label == "q8_0") {
        out = LLAMA_FTYPE_MOSTLY_Q8_0;
        return true;
    }
    if (label == "q4_k_m") {
        out = LLAMA_FTYPE_MOSTLY_Q4_K_M;
        return true;
    }
    if (label == "iq2_bn") {
        out = LLAMA_FTYPE_MOSTLY_IQ2_BN;
        return true;
    }
    if (label == "iq2_bn_r4") {
        out = LLAMA_FTYPE_MOSTLY_IQ2_BN_R4;
        return true;
    }
    return false;
}

}  // namespace

extern "C" JNIEXPORT void JNICALL Java_io_aatricks_llmedge_text_runtime_SmolLM_nativeConvertSafetensors(
        JNIEnv* env, jclass /*clazz*/, jstring modelDir, jstring outPath, jstring tokenizerPre,
        jstring precision) {
    ScopedUtfChars dir(env, modelDir);
    ScopedUtfChars out(env, outPath);
    ScopedUtfChars pre(env, tokenizerPre);
    ScopedUtfChars prec(env, precision);
    if (!dir.ok() || !out.ok() || !pre.ok() || !prec.ok()) {
        throwJavaException(env, "java/lang/IllegalArgumentException", "failed to read string arguments");
        return;
    }
    if (!dir.get() || !out.get()) {
        throwJavaException(env, "java/lang/IllegalArgumentException", "modelDir and outPath are required");
        return;
    }
    const std::string pre_s = pre.get() ? pre.get() : "";
    const std::string label = prec.get() ? prec.get() : "f16";

    try {
        // F16 is the converter's native output: no quantization step.
        if (label.empty() || label == "f16") {
            llmedge::convert::convert_llama_dir(dir.get(), out.get(), pre_s);
            return;
        }

        llama_ftype ftype;
        if (!ftypeForLabel(label, ftype)) {
            throwJavaException(env, "java/lang/IllegalArgumentException",
                              ("unsupported on-device conversion precision: " + label).c_str());
            return;
        }

        // Convert to a temporary F16 GGUF, then requantize it into the requested precision. The temp
        // file is removed on every path (success, non-zero code, or exception) so it never leaks.
        const std::string tmp_f16 = std::string(out.get()) + ".f16.tmp";
        uint32_t rc;
        try {
            llmedge::convert::convert_llama_dir(dir.get(), tmp_f16, pre_s);
            llama_backend_init();
            llama_model_quantize_params params = llama_model_quantize_default_params();
            params.ftype = ftype;
            rc = llama_model_quantize(tmp_f16.c_str(), out.get(), &params);
            llama_backend_free();
        } catch (...) {
            std::remove(tmp_f16.c_str());
            throw;
        }
        std::remove(tmp_f16.c_str());

        if (rc != 0) {
            throwJavaException(env, "java/lang/IllegalStateException",
                              ("llama_model_quantize failed (code " + std::to_string(rc) + ") for precision " +
                               label).c_str());
        }
    } catch (const std::exception& error) {
        throwJavaException(env, "java/lang/IllegalStateException", error.what());
    }
}
