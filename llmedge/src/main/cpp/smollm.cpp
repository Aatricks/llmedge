#include "LLMInference.h"
#include "llmedge_llama_compat.h"
#include <jni.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

// Include libmtmd headers via the include path so the build can switch between
// upstream tools/mtmd and fork layouts such as examples/mtmd.
#include "mtmd.h"
#include "mtmd-helper.h"

namespace {

void throwJavaException(JNIEnv* env, const char* className, const char* message) {
    if (!env) {
        return;
    }
    jclass exClass = env->FindClass(className);
    if (!exClass) {
        return;
    }
    env->ThrowNew(exClass, message);
}

void throwInvalidHandle(JNIEnv* env, const char* owner) {
    throwJavaException(env, "java/lang/IllegalStateException", owner);
}

class ScopedUtfChars {
  public:
    ScopedUtfChars(JNIEnv* env, jstring value) : env_(env), value_(value) {
        chars_ = value_ ? env_->GetStringUTFChars(value_, nullptr) : nullptr;
    }

    ~ScopedUtfChars() {
        if (chars_) {
            env_->ReleaseStringUTFChars(value_, chars_);
        }
    }

    ScopedUtfChars(const ScopedUtfChars&) = delete;
    ScopedUtfChars& operator=(const ScopedUtfChars&) = delete;

    const char* get() const { return chars_; }
    bool ok() const { return value_ == nullptr || chars_ != nullptr; }

  private:
    JNIEnv* env_;
    jstring value_;
    const char* chars_ = nullptr;
};

LLMInference* requireInference(JNIEnv* env, jlong modelPtr, const char* operation) {
    // Guard against obviously invalid sentinel/test handles so a bad JNI call becomes a
    // Java-side error instead of dereferencing near-null memory and crashing the JVM.
    if (modelPtr > 0 && modelPtr < 4096) {
        throwInvalidHandle(env, operation);
        return nullptr;
    }
    auto* llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    if (!llmInference) {
        throwInvalidHandle(env, operation);
    }
    return llmInference;
}

struct DecodeEmbeddingsBatch {
    int n_pos_per_embd;
    int n_mmproj_embd;
    std::vector<llama_pos>      pos;
    std::vector<llama_pos>      pos_view;
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id>   seq_id_0;
    std::vector<llama_seq_id *> seq_ids;
    std::vector<int8_t>         logits;
    llama_batch batch {};

    DecodeEmbeddingsBatch(float * embd, int32_t n_tokens, int n_pos_per_token, int embd_dim) :
            n_pos_per_embd(n_pos_per_token), n_mmproj_embd(embd_dim) {
        pos.resize(static_cast<size_t>(n_tokens) * static_cast<size_t>(n_pos_per_embd));
        n_seq_id.resize(n_tokens);
        seq_id_0.resize(1);
        seq_ids.resize(static_cast<size_t>(n_tokens) + 1);
        logits.resize(n_tokens);
        seq_ids[n_tokens] = nullptr;
        batch = {
                /* n_tokens = */ n_tokens,
                /* token    = */ nullptr,
                /* embd     = */ embd,
                /* pos      = */ pos.data(),
                /* n_seq_id = */ n_seq_id.data(),
                /* seq_id   = */ seq_ids.data(),
                /* logits   = */ logits.data(),
        };
    }

    void setPositionNormal(llama_pos pos_0, llama_seq_id seq_id) {
        seq_id_0[0] = seq_id;
        for (int i = 0; i < batch.n_tokens; ++i) {
            batch.pos[i] = pos_0 + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i] = seq_id_0.data();
            batch.logits[i] = false;
        }
    }

    void setPositionMRope2d(llama_pos pos_0, int nx, int ny, llama_seq_id seq_id) {
        seq_id_0[0] = seq_id;
        const int total = batch.n_tokens;
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                const int i = y * nx + x;
                if (i >= total) {
                    break;
                }
                pos[i] = pos_0;
                pos[i + total] = pos_0 + y;
                pos[i + total * 2] = pos_0 + x;
                pos[i + total * 3] = 0;
            }
        }
        for (int i = 0; i < total; ++i) {
            batch.n_seq_id[i] = 1;
            batch.seq_id[i] = seq_id_0.data();
            batch.logits[i] = false;
        }
    }

    void setPositionMRope1d(llama_pos pos_0, llama_seq_id seq_id) {
        seq_id_0[0] = seq_id;
        const int total = batch.n_tokens;
        for (int i = 0; i < total; ++i) {
            pos[i] = pos_0 + i;
            pos[i + total] = pos_0 + i;
            pos[i + total * 2] = pos_0 + i;
            pos[i + total * 3] = 0;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i] = seq_id_0.data();
            batch.logits[i] = false;
        }
    }

    llama_batch getView(int offset, int n_tokens) {
        llama_pos * pos_ptr = nullptr;
        pos_view.clear();
        if (n_pos_per_embd > 1) {
            pos_view.reserve(static_cast<size_t>(n_tokens) * static_cast<size_t>(n_pos_per_embd));
            for (int i = 0; i < n_pos_per_embd; ++i) {
                const size_t src_idx = static_cast<size_t>(i) * static_cast<size_t>(batch.n_tokens) + static_cast<size_t>(offset);
                pos_view.insert(
                        pos_view.end(),
                        pos.data() + src_idx,
                        pos.data() + src_idx + n_tokens);
            }
            pos_ptr = pos_view.data();
        } else {
            pos_ptr = pos.data() + offset;
        }
        return {
                /* n_tokens = */ n_tokens,
                /* token    = */ nullptr,
                /* embd     = */ batch.embd + static_cast<size_t>(offset) * static_cast<size_t>(n_mmproj_embd),
                /* pos      = */ pos_ptr,
                /* n_seq_id = */ batch.n_seq_id + offset,
                /* seq_id   = */ batch.seq_id + offset,
                /* logits   = */ batch.logits + offset,
        };
    }
};

bool decodeEmbeddingsIntoKv(
        llama_context * lctx,
        float * embd,
        int32_t n_tokens,
        int embd_dim,
        int nx,
        int ny,
        bool use_mrope,
        bool use_non_causal,
        int n_batch) {
    if (!lctx || !embd || n_tokens <= 0 || embd_dim <= 0 || n_batch <= 0) {
        return false;
    }

    const int effective_batch = std::max(n_batch, n_tokens);
    const llama_pos seq_pos_max = llmedge_kv_cache_seq_pos_max(lctx, 0);
    const llama_pos n_past = seq_pos_max >= 0 ? seq_pos_max + 1 : 0;
    const int n_pos_per_embd = use_mrope ? 4 : 1;
    DecodeEmbeddingsBatch batch_embd(embd, n_tokens, n_pos_per_embd, embd_dim);

    if (use_mrope) {
        if (nx > 0 && ny > 0) {
            batch_embd.setPositionMRope2d(n_past, nx, ny, 0);
        } else {
            batch_embd.setPositionMRope1d(n_past, 0);
        }
    } else {
        batch_embd.setPositionNormal(n_past, 0);
    }

    if (use_non_causal) {
        llama_set_causal_attn(lctx, false);
    }

    bool success = true;
    const int32_t n_img_batches = (n_tokens + effective_batch - 1) / effective_batch;
    for (int32_t i_batch = 0; i_batch < n_img_batches; ++i_batch) {
        const int pos_offset = i_batch * effective_batch;
        const int n_tokens_batch = std::min(effective_batch, n_tokens - pos_offset);
        llama_batch batch_view = batch_embd.getView(pos_offset, n_tokens_batch);
        if (llama_decode(lctx, batch_view) != 0) {
            success = false;
            break;
        }
    }

    if (use_non_causal) {
        llama_set_causal_attn(lctx, true);
    }

    return success;
}

} // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_loadModel(JNIEnv* env, jobject thiz, jstring modelPath, jfloat minP,
                                             jfloat temperature, jboolean storeChats, jlong contextSize,
                                             jstring chatTemplate, jint nThreads, jboolean useMmap, jboolean useMlock,
                                             jboolean useVulkan, jboolean useFlashAttn, jint kvCacheTypeK, jint kvCacheTypeV,
                                             jint nGpuLayers) {
    ScopedUtfChars modelPathCstr(env, modelPath);
    ScopedUtfChars chatTemplateCstr(env, chatTemplate);
    if (!modelPathCstr.ok() || !chatTemplateCstr.ok()) {
        return 0;
    }

    auto llmInference = std::make_unique<LLMInference>();
    try {
        llmInference->loadModel(modelPathCstr.get(), minP, temperature, storeChats, contextSize, chatTemplateCstr.get(), nThreads,
                                useMmap, useMlock, useVulkan, useFlashAttn, kvCacheTypeK, kvCacheTypeV, nGpuLayers);
    } catch (const std::exception& error) {
        throwJavaException(env, "java/lang/IllegalStateException", error.what());
        return 0;
    }
    return reinterpret_cast<jlong>(llmInference.release());
}

extern "C" JNIEXPORT jboolean JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_nativeHasVulkanBackendSupport(JNIEnv* env, jobject thiz) {
#ifdef GGML_USE_VULKAN
    return JNI_TRUE;
#else
    (void) env;
    (void) thiz;
    return JNI_FALSE;
#endif
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_addChatMessage(JNIEnv* env, jobject thiz, jlong modelPtr, jstring message,
                                                  jstring role) {
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return;
    }
    ScopedUtfChars messageCstr(env, message);
    ScopedUtfChars roleCstr(env, role);
    if (!messageCstr.ok() || !roleCstr.ok()) {
        return;
    }
    try {
        llmInference->addChatMessage(messageCstr.get(), roleCstr.get());
    } catch (const std::exception& error) {
        throwJavaException(env, "java/lang/IllegalStateException", error.what());
    }
}

extern "C" JNIEXPORT jfloat JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_getResponseGenerationSpeed(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return 0.0f;
    }
    return llmInference->getResponseGenerationTime();
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_getResponseGeneratedTokenCount(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return 0;
    }
    return llmInference->getResponseTokenCount();
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_getResponseGenerationDurationMicros(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return 0;
    }
    return llmInference->getResponseGenerationTimeMicros();
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_nativeGetLastGenerationMetrics(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return nullptr;
    }

    const int64_t elapsedMicros = llmInference->getResponseGenerationTimeMicros();
    const long tokenCount = llmInference->getResponseTokenCount();
    const float tokensPerSecond = llmInference->getResponseTokensPerSecond();

    uint32_t speedBits = 0;
    static_assert(sizeof(speedBits) == sizeof(tokensPerSecond), "Unexpected float size");
    std::memcpy(&speedBits, &tokensPerSecond, sizeof(tokensPerSecond));

    jlong values[3] = {
        static_cast<jlong>(elapsedMicros),
        static_cast<jlong>(tokenCount),
        static_cast<jlong>(speedBits),
    };
    jlongArray result = env->NewLongArray(3);
    if (!result) {
        return nullptr;
    }
    env->SetLongArrayRegion(result, 0, 3, values);
    return result;
}

extern "C" JNIEXPORT jint JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_getContextSizeUsed(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return 0;
    }
    return llmInference->getContextSizeUsed();
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_nativeConfigureThreading(JNIEnv* env, jobject thiz, jlong modelPtr,
                                                         jint generationThreads, jint promptThreads) {
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return;
    }
    llmInference->configureThreading(generationThreads, promptThreads);
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_nativeGetEstimatedMemoryBytes(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return 0;
    }
    return static_cast<jlong>(llmInference->getEstimatedMemoryBytes());
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_nativeGetEstimatedStateMemoryBytes(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return 0;
    }
    return static_cast<jlong>(llmInference->getStateMemoryBytes());
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_close(JNIEnv* env, jobject thiz, jlong modelPtr) {
    if (!modelPtr) {
        return;
    }
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return;
    }
    delete llmInference;
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_startCompletion(JNIEnv* env, jobject thiz, jlong modelPtr, jstring prompt) {
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return;
    }
    ScopedUtfChars promptCstr(env, prompt);
    if (!promptCstr.ok()) {
        return;
    }
    try {
        llmInference->startCompletion(promptCstr.get());
    } catch (const std::exception& error) {
        throwJavaException(env, "java/lang/IllegalStateException", error.what());
    }
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_setReasoningOptions(JNIEnv* env, jobject thiz, jlong modelPtr, jboolean disableThinking,
                                                    jint reasoningBudget) {
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return;
    }
    const bool disable = disableThinking == JNI_TRUE;
    llmInference->setReasoningOptions(disable, reasoningBudget);
}

extern "C" JNIEXPORT jstring JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_completionLoop(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return nullptr;
    }
    try {
        std::string response = llmInference->completionLoop();
        return env->NewStringUTF(response.c_str());
    } catch (const std::exception& error) {
        throwJavaException(env, "java/lang/IllegalStateException", error.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_completionLoopBatch(JNIEnv* env, jobject thiz, jlong modelPtr, jint maxTokens) {
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return nullptr;
    }
    try {
        std::string response = llmInference->completionLoopBatch(maxTokens);
        return env->NewStringUTF(response.c_str());
    } catch (const std::exception& error) {
        throwJavaException(env, "java/lang/IllegalStateException", error.what());
        return nullptr;
    }
}

// Byte-based batch completion: returns raw UTF-8 bytes to avoid per-call NewStringUTF overhead
extern "C" JNIEXPORT jbyteArray JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_completionLoopBatchBytes(JNIEnv* env, jobject thiz, jlong modelPtr, jint maxTokens) {
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return nullptr;
    }
    try {
        std::string response = llmInference->completionLoopBatch(maxTokens);
        jbyteArray result = env->NewByteArray(static_cast<jsize>(response.size()));
        if (result && !response.empty()) {
            env->SetByteArrayRegion(result, 0, static_cast<jsize>(response.size()),
                                    reinterpret_cast<const jbyte*>(response.c_str()));
        }
        return result;
    } catch (const std::exception& error) {
        throwJavaException(env, "java/lang/IllegalStateException", error.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_setThreadAffinity(JNIEnv* env, jobject thiz, jlong modelPtr, jlong coreMask) {
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (llmInference) {
        llmInference->setThreadAffinity(static_cast<uint64_t>(coreMask));
    }
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_stopCompletion(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return;
    }
    llmInference->stopCompletion();
}

// Projector JNI stubs used by Projector.kt. These are lightweight placeholders
// so the example can demonstrate the safe sequencing of projector usage.
// Map to keep model pointer associated with mtmd_context (for embd dim lookup)
static std::unordered_map<mtmd_context*, llama_model*> g_mtmd_model_map;
static std::mutex g_mtmd_map_mutex;

extern "C" JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_Projector_nativeInitProjector(JNIEnv* env, jobject thiz, jstring mmprojPath, jlong textModelPtr) {
    mtmd_context* ctx = nullptr;
    if (mmprojPath == nullptr) return 0;
    ScopedUtfChars mmprojC(env, mmprojPath);
    if (!mmprojC.ok()) return 0;

    mtmd_context_params params = mtmd_context_params_default();
    params.use_gpu = false; // don't attempt GPU for Android example

    try {
        ctx = mtmd_init_from_file(mmprojC.get(), reinterpret_cast<const llama_model*>(textModelPtr), params);
    } catch (...) {
        ctx = nullptr;
    }

    if (ctx && textModelPtr != 0) {
        std::lock_guard<std::mutex> lk(g_mtmd_map_mutex);
        g_mtmd_model_map[ctx] = reinterpret_cast<llama_model*>(textModelPtr);
    }

    return reinterpret_cast<jlong>(ctx);
}

extern "C" JNIEXPORT jboolean JNICALL
Java_io_aatricks_llmedge_Projector_nativeEncodeImage(JNIEnv* env, jobject thiz, jlong nativePtr, jstring imagePath, jstring outPath) {
    ScopedUtfChars inC(env, imagePath);
    ScopedUtfChars outC(env, outPath);
    if (!inC.ok() || !outC.ok()) {
        return JNI_FALSE;
    }

    mtmd_context* ctx = reinterpret_cast<mtmd_context*>(nativePtr);
    bool ok = false;

    if (ctx == nullptr) {
        // No projector available; fallback to copying file
        std::ifstream src(inC.get(), std::ios::binary);
        std::ofstream dst(outC.get(), std::ios::binary);
        if (src && dst) {
            dst << src.rdbuf();
            ok = static_cast<bool>(src) && static_cast<bool>(dst);
        }
        return ok ? JNI_TRUE : JNI_FALSE;
    }

    // Use mtmd_helper_bitmap_init_from_file to load image and preprocess it
    mtmd_bitmap* bmp = mtmd_helper_bitmap_init_from_file(ctx, inC.get());
    if (!bmp) {
        return JNI_FALSE;
    }

    const mtmd_bitmap* bitmaps[1] = { bmp };
    mtmd_input_text txt = { "<__media__>", false, false };
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    int32_t tokRes = mtmd_tokenize(ctx, chunks, &txt, bitmaps, 1);
    if (tokRes != 0) {
        mtmd_bitmap_free(bmp);
        mtmd_input_chunks_free(chunks);
        return JNI_FALSE;
    }

    // Encode image tokens
    // find first image chunk
    bool encoded = false;
    // keep these in outer scope so we can write metadata after the loop
    size_t n_tokens = 0;
    int embd_dim = 0;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
        const mtmd_input_chunk* c = mtmd_input_chunks_get(chunks, i);
        if (c && mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            int32_t res = mtmd_encode_chunk(ctx, c);
            if (res == 0) {
                float* embd = mtmd_get_output_embd(ctx);
                // Write raw float embeddings to outPath
                std::ofstream ofs(outC.get(), std::ios::binary);
                if (ofs) {
                        // We need to know size: tokens * embd_dim
                        n_tokens = static_cast<size_t>(mtmd_input_chunk_get_n_tokens(c));
                        {
                            std::lock_guard<std::mutex> lk(g_mtmd_map_mutex);
                            auto it = g_mtmd_model_map.find(ctx);
                            if (it != g_mtmd_model_map.end() && it->second) {
                                embd_dim = llama_model_n_embd(it->second);
                            }
                        }
                        if (embd_dim <= 0) {
                            // We cannot safely determine embedding dimension; abort to avoid
                            // writing an incorrect amount of data. The caller should pass
                            // the text model pointer when initializing the projector so
                            // we can validate and compute the correct size.
                            ofs.close();
                            mtmd_bitmap_free(bmp);
                            mtmd_input_chunks_free(chunks);
                            return JNI_FALSE;
                        }
                        size_t n_floats = static_cast<size_t>(n_tokens) * static_cast<size_t>(embd_dim);
                        ofs.write(reinterpret_cast<const char*>(embd), sizeof(float) * n_floats);
                    encoded = true;
                }
            }
            break;
        }
    }

        // If we encoded successfully, write a small metadata JSON file next to embeddings
        if (encoded) {
            std::string metaPath = std::string(outC.get()) + ".meta.json";
            std::ofstream mofs(metaPath, std::ios::trunc);
            if (mofs) {
                // Try to get image token shape if available
                const mtmd_image_tokens* image_tokens = nullptr;
                for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
                    const mtmd_input_chunk* c2 = mtmd_input_chunks_get(chunks, i);
                    if (c2 && mtmd_input_chunk_get_type(c2) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                        image_tokens = mtmd_input_chunk_get_tokens_image(c2);
                        break;
                    }
                }

                int nx = 0;
                int ny = 0;
                if (image_tokens) {
                    nx = static_cast<int>(mtmd_image_tokens_get_nx(image_tokens));
                    ny = static_cast<int>(mtmd_image_tokens_get_ny(image_tokens));
                }

                bool use_mrope = mtmd_decode_use_mrope(ctx);
                bool use_non_causal = mtmd_decode_use_non_causal(ctx);

                // write simple JSON
                mofs << "{\n";
                mofs << "  \"n_tokens\": " << n_tokens << ",\n";
                mofs << "  \"nx\": " << nx << ",\n";
                mofs << "  \"ny\": " << ny << ",\n";
                mofs << "  \"embd_dim\": " << embd_dim << ",\n";
                mofs << "  \"use_mrope\": " << (use_mrope ? "true" : "false") << ",\n";
                mofs << "  \"use_non_causal\": " << (use_non_causal ? "true" : "false") << "\n";
                mofs << "}\n";
                mofs.close();
            }
        }

    mtmd_bitmap_free(bmp);
    mtmd_input_chunks_free(chunks);

    return encoded ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_Projector_nativeCloseProjector(JNIEnv* env, jobject thiz, jlong nativePtr) {
    (void) env;
    (void) thiz;
    mtmd_context* ctx = reinterpret_cast<mtmd_context*>(nativePtr);
    if (ctx) {
        // Remove mapping from ctx -> llama_model* to avoid stale references
        {
            std::lock_guard<std::mutex> lk(g_mtmd_map_mutex);
            auto it = g_mtmd_model_map.find(ctx);
            if (it != g_mtmd_model_map.end()) {
                g_mtmd_model_map.erase(it);
            }
        }
        mtmd_free(ctx);
    }
}

// Backwards/alternate JNI entrypoints for Projector in package io.aatricks.llmedge.vision
extern "C" JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_vision_Projector_nativeInitProjector(JNIEnv* env, jobject thiz, jstring mmprojPath, jlong textModelPtr) {
    return Java_io_aatricks_llmedge_Projector_nativeInitProjector(env, thiz, mmprojPath, textModelPtr);
}

extern "C" JNIEXPORT jboolean JNICALL
Java_io_aatricks_llmedge_vision_Projector_nativeEncodeImage(JNIEnv* env, jobject thiz, jlong nativePtr, jstring imagePath, jstring outPath) {
    return Java_io_aatricks_llmedge_Projector_nativeEncodeImage(env, thiz, nativePtr, imagePath, outPath);
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_vision_Projector_nativeCloseProjector(JNIEnv* env, jobject thiz, jlong nativePtr) {
    Java_io_aatricks_llmedge_Projector_nativeCloseProjector(env, thiz, nativePtr);
}

// Buffer-based image encoding: accepts JPEG bytes, returns [FloatArray embeddings, IntArray metadata]
extern "C" JNIEXPORT jobjectArray JNICALL
Java_io_aatricks_llmedge_vision_Projector_nativeEncodeImageBuffer(JNIEnv* env, jobject thiz, jlong nativePtr, jbyteArray jpegData) {
    if (!jpegData) return nullptr;

    mtmd_context* ctx = reinterpret_cast<mtmd_context*>(nativePtr);
    if (!ctx) return nullptr;

    jsize dataLen = env->GetArrayLength(jpegData);
    if (dataLen <= 0) return nullptr;

    jbyte* rawBytes = env->GetByteArrayElements(jpegData, nullptr);
    if (!rawBytes) return nullptr;

    mtmd_bitmap* bmp = mtmd_helper_bitmap_init_from_buf(ctx,
        reinterpret_cast<const unsigned char*>(rawBytes), static_cast<size_t>(dataLen));
    env->ReleaseByteArrayElements(jpegData, rawBytes, JNI_ABORT);
    if (!bmp) return nullptr;

    const mtmd_bitmap* bitmaps[1] = { bmp };
    mtmd_input_text txt = { "<__media__>", false, false };
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    int32_t tokRes = mtmd_tokenize(ctx, chunks, &txt, bitmaps, 1);
    if (tokRes != 0) {
        mtmd_bitmap_free(bmp);
        mtmd_input_chunks_free(chunks);
        return nullptr;
    }

    // Find and encode the first image chunk
    float* embd = nullptr;
    size_t n_tokens = 0;
    int embd_dim = 0;
    int nx = 0, ny = 0;
    bool encoded = false;

    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
        const mtmd_input_chunk* c = mtmd_input_chunks_get(chunks, i);
        if (c && mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            int32_t res = mtmd_encode_chunk(ctx, c);
            if (res == 0) {
                embd = mtmd_get_output_embd(ctx);
                n_tokens = static_cast<size_t>(mtmd_input_chunk_get_n_tokens(c));
                {
                    std::lock_guard<std::mutex> lk(g_mtmd_map_mutex);
                    auto it = g_mtmd_model_map.find(ctx);
                    if (it != g_mtmd_model_map.end() && it->second) {
                        embd_dim = llama_model_n_embd(it->second);
                    }
                }
                if (embd_dim <= 0) {
                    mtmd_bitmap_free(bmp);
                    mtmd_input_chunks_free(chunks);
                    return nullptr;
                }
                const mtmd_image_tokens* image_tokens = mtmd_input_chunk_get_tokens_image(c);
                if (image_tokens) {
                    nx = static_cast<int>(mtmd_image_tokens_get_nx(image_tokens));
                    ny = static_cast<int>(mtmd_image_tokens_get_ny(image_tokens));
                }
                encoded = true;
            }
            break;
        }
    }

    if (!encoded || !embd) {
        mtmd_bitmap_free(bmp);
        mtmd_input_chunks_free(chunks);
        return nullptr;
    }

    bool use_mrope = mtmd_decode_use_mrope(ctx);
    bool use_non_causal = mtmd_decode_use_non_causal(ctx);
    size_t n_floats = n_tokens * static_cast<size_t>(embd_dim);

    // Build result: Object[] { float[], int[] }
    jfloatArray embdArray = env->NewFloatArray(static_cast<jsize>(n_floats));
    if (!embdArray) {
        mtmd_bitmap_free(bmp);
        mtmd_input_chunks_free(chunks);
        return nullptr;
    }
    env->SetFloatArrayRegion(embdArray, 0, static_cast<jsize>(n_floats), embd);

    // metadata: [n_tokens, nx, ny, embd_dim, use_mrope, use_non_causal]
    jint meta[6] = {
        static_cast<jint>(n_tokens), static_cast<jint>(nx), static_cast<jint>(ny),
        static_cast<jint>(embd_dim), use_mrope ? 1 : 0, use_non_causal ? 1 : 0
    };
    jintArray metaArray = env->NewIntArray(6);
    if (!metaArray) {
        mtmd_bitmap_free(bmp);
        mtmd_input_chunks_free(chunks);
        return nullptr;
    }
    env->SetIntArrayRegion(metaArray, 0, 6, meta);

    jclass objClass = env->FindClass("java/lang/Object");
    if (!objClass) {
        mtmd_bitmap_free(bmp);
        mtmd_input_chunks_free(chunks);
        return nullptr;
    }
    jobjectArray result = env->NewObjectArray(2, objClass, nullptr);
    if (!result) {
        mtmd_bitmap_free(bmp);
        mtmd_input_chunks_free(chunks);
        return nullptr;
    }
    env->SetObjectArrayElement(result, 0, embdArray);
    env->SetObjectArrayElement(result, 1, metaArray);

    mtmd_bitmap_free(bmp);
    mtmd_input_chunks_free(chunks);

    return result;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_nativePrimeImageBuffer(
        JNIEnv * env,
        jobject thiz,
        jlong modelPtr,
        jlong projectorPtr,
        jbyteArray imageData,
        jint nBatch) {
    (void) thiz;
    if (!imageData || projectorPtr == 0 || nBatch <= 0) {
        return JNI_FALSE;
    }

    auto * llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return JNI_FALSE;
    }

    llama_context * lctx = llmInference->getContext();
    mtmd_context * mtmd = reinterpret_cast<mtmd_context *>(projectorPtr);
    if (!lctx || !mtmd) {
        return JNI_FALSE;
    }

    const jsize data_len = env->GetArrayLength(imageData);
    if (data_len <= 0) {
        return JNI_FALSE;
    }

    jbyte * raw_bytes = env->GetByteArrayElements(imageData, nullptr);
    if (!raw_bytes) {
        return JNI_FALSE;
    }

    mtmd_bitmap * bmp = mtmd_helper_bitmap_init_from_buf(
            mtmd,
            reinterpret_cast<const unsigned char *>(raw_bytes),
            static_cast<size_t>(data_len));
    env->ReleaseByteArrayElements(imageData, raw_bytes, JNI_ABORT);
    if (!bmp) {
        return JNI_FALSE;
    }

    const mtmd_bitmap * bitmaps[1] = { bmp };
    mtmd_input_text txt = { "<__media__>", false, false };
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    if (!chunks) {
        mtmd_bitmap_free(bmp);
        return JNI_FALSE;
    }

    const int32_t tok_res = mtmd_tokenize(mtmd, chunks, &txt, bitmaps, 1);
    if (tok_res != 0) {
        mtmd_bitmap_free(bmp);
        mtmd_input_chunks_free(chunks);
        return JNI_FALSE;
    }

    int effective_batch = nBatch;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
        if (!chunk) {
            continue;
        }
        const auto type = mtmd_input_chunk_get_type(chunk);
        if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE || type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            effective_batch = std::max(effective_batch, static_cast<int>(mtmd_input_chunk_get_n_tokens(chunk)));
        }
    }

    const llama_pos seq_pos_max = llmedge_kv_cache_seq_pos_max(lctx, 0);
    const llama_pos n_past = seq_pos_max >= 0 ? seq_pos_max + 1 : 0;
    llama_pos new_n_past = n_past;
    const int32_t eval_res = mtmd_helper_eval_chunks(mtmd, lctx, chunks, n_past, 0, effective_batch, false, &new_n_past);

    mtmd_bitmap_free(bmp);
    mtmd_input_chunks_free(chunks);

    if (eval_res != 0) {
        return JNI_FALSE;
    }

    llmInference->markPreparedKvForNextCompletion();
    return JNI_TRUE;
}

// Buffer-based embedding decoding: accepts float array + metadata, populates KV cache
extern "C" JNIEXPORT jboolean JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_nativeDecodeEmbeddingsBuffer(JNIEnv* env, jobject thiz, jlong modelPtr,
                                                              jfloatArray embeddings, jint nTokens, jint nx, jint ny,
                                                              jint embdDim, jboolean useMrope, jboolean useNonCausal,
                                                              jint nBatch) {
    if (!embeddings || nTokens <= 0 || embdDim <= 0 || nBatch <= 0) return JNI_FALSE;

    jsize arrLen = env->GetArrayLength(embeddings);
    size_t expected = static_cast<size_t>(nTokens) * static_cast<size_t>(embdDim);
    if (static_cast<size_t>(arrLen) < expected) return JNI_FALSE;

    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) return JNI_FALSE;

    llama_context* lctx = llmInference->getContext();
    if (!lctx) return JNI_FALSE;

    jfloat* embdData = env->GetFloatArrayElements(embeddings, nullptr);
    if (!embdData) return JNI_FALSE;

    const bool success = decodeEmbeddingsIntoKv(
            lctx,
            embdData,
            nTokens,
            embdDim,
            nx,
            ny,
            useMrope == JNI_TRUE,
            useNonCausal == JNI_TRUE,
            nBatch);

    env->ReleaseFloatArrayElements(embeddings, embdData, JNI_ABORT);

    if (success) {
        llmInference->markPreparedKvForNextCompletion();
    }

    return success ? JNI_TRUE : JNI_FALSE;
}

// Return the internal llama_model* as jlong for advanced native integrations (caller must not free)
extern "C" JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_getNativeModelPtr(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) return 0;
    return reinterpret_cast<jlong>(llmInference->getModel());
}

// Decode prepared embeddings (.bin) using the already-loaded llama_context inside LLMInference
extern "C" JNIEXPORT jboolean JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_nativeDecodePreparedEmbeddings(JNIEnv* env, jobject thiz, jlong modelPtr,
                                                               jstring embdPath, jstring metaPath, jint nBatch) {
    if (!embdPath || !metaPath) return JNI_FALSE;
    ScopedUtfChars embdC(env, embdPath);
    ScopedUtfChars metaC(env, metaPath);
    if (!embdC.ok() || !metaC.ok()) {
        return JNI_FALSE;
    }

    // Read metadata JSON (very small) - parse manually
    int n_tokens = 0;
    int nx = 0, ny = 0;
    int embd_dim = 0;
    bool use_mrope = false;
    bool use_non_causal = false;

    std::ifstream mif(metaC.get());
    if (!mif) {
        return JNI_FALSE;
    }
    std::string line;
    while (std::getline(mif, line)) {
        auto pos = line.find_first_of(':');
        if (pos == std::string::npos) continue;
        std::string key = line.substr(0, pos);
        // remove quotes and spaces
        auto strip = [](std::string s) {
            while (!s.empty() && (s.front() == ' ' || s.front() == '"' || s.front() == '{' || s.front() == ',')) s.erase(s.begin());
            while (!s.empty() && (s.back() == ' ' || s.back() == '"' || s.back() == ',' || s.back() == '}')) s.pop_back();
            return s;
        };
        key = strip(key);
        std::string val = strip(line.substr(pos + 1));
        if (key == "n_tokens") n_tokens = std::stoi(val);
        else if (key == "nx") nx = std::stoi(val);
        else if (key == "ny") ny = std::stoi(val);
        else if (key == "embd_dim") embd_dim = std::stoi(val);
        else if (key == "use_mrope") use_mrope = (val == "true");
        else if (key == "use_non_causal") use_non_causal = (val == "true");
    }
    mif.close();

    if (n_tokens <= 0 || embd_dim <= 0) {
        return JNI_FALSE;
    }

    // Read embeddings
    std::ifstream ifs(embdC.get(), std::ios::binary | std::ios::ate);
    if (!ifs) {
        return JNI_FALSE;
    }
    std::streamsize size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    size_t expected = static_cast<size_t>(n_tokens) * static_cast<size_t>(embd_dim) * sizeof(float);
    if (static_cast<size_t>(size) < expected) {
        // mismatch
        ifs.close();
        return JNI_FALSE;
    }
    std::vector<float> embd_buf(n_tokens * embd_dim);
    ifs.read(reinterpret_cast<char*>(embd_buf.data()), expected);
    ifs.close();

    // Get llama_context from modelPtr
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return JNI_FALSE;
    }
    llama_context* lctx = llmInference->getContext();
    if (!lctx) {
        return JNI_FALSE;
    }

    if (!decodeEmbeddingsIntoKv(
                lctx,
                embd_buf.data(),
                n_tokens,
                embd_dim,
                nx,
                ny,
                use_mrope,
                use_non_causal,
                nBatch)) {
        return JNI_FALSE;
    }

    llmInference->markPreparedKvForNextCompletion();

    return JNI_TRUE;
}

// Retrieve the full state blob for the llama context (includes RNG, logits and KV cache)
extern "C" JNIEXPORT jbyteArray JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_nativeGetStateBytes(JNIEnv* env, jobject thiz, jlong modelPtr) {
    if (!modelPtr) return nullptr;
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return nullptr;
    }
    llama_context* ctx = llmInference->getContext();
    if (!ctx) return nullptr;
    size_t size = llama_state_get_size(ctx);
    if (size == 0) return nullptr;
    std::vector<uint8_t> buf(size);
    size_t written = llama_state_get_data(ctx, buf.data(), buf.size());
    if (written == 0) return nullptr;
    jbyteArray arr = env->NewByteArray(static_cast<jsize>(written));
    if (!arr) return nullptr;
    env->SetByteArrayRegion(arr, 0, static_cast<jsize>(written), reinterpret_cast<jbyte*>(buf.data()));
    return arr;
}

// Set the full state blob for the llama context
extern "C" JNIEXPORT jboolean JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_nativeSetStateBytes(JNIEnv* env, jobject thiz, jlong modelPtr, jbyteArray state) {
    if (!modelPtr || !state) return JNI_FALSE;
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return JNI_FALSE;
    }
    llama_context* ctx = llmInference->getContext();
    if (!ctx) return JNI_FALSE;
    jsize len = env->GetArrayLength(state);
    if (len <= 0) return JNI_FALSE;
    std::vector<uint8_t> buf(static_cast<size_t>(len));
    env->GetByteArrayRegion(state, 0, len, reinterpret_cast<jbyte*>(buf.data()));
    size_t written = llama_state_set_data(ctx, buf.data(), buf.size());
    return (written == static_cast<size_t>(len)) ? JNI_TRUE : JNI_FALSE;
}

// Get state for a specific sequence (KV slot)
extern "C" JNIEXPORT jbyteArray JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_nativeGetSequenceStateBytes(JNIEnv* env, jobject thiz, jlong modelPtr, jint seqId) {
    if (!modelPtr) return nullptr;
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return nullptr;
    }
    llama_context* ctx = llmInference->getContext();
    if (!ctx) return nullptr;
    size_t size = llmedge_state_seq_get_size(ctx, static_cast<llama_seq_id>(seqId));
    if (size == 0) return nullptr;
    std::vector<uint8_t> buf(size);
    size_t written = llmedge_state_seq_get_data(ctx, buf.data(), buf.size(), static_cast<llama_seq_id>(seqId));
    if (written == 0) return nullptr;
    jbyteArray arr = env->NewByteArray(static_cast<jsize>(written));
    if (!arr) return nullptr;
    env->SetByteArrayRegion(arr, 0, static_cast<jsize>(written), reinterpret_cast<jbyte*>(buf.data()));
    return arr;
}

// Set sequence state bytes (restore into KV slot)
extern "C" JNIEXPORT jboolean JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_nativeSetSequenceStateBytes(JNIEnv* env, jobject thiz, jlong modelPtr, jint seqId, jbyteArray state) {
    if (!modelPtr || !state) return JNI_FALSE;
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return JNI_FALSE;
    }
    llama_context* ctx = llmInference->getContext();
    if (!ctx) return JNI_FALSE;
    jsize len = env->GetArrayLength(state);
    if (len <= 0) return JNI_FALSE;
    std::vector<uint8_t> buf(static_cast<size_t>(len));
    env->GetByteArrayRegion(state, 0, len, reinterpret_cast<jbyte*>(buf.data()));
    size_t written = llmedge_state_seq_set_data(ctx, buf.data(), buf.size(), static_cast<llama_seq_id>(seqId));
    return (written == static_cast<size_t>(len)) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_nativeClearKvCache(JNIEnv* env, jobject thiz, jlong modelPtr) {
    if (!modelPtr) return;
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return;
    }
    llama_context* ctx = llmInference->getContext();
    if (!ctx) return;
    llmedge_kv_cache_clear(ctx);
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_nativeClearMessages(JNIEnv* env, jobject thiz, jlong modelPtr) {
    if (!modelPtr) return;
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return;
    }
    llmInference->clearMessages();
}
