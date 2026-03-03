#include "LLMInference.h"
#include <jni.h>
#include <fstream>
#include <memory>
#include <mutex>
#include <unordered_map>

// Include libmtmd headers from llama.cpp to enable projector-based encoding
#include "../../../../llama.cpp/tools/mtmd/mtmd.h"
#include "../../../../llama.cpp/tools/mtmd/mtmd-helper.h"

extern "C" JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_SmolLM_loadModel(JNIEnv* env, jobject thiz, jstring modelPath, jfloat minP,
                                            jfloat temperature, jboolean storeChats, jlong contextSize,
                                            jstring chatTemplate, jint nThreads, jboolean useMmap, jboolean useMlock,
                                            jboolean useVulkan) {
    jboolean    isCopy           = true;
    const char* modelPathCstr    = env->GetStringUTFChars(modelPath, &isCopy);
    auto*       llmInference     = new LLMInference();
    const char* chatTemplateCstr = env->GetStringUTFChars(chatTemplate, &isCopy);

    try {
        llmInference->loadModel(modelPathCstr, minP, temperature, storeChats, contextSize, chatTemplateCstr, nThreads,
                                useMmap, useMlock, useVulkan);
    } catch (std::runtime_error& error) {
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), error.what());
    }

    env->ReleaseStringUTFChars(modelPath, modelPathCstr);
    env->ReleaseStringUTFChars(chatTemplate, chatTemplateCstr);
    return reinterpret_cast<jlong>(llmInference);
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_SmolLM_addChatMessage(JNIEnv* env, jobject thiz, jlong modelPtr, jstring message,
                                                 jstring role) {
    jboolean    isCopy       = true;
    const char* messageCstr  = env->GetStringUTFChars(message, &isCopy);
    const char* roleCstr     = env->GetStringUTFChars(role, &isCopy);
    auto*       llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    llmInference->addChatMessage(messageCstr, roleCstr);
    env->ReleaseStringUTFChars(message, messageCstr);
    env->ReleaseStringUTFChars(role, roleCstr);
}

extern "C" JNIEXPORT jfloat JNICALL
Java_io_aatricks_llmedge_SmolLM_getResponseGenerationSpeed(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    return llmInference->getResponseGenerationTime();
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_SmolLM_getResponseGeneratedTokenCount(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    return llmInference->getResponseTokenCount();
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_SmolLM_getResponseGenerationDurationMicros(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    return llmInference->getResponseGenerationTimeMicros();
}

extern "C" JNIEXPORT jint JNICALL
Java_io_aatricks_llmedge_SmolLM_getContextSizeUsed(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    return llmInference->getContextSizeUsed();
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_SmolLM_close(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    delete llmInference;
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_SmolLM_startCompletion(JNIEnv* env, jobject thiz, jlong modelPtr, jstring prompt) {
    jboolean    isCopy       = true;
    const char* promptCstr   = env->GetStringUTFChars(prompt, &isCopy);
    auto*       llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    llmInference->startCompletion(promptCstr);
    env->ReleaseStringUTFChars(prompt, promptCstr);
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_SmolLM_setReasoningOptions(JNIEnv* env, jobject thiz, jlong modelPtr, jboolean disableThinking,
                                                    jint reasoningBudget) {
    auto* llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    if (llmInference == nullptr) {
        return;
    }
    const bool disable = disableThinking == JNI_TRUE;
    llmInference->setReasoningOptions(disable, reasoningBudget);
}

extern "C" JNIEXPORT jstring JNICALL
Java_io_aatricks_llmedge_SmolLM_completionLoop(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    try {
        std::string response = llmInference->completionLoop();
        return env->NewStringUTF(response.c_str());
    } catch (std::runtime_error& error) {
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), error.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_io_aatricks_llmedge_SmolLM_completionLoopBatch(JNIEnv* env, jobject thiz, jlong modelPtr, jint maxTokens) {
    auto* llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    try {
        std::string response = llmInference->completionLoopBatch(maxTokens);
        return env->NewStringUTF(response.c_str());
    } catch (std::runtime_error& error) {
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), error.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_SmolLM_setThreadAffinity(JNIEnv* env, jobject thiz, jlong modelPtr, jlong coreMask) {
    auto* llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    if (llmInference) {
        llmInference->setThreadAffinity(static_cast<uint64_t>(coreMask));
    }
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_SmolLM_stopCompletion(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    llmInference->stopCompletion();
}

// Projector JNI stubs used by Projector.kt. These are lightweight placeholders
// so the example can demonstrate the safe sequencing of projector usage.
// Map to keep model pointer associated with mtmd_context (for embd dim lookup)
static std::unordered_map<mtmd_context*, llama_model*> g_mtmd_model_map;
static std::mutex g_mtmd_map_mutex;

extern "C" JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_Projector_nativeInitProjector(JNIEnv* env, jobject thiz, jstring mmprojPath, jlong textModelPtr) {
    const char* mmprojC = nullptr;
    mtmd_context* ctx = nullptr;
    if (mmprojPath == nullptr) return 0;
    mmprojC = env->GetStringUTFChars(mmprojPath, nullptr);
    if (!mmprojC) return 0;

    mtmd_context_params params = mtmd_context_params_default();
    params.use_gpu = false; // don't attempt GPU for Android example

    try {
        ctx = mtmd_init_from_file(mmprojC, reinterpret_cast<const llama_model*>(textModelPtr), params);
    } catch (...) {
        ctx = nullptr;
    }

    if (ctx && textModelPtr != 0) {
        std::lock_guard<std::mutex> lk(g_mtmd_map_mutex);
        g_mtmd_model_map[ctx] = reinterpret_cast<llama_model*>(textModelPtr);
    }

    env->ReleaseStringUTFChars(mmprojPath, mmprojC);

    return reinterpret_cast<jlong>(ctx);
}

extern "C" JNIEXPORT jboolean JNICALL
Java_io_aatricks_llmedge_Projector_nativeEncodeImage(JNIEnv* env, jobject thiz, jlong nativePtr, jstring imagePath, jstring outPath) {
    const char* inC = env->GetStringUTFChars(imagePath, nullptr);
    const char* outC = env->GetStringUTFChars(outPath, nullptr);
    if (!inC || !outC) {
        if (inC) env->ReleaseStringUTFChars(imagePath, inC);
        if (outC) env->ReleaseStringUTFChars(outPath, outC);
        return JNI_FALSE;
    }

    mtmd_context* ctx = reinterpret_cast<mtmd_context*>(nativePtr);
    bool ok = false;

    if (ctx == nullptr) {
        // No projector available; fallback to copying file
        std::ifstream src(inC, std::ios::binary);
        std::ofstream dst(outC, std::ios::binary);
        if (src && dst) {
            dst << src.rdbuf();
            ok = static_cast<bool>(src) && static_cast<bool>(dst);
        }
        env->ReleaseStringUTFChars(imagePath, inC);
        env->ReleaseStringUTFChars(outPath, outC);
        return ok ? JNI_TRUE : JNI_FALSE;
    }

    // Use mtmd_helper_bitmap_init_from_file to load image and preprocess it
    mtmd_bitmap* bmp = mtmd_helper_bitmap_init_from_file(ctx, inC);
    if (!bmp) {
        env->ReleaseStringUTFChars(imagePath, inC);
        env->ReleaseStringUTFChars(outPath, outC);
        return JNI_FALSE;
    }

    const mtmd_bitmap* bitmaps[1] = { bmp };
    mtmd_input_text txt = { "<__media__>", false, false };
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    int32_t tokRes = mtmd_tokenize(ctx, chunks, &txt, bitmaps, 1);
    if (tokRes != 0) {
        mtmd_bitmap_free(bmp);
        mtmd_input_chunks_free(chunks);
        env->ReleaseStringUTFChars(imagePath, inC);
        env->ReleaseStringUTFChars(outPath, outC);
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
                std::ofstream ofs(outC, std::ios::binary);
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
                            env->ReleaseStringUTFChars(imagePath, inC);
                            env->ReleaseStringUTFChars(outPath, outC);
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
            std::string metaPath = std::string(outC) + ".meta.json";
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
    env->ReleaseStringUTFChars(imagePath, inC);
    env->ReleaseStringUTFChars(outPath, outC);

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

// Return the internal llama_model* as jlong for advanced native integrations (caller must not free)
extern "C" JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_SmolLM_getNativeModelPtr(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    if (!llmInference) return 0;
    return reinterpret_cast<jlong>(llmInference->getModel());
}

// Decode prepared embeddings (.bin) using the already-loaded llama_context inside LLMInference
extern "C" JNIEXPORT jboolean JNICALL
Java_io_aatricks_llmedge_SmolLM_nativeDecodePreparedEmbeddings(JNIEnv* env, jobject thiz, jlong modelPtr,
                                                               jstring embdPath, jstring metaPath, jint nBatch) {
    if (!embdPath || !metaPath) return JNI_FALSE;
    const char* embdC = env->GetStringUTFChars(embdPath, nullptr);
    const char* metaC = env->GetStringUTFChars(metaPath, nullptr);
    if (!embdC || !metaC) {
        if (embdC) env->ReleaseStringUTFChars(embdPath, embdC);
        if (metaC) env->ReleaseStringUTFChars(metaPath, metaC);
        return JNI_FALSE;
    }

    // Read metadata JSON (very small) - parse manually
    int n_tokens = 0;
    int nx = 0, ny = 0;
    int embd_dim = 0;
    bool use_mrope = false;
    bool use_non_causal = false;

    std::ifstream mif(metaC);
    if (!mif) {
        env->ReleaseStringUTFChars(embdPath, embdC);
        env->ReleaseStringUTFChars(metaPath, metaC);
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
        env->ReleaseStringUTFChars(embdPath, embdC);
        env->ReleaseStringUTFChars(metaPath, metaC);
        return JNI_FALSE;
    }

    // Read embeddings
    std::ifstream ifs(embdC, std::ios::binary | std::ios::ate);
    if (!ifs) {
        env->ReleaseStringUTFChars(embdPath, embdC);
        env->ReleaseStringUTFChars(metaPath, metaC);
        return JNI_FALSE;
    }
    std::streamsize size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    size_t expected = static_cast<size_t>(n_tokens) * static_cast<size_t>(embd_dim) * sizeof(float);
    if (static_cast<size_t>(size) < expected) {
        // mismatch
        ifs.close();
        env->ReleaseStringUTFChars(embdPath, embdC);
        env->ReleaseStringUTFChars(metaPath, metaC);
        return JNI_FALSE;
    }
    std::vector<float> embd_buf(n_tokens * embd_dim);
    ifs.read(reinterpret_cast<char*>(embd_buf.data()), expected);
    ifs.close();

    // Get llama_context from modelPtr
    auto* llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    if (!llmInference) {
        env->ReleaseStringUTFChars(embdPath, embdC);
        env->ReleaseStringUTFChars(metaPath, metaC);
        return JNI_FALSE;
    }
    llama_context* lctx = llmInference->getContext();
    if (!lctx) {
        env->ReleaseStringUTFChars(embdPath, embdC);
        env->ReleaseStringUTFChars(metaPath, metaC);
        return JNI_FALSE;
    }

    // Prepare batch decoding similar to mtmd_helper_decode_image_chunk
    int n_pos_per_embd = use_mrope ? 4 : 1;
    int n_mmproj_embd = embd_dim;
    int32_t i_batch = 0;
    int32_t n_img_batches = (n_tokens + nBatch - 1) / nBatch;

    // Helper to run llama_decode on a portion of embeddings
    auto run_decode_batch = [&](int offset, int n_tokens_batch) -> bool {
        // create a llama_batch that references the right slice of embd_buf
        llama_batch batch = llama_batch_init(n_tokens_batch, 0, 1);
        // tokens are not used; set embd pointer to slice
        batch.embd = embd_buf.data() + static_cast<size_t>(offset) * static_cast<size_t>(n_mmproj_embd);
        // set pos array
        std::vector<llama_pos> pos(n_tokens_batch * n_pos_per_embd);
        if (n_pos_per_embd == 1) {
            for (int i = 0; i < n_tokens_batch; ++i) pos[i] = static_cast<llama_pos>(offset + i);
        } else {
            // mrope 2d: try to reconstruct as row-major
            // If nx/ny are provided, use them; otherwise treat as linear
            if (nx > 0 && ny > 0) {
                for (int y = 0; y < ny; ++y) {
                    for (int x = 0; x < nx; ++x) {
                        int idx = y * nx + x;
                        if (idx < offset || idx >= offset + n_tokens_batch) continue;
                        int out_idx = idx - offset;
                        pos[out_idx] = static_cast<llama_pos>(0 + idx);
                        // fill the other dims similarly (pos array will be expanded later)
                    }
                }
            } else {
                for (int i = 0; i < n_tokens_batch; ++i) {
                    // fallback mapping
                    pos[i] = static_cast<llama_pos>(offset + i);
                }
            }
        }

        // We will call llama_decode with a batch that has embd pointer and pos filled
        // Note: llama_decode expects a llama_batch struct; here we craft minimal fields
        llama_batch decode_batch = {
            /*n_tokens=*/ n_tokens_batch,
            /*token=*/ nullptr,
            /*embd=*/ batch.embd,
            /*pos=*/ pos.data(),
            /*n_seq_id=*/ nullptr,
            /*seq_id=*/ nullptr,
            /*logits=*/ nullptr,
        };

        int32_t ret = llama_decode(lctx, decode_batch);
        return ret == 0;
    };

    while (i_batch < n_img_batches) {
        int pos_offset = i_batch * nBatch;
        int n_tokens_batch = std::min(static_cast<int>(nBatch), n_tokens - pos_offset);
        bool ok = run_decode_batch(pos_offset, n_tokens_batch);
        if (!ok) {
            env->ReleaseStringUTFChars(embdPath, embdC);
            env->ReleaseStringUTFChars(metaPath, metaC);
            return JNI_FALSE;
        }
        i_batch++;
    }

    env->ReleaseStringUTFChars(embdPath, embdC);
    env->ReleaseStringUTFChars(metaPath, metaC);
    return JNI_TRUE;
}

// Retrieve the full state blob for the llama context (includes RNG, logits and KV cache)
extern "C" JNIEXPORT jbyteArray JNICALL
Java_io_aatricks_llmedge_SmolLM_nativeGetStateBytes(JNIEnv* env, jobject thiz, jlong modelPtr) {
    if (!modelPtr) return nullptr;
    auto* llmInference = reinterpret_cast<LLMInference*>(modelPtr);
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
Java_io_aatricks_llmedge_SmolLM_nativeSetStateBytes(JNIEnv* env, jobject thiz, jlong modelPtr, jbyteArray state) {
    if (!modelPtr || !state) return JNI_FALSE;
    auto* llmInference = reinterpret_cast<LLMInference*>(modelPtr);
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
Java_io_aatricks_llmedge_SmolLM_nativeGetSequenceStateBytes(JNIEnv* env, jobject thiz, jlong modelPtr, jint seqId) {
    if (!modelPtr) return nullptr;
    auto* llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    llama_context* ctx = llmInference->getContext();
    if (!ctx) return nullptr;
    size_t size = llama_state_seq_get_size(ctx, static_cast<llama_seq_id>(seqId));
    if (size == 0) return nullptr;
    std::vector<uint8_t> buf(size);
    size_t written = llama_state_seq_get_data(ctx, buf.data(), buf.size(), static_cast<llama_seq_id>(seqId));
    if (written == 0) return nullptr;
    jbyteArray arr = env->NewByteArray(static_cast<jsize>(written));
    if (!arr) return nullptr;
    env->SetByteArrayRegion(arr, 0, static_cast<jsize>(written), reinterpret_cast<jbyte*>(buf.data()));
    return arr;
}

// Set sequence state bytes (restore into KV slot)
extern "C" JNIEXPORT jboolean JNICALL
Java_io_aatricks_llmedge_SmolLM_nativeSetSequenceStateBytes(JNIEnv* env, jobject thiz, jlong modelPtr, jint seqId, jbyteArray state) {
    if (!modelPtr || !state) return JNI_FALSE;
    auto* llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    llama_context* ctx = llmInference->getContext();
    if (!ctx) return JNI_FALSE;
    jsize len = env->GetArrayLength(state);
    if (len <= 0) return JNI_FALSE;
    std::vector<uint8_t> buf(static_cast<size_t>(len));
    env->GetByteArrayRegion(state, 0, len, reinterpret_cast<jbyte*>(buf.data()));
    size_t written = llama_state_seq_set_data(ctx, buf.data(), buf.size(), static_cast<llama_seq_id>(seqId));
    return (written == static_cast<size_t>(len)) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_io_aatricks_llmedge_SmolLM_nativeClearKvCache(JNIEnv* env, jobject thiz, jlong modelPtr) {
    if (!modelPtr) return;
    auto* llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    llama_context* ctx = llmInference->getContext();
    if (!ctx) return;
    llama_memory_clear(llama_get_memory(ctx), true);
}