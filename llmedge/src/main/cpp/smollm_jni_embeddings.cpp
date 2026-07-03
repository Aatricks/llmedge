#include "smollm_jni_shared.h"

#include <fstream>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "ggml-backend.h"
#include "mtmd-helper.h"
#include "mtmd.h"

namespace {
std::unordered_map<mtmd_context*, llama_model*> g_mtmd_model_map;
std::mutex g_mtmd_map_mutex;
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_Projector_nativeInitProjector(JNIEnv* env, jobject thiz, jstring mmprojPath, jlong textModelPtr) {
    mtmd_context* ctx = nullptr;
    if (mmprojPath == nullptr) return 0;
    ScopedUtfChars mmprojC(env, mmprojPath);
    if (!mmprojC.ok()) return 0;

    mtmd_context_params params = mtmd_context_params_default();
    params.use_gpu = false;

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
        std::ifstream src(inC.get(), std::ios::binary);
        std::ofstream dst(outC.get(), std::ios::binary);
        if (src && dst) {
            dst << src.rdbuf();
            ok = static_cast<bool>(src) && static_cast<bool>(dst);
        }
        return ok ? JNI_TRUE : JNI_FALSE;
    }

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

    bool encoded = false;
    size_t n_tokens = 0;
    int embd_dim = 0;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
        const mtmd_input_chunk* c = mtmd_input_chunks_get(chunks, i);
        if (c && mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            int32_t res = mtmd_encode_chunk(ctx, c);
            if (res == 0) {
                float* embd = mtmd_get_output_embd(ctx);
                std::ofstream ofs(outC.get(), std::ios::binary);
                if (ofs) {
                    n_tokens = static_cast<size_t>(mtmd_input_chunk_get_n_tokens(c));
                    {
                        std::lock_guard<std::mutex> lk(g_mtmd_map_mutex);
                        auto it = g_mtmd_model_map.find(ctx);
                        if (it != g_mtmd_model_map.end() && it->second) {
                            embd_dim = llama_model_n_embd(it->second);
                        }
                    }
                    if (embd_dim <= 0) {
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

    if (encoded) {
        std::string metaPath = std::string(outC.get()) + ".meta.json";
        std::ofstream mofs(metaPath, std::ios::trunc);
        if (mofs) {
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

extern "C" JNIEXPORT jobjectArray JNICALL
Java_io_aatricks_llmedge_vision_Projector_nativeEncodeImageBuffer(JNIEnv* env, jobject thiz, jlong nativePtr, jbyteArray jpegData) {
    if (!jpegData) return nullptr;

    mtmd_context* ctx = reinterpret_cast<mtmd_context*>(nativePtr);
    if (!ctx) return nullptr;

    jsize dataLen = env->GetArrayLength(jpegData);
    if (dataLen <= 0) return nullptr;

    jbyte* rawBytes = env->GetByteArrayElements(jpegData, nullptr);
    if (!rawBytes) return nullptr;

    mtmd_bitmap* bmp = mtmd_helper_bitmap_init_from_buf(ctx, reinterpret_cast<const unsigned char*>(rawBytes), static_cast<size_t>(dataLen));
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

    float* embd = nullptr;
    size_t n_tokens = 0;
    int embd_dim = 0;
    int nx = 0;
    int ny = 0;
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

    jfloatArray embdArray = env->NewFloatArray(static_cast<jsize>(n_floats));
    if (!embdArray) {
        mtmd_bitmap_free(bmp);
        mtmd_input_chunks_free(chunks);
        return nullptr;
    }
    env->SetFloatArrayRegion(embdArray, 0, static_cast<jsize>(n_floats), embd);

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

    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return JNI_FALSE;
    }

    llama_context* lctx = llmInference->getContext();
    mtmd_context* mtmd = reinterpret_cast<mtmd_context*>(projectorPtr);
    if (!lctx || !mtmd) {
        return JNI_FALSE;
    }

    const jsize data_len = env->GetArrayLength(imageData);
    if (data_len <= 0) {
        return JNI_FALSE;
    }

    jbyte* raw_bytes = env->GetByteArrayElements(imageData, nullptr);
    if (!raw_bytes) {
        return JNI_FALSE;
    }

    mtmd_bitmap* bmp = mtmd_helper_bitmap_init_from_buf(mtmd, reinterpret_cast<const unsigned char*>(raw_bytes), static_cast<size_t>(data_len));
    env->ReleaseByteArrayElements(imageData, raw_bytes, JNI_ABORT);
    if (!bmp) {
        return JNI_FALSE;
    }

    const mtmd_bitmap* bitmaps[1] = { bmp };
    mtmd_input_text txt = { "<__media__>", false, false };
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
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
        const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks, i);
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
        // Roll back partially primed chunks so the caller's fallback path doesn't
        // decode a second copy of the image on top of a half-decoded one.
        llmedge_kv_cache_seq_rm(lctx, 0, n_past, -1);
        return JNI_FALSE;
    }

    llmInference->markPreparedKvForNextCompletion();
    return JNI_TRUE;
}

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

extern "C" JNIEXPORT jlong JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_getNativeModelPtr(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) return 0;
    return reinterpret_cast<jlong>(llmInference->getModel());
}

extern "C" JNIEXPORT jboolean JNICALL
Java_io_aatricks_llmedge_text_runtime_SmolLM_nativeDecodePreparedEmbeddings(JNIEnv* env, jobject thiz, jlong modelPtr,
                                                               jstring embdPath, jstring metaPath, jint nBatch) {
    if (!embdPath || !metaPath) return JNI_FALSE;
    ScopedUtfChars embdC(env, embdPath);
    ScopedUtfChars metaC(env, metaPath);
    if (!embdC.ok() || !metaC.ok()) {
        return JNI_FALSE;
    }

    int n_tokens = 0;
    int nx = 0;
    int ny = 0;
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

    std::ifstream ifs(embdC.get(), std::ios::binary | std::ios::ate);
    if (!ifs) {
        return JNI_FALSE;
    }
    std::streamsize size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    size_t expected = static_cast<size_t>(n_tokens) * static_cast<size_t>(embd_dim) * sizeof(float);
    if (static_cast<size_t>(size) < expected) {
        ifs.close();
        return JNI_FALSE;
    }
    std::vector<float> embd_buf(n_tokens * embd_dim);
    ifs.read(reinterpret_cast<char*>(embd_buf.data()), expected);
    ifs.close();

    auto* llmInference = requireInference(env, modelPtr, "SmolLM model is not loaded");
    if (!llmInference) {
        return JNI_FALSE;
    }
    llama_context* lctx = llmInference->getContext();
    if (!lctx) {
        return JNI_FALSE;
    }

    if (!decodeEmbeddingsIntoKv(lctx, embd_buf.data(), n_tokens, embd_dim, nx, ny, use_mrope, use_non_causal, nBatch)) {
        return JNI_FALSE;
    }

    llmInference->markPreparedKvForNextCompletion();
    return JNI_TRUE;
}
