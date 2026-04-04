#pragma once

#include "LLMInference.h"
#include "jni_utils.h"
#include "llmedge_llama_compat.h"

#include <jni.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

inline void throwJavaException(JNIEnv* env, const char* className, const char* message) {
    llmedge_throw_java_exception(env, className, message);
}

inline void throwInvalidHandle(JNIEnv* env, const char* owner) {
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

inline LLMInference* requireInference(JNIEnv* env, jlong modelPtr, const char* operation) {
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
    std::vector<llama_pos> pos;
    std::vector<llama_pos> pos_view;
    std::vector<int32_t> n_seq_id;
    std::vector<llama_seq_id> seq_id_0;
    std::vector<llama_seq_id*> seq_ids;
    std::vector<int8_t> logits;
    llama_batch batch {};

    DecodeEmbeddingsBatch(float* embd, int32_t n_tokens, int n_pos_per_token, int embd_dim) :
            n_pos_per_embd(n_pos_per_token), n_mmproj_embd(embd_dim) {
        pos.resize(static_cast<size_t>(n_tokens) * static_cast<size_t>(n_pos_per_embd));
        n_seq_id.resize(n_tokens);
        seq_id_0.resize(1);
        seq_ids.resize(static_cast<size_t>(n_tokens) + 1);
        logits.resize(n_tokens);
        seq_ids[n_tokens] = nullptr;
        batch = {
                n_tokens,
                nullptr,
                embd,
                pos.data(),
                n_seq_id.data(),
                seq_ids.data(),
                logits.data(),
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
        llama_pos* pos_ptr = nullptr;
        pos_view.clear();
        if (n_pos_per_embd > 1) {
            pos_view.reserve(static_cast<size_t>(n_tokens) * static_cast<size_t>(n_pos_per_embd));
            for (int i = 0; i < n_pos_per_embd; ++i) {
                const size_t src_idx =
                        static_cast<size_t>(i) * static_cast<size_t>(batch.n_tokens) + static_cast<size_t>(offset);
                pos_view.insert(pos_view.end(), pos.data() + src_idx, pos.data() + src_idx + n_tokens);
            }
            pos_ptr = pos_view.data();
        } else {
            pos_ptr = pos.data() + offset;
        }
        return {
                n_tokens,
                nullptr,
                batch.embd + static_cast<size_t>(offset) * static_cast<size_t>(n_mmproj_embd),
                pos_ptr,
                batch.n_seq_id + offset,
                batch.seq_id + offset,
                batch.logits + offset,
        };
    }
};

inline bool decodeEmbeddingsIntoKv(
        llama_context* lctx,
        float* embd,
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
