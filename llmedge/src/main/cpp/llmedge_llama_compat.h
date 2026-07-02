#pragma once

#include "llama.h"

#include <cstdint>
#include <stdexcept>
#include <string>

#ifndef LLMEDGE_LLAMA_USE_MEMORY_API
#define LLMEDGE_LLAMA_USE_MEMORY_API 0
#endif

enum class llmedge_kv_cache_type_code : int32_t {
    DEFAULT = 0,
    F16 = 1,
    Q8_0 = 2,
    Q4_0 = 3,
    Q8_KV = 4,
};

inline ggml_type llmedge_resolve_kv_cache_type(int32_t code, const char * label) {
    switch (code) {
        case -1:
        case static_cast<int32_t>(llmedge_kv_cache_type_code::DEFAULT):
        case static_cast<int32_t>(llmedge_kv_cache_type_code::F16):
            return GGML_TYPE_F16;
        case static_cast<int32_t>(llmedge_kv_cache_type_code::Q8_0):
            return GGML_TYPE_Q8_0;
        case static_cast<int32_t>(llmedge_kv_cache_type_code::Q4_0):
            return GGML_TYPE_Q4_0;
        case static_cast<int32_t>(llmedge_kv_cache_type_code::Q8_KV):
            // GGML_TYPE_Q8_KV is an enum constant (not a macro), so it cannot be
            // feature-tested with #ifdef — that test was always false and disabled
            // Q8_KV entirely. The vendored ik_llama.cpp fork always provides it; a
            // submodule swap without it fails loudly at compile time instead.
            return GGML_TYPE_Q8_KV;
        default:
            throw std::runtime_error(std::string("Unknown llmedge KV cache type code for ") + label);
    }
}

inline void llmedge_kv_cache_clear(struct llama_context * ctx) {
#if LLMEDGE_LLAMA_USE_MEMORY_API
    llama_memory_clear(llama_get_memory(ctx), true);
#else
    llama_kv_cache_clear(ctx);
#endif
}

inline bool llmedge_kv_cache_seq_rm(struct llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
#if LLMEDGE_LLAMA_USE_MEMORY_API
    return llama_memory_seq_rm(llama_get_memory(ctx), seq_id, p0, p1);
#else
    return llama_kv_cache_seq_rm(ctx, seq_id, p0, p1);
#endif
}

inline llama_pos llmedge_kv_cache_seq_pos_max(struct llama_context * ctx, llama_seq_id seq_id) {
#if LLMEDGE_LLAMA_USE_MEMORY_API
    return llama_memory_seq_pos_max(llama_get_memory(ctx), seq_id);
#else
    return llama_kv_cache_seq_pos_max(ctx, seq_id);
#endif
}

inline size_t llmedge_state_seq_get_size(struct llama_context * ctx, llama_seq_id seq_id) {
    return llama_state_seq_get_size(ctx, seq_id, 0);
}

inline size_t llmedge_state_seq_get_data(struct llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_get_data(ctx, dst, size, seq_id, 0);
}

inline size_t llmedge_state_seq_set_data(struct llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_set_data(ctx, src, size, seq_id, 0);
}
