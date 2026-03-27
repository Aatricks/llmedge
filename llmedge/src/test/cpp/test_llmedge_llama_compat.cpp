#include "../../main/cpp/llmedge_llama_compat.h"

#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_stable_kv_cache_mapping() {
    require(
        llmedge_resolve_kv_cache_type(static_cast<int>(llmedge_kv_cache_type_code::DEFAULT), "k") == GGML_TYPE_F16,
        "DEFAULT KV cache type should map to F16");
    require(
        llmedge_resolve_kv_cache_type(static_cast<int>(llmedge_kv_cache_type_code::F16), "k") == GGML_TYPE_F16,
        "F16 KV cache type should map to F16");
    require(
        llmedge_resolve_kv_cache_type(static_cast<int>(llmedge_kv_cache_type_code::Q8_0), "k") == GGML_TYPE_Q8_0,
        "Q8_0 KV cache type should map to GGML_TYPE_Q8_0");
    require(
        llmedge_resolve_kv_cache_type(static_cast<int>(llmedge_kv_cache_type_code::Q4_0), "k") == GGML_TYPE_Q4_0,
        "Q4_0 KV cache type should map to GGML_TYPE_Q4_0");
#ifdef GGML_TYPE_Q8_KV
    require(
        llmedge_resolve_kv_cache_type(static_cast<int>(llmedge_kv_cache_type_code::Q8_KV), "k") == GGML_TYPE_Q8_KV,
        "Q8_KV KV cache type should map to GGML_TYPE_Q8_KV");
#endif
}

void test_unknown_kv_cache_mapping_throws() {
    bool threw = false;
    try {
        (void) llmedge_resolve_kv_cache_type(9999, "k");
    } catch (const std::exception &) {
        threw = true;
    }
    require(threw, "Unknown stable KV cache type should throw");
}

} // namespace

void test_llmedge_llama_compat() {
    test_stable_kv_cache_mapping();
    test_unknown_kv_cache_mapping_throws();
}
