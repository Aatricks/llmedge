#include "hf_to_gguf.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gguf_writer.h"
#include "nlohmann/json.hpp"
#include "safetensors_reader.h"

namespace llmedge {
namespace convert {

using json = nlohmann::json;

uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (((x >> 23) & 0xff) == 0xff) {  // inf/nan
        return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0));
    }
    if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);  // overflow -> inf
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;  // underflow -> 0
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1);
        uint32_t halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half & 1))) half++;  // round to nearest even
        return (uint16_t)(sign | half);
    }
    uint16_t h = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
    uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) h++;  // round to nearest even (may carry into exp)
    return h;
}

uint16_t bf16_to_f16(uint16_t bf) {
    uint32_t f32 = (uint32_t)bf << 16;
    float f;
    std::memcpy(&f, &f32, 4);
    return f32_to_f16(f);
}

namespace {

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("convert: cannot read " + path);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// Convert a tensor's raw bytes to F16 (from BF16/F16/F32). Returns f16 bytes.
std::vector<uint8_t> to_f16(const std::vector<uint8_t>& src, StDType dt, int64_t n) {
    std::vector<uint8_t> out(n * 2);
    uint16_t* o = (uint16_t*)out.data();
    if (dt == StDType::BF16) {
        const uint16_t* s = (const uint16_t*)src.data();
        for (int64_t i = 0; i < n; ++i) o[i] = bf16_to_f16(s[i]);
    } else if (dt == StDType::F16) {
        std::memcpy(out.data(), src.data(), n * 2);
    } else if (dt == StDType::F32) {
        const float* s = (const float*)src.data();
        for (int64_t i = 0; i < n; ++i) o[i] = f32_to_f16(s[i]);
    } else {
        throw std::runtime_error("convert: unsupported source dtype for f16");
    }
    return out;
}

// Convert a tensor's raw bytes to F32 (from BF16/F16/F32).
std::vector<uint8_t> to_f32(const std::vector<uint8_t>& src, StDType dt, int64_t n) {
    std::vector<uint8_t> out(n * 4);
    float* o = (float*)out.data();
    if (dt == StDType::BF16) {
        const uint16_t* s = (const uint16_t*)src.data();
        for (int64_t i = 0; i < n; ++i) {
            uint32_t f = (uint32_t)s[i] << 16;
            std::memcpy(&o[i], &f, 4);
        }
    } else if (dt == StDType::F32) {
        std::memcpy(out.data(), src.data(), n * 4);
    } else if (dt == StDType::F16) {
        throw std::runtime_error("convert: f16->f32 not needed here");
    } else {
        throw std::runtime_error("convert: unsupported source dtype for f32");
    }
    return out;
}

// llama.cpp Q/K RoPE permutation, applied on a [out, in] row-major matrix.
// out rows are grouped per head; within each head the two halves are interleaved.
// Mirrors convert_hf_to_gguf.py LlamaModel.permute(weights, n_head).
std::vector<uint8_t> permute_rows(const std::vector<uint8_t>& data, int64_t out_rows, int64_t in_cols,
                                  size_t elt_size, int n_head) {
    if (out_rows % n_head != 0) throw std::runtime_error("permute: out not divisible by n_head");
    int64_t head_dim = out_rows / n_head;
    if (head_dim % 2 != 0) throw std::runtime_error("permute: head_dim not even");
    int64_t hd2 = head_dim / 2;
    int64_t row_bytes = in_cols * (int64_t)elt_size;
    std::vector<uint8_t> out(data.size());
    for (int h = 0; h < n_head; ++h) {
        for (int64_t a = 0; a < 2; ++a) {
            for (int64_t j = 0; j < hd2; ++j) {
                int64_t src_row = h * head_dim + a * hd2 + j;       // input layout (n_head,2,hd2)
                int64_t dst_row = h * head_dim + j * 2 + a;          // output layout (n_head,hd2,2)
                std::memcpy(out.data() + dst_row * row_bytes,
                            data.data() + src_row * row_bytes, row_bytes);
            }
        }
    }
    return out;
}

}  // namespace

size_t convert_llama_dir(const std::string& model_dir, const std::string& out_path) {
    json cfg = json::parse(read_file(model_dir + "/config.json"));
    const std::string arch = cfg.value("model_type", "");
    if (arch != "llama") throw std::runtime_error("convert v1 supports model_type=llama only, got: " + arch);

    const int n_embd = cfg.at("hidden_size");
    const int n_layer = cfg.at("num_hidden_layers");
    const int n_head = cfg.at("num_attention_heads");
    const int n_head_kv = cfg.value("num_key_value_heads", n_head);
    const int n_ff = cfg.at("intermediate_size");
    const int n_vocab = cfg.at("vocab_size");
    const int n_ctx = cfg.value("max_position_embeddings", 2048);
    const float rms_eps = cfg.value("rms_norm_eps", 1e-5);
    const float rope_theta = cfg.value("rope_theta", 10000.0);
    const int head_dim = n_embd / n_head;

    SafetensorsFile st = read_safetensors_header(model_dir + "/model.safetensors");

    GgufWriter w;
    w.set_str("general.architecture", "llama");
    w.set_str("general.name", "converted");
    w.set_u32("llama.context_length", (uint32_t)n_ctx);
    w.set_u32("llama.embedding_length", (uint32_t)n_embd);
    w.set_u32("llama.block_count", (uint32_t)n_layer);
    w.set_u32("llama.feed_forward_length", (uint32_t)n_ff);
    w.set_u32("llama.attention.head_count", (uint32_t)n_head);
    w.set_u32("llama.attention.head_count_kv", (uint32_t)n_head_kv);
    w.set_f32("llama.attention.layer_norm_rms_epsilon", rms_eps);
    w.set_u32("llama.rope.dimension_count", (uint32_t)head_dim);
    w.set_f32("llama.rope.freq_base", rope_theta);
    w.set_u32("llama.vocab_size", (uint32_t)n_vocab);

    auto find = [&](const std::string& name) -> const StTensor* {
        for (const auto& t : st.tensors)
            if (t.name == name) return &t;
        return nullptr;
    };

    // Emit one tensor: name mapping handled by caller; perm_heads>0 applies Q/K permutation;
    // 1-D tensors stay F32, 2-D weights become F16 (values compared dtype-agnostically by the oracle).
    size_t written = 0;
    auto emit = [&](const std::string& hf_name, const std::string& gguf_name, int perm_heads) {
        const StTensor* t = find(hf_name);
        if (!t) throw std::runtime_error("convert: missing tensor " + hf_name);
        int64_t n = st_num_elements(*t);
        std::vector<uint8_t> raw = st_read_tensor_bytes(st, *t);
        const bool is_1d = t->shape.size() == 1;

        if (perm_heads > 0) {
            int64_t out_rows = t->shape[0], in_cols = t->shape[1];
            raw = permute_rows(raw, out_rows, in_cols, st_dtype_size(t->dtype), perm_heads);
        }

        std::vector<int64_t> ne;  // ggml order: ne[0] = fastest (in), ne[1] = out
        for (auto it = t->shape.rbegin(); it != t->shape.rend(); ++it) ne.push_back(*it);

        if (is_1d) {
            std::vector<uint8_t> f32 = to_f32(raw, t->dtype, n);
            w.add_tensor(gguf_name, GgmlType::F32, ne, f32.data(), f32.size());
        } else {
            std::vector<uint8_t> f16 = to_f16(raw, t->dtype, n);
            w.add_tensor(gguf_name, GgmlType::F16, ne, f16.data(), f16.size());
        }
        ++written;
    };

    emit("model.embed_tokens.weight", "token_embd.weight", 0);
    emit("model.norm.weight", "output_norm.weight", 0);
    if (find("lm_head.weight")) emit("lm_head.weight", "output.weight", 0);  // absent when tied

    for (int i = 0; i < n_layer; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        const std::string b = "blk." + std::to_string(i) + ".";
        emit(p + "self_attn.q_proj.weight", b + "attn_q.weight", n_head);
        emit(p + "self_attn.k_proj.weight", b + "attn_k.weight", n_head_kv);
        emit(p + "self_attn.v_proj.weight", b + "attn_v.weight", 0);
        emit(p + "self_attn.o_proj.weight", b + "attn_output.weight", 0);
        emit(p + "mlp.gate_proj.weight", b + "ffn_gate.weight", 0);
        emit(p + "mlp.up_proj.weight", b + "ffn_up.weight", 0);
        emit(p + "mlp.down_proj.weight", b + "ffn_down.weight", 0);
        emit(p + "input_layernorm.weight", b + "attn_norm.weight", 0);
        emit(p + "post_attention_layernorm.weight", b + "ffn_norm.weight", 0);
    }

    w.write(out_path, 32);
    return written;
}

}  // namespace convert
}  // namespace llmedge
