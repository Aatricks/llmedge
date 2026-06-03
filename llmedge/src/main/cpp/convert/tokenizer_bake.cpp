#include "tokenizer_bake.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "gguf_writer.h"
#include "nlohmann/json.hpp"

namespace llmedge {
namespace convert {

using json = nlohmann::json;

namespace {

// GGUF token types (gguf.TokenType in gguf-py / llama.cpp llama_token_type).
constexpr int32_t TOKEN_TYPE_NORMAL = 1;
constexpr int32_t TOKEN_TYPE_CONTROL = 3;

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("tokenizer: cannot read " + path);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// A token's content can be stored as a bare string or as an object {"content": "...", ...}.
std::string token_content(const json& v) {
    if (v.is_string()) return v.get<std::string>();
    if (v.is_object() && v.contains("content") && v["content"].is_string()) return v["content"].get<std::string>();
    return std::string();
}

bool get_bool(const json& o, const char* key, bool def) {
    return (o.contains(key) && o[key].is_boolean()) ? o[key].get<bool>() : def;
}

}  // namespace

void bake_gpt2_tokenizer(GgufWriter& w, const std::string& model_dir, const std::string& pre) {
    if (pre.empty()) {
        throw std::runtime_error(
            "tokenizer: tokenizer.ggml.pre hint is required (e.g. \"smollm\"); refusing to guess "
            "(a wrong pre-tokenizer loads silently and mis-tokenizes)");
    }

    json tj = json::parse(read_file(model_dir + "/tokenizer.json"));
    const json& model = tj.at("model");
    const std::string type = model.value("type", "");
    if (type != "BPE") {
        throw std::runtime_error("tokenizer: v1 bakes BPE tokenizers only, got model.type=" + type);
    }

    // ---- vocab: {token -> id}, must cover ids 0..N-1 contiguously ----
    const json& vocab = model.at("vocab");
    const int n = (int)vocab.size();
    std::vector<std::string> tokens(n);
    std::vector<bool> seen(n, false);
    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
        const int id = it.value().get<int>();
        if (id < 0 || id >= n) throw std::runtime_error("tokenizer: vocab id out of range: " + std::to_string(id));
        if (seen[id]) throw std::runtime_error("tokenizer: duplicate vocab id: " + std::to_string(id));
        tokens[id] = it.key();
        seen[id] = true;
    }
    for (int i = 0; i < n; ++i) {
        if (!seen[i]) throw std::runtime_error("tokenizer: vocab ids not contiguous (gap at " + std::to_string(i) + ")");
    }

    // ---- token_type: NORMAL, overridden to CONTROL for special added tokens ----
    std::vector<int32_t> token_type(n, TOKEN_TYPE_NORMAL);
    if (tj.contains("added_tokens")) {
        for (const json& at : tj["added_tokens"]) {
            const int id = at.at("id").get<int>();
            if (id < 0 || id >= n) {
                throw std::runtime_error("tokenizer: added token id out of range: " + std::to_string(id));
            }
            const std::string content = at.value("content", std::string());
            if (!content.empty() && tokens[id] != content) {
                throw std::runtime_error("tokenizer: added token content mismatch at id " + std::to_string(id));
            }
            if (at.value("special", false)) token_type[id] = TOKEN_TYPE_CONTROL;
        }
    }

    // ---- merges: space-joined string pairs, verbatim ----
    const json& merges_j = model.at("merges");
    std::vector<std::string> merges;
    merges.reserve(merges_j.size());
    for (const json& m : merges_j) {
        if (!m.is_string()) {
            throw std::runtime_error(
                "tokenizer: v1 expects space-joined string merges; got the array-pair form (unsupported)");
        }
        merges.push_back(m.get<std::string>());
    }

    // ---- special-token ids + flags + chat template from tokenizer_config.json (+ special_tokens_map) ----
    json cfg = json::parse(read_file(model_dir + "/tokenizer_config.json"));
    json stm;
    {
        std::ifstream stm_in(model_dir + "/special_tokens_map.json", std::ios::binary);
        if (stm_in) stm = json::parse(std::string((std::istreambuf_iterator<char>(stm_in)), std::istreambuf_iterator<char>()));
    }

    std::unordered_map<std::string, int> tok2id;
    tok2id.reserve(n * 2);
    for (int i = 0; i < n; ++i) tok2id.emplace(tokens[i], i);

    // Resolve a special token's id by its content string, preferring tokenizer_config then special_tokens_map.
    auto resolve_id = [&](const char* key) -> long {
        std::string content;
        if (cfg.contains(key)) content = token_content(cfg[key]);
        if (content.empty() && stm.is_object() && stm.contains(key)) content = token_content(stm[key]);
        if (content.empty()) return -1;
        auto found = tok2id.find(content);
        return found == tok2id.end() ? -1 : (long)found->second;
    };
    const long bos = resolve_id("bos_token");
    const long eos = resolve_id("eos_token");
    const long unk = resolve_id("unk_token");
    const long pad = resolve_id("pad_token");

    const bool add_bos = get_bool(cfg, "add_bos_token", false);
    const bool add_space_prefix = get_bool(cfg, "add_prefix_space", false);

    // ---- emit ----
    w.set_str("tokenizer.ggml.model", "gpt2");
    w.set_str("tokenizer.ggml.pre", pre);
    w.set_arr_str("tokenizer.ggml.tokens", tokens);
    w.set_arr_i32("tokenizer.ggml.token_type", token_type);
    w.set_arr_str("tokenizer.ggml.merges", merges);
    if (bos >= 0) w.set_u32("tokenizer.ggml.bos_token_id", (uint32_t)bos);
    if (eos >= 0) w.set_u32("tokenizer.ggml.eos_token_id", (uint32_t)eos);
    if (unk >= 0) w.set_u32("tokenizer.ggml.unknown_token_id", (uint32_t)unk);
    if (pad >= 0) w.set_u32("tokenizer.ggml.padding_token_id", (uint32_t)pad);
    w.set_bool("tokenizer.ggml.add_space_prefix", add_space_prefix);
    w.set_bool("tokenizer.ggml.add_bos_token", add_bos);

    if (cfg.contains("chat_template") && cfg["chat_template"].is_string()) {
        w.set_str("tokenizer.chat_template", cfg["chat_template"].get<std::string>());
    }
}

namespace {

// A byte-fallback token "<0xNN>" (exactly 6 chars: '<','0','x',hex,hex,'>').
bool is_byte_token(const std::string& t) {
    auto hex = [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
    };
    return t.size() == 6 && t[0] == '<' && t[1] == '0' && t[2] == 'x' && t[5] == '>' && hex(t[3]) && hex(t[4]);
}

}  // namespace

void bake_llama_tokenizer(GgufWriter& w, const std::string& model_dir) {
    // GGUF token types.
    constexpr int32_t TYPE_NORMAL = 1;
    constexpr int32_t TYPE_CONTROL = 3;
    constexpr int32_t TYPE_BYTE = 6;

    json tj = json::parse(read_file(model_dir + "/tokenizer.json"));
    const json& model = tj.at("model");

    // ---- vocab: {token -> id}, contiguous 0..N-1, stored verbatim (SentencePiece form) ----
    const json& vocab = model.at("vocab");
    const int n = (int)vocab.size();
    std::vector<std::string> tokens(n);
    std::vector<bool> seen(n, false);
    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
        const int id = it.value().get<int>();
        if (id < 0 || id >= n) throw std::runtime_error("tokenizer(llama): vocab id out of range: " + std::to_string(id));
        tokens[id] = it.key();
        seen[id] = true;
    }
    for (int i = 0; i < n; ++i) {
        if (!seen[i]) throw std::runtime_error("tokenizer(llama): vocab ids not contiguous (gap at " + std::to_string(i) + ")");
    }

    // ---- token_type: BYTE for <0xNN>, CONTROL for special added tokens, else NORMAL; scores constant ----
    std::vector<int32_t> token_type(n, TYPE_NORMAL);
    for (int i = 0; i < n; ++i) {
        if (is_byte_token(tokens[i])) token_type[i] = TYPE_BYTE;
    }
    if (tj.contains("added_tokens")) {
        for (const json& at : tj["added_tokens"]) {
            const int id = at.at("id").get<int>();
            if (id >= 0 && id < n && at.value("special", false)) token_type[id] = TYPE_CONTROL;
        }
    }
    // Llama/SPM checkpoints reconstructed from a tokenizer.json carry no real per-token scores; upstream
    // emits a constant sentinel. Match it.
    std::vector<float> scores(n, -1000.0f);

    // ---- special-token ids + flags from tokenizer_config.json (+ special_tokens_map) ----
    json cfg = json::parse(read_file(model_dir + "/tokenizer_config.json"));
    json stm;
    {
        std::ifstream stm_in(model_dir + "/special_tokens_map.json", std::ios::binary);
        if (stm_in) stm = json::parse(std::string((std::istreambuf_iterator<char>(stm_in)), std::istreambuf_iterator<char>()));
    }
    std::unordered_map<std::string, int> tok2id;
    tok2id.reserve(n * 2);
    for (int i = 0; i < n; ++i) tok2id.emplace(tokens[i], i);
    auto resolve_id = [&](const char* key) -> long {
        std::string content;
        if (cfg.contains(key)) content = token_content(cfg[key]);
        if (content.empty() && stm.is_object() && stm.contains(key)) content = token_content(stm[key]);
        if (content.empty()) return -1;
        auto found = tok2id.find(content);
        return found == tok2id.end() ? -1 : (long)found->second;
    };
    const long bos = resolve_id("bos_token");
    const long eos = resolve_id("eos_token");
    const long unk = resolve_id("unk_token");
    long pad = resolve_id("pad_token");
    if (pad < 0) pad = unk;  // upstream defaults padding to the unknown token when none is declared

    const bool add_bos = get_bool(cfg, "add_bos_token", false);
    const bool add_eos = get_bool(cfg, "add_eos_token", false);
    const bool add_space_prefix = get_bool(cfg, "add_prefix_space", false);

    // ---- emit ----
    w.set_str("tokenizer.ggml.model", "llama");
    w.set_str("tokenizer.ggml.pre", "default");
    w.set_arr_str("tokenizer.ggml.tokens", tokens);
    w.set_arr_f32("tokenizer.ggml.scores", scores);
    w.set_arr_i32("tokenizer.ggml.token_type", token_type);
    if (bos >= 0) w.set_u32("tokenizer.ggml.bos_token_id", (uint32_t)bos);
    if (eos >= 0) w.set_u32("tokenizer.ggml.eos_token_id", (uint32_t)eos);
    if (unk >= 0) w.set_u32("tokenizer.ggml.unknown_token_id", (uint32_t)unk);
    if (pad >= 0) w.set_u32("tokenizer.ggml.padding_token_id", (uint32_t)pad);
    w.set_bool("tokenizer.ggml.add_space_prefix", add_space_prefix);
    w.set_bool("tokenizer.ggml.add_bos_token", add_bos);
    w.set_bool("tokenizer.ggml.add_eos_token", add_eos);

    if (cfg.contains("chat_template") && cfg["chat_template"].is_string()) {
        w.set_str("tokenizer.chat_template", cfg["chat_template"].get<std::string>());
    }
}

}  // namespace convert
}  // namespace llmedge
