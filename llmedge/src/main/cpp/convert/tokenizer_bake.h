// Layer 4 of the on-device HF -> GGUF converter: bake a GPT2-BPE tokenizer into the GGUF KV store.
//
// Reads tokenizer.json / tokenizer_config.json (+ optional special_tokens_map.json) from a HF model
// dir and emits the `tokenizer.ggml.*` + `tokenizer.chat_template` keys that llama.cpp expects.
//
// SCOPE (v1): byte-level BPE only (model.type == "BPE", space-joined string merges). Anything else
// fails loudly rather than emitting a tokenizer that loads but mis-tokenizes.
#pragma once

#include <string>

namespace llmedge {
namespace convert {

class GgufWriter;

// Bake the tokenizer KVs into `w`.
//
// `pre` is the value for `tokenizer.ggml.pre` (the BPE pre-tokenizer identifier, e.g. "smollm").
// It MUST be supplied by the caller and non-empty: upstream derives it by hashing the output of the
// real tokenizer over a probe string, which we do not reimplement in v1. A wrong/guessed `pre` loads
// without error and silently mis-tokenizes, so we refuse to guess and throw when it is empty.
//
// Throws std::runtime_error on a missing file, an unsupported tokenizer, or a malformed vocab.
void bake_gpt2_tokenizer(GgufWriter& w, const std::string& model_dir, const std::string& pre);

// Bake a Llama-style (`tokenizer.ggml.model = "llama"`) tokenizer from a `tokenizer.json` whose vocab is
// the SentencePiece vocab (byte-fallback `<0xNN>` tokens, no real per-token scores). Emits tokens +
// constant scores + token_type (BYTE for `<0xNN>`, CONTROL for special added tokens, else NORMAL) and the
// special-token ids/flags — no merges, `pre="default"`. This is the family Bonsai (and other
// LlamaTokenizer checkpoints lacking a `tokenizer.model`) lands in; it matches what upstream
// `convert_hf_to_gguf.py` produces for them. Throws on a malformed/non-contiguous vocab.
void bake_llama_tokenizer(GgufWriter& w, const std::string& model_dir);

}  // namespace convert
}  // namespace llmedge
