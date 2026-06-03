// HF (Llama-family) safetensors -> GGUF orchestrator for the on-device converter (Track B / Phase B2).
//
// v1 scope: Llama architecture (tensors + hyperparameters, Layer 3) plus a baked GPT2-BPE tokenizer
// (Layer 4) when `tokenizer_pre` is supplied. Output is F16; quantization (Layer 6) is applied by the
// JNI wrapper via llama_model_quantize, not here, so this stays ggml/llama-free and host-testable.
// Tensors are verified against the upstream convert_hf_to_gguf.py output by a tensor-by-tensor value
// diff; tokenizer KVs by a KV diff (see test_convert.cpp + compare_gguf.py + compare_tokenizer_kv.py).
#pragma once

#include <cstddef>  // size_t
#include <cstdint>  // uint16_t (GCC does not pull these in transitively via <string>)
#include <string>

namespace llmedge {
namespace convert {

// Convert a Llama-arch HF model directory (config.json + *.safetensors) to a GGUF at out_path.
//
// `adapter` selects a model-specific conversion profile:
//   ""  / "none"            — stock Llama: GPT2-BPE tokenizer baked when `tokenizer_pre` is non-empty.
//   "bonsai-qlinear"        — Bonsai/QLlama: fold each weight's per-output `.scales` into the weight
//                             (in f32, before the Q/K permute), and bake the Llama-style tokenizer
//                             (`tokenizer_pre` is ignored — Bonsai uses `pre="default"`).
//
// When no tokenizer is baked (stock profile with empty `tokenizer_pre`), only tensors + llama.* hparams
// are written (the Layer 3 tensor-only path).
//
// Throws std::runtime_error on unsupported architecture or malformed input.
// Returns the number of tensors written.
size_t convert_llama_dir(const std::string& model_dir, const std::string& out_path,
                         const std::string& tokenizer_pre = "", const std::string& adapter = "");

// fp32 -> fp16 (round to nearest even) and bf16 -> fp16, exposed for tests.
uint16_t f32_to_f16(float f);
uint16_t bf16_to_f16(uint16_t bf);

}  // namespace convert
}  // namespace llmedge
