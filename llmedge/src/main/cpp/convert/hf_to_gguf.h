// HF (Llama-family) safetensors -> GGUF orchestrator for the on-device converter (Track B / Phase B2).
//
// v1 scope: Llama architecture, tensors + hyperparameters. Tokenizer is NOT baked here (Layer 4); the
// resulting GGUF carries tensors + llama.* KVs only. Verified against the upstream convert_hf_to_gguf.py
// output by a tensor-by-tensor value diff (see test_convert.cpp + compare_gguf.py).
#pragma once

#include <string>

namespace llmedge {
namespace convert {

// Convert a Llama-arch HF model directory (config.json + *.safetensors) to a GGUF at out_path.
// Throws std::runtime_error on unsupported architecture or malformed input.
// Returns the number of tensors written.
size_t convert_llama_dir(const std::string& model_dir, const std::string& out_path);

// fp32 -> fp16 (round to nearest even) and bf16 -> fp16, exposed for tests.
uint16_t f32_to_f16(float f);
uint16_t bf16_to_f16(uint16_t bf);

}  // namespace convert
}  // namespace llmedge
