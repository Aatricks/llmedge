#pragma once

#include "ggml.h"

#include <cstdint>
#include <string>

struct gguf_context * llmedge_gguf_open_file(const char * model_path);
int64_t llmedge_gguf_get_context_size(const struct gguf_context * gguf_context);
std::string llmedge_gguf_get_chat_template(const struct gguf_context * gguf_context);
std::string llmedge_gguf_get_architecture(const struct gguf_context * gguf_context);
std::string llmedge_gguf_get_parameter_count(const struct gguf_context * gguf_context);
std::string llmedge_gguf_get_model_name(const struct gguf_context * gguf_context);
int64_t llmedge_gguf_get_file_type(const struct gguf_context * gguf_context);
int32_t llmedge_gguf_get_dominant_tensor_type(const struct gguf_context * gguf_context);

// Caps the prefix set for a checkpoint with an unexpected naming scheme; real checkpoints yield
// a handful of distinct first segments.
#define LLMEDGE_GGUF_MAX_TENSOR_PREFIXES 64

// Deduplicated, newline-joined first segments of every tensor name (see the DEPTH CONTRACT note
// in the implementation). Lets a caller tell a bare denoiser from an all-in-one bundle.
std::string llmedge_gguf_get_tensor_name_prefixes(const struct gguf_context * gguf_context);
