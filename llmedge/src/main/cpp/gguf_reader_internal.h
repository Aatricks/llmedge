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
