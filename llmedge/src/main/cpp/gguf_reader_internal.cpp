#include "gguf_reader_internal.h"

#include <set>
#include <string>
#include <unordered_map>

gguf_context * llmedge_gguf_open_file(const char * model_path) {
    gguf_init_params init_params = { .no_alloc = true, .ctx = nullptr };
    return gguf_init_from_file(model_path, init_params);
}

int64_t llmedge_gguf_get_context_size(const struct gguf_context * gguf_context) {
    const int64_t architecture_key_id = gguf_find_key(gguf_context, "general.architecture");
    if (architecture_key_id == -1) {
        return -1;
    }

    const std::string architecture = gguf_get_val_str(gguf_context, architecture_key_id);
    const std::string context_length_key = architecture + ".context_length";
    const int64_t context_length_key_id = gguf_find_key(gguf_context, context_length_key.c_str());
    if (context_length_key_id == -1) {
        return -1;
    }

    return gguf_get_val_u32(gguf_context, context_length_key_id);
}

std::string llmedge_gguf_get_chat_template(const struct gguf_context * gguf_context) {
    const int64_t chat_template_key_id = gguf_find_key(gguf_context, "tokenizer.chat_template");
    return chat_template_key_id == -1 ? "" : std::string(gguf_get_val_str(gguf_context, chat_template_key_id));
}

std::string llmedge_gguf_get_architecture(const struct gguf_context * gguf_context) {
    const int64_t architecture_key_id = gguf_find_key(gguf_context, "general.architecture");
    return architecture_key_id == -1 ? "" : std::string(gguf_get_val_str(gguf_context, architecture_key_id));
}

std::string llmedge_gguf_get_parameter_count(const struct gguf_context * gguf_context) {
    const int64_t parameter_count_key_id = gguf_find_key(gguf_context, "llama.parameter_count");
    if (parameter_count_key_id == -1) {
        return "";
    }
    return std::to_string(gguf_get_val_u64(gguf_context, parameter_count_key_id));
}

std::string llmedge_gguf_get_model_name(const struct gguf_context * gguf_context) {
    const int64_t model_name_key_id = gguf_find_key(gguf_context, "general.name");
    return model_name_key_id == -1 ? "" : std::string(gguf_get_val_str(gguf_context, model_name_key_id));
}

int64_t llmedge_gguf_get_file_type(const struct gguf_context * gguf_context) {
    const int64_t file_type_key_id = gguf_find_key(gguf_context, "general.file_type");
    if (file_type_key_id == -1) {
        return -1;
    }
    return static_cast<int64_t>(gguf_get_val_u32(gguf_context, file_type_key_id));
}

std::string llmedge_gguf_get_tensor_name_prefixes(const struct gguf_context * gguf_context) {
    const int64_t tensor_count = gguf_get_n_tensors(gguf_context);
    if (tensor_count <= 0) {
        return std::string();
    }

    // DEPTH CONTRACT: exactly one segment — the tensor name up to its first '.', or the whole
    // name when it has none. Callers classify against depth-1 keys ("text_encoders",
    // "first_stage_model", "joint_blocks"); returning deeper prefixes would silently break them.
    std::set<std::string> prefixes;
    for (int64_t tensor_id = 0; tensor_id < tensor_count; ++tensor_id) {
        const char * name = gguf_get_tensor_name(gguf_context, tensor_id);
        if (name == nullptr) {
            continue;
        }
        std::string tensor_name(name);
        const size_t separator = tensor_name.find('.');
        prefixes.insert(separator == std::string::npos ? tensor_name : tensor_name.substr(0, separator));
        if (prefixes.size() >= LLMEDGE_GGUF_MAX_TENSOR_PREFIXES) {
            break;
        }
    }

    std::string joined;
    for (const std::string & prefix : prefixes) {
        if (!joined.empty()) {
            joined.push_back('\n');
        }
        joined.append(prefix);
    }
    return joined;
}

int32_t llmedge_gguf_get_dominant_tensor_type(const struct gguf_context * gguf_context) {
    const int64_t tensor_count = gguf_get_n_tensors(gguf_context);
    if (tensor_count <= 0) {
        return -1;
    }

    std::unordered_map<int32_t, int32_t> counts;
    int32_t dominant_type = -1;
    int32_t dominant_count = -1;

    for (int64_t tensor_id = 0; tensor_id < tensor_count; ++tensor_id) {
        const int32_t type = static_cast<int32_t>(gguf_get_tensor_type(gguf_context, tensor_id));
        const int32_t count = ++counts[type];
        if (count > dominant_count) {
            dominant_count = count;
            dominant_type = type;
        }
    }

    return dominant_type;
}
