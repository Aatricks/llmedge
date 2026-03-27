#include "../../main/cpp/gguf_reader_internal.h"

#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace {

struct metadata_case {
    const char * suffix;
    uint32_t file_type;
    ggml_type dominant_type;
};

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::filesystem::path write_fixture(const metadata_case & test_case) {
    ggml_init_params params = {
        /*.mem_size   =*/ 1 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };

    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "Failed to create ggml context for GGUF fixture");

    ggml_tensor * tensor_a = ggml_new_tensor_2d(ctx, test_case.dominant_type, 256, 1);
    ggml_tensor * tensor_b = ggml_new_tensor_2d(ctx, test_case.dominant_type, 256, 1);
    ggml_tensor * tensor_c = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 16, 1);
    require(tensor_a && tensor_b && tensor_c, "Failed to allocate GGUF fixture tensors");

    ggml_format_name(tensor_a, "blk.0.weight");
    ggml_format_name(tensor_b, "blk.1.weight");
    ggml_format_name(tensor_c, "output.weight");

    std::memset(tensor_a->data, 0, ggml_nbytes(tensor_a));
    std::memset(tensor_b->data, 0, ggml_nbytes(tensor_b));
    std::memset(tensor_c->data, 0, ggml_nbytes(tensor_c));

    gguf_context * gguf = gguf_init_empty();
    require(gguf != nullptr, "Failed to create GGUF fixture metadata context");

    gguf_set_val_str(gguf, "general.architecture", "llama");
    gguf_set_val_u32(gguf, "llama.context_length", 4096);
    gguf_set_val_str(gguf, "tokenizer.chat_template", "{{ messages }}");
    gguf_set_val_u64(gguf, "llama.parameter_count", 123456789ULL);
    gguf_set_val_str(gguf, "general.name", "llmedge-test-fixture");
    gguf_set_val_u32(gguf, "general.file_type", test_case.file_type);
    gguf_add_tensor(gguf, tensor_a);
    gguf_add_tensor(gguf, tensor_b);
    gguf_add_tensor(gguf, tensor_c);

    const auto output = std::filesystem::path(std::string("/tmp/llmedge-gguf-reader-fixture-") + test_case.suffix + ".gguf");
    std::error_code ec;
    std::filesystem::remove(output, ec);
    gguf_write_to_file(gguf, output.string().c_str(), false);

    gguf_free(gguf);
    ggml_free(ctx);

    return output;
}

void verify_case(const metadata_case & test_case) {
    const auto fixture_path = write_fixture(test_case);
    gguf_context * gguf = llmedge_gguf_open_file(fixture_path.string().c_str());
    require(gguf != nullptr, "Failed to reopen synthetic GGUF fixture");

    require(llmedge_gguf_get_context_size(gguf) == 4096, "Unexpected GGUF context length");
    require(llmedge_gguf_get_chat_template(gguf) == "{{ messages }}", "Unexpected GGUF chat template");
    require(llmedge_gguf_get_architecture(gguf) == "llama", "Unexpected GGUF architecture");
    require(llmedge_gguf_get_parameter_count(gguf) == "123456789", "Unexpected GGUF parameter count");
    require(llmedge_gguf_get_model_name(gguf) == "llmedge-test-fixture", "Unexpected GGUF model name");
    require(llmedge_gguf_get_file_type(gguf) == static_cast<int32_t>(test_case.file_type), "Unexpected GGUF file type");
    require(
        llmedge_gguf_get_dominant_tensor_type(gguf) == static_cast<int32_t>(test_case.dominant_type),
        "Unexpected GGUF dominant tensor type");

    gguf_free(gguf);
    std::error_code ec;
    std::filesystem::remove(fixture_path, ec);
}

} // namespace

void test_gguf_reader_metadata() {
    const metadata_case cases[] = {
        {"q8_kv", 149, GGML_TYPE_Q8_KV},
        {"iq2_k", 138, GGML_TYPE_IQ2_K},
        {"iq3_k", 139, GGML_TYPE_IQ3_K},
        {"iq4_k", 140, GGML_TYPE_IQ4_K},
        {"iq5_k", 141, GGML_TYPE_IQ5_K},
        {"iq6_k", 142, GGML_TYPE_IQ6_K},
    };

    for (const auto & test_case : cases) {
        verify_case(test_case);
    }
}
