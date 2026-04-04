#include "llm_backend_support.h"

#include "llama.h"

#include <cstring>

const char * backend_name(const RequestedBackend backend) {
    switch (backend) {
        case RequestedBackend::OPENCL:
            return "OpenCL";
        case RequestedBackend::VULKAN:
            return "Vulkan";
        case RequestedBackend::CPU:
        default:
            return "CPU";
    }
}

bool is_gpu_backend(const RequestedBackend backend) {
    return backend == RequestedBackend::OPENCL || backend == RequestedBackend::VULKAN;
}

std::string find_backend_registry_name(const RequestedBackend backend, const int desired_index) {
    if (!is_gpu_backend(backend)) {
        return {};
    }

    const char * wanted_prefix = backend_name(backend);
    const size_t wanted_prefix_len = std::strlen(wanted_prefix);
    int matched_index = 0;
    for (size_t i = 0; i < ggml_backend_reg_get_count(); ++i) {
        const char * reg_name = ggml_backend_reg_get_name(i);
        if (!reg_name) {
            continue;
        }
        if (std::strncmp(reg_name, wanted_prefix, wanted_prefix_len) != 0) {
            continue;
        }
        if (matched_index == desired_index) {
            return reg_name;
        }
        ++matched_index;
    }

    return {};
}
