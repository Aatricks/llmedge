#pragma once

#include <string>

enum class RequestedBackend : int {
    CPU = 0,
    OPENCL = 1,
    VULKAN = 2,
};

const char * backend_name(RequestedBackend backend);
bool is_gpu_backend(RequestedBackend backend);
std::string find_backend_registry_name(RequestedBackend backend, int desired_index);
