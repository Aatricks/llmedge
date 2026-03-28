#pragma once

#include "ggml-backend.h"

inline bool llmedge_backend_has_devices(const char* name, bool load_all = true) {
    if (load_all) {
        ggml_backend_load_all();
    }
    ggml_backend_reg_t reg = ggml_backend_reg_by_name(name);
    return reg && ggml_backend_reg_dev_count(reg) > 0;
}
