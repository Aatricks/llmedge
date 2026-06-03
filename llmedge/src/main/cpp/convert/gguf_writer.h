// Minimal, spec-conformant GGUF writer for the on-device HF -> GGUF converter (Track B / Phase B2).
//
// Writes GGUF v3: ["GGUF"][u32 version=3][u64 n_tensors][u64 n_kv][KV...][tensor-info...][pad][data...].
// Hand-rolled (no ggml dependency) so it is host-testable and works unchanged inside the JNI lib.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace llmedge {
namespace convert {

// ggml tensor type ids (subset we emit). Values match ggml.h GGML_TYPE_*.
enum class GgmlType : int32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q8_0 = 8,
};

class GgufWriter {
public:
    void set_str(const std::string& key, const std::string& val);
    void set_u32(const std::string& key, uint32_t val);
    void set_i32(const std::string& key, int32_t val);
    void set_f32(const std::string& key, float val);
    void set_bool(const std::string& key, bool val);
    void set_arr_u32(const std::string& key, const std::vector<uint32_t>& vals);
    void set_arr_i32(const std::string& key, const std::vector<int32_t>& vals);
    void set_arr_f32(const std::string& key, const std::vector<float>& vals);
    void set_arr_str(const std::string& key, const std::vector<std::string>& vals);

    // ne is in ggml order: ne[0] is the fastest-varying (contiguous) dimension.
    // data points to `nbytes` of tensor payload (already in `type`'s layout).
    void add_tensor(const std::string& name, GgmlType type, const std::vector<int64_t>& ne,
                    const void* data, uint64_t nbytes);

    // Serialize to `path`. Throws std::runtime_error on I/O failure.
    void write(const std::string& path, uint32_t alignment = 32) const;

    size_t kv_count() const { return kvs_.size(); }
    size_t tensor_count() const { return tensors_.size(); }

private:
    // gguf metadata value types (gguf.h gguf_type)
    enum GgufType : uint32_t {
        T_UINT8 = 0, T_INT8 = 1, T_UINT16 = 2, T_INT16 = 3,
        T_UINT32 = 4, T_INT32 = 5, T_FLOAT32 = 6, T_BOOL = 7,
        T_STRING = 8, T_ARRAY = 9, T_UINT64 = 10, T_INT64 = 11, T_FLOAT64 = 12,
    };

    struct Kv {
        uint32_t type;            // GgufType
        uint32_t arr_subtype;     // valid when type == T_ARRAY
        std::vector<uint8_t> scalar_bytes;     // for scalar types
        std::vector<std::string> str_values;   // for STRING (size 1) or ARRAY of STRING
        std::vector<uint8_t> arr_bytes;         // for ARRAY of numeric
        uint64_t arr_count = 0;
    };
    struct TensorInfo {
        std::string name;
        GgmlType type;
        std::vector<int64_t> ne;
        std::vector<uint8_t> data;
    };

    std::vector<std::pair<std::string, Kv>> kvs_;  // preserve insertion order
    std::vector<TensorInfo> tensors_;

    void put_scalar(const std::string& key, uint32_t type, const void* bytes, size_t n);
};

}  // namespace convert
}  // namespace llmedge
