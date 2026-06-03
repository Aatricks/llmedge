// Minimal safetensors reader for the on-device HF -> GGUF converter (Track B / Phase B2).
//
// safetensors layout: [u64 little-endian header_len][header_len bytes of JSON][tensor data blob].
// The JSON header maps each tensor name to {dtype, shape, data_offsets:[begin,end]} where the offsets
// are relative to the start of the data blob (i.e. file offset 8 + header_len).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace llmedge {
namespace convert {

enum class StDType { F64, F32, F16, BF16, I64, I32, I16, I8, U8, BOOL, UNKNOWN };

struct StTensor {
    std::string name;
    StDType dtype = StDType::UNKNOWN;
    std::vector<int64_t> shape;
    uint64_t data_begin = 0;  // relative to the data blob
    uint64_t data_end = 0;
};

struct SafetensorsFile {
    std::string path;
    uint64_t data_blob_start = 0;  // absolute file offset where tensor data begins (8 + header_len)
    std::string metadata_format;   // __metadata__.format, if present
    std::vector<StTensor> tensors; // in declaration order
};

size_t st_dtype_size(StDType d);
const char* st_dtype_name(StDType d);
StDType st_dtype_from_string(const std::string& s);

int64_t st_num_elements(const StTensor& t);
uint64_t st_tensor_nbytes(const StTensor& t);

// Parse only the header (no tensor data read). Throws std::runtime_error on malformed input.
SafetensorsFile read_safetensors_header(const std::string& path);

// Read a tensor's raw bytes from disk. Throws if the file is shorter than the tensor's region.
std::vector<uint8_t> st_read_tensor_bytes(const SafetensorsFile& f, const StTensor& t);

}  // namespace convert
}  // namespace llmedge
