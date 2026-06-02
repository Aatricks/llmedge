#include "safetensors_reader.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "nlohmann/json.hpp"

namespace llmedge {
namespace convert {

using json = nlohmann::json;

size_t st_dtype_size(StDType d) {
    switch (d) {
        case StDType::F64:
        case StDType::I64: return 8;
        case StDType::F32:
        case StDType::I32: return 4;
        case StDType::F16:
        case StDType::BF16:
        case StDType::I16: return 2;
        case StDType::I8:
        case StDType::U8:
        case StDType::BOOL: return 1;
        default: return 0;
    }
}

const char* st_dtype_name(StDType d) {
    switch (d) {
        case StDType::F64: return "F64";
        case StDType::F32: return "F32";
        case StDType::F16: return "F16";
        case StDType::BF16: return "BF16";
        case StDType::I64: return "I64";
        case StDType::I32: return "I32";
        case StDType::I16: return "I16";
        case StDType::I8: return "I8";
        case StDType::U8: return "U8";
        case StDType::BOOL: return "BOOL";
        default: return "UNKNOWN";
    }
}

StDType st_dtype_from_string(const std::string& s) {
    if (s == "F64") return StDType::F64;
    if (s == "F32") return StDType::F32;
    if (s == "F16") return StDType::F16;
    if (s == "BF16") return StDType::BF16;
    if (s == "I64") return StDType::I64;
    if (s == "I32") return StDType::I32;
    if (s == "I16") return StDType::I16;
    if (s == "I8") return StDType::I8;
    if (s == "U8") return StDType::U8;
    if (s == "BOOL") return StDType::BOOL;
    return StDType::UNKNOWN;
}

int64_t st_num_elements(const StTensor& t) {
    int64_t n = 1;
    for (int64_t d : t.shape) n *= d;
    return t.shape.empty() ? 0 : n;
}

uint64_t st_tensor_nbytes(const StTensor& t) {
    return t.data_end - t.data_begin;
}

SafetensorsFile read_safetensors_header(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("safetensors: cannot open " + path);

    uint64_t header_len = 0;
    in.read(reinterpret_cast<char*>(&header_len), 8);
    if (!in || in.gcount() != 8) throw std::runtime_error("safetensors: truncated length prefix");
    // safetensors is little-endian; assume host is LE (Android/ARM/x86 all are).
    if (header_len == 0 || header_len > (1ull << 32)) {
        throw std::runtime_error("safetensors: implausible header length");
    }

    std::string header_json(static_cast<size_t>(header_len), '\0');
    in.read(header_json.data(), static_cast<std::streamsize>(header_len));
    if (!in || static_cast<uint64_t>(in.gcount()) != header_len) {
        throw std::runtime_error("safetensors: truncated header");
    }

    json hdr;
    try {
        hdr = json::parse(header_json);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("safetensors: header is not valid JSON: ") + e.what());
    }
    if (!hdr.is_object()) throw std::runtime_error("safetensors: header is not a JSON object");

    SafetensorsFile out;
    out.path = path;
    out.data_blob_start = 8 + header_len;

    for (auto it = hdr.begin(); it != hdr.end(); ++it) {
        const std::string& name = it.key();
        if (name == "__metadata__") {
            if (it.value().is_object() && it.value().contains("format")) {
                out.metadata_format = it.value()["format"].get<std::string>();
            }
            continue;
        }
        const json& v = it.value();
        if (!v.is_object() || !v.contains("dtype") || !v.contains("shape") || !v.contains("data_offsets")) {
            throw std::runtime_error("safetensors: malformed entry for tensor '" + name + "'");
        }
        StTensor t;
        t.name = name;
        t.dtype = st_dtype_from_string(v["dtype"].get<std::string>());
        for (const auto& d : v["shape"]) t.shape.push_back(d.get<int64_t>());
        const auto& off = v["data_offsets"];
        if (!off.is_array() || off.size() != 2) {
            throw std::runtime_error("safetensors: bad data_offsets for '" + name + "'");
        }
        t.data_begin = off[0].get<uint64_t>();
        t.data_end = off[1].get<uint64_t>();
        if (t.data_end < t.data_begin) {
            throw std::runtime_error("safetensors: negative-length region for '" + name + "'");
        }
        out.tensors.push_back(std::move(t));
    }
    return out;
}

std::vector<uint8_t> st_read_tensor_bytes(const SafetensorsFile& f, const StTensor& t) {
    const uint64_t nbytes = st_tensor_nbytes(t);
    std::ifstream in(f.path, std::ios::binary);
    if (!in) throw std::runtime_error("safetensors: cannot reopen " + f.path);
    in.seekg(static_cast<std::streamoff>(f.data_blob_start + t.data_begin));
    std::vector<uint8_t> buf(static_cast<size_t>(nbytes));
    in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(nbytes));
    if (!in || static_cast<uint64_t>(in.gcount()) != nbytes) {
        throw std::runtime_error("safetensors: truncated tensor data for '" + t.name + "'");
    }
    return buf;
}

}  // namespace convert
}  // namespace llmedge
