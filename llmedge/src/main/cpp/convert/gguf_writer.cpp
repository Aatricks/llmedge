#include "gguf_writer.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace llmedge {
namespace convert {

namespace {
void put_u32(std::vector<uint8_t>& b, uint32_t v) { b.insert(b.end(), (uint8_t*)&v, (uint8_t*)&v + 4); }
void put_u64(std::vector<uint8_t>& b, uint64_t v) { b.insert(b.end(), (uint8_t*)&v, (uint8_t*)&v + 8); }
void put_bytes(std::vector<uint8_t>& b, const void* p, size_t n) {
    b.insert(b.end(), (const uint8_t*)p, (const uint8_t*)p + n);
}
void put_gstr(std::vector<uint8_t>& b, const std::string& s) {
    put_u64(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}
uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }
}  // namespace

void GgufWriter::put_scalar(const std::string& key, uint32_t type, const void* bytes, size_t n) {
    Kv kv;
    kv.type = type;
    kv.scalar_bytes.assign((const uint8_t*)bytes, (const uint8_t*)bytes + n);
    kvs_.emplace_back(key, std::move(kv));
}

void GgufWriter::set_u32(const std::string& k, uint32_t v) { put_scalar(k, T_UINT32, &v, 4); }
void GgufWriter::set_i32(const std::string& k, int32_t v) { put_scalar(k, T_INT32, &v, 4); }
void GgufWriter::set_f32(const std::string& k, float v) { put_scalar(k, T_FLOAT32, &v, 4); }
void GgufWriter::set_bool(const std::string& k, bool v) {
    uint8_t b = v ? 1 : 0;
    put_scalar(k, T_BOOL, &b, 1);
}
void GgufWriter::set_str(const std::string& k, const std::string& v) {
    Kv kv;
    kv.type = T_STRING;
    kv.str_values = {v};
    kvs_.emplace_back(k, std::move(kv));
}

void GgufWriter::set_arr_u32(const std::string& k, const std::vector<uint32_t>& vals) {
    Kv kv;
    kv.type = T_ARRAY;
    kv.arr_subtype = T_UINT32;
    kv.arr_count = vals.size();
    kv.arr_bytes.assign((const uint8_t*)vals.data(), (const uint8_t*)vals.data() + vals.size() * 4);
    kvs_.emplace_back(k, std::move(kv));
}
void GgufWriter::set_arr_i32(const std::string& k, const std::vector<int32_t>& vals) {
    Kv kv;
    kv.type = T_ARRAY;
    kv.arr_subtype = T_INT32;
    kv.arr_count = vals.size();
    kv.arr_bytes.assign((const uint8_t*)vals.data(), (const uint8_t*)vals.data() + vals.size() * 4);
    kvs_.emplace_back(k, std::move(kv));
}
void GgufWriter::set_arr_f32(const std::string& k, const std::vector<float>& vals) {
    Kv kv;
    kv.type = T_ARRAY;
    kv.arr_subtype = T_FLOAT32;
    kv.arr_count = vals.size();
    kv.arr_bytes.assign((const uint8_t*)vals.data(), (const uint8_t*)vals.data() + vals.size() * 4);
    kvs_.emplace_back(k, std::move(kv));
}
void GgufWriter::set_arr_str(const std::string& k, const std::vector<std::string>& vals) {
    Kv kv;
    kv.type = T_ARRAY;
    kv.arr_subtype = T_STRING;
    kv.arr_count = vals.size();
    kv.str_values = vals;
    kvs_.emplace_back(k, std::move(kv));
}

void GgufWriter::add_tensor(const std::string& name, GgmlType type, const std::vector<int64_t>& ne,
                            const void* data, uint64_t nbytes) {
    TensorInfo t;
    t.name = name;
    t.type = type;
    t.ne = ne.empty() ? std::vector<int64_t>{1} : ne;
    t.nbytes = nbytes;
    t.data.assign((const uint8_t*)data, (const uint8_t*)data + nbytes);
    tensors_.push_back(std::move(t));
}

void GgufWriter::add_tensor_info(const std::string& name, GgmlType type,
                                 const std::vector<int64_t>& ne, uint64_t nbytes) {
    TensorInfo t;
    t.name = name;
    t.type = type;
    t.ne = ne.empty() ? std::vector<int64_t>{1} : ne;
    t.nbytes = nbytes;
    tensors_.push_back(std::move(t));
}

void GgufWriter::write_begin(const std::string& path, uint32_t alignment) {
    if (out_.is_open()) throw std::runtime_error("gguf: write_begin called twice");
    alignment_ = alignment;
    next_tensor_ = 0;
    data_written_ = 0;

    std::vector<uint8_t> meta;
    // Header
    const char magic[4] = {'G', 'G', 'U', 'F'};
    put_bytes(meta, magic, 4);
    put_u32(meta, 3);  // version
    put_u64(meta, tensors_.size());
    put_u64(meta, kvs_.size());

    // KV section
    for (const auto& [key, kv] : kvs_) {
        put_gstr(meta, key);
        put_u32(meta, kv.type);
        if (kv.type == T_STRING) {
            put_gstr(meta, kv.str_values.at(0));
        } else if (kv.type == T_ARRAY) {
            put_u32(meta, kv.arr_subtype);
            put_u64(meta, kv.arr_count);
            if (kv.arr_subtype == T_STRING) {
                for (const auto& s : kv.str_values) put_gstr(meta, s);
            } else {
                put_bytes(meta, kv.arr_bytes.data(), kv.arr_bytes.size());
            }
        } else {
            put_bytes(meta, kv.scalar_bytes.data(), kv.scalar_bytes.size());
        }
    }

    // Tensor offsets (relative to the data section start), each aligned. Sizes are
    // known from the declarations, so the whole metadata block can be written before
    // any tensor payload exists in memory.
    offsets_.assign(tensors_.size(), 0);
    uint64_t cur = 0;
    for (size_t i = 0; i < tensors_.size(); ++i) {
        offsets_[i] = cur;
        cur = align_up(cur + tensors_[i].nbytes, alignment_);
    }

    // Tensor info section
    for (size_t i = 0; i < tensors_.size(); ++i) {
        const auto& t = tensors_[i];
        put_gstr(meta, t.name);
        put_u32(meta, (uint32_t)t.ne.size());
        for (int64_t d : t.ne) put_u64(meta, (uint64_t)d);
        put_u32(meta, (uint32_t)t.type);
        put_u64(meta, offsets_[i]);
    }

    out_.open(path, std::ios::binary);
    if (!out_) throw std::runtime_error("gguf: cannot open " + path + " for writing");
    out_.write((const char*)meta.data(), (std::streamsize)meta.size());

    // Pad metadata to alignment -> data section start.
    uint64_t data_start = align_up(meta.size(), alignment_);
    std::vector<uint8_t> pad(alignment_, 0);
    if (data_start > meta.size()) {
        out_.write((const char*)pad.data(), (std::streamsize)(data_start - meta.size()));
    }
    if (!out_) throw std::runtime_error("gguf: header write failed for " + path);
}

void GgufWriter::write_tensor_data(const void* data, uint64_t nbytes) {
    if (!out_.is_open()) throw std::runtime_error("gguf: write_tensor_data before write_begin");
    if (next_tensor_ >= tensors_.size()) throw std::runtime_error("gguf: more payloads than declared tensors");
    const auto& t = tensors_[next_tensor_];
    if (nbytes != t.nbytes) {
        throw std::runtime_error("gguf: payload size mismatch for " + t.name);
    }
    if (offsets_[next_tensor_] > data_written_) {
        std::vector<uint8_t> pad(offsets_[next_tensor_] - data_written_, 0);
        out_.write((const char*)pad.data(), (std::streamsize)pad.size());
        data_written_ = offsets_[next_tensor_];
    }
    out_.write((const char*)data, (std::streamsize)nbytes);
    data_written_ += nbytes;
    ++next_tensor_;
    if (!out_) throw std::runtime_error("gguf: data write failed for " + t.name);
}

void GgufWriter::finish() {
    if (!out_.is_open()) throw std::runtime_error("gguf: finish before write_begin");
    if (next_tensor_ != tensors_.size()) {
        throw std::runtime_error("gguf: finish with undelivered tensor payloads");
    }
    out_.close();
    if (!out_) throw std::runtime_error("gguf: close failed");
}

void GgufWriter::write(const std::string& path, uint32_t alignment) {
    for (const auto& t : tensors_) {
        if (t.data.size() != t.nbytes) {
            throw std::runtime_error("gguf: write() requires buffered payloads (missing for " + t.name + ")");
        }
    }
    write_begin(path, alignment);
    for (const auto& t : tensors_) {
        write_tensor_data(t.data.data(), t.nbytes);
    }
    finish();
}

}  // namespace convert
}  // namespace llmedge
