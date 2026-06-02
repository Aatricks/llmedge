// Standalone host test for safetensors_reader. Build:
//   clang++ -std=c++17 -I . -I ../../../../../../llama.cpp/vendor \
//     test_safetensors_reader.cpp safetensors_reader.cpp -o /tmp/streader_test && /tmp/streader_test
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "safetensors_reader.h"

using namespace llmedge::convert;
using json = nlohmann::json;

static int g_failures = 0;
#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("FAIL: %s (line %d)\n", #cond, __LINE__);              \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

static std::string write_synth(const std::string& path) {
    // Two tensors: F32 [2,3] (24 bytes), BF16 [4] (8 bytes). Data blob = 32 bytes.
    json hdr;
    hdr["t_f32"] = {{"dtype", "F32"}, {"shape", {2, 3}}, {"data_offsets", {0, 24}}};
    hdr["t_bf16"] = {{"dtype", "BF16"}, {"shape", {4}}, {"data_offsets", {24, 32}}};
    hdr["__metadata__"] = {{"format", "pt"}};
    std::string js = hdr.dump();

    std::vector<float> f32 = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    std::vector<uint16_t> bf16 = {0x3f80, 0x4000, 0x4040, 0x4080};  // bf16 for 1,2,3,4

    std::ofstream out(path, std::ios::binary);
    uint64_t n = js.size();
    out.write(reinterpret_cast<const char*>(&n), 8);
    out.write(js.data(), static_cast<std::streamsize>(js.size()));
    out.write(reinterpret_cast<const char*>(f32.data()), 24);
    out.write(reinterpret_cast<const char*>(bf16.data()), 8);
    out.close();
    return path;
}

static const StTensor& find(const SafetensorsFile& f, const std::string& name) {
    for (const auto& t : f.tensors)
        if (t.name == name) return t;
    throw std::runtime_error("tensor not found: " + name);
}

int main() {
    const std::string path = "/tmp/synth.safetensors";
    write_synth(path);

    SafetensorsFile f = read_safetensors_header(path);
    CHECK(f.tensors.size() == 2);
    CHECK(f.metadata_format == "pt");

    const StTensor& a = find(f, "t_f32");
    CHECK(a.dtype == StDType::F32);
    CHECK(a.shape.size() == 2 && a.shape[0] == 2 && a.shape[1] == 3);
    CHECK(st_num_elements(a) == 6);
    CHECK(st_tensor_nbytes(a) == 24);
    CHECK(st_dtype_size(a.dtype) == 4);

    const StTensor& b = find(f, "t_bf16");
    CHECK(b.dtype == StDType::BF16);
    CHECK(b.shape.size() == 1 && b.shape[0] == 4);
    CHECK(st_tensor_nbytes(b) == 8);
    CHECK(st_dtype_size(b.dtype) == 2);

    // Data round-trips.
    auto bytes_a = st_read_tensor_bytes(f, a);
    CHECK(bytes_a.size() == 24);
    float first = 0.f;
    std::memcpy(&first, bytes_a.data(), 4);
    CHECK(first == 1.f);
    float last = 0.f;
    std::memcpy(&last, bytes_a.data() + 20, 4);
    CHECK(last == 6.f);

    auto bytes_b = st_read_tensor_bytes(f, b);
    CHECK(bytes_b.size() == 8);
    uint16_t bf0 = 0;
    std::memcpy(&bf0, bytes_b.data(), 2);
    CHECK(bf0 == 0x3f80);

    // Malformed: truncated length prefix.
    {
        std::ofstream bad("/tmp/bad.safetensors", std::ios::binary);
        bad.write("\x03\x00", 2);
        bad.close();
        bool threw = false;
        try {
            read_safetensors_header("/tmp/bad.safetensors");
        } catch (const std::exception&) {
            threw = true;
        }
        CHECK(threw);
    }

    if (g_failures == 0) {
        std::printf("OK: all safetensors_reader checks passed\n");
        return 0;
    }
    std::printf("%d check(s) failed\n", g_failures);
    return 1;
}
