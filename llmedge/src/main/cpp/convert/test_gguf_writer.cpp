// Writes a sample GGUF with GgufWriter; verified by the canonical gguf-py reader (verify_gguf.py).
//   clang++ -std=c++17 -I . test_gguf_writer.cpp gguf_writer.cpp -o /tmp/gguf_test && /tmp/gguf_test
#include <cstdio>
#include <vector>

#include "gguf_writer.h"

using namespace llmedge::convert;

int main() {
    GgufWriter w;
    w.set_str("general.architecture", "llama");
    w.set_u32("llama.block_count", 16);
    w.set_u32("llama.context_length", 2048);
    w.set_f32("llama.attention.layer_norm_rms_epsilon", 1e-5f);
    w.set_bool("test.flag", true);
    w.set_arr_u32("test.arr_u32", {10, 20, 30});
    w.set_arr_str("tokenizer.ggml.tokens", {"<s>", "hello", "world"});

    // ggml order ne[0]=3 (fastest/contiguous), ne[1]=2 -> 6 F32 elements.
    std::vector<float> data = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    w.add_tensor("token_embd.weight", GgmlType::F32, {3, 2}, data.data(), data.size() * 4);

    std::vector<float> norm = {0.5f, 0.25f};
    w.add_tensor("output_norm.weight", GgmlType::F32, {2}, norm.data(), norm.size() * 4);

    w.write("/tmp/test_write.gguf", 32);
    std::printf("wrote /tmp/test_write.gguf (kv=%zu tensors=%zu)\n", w.kv_count(), w.tensor_count());
    return 0;
}
