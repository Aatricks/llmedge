// CLI: convert a Llama HF dir -> GGUF.   test_convert <model_dir> <out.gguf>
#include <cstdio>

#include "hf_to_gguf.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model_dir> <out.gguf>\n", argv[0]);
        return 2;
    }
    try {
        size_t n = llmedge::convert::convert_llama_dir(argv[1], argv[2]);
        std::printf("converted %zu tensors -> %s\n", n, argv[2]);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "convert failed: %s\n", e.what());
        return 1;
    }
}
