// CLI: convert a Llama HF dir -> GGUF.   test_convert <model_dir> <out.gguf> [tokenizer_pre]
// When tokenizer_pre is given (e.g. "smollm"), the GPT2-BPE tokenizer is baked in too (Layer 4).
#include <cstdio>

#include "hf_to_gguf.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model_dir> <out.gguf> [tokenizer_pre]\n", argv[0]);
        return 2;
    }
    const std::string pre = argc > 3 ? argv[3] : "";
    try {
        size_t n = llmedge::convert::convert_llama_dir(argv[1], argv[2], pre);
        std::printf("converted %zu tensors -> %s%s\n", n, argv[2],
                    pre.empty() ? "" : " (+tokenizer)");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "convert failed: %s\n", e.what());
        return 1;
    }
}
