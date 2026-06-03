// CLI: convert a Llama HF dir -> GGUF.  test_convert <model_dir> <out.gguf> [tokenizer_pre] [adapter]
// tokenizer_pre (e.g. "smollm") bakes the GPT2-BPE tokenizer; adapter "bonsai-qlinear" folds .scales and
// bakes the Llama-style tokenizer.
#include <cstdio>

#include "hf_to_gguf.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model_dir> <out.gguf> [tokenizer_pre] [adapter]\n", argv[0]);
        return 2;
    }
    const std::string pre = argc > 3 ? argv[3] : "";
    const std::string adapter = argc > 4 ? argv[4] : "";
    try {
        size_t n = llmedge::convert::convert_llama_dir(argv[1], argv[2], pre, adapter);
        std::printf("converted %zu tensors -> %s%s%s\n", n, argv[2],
                    pre.empty() ? "" : " (+gpt2 tokenizer)", adapter.empty() ? "" : " (+adapter)");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "convert failed: %s\n", e.what());
        return 1;
    }
}
