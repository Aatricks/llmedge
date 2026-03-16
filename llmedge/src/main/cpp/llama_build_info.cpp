#include "common.h"

// llama.cpp's build normally generates a build-info.cpp (from common/build-info.cpp.in)
// that defines these symbols. llmedge vendors llama.cpp sources directly, so we provide
// minimal definitions to satisfy the linker.
int LLAMA_BUILD_NUMBER = 0;
char const * LLAMA_COMMIT = "";
