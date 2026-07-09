# Shared llmedge native build configuration.
# The root CMakeLists.txt should set LLMEDGE_CPP_ROOT before including this file.

include("${LLMEDGE_CPP_ROOT}/cmake/llmedge-mods.cmake")

get_filename_component(LLAMA_DIR_UPSTREAM "${LLMEDGE_CPP_ROOT}/../../../../llama.cpp" ABSOLUTE)
get_filename_component(LLMEDGE_REPO_ROOT "${LLMEDGE_CPP_ROOT}/../../../.." ABSOLUTE)
set(LLMEDGE_MODS_DIR "${LLMEDGE_REPO_ROOT}/mods")
llmedge_prepare_patch_modded_tree(
        NAME "llama.cpp"
        SOURCE_ROOT "${LLAMA_DIR_UPSTREAM}"
        MODS_ROOT "${LLMEDGE_MODS_DIR}/llama.cpp"
        USE_MODS_VAR "LLMEDGE_LLAMA_USE_MODS"
        OUT_VAR LLAMA_DIR
)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")

option(LLMEDGE_ANDROID_OPENCL "Enable experimental Android OpenCL backend support" OFF)

set(GGML_DIR ${LLAMA_DIR}/ggml)
set(COMMON_DIR ${LLAMA_DIR}/common)
set(VENDOR_DIR ${LLAMA_DIR}/vendor)
if(EXISTS "${LLAMA_DIR}/tools/mtmd/mtmd.cpp")
        set(MTMD_DIR "${LLAMA_DIR}/tools/mtmd")
elseif(EXISTS "${LLAMA_DIR}/examples/mtmd/mtmd.cpp")
        set(MTMD_DIR "${LLAMA_DIR}/examples/mtmd")
else()
        message(FATAL_ERROR "No compatible mtmd sources found under ${LLAMA_DIR}")
endif()

set(LLMEDGE_LLAMA_USE_MEMORY_API OFF CACHE BOOL "Use llama_memory_* APIs instead of llama_kv_cache_* compatibility APIs" FORCE)
file(GLOB LLMEDGE_COMMON_SOURCES CONFIGURE_DEPENDS
        "${COMMON_DIR}/*.cpp"
        "${COMMON_DIR}/jinja/*.cpp"
)
# reasoning-budget.cpp is #included directly by sampling.cpp (upstream ik_llama), so
# compiling it standalone as well produces duplicate symbols at link time. Drop it
# from the glob; sampling.cpp provides its definitions.
list(REMOVE_ITEM LLMEDGE_COMMON_SOURCES "${COMMON_DIR}/reasoning-budget.cpp")
set(LLMEDGE_IQK_BASE_SOURCES
        ${GGML_DIR}/src/iqk/iqk_quantize.cpp
        ${GGML_DIR}/src/iqk/iqk_cpu_ops.cpp
        ${GGML_DIR}/src/iqk/iqk_mul_mat.cpp
        ${GGML_DIR}/src/iqk/iqk_flash_attn.cpp
)
# iqk gemm + flash-attention kernels. Globbed (not hardcoded) so new upstream
# variants (e.g. iqk_fa_512_512 added in this bump) are picked up automatically;
# a missing kernel shows up as an undefined iqk_* symbol at link time.
file(GLOB LLMEDGE_IQK_DOTPROD_SOURCES
        "${GGML_DIR}/src/iqk/iqk_gemm_*.cpp"
        "${GGML_DIR}/src/iqk/fa/*.cpp"
)
# The patched IQK sources rely on IQK_FORCE_IMPLEMENT in two cases:
# - desktop x86 host builds, where upstream gating does not enable the implementation by default
# - Android arm64 targets such as ggufreader and the baseline smollm, which compile IQK sources without
#   dotprod-specific march flags but still need the generic AArch64 NEON helper implementations
if ((NOT ANDROID AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
        OR (ANDROID AND "${ANDROID_ABI}" STREQUAL "arm64-v8a"))
        set_source_files_properties(
                ${GGML_DIR}/src/iqk/iqk_cpu_ops.cpp
                PROPERTIES COMPILE_DEFINITIONS "IQK_FORCE_IMPLEMENT"
        )
endif()
if (ANDROID AND "${ANDROID_ABI}" STREQUAL "x86_64")
        set_source_files_properties(
                ${LLMEDGE_IQK_BASE_SOURCES}
                PROPERTIES COMPILE_OPTIONS "-mavx;-mavx2;-mf16c;-mfma"
        )
endif()
set(LLMEDGE_GGML_CORE_SOURCES
        ${GGML_DIR}/src/ggml-alloc.c
        ${GGML_DIR}/src/ggml-backend.cpp
        ${GGML_DIR}/src/ggml-quants.c
        ${GGML_DIR}/src/ggml.c
)
set(LLMEDGE_GGML_VULKAN_SOURCES)
set(LLMEDGE_GGML_EXTRA_INCLUDES)
set(LLMEDGE_GGML_EXTRA_LIBS)
set(LLMEDGE_GGML_EXTRA_DEFINITIONS)
# Upstream ggml always compiles ggml-aarch64.c because it now provides generic
# quantization/repacking entry points that ggml.c references across ABIs.
set(LLMEDGE_GGML_ARCH_SOURCES
        ${GGML_DIR}/src/ggml-aarch64.c
)
set(SMOLLM_SOURCES
        ${LLMEDGE_GGML_CORE_SOURCES}
        ${LLMEDGE_GGML_ARCH_SOURCES}

        # llama.cpp core sources are globbed below (LLAMA_CPP_SOURCES) to tolerate upstream file renames.

        ${LLMEDGE_COMMON_SOURCES}

        ${LLMEDGE_CPP_ROOT}/LLMInference.cpp
        ${LLMEDGE_CPP_ROOT}/llm_backend_support.cpp
        ${LLMEDGE_CPP_ROOT}/smollm.cpp
        ${LLMEDGE_CPP_ROOT}/smollm_jni_completion.cpp
        ${LLMEDGE_CPP_ROOT}/smollm_jni_embeddings.cpp
        ${LLMEDGE_CPP_ROOT}/smollm_jni_load.cpp
        ${LLMEDGE_CPP_ROOT}/smollm_jni_state.cpp
        ${LLMEDGE_CPP_ROOT}/smollm_jni_convert.cpp
        # On-device safetensors -> GGUF converter (Track B / Phase B2). No ggml/llama dependency.
        ${LLMEDGE_CPP_ROOT}/convert/safetensors_reader.cpp
        ${LLMEDGE_CPP_ROOT}/convert/gguf_writer.cpp
        ${LLMEDGE_CPP_ROOT}/convert/tokenizer_bake.cpp
        ${LLMEDGE_CPP_ROOT}/convert/hf_to_gguf.cpp
        # libmtmd (multimodal projector) from llama.cpp
        ${MTMD_DIR}/mtmd.cpp
        ${MTMD_DIR}/mtmd-helper.cpp
        ${MTMD_DIR}/mtmd-audio.cpp
        ${MTMD_DIR}/clip.cpp
)

# mtmd may provide additional model-specific graph implementations under models/*.cpp
file(GLOB MTMD_MODEL_SOURCES "${MTMD_DIR}/models/*.cpp")
if(MTMD_MODEL_SOURCES)
        list(APPEND SMOLLM_SOURCES ${MTMD_MODEL_SOURCES})
endif()

file(GLOB LLAMA_CPP_SOURCES
        "${LLAMA_DIR}/src/*.cpp"
        "${LLAMA_DIR}/src/*.c"
)
if(LLAMA_CPP_SOURCES)
        list(APPEND SMOLLM_SOURCES ${LLAMA_CPP_SOURCES})
else()
        message(FATAL_ERROR "No llama.cpp sources found under ${LLAMA_DIR}/src")
endif()

# If llama.cpp contains per-model source files, append them to SMOLLM_SOURCES.
# Newer llama.cpp versions keep many model-specific builders in src/models/*.cpp; without
# these, link errors like llm_build_* will occur.
if(EXISTS "${LLAMA_DIR}/src/models")
        file(GLOB LLAMA_MODEL_SOURCES "${LLAMA_DIR}/src/models/*.cpp")
        if(LLAMA_MODEL_SOURCES)
                list(APPEND SMOLLM_SOURCES ${LLAMA_MODEL_SOURCES})
        endif()
endif()

# ik_llama.cpp keeps per-model graph builders in src/graphs/*.cpp (declared in
# llama-build-context.h, dispatched from llama-build-context.cpp); without these the
# link fails with undefined llm_build_context::build_<model> symbols.
if(EXISTS "${LLAMA_DIR}/src/graphs")
        file(GLOB LLAMA_GRAPH_SOURCES "${LLAMA_DIR}/src/graphs/*.cpp")
        if(LLAMA_GRAPH_SOURCES)
                list(APPEND SMOLLM_SOURCES ${LLAMA_GRAPH_SOURCES})
        endif()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/llmedge-vulkan.cmake")

include("${CMAKE_CURRENT_LIST_DIR}/llmedge-opencl.cmake")

set(GGUF_READER_SOURCES
        ${LLMEDGE_GGML_CORE_SOURCES}
        ${LLMEDGE_GGML_ARCH_SOURCES}
        ${LLMEDGE_CPP_ROOT}/gguf_reader_internal.cpp
        ${LLMEDGE_CPP_ROOT}/GGUFReader.cpp
)

add_compile_options("-ffile-prefix-map=${LLAMA_DIR}=.")
add_link_options("LINKER:--build-id=none")

# Strip symbol/debug sections at link time for Release builds. AGP's
# stripReleaseDebugSymbols step fails on these large libs ("Unable to strip ...,
# packaging them as they are"), leaving full debug_info in the APK (~115MB/lib).
# --strip-all removes .symtab + debug sections but keeps the exported .dynsym
# (JNI Java_* / JNI_OnLoad), shrinking each smollm lib ~115MB -> ~19MB.
add_link_options("$<$<CONFIG:Release>:LINKER:--strip-all>")

include("${CMAKE_CURRENT_LIST_DIR}/llmedge-smollm-targets.cmake")
