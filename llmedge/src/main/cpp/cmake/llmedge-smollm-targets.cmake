# compiling for different CPU extensions for Arm64 (aarch64)
# See docs/build_arm_flags.md for more details

function(build_library target_name)
    add_library(
            ${target_name}
            SHARED
            ${SMOLLM_SOURCES}
    )
    target_compile_features(${target_name} PUBLIC c_std_11 cxx_std_20)
    target_include_directories(
            ${target_name}
            PUBLIC
            ${COMMON_DIR}
            ${GGML_DIR}/include
            ${GGML_DIR}/src
            ${GGML_DIR}/src/iqk
            ${LLMEDGE_GGML_EXTRA_INCLUDES}
            ${LLAMA_DIR}/include
            ${LLAMA_DIR}/src
            ${MTMD_DIR}
            ${VENDOR_DIR}
    )

    # Constants GGML_COMMIT and GGML_VERSION in ggml.c
    # are supplied through llama.cpp's CMake script (that in turn gets them from git)
    # As we are NOT using llama.cpp's CMake script, we need to define them here.
    target_compile_definitions(
            ${target_name}
            PRIVATE
            GGML_COMMIT=""
            GGML_VERSION=""
            ${LLMEDGE_GGML_EXTRA_DEFINITIONS}
            LLMEDGE_LLAMA_USE_MEMORY_API=$<BOOL:${LLMEDGE_LLAMA_USE_MEMORY_API}>
    )

    target_compile_options(
            ${target_name}
            PUBLIC
            -fvisibility=hidden -fvisibility-inlines-hidden
    )
    target_compile_options(
            ${target_name}
            PUBLIC
            -ffunction-sections -fdata-sections
    )

    # Enable OpenMP for multi-threaded ggml matrix operations
    target_compile_definitions(${target_name} PRIVATE GGML_USE_OPENMP)
    target_compile_options(${target_name} PUBLIC -fopenmp)

    target_link_libraries(
            ${target_name}
            android log dl
            -fopenmp -static-openmp
            ${LLMEDGE_GGML_EXTRA_LIBS}
    )

    target_link_options(
            ${target_name}
            PRIVATE
            -Wl,--gc-sections -flto
            -Wl,--exclude-libs,ALL
    )
endfunction()

function(enable_iqk_target target_name)
    target_sources(${target_name} PRIVATE ${LLMEDGE_IQK_BASE_SOURCES})
    target_compile_definitions(
            ${target_name}
            PRIVATE
            GGML_USE_IQK_MULMAT
            GGML_IQK_FLASH_ATTENTION
    )
endfunction()

function(enable_iqk_base_target target_name)
    target_sources(${target_name} PRIVATE ${LLMEDGE_IQK_BASE_SOURCES})
endfunction()

function(build_library_arm64 target_name cpu_flags)
    build_library(${target_name})
    enable_iqk_target(${target_name})
    if("${cpu_flags}" MATCHES "dotprod")
        target_sources(${target_name} PRIVATE ${LLMEDGE_IQK_DOTPROD_SOURCES})
    endif()
    target_compile_options(
            ${target_name}
            PUBLIC
            -DGGML_USE_CPU -DGGML_USE_CPU_AARCH64 ${cpu_flags} -O3 -funroll-loops
    )
endfunction()

function(build_library_armv7a target_name cpu_flags fpu fpu_abi)
    build_library(${target_name})
    target_compile_options(
            ${target_name}
            PUBLIC
            -DGGML_USE_CPU ${cpu_flags} ${fpu} ${fpu_abi} -O3 -funroll-loops
    )
endfunction()

function(build_library_universal target_name)
    build_library(${target_name})
    set(_ggml_cpu_flags -DGGML_USE_CPU)
    if ((NOT ${ANDROID_ABI} STREQUAL "armeabi-v7a") AND (NOT ${ANDROID_ABI} STREQUAL "arm64-v8a"))
        enable_iqk_base_target(${target_name})
    endif()
    if (${ANDROID_ABI} STREQUAL "arm64-v8a")
        enable_iqk_target(${target_name})
        list(APPEND _ggml_cpu_flags -DGGML_USE_CPU_AARCH64)
    endif()
    target_compile_options(
            ${target_name}
            PUBLIC
            ${_ggml_cpu_flags} -O3 -funroll-loops
    )
endfunction()
