# ------------------------------------------------------------
# SPIRV-Headers (shared by the Android and desktop/CI builds)
# ------------------------------------------------------------
# Newer ggml-vulkan.cpp patches SPIR-V bytecode at runtime (adds RoundingModeRTE)
# and needs the Khronos `namespace spv` definitions, which it probes for via
# <spirv/unified1/spirv.hpp>. The Android NDK sysroot and the Linux CI toolchain
# do not ship those headers, so fetch them once here and attach to ggml-vulkan
# from both build entry points. Using a single pinned source keeps both
# toolchains compiling against identical definitions.
include_guard(GLOBAL)
include(FetchContent)

FetchContent_Declare(
    spirv_headers
    GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Headers.git
    GIT_TAG        vulkan-sdk-1.3.275.0  # aligned with the Vulkan-Hpp tag used elsewhere
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(spirv_headers)

set(LLMEDGE_SPIRV_INCLUDE_DIR "${spirv_headers_SOURCE_DIR}/include" CACHE INTERNAL "SPIRV-Headers include dir")

# Attach the SPIRV-Headers include dir to a Vulkan-enabled ggml target, if present.
function(llmedge_attach_spirv_headers _target)
    if (TARGET ${_target})
        target_include_directories(${_target} PRIVATE "${LLMEDGE_SPIRV_INCLUDE_DIR}")
    endif()
endfunction()
