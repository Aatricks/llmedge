# Stable Diffusion / Wan JNI target and overlay logic.

get_filename_component(SD_DIR_UPSTREAM "${LLMEDGE_CPP_ROOT}/../../../../stable-diffusion.cpp" ABSOLUTE)

# Optional: build stable-diffusion.cpp from a copied tree with overlays from the repo's
# `mods/` directory. This keeps the git submodule working tree pristine (no `-dirty`).
#
# Controls:
#   -DSD_ROOT_OVERRIDE=/abs/path      : use that stable-diffusion.cpp root directly
#   -DLLMEDGE_SDCPP_USE_MODS=ON       : enable copy+overlay
#   -DLLMEDGE_SDCPP_MODS_FILES=...    : semicolon- or comma-separated list of overlay files
set(SD_ROOT_OVERRIDE "" CACHE PATH "Override path to stable-diffusion.cpp root")

get_filename_component(LLMEDGE_REPO_ROOT "${LLMEDGE_CPP_ROOT}/../../../.." ABSOLUTE)
set(LLMEDGE_MODS_DIR "${LLMEDGE_REPO_ROOT}/mods")

set(_LLMEDGE_SDCPP_USE_MODS_DEFAULT OFF)
if (EXISTS "${LLMEDGE_MODS_DIR}/stable-diffusion.cpp" OR EXISTS "${LLMEDGE_MODS_DIR}/stable-diffusion.h")
        set(_LLMEDGE_SDCPP_USE_MODS_DEFAULT ON)
endif()
option(LLMEDGE_SDCPP_USE_MODS "Build stable-diffusion.cpp from a copied tree with mods/ overlays (keeps submodule clean)" ${_LLMEDGE_SDCPP_USE_MODS_DEFAULT})

# Default overlays: only include wan.hpp when it exists in mods/.
set(_LLMEDGE_SDCPP_MODS_FILES_DEFAULT
    stable-diffusion.cpp
    stable-diffusion.h
    ggml_extend.hpp
    util.cpp
    ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp
)
if (EXISTS "${LLMEDGE_MODS_DIR}/wan.hpp")
        set(_LLMEDGE_SDCPP_MODS_FILES_DEFAULT "wan.hpp;${_LLMEDGE_SDCPP_MODS_FILES_DEFAULT}")
endif()
set(LLMEDGE_SDCPP_MODS_FILES "${_LLMEDGE_SDCPP_MODS_FILES_DEFAULT}" CACHE STRING "Overlay files from mods/ onto the copied stable-diffusion.cpp tree (semicolon- or comma-separated)")

set(SD_DIR "${SD_DIR_UPSTREAM}")
if (SD_ROOT_OVERRIDE)
        get_filename_component(SD_DIR "${SD_ROOT_OVERRIDE}" ABSOLUTE)
elseif (LLMEDGE_SDCPP_USE_MODS OR (DEFINED ENV{LLMEDGE_SDCPP_USE_MODS} AND "$ENV{LLMEDGE_SDCPP_USE_MODS}" STREQUAL "1"))
        set(PATCHED_SD_ROOT "${CMAKE_CURRENT_BINARY_DIR}/patched-sd-src")
        message(STATUS "LLMEDGE_SDCPP_USE_MODS enabled: copying stable-diffusion.cpp sources to ${PATCHED_SD_ROOT} and overlaying mods")
        execute_process(COMMAND ${CMAKE_COMMAND} -E rm -rf "${PATCHED_SD_ROOT}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${PATCHED_SD_ROOT}")
        execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory "${SD_DIR_UPSTREAM}" "${PATCHED_SD_ROOT}")

        # Remove any stale ggml-vulkan shader-generator build/prefix directories that
        # may have been created by a previous CMake configure which referenced a
        # different stable-diffusion.cpp source directory.
        file(GLOB _vulkan_gen_dirs "${CMAKE_CURRENT_BINARY_DIR}/stable-diffusion.cpp/ggml/src/ggml-vulkan/*-gen*")
        foreach(_d IN LISTS _vulkan_gen_dirs)
                if (EXISTS "${_d}")
                        message(STATUS "LLMEDGE_SDCPP_USE_MODS: removing stale Vulkan gen dir ${_d}")
                        execute_process(COMMAND ${CMAKE_COMMAND} -E remove_directory "${_d}")
                endif()
        endforeach()

        set(_mods_files "${LLMEDGE_SDCPP_MODS_FILES}")
        string(REPLACE "," ";" _mods_files "${_mods_files}")
        foreach(f IN LISTS _mods_files)
                string(STRIP "${f}" f_trimmed)
                if (f_trimmed STREQUAL "")
                        continue()
                endif()
                if (EXISTS "${LLMEDGE_MODS_DIR}/${f_trimmed}")
                        execute_process(COMMAND ${CMAKE_COMMAND} -E copy "${LLMEDGE_MODS_DIR}/${f_trimmed}" "${PATCHED_SD_ROOT}/${f_trimmed}")
                else()
                        message(STATUS "LLMEDGE_SDCPP_USE_MODS: mods/${f_trimmed} not found; skipping")
                endif()
        endforeach()

        set(SD_DIR "${PATCHED_SD_ROOT}")
endif()

if (NOT EXISTS "${SD_DIR}/wan.hpp")
        message(FATAL_ERROR "stable-diffusion.cpp Wan headers not found at ${SD_DIR}. Run git submodule update --init --recursive (or disable overlays).")
endif()

option(WAN_SUPPORT "Enable Wan video generation support" ON)

# Configure stable-diffusion.cpp build options before adding the subdirectory.
# On Windows hosts the upstream Vulkan shader generator still tends to fail under the Android
# cross-build environment, so only force Vulkan on by default for non-Windows hosts.
set(_LLMEDGE_DEFAULT_SD_VULKAN ON)
if (CMAKE_HOST_WIN32)
        set(_LLMEDGE_DEFAULT_SD_VULKAN OFF)
endif()
set(SD_BUILD_EXAMPLES OFF CACHE BOOL "sd: build examples" FORCE)
set(SD_BUILD_SHARED_LIBS OFF CACHE BOOL "sd: build shared libs" FORCE)
set(SD_BUILD_SHARED_GGML_LIB OFF CACHE BOOL "sd: build ggml as a separate shared lib" FORCE)
set(SD_USE_SYSTEM_GGML OFF CACHE BOOL "sd: use system-installed GGML library" FORCE)
set(SD_VULKAN ${_LLMEDGE_DEFAULT_SD_VULKAN} CACHE BOOL "sd: vulkan backend" FORCE)
set(SD_CUDA OFF CACHE BOOL "sd: cuda backend" FORCE)
set(SD_METAL OFF CACHE BOOL "sd: metal backend" FORCE)
set(SD_OPENCL OFF CACHE BOOL "sd: opencl backend" FORCE)
set(SD_SYCL OFF CACHE BOOL "sd: sycl backend" FORCE)
set(SD_MUSA OFF CACHE BOOL "sd: musa backend" FORCE)

# Ensure ggml-vulkan's ExternalProject (vulkan-shaders-gen) can find the correct build tool.
if (CMAKE_MAKE_PROGRAM)
        set(_GGML_VK_HOST_TC "${CMAKE_CURRENT_BINARY_DIR}/ggml_vulkan_host_toolchain.cmake")
        file(WRITE "${_GGML_VK_HOST_TC}" "# Autogenerated by llmedge CMake\n"
                "set(CMAKE_SYSTEM_NAME \"${CMAKE_HOST_SYSTEM_NAME}\")\n"
                "set(CMAKE_SYSTEM_PROCESSOR \"${CMAKE_HOST_SYSTEM_PROCESSOR}\")\n"
                "set(CMAKE_BUILD_TYPE Release)\n"
                "set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)\n"
                "set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY NEVER)\n"
                "set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE NEVER)\n"
                "set(CMAKE_MAKE_PROGRAM \"${CMAKE_MAKE_PROGRAM}\" CACHE FILEPATH \"\" FORCE)\n"
                "set(CMAKE_GENERATOR \"Ninja\" CACHE STRING \"\" FORCE)\n"
        )
        set(GGML_VULKAN_SHADERS_GEN_TOOLCHAIN "${_GGML_VK_HOST_TC}" CACHE STRING "Toolchain for ggml-vulkan shader generator" FORCE)
endif()

# Enable ggml Vulkan for stable-diffusion's ggml only when the backend is enabled.
set(GGML_VULKAN ${SD_VULKAN} CACHE BOOL "ggml: vulkan" FORCE)
message(STATUS "Stable Diffusion Vulkan=${SD_VULKAN}; GGML Vulkan=${GGML_VULKAN}")

add_subdirectory(${SD_DIR} ${CMAKE_CURRENT_BINARY_DIR}/stable-diffusion.cpp)

if (TARGET ggml-vulkan AND vulkan_hpp_SOURCE_DIR)
    target_include_directories(ggml-vulkan PRIVATE ${vulkan_hpp_SOURCE_DIR})
endif()

add_library(sdcpp SHARED
        ${LLMEDGE_CPP_ROOT}/sdcpp_jni.cpp
)

target_include_directories(sdcpp
        PUBLIC
        ${SD_DIR}
        ${SD_DIR}/thirdparty
)

target_compile_features(sdcpp PUBLIC c_std_11 cxx_std_17)

target_compile_options(sdcpp PUBLIC -fvisibility=hidden -fvisibility-inlines-hidden -ffunction-sections -fdata-sections -O3)

target_link_libraries(sdcpp
        android log
        stable-diffusion
)
if(SD_VULKAN)
        target_link_libraries(sdcpp vulkan)
endif()

if (WAN_SUPPORT)
        target_compile_definitions(sdcpp PRIVATE WAN_SUPPORT=1)
        message(STATUS "Wan video support is enabled for sdcpp")
endif()

target_link_options(sdcpp PRIVATE -Wl,--gc-sections -flto -Wl,--exclude-libs,ALL)
