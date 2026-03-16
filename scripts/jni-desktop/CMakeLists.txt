cmake_minimum_required(VERSION 3.10)
project(sdcpp_desktop)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Point to stable-diffusion.cpp root.
set(SD_ROOT_OVERRIDE "" CACHE PATH "Override path to stable-diffusion.cpp root")
if (SD_ROOT_OVERRIDE)
    get_filename_component(SD_ROOT "${SD_ROOT_OVERRIDE}" ABSOLUTE)
else()
    get_filename_component(SD_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../stable-diffusion.cpp" ABSOLUTE)
endif()

get_filename_component(LLMEDGE_CPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../llmedge/src/main/cpp" ABSOLUTE)

# Options to control what to build
option(BUILD_SDCPP "Build stable-diffusion.cpp JNI" ON)
option(BUILD_SMOLLM "Build SmolLM (llama.cpp) JNI" OFF)

# Find JNI
find_package(JNI REQUIRED)

# ------------------------------------------------------------
# Stable Diffusion JNI
# ------------------------------------------------------------
if(BUILD_SDCPP)
    message(STATUS "Using stable-diffusion.cpp root: ${SD_ROOT}")

    # Configure stable-diffusion.cpp options
    set(SD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(SD_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    if(NOT DEFINED SD_VULKAN)
        set(SD_VULKAN ON CACHE BOOL "" FORCE)
    endif()

    # Add stable-diffusion.cpp
    add_subdirectory(${SD_ROOT} stable-diffusion)

    # Build sdcpp shared library
    add_library(sdcpp SHARED
        ${LLMEDGE_CPP_ROOT}/sdcpp_jni.cpp
    )

    target_include_directories(sdcpp PRIVATE
        ${SD_ROOT}
        ${SD_ROOT}/src
        ${SD_ROOT}/thirdparty
        ${JNI_INCLUDE_DIRS}
    )

    target_link_libraries(sdcpp PRIVATE
        stable-diffusion
        ${JNI_LIBRARIES}
    )

    if(SD_VULKAN)
        target_compile_definitions(sdcpp PRIVATE SD_USE_VULKAN)
        target_link_libraries(sdcpp PRIVATE vulkan)
        if(TARGET ggml-vulkan)
            target_link_libraries(sdcpp PRIVATE ggml-vulkan)
        endif()
    endif()

    if(WAN_SUPPORT)
        target_compile_definitions(sdcpp PRIVATE WAN_SUPPORT=1)
    endif()
endif()

# ------------------------------------------------------------
# SmolLM (llama.cpp) JNI
# ------------------------------------------------------------
if(BUILD_SMOLLM)
    get_filename_component(LLAMA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../llama.cpp" ABSOLUTE)
    message(STATUS "Using llama.cpp root: ${LLAMA_ROOT}")

    # Configure llama.cpp options
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)

    # Add llama.cpp
    add_subdirectory(${LLAMA_ROOT} llama.cpp)

    # Build smollm shared library
    add_library(smollm SHARED
        ${LLMEDGE_CPP_ROOT}/smollm.cpp
        ${LLMEDGE_CPP_ROOT}/LLMInference.cpp
        ${LLMEDGE_CPP_ROOT}/GGUFReader.cpp
    )

    target_include_directories(smollm PRIVATE
        ${LLAMA_ROOT}/include
        ${LLAMA_ROOT}/common
        ${LLAMA_ROOT} # For some internal headers if needed
        ${JNI_INCLUDE_DIRS}
    )

    target_link_libraries(smollm PRIVATE
        llama
        common
        ${JNI_LIBRARIES}
    )
endif()

# ------------------------------------------------------------
# Whisper.cpp JNI wrapper for desktop testing
# ------------------------------------------------------------

option(WHISPER_DESKTOP_JNI "Build whisper JNI for desktop" OFF)

if(WHISPER_DESKTOP_JNI)
    get_filename_component(WHISPER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../whisper.cpp" ABSOLUTE)

    if(EXISTS "${WHISPER_ROOT}/include/whisper.h")
        # Configure whisper.cpp build options
        set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
        set(WHISPER_SDL2 OFF CACHE BOOL "" FORCE)
        set(WHISPER_CURL OFF CACHE BOOL "" FORCE)

        add_subdirectory(${WHISPER_ROOT} whisper.cpp)

        add_library(whisper_jni SHARED
            ${LLMEDGE_CPP_ROOT}/whisper_jni.cpp
        )

        target_include_directories(whisper_jni PRIVATE
            ${WHISPER_ROOT}/include
            ${WHISPER_ROOT}/ggml/include
            ${JNI_INCLUDE_DIRS}
        )

        target_link_libraries(whisper_jni PRIVATE
            whisper
            ${JNI_LIBRARIES}
        )

        message(STATUS "Whisper desktop JNI configured")
    else()
        message(WARNING "whisper.cpp not found at ${WHISPER_ROOT}, skipping whisper_jni")
    endif()
endif()
