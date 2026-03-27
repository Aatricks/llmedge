cmake_minimum_required(VERSION 3.22)
project(video_jni_tests LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(JNI REQUIRED)
find_package(Java REQUIRED COMPONENTS Development)
find_package(Threads REQUIRED)

get_filename_component(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../llama.cpp" ABSOLUTE)
set(GGML_DIR ${LLAMA_DIR}/ggml)
set(COMMON_DIR ${LLAMA_DIR}/common)
set(VENDOR_DIR ${LLAMA_DIR}/vendor)
set(LLMEDGE_LLAMA_USE_MEMORY_API OFF CACHE BOOL "Use llama_memory_* APIs instead of llama_kv_cache_* compatibility APIs" FORCE)

set(TEST_JAVA_SRC ${CMAKE_CURRENT_SOURCE_DIR}/java/io/aatricks/llmedge/NativeTestProgressCallback.java)
set(TEST_JAVA_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/java-classes)
set(TEST_JAVA_CLASS ${TEST_JAVA_OUT_DIR}/io/aatricks/llmedge/NativeTestProgressCallback.class)

add_custom_command(
    OUTPUT ${TEST_JAVA_CLASS}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${TEST_JAVA_OUT_DIR}
    COMMAND ${Java_JAVAC_EXECUTABLE} -d ${TEST_JAVA_OUT_DIR} ${TEST_JAVA_SRC}
    DEPENDS ${TEST_JAVA_SRC}
    COMMENT "Compiling Java progress callback helper"
)

add_custom_target(java_test_classes ALL DEPENDS ${TEST_JAVA_CLASS})

add_library(sd_jni_under_test STATIC
    ../../main/cpp/sdcpp_jni.cpp
    sd_test_stubs.cpp
)

target_compile_definitions(sd_jni_under_test PRIVATE SD_JNI_TESTING)

target_include_directories(sd_jni_under_test PRIVATE
    ../../main/cpp
    ../../../../stable-diffusion.cpp
    ${JNI_INCLUDE_DIRS}
)

add_executable(video_jni_tests
    test_video_jni.cpp
)

target_include_directories(video_jni_tests PRIVATE
    ../../main/cpp
    ../../../../stable-diffusion.cpp
    ${JNI_INCLUDE_DIRS}
)

target_link_libraries(video_jni_tests PRIVATE
    sd_jni_under_test
    ${JNI_LIBRARIES}
)

target_compile_definitions(video_jni_tests PRIVATE
    SD_TEST_JAVA_CLASS_DIR="${TEST_JAVA_OUT_DIR}"
    SD_JNI_TESTING
)

add_dependencies(video_jni_tests java_test_classes)

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_LLGUIDANCE OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
add_subdirectory(${LLAMA_DIR} llama.cpp EXCLUDE_FROM_ALL)

add_executable(smollm_chat_template_tests
    test_smollm_chat_templates.cpp
    test_llmedge_llama_compat.cpp
    test_gguf_reader_metadata.cpp
    ../../main/cpp/gguf_reader_internal.cpp
)

target_include_directories(smollm_chat_template_tests PRIVATE
    ../../main/cpp
    ${GGML_DIR}/include
    ${GGML_DIR}/src
    ${LLAMA_DIR}/include
    ${LLAMA_DIR}/common
    ${VENDOR_DIR}
)

target_compile_definitions(smollm_chat_template_tests PRIVATE
    _GNU_SOURCE
    LLMEDGE_LLAMA_USE_MEMORY_API=$<BOOL:${LLMEDGE_LLAMA_USE_MEMORY_API}>
)

target_link_libraries(smollm_chat_template_tests PRIVATE
    common
    llama
    Threads::Threads
    dl
)
