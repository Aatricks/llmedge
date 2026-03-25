cmake_minimum_required(VERSION 3.22)
project(video_jni_tests LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(JNI REQUIRED)
find_package(Java REQUIRED COMPONENTS Development)
find_package(Threads REQUIRED)

get_filename_component(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../llama.cpp" ABSOLUTE)
set(GGML_DIR ${LLAMA_DIR}/ggml)
set(COMMON_DIR ${LLAMA_DIR}/common)
set(VENDOR_DIR ${LLAMA_DIR}/vendor)

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

set(SMOLLM_CHAT_TEMPLATE_TEST_SOURCES
    test_smollm_chat_templates.cpp
    ../../main/cpp/llama_build_info.cpp

    ${GGML_DIR}/src/ggml-alloc.c
    ${GGML_DIR}/src/ggml-backend.cpp
    ${GGML_DIR}/src/ggml-threading.cpp
    ${GGML_DIR}/src/ggml-quants.c
    ${GGML_DIR}/src/ggml-backend-reg.cpp
    ${GGML_DIR}/src/ggml-backend-dl.cpp
    ${GGML_DIR}/src/ggml-opt.cpp
    ${GGML_DIR}/src/ggml-cpu/ops.cpp
    ${GGML_DIR}/src/ggml-cpu/vec.cpp
    ${GGML_DIR}/src/ggml-cpu/quants.c
    ${GGML_DIR}/src/ggml-cpu/repack.cpp
    ${GGML_DIR}/src/ggml-cpu/traits.cpp
    ${GGML_DIR}/src/ggml-cpu/unary-ops.cpp
    ${GGML_DIR}/src/ggml-cpu/binary-ops.cpp
    ${GGML_DIR}/src/ggml-cpu/ggml-cpu.c
    ${GGML_DIR}/src/ggml-cpu/ggml-cpu.cpp
    ${GGML_DIR}/src/ggml.c
    ${GGML_DIR}/src/gguf.cpp

    ${COMMON_DIR}/chat-auto-parser-generator.cpp
    ${COMMON_DIR}/chat-auto-parser-helpers.cpp
    ${COMMON_DIR}/chat-diff-analyzer.cpp
    ${COMMON_DIR}/chat-peg-parser.cpp
    ${COMMON_DIR}/chat.cpp
    ${COMMON_DIR}/common.cpp
    ${COMMON_DIR}/json-schema-to-grammar.cpp
    ${COMMON_DIR}/log.cpp
    ${COMMON_DIR}/ngram-cache.cpp
    ${COMMON_DIR}/peg-parser.cpp
    ${COMMON_DIR}/reasoning-budget.cpp
    ${COMMON_DIR}/sampling.cpp
    ${COMMON_DIR}/unicode.cpp
    ${COMMON_DIR}/jinja/caps.cpp
    ${COMMON_DIR}/jinja/lexer.cpp
    ${COMMON_DIR}/jinja/parser.cpp
    ${COMMON_DIR}/jinja/runtime.cpp
    ${COMMON_DIR}/jinja/string.cpp
    ${COMMON_DIR}/jinja/value.cpp
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|x86)$")
    list(APPEND SMOLLM_CHAT_TEMPLATE_TEST_SOURCES
        ${GGML_DIR}/src/ggml-cpu/arch/x86/quants.c
        ${GGML_DIR}/src/ggml-cpu/arch/x86/repack.cpp
    )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)")
    list(APPEND SMOLLM_CHAT_TEMPLATE_TEST_SOURCES
        ${GGML_DIR}/src/ggml-cpu/arch/arm/quants.c
        ${GGML_DIR}/src/ggml-cpu/arch/arm/repack.cpp
    )
endif()

file(GLOB LLAMA_CORE_SOURCES
    "${LLAMA_DIR}/src/*.cpp"
    "${LLAMA_DIR}/src/*.c"
)
if(LLAMA_CORE_SOURCES)
    list(APPEND SMOLLM_CHAT_TEMPLATE_TEST_SOURCES ${LLAMA_CORE_SOURCES})
endif()

if(EXISTS "${LLAMA_DIR}/src/models")
    file(GLOB LLAMA_MODEL_SOURCES "${LLAMA_DIR}/src/models/*.cpp")
    if(LLAMA_MODEL_SOURCES)
        list(APPEND SMOLLM_CHAT_TEMPLATE_TEST_SOURCES ${LLAMA_MODEL_SOURCES})
    endif()
endif()

add_executable(smollm_chat_template_tests
    ${SMOLLM_CHAT_TEMPLATE_TEST_SOURCES}
)

target_include_directories(smollm_chat_template_tests PRIVATE
    ../../main/cpp
    ${COMMON_DIR}
    ${GGML_DIR}/include
    ${GGML_DIR}/src
    ${GGML_DIR}/src/ggml-cpu
    ${LLAMA_DIR}/include
    ${LLAMA_DIR}/src
    ${VENDOR_DIR}
)

target_compile_definitions(smollm_chat_template_tests PRIVATE
    _GNU_SOURCE
    GGML_COMMIT=""
    GGML_VERSION=""
    GGML_USE_CPU
)

target_link_libraries(smollm_chat_template_tests PRIVATE
    Threads::Threads
    dl
)
