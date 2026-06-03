# ------------------------------------------------------------
# Vulkan configuration (Global)
# ------------------------------------------------------------
option(GGML_VULKAN "Enable Vulkan support" ON)
set(VULKAN_ENABLED ${GGML_VULKAN})

if (VULKAN_ENABLED)
    # Provide Vulkan library and include path for Android NDK
    find_library(VULKAN_LIB vulkan REQUIRED)
    # Use the NDK toolchain's sysroot (set per-host by android.toolchain.cmake) so this
    # resolves correctly on Linux CI (prebuilt/linux-x86_64) and macOS (prebuilt/darwin-x86_64).
    if (CMAKE_SYSROOT)
        set(Vulkan_INCLUDE_DIR "${CMAKE_SYSROOT}/usr/include")
    else()
        file(GLOB _llmedge_ndk_sysroot_inc "${ANDROID_NDK}/toolchains/llvm/prebuilt/*/sysroot/usr/include")
        list(GET _llmedge_ndk_sysroot_inc 0 Vulkan_INCLUDE_DIR)
    endif()

    # Download Vulkan C++ headers (vulkan.hpp) which don't ship with Android NDK
    include(FetchContent)
    FetchContent_Declare(
        vulkan_hpp
        GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Hpp.git
        GIT_TAG        v1.3.275  # Must match Android NDK's Vulkan version
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(vulkan_hpp)

    # Configure Vulkan for both llama.cpp and stable-diffusion.cpp ggml-vulkan
    set(Vulkan_INCLUDE_DIRS "${Vulkan_INCLUDE_DIR};${vulkan_hpp_SOURCE_DIR}")
    set(Vulkan_LIBRARIES "${VULKAN_LIB}")
    set(Vulkan_FOUND ON)
    set(Vulkan_INCLUDE_DIR "${Vulkan_INCLUDE_DIR}" CACHE PATH "Vulkan include directory" FORCE)
    set(Vulkan_LIBRARY "${VULKAN_LIB}" CACHE FILEPATH "Vulkan library" FORCE)

    # Find glslc compiler for Vulkan shader compilation
    find_program(Vulkan_GLSLC_EXECUTABLE NAMES glslc HINTS $ENV{VULKAN_SDK}/bin)
    message(STATUS "Vulkan_GLSLC_EXECUTABLE: ${Vulkan_GLSLC_EXECUTABLE}")
    if(Vulkan_GLSLC_EXECUTABLE)
        set(Vulkan_glslc_FOUND ON)
        set(Vulkan_glslc_EXECUTABLE "${Vulkan_GLSLC_EXECUTABLE}")
    endif()

    if(NOT Vulkan_GLSLC_EXECUTABLE)
        message(FATAL_ERROR "GGML_VULKAN is enabled but glslc was not found")
    endif()

    function(llmedge_test_shader_extension_support extension_name test_shader_file result_variable)
        execute_process(
                COMMAND ${Vulkan_GLSLC_EXECUTABLE} -o - -fshader-stage=compute --target-env=vulkan1.3 "${test_shader_file}"
                OUTPUT_QUIET
                ERROR_VARIABLE glslc_error
        )

        if("${glslc_error}" MATCHES ".*extension not supported: ${extension_name}.*")
            set(${result_variable} OFF PARENT_SCOPE)
        else()
            set(${result_variable} ON PARENT_SCOPE)
        endif()
    endfunction()

    set(_llmedge_vk_shader_gen_cmake_args)
    if (NOT GGML_VULKAN_NO_COOPMAT)
        llmedge_test_shader_extension_support(
                "GL_KHR_cooperative_matrix"
                "${GGML_DIR}/src/vulkan-shaders/test_coopmat_support.comp"
                "GGML_VULKAN_COOPMAT_GLSLC_SUPPORT"
        )
        if (GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            list(APPEND LLMEDGE_GGML_EXTRA_DEFINITIONS GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            list(APPEND _llmedge_vk_shader_gen_cmake_args -DGGML_VULKAN_COOPMAT_GLSLC_SUPPORT=ON)
        endif()
    endif()

    if (NOT GGML_VULKAN_NO_COOPMAT2)
        llmedge_test_shader_extension_support(
                "GL_NV_cooperative_matrix2"
                "${GGML_DIR}/src/vulkan-shaders/test_coopmat2_support.comp"
                "GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT"
        )
        if (GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            list(APPEND LLMEDGE_GGML_EXTRA_DEFINITIONS GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            list(APPEND _llmedge_vk_shader_gen_cmake_args -DGGML_VULKAN_COOPMAT2_GLSLC_SUPPORT=ON)
        endif()
    endif()

    if (NOT GGML_VULKAN_NO_INT_DOT)
        llmedge_test_shader_extension_support(
                "GL_EXT_integer_dot_product"
                "${GGML_DIR}/src/vulkan-shaders/test_integer_dot_support.comp"
                "GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT"
        )
        if (GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            list(APPEND LLMEDGE_GGML_EXTRA_DEFINITIONS GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            list(APPEND _llmedge_vk_shader_gen_cmake_args -DGGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT=ON)
        endif()
    endif()

    if (NOT GGML_VULKAN_NO_BF16)
        llmedge_test_shader_extension_support(
                "GL_EXT_bfloat16"
                "${GGML_DIR}/src/vulkan-shaders/test_bfloat16_support.comp"
                "GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT"
        )
        if (GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
            list(APPEND LLMEDGE_GGML_EXTRA_DEFINITIONS GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
            list(APPEND _llmedge_vk_shader_gen_cmake_args -DGGML_VULKAN_BFLOAT16_GLSLC_SUPPORT=ON)
        endif()
    endif()

    list(APPEND LLMEDGE_GGML_EXTRA_DEFINITIONS GGML_USE_VULKAN)
    if (GGML_VULKAN_CHECK_RESULTS)
        list(APPEND LLMEDGE_GGML_EXTRA_DEFINITIONS GGML_VULKAN_CHECK_RESULTS)
    endif()
    if (GGML_VULKAN_DEBUG)
        list(APPEND LLMEDGE_GGML_EXTRA_DEFINITIONS GGML_VULKAN_DEBUG)
    endif()
    if (GGML_VULKAN_MEMORY_DEBUG)
        list(APPEND LLMEDGE_GGML_EXTRA_DEFINITIONS GGML_VULKAN_MEMORY_DEBUG)
    endif()
    if (GGML_VULKAN_SHADER_DEBUG_INFO)
        list(APPEND LLMEDGE_GGML_EXTRA_DEFINITIONS GGML_VULKAN_SHADER_DEBUG_INFO)
        list(APPEND _llmedge_vk_shader_gen_cmake_args -DGGML_VULKAN_SHADER_DEBUG_INFO=ON)
    endif()
    if (GGML_VULKAN_PERF)
        list(APPEND LLMEDGE_GGML_EXTRA_DEFINITIONS GGML_VULKAN_PERF)
    endif()
    if (GGML_VULKAN_VALIDATE)
        list(APPEND LLMEDGE_GGML_EXTRA_DEFINITIONS GGML_VULKAN_VALIDATE)
    endif()
    if (GGML_VULKAN_RUN_TESTS)
        list(APPEND LLMEDGE_GGML_EXTRA_DEFINITIONS GGML_VULKAN_RUN_TESTS)
    endif()

    if (CMAKE_MAKE_PROGRAM)
        set(_LLMEDGE_GGML_VK_HOST_TC "${CMAKE_CURRENT_BINARY_DIR}/ggml_vulkan_host_toolchain.cmake")
        file(WRITE "${_LLMEDGE_GGML_VK_HOST_TC}" "# Autogenerated by llmedge CMake\n"
                "set(CMAKE_SYSTEM_NAME \"${CMAKE_HOST_SYSTEM_NAME}\")\n"
                "set(CMAKE_SYSTEM_PROCESSOR \"${CMAKE_HOST_SYSTEM_PROCESSOR}\")\n"
                "set(CMAKE_BUILD_TYPE Release)\n"
                "set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)\n"
                "set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY NEVER)\n"
                "set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE NEVER)\n"
                "set(CMAKE_MAKE_PROGRAM \"${CMAKE_MAKE_PROGRAM}\" CACHE FILEPATH \"\" FORCE)\n"
                "set(CMAKE_GENERATOR \"Ninja\" CACHE STRING \"\" FORCE)\n"
        )
        set(GGML_VULKAN_SHADERS_GEN_TOOLCHAIN "${_LLMEDGE_GGML_VK_HOST_TC}" CACHE STRING "Toolchain for ggml-vulkan shader generator" FORCE)
    endif()

    include(ExternalProject)
    set(_llmedge_vk_shader_gen_install_dir "${CMAKE_CURRENT_BINARY_DIR}/llmedge-vulkan-shaders-gen")
    set(_llmedge_vk_shader_gen_build_dir "${CMAKE_CURRENT_BINARY_DIR}/llmedge-vulkan-shaders-gen-build")
    set(_llmedge_vk_shader_gen_args
            -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
            -DCMAKE_INSTALL_BINDIR=.
            -DCMAKE_BUILD_TYPE=Release
    )
    if (CMAKE_CROSSCOMPILING AND GGML_VULKAN_SHADERS_GEN_TOOLCHAIN)
        list(APPEND _llmedge_vk_shader_gen_args -DCMAKE_TOOLCHAIN_FILE=${GGML_VULKAN_SHADERS_GEN_TOOLCHAIN})
    endif()
    list(APPEND _llmedge_vk_shader_gen_args ${_llmedge_vk_shader_gen_cmake_args})

    ExternalProject_Add(
            llmedge-text-vulkan-shaders-gen
            SOURCE_DIR "${GGML_DIR}/src/vulkan-shaders"
            BINARY_DIR "${_llmedge_vk_shader_gen_build_dir}"
            INSTALL_DIR "${_llmedge_vk_shader_gen_install_dir}"
            CMAKE_ARGS ${_llmedge_vk_shader_gen_args}
            BUILD_COMMAND ${CMAKE_COMMAND} --build . --config Release
            INSTALL_COMMAND ${CMAKE_COMMAND} -E env --unset=DESTDIR ${CMAKE_COMMAND} --install . --config Release
    )

    set(_llmedge_vk_host_suffix "")
    if (CMAKE_HOST_WIN32)
        set(_llmedge_vk_host_suffix ".exe")
    endif()
    set(_llmedge_vk_genshaders_cmd "${_llmedge_vk_shader_gen_install_dir}/vulkan-shaders-gen${_llmedge_vk_host_suffix}")
    set(_llmedge_vk_header "${CMAKE_CURRENT_BINARY_DIR}/ggml-vulkan-shaders.hpp")
    set(_llmedge_vk_source "${CMAKE_CURRENT_BINARY_DIR}/ggml-vulkan-shaders.cpp")
    set(_llmedge_vk_input_dir "${GGML_DIR}/src/vulkan-shaders")
    set(_llmedge_vk_output_dir "${CMAKE_CURRENT_BINARY_DIR}/vulkan-shaders.spv")
    file(GLOB _llmedge_vk_shader_files CONFIGURE_DEPENDS "${_llmedge_vk_input_dir}/*.comp")

    add_custom_command(
            OUTPUT "${_llmedge_vk_header}" "${_llmedge_vk_source}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${_llmedge_vk_output_dir}"
            COMMAND "${_llmedge_vk_genshaders_cmd}"
                    --glslc "${Vulkan_GLSLC_EXECUTABLE}"
                    --input-dir "${_llmedge_vk_input_dir}"
                    --output-dir "${_llmedge_vk_output_dir}"
                    --target-hpp "${_llmedge_vk_header}"
                    --target-cpp "${_llmedge_vk_source}"
                    --no-clean
            DEPENDS ${_llmedge_vk_shader_files} llmedge-text-vulkan-shaders-gen
            COMMENT "Generate Vulkan shaders for llmedge text runtime"
    )

    list(APPEND LLMEDGE_GGML_VULKAN_SOURCES
            ${GGML_DIR}/src/ggml-vulkan.cpp
            "${_llmedge_vk_source}"
    )
    list(APPEND LLMEDGE_GGML_EXTRA_INCLUDES
            ${Vulkan_INCLUDE_DIRS}
            "${CMAKE_CURRENT_BINARY_DIR}"
    )
    list(APPEND LLMEDGE_GGML_EXTRA_LIBS "${VULKAN_LIB}")
endif()

if(LLMEDGE_GGML_VULKAN_SOURCES)
        list(APPEND SMOLLM_SOURCES ${LLMEDGE_GGML_VULKAN_SOURCES})
endif()
