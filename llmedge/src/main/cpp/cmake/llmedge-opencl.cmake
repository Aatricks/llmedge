# ------------------------------------------------------------
# OpenCL configuration (Android / arm64 only)
# ------------------------------------------------------------
set(LLMEDGE_OPENCL_ENABLED OFF)
set(LLMEDGE_OPENCL_HEADERS_DIR "")

if (LLMEDGE_ANDROID_OPENCL)
    if (NOT ANDROID)
        message(STATUS "LLMEDGE_ANDROID_OPENCL requested on a non-Android build; disabling")
    elseif (NOT ANDROID_ABI STREQUAL "arm64-v8a")
        message(STATUS "LLMEDGE_ANDROID_OPENCL is only supported for Android arm64-v8a; disabling for ${ANDROID_ABI}")
    else()
        include(FetchContent)

        FetchContent_Declare(
            opencl_headers
            GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-Headers.git
            GIT_TAG        v2024.10.24
            GIT_SHALLOW    TRUE
        )
        FetchContent_GetProperties(opencl_headers)
        if (NOT opencl_headers_POPULATED)
            FetchContent_Populate(opencl_headers)
        endif()

        set(LLMEDGE_OPENCL_HEADERS_DIR "${opencl_headers_SOURCE_DIR}")
        set(OPENCL_ICD_LOADER_HEADERS_DIR "${LLMEDGE_OPENCL_HEADERS_DIR}" CACHE PATH "OpenCL headers for Android loader build" FORCE)
        set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

        FetchContent_Declare(
            opencl_icd_loader
            GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-ICD-Loader.git
            GIT_TAG        v2024.10.24
            GIT_SHALLOW    TRUE
        )
        FetchContent_MakeAvailable(opencl_icd_loader)

        if (TARGET OpenCL AND NOT TARGET OpenCL::OpenCL)
            add_library(OpenCL::OpenCL ALIAS OpenCL)
        endif()

        if (TARGET OpenCL::OpenCL)
            set(LLMEDGE_OPENCL_ENABLED ON)
            message(STATUS "Experimental Android OpenCL backend support enabled")
        else()
            message(WARNING "LLMEDGE_ANDROID_OPENCL was requested, but OpenCL::OpenCL was not created by OpenCL-ICD-Loader")
        endif()
    endif()
endif()

function(llmedge_enable_android_opencl target_name ggml_root)
    if (NOT LLMEDGE_OPENCL_ENABLED)
        return()
    endif()

    find_package(Python3 REQUIRED)

    set(_ggml_opencl_dir "${ggml_root}/src/ggml-opencl")
    set(_kernel_dir "${_ggml_opencl_dir}/kernels")
    set(_embed_script "${_kernel_dir}/embed_kernel.py")
    set(_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/${target_name}-ggml-opencl")
    file(MAKE_DIRECTORY "${_generated_dir}")

    file(GLOB _kernel_sources "${_kernel_dir}/*.cl")
    set(_generated_headers)
    foreach(_kernel IN LISTS _kernel_sources)
        get_filename_component(_kernel_name "${_kernel}" NAME_WE)
        set(_generated_header "${_generated_dir}/${_kernel_name}.cl.h")
        add_custom_command(
            OUTPUT "${_generated_header}"
            COMMAND ${Python3_EXECUTABLE} "${_embed_script}" "${_kernel}" "${_generated_header}"
            DEPENDS "${_kernel}" "${_embed_script}"
            COMMENT "Embedding OpenCL kernel ${_kernel_name} for ${target_name}"
            VERBATIM
        )
        list(APPEND _generated_headers "${_generated_header}")
    endforeach()

    target_sources(
        ${target_name}
        PRIVATE
        "${_ggml_opencl_dir}/ggml-opencl.cpp"
        ${_generated_headers}
    )
    target_include_directories(
        ${target_name}
        PRIVATE
        "${ggml_root}/include"
        "${ggml_root}/src"
        "${_generated_dir}"
        "${LLMEDGE_OPENCL_HEADERS_DIR}"
    )
    target_compile_definitions(
        ${target_name}
        PRIVATE
        GGML_USE_OPENCL
        GGML_OPENCL_SOA_Q
        GGML_OPENCL_EMBED_KERNELS
        GGML_OPENCL_USE_ADRENO_KERNELS
        GGML_OPENCL_TARGET_VERSION=300
    )
    target_link_libraries(${target_name} OpenCL::OpenCL)
endfunction()
