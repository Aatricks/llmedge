# Shared OpenMP runtime for Android targets.
#
# Multiple libraries in one process must not each embed a static libomp:
# libomp aborts when a second copy registers (see #39). Instead, targets that
# need OpenMP link the NDK's shared libomp.so, and one copy is packaged into
# the AAR by copying it next to the built libraries (AGP picks up everything
# in CMAKE_LIBRARY_OUTPUT_DIRECTORY).

if(ANDROID)
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        set(_LLMEDGE_OMP_ARCH "aarch64")
    elseif(ANDROID_ABI STREQUAL "x86_64")
        set(_LLMEDGE_OMP_ARCH "x86_64")
    elseif(ANDROID_ABI STREQUAL "armeabi-v7a")
        set(_LLMEDGE_OMP_ARCH "arm")
    else()
        set(_LLMEDGE_OMP_ARCH "")
    endif()

    set(LLMEDGE_SHARED_OMP_LIB "")
    if(_LLMEDGE_OMP_ARCH)
        get_filename_component(_LLMEDGE_LLVM_BIN "${CMAKE_C_COMPILER}" DIRECTORY)
        file(GLOB _LLMEDGE_OMP_CANDIDATES
            "${_LLMEDGE_LLVM_BIN}/../lib/clang/*/lib/linux/${_LLMEDGE_OMP_ARCH}/libomp.so")
        if(_LLMEDGE_OMP_CANDIDATES)
            list(GET _LLMEDGE_OMP_CANDIDATES 0 LLMEDGE_SHARED_OMP_LIB)
        endif()
    endif()

    if(LLMEDGE_SHARED_OMP_LIB)
        file(COPY "${LLMEDGE_SHARED_OMP_LIB}" DESTINATION "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
        message(STATUS "Shared OpenMP runtime packaged from ${LLMEDGE_SHARED_OMP_LIB}")
    else()
        message(FATAL_ERROR "NDK shared libomp.so not found for ABI ${ANDROID_ABI}")
    endif()
endif()

# Link a target against the shared OpenMP runtime (compile with -fopenmp and
# resolve libomp.so from the packaged copy at runtime).
function(llmedge_link_shared_openmp target_name)
    if(NOT ANDROID)
        return()
    endif()
    target_compile_options(${target_name} PRIVATE -fopenmp)
    # Link the NDK's own copy (DT_NEEDED records just the libomp.so soname);
    # the copy in CMAKE_LIBRARY_OUTPUT_DIRECTORY is what ships in the AAR.
    target_link_libraries(${target_name} "${LLMEDGE_SHARED_OMP_LIB}")
endfunction()
