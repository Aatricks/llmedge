# Shared helper for building against copied upstream trees with local patch overlays.

include(CMakeParseArguments)

function(llmedge_prepare_patch_modded_tree)
    set(options)
    set(oneValueArgs NAME SOURCE_ROOT MODS_ROOT USE_MODS_VAR OUT_VAR)
    cmake_parse_arguments(LLM "${options}" "${oneValueArgs}" "" ${ARGN})

    if (NOT LLM_NAME OR NOT LLM_SOURCE_ROOT OR NOT LLM_MODS_ROOT OR NOT LLM_USE_MODS_VAR OR NOT LLM_OUT_VAR)
        message(FATAL_ERROR "llmedge_prepare_patch_modded_tree requires NAME, SOURCE_ROOT, MODS_ROOT, USE_MODS_VAR, and OUT_VAR")
    endif()

    file(GLOB _llmedge_patch_files CONFIGURE_DEPENDS "${LLM_MODS_ROOT}/*.patch")
    list(SORT _llmedge_patch_files)

    set(_llmedge_use_mods_default OFF)
    if (_llmedge_patch_files)
        set(_llmedge_use_mods_default ON)
    endif()

    option(${LLM_USE_MODS_VAR}
            "Build ${LLM_NAME} from a copied tree with patch overlays from ${LLM_MODS_ROOT}"
            ${_llmedge_use_mods_default})

    set(_llmedge_tree_root "${LLM_SOURCE_ROOT}")
    if (${LLM_USE_MODS_VAR} OR (DEFINED ENV{${LLM_USE_MODS_VAR}} AND "$ENV{${LLM_USE_MODS_VAR}}" STREQUAL "1"))
        if (NOT _llmedge_patch_files)
            message(STATUS "${LLM_USE_MODS_VAR} enabled but no patches were found under ${LLM_MODS_ROOT}; using upstream tree")
        else()
            find_program(LLMEDGE_PATCH_EXECUTABLE patch)
            if (NOT LLMEDGE_PATCH_EXECUTABLE)
                message(FATAL_ERROR "${LLM_USE_MODS_VAR} requires the 'patch' executable so llmedge can apply patches from ${LLM_MODS_ROOT}")
            endif()

            file(GLOB_RECURSE _llmedge_upstream_inputs CONFIGURE_DEPENDS "${LLM_SOURCE_ROOT}/*")
            set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
                    ${_llmedge_upstream_inputs}
                    ${_llmedge_patch_files}
            )

            string(REPLACE "/" "-" _llmedge_patch_name "${LLM_NAME}")
            set(_llmedge_tree_root "${CMAKE_CURRENT_BINARY_DIR}/patched-${_llmedge_patch_name}-src")
            message(STATUS "${LLM_USE_MODS_VAR} enabled: copying ${LLM_NAME} sources to ${_llmedge_tree_root} and applying patches from ${LLM_MODS_ROOT}")

            execute_process(COMMAND ${CMAKE_COMMAND} -E rm -rf "${_llmedge_tree_root}")
            execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${_llmedge_tree_root}")
            execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory "${LLM_SOURCE_ROOT}" "${_llmedge_tree_root}")
            if (EXISTS "${_llmedge_tree_root}/.git")
                execute_process(COMMAND ${CMAKE_COMMAND} -E rm -f "${_llmedge_tree_root}/.git")
                execute_process(COMMAND ${CMAKE_COMMAND} -E rm -rf "${_llmedge_tree_root}/.git")
            endif()
            foreach(_llmedge_patch_file IN LISTS _llmedge_patch_files)
                execute_process(
                        COMMAND "${CMAKE_COMMAND}" -E env TMPDIR="${CMAKE_CURRENT_BINARY_DIR}"
                        "${LLMEDGE_PATCH_EXECUTABLE}" --dry-run --batch -p1 -d "${_llmedge_tree_root}" -i "${_llmedge_patch_file}"
                        RESULT_VARIABLE _llmedge_patch_check_result
                        OUTPUT_VARIABLE _llmedge_patch_check_output
                        ERROR_VARIABLE _llmedge_patch_check_error
                )
                if (NOT _llmedge_patch_check_result EQUAL 0)
                    execute_process(
                            COMMAND "${CMAKE_COMMAND}" -E env TMPDIR="${CMAKE_CURRENT_BINARY_DIR}"
                            "${LLMEDGE_PATCH_EXECUTABLE}" --dry-run --batch -R -p1 -d "${_llmedge_tree_root}" -i "${_llmedge_patch_file}"
                            RESULT_VARIABLE _llmedge_patch_reverse_result
                            OUTPUT_VARIABLE _llmedge_patch_reverse_output
                            ERROR_VARIABLE _llmedge_patch_reverse_error
                    )
                    if (_llmedge_patch_reverse_result EQUAL 0)
                        message(STATUS "${_llmedge_patch_file} is already present in ${LLM_NAME}; skipping re-apply")
                        continue()
                    endif()
                endif()

                execute_process(
                        COMMAND "${CMAKE_COMMAND}" -E env TMPDIR="${CMAKE_CURRENT_BINARY_DIR}"
                        "${LLMEDGE_PATCH_EXECUTABLE}" --batch -p1 -d "${_llmedge_tree_root}" -i "${_llmedge_patch_file}"
                        RESULT_VARIABLE _llmedge_patch_result
                        OUTPUT_VARIABLE _llmedge_patch_output
                        ERROR_VARIABLE _llmedge_patch_error
                )
                if (NOT _llmedge_patch_result EQUAL 0)
                    message(FATAL_ERROR
                            "Failed to apply ${_llmedge_patch_file} to ${LLM_NAME}\n"
                            "check stdout:\n${_llmedge_patch_check_output}\n"
                            "check stderr:\n${_llmedge_patch_check_error}\n"
                            "reverse-check stdout:\n${_llmedge_patch_reverse_output}\n"
                            "reverse-check stderr:\n${_llmedge_patch_reverse_error}\n"
                            "apply stdout:\n${_llmedge_patch_output}\n"
                            "apply stderr:\n${_llmedge_patch_error}")
                endif()
            endforeach()
        endif()
    endif()

    set(${LLM_OUT_VAR} "${_llmedge_tree_root}" PARENT_SCOPE)
endfunction()
