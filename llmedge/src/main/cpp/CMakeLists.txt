cmake_minimum_required(VERSION 3.22.1)
project("smollm")

set(LLMEDGE_CPP_ROOT "${CMAKE_CURRENT_LIST_DIR}")

include("${LLMEDGE_CPP_ROOT}/cmake/llmedge-common.cmake")
include("${LLMEDGE_CPP_ROOT}/cmake/llmedge-targets.cmake")
include("${LLMEDGE_CPP_ROOT}/cmake/llmedge-smollm.cmake")
include("${LLMEDGE_CPP_ROOT}/cmake/llmedge-stable-diffusion.cmake")
include("${LLMEDGE_CPP_ROOT}/cmake/llmedge-whisper.cmake")
include("${LLMEDGE_CPP_ROOT}/cmake/llmedge-bark.cmake")
