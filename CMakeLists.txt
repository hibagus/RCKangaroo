cmake_minimum_required(VERSION 3.21)

project(rckangaroo
    VERSION 1.0
    LANGUAGES C CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(RCK_BUILD_CUDA "Build the legacy CUDA reference executable" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerWarnings)
include(Sanitizers)

set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

add_library(rckangaroo_core STATIC
    src/core/ec.cpp
    src/core/runtime_options.cpp
    src/core/utils.cpp
)
add_library(RCKangaroo::core ALIAS rckangaroo_core)

target_include_directories(rckangaroo_core PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)

target_link_libraries(rckangaroo_core PUBLIC Threads::Threads)
if(UNIX AND NOT APPLE)
    target_link_libraries(rckangaroo_core PUBLIC rt)
endif()

rck_set_project_warnings(rckangaroo_core)
rck_enable_sanitizers(rckangaroo_core)

if(RCK_BUILD_CUDA)
    set(CUDAToolkit_ROOT "/usr/local/cuda-12.8" CACHE PATH "CUDA Toolkit root")
    if(NOT CMAKE_CUDA_COMPILER AND EXISTS "${CUDAToolkit_ROOT}/bin/nvcc")
        set(CMAKE_CUDA_COMPILER "${CUDAToolkit_ROOT}/bin/nvcc" CACHE FILEPATH "CUDA compiler")
    endif()

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    add_executable(rckangaroo_cuda
        apps/rckangaroo.cpp
        legacy/cuda/call_cubin.cpp
        legacy/cuda/gpu_kang.cpp
        legacy/cuda/kernels.cu
    )

    target_include_directories(rckangaroo_cuda PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/legacy/cuda"
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_compile_options(rckangaroo_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_89,code=compute_89>
        $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_86,code=compute_86>
        $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_75,code=compute_75>
        $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_61,code=compute_61>
    )

    target_link_libraries(rckangaroo_cuda PRIVATE
        RCKangaroo::core
        CUDA::cudart_static
        CUDA::cuda_driver
        ${CMAKE_DL_LIBS}
    )

    rck_set_project_warnings(rckangaroo_cuda)

    set_target_properties(rckangaroo_cuda PROPERTIES
        CUDA_RUNTIME_LIBRARY Static
        OUTPUT_NAME rckangaroo
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    add_custom_command(TARGET rckangaroo_cuda POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/legacy/cuda/cubin/kernel_sm89.cubin"
            "$<TARGET_FILE_DIR:rckangaroo_cuda>/kernel_sm89.cubin"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/legacy/cuda/cubin/kernel_sm120.cubin"
            "$<TARGET_FILE_DIR:rckangaroo_cuda>/kernel_sm120.cubin"
    )
endif()

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
