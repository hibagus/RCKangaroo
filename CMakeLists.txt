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
option(RCK_BUILD_HIP "Build the native HIP executable" OFF)
option(RCK_ENABLE_SLOW_GPU_TESTS "Enable long-running HIP solver acceptance tests" OFF)
set(RCK_HIP_ARCHITECTURES "gfx942;gfx950" CACHE STRING
    "AMD GPU targets for the HIP backend")
set(RCK_BLOCK_SIZE "256" CACHE STRING "HIP kernel workgroup size")
set(RCK_POINT_GROUP_COUNT "24" CACHE STRING "Points processed by each HIP lane")
set(RCK_STEP_COUNT "1000" CACHE STRING "Kangaroo steps processed per solver launch")
set(RCK_BLOCKS_PER_CU "1" CACHE STRING "HIP workgroups launched per compute unit")
set(RCK_GFX942_POINT_GROUP_COUNT "24" CACHE STRING "Default points per lane on gfx942")
set(RCK_GFX950_POINT_GROUP_COUNT "32" CACHE STRING "Default points per lane on gfx950")
set(RCK_GFX942_STEP_COUNT "1000" CACHE STRING "Default solver steps per launch on gfx942")
set(RCK_GFX950_STEP_COUNT "2048" CACHE STRING "Default solver steps per launch on gfx950")
set(RCK_KERNEL_A_TABLE_MODE "split" CACHE STRING
    "KernelA jump-table storage: split, global, or lds64")
option(RCK_USE_ARCH_TABLE_DEFAULTS
    "Use split tables on gfx942 and 64 KiB LDS tables on gfx950" ON)
set(RCK_STATE_LAYOUT "group" CACHE STRING
    "GPU state layout: group or workgroup")
option(RCK_USE_ARCH_STATE_LAYOUT_DEFAULTS
    "Use group-major state on gfx942 and workgroup-major state on gfx950" ON)
set_property(CACHE RCK_BLOCK_SIZE PROPERTY STRINGS 64 128 256)
set_property(CACHE RCK_POINT_GROUP_COUNT PROPERTY STRINGS 8 12 16 24 32)
set_property(CACHE RCK_STEP_COUNT PROPERTY STRINGS 256 512 1000 2048)
set_property(CACHE RCK_KERNEL_A_TABLE_MODE PROPERTY STRINGS split global lds64)
set_property(CACHE RCK_STATE_LAYOUT PROPERTY STRINGS group workgroup)

if(RCK_KERNEL_A_TABLE_MODE STREQUAL "split")
    set(RCK_KERNEL_A_TABLE_MODE_VALUE 0)
elseif(RCK_KERNEL_A_TABLE_MODE STREQUAL "global")
    set(RCK_KERNEL_A_TABLE_MODE_VALUE 1)
elseif(RCK_KERNEL_A_TABLE_MODE STREQUAL "lds64")
    set(RCK_KERNEL_A_TABLE_MODE_VALUE 2)
else()
    message(FATAL_ERROR "RCK_KERNEL_A_TABLE_MODE must be split, global, or lds64")
endif()
if(RCK_USE_ARCH_TABLE_DEFAULTS)
    set(RCK_USE_ARCH_TABLE_DEFAULTS_VALUE 1)
else()
    set(RCK_USE_ARCH_TABLE_DEFAULTS_VALUE 0)
endif()
if(RCK_USE_ARCH_STATE_LAYOUT_DEFAULTS)
    set(RCK_USE_ARCH_STATE_LAYOUT_DEFAULTS_VALUE 1)
else()
    set(RCK_USE_ARCH_STATE_LAYOUT_DEFAULTS_VALUE 0)
endif()
if(RCK_STATE_LAYOUT STREQUAL "group")
    set(RCK_STATE_LAYOUT_VALUE 0)
elseif(RCK_STATE_LAYOUT STREQUAL "workgroup")
    set(RCK_STATE_LAYOUT_VALUE 1)
else()
    message(FATAL_ERROR "RCK_STATE_LAYOUT must be group or workgroup")
endif()

if(NOT RCK_BLOCK_SIZE MATCHES "^(64|128|256)$")
    message(FATAL_ERROR "RCK_BLOCK_SIZE must be 64, 128, or 256")
endif()
if(RCK_POINT_GROUP_COUNT LESS 2 OR RCK_POINT_GROUP_COUNT GREATER 32)
    message(FATAL_ERROR "RCK_POINT_GROUP_COUNT must be between 2 and 32")
endif()
math(EXPR RCK_POINT_GROUP_REMAINDER "${RCK_POINT_GROUP_COUNT} % 2")
if(NOT RCK_POINT_GROUP_REMAINDER EQUAL 0)
    message(FATAL_ERROR "RCK_POINT_GROUP_COUNT must be even")
endif()
if(RCK_STEP_COUNT LESS 1)
    message(FATAL_ERROR "RCK_STEP_COUNT must be positive")
endif()
if(RCK_BLOCKS_PER_CU LESS 1)
    message(FATAL_ERROR "RCK_BLOCKS_PER_CU must be positive")
endif()
foreach(RCK_TARGET_GROUP_COUNT IN ITEMS
        RCK_GFX942_POINT_GROUP_COUNT RCK_GFX950_POINT_GROUP_COUNT)
    if(${RCK_TARGET_GROUP_COUNT} LESS 2 OR ${RCK_TARGET_GROUP_COUNT} GREATER 32)
        message(FATAL_ERROR "${RCK_TARGET_GROUP_COUNT} must be between 2 and 32")
    endif()
    math(EXPR RCK_TARGET_GROUP_REMAINDER "${${RCK_TARGET_GROUP_COUNT}} % 2")
    if(NOT RCK_TARGET_GROUP_REMAINDER EQUAL 0)
        message(FATAL_ERROR "${RCK_TARGET_GROUP_COUNT} must be even")
    endif()
endforeach()
if(RCK_GFX942_STEP_COUNT LESS 1 OR RCK_GFX950_STEP_COUNT LESS 1)
    message(FATAL_ERROR "Architecture step counts must be positive")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerWarnings)
include(Sanitizers)

set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

add_library(rckangaroo_core STATIC
    src/core/ec.cpp
    src/core/host_pipeline.cpp
    src/core/host_topology.cpp
    src/core/runtime_options.cpp
    src/core/utils.cpp
)
add_library(RCKangaroo::core ALIAS rckangaroo_core)

target_include_directories(rckangaroo_core PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)

target_link_libraries(rckangaroo_core PUBLIC Threads::Threads)
target_compile_definitions(rckangaroo_core PUBLIC
    RCK_BLOCK_SIZE=${RCK_BLOCK_SIZE}
    RCK_POINT_GROUP_COUNT=${RCK_POINT_GROUP_COUNT}
    RCK_STEP_COUNT=${RCK_STEP_COUNT}
    RCK_BLOCKS_PER_CU=${RCK_BLOCKS_PER_CU}
    RCK_GFX942_POINT_GROUP_COUNT=${RCK_GFX942_POINT_GROUP_COUNT}
    RCK_GFX950_POINT_GROUP_COUNT=${RCK_GFX950_POINT_GROUP_COUNT}
    RCK_GFX942_STEP_COUNT=${RCK_GFX942_STEP_COUNT}
    RCK_GFX950_STEP_COUNT=${RCK_GFX950_STEP_COUNT}
    RCK_KERNEL_A_TABLE_MODE=${RCK_KERNEL_A_TABLE_MODE_VALUE}
    RCK_USE_ARCH_TABLE_DEFAULTS=${RCK_USE_ARCH_TABLE_DEFAULTS_VALUE}
    RCK_GFX942_KERNEL_A_TABLE_MODE=0
    RCK_GFX950_KERNEL_A_TABLE_MODE=2
    RCK_STATE_LAYOUT=${RCK_STATE_LAYOUT_VALUE}
    RCK_USE_ARCH_STATE_LAYOUT_DEFAULTS=${RCK_USE_ARCH_STATE_LAYOUT_DEFAULTS_VALUE}
    RCK_GFX942_STATE_LAYOUT=0
    RCK_GFX950_STATE_LAYOUT=1
)
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
        legacy/cuda/rckangaroo.cpp
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

if(RCK_BUILD_HIP)
    enable_language(HIP)

    set(CMAKE_HIP_STANDARD 20)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    set(CMAKE_HIP_EXTENSIONS OFF)

    add_library(rckangaroo_hip_backend STATIC
        src/hip/device.cpp
        src/hip/kernels.hip
    )
    add_library(RCKangaroo::hip_backend ALIAS rckangaroo_hip_backend)

    set_source_files_properties(src/hip/device.cpp PROPERTIES LANGUAGE HIP)
    target_include_directories(rckangaroo_hip_backend PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/hip"
    )
    target_link_libraries(rckangaroo_hip_backend PUBLIC RCKangaroo::core)
    set_target_properties(rckangaroo_hip_backend PROPERTIES
        HIP_ARCHITECTURES "${RCK_HIP_ARCHITECTURES}"
        POSITION_INDEPENDENT_CODE ON
    )

    add_executable(rckangaroo_hip apps/rckangaroo.cpp)
    set_source_files_properties(apps/rckangaroo.cpp PROPERTIES LANGUAGE HIP)
    target_link_libraries(rckangaroo_hip PRIVATE RCKangaroo::hip_backend)
    set_target_properties(rckangaroo_hip PROPERTIES
        HIP_ARCHITECTURES "${RCK_HIP_ARCHITECTURES}"
        OUTPUT_NAME rckangaroo
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    add_subdirectory(benchmarks)
endif()

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
