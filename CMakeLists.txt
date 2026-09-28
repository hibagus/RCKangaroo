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

option(RCK_BUILD_CUDA "Build the legacy CUDA executable" ON)

set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

add_library(rckangaroo_core STATIC
    Ec.cpp
    src/core/runtime_options.cpp
    utils.cpp
)
add_library(RCKangaroo::core ALIAS rckangaroo_core)

target_include_directories(rckangaroo_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(rckangaroo_core PUBLIC
    Threads::Threads
)

if(UNIX AND NOT APPLE)
    target_link_libraries(rckangaroo_core PUBLIC rt)
endif()

if(MSVC)
    target_compile_options(rckangaroo_core PRIVATE /W4)
else()
    target_compile_options(rckangaroo_core PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unknown-pragmas
    )
endif()

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

    add_executable(rckangaroo
        RCKangaroo.cpp
        GpuKang.cpp
        CallCubin.cpp
        RCGpuCore.cu
    )

    target_include_directories(rckangaroo PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_compile_options(rckangaroo PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_89,code=compute_89>
        $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_86,code=compute_86>
        $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_75,code=compute_75>
        $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_61,code=compute_61>
    )

    if(NOT TARGET CUDA::cudart_static)
        message(FATAL_ERROR "CUDA::cudart_static was not found. Check CUDAToolkit_ROOT and CUDA Toolkit installation.")
    endif()

    target_link_libraries(rckangaroo PRIVATE
        RCKangaroo::core
        CUDA::cudart_static
        CUDA::cuda_driver
        ${CMAKE_DL_LIBS}
    )

    if(MSVC)
        target_compile_options(rckangaroo PRIVATE /W4)
    else()
        target_compile_options(rckangaroo PRIVATE
            $<$<COMPILE_LANGUAGE:C,CXX>:-Wall>
            $<$<COMPILE_LANGUAGE:C,CXX>:-Wextra>
            $<$<COMPILE_LANGUAGE:C,CXX>:-Wpedantic>
            $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-unknown-pragmas>
        )
    endif()

    set_target_properties(rckangaroo PROPERTIES
        CUDA_RUNTIME_LIBRARY Static
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
