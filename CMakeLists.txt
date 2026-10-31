cmake_minimum_required(VERSION 3.18)
project(nano_nccl LANGUAGES C CXX CUDA)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 编译期可配置参数：默认值对应 4 个单 GPU rank / sm_70 最低架构。
# NRANKS/NCHANNELS/BLOCK_THREADS/FIFO_BUFF_BYTES 通过 compile definitions
# 注入代码；CUDA_ARCH 统一设置所有 CUDA target 的编译架构，不作为宏传递。
set(NANO_NCCL_NRANKS 4 CACHE STRING "Global communicator rank count")
set(NANO_NCCL_NCHANNELS 4 CACHE STRING "Number of channels")
set(NANO_NCCL_CUDA_ARCH 70 CACHE STRING "CUDA compute capability (minimum 70)")
set(NANO_NCCL_BLOCK_THREADS 288 CACHE STRING "Threads per block")
set(NANO_NCCL_FLOAT4_UNROLL 8 CACHE STRING "Float4 worker unroll factor (1 or 8)")
set_property(CACHE NANO_NCCL_FLOAT4_UNROLL PROPERTY STRINGS 1 8)
set(NANO_NCCL_FIFO_BUFF_BYTES 33554432 CACHE STRING "FIFO buffer size in bytes")

if(NOT NANO_NCCL_NCHANNELS MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "NANO_NCCL_NCHANNELS must be a positive integer")
endif()
if(NOT NANO_NCCL_FLOAT4_UNROLL MATCHES "^(1|8)$")
    message(FATAL_ERROR "NANO_NCCL_FLOAT4_UNROLL must be 1 or 8")
endif()

if(NOT NANO_NCCL_CUDA_ARCH MATCHES "^[0-9]+$")
    message(FATAL_ERROR "NANO_NCCL_CUDA_ARCH must be a numeric compute capability")
endif()
if(NANO_NCCL_CUDA_ARCH LESS 70)
    message(FATAL_ERROR
        "NANO_NCCL_CUDA_ARCH must be at least 70 for system-scope Simple protocol ordering")
endif()
set(CMAKE_CUDA_ARCHITECTURES ${NANO_NCCL_CUDA_ARCH})

option(NANO_NCCL_ENABLE_MPI "Build the optional MPI communicator bootstrap adapter" OFF)
option(NANO_NCCL_SOCKET_TEST_FAULT_INJECTION
       "Build test-only MPI transport fault-injection libraries" OFF)
option(NANO_NCCL_ENABLE_BENCH_PROFILING "Enable NVTX and CUDA profiler instrumentation in the all-reduce benchmark" OFF)
option(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE
       "Enable host RDMA proxy hot-path stage timeline counters" OFF)
option(NANO_NCCL_ENABLE_RDMA "Build the optional RDMA (RC send/recv) transport" OFF)

if(NANO_NCCL_SOCKET_TEST_FAULT_INJECTION AND NOT NANO_NCCL_ENABLE_MPI)
    message(FATAL_ERROR
        "NANO_NCCL_SOCKET_TEST_FAULT_INJECTION requires NANO_NCCL_ENABLE_MPI=ON")
endif()

if(NANO_NCCL_ENABLE_RDMA AND NOT NANO_NCCL_ENABLE_MPI)
    message(FATAL_ERROR
        "NANO_NCCL_ENABLE_RDMA requires NANO_NCCL_ENABLE_MPI=ON")
endif()

find_package(CUDAToolkit REQUIRED)
find_library(NUMA_LIB numa REQUIRED)
if(NANO_NCCL_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS C)
endif()

if(NANO_NCCL_ENABLE_RDMA)
    find_library(IBVERBS_LIBRARY ibverbs REQUIRED)
endif()

add_subdirectory(src)

target_compile_definitions(nano_nccl PUBLIC
    NANO_NCCL_NRANKS=${NANO_NCCL_NRANKS}
    NANO_NCCL_NCHANNELS=${NANO_NCCL_NCHANNELS}
    NANO_NCCL_BLOCK_THREADS=${NANO_NCCL_BLOCK_THREADS}
    NANO_NCCL_FLOAT4_UNROLL=${NANO_NCCL_FLOAT4_UNROLL}
    NANO_NCCL_FIFO_BUFF_BYTES=${NANO_NCCL_FIFO_BUFF_BYTES}
)

if(NANO_NCCL_SOCKET_TEST_FAULT_INJECTION)
    target_compile_definitions(nano_nccl_test_fault PUBLIC
        NANO_NCCL_NRANKS=${NANO_NCCL_NRANKS}
        NANO_NCCL_NCHANNELS=${NANO_NCCL_NCHANNELS}
        NANO_NCCL_BLOCK_THREADS=${NANO_NCCL_BLOCK_THREADS}
        NANO_NCCL_FLOAT4_UNROLL=${NANO_NCCL_FLOAT4_UNROLL}
        NANO_NCCL_FIFO_BUFF_BYTES=${NANO_NCCL_FIFO_BUFF_BYTES}
    )
endif()

add_subdirectory(sim_adapter)
add_subdirectory(benchmarks)
add_subdirectory(tests)
