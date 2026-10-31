set(nano_nccl_sources
    c_api.cc
    core/buffer.cu
    core/numa.cu
    transport/p2p/p2p_fifo.cu
    transport/p2p/p2p_step_counters.cu
    transport/p2p/p2p_topology.cu
    transport/socket/socket_endpoint.cc
    transport/socket/socket_proxy.cc
    collective/all_reduce/topology.cc
    collective/all_reduce/communicator.cu
    benchmark.cc
)

find_library(NVML_LIB nvidia-ml)
if(NOT NVML_LIB)
    find_file(NVML_LIB libnvidia-ml.so.1 REQUIRED
        PATHS /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}
              /lib/${CMAKE_LIBRARY_ARCHITECTURE})
endif()
find_package(Threads REQUIRED)

function(add_nano_nccl_library target)
    add_library(${target} STATIC ${nano_nccl_sources})
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(${target} PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
    )
    target_link_libraries(${target} PUBLIC ${NUMA_LIB} CUDA::cudart ${NVML_LIB} Threads::Threads)
endfunction()

add_nano_nccl_library(nano_nccl)

# EP arbitrary-peer exchange (B1). Host-only protocol core, separate from the
# Ring path and from the compile-time rank count.
add_library(nano_nccl_peer STATIC
    transport/peer/peer_types.cc
    transport/peer/peer_exchanger.cc
)
set_target_properties(nano_nccl_peer PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(nano_nccl_peer PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_compile_options(nano_nccl_peer PRIVATE -O2 -Wall -Wextra)

if(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
    target_sources(nano_nccl PRIVATE transport/rdma/rdma_proxy_timeline.cc)
    target_compile_definitions(nano_nccl PUBLIC
        NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE=1)
endif()

if(NANO_NCCL_SOCKET_TEST_FAULT_INJECTION)
    add_nano_nccl_library(nano_nccl_test_fault)
    target_compile_definitions(nano_nccl_test_fault PRIVATE
        NANO_NCCL_SOCKET_TEST_FAULT_INJECTION=1
    )
endif()

if(NANO_NCCL_ENABLE_MPI)
    set(nano_nccl_mpi_include_dirs ${MPI_C_INCLUDE_DIRS})
    execute_process(
        COMMAND ${MPI_C_COMPILER} --showme:incdirs
        RESULT_VARIABLE nano_nccl_mpi_showme_status
        OUTPUT_VARIABLE nano_nccl_mpi_showme_include_dirs
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(nano_nccl_mpi_showme_status EQUAL 0 AND
       NOT nano_nccl_mpi_showme_include_dirs STREQUAL "")
        separate_arguments(nano_nccl_mpi_showme_include_dirs NATIVE_COMMAND
                           "${nano_nccl_mpi_showme_include_dirs}")
        list(APPEND nano_nccl_mpi_include_dirs
             ${nano_nccl_mpi_showme_include_dirs})
    endif()
    list(REMOVE_DUPLICATES nano_nccl_mpi_include_dirs)

    set(nano_nccl_mpi_sources
        collective/all_reduce/mpi_communicator.cc
        transport/p2p/mpi_p2p.cc
        transport/shm/mpi_shm.cc
    )
    add_library(nano_nccl_mpi STATIC
        ${nano_nccl_mpi_sources}
    )
    set_target_properties(nano_nccl_mpi PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(nano_nccl_mpi PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${nano_nccl_mpi_include_dirs}
    )
    target_compile_options(nano_nccl_mpi PRIVATE -O3)
    target_compile_definitions(nano_nccl_mpi PUBLIC OMPI_SKIP_MPICXX)
    target_link_libraries(nano_nccl_mpi PUBLIC nano_nccl MPI::MPI_C)

    add_library(nano_nccl_mpi_c SHARED
        collective/all_reduce/mpi_c_api.cc
    )
    target_include_directories(nano_nccl_mpi_c PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    )
    target_compile_options(nano_nccl_mpi_c PRIVATE -O3)
    target_compile_definitions(nano_nccl_mpi_c PUBLIC OMPI_SKIP_MPICXX)
    target_link_libraries(nano_nccl_mpi_c PRIVATE nano_nccl_mpi MPI::MPI_C)

    if(NANO_NCCL_ENABLE_RDMA)
        # target_sources 由各 rdma 源文件任务渐进追加:
        #   Task 4 -> rdma_endpoint.cc, Task 5 -> rdma_qp.cc, Task 6 -> rdma_proxy.cc
        target_sources(nano_nccl PRIVATE transport/rdma/rdma_endpoint.cc)
        target_sources(nano_nccl PRIVATE transport/rdma/rdma_qp.cc)
        target_sources(nano_nccl PRIVATE transport/rdma/rdma_proxy.cc)
        target_sources(nano_nccl PRIVATE transport/rdma/rdma_progress.cc)
        target_sources(nano_nccl PRIVATE transport/rdma/rdma_gdr.cc)
        target_link_libraries(nano_nccl PRIVATE ${IBVERBS_LIBRARY} CUDA::cuda_driver)
        target_compile_definitions(nano_nccl PUBLIC NANO_NCCL_ENABLE_RDMA=1)
        target_link_libraries(nano_nccl_mpi PRIVATE ${IBVERBS_LIBRARY})
    endif()

    if(NANO_NCCL_SOCKET_TEST_FAULT_INJECTION)
        add_library(nano_nccl_mpi_test_fault STATIC ${nano_nccl_mpi_sources})
        target_include_directories(nano_nccl_mpi_test_fault PUBLIC
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/src
            ${nano_nccl_mpi_include_dirs}
        )
        target_compile_options(nano_nccl_mpi_test_fault PRIVATE -O3)
target_compile_definitions(nano_nccl_mpi_test_fault PUBLIC OMPI_SKIP_MPICXX)
target_compile_definitions(nano_nccl_mpi_test_fault PRIVATE
    NANO_NCCL_TRANSPORT_TEST_FAULT_INJECTION=1)
        target_link_libraries(nano_nccl_mpi_test_fault PUBLIC
            nano_nccl_test_fault MPI::MPI_C
        )
    endif()
endif()
