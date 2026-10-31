add_executable(nano_nccl_p2p_step_counters p2p_step_counters.cu)
target_link_libraries(nano_nccl_p2p_step_counters PRIVATE nano_nccl)

add_executable(nano_nccl_p2p_topology p2p_topology.cu)
target_link_libraries(nano_nccl_p2p_topology PRIVATE nano_nccl)

add_executable(nano_nccl_simple_protocol simple_protocol.cu)
target_link_libraries(nano_nccl_simple_protocol PRIVATE nano_nccl)

# Exercise vector bodies, scalar tails, alignment, and worker/unroll boundaries.
add_executable(nano_nccl_vector_tail_test vector_tail_test.cu)
target_link_libraries(nano_nccl_vector_tail_test PRIVATE nano_nccl)

add_executable(nano_nccl_c_api_contract c_api.c)
target_compile_features(nano_nccl_c_api_contract PRIVATE c_std_11)
set_target_properties(nano_nccl_c_api_contract PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(nano_nccl_c_api_contract PRIVATE nano_nccl)

if(NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE)
    add_executable(nano_nccl_rdma_proxy_timeline rdma_proxy_timeline_test.cc)
    target_link_libraries(nano_nccl_rdma_proxy_timeline PRIVATE nano_nccl)
    target_compile_definitions(nano_nccl_rdma_proxy_timeline PRIVATE
        NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE=1)
    add_executable(nano_nccl_ring_turnaround ring_turnaround_test.cc)
    target_link_libraries(nano_nccl_ring_turnaround PRIVATE nano_nccl)
    target_compile_definitions(nano_nccl_ring_turnaround PRIVATE
        NANO_NCCL_ENABLE_RDMA_PROXY_TIMELINE=1)
    if(BUILD_TESTING)
        add_test(NAME nano_nccl_rdma_proxy_timeline COMMAND nano_nccl_rdma_proxy_timeline)
        add_test(NAME nano_nccl_ring_turnaround COMMAND nano_nccl_ring_turnaround)
    endif()
endif()

# EP peer exchange protocol on the CPU multi-rank loopback fabric; needs no GPU.
add_executable(nano_nccl_peer_exchange_test peer_exchange_test.cc peer_loopback_fabric.cc)
target_link_libraries(nano_nccl_peer_exchange_test PRIVATE nano_nccl_peer)
target_compile_options(nano_nccl_peer_exchange_test PRIVATE -O2 -Wall -Wextra)

# EP row plans and group plans, simulated on the CPU; needs no GPU.
add_executable(nano_nccl_ep_plan_test ep_plan_test.cc)
target_link_libraries(nano_nccl_ep_plan_test PRIVATE nano_nccl_ep_plan)
target_compile_options(nano_nccl_ep_plan_test PRIVATE -O2 -Wall -Wextra)

if(BUILD_TESTING)
    add_test(NAME nano_nccl_ep_plan COMMAND nano_nccl_ep_plan_test)
    add_test(NAME nano_nccl_peer_exchange COMMAND nano_nccl_peer_exchange_test)
    set_tests_properties(nano_nccl_peer_exchange PROPERTIES TIMEOUT 600)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME nano_nccl_simple_protocol COMMAND nano_nccl_simple_protocol)
    set_tests_properties(nano_nccl_simple_protocol PROPERTIES SKIP_RETURN_CODE 77)
    add_test(NAME nano_nccl_vector_tail_test COMMAND nano_nccl_vector_tail_test)
    set_tests_properties(nano_nccl_vector_tail_test PROPERTIES SKIP_RETURN_CODE 77)
    add_test(NAME nano_nccl_c_api_contract COMMAND nano_nccl_c_api_contract)
    add_test(
        NAME nano_nccl_cuda_arch_contract
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/cuda_arch_contract_test.py
                ${CMAKE_SOURCE_DIR}
    )
    add_test(
        NAME nano_nccl_communicator_bf16_validation_static
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/communicator_bf16_validation_static.py
                ${CMAKE_SOURCE_DIR}/src/collective/all_reduce/communicator.cu
    )
    add_test(
        NAME nano_nccl_collective_scope_static
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/collective_scope_static.py
                ${CMAKE_SOURCE_DIR}/src/collective/all_reduce/communicator.cu
    )
    add_test(
        NAME nano_nccl_avg_scale_fusion_static
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/avg_scale_fusion_static.py
                ${CMAKE_SOURCE_DIR}/src/kernels/ring_simple_kernel.cuh
    )
    add_test(
        NAME nano_nccl_simple_protocol_ownership_static
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/simple_protocol_ownership_static.py
                ${CMAKE_SOURCE_DIR}
    )
    add_test(
        NAME nano_nccl_bench_profiling_static
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/bench_profiling_static.py
                ${CMAKE_SOURCE_DIR}/CMakeLists.txt
                 ${CMAKE_SOURCE_DIR}/src/CMakeLists.txt
                 ${CMAKE_SOURCE_DIR}/benchmarks/CMakeLists.txt
                 ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
                 ${CMAKE_SOURCE_DIR}/src/collective/all_reduce/bench_profiling.h
                 ${CMAKE_SOURCE_DIR}/benchmarks/all_reduce_bench.cu
    )
    add_test(
        NAME nano_nccl_rdma_transport_static
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/rdma_transport_static.py
                ${CMAKE_SOURCE_DIR}/include/nano_nccl/types.h
                ${CMAKE_SOURCE_DIR}/CMakeLists.txt
                ${CMAKE_SOURCE_DIR}/src/CMakeLists.txt
                ${CMAKE_SOURCE_DIR}/benchmarks/CMakeLists.txt
                ${CMAKE_SOURCE_DIR}/src/transport/rdma/rdma_protocol.h
                ${CMAKE_SOURCE_DIR}/src/collective/all_reduce/communicator.cu
    )
    add_test(
        NAME nano_nccl_mpi_c_api_link_static
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/mpi_c_api_link_static.py
                ${CMAKE_SOURCE_DIR}/CMakeLists.txt
                ${CMAKE_SOURCE_DIR}/src/CMakeLists.txt
                ${CMAKE_SOURCE_DIR}/benchmarks/CMakeLists.txt
                ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
    )
endif()

if(NANO_NCCL_ENABLE_MPI)
    add_executable(nano_nccl_mpi_c_api mpi_c_api.cu)
    target_link_libraries(nano_nccl_mpi_c_api PRIVATE nano_nccl_mpi_c)

    add_executable(nano_nccl_mpi_native_api mpi_native_api.cu)
    target_link_libraries(nano_nccl_mpi_native_api PRIVATE nano_nccl_mpi MPI::MPI_C)

    # EP peer exchange over ShmWire; one process per rank, ranks may share a GPU.
    add_executable(nano_nccl_peer_shm_wire_test peer_shm_wire_test.cc)
    target_link_libraries(nano_nccl_peer_shm_wire_test PRIVATE nano_nccl_peer_shm)
    target_compile_options(nano_nccl_peer_shm_wire_test PRIVATE -O2 -Wall -Wextra)
    if(BUILD_TESTING)
        foreach(np IN ITEMS 2 4)
            add_test(
                NAME nano_nccl_peer_shm_wire_np${np}
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                        --bind-to none --mca btl ^openib
                        $<TARGET_FILE:nano_nccl_peer_shm_wire_test>
            )
            set_tests_properties(nano_nccl_peer_shm_wire_np${np} PROPERTIES TIMEOUT 900)
        endforeach()
    endif()

    # EP MoE layer over ShmWire; the probe matrix is bit-exact, --ffn runs
    # real experts (checked by ep_moe_reference.py).
    add_executable(nano_nccl_ep_moe_test ep_moe_test.cu)
    target_link_libraries(nano_nccl_ep_moe_test PRIVATE nano_nccl_ep nano_nccl_peer_shm)
    if(BUILD_TESTING)
        foreach(np IN ITEMS 2 4)
            add_test(
                NAME nano_nccl_ep_moe_probe_np${np}
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                        --bind-to none --mca btl ^openib
                        $<TARGET_FILE:nano_nccl_ep_moe_test> --slots 2 --slot-bytes 65536
            )
            set_tests_properties(nano_nccl_ep_moe_probe_np${np} PROPERTIES TIMEOUT 600)
        endforeach()
    endif()

    add_executable(nano_nccl_mpi_device_selection mpi_device_selection.cc)
    target_link_libraries(nano_nccl_mpi_device_selection PRIVATE
        nano_nccl_mpi MPI::MPI_C)

    if(BUILD_TESTING AND
       (NANO_NCCL_NRANKS EQUAL 2 OR NANO_NCCL_NRANKS EQUAL 4))
        foreach(transport IN ITEMS shm auto p2p)
            add_test(
                NAME nano_nccl_mpi_native_${transport}
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG}
                        ${NANO_NCCL_NRANKS}
                        --bind-to none --mca btl ^openib
                        $<TARGET_FILE:nano_nccl_mpi_native_api> ${transport}
            )
        endforeach()
        set_tests_properties(
            nano_nccl_mpi_native_shm
            nano_nccl_mpi_native_auto
            nano_nccl_mpi_native_p2p
            PROPERTIES TIMEOUT 60
        )
        set_tests_properties(nano_nccl_mpi_native_p2p
                             PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    if(BUILD_TESTING AND NANO_NCCL_NRANKS EQUAL 2)
        add_test(
            NAME nano_nccl_mpi_duplicate_device_rejected
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                    --bind-to none --mca btl ^openib
                    $<TARGET_FILE:nano_nccl_mpi_device_selection>
        )
        add_test(
            NAME nano_nccl_mpi_native_shm_visible_local_rank
            COMMAND ${CMAKE_COMMAND} -E env
                    NANO_NCCL_TEST_VISIBLE_LOCAL_RANK=1
                    ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                    --bind-to none --mca btl ^openib
                    $<TARGET_FILE:nano_nccl_mpi_native_api> shm
        )
        foreach(transport IN ITEMS shm p2p)
            add_test(
                NAME nano_nccl_mpi_c_api_${transport}
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                        --bind-to none --mca btl ^openib
                        $<TARGET_FILE:nano_nccl_mpi_c_api> ${transport}
            )
        endforeach()
        add_test(
            NAME nano_nccl_mpi_c_api_auto
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                    --bind-to none --mca btl ^openib
                    $<TARGET_FILE:nano_nccl_mpi_c_api> auto
        )
        set_tests_properties(
            nano_nccl_mpi_native_shm
            nano_nccl_mpi_native_auto
            nano_nccl_mpi_native_shm_visible_local_rank
            nano_nccl_mpi_duplicate_device_rejected
            nano_nccl_mpi_c_api_shm
            nano_nccl_mpi_c_api_p2p
            nano_nccl_mpi_c_api_auto
            PROPERTIES TIMEOUT 60
        )
        set_tests_properties(nano_nccl_mpi_c_api_p2p
                             PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    add_executable(nano_nccl_socket_protocol socket_protocol_test.cc)
    target_link_libraries(nano_nccl_socket_protocol PRIVATE nano_nccl_mpi)

    if(NANO_NCCL_SOCKET_TEST_FAULT_INJECTION AND
       BUILD_TESTING AND NANO_NCCL_NRANKS EQUAL 2)
        add_executable(nano_nccl_mpi_transport_faults mpi_transport_faults.cc)
        target_link_libraries(nano_nccl_mpi_transport_faults PRIVATE
            nano_nccl_mpi_test_fault MPI::MPI_C)
        add_test(
            NAME nano_nccl_mpi_transport_faults
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                    --bind-to none --mca btl ^openib
                    $<TARGET_FILE:nano_nccl_mpi_transport_faults>
        )
        set_tests_properties(nano_nccl_mpi_transport_faults PROPERTIES
            TIMEOUT 60
            SKIP_RETURN_CODE 77)
    endif()

    if(NANO_NCCL_ENABLE_RDMA)
        add_test(NAME nano_nccl_rdma_gdr_static
                 COMMAND ${Python3_EXECUTABLE}
                         ${CMAKE_CURRENT_SOURCE_DIR}/rdma_gdr_static.py
                         ${CMAKE_SOURCE_DIR}/src/transport/rdma/rdma_gdr.h
                         ${CMAKE_SOURCE_DIR}/src/transport/rdma/rdma_gdr.cc
                         ${CMAKE_SOURCE_DIR}/src/CMakeLists.txt
                         ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
                         ${CMAKE_SOURCE_DIR}/src/collective/all_reduce/communicator.cu
                         ${CMAKE_CURRENT_SOURCE_DIR}/rdma_gdr.cu)

        add_executable(nano_nccl_rdma_gdr rdma_gdr.cu)
        target_link_libraries(nano_nccl_rdma_gdr PRIVATE nano_nccl ${IBVERBS_LIBRARY} CUDA::cuda_driver)
        target_compile_definitions(nano_nccl_rdma_gdr PRIVATE NANO_NCCL_ENABLE_RDMA=1)
        add_test(NAME nano_nccl_rdma_gdr COMMAND nano_nccl_rdma_gdr)
        set_tests_properties(nano_nccl_rdma_gdr PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(nano_nccl_rdma_protocol rdma_protocol.cc)
        target_link_libraries(nano_nccl_rdma_protocol PRIVATE nano_nccl ${IBVERBS_LIBRARY})
        target_compile_definitions(nano_nccl_rdma_protocol PRIVATE NANO_NCCL_ENABLE_RDMA=1)
        add_test(NAME nano_nccl_rdma_protocol COMMAND nano_nccl_rdma_protocol)

        add_executable(nano_nccl_rdma_bootstrap rdma_bootstrap.cu)
        target_link_libraries(nano_nccl_rdma_bootstrap PRIVATE nano_nccl_mpi ${IBVERBS_LIBRARY})
        target_compile_definitions(nano_nccl_rdma_bootstrap PRIVATE NANO_NCCL_ENABLE_RDMA=1)
        add_test(NAME nano_nccl_rdma_bootstrap COMMAND nano_nccl_rdma_bootstrap)
        set_tests_properties(nano_nccl_rdma_bootstrap PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(nano_nccl_rdma_proxy rdma_proxy_test.cc)
        target_link_libraries(nano_nccl_rdma_proxy PRIVATE nano_nccl_mpi ${IBVERBS_LIBRARY})
        target_compile_definitions(nano_nccl_rdma_proxy PRIVATE NANO_NCCL_ENABLE_RDMA=1)
        add_test(NAME nano_nccl_rdma_proxy COMMAND nano_nccl_rdma_proxy)
        set_tests_properties(nano_nccl_rdma_proxy PROPERTIES SKIP_RETURN_CODE 77)

        add_test(NAME nano_nccl_rdma_proxy_multi_flight_static
                 COMMAND ${Python3_EXECUTABLE}
                         ${CMAKE_CURRENT_SOURCE_DIR}/rdma_proxy_multi_flight_static.py
                         ${CMAKE_SOURCE_DIR}/src/transport/rdma/rdma_proxy.cc
                         ${CMAKE_SOURCE_DIR}/src/transport/rdma/rdma_qp.cc)

        add_test(NAME nano_nccl_rdma_proxy_send_visibility_static
                 COMMAND ${Python3_EXECUTABLE}
                         ${CMAKE_CURRENT_SOURCE_DIR}/rdma_proxy_send_visibility_static.py
                         ${CMAKE_SOURCE_DIR}/src/transport/rdma/rdma_proxy.cc)

        add_test(NAME nano_nccl_rdma_proxy_gdr_receive_flush_static
                 COMMAND ${Python3_EXECUTABLE}
                         ${CMAKE_CURRENT_SOURCE_DIR}/rdma_proxy_gdr_receive_flush_static.py
                         ${CMAKE_SOURCE_DIR}/src/transport/rdma/rdma_proxy.h
                         ${CMAKE_SOURCE_DIR}/src/transport/rdma/rdma_proxy.cc
                         ${CMAKE_SOURCE_DIR}/src/collective/all_reduce/communicator.cu
                         ${CMAKE_SOURCE_DIR}/src/transport/rdma/rdma_gdr.cc)

        add_test(NAME nano_nccl_rdma_proxy_zero_byte_static
                 COMMAND ${Python3_EXECUTABLE}
                         ${CMAKE_CURRENT_SOURCE_DIR}/rdma_proxy_zero_byte_static.py
                         ${CMAKE_SOURCE_DIR}/src/transport/rdma/rdma_proxy.cc)

        add_test(NAME nano_nccl_rdma_proxy_write_cts_static
                 COMMAND ${Python3_EXECUTABLE}
                         ${CMAKE_CURRENT_SOURCE_DIR}/rdma_proxy_write_cts_static.py
                         ${CMAKE_SOURCE_DIR}/src/transport/rdma/rdma_proxy.h
                         ${CMAKE_SOURCE_DIR}/src/transport/rdma/rdma_proxy.cc)
    endif()
endif()
