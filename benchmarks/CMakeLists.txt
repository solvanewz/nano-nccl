# A40 cost profile for the EP simulator; one process, no MPI, no peer access.
add_executable(nano_nccl_ep_profile ep_profile.cu)
target_link_libraries(nano_nccl_ep_profile PRIVATE nano_nccl_ep nano_nccl_sim nano_nccl)
target_compile_options(nano_nccl_ep_profile PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# Scheduler run times per world for the simulator (CPU only).
add_executable(nano_nccl_schedule_bench schedule_bench.cc)
target_link_libraries(nano_nccl_schedule_bench PRIVATE nano_nccl_sim nano_nccl_ep_plan)
target_compile_options(nano_nccl_schedule_bench PRIVATE -O2 -Wall -Wextra)

if(NANO_NCCL_ENABLE_MPI)
    add_executable(nano_nccl_collective_bench collective_bench.cu)
    target_link_libraries(nano_nccl_collective_bench PRIVATE
        nano_nccl_mpi MPI::MPI_C)
    target_compile_definitions(nano_nccl_collective_bench PRIVATE NANO_NCCL_ENABLE_MPI=1)
    target_include_directories(nano_nccl_collective_bench PRIVATE ${MPI_C_INCLUDE_DIRS})
    if(NANO_NCCL_ENABLE_RDMA)
        target_compile_definitions(nano_nccl_collective_bench PRIVATE NANO_NCCL_ENABLE_RDMA=1)
        target_link_libraries(nano_nccl_collective_bench PRIVATE ${IBVERBS_LIBRARY})
    endif()

    add_executable(nano_nccl_all_reduce_bench all_reduce_bench.cu)
    target_link_libraries(nano_nccl_all_reduce_bench PRIVATE
        nano_nccl_mpi MPI::MPI_C)
    target_compile_definitions(nano_nccl_all_reduce_bench PRIVATE NANO_NCCL_ENABLE_MPI=1)
    target_include_directories(nano_nccl_all_reduce_bench PRIVATE ${MPI_C_INCLUDE_DIRS})

    if(NANO_NCCL_ENABLE_BENCH_PROFILING)
        target_compile_definitions(nano_nccl_all_reduce_bench PRIVATE
            NANO_NCCL_ENABLE_BENCH_PROFILING=1)
        target_link_libraries(nano_nccl_all_reduce_bench PRIVATE CUDA::nvToolsExt)
    endif()

    if(NANO_NCCL_ENABLE_RDMA)
        target_compile_definitions(nano_nccl_all_reduce_bench PRIVATE NANO_NCCL_ENABLE_RDMA=1)
        target_link_libraries(nano_nccl_all_reduce_bench PRIVATE ${IBVERBS_LIBRARY})
    endif()
endif()
