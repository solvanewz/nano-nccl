# EP layer simulator (pure C++): runs exec::LayerProgram on every rank in
# virtual time over a simulated PeerWire, with costs from a measured profile.
add_library(nano_nccl_sim STATIC
    cost_profile.cc
    sim_wire.cc
    sim_layer.cc
)
set_target_properties(nano_nccl_sim PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(nano_nccl_sim PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(nano_nccl_sim PRIVATE -O2 -Wall -Wextra)
target_link_libraries(nano_nccl_sim PUBLIC nano_nccl_exec nano_nccl_peer)

add_executable(nano_ep_sim nano_ep_sim.cc)
target_link_libraries(nano_ep_sim PRIVATE nano_nccl_sim)
target_compile_options(nano_ep_sim PRIVATE -O2 -Wall -Wextra)
