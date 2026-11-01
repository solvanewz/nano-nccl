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

# SimAI bridge (sim_adapter/simai): NetFabric sends remote chunks through
# SimAI's AstraNetworkAPI. SimAI is third-party code that is never vendored
# here; point NANO_NCCL_SIMAI_DIR at a SimAI checkout to build the bridge and
# its CPU test (only AstraNetworkAPI.hh is needed) and, when SimAI's ns-3
# backend has been built there (scripts/build.sh -c ns3), nano_ep_ns3.
set(NANO_NCCL_SIMAI_DIR "" CACHE PATH "SimAI checkout for the AstraNetworkAPI / ns-3 bridge (empty: off)")
if(NANO_NCCL_SIMAI_DIR)
    set(_astra "${NANO_NCCL_SIMAI_DIR}/astra-sim-alibabacloud")
    if(NOT EXISTS "${_astra}/astra-sim/system/AstraNetworkAPI.hh")
        message(FATAL_ERROR "NANO_NCCL_SIMAI_DIR: no astra-sim-alibabacloud/astra-sim/system/AstraNetworkAPI.hh")
    endif()
    add_library(nano_nccl_simai STATIC simai/net_fabric.cc)
    target_include_directories(nano_nccl_simai PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/simai)
    target_include_directories(nano_nccl_simai SYSTEM PUBLIC "${_astra}")
    target_compile_options(nano_nccl_simai PRIVATE -O2 -Wall -Wextra)
    target_link_libraries(nano_nccl_simai PUBLIC nano_nccl_sim)

    # ns-3 front end, built against SimAI's ns-3 build with its flags.
    set(_ns3 "${_astra}/extern/network_backend/ns3-interface/simulation")
    file(GLOB _ns3_libs "${_ns3}/build/lib/libns3*.so")
    if(_ns3_libs)
        add_executable(nano_ep_ns3 simai/ns3/nano_ep_ns3.cc)
        target_include_directories(nano_ep_ns3 SYSTEM PRIVATE "${_ns3}/build/include"
            "${_astra}/astra-sim/network_frontend/ns3" /usr/include/libxml2)
        target_compile_definitions(nano_ep_ns3 PRIVATE NS3_MTP NS3_ASSERT_ENABLE NS3_LOG_ENABLE
            NS3_BUILD_PROFILE_DEBUG HAVE_LIBXML2 __LINUX__)
        target_compile_options(nano_ep_ns3 PRIVATE -O2 -include cstdint -Wno-deprecated-declarations)
        # The ns-3 libraries depend on each other without saying so;
        # --as-needed (Ubuntu's default) would drop some of them.
        target_link_libraries(nano_ep_ns3 PRIVATE nano_nccl_simai -Wl,--no-as-needed ${_ns3_libs} -Wl,--as-needed)
        set_target_properties(nano_ep_ns3 PROPERTIES BUILD_RPATH "${_ns3}/build/lib")
    else()
        message(STATUS "SimAI ns-3 backend not built under ${_ns3}; skipping nano_ep_ns3")
    endif()
endif()
