#pragma once

#include "event_queue.h"
#include "transport/peer/peer_wire.h"

#include <cstdint>

// What the layer model needs from the simulated transport: one
// transport::peer::PeerWire per rank (the real PeerExchanger runs on top),
// statistics, and how busy each rank's copy engines are. SimFabric
// (sim_wire.h) models ShmWire's host-staged copies; sim_adapter/simai/
// NetFabric sends remote chunks through SimAI's AstraNetworkAPI.
namespace nano_nccl::sim {

// Host side of the simulation, provided by the layer model.
class HostModel {
public:
    virtual ~HostModel() = default;
    // Virtual time of `rank`'s host thread inside its current pass; wire
    // calls charge their host cost to it.
    virtual double& host_clock(int rank) = 0;
    // Something `rank`'s progress loop would notice happened at time `at`.
    virtual void wake(int rank, double at) = 0;
    // The number of `rank`'s copy engines moving a remote chunk changed at
    // time `t` (now; see Fabric::remote_busy), so GPU work running there
    // may change speed.
    virtual void remote_busy_changed(int /*rank*/, double /*t*/) {}
    // Counts state changes, so a host pass knows whether to loop again.
    virtual void note_activity(int rank) = 0;
    // Whether `rank`'s GPU stream has work running at time `t`.
    virtual bool gpu_busy(int rank, double t) const = 0;
};

struct LinkStats {
    std::uint64_t chunks = 0;
    std::uint64_t bytes = 0;
    std::uint32_t max_in_flight = 0;  // posted, not yet released (as the sender sees it)
};

struct EngineStats {
    double busy = 0;  // total copy (or send) time
    std::uint64_t copies = 0;
};

class Fabric {
public:
    virtual ~Fabric() = default;
    virtual transport::peer::PeerWire& wire(int rank) = 0;
    virtual int nranks() const = 0;
    virtual const LinkStats& link(int src, int dst) const = 0;
    virtual const EngineStats& tx(int rank) const = 0;
    virtual const EngineStats& rx(int rank) const = 0;
    // `rank`'s copy engines moving a remote chunk right now (0..2); slows its
    // GPU work (cost::kGpuCopySlowdown).
    virtual int remote_busy(int /*rank*/) const { return 0; }
};

}  // namespace nano_nccl::sim
