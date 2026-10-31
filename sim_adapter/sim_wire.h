#pragma once

#include "cost_profile.h"
#include "event_queue.h"
#include "transport/peer/peer_wire.h"

#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

// Virtual-time PeerWire with ShmWire's structure: each rank has one tx copy
// engine (chunks into the receiver's slots) and one rx copy engine (slots into
// the receive buffer), both FIFO. A finished write is published as an arrival
// by the sender's next progress() pass, and a release reaches the sender
// after a credit latency. The real transport::peer::PeerExchanger runs on top
// unchanged, so chunking, credits, matching and the five completion events in
// the simulator are the real protocol's.
//
// Payload bytes are not moved; buffer pointers are never dereferenced.
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
    // Counts state changes, so a host pass knows whether to loop again.
    virtual void note_activity(int rank) = 0;
    // Whether `rank`'s GPU stream has work running at time `t`.
    virtual bool gpu_busy(int rank, double t) const = 0;
};

// Copy cost model. A self-link chunk costs tx_self / rx_self. A remote chunk
// costs tx_remote / rx_remote (every copy engine of both GPUs busy) or, with
// `contention`, interpolates between the *_alone tables and those by how
// many of the other remote copy engines are busy when it starts (out of the
// three others the A40 profile measures), times the compute factor when its
// rank's GPU is computing then. A chunk queued right behind one of the other
// kind (self <-> remote) on the same engine waits a switch stall first.
// Costs are per rank (the profile's per-GPU tables when it has them).
struct WireCosts {
    Table tx_remote, rx_remote, tx_self, rx_self;  // per chunk, x: bytes
    Table tx_remote_alone, rx_remote_alone;
    double tx_compute_factor = 1.0, rx_compute_factor = 1.0;
    bool contention = false;
    double tx_switch = 0, rx_switch = 0;
    double host_post = 0;      // write_chunk call
    double host_copy_out = 0;  // copy_out call
    double copy_latency = 0;   // idle engine: issue -> start
    double credit_latency = 0;
    double arrival_latency = 0;
    double copy_scale = 1.0;   // multiplies copy durations (sensitivity runs, tests)

    // Costs of `rank`. `contention` is enabled only when the profile has the
    // *_alone tables.
    static WireCosts from_profile(const CostProfile& profile, bool contention, int rank);
};

struct LinkStats {
    std::uint64_t chunks = 0;
    std::uint64_t bytes = 0;
    std::uint32_t max_in_flight = 0;  // posted, not yet released (as the sender sees it)
};

struct EngineStats {
    double busy = 0;  // total copy time
    std::uint64_t copies = 0;
};

class SimFabric {
public:
    // `costs` has one entry per rank.
    SimFabric(EventQueue& queue, HostModel& host, int nranks, int slots_per_link, std::size_t slot_bytes,
              std::vector<WireCosts> costs);
    ~SimFabric();
    SimFabric(const SimFabric&) = delete;
    SimFabric& operator=(const SimFabric&) = delete;

    transport::peer::PeerWire& wire(int rank);
    int nranks() const { return nranks_; }
    const LinkStats& link(int src, int dst) const { return links_[src * nranks_ + dst]; }
    const EngineStats& tx(int rank) const { return tx_stats_[rank]; }
    const EngineStats& rx(int rank) const { return rx_stats_[rank]; }
    // Copy engines of `rank` with remote chunks queued past time `t` (0..2).
    int remote_engines_busy(int rank, double t) const;

private:
    class Wire;
    friend class Wire;

    transport::peer::ChunkHeader& slot(int owner, int src, std::uint64_t seq);

    EventQueue& queue_;
    HostModel& host_;
    int nranks_;
    int slots_;
    std::size_t slot_bytes_;
    std::vector<WireCosts> costs_;
    std::vector<std::unique_ptr<Wire>> wires_;
    // [owner * n + src] ring of slot headers; arrived_ as the receiver sees
    // it, released_[src * n + dst] as the sender sees it.
    std::vector<std::vector<transport::peer::ChunkHeader>> headers_;
    std::vector<std::uint64_t> arrived_, released_;
    std::vector<LinkStats> links_;
    std::vector<EngineStats> tx_stats_, rx_stats_;
    // Per rank: end of the last remote chunk queued on its tx / rx engine.
    std::vector<double> tx_remote_until_, rx_remote_until_;
};

}  // namespace nano_nccl::sim
