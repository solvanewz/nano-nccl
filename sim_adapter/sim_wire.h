#pragma once

#include "cost_profile.h"
#include "fabric.h"

#include <cstdint>
#include <deque>
#include <functional>
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
// A copy is an amount of work (its time with nothing interfering) that its
// engine runs through at a rate; with in-situ costs the rate drops while the
// same GPU's other engine moves a remote chunk, so a copy's end is known only
// when it finishes and is recomputed whenever that engine starts or ends a
// remote chunk. GPU work on the rank is slowed the same way (HostModel::
// remote_busy_changed).
//
// Payload bytes are not moved; buffer pointers are never dereferenced.
namespace nano_nccl::sim {

// Copy cost model. A self-link chunk costs tx_self / rx_self. A remote chunk
// costs tx_remote / rx_remote (every copy engine of both GPUs busy) or, with
// `contention`, interpolates between the *_alone tables and those by how
// many of the other remote copy engines are busy when it starts (out of the
// three others the A40 profile measures), times the compute factor when its
// rank's GPU is computing then. A chunk queued behind another copy on the same
// engine waits the turnaround for that pair of kinds (self/remote) first.
// With in-situ costs (`insitu`) a remote chunk is insitu(0) us/MiB of work,
// scaled to its size by the all-busy table's shape, run at rate
// insitu(0) / insitu(1) while the same GPU's other engine moves a remote
// chunk and at rate 1 otherwise (on the A40 that engine, not the other GPU,
// is what slows a copy). In-situ costs come from a real run, so they already
// include any slowdown from the GPU computing; compute factors are not
// applied.
// Costs are per rank (the profile's per-GPU tables when it has them).
struct WireCosts {
    Table tx_remote, rx_remote, tx_self, rx_self;  // per chunk, x: bytes
    Table tx_remote_alone, rx_remote_alone;
    Table tx_insitu, rx_insitu;  // x: overlapped fraction (0..1), y: us per MiB
    bool insitu = false;
    double tx_compute_factor = 1.0, rx_compute_factor = 1.0;
    bool contention = false;
    // Queued copy: previous end -> start, [previous is self][this is self].
    // From the in-situ turnaround scalars, else the switch stalls on a kind
    // change and 0 otherwise.
    double tx_turn[2][2] = {{0, 0}, {0, 0}}, rx_turn[2][2] = {{0, 0}, {0, 0}};
    double host_post = 0;      // write_chunk call
    double host_copy_out = 0;  // copy_out call
    double copy_latency = 0;   // idle engine: issue -> start
    double credit_latency = 0;
    double arrival_latency = 0;
    double copy_scale = 1.0;   // multiplies copy durations (sensitivity runs, tests)

    // Costs of `rank`. `contention` is enabled only when the profile has the
    // *_alone tables; `insitu` when it also has the in-situ levels.
    static WireCosts from_profile(const CostProfile& profile, bool contention, int rank);
};

class SimFabric final : public Fabric {
public:
    // `costs` has one entry per rank.
    SimFabric(Scheduler& queue, HostModel& host, int nranks, int slots_per_link, std::size_t slot_bytes,
              std::vector<WireCosts> costs);
    ~SimFabric() override;
    SimFabric(const SimFabric&) = delete;
    SimFabric& operator=(const SimFabric&) = delete;

    transport::peer::PeerWire& wire(int rank) override;
    int nranks() const override { return nranks_; }
    const LinkStats& link(int src, int dst) const override { return links_[src * nranks_ + dst]; }
    const EngineStats& tx(int rank) const override { return tx_stats_[rank]; }
    const EngineStats& rx(int rank) const override { return rx_stats_[rank]; }
    int remote_busy(int rank) const override;

private:
    class Wire;
    friend class Wire;

    struct Job {
        std::size_t bytes;
        bool self;
        bool copies;  // false: header-only chunk, nothing to copy
        double issue;
        std::function<void(double)> done;  // called at the end time
    };
    struct Engine {
        std::deque<Job> queue;
        bool active = false;   // a start is scheduled or a job runs
        bool running = false;  // a job runs
        Job cur{};
        double start = 0, left = 0, last = 0, rate = 1;  // left: work at rate 1, as of `last`
        bool remote = false;   // the running job is a remote copy
        int last_self = -1;    // kind of the last copy (-1: none yet)
        std::uint64_t token = 0;
    };
    Engine& engine(int rank, int dir) { return engines_[static_cast<std::size_t>(rank) * 2 + dir]; }
    const Engine& engine(int rank, int dir) const { return engines_[static_cast<std::size_t>(rank) * 2 + dir]; }
    // dir: 0 tx, 1 rx.
    void enqueue(int rank, int dir, Job job);
    void start_job(int rank, int dir, double t);
    void finish_job(int rank, int dir, double t);
    double work(int rank, int dir, const Job& job, double t) const;
    double rate(int rank, int dir) const;
    void reschedule(int rank, int dir, double t);

    transport::peer::ChunkHeader& slot(int owner, int src, std::uint64_t seq);

    Scheduler& queue_;
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
    std::vector<Engine> engines_;  // [rank * 2 + dir]
};

}  // namespace nano_nccl::sim
