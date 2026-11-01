#pragma once

#include "astra-sim/system/AstraNetworkAPI.hh"
#include "fabric.h"
#include "sim_wire.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <vector>

// PeerWire per rank whose remote chunks travel through SimAI's network
// interface (AstraSim::AstraNetworkAPI, one per rank, e.g. SimAI's ns-3
// backend), with RDMA-write semantics:
//
//   write_chunk to a peer  -> sim_send(chunk + header bytes) from the sender;
//                             the write op is done (send buffer reusable) when
//                             the send completes (sim_send's handler);
//   arrival                -> the receiver's sim_recv for that chunk fires;
//                             arrivals are published in order per link, after
//                             arrival_latency;
//   copy_out               -> a local copy on the receiver's rx engine (the
//                             slot sits in device memory: rx_self cost);
//   release                -> a credit message (credit_bytes) back to the
//                             sender through the network; its arrival there is
//                             what lets the sender reuse the slots.
//   self link              -> a local copy on the tx engine (tx_self), then
//                             published by the sender's progress() as in
//                             ShmWire.
//
// Every network call is made at the host time of the wire call (through the
// Scheduler), so the network sees the runtime's issue times. The contract
// assumed of the AstraNetworkAPI: sim_send(..., count, ..., dst, tag, ...,
// handler, arg) sends `count` bytes and calls handler(arg) once the send
// completed; sim_recv(..., count, ..., src, tag, ..., handler, arg) calls
// handler(arg) once a message from `src` with `tag` of `count` bytes has
// arrived, whether it arrived before or after the call. Tags are unique per
// (src, dst) pair and direction (data: 2 * seq, credit: 2 * n + 1).
//
// Network copies do not use the PCIe copy engines here, so remote_busy() is
// 0: no GPU slowdown from them.
namespace nano_nccl::sim::simai {

struct NetConfig {
    std::size_t header_bytes = sizeof(transport::peer::ChunkHeader);  // sent with every chunk
    std::size_t credit_bytes = 64;
    double arrival_latency = 0;  // receiver sees the arrival this long after delivery (us)
};

class NetFabric final : public Fabric {
public:
    // `local` gives each rank's local copy costs (tx_self, rx_self,
    // host_post, host_copy_out, copy_latency); `net[r]` is rank r's network
    // endpoint (not owned).
    NetFabric(Scheduler& scheduler, HostModel& host, int nranks, int slots_per_link, std::size_t slot_bytes,
              std::vector<WireCosts> local, std::vector<AstraSim::AstraNetworkAPI*> net, NetConfig config = {});
    ~NetFabric() override;
    NetFabric(const NetFabric&) = delete;
    NetFabric& operator=(const NetFabric&) = delete;

    transport::peer::PeerWire& wire(int rank) override;
    int nranks() const override { return nranks_; }
    const LinkStats& link(int src, int dst) const override { return links_[index(src, dst)]; }
    const EngineStats& tx(int rank) const override { return tx_stats_[rank]; }
    const EngineStats& rx(int rank) const override { return rx_stats_[rank]; }
    // Network messages sent (chunks and credits) and completed sends/receives.
    std::uint64_t messages_sent() const { return sent_; }
    std::uint64_t sends_completed() const { return send_done_; }
    std::uint64_t receives_completed() const { return recv_done_; }

private:
    class Wire;
    friend class Wire;
    struct Callback;
    static void on_callback(void* arg);

    std::size_t index(int src, int dst) const { return static_cast<std::size_t>(src) * nranks_ + dst; }
    transport::peer::ChunkHeader& slot(int owner, int src, std::uint64_t seq);
    // Sends a message src -> dst at host time `at`; posts its receive at dst.
    void send(int src, int dst, int tag, std::uint64_t bytes, double at, std::function<void()> on_sent,
              std::function<void()> on_received);
    void delivered(int src, int dst, std::uint64_t seq);

    Scheduler& sched_;
    HostModel& host_;
    int nranks_;
    int slots_;
    std::size_t slot_bytes_;
    std::vector<WireCosts> local_;
    std::vector<AstraSim::AstraNetworkAPI*> net_;
    NetConfig config_;
    std::vector<std::unique_ptr<Wire>> wires_;
    std::vector<std::vector<transport::peer::ChunkHeader>> headers_;  // [owner * n + src] slot ring
    std::vector<std::uint64_t> arrived_, released_;                   // as the receiver / sender sees them
    std::vector<std::set<std::uint64_t>> delivered_;                  // [src * n + dst], not yet in order
    std::vector<std::uint64_t> in_order_;                             // [src * n + dst], delivered in order
    std::vector<std::uint64_t> credits_sent_;                         // [src * n + dst]: credit messages dst -> src
    std::vector<LinkStats> links_;
    std::vector<EngineStats> tx_stats_, rx_stats_;
    std::uint64_t sent_ = 0, send_done_ = 0, recv_done_ = 0;
};

}  // namespace nano_nccl::sim::simai
