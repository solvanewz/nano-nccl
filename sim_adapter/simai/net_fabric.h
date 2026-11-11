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
// With NetConfig::host_staged the remote path matches RdmaWire (host-pinned,
// src/transport/peer/rdma_wire.h) instead of a GPUDirect write:
//
//   write_chunk to a peer  -> a staging copy on the sender's tx engine
//                             (tx_remote cost: device -> pinned host); the
//                             write op is done when it ends, and only then is
//                             the chunk sent;
//   slot reuse             -> released() is min(credits, sends completed in
//                             order), as a staging slot is reused only after
//                             its send completed;
//   copy_out               -> host -> device on the rx engine (rx_remote).
//
// With NetConfig::gdr_staging (HybridWire's RdmaWire with GPUDirect) the
// remote path has the same staged semantics, but both local copies stay in
// device memory (tx_self / rx_self costs).
//
// With NetConfig::ranks_per_node > 1 the peers of a rank's node (contiguous
// blocks of ranks) are HybridWire's local links (ShmWire with device slots
// over NVLink): one direct write to the receiver's device slot, i.e. the
// plain path above, whatever the remote path is; the network backend carries
// them over its intra-node links (SimAI's NVSwitch nodes).
//
// Network transfers themselves do not occupy the PCIe copy engines, so
// remote_busy() is 0: no GPU slowdown from them.
namespace nano_nccl::sim::simai {

struct NetConfig {
    std::size_t header_bytes = sizeof(transport::peer::ChunkHeader);  // sent with every chunk
    std::size_t credit_bytes = 64;
    double arrival_latency = 0;  // receiver sees the arrival this long after delivery (us)
    // Unknown RDMA-side cost, swept as a range (plan rule): every network
    // message (chunk or credit) passes the sending rank's proxy, which hands
    // messages to the NIC one at a time, this many us each (work request
    // build, doorbell). 0: messages go to the network at their issue time.
    double post_overhead = 0;
    bool host_staged = false;    // RdmaWire semantics (see above), else GPUDirect write
    bool gdr_staging = false;    // staged semantics with device-memory copies (see above)
    int ranks_per_node = 0;      // > 1: same-node peers are direct (NVLink) writes
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
    // Network messages rank `rank` sent, and when its proxy handed the last
    // one to the NIC (NetConfig::post_overhead > 0; else 0).
    std::uint64_t messages_sent_by(int rank) const { return sent_by_[rank]; }
    double proxy_busy_until(int rank) const { return proxy_free_[rank]; }

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
    void send_completed(int src, int dst, std::uint64_t seq);

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
    std::vector<std::set<std::uint64_t>> send_done_set_;              // [src * n + dst], not yet in order
    std::vector<std::uint64_t> acked_;                                // [src * n + dst], sends completed in order
    std::vector<double> proxy_free_;                                  // [rank]: proxy busy until (post_overhead)
    std::vector<std::uint64_t> sent_by_;                              // [rank]: network messages sent
    std::vector<LinkStats> links_;
    std::vector<EngineStats> tx_stats_, rx_stats_;
    std::uint64_t sent_ = 0, send_done_ = 0, recv_done_ = 0;
};

}  // namespace nano_nccl::sim::simai
