#pragma once

#include "transport/peer/peer_wire.h"
#include "transport/peer/rdma_wire.h"
#include "transport/peer/shm_wire.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <mpi.h>

namespace nano_nccl::transport::peer {

struct HybridWireConfig {
    int slots_per_link = 4;
    std::size_t slot_bytes = std::size_t{1} << 20;
    int device = 0;  // this rank's CUDA device
    // Links between ranks of one node:
    //   kNvlink  ShmWire with payload slots in the receiver's GPU memory
    //            (CUDA IPC); every pair of distinct GPUs on the node must be
    //            NVLink-connected (NVML P2P status), otherwise construction
    //            fails before any IPC handle is opened;
    //   kHost    ShmWire through shared host memory.
    enum class Local { kNvlink, kHost };
    Local local = Local::kNvlink;
    // Links between nodes: RdmaWire, GPUDirect (device slots registered via
    // DMA-BUF or nvidia-peermem) or host-staged.
    bool gpu_direct = true;
    int debug_copy_delay_us = 0;
};

// PeerWire over two wires: ranks of the same node through a ShmWire on the
// node's communicator (NVLink or host memory), every other rank through an
// RdmaWire on the whole communicator. Each link uses exactly one of them for
// its lifetime; the self link is the ShmWire's. The RdmaWire still connects
// a QP to every peer, so the nodes' local links stay unused there.
//
// Nodes are MPI_COMM_TYPE_SHARED groups. Tests may override the grouping with
// NANO_NCCL_EP_TEST_NODE_OF_RANK="n0,n1,..." (node id per global rank), so
// one host can stand in for several nodes.
//
// Collective, like both parts. Before destroying it: finish() on every rank,
// then a barrier.
class HybridWire final : public PeerWire {
public:
    HybridWire(MPI_Comm comm, const HybridWireConfig& config);
    ~HybridWire() override;
    HybridWire(const HybridWire&) = delete;
    HybridWire& operator=(const HybridWire&) = delete;

    int rank() const override { return rank_; }
    int nranks() const override { return nranks_; }
    int slots_per_link() const override { return slots_; }
    std::size_t slot_bytes() const override { return slot_bytes_; }

    std::uint64_t released(int dst) override;
    OpId write_chunk(int dst, std::uint64_t seq, const ChunkHeader& header, const std::byte* src) override;
    std::uint64_t arrived(int src) override;
    const ChunkHeader& slot_header(int src, std::uint64_t seq) override;
    OpId copy_out(int src, std::uint64_t seq, std::byte* dst, std::size_t bytes) override;
    void release(int src, std::uint64_t upto) override;
    bool done(OpId op) override;
    void progress() override;

    // Completes every network write (RdmaWire::drain) and every local copy,
    // and unmaps the peers' device slots.
    void finish();

    // Ranks per node when every node holds the same number of consecutive
    // ranks (the cost model's layout), else 0.
    int ranks_per_node() const;
    bool is_local(int peer) const { return node_of_[static_cast<std::size_t>(peer)] == node_of_[static_cast<std::size_t>(rank_)]; }
    // The backend of each kind of link actually in use, e.g.
    // "local=nvlink(2 ranks) remote=rdma/gpudirect(dmabuf) mlx5_0".
    const std::string& description() const { return description_; }

private:
    int local_index(int peer) const { return local_rank_of_[static_cast<std::size_t>(peer)]; }

    int rank_ = 0;
    int nranks_ = 0;
    int slots_ = 0;
    std::size_t slot_bytes_ = 0;
    std::vector<int> node_of_;        // [global rank]
    std::vector<int> local_rank_of_;  // [global rank] rank in its node's communicator
    MPI_Comm node_comm_ = MPI_COMM_NULL;
    std::unique_ptr<ShmWire> local_;
    std::unique_ptr<RdmaWire> remote_;  // null on a single node
    std::string description_;
};

}  // namespace nano_nccl::transport::peer
