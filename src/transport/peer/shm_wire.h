#pragma once

#include "transport/peer/peer_wire.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

namespace nano_nccl::transport::peer {

struct ShmWireConfig {
    int slots_per_link = 4;
    std::size_t slot_bytes = std::size_t{1} << 20;
    int device = 0;  // CUDA device this rank's copies run on
    // Tests only: every copy first waits this long on its stream, so a
    // completion reported before its copy finished shows up as stale data
    // instead of depending on timing luck.
    int debug_copy_delay_us = 0;
};

// PeerWire between processes on one host, staged through POSIX shared
// memory. Every byte crosses host memory: the sender's GPU copies a chunk
// into the receiver's slot (D2H, or H2H for host buffers) and the receiver's
// GPU copies it out (H2D). No CUDA peer access is used, so it is safe on
// hosts where GPU peer writes are blocked by the IOMMU.
//
// Each rank owns one region holding the receive side of every link
// src -> rank: per link an `arrived` counter (written by src), a `released`
// counter (written by rank), S chunk headers and S payload slots. Senders map
// the receivers' regions. Both mappings are registered with CUDA so the
// copies are asynchronous DMA. The self link rank -> rank keeps its counters
// and headers in the region but its payload slots in this rank's device
// memory, so data a rank sends to itself is copied on the device instead of
// crossing PCIe twice.
//
// Completion: a write op is done when its copy has finished and the chunk's
// arrival is published, which also means the source buffer is no longer read.
// A copy-out op is done when the copy into the destination has finished
// (cudaEventQuery on the receive copy stream), so the data is visible to any
// later work on the device.
//
// Arrivals are published only from progress(): a rank that stops driving
// its exchanger (for example while blocked on a stream) delays its peers by
// that long, so callers poll instead of blocking.
//
// Each receive region is first touched on the NUMA node of its owner's GPU.
//
// Collective: every rank of `comm` constructs it with the same
// slots_per_link and slot_bytes. All ranks must share one host.
class ShmWire final : public PeerWire {
public:
    ShmWire(MPI_Comm comm, const ShmWireConfig& config);
    ~ShmWire() override;
    ShmWire(const ShmWire&) = delete;
    ShmWire& operator=(const ShmWire&) = delete;

    int rank() const override { return rank_; }
    int nranks() const override { return nranks_; }
    int slots_per_link() const override { return slots_; }
    std::size_t slot_bytes() const override { return slot_bytes_; }

    std::uint64_t released(int dst) override;
    OpId write_chunk(int dst, std::uint64_t seq, const ChunkHeader& header,
                     const std::byte* src) override;
    std::uint64_t arrived(int src) override;
    const ChunkHeader& slot_header(int src, std::uint64_t seq) override;
    OpId copy_out(int src, std::uint64_t seq, std::byte* dst, std::size_t bytes) override;
    void release(int src, std::uint64_t upto) override;
    bool done(OpId op) override;
    void progress() override;

    // Bytes of shared memory each rank allocates for its receive region.
    std::size_t region_bytes() const { return region_bytes_; }

private:
    struct Region;
    struct PendingWrite {
        std::uint64_t seq;
        OpId op;
    };
    struct Op {
        cudaEvent_t event = nullptr;
        bool is_write = false;
        bool complete = false;  // write ops: arrival published
    };

    std::byte* link_block(int owner, int src) const;
    std::uint64_t* arrived_word(int owner, int src) const;
    std::uint64_t* released_word(int owner, int src) const;
    ChunkHeader* header_slot(int owner, int src, std::uint64_t seq) const;
    std::byte* payload_slot(int owner, int src, std::uint64_t seq) const;
    cudaEvent_t take_event();
    void recycle_event(cudaEvent_t event);
    OpId record(bool is_write, cudaStream_t stream);
    void check_link(int peer, const char* what) const;
    void debug_delay(cudaStream_t stream);

    int rank_ = 0;
    int nranks_ = 0;
    int slots_ = 0;
    std::size_t slot_bytes_ = 0;
    std::size_t slot_stride_ = 0;
    std::size_t payload_offset_ = 0;
    std::size_t link_bytes_ = 0;
    std::size_t region_bytes_ = 0;
    int device_ = 0;
    int debug_delay_us_ = 0;

    std::vector<std::unique_ptr<Region>> regions_;  // indexed by owning rank
    std::byte* self_payload_ = nullptr;  // device memory, slots of the self link
    cudaStream_t tx_stream_ = nullptr;
    cudaStream_t rx_stream_ = nullptr;

    // Sender side, per destination.
    std::vector<std::uint64_t> tx_next_;
    std::vector<std::deque<PendingWrite>> tx_pending_;
    // Receiver side, per source.
    std::vector<std::uint64_t> rx_arrived_;
    std::vector<std::uint64_t> rx_released_;
    std::vector<std::vector<ChunkHeader>> rx_headers_;  // snapshot per slot

    OpId next_op_ = 1;
    std::unordered_map<OpId, Op> ops_;
    std::vector<cudaEvent_t> free_events_;
};

}  // namespace nano_nccl::transport::peer
