#pragma once

#include "transport/peer/local_copier.h"
#include "transport/peer/peer_wire.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <mpi.h>

struct ibv_pd;
struct ibv_mr;

namespace nano_nccl::transport::peer {

// Registers `bytes` of CUDA device memory at `addr` with `pd` (GPUDirect
// RDMA: DMA-BUF or nvidia-peermem) and names the method in `*method`.
// Returns the MR, deregistered when the last reference goes; throws when the
// device memory cannot be registered. Injected so this library needs no CUDA
// driver API.
using DeviceMemoryRegistrar =
    std::function<std::shared_ptr<ibv_mr>(ibv_pd* pd, void* addr, std::size_t bytes, std::string* method)>;

struct RdmaWireConfig {
    int slots_per_link = 4;
    std::size_t slot_bytes = std::size_t{1} << 20;
    // Credit writes in flight per link; releases made meanwhile are folded
    // into the next write.
    int credit_writes_in_flight = 2;
    // GPUDirect RDMA when set: staging and receive slots live in device
    // memory (LocalCopier::alloc_local) registered with this function, so the
    // NIC reads and writes GPU memory and the local copies stay on the
    // device. Chunk headers then go in host memory, written ahead of the
    // payload by a separate (inline) RDMA WRITE on the same QP.
    DeviceMemoryRegistrar register_device;
    // CUDA device of the slots; GPUDirect only (receive write flush).
    int device = 0;
};

// PeerWire over RDMA verbs (RC, one QP per remote peer). Host-pinned by
// default: every byte is staged through registered host memory. With
// RdmaWireConfig::register_device (GPUDirect) the slots are device memory and
// only the 64-byte headers and credit words stay in host memory:
//   write_chunk  header -> host header slot (CPU), payload -> device staging
//                slot (copier, device to device);
//   progress     posts an RDMA WRITE of the header into the receiver's host
//                header slot, then the WRITE_WITH_IMM of the payload; RC
//                places them in order, so the payload's receive completion
//                means the header is in place too. The receiver flushes
//                GPUDirect writes (cudaDeviceFlushGPUDirectRDMAWrites) when
//                the device does not order them natively, before the arrival
//                is visible to copy_out.
// Everything else is as below.
//
// Each slot holds a 64-byte header followed by the payload. Per remote link
// src -> dst:
//   write_chunk  copies header (CPU) and payload (copier tx queue) into the
//                sender's staging slot seq % S. The op is done when that copy
//                finished (the source buffer is no longer read) and the
//                chunk's WRITE has been posted, so a rank that sees its
//                exchange finished and stops polling leaves nothing unsent;
//   progress     posts, in seq order, one RDMA WRITE_WITH_IMM (imm = seq) of
//                header + payload from the staging slot into the receiver's
//                slot seq % S, once its staging copy finished;
//   arrival      is the receiver's completion of that WRITE (data placed),
//                seen in progress(); RC keeps a link's arrivals in order;
//   copy_out     copies the payload out (copier rx queue); done when visible;
//   release      records the count; progress() RDMA-writes the latest one
//                into the sender's credit word for the link (at most
//                credit_writes_in_flight writes in flight per link).
//   released(d)  on the sender is min(credit word, data WRITEs completed):
//                a staging slot is reused only after its WRITE completed.
// The self link copies on the copier's tx queue into local slots
// (alloc_local) and publishes the arrival from progress(), like ShmWire.
//
// Arrivals and credit writes happen in progress() (a release takes effect at
// the next call), so callers keep calling it while exchanges are in flight
// instead of blocking. Once an exchange reports kSendBufferReusable, all of
// its chunks are on the wire: a WRITE in flight completes without polling.
//
// Collective: every rank of `comm` constructs it with the same geometry. The
// HCA is chosen as in the B0 RDMA transport (NANO_NCCL_RDMA_IFNAME,
// NANO_NCCL_RDMA_GID_INDEX); construction fails on every rank if any rank has
// no usable device. Ranks may share one HCA.
class RdmaWire final : public PeerWire {
public:
    RdmaWire(MPI_Comm comm, const RdmaWireConfig& config, std::unique_ptr<LocalCopier> copier);
    ~RdmaWire() override;
    RdmaWire(const RdmaWire&) = delete;
    RdmaWire& operator=(const RdmaWire&) = delete;

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

    // Waits until every queued copy and network write (data and credits,
    // including deferred ones) has completed locally. Call it, then a
    // barrier, before destroying the wires of a job, so no write is still
    // headed for a peer that is tearing down. The destructor itself posts
    // nothing and waits only briefly for writes already posted.
    void drain();

    // The HCA in use and the copier, for reporting the actual backend.
    const std::string& device_name() const { return device_name_; }
    const char* copier_name() const { return copier_->name(); }
    bool gpu_direct() const { return gdr_; }
    // How the device slots were registered ("dmabuf", "device-verbs"), or
    // "host" without GPUDirect.
    const std::string& registration() const { return registration_; }
    std::uint64_t data_writes() const { return data_writes_; }
    // Per-link counters, for error messages and stall reports.
    std::string debug_state() const;
    std::uint64_t credit_writes() const { return credit_writes_; }

private:
    struct Verbs;
    struct Op {
        LocalCopier::Queue queue;
        std::uint64_t ticket;
    };
    struct PendingWrite {
        std::uint64_t seq;
        std::uint64_t ticket;  // staging copy
        std::size_t bytes;     // payload bytes
    };

    // Memory from the copier, freed through it.
    struct Region {
        LocalCopier* copier = nullptr;
        std::byte* ptr = nullptr;
        bool local = false;
        Region() = default;
        Region(const Region&) = delete;
        Region& operator=(const Region&) = delete;
        ~Region();
        void allocate(LocalCopier* owner, std::size_t bytes, bool on_consumer_side);
    };

    std::byte* recv_slot(int src, std::uint64_t seq) const;
    std::byte* staging_slot(int dst, std::uint64_t seq) const;
    std::byte* self_slot(std::uint64_t seq) const;
    std::uint64_t* credit_word(int dst) const;
    ChunkHeader* tx_header(int dst, std::uint64_t seq) const;  // GPUDirect only
    ChunkHeader* rx_header(int src, std::uint64_t seq) const;  // GPUDirect only
    std::uint64_t* credit_source(int src) const;
    void check_link(int peer, const char* what) const;
    OpId record(LocalCopier::Queue queue, std::uint64_t ticket);
    // Posts the WRITE of (or, self link, publishes) every chunk whose staging
    // copy finished; returns that tx-queue snapshot.
    std::uint64_t post_staged();
    void post_data(int dst, const PendingWrite& write);
    void post_credit(int src);
    void poll_completions();

    int rank_ = 0;
    int nranks_ = 0;
    int slots_ = 0;
    std::size_t slot_bytes_ = 0;
    std::size_t slot_stride_ = 0;
    std::size_t block_bytes_ = 0;  // one link's slots
    int credit_cap_ = 0;
    bool gdr_ = false;
    bool gdr_flush_ = false;  // GPUDirect receive writes need an explicit flush
    int device_ = 0;
    std::size_t header_offset_ = 0;  // in control_: GPUDirect header slots
    std::string device_name_;
    std::string registration_ = "host";

    // Destroyed in reverse: the verbs (MRs, QPs) before the memory they
    // cover, the memory before its copier.
    std::unique_ptr<LocalCopier> copier_;
    Region recv_;     // [src] blocks; the NIC writes here
    Region staging_;  // [dst] blocks; the NIC reads here
    Region control_;  // [dst] credit words, [src] credit sources; GPUDirect: then
                      // [dst][slot] tx headers, [src][slot] rx headers
    Region self_;     // the self link's slots (alloc_local)
    std::unique_ptr<Verbs> verbs_;

    // Sender side, per destination.
    std::vector<std::uint64_t> tx_next_;           // chunks written
    std::vector<std::deque<PendingWrite>> tx_pending_;  // staged, WRITE not posted yet
    std::vector<std::uint64_t> tx_posted_;
    std::vector<std::uint64_t> tx_acked_;          // data WRITEs completed
    // Receiver side, per source.
    std::vector<std::uint64_t> rx_arrived_;
    std::vector<std::uint64_t> rx_released_;
    std::vector<std::uint64_t> credit_sent_;       // value of the last credit write posted
    std::vector<int> credit_inflight_;
    std::vector<std::vector<ChunkHeader>> rx_headers_;  // snapshot per slot
    std::vector<ChunkHeader> self_headers_;

    OpId next_op_ = 1;
    std::unordered_map<OpId, Op> ops_;
    std::uint64_t data_writes_ = 0;
    std::uint64_t credit_writes_ = 0;
};

}  // namespace nano_nccl::transport::peer
