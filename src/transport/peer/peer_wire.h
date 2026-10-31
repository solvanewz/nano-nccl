#pragma once

#include "transport/peer/peer_types.h"

#include <cstddef>
#include <cstdint>

namespace nano_nccl::transport::peer {

// One rank's endpoint of the medium that carries chunks between ranks.
//
// Each link src -> dst has `slots_per_link()` receive slots on dst. Chunk
// number `seq` of a link (monotonic across exchanges) occupies slot
// `seq % slots_per_link()`. Two counters, in the spirit of the Simple FIFO
// head/tail, coordinate a link:
//   arrived(src)  on dst: chunks [0, arrived) have landed, in order;
//   released(dst) on src: dst has freed the slots of chunks [0, released).
// A sender may write chunk `seq` only while seq - released < slots.
//
// Implementations: the CPU loopback fabric (tests) and ShmWire (host-staged
// POSIX shared memory between processes on one host).
class PeerWire {
public:
    using OpId = std::uint64_t;
    // Never returned by write_chunk/copy_out; callers use it for ops that need
    // no wire work (zero-byte chunks).
    static constexpr OpId kNoOp = 0;

    virtual ~PeerWire() = default;

    virtual int rank() const = 0;
    virtual int nranks() const = 0;
    virtual int slots_per_link() const = 0;
    virtual std::size_t slot_bytes() const = 0;

    // Sender side of link rank() -> dst.
    virtual std::uint64_t released(int dst) = 0;
    // Writes header and `header.bytes` payload bytes from `src` (null when the
    // chunk is empty) into slot seq % slots of link rank() -> dst. The op is
    // done once `src` is no longer read; arrival is reported separately.
    virtual OpId write_chunk(int dst, std::uint64_t seq, const ChunkHeader& header,
                             const std::byte* src) = 0;

    // Receiver side of link src -> rank().
    virtual std::uint64_t arrived(int src) = 0;
    // Valid for an arrived, unreleased chunk.
    virtual const ChunkHeader& slot_header(int src, std::uint64_t seq) = 0;
    // Copies the chunk payload into `dst`. The op is done once `dst` holds the
    // data and is visible to its consumer.
    virtual OpId copy_out(int src, std::uint64_t seq, std::byte* dst, std::size_t bytes) = 0;
    // Frees the slots of chunks [0, upto) of link src -> rank(). Monotonic.
    virtual void release(int src, std::uint64_t upto) = 0;

    virtual bool done(OpId op) = 0;

    // Non-blocking. Lets a wire with asynchronous work of its own (for
    // example publishing arrivals once a DMA finishes) advance it.
    // PeerExchanger::progress() calls it first.
    virtual void progress() {}
};

}  // namespace nano_nccl::transport::peer
