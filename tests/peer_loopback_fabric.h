#pragma once

#include "transport/peer/peer_wire.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

namespace nano_nccl::test {

struct LoopbackConfig {
    int nranks = 2;
    int slots = 2;
    std::size_t slot_bytes = 64;
    std::uint64_t seed = 1;
    // Each stage (source read, landing, copy-out, credit return) completes
    // after a random 1..1+max_delay steps, so the five completion events are
    // reached at independent times and in shuffled order across links.
    int max_delay = 4;
};

// CPU multi-rank test backend: N logical ranks in one process sharing one
// deterministic, single-threaded fabric. Payload bytes are really copied.
// The fabric also checks the protocol from the medium's side: writing into a
// slot that was not released, reading a slot that was already reused, and
// releasing chunks that have not arrived all throw std::logic_error.
class LoopbackFabric {
public:
    explicit LoopbackFabric(const LoopbackConfig& config);
    ~LoopbackFabric();
    LoopbackFabric(const LoopbackFabric&) = delete;
    LoopbackFabric& operator=(const LoopbackFabric&) = delete;

    transport::peer::PeerWire& endpoint(int rank);

    // Advances time by one step and runs every event that became due, in a
    // random order.
    void step();
    bool idle() const { return events_.empty(); }
    std::uint64_t now() const { return now_; }
    std::uint64_t chunks_landed() const { return chunks_landed_; }

    // Applies `mutate` to the header of the next chunk landing on src -> dst.
    void corrupt_next_header(int src, int dst,
                             std::function<void(transport::peer::ChunkHeader&)> mutate);

    // Every link drained: all written chunks arrived and were released, and
    // the sender has seen the release.
    bool drained() const;
    std::string describe() const;

private:
    class Endpoint;
    friend class Endpoint;

    struct Link {
        std::vector<std::byte> payload;
        std::vector<transport::peer::ChunkHeader> headers;
        std::vector<std::uint64_t> slot_seq;
        std::set<std::uint64_t> landed;  // landed beyond the in-order prefix
        std::uint64_t written = 0;
        std::uint64_t arrived = 0;
        std::uint64_t released = 0;          // receiver's actual release
        std::uint64_t released_visible = 0;  // what the sender currently sees
        std::function<void(transport::peer::ChunkHeader&)> corrupt;
    };
    struct Event {
        std::uint64_t due;
        std::function<void()> action;
    };

    Link& link(int src, int dst);
    const Link& link(int src, int dst) const;
    void schedule(std::function<void()> action);
    std::uint64_t new_op() { return next_op_++; }

    // PeerWire operations, called through Endpoint.
    std::uint64_t write_chunk(int src, int dst, std::uint64_t seq,
                              const transport::peer::ChunkHeader& header, const std::byte* data);
    std::uint64_t copy_out(int src, int dst, std::uint64_t seq, std::byte* out, std::size_t bytes);
    void release(int src, int dst, std::uint64_t upto);
    const transport::peer::ChunkHeader& slot_header(int src, int dst, std::uint64_t seq) const;

    LoopbackConfig config_;
    std::vector<Link> links_;
    std::vector<std::unique_ptr<Endpoint>> endpoints_;
    std::vector<Event> events_;
    std::unordered_set<std::uint64_t> done_ops_;
    std::mt19937_64 rng_;
    std::uint64_t now_ = 0;
    std::uint64_t next_op_ = 1;
    std::uint64_t chunks_landed_ = 0;
};

}  // namespace nano_nccl::test
