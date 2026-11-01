#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace nano_nccl::transport::peer {

// The local copies a network wire needs around its transfers: staging a
// chunk into registered host memory before the NIC sends it (tx queue), and
// copying an arrived chunk out to its consumer (rx queue).
//
// Every copy gets a ticket; the tickets of a queue are 0, 1, 2, ... in the
// order the copies were queued, and they finish in that order. A copy reads
// `src` until it has finished, so the source must stay unchanged until then.
class LocalCopier {
public:
    enum Queue : int { kTx = 0, kRx = 1 };

    virtual ~LocalCopier() = default;

    // Queues a copy and returns its ticket. `bytes` may be 0 (an ordering
    // marker; `dst` and `src` are then ignored).
    virtual std::uint64_t copy(Queue queue, void* dst, const void* src, std::size_t bytes) = 0;
    // Non-blocking: copies [0, finished) of `queue` are done, and their data
    // is visible to the consumer of `dst`.
    virtual std::uint64_t finished(Queue queue) = 0;
    // Blocks until every queued copy has finished.
    virtual void drain() = 0;

    // Zeroed host memory that both this copier and a NIC can access (pinned
    // for CUDA). Freed with free_host.
    virtual void* alloc_host(std::size_t bytes) = 0;
    virtual void free_host(void* ptr) noexcept = 0;
    // Zeroed memory on the consumer's side (device memory for CUDA): the
    // self link's slots, so data a rank sends to itself stays local.
    virtual void* alloc_local(std::size_t bytes) = 0;
    virtual void free_local(void* ptr) noexcept = 0;

    virtual const char* name() const = 0;
};

// cudaMemcpyAsync on two non-blocking streams of `device`; buffers may be
// device or host memory. debug_copy_delay_us > 0 makes every copy first wait
// that long on its stream (tests: a completion reported too early then shows
// up as stale data).
std::unique_ptr<LocalCopier> make_cuda_copier(int device, int debug_copy_delay_us = 0);

// Plain memcpy for CPU-only tests, run late: a copy executes when finished()
// is polled at least a pseudo-random 0..max_delay_us after it was queued (and
// after the copies ahead of it), so a source reused too early or a slot read
// before its copy ran is seen as wrong data.
std::unique_ptr<LocalCopier> make_host_copier(std::uint64_t seed, int max_delay_us);

}  // namespace nano_nccl::transport::peer
