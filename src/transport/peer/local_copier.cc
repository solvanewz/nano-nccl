#include "transport/peer/local_copier.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <new>
#include <random>
#include <stdexcept>

namespace nano_nccl::transport::peer {

namespace {

using Clock = std::chrono::steady_clock;

void* zeroed_pages(std::size_t bytes) {
    constexpr std::size_t kPage = 4096;
    const std::size_t rounded = (std::max<std::size_t>(bytes, 1) + kPage - 1) / kPage * kPage;
    void* ptr = std::aligned_alloc(kPage, rounded);
    if (ptr == nullptr) throw std::bad_alloc();
    std::memset(ptr, 0, rounded);
    return ptr;
}

class HostCopier final : public LocalCopier {
public:
    HostCopier(std::uint64_t seed, int max_delay_us) : rng_(seed), max_delay_us_(std::max(0, max_delay_us)) {}

    std::uint64_t copy(Queue queue, void* dst, const void* src, std::size_t bytes) override {
        if (bytes != 0 && (dst == nullptr || src == nullptr)) {
            throw std::invalid_argument("HostCopier::copy: null buffer");
        }
        auto& q = queues_[queue];
        const auto delay = std::chrono::microseconds(
            max_delay_us_ == 0 ? 0 : static_cast<long>(rng_() % static_cast<std::uint64_t>(max_delay_us_ + 1)));
        const Clock::time_point due = std::max(Clock::now() + delay, q.last_due);
        q.last_due = due;
        q.pending.push_back({dst, src, bytes, due});
        return q.queued++;
    }

    std::uint64_t finished(Queue queue) override {
        auto& q = queues_[queue];
        const Clock::time_point now = Clock::now();
        while (!q.pending.empty() && q.pending.front().due <= now) run_front(q);
        return q.finished;
    }

    void drain() override {
        for (auto& q : queues_) {
            while (!q.pending.empty()) run_front(q);
        }
    }

    void* alloc_host(std::size_t bytes) override { return zeroed_pages(bytes); }
    void free_host(void* ptr) noexcept override { std::free(ptr); }
    void* alloc_local(std::size_t bytes) override { return zeroed_pages(bytes); }
    void free_local(void* ptr) noexcept override { std::free(ptr); }
    const char* name() const override { return "host-memcpy"; }

private:
    struct Pending {
        void* dst;
        const void* src;
        std::size_t bytes;
        Clock::time_point due;
    };
    struct QueueState {
        std::deque<Pending> pending;
        std::uint64_t queued = 0;
        std::uint64_t finished = 0;
        Clock::time_point last_due{};
    };

    static void run_front(QueueState& q) {
        const Pending& p = q.pending.front();
        if (p.bytes != 0) std::memcpy(p.dst, p.src, p.bytes);
        q.pending.pop_front();
        ++q.finished;
    }

    std::mt19937_64 rng_;
    int max_delay_us_;
    std::array<QueueState, 2> queues_;
};

}  // namespace

std::unique_ptr<LocalCopier> make_host_copier(std::uint64_t seed, int max_delay_us) {
    return std::make_unique<HostCopier>(seed, max_delay_us);
}

}  // namespace nano_nccl::transport::peer
