#include "transport/peer/local_copier.h"

#include "core/buffer.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <deque>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <unistd.h>

namespace nano_nccl::transport::peer {

namespace {

void CUDART_CB sleep_on_stream(void* micros) { usleep(static_cast<useconds_t>(*static_cast<int*>(micros))); }

class CudaCopier final : public LocalCopier {
public:
    CudaCopier(int device, int debug_copy_delay_us) : device_(device), delay_us_(debug_copy_delay_us) {
        CUDA_CHECK_THROW(cudaSetDevice(device_));
        for (auto& q : queues_) CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&q.stream, cudaStreamNonBlocking));
    }

    ~CudaCopier() override {
        cudaSetDevice(device_);
        for (auto& q : queues_) {
            if (q.stream == nullptr) continue;
            cudaStreamSynchronize(q.stream);
            for (auto& [ticket, event] : q.events) cudaEventDestroy(event);
            cudaStreamDestroy(q.stream);
        }
        for (cudaEvent_t event : free_events_) cudaEventDestroy(event);
    }

    std::uint64_t copy(Queue queue, void* dst, const void* src, std::size_t bytes) override {
        auto& q = queues_[queue];
        CUDA_CHECK_THROW(cudaSetDevice(device_));
        if (delay_us_ > 0) CUDA_CHECK_THROW(cudaLaunchHostFunc(q.stream, sleep_on_stream, &delay_us_));
        if (bytes != 0) CUDA_CHECK_THROW(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, q.stream));
        cudaEvent_t event = take_event();
        const cudaError_t status = cudaEventRecord(event, q.stream);
        if (status != cudaSuccess) {
            free_events_.push_back(event);
            CUDA_CHECK_THROW(status);
        }
        q.events.emplace_back(q.queued, event);
        return q.queued++;
    }

    std::uint64_t finished(Queue queue) override {
        auto& q = queues_[queue];
        while (!q.events.empty()) {
            const cudaError_t status = cudaEventQuery(q.events.front().second);
            if (status == cudaErrorNotReady) break;
            CUDA_CHECK_THROW(status);
            free_events_.push_back(q.events.front().second);
            q.events.pop_front();
            ++q.finished;
        }
        return q.finished;
    }

    void drain() override {
        CUDA_CHECK_THROW(cudaSetDevice(device_));
        for (auto& q : queues_) CUDA_CHECK_THROW(cudaStreamSynchronize(q.stream));
        finished(kTx);
        finished(kRx);
    }

    void* alloc_host(std::size_t bytes) override {
        CUDA_CHECK_THROW(cudaSetDevice(device_));
        void* ptr = nullptr;
        CUDA_CHECK_THROW(cudaHostAlloc(&ptr, bytes, cudaHostAllocPortable));
        std::memset(ptr, 0, bytes);
        return ptr;
    }
    void free_host(void* ptr) noexcept override {
        if (ptr != nullptr && cudaFreeHost(ptr) != cudaSuccess) std::fprintf(stderr, "CudaCopier: cudaFreeHost failed\n");
    }
    void* alloc_local(std::size_t bytes) override {
        CUDA_CHECK_THROW(cudaSetDevice(device_));
        void* ptr = nullptr;
        CUDA_CHECK_THROW(cudaMalloc(&ptr, bytes));
        CUDA_CHECK_THROW(cudaMemset(ptr, 0, bytes));
        CUDA_CHECK_THROW(cudaDeviceSynchronize());
        return ptr;
    }
    void free_local(void* ptr) noexcept override {
        if (ptr != nullptr && cudaFree(ptr) != cudaSuccess) std::fprintf(stderr, "CudaCopier: cudaFree failed\n");
    }
    const char* name() const override { return "cuda"; }

private:
    struct QueueState {
        cudaStream_t stream = nullptr;
        std::deque<std::pair<std::uint64_t, cudaEvent_t>> events;
        std::uint64_t queued = 0;
        std::uint64_t finished = 0;
    };

    cudaEvent_t take_event() {
        if (!free_events_.empty()) {
            cudaEvent_t event = free_events_.back();
            free_events_.pop_back();
            return event;
        }
        cudaEvent_t event = nullptr;
        CUDA_CHECK_THROW(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        return event;
    }

    int device_;
    int delay_us_;
    std::array<QueueState, 2> queues_;
    std::vector<cudaEvent_t> free_events_;
};

}  // namespace

std::unique_ptr<LocalCopier> make_cuda_copier(int device, int debug_copy_delay_us) {
    return std::make_unique<CudaCopier>(device, debug_copy_delay_us);
}

}  // namespace nano_nccl::transport::peer
