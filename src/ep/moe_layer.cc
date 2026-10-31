#include "ep/moe_layer.h"

#include "core/buffer.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace nano_nccl::ep {

namespace {

using transport::peer::CompletionEvent;
using transport::peer::ExchangeId;
using Clock = std::chrono::steady_clock;

constexpr std::uint32_t kDispatchTag = 1;
constexpr std::uint32_t kCombineTag = 2;

double micros(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

template <typename T>
T* device_alloc(std::size_t count) {
    void* ptr = nullptr;
    CUDA_CHECK_THROW(cudaMalloc(&ptr, std::max<std::size_t>(count, 1) * sizeof(T)));
    return static_cast<T*>(ptr);
}

template <typename T>
T* host_alloc(std::size_t count) {
    void* ptr = nullptr;
    CUDA_CHECK_THROW(cudaMallocHost(&ptr, std::max<std::size_t>(count, 1) * sizeof(T)));
    return static_cast<T*>(ptr);
}

}  // namespace

MoeLayer::MoeLayer(transport::peer::PeerExchanger& exchanger, const MoeLayerConfig& config, ExpertFfn* ffn,
                   int device)
    : ex_(exchanger), config_(config), ffn_(ffn), device_(device) {
    const EpShape& s = config_.shape;
    validate_shape(s);
    if (s.world != ex_.nranks() || s.rank != ex_.rank()) {
        throw std::invalid_argument("MoeLayer: shape world/rank differ from the exchanger's");
    }
    if (config_.max_tokens < 0) throw std::invalid_argument("MoeLayer: negative max_tokens");
    if (config_.mode == ExpertMode::kFfn) {
        if (ffn_ == nullptr) throw std::invalid_argument("MoeLayer: kFfn needs an ExpertFfn");
        if (ffn_->local_experts() != s.experts_per_rank() || ffn_->hidden() != s.hidden) {
            throw std::invalid_argument("MoeLayer: ExpertFfn shape differs from the layer's");
        }
    }
    const auto entries = static_cast<std::size_t>(config_.max_tokens) * s.top_k;
    if (config_.max_recv_rows == 0) {
        config_.max_recv_rows = static_cast<std::int64_t>(entries) * s.world;
    }
    const auto recv_rows = static_cast<std::size_t>(config_.max_recv_rows);
    const auto h = static_cast<std::size_t>(s.hidden);
    const auto segments = static_cast<std::size_t>(s.world) * s.experts_per_rank();
    tables_bytes_ = 2 * segments * sizeof(RowSegment) + (s.experts_per_rank() + 1) * sizeof(std::int64_t);

    CUDA_CHECK_THROW(cudaSetDevice(device_));
    send_ = device_alloc<bf16>(entries * h);
    recv_ = device_alloc<bf16>(recv_rows * h);
    expert_ = device_alloc<bf16>(recv_rows * h);
    ret_ = device_alloc<bf16>(entries * h);
    positions_ = device_alloc<std::int32_t>(entries);
    counts_dev_ = device_alloc<std::int32_t>(s.num_experts + 1);
    tables_dev_ = device_alloc<std::byte>(tables_bytes_);
    counts_host_ = host_alloc<std::int32_t>(s.num_experts + 1);
    send_counts_ = host_alloc<std::uint64_t>(segments);
    recv_counts_ = host_alloc<std::uint64_t>(segments);
    tables_host_ = host_alloc<std::byte>(tables_bytes_);
    CUDA_CHECK_THROW(cudaEventCreateWithFlags(&stream_done_, cudaEventDisableTiming));
}

MoeLayer::~MoeLayer() {
    cudaSetDevice(device_);
    for (void* p : {static_cast<void*>(send_), static_cast<void*>(recv_), static_cast<void*>(expert_),
                    static_cast<void*>(ret_), static_cast<void*>(positions_), static_cast<void*>(counts_dev_),
                    tables_dev_}) {
        cudaFree(p);
    }
    for (void* p : {static_cast<void*>(counts_host_), static_cast<void*>(send_counts_),
                    static_cast<void*>(recv_counts_), tables_host_}) {
        cudaFreeHost(p);
    }
    cudaEventDestroy(stream_done_);
}

void MoeLayer::wait(ExchangeId id, CompletionEvent event) {
    while (!ex_.query_completion(id, event)) ex_.progress();
}

void MoeLayer::sync(cudaStream_t stream) {
    CUDA_CHECK_THROW(cudaEventRecord(stream_done_, stream));
    for (;;) {
        const cudaError_t status = cudaEventQuery(stream_done_);
        if (status == cudaSuccess) return;
        if (status != cudaErrorNotReady) CUDA_CHECK_THROW(status);
        ex_.progress();
    }
}

StageTimes MoeLayer::forward(std::uint64_t epoch, const bf16* x, int tokens, const std::int32_t* topk_ids,
                             const float* topk_weights, bf16* out, cudaStream_t stream) {
    const EpShape& s = config_.shape;
    const int E = s.num_experts;
    const int W = s.world;
    const int epr = s.experts_per_rank();
    const int H = s.hidden;
    if (tokens < 0 || tokens > config_.max_tokens) {
        throw std::invalid_argument("MoeLayer::forward: " + std::to_string(tokens) + " tokens exceed max_tokens " +
                                    std::to_string(config_.max_tokens));
    }
    const int entries = tokens * s.top_k;
    StageTimes times;
    CUDA_CHECK_THROW(cudaSetDevice(device_));
    const auto t0 = Clock::now();

    // Route and pack: send rows ordered by global expert.
    launch_count_experts(topk_ids, entries, E, counts_dev_, stream);
    launch_place_entries(topk_ids, entries, E, counts_dev_, positions_, stream);
    launch_pack_rows(x, H, positions_, entries, s.top_k, send_, stream);
    CUDA_CHECK_THROW(cudaMemcpyAsync(counts_host_, counts_dev_, (E + 1) * sizeof(std::int32_t),
                                     cudaMemcpyDeviceToHost, stream));
    sync(stream);
    if (counts_host_[E] != 0) {
        throw std::invalid_argument("MoeLayer::forward: " + std::to_string(counts_host_[E]) +
                                    " expert ids out of range");
    }
    const auto t1 = Clock::now();

    // Counts: peer d receives N(rank, e) for its experts e.
    for (int e = 0; e < E; ++e) send_counts_[e] = static_cast<std::uint64_t>(counts_host_[e]);
    const ExchangeId counts_id = transport::peer::exchange_counts_async(ex_, epoch, send_counts_, recv_counts_,
                                                                        static_cast<std::size_t>(epr));
    wait(counts_id, CompletionEvent::kConsumerReadable);
    const auto t2 = Clock::now();

    std::vector<std::int64_t> send_counts(counts_host_, counts_host_ + E);
    std::vector<std::int64_t> recv_counts(recv_counts_, recv_counts_ + W * epr);
    plan_ = make_dispatch_plan(s, send_counts, recv_counts);
    if (plan_.recv_rows > config_.max_recv_rows) {
        throw std::length_error("MoeLayer::forward: " + std::to_string(plan_.recv_rows) +
                                " received rows exceed max_recv_rows " + std::to_string(config_.max_recv_rows));
    }
    const ExchangeId dispatch_id = ex_.exchangev_async(epoch, kDispatchTag, plan_.dispatch,
                                                       reinterpret_cast<const std::byte*>(send_),
                                                       reinterpret_cast<std::byte*>(recv_));
    wait(dispatch_id, CompletionEvent::kConsumerReadable);
    const auto t3 = Clock::now();

    // Unpack to expert-major, run the experts in place, repack into
    // receive-buffer order (recv_ is free again after the unpack).
    auto* unpack = static_cast<RowSegment*>(tables_host_);
    auto* repack = unpack + W * epr;
    auto* offsets = reinterpret_cast<std::int64_t*>(repack + W * epr);
    std::memcpy(unpack, plan_.unpack.data(), plan_.unpack.size() * sizeof(RowSegment));
    std::memcpy(repack, plan_.repack.data(), plan_.repack.size() * sizeof(RowSegment));
    std::memcpy(offsets, plan_.expert_offsets.data(), plan_.expert_offsets.size() * sizeof(std::int64_t));
    CUDA_CHECK_THROW(cudaMemcpyAsync(tables_dev_, tables_host_, tables_bytes_, cudaMemcpyHostToDevice, stream));
    auto* unpack_dev = static_cast<const RowSegment*>(tables_dev_);
    auto* repack_dev = unpack_dev + W * epr;
    auto* offsets_dev = reinterpret_cast<const std::int64_t*>(repack_dev + W * epr);
    const int segments = static_cast<int>(plan_.unpack.size());
    launch_permute_rows(recv_, expert_, unpack_dev, segments, H, stream);
    if (config_.mode == ExpertMode::kFfn) {
        ffn_->forward(expert_, plan_.expert_offsets, stream);
    } else {
        launch_probe_experts(expert_, offsets_dev, epr, s.first_local_expert(), H, stream);
    }
    launch_permute_rows(expert_, recv_, repack_dev, segments, H, stream);
    sync(stream);
    const auto t4 = Clock::now();

    const ExchangeId combine_id = ex_.exchangev_async(epoch, kCombineTag, plan_.combine,
                                                      reinterpret_cast<const std::byte*>(recv_),
                                                      reinterpret_cast<std::byte*>(ret_));
    wait(combine_id, CompletionEvent::kConsumerReadable);
    const auto t5 = Clock::now();

    launch_combine_reduce(ret_, positions_, topk_weights, tokens, s.top_k, H, out, stream);
    sync(stream);
    const auto t6 = Clock::now();

    for (ExchangeId id : {counts_id, dispatch_id, combine_id}) {
        wait(id, CompletionEvent::kSendBufferReusable);
        wait(id, CompletionEvent::kReceiveSlotsReleased);
        ex_.retire(id);
    }
    const auto t7 = Clock::now();

    times.route = micros(t0, t1);
    times.counts = micros(t1, t2);
    times.dispatch = micros(t2, t3);
    times.expert = micros(t3, t4);
    times.combine = micros(t4, t5);
    times.reduce = micros(t5, t6);
    times.total = micros(t0, t7);
    return times;
}

}  // namespace nano_nccl::ep
