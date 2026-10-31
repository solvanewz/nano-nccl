#include "ep/moe_layer.h"

#include "core/buffer.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nano_nccl::ep {

namespace {

using transport::peer::CompletionEvent;
using transport::peer::ExchangeId;
using Clock = std::chrono::steady_clock;

// Tags carry the step so a mismatch names it.
constexpr std::uint32_t kDispatchTag = 1u << 16;
constexpr std::uint32_t kCombineTag = 2u << 16;

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

bool event_done(cudaEvent_t event) {
    const cudaError_t status = cudaEventQuery(event);
    if (status == cudaSuccess) return true;
    if (status != cudaErrorNotReady) CUDA_CHECK_THROW(status);
    return false;
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
    // Every (source, local expert) pair belongs to exactly one step.
    const auto segments = static_cast<std::size_t>(s.world) * s.experts_per_rank();
    tables_bytes_ = 2 * segments * sizeof(RowSegment) + s.experts_per_rank() * sizeof(ExpertSegment);

    CUDA_CHECK_THROW(cudaSetDevice(device_));
    send_ = device_alloc<bf16>(entries * h);
    recv_ = device_alloc<bf16>(recv_rows * h);
    expert_ = device_alloc<bf16>(recv_rows * h);
    ret_ = device_alloc<bf16>(entries * h);
    positions_ = device_alloc<std::int32_t>(entries);
    counts_dev_ = device_alloc<std::int32_t>(s.num_experts + 1);
    order_dev_ = device_alloc<std::int32_t>(s.num_experts);
    tables_dev_ = device_alloc<std::byte>(tables_bytes_);
    counts_host_ = host_alloc<std::int32_t>(s.num_experts + 1);
    order_host_ = host_alloc<std::int32_t>(s.num_experts);
    send_counts_ = host_alloc<std::uint64_t>(s.num_experts);
    recv_counts_ = host_alloc<std::uint64_t>(segments);
    tables_host_ = host_alloc<std::byte>(tables_bytes_);
    CUDA_CHECK_THROW(cudaEventCreateWithFlags(&stream_done_, cudaEventDisableTiming));
}

MoeLayer::~MoeLayer() {
    cudaSetDevice(device_);
    for (void* p : {static_cast<void*>(send_), static_cast<void*>(recv_), static_cast<void*>(expert_),
                    static_cast<void*>(ret_), static_cast<void*>(positions_), static_cast<void*>(counts_dev_),
                    static_cast<void*>(order_dev_), tables_dev_}) {
        cudaFree(p);
    }
    for (void* p : {static_cast<void*>(counts_host_), static_cast<void*>(order_host_),
                    static_cast<void*>(send_counts_), static_cast<void*>(recv_counts_), tables_host_}) {
        cudaFreeHost(p);
    }
    cudaEventDestroy(stream_done_);
    for (cudaEvent_t e : step_done_) cudaEventDestroy(e);
}

void MoeLayer::wait(ExchangeId id, CompletionEvent event) {
    while (!ex_.query_completion(id, event)) ex_.progress();
}

void MoeLayer::sync(cudaStream_t stream) {
    CUDA_CHECK_THROW(cudaEventRecord(stream_done_, stream));
    while (!event_done(stream_done_)) ex_.progress();
}

// All steps' segment tables in one upload: [unpack...][repack...][experts...].
void MoeLayer::upload_step_tables(cudaStream_t stream) {
    const int n = static_cast<int>(plan_.steps.size());
    unpack_dev_.assign(n, nullptr);
    repack_dev_.assign(n, nullptr);
    experts_dev_.assign(n, nullptr);
    std::size_t unpack_total = 0;
    for (const GroupStep& st : plan_.steps) unpack_total += st.unpack.size();
    auto* host_unpack = static_cast<RowSegment*>(tables_host_);
    auto* host_repack = host_unpack + unpack_total;
    auto* host_experts = reinterpret_cast<ExpertSegment*>(host_repack + unpack_total);
    const auto* dev_unpack = static_cast<const RowSegment*>(tables_dev_);
    const auto* dev_repack = dev_unpack + unpack_total;
    const auto* dev_experts = reinterpret_cast<const ExpertSegment*>(dev_repack + unpack_total);
    std::size_t rows = 0, experts = 0;
    for (int i = 0; i < n; ++i) {
        const GroupStep& st = plan_.steps[i];
        std::copy(st.unpack.begin(), st.unpack.end(), host_unpack + rows);
        std::copy(st.repack.begin(), st.repack.end(), host_repack + rows);
        std::copy(st.experts.begin(), st.experts.end(), host_experts + experts);
        unpack_dev_[i] = dev_unpack + rows;
        repack_dev_[i] = dev_repack + rows;
        experts_dev_[i] = dev_experts + experts;
        rows += st.unpack.size();
        experts += st.experts.size();
    }
    const std::size_t bytes = 2 * unpack_total * sizeof(RowSegment) + experts * sizeof(ExpertSegment);
    if (bytes > tables_bytes_) throw std::logic_error("MoeLayer: step tables exceed their buffer");
    CUDA_CHECK_THROW(cudaMemcpyAsync(tables_dev_, tables_host_, bytes, cudaMemcpyHostToDevice, stream));
}

// Unpack -> experts -> repack for one step, on `stream`.
void MoeLayer::launch_step(int i, cudaStream_t stream) {
    const GroupStep& st = plan_.steps[i];
    const EpShape& s = config_.shape;
    const int segments = static_cast<int>(st.unpack.size());
    launch_permute_rows(recv_, expert_, unpack_dev_[i], segments, s.hidden, stream);
    if (config_.mode == ExpertMode::kFfn) {
        ffn_->forward(expert_, st.experts, stream);
    } else {
        launch_probe_experts(expert_, experts_dev_[i], static_cast<int>(st.experts.size()),
                             s.first_local_expert(), s.hidden, stream);
    }
    launch_permute_rows(expert_, recv_, repack_dev_[i], segments, s.hidden, stream);
}

ExchangeId MoeLayer::issue_dispatch(std::uint64_t epoch, int i) {
    return ex_.exchangev_async(epoch, kDispatchTag | static_cast<std::uint32_t>(i), plan_.steps[i].dispatch,
                               reinterpret_cast<const std::byte*>(send_), reinterpret_cast<std::byte*>(recv_));
}

ExchangeId MoeLayer::issue_combine(std::uint64_t epoch, int i) {
    return ex_.exchangev_async(epoch, kCombineTag | static_cast<std::uint32_t>(i), plan_.steps[i].combine,
                               reinterpret_cast<const std::byte*>(recv_), reinterpret_cast<std::byte*>(ret_));
}

ForwardTimes MoeLayer::forward(std::uint64_t epoch, const bf16* x, int tokens, const std::int32_t* topk_ids,
                               const float* topk_weights, bf16* out, const planner::GroupPlan& groups,
                               const PipelineOptions& options, cudaStream_t stream) {
    const EpShape& s = config_.shape;
    const int E = s.num_experts;
    const int W = s.world;
    const int epr = s.experts_per_rank();
    if (tokens < 0 || tokens > config_.max_tokens) {
        throw std::invalid_argument("MoeLayer::forward: " + std::to_string(tokens) + " tokens exceed max_tokens " +
                                    std::to_string(config_.max_tokens));
    }
    if (options.lookahead < 0) throw std::invalid_argument("MoeLayer::forward: negative lookahead");
    const std::vector<std::int32_t> order = send_expert_order(s, groups);  // validates the plan
    const int entries = tokens * s.top_k;
    ForwardTimes times;
    CUDA_CHECK_THROW(cudaSetDevice(device_));
    const auto t0 = Clock::now();
    auto now = [t0] { return micros(t0, Clock::now()); };

    // Route and pack: send rows ordered by step, then expert.
    std::copy(order.begin(), order.end(), order_host_);
    CUDA_CHECK_THROW(cudaMemcpyAsync(order_dev_, order_host_, E * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                     stream));
    launch_count_experts(topk_ids, entries, E, counts_dev_, stream);
    launch_place_entries(topk_ids, entries, E, order_dev_, counts_dev_, positions_, stream);
    launch_pack_rows(x, s.hidden, positions_, entries, s.top_k, send_, stream);
    CUDA_CHECK_THROW(cudaMemcpyAsync(counts_host_, counts_dev_, (E + 1) * sizeof(std::int32_t),
                                     cudaMemcpyDeviceToHost, stream));
    sync(stream);
    if (counts_host_[E] != 0) {
        throw std::invalid_argument("MoeLayer::forward: " + std::to_string(counts_host_[E]) +
                                    " expert ids out of range");
    }
    times.route = now();

    // Counts: peer d receives N(rank, e) for its experts e.
    for (int e = 0; e < E; ++e) send_counts_[e] = static_cast<std::uint64_t>(counts_host_[e]);
    const ExchangeId counts_id = transport::peer::exchange_counts_async(ex_, epoch, send_counts_, recv_counts_,
                                                                        static_cast<std::size_t>(epr));
    wait(counts_id, CompletionEvent::kConsumerReadable);
    times.counts = now() - times.route;

    std::vector<std::int64_t> send_counts(counts_host_, counts_host_ + E);
    std::vector<std::int64_t> recv_counts(recv_counts_, recv_counts_ + W * epr);
    plan_ = make_dispatch_plan(s, groups, send_counts, recv_counts);
    if (plan_.recv_rows > config_.max_recv_rows) {
        throw std::length_error("MoeLayer::forward: " + std::to_string(plan_.recv_rows) +
                                " received rows exceed max_recv_rows " + std::to_string(config_.max_recv_rows));
    }
    const int G = static_cast<int>(plan_.steps.size());
    while (static_cast<int>(step_done_.size()) < G) {
        cudaEvent_t e = nullptr;
        CUDA_CHECK_THROW(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        step_done_.push_back(e);
    }
    upload_step_tables(stream);
    times.steps.resize(G);
    for (int i = 0; i < G; ++i) {
        times.steps[i].group = plan_.steps[i].group;
        times.steps[i].expert_rows = plan_.steps[i].expert_rows;
    }
    std::vector<ExchangeId> dispatch_id(G, 0), combine_id(G, 0);
    const double pipeline_start = now();

    if (!options.overlap) {
        for (int i = 0; i < G; ++i) {
            StepTimes& t = times.steps[i];
            t.dispatch_issued = now();
            dispatch_id[i] = issue_dispatch(epoch, i);
            wait(dispatch_id[i], CompletionEvent::kConsumerReadable);
            t.dispatch_readable = t.expert_launched = now();
            launch_step(i, stream);
            sync(stream);
            t.expert_done = t.combine_issued = now();
            combine_id[i] = issue_combine(epoch, i);
            wait(combine_id[i], CompletionEvent::kConsumerReadable);
            t.combine_readable = now();
        }
    } else {
        // Fixed issue order, identical on every rank: D0 .. D(L), then
        // C0, D(L+1), C1, D(L+2), ...
        std::vector<std::pair<bool, int>> issue;  // (is_combine, step)
        const int lead = std::min(options.lookahead, G - 1);
        for (int i = 0; i <= lead; ++i) issue.push_back({false, i});
        for (int i = 0; i < G; ++i) {
            issue.push_back({true, i});
            if (i + lead + 1 < G) issue.push_back({false, i + lead + 1});
        }
        std::size_t next_issue = 0;
        int next_launch = 0, next_done = 0, readable = 0;
        std::vector<char> combine_seen(G, 0);
        while (readable < G) {
            ex_.progress();
            while (next_issue < issue.size()) {
                const auto [is_combine, i] = issue[next_issue];
                if (is_combine) {
                    if (i >= next_done) break;  // its experts have not finished
                    times.steps[i].combine_issued = now();
                    combine_id[i] = issue_combine(epoch, i);
                } else {
                    times.steps[i].dispatch_issued = now();
                    dispatch_id[i] = issue_dispatch(epoch, i);
                }
                ++next_issue;
            }
            // Experts run in step order on one stream; enqueue the next step
            // as soon as its data is readable.
            if (next_launch < G && dispatch_id[next_launch] != 0 &&
                ex_.query_completion(dispatch_id[next_launch], CompletionEvent::kConsumerReadable)) {
                StepTimes& t = times.steps[next_launch];
                t.dispatch_readable = t.expert_launched = now();
                launch_step(next_launch, stream);
                CUDA_CHECK_THROW(cudaEventRecord(step_done_[next_launch], stream));
                ++next_launch;
            }
            while (next_done < next_launch && event_done(step_done_[next_done])) {
                times.steps[next_done].expert_done = now();
                ++next_done;
            }
            for (int i = 0; i < G; ++i) {
                if (combine_id[i] != 0 && !combine_seen[i] &&
                    ex_.query_completion(combine_id[i], CompletionEvent::kConsumerReadable)) {
                    combine_seen[i] = 1;
                    times.steps[i].combine_readable = now();
                    ++readable;
                }
            }
        }
    }
    times.pipeline = now() - pipeline_start;

    const double reduce_start = now();
    launch_combine_reduce(ret_, positions_, topk_weights, tokens, s.top_k, s.hidden, out, stream);
    sync(stream);
    times.reduce = now() - reduce_start;

    std::vector<ExchangeId> all{counts_id};
    all.insert(all.end(), dispatch_id.begin(), dispatch_id.end());
    all.insert(all.end(), combine_id.begin(), combine_id.end());
    for (ExchangeId id : all) {
        wait(id, CompletionEvent::kSendBufferReusable);
        wait(id, CompletionEvent::kReceiveSlotsReleased);
        ex_.retire(id);
    }
    times.total = now();
    return times;
}

}  // namespace nano_nccl::ep
