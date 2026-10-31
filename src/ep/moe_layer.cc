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

using transport::peer::ExchangeId;
using Clock = std::chrono::steady_clock;

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
    for (cudaEvent_t e : op_done_) cudaEventDestroy(e);
}

// All steps' segment tables in one upload: [unpack...][repack...][experts...].
void MoeLayer::upload_step_tables(cudaStream_t stream) {
    const int n = static_cast<int>(plan_.rows.steps.size());
    unpack_dev_.assign(n, nullptr);
    repack_dev_.assign(n, nullptr);
    experts_dev_.assign(n, nullptr);
    std::size_t unpack_total = 0;
    for (const GroupStep& st : plan_.rows.steps) unpack_total += st.unpack.size();
    auto* host_unpack = static_cast<RowSegment*>(tables_host_);
    auto* host_repack = host_unpack + unpack_total;
    auto* host_experts = reinterpret_cast<ExpertSegment*>(host_repack + unpack_total);
    const auto* dev_unpack = static_cast<const RowSegment*>(tables_dev_);
    const auto* dev_repack = dev_unpack + unpack_total;
    const auto* dev_experts = reinterpret_cast<const ExpertSegment*>(dev_repack + unpack_total);
    std::size_t rows = 0, experts = 0;
    for (int i = 0; i < n; ++i) {
        const GroupStep& st = plan_.rows.steps[i];
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
    const GroupStep& st = plan_.rows.steps[i];
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

ForwardTimes MoeLayer::forward(std::uint64_t epoch, const bf16* x, int tokens, const std::int32_t* topk_ids,
                               const float* topk_weights, bf16* out, const planner::GroupPlan& groups,
                               const PipelineOptions& options, cudaStream_t stream) {
    using exec::Condition;
    using exec::OpKind;
    const EpShape& s = config_.shape;
    const int E = s.num_experts;
    const int W = s.world;
    const int epr = s.experts_per_rank();
    if (tokens < 0 || tokens > config_.max_tokens) {
        throw std::invalid_argument("MoeLayer::forward: " + std::to_string(tokens) + " tokens exceed max_tokens " +
                                    std::to_string(config_.max_tokens));
    }
    const std::vector<std::int32_t> order = send_expert_order(s, groups);  // validates the plan
    program_ = exec::build_layer_program(s.rank, groups.order, options);
    const exec::LayerProgram& p = program_;
    const int n = static_cast<int>(p.ops.size());
    while (static_cast<int>(op_done_.size()) < n) {
        cudaEvent_t e = nullptr;
        CUDA_CHECK_THROW(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        op_done_.push_back(e);
    }
    const int entries = tokens * s.top_k;
    CUDA_CHECK_THROW(cudaSetDevice(device_));
    const auto t0 = Clock::now();
    auto now = [t0] { return micros(t0, Clock::now()); };

    ForwardTimes times;
    times.ops.assign(n, OpTimes{});
    std::vector<ExchangeId> ids(n, 0);
    std::vector<char> done(n, 0);       // Condition::kDone of host and stream ops
    std::vector<int> stream_ops;        // issued stream ops, in issue (= completion) order
    std::size_t stream_polled = 0;
    std::vector<int> exchanges_open;    // issued exchanges not yet seen readable
    exec::ProgramCursor cursor(p);
    auto satisfied = [&](const exec::Completion& c) {
        if (c.condition == Condition::kDone) return done[c.op] != 0;
        return ids[c.op] != 0 && ex_.query_completion(ids[c.op], exec::to_completion_event(c.condition));
    };
    auto record = [&](int op) {
        CUDA_CHECK_THROW(cudaEventRecord(op_done_[op], stream));
        stream_ops.push_back(op);
    };

    auto issue = [&](int op) {
        const exec::Operation& o = p.ops[op];
        switch (o.kind) {
            case OpKind::kRoute:
                // Send rows ordered by step, then expert.
                std::copy(order.begin(), order.end(), order_host_);
                CUDA_CHECK_THROW(cudaMemcpyAsync(order_dev_, order_host_, E * sizeof(std::int32_t),
                                                 cudaMemcpyHostToDevice, stream));
                launch_count_experts(topk_ids, entries, E, counts_dev_, stream);
                launch_place_entries(topk_ids, entries, E, order_dev_, counts_dev_, positions_, stream);
                launch_pack_rows(x, s.hidden, positions_, entries, s.top_k, send_, stream);
                CUDA_CHECK_THROW(cudaMemcpyAsync(counts_host_, counts_dev_, (E + 1) * sizeof(std::int32_t),
                                                 cudaMemcpyDeviceToHost, stream));
                record(op);
                break;
            case OpKind::kCounts:
                // Peer d receives N(rank, e) for its experts e.
                if (counts_host_[E] != 0) {
                    throw std::invalid_argument("MoeLayer::forward: " + std::to_string(counts_host_[E]) +
                                                " expert ids out of range");
                }
                for (int e = 0; e < E; ++e) send_counts_[e] = static_cast<std::uint64_t>(counts_host_[e]);
                ids[op] = transport::peer::exchange_counts_async(ex_, epoch, send_counts_, recv_counts_,
                                                                 static_cast<std::size_t>(epr));
                break;
            case OpKind::kPlan: {
                const std::vector<std::int64_t> send_counts(counts_host_, counts_host_ + E);
                const std::vector<std::int64_t> recv_counts(recv_counts_, recv_counts_ + W * epr);
                plan_ = exec::make_ep_plan(s, groups, options, send_counts, recv_counts, ex_.slot_bytes());
                if (plan_.buffers.recv_rows > config_.max_recv_rows) {
                    throw std::length_error("MoeLayer::forward: " + std::to_string(plan_.buffers.recv_rows) +
                                            " received rows exceed max_recv_rows " +
                                            std::to_string(config_.max_recv_rows));
                }
                upload_step_tables(stream);  // ahead of every expert op on the stream
                done[op] = 1;
                break;
            }
            case OpKind::kDispatch:
                ids[op] = ex_.exchangev_async(epoch, exec::exchange_tag(p, op), exec::exchange_layout(p, op, plan_),
                                              reinterpret_cast<const std::byte*>(send_),
                                              reinterpret_cast<std::byte*>(recv_));
                break;
            case OpKind::kCombine:
                ids[op] = ex_.exchangev_async(epoch, exec::exchange_tag(p, op), exec::exchange_layout(p, op, plan_),
                                              reinterpret_cast<const std::byte*>(recv_),
                                              reinterpret_cast<std::byte*>(ret_));
                break;
            case OpKind::kExpert:
                launch_step(o.step, stream);
                record(op);
                break;
            case OpKind::kReduce:
                launch_combine_reduce(ret_, positions_, topk_weights, tokens, s.top_k, s.hidden, out, stream);
                record(op);
                break;
            case OpKind::kRetire:
                for (int i = 0; i < n; ++i) {
                    if (ids[i] != 0) ex_.retire(ids[i]);
                }
                done[op] = 1;
                break;
        }
        if (o.queue == exec::Queue::kExchange) exchanges_open.push_back(op);
    };

    // Never block on the stream: peers depend on this rank's progress() to
    // see their chunks arrive and their slots return.
    while (!done[p.retire]) {
        ex_.progress();
        while (stream_polled < stream_ops.size() && event_done(op_done_[stream_ops[stream_polled]])) {
            const int op = stream_ops[stream_polled++];
            done[op] = 1;
            times.ops[op].complete = now();
        }
        for (std::size_t i = 0; i < exchanges_open.size();) {
            const int op = exchanges_open[i];
            if (ex_.query_completion(ids[op], transport::peer::CompletionEvent::kConsumerReadable)) {
                times.ops[op].complete = now();
                exchanges_open[i] = exchanges_open.back();
                exchanges_open.pop_back();
            } else {
                ++i;
            }
        }
        for (int op; (op = cursor.next_ready(satisfied)) != -1;) {
            OpTimes& t = times.ops[op];
            t.issue_begin = now();
            issue(op);
            cursor.mark_issued(op);
            t.issue_end = now();
            if (p.ops[op].queue == exec::Queue::kHost) t.complete = t.issue_end;
        }
    }

    const auto& ot = times.ops;
    times.route = ot[p.route].complete;
    times.counts = ot[p.counts].complete - times.route;
    double last_combine = 0;
    for (int op : p.combine) last_combine = std::max(last_combine, ot[op].complete);
    times.pipeline = last_combine - ot[p.dispatch[0]].issue_begin;
    times.reduce = ot[p.reduce].complete - ot[p.reduce].issue_begin;
    times.total = ot[p.retire].issue_end;
    times.steps.resize(p.steps);
    for (int i = 0; i < p.steps; ++i) {
        StepTimes& st = times.steps[i];
        st.group = plan_.rows.steps[i].group;
        st.expert_rows = plan_.rows.steps[i].expert_rows;
        st.dispatch_issued = ot[p.dispatch[i]].issue_begin;
        st.dispatch_readable = ot[p.dispatch[i]].complete;
        st.expert_launched = ot[p.expert[i]].issue_begin;
        st.expert_done = ot[p.expert[i]].complete;
        st.combine_issued = ot[p.combine[i]].issue_begin;
        st.combine_readable = ot[p.combine[i]].complete;
    }
    return times;
}

}  // namespace nano_nccl::ep
