#pragma once

#include "ep/ep_kernels.h"
#include "ep/ep_plan.h"
#include "ep/expert_ffn.h"
#include "planner/group_plan.h"
#include "transport/peer/peer_exchanger.h"

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

namespace nano_nccl::ep {

enum class ExpertMode {
    kFfn,    // real BF16 SwiGLU experts (ExpertFfn)
    kProbe,  // each expert scales its rows by probe_scale(): exact data-plane checks
};

struct MoeLayerConfig {
    EpShape shape;
    int max_tokens = 0;
    // Receive-side row capacity; 0 means world * max_tokens * top_k, which
    // fits any routing.
    std::int64_t max_recv_rows = 0;
    ExpertMode mode = ExpertMode::kFfn;
};

// How the steps of a GroupPlan are scheduled. Every rank must use the same
// options for the same forward() call: they fix the exchange issue order.
struct PipelineOptions {
    // false: each step runs dispatch -> experts -> combine to completion before
    // the next starts (no communication/computation overlap).
    bool overlap = true;
    // With overlap: dispatches of steps i+1 .. i+lookahead may be in flight
    // while step i computes. Issue order: D0 .. D(L), then C0, D(L+1), C1, ...
    int lookahead = 1;
};

// Host-observed times of one step, in microseconds since forward() started.
struct StepTimes {
    int group = 0;
    std::int64_t expert_rows = 0;
    double dispatch_issued = 0;
    double dispatch_readable = 0;
    double expert_launched = 0;
    double expert_done = 0;
    double combine_issued = 0;
    double combine_readable = 0;
};

struct ForwardTimes {
    double route = 0;     // count + place + pack kernels, counts copied to the host
    double counts = 0;    // exchange_counts until readable
    double pipeline = 0;  // first dispatch issued .. last combine readable
    double reduce = 0;    // weighted top-k sum
    double total = 0;     // including waiting for every exchange to finish
    std::vector<StepTimes> steps;  // in execution order
};

// One MoE layer over an arbitrary-peer exchanger: Counts/Offsets -> Pack ->
// per step (Dispatch -> Expert FFN -> Combine) -> Reduce.
//
// The GroupPlan passed to forward() is fixed for that call. With overlap the
// host drives the exchanger continuously and launches step i's experts on
// `stream` as soon as its dispatch is readable, so later dispatches and
// earlier combines run on the copy engines while experts run.
//
// Collective: every rank calls forward() with the same epoch, GroupPlan and
// PipelineOptions, in the same sequence.
//
// Buffers (each step uses its own disjoint rows, so steps never share them):
//   send_      packed token rows   pack .. dispatch kSendBufferReusable
//   recv_      dispatch receive    copy-out .. unpack; then repacked outputs
//                                  repack .. combine kSendBufferReusable
//   expert_    expert-major rows   unpack .. repack (experts work in place)
//   ret_       combine receive     copy-out .. reduce (send-buffer order)
//   positions_ entry -> send row   place .. reduce
// forward() returns only after every exchange finished, so the next call may
// reuse all of them.
class MoeLayer {
public:
    // `ffn` is required for ExpertMode::kFfn and ignored otherwise.
    MoeLayer(transport::peer::PeerExchanger& exchanger, const MoeLayerConfig& config, ExpertFfn* ffn,
             int device);
    ~MoeLayer();
    MoeLayer(const MoeLayer&) = delete;
    MoeLayer& operator=(const MoeLayer&) = delete;

    // x: [tokens, hidden] bf16, topk_ids: [tokens, top_k] int32 global expert
    // ids (distinct per token), topk_weights: [tokens, top_k] fp32, out:
    // [tokens, hidden] bf16; all device memory. Blocks until done.
    ForwardTimes forward(std::uint64_t epoch, const bf16* x, int tokens, const std::int32_t* topk_ids,
                         const float* topk_weights, bf16* out, const planner::GroupPlan& groups,
                         const PipelineOptions& options, cudaStream_t stream);

    const DispatchPlan& last_plan() const { return plan_; }

private:
    void wait(transport::peer::ExchangeId id, transport::peer::CompletionEvent event);
    // Waits for `stream` while driving the exchanger: peers depend on this
    // rank's progress (e.g. to see its last chunks arrive).
    void sync(cudaStream_t stream);
    void upload_step_tables(cudaStream_t stream);
    void launch_step(int step, cudaStream_t stream);
    transport::peer::ExchangeId issue_dispatch(std::uint64_t epoch, int step);
    transport::peer::ExchangeId issue_combine(std::uint64_t epoch, int step);

    transport::peer::PeerExchanger& ex_;
    MoeLayerConfig config_;
    ExpertFfn* ffn_;
    int device_;
    DispatchPlan plan_;

    bf16* send_ = nullptr;
    bf16* recv_ = nullptr;
    bf16* expert_ = nullptr;
    bf16* ret_ = nullptr;
    std::int32_t* positions_ = nullptr;
    std::int32_t* counts_dev_ = nullptr;
    std::int32_t* order_dev_ = nullptr;
    void* tables_dev_ = nullptr;  // per-step segments
    // Pinned host memory.
    std::int32_t* counts_host_ = nullptr;
    std::int32_t* order_host_ = nullptr;
    std::uint64_t* send_counts_ = nullptr;
    std::uint64_t* recv_counts_ = nullptr;
    void* tables_host_ = nullptr;
    std::size_t tables_bytes_ = 0;
    // Device addresses of each step's tables inside tables_dev_.
    std::vector<const RowSegment*> unpack_dev_, repack_dev_;
    std::vector<const ExpertSegment*> experts_dev_;
    cudaEvent_t stream_done_ = nullptr;
    std::vector<cudaEvent_t> step_done_;
};

}  // namespace nano_nccl::ep
