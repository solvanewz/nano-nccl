#pragma once

#include "ep/ep_kernels.h"
#include "ep/ep_plan.h"
#include "ep/expert_ffn.h"
#include "exec/exec_program.h"
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

// Shared with the simulator (exec/exec_program.h).
using PipelineOptions = exec::PipelineOptions;

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

// Host-observed times of one LayerProgram operation, in microseconds since
// forward() started; -1 when not reached.
struct OpTimes {
    double issue_begin = -1;
    double issue_end = -1;
    // When the condition later ops wait for was first observed: kDone for
    // host/stream ops, kConsumerReadable for exchanges.
    double complete = -1;
};

struct ForwardTimes {
    double route = 0;     // count [+ place + pack (kAhead)] kernels, counts copied to the host
    double counts = 0;    // exchange_counts until readable
    double plan = 0;      // plan op (host): [scheduler +] row plan + table upload issue
    double pack = 0;      // kFromCounts: pack op issue .. done; 0 for kAhead
    double pipeline = 0;  // first dispatch issued .. last combine readable
    double reduce = 0;    // weighted top-k sum
    double total = 0;     // including waiting for every exchange to finish
    std::vector<StepTimes> steps;  // in execution order
    std::vector<OpTimes> ops;      // per LayerProgram op
};

// One MoE layer over an arbitrary-peer exchanger: Counts/Offsets -> Pack ->
// per step (Dispatch -> Expert FFN -> Combine) -> Reduce.
//
// forward() builds the invocation's exec::LayerProgram and executes it: the
// host drives the exchanger continuously and issues each operation when
// exec::ProgramCursor says its dependencies hold, so with overlap step i's
// experts start as soon as its dispatch is readable while later dispatches
// and earlier combines run on the copy engines. The simulator executes the
// same program with the same rule. The GroupPlan is fixed for the call: given
// up front (PlanTiming::kAhead) or computed by the scheduler from this call's
// global counts at the plan op, before any row is packed (kFromCounts).
//
// Collective: every rank calls forward() with the same epoch, PlanSource and
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
                         const float* topk_weights, bf16* out, const exec::PlanSource& source,
                         const PipelineOptions& options, cudaStream_t stream);
    // PlanSource::ahead(groups).
    ForwardTimes forward(std::uint64_t epoch, const bf16* x, int tokens, const std::int32_t* topk_ids,
                         const float* topk_weights, bf16* out, const planner::GroupPlan& groups,
                         const PipelineOptions& options, cudaStream_t stream);

    const DispatchPlan& last_plan() const { return plan_.rows; }
    const exec::EpPlan& last_ep_plan() const { return plan_; }
    const exec::LayerProgram& last_program() const { return program_; }

private:
    void upload_step_tables(cudaStream_t stream);
    void launch_step(int step, cudaStream_t stream);

    transport::peer::PeerExchanger& ex_;
    MoeLayerConfig config_;
    ExpertFfn* ffn_;
    int device_;
    exec::EpPlan plan_;
    exec::LayerProgram program_;

    bf16* send_ = nullptr;
    bf16* recv_ = nullptr;
    bf16* expert_ = nullptr;
    bf16* ret_ = nullptr;
    std::int32_t* positions_ = nullptr;
    std::int32_t* counts_dev_ = nullptr;
    std::int32_t* order_dev_ = nullptr;
    void* tables_dev_ = nullptr;  // per-step segments
    // Pinned host memory. send/recv counts hold W * E values (kFromCounts:
    // the full row to and from every peer; kAhead uses W * epr of them).
    std::int32_t* counts_host_ = nullptr;
    std::int32_t* order_host_ = nullptr;
    std::uint64_t* send_counts_ = nullptr;
    std::uint64_t* recv_counts_ = nullptr;
    void* tables_host_ = nullptr;
    std::size_t tables_bytes_ = 0;
    // Device addresses of each step's tables inside tables_dev_.
    std::vector<const RowSegment*> unpack_dev_, repack_dev_;
    std::vector<const ExpertSegment*> experts_dev_;
    std::vector<cudaEvent_t> op_done_;  // per program op; used by stream ops
};

}  // namespace nano_nccl::ep
