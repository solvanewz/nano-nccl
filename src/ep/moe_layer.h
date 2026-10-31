#pragma once

#include "ep/ep_kernels.h"
#include "ep/ep_plan.h"
#include "ep/expert_ffn.h"
#include "transport/peer/peer_exchanger.h"

#include <cstdint>

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

// Wall-clock microseconds of each stage on this rank.
struct StageTimes {
    double route = 0;     // count + place + pack kernels, counts copied to the host
    double counts = 0;    // exchange_counts until readable
    double dispatch = 0;  // token rows until readable
    double expert = 0;    // unpack + experts + repack
    double combine = 0;   // expert outputs back until readable
    double reduce = 0;    // weighted top-k sum
    double total = 0;     // including waiting for every exchange to finish
};

// One MoE layer over an arbitrary-peer exchanger: Counts/Offsets -> Pack ->
// Dispatch -> Expert FFN -> Combine -> Reduce, run stage after stage. The
// fixed-grouping pipeline builds on the same buffers and kernels.
//
// Collective: every rank of the exchanger calls forward() with the same
// sequence of epochs. Buffers and their lifetimes in one forward():
//   send_     packed token rows     pack .. dispatch kSendBufferReusable
//   recv_     dispatch receive      dispatch copy-out .. unpack; then holds the
//             repacked outputs      repack .. combine kSendBufferReusable
//   expert_   expert-major rows     unpack .. repack (the FFN works in place)
//   ret_      combine receive       combine copy-out .. reduce
//   positions_ entry -> send row    place .. reduce
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
    StageTimes forward(std::uint64_t epoch, const bf16* x, int tokens, const std::int32_t* topk_ids,
                       const float* topk_weights, bf16* out, cudaStream_t stream);

    const DispatchPlan& last_plan() const { return plan_; }

private:
    void wait(transport::peer::ExchangeId id, transport::peer::CompletionEvent event);
    // Waits for `stream` while driving the exchanger: peers depend on this
    // rank's progress (e.g. to see its last chunks arrive).
    void sync(cudaStream_t stream);

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
    void* tables_dev_ = nullptr;   // segments and expert offsets
    // Pinned host memory.
    std::int32_t* counts_host_ = nullptr;
    std::uint64_t* send_counts_ = nullptr;
    std::uint64_t* recv_counts_ = nullptr;
    void* tables_host_ = nullptr;
    std::size_t tables_bytes_ = 0;
    cudaEvent_t stream_done_ = nullptr;
};

}  // namespace nano_nccl::ep
