#pragma once

#include "ep/ep_plan.h"

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// CUDA kernels of the EP data plane. All launchers are asynchronous on
// `stream` and accept zero-sized work.
namespace nano_nccl::ep {

using bf16 = __nv_bfloat16;

// counts[e] = number of entries routed to expert e, for e < E; counts[E]
// counts expert ids outside [0, E). `counts` has E + 1 slots.
void launch_count_experts(const std::int32_t* expert_ids, int entries, int num_experts,
                          std::int32_t* counts, cudaStream_t stream);

// positions[i] = send-buffer row of entry i: experts follow `expert_order`
// (DispatchPlan::expert_order, E entries), and entries of one expert keep
// their index order (deterministic). Needs the counts above.
void launch_place_entries(const std::int32_t* expert_ids, int entries, int num_experts,
                          const std::int32_t* expert_order, const std::int32_t* counts,
                          std::int32_t* positions, cudaStream_t stream);

// send[positions[i]] = x[i / top_k].
void launch_pack_rows(const bf16* x, int hidden, const std::int32_t* positions, int entries,
                      int top_k, bf16* send, cudaStream_t stream);

// Row copies described by device-resident segments.
void launch_permute_rows(const bf16* src, bf16* dst, const RowSegment* segments, int num_segments,
                         int hidden, cudaStream_t stream);

// a[m, f] = silu(h[m, f]) * h[m, ffn + f], with h = [gate | up] of width 2 * ffn.
void launch_swiglu(const bf16* h, std::int64_t rows, int ffn, bf16* a, cudaStream_t stream);

// Probe experts for exact data-plane checks: the rows of each segment are
// multiplied by probe_scale(first_expert + segment.expert), a power of two.
__host__ __device__ inline float probe_scale(int expert) {
    const int m = expert % 3;
    return m == 0 ? 0.5f : (m == 1 ? 1.0f : 2.0f);
}
void launch_probe_experts(bf16* rows, const ExpertSegment* segments, int num_segments, int first_expert,
                          int hidden, cudaStream_t stream);

// out[t] = sum over k (in order, fp32 fma) of weights[t, k] * returned[positions[t * top_k + k]].
void launch_combine_reduce(const bf16* returned, const std::int32_t* positions, const float* weights,
                           int tokens, int top_k, int hidden, bf16* out, cudaStream_t stream);

}  // namespace nano_nccl::ep
