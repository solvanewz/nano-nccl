#include "ep/ep_kernels.h"

#include "core/buffer.h"

#include <algorithm>

#include <cub/block/block_scan.cuh>

namespace nano_nccl::ep {

namespace {

constexpr int kThreads = 256;
constexpr int kRowBlocksPerSegment = 32;

__global__ void count_experts_kernel(const std::int32_t* ids, int entries, int num_experts,
                                     std::int32_t* counts) {
    // Block e counts expert e; block E counts out-of-range ids.
    const int e = blockIdx.x;
    __shared__ int total;
    if (threadIdx.x == 0) total = 0;
    __syncthreads();
    int mine = 0;
    for (int i = threadIdx.x; i < entries; i += blockDim.x) {
        const int id = ids[i];
        const bool match = e < num_experts ? id == e : (id < 0 || id >= num_experts);
        mine += match ? 1 : 0;
    }
    atomicAdd(&total, mine);
    __syncthreads();
    if (threadIdx.x == 0) counts[e] = total;
}

__global__ void place_entries_kernel(const std::int32_t* ids, int entries, const std::int32_t* counts,
                                     std::int32_t* positions) {
    using Scan = cub::BlockScan<int, kThreads>;
    __shared__ typename Scan::TempStorage scan;
    __shared__ int base;
    const int e = blockIdx.x;
    if (threadIdx.x == 0) {
        int sum = 0;
        for (int i = 0; i < e; ++i) sum += counts[i];
        base = sum;
    }
    __syncthreads();
    int running = 0;
    for (int tile = 0; tile < entries; tile += kThreads) {
        const int i = tile + static_cast<int>(threadIdx.x);
        const int flag = (i < entries && ids[i] == e) ? 1 : 0;
        int prefix = 0, aggregate = 0;
        Scan(scan).ExclusiveSum(flag, prefix, aggregate);
        if (flag) positions[i] = base + running + prefix;
        running += aggregate;
        __syncthreads();  // TempStorage is reused by the next tile
    }
}

__device__ inline void copy_row(const bf16* src, bf16* dst, int hidden) {
    const auto* s = reinterpret_cast<const uint4*>(src);
    auto* d = reinterpret_cast<uint4*>(dst);
    for (int v = threadIdx.x; v < hidden / 8; v += blockDim.x) d[v] = s[v];
}

__global__ void pack_rows_kernel(const bf16* x, int hidden, const std::int32_t* positions, int top_k,
                                 bf16* send) {
    const int i = blockIdx.x;
    copy_row(x + static_cast<std::int64_t>(i / top_k) * hidden,
             send + static_cast<std::int64_t>(positions[i]) * hidden, hidden);
}

__global__ void permute_rows_kernel(const bf16* src, bf16* dst, const RowSegment* segments, int hidden) {
    const RowSegment seg = segments[blockIdx.x];
    for (std::int64_t r = blockIdx.y; r < seg.rows; r += gridDim.y) {
        copy_row(src + (seg.src_row + r) * hidden, dst + (seg.dst_row + r) * hidden, hidden);
    }
}

__global__ void swiglu_kernel(const bf16* h, std::int64_t rows, int ffn, bf16* a) {
    const std::int64_t total = rows * ffn;
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < total;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t m = i / ffn;
        const std::int64_t f = i % ffn;
        const float gate = __bfloat162float(h[m * 2 * ffn + f]);
        const float up = __bfloat162float(h[m * 2 * ffn + ffn + f]);
        a[i] = __float2bfloat16_rn(gate / (1.0f + __expf(-gate)) * up);
    }
}

__global__ void probe_experts_kernel(bf16* rows, const std::int64_t* expert_offsets, int first_expert,
                                     int hidden) {
    const int j = blockIdx.x;
    const float scale = probe_scale(first_expert + j);
    const std::int64_t begin = expert_offsets[j];
    const std::int64_t end = expert_offsets[j + 1];
    for (std::int64_t r = begin + blockIdx.y; r < end; r += gridDim.y) {
        bf16* row = rows + r * hidden;
        for (int c = threadIdx.x; c < hidden; c += blockDim.x) {
            row[c] = __float2bfloat16_rn(__bfloat162float(row[c]) * scale);
        }
    }
}

__global__ void combine_reduce_kernel(const bf16* returned, const std::int32_t* positions,
                                      const float* weights, int top_k, int hidden, bf16* out) {
    const int t = blockIdx.x;
    for (int c = threadIdx.x; c < hidden; c += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            const std::int64_t row = positions[t * top_k + k];
            acc = fmaf(weights[t * top_k + k], __bfloat162float(returned[row * hidden + c]), acc);
        }
        out[static_cast<std::int64_t>(t) * hidden + c] = __float2bfloat16_rn(acc);
    }
}

}  // namespace

void launch_count_experts(const std::int32_t* expert_ids, int entries, int num_experts,
                          std::int32_t* counts, cudaStream_t stream) {
    count_experts_kernel<<<num_experts + 1, kThreads, 0, stream>>>(expert_ids, entries, num_experts, counts);
    CUDA_CHECK_THROW(cudaGetLastError());
}

void launch_place_entries(const std::int32_t* expert_ids, int entries, int num_experts,
                          const std::int32_t* counts, std::int32_t* positions, cudaStream_t stream) {
    if (entries == 0) return;
    place_entries_kernel<<<num_experts, kThreads, 0, stream>>>(expert_ids, entries, counts, positions);
    CUDA_CHECK_THROW(cudaGetLastError());
}

void launch_pack_rows(const bf16* x, int hidden, const std::int32_t* positions, int entries, int top_k,
                      bf16* send, cudaStream_t stream) {
    if (entries == 0) return;
    pack_rows_kernel<<<entries, 128, 0, stream>>>(x, hidden, positions, top_k, send);
    CUDA_CHECK_THROW(cudaGetLastError());
}

void launch_permute_rows(const bf16* src, bf16* dst, const RowSegment* segments, int num_segments,
                         int hidden, cudaStream_t stream) {
    if (num_segments == 0) return;
    permute_rows_kernel<<<dim3(num_segments, kRowBlocksPerSegment), 128, 0, stream>>>(src, dst, segments,
                                                                                      hidden);
    CUDA_CHECK_THROW(cudaGetLastError());
}

void launch_swiglu(const bf16* h, std::int64_t rows, int ffn, bf16* a, cudaStream_t stream) {
    const std::int64_t total = rows * ffn;
    if (total == 0) return;
    const auto blocks = static_cast<int>(std::min<std::int64_t>((total + kThreads - 1) / kThreads, 4096));
    swiglu_kernel<<<blocks, kThreads, 0, stream>>>(h, rows, ffn, a);
    CUDA_CHECK_THROW(cudaGetLastError());
}

void launch_probe_experts(bf16* rows, const std::int64_t* expert_offsets, int local_experts,
                          int first_expert, int hidden, cudaStream_t stream) {
    if (local_experts == 0) return;
    probe_experts_kernel<<<dim3(local_experts, kRowBlocksPerSegment), 128, 0, stream>>>(
        rows, expert_offsets, first_expert, hidden);
    CUDA_CHECK_THROW(cudaGetLastError());
}

void launch_combine_reduce(const bf16* returned, const std::int32_t* positions, const float* weights,
                           int tokens, int top_k, int hidden, bf16* out, cudaStream_t stream) {
    if (tokens == 0) return;
    combine_reduce_kernel<<<tokens, 128, 0, stream>>>(returned, positions, weights, top_k, hidden, out);
    CUDA_CHECK_THROW(cudaGetLastError());
}

}  // namespace nano_nccl::ep
