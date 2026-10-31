#pragma once

#include "ep/ep_kernels.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace nano_nccl::ep {

// BF16 SwiGLU experts hosted by one rank, on cuBLAS (fp32 accumulation, with
// reduced-precision split-K reduction disabled):
//   h = x @ w13   (w13 = [gate | up], [hidden, 2 * ffn], row-major)
//   a = silu(h[:, :ffn]) * h[:, ffn:]          rounded to bf16
//   y = a @ w2    (w2: [ffn, hidden], row-major)
// GEMM outputs are rounded to bf16, as a PyTorch bf16 matmul does.
class ExpertFfn {
public:
    ExpertFfn(int device, int local_experts, int hidden, int ffn, std::int64_t max_rows);
    ~ExpertFfn();
    ExpertFfn(const ExpertFfn&) = delete;
    ExpertFfn& operator=(const ExpertFfn&) = delete;

    bf16* w13(int expert) { return w13_ + static_cast<std::size_t>(expert) * hidden_ * 2 * ffn_; }
    bf16* w2(int expert) { return w2_ + static_cast<std::size_t>(expert) * ffn_ * hidden_; }
    int local_experts() const { return local_experts_; }
    int hidden() const { return hidden_; }
    int ffn() const { return ffn_; }

    // `rows` is expert-major; expert j owns rows [expert_offsets[j], expert_offsets[j + 1]).
    // Outputs overwrite the inputs in place. Asynchronous on `stream`.
    void forward(bf16* rows, const std::vector<std::int64_t>& expert_offsets, cudaStream_t stream);

private:
    void gemm(const bf16* a, const bf16* b, bf16* c, std::int64_t m, int n, int k);

    int device_;
    int local_experts_;
    int hidden_;
    int ffn_;
    std::int64_t max_rows_;
    cublasHandle_t handle_ = nullptr;
    bf16* w13_ = nullptr;
    bf16* w2_ = nullptr;
    bf16* h_ = nullptr;  // [max_rows, 2 * ffn]
    bf16* a_ = nullptr;  // [max_rows, ffn]
};

}  // namespace nano_nccl::ep
