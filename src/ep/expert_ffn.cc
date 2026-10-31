#include "ep/expert_ffn.h"

#include "core/buffer.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace nano_nccl::ep {

namespace {

void cublas_check(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + " failed: cuBLAS status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

template <typename T>
T* device_alloc(std::size_t count) {
    void* ptr = nullptr;
    CUDA_CHECK_THROW(cudaMalloc(&ptr, std::max<std::size_t>(count, 1) * sizeof(T)));
    return static_cast<T*>(ptr);
}

}  // namespace

ExpertFfn::ExpertFfn(int device, int local_experts, int hidden, int ffn, std::int64_t max_rows)
    : device_(device), local_experts_(local_experts), hidden_(hidden), ffn_(ffn), max_rows_(max_rows) {
    if (local_experts <= 0 || hidden <= 0 || ffn <= 0 || max_rows < 0) {
        throw std::invalid_argument("ExpertFfn: bad shape");
    }
    CUDA_CHECK_THROW(cudaSetDevice(device_));
    const auto e = static_cast<std::size_t>(local_experts);
    const auto h = static_cast<std::size_t>(hidden);
    const auto f = static_cast<std::size_t>(ffn);
    const auto m = static_cast<std::size_t>(max_rows);
    try {
        cublas_check(cublasCreate(&handle_), "cublasCreate");
        // By default cuBLAS may reduce split-K partial sums in bf16, which
        // breaks the fp32-accumulation contract (measured on A40: relative L2
        // error 3.9e-3 against a reference with fp32 reductions, 6e-5 with
        // this mode).
        cublas_check(cublasSetMathMode(handle_, static_cast<cublasMath_t>(
                                                    CUBLAS_DEFAULT_MATH |
                                                    CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)),
                     "cublasSetMathMode");
        w13_ = device_alloc<bf16>(e * h * 2 * f);
        w2_ = device_alloc<bf16>(e * f * h);
        h_ = device_alloc<bf16>(m * 2 * f);
        a_ = device_alloc<bf16>(m * f);
    } catch (...) {
        cudaFree(w13_);
        cudaFree(w2_);
        cudaFree(h_);
        if (handle_ != nullptr) cublasDestroy(handle_);
        throw;
    }
}

ExpertFfn::~ExpertFfn() {
    cudaSetDevice(device_);
    cudaFree(w13_);
    cudaFree(w2_);
    cudaFree(h_);
    cudaFree(a_);
    cublasDestroy(handle_);
}

// Row-major C[m, n] = A[m, k] @ B[k, n], i.e. column-major C^T = B^T A^T.
void ExpertFfn::gemm(const bf16* a, const bf16* b, bf16* c, std::int64_t m, int n, int k) {
    const float alpha = 1.0f, beta = 0.0f;
    cublas_check(cublasGemmEx(handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, static_cast<int>(m), k, &alpha, b,
                              CUDA_R_16BF, n, a, CUDA_R_16BF, k, &beta, c, CUDA_R_16BF, n,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                 "cublasGemmEx");
}

void ExpertFfn::forward(bf16* rows, const std::vector<ExpertSegment>& segments, cudaStream_t stream) {
    for (const ExpertSegment& seg : segments) {
        if (seg.expert < 0 || seg.expert >= local_experts_ || seg.begin < 0 || seg.rows < 0 ||
            seg.begin + seg.rows > max_rows_) {
            throw std::invalid_argument("ExpertFfn::forward: segment for expert " + std::to_string(seg.expert) +
                                        " rows [" + std::to_string(seg.begin) + ", " +
                                        std::to_string(seg.begin + seg.rows) + ") is outside capacity " +
                                        std::to_string(max_rows_));
        }
    }
    CUDA_CHECK_THROW(cudaSetDevice(device_));
    cublas_check(cublasSetStream(handle_, stream), "cublasSetStream");
    // The workspace rows mirror the expert-major rows, so segments never share them.
    for (const ExpertSegment& seg : segments) {
        if (seg.rows == 0) continue;
        gemm(rows + seg.begin * hidden_, w13(seg.expert), h_ + seg.begin * 2 * ffn_, seg.rows, 2 * ffn_, hidden_);
        launch_swiglu(h_ + seg.begin * 2 * ffn_, seg.rows, ffn_, a_ + seg.begin * ffn_, stream);
        gemm(a_ + seg.begin * ffn_, w2(seg.expert), rows + seg.begin * hidden_, seg.rows, hidden_, ffn_);
    }
}

}  // namespace nano_nccl::ep
