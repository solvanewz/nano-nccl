// Expert FFN concurrency microbenchmark: k experts with few rows each, run
//   serial   one cuBLAS handle on one stream, expert after expert; the same
//            launch sequence as ExpertFfn::forward (2 GEMMs + SwiGLU each);
//   streams  expert j on stream j % S (one handle per stream), forked from and
//            joined back to the main stream every iteration;
//   grouped  cublasGemmGroupedBatchedEx (one group per expert) for each GEMM,
//            one SwiGLU over all rows: 3 launches in total;
//   batched  cublasGemmStridedBatchedEx, only when every expert has the same
//            rows (the ideal case for batching).
//
//   nano_nccl_expert_concurrency_bench [--hidden H --ffn-dim F --device D
//       --streams S --trials N --reps R --case ROWS[,ROWS...] ... --out FILE]
//
// Each --case lists the rows of each expert (default: a built-in set). Every
// expert has its own weights, so concurrent experts do not share L2 lines.
// The GEMMs use ExpertFfn's settings (bf16 in/out, fp32 compute, reduced
// precision split-K reduction disallowed).
//
// Device time: the iterations are queued behind a spin kernel so host issue
// time is hidden; a run whose queue drained before the host finished issuing
// is flagged "drained" (its device time then includes host gaps). Host time:
// wall time of the issuing calls per iteration. The outputs of every mode are
// compared bitwise with serial.
//
// Single GPU, no peer access, no MPI.
#include "ep/ep_kernels.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

using nano_nccl::ep::bf16;
using Clock = std::chrono::steady_clock;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

void cublas_check(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": cuBLAS status " + std::to_string(static_cast<int>(s)));
    }
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0 : v[v.size() / 2];
}

__global__ void spin_kernel(long long cycles) {
    const long long start = clock64();
    while (clock64() - start < cycles) {
    }
}

// Deterministic values in about [-scale, scale].
__global__ void fill_kernel(bf16* p, std::size_t n, std::uint32_t seed, float scale) {
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        std::uint32_t x = static_cast<std::uint32_t>(i) * 2654435761u ^ seed;
        x ^= x >> 15;
        x *= 2246822519u;
        x ^= x >> 13;
        p[i] = __float2bfloat16(scale * (static_cast<float>(x & 0xffff) / 32768.0f - 1.0f));
    }
}

void fill(bf16* p, std::size_t n, std::uint32_t seed, float scale) {
    fill_kernel<<<256, 256>>>(p, n, seed, scale);
    check(cudaGetLastError(), "fill");
}

cublasHandle_t make_handle() {
    cublasHandle_t h = nullptr;
    cublas_check(cublasCreate(&h), "cublasCreate");
    cublas_check(cublasSetMathMode(h, static_cast<cublasMath_t>(CUBLAS_DEFAULT_MATH |
                                                                CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)),
                 "cublasSetMathMode");
    return h;
}

// Row-major C[m, n] = A[m, k] @ B[k, n], exactly as ExpertFfn::gemm.
void gemm(cublasHandle_t h, const bf16* a, const bf16* b, bf16* c, std::int64_t m, int n, int k) {
    const float alpha = 1.0f, beta = 0.0f;
    cublas_check(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, static_cast<int>(m), k, &alpha, b, CUDA_R_16BF, n, a,
                              CUDA_R_16BF, k, &beta, c, CUDA_R_16BF, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                 "cublasGemmEx");
}

struct Args {
    int hidden = 2048;
    int ffn = 1408;
    int device = 0;
    int streams = 8;
    int trials = 7;
    int reps = 10;
    std::vector<std::vector<int>> cases;
    std::string out;
};

std::vector<std::vector<int>> default_cases() {
    std::vector<std::vector<int>> c;
    for (int k : {1, 2, 4, 8}) {
        for (int r : {16, 64, 256}) c.push_back(std::vector<int>(k, r));
    }
    c.push_back(std::vector<int>(4, 512));
    c.push_back({1024, 16, 16, 16, 16, 16, 16, 16});
    c.push_back({512, 64, 64, 64});
    c.push_back({256, 128, 64, 32, 16, 8, 4, 1});
    return c;
}

std::string case_name(const std::vector<int>& rows) {
    std::string s;
    for (std::size_t i = 0; i < rows.size(); ++i) s += (i ? "," : "") + std::to_string(rows[i]);
    return s;
}

struct Result {
    std::string mode;
    bool supported = true;
    std::string note;
    double device_us = 0, device_min_us = 0, host_us = 0;
    bool drained = false;
    std::size_t mismatches = 0;
    double max_abs_diff = 0;
};

float bf16_bits_to_float(std::uint16_t b) {
    const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

class Bench {
public:
    explicit Bench(const Args& a) : a_(a) {
        check(cudaSetDevice(a.device), "cudaSetDevice");
        int khz = 0;
        check(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, a.device), "clock rate");
        clock_khz_ = khz;
        for (const auto& c : a.cases) {
            max_experts_ = std::max<int>(max_experts_, static_cast<int>(c.size()));
            std::int64_t r = 0;
            for (int x : c) r += x;
            max_rows_ = std::max(max_rows_, r);
        }
        const std::size_t h = a.hidden, f = a.ffn, e = max_experts_, m = std::max<std::int64_t>(max_rows_, 1);
        check(cudaMalloc(&w13_, e * h * 2 * f * sizeof(bf16)), "w13");
        check(cudaMalloc(&w2_, e * f * h * sizeof(bf16)), "w2");
        check(cudaMalloc(&x_, m * h * sizeof(bf16)), "x");
        check(cudaMalloc(&y_, m * h * sizeof(bf16)), "y");
        check(cudaMalloc(&h_, m * 2 * f * sizeof(bf16)), "h");
        check(cudaMalloc(&act_, m * f * sizeof(bf16)), "act");
        check(cudaMalloc(&ptrs_, 6 * e * sizeof(void*)), "ptrs");
        fill(w13_, e * h * 2 * f, 1u, 0.05f);
        fill(w2_, e * f * h, 2u, 0.05f);
        fill(x_, m * h, 3u, 1.0f);
        check(cudaDeviceSynchronize(), "init");
        check(cudaStreamCreateWithFlags(&main_, cudaStreamNonBlocking), "stream");
        main_handle_ = make_handle();
        cublas_check(cublasSetStream(main_handle_, main_), "cublasSetStream");
        for (int i = 0; i < a.streams; ++i) {
            cudaStream_t s;
            check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
            side_.push_back(s);
            cublasHandle_t hd = make_handle();
            cublas_check(cublasSetStream(hd, s), "cublasSetStream");
            side_handles_.push_back(hd);
            cudaEvent_t ev;
            check(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "event");
            side_done_.push_back(ev);
        }
        check(cudaEventCreateWithFlags(&fork_, cudaEventDisableTiming), "event");
        check(cudaEventCreate(&t0_), "event");
        check(cudaEventCreate(&t1_), "event");
        check(cudaEventCreateWithFlags(&spin_done_, cudaEventDisableTiming), "event");
    }

    ~Bench() {
        cudaSetDevice(a_.device);
        cudaDeviceSynchronize();
        for (auto hd : side_handles_) cublasDestroy(hd);
        cublasDestroy(main_handle_);
        for (auto s : side_) cudaStreamDestroy(s);
        for (auto ev : side_done_) cudaEventDestroy(ev);
        cudaEventDestroy(fork_);
        cudaEventDestroy(t0_);
        cudaEventDestroy(t1_);
        cudaEventDestroy(spin_done_);
        cudaStreamDestroy(main_);
        cudaFree(w13_);
        cudaFree(w2_);
        cudaFree(x_);
        cudaFree(y_);
        cudaFree(h_);
        cudaFree(act_);
        cudaFree(ptrs_);
    }

    std::vector<Result> run_case(const std::vector<int>& rows) {
        set_case(rows);
        std::vector<Result> out;
        out.push_back(measure("serial", [&] { serial(); }));
        const std::vector<std::uint16_t> ref = output();
        out.back().mismatches = 0;
        const int s = std::min<int>(a_.streams, static_cast<int>(rows.size()));
        out.push_back(measure("streams" + std::to_string(s), [&] { streams(s); }));
        compare(out.back(), ref);
        out.push_back(try_measure("grouped", [&] { grouped(); }, ref));
        const bool equal = std::all_of(rows.begin(), rows.end(), [&](int r) { return r == rows[0]; });
        if (equal && rows[0] > 0) {
            out.push_back(try_measure("batched", [&] { batched(); }, ref));
        }
        return out;
    }

    double weight_bytes(const std::vector<int>& rows) const {
        int active = 0;
        for (int r : rows) active += r > 0;
        return static_cast<double>(active) * 3.0 * a_.hidden * a_.ffn * sizeof(bf16);
    }

private:
    bf16* w13(int j) const { return w13_ + static_cast<std::size_t>(j) * a_.hidden * 2 * a_.ffn; }
    bf16* w2(int j) const { return w2_ + static_cast<std::size_t>(j) * a_.ffn * a_.hidden; }

    void set_case(const std::vector<int>& rows) {
        rows_ = rows;
        begin_.assign(rows.size(), 0);
        total_rows_ = 0;
        for (std::size_t j = 0; j < rows.size(); ++j) {
            begin_[j] = total_rows_;
            total_rows_ += rows[j];
        }
        // Device pointer arrays for the grouped GEMMs (active experts only),
        // uploaded once per case as a runtime would upload them with its step tables.
        g_ = {};
        std::vector<const void*> host(6 * rows.size(), nullptr);
        int n = 0;
        for (std::size_t j = 0; j < rows.size(); ++j) {
            if (rows[j] == 0) continue;
            const std::size_t b = begin_[j];
            g_.rows.push_back(rows[j]);
            host[0 * rows.size() + n] = w13(static_cast<int>(j));
            host[1 * rows.size() + n] = x_ + b * a_.hidden;
            host[2 * rows.size() + n] = h_ + b * 2 * a_.ffn;
            host[3 * rows.size() + n] = w2(static_cast<int>(j));
            host[4 * rows.size() + n] = act_ + b * a_.ffn;
            host[5 * rows.size() + n] = y_ + b * a_.hidden;
            ++n;
        }
        g_.count = n;
        stride_ = rows.size();
        check(cudaMemcpy(ptrs_, host.data(), host.size() * sizeof(void*), cudaMemcpyHostToDevice), "ptrs");
    }

    void expert(cublasHandle_t hd, cudaStream_t s, std::size_t j) {
        if (rows_[j] == 0) return;
        const std::size_t b = begin_[j];
        gemm(hd, x_ + b * a_.hidden, w13(static_cast<int>(j)), h_ + b * 2 * a_.ffn, rows_[j], 2 * a_.ffn, a_.hidden);
        nano_nccl::ep::launch_swiglu(h_ + b * 2 * a_.ffn, rows_[j], a_.ffn, act_ + b * a_.ffn, s);
        gemm(hd, act_ + b * a_.ffn, w2(static_cast<int>(j)), y_ + b * a_.hidden, rows_[j], a_.hidden, a_.ffn);
    }

    void serial() {
        for (std::size_t j = 0; j < rows_.size(); ++j) expert(main_handle_, main_, j);
    }

    void streams(int s) {
        check(cudaEventRecord(fork_, main_), "record");
        for (int i = 0; i < s; ++i) check(cudaStreamWaitEvent(side_[i], fork_, 0), "wait");
        for (std::size_t j = 0; j < rows_.size(); ++j) expert(side_handles_[j % s], side_[j % s], j);
        for (int i = 0; i < s; ++i) {
            check(cudaEventRecord(side_done_[i], side_[i]), "record");
            check(cudaStreamWaitEvent(main_, side_done_[i], 0), "wait");
        }
    }

    void grouped_gemm(const void* const* a, const void* const* b, void* const* c, int n, int k) {
        // Row-major C[m, n] = A[m, k] @ B[k, n] as column-major C^T = B^T A^T,
        // with the weights (B) as cuBLAS's first operand, as in gemm().
        const int g = g_.count;
        std::vector<cublasOperation_t> op(g, CUBLAS_OP_N);
        std::vector<int> mm(g, n), nn(g_.rows.begin(), g_.rows.end()), kk(g, k), lda(g, n), ldb(g, k), ldc(g, n),
            size(g, 1);
        std::vector<float> alpha(g, 1.0f), beta(g, 0.0f);
        cublas_check(cublasGemmGroupedBatchedEx(main_handle_, op.data(), op.data(), mm.data(), nn.data(), kk.data(),
                                                alpha.data(), b, CUDA_R_16BF, lda.data(), a, CUDA_R_16BF, ldb.data(),
                                                beta.data(), c, CUDA_R_16BF, ldc.data(), g, size.data(),
                                                CUBLAS_COMPUTE_32F),
                     "cublasGemmGroupedBatchedEx");
    }

    void grouped() {
        if (g_.count == 0) return;
        void** p = ptrs_;
        grouped_gemm(p + 1 * stride_, p + 0 * stride_, p + 2 * stride_, 2 * a_.ffn, a_.hidden);
        nano_nccl::ep::launch_swiglu(h_, total_rows_, a_.ffn, act_, main_);
        grouped_gemm(p + 4 * stride_, p + 3 * stride_, p + 5 * stride_, a_.hidden, a_.ffn);
    }

    void batched() {
        const int k = static_cast<int>(rows_.size());
        const int m = rows_[0];
        const float alpha = 1.0f, beta = 0.0f;
        const long long hw = static_cast<long long>(a_.hidden) * 2 * a_.ffn, fw = static_cast<long long>(a_.ffn) * a_.hidden;
        cublas_check(cublasGemmStridedBatchedEx(main_handle_, CUBLAS_OP_N, CUBLAS_OP_N, 2 * a_.ffn, m, a_.hidden,
                                                &alpha, w13_, CUDA_R_16BF, 2 * a_.ffn, hw, x_, CUDA_R_16BF, a_.hidden,
                                                static_cast<long long>(m) * a_.hidden, &beta, h_, CUDA_R_16BF,
                                                2 * a_.ffn, static_cast<long long>(m) * 2 * a_.ffn, k,
                                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                     "cublasGemmStridedBatchedEx");
        nano_nccl::ep::launch_swiglu(h_, total_rows_, a_.ffn, act_, main_);
        cublas_check(cublasGemmStridedBatchedEx(main_handle_, CUBLAS_OP_N, CUBLAS_OP_N, a_.hidden, m, a_.ffn, &alpha,
                                                w2_, CUDA_R_16BF, a_.hidden, fw, act_, CUDA_R_16BF, a_.ffn,
                                                static_cast<long long>(m) * a_.ffn, &beta, y_, CUDA_R_16BF, a_.hidden,
                                                static_cast<long long>(m) * a_.hidden, k, CUBLAS_COMPUTE_32F,
                                                CUBLAS_GEMM_DEFAULT),
                     "cublasGemmStridedBatchedEx");
    }

    void spin(double us) {
        spin_kernel<<<1, 1, 0, main_>>>(static_cast<long long>(us * clock_khz_ / 1000.0));
        check(cudaGetLastError(), "spin");
    }

    Result measure(const std::string& mode, const std::function<void()>& enqueue) {
        Result r;
        r.mode = mode;
        // Warm-up, and a host-issue estimate to size the spin.
        for (int i = 0; i < 3; ++i) enqueue();
        check(cudaStreamSynchronize(main_), "sync");
        const auto w0 = Clock::now();
        enqueue();
        const double issue_us = std::chrono::duration<double, std::micro>(Clock::now() - w0).count();
        check(cudaStreamSynchronize(main_), "sync");
        const double spin_us = std::max(3000.0, 3.0 * issue_us * a_.reps + 1000.0);
        std::vector<double> dev, host;
        for (int t = 0; t < a_.trials; ++t) {
            spin(spin_us);
            check(cudaEventRecord(spin_done_, main_), "record");
            check(cudaEventRecord(t0_, main_), "record");
            const auto h0 = Clock::now();
            for (int i = 0; i < a_.reps; ++i) enqueue();
            const auto h1 = Clock::now();
            if (cudaEventQuery(spin_done_) == cudaSuccess) r.drained = true;
            check(cudaEventRecord(t1_, main_), "record");
            check(cudaEventSynchronize(t1_), "sync");
            float ms = 0;
            check(cudaEventElapsedTime(&ms, t0_, t1_), "elapsed");
            dev.push_back(1000.0 * ms / a_.reps);
            host.push_back(std::chrono::duration<double, std::micro>(h1 - h0).count() / a_.reps);
        }
        r.device_us = median(dev);
        r.device_min_us = *std::min_element(dev.begin(), dev.end());
        r.host_us = median(host);
        // Leave this mode's output in y_ for the comparison.
        check(cudaMemsetAsync(y_, 0, static_cast<std::size_t>(std::max<std::int64_t>(total_rows_, 1)) * a_.hidden *
                                         sizeof(bf16),
                              main_),
              "memset");
        enqueue();
        check(cudaStreamSynchronize(main_), "sync");
        return r;
    }

    Result try_measure(const std::string& mode, const std::function<void()>& enqueue,
                       const std::vector<std::uint16_t>& ref) {
        try {
            Result r = measure(mode, enqueue);
            compare(r, ref);
            return r;
        } catch (const std::runtime_error& e) {
            // An unsupported cuBLAS call fails at issue time, before any work is queued.
            check(cudaDeviceSynchronize(), "sync after failed mode");
            Result r;
            r.mode = mode;
            r.supported = false;
            r.note = e.what();
            return r;
        }
    }

    std::vector<std::uint16_t> output() {
        std::vector<std::uint16_t> v(static_cast<std::size_t>(total_rows_) * a_.hidden);
        check(cudaMemcpy(v.data(), y_, v.size() * sizeof(bf16), cudaMemcpyDeviceToHost), "copy out");
        return v;
    }

    void compare(Result& r, const std::vector<std::uint16_t>& ref) {
        const std::vector<std::uint16_t> got = output();
        for (std::size_t i = 0; i < ref.size(); ++i) {
            if (got[i] == ref[i]) continue;
            ++r.mismatches;
            r.max_abs_diff = std::max<double>(r.max_abs_diff, std::abs(bf16_bits_to_float(got[i]) -
                                                                      bf16_bits_to_float(ref[i])));
        }
    }

    const Args& a_;
    double clock_khz_ = 1e6;
    int max_experts_ = 1;
    std::int64_t max_rows_ = 0;
    bf16 *w13_ = nullptr, *w2_ = nullptr, *x_ = nullptr, *y_ = nullptr, *h_ = nullptr, *act_ = nullptr;
    void** ptrs_ = nullptr;
    std::size_t stride_ = 0;
    cudaStream_t main_ = nullptr;
    cublasHandle_t main_handle_ = nullptr;
    std::vector<cudaStream_t> side_;
    std::vector<cublasHandle_t> side_handles_;
    std::vector<cudaEvent_t> side_done_;
    cudaEvent_t fork_ = nullptr, t0_ = nullptr, t1_ = nullptr, spin_done_ = nullptr;
    std::vector<int> rows_;
    std::vector<std::int64_t> begin_;
    std::int64_t total_rows_ = 0;
    struct {
        int count = 0;
        std::vector<int> rows;
    } g_;
};

std::vector<int> parse_rows(const std::string& s) {
    std::vector<int> v;
    std::size_t pos = 0;
    while (pos <= s.size()) {
        const std::size_t comma = s.find(',', pos);
        const std::string tok = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        const int r = std::atoi(tok.c_str());
        if (tok.empty() || r < 0) throw std::invalid_argument("--case: bad row count '" + tok + "'");
        v.push_back(r);
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return v;
}

int run(Args a) {
    if (a.cases.empty()) a.cases = default_cases();
    if (a.streams < 1 || a.trials < 1 || a.reps < 1) throw std::invalid_argument("streams/trials/reps must be >= 1");
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, a.device), "props");
    Bench bench(a);
    FILE* out = a.out.empty() ? nullptr : std::fopen(a.out.c_str(), "w");
    if (!a.out.empty() && out == nullptr) throw std::runtime_error("cannot open " + a.out);
    std::printf("# device %s, hidden %d, ffn %d, streams <= %d, trials %d, reps %d\n", prop.name, a.hidden, a.ffn,
                a.streams, a.trials, a.reps);
    std::printf("%-34s %-9s %10s %10s %9s %7s %7s %s\n", "case(rows per expert)", "mode", "device_us", "min_us",
                "host_us", "speedup", "GB/s", "notes");
    if (out) {
        std::fprintf(out, "# device %s hidden %d ffn %d streams %d trials %d reps %d\n", prop.name, a.hidden, a.ffn,
                     a.streams, a.trials, a.reps);
        std::fprintf(out, "case\texperts\trows\tmode\tdevice_us\tdevice_min_us\thost_us\tspeedup\tweight_gbps\t"
                          "drained\tmismatches\tmax_abs_diff\tnote\n");
    }
    for (const auto& c : a.cases) {
        const std::vector<Result> rs = bench.run_case(c);
        const double serial_us = rs[0].device_us;
        const double bytes = bench.weight_bytes(c);
        std::int64_t total = 0;
        for (int r : c) total += r;
        for (const Result& r : rs) {
            const std::string name = case_name(c);
            if (!r.supported) {
                std::printf("%-34s %-9s %10s %10s %9s %7s %7s unsupported: %s\n", name.c_str(), r.mode.c_str(), "-",
                            "-", "-", "-", "-", r.note.c_str());
                if (out) {
                    std::fprintf(out, "%s\t%zu\t%lld\t%s\t\t\t\t\t\t\t\t\tunsupported: %s\n", name.c_str(), c.size(),
                                 static_cast<long long>(total), r.mode.c_str(), r.note.c_str());
                }
                continue;
            }
            const double speedup = r.device_us > 0 ? serial_us / r.device_us : 0;
            const double gbps = r.device_us > 0 ? bytes / (r.device_us * 1e3) : 0;
            std::string notes;
            if (r.drained) notes += "drained ";
            notes += r.mismatches == 0 ? "bitwise" : "differs(" + std::to_string(r.mismatches) + ", max " +
                                                         std::to_string(r.max_abs_diff) + ")";
            std::printf("%-34s %-9s %10.1f %10.1f %9.1f %7.2f %7.0f %s\n", name.c_str(), r.mode.c_str(), r.device_us,
                        r.device_min_us, r.host_us, speedup, gbps, notes.c_str());
            if (out) {
                std::fprintf(out, "%s\t%zu\t%lld\t%s\t%.2f\t%.2f\t%.2f\t%.4f\t%.1f\t%d\t%zu\t%g\t\n", name.c_str(),
                             c.size(), static_cast<long long>(total), r.mode.c_str(), r.device_us, r.device_min_us,
                             r.host_us, speedup, gbps, r.drained ? 1 : 0, r.mismatches, r.max_abs_diff);
            }
        }
        std::fflush(stdout);
    }
    if (out) std::fclose(out);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args a;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto next = [&]() -> const char* {
                if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
                return argv[++i];
            };
            if (arg == "--hidden") a.hidden = std::atoi(next());
            else if (arg == "--ffn-dim") a.ffn = std::atoi(next());
            else if (arg == "--device") a.device = std::atoi(next());
            else if (arg == "--streams") a.streams = std::atoi(next());
            else if (arg == "--trials") a.trials = std::atoi(next());
            else if (arg == "--reps") a.reps = std::atoi(next());
            else if (arg == "--case") a.cases.push_back(parse_rows(next()));
            else if (arg == "--out") a.out = next();
            else throw std::invalid_argument("unknown argument " + arg);
        }
        return run(a);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "expert_concurrency_bench: %s\n", e.what());
        return 1;
    }
}
