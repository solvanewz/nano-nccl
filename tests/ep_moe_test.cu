// EP MoE layer over ShmWire (B1), one process per rank under mpirun.
//
//   ep_moe_test [--slots S] [--slot-bytes B] [--copy-delay-us D]
//       probe matrix: experts scale rows by powers of two, so every output
//       is checked bit-exactly against a host reference. Covers skewed
//       routing, ranks with no tokens, all tokens routed to one rank's
//       experts, unequal token counts, and several forwards per layer, each
//       under several group plans and pipeline schedules (serial, fixed
//       groups with and without overlap, all dispatches ahead, and an
//       arbitrary expert->group mapping run in reverse order). D slows every
//       ShmWire copy (ShmWireConfig::debug_copy_delay_us), so data used
//       before it is readable shows up as wrong output.
//   ep_moe_test --ffn [--tokens T --experts E --topk K --hidden H --ffn-dim F
//                      --iters N --skew X --groups G --lookahead L
//                      --no-overlap --dump DIR --trace FILE]
//       real BF16 SwiGLU experts with the B1 fixed group plan; prints stage
//       times and the step timeline, and checks that the output is
//       bit-identical to a serial (one group, no overlap) forward. With
//       --dump, writes inputs, weights and outputs for
//       tests/ep_moe_reference.py.
#include "ep/moe_layer.h"
#include "ep/synthetic_routing.h"
#include "exec/exec_trace.h"
#include "transport/peer/peer_exchanger.h"
#include "transport/peer/shm_wire.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mpi.h>

namespace {

using nano_nccl::ep::bf16;
using nano_nccl::ep::EpShape;
using nano_nccl::ep::ExpertFfn;
using nano_nccl::ep::ExpertMode;
using nano_nccl::ep::MoeLayer;
using nano_nccl::ep::MoeLayerConfig;
using nano_nccl::ep::ForwardTimes;
using nano_nccl::ep::PipelineOptions;
using nano_nccl::planner::GroupPlan;
using nano_nccl::ep::probe_scale;
using nano_nccl::transport::peer::PeerExchanger;
using nano_nccl::transport::peer::ShmWire;
namespace exec = nano_nccl::exec;

int g_rank = 0;
int g_world = 0;
int g_device = 0;

void cuda_check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

template <typename T>
struct DeviceArray {
    T* ptr = nullptr;
    explicit DeviceArray(std::size_t n) { cuda_check(cudaMalloc(&ptr, std::max<std::size_t>(n, 1) * sizeof(T)), "cudaMalloc"); }
    ~DeviceArray() { cudaFree(ptr); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    void upload(const std::vector<T>& v) {
        cuda_check(cudaMemcpy(ptr, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    }
    std::vector<T> download(std::size_t n) const {
        std::vector<T> v(n);
        cuda_check(cudaMemcpy(v.data(), ptr, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
        return v;
    }
};

float uniform(std::uint64_t seed, std::uint64_t i) {  // [-1, 1)
    std::uint64_t x = seed * 0x9E3779B97F4A7C15ull + i * 0xD1B54A32D192ED03ull;
    x ^= x >> 33;
    x *= 0xFF51AFD7ED558CCDull;
    x ^= x >> 33;
    x *= 0xC4CEB9FE1A85EC53ull;
    x ^= x >> 33;
    return static_cast<float>(x >> 40) / static_cast<float>(1ull << 23) - 1.0f;
}

std::vector<bf16> random_bf16(std::size_t n, std::uint64_t seed, float scale) {
    std::vector<bf16> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = __float2bfloat16_rn(uniform(seed, i) * scale);
    return v;
}

using nano_nccl::ep::make_routing;
using nano_nccl::ep::Routing;

std::unique_ptr<PeerExchanger> g_exchanger;

struct ProbeCase {
    const char* name;
    int experts_per_rank;
    int top_k;
    int hidden;
    double skew;
    int only_rank;     // -1: any expert
    int base_tokens;   // rank r gets base + r * step tokens, except zero_rank
    int step_tokens;
    int zero_rank;     // -1: none
};

int tokens_of(const ProbeCase& c, int rank) {
    if (rank == c.zero_rank) return 0;
    return c.base_tokens + rank * c.step_tokens;
}

struct Variant {
    const char* name;
    int groups;      // 0: one group per local expert
    bool overlap;
    int lookahead;
    bool scrambled;  // arbitrary expert -> group mapping, groups run in reverse
};

constexpr Variant kVariants[] = {
    {"serial", 1, false, 0, false},
    {"g2_overlap", 2, true, 1, false},
    {"gmax_overlap", 0, true, 1, false},
    {"gmax_no_overlap", 0, false, 0, false},
    {"gmax_all_ahead", 0, true, 64, false},
    {"scrambled_l0", 3, true, 0, true},
};

// Identical on every rank: deterministic in the shape.
GroupPlan make_group_plan(const EpShape& s, const Variant& v) {
    const int epr = s.experts_per_rank();
    if (!v.scrambled) {
        const int groups = std::min(v.groups == 0 ? epr : v.groups, epr);
        return nano_nccl::planner::fixed_group_plan(s.world, s.num_experts, groups);
    }
    const int groups = std::min(v.groups, s.num_experts);
    std::mt19937_64 rng(static_cast<std::uint64_t>(s.num_experts) * 31 + s.top_k);
    GroupPlan plan;
    for (int e = 0; e < s.num_experts; ++e) plan.expert_group.push_back(static_cast<int>(rng() % groups));
    for (int g = groups - 1; g >= 0; --g) plan.order.push_back(g);
    return plan;
}

// Returns the number of output values checked on this rank.
std::uint64_t run_probe_case(const ProbeCase& c, const Variant& variant, std::uint64_t* epoch) {
    EpShape s{g_world, g_rank, c.experts_per_rank * g_world, c.top_k, c.hidden, 0};
    int max_tokens = 0;
    for (int r = 0; r < g_world; ++r) max_tokens = std::max(max_tokens, tokens_of(c, r));
    MoeLayerConfig cfg{s, max_tokens, 0, ExpertMode::kProbe};
    MoeLayer layer(*g_exchanger, cfg, nullptr, g_device);
    cudaStream_t stream;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
    const GroupPlan plan = make_group_plan(s, variant);
    const PipelineOptions options{variant.overlap, variant.lookahead};
    std::uint64_t checked = 0;
    for (int iter = 0; iter < 3; ++iter) {
        // Different token counts per iteration so stale rows would show up.
        const int tokens = std::max(0, tokens_of(c, g_rank) - (g_rank == c.zero_rank ? 0 : iter * 7));
        const std::uint64_t seed = 1000003ull * (iter + 1) + 7919ull * g_rank + c.hidden;
        const auto x = random_bf16(static_cast<std::size_t>(tokens) * c.hidden, seed, 1.0f);
        const Routing routing = make_routing(s, tokens, c.skew, c.only_rank, seed ^ 0xABCDEFull);
        DeviceArray<bf16> dx(x.size()), dout(x.size());
        DeviceArray<std::int32_t> dids(routing.ids.size());
        DeviceArray<float> dw(routing.weights.size());
        dx.upload(x);
        dids.upload(routing.ids);
        dw.upload(routing.weights);
        cuda_check(cudaMemset(dout.ptr, 0xFF, std::max<std::size_t>(x.size(), 1) * sizeof(bf16)), "memset");
        layer.forward((*epoch)++, dx.ptr, tokens, dids.ptr, dw.ptr, dout.ptr, plan, options, stream);
        const auto out = dout.download(x.size());
        for (int t = 0; t < tokens; ++t) {
            for (int h = 0; h < c.hidden; ++h) {
                float acc = 0.0f;
                for (int k = 0; k < c.top_k; ++k) {
                    const int e = routing.ids[t * c.top_k + k];
                    const float v = __bfloat162float(
                        __float2bfloat16_rn(__bfloat162float(x[t * c.hidden + h]) * probe_scale(e)));
                    acc = std::fma(routing.weights[t * c.top_k + k], v, acc);
                }
                const bf16 want = __float2bfloat16_rn(acc);
                const bf16 got = out[t * c.hidden + h];
                if (std::memcmp(&want, &got, sizeof(bf16)) != 0) {
                    throw std::runtime_error(std::string(c.name) + "/" + variant.name + " iter " +
                                             std::to_string(iter) + ": token " +
                                             std::to_string(t) + " col " + std::to_string(h) + " got " +
                                             std::to_string(__bfloat162float(got)) + " want " +
                                             std::to_string(__bfloat162float(want)));
                }
            }
        }
        checked += static_cast<std::uint64_t>(tokens) * c.hidden;
    }
    cudaStreamDestroy(stream);
    return checked;
}

int run_probe() {
    const ProbeCase cases[] = {
        {"uniform", 2, 2, 64, 0.0, -1, 257, 0, -1},
        {"skewed_k4", 4, 4, 128, 1.5, -1, 1000, 0, -1},
        {"top1_zero_rank", 1, 1, 256, 0.0, -1, 300, 0, 1},
        {"all_to_rank0", 2, 2, 64, 0.0, 0, 500, 0, -1},
        {"unequal_tokens_k8", 8, 8, 32, 0.8, -1, 100, 37, -1},
        {"single_token", 2, 2, 1024, 0.0, -1, 1, 0, -1},
        {"wide_rows", 2, 2, 4096, 0.5, -1, 200, 50, -1},
    };
    std::uint64_t epoch = 1, checked = 0;
    int runs = 0;
    for (const ProbeCase& c : cases) {
        for (const Variant& v : kVariants) {
            try {
                checked += run_probe_case(c, v, &epoch);
            } catch (const std::exception& error) {
                std::printf("FAIL rank %d probe %s/%s: %s\n", g_rank, c.name, v.name, error.what());
                std::fflush(stdout);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            ++runs;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (g_rank == 0) {
            std::printf("PASS probe %s (%zu plans)\n", c.name, sizeof(kVariants) / sizeof(kVariants[0]));
            std::fflush(stdout);
        }
    }
    std::uint64_t all = 0;
    MPI_Reduce(&checked, &all, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    if (g_rank == 0) {
        std::printf("probe: %d case x plan runs, %llu output values bit-exact\n", runs,
                    static_cast<unsigned long long>(all));
        std::printf("ep_moe_probe=PASS\n");
    }
    return 0;
}

void write_file(const std::string& path, const void* data, std::size_t bytes) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) throw std::runtime_error("cannot open " + path);
    const bool ok = std::fwrite(data, 1, bytes, f) == bytes;
    std::fclose(f);
    if (!ok) throw std::runtime_error("short write to " + path);
}

struct FfnArgs {
    int tokens = 2048;
    int experts = 8;
    int top_k = 2;
    int hidden = 1024;
    int ffn = 2048;
    int iters = 10;
    double skew = 0.0;
    int groups = 1;
    int lookahead = 1;
    bool overlap = true;
    std::string dump;
    std::string trace;  // exec::LayerTrace of the timed runs, written by rank 0
    int slots = 0;
    std::size_t slot_bytes = 0;
};

// Gathers every rank's counts, stage means and last-iteration op times.
void write_trace(const FfnArgs& a, const EpShape& s, const MoeLayer& layer, const ForwardTimes& last,
                 const double* sum, const GroupPlan& plan, const PipelineOptions& options) {
    const exec::LayerProgram& prog = layer.last_program();
    const std::size_t n = prog.ops.size();
    std::vector<double> mine{static_cast<double>(a.tokens)};
    for (int i = 0; i < 5; ++i) mine.push_back(sum[i] / a.iters);
    for (const auto& t : last.ops) mine.insert(mine.end(), {t.issue_begin, t.issue_end, t.complete});
    const std::vector<std::int64_t>& counts = layer.last_plan().send_counts;
    std::vector<double> all(mine.size() * g_world);
    std::vector<std::int64_t> all_counts(counts.size() * g_world);
    MPI_Gather(mine.data(), static_cast<int>(mine.size()), MPI_DOUBLE, all.data(), static_cast<int>(mine.size()),
               MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(counts.data(), static_cast<int>(counts.size()), MPI_INT64_T, all_counts.data(),
               static_cast<int>(counts.size()), MPI_INT64_T, 0, MPI_COMM_WORLD);
    if (g_rank != 0) return;
    exec::LayerTrace t;
    t.key = exec::ProfileKey{g_world, s.num_experts, s.top_k, s.hidden, s.ffn, a.tokens, "zipf", a.skew,
                             "bf16", "ffn", "shm", a.slots, a.slot_bytes};
    t.groups = plan;
    t.options = options;
    t.iters = a.iters;
    for (int r = 0; r < g_world; ++r) {
        const double* m = all.data() + r * mine.size();
        exec::TraceRank tr;
        tr.tokens = static_cast<int>(m[0]);
        tr.route = m[1];
        tr.counts = m[2];
        tr.pipeline = m[3];
        tr.reduce = m[4];
        tr.total = m[5];
        tr.expert_counts.assign(all_counts.begin() + r * counts.size(), all_counts.begin() + (r + 1) * counts.size());
        t.ranks.push_back(tr);
        for (std::size_t i = 0; i < n; ++i) {
            exec::TraceOp op;
            op.rank = r;
            op.op = static_cast<int>(i);
            op.kind = prog.ops[i].kind;
            op.step = prog.ops[i].step;
            op.issue_begin = m[6 + 3 * i];
            op.issue_end = m[7 + 3 * i];
            op.complete = m[8 + 3 * i];
            t.ops.push_back(op);
        }
    }
    t.save(a.trace);
    std::printf("trace written to %s\n", a.trace.c_str());
}

int run_ffn(const FfnArgs& a) {
    EpShape s{g_world, g_rank, a.experts, a.top_k, a.hidden, a.ffn};
    nano_nccl::ep::validate_shape(s);
    const int epr = s.experts_per_rank();
    MoeLayerConfig cfg{s, a.tokens, 0, ExpertMode::kFfn};
    const std::int64_t max_rows = static_cast<std::int64_t>(a.tokens) * a.top_k * g_world;
    ExpertFfn ffn(g_device, epr, a.hidden, a.ffn, max_rows);
    std::vector<std::vector<bf16>> w13(epr), w2(epr);
    for (int j = 0; j < epr; ++j) {
        const int e = s.first_local_expert() + j;
        w13[j] = random_bf16(static_cast<std::size_t>(a.hidden) * 2 * a.ffn, 1000 + e, 1.0f / std::sqrt(a.hidden));
        w2[j] = random_bf16(static_cast<std::size_t>(a.ffn) * a.hidden, 5000 + e, 1.0f / std::sqrt(a.ffn));
        cuda_check(cudaMemcpy(ffn.w13(j), w13[j].data(), w13[j].size() * sizeof(bf16), cudaMemcpyHostToDevice), "w13");
        cuda_check(cudaMemcpy(ffn.w2(j), w2[j].data(), w2[j].size() * sizeof(bf16), cudaMemcpyHostToDevice), "w2");
    }
    MoeLayer layer(*g_exchanger, cfg, &ffn, g_device);

    const auto x = random_bf16(static_cast<std::size_t>(a.tokens) * a.hidden, 77 + g_rank, 1.0f);
    const Routing routing = make_routing(s, a.tokens, a.skew, -1, nano_nccl::ep::benchmark_routing_seed(g_rank));
    DeviceArray<bf16> dx(x.size()), dout(x.size());
    DeviceArray<std::int32_t> dids(routing.ids.size());
    DeviceArray<float> dw(routing.weights.size());
    dx.upload(x);
    dids.upload(routing.ids);
    dw.upload(routing.weights);
    cudaStream_t stream;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");

    const GroupPlan plan = nano_nccl::planner::fixed_group_plan(g_world, a.experts, a.groups);
    const PipelineOptions options{a.overlap, a.lookahead};
    constexpr int kStages = 5;
    double sum[kStages] = {};
    ForwardTimes last;
    const int warmup = 2;
    std::uint64_t epoch = 1;
    for (int it = 0; it < warmup + a.iters; ++it) {
        MPI_Barrier(MPI_COMM_WORLD);
        const ForwardTimes t =
            layer.forward(epoch++, dx.ptr, a.tokens, dids.ptr, dw.ptr, dout.ptr, plan, options, stream);
        if (it < warmup) continue;
        const double v[kStages] = {t.route, t.counts, t.pipeline, t.reduce, t.total};
        for (int i = 0; i < kStages; ++i) sum[i] += v[i];
        last = t;
    }
    if (!a.trace.empty()) write_trace(a, s, layer, last, sum, plan, options);
    double slowest[kStages];
    MPI_Reduce(sum, slowest, kStages, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    std::int64_t recv_rows = layer.last_plan().recv_rows, max_recv = 0, min_recv = 0;
    MPI_Reduce(&recv_rows, &max_recv, 1, MPI_INT64_T, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&recv_rows, &min_recv, 1, MPI_INT64_T, MPI_MIN, 0, MPI_COMM_WORLD);

    // Grouping and overlap must not change a single output bit.
    const auto pipelined = dout.download(x.size());
    MPI_Barrier(MPI_COMM_WORLD);
    layer.forward(epoch++, dx.ptr, a.tokens, dids.ptr, dw.ptr, dout.ptr,
                  nano_nccl::planner::fixed_group_plan(g_world, a.experts, 1), PipelineOptions{false, 0}, stream);
    const auto serial = dout.download(x.size());
    int same = std::memcmp(pipelined.data(), serial.data(), x.size() * sizeof(bf16)) == 0 ? 1 : 0, all_same = 0;
    MPI_Allreduce(&same, &all_same, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (g_rank == 0) {
        std::printf("# EP MoE: ranks=%d tokens/rank=%d experts=%d top_k=%d hidden=%d ffn=%d skew=%.2f "
                    "groups=%d overlap=%d lookahead=%d\n",
                    g_world, a.tokens, a.experts, a.top_k, a.hidden, a.ffn, a.skew, a.groups, a.overlap ? 1 : 0,
                    a.lookahead);
        std::printf("# expert rows per rank: min %lld max %lld\n", static_cast<long long>(min_recv),
                    static_cast<long long>(max_recv));
        const char* names[kStages] = {"route", "counts", "pipeline", "reduce", "total"};
        std::printf("# stage times, us (mean over %d iters, slowest rank per stage)\n", a.iters);
        for (int i = 0; i < kStages; ++i) std::printf("  %-9s %10.1f\n", names[i], slowest[i] / a.iters);
        std::printf("# rank 0 step timeline of the last iteration, us since forward start\n");
        std::printf("  %4s %5s %7s %9s %9s %9s %9s %9s %9s\n", "step", "group", "rows", "D_issue", "D_ready",
                    "E_launch", "E_done", "C_issue", "C_ready");
        for (std::size_t i = 0; i < last.steps.size(); ++i) {
            const auto& t = last.steps[i];
            std::printf("  %4zu %5d %7lld %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f\n", i, t.group,
                        static_cast<long long>(t.expert_rows), t.dispatch_issued, t.dispatch_readable,
                        t.expert_launched, t.expert_done, t.combine_issued, t.combine_readable);
        }
        std::printf("bitwise identical to serial: %s\n", all_same ? "yes" : "NO");
    }
    if (!all_same) throw std::runtime_error("pipelined output differs from the serial forward");

    if (!a.dump.empty()) {
        const std::string p = a.dump + "/";
        const std::string r = std::to_string(g_rank);
        const auto& out = pipelined;
        write_file(p + "x_" + r + ".bin", x.data(), x.size() * sizeof(bf16));
        write_file(p + "ids_" + r + ".bin", routing.ids.data(), routing.ids.size() * sizeof(std::int32_t));
        write_file(p + "weights_" + r + ".bin", routing.weights.data(), routing.weights.size() * sizeof(float));
        write_file(p + "out_" + r + ".bin", out.data(), out.size() * sizeof(bf16));
        for (int j = 0; j < epr; ++j) {
            const std::string e = std::to_string(s.first_local_expert() + j);
            write_file(p + "w13_" + e + ".bin", w13[j].data(), w13[j].size() * sizeof(bf16));
            write_file(p + "w2_" + e + ".bin", w2[j].data(), w2[j].size() * sizeof(bf16));
        }
        if (g_rank == 0) {
            const std::string meta = "{\"world\": " + std::to_string(g_world) + ", \"tokens\": " +
                                     std::to_string(a.tokens) + ", \"experts\": " + std::to_string(a.experts) +
                                     ", \"top_k\": " + std::to_string(a.top_k) + ", \"hidden\": " +
                                     std::to_string(a.hidden) + ", \"ffn\": " + std::to_string(a.ffn) + "}\n";
            write_file(p + "meta.json", meta.data(), meta.size());
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (g_rank == 0) std::printf("dumped to %s\n", a.dump.c_str());
    }
    cudaStreamDestroy(stream);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_world);
    int status = 1;
    try {
        MPI_Comm node;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
        int local_rank = 0;
        MPI_Comm_rank(node, &local_rank);
        MPI_Comm_free(&node);
        int devices = 0;
        cuda_check(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
        g_device = local_rank % devices;
        cuda_check(cudaSetDevice(g_device), "cudaSetDevice");

        bool ffn_mode = false;
        int slots = 4;
        int copy_delay_us = 0;
        std::size_t slot_bytes = std::size_t{1} << 20;
        FfnArgs a;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto next = [&]() -> const char* {
                if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
                return argv[++i];
            };
            if (arg == "--ffn") ffn_mode = true;
            else if (arg == "--slots") slots = std::atoi(next());
            else if (arg == "--slot-bytes") slot_bytes = std::strtoull(next(), nullptr, 0);
            else if (arg == "--copy-delay-us") copy_delay_us = std::atoi(next());
            else if (arg == "--tokens") a.tokens = std::atoi(next());
            else if (arg == "--experts") a.experts = std::atoi(next());
            else if (arg == "--topk") a.top_k = std::atoi(next());
            else if (arg == "--hidden") a.hidden = std::atoi(next());
            else if (arg == "--ffn-dim") a.ffn = std::atoi(next());
            else if (arg == "--iters") a.iters = std::atoi(next());
            else if (arg == "--skew") a.skew = std::atof(next());
            else if (arg == "--dump") a.dump = next();
            else if (arg == "--groups") a.groups = std::atoi(next());
            else if (arg == "--lookahead") a.lookahead = std::atoi(next());
            else if (arg == "--no-overlap") a.overlap = false;
            else if (arg == "--trace") a.trace = next();
            else throw std::invalid_argument("unknown argument " + arg);
        }
        a.slots = slots;
        a.slot_bytes = slot_bytes;
        ShmWire wire(MPI_COMM_WORLD, {slots, slot_bytes, g_device, copy_delay_us});
        g_exchanger = std::make_unique<PeerExchanger>(&wire);
        status = ffn_mode ? run_ffn(a) : run_probe();
        MPI_Barrier(MPI_COMM_WORLD);  // peers may still read this rank's region
        g_exchanger.reset();
    } catch (const std::exception& error) {
        std::printf("rank %d error: %s\n", g_rank, error.what());
        std::fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
