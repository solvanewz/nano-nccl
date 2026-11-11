// EP MoE layer over ShmWire or RdmaWire (B1), one process per rank under mpirun.
//
//   ep_moe_test [--wire shm|rdma] [--slots S] [--slot-bytes B] [--copy-delay-us D]
//       probe matrix: experts scale rows by powers of two, so every output
//       is checked bit-exactly against a host reference. Covers skewed
//       routing, ranks with no tokens, all tokens routed to one rank's
//       experts, unequal token counts, and several forwards per layer, each
//       under several group plans and pipeline schedules (serial, fixed
//       groups with and without overlap, all dispatches ahead, arbitrary
//       expert->group mappings in arbitrary order with and without overlap,
//       and an empty group followed by one group per rank's experts, i.e.
//       the plan shapes a B2 scheduler may produce), plus plans computed from
//       each forward's counts (exec::PlanTiming::kFromCounts: token_greedy,
//       cost_aware with a communication-heavy synthetic cost model and no
//       fallback, and fixed through the same program). D slows every
//       ShmWire copy (ShmWireConfig::debug_copy_delay_us), so data used
//       before it is readable shows up as wrong output.
//   ep_moe_test --ffn [--tokens T --experts E --topk K --hidden H --ffn-dim F
//                      --iters N --skew X --groups G --lookahead L
//                      --no-overlap --scheduler NAME --plan-from-counts
//                      --random-plan SEED --workload NAME --dump DIR --trace FILE
//                      --cost-profile P --fallback-permille N
//                      --cost-link GBPS[:EXCHANGE_US]]
//       real BF16 SwiGLU experts; the group plan comes from the scheduler
//       (planner::make_schedule, default fixed = B1) with G groups, computed
//       from each forward's counts when the scheduler reads them or with
//       --plan-from-counts (exec::plan_source_for), or with --random-plan
//       from planner::random_group_plan over at most G groups. cost_aware
//       needs --cost-profile (an ep_profile/calibrated profile): its model is
//       sim::planner_cost_model over the profile's host-staged copy rates,
//       the same one nano_ep_sim builds, so a replay reproduces the plan;
//       --cost-link GBPS[:EXCHANGE_US[:MESSAGE_US]] replaces the copy rates
//       by sim::link_rates (e.g. a fit of the measured RdmaWire rate), and
//       --cost-local GBPS[:MESSAGE_US] (--wire hybrid) adds the same-node
//       links as a second fabric (sim::with_local_link). --trace - prints
//       the trace on stdout between ==TRACE_BEGIN== / ==TRACE_END==.
//       Prints stage times and the step timeline, and checks that the output
//       is bit-identical to a serial (one group, no overlap) forward. With
//       --dump, writes inputs, weights and outputs for
//       tests/ep_moe_reference.py. --warmup W untimed forwards (default 2);
//       --iter-stats prints one "iter," line per timed forward and rank
//       (stage times, exposed communication wait; see iter_stats()).
//
// --wire shm (default): ShmWire, all ranks on one host. --wire rdma
// (RDMA builds): RdmaWire with the CUDA copier, ranks on any hosts; the HCA
// comes from NANO_NCCL_RDMA_IFNAME / NANO_NCCL_RDMA_GID_INDEX. --wire hybrid
// (RDMA builds): HybridWire, same-node links --local nvlink (default; fails
// unless every pair of the node's GPUs has NVLink P2P) or host, other links
// --remote gdr (default; GPUDirect RDMA) or host. Each rank uses GPU
// (local rank % visible GPUs).
//
//   ep_moe_test --exchange-bench [--max-bytes M] (with any --wire)
//       uniform exchange (every rank to every rank, B bytes per peer, B from
//       256 KiB to M by 4x): slowest rank's time per exchange, mean of 20
//       after 5 warmup; the same table as peer_rdma_wire_test --bench.
#include "ep/moe_layer.h"
#include "ep/synthetic_routing.h"
#include "exec/exec_trace.h"
#include "cost_profile.h"
#include "planner/scheduler.h"
#include "transport/peer/peer_exchanger.h"
#include "transport/peer/shm_wire.h"
#ifdef NANO_NCCL_EP_RDMA_WIRE
#include "transport/peer/hybrid_wire.h"
#include "transport/peer/rdma_wire.h"
#endif

#include <chrono>

#include <unistd.h>

#include <algorithm>
#include <utility>
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
using nano_nccl::transport::peer::ShmWireConfig;
namespace exec = nano_nccl::exec;

int g_rank = 0;
int g_world = 0;
int g_device = 0;
int g_ranks_per_node = 0;  // HybridWire with equal, contiguous nodes; else 0

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

enum class PlanKind {
    kFixed,      // fixed_group_plan
    kScrambled,  // arbitrary expert -> group mapping, groups run in reverse
    kRandom,     // planner::random_group_plan: may have empty and one-rank groups
    kRankMajor,  // an empty group, then one group per rank's experts, last rank first
    kGreedy,     // token_greedy from each forward's counts; groups 0: num_experts / 2
    kFixedFromCounts,  // fixed (groups 0: experts per rank) through the kFromCounts program
    kCostAware,  // cost_aware from each forward's counts (probe_cost_model); groups 0: num_experts / 2
};

struct Variant {
    const char* name;
    int groups;      // kFixed: 0 = one group per local expert; kScrambled: group count
    bool overlap;
    int lookahead;
    PlanKind kind;
    std::uint64_t seed;  // kRandom
};

constexpr Variant kVariants[] = {
    {"serial", 1, false, 0, PlanKind::kFixed, 0},
    {"g2_overlap", 2, true, 1, PlanKind::kFixed, 0},
    {"gmax_overlap", 0, true, 1, PlanKind::kFixed, 0},
    {"gmax_no_overlap", 0, false, 0, PlanKind::kFixed, 0},
    {"gmax_all_ahead", 0, true, 64, PlanKind::kFixed, 0},
    {"scrambled_l0", 3, true, 0, PlanKind::kScrambled, 0},
    {"random_l1", 0, true, 1, PlanKind::kRandom, 1},
    {"random_l2", 0, true, 2, PlanKind::kRandom, 2},
    {"random_no_overlap", 0, false, 0, PlanKind::kRandom, 3},
    {"rank_major_all_ahead", 0, true, 64, PlanKind::kRankMajor, 0},
    {"greedy_l2", 0, true, 2, PlanKind::kGreedy, 0},
    {"greedy_no_overlap", 3, false, 0, PlanKind::kGreedy, 0},
    {"fixed_from_counts_l1", 0, true, 1, PlanKind::kFixedFromCounts, 0},
    {"aware_l2", 0, true, 2, PlanKind::kCostAware, 0},
    {"aware_no_overlap", 3, false, 0, PlanKind::kCostAware, 0},
};

// cost_aware model for the probe runs: traffic dominates and there is no
// fallback, so the search moves experts and reorders steps (a data-path
// test, not a performance model).
nano_nccl::planner::CostModel probe_cost_model(const EpShape& s) {
    namespace pl = nano_nccl::planner;
    pl::CostModel m;
    m.row_bytes = static_cast<std::int64_t>(s.hidden) * 2;
    m.expert_ns = pl::IntTable({{1, 10000}, {1024, 30000}});
    m.permute_ns = pl::IntTable({{1, 1000}});
    m.expert_issue_ns = pl::IntTable({{0, 1000}});
    m.tx_ps_per_byte = m.rx_ps_per_byte = 2000;
    m.message_ns = 1000;
    m.exchange_ns = 1000;
    m.fallback_permille = 0;
    return m;
}

GroupPlan make_group_plan(const EpShape& s, const Variant& v);

// Identical on every rank: deterministic in the shape.
exec::PlanSource make_plan_source(const EpShape& s, const Variant& v) {
    namespace pl = nano_nccl::planner;
    const int epr = s.experts_per_rank();
    if (v.kind == PlanKind::kGreedy) {
        const int groups = std::min(v.groups == 0 ? std::max(1, s.num_experts / 2) : v.groups, s.num_experts);
        return exec::PlanSource::from_counts(pl::SchedulerKind::kTokenGreedy, groups);
    }
    if (v.kind == PlanKind::kCostAware) {
        auto src = exec::PlanSource::from_counts(
            pl::SchedulerKind::kCostAware,
            std::min(v.groups == 0 ? std::max(1, s.num_experts / 2) : v.groups, s.num_experts));
        src.cost = std::make_shared<const pl::CostModel>(probe_cost_model(s));
        return src;
    }
    if (v.kind == PlanKind::kFixedFromCounts) {
        return exec::PlanSource::from_counts(pl::SchedulerKind::kFixed, v.groups == 0 ? epr : v.groups);
    }
    return exec::PlanSource::ahead(make_group_plan(s, v));
}

GroupPlan make_group_plan(const EpShape& s, const Variant& v) {
    const int epr = s.experts_per_rank();
    switch (v.kind) {
        case PlanKind::kFixed: {
            const int groups = std::min(v.groups == 0 ? epr : v.groups, epr);
            return nano_nccl::planner::fixed_group_plan(s.world, s.num_experts, groups);
        }
        case PlanKind::kScrambled: {
            const int groups = std::min(v.groups, s.num_experts);
            std::mt19937_64 rng(static_cast<std::uint64_t>(s.num_experts) * 31 + s.top_k);
            GroupPlan plan;
            for (int e = 0; e < s.num_experts; ++e) plan.expert_group.push_back(static_cast<int>(rng() % groups));
            for (int g = groups - 1; g >= 0; --g) plan.order.push_back(g);
            return plan;
        }
        case PlanKind::kRandom:
            return nano_nccl::planner::random_group_plan(
                s.num_experts, s.num_experts + 1,
                v.seed * 1000003ull + static_cast<std::uint64_t>(s.num_experts) * 31 + s.top_k);
        case PlanKind::kRankMajor: {
            GroupPlan plan;  // group r: rank r's experts; group W: empty
            for (int e = 0; e < s.num_experts; ++e) plan.expert_group.push_back(s.owner(e));
            plan.order.push_back(s.world);
            for (int r = s.world - 1; r >= 0; --r) plan.order.push_back(r);
            return plan;
        }
        case PlanKind::kGreedy:
        case PlanKind::kFixedFromCounts:
        case PlanKind::kCostAware:
            break;
    }
    throw std::logic_error("make_group_plan: not an ahead plan kind");
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
    const exec::PlanSource plan = make_plan_source(s, variant);
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
    std::string scheduler = "fixed";
    std::string workload = "zipf";  // ep::Workload name
    bool from_counts = false;
    long long random_plan = -1;  // >= 0: planner::random_group_plan seed
    std::string dump;
    std::string trace;  // exec::LayerTrace of the timed runs, written by rank 0
    int slots = 0;
    std::size_t slot_bytes = 0;
    std::string cost_profile;     // cost_aware: profile for its cost model
    int fallback_permille = -1;   // cost_aware: override the model's default
    double cost_link_gbps = 0;        // cost_aware: > 0 models the fabric as this link
    double cost_link_exchange_us = 0;  // ... with this fixed cost per exchange
    double cost_link_message_us = 0;   // ... and this cost per message
    double cost_local_gbps = 0;        // cost_aware: > 0 adds a local fabric within nodes (hybrid wire)
    double cost_local_message_us = 0;
    int warmup = 2;
    bool iter_stats = false;
    std::string wire = "shm";
};

// Measure of the union of [begin, end) intervals.
double union_length(std::vector<std::pair<double, double>> v) {
    std::sort(v.begin(), v.end());
    double total = 0, lo = 0, hi = -1;
    for (const auto& [b, e] : v) {
        if (e <= b) continue;
        if (b > hi) {
            if (hi > lo) total += hi - lo;
            lo = b;
            hi = e;
        } else {
            hi = std::max(hi, e);
        }
    }
    if (hi > lo) total += hi - lo;
    return total;
}

// Measure of union(a) minus union(b).
double exposed_length(const std::vector<std::pair<double, double>>& a, std::vector<std::pair<double, double>> b) {
    std::vector<std::pair<double, double>> both = a;
    both.insert(both.end(), b.begin(), b.end());
    return union_length(both) - union_length(std::move(b));
}

constexpr int kIterFields = 12;
const char* const kIterNames[kIterFields] = {"total", "route", "counts", "plan", "pack", "pipeline", "reduce",
                                             "comm_wait", "dispatch", "combine", "expert_busy", "recv_rows"};

// One forward on this rank, in us. comm_wait: time during which at least one
// exchange (counts, dispatch, combine; issue .. consumer readable) is in
// flight and no compute is (stream ops: route, pack, expert, reduce, issue ..
// observed done; host plan op), i.e. communication not hidden behind work.
// dispatch / combine: sums of their exchanges' issue .. readable times;
// expert_busy: union of the expert ops.
std::vector<double> iter_stats(const ForwardTimes& t, const exec::LayerProgram& prog, std::int64_t recv_rows) {
    std::vector<std::pair<double, double>> comm, work, expert;
    double dispatch = 0, combine = 0;
    for (std::size_t i = 0; i < prog.ops.size() && i < t.ops.size(); ++i) {
        const auto& o = t.ops[i];
        if (o.issue_begin < 0) continue;
        const double end = o.complete >= 0 ? o.complete : o.issue_end;
        if (end < 0) continue;
        const exec::OpKind kind = prog.ops[i].kind;
        if (prog.ops[i].queue == exec::Queue::kExchange) {
            comm.emplace_back(o.issue_begin, end);
            if (kind == exec::OpKind::kDispatch) dispatch += end - o.issue_begin;
            if (kind == exec::OpKind::kCombine) combine += end - o.issue_begin;
        } else if (kind != exec::OpKind::kRetire) {
            work.emplace_back(o.issue_begin, end);
            if (kind == exec::OpKind::kExpert) expert.emplace_back(o.issue_begin, end);
        }
    }
    return {t.total, t.route, t.counts, t.plan, t.pack, t.pipeline, t.reduce, exposed_length(comm, work),
            dispatch, combine, union_length(expert), static_cast<double>(recv_rows)};
}

// Gathers every rank's counts, stage means and last-iteration op times.
void write_trace(const FfnArgs& a, const EpShape& s, const MoeLayer& layer, const ForwardTimes& last,
                 const double* sum, const exec::PlanSource& source, const PipelineOptions& options) {
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
    t.key = exec::ProfileKey{g_world, s.num_experts, s.top_k, s.hidden, s.ffn, a.tokens, nano_nccl::ep::parse_workload(a.workload, a.skew).name(), a.skew,
                             "bf16", "ffn", a.wire, a.slots, a.slot_bytes};
    t.groups = layer.last_ep_plan().groups;  // the plan that ran
    t.timing = source.timing;
    t.scheduler = source.scheduler;
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
    if (a.trace == "-") {  // on stdout, for launchers on another host than rank 0
        std::printf("==TRACE_BEGIN==\n%s==TRACE_END==\n", t.serialize().c_str());
        return;
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
    const Routing routing =
        make_routing(s, a.tokens, nano_nccl::ep::source_expert_weights(nano_nccl::ep::parse_workload(a.workload, a.skew), s),
                     nano_nccl::ep::benchmark_routing_seed(g_rank));
    DeviceArray<bf16> dx(x.size()), dout(x.size());
    DeviceArray<std::int32_t> dids(routing.ids.size());
    DeviceArray<float> dw(routing.weights.size());
    dx.upload(x);
    dids.upload(routing.ids);
    dw.upload(routing.weights);
    cudaStream_t stream;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");

    namespace pl = nano_nccl::planner;
    exec::PlanSource plan;
    std::string plan_name;
    if (a.random_plan >= 0) {
        plan = exec::PlanSource::ahead(
            pl::random_group_plan(a.experts, a.groups, static_cast<std::uint64_t>(a.random_plan)));
        plan_name = "random:" + std::to_string(a.random_plan);
    } else {
        plan = exec::plan_source_for(pl::parse_scheduler(a.scheduler), g_world, a.experts, a.groups, a.from_counts);
        plan_name = a.scheduler;
        if (plan.scheduler == pl::SchedulerKind::kCostAware) {
            if (a.cost_profile.empty()) throw std::invalid_argument("--scheduler cost_aware needs --cost-profile");
            const auto prof = nano_nccl::sim::CostProfile::load(a.cost_profile);
            // Fabric: the profile's host-staged copies (ShmWire), or a measured
            // network link (--cost-link, e.g. RdmaWire between hosts).
            auto fabric = a.cost_link_gbps > 0
                              ? nano_nccl::sim::link_rates(a.cost_link_gbps, a.cost_link_message_us,
                                                           g_exchanger->slot_bytes(), a.cost_link_exchange_us)
                              : nano_nccl::sim::copy_rates(prof, g_exchanger->slot_bytes());
            // --cost-local: same-node links (HybridWire) as a second fabric.
            if (a.cost_local_gbps > 0) {
                if (g_ranks_per_node <= 1) {
                    throw std::invalid_argument("--cost-local needs --wire hybrid with equal, contiguous nodes");
                }
                fabric = nano_nccl::sim::with_local_link(fabric, g_ranks_per_node, a.cost_local_gbps,
                                                         a.cost_local_message_us);
            }
            auto m = nano_nccl::sim::planner_cost_model(prof, a.hidden, 1.0, fabric);
            if (g_rank == 0) {
                std::printf("# cost_model network %.1f ps/B, %lld ns/msg, %lld ns/exchange; local %d ranks/node "
                            "%lld ps/B, %lld ns/msg\n",
                            static_cast<double>(m.tx_ps_per_byte), static_cast<long long>(m.message_ns),
                            static_cast<long long>(m.exchange_ns), m.ranks_per_node,
                            static_cast<long long>(m.local_tx_ps_per_byte), static_cast<long long>(m.local_message_ns));
            }
            if (a.fallback_permille >= 0) m.fallback_permille = a.fallback_permille;
            plan.cost = std::make_shared<const pl::CostModel>(m);
        }
    }
    plan_name += std::string(" (") + exec::plan_timing_name(plan.timing) + ")";
    const PipelineOptions options{a.overlap, a.lookahead};
    // The first five are the trace's stage means.
    constexpr int kStages = 7;
    double sum[kStages] = {};
    ForwardTimes last;
    const int warmup = a.warmup;
    std::uint64_t epoch = 1;
    std::vector<double> iter_rows;  // rank 0: [iters][world][kIterFields]
    if (a.iter_stats && g_rank == 0) iter_rows.resize(static_cast<std::size_t>(a.iters) * g_world * kIterFields);
    for (int it = 0; it < warmup + a.iters; ++it) {
        MPI_Barrier(MPI_COMM_WORLD);
        const ForwardTimes t =
            layer.forward(epoch++, dx.ptr, a.tokens, dids.ptr, dw.ptr, dout.ptr, plan, options, stream);
        if (it < warmup) continue;
        if (a.iter_stats) {
            const auto mine = iter_stats(t, layer.last_program(), layer.last_plan().recv_rows);
            double* dst = g_rank == 0 ? iter_rows.data() + static_cast<std::size_t>(it - warmup) * g_world * kIterFields
                                      : nullptr;
            MPI_Gather(mine.data(), kIterFields, MPI_DOUBLE, dst, kIterFields, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        const double v[kStages] = {t.route, t.counts, t.pipeline, t.reduce, t.total, t.plan, t.pack};
        for (int i = 0; i < kStages; ++i) sum[i] += v[i];
        last = t;
    }
    if (!a.trace.empty()) write_trace(a, s, layer, last, sum, plan, options);
    double slowest[kStages];
    MPI_Reduce(sum, slowest, kStages, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    std::int64_t recv_rows = layer.last_plan().recv_rows, max_recv = 0, min_recv = 0;
    MPI_Reduce(&recv_rows, &max_recv, 1, MPI_INT64_T, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&recv_rows, &min_recv, 1, MPI_INT64_T, MPI_MIN, 0, MPI_COMM_WORLD);

    const GroupPlan ran = layer.last_ep_plan().groups;
    // Grouping and overlap must not change a single output bit.
    const auto pipelined = dout.download(x.size());
    MPI_Barrier(MPI_COMM_WORLD);
    layer.forward(epoch++, dx.ptr, a.tokens, dids.ptr, dw.ptr, dout.ptr,
                  nano_nccl::planner::fixed_group_plan(g_world, a.experts, 1), PipelineOptions{false, 0}, stream);
    const auto serial = dout.download(x.size());
    int same = std::memcmp(pipelined.data(), serial.data(), x.size() * sizeof(bf16)) == 0 ? 1 : 0, all_same = 0;
    MPI_Allreduce(&same, &all_same, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (g_rank == 0) {
        std::printf("# workload %s\n", a.workload.c_str());
        std::printf("# EP MoE: ranks=%d tokens/rank=%d experts=%d top_k=%d hidden=%d ffn=%d skew=%.2f "
                    "plan=%s groups=%d overlap=%d lookahead=%d\n",
                    g_world, a.tokens, a.experts, a.top_k, a.hidden, a.ffn, a.skew, plan_name.c_str(),
                    ran.groups(), a.overlap ? 1 : 0, a.lookahead);
        std::printf("# expert -> group:");
        for (int g : ran.expert_group) std::printf(" %d", g);
        std::printf("; order:");
        for (int g : ran.order) std::printf(" %d", g);
        std::printf("\n");
        std::printf("# expert rows per rank: min %lld max %lld\n", static_cast<long long>(min_recv),
                    static_cast<long long>(max_recv));
        const char* names[kStages] = {"route", "counts", "pipeline", "reduce", "total", "plan", "pack"};
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
        if (a.iter_stats) {
            std::printf("iter_header,iter,rank");
            for (const char* n : kIterNames) std::printf(",%s", n);
            std::printf("\n");
            for (int it = 0; it < a.iters; ++it) {
                for (int r = 0; r < g_world; ++r) {
                    const double* v = iter_rows.data() + (static_cast<std::size_t>(it) * g_world + r) * kIterFields;
                    std::printf("iter,%d,%d", it, r);
                    for (int f = 0; f < kIterFields; ++f) std::printf(",%.2f", v[f]);
                    std::printf("\n");
                }
            }
        }
        std::fflush(stdout);
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

// Uniform exchange over the selected wire (device buffers).
int run_exchange_bench(std::size_t max_bytes) {
    namespace tp = nano_nccl::transport::peer;
    using Clock = std::chrono::steady_clock;
    const auto n = static_cast<std::size_t>(g_world);
    DeviceArray<std::byte> send(max_bytes * n), recv(max_bytes * n);
    cuda_check(cudaMemset(send.ptr, 1, max_bytes * n), "memset");
    constexpr int kWarmup = 5, kIters = 20;
    constexpr std::uint32_t kTag = 7;
    PeerExchanger& e = *g_exchanger;
    if (g_rank == 0) {
        std::printf("# uniform exchange: ranks=%d slots=%d slot_bytes=%zu\n", g_world, e.slots_per_link(),
                    e.slot_bytes());
        std::printf("# %12s %12s %14s %14s\n", "bytes/peer", "time_us", "out_GB/s", "remote_GB/s");
    }
    std::uint64_t epoch = 1;
    for (std::size_t bytes = 256 << 10; bytes <= max_bytes; bytes *= 4) {
        const auto layout = tp::uniform_layout(g_world, bytes);
        double total = 0;
        for (int it = 0; it < kWarmup + kIters; ++it) {
            MPI_Barrier(MPI_COMM_WORLD);
            const auto t0 = Clock::now();
            const auto id = e.exchangev_async(epoch++, kTag, layout, send.ptr, recv.ptr);
            while (!(e.query_completion(id, tp::CompletionEvent::kSendBufferReusable) &&
                     e.query_completion(id, tp::CompletionEvent::kReceiveSlotsReleased))) {
                e.progress();
            }
            double elapsed = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
            e.retire(id);
            double slowest = 0;
            MPI_Allreduce(&elapsed, &slowest, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (it >= kWarmup) total += slowest;
        }
        const double us = total / kIters;
        if (g_rank == 0) {
            std::printf("  %12zu %12.2f %14.2f %14.2f\n", bytes, us, bytes * n / us / 1e3, bytes * (n - 1) / us / 1e3);
        }
    }
    if (g_rank == 0) std::printf("exchange_bench=DONE\n");
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
            else if (arg == "--scheduler") a.scheduler = next();
            else if (arg == "--workload") a.workload = next();
            else if (arg == "--plan-from-counts") a.from_counts = true;
            else if (arg == "--random-plan") a.random_plan = std::atoll(next());
            else if (arg == "--trace") a.trace = next();
            else if (arg == "--cost-profile") a.cost_profile = next();
            else if (arg == "--fallback-permille") a.fallback_permille = std::atoi(next());
            else if (arg == "--cost-link") {
                const std::string v = next();  // GBPS[:EXCHANGE_US[:MESSAGE_US]]
                a.cost_link_gbps = std::atof(v.c_str());
                const auto c1 = v.find(':');
                if (c1 != std::string::npos) {
                    a.cost_link_exchange_us = std::atof(v.c_str() + c1 + 1);
                    const auto c2 = v.find(':', c1 + 1);
                    if (c2 != std::string::npos) a.cost_link_message_us = std::atof(v.c_str() + c2 + 1);
                }
                if (!(a.cost_link_gbps > 0)) throw std::invalid_argument("--cost-link needs GBPS > 0");
            } else if (arg == "--cost-local") {
                const std::string v = next();  // GBPS[:MESSAGE_US]
                a.cost_local_gbps = std::atof(v.c_str());
                const auto c1 = v.find(':');
                if (c1 != std::string::npos) a.cost_local_message_us = std::atof(v.c_str() + c1 + 1);
                if (!(a.cost_local_gbps > 0)) throw std::invalid_argument("--cost-local needs GBPS > 0");
            }
            else if (arg == "--warmup") a.warmup = std::atoi(next());
            else if (arg == "--iter-stats") a.iter_stats = true;
            else if (arg == "--wire") a.wire = next();
            else if (arg == "--local" || arg == "--remote" || arg == "--max-bytes") next();
            else if (arg == "--exchange-bench") {}
            else throw std::invalid_argument("unknown argument " + arg);
        }
        a.slots = slots;
        a.slot_bytes = slot_bytes;
        std::unique_ptr<nano_nccl::transport::peer::PeerWire> wire;
        std::string wire_device = "shm";
        std::string local_kind = "nvlink", remote_kind = "gdr";
        bool exchange_bench = false;
        std::size_t bench_max_bytes = std::size_t{64} << 20;
        for (int i = 1; i < argc; ++i) {  // options of this block only (parsed above as well)
            const std::string arg = argv[i];
            if (arg == "--local" && i + 1 < argc) local_kind = argv[++i];
            else if (arg == "--remote" && i + 1 < argc) remote_kind = argv[++i];
            else if (arg == "--exchange-bench") exchange_bench = true;
            else if (arg == "--max-bytes" && i + 1 < argc) bench_max_bytes = std::strtoull(argv[++i], nullptr, 0);
        }
        if (a.wire == "shm") {
            wire = std::make_unique<ShmWire>(MPI_COMM_WORLD, ShmWireConfig{slots, slot_bytes, g_device, copy_delay_us});
        } else if (a.wire == "rdma") {
#ifdef NANO_NCCL_EP_RDMA_WIRE
            namespace tp = nano_nccl::transport::peer;
            auto rdma = std::make_unique<tp::RdmaWire>(MPI_COMM_WORLD, tp::RdmaWireConfig{slots, slot_bytes, 2},
                                                       tp::make_cuda_copier(g_device, copy_delay_us));
            wire_device = rdma->device_name() + "/" + rdma->copier_name();
            wire = std::move(rdma);
#else
            throw std::invalid_argument("--wire rdma needs an RDMA build (NANO_NCCL_ENABLE_RDMA=ON)");
#endif
        } else if (a.wire == "hybrid") {
#ifdef NANO_NCCL_EP_RDMA_WIRE
            namespace tp = nano_nccl::transport::peer;
            tp::HybridWireConfig hc;
            hc.slots_per_link = slots;
            hc.slot_bytes = slot_bytes;
            hc.device = g_device;
            hc.debug_copy_delay_us = copy_delay_us;
            if (local_kind == "nvlink") hc.local = tp::HybridWireConfig::Local::kNvlink;
            else if (local_kind == "host") hc.local = tp::HybridWireConfig::Local::kHost;
            else throw std::invalid_argument("--local must be nvlink or host");
            if (remote_kind == "gdr") hc.gpu_direct = true;
            else if (remote_kind == "host") hc.gpu_direct = false;
            else throw std::invalid_argument("--remote must be gdr or host");
            auto hybrid = std::make_unique<tp::HybridWire>(MPI_COMM_WORLD, hc);
            wire_device = hybrid->description();
            g_ranks_per_node = hybrid->ranks_per_node();
            wire = std::move(hybrid);
#else
            throw std::invalid_argument("--wire hybrid needs an RDMA build (NANO_NCCL_ENABLE_RDMA=ON)");
#endif
        } else {
            throw std::invalid_argument("--wire must be shm, rdma or hybrid");
        }
        {
            // Where every rank runs, as the backend actually in use.
            char host[256] = {};
            gethostname(host, sizeof(host) - 1);
            cudaDeviceProp prop{};
            cuda_check(cudaGetDeviceProperties(&prop, g_device), "cudaGetDeviceProperties");
            char line[512];
            std::snprintf(line, sizeof(line), "rank %d host %s local_rank %d device %d (%s, sm_%d%d) wire %s %s",
                          g_rank, host, local_rank, g_device, prop.name, prop.major, prop.minor, a.wire.c_str(),
                          wire_device.c_str());
            std::vector<char> all(g_rank == 0 ? sizeof(line) * g_world : 1);
            MPI_Gather(line, sizeof(line), MPI_CHAR, all.data(), sizeof(line), MPI_CHAR, 0, MPI_COMM_WORLD);
            if (g_rank == 0) {
                for (int r = 0; r < g_world; ++r) std::printf("# %s\n", all.data() + r * sizeof(line));
                std::fflush(stdout);
            }
        }
        g_exchanger = std::make_unique<PeerExchanger>(wire.get());
        status = exchange_bench ? run_exchange_bench(bench_max_bytes) : ffn_mode ? run_ffn(a) : run_probe();
#ifdef NANO_NCCL_EP_RDMA_WIRE
        // No write may still be headed for a peer that tears down, and no
        // peer may still map device slots that their owner frees.
        if (auto* rdma = dynamic_cast<nano_nccl::transport::peer::RdmaWire*>(wire.get())) rdma->drain();
        if (auto* hybrid = dynamic_cast<nano_nccl::transport::peer::HybridWire*>(wire.get())) hybrid->finish();
#endif
        if (auto* shm = dynamic_cast<ShmWire*>(wire.get())) shm->finish();
        MPI_Barrier(MPI_COMM_WORLD);  // peers may still read this rank's region
        g_exchanger.reset();
        wire.reset();
    } catch (const std::exception& error) {
        std::printf("rank %d error: %s\n", g_rank, error.what());
        std::fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
