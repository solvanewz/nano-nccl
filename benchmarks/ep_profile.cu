// Collects the A40 cost profile that drives the EP simulator (sim_adapter).
//
//   nano_nccl_ep_profile --out FILE [--world W --experts E --topk K --hidden H
//                        --ffn-dim F --tokens T --slots S --slot-bytes B --quick
//                        --host-memory shm|private]
//
// One process, GPUs 0 and 1 (GPU 0 twice if only one exists). Peer access is
// never enabled: inter-GPU traffic is staged through pinned host memory
// exactly as ShmWire does it (a sender's GPU writes into a host region placed
// on the receiver GPU's NUMA node; the receiver copies it in).
//
// Measured (see sim_adapter/cost_profile.h for the names):
//   GPU op tables   device time of route / permute / expert FFN / probe /
//                   reduce, with launches queued behind a spin kernel so host
//                   enqueue time is excluded;
//   copy tables     per-chunk time of each copy-engine role, with all four
//                   directions (2 GPUs x tx/rx) busy and alone, through host
//                   regions allocated like ShmWire's (POSIX shm, rotating
//                   slots; --host-memory private for comparison);
//   interference    copy slowdown while both GPUs compute, expert slowdown
//                   while every copy engine is busy;
//   host scalars    wall time of the issuing calls, idle-stream launch
//                   latencies, one exchanger issue.
#include "cost_profile.h"
#include "core/numa.h"
#include "ep/ep_kernels.h"
#include "ep/expert_ffn.h"
#include "ep/synthetic_routing.h"
#include "exec/exec_program.h"
#include "planner/group_plan.h"
#include "transport/peer/peer_exchanger.h"
#include "transport/peer/peer_wire.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

using nano_nccl::ep::bf16;
using nano_nccl::ep::EpShape;
using nano_nccl::sim::CostProfile;
using nano_nccl::sim::Table;
namespace cost = nano_nccl::sim::cost;
using Clock = std::chrono::steady_clock;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

double wall_us(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
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

__global__ void empty_kernel() {}

struct Gpu {
    int device = 0;
    double clock_khz = 1e6;
    cudaStream_t compute = nullptr, tx = nullptr, rx = nullptr;

    void init(int d) {
        device = d;
        check(cudaSetDevice(d), "cudaSetDevice");
        int khz = 0;
        check(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, d), "clock rate");
        clock_khz = khz;
        check(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking), "stream");
        check(cudaStreamCreateWithFlags(&tx, cudaStreamNonBlocking), "stream");
        check(cudaStreamCreateWithFlags(&rx, cudaStreamNonBlocking), "stream");
    }
    // Holds `stream` busy for about `us` so later work queues up.
    void spin(cudaStream_t stream, double us) const {
        spin_kernel<<<1, 1, 0, stream>>>(static_cast<long long>(us * clock_khz / 1000.0));
        check(cudaGetLastError(), "spin");
    }
};

// Device time per call of `enqueue` (reps calls queued behind a spin).
double device_us(const Gpu& g, cudaStream_t stream, int reps, const std::function<void()>& enqueue) {
    check(cudaSetDevice(g.device), "cudaSetDevice");
    cudaEvent_t a, b;
    check(cudaEventCreate(&a), "event");
    check(cudaEventCreate(&b), "event");
    enqueue();  // warm-up
    check(cudaStreamSynchronize(stream), "sync");
    std::vector<double> samples;
    for (int trial = 0; trial < 3; ++trial) {
        g.spin(stream, 3000);
        check(cudaEventRecord(a, stream), "record");
        for (int i = 0; i < reps; ++i) enqueue();
        check(cudaEventRecord(b, stream), "record");
        check(cudaEventSynchronize(b), "sync");
        float ms = 0;
        check(cudaEventElapsedTime(&ms, a, b), "elapsed");
        samples.push_back(1000.0 * ms / reps);
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return median(samples);
}

// Host wall time per call of `issue`, with the stream held busy so calls
// never wait for the device.
double host_us(const Gpu& g, cudaStream_t stream, int reps, const std::function<void()>& issue) {
    check(cudaSetDevice(g.device), "cudaSetDevice");
    issue();
    check(cudaStreamSynchronize(stream), "sync");
    std::vector<double> samples;
    for (int trial = 0; trial < 5; ++trial) {
        g.spin(stream, 20000);
        for (int i = 0; i < reps; ++i) {
            const auto t0 = Clock::now();
            issue();
            samples.push_back(wall_us(t0, Clock::now()));
        }
        check(cudaStreamSynchronize(stream), "sync");
    }
    return median(samples);
}

// Pinned host memory first-touched on `node`. With `shm`, a POSIX shared
// memory mapping like ShmWire's receive regions (MAP_SHARED, unlinked at
// once); otherwise private anonymous memory, which may get huge pages.
struct HostRegion {
    void* ptr = nullptr;
    std::size_t bytes = 0;
    bool shm = false;
    HostRegion(std::size_t n, int node, bool use_shm) : bytes((n + 4095) / 4096 * 4096), shm(use_shm) {
        if (shm) {
            const std::string name = "/nano_ep_profile_" + std::to_string(getpid()) + "_" + std::to_string(node) + "_" +
                                     std::to_string(reinterpret_cast<std::uintptr_t>(this) & 0xffff);
            const int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, S_IRUSR | S_IWUSR);
            if (fd < 0) throw std::runtime_error("shm_open failed");
            shm_unlink(name.c_str());
            if (ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
                close(fd);
                throw std::runtime_error("ftruncate failed");
            }
            ptr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            close(fd);
            if (ptr == MAP_FAILED) throw std::runtime_error("mmap failed");
        } else {
            ptr = std::aligned_alloc(4096, bytes);
            if (ptr == nullptr) throw std::bad_alloc();
        }
        nano_nccl::core::numa_set_prefer(node);
        std::memset(ptr, 0, bytes);
        nano_nccl::core::numa_set_prefer(-1);
        check(cudaHostRegister(ptr, bytes, cudaHostRegisterDefault), "cudaHostRegister");
    }
    ~HostRegion() {
        cudaHostUnregister(ptr);
        if (shm) {
            munmap(ptr, bytes);
        } else {
            std::free(ptr);
        }
    }
    void* slot(std::size_t i, std::size_t stride) const { return static_cast<std::byte*>(ptr) + i * stride; }
};

template <typename T>
T* dalloc(std::size_t n) {
    void* p = nullptr;
    check(cudaMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T)), "cudaMalloc");
    check(cudaMemset(p, 0, std::max<std::size_t>(n, 1) * sizeof(T)), "cudaMemset");
    return static_cast<T*>(p);
}

// A wire that accepts nothing: only exchangev_async's host cost is timed.
class NullWire final : public nano_nccl::transport::peer::PeerWire {
public:
    explicit NullWire(int n) : n_(n) {}
    int rank() const override { return 0; }
    int nranks() const override { return n_; }
    int slots_per_link() const override { return 4; }
    std::size_t slot_bytes() const override { return 1 << 20; }
    std::uint64_t released(int) override { return 0; }
    OpId write_chunk(int, std::uint64_t, const nano_nccl::transport::peer::ChunkHeader&, const std::byte*) override {
        throw std::logic_error("NullWire");
    }
    std::uint64_t arrived(int) override { return 0; }
    const nano_nccl::transport::peer::ChunkHeader& slot_header(int, std::uint64_t) override {
        throw std::logic_error("NullWire");
    }
    OpId copy_out(int, std::uint64_t, std::byte*, std::size_t) override { throw std::logic_error("NullWire"); }
    void release(int, std::uint64_t) override {}
    bool done(OpId) override { return true; }

private:
    int n_;
};

struct Args {
    int world = 2, experts = 16, top_k = 2, hidden = 2048, ffn = 1408, tokens = 4096;
    int slots = 4;
    std::size_t slot_bytes = std::size_t{1} << 20;
    bool quick = false;
    bool shm = true;  // host regions as ShmWire allocates them
    std::string out;
};

void add_table(CostProfile& p, const std::string& name, const std::string& xu, std::vector<std::pair<double, double>> pts) {
    p.tables[name] = Table(std::move(pts));
    p.units[name] = {xu, "us"};
    std::printf("table %-24s", name.c_str());
    for (const auto& [x, y] : p.tables[name].points()) std::printf(" %g:%.2f", x, y);
    std::printf("\n");
    std::fflush(stdout);
}

void add_scalar(CostProfile& p, const std::string& name, double v) {
    p.scalars[name] = v;
    std::printf("scalar %-24s %.3f\n", name.c_str(), v);
    std::fflush(stdout);
}

int run(const Args& a) {
    int devices = 0;
    check(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    Gpu g[2];
    g[0].init(0);
    g[1].init(devices > 1 ? 1 : 0);
    const EpShape s{a.world, 0, a.experts, a.top_k, a.hidden, a.ffn};
    nano_nccl::ep::validate_shape(s);
    const int epr = s.experts_per_rank();
    const std::size_t row_bytes = s.row_bytes();

    CostProfile p;
    p.key = nano_nccl::exec::ProfileKey{a.world, a.experts, a.top_k, a.hidden, a.ffn, a.tokens, "zipf", 0.0,
                                        "bf16", "ffn", "shm", a.slots, a.slot_bytes};
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0), "props");
    p.meta.push_back(std::string("device ") + prop.name + " x" + std::to_string(devices));
    p.meta.push_back("gpu numa nodes " + std::to_string(nano_nccl::core::gpu_numa_node(g[0].device)) + " " +
                     std::to_string(nano_nccl::core::gpu_numa_node(g[1].device)));
    p.meta.push_back("copies staged through pinned host memory; no peer access");

    // ---- GPU ops on GPU 0 ----
    const std::int64_t max_rows = static_cast<std::int64_t>(a.tokens) * a.top_k * a.world;
    nano_nccl::ep::ExpertFfn ffn(g[0].device, epr, a.hidden, a.ffn, max_rows);
    for (int j = 0; j < epr; ++j) {
        check(cudaMemset(ffn.w13(j), 0, static_cast<std::size_t>(a.hidden) * 2 * a.ffn * sizeof(bf16)), "w13");
        check(cudaMemset(ffn.w2(j), 0, static_cast<std::size_t>(a.ffn) * a.hidden * sizeof(bf16)), "w2");
    }
    check(cudaSetDevice(g[0].device), "cudaSetDevice");
    bf16* rows = dalloc<bf16>(static_cast<std::size_t>(max_rows) * a.hidden);
    bf16* rows2 = dalloc<bf16>(static_cast<std::size_t>(max_rows) * a.hidden);
    cudaStream_t cs = g[0].compute;

    std::vector<double> row_points{1, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192};
    for (double r : {12288.0, 16384.0, 24576.0, 32768.0}) {
        if (r <= max_rows) row_points.push_back(r);
    }
    if (a.quick) row_points = {1, 256, 1024, 4096};
    {
        std::vector<std::pair<double, double>> pts;
        for (double r : row_points) {
            const std::vector<nano_nccl::ep::ExpertSegment> seg{{0, 0, 0, static_cast<std::int64_t>(r)}};
            pts.push_back({r, device_us(g[0], cs, 10, [&] { ffn.forward(rows, seg, cs); })});
        }
        add_table(p, cost::kExpertFfn, "rows", pts);
    }
    {
        // One unpack/repack: rows spread over W * epr segments.
        const int nseg = a.world * epr;
        std::vector<nano_nccl::ep::RowSegment> segs(nseg);
        auto* dsegs = dalloc<nano_nccl::ep::RowSegment>(nseg);
        auto* dexp = dalloc<nano_nccl::ep::ExpertSegment>(1);
        std::vector<std::pair<double, double>> perm, probe;
        for (double r : row_points) {
            const auto total = static_cast<std::int64_t>(r);
            std::int64_t at = 0;
            for (int i = 0; i < nseg; ++i) {
                const std::int64_t n = total / nseg + (i < total % nseg ? 1 : 0);
                segs[i] = {at, at, n};
                at += n;
            }
            check(cudaMemcpy(dsegs, segs.data(), nseg * sizeof(segs[0]), cudaMemcpyHostToDevice), "segs");
            perm.push_back({r, device_us(g[0], cs, 20, [&] {
                                nano_nccl::ep::launch_permute_rows(rows, rows2, dsegs, nseg, a.hidden, cs);
                            })});
            const nano_nccl::ep::ExpertSegment e{0, 0, 0, total};
            check(cudaMemcpy(dexp, &e, sizeof(e), cudaMemcpyHostToDevice), "exp");
            probe.push_back({r, device_us(g[0], cs, 20, [&] {
                                 nano_nccl::ep::launch_probe_experts(rows, dexp, 1, 0, a.hidden, cs);
                             })});
        }
        add_table(p, cost::kPermute, "rows", perm);
        add_table(p, cost::kExpertProbe, "rows", probe);

        // Host cost of issuing one expert op with n active experts.
        std::vector<std::pair<double, double>> issue;
        cudaEvent_t ev;
        check(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "event");
        std::vector<int> active{0};
        for (int n = 1; n < epr; n *= 2) active.push_back(n);
        active.push_back(epr);
        for (int n : active) {
            std::vector<nano_nccl::ep::ExpertSegment> es;
            for (int j = 0; j < n; ++j) es.push_back({j, 0, j * 256, 256});
            issue.push_back({static_cast<double>(n), host_us(g[0], cs, 20, [&] {
                                 nano_nccl::ep::launch_permute_rows(rows, rows2, dsegs, nseg, a.hidden, cs);
                                 ffn.forward(rows2, es, cs);
                                 nano_nccl::ep::launch_permute_rows(rows2, rows, dsegs, nseg, a.hidden, cs);
                                 check(cudaEventRecord(ev, cs), "record");
                             })});
        }
        add_table(p, cost::kHostExpertIssue, "experts", issue);
        cudaEventDestroy(ev);
        cudaFree(dsegs);
        cudaFree(dexp);
    }
    {
        // Route (count + place + pack + counts to host) and reduce.
        const int max_tokens = a.tokens;
        const auto routing = nano_nccl::ep::make_routing(s, max_tokens, 0.0, -1, 1);
        auto* ids = dalloc<std::int32_t>(routing.ids.size());
        auto* weights = dalloc<float>(routing.weights.size());
        check(cudaMemcpy(ids, routing.ids.data(), routing.ids.size() * 4, cudaMemcpyHostToDevice), "ids");
        check(cudaMemcpy(weights, routing.weights.data(), routing.weights.size() * 4, cudaMemcpyHostToDevice), "w");
        auto* counts = dalloc<std::int32_t>(a.experts + 1);
        auto* order = dalloc<std::int32_t>(a.experts);
        auto* positions = dalloc<std::int32_t>(routing.ids.size());
        auto* x = dalloc<bf16>(static_cast<std::size_t>(max_tokens) * a.hidden);
        auto* out = dalloc<bf16>(static_cast<std::size_t>(max_tokens) * a.hidden);
        std::int32_t *counts_host = nullptr, *order_host = nullptr;
        check(cudaMallocHost(&counts_host, (a.experts + 1) * 4), "pinned");
        check(cudaMallocHost(&order_host, a.experts * 4), "pinned");
        for (int e = 0; e < a.experts; ++e) order_host[e] = e;
        cudaEvent_t ev;
        check(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "event");
        auto route = [&](int tokens) {
            const int entries = tokens * a.top_k;
            check(cudaMemcpyAsync(order, order_host, a.experts * 4, cudaMemcpyHostToDevice, cs), "order");
            nano_nccl::ep::launch_count_experts(ids, entries, a.experts, counts, cs);
            nano_nccl::ep::launch_place_entries(ids, entries, a.experts, order, counts, positions, cs);
            nano_nccl::ep::launch_pack_rows(x, a.hidden, positions, entries, a.top_k, rows, cs);
            check(cudaMemcpyAsync(counts_host, counts, (a.experts + 1) * 4, cudaMemcpyDeviceToHost, cs), "counts");
            check(cudaEventRecord(ev, cs), "record");
        };
        auto route_and_wait = [&](int tokens) {  // leaves valid positions for the reduce
            route(tokens);
            check(cudaStreamSynchronize(cs), "sync");
        };
        std::vector<std::pair<double, double>> rt, rd;
        std::vector<int> token_points{128, 512, 1024, 2048, 4096, 8192};
        if (a.quick) token_points = {512, 4096};
        for (int t : token_points) {
            if (t > max_tokens) continue;
            rt.push_back({static_cast<double>(t) * a.top_k, device_us(g[0], cs, 10, [&] { route(t); })});
            route_and_wait(t);
            rd.push_back({static_cast<double>(t), device_us(g[0], cs, 10, [&] {
                              nano_nccl::ep::launch_combine_reduce(rows, positions, weights, t, a.top_k, a.hidden,
                                                                   out, cs);
                          })});
        }
        add_table(p, cost::kRoute, "entries", rt);
        add_table(p, cost::kReduce, "tokens", rd);
        add_scalar(p, cost::kHostRouteIssue, host_us(g[0], cs, 20, [&] { route(max_tokens); }));
        add_scalar(p, cost::kHostReduceIssue, host_us(g[0], cs, 20, [&] {
                       nano_nccl::ep::launch_combine_reduce(rows, positions, weights, max_tokens, a.top_k, a.hidden,
                                                            out, cs);
                       check(cudaEventRecord(ev, cs), "record");
                   }));

        // Idle-stream launch latency: issue -> observed done, minus device time.
        std::vector<double> lat;
        for (int i = 0; i < 200; ++i) {
            check(cudaStreamSynchronize(cs), "sync");
            const auto t0 = Clock::now();
            empty_kernel<<<1, 1, 0, cs>>>();
            check(cudaEventRecord(ev, cs), "record");
            while (cudaEventQuery(ev) == cudaErrorNotReady) {
            }
            lat.push_back(wall_us(t0, Clock::now()));
        }
        add_scalar(p, cost::kLaunchLatency, median(lat));
        // cudaEventQuery on a pending event: the unit of a progress pass.
        g[0].spin(cs, 5000);
        check(cudaEventRecord(ev, cs), "record");
        const auto q0 = Clock::now();
        constexpr int kQueries = 2000;
        for (int i = 0; i < kQueries; ++i) (void)cudaEventQuery(ev);
        const double query = wall_us(q0, Clock::now()) / kQueries;
        check(cudaStreamSynchronize(cs), "sync");
        // A ShmWire progress pass queries about one event per busy link and engine.
        add_scalar(p, cost::kHostPoll, 0.3 + 2.0 * a.world * query);
        p.meta.push_back("host_poll_us = 0.3 + 2 * world * cudaEventQuery(" + std::to_string(query) + " us)");

        cudaEventDestroy(ev);
        for (void* q : {static_cast<void*>(ids), static_cast<void*>(weights), static_cast<void*>(counts),
                        static_cast<void*>(order), static_cast<void*>(positions), static_cast<void*>(x),
                        static_cast<void*>(out)}) {
            cudaFree(q);
        }
        cudaFreeHost(counts_host);
        cudaFreeHost(order_host);
    }
    {
        // Host: plan building (+ its table upload) per step count, and one
        // exchange issue.
        std::vector<std::vector<std::int64_t>> n(a.world);
        for (int r = 0; r < a.world; ++r) {
            EpShape sr = s;
            sr.rank = r;
            n[r] = nano_nccl::ep::expert_counts(nano_nccl::ep::make_routing(sr, a.tokens, 0.0, -1, 7 + r).ids,
                                                a.experts);
        }
        std::vector<std::int64_t> recv(static_cast<std::size_t>(a.world) * epr);
        for (int src = 0; src < a.world; ++src) {
            for (int j = 0; j < epr; ++j) recv[src * epr + j] = n[src][j];
        }
        void* tables_host = nullptr;
        void* tables_dev = nullptr;
        const std::size_t tbytes = 64 * 1024;
        check(cudaMallocHost(&tables_host, tbytes), "pinned");
        check(cudaMalloc(&tables_dev, tbytes), "cudaMalloc");
        std::vector<std::pair<double, double>> plan_pts;
        for (int groups = 1; groups <= epr; groups *= 2) {
            const auto gp = nano_nccl::planner::fixed_group_plan(a.world, a.experts, groups);
            plan_pts.push_back({static_cast<double>(groups), host_us(g[0], cs, 20, [&] {
                                    const auto plan = nano_nccl::exec::make_ep_plan(s, gp, {}, n[0], recv, a.slot_bytes);
                                    std::size_t bytes = 0;
                                    for (const auto& st : plan.rows.steps) {
                                        bytes += (st.unpack.size() + st.repack.size()) * sizeof(st.unpack[0]) +
                                                 st.experts.size() * sizeof(st.experts[0]);
                                    }
                                    std::memset(tables_host, 0, std::min(bytes, tbytes));
                                    check(cudaMemcpyAsync(tables_dev, tables_host, std::min(bytes, tbytes),
                                                          cudaMemcpyHostToDevice, cs),
                                          "tables");
                                })});
        }
        add_table(p, cost::kHostPlan, "steps", plan_pts);
        cudaFreeHost(tables_host);
        cudaFree(tables_dev);

        NullWire wire(a.world);
        nano_nccl::transport::peer::PeerExchanger ex(&wire);
        const auto layout = nano_nccl::transport::peer::uniform_layout(a.world, 0);
        std::vector<double> t;
        for (int i = 0; i < 2000; ++i) {
            const auto t0 = Clock::now();
            (void)ex.exchangev_async(1, 1, layout, nullptr, nullptr);
            t.push_back(wall_us(t0, Clock::now()));
        }
        add_scalar(p, cost::kHostExchangeIssue, median(t));
    }
    cudaFree(rows);
    cudaFree(rows2);

    // ---- Copy engines ----
    // region[r] is GPU r's receive region (on GPU r's node) with `slots`
    // slots; copies rotate through the slots and through larger device
    // buffers, as a stream of chunks does.
    const std::size_t max_chunk = std::max<std::size_t>(a.slot_bytes, 4u << 20);
    const int nslots = std::max(a.slots, 1);
    constexpr int kDevSpan = 8;  // device buffers hold this many chunks
    HostRegion region0(max_chunk * nslots, nano_nccl::core::gpu_numa_node(g[0].device), a.shm);
    HostRegion region1(max_chunk * nslots, nano_nccl::core::gpu_numa_node(g[1].device), a.shm);
    const HostRegion* region[2] = {&region0, &region1};
    p.meta.push_back(std::string("host regions: ") + (a.shm ? "POSIX shm (MAP_SHARED), as ShmWire" : "private memory") +
                     ", " + std::to_string(nslots) + " rotating slots");
    std::byte* src[2];
    std::byte* dst[2];
    std::byte* self_slot[2];
    bf16* load_rows[2];
    for (int i = 0; i < 2; ++i) {
        check(cudaSetDevice(g[i].device), "cudaSetDevice");
        src[i] = dalloc<std::byte>(max_chunk * kDevSpan);
        dst[i] = dalloc<std::byte>(max_chunk * kDevSpan);
        self_slot[i] = dalloc<std::byte>(max_chunk * nslots);
        load_rows[i] = dalloc<bf16>(static_cast<std::size_t>(1024) * a.hidden);
    }
    // Compute load for the interference runs: one 1024-row expert per GPU.
    nano_nccl::ep::ExpertFfn ffn1(g[1].device, 1, a.hidden, a.ffn, 1024);
    nano_nccl::ep::ExpertFfn* load_ffn[2] = {&ffn, &ffn1};
    const std::vector<nano_nccl::ep::ExpertSegment> load_seg{{0, 0, 0, 1024}};

    std::vector<double> sizes{4096, 16384, 65536, 262144, 524288, 1048576, 2097152, 4194304};
    if (a.quick) sizes = {65536, 1048576};
    std::vector<std::pair<double, double>> tx_c, rx_c, tx_a, rx_a, self_t, self_r;
    cudaEvent_t ev_a[6], ev_b[6];
    for (int i = 0; i < 6; ++i) {
        check(cudaSetDevice(g[i < 4 ? i / 2 : i - 4].device), "cudaSetDevice");
        check(cudaEventCreate(&ev_a[i]), "event");
        check(cudaEventCreate(&ev_b[i]), "event");
    }
    // Roles in order: GPU0 tx, GPU0 rx, GPU1 tx, GPU1 rx. GPU i sends into
    // the other GPU's region and receives from its own.
    struct Role {
        int gpu;
        bool tx;
    };
    const Role roles[4] = {{0, true}, {0, false}, {1, true}, {1, false}};
    std::size_t seq[4] = {};
    // ShmWire records an event after every chunk copy; so do these.
    cudaEvent_t chunk_ev[4];
    for (int i = 0; i < 4; ++i) {
        check(cudaSetDevice(g[i / 2].device), "cudaSetDevice");
        check(cudaEventCreateWithFlags(&chunk_ev[i], cudaEventDisableTiming), "event");
    }
    // One chunk of `role`: remote (into / out of a host slot) or, with
    // `self`, the self link's device slot.
    auto copy_once = [&](int role, std::size_t bytes, bool self = false) {
        const Role& ro = roles[role];
        const int i = ro.gpu;
        const std::size_t k = seq[role]++;
        cudaStream_t st = ro.tx ? g[i].tx : g[i].rx;
        void* slot = self ? static_cast<void*>(self_slot[i] + (k % nslots) * max_chunk)
                          : (ro.tx ? region[1 - i] : region[i])->slot(k % nslots, max_chunk);
        std::byte* dev = (ro.tx ? src[i] : dst[i]) + (k % kDevSpan) * max_chunk;
        if (ro.tx) {
            check(cudaMemcpyAsync(slot, dev, bytes, cudaMemcpyDefault, st), "tx copy");
        } else {
            check(cudaMemcpyAsync(dev, slot, bytes, cudaMemcpyDefault, st), "rx copy");
        }
        check(cudaEventRecord(chunk_ev[role], st), "record");
    };
    // Per-copy time of each active role, all started together behind spins.
    // With calls > 0, both GPUs' compute streams also run that many expert
    // calls from the same start; `ffn_us` (if set) gets GPU 0's time per call.
    auto measure = [&](const std::vector<int>& active, std::size_t bytes, double* per_copy, int calls = 0,
                       int reps = 0, double* ffn_us = nullptr) {
        if (reps == 0) reps = static_cast<int>(std::clamp<std::size_t>((32u << 20) / bytes, 16, 1024));
        const double gate = 3000 + 12.0 * reps * active.size() + 30.0 * calls;
        const bool compute = calls > 0;
        for (int role : active) {
            const Role& ro = roles[role];
            check(cudaSetDevice(g[ro.gpu].device), "cudaSetDevice");
            cudaStream_t st = ro.tx ? g[ro.gpu].tx : g[ro.gpu].rx;
            g[ro.gpu].spin(st, gate);
            check(cudaEventRecord(ev_a[role], st), "record");
        }
        if (compute) {
            for (int i = 0; i < 2; ++i) {
                check(cudaSetDevice(g[i].device), "cudaSetDevice");
                g[i].spin(g[i].compute, gate);
                check(cudaEventRecord(ev_a[4 + i], g[i].compute), "record");
                for (int c = 0; c < calls; ++c) load_ffn[i]->forward(load_rows[i], load_seg, g[i].compute);
                check(cudaEventRecord(ev_b[4 + i], g[i].compute), "record");
            }
        }
        for (int i = 0; i < reps; ++i) {
            for (int role : active) {
                check(cudaSetDevice(g[roles[role].gpu].device), "cudaSetDevice");
                copy_once(role, bytes);
            }
        }
        for (int role : active) {
            const Role& ro = roles[role];
            check(cudaSetDevice(g[ro.gpu].device), "cudaSetDevice");
            check(cudaEventRecord(ev_b[role], ro.tx ? g[ro.gpu].tx : g[ro.gpu].rx), "record");
        }
        for (int role : active) {
            check(cudaEventSynchronize(ev_b[role]), "sync");
            float ms = 0;
            check(cudaEventElapsedTime(&ms, ev_a[role], ev_b[role]), "elapsed");
            per_copy[role] = 1000.0 * ms / reps;
        }
        if (compute) {
            for (int i = 0; i < 2; ++i) check(cudaEventSynchronize(ev_b[4 + i]), "sync");
            float ms = 0;
            check(cudaEventElapsedTime(&ms, ev_a[4], ev_b[4]), "elapsed");
            if (ffn_us != nullptr) *ffn_us = 1000.0 * ms / calls;
        }
    };
    // Per-copy time of one engine running `pattern` alone: 0 remote chunks,
    // 1 self chunks, 2 alternating remote/self (time per pair).
    auto engine_seq = [&](int role, std::size_t bytes, int pattern, int reps) {
        const Role& ro = roles[role];
        check(cudaSetDevice(g[ro.gpu].device), "cudaSetDevice");
        cudaStream_t st = ro.tx ? g[ro.gpu].tx : g[ro.gpu].rx;
        std::vector<double> t;
        for (int trial = 0; trial < 3; ++trial) {
            g[ro.gpu].spin(st, 3000 + 12.0 * reps);
            check(cudaEventRecord(ev_a[role], st), "record");
            for (int i = 0; i < reps; ++i) {
                if (pattern != 1) copy_once(role, bytes, false);
                if (pattern != 0) copy_once(role, bytes, true);
            }
            check(cudaEventRecord(ev_b[role], st), "record");
            check(cudaEventSynchronize(ev_b[role]), "sync");
            float ms = 0;
            check(cudaEventElapsedTime(&ms, ev_a[role], ev_b[role]), "elapsed");
            t.push_back(1000.0 * ms / reps);
        }
        return median(t);
    };
    std::vector<std::pair<double, double>> tx_cg[2], rx_cg[2], tx_ag[2], rx_ag[2];
    for (double b : sizes) {
        const auto bytes = static_cast<std::size_t>(b);
        double all[4] = {}, alone[4] = {};
        measure({0, 1, 2, 3}, bytes, all);  // warm-up
        std::vector<double> t0, t1, per[4];
        for (int trial = 0; trial < 3; ++trial) {
            measure({0, 1, 2, 3}, bytes, all);
            t0.push_back((all[0] + all[2]) / 2);
            t1.push_back((all[1] + all[3]) / 2);
            for (int r = 0; r < 4; ++r) per[r].push_back(all[r]);
        }
        tx_c.push_back({b, median(t0)});
        rx_c.push_back({b, median(t1)});
        for (int r = 0; r < 4; ++r) {
            measure({r}, bytes, alone);
            (r % 2 == 0 ? tx_ag : rx_ag)[r / 2].push_back({b, alone[r]});
            (r % 2 == 0 ? tx_cg : rx_cg)[r / 2].push_back({b, median(per[r])});
        }
        tx_a.push_back({b, (tx_ag[0].back().second + tx_ag[1].back().second) / 2});
        rx_a.push_back({b, (rx_ag[0].back().second + rx_ag[1].back().second) / 2});
        // Self link: device slot, on the same engines (D2D).
        self_t.push_back({b, (engine_seq(0, bytes, 1, 32) + engine_seq(2, bytes, 1, 32)) / 2});
        self_r.push_back({b, (engine_seq(1, bytes, 1, 32) + engine_seq(3, bytes, 1, 32)) / 2});
    }
    add_table(p, cost::kTxRemote, "bytes", tx_c);
    add_table(p, cost::kRxRemote, "bytes", rx_c);
    add_table(p, cost::kTxRemoteAlone, "bytes", tx_a);
    add_table(p, cost::kRxRemoteAlone, "bytes", rx_a);
    add_table(p, cost::kTxSelf, "bytes", self_t);
    add_table(p, cost::kRxSelf, "bytes", self_r);
    // Per GPU (the simulator uses "<name>@<rank>" for rank r when present):
    // the two A40s sit on different sockets and do not copy equally fast.
    for (int i = 0; i < 2; ++i) {
        const std::string at = "@" + std::to_string(i);
        add_table(p, cost::kTxRemote + at, "bytes", tx_cg[i]);
        add_table(p, cost::kRxRemote + at, "bytes", rx_cg[i]);
        add_table(p, cost::kTxRemoteAlone + at, "bytes", tx_ag[i]);
        add_table(p, cost::kRxRemoteAlone + at, "bytes", rx_ag[i]);
    }
    // Switching one engine between remote and self chunks stalls it: time of
    // an alternating pair minus one remote and one self chunk, per switch.
    {
        const std::size_t bytes = std::min<std::size_t>(a.slot_bytes, max_chunk);
        double sw[2][2] = {};
        for (int r = 0; r < 4; ++r) {
            const double remote = engine_seq(r, bytes, 0, 32), self = engine_seq(r, bytes, 1, 32);
            const double pair = engine_seq(r, bytes, 2, 32);
            sw[r / 2][r % 2] = std::max(0.0, (pair - remote - self) / 2);
        }
        add_scalar(p, cost::kTxSwitch, (sw[0][0] + sw[1][0]) / 2);
        add_scalar(p, cost::kRxSwitch, (sw[0][1] + sw[1][1]) / 2);
        for (int i = 0; i < 2; ++i) {
            add_scalar(p, std::string(cost::kTxSwitch) + "@" + std::to_string(i), sw[i][0]);
            add_scalar(p, std::string(cost::kRxSwitch) + "@" + std::to_string(i), sw[i][1]);
        }
    }

    // Interference at the slot size: copies while both GPUs run experts, and
    // experts while every copy engine is busy.
    {
        const std::size_t bytes = std::min<std::size_t>(a.slot_bytes, max_chunk);
        std::vector<double> ftx, frx, fgpu;
        double ffn_alone = 0;
        {
            const std::vector<nano_nccl::ep::ExpertSegment> seg = load_seg;
            ffn_alone = device_us(g[0], g[0].compute, 20, [&] { ffn.forward(load_rows[0], seg, g[0].compute); });
        }
        constexpr int kReps = 64;
        for (int trial = 0; trial < 3; ++trial) {
            double busy[4] = {}, loaded[4] = {}, spare[4] = {}, ffn_loaded = 0;
            measure({0, 1, 2, 3}, bytes, busy, 0, kReps);
            const double copy_span = kReps * std::max(busy[0], busy[1]);
            // Experts outlast the copies: every copy runs under compute.
            measure({0, 1, 2, 3}, bytes, loaded, static_cast<int>(1.5 * copy_span / ffn_alone) + 4, kReps);
            // Copies outlast the experts: every expert runs under copies.
            const int calls = std::max(2, static_cast<int>(0.6 * copy_span / ffn_alone));
            measure({0, 1, 2, 3}, bytes, spare, calls, kReps, &ffn_loaded);
            ftx.push_back((loaded[0] + loaded[2]) / (busy[0] + busy[2]));
            frx.push_back((loaded[1] + loaded[3]) / (busy[1] + busy[3]));
            fgpu.push_back(ffn_loaded / ffn_alone);
        }
        add_scalar(p, cost::kCopyTxComputeFactor, median(ftx));
        add_scalar(p, cost::kCopyRxComputeFactor, median(frx));
        add_scalar(p, cost::kGpuCopySlowdown, median(fgpu));
        p.meta.push_back("interference measured at " + std::to_string(bytes) +
                         " B chunks with 1024-row experts on both GPUs; expert alone " + std::to_string(ffn_alone) +
                         " us");
    }

    // Host side of one chunk: ShmWire::write_chunk (header store, copy,
    // event record) and copy_out (copy, event record).
    {
        check(cudaSetDevice(g[0].device), "cudaSetDevice");
        cudaEvent_t ev;
        check(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "event");
        alignas(64) unsigned char header[64] = {};
        add_scalar(p, cost::kHostPost, host_us(g[0], g[0].tx, 50, [&] {
                       std::memcpy(static_cast<unsigned char*>(region[1]->ptr) + max_chunk - 64, header, 56);
                       check(cudaMemcpyAsync(region[1]->ptr, src[0], 65536, cudaMemcpyDefault, g[0].tx), "copy");
                       check(cudaEventRecord(ev, g[0].tx), "record");
                   }));
        add_scalar(p, cost::kHostCopyOut, host_us(g[0], g[0].rx, 50, [&] {
                       check(cudaMemcpyAsync(dst[0], region[0]->ptr, 65536, cudaMemcpyDefault, g[0].rx), "copy");
                       check(cudaEventRecord(ev, g[0].rx), "record");
                   }));
        // Idle copy engine: issue -> observed done, minus the copy itself.
        std::vector<double> lat;
        for (int i = 0; i < 200; ++i) {
            check(cudaStreamSynchronize(g[0].tx), "sync");
            const auto t0 = Clock::now();
            check(cudaMemcpyAsync(region[1]->ptr, src[0], 4096, cudaMemcpyDefault, g[0].tx), "copy");
            check(cudaEventRecord(ev, g[0].tx), "record");
            while (cudaEventQuery(ev) == cudaErrorNotReady) {
            }
            lat.push_back(wall_us(t0, Clock::now()));
        }
        add_scalar(p, cost::kCopyLatency, std::max(0.0, median(lat) - tx_a.front().second));
        cudaEventDestroy(ev);
    }
    // Shared-memory counter visibility between processes on one host is far
    // below a microsecond; not measured here.
    add_scalar(p, cost::kCreditLatency, 0.2);
    add_scalar(p, cost::kArrivalLatency, 0.2);
    p.meta.push_back("credit_latency_us and arrival_latency_us are assumed (0.2), not measured");

    for (int i = 0; i < 2; ++i) {
        cudaSetDevice(g[i].device);
        cudaFree(src[i]);
        cudaFree(dst[i]);
        cudaFree(self_slot[i]);
        cudaFree(load_rows[i]);
    }
    if (!a.out.empty()) {
        p.save(a.out);
        std::printf("profile written to %s\n", a.out.c_str());
    }
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
            if (arg == "--out") a.out = next();
            else if (arg == "--world") a.world = std::atoi(next());
            else if (arg == "--experts") a.experts = std::atoi(next());
            else if (arg == "--topk") a.top_k = std::atoi(next());
            else if (arg == "--hidden") a.hidden = std::atoi(next());
            else if (arg == "--ffn-dim") a.ffn = std::atoi(next());
            else if (arg == "--tokens") a.tokens = std::atoi(next());
            else if (arg == "--slots") a.slots = std::atoi(next());
            else if (arg == "--slot-bytes") a.slot_bytes = std::strtoull(next(), nullptr, 0);
            else if (arg == "--quick") a.quick = true;
            else if (arg == "--host-memory") {
                const std::string v = next();
                if (v != "shm" && v != "private") throw std::invalid_argument("--host-memory shm|private");
                a.shm = v == "shm";
            }
            else throw std::invalid_argument("unknown argument " + arg);
        }
        return run(a);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ep_profile: %s\n", e.what());
        return 1;
    }
}
