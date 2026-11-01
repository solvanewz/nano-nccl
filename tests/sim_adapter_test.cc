// CPU test of the shared execution description (src/exec) and the EP layer
// simulator (sim_adapter), with a synthetic cost profile.
//
// Program: queue/dependency invariants and the exchange issue order for every
// group count, lookahead and overlap setting.
// Simulator, on random worlds, counts (with zeros), group plans, slots and
// options: every rank finishes; every op is issued only after each of its
// dependencies was observed; each exchange's completion events come in
// protocol order; the messages of both ends of every link agree; no link
// ever has more chunks in flight than slots. Plus the plan's phase-one
// timing checks: a slower network delays the experts, combines start only
// after their experts, smaller buffers cause waiting, no overlap gives
// strictly serial steps, overlap beats serial when both sides cost the same,
// and runs are deterministic. Plus the in-situ copy model on its own: two
// remote copies on one GPU slow each other by the profile's factor, and the
// survivor speeds up again when the other ends (exact times); per-rank GPU
// slowdowns apply to their rank only.
#include "cost_profile.h"
#include "exec/exec_program.h"
#include "exec/exec_trace.h"
#include "planner/group_plan.h"
#include "sim_layer.h"
#include "sim_wire.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <exception>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using nano_nccl::exec::Condition;
using nano_nccl::exec::LayerProgram;
using nano_nccl::exec::OpKind;
using nano_nccl::exec::Queue;
using nano_nccl::sim::CostProfile;
using nano_nccl::sim::EventQueue;
using nano_nccl::sim::HostModel;
using nano_nccl::sim::SimFabric;
using nano_nccl::sim::WireCosts;
using nano_nccl::transport::peer::ChunkHeader;
using nano_nccl::sim::SimLayerConfig;
using nano_nccl::sim::SimResult;
using nano_nccl::sim::Table;
namespace cost = nano_nccl::sim::cost;

int g_failed = 0;

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
}

template <typename Fn>
void check(const char* name, Fn fn) {
    try {
        fn();
        std::printf("PASS %s\n", name);
    } catch (const std::exception& e) {
        ++g_failed;
        std::printf("FAIL %s: %s\n", name, e.what());
    }
}

// With `contention`, also the *_alone copy tables (cheaper) and the
// interference factors, which enable the simulator's contention model.
CostProfile synthetic_profile(bool contention = false) {
    CostProfile p;
    p.key = nano_nccl::exec::ProfileKey{2, 8, 2, 256, 512, 512, "zipf", 0, "bf16", "ffn", "shm", 4, 65536};
    auto t = [&](const char* name, std::vector<std::pair<double, double>> pts) { p.tables[name] = Table(pts); };
    t(cost::kRoute, {{0, 5}, {100000, 105}});
    t(cost::kPermute, {{0, 1}, {10000, 11}});
    t(cost::kExpertFfn, {{1, 10}, {1000, 30}, {10000, 200}});
    t(cost::kExpertProbe, {{0, 1}, {10000, 5}});
    t(cost::kReduce, {{0, 2}, {10000, 12}});
    t(cost::kTxRemote, {{4096, 1.0}, {1 << 20, 80}});
    t(cost::kRxRemote, {{4096, 1.0}, {1 << 20, 80}});
    t(cost::kTxSelf, {{4096, 0.5}, {1 << 20, 4}});
    t(cost::kRxSelf, {{4096, 0.5}, {1 << 20, 4}});
    t(cost::kHostExpertIssue, {{0, 5}, {8, 40}});
    t(cost::kHostPlan, {{1, 10}, {8, 30}});
    p.scalars = {{cost::kHostRouteIssue, 10}, {cost::kHostReduceIssue, 3}, {cost::kHostExchangeIssue, 1},
                 {cost::kHostPost, 2},        {cost::kHostCopyOut, 2},     {cost::kHostPoll, 0.5},
                 {cost::kLaunchLatency, 4},   {cost::kCopyLatency, 2},     {cost::kCreditLatency, 0.2},
                 {cost::kArrivalLatency, 0.2}};
    if (contention) {
        t(cost::kTxRemoteAlone, {{4096, 0.6}, {1 << 20, 45}});
        t(cost::kRxRemoteAlone, {{4096, 0.6}, {1 << 20, 45}});
        p.scalars[cost::kCopyTxComputeFactor] = 1.3;
        p.scalars[cost::kCopyRxComputeFactor] = 1.4;
        p.scalars[cost::kGpuCopySlowdown] = 1.2;
    }
    return p;
}

struct Case {
    SimLayerConfig config;
    std::vector<std::vector<std::int64_t>> counts;
    std::vector<int> tokens;
};

Case balanced_case(int world, int epr, int rows_per_expert, int hidden) {
    Case c;
    SimLayerConfig& s = c.config;
    s.world = world;
    s.num_experts = world * epr;
    s.top_k = 2;
    s.hidden = hidden;
    s.ffn = 512;
    s.groups = nano_nccl::planner::fixed_group_plan(world, s.num_experts, 1);
    s.options = {false, 0};
    s.slots_per_link = 4;
    s.slot_bytes = 65536;
    for (int r = 0; r < world; ++r) {
        c.counts.push_back(std::vector<std::int64_t>(s.num_experts, rows_per_expert));
        c.tokens.push_back(rows_per_expert * s.num_experts / s.top_k);
    }
    return c;
}

SimResult run(const Case& c, const CostProfile& p) { return nano_nccl::sim::simulate_layer(c.config, c.counts, c.tokens, p); }

// Invariants every simulated run must satisfy.
void check_invariants(const Case& c, const SimResult& res, const CostProfile& prof) {
    const int W = c.config.world;
    require(static_cast<int>(res.ranks.size()) == W, "rank count");
    for (int r = 0; r < W; ++r) {
        const auto& rr = res.ranks[r];
        const LayerProgram& p = rr.program;
        for (std::size_t i = 0; i < p.ops.size(); ++i) {
            const auto& t = rr.ops[i];
            const std::string at = "rank " + std::to_string(r) + " op " + std::to_string(i) + " (" +
                                   nano_nccl::exec::op_kind_name(p.ops[i].kind) + ")";
            require(t.issue_begin >= 0 && t.issue_end >= t.issue_begin, at + ": not issued");
            require(t.complete >= t.issue_end, at + ": completes before its issue ends");
            for (const auto& dep : p.ops[i].deps) {
                const double seen = rr.ops[dep.op].condition[static_cast<int>(dep.condition)];
                require(seen >= 0 && seen <= t.issue_begin,
                        at + ": issued before " + nano_nccl::exec::condition_name(dep.condition) + " of op " +
                            std::to_string(dep.op));
            }
            if (p.ops[i].queue == Queue::kExchange) {
                const double* e = t.condition;
                for (int k = 0; k < 5; ++k) require(e[k] >= 0, at + ": event " + std::to_string(k) + " never seen");
                require(e[0] <= e[1], at + ": send buffer reusable before submitted");
                require(e[2] <= e[3] && e[3] <= e[4], at + ": receive events out of order");
            }
        }
        // One stream: device intervals in issue order, never overlapping.
        double stream_free = 0;
        for (std::size_t i = 0; i < p.ops.size(); ++i) {
            if (p.ops[i].queue != Queue::kStream) continue;
            const auto& t = rr.ops[i];
            require(t.gpu_start >= t.issue_end && t.gpu_end >= t.gpu_start, "stream op runs before its launch");
            require(t.gpu_start >= stream_free, "stream ops overlap on rank " + std::to_string(r));
            require(t.complete >= t.gpu_end, "stream op observed before it finished");
            stream_free = t.gpu_end;
        }
        // Same exchange order on every rank.
        for (std::size_t i = 0; i < p.ops.size(); ++i) {
            require(p.ops[i].kind == res.ranks[0].program.ops[i].kind, "programs differ between ranks");
        }
    }
    // Both ends of every message agree.
    for (int r = 0; r < W; ++r) {
        const auto& prog = res.ranks[r].program;
        for (std::size_t i = 0; i < prog.ops.size(); ++i) {
            if (prog.ops[i].queue != Queue::kExchange) continue;
            const auto& msgs = res.ranks[r].messages[i];
            require(static_cast<int>(msgs.size()) == W, "message count");
            for (const auto& m : msgs) {
                const auto& dst_plan = res.ranks[m.dst].plan;
                const std::size_t expect =
                    prog.ops[i].kind == OpKind::kCounts
                        ? nano_nccl::exec::counts_layout(dst_plan.shape).recv_bytes[r]
                        : nano_nccl::exec::exchange_layout(res.ranks[m.dst].program, static_cast<int>(i), dst_plan)
                              .recv_bytes[r];
                require(m.src == r && m.bytes == expect, "message " + std::to_string(r) + "->" +
                                                             std::to_string(m.dst) + " of op " + std::to_string(i) +
                                                             " disagrees with the receiver");
                require(m.sequence == prog.exchange_seq[i], "message sequence");
            }
        }
    }
    // A payload cannot arrive before its sender issued it and one chunk crossed.
    for (int d = 0; d < W; ++d) {
        const auto& prog = res.ranks[d].program;
        for (std::size_t i = 0; i < prog.ops.size(); ++i) {
            if (prog.ops[i].kind != OpKind::kDispatch && prog.ops[i].kind != OpKind::kCombine) continue;
            const auto& in = nano_nccl::exec::exchange_layout(prog, static_cast<int>(i), res.ranks[d].plan);
            const double arrived = res.ranks[d].ops[i].condition[static_cast<int>(Condition::kPayloadArrived)];
            for (int s = 0; s < W; ++s) {
                if (in.recv_bytes[s] == 0) continue;
                // The sender's tx engine copies the message's chunks one after
                // another, each at least at its cheapest (uncontended) cost.
                const bool alone = s != d && c.config.contention && prof.tables.count(cost::kTxRemoteAlone);
                const Table& tx = prof.table(s == d ? cost::kTxSelf : (alone ? cost::kTxRemoteAlone : cost::kTxRemote));
                const std::size_t bytes = in.recv_bytes[s], slot = c.config.slot_bytes;
                double cross = static_cast<double>(bytes / slot) * tx(static_cast<double>(slot));
                if (bytes % slot != 0) cross += tx(static_cast<double>(bytes % slot));
                cross *= c.config.copy_scale;
                require(arrived >= res.ranks[s].ops[i].issue_begin + cross - 1e-9,
                        "payload " + std::to_string(s) + "->" + std::to_string(d) + " of op " + std::to_string(i) +
                            " arrived before it could cross");
            }
        }
    }
    for (const auto& l : res.links) {
        require(l.max_in_flight <= static_cast<std::uint32_t>(c.config.slots_per_link), "more chunks in flight than slots");
    }
}

void test_program() {
    for (int G = 1; G <= 8; ++G) {
        for (int L = 0; L <= 9; ++L) {
            for (bool overlap : {false, true}) {
                std::vector<int> order(G);
                std::iota(order.begin(), order.end(), 0);
                std::reverse(order.begin(), order.end());
                const LayerProgram p = nano_nccl::exec::build_layer_program(3, order, {overlap, L});
                const LayerProgram q = nano_nccl::exec::build_layer_program(0, order, {overlap, L});
                // Exchange issue order: identical across ranks, and the documented one.
                std::vector<std::pair<OpKind, int>> ex, exq, want;
                for (const auto& op : p.ops) {
                    if (op.queue == Queue::kExchange) ex.push_back({op.kind, op.step});
                }
                for (const auto& op : q.ops) {
                    if (op.queue == Queue::kExchange) exq.push_back({op.kind, op.step});
                }
                require(ex == exq, "exchange order depends on the rank");
                want.push_back({OpKind::kCounts, -1});
                if (!overlap) {
                    for (int i = 0; i < G; ++i) {
                        want.push_back({OpKind::kDispatch, i});
                        want.push_back({OpKind::kCombine, i});
                    }
                } else {
                    const int lead = std::min(L, G - 1);
                    for (int i = 0; i <= lead; ++i) want.push_back({OpKind::kDispatch, i});
                    for (int i = 0; i < G; ++i) {
                        want.push_back({OpKind::kCombine, i});
                        if (i + lead + 1 < G) want.push_back({OpKind::kDispatch, i + lead + 1});
                    }
                }
                require(ex == want, "exchange order G=" + std::to_string(G) + " L=" + std::to_string(L));
                for (int i = 0; i < G; ++i) require(p.ops[p.dispatch[i]].group == order[i], "step group");
                for (int i = 1; i < G && !overlap; ++i) {
                    bool waits = false;
                    for (const auto& d : p.ops[p.dispatch[i]].deps) {
                        waits |= d.op == p.combine[i - 1] && d.condition == Condition::kConsumerReadable;
                    }
                    require(waits, "serial dispatch does not wait for the previous combine");
                }
            }
        }
    }
    bool threw = false;
    try {
        nano_nccl::exec::build_layer_program(0, {0}, {true, -1});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "negative lookahead accepted");
    threw = false;
    try {
        nano_nccl::exec::build_layer_program(0, {}, {});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "empty group order accepted");
}

void test_formats() {
    const CostProfile p = synthetic_profile();
    const CostProfile q = CostProfile::parse(p.serialize());
    require(q.serialize() == p.serialize(), "profile round trip");
    require(q.key.to_string() == p.key.to_string(), "key round trip");
    const Table t({{0, 1}, {10, 11}});
    require(t(5) == 6 && t(20) == 21 && t(-3) == 1, "table interpolation");
    nano_nccl::exec::LayerTrace tr;
    tr.key = p.key;
    tr.groups = nano_nccl::planner::fixed_group_plan(2, 8, 2);
    tr.options = {true, 1};
    tr.iters = 3;
    for (int r = 0; r < 2; ++r) {
        nano_nccl::exec::TraceRank rank;
        rank.tokens = 100 + r;
        rank.total = 12.5 + r;
        rank.expert_counts.assign(8, 3 + r);
        tr.ranks.push_back(rank);
        tr.ops.push_back({r, 0, OpKind::kRoute, -1, 0, 1.5, 7.25});
    }
    const auto back = nano_nccl::exec::LayerTrace::parse(tr.serialize());
    require(back.serialize() == tr.serialize(), "trace round trip");
    bool threw = false;
    try {
        nano_nccl::exec::ProfileKey::parse("world=2 bogus=1");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "bad profile key accepted");
}

void test_random(int trials) {
    const CostProfile plain = synthetic_profile(false), contended = synthetic_profile(true);
    std::mt19937_64 rng(2026);
    const int worlds[] = {1, 2, 3, 4, 8};
    for (int t = 0; t < trials; ++t) {
        Case c;
        SimLayerConfig& s = c.config;
        s.world = worlds[rng() % 5];
        const int epr = 1 + static_cast<int>(rng() % 4);
        s.num_experts = s.world * epr;
        s.top_k = 1;
        s.hidden = 8 * (1 + static_cast<int>(rng() % 64));
        s.ffn = 64;
        s.mode = rng() % 2 ? "ffn" : "probe";
        if (rng() % 2) {
            s.groups = nano_nccl::planner::fixed_group_plan(s.world, s.num_experts, 1 + static_cast<int>(rng() % epr));
        } else {
            const int G = 1 + static_cast<int>(rng() % (s.num_experts + 1));
            for (int e = 0; e < s.num_experts; ++e) s.groups.expert_group.push_back(static_cast<int>(rng() % G));
            s.groups.order.resize(G);
            std::iota(s.groups.order.begin(), s.groups.order.end(), 0);
            std::shuffle(s.groups.order.begin(), s.groups.order.end(), rng);
        }
        s.options = {rng() % 3 != 0, static_cast<int>(rng() % 5)};
        s.slots_per_link = 1 + static_cast<int>(rng() % 4);
        s.slot_bytes = std::size_t{1} << (9 + rng() % 8);  // 512 B .. 64 KiB: many chunks per message
        for (int r = 0; r < s.world; ++r) {
            std::vector<std::int64_t> n(s.num_experts);
            for (auto& x : n) x = rng() % 3 == 0 ? 0 : static_cast<std::int64_t>(rng() % 40);
            c.counts.push_back(n);
            c.tokens.push_back(static_cast<int>(std::accumulate(n.begin(), n.end(), std::int64_t{0})));
        }
        const CostProfile& prof = t % 2 ? contended : plain;
        try {
            check_invariants(c, run(c, prof), prof);
        } catch (const std::exception& e) {
            throw std::runtime_error("trial " + std::to_string(t) + " (W=" + std::to_string(s.world) +
                                     " groups=" + std::to_string(s.groups.groups()) + " slots=" +
                                     std::to_string(s.slots_per_link) + " slot_bytes=" + std::to_string(s.slot_bytes) +
                                     "): " + e.what());
        }
    }
}

double expert_launch(const SimResult& r, int rank, int step) {
    return r.ranks[rank].ops[r.ranks[rank].program.expert[step]].issue_begin;
}

void test_timing() {
    const CostProfile prof = synthetic_profile();
    // Communication and computation of similar size: W=2, 4 experts/rank,
    // 512 rows each, 1 KiB rows -> 2 MiB per peer per layer.
    Case base = balanced_case(2, 4, 512, 512);
    base.config.groups = nano_nccl::planner::fixed_group_plan(2, 8, 4);
    base.config.options = {true, 1};
    const SimResult r0 = run(base, prof);
    check_invariants(base, r0, prof);

    // Deterministic.
    const SimResult r0b = run(base, prof);
    for (std::size_t i = 0; i < r0.ranks[0].ops.size(); ++i) {
        require(r0.ranks[0].ops[i].complete == r0b.ranks[0].ops[i].complete, "simulation is not deterministic");
    }

    // A slower network delays the experts of later steps and the layer.
    Case slow = base;
    slow.config.copy_scale = 4.0;
    const SimResult r1 = run(slow, prof);
    check_invariants(slow, r1, prof);
    require(expert_launch(r1, 0, 3) > expert_launch(r0, 0, 3) + 100, "slower network did not delay the experts");
    require(r1.makespan > r0.makespan * 1.5, "slower network did not slow the layer");

    // Combines start only after their experts finished.
    for (const SimResult* r : {&r0, &r1}) {
        for (const auto& rank : r->ranks) {
            for (int i = 0; i < rank.program.steps; ++i) {
                require(rank.ops[rank.program.combine[i]].issue_begin >= rank.ops[rank.program.expert[i]].complete,
                        "combine issued before its experts finished");
            }
        }
    }

    // Smaller buffers make senders wait for credits.
    Case tight = base;
    tight.config.slots_per_link = 1;
    tight.config.slot_bytes = 4096;
    Case roomy = tight;
    roomy.config.slots_per_link = 16;
    const SimResult rt = run(tight, prof), rr = run(roomy, prof);
    check_invariants(tight, rt, prof);
    check_invariants(roomy, rr, prof);
    std::uint32_t max_tight = 0;
    for (const auto& l : rt.links) max_tight = std::max(max_tight, l.max_in_flight);
    require(max_tight == 1, "one slot per link must allow one chunk in flight");
    require(rt.makespan > rr.makespan * 1.2, "fewer slots did not cost time (" + std::to_string(rt.makespan) + " vs " +
                                                 std::to_string(rr.makespan) + ")");

    // No overlap: every step finishes before the next dispatch is issued,
    // and no expert runs while another step's exchange is in flight.
    Case serial = base;
    serial.config.options = {false, 0};
    const SimResult rs = run(serial, prof);
    check_invariants(serial, rs, prof);
    for (const auto& rank : rs.ranks) {
        const auto& p = rank.program;
        for (int i = 1; i < p.steps; ++i) {
            require(rank.ops[p.dispatch[i]].issue_begin >= rank.ops[p.combine[i - 1]].complete,
                    "serial dispatch issued before the previous step finished");
        }
    }
    // Overlap beats serial when communication and computation are comparable.
    Case one = base;
    one.config.groups = nano_nccl::planner::fixed_group_plan(2, 8, 1);
    one.config.options = {false, 0};
    const SimResult r_one = run(one, prof);
    Case over = base;
    over.config.options = {true, 3};
    const SimResult r_over = run(over, prof);
    require(r_over.makespan < r_one.makespan * 0.9, "overlap did not beat serial (" + std::to_string(r_over.makespan) +
                                                        " vs " + std::to_string(r_one.makespan) + ")");
    std::printf("  serial %.0f us, G4 L1 %.0f us, G4 L3 %.0f us, G4 L1 4x slower copies %.0f us\n", r_one.makespan,
                r0.makespan, r_over.makespan, r1.makespan);

    // Contention model: the same invariants hold, and it changes the timing
    // (here the alone tables are cheaper than the all-busy ones it replaces).
    const CostProfile cont = synthetic_profile(true);
    const SimResult rc = run(over, cont);
    check_invariants(over, rc, cont);
    Case off = over;
    off.config.contention = false;
    const SimResult rc_off = run(off, cont);
    require(rc.makespan != rc_off.makespan, "contention model had no effect");
    std::printf("  contention on %.0f us, off %.0f us\n", rc.makespan, rc_off.makespan);

    // A self <-> remote switch stall on the copy engines slows a two-rank
    // layer, whose engines alternate between the two kinds.
    CostProfile stall = prof;
    stall.scalars[cost::kTxSwitch] = 10;
    stall.scalars[cost::kRxSwitch] = 10;
    const SimResult rsw = run(base, stall);
    check_invariants(base, rsw, stall);
    require(rsw.makespan > r0.makespan + 50, "switch stalls had no effect");
}

// In-situ copy model, driven directly through the fabric: rank 1 sends rank 0
// one 1 MiB chunk; then rank 0 copies it out (rx) while it sends `tx_bytes`
// to rank 1 (tx). Remote copies cost 50 us/MiB alone and 100 us/MiB while
// the same GPU's other engine moves a remote chunk. Returns rank 0's
// {tx, rx} engine busy time for the overlapped pair.
std::pair<double, double> fluid_pair(std::size_t tx_bytes, int* gpu_changes) {
    struct Host final : HostModel {
        double clk[2] = {0, 0};
        int changes = 0;
        double& host_clock(int r) override { return clk[r]; }
        void wake(int, double) override {}
        void note_activity(int) override {}
        bool gpu_busy(int, double) const override { return false; }
        void remote_busy_changed(int rank, double) override { changes += rank == 0 ? 1 : 0; }
    } host;
    constexpr double kMiB = 1 << 20;
    WireCosts c;
    c.tx_remote = c.rx_remote = Table({{0, 0}, {kMiB, 80}});  // only its shape is used: linear
    c.tx_self = c.rx_self = Table({{0, 0}, {kMiB, 4}});
    c.tx_insitu = c.rx_insitu = Table({{0, 50}, {1, 100}});
    c.contention = c.insitu = true;
    EventQueue q;
    SimFabric f(q, host, 2, 4, 1 << 20, {c, c});
    auto header = [](int src, int dst, std::size_t bytes) {
        ChunkHeader h;
        h.src = static_cast<std::uint32_t>(src);
        h.dst = static_cast<std::uint32_t>(dst);
        h.chunk_count = 1;
        h.bytes = h.total_bytes = bytes;
        return h;
    };
    f.wire(1).write_chunk(0, 0, header(1, 0, 1 << 20), nullptr);
    q.run();
    require(std::fabs(f.tx(1).busy - 50) < 1e-9, "a lone remote copy must cost its alone time");
    host.clk[1] = q.now();
    f.wire(1).progress();  // publishes the arrival
    q.run();
    require(f.wire(0).arrived(1) == 1, "chunk did not arrive");
    host.clk[0] = q.now();
    const int before = host.changes;
    f.wire(0).copy_out(1, 0, nullptr, 1 << 20);
    f.wire(0).write_chunk(1, 0, header(0, 1, tx_bytes), nullptr);
    q.run();
    *gpu_changes = host.changes - before;
    return {f.tx(0).busy, f.rx(0).busy};
}

void test_fluid() {
    int changes = 0;
    // Both 1 MiB: each runs at half speed for all of it.
    auto [tx1, rx1] = fluid_pair(1 << 20, &changes);
    require(std::fabs(tx1 - 100) < 1e-6 && std::fabs(rx1 - 100) < 1e-6,
            "overlapped copies: tx " + std::to_string(tx1) + " rx " + std::to_string(rx1) + " us, want 100 and 100");
    require(changes >= 2, "GPU was not told about remote copies starting and ending");
    // tx 0.5 MiB: both at half speed until tx ends at +50 (25 us of work),
    // then rx runs its remaining 25 us of work at full speed: ends at +75.
    auto [tx2, rx2] = fluid_pair(1 << 19, &changes);
    require(std::fabs(tx2 - 50) < 1e-6 && std::fabs(rx2 - 75) < 1e-6,
            "rx did not speed up after tx ended: tx " + std::to_string(tx2) + " rx " + std::to_string(rx2) +
                " us, want 50 and 75");

    // Per-rank GPU slowdown: only rank 0's GPU work gets slower.
    CostProfile p = synthetic_profile(true);
    p.tables[cost::kTxRemoteInsitu] = Table({{0, 45}, {1, 90}});
    p.tables[cost::kRxRemoteInsitu] = Table({{0, 45}, {1, 90}});
    p.scalars[std::string(cost::kGpuCopySlowdown) + "@0"] = 3.0;
    p.scalars[std::string(cost::kGpuCopySlowdown) + "@1"] = 1.0;
    CostProfile quiet = p;
    quiet.scalars[std::string(cost::kGpuCopySlowdown) + "@0"] = 1.0;
    Case c = balanced_case(2, 4, 512, 512);
    c.config.groups = nano_nccl::planner::fixed_group_plan(2, 8, 4);
    c.config.options = {true, 3};
    const SimResult slow = run(c, p), base = run(c, quiet);
    check_invariants(c, slow, p);
    require(std::fabs(slow.ranks[1].gpu_busy - base.ranks[1].gpu_busy) < 1e-6, "rank 1 GPU work changed");
    require(slow.ranks[0].gpu_busy > base.ranks[0].gpu_busy * 1.05, "rank 0 GPU slowdown had no effect");
    require(slow.ranks[0].gpu_busy <= base.ranks[0].gpu_busy * 3.0 + 1e-6, "GPU slowed beyond its factor");
    std::printf("  in-situ GPU busy rank 0 %.0f us (quiet %.0f), rank 1 %.0f us\n", slow.ranks[0].gpu_busy,
                base.ranks[0].gpu_busy, slow.ranks[1].gpu_busy);
}

}  // namespace

int main() {
    check("program_structure", test_program);
    check("formats", test_formats);
    check("random_worlds_invariants", [] { test_random(400); });
    check("timing_dependencies", test_timing);
    check("insitu_copy_model", test_fluid);
    std::printf("sim_adapter=%s\n", g_failed == 0 ? "PASS" : "FAIL");
    return g_failed == 0 ? 0 : 1;
}
