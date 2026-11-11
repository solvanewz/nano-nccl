// CPU test of the SimAI bridge (sim_adapter/simai): the layer model and the
// real PeerExchanger running over NetFabric, whose remote chunks and credits
// are messages through SimAI's AstraNetworkAPI, here backed by the analytical
// network on the simulator's EventQueue and driven through AstraScheduler
// (the same path SimAI's ns-3 backend takes in nano_ep_ns3).
//
// 1. Minimal send -> completion callbacks: a message's send handler fires when
//    it left the sender, its receive handler when it arrived, whether the
//    receive was posted before or after the arrival.
// 2. Whole layers on random worlds: every rank retires; completion events of
//    each exchange in protocol order; no link has more chunks in flight than
//    slots; every message sent completed at both ends; a payload never
//    arrives before its bytes could cross the network.
// 3. Timing: a slower network delays the experts and the layer, and the
//    results are deterministic; with one slot per link, every chunk after the
//    first waits for the previous one's credit to come back through the
//    network (a data and a credit latency).
#include "analytical_network.h"
#include "astra_scheduler.h"
#include "net_fabric.h"
#include "planner/group_plan.h"
#include "sim_layer.h"

#include <cmath>
#include <cstdio>
#include <exception>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using nano_nccl::exec::Condition;
using nano_nccl::exec::OpKind;
using nano_nccl::sim::CostProfile;
using nano_nccl::sim::EventQueue;
using nano_nccl::sim::Fabric;
using nano_nccl::sim::HostModel;
using nano_nccl::sim::LayerSimulation;
using nano_nccl::sim::Scheduler;
using nano_nccl::sim::SimLayerConfig;
using nano_nccl::sim::SimResult;
using nano_nccl::sim::Table;
using nano_nccl::sim::WireCosts;
using nano_nccl::sim::simai::AnalyticalNetwork;
using nano_nccl::sim::simai::AstraScheduler;
using nano_nccl::sim::simai::NetConfig;
using nano_nccl::sim::simai::NetFabric;
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

CostProfile profile() {
    CostProfile p;
    p.key = nano_nccl::exec::ProfileKey{2, 8, 2, 256, 512, 512, "zipf", 0, "bf16", "ffn", "rdma", 4, 65536};
    auto t = [&](const char* name, std::vector<std::pair<double, double>> pts) { p.tables[name] = Table(pts); };
    t(cost::kRoute, {{0, 5}, {100000, 105}});
    t(cost::kPermute, {{0, 1}, {10000, 11}});
    t(cost::kExpertFfn, {{1, 10}, {1000, 30}, {10000, 200}});
    t(cost::kExpertProbe, {{0, 1}, {10000, 5}});
    t(cost::kReduce, {{0, 2}, {10000, 12}});
    t(cost::kTxRemote, {{4096, 1.0}, {1 << 20, 80}});  // NetFabric: host-staged only
    t(cost::kRxRemote, {{4096, 1.0}, {1 << 20, 80}});
    t(cost::kTxSelf, {{4096, 0.5}, {1 << 20, 4}});
    t(cost::kRxSelf, {{4096, 0.5}, {1 << 20, 4}});
    t(cost::kHostExpertIssue, {{0, 5}, {8, 40}});
    t(cost::kHostPlan, {{1, 10}, {8, 30}});
    p.scalars = {{cost::kHostRouteIssue, 10}, {cost::kHostReduceIssue, 3}, {cost::kHostExchangeIssue, 1},
                 {cost::kHostPost, 2},        {cost::kHostCopyOut, 2},     {cost::kHostPoll, 0.5},
                 {cost::kLaunchLatency, 4},   {cost::kCopyLatency, 2},     {cost::kCreditLatency, 0.2},
                 {cost::kArrivalLatency, 0.2}};
    return p;
}

// 1. Minimal send -> completion callbacks through the interface.
void test_callbacks() {
    EventQueue q;
    AnalyticalNetwork net(q, 2, 80.0, 2.0);  // 10 bytes/ns, 2 us
    AstraScheduler sched(net.endpoint(0));
    struct Log {
        double sent = -1, received = -1, late_received = -1;
        Scheduler* s;
    } log{-1, -1, -1, &sched};
    struct Arg {
        Log* log;
        int which;
    };
    auto handler = [](void* p) {
        Arg* a = static_cast<Arg*>(p);
        const double t = a->log->s->now();
        (a->which == 0 ? a->log->sent : a->which == 1 ? a->log->received : a->log->late_received) = t;
        delete a;
    };
    AstraSim::sim_request req{};
    // 1 MB at 10 GB/s: 100 us on the wire, arrival 2 us later.
    net.endpoint(1).sim_recv(nullptr, 1000000, AstraSim::UINT8, 0, 7, &req, handler, new Arg{&log, 1});
    net.endpoint(0).sim_send(nullptr, 1000000, AstraSim::UINT8, 1, 7, &req, handler, new Arg{&log, 0});
    // A second message whose receive is posted only after it arrived.
    net.endpoint(0).sim_send(nullptr, 1000, AstraSim::UINT8, 1, 9, &req, [](void* p) { delete static_cast<Arg*>(p); },
                             new Arg{&log, 3});
    sched.schedule_at(500, [&] {
        net.endpoint(1).sim_recv(nullptr, 1000, AstraSim::UINT8, 0, 9, &req, handler, new Arg{&log, 2});
    });
    q.run();
    require(std::fabs(log.sent - 100) < 1e-9, "send handler at " + std::to_string(log.sent) + " us, want 100");
    require(std::fabs(log.received - 102) < 1e-9, "receive handler at " + std::to_string(log.received) + " us, want 102");
    require(std::fabs(log.late_received - 500) < 1e-9, "late-posted receive did not complete at once");
}

struct Run {
    SimResult res;
    std::uint64_t sent = 0, send_done = 0, recv_done = 0;
    std::vector<std::uint64_t> sent_by;
    std::vector<double> proxy_busy_until;
};

Run run_layer(const SimLayerConfig& c, const std::vector<std::vector<std::int64_t>>& counts,
              const std::vector<int>& tokens, const CostProfile& p, double gbps, double latency_us,
              NetConfig config = {}) {
    EventQueue q;
    AnalyticalNetwork net(q, c.world, gbps, latency_us);
    AstraScheduler sched(net.endpoint(0));
    NetFabric* fabric = nullptr;
    auto factory = [&](Scheduler& s, HostModel& host, int world, int slots, std::size_t slot_bytes,
                       std::vector<WireCosts> costs) -> std::unique_ptr<Fabric> {
        auto f = std::make_unique<NetFabric>(s, host, world, slots, slot_bytes, std::move(costs), net.endpoints(),
                                             config);
        fabric = f.get();
        return f;
    };
    LayerSimulation sim(c, counts, tokens, p, sched, factory);
    bool all_retired = false;
    sim.start([&] { all_retired = true; });
    const std::uint64_t events = q.run();
    Run r;
    r.res = sim.finish(events);
    require(all_retired, "on_all_retired was not called");
    r.sent = fabric->messages_sent();
    r.send_done = fabric->sends_completed();
    r.recv_done = fabric->receives_completed();
    for (int k = 0; k < c.world; ++k) {
        r.sent_by.push_back(fabric->messages_sent_by(k));
        r.proxy_busy_until.push_back(fabric->proxy_busy_until(k));
    }
    return r;
}

// `staged` (NetConfig::host_staged) needs the profile for the staging cost.
// staged: remote sends go out after a staging copy (tx_remote, or tx_self with
// device_staging); same-node peers of nodes of ranks_per_node are direct.
void check_run(const SimLayerConfig& c, const Run& r, double gbps, double latency_us, bool staged = false,
               const CostProfile* p = nullptr, int ranks_per_node = 0, bool device_staging = false) {
    const int W = c.world;
    require(r.sent == r.send_done && r.sent == r.recv_done,
            "messages sent " + std::to_string(r.sent) + ", sends completed " + std::to_string(r.send_done) +
                ", receives completed " + std::to_string(r.recv_done));
    for (const auto& l : r.res.links) {
        require(l.max_in_flight <= static_cast<std::uint32_t>(c.slots_per_link), "more chunks in flight than slots");
    }
    const double bytes_per_us = gbps * 1e3 / 8.0;
    for (int d = 0; d < W; ++d) {
        const auto& rk = r.res.ranks[d];
        require(rk.ops[rk.program.retire].issue_begin >= 0, "rank did not retire");
        for (std::size_t i = 0; i < rk.program.ops.size(); ++i) {
            const auto kind = rk.program.ops[i].kind;
            if (kind != OpKind::kDispatch && kind != OpKind::kCombine && kind != OpKind::kCounts) continue;
            const auto& cond = rk.ops[i].condition;
            auto at = [&](Condition x) { return cond[static_cast<int>(x)]; };
            require(at(Condition::kSubmitted) <= at(Condition::kSendBufferReusable), "submitted after send reusable");
            require(at(Condition::kPayloadArrived) <= at(Condition::kConsumerReadable), "readable before arrived");
            require(at(Condition::kConsumerReadable) <= at(Condition::kReceiveSlotsReleased), "released before readable");
            if (kind == OpKind::kCounts) continue;
            // A remote payload cannot arrive before its bytes crossed the wire.
            const auto& in = nano_nccl::exec::exchange_layout(rk.program, static_cast<int>(i), rk.plan);
            // The send buffer is reusable only once every remote byte left
            // this rank's port.
            // (Staged: once it was copied to the staging slots, earlier.)
            double out_bytes = 0;
            for (int s = 0; s < W; ++s) out_bytes += s == d ? 0.0 : static_cast<double>(in.send_bytes[s]);
            if (!staged) {
                require(at(Condition::kSendBufferReusable) >= rk.ops[i].issue_begin + out_bytes / bytes_per_us - 1e-6,
                        "send buffer of op " + std::to_string(i) + " reusable before its bytes were sent");
            }
            for (int s = 0; s < W; ++s) {
                if (s == d || in.recv_bytes[s] == 0) continue;
                double cross = static_cast<double>(in.recv_bytes[s]) / bytes_per_us + latency_us;
                const bool same_node = ranks_per_node > 1 && s / ranks_per_node == d / ranks_per_node;
                if (staged && !same_node) {
                    // The first chunk is staged before anything is sent.
                    const double first = static_cast<double>(std::min(in.recv_bytes[s], c.slot_bytes));
                    const WireCosts wc = WireCosts::from_profile(*p, false, s);
                    cross += device_staging ? wc.tx_self(first) : wc.tx_remote(first);
                }
                require(at(Condition::kPayloadArrived) >= r.res.ranks[s].ops[i].issue_begin + cross - 1e-6,
                        "payload " + std::to_string(s) + "->" + std::to_string(d) + " of op " + std::to_string(i) +
                            " arrived before it could cross the network");
            }
        }
    }
}

// 2. Random worlds.
void test_random(int trials) {
    std::mt19937 rng(20261101);
    const CostProfile p = profile();
    for (int t = 0; t < trials; ++t) {
        SimLayerConfig c;
        const int worlds[] = {2, 3, 4, 8};
        c.world = worlds[rng() % 4];
        const int epr = 1 + static_cast<int>(rng() % 3);
        c.num_experts = c.world * epr;
        c.top_k = 2;
        c.hidden = 256;
        c.ffn = 512;
        const int groups = 1 + static_cast<int>(rng() % epr);  // at most experts per rank
        c.plan = nano_nccl::exec::PlanSource::ahead(nano_nccl::planner::fixed_group_plan(c.world, c.num_experts, groups));
        const bool overlap = rng() % 2 == 0;
        c.options = {overlap, overlap ? static_cast<int>(rng() % 3) : 0};
        c.slots_per_link = 1 + static_cast<int>(rng() % 4);
        c.slot_bytes = std::size_t{1} << (10 + rng() % 7);
        std::vector<std::vector<std::int64_t>> counts(c.world, std::vector<std::int64_t>(c.num_experts));
        std::vector<int> tokens(c.world);
        for (int r = 0; r < c.world; ++r) {
            int sum = 0;
            for (auto& n : counts[r]) {
                n = rng() % 4 == 0 ? 0 : static_cast<std::int64_t>(rng() % 64);
                sum += static_cast<int>(n);
            }
            tokens[r] = (sum + 1) / 2;
        }
        const double gbps = 25.0 * (1 + rng() % 16), lat = 0.5 * (rng() % 8);
        NetConfig config;
        config.host_staged = rng() % 2 == 0;
        try {
            const Run r = run_layer(c, counts, tokens, p, gbps, lat, config);
            check_run(c, r, gbps, lat, config.host_staged, &p);
        } catch (const std::exception& e) {
            throw std::runtime_error("trial " + std::to_string(t) + " (W=" + std::to_string(c.world) +
                                     (config.host_staged ? " staged" : "") + " groups=" +
                                     std::to_string(groups) + " slots=" + std::to_string(c.slots_per_link) +
                                     " slot_bytes=" + std::to_string(c.slot_bytes) + "): " + e.what());
        }
    }
}

// 3. Network speed matters, and runs are deterministic.
void test_timing() {
    const CostProfile p = profile();
    SimLayerConfig c;
    c.world = 4;
    c.num_experts = 16;
    c.top_k = 2;
    c.hidden = 1024;
    c.ffn = 512;
    c.plan = nano_nccl::exec::PlanSource::ahead(nano_nccl::planner::fixed_group_plan(4, 16, 4));
    c.options = {true, 1};
    c.slots_per_link = 4;
    c.slot_bytes = 65536;
    std::vector<std::vector<std::int64_t>> counts(4, std::vector<std::int64_t>(16, 256));
    std::vector<int> tokens(4, 256 * 16 / 2);
    const Run fast = run_layer(c, counts, tokens, p, 200, 1), slow = run_layer(c, counts, tokens, p, 25, 1);
    const Run again = run_layer(c, counts, tokens, p, 200, 1);
    check_run(c, fast, 200, 1);
    check_run(c, slow, 25, 1);
    require(fast.res.makespan == again.res.makespan, "not deterministic");
    const auto& r0 = fast.res.ranks[0], s0 = slow.res.ranks[0];
    const int e3 = r0.program.expert[3];
    require(s0.ops[e3].issue_begin > r0.ops[e3].issue_begin + 100, "slower network did not delay the experts");
    require(slow.res.makespan > fast.res.makespan * 1.5, "slower network did not slow the layer");
    std::printf("  W=4 G4 L1: 200 Gb/s %.0f us, 25 Gb/s %.0f us; %llu messages\n", fast.res.makespan,
                slow.res.makespan, static_cast<unsigned long long>(fast.sent));
}

// NetConfig::post_overhead: each rank's proxy hands one message at a time to
// the NIC, so a rank that sends n messages needs at least n * overhead.
void test_post_overhead() {
    const CostProfile p = profile();
    SimLayerConfig c;
    c.world = 4;
    c.num_experts = 16;
    c.top_k = 2;
    c.hidden = 1024;
    c.ffn = 512;
    c.plan = nano_nccl::exec::PlanSource::ahead(nano_nccl::planner::fixed_group_plan(4, 16, 4));
    c.options = {true, 1};
    c.slots_per_link = 4;
    c.slot_bytes = 65536;
    std::vector<std::vector<std::int64_t>> counts(4, std::vector<std::int64_t>(16, 256));
    std::vector<int> tokens(4, 256 * 16 / 2);
    const Run base = run_layer(c, counts, tokens, p, 200, 1);
    NetConfig zero;
    zero.post_overhead = 0;
    require(run_layer(c, counts, tokens, p, 200, 1, zero).res.makespan == base.res.makespan,
            "zero overhead changes the run");
    NetConfig slow;
    slow.post_overhead = 20;
    const Run r = run_layer(c, counts, tokens, p, 200, 1, slow);
    check_run(c, r, 200, 1);
    // Serialized: the n-th message leaves the proxy no earlier than n
    // overheads (some trailing credits go out after the layer retired, so
    // the makespan itself is not bounded by it).
    for (int k = 0; k < c.world; ++k) {
        require(r.proxy_busy_until[k] >= static_cast<double>(r.sent_by[k]) * slow.post_overhead - 1e-6,
                "proxy of rank " + std::to_string(k) + " does not serialize its messages");
    }
    require(r.res.makespan > base.res.makespan, "overhead did not slow the layer");
    std::printf("  post overhead 0: %.0f us, 20 us/message: %.0f us (%llu messages from rank 0)\n",
                base.res.makespan, r.res.makespan, static_cast<unsigned long long>(r.sent_by[0]));
}

// One slot per link and a long latency: a message of n chunks needs at least
// n data latencies and n - 1 credit latencies.
void test_credits() {
    const CostProfile p = profile();
    SimLayerConfig c;
    c.world = 2;
    c.num_experts = 2;
    c.top_k = 2;
    c.hidden = 256;
    c.ffn = 512;
    c.plan = nano_nccl::exec::PlanSource::ahead(nano_nccl::planner::fixed_group_plan(2, 2, 1));
    c.options = {false, 0};
    c.slots_per_link = 1;
    c.slot_bytes = 4096;
    std::vector<std::vector<std::int64_t>> counts(2, std::vector<std::int64_t>(2, 64));
    std::vector<int> tokens(2, 64);
    const double latency = 50;
    for (bool staged : {false, true}) {
        NetConfig config;
        config.host_staged = staged;
        const Run r = run_layer(c, counts, tokens, p, 100, latency, config);
        check_run(c, r, 100, latency, staged, &p);
        const auto& rk = r.res.ranks[1];
        const int d = rk.program.dispatch[0];
        const auto& in = nano_nccl::exec::exchange_layout(rk.program, d, rk.plan);
        const std::size_t chunks = (in.recv_bytes[0] + c.slot_bytes - 1) / c.slot_bytes;
        require(chunks >= 4, "test needs several chunks");
        const double arrived = rk.ops[d].condition[static_cast<int>(Condition::kPayloadArrived)];
        const double bound = r.res.ranks[0].ops[d].issue_begin + static_cast<double>(2 * chunks - 1) * latency;
        require(arrived >= bound, std::string(staged ? "staged: " : "") + "chunks did not wait for credits: arrived at " +
                                      std::to_string(arrived) + " us, bound " + std::to_string(bound) + " us");
    }
}

// 5. Host-staged sends (RdmaWire semantics) against GPUDirect writes, on a
// slow network: the staged send buffer is free once it is copied, long before
// its bytes could cross the wire, and copy-outs come from host memory (more
// rx engine time). The layer is not necessarily slower: freeing send buffers
// early can let later steps start sooner.
void test_staged() {
    const CostProfile p = profile();
    SimLayerConfig c;
    c.world = 4;
    c.num_experts = 16;
    c.top_k = 2;
    c.hidden = 1024;
    c.ffn = 512;
    c.plan = nano_nccl::exec::PlanSource::ahead(nano_nccl::planner::fixed_group_plan(4, 16, 4));
    c.options = {true, 1};
    c.slots_per_link = 4;
    c.slot_bytes = 65536;
    std::vector<std::vector<std::int64_t>> counts(4, std::vector<std::int64_t>(16, 256));
    std::vector<int> tokens(4, 256 * 16 / 2);
    const double gbps = 25, latency = 1;
    NetConfig staged_config;
    staged_config.host_staged = true;
    const Run direct = run_layer(c, counts, tokens, p, gbps, latency);
    const Run staged = run_layer(c, counts, tokens, p, gbps, latency, staged_config);
    check_run(c, direct, gbps, latency);
    check_run(c, staged, gbps, latency, true, &p);
    double rx_direct = 0, rx_staged = 0;
    for (int d = 0; d < c.world; ++d) {
        rx_direct += direct.res.rx[d].busy;
        rx_staged += staged.res.rx[d].busy;
    }
    require(rx_staged > rx_direct * 2, "staged copy-outs did not read from host memory (rx busy " +
                                           std::to_string(rx_staged) + " vs " + std::to_string(rx_direct) + " us)");
    const double bytes_per_us = gbps * 1e3 / 8.0;
    int early = 0;
    for (int d = 0; d < c.world; ++d) {
        const auto& rk = staged.res.ranks[d];
        for (int i : rk.program.dispatch) {
            const auto& in = nano_nccl::exec::exchange_layout(rk.program, i, rk.plan);
            double out_bytes = 0;
            for (int s = 0; s < c.world; ++s) out_bytes += s == d ? 0.0 : static_cast<double>(in.send_bytes[s]);
            const double reusable = rk.ops[i].condition[static_cast<int>(Condition::kSendBufferReusable)];
            if (reusable < rk.ops[i].issue_begin + out_bytes / bytes_per_us) ++early;
        }
    }
    require(early > 0, "no staged send buffer was free before its bytes could be sent");
    std::printf("  W=4 G4 L1 25 Gb/s: GPUDirect %.0f us, host-staged %.0f us; %d dispatches freed early\n",
                direct.res.makespan, staged.res.makespan, early);
}

// Dispatches whose send buffer was reusable before their remote bytes could
// have crossed the link (only a staging copy makes that possible).
int early_frees(const SimLayerConfig& c, const Run& run, double gbps) {
    const double bytes_per_us = gbps * 1e3 / 8.0;
    int early = 0;
    for (int d = 0; d < c.world; ++d) {
        const auto& rk = run.res.ranks[d];
        for (int i : rk.program.dispatch) {
            const auto& in = nano_nccl::exec::exchange_layout(rk.program, i, rk.plan);
            double out_bytes = 0;
            for (int s = 0; s < c.world; ++s) out_bytes += s == d ? 0.0 : static_cast<double>(in.send_bytes[s]);
            const double reusable = rk.ops[i].condition[static_cast<int>(Condition::kSendBufferReusable)];
            if (reusable < rk.ops[i].issue_begin + out_bytes / bytes_per_us) ++early;
        }
    }
    return early;
}

// HybridWire semantics: remote chunks staged in device memory (GPUDirect),
// same-node chunks written directly.
void test_hybrid() {
    const CostProfile p = profile();
    SimLayerConfig c;
    c.world = 4;
    c.num_experts = 16;
    c.top_k = 2;
    c.hidden = 1024;
    c.ffn = 512;
    c.plan = nano_nccl::exec::PlanSource::ahead(nano_nccl::planner::fixed_group_plan(4, 16, 4));
    c.options = {true, 1};
    c.slots_per_link = 4;
    c.slot_bytes = 65536;
    std::vector<std::vector<std::int64_t>> counts(4, std::vector<std::int64_t>(16, 256));
    std::vector<int> tokens(4, 256 * 16 / 2);
    const double gbps = 25, latency = 1;
    NetConfig staged_config, pairs, one_node;
    staged_config.host_staged = true;
    pairs.gdr_staging = true;
    pairs.ranks_per_node = 2;
    one_node.gdr_staging = true;
    one_node.ranks_per_node = 4;
    const Run direct = run_layer(c, counts, tokens, p, gbps, latency);
    const Run staged = run_layer(c, counts, tokens, p, gbps, latency, staged_config);
    const Run hybrid = run_layer(c, counts, tokens, p, gbps, latency, pairs);
    const Run local = run_layer(c, counts, tokens, p, gbps, latency, one_node);
    check_run(c, hybrid, gbps, latency, true, &p, 2, true);
    check_run(c, local, gbps, latency);
    double rx_staged = 0, rx_hybrid = 0;
    for (int d = 0; d < c.world; ++d) {
        rx_staged += staged.res.rx[d].busy;
        rx_hybrid += hybrid.res.rx[d].busy;
    }
    require(rx_hybrid * 2 < rx_staged, "GPUDirect copy-outs read host memory (rx busy " + std::to_string(rx_hybrid) +
                                           " vs host-staged " + std::to_string(rx_staged) + " us)");
    const int early = early_frees(c, hybrid, gbps);
    require(early > 0, "no GPUDirect remote send buffer was freed by its staging copy");
    // One node: every link is a direct write, exactly the plain path.
    require(early_frees(c, local, gbps) == 0, "a same-node chunk was staged");
    require(local.res.makespan == direct.res.makespan,
            "one node differs from direct writes: " + std::to_string(local.res.makespan) + " vs " +
                std::to_string(direct.res.makespan) + " us");
    std::printf("  W=4 G4 L1 25 Gb/s: direct %.0f us, nodes of 2 (GPUDirect staging) %.0f us; %d dispatches freed "
                "early\n",
                direct.res.makespan, hybrid.res.makespan, early);
}

}  // namespace

int main() {
    check("send_completion_callbacks", test_callbacks);
    check("random_worlds", [] { test_random(200); });
    check("timing", test_timing);
    check("credit_round_trips", test_credits);
    check("post_overhead", test_post_overhead);
    check("staged_vs_gpudirect", test_staged);
    check("hybrid_semantics", test_hybrid);
    std::printf("simai_bridge=%s\n", g_failed == 0 ? "PASS" : "FAIL");
    return g_failed == 0 ? 0 : 1;
}
