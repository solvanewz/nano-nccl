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
    t(cost::kTxRemote, {{4096, 1.0}, {1 << 20, 80}});  // unused by NetFabric
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
};

Run run_layer(const SimLayerConfig& c, const std::vector<std::vector<std::int64_t>>& counts,
              const std::vector<int>& tokens, const CostProfile& p, double gbps, double latency_us) {
    EventQueue q;
    AnalyticalNetwork net(q, c.world, gbps, latency_us);
    AstraScheduler sched(net.endpoint(0));
    NetFabric* fabric = nullptr;
    auto factory = [&](Scheduler& s, HostModel& host, int world, int slots, std::size_t slot_bytes,
                       std::vector<WireCosts> costs) -> std::unique_ptr<Fabric> {
        auto f = std::make_unique<NetFabric>(s, host, world, slots, slot_bytes, std::move(costs), net.endpoints());
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
    return r;
}

void check_run(const SimLayerConfig& c, const Run& r, double gbps, double latency_us) {
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
            double out_bytes = 0;
            for (int s = 0; s < W; ++s) out_bytes += s == d ? 0.0 : static_cast<double>(in.send_bytes[s]);
            require(at(Condition::kSendBufferReusable) >= rk.ops[i].issue_begin + out_bytes / bytes_per_us - 1e-6,
                    "send buffer of op " + std::to_string(i) + " reusable before its bytes were sent");
            for (int s = 0; s < W; ++s) {
                if (s == d || in.recv_bytes[s] == 0) continue;
                const double cross = static_cast<double>(in.recv_bytes[s]) / bytes_per_us + latency_us;
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
        c.groups = nano_nccl::planner::fixed_group_plan(c.world, c.num_experts, groups);
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
        try {
            const Run r = run_layer(c, counts, tokens, p, gbps, lat);
            check_run(c, r, gbps, lat);
        } catch (const std::exception& e) {
            throw std::runtime_error("trial " + std::to_string(t) + " (W=" + std::to_string(c.world) + " groups=" +
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
    c.groups = nano_nccl::planner::fixed_group_plan(4, 16, 4);
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
    c.groups = nano_nccl::planner::fixed_group_plan(2, 2, 1);
    c.options = {false, 0};
    c.slots_per_link = 1;
    c.slot_bytes = 4096;
    std::vector<std::vector<std::int64_t>> counts(2, std::vector<std::int64_t>(2, 64));
    std::vector<int> tokens(2, 64);
    const double latency = 50;
    const Run r = run_layer(c, counts, tokens, p, 100, latency);
    check_run(c, r, 100, latency);
    const auto& rk = r.res.ranks[1];
    const int d = rk.program.dispatch[0];
    const auto& in = nano_nccl::exec::exchange_layout(rk.program, d, rk.plan);
    const std::size_t chunks = (in.recv_bytes[0] + c.slot_bytes - 1) / c.slot_bytes;
    require(chunks >= 4, "test needs several chunks");
    const double arrived = rk.ops[d].condition[static_cast<int>(Condition::kPayloadArrived)];
    const double bound = r.res.ranks[0].ops[d].issue_begin + static_cast<double>(2 * chunks - 1) * latency;
    require(arrived >= bound, "chunks did not wait for credits: arrived at " + std::to_string(arrived) +
                                  " us, bound " + std::to_string(bound) + " us");
}

}  // namespace

int main() {
    check("send_completion_callbacks", test_callbacks);
    check("random_worlds", [] { test_random(200); });
    check("timing", test_timing);
    check("credit_round_trips", test_credits);
    std::printf("simai_bridge=%s\n", g_failed == 0 ? "PASS" : "FAIL");
    return g_failed == 0 ? 0 : 1;
}
