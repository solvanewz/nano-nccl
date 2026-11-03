// EP layer simulation on SimAI's ns-3 network backend.
//
// The layer model, the real PeerExchanger and NetFabric run on ns-3's event
// loop (AstraScheduler); every remote chunk and every credit is an RDMA flow
// in SimAI's ns-3 (RdmaClient over the qbb/RdmaHw stack that SimAI's own
// front end uses), set up by SimAI's network_frontend/ns3/common.h. This file
// implements AstraSim::AstraNetworkAPI on that network itself, because
// SimAI's ASTRASimNetwork ties its callbacks to astra-sim's Sys objects.
//
//   nano_ep_ns3 --selftest [--world W --gbps G --latency-us L --out DIR]
//       one 1 MiB message rank 0 -> 1: prints when its send and its receive
//       completed (the minimal send -> completion callback case).
//   nano_ep_ns3 --profile P [--trace T | synthetic options as nano_ep_sim]
//               [--world W --gbps G --latency-us L --out DIR]
//               [--backend ns3|analytical] [--staged]
//       one layer invocation; host and GPU costs from the profile, network
//       from ns-3. The topology is a star: W GPU nodes on one switch, links of
//       G Gb/s and L us; the ns-3 inputs and outputs go to DIR. With
//       --backend analytical the same layer runs over the analytical network
//       (one port per node at G Gb/s, 2 L per message) on the simulator's
//       own event loop, as a cross-check. --staged models RdmaWire's
//       host-pinned path (staging copy, then send) instead of GPUDirect
//       writes (NetConfig::host_staged).
#include "analytical_network.h"
#include "astra_scheduler.h"
#include "cost_profile.h"
#include "ep/synthetic_routing.h"
#include "exec/exec_trace.h"
#include "net_fabric.h"
#include "planner/group_plan.h"
#include "sim_layer.h"

#include "astra-sim/system/Common.hh"
// SimAI's ns-3 network setup (ReadConf, SetConfig, SetupNetwork and its
// globals); it brings `using namespace ns3` and `std`, so it comes last.
#include "common.h"

#include <sys/stat.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

namespace nsim = nano_nccl::sim;

// ---- AstraNetworkAPI on SimAI's ns-3 network --------------------------------

using Handler = void (*)(void*);
struct Flow {
    int tag;
    std::uint64_t bytes;
    Handler on_sent;
    void* sent_arg;
    bool sent = false, received = false;
};
struct Posted {
    std::uint64_t count;
    Handler fn;
    void* arg;
};
std::map<std::tuple<uint32_t, uint32_t, uint32_t>, Flow> g_flows;          // sport, src, dst
std::map<std::tuple<int, int, int>, Posted> g_posted;                      // src, dst, tag
std::map<std::tuple<int, int, int>, std::uint64_t> g_unexpected;           // src, dst, tag
int g_world = 0;
std::uint64_t g_sent_on_ack = 0;  // messages whose send completion came from their acknowledgement

void deliver(int src, int dst, int tag, std::uint64_t bytes) {
    const auto k = std::make_tuple(src, dst, tag);
    auto p = g_posted.find(k);
    if (p == g_posted.end()) {
        if (!g_unexpected.emplace(k, bytes).second) throw std::logic_error("ns3: duplicate message");
        return;
    }
    if (p->second.count != bytes) throw std::logic_error("ns3: receive size mismatch");
    const Posted got = p->second;
    g_posted.erase(p);
    got.fn(got.arg);
}

void finish_flow(std::map<std::tuple<uint32_t, uint32_t, uint32_t>, Flow>::iterator it) {
    if (it->second.sent && it->second.received) g_flows.erase(it);
}

// SimAI's send-complete hook (qbb-net-device.cc) fires after the wire time
// of every packet a host NIC sends whose size is in (60, 9000) bytes; with
// SimAI.conf's 9000-byte payloads that is a message's last, partial packet.
// A message whose last packet is full, or that is a single tiny packet, gets
// no hook at all (seen as a hang at W=8, E=64): its full acknowledgement
// (qp_finish) then implies it was sent. The bytes-left check keeps any other
// packet's hook from completing a message early.
void mark_sent(std::map<std::tuple<uint32_t, uint32_t, uint32_t>, Flow>::iterator it) {
    it->second.sent = true;
    const Handler fn = it->second.on_sent;
    void* arg = it->second.sent_arg;
    finish_flow(it);
    fn(arg);
}

// QpComplete (sender side, every byte acknowledged): the receiver has the
// whole message.
void qp_finish(FILE*, Ptr<RdmaQueuePair> q) {
    const uint32_t sid = ip_to_node_id(q->sip), did = ip_to_node_id(q->dip);
    Ptr<RdmaDriver> rdma = n.Get(did)->GetObject<RdmaDriver>();
    rdma->m_rdma->DeleteRxQp(q->sip.Get(), q->m_pg, q->sport);
    const auto key = std::make_tuple(static_cast<uint32_t>(q->sport), sid, did);
    auto it = g_flows.find(key);
    if (it == g_flows.end()) throw std::logic_error("ns3: completion of an unknown flow");
    const Flow f = it->second;
    if (!f.sent) {
        ++g_sent_on_ack;
        mark_sent(it);
        it = g_flows.find(key);
    }
    it->second.received = true;
    finish_flow(it);
    deliver(static_cast<int>(sid), static_cast<int>(did), f.tag, f.bytes);
}

void send_finish(FILE*, Ptr<RdmaQueuePair> q) {
    if (q->GetBytesLeft() != 0) return;  // not the message's last packet
    const uint32_t sid = ip_to_node_id(q->sip), did = ip_to_node_id(q->dip);
    auto it = g_flows.find(std::make_tuple(static_cast<uint32_t>(q->sport), sid, did));
    if (it == g_flows.end() || it->second.sent) return;
    mark_sent(it);
}

class Ns3Network final : public AstraSim::AstraNetworkAPI {
public:
    explicit Ns3Network(int rank) : AstraNetworkAPI(rank) {}
    BackendType get_backend_type() override { return BackendType::NS3; }
    int sim_comm_size(AstraSim::sim_comm, int* size) override {
        *size = g_world;
        return 0;
    }
    int sim_finish() override { return 0; }
    double sim_time_resolution() override { return 0.001; }  // ns per tick (ps)
    int sim_init(AstraSim::AstraMemoryAPI*) override { return 0; }
    AstraSim::timespec_t sim_get_time() override {
        AstraSim::timespec_t t;
        t.time_res = AstraSim::NS;
        t.time_val = static_cast<double>(Simulator::Now().GetPicoSeconds()) / 1000.0;
        return t;
    }
    void sim_schedule(AstraSim::timespec_t delta, void (*fn)(void*), void* arg) override {
        Simulator::Schedule(PicoSeconds(static_cast<uint64_t>(std::llround(delta.time_val * 1000.0))), fn, arg);
    }
    int sim_send(void*, uint64_t count, int, int dst, int tag, AstraSim::sim_request*, void (*fn)(void*),
                 void* arg) override {
        const int src = rank;
        uint16_t& next = portNumber[src][dst];
        if (next == 65535) throw std::runtime_error("ns3: out of ports for one rank pair");
        const uint16_t port = next++;
        g_flows[std::make_tuple(static_cast<uint32_t>(port), static_cast<uint32_t>(src), static_cast<uint32_t>(dst))] =
            Flow{tag, count, fn, arg};
        RdmaClientHelper client(3, serverAddress[src], serverAddress[dst], port, 100, count,
                                has_win ? (global_t == 1 ? maxBdp : pairBdp[n.Get(src)][n.Get(dst)]) : 0,
                                global_t == 1 ? maxRtt : pairRtt[src][dst], nullptr, nullptr, tag, src, dst);
        ApplicationContainer app = client.Install(n.Get(src));
        app.Start(Time(0));
        return 0;
    }
    int sim_recv(void*, uint64_t count, int, int src, int tag, AstraSim::sim_request*, void (*fn)(void*),
                 void* arg) override {
        const auto k = std::make_tuple(src, rank, tag);
        auto u = g_unexpected.find(k);
        if (u != g_unexpected.end()) {
            if (u->second != count) throw std::logic_error("ns3: receive size mismatch");
            g_unexpected.erase(u);
            fn(arg);
            return 0;
        }
        if (!g_posted.emplace(k, Posted{count, fn, arg}).second) throw std::logic_error("ns3: receive posted twice");
        return 0;
    }
};

// ---- inputs -----------------------------------------------------------------

struct Args {
    std::string profile, trace, out = "nano_ep_ns3_out", backend = "ns3";
    bool selftest = false;
    int world = 2, experts = 16, top_k = 2, hidden = 2048, ffn = 1408, tokens = 4096;
    double skew = 0.0;
    std::string scheduler = "fixed";
    std::string workload = "zipf";
    bool from_counts = false;
    int groups = 1, lookahead = 1;
    bool overlap = true;
    int slots = 4;
    std::size_t slot_bytes = std::size_t{1} << 20;
    std::string mode = "ffn";
    int gbps = 100;
    double latency_us = 1.0;
    bool staged = false;  // host-staged sends (RdmaWire semantics), else GPUDirect writes
    double max_sim_ms = 1000;  // ns-3: stop (and report a deadlock) after this much simulated time
};

// SimAI's example ns-3 configuration (inputs/config/SimAI.conf) with the
// output files moved into `dir` and tracing off.
void write_inputs(const Args& a, std::string& topo, std::string& conf) {
    mkdir(a.out.c_str(), 0755);
    topo = a.out + "/topo.txt";
    conf = a.out + "/ns3.conf";
    const int sw = a.world;
    {
        std::ofstream t(topo);
        // nodes, gpus per server, nvswitches, switches, links, gpu type
        t << a.world + 1 << " 1 0 1 " << a.world << " NONE\n" << sw << "\n";
        char delay[64];
        std::snprintf(delay, sizeof delay, "%.6fms", a.latency_us / 1000.0);
        for (int r = 0; r < a.world; ++r) t << r << " " << sw << " " << a.gbps << "Gbps " << delay << " 0\n";
    }
    std::ofstream(a.out + "/flow.txt") << "0\n";
    std::ofstream(a.out + "/trace.txt") << "0\n";
    std::ofstream c(conf);
    c << "ENABLE_QCN 1\nUSE_DYNAMIC_PFC_THRESHOLD 1\nPACKET_PAYLOAD_SIZE 9000\n"
      << "FLOW_FILE " << a.out << "/flow.txt\nTRACE_FILE " << a.out << "/trace.txt\n"
      << "TRACE_OUTPUT_FILE " << a.out << "/mix.tr\nFCT_OUTPUT_FILE " << a.out << "/fct.txt\n"
      << "PFC_OUTPUT_FILE " << a.out << "/pfc.txt\nSIMULATOR_STOP_TIME 40000000000000.00\n"
      << "CC_MODE 1\nALPHA_RESUME_INTERVAL 1\nRATE_DECREASE_INTERVAL 4\nCLAMP_TARGET_RATE 0\nRP_TIMER 900\n"
      << "EWMA_GAIN 0.00390625\nFAST_RECOVERY_TIMES 1\nRATE_AI 50Mb/s\nRATE_HAI 100Mb/s\nMIN_RATE 100Mb/s\n"
      << "DCTCP_RATE_AI 1000Mb/s\nERROR_RATE_PER_LINK 0.0000\nL2_CHUNK_SIZE 4000\nL2_ACK_INTERVAL 1\n"
      << "L2_BACK_TO_ZERO 0\nHAS_WIN 1\nGLOBAL_T 0\nVAR_WIN 1\nFAST_REACT 1\nU_TARGET 0.95\nMI_THRESH 0\n"
      << "INT_MULTI 1\nMULTI_RATE 0\nSAMPLE_FEEDBACK 0\nPINT_LOG_BASE 1.05\nPINT_PROB 1.0\nRATE_BOUND 1\n"
      << "ACK_HIGH_PRIO 0\nLINK_DOWN 0 0 0\nENABLE_TRACE 0\n"
      << "KMAX_MAP 6 25000000000 400 50000000000 800 100000000000 1600 200000000000 1200 400000000000 3200 "
         "1600000000000 2400\n"
      << "KMIN_MAP 6 25000000000 100 50000000000 200 100000000000 400 200000000000 300 400000000000 800 "
         "1600000000000 600\n"
      << "PMAX_MAP 6 25000000000 0.2 50000000000 0.2 100000000000 0.2 200000000000 0.8 400000000000 0.2 "
         "1600000000000 0.2\n"
      << "BUFFER_SIZE 32\nQLEN_MON_FILE " << a.out << "/qlen.txt\nBW_MON_FILE " << a.out << "/bw.txt\n"
      << "RATE_MON_FILE " << a.out << "/rate.txt\nCNP_MON_FILE " << a.out << "/cnp.txt\n"
      << "MON_START 0\nMON_END 20000\nQP_MON_INTERVAL 100\nQLEN_MON_INTERVAL 10000\nBW_MON_INTERVAL 10000\n";
}

struct Stages {
    double route = 0, counts = 0, pipeline = 0, reduce = 0, total = 0;
};

Stages slowest_stages(const nsim::SimResult& res) {
    Stages m;
    for (const auto& r : res.ranks) {
        const auto& p = r.program;
        const auto& o = r.ops;
        double last = 0;
        for (int op : p.combine) last = std::max(last, o[op].complete);
        m.route = std::max(m.route, o[p.route].complete);
        m.counts = std::max(m.counts, o[p.counts].complete - o[p.route].complete);
        m.pipeline = std::max(m.pipeline, last - o[p.dispatch[0]].issue_begin);
        m.reduce = std::max(m.reduce, o[p.reduce].complete - o[p.reduce].issue_begin);
        m.total = std::max(m.total, o[p.retire].issue_end);
    }
    return m;
}

nsim::SimResult run_layer(const Args& a, nsim::Scheduler& sched, std::vector<AstraSim::AstraNetworkAPI*> eps,
                          const std::function<void(nsim::LayerSimulation&)>& drive,
                          nsim::simai::NetFabric** fabric_out);
int report(const Args& a, const nsim::SimResult& res, const nsim::simai::NetFabric& fabric);

int run(const Args& a) {
    if (a.backend == "analytical") {
        nsim::EventQueue q;
        nsim::simai::AnalyticalNetwork net(q, a.world, a.gbps, 2 * a.latency_us);
        nsim::simai::AstraScheduler sched(net.endpoint(0));
        nsim::simai::NetFabric* fabric = nullptr;
        const nsim::SimResult res = run_layer(a, sched, net.endpoints(), [&](nsim::LayerSimulation& sim) {
            sim.start();
            q.run();
        }, &fabric);
        return report(a, res, *fabric);
    }
    std::string topo, conf;
    write_inputs(a, topo, conf);
    if (!ReadConf(topo, conf)) throw std::runtime_error("ns3: cannot read " + conf);
    SetConfig();
    SetupNetwork(qp_finish, send_finish);
    g_world = a.world;
    std::vector<std::unique_ptr<Ns3Network>> nets;
    std::vector<AstraSim::AstraNetworkAPI*> eps;
    for (int r = 0; r < a.world; ++r) {
        nets.push_back(std::make_unique<Ns3Network>(r));
        eps.push_back(nets.back().get());
    }
    nsim::simai::AstraScheduler sched(*nets[0]);

    if (a.selftest) {
        const std::uint64_t bytes = 1 << 20;
        struct Seen {
            nsim::Scheduler* s;
            double sent = -1, received = -1;
        } seen{&sched};
        AstraSim::sim_request req{};
        nets[1]->sim_recv(nullptr, bytes, AstraSim::UINT8, 0, 1, &req,
                          [](void* p) { static_cast<Seen*>(p)->received = static_cast<Seen*>(p)->s->now(); }, &seen);
        nets[0]->sim_send(nullptr, bytes, AstraSim::UINT8, 1, 1, &req,
                          [](void* p) { static_cast<Seen*>(p)->sent = static_cast<Seen*>(p)->s->now(); }, &seen);
        Simulator::Stop(Seconds(1));
        Simulator::Run();
        const double wire = bytes * 8.0 / (a.gbps * 1e3);
        std::printf("selftest 1 MiB rank 0 -> 1 over %d Gb/s, %.2f us links: send completed at %.2f us, receive at "
                    "%.2f us (wire time %.2f us, 2 hops)\n",
                    a.gbps, a.latency_us, seen.sent, seen.received, wire);
        std::printf("selftest=%s\n", seen.sent > 0 && seen.received >= seen.sent ? "PASS" : "FAIL");
        return seen.sent > 0 && seen.received >= seen.sent ? 0 : 1;
    }

    nsim::simai::NetFabric* fabric = nullptr;
    const double limit_ms = a.max_sim_ms;
    const nsim::SimResult res = run_layer(a, sched, eps, [limit_ms](nsim::LayerSimulation& sim) {
        sim.start([] { Simulator::Stop(); });
        // DCQCN timers keep the event loop alive forever, so a message that
        // never completes would hang the run; stop and let finish() report
        // where the layer stopped.
        Simulator::Stop(MicroSeconds(static_cast<uint64_t>(limit_ms * 1000.0)));
        Simulator::Run();
    }, &fabric);
    return report(a, res, *fabric);
}

nsim::SimResult run_layer(const Args& a, nsim::Scheduler& sched, std::vector<AstraSim::AstraNetworkAPI*> eps,
                          const std::function<void(nsim::LayerSimulation&)>& drive,
                          nsim::simai::NetFabric** fabric_out) {
    const nsim::CostProfile profile = nsim::CostProfile::load(a.profile);
    nsim::SimLayerConfig c;
    std::vector<std::vector<std::int64_t>> counts;
    std::vector<int> tokens;
    if (!a.trace.empty()) {
        const auto trace = nano_nccl::exec::LayerTrace::load(a.trace);
        const auto& k = trace.key;
        c.world = k.world;
        c.num_experts = k.num_experts;
        c.top_k = k.top_k;
        c.hidden = k.hidden;
        c.ffn = k.ffn;
        c.mode = k.mode;
        c.plan = trace.source();
        c.options = trace.options;
        c.slots_per_link = k.slots_per_link;
        c.slot_bytes = k.slot_bytes;
        for (const auto& r : trace.ranks) {
            counts.push_back(r.expert_counts);
            tokens.push_back(r.tokens);
        }
        if (c.world != a.world) throw std::invalid_argument("--world must match the trace's world");
    } else {
        c.world = a.world;
        c.num_experts = a.experts;
        c.top_k = a.top_k;
        c.hidden = a.hidden;
        c.ffn = a.ffn;
        c.mode = a.mode;
        c.options = {a.overlap, a.lookahead};
        c.slots_per_link = a.slots;
        c.slot_bytes = a.slot_bytes;
        tokens.assign(a.world, a.tokens);
        counts = nano_nccl::ep::workload_counts(nano_nccl::ep::parse_workload(a.workload, a.skew),
                                                {a.world, 0, a.experts, a.top_k, a.hidden, a.ffn}, a.tokens);
        c.plan = nano_nccl::exec::plan_source_for(nano_nccl::planner::parse_scheduler(a.scheduler), a.world,
                                                  a.experts, a.groups, a.from_counts);
    }
    auto factory = [&](nsim::Scheduler& s, nsim::HostModel& host, int world, int slots, std::size_t slot_bytes,
                       std::vector<nsim::WireCosts> costs) -> std::unique_ptr<nsim::Fabric> {
        nsim::simai::NetConfig config;
        config.host_staged = a.staged;
        auto f = std::make_unique<nsim::simai::NetFabric>(s, host, world, slots, slot_bytes, std::move(costs), eps,
                                                          config);
        *fabric_out = f.get();
        return f;
    };
    nsim::LayerSimulation sim(c, counts, tokens, profile, sched, factory);
    drive(sim);
    return sim.finish();
}

int report(const Args& a, const nsim::SimResult& res, const nsim::simai::NetFabric& fabric_ref) {
    const nsim::simai::NetFabric* fabric = &fabric_ref;
    const nsim::SimRankResult& r0 = res.ranks[0];
    const Stages s = slowest_stages(res);
    std::printf("# %s star: %d GPUs, %d Gb/s, %.2f us per link, %s; %d steps, %zu ops per rank\n",
                a.backend.c_str(), static_cast<int>(res.ranks.size()), a.gbps, a.latency_us,
                a.staged ? "host-staged" : "GPUDirect", r0.program.steps, r0.program.ops.size());
    std::printf("  route %.1f  counts %.1f  pipeline %.1f  reduce %.1f  total %.1f (us, slowest rank)\n", s.route,
                s.counts, s.pipeline, s.reduce, s.total);
    std::printf("# network messages %llu, sends completed %llu, receives completed %llu\n",
                static_cast<unsigned long long>(fabric->messages_sent()),
                static_cast<unsigned long long>(fabric->sends_completed()),
                static_cast<unsigned long long>(fabric->receives_completed()));
    if (a.backend == "ns3") {
        std::printf("# sends completed by their acknowledgement (single tiny packet): %llu\n",
                    static_cast<unsigned long long>(g_sent_on_ack));
    }
    const bool ok = fabric->messages_sent() == fabric->sends_completed() &&
                    fabric->messages_sent() == fabric->receives_completed();
    std::printf("%s_layer=%s total=%.1f\n", a.backend.c_str(), ok ? "PASS" : "FAIL", s.total);
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args a;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto next = [&]() -> const char* {
                if (i + 1 >= argc) throw std::invalid_argument(arg + " needs a value");
                return argv[++i];
            };
            if (arg == "--profile") a.profile = next();
            else if (arg == "--trace") a.trace = next();
            else if (arg == "--out") a.out = next();
            else if (arg == "--selftest") a.selftest = true;
            else if (arg == "--world") a.world = std::atoi(next());
            else if (arg == "--experts") a.experts = std::atoi(next());
            else if (arg == "--topk") a.top_k = std::atoi(next());
            else if (arg == "--hidden") a.hidden = std::atoi(next());
            else if (arg == "--ffn-dim") a.ffn = std::atoi(next());
            else if (arg == "--tokens") a.tokens = std::atoi(next());
            else if (arg == "--skew") a.skew = std::atof(next());
            else if (arg == "--scheduler") a.scheduler = next();
            else if (arg == "--workload") a.workload = next();
            else if (arg == "--plan-from-counts") a.from_counts = true;
            else if (arg == "--groups") a.groups = std::atoi(next());
            else if (arg == "--lookahead") a.lookahead = std::atoi(next());
            else if (arg == "--no-overlap") a.overlap = false;
            else if (arg == "--slots") a.slots = std::atoi(next());
            else if (arg == "--slot-bytes") a.slot_bytes = std::strtoull(next(), nullptr, 0);
            else if (arg == "--mode") a.mode = next();
            else if (arg == "--gbps") a.gbps = std::atoi(next());
            else if (arg == "--latency-us") a.latency_us = std::atof(next());
            else if (arg == "--backend") a.backend = next();
            else if (arg == "--staged") a.staged = true;
            else if (arg == "--max-sim-ms") a.max_sim_ms = std::atof(next());
            else throw std::invalid_argument("unknown option " + arg);
        }
        const int speeds[] = {25, 50, 100, 200, 400, 1600};  // ECN maps of SimAI.conf
        if (std::find(std::begin(speeds), std::end(speeds), a.gbps) == std::end(speeds)) {
            throw std::invalid_argument("--gbps must be one of 25 50 100 200 400 1600");
        }
        if (a.world < 2) throw std::invalid_argument("--world must be >= 2");
        if (a.backend != "ns3" && a.backend != "analytical") throw std::invalid_argument("--backend ns3|analytical");
        if (a.selftest && a.backend != "ns3") throw std::invalid_argument("--selftest runs on ns3");
        if (!a.selftest && a.profile.empty()) throw std::invalid_argument("--profile is required");
        return run(a);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "nano_ep_ns3: %s\n", e.what());
        return 2;
    }
}
