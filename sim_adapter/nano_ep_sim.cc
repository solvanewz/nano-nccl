// EP layer simulator front end.
//
//   nano_ep_sim --profile P --trace T [--timeline] [--copy-scale X --compute-scale Y]
//               [--no-contention]
//       replays a measured run (tests/ep_moe_test --ffn --trace): same shape,
//       counts, group plan, options and slots; prints measured vs predicted
//       stage times and, with --timeline, rank 0's per-op times.
//   nano_ep_sim --profile P [--world W --experts E --topk K --hidden H --ffn-dim F
//               --tokens T --skew S --scheduler NAME --plan-from-counts --groups G
//               --lookahead L --no-overlap --slots N --slot-bytes B
//               --mode ffn|probe --sweep --timeline]
//       synthetic routing (ep::make_routing with the benchmark seeds), one
//       configuration or, with --sweep, the A40 benchmark sweep. The group
//       plan comes from the scheduler switch (planner::make_schedule, default
//       fixed = B1) with G groups (exec::plan_source_for: schedulers that read
//       the counts, or any with --plan-from-counts, run at the plan op).
#include "cost_profile.h"
#include "ep/synthetic_routing.h"
#include "exec/exec_trace.h"
#include "sim_layer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using nano_nccl::exec::OpKind;
using nano_nccl::sim::SimLayerConfig;
using nano_nccl::sim::SimRankResult;
using nano_nccl::sim::SimResult;

struct Stages {
    double route = 0, counts = 0, pipeline = 0, reduce = 0, total = 0;
};

Stages stages_of(const SimRankResult& r) {
    const auto& p = r.program;
    const auto& o = r.ops;
    Stages s;
    s.route = o[p.route].complete;
    s.counts = o[p.counts].complete - s.route;
    double last = 0;
    for (int op : p.combine) last = std::max(last, o[op].complete);
    s.pipeline = last - o[p.dispatch[0]].issue_begin;
    s.reduce = o[p.reduce].complete - o[p.reduce].issue_begin;
    s.total = o[p.retire].issue_end;
    return s;
}

// Slowest rank per stage, as tests/ep_moe_test prints it.
Stages slowest(const std::vector<Stages>& all) {
    Stages m;
    for (const Stages& s : all) {
        m.route = std::max(m.route, s.route);
        m.counts = std::max(m.counts, s.counts);
        m.pipeline = std::max(m.pipeline, s.pipeline);
        m.reduce = std::max(m.reduce, s.reduce);
        m.total = std::max(m.total, s.total);
    }
    return m;
}

void print_timeline(const SimResult& res, int rank, const nano_nccl::exec::LayerTrace* trace) {
    const SimRankResult& r = res.ranks.at(rank);
    std::printf("# rank %d op timeline, us since forward start%s\n", rank,
                trace ? " (measured = last iteration of the trace)" : "");
    if (trace) {
        std::printf("  %3s %-9s %4s %10s %10s %10s %10s\n", "op", "kind", "step", "issue_m", "issue_p", "done_m",
                    "done_p");
    } else {
        std::printf("  %3s %-9s %4s %10s %10s\n", "op", "kind", "step", "issue", "done");
    }
    for (std::size_t i = 0; i < r.ops.size(); ++i) {
        const auto& o = r.program.ops[i];
        if (trace) {
            const nano_nccl::exec::TraceOp* m = nullptr;
            for (const auto& t : trace->ops) {
                if (t.rank == rank && t.op == static_cast<int>(i)) m = &t;
            }
            std::printf("  %3zu %-9s %4d %10.1f %10.1f %10.1f %10.1f\n", i, nano_nccl::exec::op_kind_name(o.kind),
                        o.step, m ? m->issue_begin : -1, r.ops[i].issue_begin, m ? m->complete : -1,
                        r.ops[i].complete);
        } else {
            std::printf("  %3zu %-9s %4d %10.1f %10.1f\n", i, nano_nccl::exec::op_kind_name(o.kind), o.step,
                        r.ops[i].issue_begin, r.ops[i].complete);
        }
    }
}

void print_utilization(const SimResult& res) {
    for (std::size_t r = 0; r < res.ranks.size(); ++r) {
        const double t = res.ranks[r].total;
        std::printf("# rank %zu busy: gpu %.0f us (%.0f%%), tx %.0f us (%.0f%%), rx %.0f us (%.0f%%)\n", r,
                    res.ranks[r].gpu_busy, 100 * res.ranks[r].gpu_busy / t, res.tx[r].busy, 100 * res.tx[r].busy / t,
                    res.rx[r].busy, 100 * res.rx[r].busy / t);
    }
}

struct Args {
    std::string profile, trace;
    int world = 2, experts = 16, top_k = 2, hidden = 2048, ffn = 1408, tokens = 4096;
    double skew = 0.0;
    std::string scheduler = "fixed";
    bool from_counts = false;
    int groups = 1, lookahead = 1;
    bool overlap = true, sweep = false, timeline = false;
    int slots = 4;
    std::size_t slot_bytes = std::size_t{1} << 20;
    std::string mode = "ffn";
    double copy_scale = 1.0, compute_scale = 1.0;
    bool contention = true;
};

int replay(const Args& a, const nano_nccl::sim::CostProfile& profile) {
    const auto trace = nano_nccl::exec::LayerTrace::load(a.trace);
    const auto& k = trace.key;
    if (!nano_nccl::exec::cost_compatible(profile.key, k)) {
        std::fprintf(stderr, "warning: profile key (%s) is not cost-compatible with the trace (%s)\n",
                     profile.key.to_string().c_str(), k.to_string().c_str());
    }
    SimLayerConfig c;
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
    c.copy_scale = a.copy_scale;
    c.compute_scale = a.compute_scale;
    c.contention = a.contention;
    std::vector<std::vector<std::int64_t>> counts;
    std::vector<int> tokens;
    for (const auto& r : trace.ranks) {
        counts.push_back(r.expert_counts);
        tokens.push_back(r.tokens);
    }
    const SimResult res = nano_nccl::sim::simulate_layer(c, counts, tokens, profile);
    // A scheduled run: the simulator's scheduler call must give the plan the
    // runtime ran.
    for (const auto& r : res.ranks) {
        if (r.plan.groups.expert_group != trace.groups.expert_group || r.plan.groups.order != trace.groups.order) {
            throw std::runtime_error("replay: the simulated plan differs from the measured one");
        }
    }
    std::vector<Stages> pred, meas;
    for (const auto& r : res.ranks) pred.push_back(stages_of(r));
    for (const auto& r : trace.ranks) meas.push_back({r.route, r.counts, r.pipeline, r.reduce, r.total});
    const Stages p = slowest(pred), m = slowest(meas);
    std::printf("# replay of %s\n# key %s\n# plan=%s scheduler=%s groups=%d overlap=%d lookahead=%d\n",
                a.trace.c_str(), k.to_string().c_str(), nano_nccl::exec::plan_timing_name(trace.timing),
                nano_nccl::planner::scheduler_name(trace.scheduler), trace.groups.groups(),
                trace.options.overlap ? 1 : 0, trace.options.lookahead);
    std::printf("  %-9s %10s %10s %8s\n", "stage", "measured", "predicted", "error");
    const char* names[] = {"route", "counts", "pipeline", "reduce", "total"};
    const double mv[] = {m.route, m.counts, m.pipeline, m.reduce, m.total};
    const double pv[] = {p.route, p.counts, p.pipeline, p.reduce, p.total};
    for (int i = 0; i < 5; ++i) {
        std::printf("  %-9s %10.1f %10.1f %7.1f%%\n", names[i], mv[i], pv[i], 100 * (pv[i] - mv[i]) / mv[i]);
    }
    print_utilization(res);
    if (a.timeline) print_timeline(res, 0, &trace);
    std::printf("compare scheduler=%s plan=%s groups=%d overlap=%d lookahead=%d skew=%g contention=%d measured=%.1f "
                "predicted=%.1f error=%.1f%%\n",
                nano_nccl::planner::scheduler_name(trace.scheduler), nano_nccl::exec::plan_timing_name(trace.timing),
                trace.groups.groups(), trace.options.overlap ? 1 : 0, trace.options.lookahead, k.skew,
                a.contention ? 1 : 0, m.total, p.total, 100 * (p.total - m.total) / m.total);
    return 0;
}

double synthetic(const Args& a, const nano_nccl::sim::CostProfile& profile, int groups, int lookahead, bool overlap,
                 bool verbose) {
    SimLayerConfig c;
    c.world = a.world;
    c.num_experts = a.experts;
    c.top_k = a.top_k;
    c.hidden = a.hidden;
    c.ffn = a.ffn;
    c.mode = a.mode;
    std::vector<std::vector<std::int64_t>> counts;
    std::vector<int> tokens(a.world, a.tokens);
    for (int r = 0; r < a.world; ++r) {
        const nano_nccl::ep::EpShape s{a.world, r, a.experts, a.top_k, a.hidden, a.ffn};
        counts.push_back(nano_nccl::ep::expert_counts(
            nano_nccl::ep::make_routing(s, a.tokens, a.skew, -1, nano_nccl::ep::benchmark_routing_seed(r)).ids,
            a.experts));
    }
    c.plan = nano_nccl::exec::plan_source_for(nano_nccl::planner::parse_scheduler(a.scheduler), a.world, a.experts,
                                              groups, a.from_counts);
    c.options = {overlap, lookahead};
    c.slots_per_link = a.slots;
    c.slot_bytes = a.slot_bytes;
    c.copy_scale = a.copy_scale;
    c.compute_scale = a.compute_scale;
    c.contention = a.contention;
    const SimResult res = nano_nccl::sim::simulate_layer(c, counts, tokens, profile);
    std::vector<Stages> all;
    for (const auto& r : res.ranks) all.push_back(stages_of(r));
    const Stages s = slowest(all);
    if (verbose) {
        std::printf("# simulated: world=%d experts=%d top_k=%d hidden=%d ffn=%d tokens=%d skew=%g scheduler=%s "
                    "plan=%s groups=%d overlap=%d lookahead=%d slots=%d slot_bytes=%zu\n",
                    a.world, a.experts, a.top_k, a.hidden, a.ffn, a.tokens, a.skew, a.scheduler.c_str(),
                    nano_nccl::exec::plan_timing_name(c.plan.timing), c.plan.steps(), overlap ? 1 : 0, lookahead,
                    a.slots, a.slot_bytes);
        const auto& ran = res.ranks[0].plan.groups;
        std::printf("# expert -> group:");
        for (int g : ran.expert_group) std::printf(" %d", g);
        std::printf("; order:");
        for (int g : ran.order) std::printf(" %d", g);
        std::printf("\n");
        std::printf("  route %.1f  counts %.1f  pipeline %.1f  reduce %.1f  total %.1f (us, slowest rank)\n", s.route,
                    s.counts, s.pipeline, s.reduce, s.total);
        print_utilization(res);
        if (a.timeline) print_timeline(res, 0, nullptr);
    }
    return s.total;
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
            if (arg == "--profile") a.profile = next();
            else if (arg == "--trace") a.trace = next();
            else if (arg == "--world") a.world = std::atoi(next());
            else if (arg == "--experts") a.experts = std::atoi(next());
            else if (arg == "--topk") a.top_k = std::atoi(next());
            else if (arg == "--hidden") a.hidden = std::atoi(next());
            else if (arg == "--ffn-dim") a.ffn = std::atoi(next());
            else if (arg == "--tokens") a.tokens = std::atoi(next());
            else if (arg == "--skew") a.skew = std::atof(next());
            else if (arg == "--scheduler") a.scheduler = next();
            else if (arg == "--plan-from-counts") a.from_counts = true;
            else if (arg == "--groups") a.groups = std::atoi(next());
            else if (arg == "--lookahead") a.lookahead = std::atoi(next());
            else if (arg == "--no-overlap") a.overlap = false;
            else if (arg == "--slots") a.slots = std::atoi(next());
            else if (arg == "--slot-bytes") a.slot_bytes = std::strtoull(next(), nullptr, 0);
            else if (arg == "--mode") a.mode = next();
            else if (arg == "--sweep") a.sweep = true;
            else if (arg == "--timeline") a.timeline = true;
            else if (arg == "--copy-scale") a.copy_scale = std::atof(next());
            else if (arg == "--compute-scale") a.compute_scale = std::atof(next());
            else if (arg == "--no-contention") a.contention = false;
            else throw std::invalid_argument("unknown argument " + arg);
        }
        if (a.profile.empty()) throw std::invalid_argument("--profile is required");
        const auto profile = nano_nccl::sim::CostProfile::load(a.profile);
        if (!a.trace.empty()) return replay(a, profile);
        if (!a.sweep) {
            synthetic(a, profile, a.groups, a.lookahead, a.overlap, true);
            return 0;
        }
        // The A40 benchmark sweep (scripts/a40_run_b1.sh --bench).
        struct Cfg {
            int g, l;
            bool overlap;
        };
        const Cfg cfgs[] = {{1, 0, false}, {8, 0, false}, {2, 1, true}, {4, 1, true},
                            {8, 1, true},  {4, 2, true},  {8, 2, true}, {8, 7, true}};
        std::printf("# simulated sweep, skew=%g (total us, slowest rank)\n", a.skew);
        const double serial = synthetic(a, profile, 1, 0, false, false);
        for (const Cfg& c : cfgs) {
            if (c.g > a.experts / a.world) continue;
            const double t = synthetic(a, profile, c.g, c.l, c.overlap, false);
            std::printf("  groups=%d lookahead=%d overlap=%d  total %8.1f  speedup %.2fx\n", c.g, c.l,
                        c.overlap ? 1 : 0, t, serial / t);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "nano_ep_sim: %s\n", e.what());
        return 1;
    }
}
