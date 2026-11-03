// Scheduler run time for the simulator's plan op (CPU only, no GPU, no MPI).
//
//   nano_nccl_schedule_bench --profile IN --out OUT [--worlds "2 4 8 16"]
//                            [--gbps "25 100 400"] [--reps N]
//
// Reads an EP cost profile (benchmarks/ep_profile, possibly calibrated) and
// writes it again to OUT with one table per scheduler and world:
// "host_schedule_us_<name>_w<W>", x = num_experts * groups, y = us. The
// shape is the profile's (experts per rank, top-k, hidden, tokens per rank)
// scaled to W ranks (E = W * experts per rank). Each point is the slowest,
// over the workload families and link speeds below, of the median of N runs
// of planner::make_schedule on that workload's counts; cost_aware runs with
// sim::planner_cost_model for a GPUDirect link of that speed (its search
// length depends on the counts and the model, hence the maximum). Also
// writes "host_schedule_us_cost_aware" for the profile's own world when
// missing. Run it on the host whose CPU the profile describes.
#include "cost_profile.h"
#include "ep/synthetic_routing.h"
#include "planner/scheduler.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace pl = nano_nccl::planner;
namespace nsim = nano_nccl::sim;

std::vector<int> parse_ints(const std::string& s) {
    std::istringstream in(s);
    std::vector<int> v;
    for (int x; in >> x;) v.push_back(x);
    if (v.empty()) throw std::invalid_argument("empty list '" + s + "'");
    return v;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string in, out, worlds = "2 4 8 16", speeds = "25 100 400";
        int reps = 30;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto next = [&]() -> const char* {
                if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
                return argv[++i];
            };
            if (arg == "--profile") in = next();
            else if (arg == "--out") out = next();
            else if (arg == "--worlds") worlds = next();
            else if (arg == "--gbps") speeds = next();
            else if (arg == "--reps") reps = std::atoi(next());
            else throw std::invalid_argument("unknown argument " + arg);
        }
        if (in.empty() || out.empty() || reps < 1) throw std::invalid_argument("--profile and --out are required");
        nsim::CostProfile p = nsim::CostProfile::load(in);
        const auto& key = p.key;
        const int epr = key.num_experts / key.world;
        const char* workloads[][2] = {{"zipf", "1.0"},    {"zipf_spread", "1.0"}, {"rank_skew", "2.0"},
                                      {"local:4", "1.0"}, {"pair:4", "1.0"},      {"hot_rank:3", "1.0"}};
        const pl::SchedulerKind kinds[] = {pl::SchedulerKind::kFixed, pl::SchedulerKind::kTokenGreedy,
                                           pl::SchedulerKind::kCostAware};
        using Clock = std::chrono::steady_clock;
        for (int W : parse_ints(worlds)) {
            const int E = W * epr;
            std::vector<std::vector<std::int64_t>> per_workload;
            for (const auto& w : workloads) {
                const auto rows = nano_nccl::ep::workload_counts(nano_nccl::ep::parse_workload(w[0], std::atof(w[1])),
                                                                 {W, 0, E, key.top_k, key.hidden, key.ffn}, key.tokens);
                std::vector<std::int64_t> flat;
                for (const auto& r : rows) flat.insert(flat.end(), r.begin(), r.end());
                per_workload.push_back(std::move(flat));
            }
            for (pl::SchedulerKind kind : kinds) {
                const int max_groups = kind == pl::SchedulerKind::kFixed ? epr : std::min(E, 16);
                std::vector<std::pair<double, double>> pts;
                for (int G = 1;; G = std::min(G * 2, max_groups)) {
                    double worst = 0;
                    for (const auto& counts : per_workload) {
                        for (int gbps : parse_ints(speeds)) {
                            pl::ScheduleRequest req{W, E, G, counts};
                            req.overlap = true;
                            req.lookahead = std::min(G - 1, 2);
                            if (kind == pl::SchedulerKind::kCostAware) {
                                req.cost = std::make_shared<const pl::CostModel>(nsim::planner_cost_model(
                                    p, key.hidden, 1.0, nsim::link_rates(gbps, 0, key.slot_bytes, 2.0)));
                            }
                            std::vector<double> t;
                            for (int i = 0; i < reps; ++i) {
                                const auto t0 = Clock::now();
                                const pl::GroupPlan plan = pl::make_schedule(kind, req);
                                t.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
                                if (plan.groups() != G) throw std::logic_error("scheduler group count");
                            }
                            worst = std::max(worst, median(t));
                        }
                    }
                    pts.push_back({static_cast<double>(E) * G, worst});
                    std::printf("world=%d experts=%d scheduler=%s groups=%d us=%.2f\n", W, E, pl::scheduler_name(kind),
                                G, worst);
                    if (G == max_groups) break;
                }
                const std::string name = std::string(nsim::cost::kHostSchedule) + pl::scheduler_name(kind);
                p.tables[name + "_w" + std::to_string(W)] = nsim::Table(pts);
                p.units[name + "_w" + std::to_string(W)] = {"expert_groups", "us"};
                if (W == key.world && !p.tables.count(name)) {
                    p.tables[name] = nsim::Table(pts);
                    p.units[name] = {"expert_groups", "us"};
                }
            }
        }
        p.meta.push_back("scheduler run times per world from nano_nccl_schedule_bench (worlds " + worlds +
                         ", gbps " + speeds + ")");
        p.save(out);
        std::printf("profile written to %s\n", out.c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "nano_nccl_schedule_bench: %s\n", e.what());
        return 1;
    }
}
