// CPU test of the EP row plan (src/ep/ep_plan) and group plans
// (src/planner): for random shapes, group plans and count matrices it builds
// every rank's DispatchPlan and simulates the whole layer on row tags:
// pack -> per step dispatch -> unpack -> "experts" -> repack -> combine.
// Checks that both ends of every message agree on its size, that every row
// reaches its expert exactly once and in a deterministic order, that every
// layout passes the exchanger's validation, and that every row returns to the
// position it was sent from. Also checks the scheduler switch
// (src/planner/scheduler) and its random test plans.
#include "ep/ep_plan.h"
#include "ep/synthetic_routing.h"
#include "planner/group_plan.h"
#include "planner/scheduler.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using nano_nccl::ep::DispatchPlan;
using nano_nccl::ep::EpShape;
using nano_nccl::planner::GroupPlan;

struct Tag {
    int src = -1;
    int expert = -1;
    std::int64_t index = -1;  // row index within (src, expert)
    int visits = 0;           // times an expert processed this row
    bool operator==(const Tag& o) const { return src == o.src && expert == o.expert && index == o.index; }
};

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
}

// Copies `bytes` (whole rows) between row buffers.
void copy_rows(const std::vector<Tag>& src, std::size_t src_off, std::vector<Tag>& dst, std::size_t dst_off,
               std::size_t bytes, std::size_t rb, const std::string& what) {
    require(bytes % rb == 0 && src_off % rb == 0 && dst_off % rb == 0, what + ": not row aligned");
    const std::size_t rows = bytes / rb;
    require(src_off / rb + rows <= src.size() && dst_off / rb + rows <= dst.size(), what + ": out of bounds");
    for (std::size_t i = 0; i < rows; ++i) {
        Tag& d = dst[dst_off / rb + i];
        require(d.src == -1, what + ": row written twice");
        d = src[src_off / rb + i];
    }
}

GroupPlan random_plan(std::mt19937_64& rng, int world, int experts) {
    if (rng() % 3 == 0) return nano_nccl::planner::random_group_plan(experts, experts + 1, rng());
    if (rng() % 2 == 0) {
        const int per_rank = experts / world;
        return nano_nccl::planner::fixed_group_plan(world, experts, 1 + static_cast<int>(rng() % per_rank));
    }
    GroupPlan plan;
    const int groups = 1 + static_cast<int>(rng() % (experts + 1));  // may leave groups empty
    for (int e = 0; e < experts; ++e) plan.expert_group.push_back(static_cast<int>(rng() % groups));
    plan.order.resize(groups);
    std::iota(plan.order.begin(), plan.order.end(), 0);
    std::shuffle(plan.order.begin(), plan.order.end(), rng);
    return plan;
}

void run_trial(std::mt19937_64& rng, int trial) {
    const int worlds[] = {1, 2, 3, 4, 8};
    const int W = worlds[rng() % 5];
    const int epr = 1 + static_cast<int>(rng() % 4);
    const int E = W * epr;
    GroupPlan groups = random_plan(rng, W, E);
    // N[s][e], with many zeros.
    std::vector<std::vector<std::int64_t>> N(W, std::vector<std::int64_t>(E));
    for (auto& row : N) {
        for (auto& n : row) n = rng() % 3 == 0 ? 0 : static_cast<std::int64_t>(rng() % 6);
    }
    if (rng() % 4 == 0) {  // a plan that depends on these counts
        nano_nccl::planner::ScheduleRequest req{W, E, 1 + static_cast<int>(rng() % E), {}};
        for (const auto& row : N) req.counts.insert(req.counts.end(), row.begin(), row.end());
        groups = nano_nccl::planner::make_schedule(nano_nccl::planner::SchedulerKind::kTokenGreedy, req);
    }

    std::vector<EpShape> shapes;
    std::vector<DispatchPlan> plans;
    for (int r = 0; r < W; ++r) {
        EpShape s{W, r, E, 1, 8, 0};
        std::vector<std::int64_t> recv(static_cast<std::size_t>(W) * epr);
        for (int src = 0; src < W; ++src) {
            for (int j = 0; j < epr; ++j) recv[src * epr + j] = N[src][r * epr + j];
        }
        shapes.push_back(s);
        plans.push_back(nano_nccl::ep::make_dispatch_plan(s, groups, N[r], recv));
    }
    const std::size_t rb = shapes[0].row_bytes();
    const int steps = groups.groups();
    const std::string where = "trial " + std::to_string(trial) + " (W=" + std::to_string(W) +
                              " epr=" + std::to_string(epr) + " groups=" + std::to_string(steps) + ")";

    // Pack: send rows tagged by (rank, expert, index).
    std::vector<std::vector<Tag>> send(W), recv(W), expert(W), back(W), ret(W);
    for (int r = 0; r < W; ++r) {
        const DispatchPlan& p = plans[r];
        require(static_cast<int>(p.steps.size()) == steps, where + ": step count");
        send[r].assign(static_cast<std::size_t>(p.send_rows), Tag{});
        for (int e = 0; e < E; ++e) {
            for (std::int64_t i = 0; i < N[r][e]; ++i) {
                Tag& t = send[r][static_cast<std::size_t>(p.send_offsets[e] + i)];
                require(t.src == -1, where + ": send rows overlap");
                t = {r, e, i, 0};
            }
        }
        for (const Tag& t : send[r]) require(t.src != -1, where + ": send row unused");
        recv[r].assign(static_cast<std::size_t>(p.recv_rows), Tag{});
        expert[r].assign(static_cast<std::size_t>(p.recv_rows), Tag{});
        back[r].assign(static_cast<std::size_t>(p.recv_rows), Tag{});
        ret[r].assign(static_cast<std::size_t>(p.send_rows), Tag{});
    }

    for (int i = 0; i < steps; ++i) {
        for (int r = 0; r < W; ++r) {
            const auto& st = plans[r].steps[i];
            require(st.group == groups.order[i], where + ": step order");
            nano_nccl::transport::peer::validate_layout(st.dispatch, W);
            nano_nccl::transport::peer::validate_layout(st.combine, W);
        }
        // Dispatch: r -> d.
        for (int r = 0; r < W; ++r) {
            for (int d = 0; d < W; ++d) {
                const auto& out = plans[r].steps[i].dispatch;
                const auto& in = plans[d].steps[i].dispatch;
                require(out.send_bytes[d] == in.recv_bytes[r], where + ": dispatch size mismatch");
                copy_rows(send[r], out.send_offsets[d], recv[d], in.recv_offsets[r], out.send_bytes[d], rb,
                          where + " dispatch");
            }
        }
        // Unpack, run "experts", repack.
        for (int r = 0; r < W; ++r) {
            const auto& st = plans[r].steps[i];
            for (const auto& seg : st.unpack) {
                for (std::int64_t k = 0; k < seg.rows; ++k) {
                    Tag& d = expert[r][static_cast<std::size_t>(seg.dst_row + k)];
                    require(d.src == -1, where + ": unpack overlap");
                    d = recv[r][static_cast<std::size_t>(seg.src_row + k)];
                }
            }
            for (const auto& seg : st.experts) {
                const int global = shapes[r].first_local_expert() + seg.expert;
                require(groups.expert_group[global] == st.group, where + ": expert in the wrong step");
                for (std::int64_t k = 0; k < seg.rows; ++k) {
                    Tag& t = expert[r][static_cast<std::size_t>(seg.begin + k)];
                    require(t.expert == global, where + ": row reached the wrong expert");
                    if (k > 0) {
                        const Tag& prev = expert[r][static_cast<std::size_t>(seg.begin + k - 1)];
                        require(prev.src < t.src || (prev.src == t.src && prev.index + 1 == t.index),
                                where + ": expert rows not in (source, index) order");
                    }
                    ++t.visits;
                }
            }
            for (const auto& seg : st.repack) {
                for (std::int64_t k = 0; k < seg.rows; ++k) {
                    Tag& d = back[r][static_cast<std::size_t>(seg.dst_row + k)];
                    require(d.src == -1, where + ": repack overlap");
                    d = expert[r][static_cast<std::size_t>(seg.src_row + k)];
                }
            }
        }
        // Combine: d -> r, back into r's send-buffer order.
        for (int d = 0; d < W; ++d) {
            for (int r = 0; r < W; ++r) {
                const auto& out = plans[d].steps[i].combine;
                const auto& in = plans[r].steps[i].combine;
                require(out.send_bytes[r] == in.recv_bytes[d], where + ": combine size mismatch");
                copy_rows(back[d], out.send_offsets[r], ret[r], in.recv_offsets[d], out.send_bytes[r], rb,
                          where + " combine");
            }
        }
    }
    for (int r = 0; r < W; ++r) {
        for (std::size_t k = 0; k < send[r].size(); ++k) {
            require(ret[r][k] == send[r][k], where + ": row did not return to its send position");
            require(ret[r][k].visits == 1, where + ": row processed " + std::to_string(ret[r][k].visits) + " times");
        }
    }
}

bool same_plan(const GroupPlan& a, const GroupPlan& b) {
    return a.expert_group == b.expert_group && a.order == b.order;
}

// The scheduler switch: names, the fixed scheduler, determinism, and that the
// random test plans reach the shapes a dynamic scheduler may produce.
bool test_scheduler() {
    namespace pl = nano_nccl::planner;
    bool ok = true;
    auto check = [&ok](bool cond, const std::string& what) {
        if (!cond) {
            std::printf("FAIL scheduler: %s\n", what.c_str());
            ok = false;
        }
    };
    for (pl::SchedulerKind k : {pl::SchedulerKind::kFixed, pl::SchedulerKind::kTokenGreedy}) {
        check(pl::parse_scheduler(pl::scheduler_name(k)) == k, std::string("name round trip ") + pl::scheduler_name(k));
    }
    for (int W : {1, 2, 4, 8}) {
        for (int epr : {1, 2, 8}) {
            for (int g = 1; g <= epr; ++g) {
                const pl::ScheduleRequest req{W, W * epr, g, {}};
                check(same_plan(pl::make_schedule(pl::SchedulerKind::kFixed, req), pl::fixed_group_plan(W, W * epr, g)),
                      "fixed scheduler differs from fixed_group_plan");
                pl::ScheduleRequest with_counts = req;
                with_counts.counts.assign(static_cast<std::size_t>(W) * W * epr, 3);
                check(same_plan(pl::make_schedule(pl::SchedulerKind::kFixed, with_counts), pl::fixed_group_plan(W, W * epr, g)),
                      "fixed scheduler depends on the counts");
            }
        }
    }
    // Random test plans: deterministic, valid, and covering empty groups,
    // groups with one rank's experts only, and groups spanning every rank.
    const int W = 2, E = 8, epr = E / W;
    int empty = 0, one_rank = 0, all_ranks = 0, reordered = 0;
    for (std::uint64_t seed = 0; seed < 400; ++seed) {
        const GroupPlan a = pl::random_group_plan(E, E + 1, seed);
        check(same_plan(a, pl::random_group_plan(E, E + 1, seed)), "random plan not deterministic");
        check(a.groups() >= 1 && a.groups() <= E + 1, "random plan group count");
        for (int g = 0; g < a.groups(); ++g) {
            int ranks = 0;
            for (int r = 0; r < W; ++r) {
                bool has = false;
                for (int j = 0; j < epr; ++j) has |= a.expert_group[r * epr + j] == g;
                ranks += has ? 1 : 0;
            }
            empty += ranks == 0;
            one_rank += ranks == 1;
            all_ranks += ranks == W;
        }
        for (int i = 0; i < a.groups(); ++i) reordered += a.order[i] != i;
    }
    check(empty > 0 && one_rank > 0 && all_ranks > 0 && reordered > 0,
          "random plans miss a shape: empty " + std::to_string(empty) + ", one rank " + std::to_string(one_rank) +
              ", all ranks " + std::to_string(all_ranks) + ", reordered " + std::to_string(reordered));
    if (ok) std::printf("PASS scheduler\n");
    return ok;
}

// sum over groups of the busiest rank's rows (token_greedy's objective).
std::int64_t step_max_sum(const GroupPlan& plan, const std::vector<std::int64_t>& counts, int W, int E) {
    const int epr = E / W;
    std::vector<std::int64_t> load(static_cast<std::size_t>(W) * plan.groups(), 0);
    for (int s = 0; s < W; ++s) {
        for (int e = 0; e < E; ++e) load[(e / epr) * plan.groups() + plan.expert_group[e]] += counts[s * E + e];
    }
    std::int64_t sum = 0;
    for (int g = 0; g < plan.groups(); ++g) {
        std::int64_t m = 0;
        for (int r = 0; r < W; ++r) m = std::max(m, load[r * plan.groups() + g]);
        sum += m;
    }
    return sum;
}

bool test_token_greedy() {
    namespace pl = nano_nccl::planner;
    bool ok = true;
    auto check = [&ok](bool cond, const std::string& what) {
        if (!cond) {
            std::printf("FAIL token_greedy: %s\n", what.c_str());
            ok = false;
        }
    };
    const auto kind = pl::SchedulerKind::kTokenGreedy;
    check(pl::scheduler_reads_counts(kind), "must read the counts");
    // Hand example: rank 0 hosts experts 0 (10 rows) and 1 (1 row), rank 1
    // hosts 2 (1 row) and 3 (10 rows). Fixed pairs 0 with 2 and 1 with 3:
    // 10 + 10 = 20. Pairing the two heavy experts reaches the bound 10 + 1.
    {
        const pl::ScheduleRequest req{2, 4, 2, {10, 1, 1, 10, 0, 0, 0, 0}};
        const GroupPlan p = pl::make_schedule(kind, req);
        check(p.expert_group == std::vector<int>({0, 1, 1, 0}), "hand example mapping");
        check(p.order == std::vector<int>({1, 0}), "hand example order: small step first");
        check(step_max_sum(p, req.counts, 2, 4) == 11, "hand example objective");
        check(step_max_sum(pl::fixed_group_plan(2, 4, 2), req.counts, 2, 4) == 20, "hand example fixed objective");
    }
    std::mt19937_64 rng(777);
    double greedy_sum = 0, fixed_sum = 0;
    int compared = 0, worse = 0;
    for (int trial = 0; trial < 2000; ++trial) {
        const int W = 1 << (rng() % 4), epr = 1 + static_cast<int>(rng() % 8), E = W * epr;
        const int G = 1 + static_cast<int>(rng() % E);
        const bool uniform = rng() % 5 == 0;
        pl::ScheduleRequest req{W, E, G, std::vector<std::int64_t>(static_cast<std::size_t>(W) * E)};
        for (auto& n : req.counts) {
            n = uniform ? 7 : (rng() % 4 == 0 ? 0 : static_cast<std::int64_t>(rng() % (1 + rng() % 1000)));
        }
        const GroupPlan p = pl::make_schedule(kind, req);
        const std::string where = "trial " + std::to_string(trial);
        check(p.groups() == G, where + ": group count");
        GroupPlan q = pl::make_schedule(kind, req);
        check(q.expert_group == p.expert_group && q.order == p.order, where + ": not deterministic");
        // Lower bounds: the busiest rank's total, and the largest expert.
        std::vector<std::int64_t> total(static_cast<std::size_t>(W), 0);
        std::int64_t largest = 0;
        for (int e = 0; e < E; ++e) {
            std::int64_t rows = 0;
            for (int s = 0; s < W; ++s) rows += req.counts[s * E + e];
            total[e / epr] += rows;
            largest = std::max(largest, rows);
        }
        const std::int64_t bound = std::max(*std::max_element(total.begin(), total.end()), largest);
        const std::int64_t got = step_max_sum(p, req.counts, W, E);
        check(got >= bound, where + ": below the lower bound");
        if (G == 1) check(got == bound, where + ": one group must equal the busiest rank");
        // Equal experts and groups dividing epr: every step equal, at the bound.
        if (uniform && epr % G == 0) check(got == bound, where + ": uniform load not at the bound");
        if (G <= epr) {
            const std::int64_t fixed = step_max_sum(pl::fixed_group_plan(W, E, G), req.counts, W, E);
            greedy_sum += static_cast<double>(got);
            fixed_sum += static_cast<double>(fixed);
            ++compared;
            worse += got > fixed ? 1 : 0;
        }
    }
    std::printf("token_greedy vs fixed objective over %d trials: %.3f of fixed on average, worse in %d\n", compared,
                greedy_sum / fixed_sum, worse);
    check(greedy_sum < fixed_sum, "not better than fixed on average");
    if (ok) std::printf("PASS token_greedy\n");
    return ok;
}

// Workload families (ep::Workload): expected expert loads per rank.
bool test_workloads() {
    namespace ep = nano_nccl::ep;
    bool ok = true;
    auto check = [&ok](bool cond, const std::string& what) {
        if (!cond) {
            std::printf("FAIL workloads: %s\n", what.c_str());
            ok = false;
        }
    };
    const int W = 8, epr = 8, E = W * epr;
    EpShape s{W, 0, E, 2, 64, 0};
    // zipf is the original benchmark distribution, draw for draw.
    {
        const auto a = ep::make_routing(s, 300, 1.2, -1, 99);
        const auto b = ep::make_routing(s, 300, ep::source_expert_weights(ep::parse_workload("zipf", 1.2), s), 99);
        check(a.ids == b.ids && a.weights == b.weights, "zipf differs from make_routing(skew)");
    }
    // Expected load of rank r's experts: sum over sources of the share of
    // their weight on r (each source's distribution normalized).
    auto rank_load = [&](const ep::Workload& w) {
        std::vector<double> load(W, 0);
        for (int src = 0; src < W; ++src) {
            EpShape t = s;
            t.rank = src;
            const auto wt = ep::source_expert_weights(w, t);
            double total = 0;
            for (double x : wt) total += x;
            for (int e = 0; e < E; ++e) load[e / epr] += wt[e] / total;
        }
        return load;
    };
    auto equal = [](const std::vector<double>& v) {
        for (double x : v) {
            if (std::abs(x - v[0]) > 1e-9 * std::abs(v[0])) return false;
        }
        return true;
    };
    for (const char* name : {"rank_skew", "local:4", "pair:4"}) {
        check(equal(rank_load(ep::parse_workload(name, 1.5))), std::string(name) + ": rank loads differ");
    }
    {
        const auto l = rank_load(ep::parse_workload("hot_rank:3", 1.0));
        check(std::abs(l[0] / l[1] - 3.0) < 1e-9 && equal(std::vector<double>(l.begin() + 1, l.end())),
              "hot_rank:3: rank 0 is not 3x the others");
    }
    {
        const auto z = rank_load(ep::parse_workload("zipf", 1.0));
        check(z[0] > z[W - 1] * 2, "zipf: rank 0 not hot");
    }
    // rank_skew: different hot local expert per rank; local/pair: the bias.
    {
        const auto wt = ep::source_expert_weights(ep::parse_workload("rank_skew", 2.0), s);
        int distinct = 0;
        for (int r = 1; r < W; ++r) {
            distinct += std::max_element(wt.begin() + r * epr, wt.begin() + (r + 1) * epr) - (wt.begin() + r * epr) !=
                        std::max_element(wt.begin(), wt.begin() + epr) - wt.begin();
        }
        check(distinct > 0, "rank_skew: every rank has the same hot local expert");
        EpShape t = s;
        t.rank = 3;
        const auto loc = ep::source_expert_weights(ep::parse_workload("local:5", 2.0), t);
        const auto pair = ep::source_expert_weights(ep::parse_workload("pair:5", 2.0), t);
        check(std::abs(loc[3 * epr] - 5 * wt[3 * epr]) < 1e-12 && loc[4 * epr] == wt[4 * epr], "local:5 bias");
        check(std::abs(pair[4 * epr] - 5 * wt[4 * epr]) < 1e-12 && pair[3 * epr] == wt[3 * epr], "pair:5 bias");
    }
    const auto counts = ep::workload_counts(ep::parse_workload("pair:4", 1.0), s, 500);
    for (const auto& row : counts) {
        std::int64_t sum = 0;
        for (auto n : row) sum += n;
        check(sum == 500 * 2, "workload_counts: entries per rank");
    }
    check(ep::parse_workload("pair:4", 0).name() == "pair:4" && ep::parse_workload("rank_skew", 0).name() == "rank_skew",
          "workload names");
    if (ok) std::printf("PASS workloads\n");
    return ok;
}

template <typename Fn>
bool expect_invalid(const char* name, Fn fn) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        std::printf("PASS negative %s\n", name);
        return true;
    }
    std::printf("FAIL negative %s: accepted\n", name);
    return false;
}

}  // namespace

int main() {
    std::mt19937_64 rng(12345);
    constexpr int kTrials = 3000;
    int failed = 0;
    for (int t = 0; t < kTrials; ++t) {
        try {
            run_trial(rng, t);
        } catch (const std::exception& error) {
            if (failed++ < 5) std::printf("FAIL %s\n", error.what());
        }
    }
    std::printf("trials: %d passed, %d failed\n", kTrials - failed, failed);
    bool ok = failed == 0;
    ok &= test_scheduler();
    ok &= test_token_greedy();
    ok &= test_workloads();
    ok &= expect_invalid("group_out_of_range", [] {
        nano_nccl::planner::validate_group_plan({{0, 2}, {0, 1}}, 2);
    });
    ok &= expect_invalid("order_not_permutation", [] {
        nano_nccl::planner::validate_group_plan({{0, 1}, {1, 1}}, 2);
    });
    ok &= expect_invalid("wrong_expert_count", [] {
        nano_nccl::planner::validate_group_plan({{0}, {0}}, 2);
    });
    ok &= expect_invalid("fixed_too_many_groups", [] { nano_nccl::planner::fixed_group_plan(2, 4, 3); });
    ok &= expect_invalid("unknown_scheduler", [] { nano_nccl::planner::parse_scheduler("greedy"); });
    ok &= expect_invalid("schedule_counts_size", [] {
        nano_nccl::planner::make_schedule(nano_nccl::planner::SchedulerKind::kFixed, {2, 4, 2, {1, 2, 3}});
    });
    ok &= expect_invalid("schedule_negative_count", [] {
        nano_nccl::planner::make_schedule(nano_nccl::planner::SchedulerKind::kFixed,
                                          {2, 4, 2, {1, 2, 3, 4, 5, 6, 7, -1}});
    });
    ok &= expect_invalid("token_greedy_needs_counts", [] {
        nano_nccl::planner::make_schedule(nano_nccl::planner::SchedulerKind::kTokenGreedy, {2, 4, 2, {}});
    });
    ok &= expect_invalid("token_greedy_too_many_groups", [] {
        nano_nccl::planner::make_schedule(nano_nccl::planner::SchedulerKind::kTokenGreedy,
                                          {2, 4, 5, {1, 1, 1, 1, 1, 1, 1, 1}});
    });
    ok &= expect_invalid("workload_unknown", [] { nano_nccl::ep::parse_workload("uniform", 0); });
    ok &= expect_invalid("workload_missing_param", [] { nano_nccl::ep::parse_workload("pair", 0); });
    ok &= expect_invalid("workload_extra_param", [] { nano_nccl::ep::parse_workload("zipf:2", 0); });
    ok &= expect_invalid("workload_bad_param", [] { nano_nccl::ep::parse_workload("local:-1", 0); });
    ok &= expect_invalid("schedule_bad_world", [] {
        nano_nccl::planner::make_schedule(nano_nccl::planner::SchedulerKind::kFixed, {3, 4, 1, {}});
    });
    std::printf("ep_plan=%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
