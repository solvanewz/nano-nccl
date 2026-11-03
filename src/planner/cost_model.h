#pragma once

#include "planner/group_plan.h"

#include <climits>
#include <cstdint>
#include <utility>
#include <vector>

// Lightweight layer cost model for the cost_aware scheduler (pure C++, no
// CUDA, integer arithmetic only so every rank that evaluates the same request
// gets the same numbers and hence the same plan).
//
// It is an online estimate, not the simulator: it sees only what a real
// deployment has when it plans (this invocation's N(s, e), profiled op costs,
// the link rate and per-message overhead of the fabric), never switch queues
// or future routing. What it models, per step k of the plan's order:
//   compute(r, k)  = 2 * permute(rows(r, k)) + sum over rank r's experts e of
//                    step k with rows(e) > 0 of expert(rows(e)); the host
//                    issues the step first (expert_issue(active experts)).
//   dispatch(k)    = every rank r sends tx(r, k) bytes (its rows for other
//                    ranks' experts of step k) and receives rx(r, k) bytes;
//                    rank r's send side and receive side are each a serial
//                    resource (rate + per-message overhead, one message per
//                    peer plus bytes / chunk_bytes, prorated) shared by all of
//                    its exchanges; an exchange ends when its slowest side
//                    does, plus a fixed exchange latency.
//   combine(k)     = the same bytes in the other direction.
// and the issue order of exec::PipelineOptions (D0..D(L), then per step i:
// experts of i, C(i), D(i+L+1); combines and later dispatches are issued once
// step i's experts are done on every rank). The rank's own rows (self link)
// are not counted as traffic. The constant route/counts/plan/pack/reduce part
// of the layer does not depend on the plan and is left out.
namespace nano_nccl::planner {

// Piecewise-linear table over integers, same rule as the simulator's tables:
// constant below the first point, linear between points, extrapolated from
// the last two (a single point is a constant).
class IntTable {
public:
    IntTable() = default;
    // Ascending x, at least one point; throws std::invalid_argument otherwise.
    explicit IntTable(std::vector<std::pair<std::int64_t, std::int64_t>> points);
    std::int64_t operator()(std::int64_t x) const;
    bool empty() const { return points_.empty(); }
    const std::vector<std::pair<std::int64_t, std::int64_t>>& points() const { return points_; }

private:
    std::vector<std::pair<std::int64_t, std::int64_t>> points_;
};

struct CostModel {
    std::int64_t row_bytes = 0;  // bytes of one token row in a dispatch / combine
    // GPU, ns.
    IntTable expert_ns;   // one expert, x = its rows (experts without rows are skipped)
    IntTable permute_ns;  // one unpack (and one repack) of a step, x = the rank's rows of the step
    // Host, ns: issuing one step's expert work, x = experts with rows.
    IntTable expert_issue_ns;
    // Fabric, per rank and direction.
    std::int64_t tx_ps_per_byte = 0;
    std::int64_t rx_ps_per_byte = 0;
    std::int64_t message_ns = 0;   // per message on that side (chunks out; credits back)
    std::int64_t chunk_bytes = 0;  // payload per message; 0: one message per peer
    std::int64_t exchange_ns = 0;  // fixed per exchange (latency, completion notice)
    // cost_aware: a plan replaces the baseline (token_greedy, or fixed) only
    // when estimated this many permille faster (the estimate is not exact).
    int fallback_permille = 20;
    // cost_aware: local search passes (each applies the best single move) and
    // the total number of candidate estimates; together they bound its run time.
    int max_passes = 8;
    int max_evaluations = 160;

    // Throws std::invalid_argument when a required table is missing or a
    // value is negative.
    void validate() const;
};

// Estimated plan-dependent time of one invocation (ns): from the first
// dispatch's issue to the last combine's end on the slowest rank.
// `counts` is N(s, e) at s * num_experts + e.
std::int64_t estimate_layer_ns(const CostModel& model, int world, int num_experts,
                               const std::vector<std::int64_t>& counts, const GroupPlan& plan, bool overlap,
                               int lookahead);

// Incremental form used by the search: keeps per (group, rank) aggregates and
// their times, so moving one expert costs O(world) and an estimate
// O(groups * world) additions and maxima.
class PlanEvaluator {
public:
    static constexpr std::int64_t kNoBound = INT64_MAX;

    PlanEvaluator(const CostModel& model, int world, int num_experts, const std::vector<std::int64_t>& counts,
                  bool overlap, int lookahead);

    void load(const GroupPlan& plan);  // validated; plan.groups() fixes the group count
    // Estimate of the current plan. With a bound it may stop early and then
    // returns some value >= bound (enough to reject a candidate).
    std::int64_t estimate(std::int64_t bound = kNoBound) const;
    void move(int expert, int group);  // expert -> group (any group of the plan)
    void swap_order(int i, int j);     // exchange positions i and j of the order
    const GroupPlan& plan() const { return plan_; }

    // The (group, rank) cell with the largest compute time / dispatch
    // receive volume over the whole plan (lowest group, then rank, on ties).
    std::pair<int, int> busiest_compute_cell() const;
    std::pair<int, int> busiest_receive_cell() const;

private:
    std::size_t at(int group, int rank) const { return static_cast<std::size_t>(group) * world_ + rank; }
    std::int64_t side_ps(std::int64_t rows, std::int64_t row_ps) const;
    static std::int64_t to_ns(std::int64_t ps) { return (ps + 999) / 1000; }
    void refresh_owner(std::size_t cell);
    void refresh_sender(std::size_t cell);

    const CostModel& m_;
    int world_, experts_, epr_;
    bool overlap_;
    int lookahead_;
    std::vector<std::int64_t> counts_;
    std::vector<std::int64_t> rows_;       // [E] rows of expert e (sum over sources)
    std::vector<std::int64_t> in_rows_;    // [E] rows of e from other ranks
    std::vector<std::int64_t> expert_ns_;  // [E] expert_ns(rows) or 0
    std::int64_t tx_row_ps_ = 0, rx_row_ps_ = 0, base_ps_ = 0, exchange_ps_ = 0;
    GroupPlan plan_;
    // [G * W] aggregates of the current mapping (rows) ...
    std::vector<std::int64_t> ffn_, step_rows_, active_, disp_tx_, disp_rx_;
    // ... and their times (ps): GPU work and host issue of the step on the
    // rank; the rank's send / receive side in the step's dispatch and combine.
    std::vector<std::int64_t> compute_, issue_, dtx_, drx_, ctx_, crx_;
    mutable std::vector<std::int64_t> tx_free_, rx_free_, gpu_, dispatch_end_;  // estimate() scratch
};

}  // namespace nano_nccl::planner
