#pragma once

#include "planner/cost_model.h"
#include "planner/group_plan.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Scheduler switch shared by the real runtime and the simulator: one executor,
// and the expert grouping of each layer invocation comes from the selected
// scheduler. Pure C++, no CUDA.
//
//   fixed         B1: fixed_group_plan(), independent of the routing.
//   token_greedy  simple strong baseline for B2: balances expert rows per
//                 step across ranks (see make_schedule()); ignores where the
//                 rows come from, i.e. communication.
//   cost_aware    B2: picks the plan with the smallest estimated layer time
//                 under a CostModel (compute, traffic per rank and step, the
//                 pipeline's issue order); keeps its baseline (token_greedy,
//                 or fixed when clearly better) when the gain is too small to
//                 trust the estimate.
//
// Schedulers take the same request and return the same GroupPlan, so the
// layer and the simulator need no changes beyond selecting them.
namespace nano_nccl::planner {

enum class SchedulerKind : int {
    kFixed = 0,
    kTokenGreedy,
    kCostAware,
};

const char* scheduler_name(SchedulerKind kind);
// Inverse of scheduler_name(); throws std::invalid_argument for unknown names.
SchedulerKind parse_scheduler(const std::string& name);

struct ScheduleRequest {
    int world = 0;
    int num_experts = 0;  // global; a multiple of world (contiguous placement)
    int groups = 1;       // the group count of the plan (fixed: <= experts per rank; others: <= num_experts)
    // N(s, e) at index s * num_experts + e: entries rank s routes to expert e.
    // Every rank must pass the same values. May be empty for a scheduler that
    // does not read it (scheduler_reads_counts() == false).
    std::vector<std::int64_t> counts;
    // cost_aware only: the pipeline the plan will run with (exec::PipelineOptions)
    // and the cost model; both must be identical on every rank.
    bool overlap = true;
    int lookahead = 1;
    std::shared_ptr<const CostModel> cost = nullptr;
};

bool scheduler_reads_counts(SchedulerKind kind);

// Deterministic (integer arithmetic only): equal requests give equal plans,
// so every rank that sees the same request derives the same plan without
// publishing it. Every scheduler returns exactly request.groups groups (some
// may be empty). Throws std::invalid_argument for invalid requests.
//
// token_greedy: rows(e) = sum over s of N(s, e); load(r, g) = rows of rank r's
// experts in group g. A step is as slow as its busiest rank and the pipeline
// wants equal steps, so: each rank spreads its own experts over the G groups
// by LPT (heaviest first into its lightest bin, ties: lower id / lower bin),
// then the ranks are aligned: every rank's k-th heaviest bin forms step group
// k. Order: the group with the smallest step maximum (max over r of
// load(r, g)) first, so a short first dispatch starts the pipeline early, then
// descending step maximum, so the last, exposed step is short; ties by id.
//
// cost_aware (needs request.cost): the baseline is token_greedy, or fixed
// (groups <= experts per rank) when the CostModel estimates it at least
// CostModel::fallback_permille faster. From the baseline it runs a bounded
// local search: each pass tries moving every expert of the two bottleneck
// cells (the (group, rank) with the most compute and the one receiving the
// most rows) to every other group, and swapping adjacent steps, then applies
// the single best strict improvement; at most CostModel::max_passes passes
// and max_evaluations estimates. Returns the baseline unless the result is
// estimated at least fallback_permille faster than it.
GroupPlan make_schedule(SchedulerKind kind, const ScheduleRequest& request);

// Test generator, not a scheduler: a random expert -> group mapping over
// 1 .. max_groups groups (some groups may be empty or hold one rank's experts
// only) and a random group order. Deterministic in its arguments on every
// platform (uses only std::mt19937_64's raw output).
GroupPlan random_group_plan(int num_experts, int max_groups, std::uint64_t seed);

}  // namespace nano_nccl::planner
