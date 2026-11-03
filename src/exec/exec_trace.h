#pragma once

#include "exec/exec_program.h"

#include <cstdint>
#include <string>
#include <vector>

// A measured run of one layer configuration on every rank: what the
// simulator needs to replay it (shape, counts, plan, options) and what the
// real code observed (per-op times of the last iteration, stage means).
//
// Text format:
//   nano-ep-trace 1
//   key <exec::ProfileKey::to_string()>
//   options overlap=<0|1> lookahead=<L> iters=<n>
//   plan timing=<ahead|from_counts> scheduler=<name>   (optional; default ahead/fixed)
//   expert_group <g_0> ... <g_{E-1}>                     (the plan that ran)
//   order <group> ...
//   rank <r> tokens=<t> route=<us> counts=<us> pipeline=<us> reduce=<us> total=<us>
//   counts <r> <N(r,0)> ... <N(r,E-1)>
//   op <r> <index> <kind> <step> <issue_begin> <issue_end> <complete>
namespace nano_nccl::exec {

struct TraceOp {
    int rank = 0;
    int op = 0;
    OpKind kind = OpKind::kRoute;
    int step = -1;
    double issue_begin = -1, issue_end = -1, complete = -1;
};

struct TraceRank {
    int tokens = 0;
    // Means over the timed iterations (ep::ForwardTimes fields).
    double route = 0, counts = 0, pipeline = 0, reduce = 0, total = 0;
    std::vector<std::int64_t> expert_counts;  // N(r, e), [E]
};

struct LayerTrace {
    ProfileKey key;
    planner::GroupPlan groups;  // the plan that ran
    PipelineOptions options;
    // How it was obtained. kFromCounts: the scheduler ran on the global counts
    // at the plan op, with groups.groups() groups.
    PlanTiming timing = PlanTiming::kAhead;
    planner::SchedulerKind scheduler = planner::SchedulerKind::kFixed;
    // The source that reproduces the run's program and plan.
    PlanSource source() const {
        return timing == PlanTiming::kAhead ? PlanSource::ahead(groups)
                                            : PlanSource::from_counts(scheduler, groups.groups());
    }
    int iters = 0;
    std::vector<TraceRank> ranks;
    std::vector<TraceOp> ops;  // last timed iteration, every rank

    std::string serialize() const;
    static LayerTrace parse(const std::string& text, const std::string& where = "trace");
    static LayerTrace load(const std::string& path);
    void save(const std::string& path) const;
};

OpKind parse_op_kind(const std::string& name);  // inverse of op_kind_name

}  // namespace nano_nccl::exec
