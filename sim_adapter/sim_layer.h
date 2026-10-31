#pragma once

#include "cost_profile.h"
#include "exec/exec_program.h"
#include "sim_wire.h"

#include <cstdint>
#include <string>
#include <vector>

// Virtual-time execution of one EP MoE layer invocation on every rank.
//
// Each rank builds the same exec::LayerProgram the runtime builds and issues
// it with the same exec::ProgramCursor rule. A rank's host is one thread: a
// pass runs PeerExchanger::progress() over the simulated wire, observes
// finished GPU work, then issues every ready op, charging profiled host costs
// to the rank's host clock. GPU ops run in issue order on one stream with
// profiled durations. The counts exchange is simulated for its timing; the
// plan uses the true counts (what that exchange delivers).
namespace nano_nccl::sim {

struct SimLayerConfig {
    int world = 0;
    int num_experts = 0;
    int top_k = 0;
    int hidden = 0;
    int ffn = 0;
    std::string mode = "ffn";  // ffn | probe: which expert cost table applies
    planner::GroupPlan groups;
    exec::PipelineOptions options;
    int slots_per_link = 4;
    std::size_t slot_bytes = std::size_t{1} << 20;
    double copy_scale = 1.0;     // multiplies copy-engine durations
    double compute_scale = 1.0;  // multiplies GPU op durations
    // First-order copy/compute interference from the profile (WireCosts and
    // cost::kGpuCopySlowdown); off: every remote chunk costs the all-busy
    // table and GPU ops their isolated time.
    bool contention = true;
    std::uint64_t iteration = 1;
};

struct SimOpTimes {
    double issue_begin = -1;
    double issue_end = -1;
    double complete = -1;  // same meaning as ep::OpTimes::complete
    // First host observation of each exec::Condition (index = its value).
    double condition[6] = {-1, -1, -1, -1, -1, -1};
    double gpu_start = -1, gpu_end = -1;  // stream ops: device interval
};

struct SimRankResult {
    exec::LayerProgram program;
    exec::EpPlan plan;
    std::vector<SimOpTimes> ops;
    std::vector<std::vector<exec::Message>> messages;  // per op (exchange ops only)
    double gpu_busy = 0;
    double total = 0;  // retire issued
};

struct SimResult {
    std::vector<SimRankResult> ranks;
    double makespan = 0;
    std::vector<LinkStats> links;  // [src * world + dst]
    std::vector<EngineStats> tx, rx;
    std::uint64_t events = 0;
};

// counts[r][e] = N(r, e); tokens[r] = tokens of rank r (only the route and
// reduce costs use it). Throws std::runtime_error on deadlock.
SimResult simulate_layer(const SimLayerConfig& config, const std::vector<std::vector<std::int64_t>>& counts,
                         const std::vector<int>& tokens, const CostProfile& profile);

// GPU duration of an op under the profile (exposed for reports and tests).
double gpu_cost(const CostProfile& profile, const std::string& mode, exec::OpKind kind, const exec::ComputeWork& work);

}  // namespace nano_nccl::sim
