#include "planner/scheduler.h"

#include <random>
#include <stdexcept>
#include <string>

namespace nano_nccl::planner {

namespace {

void validate_request(SchedulerKind kind, const ScheduleRequest& r) {
    if (r.world <= 0 || r.num_experts <= 0 || r.num_experts % r.world != 0) {
        throw std::invalid_argument("ScheduleRequest: num_experts must be a positive multiple of world");
    }
    const std::size_t want = static_cast<std::size_t>(r.world) * r.num_experts;
    if (r.counts.empty()) {
        if (scheduler_reads_counts(kind)) {
            throw std::invalid_argument(std::string("ScheduleRequest: scheduler ") + scheduler_name(kind) +
                                        " needs the counts");
        }
        return;
    }
    if (r.counts.size() != want) {
        throw std::invalid_argument("ScheduleRequest: counts has " + std::to_string(r.counts.size()) +
                                    " entries, want world * num_experts = " + std::to_string(want));
    }
    for (std::int64_t c : r.counts) {
        if (c < 0) throw std::invalid_argument("ScheduleRequest: negative count");
    }
}

}  // namespace

const char* scheduler_name(SchedulerKind kind) {
    switch (kind) {
        case SchedulerKind::kFixed:
            return "fixed";
    }
    return "?";
}

SchedulerKind parse_scheduler(const std::string& name) {
    for (SchedulerKind k : {SchedulerKind::kFixed}) {
        if (name == scheduler_name(k)) return k;
    }
    throw std::invalid_argument("unknown scheduler '" + name + "' (known: fixed)");
}

bool scheduler_reads_counts(SchedulerKind kind) {
    switch (kind) {
        case SchedulerKind::kFixed:
            return false;
    }
    return false;
}

GroupPlan make_schedule(SchedulerKind kind, const ScheduleRequest& request) {
    validate_request(kind, request);
    GroupPlan plan;
    switch (kind) {
        case SchedulerKind::kFixed:
            plan = fixed_group_plan(request.world, request.num_experts, request.groups);
            break;
    }
    validate_group_plan(plan, request.num_experts);
    return plan;
}

GroupPlan random_group_plan(int num_experts, int max_groups, std::uint64_t seed) {
    if (num_experts <= 0 || max_groups <= 0) {
        throw std::invalid_argument("random_group_plan: num_experts and max_groups must be positive");
    }
    std::mt19937_64 rng(seed);
    const int groups = 1 + static_cast<int>(rng() % static_cast<std::uint64_t>(max_groups));
    GroupPlan plan;
    plan.expert_group.reserve(static_cast<std::size_t>(num_experts));
    for (int e = 0; e < num_experts; ++e) {
        plan.expert_group.push_back(static_cast<int>(rng() % static_cast<std::uint64_t>(groups)));
    }
    plan.order.resize(static_cast<std::size_t>(groups));
    for (int g = 0; g < groups; ++g) plan.order[g] = g;
    for (int i = groups - 1; i > 0; --i) {  // Fisher-Yates
        const int j = static_cast<int>(rng() % static_cast<std::uint64_t>(i + 1));
        std::swap(plan.order[i], plan.order[j]);
    }
    validate_group_plan(plan, num_experts);
    return plan;
}

}  // namespace nano_nccl::planner
