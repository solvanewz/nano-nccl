#include "planner/scheduler.h"

#include <algorithm>
#include <numeric>
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

constexpr SchedulerKind kAllKinds[] = {SchedulerKind::kFixed, SchedulerKind::kTokenGreedy};

GroupPlan token_greedy_plan(const ScheduleRequest& r) {
    const int W = r.world, E = r.num_experts, G = r.groups, epr = E / W;
    if (G < 1 || G > E) {
        throw std::invalid_argument("token_greedy: groups must be in [1, num_experts = " + std::to_string(E) + "]");
    }
    std::vector<std::int64_t> rows(static_cast<std::size_t>(E), 0);
    for (int s = 0; s < W; ++s) {
        for (int e = 0; e < E; ++e) rows[e] += r.counts[static_cast<std::size_t>(s) * E + e];
    }
    GroupPlan plan;
    plan.expert_group.assign(static_cast<std::size_t>(E), -1);
    std::vector<std::int64_t> step_max(static_cast<std::size_t>(G), 0);
    for (int rank = 0; rank < W; ++rank) {
        // LPT: this rank's experts, heaviest first, each into its lightest bin.
        std::vector<int> experts(static_cast<std::size_t>(epr));
        std::iota(experts.begin(), experts.end(), rank * epr);
        std::stable_sort(experts.begin(), experts.end(), [&](int a, int b) { return rows[a] > rows[b]; });
        std::vector<std::int64_t> bin(static_cast<std::size_t>(G), 0);
        std::vector<int> bin_of(static_cast<std::size_t>(epr), -1);
        for (int e : experts) {
            const int b = static_cast<int>(std::min_element(bin.begin(), bin.end()) - bin.begin());
            bin[b] += rows[e];
            bin_of[e - rank * epr] = b;
        }
        // Align ranks: this rank's k-th heaviest bin joins step group k.
        std::vector<int> by_load(static_cast<std::size_t>(G));
        std::iota(by_load.begin(), by_load.end(), 0);
        std::stable_sort(by_load.begin(), by_load.end(), [&](int a, int b) { return bin[a] > bin[b]; });
        std::vector<int> group_of_bin(static_cast<std::size_t>(G));
        for (int k = 0; k < G; ++k) {
            group_of_bin[by_load[k]] = k;
            step_max[k] = std::max(step_max[k], bin[by_load[k]]);
        }
        for (int j = 0; j < epr; ++j) plan.expert_group[rank * epr + j] = group_of_bin[bin_of[j]];
    }
    const int first = static_cast<int>(std::min_element(step_max.begin(), step_max.end()) - step_max.begin());
    std::vector<int> rest;
    for (int g = 0; g < G; ++g) {
        if (g != first) rest.push_back(g);
    }
    std::stable_sort(rest.begin(), rest.end(), [&](int a, int b) { return step_max[a] > step_max[b]; });
    plan.order.push_back(first);
    plan.order.insert(plan.order.end(), rest.begin(), rest.end());
    return plan;
}

}  // namespace

const char* scheduler_name(SchedulerKind kind) {
    switch (kind) {
        case SchedulerKind::kFixed:
            return "fixed";
        case SchedulerKind::kTokenGreedy:
            return "token_greedy";
    }
    return "?";
}

SchedulerKind parse_scheduler(const std::string& name) {
    std::string known;
    for (SchedulerKind k : kAllKinds) {
        if (name == scheduler_name(k)) return k;
        known += std::string(known.empty() ? "" : ", ") + scheduler_name(k);
    }
    throw std::invalid_argument("unknown scheduler '" + name + "' (known: " + known + ")");
}

bool scheduler_reads_counts(SchedulerKind kind) {
    switch (kind) {
        case SchedulerKind::kFixed:
            return false;
        case SchedulerKind::kTokenGreedy:
            return true;
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
        case SchedulerKind::kTokenGreedy:
            plan = token_greedy_plan(request);
            break;
    }
    validate_group_plan(plan, request.num_experts);
    if (plan.groups() != request.groups) {
        throw std::logic_error(std::string("make_schedule: ") + scheduler_name(kind) + " returned " +
                               std::to_string(plan.groups()) + " groups, want " + std::to_string(request.groups));
    }
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
