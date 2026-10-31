#include "planner/group_plan.h"

#include <stdexcept>
#include <string>

namespace nano_nccl::planner {

void validate_group_plan(const GroupPlan& plan, int num_experts) {
    const int groups = plan.groups();
    if (groups <= 0) throw std::invalid_argument("GroupPlan: no groups");
    if (static_cast<int>(plan.expert_group.size()) != num_experts) {
        throw std::invalid_argument("GroupPlan: expert_group has " + std::to_string(plan.expert_group.size()) +
                                    " entries for " + std::to_string(num_experts) + " experts");
    }
    for (int g : plan.expert_group) {
        if (g < 0 || g >= groups) throw std::invalid_argument("GroupPlan: group " + std::to_string(g) + " out of range");
    }
    std::vector<char> seen(static_cast<std::size_t>(groups), 0);
    for (int g : plan.order) {
        if (g < 0 || g >= groups || seen[static_cast<std::size_t>(g)]) {
            throw std::invalid_argument("GroupPlan: order is not a permutation of the groups");
        }
        seen[static_cast<std::size_t>(g)] = 1;
    }
}

GroupPlan fixed_group_plan(int world, int num_experts, int groups) {
    if (world <= 0 || num_experts <= 0 || num_experts % world != 0) {
        throw std::invalid_argument("fixed_group_plan: num_experts must be a positive multiple of world");
    }
    const int per_rank = num_experts / world;
    if (groups < 1 || groups > per_rank) {
        throw std::invalid_argument("fixed_group_plan: groups must be in [1, experts per rank = " +
                                    std::to_string(per_rank) + "]");
    }
    GroupPlan plan;
    plan.expert_group.resize(static_cast<std::size_t>(num_experts));
    for (int e = 0; e < num_experts; ++e) plan.expert_group[e] = (e % per_rank) * groups / per_rank;
    for (int g = 0; g < groups; ++g) plan.order.push_back(g);
    return plan;
}

}  // namespace nano_nccl::planner
