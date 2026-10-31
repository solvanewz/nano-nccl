#pragma once

#include <vector>

// Expert grouping and execution order of one MoE layer invocation, shared by
// the real runtime and the simulator (pure C++, no CUDA).
//
// A plan is fixed for one layer invocation: the expert -> group mapping must
// not change while that invocation's messages are in flight. B1 uses the
// `fixed` scheduler below; B2's schedulers produce the same structure.
namespace nano_nccl::planner {

struct GroupPlan {
    std::vector<int> expert_group;  // [num_experts]: global expert -> group
    std::vector<int> order;         // groups in execution order; a permutation of [0, groups)

    int groups() const { return static_cast<int>(order.size()); }
};

// Throws std::invalid_argument unless every expert maps to a group in
// [0, groups) and `order` is a permutation. Empty groups are allowed.
void validate_group_plan(const GroupPlan& plan, int num_experts);

// B1 `fixed` scheduler: local expert j of every rank goes to group
// j * groups / experts_per_rank, and groups run in index order. Every group
// then has experts on every rank, so each step exchanges with every peer.
// Requires 1 <= groups <= num_experts / world.
GroupPlan fixed_group_plan(int world, int num_experts, int groups);

}  // namespace nano_nccl::planner
