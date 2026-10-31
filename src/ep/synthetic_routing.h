#pragma once

#include "ep/ep_plan.h"

#include <cstdint>
#include <vector>

// Synthetic router output for tests, benchmarks, profiling and the simulator,
// so all of them see the same token -> expert assignment for a given seed.
namespace nano_nccl::ep {

struct Routing {
    std::vector<std::int32_t> ids;  // [tokens, top_k] global expert ids, distinct per token
    std::vector<float> weights;     // [tokens, top_k], each row sums to 1
};

// Top-k distinct experts per token, drawn with probability ~ (e + 1)^-skew;
// with only_rank >= 0, only that rank's experts are drawn.
Routing make_routing(const EpShape& shape, int tokens, double skew, int only_rank, std::uint64_t seed);

// counts[e] = number of entries routed to expert e.
std::vector<std::int64_t> expert_counts(const std::vector<std::int32_t>& ids, int num_experts);

// Seed of rank r's routing in the benchmark runs (tests/ep_moe_test --ffn).
inline std::uint64_t benchmark_routing_seed(int rank) { return 4242 + static_cast<std::uint64_t>(rank); }

}  // namespace nano_nccl::ep
