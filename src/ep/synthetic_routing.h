#pragma once

#include "ep/ep_plan.h"

#include <cstdint>
#include <string>
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

// Top-k distinct experts per token, drawn with probability ~ expert_weights[e]
// (E entries, non-negative, at least top_k positive). make_routing() above is
// this with zipf weights (same draws for the same seed).
Routing make_routing(const EpShape& shape, int tokens, const std::vector<double>& expert_weights,
                     std::uint64_t seed);

// Token -> expert distribution families (plan §D workloads). A name is
// "<family>" or "<family>:<param>"; `skew` is the zipf exponent.
//   zipf          w(e) = (e + 1)^-skew over global ids: the hottest experts
//                 all sit on rank 0 (the B1 benchmark distribution).
//   zipf_spread   zipf over a fixed pseudo-random permutation of all experts:
//                 hot experts spread over the ranks, rank totals differ.
//   rank_skew     every rank equally popular in total; inside rank r, local
//                 expert j has weight (pi_r(j) + 1)^-skew with a different
//                 permutation pi_r per rank: balanced ranks, skewed steps.
//   local:P       rank_skew, and each source weights the experts of its own
//                 rank P times: same expert loads, less remote traffic.
//   pair:P        rank_skew, and source s weights the experts of rank
//                 (s + 1) % W P times: same expert loads, traffic
//                 concentrated on one destination per source.
//   hot_rank:P    rank_skew with rank 0's experts weighted P times: one rank
//                 persistently hot (grouping cannot move its compute).
// All sources use the same expert weights except local and pair.
struct Workload {
    std::string family = "zipf";
    double param = 0;  // P, where the family has one
    double skew = 0;

    std::string name() const;  // "<family>[:<param>]"
};
Workload parse_workload(const std::string& name, double skew);  // throws std::invalid_argument

// Weights of global experts for tokens of source rank shape.rank.
std::vector<double> source_expert_weights(const Workload& workload, const EpShape& shape);

// counts[r][e] = N(r, e) when every rank r routes `tokens` tokens of
// `workload` with routing_seed(r, seed) (shape.rank is ignored).
std::vector<std::vector<std::int64_t>> workload_counts(const Workload& workload, const EpShape& shape, int tokens,
                                                       std::uint64_t seed = 0);

// counts[e] = number of entries routed to expert e.
std::vector<std::int64_t> expert_counts(const std::vector<std::int32_t>& ids, int num_experts);

// Seed of rank r's routing in the benchmark runs (tests/ep_moe_test --ffn).
inline std::uint64_t benchmark_routing_seed(int rank) { return 4242 + static_cast<std::uint64_t>(rank); }
// Rank r's routing seed for sample `seed` of a workload; seed 0 is the
// benchmark seed.
inline std::uint64_t routing_seed(int rank, std::uint64_t seed) {
    return benchmark_routing_seed(rank) + seed * 1000003ull;
}

}  // namespace nano_nccl::ep
