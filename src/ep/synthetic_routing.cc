#include "ep/synthetic_routing.h"

#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

namespace nano_nccl::ep {

Routing make_routing(const EpShape& s, int tokens, double skew, int only_rank, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    const int first = only_rank >= 0 ? only_rank * s.experts_per_rank() : 0;
    const int count = only_rank >= 0 ? s.experts_per_rank() : s.num_experts;
    if (s.top_k > count) throw std::invalid_argument("make_routing: top_k exceeds the experts to draw from");
    std::vector<double> base(count);
    for (int e = 0; e < count; ++e) base[e] = std::pow(e + 1.0, -skew);
    Routing r;
    for (int t = 0; t < tokens; ++t) {
        std::vector<double> p = base;
        float sum = 0;
        std::vector<float> w(s.top_k);
        for (int k = 0; k < s.top_k; ++k) {
            std::discrete_distribution<int> pick(p.begin(), p.end());
            const int e = pick(rng);
            p[e] = 0;
            r.ids.push_back(first + e);
            w[k] = 0.1f + static_cast<float>(rng() % 1000) / 1000.0f;
            sum += w[k];
        }
        for (float x : w) r.weights.push_back(x / sum);
    }
    return r;
}

std::vector<std::int64_t> expert_counts(const std::vector<std::int32_t>& ids, int num_experts) {
    std::vector<std::int64_t> counts(num_experts, 0);
    for (std::int32_t e : ids) {
        if (e < 0 || e >= num_experts) throw std::invalid_argument("expert_counts: id " + std::to_string(e));
        ++counts[e];
    }
    return counts;
}

}  // namespace nano_nccl::ep
