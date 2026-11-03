#include "ep/synthetic_routing.h"

#include <cmath>
#include <cstdlib>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

namespace nano_nccl::ep {

namespace {

// Fixed pseudo-random permutation of [0, n), the same on every platform
// (mt19937_64 raw output only).
std::vector<int> permutation(int n, std::uint64_t seed) {
    std::vector<int> p(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) p[i] = i;
    std::mt19937_64 rng(seed);
    for (int i = n - 1; i > 0; --i) std::swap(p[i], p[static_cast<int>(rng() % static_cast<std::uint64_t>(i + 1))]);
    return p;
}

}  // namespace

Routing make_routing(const EpShape& s, int tokens, const std::vector<double>& expert_weights, std::uint64_t seed) {
    if (static_cast<int>(expert_weights.size()) != s.num_experts) {
        throw std::invalid_argument("make_routing: need one weight per expert");
    }
    int positive = 0;
    for (double w : expert_weights) {
        if (!(w >= 0)) throw std::invalid_argument("make_routing: negative or NaN weight");
        positive += w > 0 ? 1 : 0;
    }
    if (s.top_k > positive) throw std::invalid_argument("make_routing: top_k exceeds the experts to draw from");
    std::mt19937_64 rng(seed);
    Routing r;
    for (int t = 0; t < tokens; ++t) {
        std::vector<double> p = expert_weights;
        float sum = 0;
        std::vector<float> w(s.top_k);
        for (int k = 0; k < s.top_k; ++k) {
            std::discrete_distribution<int> pick(p.begin(), p.end());
            const int e = pick(rng);
            p[e] = 0;
            r.ids.push_back(e);
            w[k] = 0.1f + static_cast<float>(rng() % 1000) / 1000.0f;
            sum += w[k];
        }
        for (float x : w) r.weights.push_back(x / sum);
    }
    return r;
}

Routing make_routing(const EpShape& s, int tokens, double skew, int only_rank, std::uint64_t seed) {
    if (only_rank < 0) {
        std::vector<double> base(static_cast<std::size_t>(s.num_experts));
        for (int e = 0; e < s.num_experts; ++e) base[e] = std::pow(e + 1.0, -skew);
        return make_routing(s, tokens, base, seed);
    }
    // Only rank only_rank's experts, zipf over their local index.
    std::vector<double> base(static_cast<std::size_t>(s.num_experts), 0.0);
    const int first = only_rank * s.experts_per_rank();
    for (int j = 0; j < s.experts_per_rank(); ++j) base[first + j] = std::pow(j + 1.0, -skew);
    return make_routing(s, tokens, base, seed);
}

std::string Workload::name() const {
    if (family == "zipf" || family == "zipf_spread" || family == "rank_skew") return family;
    std::ostringstream o;
    o << family << ":" << param;
    return o.str();
}

Workload parse_workload(const std::string& name, double skew) {
    Workload w;
    w.skew = skew;
    const auto colon = name.find(':');
    w.family = name.substr(0, colon);
    const bool has_param = w.family == "local" || w.family == "pair" || w.family == "hot_rank";
    const bool no_param = w.family == "zipf" || w.family == "zipf_spread" || w.family == "rank_skew";
    if (!has_param && !no_param) {
        throw std::invalid_argument("unknown workload '" + name +
                                    "' (zipf, zipf_spread, rank_skew, local:P, pair:P, hot_rank:P)");
    }
    if (has_param != (colon != std::string::npos)) {
        throw std::invalid_argument("workload '" + name + "': " + w.family +
                                    (has_param ? " needs a parameter" : " takes no parameter"));
    }
    if (has_param) {
        char* end = nullptr;
        w.param = std::strtod(name.c_str() + colon + 1, &end);
        if (*end != '\0' || !(w.param > 0)) throw std::invalid_argument("workload '" + name + "': bad parameter");
    }
    return w;
}

std::vector<double> source_expert_weights(const Workload& w, const EpShape& s) {
    validate_shape(s);
    const int E = s.num_experts, epr = s.experts_per_rank();
    std::vector<double> out(static_cast<std::size_t>(E));
    if (w.family == "zipf") {
        for (int e = 0; e < E; ++e) out[e] = std::pow(e + 1.0, -w.skew);
        return out;
    }
    if (w.family == "zipf_spread") {
        const std::vector<int> pi = permutation(E, 0x5eed0001ull + static_cast<std::uint64_t>(E));
        for (int e = 0; e < E; ++e) out[e] = std::pow(pi[e] + 1.0, -w.skew);
        return out;
    }
    // rank_skew and the families built on it.
    for (int r = 0; r < s.world; ++r) {
        const std::vector<int> pi = permutation(epr, 0x5eed1000ull + static_cast<std::uint64_t>(r) * 7919);
        for (int j = 0; j < epr; ++j) out[r * epr + j] = std::pow(pi[j] + 1.0, -w.skew);
    }
    if (w.family == "rank_skew") return out;
    int boosted = -1;
    if (w.family == "local") boosted = s.rank;
    else if (w.family == "pair") boosted = (s.rank + 1) % s.world;
    else if (w.family == "hot_rank") boosted = 0;
    else throw std::invalid_argument("unknown workload family '" + w.family + "'");
    for (int j = 0; j < epr; ++j) out[boosted * epr + j] *= w.param;
    return out;
}

std::vector<std::vector<std::int64_t>> workload_counts(const Workload& workload, const EpShape& shape, int tokens) {
    std::vector<std::vector<std::int64_t>> counts;
    for (int r = 0; r < shape.world; ++r) {
        EpShape s = shape;
        s.rank = r;
        counts.push_back(expert_counts(
            make_routing(s, tokens, source_expert_weights(workload, s), benchmark_routing_seed(r)).ids,
            shape.num_experts));
    }
    return counts;
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
