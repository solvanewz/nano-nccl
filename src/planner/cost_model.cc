#include "planner/cost_model.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace nano_nccl::planner {

IntTable::IntTable(std::vector<std::pair<std::int64_t, std::int64_t>> points) : points_(std::move(points)) {
    if (points_.empty()) throw std::invalid_argument("IntTable: no points");
    for (std::size_t i = 1; i < points_.size(); ++i) {
        if (points_[i].first <= points_[i - 1].first) throw std::invalid_argument("IntTable: x must ascend");
    }
}

std::int64_t IntTable::operator()(std::int64_t x) const {
    if (points_.empty()) throw std::logic_error("IntTable: empty");
    if (points_.size() == 1 || x <= points_.front().first) return points_.front().second;
    auto lerp = [x](const std::pair<std::int64_t, std::int64_t>& a, const std::pair<std::int64_t, std::int64_t>& b) {
        return a.second + (b.second - a.second) * (x - a.first) / (b.first - a.first);
    };
    const auto hi = std::lower_bound(points_.begin(), points_.end(), x,
                                     [](const std::pair<std::int64_t, std::int64_t>& p, std::int64_t v) {
                                         return p.first < v;
                                     });
    if (hi == points_.end()) return std::max<std::int64_t>(0, lerp(points_[points_.size() - 2], points_.back()));
    if (hi->first == x) return hi->second;
    return lerp(*(hi - 1), *hi);
}

void CostModel::validate() const {
    auto fail = [](const std::string& what) { throw std::invalid_argument("CostModel: " + what); };
    if (row_bytes <= 0) fail("row_bytes must be positive");
    if (expert_ns.empty()) fail("no expert table");
    if (permute_ns.empty()) fail("no permute table");
    if (expert_issue_ns.empty()) fail("no expert issue table");
    if (tx_ps_per_byte < 0 || rx_ps_per_byte < 0 || message_ns < 0 || chunk_bytes < 0 || exchange_ns < 0 ||
        ranks_per_node < 0 || local_tx_ps_per_byte < 0 || local_rx_ps_per_byte < 0 || local_message_ns < 0) {
        fail("negative fabric cost");
    }
    if (fallback_permille < 0 || fallback_permille >= 1000) fail("fallback_permille must be in [0, 1000)");
    if (max_passes < 0 || max_evaluations < 0) fail("negative search budget");
}

PlanEvaluator::PlanEvaluator(const CostModel& model, int world, int num_experts,
                             const std::vector<std::int64_t>& counts, bool overlap, int lookahead)
    : m_(model), world_(world), experts_(num_experts), epr_(0), rpn_(model.ranks_per_node), overlap_(overlap),
      lookahead_(lookahead), counts_(counts) {
    model.validate();
    if (world <= 0 || num_experts <= 0 || num_experts % world != 0) {
        throw std::invalid_argument("PlanEvaluator: num_experts must be a positive multiple of world");
    }
    if (counts.size() != static_cast<std::size_t>(world) * num_experts) {
        throw std::invalid_argument("PlanEvaluator: counts must have world * num_experts entries");
    }
    if (lookahead < 0) throw std::invalid_argument("PlanEvaluator: negative lookahead");
    if (rpn_ > 1 && world % rpn_ != 0) {
        throw std::invalid_argument("PlanEvaluator: world must be a multiple of ranks_per_node");
    }
    epr_ = num_experts / world;
    rows_.assign(static_cast<std::size_t>(num_experts), 0);
    in_rows_.assign(static_cast<std::size_t>(num_experts), 0);
    in_lrows_.assign(static_cast<std::size_t>(num_experts), 0);
    expert_ns_.assign(static_cast<std::size_t>(num_experts), 0);
    for (int e = 0; e < num_experts; ++e) {
        const int owner = e / epr_;
        for (int s = 0; s < world; ++s) {
            const std::int64_t n = counts_[static_cast<std::size_t>(s) * num_experts + e];
            rows_[e] += n;
            if (s == owner) continue;
            (node(s) == node(owner) ? in_lrows_ : in_rows_)[e] += n;
        }
        expert_ns_[e] = rows_[e] > 0 ? m_.expert_ns(rows_[e]) : 0;
    }
    // Per row on each side: its bytes at the side's rate plus its share of
    // the per-message cost (one message per chunk_bytes); per exchange and
    // side: one message to or from each peer.
    auto row_ps = [&](std::int64_t ps_per_byte, std::int64_t message_ns) {
        const std::int64_t share = m_.chunk_bytes > 0 ? 1000 * message_ns * m_.row_bytes / m_.chunk_bytes : 0;
        return m_.row_bytes * ps_per_byte + share;
    };
    tx_row_ps_ = row_ps(m_.tx_ps_per_byte, m_.message_ns);
    rx_row_ps_ = row_ps(m_.rx_ps_per_byte, m_.message_ns);
    const int local_peers = rpn_ > 1 ? rpn_ - 1 : 0;
    base_ps_ = 1000 * m_.message_ns * (world - 1 - local_peers);
    ltx_row_ps_ = row_ps(m_.local_tx_ps_per_byte, m_.local_message_ns);
    lrx_row_ps_ = row_ps(m_.local_rx_ps_per_byte, m_.local_message_ns);
    lbase_ps_ = 1000 * m_.local_message_ns * local_peers;
    exchange_ps_ = 1000 * m_.exchange_ns;
}

void PlanEvaluator::load(const GroupPlan& plan) {
    validate_group_plan(plan, experts_);
    plan_ = plan;
    const std::size_t n = static_cast<std::size_t>(plan.groups()) * world_;
    for (auto* v : {&ffn_, &step_rows_, &active_, &disp_tx_, &disp_rx_, &disp_ltx_, &disp_lrx_, &compute_, &issue_,
                    &dtx_, &drx_, &ctx_, &crx_, &ldtx_, &ldrx_, &lctx_, &lcrx_}) {
        v->assign(n, 0);
    }
    for (int e = 0; e < experts_; ++e) {
        const int g = plan_.expert_group[e], owner = e / epr_;
        const std::size_t k = at(g, owner);
        ffn_[k] += expert_ns_[e];
        step_rows_[k] += rows_[e];
        active_[k] += rows_[e] > 0 ? 1 : 0;
        disp_rx_[k] += in_rows_[e];
        disp_lrx_[k] += in_lrows_[e];
        for (int s = 0; s < world_; ++s) {
            if (s == owner) continue;
            (node(s) == node(owner) ? disp_ltx_ : disp_tx_)[at(g, s)] +=
                counts_[static_cast<std::size_t>(s) * experts_ + e];
        }
    }
    for (std::size_t k = 0; k < n; ++k) {
        refresh_owner(k);
        refresh_sender(k);
    }
}

std::int64_t PlanEvaluator::side_ps(std::int64_t rows, std::int64_t row_ps, std::int64_t base_ps) const {
    return rows * row_ps + base_ps;
}

// The owner's cell: its compute, and what it receives in the dispatch and
// sends back in the combine.
void PlanEvaluator::refresh_owner(std::size_t k) {
    compute_[k] = 1000 * (2 * m_.permute_ns(step_rows_[k]) + ffn_[k]);
    issue_[k] = 1000 * m_.expert_issue_ns(active_[k]);
    drx_[k] = side_ps(disp_rx_[k], rx_row_ps_, base_ps_);
    ctx_[k] = side_ps(disp_rx_[k], tx_row_ps_, base_ps_);
    ldrx_[k] = side_ps(disp_lrx_[k], lrx_row_ps_, lbase_ps_);
    lctx_[k] = side_ps(disp_lrx_[k], ltx_row_ps_, lbase_ps_);
}

// A source's cell: what it sends in the dispatch and gets back in the combine.
void PlanEvaluator::refresh_sender(std::size_t k) {
    dtx_[k] = side_ps(disp_tx_[k], tx_row_ps_, base_ps_);
    crx_[k] = side_ps(disp_tx_[k], rx_row_ps_, base_ps_);
    ldtx_[k] = side_ps(disp_ltx_[k], ltx_row_ps_, lbase_ps_);
    lcrx_[k] = side_ps(disp_ltx_[k], lrx_row_ps_, lbase_ps_);
}

void PlanEvaluator::move(int e, int group) {
    const int from = plan_.expert_group[e];
    if (from == group) return;
    const int owner = e / epr_;
    auto apply = [&](int g, std::int64_t sign) {
        const std::size_t k = at(g, owner);
        ffn_[k] += sign * expert_ns_[e];
        step_rows_[k] += sign * rows_[e];
        active_[k] += sign * (rows_[e] > 0 ? 1 : 0);
        disp_rx_[k] += sign * in_rows_[e];
        disp_lrx_[k] += sign * in_lrows_[e];
        refresh_owner(k);
        for (int s = 0; s < world_; ++s) {
            const std::int64_t n = counts_[static_cast<std::size_t>(s) * experts_ + e];
            if (s == owner || n == 0) continue;
            (node(s) == node(owner) ? disp_ltx_ : disp_tx_)[at(g, s)] += sign * n;
            refresh_sender(at(g, s));
        }
    };
    if (from >= 0) apply(from, -1);
    apply(group, +1);
    plan_.expert_group[e] = group;
}

void PlanEvaluator::swap_order(int i, int j) { std::swap(plan_.order[i], plan_.order[j]); }

std::pair<int, int> PlanEvaluator::busiest_compute_cell() const {
    const std::size_t k = static_cast<std::size_t>(std::max_element(compute_.begin(), compute_.end()) - compute_.begin());
    return {static_cast<int>(k / world_), static_cast<int>(k % world_)};
}

std::pair<int, int> PlanEvaluator::busiest_receive_cell() const {
    // By receive time over both fabrics (rows alone when there is one).
    std::size_t best = 0;
    for (std::size_t k = 1; k < drx_.size(); ++k) {
        if (std::max(drx_[k], ldrx_[k]) > std::max(drx_[best], ldrx_[best])) best = k;
    }
    if (rpn_ <= 1) {
        best = static_cast<std::size_t>(std::max_element(disp_rx_.begin(), disp_rx_.end()) - disp_rx_.begin());
    }
    return {static_cast<int>(best / world_), static_cast<int>(best % world_)};
}

std::int64_t PlanEvaluator::estimate(std::int64_t bound_ns) const {
    const int G = plan_.groups(), W = world_;
    // Internally ps (no division on the hot path); the result is ns, rounded up.
    const std::int64_t bound = bound_ns == kNoBound ? kNoBound : bound_ns * 1000;
    tx_free_.assign(W, 0);
    rx_free_.assign(W, 0);
    ltx_free_.assign(W, 0);
    lrx_free_.assign(W, 0);
    gpu_.assign(W, 0);
    dispatch_end_.assign(static_cast<std::size_t>(G), 0);
    // Every time computed below is a lower bound of the result (each feeds a
    // later combine through a maximum), so a value past `bound` ends the run.
    const bool local = rpn_ > 1;
    auto exchange = [&](std::int64_t issue, const std::int64_t* tx, const std::int64_t* rx, const std::int64_t* ltx,
                        const std::int64_t* lrx) {
        std::int64_t end = issue;
        for (int r = 0; r < W; ++r) {
            tx_free_[r] = std::max(issue, tx_free_[r]) + tx[r];
            rx_free_[r] = std::max(issue, rx_free_[r]) + rx[r];
            end = std::max({end, tx_free_[r], rx_free_[r]});
            if (local) {
                ltx_free_[r] = std::max(issue, ltx_free_[r]) + ltx[r];
                lrx_free_[r] = std::max(issue, lrx_free_[r]) + lrx[r];
                end = std::max({end, ltx_free_[r], lrx_free_[r]});
            }
        }
        return end + exchange_ps_;
    };
    auto dispatch = [&](int i, std::int64_t issue) {
        const std::size_t k = at(plan_.order[i], 0);
        dispatch_end_[i] = exchange(issue, &dtx_[k], &drx_[k], &ldtx_[k], &ldrx_[k]);
    };
    auto experts = [&](int i) {
        const std::size_t k = at(plan_.order[i], 0);
        std::int64_t done = 0;
        for (int r = 0; r < W; ++r) {
            gpu_[r] = std::max(dispatch_end_[i] + issue_[k + r], gpu_[r]) + compute_[k + r];
            done = std::max(done, gpu_[r]);
        }
        return done;
    };
    // Back to the sources: each owner sends what it received.
    auto combine = [&](int i, std::int64_t issue) {
        const std::size_t k = at(plan_.order[i], 0);
        return exchange(issue, &ctx_[k], &crx_[k], &lctx_[k], &lcrx_[k]);
    };
    std::int64_t total = 0;
    if (!overlap_) {
        std::int64_t t = 0;
        for (int i = 0; i < G; ++i) {
            dispatch(i, t);
            t = combine(i, experts(i));
            total = std::max(total, t);
            if (total >= bound) return to_ns(total);
        }
        return to_ns(total);
    }
    const int lead = std::min(lookahead_, G - 1);
    for (int i = 0; i <= lead; ++i) dispatch(i, 0);
    for (int i = 0; i < G; ++i) {
        const std::int64_t done = experts(i);
        total = std::max(total, combine(i, done));
        if (total >= bound) return to_ns(total);
        if (i + lead + 1 < G) dispatch(i + lead + 1, done);
    }
    return to_ns(total);
}

std::int64_t estimate_layer_ns(const CostModel& model, int world, int num_experts,
                               const std::vector<std::int64_t>& counts, const GroupPlan& plan, bool overlap,
                               int lookahead) {
    PlanEvaluator ev(model, world, num_experts, counts, overlap, lookahead);
    ev.load(plan);
    return ev.estimate();
}

}  // namespace nano_nccl::planner
