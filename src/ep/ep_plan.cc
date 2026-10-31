#include "ep/ep_plan.h"

#include <stdexcept>
#include <string>

namespace nano_nccl::ep {

void validate_shape(const EpShape& s) {
    auto fail = [](const std::string& what) { throw std::invalid_argument("EpShape: " + what); };
    if (s.world <= 0 || s.rank < 0 || s.rank >= s.world) fail("bad world/rank");
    if (s.num_experts <= 0 || s.num_experts % s.world != 0) {
        fail("num_experts must be a positive multiple of world");
    }
    if (s.top_k <= 0 || s.top_k > s.num_experts) fail("top_k must be in [1, num_experts]");
    if (s.hidden <= 0 || s.hidden % 8 != 0) fail("hidden must be a positive multiple of 8");
    if (s.ffn < 0) fail("ffn must not be negative");
}

DispatchPlan make_dispatch_plan(const EpShape& shape, const std::vector<std::int64_t>& send_counts,
                                const std::vector<std::int64_t>& recv_counts) {
    validate_shape(shape);
    const int W = shape.world;
    const int E = shape.num_experts;
    const int epr = shape.experts_per_rank();
    const auto rb = static_cast<std::int64_t>(shape.row_bytes());
    if (static_cast<int>(send_counts.size()) != E || static_cast<int>(recv_counts.size()) != W * epr) {
        throw std::invalid_argument("make_dispatch_plan: count vectors have the wrong size");
    }
    for (std::int64_t c : send_counts) {
        if (c < 0) throw std::invalid_argument("make_dispatch_plan: negative send count");
    }
    for (std::int64_t c : recv_counts) {
        if (c < 0) throw std::invalid_argument("make_dispatch_plan: negative receive count");
    }

    DispatchPlan p;
    p.send_counts = send_counts;
    p.send_offsets.assign(static_cast<std::size_t>(E) + 1, 0);
    for (int e = 0; e < E; ++e) p.send_offsets[e + 1] = p.send_offsets[e] + send_counts[e];
    p.send_rows = p.send_offsets[E];

    p.recv_counts = recv_counts;
    p.recv_offsets.assign(static_cast<std::size_t>(W) * epr, 0);
    std::int64_t row = 0;
    for (int i = 0; i < W * epr; ++i) {
        p.recv_offsets[i] = row;
        row += recv_counts[i];
    }
    p.recv_rows = row;

    p.expert_offsets.assign(static_cast<std::size_t>(epr) + 1, 0);
    for (int j = 0; j < epr; ++j) {
        std::int64_t rows = 0;
        for (int s = 0; s < W; ++s) rows += recv_counts[s * epr + j];
        p.expert_offsets[j + 1] = p.expert_offsets[j] + rows;
    }

    auto& d = p.dispatch;
    d.send_bytes.assign(W, 0);
    d.send_offsets.assign(W, 0);
    d.recv_bytes.assign(W, 0);
    d.recv_offsets.assign(W, 0);
    for (int peer = 0; peer < W; ++peer) {
        const std::int64_t first = p.send_offsets[peer * epr];
        const std::int64_t last = p.send_offsets[(peer + 1) * epr];
        d.send_offsets[peer] = static_cast<std::size_t>(first * rb);
        d.send_bytes[peer] = static_cast<std::size_t>((last - first) * rb);
        std::int64_t rows = 0;
        for (int j = 0; j < epr; ++j) rows += recv_counts[peer * epr + j];
        d.recv_offsets[peer] = static_cast<std::size_t>(p.recv_offsets[peer * epr] * rb);
        d.recv_bytes[peer] = static_cast<std::size_t>(rows * rb);
    }
    p.combine = {d.recv_bytes, d.recv_offsets, d.send_bytes, d.send_offsets};

    for (int j = 0; j < epr; ++j) {
        std::int64_t dst = p.expert_offsets[j];
        for (int s = 0; s < W; ++s) {
            const std::int64_t rows = recv_counts[s * epr + j];
            if (rows != 0) {
                const std::int64_t src = p.recv_offsets[s * epr + j];
                p.unpack.push_back({src, dst, rows});
                p.repack.push_back({dst, src, rows});
            }
            dst += rows;
        }
    }
    return p;
}

}  // namespace nano_nccl::ep
