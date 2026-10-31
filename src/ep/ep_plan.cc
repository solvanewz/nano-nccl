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

std::vector<std::int32_t> send_expert_order(const EpShape& shape, const planner::GroupPlan& groups) {
    validate_shape(shape);
    planner::validate_group_plan(groups, shape.num_experts);
    std::vector<std::int32_t> order;
    order.reserve(static_cast<std::size_t>(shape.num_experts));
    for (int g : groups.order) {
        for (int e = 0; e < shape.num_experts; ++e) {
            if (groups.expert_group[e] == g) order.push_back(e);
        }
    }
    return order;
}

DispatchPlan make_dispatch_plan(const EpShape& shape, const planner::GroupPlan& groups,
                                const std::vector<std::int64_t>& send_counts,
                                const std::vector<std::int64_t>& recv_counts) {
    const int W = shape.world;
    const int E = shape.num_experts;
    const int epr = shape.experts_per_rank();
    const auto rb = static_cast<std::int64_t>(shape.row_bytes());

    DispatchPlan p;
    p.expert_order = send_expert_order(shape, groups);  // validates shape and plan
    if (static_cast<int>(send_counts.size()) != E || static_cast<int>(recv_counts.size()) != W * epr) {
        throw std::invalid_argument("make_dispatch_plan: count vectors have the wrong size");
    }
    for (std::int64_t c : send_counts) {
        if (c < 0) throw std::invalid_argument("make_dispatch_plan: negative send count");
    }
    for (std::int64_t c : recv_counts) {
        if (c < 0) throw std::invalid_argument("make_dispatch_plan: negative receive count");
    }
    p.send_counts = send_counts;
    p.recv_counts = recv_counts;
    p.send_offsets.assign(static_cast<std::size_t>(E), 0);

    auto bytes = [rb](std::int64_t rows) { return static_cast<std::size_t>(rows * rb); };
    std::int64_t send_cursor = 0, recv_cursor = 0, expert_cursor = 0;
    std::vector<std::int64_t> recv_row(static_cast<std::size_t>(W) * epr, 0);  // [s * epr + j]

    for (int g : groups.order) {
        GroupStep st;
        st.group = g;
        auto& d = st.dispatch;
        d.send_bytes.assign(W, 0);
        d.send_offsets.assign(W, 0);
        d.recv_bytes.assign(W, 0);
        d.recv_offsets.assign(W, 0);

        // Send side: experts of g in id order, hence grouped by owner.
        st.send_begin = send_cursor;
        for (int peer = 0; peer < W; ++peer) d.send_offsets[peer] = bytes(send_cursor);
        int last_peer = -1;
        for (int e = 0; e < E; ++e) {
            if (groups.expert_group[e] != g) continue;
            const int peer = shape.owner(e);
            if (peer != last_peer) {
                // Peers without experts in g keep a zero-byte message at the
                // current position.
                for (int q = last_peer + 1; q <= peer; ++q) d.send_offsets[q] = bytes(send_cursor);
                last_peer = peer;
            }
            p.send_offsets[e] = send_cursor;
            d.send_bytes[peer] += bytes(send_counts[e]);
            send_cursor += send_counts[e];
        }
        for (int q = last_peer + 1; q < W; ++q) d.send_offsets[q] = bytes(send_cursor);
        st.send_rows = send_cursor - st.send_begin;

        // Receive side: this rank's experts of g, by source then expert.
        std::vector<int> local;
        for (int j = 0; j < epr; ++j) {
            if (groups.expert_group[shape.first_local_expert() + j] == g) local.push_back(j);
        }
        st.recv_begin = recv_cursor;
        for (int s = 0; s < W; ++s) {
            d.recv_offsets[s] = bytes(recv_cursor);
            for (int j : local) {
                recv_row[s * epr + j] = recv_cursor;
                d.recv_bytes[s] += bytes(recv_counts[s * epr + j]);
                recv_cursor += recv_counts[s * epr + j];
            }
        }
        st.recv_rows = recv_cursor - st.recv_begin;

        st.expert_begin = expert_cursor;
        for (int j : local) {
            ExpertSegment seg;
            seg.expert = j;
            seg.begin = expert_cursor;
            for (int s = 0; s < W; ++s) {
                const std::int64_t rows = recv_counts[s * epr + j];
                if (rows != 0) {
                    st.unpack.push_back({recv_row[s * epr + j], expert_cursor, rows});
                    st.repack.push_back({expert_cursor, recv_row[s * epr + j], rows});
                }
                expert_cursor += rows;
            }
            seg.rows = expert_cursor - seg.begin;
            st.experts.push_back(seg);
        }
        st.expert_rows = expert_cursor - st.expert_begin;

        st.combine = {d.recv_bytes, d.recv_offsets, d.send_bytes, d.send_offsets};
        p.steps.push_back(std::move(st));
    }
    p.send_rows = send_cursor;
    p.recv_rows = recv_cursor;
    return p;
}

}  // namespace nano_nccl::ep
