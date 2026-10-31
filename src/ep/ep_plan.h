#pragma once

#include "planner/group_plan.h"
#include "transport/peer/peer_types.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// Host-side shapes and row layouts of one EP MoE layer (B1/B2).
//
// Experts are placed contiguously: rank r hosts global experts
// [r * E/W, (r + 1) * E/W). A routed token copy ("entry") is one (token, k)
// pair and travels as one row of `hidden` bf16 values.
//
// A GroupPlan splits the experts into groups that run as steps, in the plan's
// order; each step has its own dispatch and combine exchange. Row orders:
//   send buffer      by step, then global expert (so grouped by destination
//                    rank inside a step); combine results land here too
//   receive buffer   by step, then source rank, then local expert
//   expert-major     by step, then local expert, then source rank (FFN input)
// Combine sends the expert outputs back in receive-buffer order, so the
// source receives them in its own send-buffer order. With a single group this
// is the plain serial layer.
namespace nano_nccl::ep {

struct EpShape {
    int world = 0;
    int rank = 0;
    int num_experts = 0;  // global
    int top_k = 0;
    int hidden = 0;       // bf16 values per row; a multiple of 8 (16-byte rows)
    int ffn = 0;          // expert intermediate size

    int experts_per_rank() const { return num_experts / world; }
    int owner(int expert) const { return expert / experts_per_rank(); }
    int first_local_expert() const { return rank * experts_per_rank(); }
    std::size_t row_bytes() const { return static_cast<std::size_t>(hidden) * 2; }
};

// Throws std::invalid_argument for shapes the data plane does not support.
void validate_shape(const EpShape& shape);

// Copies `rows` rows from row `src_row` of one buffer to row `dst_row` of another.
struct RowSegment {
    std::int64_t src_row = 0;
    std::int64_t dst_row = 0;
    std::int64_t rows = 0;
};

// Rows [begin, begin + rows) of the expert-major buffer belong to local expert
// `expert`. Also copied to the device.
struct ExpertSegment {
    std::int32_t expert = 0;
    std::int32_t reserved = 0;
    std::int64_t begin = 0;
    std::int64_t rows = 0;
};

struct GroupStep {
    int group = 0;
    std::int64_t send_begin = 0, send_rows = 0;      // send buffer rows
    std::int64_t recv_begin = 0, recv_rows = 0;      // receive buffer rows
    std::int64_t expert_begin = 0, expert_rows = 0;  // expert-major rows
    transport::peer::ExchangeLayout dispatch;        // bytes, whole buffers
    transport::peer::ExchangeLayout combine;         // dispatch reversed
    std::vector<RowSegment> unpack;                  // receive buffer -> expert-major
    std::vector<RowSegment> repack;                  // expert-major -> receive-buffer order
    std::vector<ExpertSegment> experts;              // this rank's experts of the group
};

struct DispatchPlan {
    // This rank as a source: send_counts[e] = N(rank, e).
    std::vector<std::int64_t> send_counts;   // [E]
    // This rank as an expert host: index s * epr + j is N(s, first_local + j).
    std::vector<std::int64_t> recv_counts;   // [W * epr]
    std::vector<std::int32_t> expert_order;  // [E] send-buffer order of experts
    std::vector<std::int64_t> send_offsets;  // [E] first send row of expert e
    std::int64_t send_rows = 0;
    std::int64_t recv_rows = 0;
    std::vector<GroupStep> steps;            // in execution order
};

// Order of experts in the send buffer: by step, then expert id. Needed before
// the counts are known (placement runs first).
std::vector<std::int32_t> send_expert_order(const EpShape& shape, const planner::GroupPlan& groups);

// `send_counts` has E entries; `recv_counts` has W * epr entries as above.
DispatchPlan make_dispatch_plan(const EpShape& shape, const planner::GroupPlan& groups,
                                const std::vector<std::int64_t>& send_counts,
                                const std::vector<std::int64_t>& recv_counts);

}  // namespace nano_nccl::ep
