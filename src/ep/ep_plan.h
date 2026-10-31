#pragma once

#include "transport/peer/peer_types.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// Host-side shapes and row layouts of one EP MoE layer (B1).
//
// Experts are placed contiguously: rank r hosts global experts
// [r * E/W, (r + 1) * E/W). A routed token copy ("entry") is one (token, k)
// pair and travels as one row of `hidden` bf16 values.
//
// Row orders used by the data plane:
//   send buffer      by global expert, so also grouped by destination rank
//   receive buffer   [source rank][local expert]   (one message per source)
//   expert-major     [local expert][source rank]   (what the expert FFN reads)
// Combine sends the expert outputs back in receive-buffer order, so the
// source receives them in its own send-buffer order.
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

struct DispatchPlan {
    // This rank as a source: send_counts[e] = N(rank, e) and its row offsets.
    std::vector<std::int64_t> send_counts;    // [E]
    std::vector<std::int64_t> send_offsets;   // [E + 1]
    // This rank as an expert host: index s * epr + j is N(s, first_local + j).
    std::vector<std::int64_t> recv_counts;    // [W * epr]
    std::vector<std::int64_t> recv_offsets;   // [W * epr], receive-buffer rows
    std::vector<std::int64_t> expert_offsets; // [epr + 1], expert-major rows
    std::int64_t send_rows = 0;
    std::int64_t recv_rows = 0;

    transport::peer::ExchangeLayout dispatch;  // bytes
    transport::peer::ExchangeLayout combine;   // dispatch reversed

    std::vector<RowSegment> unpack;  // receive buffer -> expert-major
    std::vector<RowSegment> repack;  // expert-major -> receive-buffer order
};

// `send_counts` has E entries; `recv_counts` has W * epr entries as above.
DispatchPlan make_dispatch_plan(const EpShape& shape, const std::vector<std::int64_t>& send_counts,
                                const std::vector<std::int64_t>& recv_counts);

}  // namespace nano_nccl::ep
