#include "transport/peer/peer_types.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace nano_nccl::transport::peer {

const char* completion_event_name(CompletionEvent event) {
    switch (event) {
        case CompletionEvent::kSubmitted: return "submitted";
        case CompletionEvent::kSendBufferReusable: return "send_buffer_reusable";
        case CompletionEvent::kPayloadArrived: return "payload_arrived";
        case CompletionEvent::kConsumerReadable: return "consumer_readable";
        case CompletionEvent::kReceiveSlotsReleased: return "receive_slots_released";
    }
    return "unknown";
}

ExchangeLayout uniform_layout(int nranks, std::size_t bytes_per_peer) {
    if (nranks <= 0) throw std::invalid_argument("nranks must be positive");
    ExchangeLayout layout;
    layout.send_bytes.assign(static_cast<std::size_t>(nranks), bytes_per_peer);
    layout.recv_bytes = layout.send_bytes;
    for (int peer = 0; peer < nranks; ++peer) {
        layout.send_offsets.push_back(static_cast<std::size_t>(peer) * bytes_per_peer);
    }
    layout.recv_offsets = layout.send_offsets;
    return layout;
}

void validate_layout(const ExchangeLayout& layout, int nranks) {
    const auto n = static_cast<std::size_t>(nranks);
    if (layout.send_bytes.size() != n || layout.send_offsets.size() != n ||
        layout.recv_bytes.size() != n || layout.recv_offsets.size() != n) {
        throw std::invalid_argument(
            "exchange layout vectors must have one entry per rank (" +
            std::to_string(nranks) + ")");
    }
    constexpr auto kMax = std::numeric_limits<std::size_t>::max();
    std::vector<std::pair<std::size_t, std::size_t>> recv_regions;
    for (std::size_t peer = 0; peer < n; ++peer) {
        if (layout.send_offsets[peer] > kMax - layout.send_bytes[peer] ||
            layout.recv_offsets[peer] > kMax - layout.recv_bytes[peer]) {
            throw std::invalid_argument("exchange layout offset overflows for peer " +
                                        std::to_string(peer));
        }
        if (layout.recv_bytes[peer] != 0) {
            recv_regions.emplace_back(layout.recv_offsets[peer],
                                      layout.recv_offsets[peer] + layout.recv_bytes[peer]);
        }
    }
    // Two peers writing the same receive bytes would race.
    std::sort(recv_regions.begin(), recv_regions.end());
    for (std::size_t i = 1; i < recv_regions.size(); ++i) {
        if (recv_regions[i].first < recv_regions[i - 1].second) {
            throw std::invalid_argument("exchange layout receive regions overlap");
        }
    }
}

std::uint32_t chunk_count(std::size_t bytes, std::size_t slot_bytes) {
    if (slot_bytes == 0) throw std::invalid_argument("slot_bytes must be positive");
    if (bytes == 0) return 1;
    const std::size_t chunks = (bytes - 1) / slot_bytes + 1;
    if (chunks > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("message needs more than 2^32 chunks");
    }
    return static_cast<std::uint32_t>(chunks);
}

}  // namespace nano_nccl::transport::peer
