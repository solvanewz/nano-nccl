#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// Arbitrary-peer variable-length exchange for the EP path (B1).
//
// Unlike the Ring path, every ordered rank pair (including a rank with itself)
// is a link, the rank count is a runtime value, and per-peer message sizes may
// differ or be zero. See docs/design/b1-peer-exchange.md in the workspace.
namespace nano_nccl::transport::peer {

// The five completion conditions an exchange exposes. They are distinct
// events; a backend may reach them at different times, and callers must wait
// for the one that matches what they are about to do.
enum class CompletionEvent : int {
    kSubmitted = 0,           // every outgoing chunk handed to the wire
    kSendBufferReusable,      // the wire finished reading the send buffer
    kPayloadArrived,          // every incoming chunk landed in a receive slot
    kConsumerReadable,        // incoming data copied into the receive buffer and visible
    kReceiveSlotsReleased,    // every receive slot used by the exchange returned as credit
};

const char* completion_event_name(CompletionEvent event);

// Exchanges whose tag has this bit set are reserved for the library.
constexpr std::uint32_t kReservedTagBit = 0x80000000u;
constexpr std::uint32_t kCountsTag = kReservedTagBit | 1u;

// Travels with every chunk; the receiver checks each field against the
// exchange it expects next on that link.
struct ChunkHeader {
    std::uint64_t epoch = 0;
    std::uint32_t tag = 0;
    std::uint32_t src = 0;
    std::uint32_t dst = 0;
    std::uint32_t chunk_index = 0;
    std::uint32_t chunk_count = 0;
    std::uint32_t reserved = 0;
    std::uint64_t offset = 0;       // byte offset of this chunk inside the message
    std::uint64_t bytes = 0;        // payload bytes in this chunk
    std::uint64_t total_bytes = 0;  // whole message
};
static_assert(sizeof(ChunkHeader) == 56);

// Per-peer byte counts and offsets into one send and one receive buffer,
// indexed by peer rank.
struct ExchangeLayout {
    std::vector<std::size_t> send_bytes;
    std::vector<std::size_t> send_offsets;
    std::vector<std::size_t> recv_bytes;
    std::vector<std::size_t> recv_offsets;
};

// Contiguous layout with the same byte count for every peer.
ExchangeLayout uniform_layout(int nranks, std::size_t bytes_per_peer);

// Throws std::invalid_argument on size mismatch, offset overflow, or
// overlapping receive regions.
void validate_layout(const ExchangeLayout& layout, int nranks);

// Chunks needed for one message. A zero-byte message still sends one
// header-only chunk so the receiver learns the size explicitly.
std::uint32_t chunk_count(std::size_t bytes, std::size_t slot_bytes);

}  // namespace nano_nccl::transport::peer
