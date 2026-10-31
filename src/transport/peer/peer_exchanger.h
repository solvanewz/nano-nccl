#pragma once

#include "transport/peer/peer_types.h"
#include "transport/peer/peer_wire.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <vector>

namespace nano_nccl::transport::peer {

using ExchangeId = std::uint64_t;

// Host-side protocol engine of one rank: splits messages into slot-sized
// chunks, posts them as credits allow, matches arrivals to exchanges, copies
// them out and returns the slots.
//
// Ordering contract: every rank must issue its exchanges in the same order
// (as with collectives). Chunks on one link are matched to exchanges in issue
// order, and a mismatch in epoch, tag or size is reported as an error.
//
// Not thread-safe: one thread drives progress() and the other calls.
class PeerExchanger {
public:
    explicit PeerExchanger(PeerWire* wire);
    PeerExchanger(const PeerExchanger&) = delete;
    PeerExchanger& operator=(const PeerExchanger&) = delete;

    // `send_buffer` must stay unchanged until kSendBufferReusable and
    // `recv_buffer` must not be read before kConsumerReadable.
    ExchangeId exchangev_async(std::uint64_t epoch, std::uint32_t tag,
                               const ExchangeLayout& layout,
                               const std::byte* send_buffer, std::byte* recv_buffer);

    // Non-blocking. Throws std::runtime_error when an arriving chunk does not
    // match the exchange expected on its link.
    void progress();

    bool query_completion(ExchangeId id, CompletionEvent event) const;
    // `peer` is the destination for send-side events and the source for
    // receive-side events.
    bool query_peer_completion(ExchangeId id, int peer, CompletionEvent event) const;

    // Drops the bookkeeping of an exchange that reached both
    // kSendBufferReusable and kReceiveSlotsReleased.
    void retire(ExchangeId id);

    int rank() const { return rank_; }
    int nranks() const { return nranks_; }
    int slots_per_link() const { return slots_; }
    std::size_t slot_bytes() const { return slot_bytes_; }
    std::size_t active_exchanges() const { return exchanges_.size(); }

private:
    struct Exchange {
        std::uint64_t epoch = 0;
        std::uint32_t tag = 0;
        ExchangeLayout layout;
        const std::byte* send = nullptr;
        std::byte* recv = nullptr;
        // Per peer, in chunks.
        std::vector<std::uint32_t> tx_total, tx_posted, tx_done;
        std::vector<std::uint32_t> rx_total, rx_arrived, rx_copied, rx_released;
    };
    struct PendingWrite {
        ExchangeId id;
        int dst;
        PeerWire::OpId op;
    };
    struct PendingCopy {
        std::uint64_t seq;
        ExchangeId id;
        PeerWire::OpId op;
        bool done;
    };
    struct TxLink {
        std::uint64_t next_seq = 0;
        std::deque<ExchangeId> queue;  // exchanges with chunks left to post
    };
    struct RxLink {
        std::uint64_t next_seq = 0;    // next arrived chunk to match
        std::uint64_t released = 0;
        std::deque<ExchangeId> queue;  // exchanges still expecting chunks
        std::deque<PendingCopy> copies;  // in seq order; slots are freed in order
    };

    bool post_one(int dst);
    void accept_arrivals(int src);
    void finish_copies(int src);
    void check_header(const Exchange& exchange, int src, std::uint64_t seq,
                      const ChunkHeader& header) const;
    Exchange& get(ExchangeId id);
    const Exchange& get(ExchangeId id) const;
    static bool peer_reached(const Exchange& exchange, int peer, CompletionEvent event);

    PeerWire* wire_;
    int rank_;
    int nranks_;
    int slots_;
    std::size_t slot_bytes_;
    ExchangeId next_id_ = 1;
    std::map<ExchangeId, Exchange> exchanges_;
    std::vector<TxLink> tx_;
    std::vector<RxLink> rx_;
    std::vector<PendingWrite> pending_writes_;
};

// exchange_counts(): every rank sends `counts_per_peer` 64-bit values to every
// peer, carried by the same exchange machinery so its cost shows up like any
// other traffic. recv_counts[p * counts_per_peer + i] receives peer p's values.
ExchangeId exchange_counts_async(PeerExchanger& exchanger, std::uint64_t epoch,
                                 const std::uint64_t* send_counts,
                                 std::uint64_t* recv_counts,
                                 std::size_t counts_per_peer);

}  // namespace nano_nccl::transport::peer
