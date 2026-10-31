#include "transport/peer/peer_exchanger.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>

namespace nano_nccl::transport::peer {

namespace {

std::string describe(const ChunkHeader& h) {
    std::ostringstream out;
    out << "{epoch=" << h.epoch << " tag=" << h.tag << " src=" << h.src << " dst=" << h.dst
        << " chunk=" << h.chunk_index << "/" << h.chunk_count << " offset=" << h.offset
        << " bytes=" << h.bytes << " total=" << h.total_bytes << "}";
    return out.str();
}

}  // namespace

PeerExchanger::PeerExchanger(PeerWire* wire) : wire_(wire) {
    if (wire_ == nullptr) throw std::invalid_argument("PeerExchanger needs a wire");
    rank_ = wire_->rank();
    nranks_ = wire_->nranks();
    slots_ = wire_->slots_per_link();
    slot_bytes_ = wire_->slot_bytes();
    if (nranks_ <= 0 || rank_ < 0 || rank_ >= nranks_ || slots_ <= 0 || slot_bytes_ == 0) {
        throw std::invalid_argument("PeerExchanger: invalid wire geometry");
    }
    tx_.resize(static_cast<std::size_t>(nranks_));
    rx_.resize(static_cast<std::size_t>(nranks_));
}

ExchangeId PeerExchanger::exchangev_async(std::uint64_t epoch, std::uint32_t tag,
                                          const ExchangeLayout& layout,
                                          const std::byte* send_buffer,
                                          std::byte* recv_buffer) {
    validate_layout(layout, nranks_);
    const auto n = static_cast<std::size_t>(nranks_);
    Exchange exchange;
    exchange.epoch = epoch;
    exchange.tag = tag;
    exchange.layout = layout;
    exchange.send = send_buffer;
    exchange.recv = recv_buffer;
    exchange.tx_total.resize(n);
    exchange.rx_total.resize(n);
    for (std::size_t peer = 0; peer < n; ++peer) {
        if (layout.send_bytes[peer] != 0 && send_buffer == nullptr) {
            throw std::invalid_argument("exchangev_async: null send buffer with data to send");
        }
        if (layout.recv_bytes[peer] != 0 && recv_buffer == nullptr) {
            throw std::invalid_argument("exchangev_async: null receive buffer with data to receive");
        }
        exchange.tx_total[peer] = chunk_count(layout.send_bytes[peer], slot_bytes_);
        exchange.rx_total[peer] = chunk_count(layout.recv_bytes[peer], slot_bytes_);
    }
    exchange.tx_posted.assign(n, 0);
    exchange.tx_done.assign(n, 0);
    exchange.rx_arrived.assign(n, 0);
    exchange.rx_copied.assign(n, 0);
    exchange.rx_released.assign(n, 0);

    const ExchangeId id = next_id_++;
    exchanges_.emplace(id, std::move(exchange));
    for (std::size_t peer = 0; peer < n; ++peer) {
        tx_[peer].queue.push_back(id);
        rx_[peer].queue.push_back(id);
    }
    return id;
}

void PeerExchanger::progress() {
    // Post round-robin, one chunk per link per pass, so a long message on one
    // link does not monopolize the wire.
    bool posted = true;
    while (posted) {
        posted = false;
        for (int i = 0; i < nranks_; ++i) posted |= post_one((rank_ + i) % nranks_);
    }

    for (std::size_t i = 0; i < pending_writes_.size();) {
        if (wire_->done(pending_writes_[i].op)) {
            ++get(pending_writes_[i].id).tx_done[static_cast<std::size_t>(pending_writes_[i].dst)];
            pending_writes_[i] = pending_writes_.back();
            pending_writes_.pop_back();
        } else {
            ++i;
        }
    }

    for (int src = 0; src < nranks_; ++src) {
        accept_arrivals(src);
        finish_copies(src);
    }
}

// Link queues hold only exchanges with work left on that link: an entry is
// popped as soon as its last chunk is posted or matched, so a retired
// exchange is never referenced again.
bool PeerExchanger::post_one(int dst) {
    TxLink& link = tx_[static_cast<std::size_t>(dst)];
    if (link.queue.empty()) return false;

    const std::uint64_t released = wire_->released(dst);
    if (released > link.next_seq) {
        throw std::logic_error("peer wire released more chunks than were posted on link " +
                               std::to_string(rank_) + "->" + std::to_string(dst));
    }
    if (link.next_seq - released >= static_cast<std::uint64_t>(slots_)) return false;

    const ExchangeId id = link.queue.front();
    Exchange& exchange = get(id);
    const auto d = static_cast<std::size_t>(dst);
    const std::uint32_t index = exchange.tx_posted[d];
    const std::size_t total = exchange.layout.send_bytes[d];
    const std::size_t offset = static_cast<std::size_t>(index) * slot_bytes_;
    const std::size_t bytes = total == 0 ? 0 : std::min(slot_bytes_, total - offset);

    ChunkHeader header;
    header.epoch = exchange.epoch;
    header.tag = exchange.tag;
    header.src = static_cast<std::uint32_t>(rank_);
    header.dst = static_cast<std::uint32_t>(dst);
    header.chunk_index = index;
    header.chunk_count = exchange.tx_total[d];
    header.offset = offset;
    header.bytes = bytes;
    header.total_bytes = total;
    const std::byte* src =
        bytes == 0 ? nullptr : exchange.send + exchange.layout.send_offsets[d] + offset;

    const PeerWire::OpId op = wire_->write_chunk(dst, link.next_seq, header, src);
    if (op == PeerWire::kNoOp) throw std::logic_error("peer wire returned kNoOp for a write");
    pending_writes_.push_back({id, dst, op});
    ++link.next_seq;
    if (++exchange.tx_posted[d] == exchange.tx_total[d]) link.queue.pop_front();
    return true;
}

void PeerExchanger::check_header(const Exchange& exchange, int src, std::uint64_t seq,
                                 const ChunkHeader& header) const {
    const auto s = static_cast<std::size_t>(src);
    ChunkHeader expected;
    expected.epoch = exchange.epoch;
    expected.tag = exchange.tag;
    expected.src = static_cast<std::uint32_t>(src);
    expected.dst = static_cast<std::uint32_t>(rank_);
    expected.chunk_index = exchange.rx_arrived[s];
    expected.chunk_count = exchange.rx_total[s];
    expected.total_bytes = exchange.layout.recv_bytes[s];
    expected.offset = static_cast<std::size_t>(expected.chunk_index) * slot_bytes_;
    expected.bytes = expected.total_bytes == 0
                         ? 0
                         : std::min<std::uint64_t>(slot_bytes_, expected.total_bytes - expected.offset);
    if (header.epoch != expected.epoch || header.tag != expected.tag ||
        header.src != expected.src || header.dst != expected.dst ||
        header.chunk_index != expected.chunk_index ||
        header.chunk_count != expected.chunk_count ||
        header.total_bytes != expected.total_bytes || header.offset != expected.offset ||
        header.bytes != expected.bytes) {
        throw std::runtime_error("peer exchange: unexpected chunk on link " +
                                 std::to_string(src) + "->" + std::to_string(rank_) +
                                 " seq " + std::to_string(seq) + ": expected " +
                                 describe(expected) + " got " + describe(header));
    }
}

void PeerExchanger::accept_arrivals(int src) {
    RxLink& link = rx_[static_cast<std::size_t>(src)];
    const auto s = static_cast<std::size_t>(src);
    const std::uint64_t arrived = wire_->arrived(src);
    if (arrived < link.next_seq) {
        throw std::logic_error("peer wire arrival counter went backwards on link " +
                               std::to_string(src) + "->" + std::to_string(rank_));
    }
    while (link.next_seq < arrived) {
        // A chunk for an exchange this rank has not issued yet stays in its
        // slot; the sender is held back by credits until it is issued.
        if (link.queue.empty()) break;

        const ExchangeId id = link.queue.front();
        Exchange& exchange = get(id);
        const ChunkHeader& header = wire_->slot_header(src, link.next_seq);
        check_header(exchange, src, link.next_seq, header);
        PeerWire::OpId op = PeerWire::kNoOp;
        if (header.bytes != 0) {
            std::byte* dst = exchange.recv + exchange.layout.recv_offsets[s] + header.offset;
            op = wire_->copy_out(src, link.next_seq, dst, header.bytes);
            if (op == PeerWire::kNoOp) throw std::logic_error("peer wire returned kNoOp for a copy");
        }
        link.copies.push_back({link.next_seq, id, op, false});
        ++link.next_seq;
        if (++exchange.rx_arrived[s] == exchange.rx_total[s]) link.queue.pop_front();
    }
}

void PeerExchanger::finish_copies(int src) {
    RxLink& link = rx_[static_cast<std::size_t>(src)];
    const auto s = static_cast<std::size_t>(src);
    for (PendingCopy& copy : link.copies) {
        if (!copy.done && (copy.op == PeerWire::kNoOp || wire_->done(copy.op))) {
            copy.done = true;
            ++get(copy.id).rx_copied[s];
        }
    }
    // The wire's credit is a counter, so slots are returned in chunk order
    // even when copies finish out of order.
    const std::uint64_t before = link.released;
    while (!link.copies.empty() && link.copies.front().done) {
        ++get(link.copies.front().id).rx_released[s];
        link.released = link.copies.front().seq + 1;
        link.copies.pop_front();
    }
    if (link.released != before) wire_->release(src, link.released);
}

bool PeerExchanger::peer_reached(const Exchange& e, int peer, CompletionEvent event) {
    const auto p = static_cast<std::size_t>(peer);
    switch (event) {
        case CompletionEvent::kSubmitted: return e.tx_posted[p] == e.tx_total[p];
        case CompletionEvent::kSendBufferReusable: return e.tx_done[p] == e.tx_total[p];
        case CompletionEvent::kPayloadArrived: return e.rx_arrived[p] == e.rx_total[p];
        case CompletionEvent::kConsumerReadable: return e.rx_copied[p] == e.rx_total[p];
        case CompletionEvent::kReceiveSlotsReleased: return e.rx_released[p] == e.rx_total[p];
    }
    throw std::invalid_argument("unknown completion event");
}

bool PeerExchanger::query_completion(ExchangeId id, CompletionEvent event) const {
    const Exchange& exchange = get(id);
    for (int peer = 0; peer < nranks_; ++peer) {
        if (!peer_reached(exchange, peer, event)) return false;
    }
    return true;
}

bool PeerExchanger::query_peer_completion(ExchangeId id, int peer, CompletionEvent event) const {
    if (peer < 0 || peer >= nranks_) throw std::invalid_argument("peer out of range");
    return peer_reached(get(id), peer, event);
}

void PeerExchanger::retire(ExchangeId id) {
    if (!query_completion(id, CompletionEvent::kSendBufferReusable) ||
        !query_completion(id, CompletionEvent::kReceiveSlotsReleased)) {
        throw std::logic_error("retire: exchange " + std::to_string(id) + " is still in flight");
    }
    exchanges_.erase(id);
}

PeerExchanger::Exchange& PeerExchanger::get(ExchangeId id) {
    auto it = exchanges_.find(id);
    if (it == exchanges_.end()) {
        throw std::invalid_argument("unknown or retired exchange " + std::to_string(id));
    }
    return it->second;
}

const PeerExchanger::Exchange& PeerExchanger::get(ExchangeId id) const {
    auto it = exchanges_.find(id);
    if (it == exchanges_.end()) {
        throw std::invalid_argument("unknown or retired exchange " + std::to_string(id));
    }
    return it->second;
}

ExchangeId exchange_counts_async(PeerExchanger& exchanger, std::uint64_t epoch,
                                 const std::uint64_t* send_counts,
                                 std::uint64_t* recv_counts,
                                 std::size_t counts_per_peer) {
    const ExchangeLayout layout =
        uniform_layout(exchanger.nranks(), counts_per_peer * sizeof(std::uint64_t));
    return exchanger.exchangev_async(epoch, kCountsTag, layout,
                                     reinterpret_cast<const std::byte*>(send_counts),
                                     reinterpret_cast<std::byte*>(recv_counts));
}

}  // namespace nano_nccl::transport::peer
