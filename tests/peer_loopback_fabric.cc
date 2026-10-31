#include "peer_loopback_fabric.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace nano_nccl::test {

using transport::peer::ChunkHeader;
using transport::peer::PeerWire;

namespace {
constexpr std::uint64_t kEmptySlot = std::numeric_limits<std::uint64_t>::max();
}

class LoopbackFabric::Endpoint final : public PeerWire {
public:
    Endpoint(LoopbackFabric* fabric, int rank) : fabric_(fabric), rank_(rank) {}

    int rank() const override { return rank_; }
    int nranks() const override { return fabric_->config_.nranks; }
    int slots_per_link() const override { return fabric_->config_.slots; }
    std::size_t slot_bytes() const override { return fabric_->config_.slot_bytes; }

    std::uint64_t released(int dst) override {
        return fabric_->link(rank_, dst).released_visible;
    }
    OpId write_chunk(int dst, std::uint64_t seq, const ChunkHeader& header,
                     const std::byte* src) override {
        return fabric_->write_chunk(rank_, dst, seq, header, src);
    }
    std::uint64_t arrived(int src) override { return fabric_->link(src, rank_).arrived; }
    const ChunkHeader& slot_header(int src, std::uint64_t seq) override {
        return fabric_->slot_header(src, rank_, seq);
    }
    OpId copy_out(int src, std::uint64_t seq, std::byte* dst, std::size_t bytes) override {
        return fabric_->copy_out(src, rank_, seq, dst, bytes);
    }
    void release(int src, std::uint64_t upto) override { fabric_->release(src, rank_, upto); }
    bool done(OpId op) override { return fabric_->done_ops_.count(op) != 0; }

private:
    LoopbackFabric* fabric_;
    int rank_;
};

LoopbackFabric::LoopbackFabric(const LoopbackConfig& config) : config_(config), rng_(config.seed) {
    if (config_.nranks <= 0 || config_.slots <= 0 || config_.slot_bytes == 0 ||
        config_.max_delay < 0) {
        throw std::invalid_argument("LoopbackFabric: invalid config");
    }
    const auto n = static_cast<std::size_t>(config_.nranks);
    const auto slots = static_cast<std::size_t>(config_.slots);
    links_.resize(n * n);
    for (Link& l : links_) {
        l.payload.resize(slots * config_.slot_bytes);
        l.headers.resize(slots);
        l.slot_seq.assign(slots, kEmptySlot);
    }
    for (int rank = 0; rank < config_.nranks; ++rank) {
        endpoints_.push_back(std::make_unique<Endpoint>(this, rank));
    }
}

LoopbackFabric::~LoopbackFabric() = default;

PeerWire& LoopbackFabric::endpoint(int rank) {
    return *endpoints_.at(static_cast<std::size_t>(rank));
}

LoopbackFabric::Link& LoopbackFabric::link(int src, int dst) {
    if (src < 0 || dst < 0 || src >= config_.nranks || dst >= config_.nranks) {
        throw std::out_of_range("LoopbackFabric: rank out of range");
    }
    return links_[static_cast<std::size_t>(src) * static_cast<std::size_t>(config_.nranks) +
                  static_cast<std::size_t>(dst)];
}

const LoopbackFabric::Link& LoopbackFabric::link(int src, int dst) const {
    return const_cast<LoopbackFabric*>(this)->link(src, dst);
}

void LoopbackFabric::schedule(std::function<void()> action) {
    std::uniform_int_distribution<int> delay(0, config_.max_delay);
    events_.push_back({now_ + 1 + static_cast<std::uint64_t>(delay(rng_)), std::move(action)});
}

void LoopbackFabric::step() {
    ++now_;
    std::vector<Event> due;
    for (std::size_t i = 0; i < events_.size();) {
        if (events_[i].due <= now_) {
            due.push_back(std::move(events_[i]));
            events_[i] = std::move(events_.back());
            events_.pop_back();
        } else {
            ++i;
        }
    }
    std::shuffle(due.begin(), due.end(), rng_);
    for (Event& event : due) event.action();
}

std::uint64_t LoopbackFabric::write_chunk(int src, int dst, std::uint64_t seq,
                                          const ChunkHeader& header, const std::byte* data) {
    Link& l = link(src, dst);
    const auto slots = static_cast<std::uint64_t>(config_.slots);
    if (seq != l.written) {
        throw std::logic_error("loopback: link " + std::to_string(src) + "->" +
                               std::to_string(dst) + " wrote seq " + std::to_string(seq) +
                               ", expected " + std::to_string(l.written));
    }
    if (seq >= l.released + slots) {
        throw std::logic_error("loopback: link " + std::to_string(src) + "->" +
                               std::to_string(dst) + " overwrote unreleased slot (seq " +
                               std::to_string(seq) + ", released " +
                               std::to_string(l.released) + ")");
    }
    if (header.bytes > config_.slot_bytes || (header.bytes != 0 && data == nullptr)) {
        throw std::logic_error("loopback: bad chunk payload");
    }
    ++l.written;
    const std::uint64_t op = new_op();
    // Stage 1: the source buffer is read (snapshot taken now, not at post
    // time, so reusing the buffer early corrupts the data as it would on a
    // real NIC). Stage 2: the chunk lands in the slot.
    schedule([this, src, dst, seq, header, data, op] {
        auto snapshot = std::make_shared<std::vector<std::byte>>(data, data + header.bytes);
        done_ops_.insert(op);
        schedule([this, src, dst, seq, header, snapshot] {
            Link& lk = link(src, dst);
            const auto slot = static_cast<std::size_t>(seq % static_cast<std::uint64_t>(config_.slots));
            std::copy(snapshot->begin(), snapshot->end(),
                      lk.payload.begin() + static_cast<std::ptrdiff_t>(slot * config_.slot_bytes));
            ChunkHeader landed = header;
            if (lk.corrupt) {
                lk.corrupt(landed);
                lk.corrupt = nullptr;
            }
            lk.headers[slot] = landed;
            lk.slot_seq[slot] = seq;
            lk.landed.insert(seq);
            while (!lk.landed.empty() && *lk.landed.begin() == lk.arrived) {
                lk.landed.erase(lk.landed.begin());
                ++lk.arrived;
            }
            ++chunks_landed_;
        });
    });
    return op;
}

const ChunkHeader& LoopbackFabric::slot_header(int src, int dst, std::uint64_t seq) const {
    const Link& l = link(src, dst);
    const auto slot = static_cast<std::size_t>(seq % static_cast<std::uint64_t>(config_.slots));
    if (seq >= l.arrived || l.slot_seq[slot] != seq) {
        throw std::logic_error("loopback: header of seq " + std::to_string(seq) + " on link " +
                               std::to_string(src) + "->" + std::to_string(dst) +
                               " is not readable");
    }
    return l.headers[slot];
}

std::uint64_t LoopbackFabric::copy_out(int src, int dst, std::uint64_t seq, std::byte* out,
                                       std::size_t bytes) {
    const Link& l = link(src, dst);
    if (seq >= l.arrived || seq < l.released) {
        throw std::logic_error("loopback: copy_out of seq " + std::to_string(seq) +
                               " outside the arrived, unreleased window");
    }
    const std::uint64_t op = new_op();
    schedule([this, src, dst, seq, out, bytes, op] {
        Link& lk = link(src, dst);
        const auto slot = static_cast<std::size_t>(seq % static_cast<std::uint64_t>(config_.slots));
        if (lk.slot_seq[slot] != seq) {
            throw std::logic_error("loopback: slot of seq " + std::to_string(seq) + " on link " +
                                   std::to_string(src) + "->" + std::to_string(dst) +
                                   " was reused before copy-out finished");
        }
        std::memcpy(out, lk.payload.data() + slot * config_.slot_bytes, bytes);
        done_ops_.insert(op);
    });
    return op;
}

void LoopbackFabric::release(int src, int dst, std::uint64_t upto) {
    Link& l = link(src, dst);
    if (upto < l.released || upto > l.arrived) {
        throw std::logic_error("loopback: release to " + std::to_string(upto) + " on link " +
                               std::to_string(src) + "->" + std::to_string(dst) +
                               " outside [released " + std::to_string(l.released) +
                               ", arrived " + std::to_string(l.arrived) + "]");
    }
    l.released = upto;
    // The credit reaches the sender later.
    schedule([this, src, dst, upto] {
        Link& lk = link(src, dst);
        lk.released_visible = std::max(lk.released_visible, upto);
    });
}

void LoopbackFabric::corrupt_next_header(int src, int dst, std::function<void(ChunkHeader&)> mutate) {
    link(src, dst).corrupt = std::move(mutate);
}

bool LoopbackFabric::drained() const {
    for (const Link& l : links_) {
        if (l.written != l.arrived || l.arrived != l.released || l.released != l.released_visible) {
            return false;
        }
    }
    return idle();
}

std::string LoopbackFabric::describe() const {
    std::ostringstream out;
    out << "t=" << now_ << " pending_events=" << events_.size() << "\n";
    for (int src = 0; src < config_.nranks; ++src) {
        for (int dst = 0; dst < config_.nranks; ++dst) {
            const Link& l = link(src, dst);
            out << "  link " << src << "->" << dst << ": written=" << l.written
                << " arrived=" << l.arrived << " released=" << l.released
                << " released_visible=" << l.released_visible << "\n";
        }
    }
    return out.str();
}

}  // namespace nano_nccl::test
