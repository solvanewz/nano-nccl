#include "sim_wire.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace nano_nccl::sim {

using transport::peer::ChunkHeader;
using transport::peer::PeerWire;

WireCosts WireCosts::from_profile(const CostProfile& p, bool contention, int rank) {
    WireCosts c;
    if (contention && p.tables.count(cost::kTxRemoteAlone) && p.tables.count(cost::kRxRemoteAlone)) {
        c.contention = true;
        c.tx_remote_alone = p.table_for(cost::kTxRemoteAlone, rank);
        c.rx_remote_alone = p.table_for(cost::kRxRemoteAlone, rank);
        c.tx_compute_factor = p.scalar_or(cost::kCopyTxComputeFactor, 1.0);
        c.rx_compute_factor = p.scalar_or(cost::kCopyRxComputeFactor, 1.0);
    }
    c.tx_remote = p.table_for(cost::kTxRemote, rank);
    c.rx_remote = p.table_for(cost::kRxRemote, rank);
    c.tx_switch = p.scalar_for(cost::kTxSwitch, rank, 0);
    c.rx_switch = p.scalar_for(cost::kRxSwitch, rank, 0);
    c.tx_self = p.table(cost::kTxSelf);
    c.rx_self = p.table(cost::kRxSelf);
    c.host_post = p.scalar(cost::kHostPost);
    c.host_copy_out = p.scalar(cost::kHostCopyOut);
    c.copy_latency = p.scalar_or(cost::kCopyLatency, 0);
    c.credit_latency = p.scalar_or(cost::kCreditLatency, 0);
    c.arrival_latency = p.scalar_or(cost::kArrivalLatency, 0);
    return c;
}

class SimFabric::Wire final : public PeerWire {
public:
    Wire(SimFabric& f, int rank) : f_(f), rank_(rank), tx_pending_(f.nranks_), tx_next_(f.nranks_, 0),
                                   rx_released_(f.nranks_, 0) {
        ops_.push_back(true);  // kNoOp
    }

    int rank() const override { return rank_; }
    int nranks() const override { return f_.nranks_; }
    int slots_per_link() const override { return f_.slots_; }
    std::size_t slot_bytes() const override { return f_.slot_bytes_; }

    std::uint64_t released(int dst) override { return f_.released_[link(rank_, dst)]; }

    OpId write_chunk(int dst, std::uint64_t seq, const ChunkHeader& header, const std::byte*) override {
        check(dst);
        if (seq != tx_next_[dst]) throw std::logic_error("SimWire::write_chunk: out-of-order seq");
        if (seq - released(dst) >= static_cast<std::uint64_t>(f_.slots_)) {
            throw std::logic_error("SimWire::write_chunk: no free slot on link " + std::to_string(rank_) + "->" +
                                   std::to_string(dst));
        }
        if (header.bytes > f_.slot_bytes_) throw std::invalid_argument("SimWire::write_chunk: chunk exceeds slot");
        double& clock = f_.host_.host_clock(rank_);
        clock += costs().host_post;
        f_.slot(dst, rank_, seq) = header;
        const bool self = dst == rank_;
        const double start = engine_start(tx_free_, clock, tx_last_self_, self && header.bytes != 0, header.bytes != 0, true);
        const double dur = header.bytes == 0 ? 0.0 : copy_cost(true, self, header.bytes, start);
        const double end = start + dur;
        tx_free_ = end;
        if (!self && header.bytes != 0) f_.tx_remote_until_[rank_] = end;
        auto& st = f_.tx_stats_[rank_];
        st.busy += dur;
        ++st.copies;
        LinkStats& ls = f_.links_[link(rank_, dst)];
        ++ls.chunks;
        ls.bytes += header.bytes;
        ls.max_in_flight = std::max<std::uint32_t>(ls.max_in_flight, static_cast<std::uint32_t>(seq + 1 - released(dst)));
        const OpId op = new_op();
        f_.queue_.schedule_at(end, [this, op, end] {
            ops_[op] = true;
            f_.host_.wake(rank_, end);
        });
        tx_pending_[dst].push_back({seq, op});
        ++tx_next_[dst];
        f_.host_.note_activity(rank_);
        return op;
    }

    std::uint64_t arrived(int src) override { return f_.arrived_[link(src, rank_)]; }

    const ChunkHeader& slot_header(int src, std::uint64_t seq) override {
        check(src);
        if (seq < rx_released_[src] || seq >= arrived(src)) throw std::logic_error("SimWire::slot_header: not held");
        return f_.slot(rank_, src, seq);
    }

    OpId copy_out(int src, std::uint64_t seq, std::byte*, std::size_t bytes) override {
        check(src);
        if (seq < rx_released_[src] || seq >= arrived(src)) throw std::logic_error("SimWire::copy_out: not held");
        if (bytes == 0 || bytes > f_.slot_bytes_) throw std::invalid_argument("SimWire::copy_out: bad size");
        double& clock = f_.host_.host_clock(rank_);
        clock += costs().host_copy_out;
        const bool self = src == rank_;
        const double start = engine_start(rx_free_, clock, rx_last_self_, self, true, false);
        const double dur = copy_cost(false, self, bytes, start);
        const double end = start + dur;
        rx_free_ = end;
        if (!self) f_.rx_remote_until_[rank_] = end;
        auto& st = f_.rx_stats_[rank_];
        st.busy += dur;
        ++st.copies;
        const OpId op = new_op();
        f_.queue_.schedule_at(end, [this, op, end] {
            ops_[op] = true;
            f_.host_.wake(rank_, end);
        });
        f_.host_.note_activity(rank_);
        return op;
    }

    void release(int src, std::uint64_t upto) override {
        check(src);
        if (upto < rx_released_[src] || upto > arrived(src)) throw std::logic_error("SimWire::release: bad counter");
        if (upto == rx_released_[src]) return;
        rx_released_[src] = upto;
        const double at = f_.host_.host_clock(rank_) + costs().credit_latency;
        const std::size_t l = link(src, rank_);
        f_.queue_.schedule_at(at, [this, l, upto, src, at] {
            f_.released_[l] = std::max(f_.released_[l], upto);
            f_.host_.wake(src, at);
        });
        f_.host_.note_activity(rank_);
    }

    bool done(OpId op) override { return ops_.at(op); }

    // Publishes finished writes as arrivals, in order per link (ShmWire
    // publishes from its progress() once the copy's event completed).
    void progress() override {
        for (int dst = 0; dst < f_.nranks_; ++dst) {
            auto& pending = tx_pending_[dst];
            while (!pending.empty() && ops_[pending.front().op]) {
                const std::uint64_t upto = pending.front().seq + 1;
                pending.pop_front();
                const double at = f_.host_.host_clock(rank_) + costs().arrival_latency;
                const std::size_t l = link(rank_, dst);
                f_.queue_.schedule_at(at, [this, l, upto, dst, at] {
                    f_.arrived_[l] = std::max(f_.arrived_[l], upto);
                    f_.host_.wake(dst, at);
                });
                f_.host_.note_activity(rank_);
            }
        }
    }

private:
    struct Pending {
        std::uint64_t seq;
        OpId op;
    };

    std::size_t link(int src, int dst) const { return static_cast<std::size_t>(src) * f_.nranks_ + dst; }
    void check(int peer) const {
        if (peer < 0 || peer >= f_.nranks_) throw std::out_of_range("SimWire: bad peer " + std::to_string(peer));
    }
    OpId new_op() {
        ops_.push_back(false);
        return ops_.size() - 1;
    }
    // FIFO engine: a copy starts at its issue time (plus the launch latency
    // when the engine is idle) or when the previous copy ends.
    // A queued copy of the other kind than the engine's previous one waits
    // the switch stall; `copies` is false for header-only chunks.
    double engine_start(double free_at, double issue, int& last_self, bool self, bool copies, bool tx) const {
        if (free_at <= issue) {
            if (copies) last_self = self ? 1 : 0;
            return issue + costs().copy_latency;
        }
        double start = free_at;
        if (copies) {
            if (last_self >= 0 && last_self != (self ? 1 : 0)) start += tx ? costs().tx_switch : costs().rx_switch;
            last_self = self ? 1 : 0;
        }
        return start;
    }
    const WireCosts& costs() const { return f_.costs_[rank_]; }
    double copy_cost(bool tx, bool self, std::size_t bytes, double start) const {
        const WireCosts& c = costs();
        const double b = static_cast<double>(bytes);
        if (self) return c.copy_scale * (tx ? c.tx_self : c.rx_self)(b);
        const double busy = (tx ? c.tx_remote : c.rx_remote)(b);
        if (!c.contention) return c.copy_scale * busy;
        int others = 0;
        for (int r = 0; r < f_.nranks_; ++r) {
            others += (f_.tx_remote_until_[r] > start && !(tx && r == rank_)) ? 1 : 0;
            others += (f_.rx_remote_until_[r] > start && !(!tx && r == rank_)) ? 1 : 0;
        }
        const double alone = (tx ? c.tx_remote_alone : c.rx_remote_alone)(b);
        double t = alone + (busy - alone) * std::min(1.0, others / 3.0);
        if (f_.host_.gpu_busy(rank_, start)) t *= tx ? c.tx_compute_factor : c.rx_compute_factor;
        return c.copy_scale * t;
    }

    SimFabric& f_;
    int rank_;
    std::vector<bool> ops_;
    std::vector<std::deque<Pending>> tx_pending_;
    std::vector<std::uint64_t> tx_next_;
    std::vector<std::uint64_t> rx_released_;
    double tx_free_ = 0, rx_free_ = 0;
    int tx_last_self_ = -1, rx_last_self_ = -1;  // kind of the engine's last copy
};

SimFabric::SimFabric(EventQueue& queue, HostModel& host, int nranks, int slots_per_link, std::size_t slot_bytes,
                     std::vector<WireCosts> costs)
    : queue_(queue), host_(host), nranks_(nranks), slots_(slots_per_link), slot_bytes_(slot_bytes),
      costs_(std::move(costs)) {
    if (nranks < 1 || slots_per_link < 1 || slot_bytes == 0) throw std::invalid_argument("SimFabric: bad geometry");
    if (static_cast<int>(costs_.size()) != nranks) throw std::invalid_argument("SimFabric: one WireCosts per rank");
    const auto links = static_cast<std::size_t>(nranks) * nranks;
    headers_.assign(links, std::vector<ChunkHeader>(slots_per_link));
    arrived_.assign(links, 0);
    released_.assign(links, 0);
    links_.assign(links, {});
    tx_stats_.assign(nranks, {});
    rx_stats_.assign(nranks, {});
    tx_remote_until_.assign(nranks, 0);
    rx_remote_until_.assign(nranks, 0);
    for (int r = 0; r < nranks; ++r) wires_.push_back(std::make_unique<Wire>(*this, r));
}

SimFabric::~SimFabric() = default;

PeerWire& SimFabric::wire(int rank) { return *wires_.at(rank); }

int SimFabric::remote_engines_busy(int rank, double t) const {
    return (tx_remote_until_.at(rank) > t ? 1 : 0) + (rx_remote_until_.at(rank) > t ? 1 : 0);
}

ChunkHeader& SimFabric::slot(int owner, int src, std::uint64_t seq) {
    return headers_[static_cast<std::size_t>(owner) * nranks_ + src][seq % static_cast<std::uint64_t>(slots_)];
}

}  // namespace nano_nccl::sim
