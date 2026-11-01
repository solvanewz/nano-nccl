#include "sim_wire.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace nano_nccl::sim {

using transport::peer::ChunkHeader;
using transport::peer::PeerWire;

namespace {
constexpr double kMiB = 1024.0 * 1024.0;
}  // namespace

WireCosts WireCosts::from_profile(const CostProfile& p, bool contention, int rank) {
    WireCosts c;
    if (contention && p.tables.count(cost::kTxRemoteAlone) && p.tables.count(cost::kRxRemoteAlone)) {
        c.contention = true;
        c.tx_remote_alone = p.table_for(cost::kTxRemoteAlone, rank);
        c.rx_remote_alone = p.table_for(cost::kRxRemoteAlone, rank);
        c.tx_compute_factor = p.scalar_or(cost::kCopyTxComputeFactor, 1.0);
        c.rx_compute_factor = p.scalar_or(cost::kCopyRxComputeFactor, 1.0);
        if (p.tables.count(cost::kTxRemoteInsitu) && p.tables.count(cost::kRxRemoteInsitu)) {
            c.insitu = true;
            c.tx_insitu = p.table_for(cost::kTxRemoteInsitu, rank);
            c.rx_insitu = p.table_for(cost::kRxRemoteInsitu, rank);
        }
    }
    c.tx_remote = p.table_for(cost::kTxRemote, rank);
    c.rx_remote = p.table_for(cost::kRxRemote, rank);
    auto turnaround = [&](double (&turn)[2][2], const std::string& prefix, const char* sw) {
        const char* kinds[2] = {"r", "s"};
        bool insitu = true;
        for (const char* a : kinds) {
            for (const char* b : kinds) insitu = insitu && p.scalars.count(prefix + a + b);
        }
        for (int a = 0; a < 2; ++a) {
            for (int b = 0; b < 2; ++b) {
                turn[a][b] = insitu ? p.scalar_for(prefix + kinds[a] + kinds[b], rank, 0)
                                    : (a != b ? p.scalar_for(sw, rank, 0) : 0.0);
            }
        }
    };
    turnaround(c.tx_turn, cost::kTxTurnaround, cost::kTxSwitch);
    turnaround(c.rx_turn, cost::kRxTurnaround, cost::kRxSwitch);
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
        ++f_.tx_stats_[rank_].copies;
        LinkStats& ls = f_.links_[link(rank_, dst)];
        ++ls.chunks;
        ls.bytes += header.bytes;
        ls.max_in_flight = std::max<std::uint32_t>(ls.max_in_flight, static_cast<std::uint32_t>(seq + 1 - released(dst)));
        const OpId op = new_op();
        f_.enqueue(rank_, 0, {header.bytes, self, header.bytes != 0, clock, [this, op](double end) {
                                  ops_[op] = true;
                                  f_.host_.wake(rank_, end);
                              }});
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
        ++f_.rx_stats_[rank_].copies;
        const OpId op = new_op();
        f_.enqueue(rank_, 1, {bytes, self, true, clock, [this, op](double end) {
                                  ops_[op] = true;
                                  f_.host_.wake(rank_, end);
                              }});
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
    const WireCosts& costs() const { return f_.costs_[rank_]; }

    SimFabric& f_;
    int rank_;
    std::vector<bool> ops_;
    std::vector<std::deque<Pending>> tx_pending_;
    std::vector<std::uint64_t> tx_next_;
    std::vector<std::uint64_t> rx_released_;
};

SimFabric::SimFabric(Scheduler& queue, HostModel& host, int nranks, int slots_per_link, std::size_t slot_bytes,
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
    engines_.resize(static_cast<std::size_t>(nranks) * 2);
    for (int r = 0; r < nranks; ++r) wires_.push_back(std::make_unique<Wire>(*this, r));
}

SimFabric::~SimFabric() = default;

PeerWire& SimFabric::wire(int rank) { return *wires_.at(rank); }

int SimFabric::remote_busy(int rank) const {
    return (engine(rank, 0).remote ? 1 : 0) + (engine(rank, 1).remote ? 1 : 0);
}

// FIFO engine: a copy issued to an idle engine starts after the launch
// latency; a queued one after the previous copy ends plus the turnaround for
// the two kinds. Header-only chunks take no engine time.
void SimFabric::enqueue(int rank, int dir, Job job) {
    Engine& e = engine(rank, dir);
    e.queue.push_back(std::move(job));
    if (e.active) return;
    e.active = true;
    const double t = std::max(queue_.now(), e.queue.front().issue + costs_[rank].copy_latency);
    queue_.schedule_at(t, [this, rank, dir, t] { start_job(rank, dir, t); });
}

void SimFabric::start_job(int rank, int dir, double t) {
    Engine& e = engine(rank, dir);
    e.cur = std::move(e.queue.front());
    e.queue.pop_front();
    e.running = true;
    e.start = e.last = t;
    if (!e.cur.copies) {
        e.left = 0;
        e.rate = 1;
        finish_job(rank, dir, t);
        return;
    }
    e.last_self = e.cur.self ? 1 : 0;
    e.left = work(rank, dir, e.cur, t);
    e.remote = !e.cur.self;
    if (e.remote) {
        // Both engines of the rank change speed, and its GPU work.
        reschedule(rank, 0, t);
        reschedule(rank, 1, t);
        host_.remote_busy_changed(rank, t);
    } else {
        reschedule(rank, dir, t);
    }
}

void SimFabric::finish_job(int rank, int dir, double t) {
    Engine& e = engine(rank, dir);
    const bool was_remote = e.remote;
    e.running = false;
    e.remote = false;
    ++e.token;  // no end event of this job is pending any more
    (dir == 0 ? tx_stats_ : rx_stats_)[rank].busy += t - e.start;
    Job job = std::move(e.cur);
    e.cur = Job{};
    if (was_remote) {
        reschedule(rank, 1 - dir, t);
        host_.remote_busy_changed(rank, t);
    }
    if (e.queue.empty()) {
        e.active = false;
    } else {
        const Job& next = e.queue.front();
        const WireCosts& c = costs_[rank];
        double at;
        if (next.issue + c.copy_latency > t) {
            at = next.issue + c.copy_latency;  // issued after the engine went idle
        } else {
            at = t;
            if (next.copies && e.last_self >= 0) at += (dir == 0 ? c.tx_turn : c.rx_turn)[e.last_self][next.self ? 1 : 0];
        }
        queue_.schedule_at(at, [this, rank, dir, at] { start_job(rank, dir, at); });
    }
    job.done(t);
}

double SimFabric::rate(int rank, int dir) const {
    const Engine& e = engine(rank, dir);
    const WireCosts& c = costs_[rank];
    if (!e.remote || !c.insitu) return 1.0;
    if (!engine(rank, 1 - dir).remote) return 1.0;
    const Table& line = dir == 0 ? c.tx_insitu : c.rx_insitu;
    return line(0) / line(1);
}

// Brings the running job of the engine up to time t at its old rate, then
// schedules its end at the current rate.
void SimFabric::reschedule(int rank, int dir, double t) {
    Engine& e = engine(rank, dir);
    if (!e.running) return;
    e.left = std::max(0.0, e.left - (t - e.last) * e.rate);
    e.last = t;
    e.rate = rate(rank, dir);
    const double end = t + e.left / e.rate;
    const std::uint64_t token = ++e.token;
    queue_.schedule_at(end, [this, rank, dir, end, token] {
        if (engine(rank, dir).token == token) finish_job(rank, dir, end);
    });
}

// Work of a copy: its duration at rate 1, priced when it starts.
double SimFabric::work(int rank, int dir, const Job& job, double t) const {
    const WireCosts& c = costs_[rank];
    const bool tx = dir == 0;
    const double b = static_cast<double>(job.bytes);
    if (job.self) return c.copy_scale * (tx ? c.tx_self : c.rx_self)(b);
    const double busy = (tx ? c.tx_remote : c.rx_remote)(b);
    if (!c.contention) return c.copy_scale * busy;
    if (c.insitu) {
        const Table& shape = tx ? c.tx_remote : c.rx_remote;
        return c.copy_scale * (tx ? c.tx_insitu : c.rx_insitu)(0) * shape(b) / shape(kMiB);
    }
    int others = 0;
    for (int r = 0; r < nranks_; ++r) {
        for (int d = 0; d < 2; ++d) {
            if (!(r == rank && d == dir) && engine(r, d).remote) ++others;
        }
    }
    const double alone = (tx ? c.tx_remote_alone : c.rx_remote_alone)(b);
    double w = alone + (busy - alone) * std::min(1.0, others / 3.0);
    if (host_.gpu_busy(rank, t)) w *= tx ? c.tx_compute_factor : c.rx_compute_factor;
    return c.copy_scale * w;
}

ChunkHeader& SimFabric::slot(int owner, int src, std::uint64_t seq) {
    return headers_[static_cast<std::size_t>(owner) * nranks_ + src][seq % static_cast<std::uint64_t>(slots_)];
}

}  // namespace nano_nccl::sim
