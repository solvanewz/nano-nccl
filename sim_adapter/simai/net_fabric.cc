#include "net_fabric.h"

#include <algorithm>
#include <deque>
#include <functional>
#include <stdexcept>
#include <string>

namespace nano_nccl::sim::simai {

using transport::peer::ChunkHeader;
using transport::peer::PeerWire;

// Heap-allocated argument of an AstraNetworkAPI callback.
struct NetFabric::Callback {
    NetFabric* fabric;
    std::function<void()> fn;
    bool send;  // a send completion (else a receive)
};

void NetFabric::on_callback(void* arg) {
    Callback* c = static_cast<Callback*>(arg);
    if (c->send) {
        ++c->fabric->send_done_;
    } else {
        ++c->fabric->recv_done_;
    }
    c->fn();
    delete c;
}

class NetFabric::Wire final : public PeerWire {
public:
    Wire(NetFabric& f, int rank)
        : f_(f), rank_(rank), self_pending_(), tx_next_(f.nranks_, 0), rx_released_(f.nranks_, 0) {
        ops_.push_back(true);  // kNoOp
    }

    int rank() const override { return rank_; }
    int nranks() const override { return f_.nranks_; }
    int slots_per_link() const override { return f_.slots_; }
    std::size_t slot_bytes() const override { return f_.slot_bytes_; }

    std::uint64_t released(int dst) override {
        const std::size_t l = f_.index(rank_, dst);
        if (f_.config_.host_staged && dst != rank_) return std::min(f_.released_[l], f_.acked_[l]);
        return f_.released_[l];
    }

    OpId write_chunk(int dst, std::uint64_t seq, const ChunkHeader& header, const std::byte*) override {
        check(dst);
        if (seq != tx_next_[dst]) throw std::logic_error("NetWire::write_chunk: out-of-order seq");
        if (seq - released(dst) >= static_cast<std::uint64_t>(f_.slots_)) {
            throw std::logic_error("NetWire::write_chunk: no free slot on link " + std::to_string(rank_) + "->" +
                                   std::to_string(dst));
        }
        if (header.bytes > f_.slot_bytes_) throw std::invalid_argument("NetWire::write_chunk: chunk exceeds slot");
        double& clock = f_.host_.host_clock(rank_);
        clock += costs().host_post;
        f_.slot(dst, rank_, seq) = header;
        LinkStats& ls = f_.links_[f_.index(rank_, dst)];
        ++ls.chunks;
        ls.bytes += header.bytes;
        ls.max_in_flight = std::max<std::uint32_t>(ls.max_in_flight, static_cast<std::uint32_t>(seq + 1 - released(dst)));
        ++f_.tx_stats_[rank_].copies;
        const OpId op = new_op();
        ++tx_next_[dst];
        if (dst == rank_) {
            // Self link: a local copy, published by progress() once done.
            const double dur = header.bytes == 0 ? 0.0 : costs().tx_self(static_cast<double>(header.bytes));
            const double start = std::max(clock + (tx_free_ <= clock ? costs().copy_latency : 0.0), tx_free_);
            tx_free_ = start + dur;
            f_.tx_stats_[rank_].busy += dur;
            const double end = tx_free_;
            f_.sched_.schedule_at(end, [this, op, end] {
                ops_[op] = true;
                f_.host_.wake(rank_, end);
            });
            self_pending_.push_back({seq, op});
        } else if (f_.config_.host_staged) {
            // RdmaWire: stage into pinned host memory, then send from there.
            const double dur = header.bytes == 0 ? 0.0 : costs().tx_remote(static_cast<double>(header.bytes));
            const double start = std::max(clock + (tx_free_ <= clock ? costs().copy_latency : 0.0), tx_free_);
            tx_free_ = start + dur;
            f_.tx_stats_[rank_].busy += dur;
            const double end = tx_free_;
            f_.sched_.schedule_at(end, [this, op, end] {
                ops_[op] = true;
                f_.host_.wake(rank_, end);
            });
            f_.send(rank_, dst, static_cast<int>(2 * seq), header.bytes + f_.config_.header_bytes, end,
                    [this, dst, seq] { f_.send_completed(rank_, dst, seq); },
                    [this, dst, seq] { f_.delivered(rank_, dst, seq); });
        } else {
            const double at = clock;
            f_.send(rank_, dst, static_cast<int>(2 * seq), header.bytes + f_.config_.header_bytes, at,
                    [this, op, at] {
                        ops_[op] = true;
                        const double now = f_.sched_.now();
                        f_.tx_stats_[rank_].busy += now - at;
                        f_.host_.wake(rank_, now);
                    },
                    [this, dst, seq] { f_.delivered(rank_, dst, seq); });
        }
        f_.host_.note_activity(rank_);
        return op;
    }

    std::uint64_t arrived(int src) override { return f_.arrived_[f_.index(src, rank_)]; }

    const ChunkHeader& slot_header(int src, std::uint64_t seq) override {
        check(src);
        if (seq < rx_released_[src] || seq >= arrived(src)) throw std::logic_error("NetWire::slot_header: not held");
        return f_.slot(rank_, src, seq);
    }

    OpId copy_out(int src, std::uint64_t seq, std::byte*, std::size_t bytes) override {
        check(src);
        if (seq < rx_released_[src] || seq >= arrived(src)) throw std::logic_error("NetWire::copy_out: not held");
        if (bytes == 0 || bytes > f_.slot_bytes_) throw std::invalid_argument("NetWire::copy_out: bad size");
        double& clock = f_.host_.host_clock(rank_);
        clock += costs().host_copy_out;
        // The slot is in device memory (GPUDirect) or pinned host memory (staged).
        const bool host_slot = f_.config_.host_staged && src != rank_;
        const double dur = host_slot ? costs().rx_remote(static_cast<double>(bytes))
                                     : costs().rx_self(static_cast<double>(bytes));
        const double start = std::max(clock + (rx_free_ <= clock ? costs().copy_latency : 0.0), rx_free_);
        rx_free_ = start + dur;
        auto& st = f_.rx_stats_[rank_];
        st.busy += dur;
        ++st.copies;
        const OpId op = new_op();
        const double end = rx_free_;
        f_.sched_.schedule_at(end, [this, op, end] {
            ops_[op] = true;
            f_.host_.wake(rank_, end);
        });
        f_.host_.note_activity(rank_);
        return op;
    }

    void release(int src, std::uint64_t upto) override {
        check(src);
        if (upto < rx_released_[src] || upto > arrived(src)) throw std::logic_error("NetWire::release: bad counter");
        if (upto == rx_released_[src]) return;
        rx_released_[src] = upto;
        const std::size_t l = f_.index(src, rank_);
        auto apply = [this, l, upto, src] {
            f_.released_[l] = std::max(f_.released_[l], upto);
            f_.host_.wake(src, f_.sched_.now());
        };
        const double at = f_.host_.host_clock(rank_);
        if (src == rank_) {
            f_.sched_.schedule_at(at, apply);
        } else {
            const std::uint64_t n = f_.credits_sent_[l]++;
            f_.send(rank_, src, static_cast<int>(2 * n + 1), f_.config_.credit_bytes, at, [] {}, apply);
        }
        f_.host_.note_activity(rank_);
    }

    bool done(OpId op) override { return ops_.at(op); }

    // Publishes finished self-link copies as arrivals, in order.
    void progress() override {
        while (!self_pending_.empty() && ops_[self_pending_.front().op]) {
            const std::uint64_t upto = self_pending_.front().seq + 1;
            self_pending_.pop_front();
            const std::size_t l = f_.index(rank_, rank_);
            f_.arrived_[l] = std::max(f_.arrived_[l], upto);
            f_.host_.note_activity(rank_);
        }
    }

private:
    struct Pending {
        std::uint64_t seq;
        OpId op;
    };
    void check(int peer) const {
        if (peer < 0 || peer >= f_.nranks_) throw std::out_of_range("NetWire: bad peer " + std::to_string(peer));
    }
    OpId new_op() {
        ops_.push_back(false);
        return ops_.size() - 1;
    }
    const WireCosts& costs() const { return f_.local_[rank_]; }

    NetFabric& f_;
    int rank_;
    std::vector<bool> ops_;
    std::deque<Pending> self_pending_;
    std::vector<std::uint64_t> tx_next_;
    std::vector<std::uint64_t> rx_released_;
    double tx_free_ = 0, rx_free_ = 0;  // local engines (self copies, copy-outs)
};

NetFabric::NetFabric(Scheduler& scheduler, HostModel& host, int nranks, int slots_per_link, std::size_t slot_bytes,
                     std::vector<WireCosts> local, std::vector<AstraSim::AstraNetworkAPI*> net, NetConfig config)
    : sched_(scheduler), host_(host), nranks_(nranks), slots_(slots_per_link), slot_bytes_(slot_bytes),
      local_(std::move(local)), net_(std::move(net)), config_(config) {
    if (nranks < 1 || slots_per_link < 1 || slot_bytes == 0) throw std::invalid_argument("NetFabric: bad geometry");
    if (static_cast<int>(local_.size()) != nranks || static_cast<int>(net_.size()) != nranks) {
        throw std::invalid_argument("NetFabric: one WireCosts and one network endpoint per rank");
    }
    for (auto* n : net_) {
        if (n == nullptr) throw std::invalid_argument("NetFabric: null network endpoint");
    }
    const auto links = static_cast<std::size_t>(nranks) * nranks;
    headers_.assign(links, std::vector<ChunkHeader>(slots_per_link));
    arrived_.assign(links, 0);
    released_.assign(links, 0);
    delivered_.assign(links, {});
    in_order_.assign(links, 0);
    credits_sent_.assign(links, 0);
    send_done_set_.assign(links, {});
    acked_.assign(links, 0);
    links_.assign(links, {});
    tx_stats_.assign(nranks, {});
    rx_stats_.assign(nranks, {});
    for (int r = 0; r < nranks; ++r) wires_.push_back(std::make_unique<Wire>(*this, r));
}

NetFabric::~NetFabric() = default;

PeerWire& NetFabric::wire(int rank) { return *wires_.at(rank); }

ChunkHeader& NetFabric::slot(int owner, int src, std::uint64_t seq) {
    return headers_[index(owner, src)][seq % static_cast<std::uint64_t>(slots_)];
}

void NetFabric::send(int src, int dst, int tag, std::uint64_t bytes, double at, std::function<void()> on_sent,
                     std::function<void()> on_received) {
    sched_.schedule_at(at, [this, src, dst, tag, bytes, on_sent = std::move(on_sent),
                            on_received = std::move(on_received)]() mutable {
        AstraSim::sim_request req{};
        req.srcRank = static_cast<std::uint32_t>(src);
        req.dstRank = static_cast<std::uint32_t>(dst);
        req.tag = static_cast<std::uint32_t>(tag);
        req.reqType = AstraSim::UINT8;
        req.reqCount = bytes;
        req.flowTag.sender_node = src;
        req.flowTag.receiver_node = dst;
        req.flowTag.flow_size = bytes;
        req.flowTag.tag_id = tag;
        AstraSim::sim_request rreq = req;
        // Post the receive first: the backend may deliver within sim_send.
        net_[dst]->sim_recv(nullptr, bytes, AstraSim::UINT8, src, tag, &rreq, &NetFabric::on_callback,
                            new Callback{this, std::move(on_received), false});
        ++sent_;
        net_[src]->sim_send(nullptr, bytes, AstraSim::UINT8, dst, tag, &req, &NetFabric::on_callback,
                            new Callback{this, std::move(on_sent), true});
    });
}

// Staged mode: the send of chunk `seq` src -> dst completed; the sender sees
// completions in order (one RC QP per link), and may reuse staging slots.
void NetFabric::send_completed(int src, int dst, std::uint64_t seq) {
    const std::size_t l = index(src, dst);
    const double now = sched_.now();
    send_done_set_[l].insert(seq);
    std::uint64_t& upto = acked_[l];
    while (!send_done_set_[l].empty() && *send_done_set_[l].begin() == upto) {
        send_done_set_[l].erase(send_done_set_[l].begin());
        ++upto;
    }
    host_.wake(src, now);
}

// A chunk src -> dst was delivered; arrivals become visible in order.
void NetFabric::delivered(int src, int dst, std::uint64_t seq) {
    const std::size_t l = index(src, dst);
    delivered_[l].insert(seq);
    std::uint64_t& upto = in_order_[l];
    const std::uint64_t before = upto;
    while (!delivered_[l].empty() && *delivered_[l].begin() == upto) {
        delivered_[l].erase(delivered_[l].begin());
        ++upto;
    }
    if (upto == before) return;
    const double at = sched_.now() + config_.arrival_latency;
    sched_.schedule_at(at, [this, l, upto = upto, dst, at] {
        arrived_[l] = std::max(arrived_[l], upto);
        host_.wake(dst, at);
    });
}

}  // namespace nano_nccl::sim::simai
