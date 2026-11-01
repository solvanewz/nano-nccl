#pragma once

#include "astra-sim/system/AstraNetworkAPI.hh"
#include "event_queue.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

// A small analytical network behind SimAI's AstraNetworkAPI, on the
// simulator's own EventQueue: every node has one tx and one rx NIC port of
// `gbps`, FIFO; a message's send completes when its last byte left the tx
// port, and it arrives `latency_us` later, or when the rx port finished the
// messages ahead of it. Used to test NetFabric without ns-3 and to sweep
// network parameters quickly; SimAI's ns-3 backend implements the same
// interface (sim_adapter/simai/ns3/nano_ep_ns3.cc).
namespace nano_nccl::sim::simai {

class AnalyticalNetwork {
public:
    AnalyticalNetwork(EventQueue& q, int nodes, double gbps, double latency_us)
        : q_(q), bytes_per_us_(gbps * 1e3 / 8.0), latency_(latency_us), tx_free_(nodes, 0), rx_free_(nodes, 0) {
        if (nodes < 1 || gbps <= 0 || latency_us < 0) throw std::invalid_argument("AnalyticalNetwork: bad parameters");
        for (int r = 0; r < nodes; ++r) endpoints_.push_back(std::make_unique<Endpoint>(*this, r));
    }

    AstraSim::AstraNetworkAPI& endpoint(int rank) { return *endpoints_.at(rank); }
    std::vector<AstraSim::AstraNetworkAPI*> endpoints() {
        std::vector<AstraSim::AstraNetworkAPI*> out;
        for (auto& e : endpoints_) out.push_back(e.get());
        return out;
    }
    double bytes_per_us() const { return bytes_per_us_; }
    double latency_us() const { return latency_; }
    std::uint64_t messages() const { return messages_; }

private:
    using Handler = void (*)(void*);
    using Key = std::tuple<int, int, int>;  // src, dst, tag
    struct Posted {
        std::uint64_t count;
        Handler fn;
        void* arg;
    };

    class Endpoint final : public AstraSim::AstraNetworkAPI {
    public:
        Endpoint(AnalyticalNetwork& n, int rank) : AstraNetworkAPI(rank), n_(n) {}
        int sim_comm_size(AstraSim::sim_comm, int* size) override {
            *size = static_cast<int>(n_.endpoints_.size());
            return 0;
        }
        int sim_finish() override { return 0; }
        double sim_time_resolution() override { return 1; }
        int sim_init(AstraSim::AstraMemoryAPI*) override { return 0; }
        AstraSim::timespec_t sim_get_time() override {
            AstraSim::timespec_t t;
            t.time_res = AstraSim::NS;
            t.time_val = n_.q_.now() * 1000.0;
            return t;
        }
        void sim_schedule(AstraSim::timespec_t delta, void (*fn)(void*), void* arg) override {
            n_.q_.schedule(delta.time_val / 1000.0, [fn, arg] { fn(arg); });
        }
        int sim_send(void*, std::uint64_t count, int, int dst, int tag, AstraSim::sim_request*, void (*fn)(void*),
                     void* arg) override {
            n_.send(rank, dst, tag, count, fn, arg);
            return 0;
        }
        int sim_recv(void*, std::uint64_t count, int, int src, int tag, AstraSim::sim_request*, void (*fn)(void*),
                     void* arg) override {
            n_.recv(src, rank, tag, count, fn, arg);
            return 0;
        }

    private:
        AnalyticalNetwork& n_;
    };

    void send(int src, int dst, int tag, std::uint64_t count, Handler fn, void* arg) {
        check(src);
        check(dst);
        ++messages_;
        const double wire = static_cast<double>(count) / bytes_per_us_;
        const double start = std::max(q_.now(), tx_free_[src]);
        const double sent = start + wire;
        tx_free_[src] = sent;
        const double arrive = std::max(sent + latency_, rx_free_[dst] + wire);
        rx_free_[dst] = arrive;
        q_.schedule_at(sent, [fn, arg] { fn(arg); });
        q_.schedule_at(arrive, [this, src, dst, tag, count] { arrived(Key{src, dst, tag}, count); });
    }

    void recv(int src, int dst, int tag, std::uint64_t count, Handler fn, void* arg) {
        const Key k{src, dst, tag};
        auto u = unexpected_.find(k);
        if (u != unexpected_.end()) {
            if (u->second != count) throw std::logic_error("AnalyticalNetwork: receive size mismatch");
            unexpected_.erase(u);
            q_.schedule(0, [fn, arg] { fn(arg); });
            return;
        }
        if (!posted_.emplace(k, Posted{count, fn, arg}).second) {
            throw std::logic_error("AnalyticalNetwork: receive posted twice");
        }
    }

    void arrived(const Key& k, std::uint64_t count) {
        auto p = posted_.find(k);
        if (p == posted_.end()) {
            if (!unexpected_.emplace(k, count).second) throw std::logic_error("AnalyticalNetwork: duplicate message");
            return;
        }
        if (p->second.count != count) throw std::logic_error("AnalyticalNetwork: receive size mismatch");
        const Posted got = p->second;
        posted_.erase(p);
        got.fn(got.arg);
    }

    void check(int r) const {
        if (r < 0 || r >= static_cast<int>(endpoints_.size())) throw std::out_of_range("AnalyticalNetwork: bad node");
    }

    EventQueue& q_;
    double bytes_per_us_, latency_;
    std::vector<double> tx_free_, rx_free_;
    std::vector<std::unique_ptr<Endpoint>> endpoints_;
    std::map<Key, Posted> posted_;
    std::map<Key, std::uint64_t> unexpected_;
    std::uint64_t messages_ = 0;
};

}  // namespace nano_nccl::sim::simai
