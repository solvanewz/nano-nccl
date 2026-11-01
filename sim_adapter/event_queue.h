#pragma once

#include <cstdint>
#include <functional>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

// Discrete-event core of the EP simulator. Time is in microseconds.
//
// The layer model and the wires only talk to a Scheduler, so they run on this
// queue or on another event loop: sim_adapter/simai/ adapts SimAI's
// AstraNetworkAPI (sim_get_time / sim_schedule), which drives them from
// SimAI's ns-3 event loop.
namespace nano_nccl::sim {

class Scheduler {
public:
    using Fn = std::function<void()>;
    virtual ~Scheduler() = default;
    // Time of the event running now (or of the last one).
    virtual double now() const = 0;
    // Runs `fn` at absolute time `t` (>= now). Events at equal times run in
    // scheduling order.
    virtual void schedule_at(double t, Fn fn) = 0;
    void schedule(double delay, Fn fn) { schedule_at(now() + delay, std::move(fn)); }
};

// The simulator's own event loop; deterministic.
class EventQueue final : public Scheduler {
public:
    double now() const override { return now_; }

    void schedule_at(double t, Fn fn) override {
        if (t < now_) throw std::logic_error("EventQueue: event scheduled in the past");
        heap_.push({t, next_seq_++, std::move(fn)});
    }

    // Runs events until none are left or `max_events` ran; returns how many ran.
    std::uint64_t run(std::uint64_t max_events = UINT64_MAX) {
        std::uint64_t ran = 0;
        while (!heap_.empty() && ran < max_events) {
            Event e = heap_.top();
            heap_.pop();
            now_ = e.time;
            e.fn();
            ++ran;
        }
        return ran;
    }
    bool empty() const { return heap_.empty(); }

private:
    struct Event {
        double time;
        std::uint64_t seq;
        Fn fn;
    };
    struct Later {
        bool operator()(const Event& a, const Event& b) const {
            return a.time != b.time ? a.time > b.time : a.seq > b.seq;
        }
    };
    double now_ = 0;
    std::uint64_t next_seq_ = 0;
    std::priority_queue<Event, std::vector<Event>, Later> heap_;
};

}  // namespace nano_nccl::sim
