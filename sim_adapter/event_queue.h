#pragma once

#include <cstdint>
#include <functional>
#include <queue>
#include <stdexcept>
#include <vector>

// Discrete-event core of the EP simulator. Time is in microseconds.
//
// It has the shape of SimAI's AstraNetworkAPI scheduling calls
// (sim_get_time / sim_schedule), so the layer model can later run on SimAI's
// ns-3 event loop through a bridge instead of this queue.
namespace nano_nccl::sim {

class EventQueue {
public:
    using Fn = std::function<void()>;

    double now() const { return now_; }

    // Runs `fn` at absolute time `t` (>= now). Events at equal times run in
    // scheduling order, so a run is deterministic.
    void schedule_at(double t, Fn fn) {
        if (t < now_) throw std::logic_error("EventQueue: event scheduled in the past");
        heap_.push({t, next_seq_++, std::move(fn)});
    }
    void schedule(double delay, Fn fn) { schedule_at(now_ + delay, std::move(fn)); }

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
