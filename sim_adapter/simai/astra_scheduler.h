#pragma once

#include "astra-sim/system/AstraNetworkAPI.hh"
#include "event_queue.h"

#include <stdexcept>
#include <utility>

// sim::Scheduler on top of SimAI's AstraNetworkAPI (sim_get_time and
// sim_schedule, both in nanoseconds as SimAI's ns-3 front end uses them), so
// the layer model and NetFabric run on the network backend's event loop.
// While one of its events runs, now() is that event's exact time in us, so
// rounding of the backend's clock never moves the model's times.
namespace nano_nccl::sim::simai {

class AstraScheduler final : public Scheduler {
public:
    explicit AstraScheduler(AstraSim::AstraNetworkAPI& api) : api_(api) {}

    double now() const override {
        if (in_event_) return current_;
        return api_.sim_get_time().time_val / 1000.0;
    }

    void schedule_at(double t, Fn fn) override {
        const double base = now();
        if (t < base) throw std::logic_error("AstraScheduler: event scheduled in the past");
        AstraSim::timespec_t delta;
        delta.time_res = AstraSim::NS;
        delta.time_val = (t - base) * 1000.0;
        api_.sim_schedule(delta, &AstraScheduler::trampoline, new Pending{this, t, std::move(fn)});
    }

private:
    struct Pending {
        AstraScheduler* self;
        double t;
        Fn fn;
    };
    static void trampoline(void* arg) {
        Pending* p = static_cast<Pending*>(arg);
        AstraScheduler& s = *p->self;
        const bool outer = s.in_event_;
        const double saved = s.current_;
        s.in_event_ = true;
        s.current_ = p->t;
        p->fn();
        s.in_event_ = outer;
        s.current_ = saved;
        delete p;
    }

    AstraSim::AstraNetworkAPI& api_;
    bool in_event_ = false;
    double current_ = 0;
};

}  // namespace nano_nccl::sim::simai
