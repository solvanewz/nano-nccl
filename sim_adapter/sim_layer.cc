#include "sim_layer.h"

#include "transport/peer/peer_exchanger.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <memory>
#include <stdexcept>
#include <string>

namespace nano_nccl::sim {

namespace {

using exec::Condition;
using exec::OpKind;
using transport::peer::ExchangeId;
using transport::peer::PeerExchanger;

// Stand-ins for device buffers: the simulated wire never dereferences them.
std::byte* fake_buffer(std::uintptr_t base) { return reinterpret_cast<std::byte*>(base); }

}  // namespace

double gpu_cost(const CostProfile& p, const std::string& mode, OpKind kind, const exec::ComputeWork& w) {
    switch (kind) {
        case OpKind::kRoute:
            return p.table(w.route_packs ? cost::kRoute : cost::kCount)(static_cast<double>(w.entries));
        case OpKind::kPack: return p.table(cost::kPack)(static_cast<double>(w.entries));
        case OpKind::kReduce: return p.table(cost::kReduce)(static_cast<double>(w.tokens));
        case OpKind::kExpert: {
            double t = 2 * p.table(cost::kPermute)(static_cast<double>(w.permute_rows));
            if (mode == "probe") {
                std::int64_t rows = 0;
                for (std::int64_t r : w.expert_rows) rows += r;
                t += p.table(cost::kExpertProbe)(static_cast<double>(rows));
            } else {
                const Table& ffn = p.table(cost::kExpertFfn);
                for (std::int64_t r : w.expert_rows) {
                    if (r > 0) t += ffn(static_cast<double>(r));
                }
            }
            return t;
        }
        default: throw std::invalid_argument("gpu_cost: not a GPU op");
    }
}

std::unique_ptr<Fabric> make_sim_fabric(Scheduler& scheduler, HostModel& host, int world, int slots_per_link,
                                        std::size_t slot_bytes, std::vector<WireCosts> costs) {
    return std::make_unique<SimFabric>(scheduler, host, world, slots_per_link, slot_bytes, std::move(costs));
}

class LayerSimulation::Impl final : public HostModel {
public:
    Impl(const SimLayerConfig& c, const std::vector<std::vector<std::int64_t>>& counts,
         const std::vector<int>& tokens, const CostProfile& profile, Scheduler& q, const FabricFactory& fabric)
        : c_(c), counts_(counts), tokens_(tokens), p_(profile), q_(q) {
        const int W = c.world;
        if (W < 1 || static_cast<int>(counts.size()) != W || static_cast<int>(tokens.size()) != W) {
            throw std::invalid_argument("simulate_layer: counts/tokens need one entry per rank");
        }
        for (const auto& row : counts) {
            if (static_cast<int>(row.size()) != c.num_experts) {
                throw std::invalid_argument("simulate_layer: counts rows need num_experts entries");
            }
            global_counts_.insert(global_counts_.end(), row.begin(), row.end());
        }
        std::vector<WireCosts> wc;
        for (int r = 0; r < W; ++r) {
            wc.push_back(WireCosts::from_profile(profile, c.contention, r));
            wc.back().copy_scale = c.copy_scale;
        }
        for (int r = 0; r < W; ++r) {
            gpu_copy_slowdown_.push_back(c.contention ? profile.scalar_for(cost::kGpuCopySlowdown, r, 1.0) : 1.0);
        }
        host_poll_ = profile.scalar(cost::kHostPoll);
        launch_latency_ = profile.scalar_or(cost::kLaunchLatency, 0);
        fabric_ = fabric(q_, *this, W, c.slots_per_link, c.slot_bytes, wc);
        if (!fabric_ || fabric_->nranks() != W) throw std::invalid_argument("simulate_layer: bad fabric");
        ranks_.resize(W);
        for (int r = 0; r < W; ++r) {
            Rank& k = ranks_[r];
            k.shape = ep::EpShape{W, r, c.num_experts, c.top_k, c.hidden, c.ffn};
            ep::validate_shape(k.shape);
            k.result.program = exec::build_program(r, c.plan, c.options);
            k.cursor = std::make_unique<exec::ProgramCursor>(k.result.program);
            k.ex = std::make_unique<PeerExchanger>(&fabric_->wire(r));
            const std::size_t n = k.result.program.ops.size();
            k.result.ops.assign(n, {});
            k.result.messages.assign(n, {});
            k.ids.assign(n, 0);
            k.done.assign(n, 0);
            k.gpu_finished.assign(n, 0);
            k.send_counts.assign(static_cast<std::size_t>(W) * c.num_experts, 0);
            k.recv_counts.assign(static_cast<std::size_t>(W) * c.num_experts, 0);
        }
    }

    void start(std::function<void()> on_done) {
        on_done_ = std::move(on_done);
        for (int r = 0; r < c_.world; ++r) wake(r, q_.now());
    }

    SimResult finish(std::uint64_t events) {
        SimResult out;
        out.events = events;
        for (int r = 0; r < c_.world; ++r) {
            Rank& k = ranks_[r];
            if (!k.done[k.result.program.retire]) {
                std::string where;
                for (std::size_t i = 0; i < k.result.ops.size(); ++i) {
                    if (!k.cursor->issued(static_cast<int>(i))) {
                        where = exec::op_kind_name(k.result.program.ops[i].kind);
                        where += " (op " + std::to_string(i) + ")";
                        break;
                    }
                }
                // Exchanges that never reached a condition retire waits for.
                std::string open;
                const auto& prog = k.result.program;
                for (std::size_t i = 0; i < prog.ops.size(); ++i) {
                    if (prog.ops[i].queue != exec::Queue::kExchange || k.ids[i] == 0) continue;
                    for (Condition c : {Condition::kConsumerReadable, Condition::kSendBufferReusable,
                                        Condition::kReceiveSlotsReleased}) {
                        if (!k.ex->query_completion(k.ids[i], exec::to_completion_event(c))) {
                            open += std::string(open.empty() ? "" : ", ") + exec::op_kind_name(prog.ops[i].kind) +
                                    " op " + std::to_string(i) + " step " + std::to_string(prog.ops[i].step) +
                                    " not " + exec::condition_name(c);
                        }
                    }
                }
                throw std::runtime_error("simulate_layer: deadlock, rank " + std::to_string(r) +
                                         " stopped before " + where + (open.empty() ? "" : "; " + open));
            }
            k.result.total = k.result.ops[k.result.program.retire].issue_end;
            out.makespan = std::max(out.makespan, k.result.total);
            out.ranks.push_back(std::move(k.result));
        }
        const int W = c_.world;
        for (int s = 0; s < W; ++s) {
            for (int d = 0; d < W; ++d) out.links.push_back(fabric_->link(s, d));
            out.tx.push_back(fabric_->tx(s));
            out.rx.push_back(fabric_->rx(s));
        }
        return out;
    }

    double& host_clock(int rank) override { return ranks_[rank].clock; }
    bool gpu_busy(int rank, double t) const override {
        const Rank& k = ranks_[rank];
        return k.gpu_free > t || !k.gpu_queue.empty();
    }
    void note_activity(int rank) override { ++ranks_[rank].activity; }

    void wake(int rank, double at) override {
        Rank& k = ranks_[rank];
        at = std::max({at, k.busy_until, q_.now()});
        if (k.wake_pending && k.wake_at <= at) return;
        k.wake_pending = true;
        k.wake_at = at;
        const std::uint64_t token = ++k.wake_token;
        q_.schedule_at(at, [this, rank, token] {
            if (ranks_[rank].wake_token == token) host_pass(rank);
        });
    }

private:
    struct GpuJob {
        int op;
        double dur;     // before interference
        double issued;  // host time of the issue
    };
    struct Rank {
        ep::EpShape shape;
        SimRankResult result;
        std::unique_ptr<exec::ProgramCursor> cursor;
        std::unique_ptr<PeerExchanger> ex;
        bool have_plan = false;
        std::vector<ExchangeId> ids;
        std::vector<char> done;          // Condition::kDone as the host sees it
        std::vector<char> gpu_finished;  // device finished (host may not have looked yet)
        std::vector<int> stream_ops;     // issued, in order
        std::size_t stream_polled = 0;
        std::vector<std::uint64_t> send_counts, recv_counts;  // counts exchange stand-ins
        double clock = 0, busy_until = 0;
        std::deque<GpuJob> gpu_queue;  // issued, not started
        bool gpu_running = false;      // an op is started (or about to start)
        GpuJob gpu_current{};          // the running op
        bool gpu_in_op = false;
        double gpu_current_start = 0;
        double gpu_left = 0, gpu_last = 0, gpu_rate = 1;  // left: work at rate 1, as of gpu_last
        std::uint64_t gpu_end_token = 0;
        double gpu_free = 0;  // (expected) end of the last started op
        bool wake_pending = false;
        double wake_at = 0;
        std::uint64_t wake_token = 0;
        std::uint64_t activity = 0;
    };

    // One wake of a rank's host thread: loops like the runtime's forward()
    // until a pass changes nothing.
    void host_pass(int r) {
        Rank& k = ranks_[r];
        k.wake_pending = false;
        const exec::LayerProgram& prog = k.result.program;
        // Retired: every exchange finished, including the arrivals peers
        // needed from this rank, so its host has nothing left to drive.
        if (k.done[prog.retire]) return;
        k.clock = q_.now();
        auto satisfied = [&](const exec::Completion& dep) {
            bool ok;
            if (dep.condition == Condition::kDone) {
                ok = k.done[dep.op] != 0;
            } else {
                ok = k.ids[dep.op] != 0 &&
                     k.ex->query_completion(k.ids[dep.op], exec::to_completion_event(dep.condition));
            }
            if (ok) observe(k, dep.op, dep.condition);
            return ok;
        };
        for (int pass = 0;; ++pass) {
            if (pass > 1000000) throw std::logic_error("simulate_layer: host pass does not settle");
            const std::uint64_t before = k.activity;
            k.clock += host_poll_;
            k.ex->progress();
            while (k.stream_polled < k.stream_ops.size() && k.gpu_finished[k.stream_ops[k.stream_polled]]) {
                const int op = k.stream_ops[k.stream_polled++];
                k.done[op] = 1;
                observe(k, op, Condition::kDone);
                ++k.activity;
            }
            for (std::size_t op = 0; op < prog.ops.size(); ++op) {
                if (k.ids[op] == 0) continue;
                for (int cond = 0; cond < 5; ++cond) {
                    if (k.result.ops[op].condition[cond] < 0 &&
                        k.ex->query_completion(k.ids[op], static_cast<transport::peer::CompletionEvent>(cond))) {
                        observe(k, static_cast<int>(op), static_cast<Condition>(cond));
                    }
                }
            }
            for (int op; (op = k.cursor->next_ready(satisfied)) != -1;) {
                SimOpTimes& t = k.result.ops[op];
                t.issue_begin = k.clock;
                issue(r, op);
                k.cursor->mark_issued(op);
                t.issue_end = k.clock;
                if (prog.ops[op].queue == exec::Queue::kHost) {
                    k.done[op] = 1;
                    observe(k, op, Condition::kDone);
                }
                ++k.activity;
            }
            if (k.activity == before || k.done[prog.retire]) break;
        }
        k.busy_until = k.clock;
        if (k.done[prog.retire] && ++retired_ == c_.world && on_done_) on_done_();
    }

    void observe(Rank& k, int op, Condition cond) {
        SimOpTimes& t = k.result.ops[op];
        double& at = t.condition[static_cast<int>(cond)];
        if (at >= 0) return;
        at = k.clock;
        const bool exchange = k.result.program.ops[op].queue == exec::Queue::kExchange;
        if ((exchange && cond == Condition::kConsumerReadable) || (!exchange && cond == Condition::kDone)) {
            t.complete = k.clock;
        }
    }

    void issue(int r, int op) {
        Rank& k = ranks_[r];
        const exec::LayerProgram& prog = k.result.program;
        const exec::Operation& o = prog.ops[op];
        switch (o.kind) {
            case OpKind::kRoute:
                k.clock += p_.scalar(prog.timing == exec::PlanTiming::kAhead ? cost::kHostRouteIssue
                                                                             : cost::kHostCountIssue);
                gpu(r, op, gpu_cost(p_, c_.mode, o.kind, work(k, op)));
                break;
            case OpKind::kPack:
                k.clock += p_.scalar(cost::kHostPackIssue);
                gpu(r, op, gpu_cost(p_, c_.mode, o.kind, work(k, op)));
                break;
            case OpKind::kCounts: {
                k.clock += p_.scalar(cost::kHostExchangeIssue);
                k.ids[op] = transport::peer::exchange_counts_async(*k.ex, c_.iteration, k.send_counts.data(),
                                                                   k.recv_counts.data(),
                                                                   exec::counts_per_peer(k.shape, prog.timing));
                exec::EpPlan shape_only;
                shape_only.shape = k.shape;
                k.result.messages[op] = exec::outgoing_messages(prog, op, shape_only, c_.iteration);
                break;
            }
            case OpKind::kPlan: {
                k.clock += p_.table(cost::kHostPlan)(prog.steps);
                if (prog.timing == exec::PlanTiming::kFromCounts) {
                    k.clock += p_.table(std::string(cost::kHostSchedule) + planner::scheduler_name(c_.plan.scheduler))(
                        static_cast<double>(c_.num_experts) * prog.steps);
                }
                const int epr = k.shape.experts_per_rank();
                std::vector<std::int64_t> recv(static_cast<std::size_t>(c_.world) * epr);
                for (int s = 0; s < c_.world; ++s) {
                    for (int j = 0; j < epr; ++j) recv[s * epr + j] = counts_[s][r * epr + j];
                }
                // The scheduler itself: the same code the runtime's plan op runs.
                const planner::GroupPlan groups = exec::resolve_plan(c_.plan, c_.world, c_.num_experts, global_counts_);
                k.result.plan = exec::make_ep_plan(k.shape, groups, c_.options, counts_[r], recv, c_.slot_bytes,
                                                   prog.timing);
                k.have_plan = true;
                break;
            }
            case OpKind::kDispatch:
            case OpKind::kCombine: {
                k.clock += p_.scalar(cost::kHostExchangeIssue);
                const bool d = o.kind == OpKind::kDispatch;
                k.ids[op] = k.ex->exchangev_async(c_.iteration, exec::exchange_tag(prog, op),
                                                  exec::exchange_layout(prog, op, k.result.plan),
                                                  fake_buffer(d ? 0x100000000ull : 0x200000000ull),
                                                  fake_buffer(d ? 0x200000000ull : 0x300000000ull));
                k.result.messages[op] = exec::outgoing_messages(prog, op, k.result.plan, c_.iteration);
                break;
            }
            case OpKind::kExpert: {
                const exec::ComputeWork w = work(k, op);
                int active = 0;
                for (std::int64_t rows : w.expert_rows) active += rows > 0 ? 1 : 0;
                k.clock += p_.table(cost::kHostExpertIssue)(active);
                gpu(r, op, gpu_cost(p_, c_.mode, o.kind, w));
                break;
            }
            case OpKind::kReduce:
                k.clock += p_.scalar(cost::kHostReduceIssue);
                gpu(r, op, gpu_cost(p_, c_.mode, o.kind, work(k, op)));
                break;
            case OpKind::kRetire:
                for (std::size_t i = 0; i < k.ids.size(); ++i) {
                    if (k.ids[i] != 0) k.ex->retire(k.ids[i]);
                }
                break;
        }
    }

    exec::ComputeWork work(const Rank& k, int op) const {
        const OpKind kind = k.result.program.ops[op].kind;
        if (kind == OpKind::kRoute || kind == OpKind::kPack) {  // may run before the plan exists
            exec::EpPlan shape_only;
            shape_only.shape = k.shape;
            return exec::compute_work(k.result.program, op, shape_only, tokens_[k.shape.rank]);
        }
        return exec::compute_work(k.result.program, op, k.result.plan, tokens_[k.shape.rank]);
    }

    // Enqueues GPU work on rank r's stream at the current host time. The
    // stream runs its ops in issue order; an op issued to an idle stream
    // starts after the launch latency.
    void gpu(int r, int op, double dur) {
        Rank& k = ranks_[r];
        k.stream_ops.push_back(op);
        k.gpu_queue.push_back({op, dur * c_.compute_scale, k.clock});
        if (!k.gpu_running) gpu_start_next(r, k.clock + launch_latency_);
    }

    // Starts the next queued op of rank r's stream at time t. It runs at
    // 1 / (1 + (gpu_copy_slowdown - 1) * k / 2) while k of the rank's copy
    // engines move remote chunks, so its end moves whenever k changes
    // (remote_busy_changed).
    void gpu_start_next(int r, double t) {
        Rank& k = ranks_[r];
        k.gpu_running = true;
        q_.schedule_at(t, [this, r, t] {
            Rank& k = ranks_[r];
            if (k.gpu_queue.empty()) return;
            k.gpu_current = k.gpu_queue.front();
            k.gpu_queue.pop_front();
            k.gpu_current_start = t;
            k.gpu_left = k.gpu_current.dur;
            k.gpu_last = t;
            k.gpu_in_op = true;
            k.result.ops[k.gpu_current.op].gpu_start = t;
            schedule_gpu_end(r, t);
        });
    }

    double gpu_rate(int r) const {
        return 1.0 / (1.0 + (gpu_copy_slowdown_[r] - 1.0) * fabric_->remote_busy(r) / 2.0);
    }

    // Brings the running op up to time t at its old rate, then schedules its
    // end at the current rate.
    void schedule_gpu_end(int r, double t) {
        Rank& k = ranks_[r];
        k.gpu_left = std::max(0.0, k.gpu_left - (t - k.gpu_last) * k.gpu_rate);
        k.gpu_last = t;
        k.gpu_rate = gpu_rate(r);
        const double end = t + k.gpu_left / k.gpu_rate;
        k.gpu_free = end;
        const std::uint64_t token = ++k.gpu_end_token;
        q_.schedule_at(end, [this, r, end, token] {
            Rank& k = ranks_[r];
            if (k.gpu_end_token != token) return;  // superseded by a later estimate
            k.gpu_in_op = false;
            const int op = k.gpu_current.op;
            k.result.gpu_busy += end - k.gpu_current_start;
            k.result.ops[op].gpu_end = end;
            k.gpu_finished[op] = 1;
            k.gpu_running = false;
            if (!k.gpu_queue.empty()) gpu_start_next(r, std::max(end, k.gpu_queue.front().issued + launch_latency_));
            wake(r, end);
        });
    }

    void remote_busy_changed(int rank, double t) override {
        if (ranks_.empty()) return;
        if (ranks_[rank].gpu_in_op) schedule_gpu_end(rank, t);
    }

    const SimLayerConfig& c_;
    const std::vector<std::vector<std::int64_t>>& counts_;
    std::vector<std::int64_t> global_counts_;  // N(s, e) at s * E + e
    const std::vector<int>& tokens_;
    const CostProfile& p_;
    double host_poll_ = 0, launch_latency_ = 0;
    std::vector<double> gpu_copy_slowdown_;  // per rank
    Scheduler& q_;
    std::function<void()> on_done_;
    int retired_ = 0;
    std::unique_ptr<Fabric> fabric_;
    std::vector<Rank> ranks_;
};

LayerSimulation::LayerSimulation(const SimLayerConfig& config, const std::vector<std::vector<std::int64_t>>& counts,
                                 const std::vector<int>& tokens, const CostProfile& profile, Scheduler& scheduler,
                                 const FabricFactory& fabric) {
    if (config.plan.timing == exec::PlanTiming::kAhead) {
        planner::validate_group_plan(config.plan.plan, config.num_experts);
    }
    impl_ = std::make_unique<Impl>(config, counts, tokens, profile, scheduler, fabric);
}

LayerSimulation::~LayerSimulation() = default;

void LayerSimulation::start(std::function<void()> on_all_retired) { impl_->start(std::move(on_all_retired)); }

SimResult LayerSimulation::finish(std::uint64_t events) { return impl_->finish(events); }

SimResult simulate_layer(const SimLayerConfig& config, const std::vector<std::vector<std::int64_t>>& counts,
                         const std::vector<int>& tokens, const CostProfile& profile) {
    EventQueue q;
    LayerSimulation sim(config, counts, tokens, profile, q);
    sim.start();
    const std::uint64_t events = q.run();
    return sim.finish(events);
}

}  // namespace nano_nccl::sim
