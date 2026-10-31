#include "sim_layer.h"

#include "transport/peer/peer_exchanger.h"

#include <algorithm>
#include <cstdint>
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
        case OpKind::kRoute: return p.table(cost::kRoute)(static_cast<double>(w.entries));
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

namespace {

class Simulation final : public HostModel {
public:
    Simulation(const SimLayerConfig& c, const std::vector<std::vector<std::int64_t>>& counts,
               const std::vector<int>& tokens, const CostProfile& profile)
        : c_(c), counts_(counts), tokens_(tokens), p_(profile) {
        const int W = c.world;
        if (W < 1 || static_cast<int>(counts.size()) != W || static_cast<int>(tokens.size()) != W) {
            throw std::invalid_argument("simulate_layer: counts/tokens need one entry per rank");
        }
        for (const auto& row : counts) {
            if (static_cast<int>(row.size()) != c.num_experts) {
                throw std::invalid_argument("simulate_layer: counts rows need num_experts entries");
            }
        }
        std::vector<WireCosts> wc;
        for (int r = 0; r < W; ++r) {
            wc.push_back(WireCosts::from_profile(profile, c.contention, r));
            wc.back().copy_scale = c.copy_scale;
        }
        gpu_copy_slowdown_ = c.contention ? profile.scalar_or(cost::kGpuCopySlowdown, 1.0) : 1.0;
        host_poll_ = profile.scalar(cost::kHostPoll);
        launch_latency_ = profile.scalar_or(cost::kLaunchLatency, 0);
        fabric_ = std::make_unique<SimFabric>(q_, *this, W, c.slots_per_link, c.slot_bytes, wc);
        ranks_.resize(W);
        for (int r = 0; r < W; ++r) {
            Rank& k = ranks_[r];
            k.shape = ep::EpShape{W, r, c.num_experts, c.top_k, c.hidden, c.ffn};
            ep::validate_shape(k.shape);
            k.result.program = exec::build_layer_program(r, c.groups.order, c.options);
            k.cursor = std::make_unique<exec::ProgramCursor>(k.result.program);
            k.ex = std::make_unique<PeerExchanger>(&fabric_->wire(r));
            const std::size_t n = k.result.program.ops.size();
            k.result.ops.assign(n, {});
            k.result.messages.assign(n, {});
            k.ids.assign(n, 0);
            k.done.assign(n, 0);
            k.gpu_finished.assign(n, 0);
            k.send_counts.assign(static_cast<std::size_t>(c.num_experts), 0);
            k.recv_counts.assign(static_cast<std::size_t>(c.num_experts), 0);
        }
    }

    SimResult run() {
        for (int r = 0; r < c_.world; ++r) wake(r, 0);
        const std::uint64_t events = q_.run();
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
                throw std::runtime_error("simulate_layer: deadlock, rank " + std::to_string(r) +
                                         " stopped before " + where);
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
    bool gpu_busy(int rank, double t) const override { return ranks_[rank].gpu_free > t; }
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
        double clock = 0, busy_until = 0, gpu_free = 0;
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
                k.clock += p_.scalar(cost::kHostRouteIssue);
                gpu(r, op, gpu_cost(p_, c_.mode, o.kind, work(k, op)));
                break;
            case OpKind::kCounts: {
                k.clock += p_.scalar(cost::kHostExchangeIssue);
                k.ids[op] = transport::peer::exchange_counts_async(*k.ex, c_.iteration, k.send_counts.data(),
                                                                   k.recv_counts.data(),
                                                                   static_cast<std::size_t>(k.shape.experts_per_rank()));
                exec::EpPlan shape_only;
                shape_only.shape = k.shape;
                k.result.messages[op] = exec::outgoing_messages(prog, op, shape_only, c_.iteration);
                break;
            }
            case OpKind::kPlan: {
                k.clock += p_.table(cost::kHostPlan)(prog.steps);
                const int epr = k.shape.experts_per_rank();
                std::vector<std::int64_t> recv(static_cast<std::size_t>(c_.world) * epr);
                for (int s = 0; s < c_.world; ++s) {
                    for (int j = 0; j < epr; ++j) recv[s * epr + j] = counts_[s][r * epr + j];
                }
                k.result.plan = exec::make_ep_plan(k.shape, c_.groups, c_.options, counts_[r], recv, c_.slot_bytes);
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
        if (kind == OpKind::kRoute) {
            exec::ComputeWork w;
            w.entries = static_cast<std::int64_t>(tokens_[k.shape.rank]) * c_.top_k;
            return w;
        }
        return exec::compute_work(k.result.program, op, k.result.plan, tokens_[k.shape.rank]);
    }

    // Enqueues GPU work on rank r's stream at the current host time. It runs
    // slower while the rank's copy engines move remote chunks.
    void gpu(int r, int op, double dur) {
        Rank& k = ranks_[r];
        dur *= c_.compute_scale;
        const double start = k.gpu_free <= k.clock ? k.clock + launch_latency_ : k.gpu_free;
        dur *= 1.0 + (gpu_copy_slowdown_ - 1.0) * fabric_->remote_engines_busy(r, start) / 2.0;
        k.gpu_free = start + dur;
        k.result.gpu_busy += dur;
        k.result.ops[op].gpu_start = start;
        k.result.ops[op].gpu_end = k.gpu_free;
        k.stream_ops.push_back(op);
        const double end = k.gpu_free;
        q_.schedule_at(end, [this, r, op, end] {
            ranks_[r].gpu_finished[op] = 1;
            wake(r, end);
        });
    }

    const SimLayerConfig& c_;
    const std::vector<std::vector<std::int64_t>>& counts_;
    const std::vector<int>& tokens_;
    const CostProfile& p_;
    double host_poll_ = 0, launch_latency_ = 0, gpu_copy_slowdown_ = 1.0;
    EventQueue q_;
    std::unique_ptr<SimFabric> fabric_;
    std::vector<Rank> ranks_;
};

}  // namespace

SimResult simulate_layer(const SimLayerConfig& config, const std::vector<std::vector<std::int64_t>>& counts,
                         const std::vector<int>& tokens, const CostProfile& profile) {
    planner::validate_group_plan(config.groups, config.num_experts);
    Simulation sim(config, counts, tokens, profile);
    return sim.run();
}

}  // namespace nano_nccl::sim
