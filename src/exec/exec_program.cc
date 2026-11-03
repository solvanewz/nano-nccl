#include "exec/exec_program.h"

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace nano_nccl::exec {

namespace {

constexpr std::uint32_t kDispatchTag = 1u << 16;
constexpr std::uint32_t kCombineTag = 2u << 16;

Queue queue_of(OpKind kind) {
    switch (kind) {
        case OpKind::kPlan:
        case OpKind::kRetire: return Queue::kHost;
        case OpKind::kCounts:
        case OpKind::kDispatch:
        case OpKind::kCombine: return Queue::kExchange;
        case OpKind::kRoute:
        case OpKind::kPack:
        case OpKind::kExpert:
        case OpKind::kReduce: return Queue::kStream;
    }
    throw std::invalid_argument("unknown op kind");
}

bool is_exchange_condition(Condition c) { return c != Condition::kDone; }

}  // namespace

const char* op_kind_name(OpKind kind) {
    switch (kind) {
        case OpKind::kRoute: return "route";
        case OpKind::kCounts: return "counts";
        case OpKind::kPlan: return "plan";
        case OpKind::kDispatch: return "dispatch";
        case OpKind::kExpert: return "expert";
        case OpKind::kCombine: return "combine";
        case OpKind::kReduce: return "reduce";
        case OpKind::kRetire: return "retire";
        case OpKind::kPack: return "pack";
    }
    return "?";
}

const char* plan_timing_name(PlanTiming timing) {
    switch (timing) {
        case PlanTiming::kAhead: return "ahead";
        case PlanTiming::kFromCounts: return "from_counts";
    }
    return "?";
}

PlanTiming parse_plan_timing(const std::string& name) {
    for (PlanTiming t : {PlanTiming::kAhead, PlanTiming::kFromCounts}) {
        if (name == plan_timing_name(t)) return t;
    }
    throw std::invalid_argument("unknown plan timing '" + name + "' (known: ahead, from_counts)");
}

PlanSource PlanSource::ahead(planner::GroupPlan plan) {
    PlanSource s;
    s.timing = PlanTiming::kAhead;
    s.plan = std::move(plan);
    return s;
}

PlanSource PlanSource::from_counts(planner::SchedulerKind scheduler, int groups) {
    if (groups < 1) throw std::invalid_argument("PlanSource::from_counts: groups must be positive");
    PlanSource s;
    s.timing = PlanTiming::kFromCounts;
    s.scheduler = scheduler;
    s.groups = groups;
    return s;
}

PlanSource plan_source_for(planner::SchedulerKind kind, int world, int num_experts, int groups, bool from_counts) {
    if (from_counts || planner::scheduler_reads_counts(kind)) return PlanSource::from_counts(kind, groups);
    return PlanSource::ahead(planner::make_schedule(kind, {world, num_experts, groups, {}}));
}

planner::GroupPlan resolve_plan(const PlanSource& source, const PipelineOptions& options, int world,
                                int num_experts, const std::vector<std::int64_t>& global_counts) {
    if (source.timing == PlanTiming::kAhead) {
        planner::validate_group_plan(source.plan, num_experts);
        return source.plan;
    }
    planner::ScheduleRequest request{world, num_experts, source.groups, {}};
    if (planner::scheduler_reads_counts(source.scheduler)) request.counts = global_counts;
    request.overlap = options.overlap;
    request.lookahead = options.lookahead;
    request.cost = source.cost;
    return planner::make_schedule(source.scheduler, request);
}

const char* queue_name(Queue queue) {
    switch (queue) {
        case Queue::kHost: return "host";
        case Queue::kExchange: return "exchange";
        case Queue::kStream: return "stream";
    }
    return "?";
}

const char* condition_name(Condition condition) {
    switch (condition) {
        case Condition::kSubmitted: return "submitted";
        case Condition::kSendBufferReusable: return "send_buffer_reusable";
        case Condition::kPayloadArrived: return "payload_arrived";
        case Condition::kConsumerReadable: return "consumer_readable";
        case Condition::kReceiveSlotsReleased: return "receive_slots_released";
        case Condition::kDone: return "done";
    }
    return "?";
}

transport::peer::CompletionEvent to_completion_event(Condition condition) {
    if (!is_exchange_condition(condition)) {
        throw std::invalid_argument("to_completion_event: kDone is not an exchange event");
    }
    return static_cast<transport::peer::CompletionEvent>(static_cast<int>(condition));
}

static_assert(static_cast<int>(Condition::kSubmitted) ==
              static_cast<int>(transport::peer::CompletionEvent::kSubmitted));
static_assert(static_cast<int>(Condition::kSendBufferReusable) ==
              static_cast<int>(transport::peer::CompletionEvent::kSendBufferReusable));
static_assert(static_cast<int>(Condition::kPayloadArrived) ==
              static_cast<int>(transport::peer::CompletionEvent::kPayloadArrived));
static_assert(static_cast<int>(Condition::kConsumerReadable) ==
              static_cast<int>(transport::peer::CompletionEvent::kConsumerReadable));
static_assert(static_cast<int>(Condition::kReceiveSlotsReleased) ==
              static_cast<int>(transport::peer::CompletionEvent::kReceiveSlotsReleased));

namespace {

// group_order empty: kFromCounts (groups unknown, -1).
LayerProgram build(int rank, int G, const std::vector<int>& group_order, const PipelineOptions& options,
                   PlanTiming timing) {
    if (G < 1) throw std::invalid_argument("build_layer_program: a plan needs at least one group");
    if (options.lookahead < 0) throw std::invalid_argument("build_layer_program: negative lookahead");
    LayerProgram p;
    p.rank = rank;
    p.steps = G;
    p.options = options;
    p.timing = timing;
    p.dispatch.assign(G, -1);
    p.expert.assign(G, -1);
    p.combine.assign(G, -1);
    auto add = [&](OpKind kind, int step, std::vector<Completion> deps) {
        Operation op;
        op.kind = kind;
        op.queue = queue_of(kind);
        op.rank = rank;
        op.step = step;
        op.group = step >= 0 && !group_order.empty() ? group_order[step] : -1;
        op.deps = std::move(deps);
        p.ops.push_back(std::move(op));
        return static_cast<int>(p.ops.size()) - 1;
    };
    p.route = add(OpKind::kRoute, -1, {});
    p.counts = add(OpKind::kCounts, -1, {{p.route, Condition::kDone}});
    p.plan = add(OpKind::kPlan, -1, {{p.counts, Condition::kConsumerReadable}});
    // The rows a dispatch sends: packed by the route (kAhead) or by the pack.
    if (timing == PlanTiming::kFromCounts) p.pack = add(OpKind::kPack, -1, {{p.plan, Condition::kDone}});
    auto dispatch = [&](int i) {
        std::vector<Completion> deps{{p.plan, Condition::kDone}};
        if (p.pack >= 0) deps.push_back({p.pack, Condition::kDone});
        if (!options.overlap && i > 0) deps.push_back({p.combine[i - 1], Condition::kConsumerReadable});
        p.dispatch[i] = add(OpKind::kDispatch, i, std::move(deps));
    };
    auto expert = [&](int i) { p.expert[i] = add(OpKind::kExpert, i, {{p.dispatch[i], Condition::kConsumerReadable}}); };
    auto combine = [&](int i) { p.combine[i] = add(OpKind::kCombine, i, {{p.expert[i], Condition::kDone}}); };
    if (!options.overlap) {
        for (int i = 0; i < G; ++i) {
            dispatch(i);
            expert(i);
            combine(i);
        }
    } else {
        const int lead = std::min(options.lookahead, G - 1);
        for (int i = 0; i <= lead; ++i) dispatch(i);
        for (int i = 0; i < G; ++i) {
            expert(i);
            combine(i);
            if (i + lead + 1 < G) dispatch(i + lead + 1);
        }
    }
    std::vector<Completion> reduce_deps;
    for (int i = 0; i < G; ++i) reduce_deps.push_back({p.combine[i], Condition::kConsumerReadable});
    p.reduce = add(OpKind::kReduce, -1, std::move(reduce_deps));
    std::vector<Completion> retire_deps{{p.reduce, Condition::kDone}};
    p.exchange_seq.assign(p.ops.size() + 1, -1);
    int seq = 0;
    for (int i = 0; i < static_cast<int>(p.ops.size()); ++i) {
        if (p.ops[i].queue != Queue::kExchange) continue;
        p.exchange_seq[i] = seq++;
        retire_deps.push_back({i, Condition::kSendBufferReusable});
        retire_deps.push_back({i, Condition::kReceiveSlotsReleased});
    }
    p.retire = add(OpKind::kRetire, -1, std::move(retire_deps));
    validate_program(p);
    return p;
}

}  // namespace

LayerProgram build_layer_program(int rank, const std::vector<int>& group_order, const PipelineOptions& options) {
    return build(rank, static_cast<int>(group_order.size()), group_order, options, PlanTiming::kAhead);
}

LayerProgram build_scheduled_program(int rank, int steps, const PipelineOptions& options) {
    return build(rank, steps, {}, options, PlanTiming::kFromCounts);
}

LayerProgram build_program(int rank, const PlanSource& source, const PipelineOptions& options) {
    if (source.timing == PlanTiming::kAhead) return build_layer_program(rank, source.plan.order, options);
    return build_scheduled_program(rank, source.groups, options);
}

void validate_program(const LayerProgram& p) {
    auto fail = [](const std::string& what) { throw std::logic_error("LayerProgram: " + what); };
    const int n = static_cast<int>(p.ops.size());
    if (static_cast<int>(p.exchange_seq.size()) != n) fail("exchange_seq size");
    int seq = 0;
    for (int i = 0; i < n; ++i) {
        const Operation& op = p.ops[i];
        if (op.queue != queue_of(op.kind)) fail(std::string("op ") + std::to_string(i) + " on the wrong queue");
        if (op.rank != p.rank) fail("op " + std::to_string(i) + " has another rank");
        const bool stepped = op.kind == OpKind::kDispatch || op.kind == OpKind::kExpert || op.kind == OpKind::kCombine;
        if (stepped != (op.step >= 0) || op.step >= p.steps) fail("op " + std::to_string(i) + " step");
        if (op.queue == Queue::kExchange) {
            if (p.exchange_seq[i] != seq++) fail("exchange_seq of op " + std::to_string(i));
        } else if (p.exchange_seq[i] != -1) {
            fail("exchange_seq of non-exchange op " + std::to_string(i));
        }
        for (const Completion& dep : op.deps) {
            if (dep.op < 0 || dep.op >= i) fail("op " + std::to_string(i) + " depends on a later op");
            const bool exchange = p.ops[dep.op].queue == Queue::kExchange;
            if (exchange != is_exchange_condition(dep.condition)) {
                fail("op " + std::to_string(i) + " waits for " + condition_name(dep.condition) + " of a " +
                     op_kind_name(p.ops[dep.op].kind) + " op");
            }
        }
    }
    auto check_role = [&](int op, OpKind kind, int step) {
        if (op < 0 || op >= n || p.ops[op].kind != kind || p.ops[op].step != step) fail("role index");
    };
    check_role(p.route, OpKind::kRoute, -1);
    check_role(p.counts, OpKind::kCounts, -1);
    check_role(p.plan, OpKind::kPlan, -1);
    check_role(p.reduce, OpKind::kReduce, -1);
    check_role(p.retire, OpKind::kRetire, -1);
    if (p.timing == PlanTiming::kFromCounts) {
        check_role(p.pack, OpKind::kPack, -1);
        for (int i = 0; i < p.steps; ++i) {
            bool waits = false;
            for (const Completion& dep : p.ops[p.dispatch[i]].deps) waits |= dep.op == p.pack;
            if (!waits) fail("dispatch " + std::to_string(i) + " does not wait for the pack");
        }
    } else if (p.pack != -1) {
        fail("pack op in a kAhead program");
    }
    for (int i = 0; i < n; ++i) {
        if (p.ops[i].kind == OpKind::kPack && i != p.pack) fail("stray pack op");
    }
    if (static_cast<int>(p.dispatch.size()) != p.steps || static_cast<int>(p.expert.size()) != p.steps ||
        static_cast<int>(p.combine.size()) != p.steps) {
        fail("step role sizes");
    }
    for (int i = 0; i < p.steps; ++i) {
        check_role(p.dispatch[i], OpKind::kDispatch, i);
        check_role(p.expert[i], OpKind::kExpert, i);
        check_role(p.combine[i], OpKind::kCombine, i);
    }
}

ProgramCursor::ProgramCursor(const LayerProgram& program) : program_(&program) {
    issued_.assign(program.ops.size(), 0);
    remaining_ = static_cast<int>(program.ops.size());
    for (int i = 0; i < static_cast<int>(program.ops.size()); ++i) {
        queues_[static_cast<int>(program.ops[i].queue)].push_back(i);
    }
}

void ProgramCursor::mark_issued(int op) {
    auto& queue = queues_[static_cast<int>(program_->ops.at(op).queue)];
    std::size_t& head = head_[static_cast<int>(program_->ops[op].queue)];
    if (head >= queue.size() || queue[head] != op) {
        throw std::logic_error("ProgramCursor::mark_issued: op " + std::to_string(op) + " is not its queue's head");
    }
    ++head;
    issued_[op] = 1;
    --remaining_;
}

const char* dtype_name(Dtype dtype) {
    switch (dtype) {
        case Dtype::kBf16: return "bf16";
        case Dtype::kU64: return "u64";
    }
    return "?";
}

EpPlan make_ep_plan(const ep::EpShape& shape, const planner::GroupPlan& groups, const PipelineOptions& options,
                    const std::vector<std::int64_t>& send_counts, const std::vector<std::int64_t>& recv_counts,
                    std::size_t slot_bytes, PlanTiming timing) {
    if (options.lookahead < 0) throw std::invalid_argument("make_ep_plan: negative lookahead");
    if (slot_bytes == 0) throw std::invalid_argument("make_ep_plan: zero slot size");
    EpPlan plan;
    plan.shape = shape;
    plan.groups = groups;
    plan.options = options;
    plan.timing = timing;
    plan.rows = ep::make_dispatch_plan(shape, groups, send_counts, recv_counts);
    BufferNeeds& b = plan.buffers;
    b.send_rows = plan.rows.send_rows;
    b.recv_rows = plan.rows.recv_rows;
    b.counts_bytes = counts_per_peer(shape, timing) * sizeof(std::uint64_t);
    b.max_message_bytes = b.counts_bytes;
    b.max_chunks_per_link = transport::peer::chunk_count(b.counts_bytes, slot_bytes);
    for (const ep::GroupStep& st : plan.rows.steps) {
        for (const auto* layout : {&st.dispatch, &st.combine}) {
            for (std::size_t bytes : layout->send_bytes) {
                b.max_message_bytes = std::max(b.max_message_bytes, bytes);
                b.max_chunks_per_link = std::max(b.max_chunks_per_link, transport::peer::chunk_count(bytes, slot_bytes));
            }
        }
    }
    return plan;
}

std::uint32_t exchange_tag(const LayerProgram& program, int op) {
    const Operation& o = program.ops.at(op);
    switch (o.kind) {
        case OpKind::kCounts: return transport::peer::kCountsTag;
        case OpKind::kDispatch: return kDispatchTag | static_cast<std::uint32_t>(o.step);
        case OpKind::kCombine: return kCombineTag | static_cast<std::uint32_t>(o.step);
        default: throw std::invalid_argument("exchange_tag: op " + std::to_string(op) + " is not an exchange");
    }
}

std::size_t counts_per_peer(const ep::EpShape& shape, PlanTiming timing) {
    return static_cast<std::size_t>(timing == PlanTiming::kAhead ? shape.experts_per_rank() : shape.num_experts);
}

transport::peer::ExchangeLayout counts_layout(const ep::EpShape& shape, PlanTiming timing) {
    return transport::peer::uniform_layout(shape.world, counts_per_peer(shape, timing) * sizeof(std::uint64_t));
}

const transport::peer::ExchangeLayout& exchange_layout(const LayerProgram& program, int op, const EpPlan& plan) {
    const Operation& o = program.ops.at(op);
    if (o.kind == OpKind::kDispatch) return plan.rows.steps.at(o.step).dispatch;
    if (o.kind == OpKind::kCombine) return plan.rows.steps.at(o.step).combine;
    throw std::invalid_argument("exchange_layout: op " + std::to_string(op) + " is not a dispatch or combine");
}

std::vector<Message> outgoing_messages(const LayerProgram& program, int op, const EpPlan& plan,
                                       std::uint64_t iteration) {
    const Operation& o = program.ops.at(op);
    const bool counts = o.kind == OpKind::kCounts;
    const transport::peer::ExchangeLayout layout =
        counts ? counts_layout(plan.shape, program.timing) : exchange_layout(program, op, plan);
    std::vector<Message> out;
    for (int peer = 0; peer < plan.shape.world; ++peer) {
        Message m;
        m.src = program.rank;
        m.dst = peer;
        m.bytes = layout.send_bytes[peer];
        m.dtype = counts ? Dtype::kU64 : Dtype::kBf16;
        m.iteration = iteration;
        m.tag = exchange_tag(program, op);
        m.sequence = program.exchange_seq[op];
        out.push_back(m);
    }
    return out;
}

ComputeWork compute_work(const LayerProgram& program, int op, const EpPlan& plan, int tokens) {
    const Operation& o = program.ops.at(op);
    ComputeWork w;
    switch (o.kind) {
        case OpKind::kRoute:
            w.entries = static_cast<std::int64_t>(tokens) * plan.shape.top_k;
            w.route_packs = program.timing == PlanTiming::kAhead;
            break;
        case OpKind::kPack: w.entries = static_cast<std::int64_t>(tokens) * plan.shape.top_k; break;
        case OpKind::kReduce: w.tokens = tokens; break;
        case OpKind::kExpert: {
            const ep::GroupStep& st = plan.rows.steps.at(o.step);
            for (const ep::RowSegment& seg : st.unpack) w.permute_rows += seg.rows;
            w.permute_segments = static_cast<std::int64_t>(st.unpack.size());
            for (const ep::ExpertSegment& seg : st.experts) w.expert_rows.push_back(seg.rows);
            break;
        }
        default: throw std::invalid_argument("compute_work: op " + std::to_string(op) + " is not GPU work");
    }
    return w;
}

std::string ProfileKey::to_string() const {
    std::ostringstream s;
    s << "world=" << world << " num_experts=" << num_experts << " top_k=" << top_k << " hidden=" << hidden
      << " ffn=" << ffn << " tokens=" << tokens << " distribution=" << distribution << " skew=" << skew
      << " dtype=" << dtype << " mode=" << mode << " transport=" << transport << " slots_per_link=" << slots_per_link
      << " slot_bytes=" << slot_bytes;
    return s.str();
}

ProfileKey ProfileKey::parse(const std::string& text) {
    ProfileKey k;
    std::istringstream in(text);
    std::string item;
    int seen = 0;
    while (in >> item) {
        const auto eq = item.find('=');
        if (eq == std::string::npos) throw std::invalid_argument("ProfileKey: '" + item + "' is not key=value");
        const std::string name = item.substr(0, eq), value = item.substr(eq + 1);
        auto num = [&](auto& field) {
            char* end = nullptr;
            const double v = std::strtod(value.c_str(), &end);
            if (value.empty() || *end != '\0') throw std::invalid_argument("ProfileKey: bad number in '" + item + "'");
            field = static_cast<std::remove_reference_t<decltype(field)>>(v);
        };
        if (name == "world") num(k.world);
        else if (name == "num_experts") num(k.num_experts);
        else if (name == "top_k") num(k.top_k);
        else if (name == "hidden") num(k.hidden);
        else if (name == "ffn") num(k.ffn);
        else if (name == "tokens") num(k.tokens);
        else if (name == "distribution") k.distribution = value;
        else if (name == "skew") num(k.skew);
        else if (name == "dtype") k.dtype = value;
        else if (name == "mode") k.mode = value;
        else if (name == "transport") k.transport = value;
        else if (name == "slots_per_link") num(k.slots_per_link);
        else if (name == "slot_bytes") num(k.slot_bytes);
        else throw std::invalid_argument("ProfileKey: unknown field '" + name + "'");
        ++seen;
    }
    if (seen != 13) throw std::invalid_argument("ProfileKey: expected 13 fields, got " + std::to_string(seen));
    return k;
}

bool cost_compatible(const ProfileKey& a, const ProfileKey& b) {
    return a.hidden == b.hidden && a.ffn == b.ffn && a.world > 0 && b.world > 0 &&
           a.num_experts / a.world == b.num_experts / b.world && a.dtype == b.dtype && a.mode == b.mode;
}

}  // namespace nano_nccl::exec
