#pragma once

#include "ep/ep_plan.h"
#include "planner/group_plan.h"
#include "planner/scheduler.h"
#include "transport/peer/peer_types.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Execution description of one EP MoE layer invocation, shared by the real
// runtime (ep::MoeLayer) and the simulator (sim_adapter). Pure C++.
//
// One rank's invocation is a LayerProgram: a list of Operations, each on one
// issue queue, with dependencies on Completions of earlier operations. The
// runtime turns operations into CUDA work and PeerExchanger calls; the
// simulator turns them into virtual-time work over a simulated wire. Both
// issue with the same rule (ProgramCursor) and wait for the same conditions,
// so what the simulator lets proceed is exactly what the real code lets
// proceed.
namespace nano_nccl::exec {

// How the steps of a GroupPlan are scheduled. Every rank must use the same
// options for the same invocation: they fix the exchange issue order.
struct PipelineOptions {
    // false: each step runs dispatch -> experts -> combine to completion before
    // the next starts (no communication/computation overlap).
    bool overlap = true;
    // With overlap: dispatches of steps i+1 .. i+lookahead may be in flight
    // while step i computes. Exchange issue order: D0 .. D(L), then C0,
    // D(L+1), C1, D(L+2), ...
    int lookahead = 1;
};

// When the invocation's GroupPlan is known.
enum class PlanTiming : int {
    // Before routing (B1 fixed, test plans): kRoute also places and packs the
    // rows in plan order, and kCounts carries to each peer only N(rank, e) for
    // that peer's experts.
    kAhead = 0,
    // From this invocation's counts (B2): kRoute only counts; kCounts
    // all-gathers every rank's full row N(rank, .); kPlan runs the scheduler
    // on the global N; kPack then places and packs. Everything between the
    // counts and the first dispatch is the cost of the dynamic plan.
    kFromCounts,
};
const char* plan_timing_name(PlanTiming timing);
PlanTiming parse_plan_timing(const std::string& name);  // throws std::invalid_argument

// Where an invocation's GroupPlan comes from. Every rank must pass the same
// source for the same invocation.
struct PlanSource {
    PlanTiming timing = PlanTiming::kAhead;
    planner::GroupPlan plan;  // kAhead
    // kFromCounts: the scheduler and its group count (= the program's steps;
    // the scheduler must return exactly that many groups).
    planner::SchedulerKind scheduler = planner::SchedulerKind::kFixed;
    int groups = 0;

    static PlanSource ahead(planner::GroupPlan plan);
    static PlanSource from_counts(planner::SchedulerKind scheduler, int groups);
    int steps() const { return timing == PlanTiming::kAhead ? plan.groups() : groups; }
};

// The source front ends use for scheduler `kind` with `groups` groups:
// kFromCounts when the scheduler reads the counts or `from_counts` is set
// (e.g. fixed with the dynamic program, to measure that program's cost),
// else kAhead with make_schedule's plan.
PlanSource plan_source_for(planner::SchedulerKind kind, int world, int num_experts, int groups, bool from_counts);

// The invocation's plan. kAhead: source.plan (validated); kFromCounts:
// planner::make_schedule on `global_counts` (N(s, e) at s * E + e).
planner::GroupPlan resolve_plan(const PlanSource& source, int world, int num_experts,
                                const std::vector<std::int64_t>& global_counts);

enum class OpKind : int {
    kRoute = 0,  // count kernel, counts copied to the host; kAhead: also place + pack
    kCounts,     // exchange of N(s, e) (see PlanTiming)
    kPlan,       // host: [kFromCounts: run the scheduler,] build the EpPlan, upload its tables
    kDispatch,   // exchange of one step's token rows
    kExpert,     // unpack -> expert FFN -> repack of one step
    kCombine,    // exchange of one step's expert outputs, back to the sources
    kReduce,     // weighted top-k sum
    kRetire,     // host: every exchange finished; buffers may be reused
    kPack,       // kFromCounts only: send order upload + place + pack
};
const char* op_kind_name(OpKind kind);

// Where an operation is issued. Operations of one queue are issued in program
// order; different queues are independent.
enum class Queue : int {
    kHost = 0,  // host work, done when its issue returns
    kExchange,  // PeerExchanger exchanges; the order must match on every rank
    kStream,    // GPU work on the layer's one CUDA stream; runs in issue order
};
const char* queue_name(Queue queue);

// A condition a later operation waits for. The first five are the exchange
// completion events (transport::peer::CompletionEvent, same values) and apply
// only to kExchange operations; kDone applies to the other queues and means
// the work finished (for kStream: the host observed it finish).
enum class Condition : int {
    kSubmitted = 0,
    kSendBufferReusable,
    kPayloadArrived,
    kConsumerReadable,
    kReceiveSlotsReleased,
    kDone,
};
const char* condition_name(Condition condition);
transport::peer::CompletionEvent to_completion_event(Condition condition);  // throws for kDone

struct Completion {
    int op = -1;
    Condition condition = Condition::kDone;
};

struct Operation {
    OpKind kind = OpKind::kRoute;
    Queue queue = Queue::kHost;
    int rank = 0;
    int step = -1;   // kDispatch / kExpert / kCombine: index in execution order
    int group = -1;  // the GroupPlan group of that step
    std::vector<Completion> deps;
};

// One rank's program. `ops` is in a topological order (every dependency
// points to a lower index), and per queue it is the issue order.
struct LayerProgram {
    int rank = 0;
    int steps = 0;
    PipelineOptions options;
    PlanTiming timing = PlanTiming::kAhead;
    std::vector<Operation> ops;
    // Op indices by role; pack is -1 for kAhead.
    int route = -1, counts = -1, plan = -1, pack = -1, reduce = -1, retire = -1;
    std::vector<int> dispatch, expert, combine;  // [steps]
    // Index of each kExchange op among this rank's exchanges, in issue order;
    // -1 for other ops. Equal on every rank: it is the message sequence.
    std::vector<int> exchange_seq;
};

// PlanTiming::kAhead. `group_order` is the GroupPlan's order (one entry per step).
LayerProgram build_layer_program(int rank, const std::vector<int>& group_order, const PipelineOptions& options);
// PlanTiming::kFromCounts with `steps` steps: route -> counts -> plan -> pack,
// and every dispatch waits for the pack. Operation::group is -1 (the plan is
// not known yet).
LayerProgram build_scheduled_program(int rank, int steps, const PipelineOptions& options);
// Either of the above, as `source` asks.
LayerProgram build_program(int rank, const PlanSource& source, const PipelineOptions& options);

// Throws std::logic_error unless the dependency and queue invariants hold.
void validate_program(const LayerProgram& program);

// Issue rule shared by the runtime and the simulator: the next operation of
// each queue is issued once all its dependencies are satisfied. Among ready
// queue heads the lowest op index goes first, so the order is deterministic.
class ProgramCursor {
public:
    explicit ProgramCursor(const LayerProgram& program);

    // Returns the next op to issue, or -1. `satisfied(const Completion&)`
    // reports whether a dependency holds now.
    template <typename Satisfied>
    int next_ready(Satisfied&& satisfied) const {
        int best = -1;
        for (int q = 0; q < kQueues; ++q) {
            const auto& queue = queues_[q];
            if (head_[q] >= queue.size()) continue;
            const int op = queue[head_[q]];
            if (best != -1 && op > best) continue;
            bool ready = true;
            for (const Completion& dep : program_->ops[op].deps) {
                if (!satisfied(dep)) {
                    ready = false;
                    break;
                }
            }
            if (ready) best = op;
        }
        return best;
    }
    // `op` must be the value next_ready() just returned.
    void mark_issued(int op);
    bool issued(int op) const { return issued_[op] != 0; }
    bool all_issued() const { return remaining_ == 0; }

private:
    static constexpr int kQueues = 3;
    const LayerProgram* program_;
    std::vector<int> queues_[kQueues];
    std::size_t head_[kQueues] = {};
    std::vector<char> issued_;
    int remaining_ = 0;
};

// ---- Messages ------------------------------------------------------------

enum class Dtype : int { kBf16 = 0, kU64 };
const char* dtype_name(Dtype dtype);

// One directed message of an exchange.
struct Message {
    int src = 0;
    int dst = 0;
    std::size_t bytes = 0;
    Dtype dtype = Dtype::kBf16;
    std::uint64_t iteration = 0;  // the layer invocation's epoch
    std::uint32_t tag = 0;
    int sequence = 0;             // LayerProgram::exchange_seq of the exchange
};

// Plan of one rank for one invocation: grouping, step order and row layouts,
// plus the buffer capacities they need.
struct BufferNeeds {
    std::int64_t send_rows = 0;    // packed rows; combine results land here too
    std::int64_t recv_rows = 0;    // dispatch receive = expert-major = combine send
    std::size_t counts_bytes = 0;  // per peer, each way
    std::size_t max_message_bytes = 0;
    std::uint32_t max_chunks_per_link = 0;  // over one exchange, for the given slot size
};

struct EpPlan {
    ep::EpShape shape;
    planner::GroupPlan groups;
    PipelineOptions options;
    PlanTiming timing = PlanTiming::kAhead;
    ep::DispatchPlan rows;  // rows.steps follow groups.order
    BufferNeeds buffers;
};

EpPlan make_ep_plan(const ep::EpShape& shape, const planner::GroupPlan& groups, const PipelineOptions& options,
                    const std::vector<std::int64_t>& send_counts, const std::vector<std::int64_t>& recv_counts,
                    std::size_t slot_bytes, PlanTiming timing = PlanTiming::kAhead);

// Tags carry the step so a mismatch names it.
std::uint32_t exchange_tag(const LayerProgram& program, int op);
// 64-bit counts per peer in the counts exchange: experts_per_rank (kAhead)
// or num_experts (kFromCounts: the full row to every peer).
std::size_t counts_per_peer(const ep::EpShape& shape, PlanTiming timing);
// Counts exchange layout: counts_per_peer() 64-bit counts to and from every peer.
transport::peer::ExchangeLayout counts_layout(const ep::EpShape& shape, PlanTiming timing);
// Byte layout of a kDispatch or kCombine op (kCounts: counts_layout()).
const transport::peer::ExchangeLayout& exchange_layout(const LayerProgram& program, int op, const EpPlan& plan);
// The outgoing messages of an exchange op, one per peer (including self and
// zero-byte ones, which still travel as a header-only chunk).
std::vector<Message> outgoing_messages(const LayerProgram& program, int op, const EpPlan& plan,
                                       std::uint64_t iteration);

// ---- Work sizes for cost models -------------------------------------------

struct ComputeWork {
    std::int64_t entries = 0;                // kRoute, kPack: tokens * top_k
    bool route_packs = false;                // kRoute: also places and packs (kAhead)
    std::int64_t tokens = 0;                 // kReduce
    std::int64_t permute_rows = 0;           // kExpert: rows unpacked (= rows repacked)
    std::int64_t permute_segments = 0;       // kExpert
    std::vector<std::int64_t> expert_rows;   // kExpert: rows per local expert of the step
};
// kRoute and kPack need only the shape of `plan` (they run before it exists).
ComputeWork compute_work(const LayerProgram& program, int op, const EpPlan& plan, int tokens);

// ---- Profiles ---------------------------------------------------------------

// Identifies the setting a profile or a measured run belongs to.
struct ProfileKey {
    int world = 0;
    int num_experts = 0;
    int top_k = 0;
    int hidden = 0;
    int ffn = 0;
    int tokens = 0;                       // per rank
    std::string distribution = "zipf";    // token -> expert distribution family
    double skew = 0.0;                    // its parameter
    std::string dtype = "bf16";
    std::string mode = "ffn";             // expert mode: ffn | probe
    std::string transport = "shm";
    int slots_per_link = 0;
    std::size_t slot_bytes = 0;

    std::string to_string() const;  // "world=2 num_experts=16 ..." (space-separated key=value)
    static ProfileKey parse(const std::string& text);  // inverse of to_string; throws on bad input
};

// True when op costs measured under `a` apply to `b`: same per-expert GEMM
// shapes, expert count per rank, dtype, expert mode and transport.
bool cost_compatible(const ProfileKey& a, const ProfileKey& b);

}  // namespace nano_nccl::exec
