#pragma once

#include "exec/exec_program.h"
#include "planner/cost_model.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

// Measured operation costs that drive the simulator (A40 profiles collected
// by benchmarks/ep_profile, or synthetic ones in tests).
//
// File format (text, '#' starts a comment line):
//   nano-ep-profile 1
//   key <exec::ProfileKey::to_string()>
//   meta <free text>                       (any number; kept, not interpreted)
//   scalar <name> <value>
//   table <name> <x-unit> <y-unit>
//   <x> <y>                                (ascending x, at least one row)
//   end
// Tables are interpolated piecewise-linearly, constant below their first
// point and extrapolated linearly from their last two points (a single point
// is a constant). All times are in microseconds.
namespace nano_nccl::sim {

class Table {
public:
    Table() = default;
    explicit Table(std::vector<std::pair<double, double>> points);
    double operator()(double x) const;
    const std::vector<std::pair<double, double>>& points() const { return points_; }
    bool empty() const { return points_.empty(); }

private:
    std::vector<std::pair<double, double>> points_;
};

struct CostProfile {
    exec::ProfileKey key;
    std::vector<std::string> meta;
    std::map<std::string, double> scalars;
    std::map<std::string, std::pair<std::string, std::string>> units;  // table -> (x unit, y unit)
    std::map<std::string, Table> tables;

    // Throws std::out_of_range naming the missing entry.
    double scalar(const std::string& name) const;
    const Table& table(const std::string& name) const;
    double scalar_or(const std::string& name, double fallback) const;
    // "<name>@<rank>" when present, else "<name>".
    const Table& table_for(const std::string& name, int rank) const;
    double scalar_for(const std::string& name, int rank, double fallback) const;

    static CostProfile load(const std::string& path);
    static CostProfile parse(const std::string& text, const std::string& where = "profile");
    std::string serialize() const;
    void save(const std::string& path) const;
};

// Names the simulator reads. GPU durations are pure device time of the op on
// an otherwise idle stream; host costs are wall time of the issuing call.
// Copy entries may also exist per rank as "<name>@<rank>"; the simulator
// prefers those (the profile's GPU i plays rank i).
namespace cost {
// GPU, tables over the given x.
inline constexpr const char* kRoute = "gpu_route_us";             // x: entries (count+place+pack+counts D2H)
// PlanTiming::kFromCounts splits the route: count + counts D2H, then (after
// the plan) send-order upload + place + pack.
inline constexpr const char* kCount = "gpu_count_us";             // x: entries
inline constexpr const char* kPack = "gpu_pack_us";               // x: entries
inline constexpr const char* kPermute = "gpu_permute_us";         // x: rows (one unpack or repack)
inline constexpr const char* kExpertFfn = "gpu_expert_ffn_us";    // x: rows of one expert
inline constexpr const char* kExpertProbe = "gpu_expert_probe_us";  // x: rows of one step
inline constexpr const char* kReduce = "gpu_reduce_us";           // x: tokens
// Copy engines, per chunk, x: bytes. "remote": GPU <-> another rank's host
// slot; "self": device slot of the self link. Measured with every copy engine
// of both GPUs busy, which is the EP steady state.
inline constexpr const char* kTxRemote = "copy_tx_remote_us";
inline constexpr const char* kRxRemote = "copy_rx_remote_us";
inline constexpr const char* kTxRemoteAlone = "copy_tx_remote_us_alone";  // only this engine busy
inline constexpr const char* kRxRemoteAlone = "copy_rx_remote_us_alone";
inline constexpr const char* kTxSelf = "copy_tx_self_us";
// Stall when an engine switches between a remote and a self chunk (scalar).
inline constexpr const char* kTxSwitch = "copy_tx_switch_us";
inline constexpr const char* kRxSwitch = "copy_rx_switch_us";
// In-situ engine turnaround, optional scalars: previous end -> start of a copy
// that was already queued, by kind of the previous and this copy (r = remote,
// s = self), e.g. "copy_tx_turnaround_us_rs" = remote then self. Measured by
// sim_adapter/calibrate_copies.py; when present they replace the switch
// stalls (and add a same-kind turnaround).
inline constexpr const char* kTxTurnaround = "copy_tx_turnaround_us_";
inline constexpr const char* kRxTurnaround = "copy_rx_turnaround_us_";
inline constexpr const char* kRxSelf = "copy_rx_self_us";
// In-situ remote copy cost, optional: x = fraction of the chunk overlapped by
// the same GPU's remote copies in the other direction (0..1), y = us per MiB,
// fitted on real multi-process runs (sim_adapter/calibrate_copies.py over
// nsys reports). When present the contention model uses these instead of
// interpolating between *_alone and the all-busy table; the all-busy table
// then only gives the size dependence.
inline constexpr const char* kTxRemoteInsitu = "copy_tx_remote_insitu_us_per_mib";
inline constexpr const char* kRxRemoteInsitu = "copy_rx_remote_insitu_us_per_mib";
// Host, scalars unless noted.
inline constexpr const char* kHostRouteIssue = "host_route_issue_us";
inline constexpr const char* kHostExpertIssue = "host_expert_issue_us";  // table, x: experts with rows
inline constexpr const char* kHostReduceIssue = "host_reduce_issue_us";
inline constexpr const char* kHostPlan = "host_plan_us";                 // table, x: steps
inline constexpr const char* kHostCountIssue = "host_count_issue_us";    // kFromCounts route
inline constexpr const char* kHostPackIssue = "host_pack_issue_us";
// Scheduler run time, table per scheduler: "host_schedule_us_<name>",
// x: num_experts * groups (kFromCounts plan op, added to kHostPlan).
inline constexpr const char* kHostSchedule = "host_schedule_us_";
inline constexpr const char* kHostExchangeIssue = "host_exchange_issue_us";
inline constexpr const char* kHostPost = "host_post_chunk_us";           // write_chunk call
inline constexpr const char* kHostCopyOut = "host_copy_out_us";          // copy_out call
inline constexpr const char* kHostPoll = "host_poll_us";                 // one progress-loop pass
inline constexpr const char* kLaunchLatency = "gpu_launch_latency_us";   // idle stream: issue -> start
inline constexpr const char* kCopyLatency = "copy_launch_latency_us";    // idle copy engine: issue -> start
inline constexpr const char* kCreditLatency = "credit_latency_us";       // release -> sender sees it
inline constexpr const char* kArrivalLatency = "arrival_latency_us";     // publish -> receiver sees it
// Interference (ratios, optional; 1 when absent).
inline constexpr const char* kCopyTxComputeFactor = "copy_tx_compute_factor";  // tx copy while its GPU computes
inline constexpr const char* kCopyRxComputeFactor = "copy_rx_compute_factor";
// GPU op while both its engines copy remote chunks for all of it (scaled by
// the covered fraction). benchmarks/ep_profile measures it in isolation;
// sim_adapter/calibrate_copies.py --gpu-pair measures it per rank in situ.
inline constexpr const char* kGpuCopySlowdown = "gpu_copy_slowdown";
}  // namespace cost

// What a deployment knows about its fabric, for the planner's cost model:
// per rank and direction, the transfer rate and the cost of each message.
struct FabricRates {
    double tx_us_per_mib = 0;
    double rx_us_per_mib = 0;
    double message_us = 0;        // per message on the sending (chunk) / receiving (credit) side
    std::size_t chunk_bytes = 0;  // payload per message
    double exchange_us = 0;       // fixed per exchange
    // A second fabric between the ranks of a node (NVLink), for nodes of
    // ranks_per_node contiguous ranks; ranks_per_node <= 1: none.
    int ranks_per_node = 0;
    double local_tx_us_per_mib = 0;
    double local_rx_us_per_mib = 0;
    double local_message_us = 0;
};
// A network link of `gbps` in each direction.
FabricRates link_rates(double gbps, double message_us, std::size_t chunk_bytes, double exchange_us);
// `f` plus a local fabric of `gbps` per direction between the ranks of each
// node of `ranks_per_node`.
FabricRates with_local_link(FabricRates f, int ranks_per_node, double gbps, double message_us);
// The profile's host-staged local copies (A40 ShmWire): a remote chunk is a
// tx copy on the sender and an rx copy on the receiver, measured per chunk.
FabricRates copy_rates(const CostProfile& p, std::size_t chunk_bytes);

// planner::CostModel from the profile's expert, permute and expert-issue
// tables (ffn mode; scaled by compute_scale like the simulator's GPU ops)
// and the fabric. Integer ns / ps, rounded.
planner::CostModel planner_cost_model(const CostProfile& p, int hidden, double compute_scale,
                                      const FabricRates& fabric);

// Scheduler run time table for `scheduler` at `world`: "host_schedule_us_<name>_w<world>"
// when the profile has it (measured at that world), else "host_schedule_us_<name>".
const Table& schedule_table(const CostProfile& p, const std::string& scheduler, int world);

}  // namespace nano_nccl::sim
