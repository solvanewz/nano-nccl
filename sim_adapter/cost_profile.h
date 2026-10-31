#pragma once

#include "exec/exec_program.h"

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
inline constexpr const char* kRxSelf = "copy_rx_self_us";
// Host, scalars unless noted.
inline constexpr const char* kHostRouteIssue = "host_route_issue_us";
inline constexpr const char* kHostExpertIssue = "host_expert_issue_us";  // table, x: experts with rows
inline constexpr const char* kHostReduceIssue = "host_reduce_issue_us";
inline constexpr const char* kHostPlan = "host_plan_us";                 // table, x: steps
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
inline constexpr const char* kGpuCopySlowdown = "gpu_copy_slowdown";  // GPU op while both its engines copy
}  // namespace cost

}  // namespace nano_nccl::sim
