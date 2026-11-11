#include "transport/peer/hybrid_wire.h"

#include "transport/peer/local_copier.h"
#include "transport/rdma/rdma_gdr.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include <strings.h>

#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <nvml.h>

namespace nano_nccl::transport::peer {

namespace {

constexpr std::size_t kErrorBytes = 512;
constexpr int kBusIdBytes = 32;

void mpi_check(int status, const char* operation) {
    if (status == MPI_SUCCESS) return;
    char error[MPI_MAX_ERROR_STRING]{};
    int length = 0;
    MPI_Error_string(status, error, &length);
    throw std::runtime_error(std::string(operation) + ": " + std::string(error, static_cast<std::size_t>(length)));
}

// Collective: throws the same error on every rank if any rank failed.
void agree(MPI_Comm comm, const std::string& local_error, const char* stage) {
    int ok = local_error.empty() ? 1 : 0;
    int all_ok = 0;
    mpi_check(MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, comm), "MPI_Allreduce(HybridWire)");
    if (all_ok == 1) return;
    int nranks = 0;
    MPI_Comm_size(comm, &nranks);
    std::array<char, kErrorBytes> mine{};
    std::snprintf(mine.data(), mine.size(), "%s", local_error.c_str());
    std::vector<std::array<char, kErrorBytes>> all(static_cast<std::size_t>(nranks));
    mpi_check(MPI_Allgather(mine.data(), static_cast<int>(kErrorBytes), MPI_CHAR, all.data(),
                            static_cast<int>(kErrorBytes), MPI_CHAR, comm),
              "MPI_Allgather(HybridWire errors)");
    for (int r = 0; r < nranks; ++r) {
        if (all[static_cast<std::size_t>(r)][0] != '\0') {
            throw std::runtime_error(std::string("HybridWire ") + stage + " failed on rank " + std::to_string(r) +
                                     ": " + all[static_cast<std::size_t>(r)].data());
        }
    }
    throw std::runtime_error(std::string("HybridWire ") + stage + " failed");
}

// Node id of every rank: NANO_NCCL_EP_TEST_NODE_OF_RANK, or the lowest global
// rank of each MPI_COMM_TYPE_SHARED group.
std::vector<int> node_ids(MPI_Comm comm, int rank, int nranks) {
    std::vector<int> ids(static_cast<std::size_t>(nranks), 0);
    if (const char* env = std::getenv("NANO_NCCL_EP_TEST_NODE_OF_RANK"); env != nullptr && env[0] != '\0') {
        std::string list = env;
        std::size_t pos = 0;
        for (int r = 0; r < nranks; ++r) {
            const std::size_t comma = list.find(',', pos);
            const std::string item = list.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
            if (item.empty()) throw std::invalid_argument("NANO_NCCL_EP_TEST_NODE_OF_RANK needs one id per rank");
            ids[static_cast<std::size_t>(r)] = std::stoi(item);
            if (comma == std::string::npos && r + 1 < nranks) {
                throw std::invalid_argument("NANO_NCCL_EP_TEST_NODE_OF_RANK needs one id per rank");
            }
            pos = comma + 1;
        }
        return ids;
    }
    MPI_Comm shared = MPI_COMM_NULL;
    mpi_check(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shared), "MPI_Comm_split_type");
    int leader = rank;
    const int status = MPI_Allreduce(&rank, &leader, 1, MPI_INT, MPI_MIN, shared);
    MPI_Comm_free(&shared);
    mpi_check(status, "MPI_Allreduce(HybridWire node leader)");
    mpi_check(MPI_Allgather(&leader, 1, MPI_INT, ids.data(), 1, MPI_INT, comm), "MPI_Allgather(HybridWire nodes)");
    return ids;
}

// Whether NVML reports NVLink peer-to-peer between the GPUs at these PCI bus
// ids (direct bridge or NVSwitch), both ways.
bool nvlink_p2p(const char* a, const char* b, std::string* why) {
    if (nvmlInit_v2() != NVML_SUCCESS) {
        *why = "NVML unavailable";
        return false;
    }
    bool ok = false;
    nvmlDevice_t da{}, db{};
    if (nvmlDeviceGetHandleByPciBusId_v2(a, &da) != NVML_SUCCESS ||
        nvmlDeviceGetHandleByPciBusId_v2(b, &db) != NVML_SUCCESS) {
        *why = std::string("NVML cannot find GPU ") + a + " or " + b;
    } else {
        nvmlGpuP2PStatus_t ab = NVML_P2P_STATUS_UNKNOWN, ba = NVML_P2P_STATUS_UNKNOWN;
        const bool queried = nvmlDeviceGetP2PStatus(da, db, NVML_P2P_CAPS_INDEX_NVLINK, &ab) == NVML_SUCCESS &&
                             nvmlDeviceGetP2PStatus(db, da, NVML_P2P_CAPS_INDEX_NVLINK, &ba) == NVML_SUCCESS;
        ok = queried && ab == NVML_P2P_STATUS_OK && ba == NVML_P2P_STATUS_OK;
        if (!ok) *why = std::string("no NVLink peer-to-peer between GPUs ") + a + " and " + b;
    }
    nvmlShutdown();
    return ok;
}

}  // namespace

HybridWire::HybridWire(MPI_Comm comm, const HybridWireConfig& config)
    : slots_(config.slots_per_link), slot_bytes_(config.slot_bytes) {
    mpi_check(MPI_Comm_rank(comm, &rank_), "MPI_Comm_rank");
    mpi_check(MPI_Comm_size(comm, &nranks_), "MPI_Comm_size");
    const auto n = static_cast<std::size_t>(nranks_);

    std::string error;
    try {
        node_of_ = node_ids(comm, rank_, nranks_);
    } catch (const std::exception& e) {
        error = e.what();
        node_of_.assign(n, rank_);
    }
    agree(comm, error, "node grouping");
    mpi_check(MPI_Comm_split(comm, node_of_[static_cast<std::size_t>(rank_)], rank_, &node_comm_),
              "MPI_Comm_split(HybridWire node)");
    int my_local = 0, local_size = 0;
    MPI_Comm_rank(node_comm_, &my_local);
    MPI_Comm_size(node_comm_, &local_size);
    local_rank_of_.assign(n, 0);
    mpi_check(MPI_Allgather(&my_local, 1, MPI_INT, local_rank_of_.data(), 1, MPI_INT, comm),
              "MPI_Allgather(HybridWire local ranks)");
    const bool multi_node =
        std::any_of(node_of_.begin(), node_of_.end(), [&](int id) { return id != node_of_[0]; });

    // NVLink must hold between every pair of distinct GPUs on the node before
    // any IPC mapping (and so any peer access) exists. Ranks sharing a GPU
    // need no peer access.
    const bool nvlink = config.local == HybridWireConfig::Local::kNvlink;
    if (nvlink) {
        std::array<char, kBusIdBytes> mine{};
        std::vector<std::array<char, kBusIdBytes>> ids(static_cast<std::size_t>(local_size));
        try {
            if (cudaDeviceGetPCIBusId(mine.data(), kBusIdBytes, config.device) != cudaSuccess) {
                throw std::runtime_error("cudaDeviceGetPCIBusId failed");
            }
        } catch (const std::exception& e) {
            error = e.what();
        }
        mpi_check(MPI_Allgather(mine.data(), kBusIdBytes, MPI_CHAR, ids.data(), kBusIdBytes, MPI_CHAR, node_comm_),
                  "MPI_Allgather(HybridWire GPU ids)");
        for (int a = 0; a < local_size && error.empty(); ++a) {
            for (int b = a + 1; b < local_size && error.empty(); ++b) {
                const char* ia = ids[static_cast<std::size_t>(a)].data();
                const char* ib = ids[static_cast<std::size_t>(b)].data();
                if (strcasecmp(ia, ib) == 0) continue;  // same GPU
                std::string why;
                if (!nvlink_p2p(ia, ib, &why)) error = why;
            }
        }
    }
    agree(comm, error, "NVLink check");

    ShmWireConfig local_config;
    local_config.slots_per_link = slots_;
    local_config.slot_bytes = slot_bytes_;
    local_config.device = config.device;
    local_config.debug_copy_delay_us = config.debug_copy_delay_us;
    local_config.device_payload = nvlink;
    try {
        local_ = std::make_unique<ShmWire>(node_comm_, local_config);
    } catch (const std::exception& e) {
        error = e.what();
    }
    agree(comm, error, "local wire");

    std::string remote_name = "none";
    if (multi_node) {
        RdmaWireConfig remote_config;
        remote_config.slots_per_link = slots_;
        remote_config.slot_bytes = slot_bytes_;
        remote_config.device = config.device;
        if (config.gpu_direct) {
            remote_config.register_device = [](ibv_pd* pd, void* addr, std::size_t bytes, std::string* method) {
                auto reg = std::make_shared<rdma::RdmaRegisteredMemory>(
                    rdma::RdmaRegisteredMemory::register_device(pd, addr, bytes));
                if (method != nullptr) *method = reg->registration_method_name();
                return std::shared_ptr<ibv_mr>(reg, reg->mr());
            };
        }
        // RdmaWire agrees on its own errors over `comm`.
        remote_ = std::make_unique<RdmaWire>(comm, remote_config,
                                             make_cuda_copier(config.device, config.debug_copy_delay_us));
        remote_name = std::string(remote_->gpu_direct() ? "rdma/gpudirect(" + remote_->registration() + ")"
                                                        : "rdma/host-staged") +
                      " " + remote_->device_name();
    }
    description_ = std::string("local=") + (nvlink ? "nvlink" : "host-shm") + "(" + std::to_string(local_size) +
                   " ranks) remote=" + remote_name;
}

HybridWire::~HybridWire() {
    local_.reset();
    remote_.reset();
    if (node_comm_ != MPI_COMM_NULL) MPI_Comm_free(&node_comm_);
}

int HybridWire::ranks_per_node() const {
    int size = 0;
    while (size < nranks_ && node_of_[static_cast<std::size_t>(size)] == node_of_[0]) ++size;
    if (size == 0 || nranks_ % size != 0) return 0;
    for (int r = 0; r < nranks_; ++r) {
        if (node_of_[static_cast<std::size_t>(r)] != node_of_[static_cast<std::size_t>(r - r % size)]) return 0;
        if (r % size == 0 && r > 0 && node_of_[static_cast<std::size_t>(r)] == node_of_[static_cast<std::size_t>(r - 1)]) {
            return 0;
        }
    }
    return size;
}

void HybridWire::finish() {
    if (remote_) remote_->drain();
    local_->finish();
}

std::uint64_t HybridWire::released(int dst) {
    return is_local(dst) ? local_->released(local_index(dst)) : remote_->released(dst);
}

// Op ids: the part's id times two, plus 1 for the remote part (never kNoOp).
PeerWire::OpId HybridWire::write_chunk(int dst, std::uint64_t seq, const ChunkHeader& header, const std::byte* src) {
    if (is_local(dst)) return local_->write_chunk(local_index(dst), seq, header, src) * 2;
    return remote_->write_chunk(dst, seq, header, src) * 2 + 1;
}

std::uint64_t HybridWire::arrived(int src) {
    return is_local(src) ? local_->arrived(local_index(src)) : remote_->arrived(src);
}

const ChunkHeader& HybridWire::slot_header(int src, std::uint64_t seq) {
    return is_local(src) ? local_->slot_header(local_index(src), seq) : remote_->slot_header(src, seq);
}

PeerWire::OpId HybridWire::copy_out(int src, std::uint64_t seq, std::byte* dst, std::size_t bytes) {
    if (is_local(src)) return local_->copy_out(local_index(src), seq, dst, bytes) * 2;
    return remote_->copy_out(src, seq, dst, bytes) * 2 + 1;
}

void HybridWire::release(int src, std::uint64_t upto) {
    if (is_local(src)) local_->release(local_index(src), upto);
    else remote_->release(src, upto);
}

bool HybridWire::done(OpId op) {
    if (op % 2 == 1) return remote_->done(op / 2);
    return local_->done(op / 2);
}

void HybridWire::progress() {
    local_->progress();
    if (remote_) remote_->progress();
}

}  // namespace nano_nccl::transport::peer
