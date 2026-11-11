#include "transport/peer/rdma_wire.h"

#include "transport/rdma/rdma_endpoint.h"
#include "transport/rdma/rdma_qp.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>

namespace nano_nccl::transport::peer {

namespace {

constexpr std::size_t kHeaderBytes = 64;  // in front of every slot's payload
constexpr std::size_t kSlotAlign = 256;
constexpr std::size_t kLine = 64;
constexpr std::size_t kErrorBytes = 512;
constexpr std::uint64_t kCreditBit = std::uint64_t{1} << 63;  // wr_id of credit writes
constexpr std::uint64_t kHeaderBit = std::uint64_t{1} << 62;  // wr_id of GPUDirect header writes (unsignaled)
constexpr int kPollBatch = 16;
constexpr double kDrainTimeoutSeconds = 30.0;
constexpr double kTeardownWaitSeconds = 1.0;
static_assert(sizeof(ChunkHeader) <= kHeaderBytes);

std::size_t round_up(std::size_t value, std::size_t align) { return (value + align - 1) / align * align; }

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
    mpi_check(MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, comm), "MPI_Allreduce(RdmaWire)");
    if (all_ok == 1) return;
    int nranks = 0;
    MPI_Comm_size(comm, &nranks);
    std::array<char, kErrorBytes> mine{};
    std::snprintf(mine.data(), mine.size(), "%s", local_error.c_str());
    std::vector<std::array<char, kErrorBytes>> all(static_cast<std::size_t>(nranks));
    mpi_check(MPI_Allgather(mine.data(), static_cast<int>(kErrorBytes), MPI_CHAR, all.data(),
                            static_cast<int>(kErrorBytes), MPI_CHAR, comm),
              "MPI_Allgather(RdmaWire errors)");
    for (int r = 0; r < nranks; ++r) {
        if (all[static_cast<std::size_t>(r)][0] != '\0') {
            throw std::runtime_error(std::string("RdmaWire ") + stage + " failed on rank " + std::to_string(r) +
                                     ": " + all[static_cast<std::size_t>(r)].data());
        }
    }
    throw std::runtime_error(std::string("RdmaWire ") + stage + " failed");
}

// What rank r tells peer p about the link pair (r, p).
struct LinkInfo {
    rdma::RdmaPeerInfo qp;        // r's QP for p
    std::uint64_t recv_addr = 0;  // r's slots of link p -> r
    std::uint32_t recv_rkey = 0;
    std::uint32_t credit_rkey = 0;
    std::uint64_t credit_addr = 0;  // r's credit word of link r -> p (p writes it)
    std::uint64_t header_addr = 0;  // GPUDirect: r's host header slots of link p -> r (control rkey)
};

// Whether NIC writes into `device`'s memory need cudaDeviceFlushGPUDirectRDMAWrites
// before other work may read them; throws when they need one the device lacks.
bool gdr_flush_required(int device) {
    int ordering = 0;
    if (cudaDeviceGetAttribute(&ordering, cudaDevAttrGPUDirectRDMAWritesOrdering, device) != cudaSuccess) {
        throw std::runtime_error("RdmaWire: cannot query GPUDirect RDMA write ordering");
    }
    if (ordering >= cudaGPUDirectRDMAWritesOrderingOwner) return false;
    int options = 0;
    if (cudaDeviceGetAttribute(&options, cudaDevAttrGPUDirectRDMAFlushWritesOptions, device) != cudaSuccess ||
        (options & cudaFlushGPUDirectRDMAWritesOptionHost) == 0) {
        throw std::runtime_error("RdmaWire: GPUDirect receive needs a write flush the device does not support");
    }
    return true;
}

std::uint32_t initial_psn(int from, int to) {
    return (static_cast<std::uint32_t>(from) * 7919u + static_cast<std::uint32_t>(to) * 104729u + 1u) & 0xffffffu;
}

ibv_mr* register_mr(ibv_pd* pd, void* addr, std::size_t bytes, const char* what) {
    ibv_mr* mr = ibv_reg_mr(pd, addr, bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (mr == nullptr) {
        throw std::runtime_error(std::string("RdmaWire: ibv_reg_mr(") + what + ") failed: " + std::strerror(errno));
    }
    return mr;
}

std::uint64_t load_acquire(const std::uint64_t* word) { return __atomic_load_n(word, __ATOMIC_ACQUIRE); }

}  // namespace

struct RdmaWire::Verbs {
    rdma::RdmaEndpoint endpoint;
    std::vector<rdma::RdmaQp> qps;  // [peer]; the self entry stays empty
    std::vector<LinkInfo> remote;   // [peer]
    ibv_mr* recv_mr = nullptr;
    ibv_mr* staging_mr = nullptr;
    ibv_mr* control_mr = nullptr;
    // GPUDirect: device registrations; recv_mr / staging_mr point into them.
    std::shared_ptr<ibv_mr> recv_device, staging_device;
    bool inline_credits = false;
    bool inline_headers = false;

    ~Verbs() {
        qps.clear();  // QPs first, then the MRs, then (endpoint) the PD
        if (recv_device) recv_mr = nullptr;
        if (staging_device) staging_mr = nullptr;
        recv_device.reset();
        staging_device.reset();
        for (ibv_mr* mr : {recv_mr, staging_mr, control_mr}) {
            if (mr != nullptr && ibv_dereg_mr(mr) != 0) std::fprintf(stderr, "RdmaWire: ibv_dereg_mr failed\n");
        }
    }
};

RdmaWire::Region::~Region() {
    if (ptr == nullptr) return;
    if (local) copier->free_local(ptr); else copier->free_host(ptr);
}

void RdmaWire::Region::allocate(LocalCopier* owner, std::size_t bytes, bool on_consumer_side) {
    copier = owner;
    local = on_consumer_side;
    ptr = static_cast<std::byte*>(local ? owner->alloc_local(bytes) : owner->alloc_host(bytes));
}

RdmaWire::RdmaWire(MPI_Comm comm, const RdmaWireConfig& config, std::unique_ptr<LocalCopier> copier)
    : slots_(config.slots_per_link),
      slot_bytes_(config.slot_bytes),
      credit_cap_(config.credit_writes_in_flight),
      gdr_(static_cast<bool>(config.register_device)),
      device_(config.device),
      copier_(std::move(copier)) {
    mpi_check(MPI_Comm_rank(comm, &rank_), "MPI_Comm_rank");
    mpi_check(MPI_Comm_size(comm, &nranks_), "MPI_Comm_size");
    const auto n = static_cast<std::size_t>(nranks_);

    std::string error;
    if (slots_ <= 0 || slot_bytes_ == 0 || slot_bytes_ > 0xffffffffu - kHeaderBytes) {
        error = "slots_per_link must be positive and slot_bytes in (0, 4 GiB - 64)";
    } else if (credit_cap_ <= 0) {
        error = "credit_writes_in_flight must be positive";
    } else if (!copier_) {
        error = "no local copier";
    }
    // Every rank must use the same geometry; a mismatch would corrupt slots.
    // GPUDirect changes the message format, so it must match as well.
    const std::array<std::uint64_t, 3> mine{static_cast<std::uint64_t>(config.slots_per_link),
                                            static_cast<std::uint64_t>(config.slot_bytes), gdr_ ? 1u : 0u};
    std::vector<std::array<std::uint64_t, 3>> all(n);
    mpi_check(MPI_Allgather(mine.data(), 3, MPI_UINT64_T, all.data(), 3, MPI_UINT64_T, comm),
              "MPI_Allgather(RdmaWire geometry)");
    for (std::size_t r = 0; r < n && error.empty(); ++r) {
        if (all[r] != all[0]) {
            error = "rank " + std::to_string(r) + " uses slots_per_link=" + std::to_string(all[r][0]) +
                    " slot_bytes=" + std::to_string(all[r][1]) + " gpudirect=" + std::to_string(all[r][2]) +
                    ", rank 0 uses " + std::to_string(all[0][0]) + "/" + std::to_string(all[0][1]) + "/" +
                    std::to_string(all[0][2]);
        }
    }
    agree(comm, error, "configuration");

    slot_stride_ = round_up(kHeaderBytes + slot_bytes_, kSlotAlign);
    block_bytes_ = static_cast<std::size_t>(slots_) * slot_stride_;
    const std::size_t region_bytes = block_bytes_ * n;
    header_offset_ = 2 * n * kLine;
    const std::size_t header_bytes = gdr_ ? 2 * n * static_cast<std::size_t>(slots_) * kHeaderBytes : 0;
    const std::size_t control_bytes = header_offset_ + header_bytes;

    // Stage 1: device, memory, registrations, QPs with their receives posted.
    std::vector<LinkInfo> mine_info(n);
    try {
        verbs_ = std::make_unique<Verbs>();
        Verbs& v = *verbs_;
        v.endpoint = rdma::RdmaEndpoint::create_from_environment();
        const char* name = ibv_get_device_name(v.endpoint.context()->device);
        device_name_ = name != nullptr ? name : "?";

        recv_.allocate(copier_.get(), region_bytes, gdr_);
        staging_.allocate(copier_.get(), region_bytes, gdr_);
        control_.allocate(copier_.get(), control_bytes, false);
        self_.allocate(copier_.get(), block_bytes_, true);
        if (gdr_) {
            gdr_flush_ = gdr_flush_required(device_);
            v.recv_device = config.register_device(v.endpoint.pd(), recv_.ptr, region_bytes, &registration_);
            v.staging_device = config.register_device(v.endpoint.pd(), staging_.ptr, region_bytes, &registration_);
            if (!v.recv_device || !v.staging_device) throw std::runtime_error("RdmaWire: device registration failed");
            v.recv_mr = v.recv_device.get();
            v.staging_mr = v.staging_device.get();
        } else {
            v.recv_mr = register_mr(v.endpoint.pd(), recv_.ptr, region_bytes, "receive slots");
            v.staging_mr = register_mr(v.endpoint.pd(), staging_.ptr, region_bytes, "staging slots");
        }
        v.control_mr = register_mr(v.endpoint.pd(), control_.ptr, control_bytes, "credit words");

        v.qps.resize(n);
        bool inline_ok = true, inline_headers = true;
        for (int p = 0; p < nranks_; ++p) {
            if (p == rank_) continue;
            auto& qp = v.qps[static_cast<std::size_t>(p)];
            // GPUDirect: two writes per chunk (header, payload).
            qp = rdma::RdmaQp::create_init(v.endpoint, (gdr_ ? 2 : 1) * slots_ + credit_cap_, slots_);
            inline_ok = inline_ok && qp.max_inline_data() >= sizeof(std::uint64_t);
            inline_headers = inline_headers && qp.max_inline_data() >= sizeof(ChunkHeader);
            for (int s = 0; s < slots_; ++s) {
                ibv_recv_wr wr{};
                wr.wr_id = static_cast<std::uint64_t>(p);
                ibv_recv_wr* bad = nullptr;
                const int rc = ibv_post_recv(qp.qp(), &wr, &bad);
                if (rc != 0) throw std::runtime_error(std::string("RdmaWire: ibv_post_recv failed: ") + std::strerror(rc));
            }
            LinkInfo& info = mine_info[static_cast<std::size_t>(p)];
            info.qp = qp.local_info();
            info.qp.psn = initial_psn(rank_, p);
            info.qp.port_lid = v.endpoint.port_lid();
            info.qp.gid_index = v.endpoint.gid_index();
            std::memcpy(info.qp.gid, v.endpoint.gid(), sizeof(info.qp.gid));
            info.qp.active_mtu = v.endpoint.active_mtu();
            info.recv_addr = reinterpret_cast<std::uint64_t>(recv_.ptr + static_cast<std::size_t>(p) * block_bytes_);
            info.recv_rkey = v.recv_mr->rkey;
            info.credit_addr = reinterpret_cast<std::uint64_t>(control_.ptr + static_cast<std::size_t>(p) * kLine);
            info.credit_rkey = v.control_mr->rkey;
            if (gdr_) info.header_addr = reinterpret_cast<std::uint64_t>(rx_header(p, 0));
        }
        v.inline_credits = inline_ok;
        v.inline_headers = inline_headers;
    } catch (const std::exception& e) {
        error = e.what();
    }
    agree(comm, error, "setup");

    // Stage 2: exchange link information and connect every QP.
    verbs_->remote.resize(n);
    mpi_check(MPI_Alltoall(mine_info.data(), static_cast<int>(sizeof(LinkInfo)), MPI_BYTE, verbs_->remote.data(),
                           static_cast<int>(sizeof(LinkInfo)), MPI_BYTE, comm),
              "MPI_Alltoall(RdmaWire links)");
    try {
        Verbs& v = *verbs_;
        for (int p = 0; p < nranks_; ++p) {
            if (p == rank_) continue;
            auto& qp = v.qps[static_cast<std::size_t>(p)];
            const LinkInfo& remote = v.remote[static_cast<std::size_t>(p)];
            qp.transition_to_rtr(remote.qp, v.endpoint.gid_index(), v.endpoint.active_mtu());
            qp.transition_to_rts(initial_psn(rank_, p), remote.qp.psn);
        }
    } catch (const std::exception& e) {
        error = e.what();
    }
    agree(comm, error, "connection");
    // No rank writes before every QP is ready to receive.
    mpi_check(MPI_Barrier(comm), "MPI_Barrier(RdmaWire)");

    tx_next_.assign(n, 0);
    tx_pending_.resize(n);
    tx_posted_.assign(n, 0);
    tx_acked_.assign(n, 0);
    rx_arrived_.assign(n, 0);
    rx_released_.assign(n, 0);
    credit_sent_.assign(n, 0);
    credit_inflight_.assign(n, 0);
    rx_headers_.assign(n, std::vector<ChunkHeader>(static_cast<std::size_t>(slots_)));
    self_headers_.assign(static_cast<std::size_t>(slots_), ChunkHeader{});
}

RdmaWire::~RdmaWire() {
    // Local copies still read or write the regions; finish them first.
    try {
        copier_->drain();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "RdmaWire: copier drain failed: %s\n", e.what());
    }
    // Best effort, no new writes: a peer may already be gone (after an error,
    // or when the caller skipped drain() + barrier), and a write to a
    // destroyed QP may never complete. Writes still in flight are harmless
    // once both QPs are destroyed.
    const auto start = std::chrono::steady_clock::now();
    try {
        for (;;) {
            bool idle = true;
            for (int p = 0; p < nranks_ && idle; ++p) {
                const auto i = static_cast<std::size_t>(p);
                idle = tx_posted_[i] == tx_acked_[i] && credit_inflight_[i] == 0;
            }
            if (idle || std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() >
                            kTeardownWaitSeconds) {
                break;
            }
            poll_completions();
        }
    } catch (const std::exception&) {
        // A failed write at teardown: the peer is gone; nothing to recover.
    }
}

std::byte* RdmaWire::recv_slot(int src, std::uint64_t seq) const {
    return recv_.ptr + static_cast<std::size_t>(src) * block_bytes_ +
           static_cast<std::size_t>(seq % static_cast<std::uint64_t>(slots_)) * slot_stride_;
}

std::byte* RdmaWire::staging_slot(int dst, std::uint64_t seq) const {
    return staging_.ptr + static_cast<std::size_t>(dst) * block_bytes_ +
           static_cast<std::size_t>(seq % static_cast<std::uint64_t>(slots_)) * slot_stride_;
}

std::byte* RdmaWire::self_slot(std::uint64_t seq) const {
    return self_.ptr + static_cast<std::size_t>(seq % static_cast<std::uint64_t>(slots_)) * slot_stride_;
}

std::uint64_t* RdmaWire::credit_word(int dst) const {
    return reinterpret_cast<std::uint64_t*>(control_.ptr + static_cast<std::size_t>(dst) * kLine);
}

std::uint64_t* RdmaWire::credit_source(int src) const {
    return reinterpret_cast<std::uint64_t*>(control_.ptr +
                                            (static_cast<std::size_t>(nranks_) + static_cast<std::size_t>(src)) * kLine);
}

ChunkHeader* RdmaWire::tx_header(int dst, std::uint64_t seq) const {
    const std::size_t slot = static_cast<std::size_t>(dst) * static_cast<std::size_t>(slots_) +
                             static_cast<std::size_t>(seq % static_cast<std::uint64_t>(slots_));
    return reinterpret_cast<ChunkHeader*>(control_.ptr + header_offset_ + slot * kHeaderBytes);
}

ChunkHeader* RdmaWire::rx_header(int src, std::uint64_t seq) const {
    const std::size_t slot = (static_cast<std::size_t>(nranks_) + static_cast<std::size_t>(src)) *
                                 static_cast<std::size_t>(slots_) +
                             static_cast<std::size_t>(seq % static_cast<std::uint64_t>(slots_));
    return reinterpret_cast<ChunkHeader*>(control_.ptr + header_offset_ + slot * kHeaderBytes);
}

void RdmaWire::check_link(int peer, const char* what) const {
    if (peer < 0 || peer >= nranks_) {
        throw std::invalid_argument(std::string("RdmaWire::") + what + ": peer " + std::to_string(peer) +
                                    " out of range");
    }
}

PeerWire::OpId RdmaWire::record(LocalCopier::Queue queue, std::uint64_t ticket) {
    const OpId id = next_op_++;
    ops_.emplace(id, Op{queue, ticket});
    return id;
}

std::uint64_t RdmaWire::released(int dst) {
    check_link(dst, "released");
    const auto d = static_cast<std::size_t>(dst);
    if (dst == rank_) return rx_released_[d];
    const std::uint64_t credit = load_acquire(credit_word(dst));
    if (credit > tx_next_[d]) {
        throw std::runtime_error("RdmaWire: link " + std::to_string(rank_) + "->" + std::to_string(dst) +
                                 " released " + std::to_string(credit) + " chunks but only " +
                                 std::to_string(tx_next_[d]) + " were written");
    }
    // A staging slot is free again only once the WRITE that read it completed.
    return std::min(credit, tx_acked_[d]);
}

PeerWire::OpId RdmaWire::write_chunk(int dst, std::uint64_t seq, const ChunkHeader& header, const std::byte* src) {
    check_link(dst, "write_chunk");
    const auto d = static_cast<std::size_t>(dst);
    if (seq != tx_next_[d]) {
        throw std::logic_error("RdmaWire::write_chunk: link " + std::to_string(rank_) + "->" + std::to_string(dst) +
                               " expected seq " + std::to_string(tx_next_[d]) + ", got " + std::to_string(seq));
    }
    if (seq - released(dst) >= static_cast<std::uint64_t>(slots_)) {
        throw std::logic_error("RdmaWire::write_chunk: no free slot on link " + std::to_string(rank_) + "->" +
                               std::to_string(dst));
    }
    if (header.bytes > slot_bytes_ || (header.bytes != 0 && src == nullptr)) {
        throw std::invalid_argument("RdmaWire::write_chunk: bad payload for chunk");
    }
    const auto bytes = static_cast<std::size_t>(header.bytes);
    std::uint64_t ticket;
    if (dst == rank_) {
        self_headers_[static_cast<std::size_t>(seq % static_cast<std::uint64_t>(slots_))] = header;
        ticket = copier_->copy(LocalCopier::kTx, self_slot(seq), src, bytes);
    } else {
        std::byte* slot = staging_slot(dst, seq);
        // GPUDirect: the slot is device memory; the header goes separately.
        std::memcpy(gdr_ ? static_cast<void*>(tx_header(dst, seq)) : static_cast<void*>(slot), &header,
                    sizeof(ChunkHeader));
        ticket = copier_->copy(LocalCopier::kTx, slot + kHeaderBytes, src, bytes);
    }
    tx_pending_[d].push_back({seq, ticket, bytes});
    ++tx_next_[d];
    return record(LocalCopier::kTx, ticket);
}

void RdmaWire::post_data(int dst, const PendingWrite& write) {
    const auto d = static_cast<std::size_t>(dst);
    const LinkInfo& remote = verbs_->remote[d];
    const std::uint64_t slot = write.seq % static_cast<std::uint64_t>(slots_);
    ibv_sge sge{};
    ibv_send_wr wr{};
    wr.wr_id = write.seq;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.imm_data = htonl(static_cast<std::uint32_t>(write.seq));
    wr.wr.rdma.rkey = remote.recv_rkey;
    // Header write ahead of the payload (GPUDirect), same QP, unsignaled: the
    // payload's completion covers it, and RC places it first.
    ibv_sge header_sge{};
    ibv_send_wr header_wr{};
    if (gdr_) {
        sge.addr = reinterpret_cast<std::uint64_t>(staging_slot(dst, write.seq) + kHeaderBytes);
        sge.length = static_cast<std::uint32_t>(write.bytes);
        wr.num_sge = write.bytes == 0 ? 0 : 1;
        wr.wr.rdma.remote_addr = remote.recv_addr + slot * static_cast<std::uint64_t>(slot_stride_) + kHeaderBytes;
        header_sge.addr = reinterpret_cast<std::uint64_t>(tx_header(dst, write.seq));
        header_sge.length = sizeof(ChunkHeader);
        header_sge.lkey = verbs_->control_mr->lkey;
        header_wr.wr_id = kHeaderBit | write.seq;
        header_wr.sg_list = &header_sge;
        header_wr.num_sge = 1;
        header_wr.opcode = IBV_WR_RDMA_WRITE;
        header_wr.send_flags = verbs_->inline_headers ? IBV_SEND_INLINE : 0;
        header_wr.wr.rdma.remote_addr = remote.header_addr + slot * kHeaderBytes;
        header_wr.wr.rdma.rkey = remote.credit_rkey;
        header_wr.next = &wr;
    } else {
        sge.addr = reinterpret_cast<std::uint64_t>(staging_slot(dst, write.seq));
        sge.length = static_cast<std::uint32_t>(kHeaderBytes + write.bytes);
        wr.wr.rdma.remote_addr = remote.recv_addr + slot * static_cast<std::uint64_t>(slot_stride_);
    }
    sge.lkey = verbs_->staging_mr->lkey;
    ibv_send_wr* bad = nullptr;
    const int rc = ibv_post_send(verbs_->qps[d].qp(), gdr_ ? &header_wr : &wr, &bad);
    if (rc != 0) {
        throw std::runtime_error("RdmaWire: posting the data write of link " + std::to_string(rank_) + "->" +
                                 std::to_string(dst) + " failed: " + std::strerror(rc));
    }
    ++tx_posted_[d];
    ++data_writes_;
}

void RdmaWire::post_credit(int src) {
    const auto s = static_cast<std::size_t>(src);
    const std::uint64_t value = rx_released_[s];
    std::uint64_t* source = credit_source(src);
    __atomic_store_n(source, value, __ATOMIC_RELEASE);
    ibv_sge sge{};
    sge.addr = reinterpret_cast<std::uint64_t>(source);
    sge.length = sizeof(std::uint64_t);
    sge.lkey = verbs_->control_mr->lkey;
    ibv_send_wr wr{};
    wr.wr_id = kCreditBit | value;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED | (verbs_->inline_credits ? IBV_SEND_INLINE : 0);
    wr.wr.rdma.remote_addr = verbs_->remote[s].credit_addr;
    wr.wr.rdma.rkey = verbs_->remote[s].credit_rkey;
    ibv_send_wr* bad = nullptr;
    const int rc = ibv_post_send(verbs_->qps[s].qp(), &wr, &bad);
    if (rc != 0) {
        throw std::runtime_error("RdmaWire: posting the credit write of link " + std::to_string(src) + "->" +
                                 std::to_string(rank_) + " failed: " + std::strerror(rc));
    }
    credit_sent_[s] = value;
    ++credit_inflight_[s];
    ++credit_writes_;
}

void RdmaWire::poll_completions() {
    ibv_wc wcs[kPollBatch];
    bool arrivals = false;
    for (int p = 0; p < nranks_; ++p) {
        if (p == rank_) continue;
        const auto i = static_cast<std::size_t>(p);
        auto& qp = verbs_->qps[i];
        const int got = ibv_poll_cq(qp.cq(), kPollBatch, wcs);
        if (got < 0) throw std::runtime_error("RdmaWire: ibv_poll_cq failed for peer " + std::to_string(p));
        for (int k = 0; k < got; ++k) {
            const ibv_wc& wc = wcs[k];
            if (wc.status != IBV_WC_SUCCESS) {
                throw std::runtime_error("RdmaWire: work completion error with peer " + std::to_string(p) + ": " +
                                         ibv_wc_status_str(wc.status) + " (opcode " + std::to_string(wc.opcode) +
                                         ", wr_id " + std::to_string(wc.wr_id) + ")");
            }
            if (wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
                const auto expected = static_cast<std::uint32_t>(rx_arrived_[i]);
                const std::uint32_t seq = ntohl(wc.imm_data);
                if (seq != expected) {
                    throw std::runtime_error("RdmaWire: link " + std::to_string(p) + "->" + std::to_string(rank_) +
                                             " delivered chunk " + std::to_string(seq) + ", expected " +
                                             std::to_string(expected));
                }
                if (rx_arrived_[i] - rx_released_[i] >= static_cast<std::uint64_t>(slots_)) {
                    throw std::runtime_error("RdmaWire: link " + std::to_string(p) + "->" + std::to_string(rank_) +
                                             " delivered more chunks than it has free slots");
                }
                ++rx_arrived_[i];
                arrivals = true;
                ibv_recv_wr wr{};
                wr.wr_id = static_cast<std::uint64_t>(p);
                ibv_recv_wr* bad = nullptr;
                const int rc = ibv_post_recv(qp.qp(), &wr, &bad);
                if (rc != 0) throw std::runtime_error(std::string("RdmaWire: ibv_post_recv failed: ") + std::strerror(rc));
            } else if (wc.opcode == IBV_WC_RDMA_WRITE && (wc.wr_id & kCreditBit) != 0) {
                --credit_inflight_[i];
            } else if (wc.opcode == IBV_WC_RDMA_WRITE && (wc.wr_id & kHeaderBit) != 0) {
                // Unsignaled header writes complete only on error (above).
            } else if (wc.opcode == IBV_WC_RDMA_WRITE) {
                if (wc.wr_id != tx_acked_[i]) {
                    throw std::runtime_error("RdmaWire: data write " + std::to_string(wc.wr_id) + " to peer " +
                                             std::to_string(p) + " completed out of order");
                }
                ++tx_acked_[i];
            } else {
                throw std::runtime_error("RdmaWire: unexpected completion opcode " + std::to_string(wc.opcode));
            }
        }
    }
    // Arrived GPUDirect payloads must be visible to the copy-outs that read
    // them; without native ordering the device needs an explicit flush.
    if (arrivals && gdr_flush_) {
        const cudaError_t status = cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
                                                                      cudaFlushGPUDirectRDMAWritesToOwner);
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("RdmaWire: GPUDirect receive flush failed: ") +
                                     cudaGetErrorString(status));
        }
    }
}

std::uint64_t RdmaWire::post_staged() {
    // Staged chunks, in seq order per link (one tx queue, so in order).
    const std::uint64_t staged = copier_->finished(LocalCopier::kTx);
    for (int dst = 0; dst < nranks_; ++dst) {
        auto& pending = tx_pending_[static_cast<std::size_t>(dst)];
        while (!pending.empty() && pending.front().ticket < staged) {
            if (dst == rank_) {
                rx_arrived_[static_cast<std::size_t>(rank_)] = pending.front().seq + 1;
            } else {
                post_data(dst, pending.front());
            }
            pending.pop_front();
        }
    }
    return staged;
}

void RdmaWire::progress() {
    post_staged();
    poll_completions();
    for (int src = 0; src < nranks_; ++src) {
        const auto s = static_cast<std::size_t>(src);
        if (src != rank_ && rx_released_[s] > credit_sent_[s] && credit_inflight_[s] < credit_cap_) post_credit(src);
    }
}

std::uint64_t RdmaWire::arrived(int src) {
    check_link(src, "arrived");
    return rx_arrived_[static_cast<std::size_t>(src)];
}

const ChunkHeader& RdmaWire::slot_header(int src, std::uint64_t seq) {
    check_link(src, "slot_header");
    const auto s = static_cast<std::size_t>(src);
    if (seq < rx_released_[s] || seq >= rx_arrived_[s]) {
        throw std::logic_error("RdmaWire::slot_header: chunk " + std::to_string(seq) + " of link " +
                               std::to_string(src) + "->" + std::to_string(rank_) + " is not held");
    }
    const auto slot = static_cast<std::size_t>(seq % static_cast<std::uint64_t>(slots_));
    // Snapshot, so the caller checks and uses the same bytes.
    ChunkHeader& copy = rx_headers_[s][slot];
    if (src == rank_) {
        copy = self_headers_[slot];
    } else {
        std::memcpy(&copy, gdr_ ? static_cast<const void*>(rx_header(src, seq)) : recv_slot(src, seq),
                    sizeof(ChunkHeader));
    }
    return copy;
}

PeerWire::OpId RdmaWire::copy_out(int src, std::uint64_t seq, std::byte* dst, std::size_t bytes) {
    check_link(src, "copy_out");
    const auto s = static_cast<std::size_t>(src);
    if (seq < rx_released_[s] || seq >= rx_arrived_[s]) {
        throw std::logic_error("RdmaWire::copy_out: chunk " + std::to_string(seq) + " of link " + std::to_string(src) +
                               "->" + std::to_string(rank_) + " is not held");
    }
    if (bytes == 0 || bytes > slot_bytes_ || dst == nullptr) {
        throw std::invalid_argument("RdmaWire::copy_out: bad destination");
    }
    const std::byte* from = src == rank_ ? self_slot(seq) : recv_slot(src, seq) + kHeaderBytes;
    return record(LocalCopier::kRx, copier_->copy(LocalCopier::kRx, dst, from, bytes));
}

void RdmaWire::release(int src, std::uint64_t upto) {
    check_link(src, "release");
    const auto s = static_cast<std::size_t>(src);
    if (upto < rx_released_[s] || upto > rx_arrived_[s]) {
        throw std::logic_error("RdmaWire::release: link " + std::to_string(src) + "->" + std::to_string(rank_) +
                               " release to " + std::to_string(upto) + " outside [" +
                               std::to_string(rx_released_[s]) + ", " + std::to_string(rx_arrived_[s]) + "]");
    }
    // The credit write goes out from the next progress(), folded with any
    // later release of the link.
    rx_released_[s] = upto;
}

bool RdmaWire::done(OpId id) {
    auto it = ops_.find(id);
    if (it == ops_.end()) throw std::invalid_argument("RdmaWire::done: unknown op " + std::to_string(id));
    // A write is done once its chunk is posted (or, on the self link,
    // published), not merely staged: a caller that sees its exchange finished
    // may stop driving progress(), and the chunk must not be left behind.
    const std::uint64_t finished =
        it->second.queue == LocalCopier::kTx ? post_staged() : copier_->finished(LocalCopier::kRx);
    if (finished <= it->second.ticket) return false;
    ops_.erase(it);
    return true;
}

void RdmaWire::drain() {
    const auto start = std::chrono::steady_clock::now();
    for (;;) {
        progress();
        bool idle = true;
        for (int p = 0; p < nranks_ && idle; ++p) {
            const auto i = static_cast<std::size_t>(p);
            idle = tx_pending_[i].empty() && tx_posted_[i] == tx_acked_[i] && credit_inflight_[i] == 0 &&
                   (p == rank_ || credit_sent_[i] == rx_released_[i]);
        }
        if (idle) break;
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() > kDrainTimeoutSeconds) {
            throw std::runtime_error("RdmaWire::drain: network writes did not complete in time: " + debug_state());
        }
    }
    copier_->drain();
}

std::string RdmaWire::debug_state() const {
    std::string state = "rank " + std::to_string(rank_);
    for (int p = 0; p < nranks_; ++p) {
        const auto i = static_cast<std::size_t>(p);
        state += " [peer " + std::to_string(p) + ": tx written " + std::to_string(tx_next_[i]) + " staged-unposted " +
                 std::to_string(tx_pending_[i].size()) + " acked/posted " + std::to_string(tx_acked_[i]) + "/" +
                 std::to_string(tx_posted_[i]) + " credit-word " +
                 std::to_string(p == rank_ ? rx_released_[i] : load_acquire(credit_word(p))) + "; rx arrived " +
                 std::to_string(rx_arrived_[i]) + " released " + std::to_string(rx_released_[i]) + " credit sent " +
                 std::to_string(credit_sent_[i]) + " in flight " + std::to_string(credit_inflight_[i]) + "]";
    }
    return state;
}

}  // namespace nano_nccl::transport::peer
