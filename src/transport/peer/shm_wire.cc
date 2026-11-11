#include "transport/peer/shm_wire.h"

#include "core/buffer.h"
#include "core/numa.h"

#include <array>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace nano_nccl::transport::peer {

namespace {

constexpr std::size_t kLine = 64;
constexpr std::size_t kPage = 4096;
constexpr std::size_t kSlotAlign = 256;
constexpr std::size_t kNameBytes = 128;
constexpr std::size_t kErrorBytes = 512;
static_assert(sizeof(ChunkHeader) <= kLine);

std::size_t round_up(std::size_t value, std::size_t align) {
    return (value + align - 1) / align * align;
}

std::uint64_t load_acquire(const std::uint64_t* word) {
    return __atomic_load_n(word, __ATOMIC_ACQUIRE);
}

void store_release(std::uint64_t* word, std::uint64_t value) {
    __atomic_store_n(word, value, __ATOMIC_RELEASE);
}

void mpi_check(int status, const char* operation) {
    if (status == MPI_SUCCESS) return;
    char error[MPI_MAX_ERROR_STRING]{};
    int length = 0;
    MPI_Error_string(status, error, &length);
    throw std::runtime_error(std::string(operation) + ": " +
                             std::string(error, static_cast<std::size_t>(length)));
}

// Collective: throws the same error on every rank if any rank failed.
void agree(MPI_Comm comm, const std::string& local_error, const char* stage) {
    int ok = local_error.empty() ? 1 : 0;
    int all_ok = 0;
    mpi_check(MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, comm), "MPI_Allreduce(ShmWire)");
    if (all_ok == 1) return;
    int nranks = 0;
    MPI_Comm_size(comm, &nranks);
    std::array<char, kErrorBytes> mine{};
    std::snprintf(mine.data(), mine.size(), "%s", local_error.c_str());
    std::vector<std::array<char, kErrorBytes>> all(static_cast<std::size_t>(nranks));
    mpi_check(MPI_Allgather(mine.data(), static_cast<int>(kErrorBytes), MPI_CHAR, all.data(),
                            static_cast<int>(kErrorBytes), MPI_CHAR, comm),
              "MPI_Allgather(ShmWire errors)");
    for (int r = 0; r < nranks; ++r) {
        if (all[static_cast<std::size_t>(r)][0] != '\0') {
            throw std::runtime_error(std::string("ShmWire ") + stage + " failed on rank " +
                                     std::to_string(r) + ": " +
                                     all[static_cast<std::size_t>(r)].data());
        }
    }
    throw std::runtime_error(std::string("ShmWire ") + stage + " failed");
}

void CUDART_CB sleep_on_stream(void* micros) { usleep(static_cast<useconds_t>(*static_cast<int*>(micros))); }

bool event_done(cudaEvent_t event) {
    const cudaError_t status = cudaEventQuery(event);
    if (status == cudaSuccess) return true;
    if (status == cudaErrorNotReady) return false;
    CUDA_CHECK_THROW(status);
    return false;
}

}  // namespace

struct ShmWire::Region {
    std::string name;  // set while this rank still has to unlink it
    void* base = nullptr;
    std::size_t bytes = 0;
    int fd = -1;
    bool registered = false;

    ~Region() {
        if (registered && cudaHostUnregister(base) != cudaSuccess) {
            std::fprintf(stderr, "ShmWire: cudaHostUnregister failed during cleanup\n");
        }
        if (base != nullptr) munmap(base, bytes);
        if (fd >= 0) close(fd);
        if (!name.empty()) shm_unlink(name.c_str());
    }
};

ShmWire::ShmWire(MPI_Comm comm, const ShmWireConfig& config)
    : slots_(config.slots_per_link),
      slot_bytes_(config.slot_bytes),
      device_(config.device),
      debug_delay_us_(config.debug_copy_delay_us),
      device_payload_(config.device_payload) {
    mpi_check(MPI_Comm_rank(comm, &rank_), "MPI_Comm_rank");
    mpi_check(MPI_Comm_size(comm, &nranks_), "MPI_Comm_size");
    const auto n = static_cast<std::size_t>(nranks_);

    std::string error;
    if (slots_ <= 0 || slot_bytes_ == 0) error = "slots_per_link and slot_bytes must be positive";

    // Every rank must use the same geometry; a mismatch would corrupt slots.
    const std::array<std::uint64_t, 3> mine{static_cast<std::uint64_t>(config.slots_per_link),
                                            static_cast<std::uint64_t>(config.slot_bytes),
                                            config.device_payload ? 1u : 0u};
    std::vector<std::array<std::uint64_t, 3>> all(n);
    mpi_check(MPI_Allgather(mine.data(), 3, MPI_UINT64_T, all.data(), 3, MPI_UINT64_T, comm),
              "MPI_Allgather(ShmWire geometry)");
    for (std::size_t r = 0; r < n && error.empty(); ++r) {
        if (all[r] != all[0]) {
            error = "rank " + std::to_string(r) + " uses slots_per_link=" + std::to_string(all[r][0]) +
                    " slot_bytes=" + std::to_string(all[r][1]) + " device_payload=" + std::to_string(all[r][2]) +
                    ", rank 0 uses " + std::to_string(all[0][0]) + "/" + std::to_string(all[0][1]) + "/" +
                    std::to_string(all[0][2]);
        }
    }

    MPI_Comm node = MPI_COMM_NULL;
    mpi_check(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node),
              "MPI_Comm_split_type");
    int node_size = 0;
    MPI_Comm_size(node, &node_size);
    MPI_Comm_free(&node);
    if (error.empty() && node_size != nranks_) {
        error = "all ranks must share one host (" + std::to_string(node_size) + " of " +
                std::to_string(nranks_) + " are local)";
    }
    agree(comm, error, "configuration");

    payload_offset_ = round_up(2 * kLine + static_cast<std::size_t>(slots_) * kLine, kPage);
    slot_stride_ = round_up(slot_bytes_, kSlotAlign);
    // device_payload: the host region holds only counters and headers.
    link_bytes_ = round_up(payload_offset_ + (device_payload_ ? 0 : static_cast<std::size_t>(slots_) * slot_stride_),
                           kPage);
    device_link_bytes_ = static_cast<std::size_t>(slots_) * slot_stride_;
    region_bytes_ = link_bytes_ * n;
    regions_.resize(n);

    // Stage 1: create and zero this rank's receive region.
    static std::atomic<unsigned long> sequence{0};
    std::array<char, kNameBytes> my_name{};
    try {
        CUDA_CHECK_THROW(cudaSetDevice(device_));
        auto region = std::make_unique<Region>();
        region->bytes = region_bytes_;
        for (int attempt = 0; attempt < 32 && region->fd < 0; ++attempt) {
            const std::string name = "/nano_nccl_peer_" + std::to_string(getpid()) + "_" +
                                     std::to_string(rank_) + "_" +
                                     std::to_string(sequence.fetch_add(1));
            region->fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, S_IRUSR | S_IWUSR);
            if (region->fd >= 0) {
                region->name = name;
            } else if (errno != EEXIST) {
                throw std::system_error(errno, std::generic_category(), "shm_open(create)");
            }
        }
        if (region->fd < 0) throw std::runtime_error("no unique shared memory name");
        if (ftruncate(region->fd, static_cast<off_t>(region_bytes_)) != 0) {
            throw std::system_error(errno, std::generic_category(), "ftruncate");
        }
        region->base = mmap(nullptr, region_bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, region->fd, 0);
        if (region->base == MAP_FAILED) {
            region->base = nullptr;
            throw std::system_error(errno, std::generic_category(), "mmap(own region)");
        }
        // First touch on the receiving GPU's NUMA node: its copy-outs read
        // the slots locally, and the senders' writes cross sockets as posted
        // writes, which tolerate the distance better than reads.
        core::numa_set_prefer(core::gpu_numa_node(device_));
        std::memset(region->base, 0, region_bytes_);
        core::numa_set_prefer(-1);
        std::snprintf(my_name.data(), my_name.size(), "%s", region->name.c_str());
        regions_[static_cast<std::size_t>(rank_)] = std::move(region);
    } catch (const std::exception& e) {
        error = e.what();
    }
    agree(comm, error, "region creation");

    // Stage 2: map every peer's region and register all of them with CUDA.
    std::vector<std::array<char, kNameBytes>> names(n);
    mpi_check(MPI_Allgather(my_name.data(), static_cast<int>(kNameBytes), MPI_CHAR, names.data(),
                            static_cast<int>(kNameBytes), MPI_CHAR, comm),
              "MPI_Allgather(ShmWire names)");
    try {
        for (int owner = 0; owner < nranks_; ++owner) {
            if (owner == rank_) continue;
            auto region = std::make_unique<Region>();
            region->bytes = region_bytes_;
            region->fd = shm_open(names[static_cast<std::size_t>(owner)].data(), O_RDWR | O_CLOEXEC, 0);
            if (region->fd < 0) {
                throw std::system_error(errno, std::generic_category(),
                                        "shm_open(region of rank " + std::to_string(owner) + ")");
            }
            region->base = mmap(nullptr, region_bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, region->fd, 0);
            if (region->base == MAP_FAILED) {
                region->base = nullptr;
                throw std::system_error(errno, std::generic_category(), "mmap(peer region)");
            }
            regions_[static_cast<std::size_t>(owner)] = std::move(region);
        }
        for (auto& region : regions_) {
            CUDA_CHECK_THROW(cudaHostRegister(region->base, region->bytes, cudaHostRegisterPortable));
            region->registered = true;
        }
    } catch (const std::exception& e) {
        error = e.what();
    }
    agree(comm, error, "region mapping");

    // Every rank has opened every region; the names are no longer needed.
    Region& own = *regions_[static_cast<std::size_t>(rank_)];
    shm_unlink(own.name.c_str());
    own.name.clear();

    // Stage 3 (device_payload): export this rank's device slots, map the peers'.
    if (device_payload_) {
        peer_slots_.assign(n, nullptr);
        cudaIpcMemHandle_t handle{};
        try {
            void* slots = nullptr;
            CUDA_CHECK_THROW(cudaMalloc(&slots, device_link_bytes_ * n));
            device_slots_ = static_cast<std::byte*>(slots);
            CUDA_CHECK_THROW(cudaMemset(device_slots_, 0, device_link_bytes_ * n));
            CUDA_CHECK_THROW(cudaIpcGetMemHandle(&handle, device_slots_));
        } catch (const std::exception& e) {
            error = e.what();
        }
        agree(comm, error, "device slot allocation");
        std::vector<cudaIpcMemHandle_t> handles(n);
        mpi_check(MPI_Allgather(&handle, sizeof(handle), MPI_BYTE, handles.data(), sizeof(handle), MPI_BYTE, comm),
                  "MPI_Allgather(ShmWire IPC handles)");
        try {
            for (int owner = 0; owner < nranks_; ++owner) {
                if (owner == rank_) {
                    peer_slots_[static_cast<std::size_t>(owner)] = device_slots_;
                    continue;
                }
                void* mapped = nullptr;
                CUDA_CHECK_THROW(cudaIpcOpenMemHandle(&mapped, handles[static_cast<std::size_t>(owner)],
                                                      cudaIpcMemLazyEnablePeerAccess));
                peer_slots_[static_cast<std::size_t>(owner)] = static_cast<std::byte*>(mapped);
            }
        } catch (const std::exception& e) {
            error = e.what();
        }
        agree(comm, error, "device slot mapping");
    }

    void* self_payload = nullptr;
    CUDA_CHECK_THROW(cudaMalloc(&self_payload, static_cast<std::size_t>(slots_) * slot_stride_));
    self_payload_ = static_cast<std::byte*>(self_payload);
    cudaError_t status = cudaStreamCreateWithFlags(&tx_stream_, cudaStreamNonBlocking);
    if (status == cudaSuccess) status = cudaStreamCreateWithFlags(&rx_stream_, cudaStreamNonBlocking);
    if (status != cudaSuccess) {
        if (tx_stream_ != nullptr) cudaStreamDestroy(tx_stream_);
        cudaFree(self_payload_);
        CUDA_CHECK_THROW(status);
    }

    tx_next_.assign(n, 0);
    tx_pending_.resize(n);
    rx_arrived_.assign(n, 0);
    rx_released_.assign(n, 0);
    rx_headers_.assign(n, std::vector<ChunkHeader>(static_cast<std::size_t>(slots_)));
}

void ShmWire::finish() {
    CUDA_CHECK_THROW(cudaSetDevice(device_));
    CUDA_CHECK_THROW(cudaStreamSynchronize(tx_stream_));
    CUDA_CHECK_THROW(cudaStreamSynchronize(rx_stream_));
    for (int owner = 0; owner < static_cast<int>(peer_slots_.size()); ++owner) {
        std::byte*& mapped = peer_slots_[static_cast<std::size_t>(owner)];
        if (owner != rank_ && mapped != nullptr) CUDA_CHECK_THROW(cudaIpcCloseMemHandle(mapped));
        mapped = nullptr;
    }
}

ShmWire::~ShmWire() {
    // Copies in flight still target the mapped regions; finish them first.
    cudaSetDevice(device_);
    cudaStreamSynchronize(tx_stream_);
    cudaStreamSynchronize(rx_stream_);
    for (int owner = 0; owner < static_cast<int>(peer_slots_.size()); ++owner) {
        std::byte* mapped = peer_slots_[static_cast<std::size_t>(owner)];
        if (owner != rank_ && mapped != nullptr && cudaIpcCloseMemHandle(mapped) != cudaSuccess) {
            std::fprintf(stderr, "ShmWire: cudaIpcCloseMemHandle failed during cleanup\n");
        }
    }
    if (device_slots_ != nullptr) cudaFree(device_slots_);
    for (auto& [id, op] : ops_) cudaEventDestroy(op.event);
    for (cudaEvent_t event : free_events_) cudaEventDestroy(event);
    cudaStreamDestroy(tx_stream_);
    cudaStreamDestroy(rx_stream_);
    cudaFree(self_payload_);
    regions_.clear();
}

std::byte* ShmWire::link_block(int owner, int src) const {
    return static_cast<std::byte*>(regions_[static_cast<std::size_t>(owner)]->base) +
           static_cast<std::size_t>(src) * link_bytes_;
}

std::uint64_t* ShmWire::arrived_word(int owner, int src) const {
    return reinterpret_cast<std::uint64_t*>(link_block(owner, src));
}

std::uint64_t* ShmWire::released_word(int owner, int src) const {
    return reinterpret_cast<std::uint64_t*>(link_block(owner, src) + kLine);
}

ChunkHeader* ShmWire::header_slot(int owner, int src, std::uint64_t seq) const {
    const auto slot = static_cast<std::size_t>(seq % static_cast<std::uint64_t>(slots_));
    return reinterpret_cast<ChunkHeader*>(link_block(owner, src) + 2 * kLine + slot * kLine);
}

std::byte* ShmWire::payload_slot(int owner, int src, std::uint64_t seq) const {
    const auto slot = static_cast<std::size_t>(seq % static_cast<std::uint64_t>(slots_));
    if (owner == rank_ && src == rank_) return self_payload_ + slot * slot_stride_;
    if (device_payload_) {
        std::byte* base = peer_slots_[static_cast<std::size_t>(owner)];
        if (base == nullptr) throw std::logic_error("ShmWire: device slots used after finish()");
        return base + static_cast<std::size_t>(src) * device_link_bytes_ + slot * slot_stride_;
    }
    return link_block(owner, src) + payload_offset_ + slot * slot_stride_;
}

void ShmWire::check_link(int peer, const char* what) const {
    if (peer < 0 || peer >= nranks_) {
        throw std::invalid_argument(std::string("ShmWire::") + what + ": peer " +
                                    std::to_string(peer) + " out of range");
    }
}

void ShmWire::debug_delay(cudaStream_t stream) {
    if (debug_delay_us_ > 0) CUDA_CHECK_THROW(cudaLaunchHostFunc(stream, sleep_on_stream, &debug_delay_us_));
}

cudaEvent_t ShmWire::take_event() {
    if (!free_events_.empty()) {
        cudaEvent_t event = free_events_.back();
        free_events_.pop_back();
        return event;
    }
    cudaEvent_t event = nullptr;
    CUDA_CHECK_THROW(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    return event;
}

void ShmWire::recycle_event(cudaEvent_t event) { free_events_.push_back(event); }

PeerWire::OpId ShmWire::record(bool is_write, cudaStream_t stream) {
    cudaEvent_t event = take_event();
    const cudaError_t status = cudaEventRecord(event, stream);
    if (status != cudaSuccess) {
        recycle_event(event);
        CUDA_CHECK_THROW(status);
    }
    const OpId id = next_op_++;
    ops_.emplace(id, Op{event, is_write, false});
    return id;
}

std::uint64_t ShmWire::released(int dst) {
    check_link(dst, "released");
    const std::uint64_t value = load_acquire(released_word(dst, rank_));
    if (value > tx_next_[static_cast<std::size_t>(dst)]) {
        throw std::runtime_error("ShmWire: link " + std::to_string(rank_) + "->" + std::to_string(dst) +
                                 " released " + std::to_string(value) + " chunks but only " +
                                 std::to_string(tx_next_[static_cast<std::size_t>(dst)]) +
                                 " were written");
    }
    return value;
}

PeerWire::OpId ShmWire::write_chunk(int dst, std::uint64_t seq, const ChunkHeader& header,
                                    const std::byte* src) {
    check_link(dst, "write_chunk");
    const auto d = static_cast<std::size_t>(dst);
    if (seq != tx_next_[d]) {
        throw std::logic_error("ShmWire::write_chunk: link " + std::to_string(rank_) + "->" +
                               std::to_string(dst) + " expected seq " + std::to_string(tx_next_[d]) +
                               ", got " + std::to_string(seq));
    }
    if (seq - released(dst) >= static_cast<std::uint64_t>(slots_)) {
        throw std::logic_error("ShmWire::write_chunk: no free slot on link " + std::to_string(rank_) +
                               "->" + std::to_string(dst));
    }
    if (header.bytes > slot_bytes_ || (header.bytes != 0 && src == nullptr)) {
        throw std::invalid_argument("ShmWire::write_chunk: bad payload for chunk");
    }
    // The header is read only after the arrival counter covers this chunk,
    // and that counter is published after the payload copy finishes.
    std::memcpy(header_slot(dst, rank_, seq), &header, sizeof(ChunkHeader));
    CUDA_CHECK_THROW(cudaSetDevice(device_));
    debug_delay(tx_stream_);
    if (header.bytes != 0) {
        CUDA_CHECK_THROW(cudaMemcpyAsync(payload_slot(dst, rank_, seq), src, header.bytes,
                                         cudaMemcpyDefault, tx_stream_));
    }
    const OpId op = record(true, tx_stream_);
    tx_pending_[d].push_back({seq, op});
    ++tx_next_[d];
    return op;
}

void ShmWire::progress() {
    // All writes share one stream, so they finish in order on every link.
    for (int dst = 0; dst < nranks_; ++dst) {
        auto& pending = tx_pending_[static_cast<std::size_t>(dst)];
        while (!pending.empty()) {
            Op& op = ops_.at(pending.front().op);
            if (!event_done(op.event)) break;
            store_release(arrived_word(dst, rank_), pending.front().seq + 1);
            op.complete = true;
            pending.pop_front();
        }
    }
}

std::uint64_t ShmWire::arrived(int src) {
    check_link(src, "arrived");
    const auto s = static_cast<std::size_t>(src);
    const std::uint64_t value = load_acquire(arrived_word(rank_, src));
    if (value < rx_arrived_[s] || value - rx_released_[s] > static_cast<std::uint64_t>(slots_)) {
        throw std::runtime_error("ShmWire: link " + std::to_string(src) + "->" + std::to_string(rank_) +
                                 " arrival counter " + std::to_string(value) + " is inconsistent (seen " +
                                 std::to_string(rx_arrived_[s]) + ", released " +
                                 std::to_string(rx_released_[s]) + ")");
    }
    rx_arrived_[s] = value;
    return value;
}

const ChunkHeader& ShmWire::slot_header(int src, std::uint64_t seq) {
    check_link(src, "slot_header");
    const auto s = static_cast<std::size_t>(src);
    if (seq < rx_released_[s] || seq >= rx_arrived_[s]) {
        throw std::logic_error("ShmWire::slot_header: chunk " + std::to_string(seq) + " of link " +
                               std::to_string(src) + "->" + std::to_string(rank_) + " is not held");
    }
    // Snapshot, so the caller checks and uses the same bytes.
    ChunkHeader& copy = rx_headers_[s][static_cast<std::size_t>(seq % static_cast<std::uint64_t>(slots_))];
    std::memcpy(&copy, header_slot(rank_, src, seq), sizeof(ChunkHeader));
    return copy;
}

PeerWire::OpId ShmWire::copy_out(int src, std::uint64_t seq, std::byte* dst, std::size_t bytes) {
    check_link(src, "copy_out");
    const auto s = static_cast<std::size_t>(src);
    if (seq < rx_released_[s] || seq >= rx_arrived_[s]) {
        throw std::logic_error("ShmWire::copy_out: chunk " + std::to_string(seq) + " of link " +
                               std::to_string(src) + "->" + std::to_string(rank_) + " is not held");
    }
    if (bytes == 0 || bytes > slot_bytes_ || dst == nullptr) {
        throw std::invalid_argument("ShmWire::copy_out: bad destination");
    }
    CUDA_CHECK_THROW(cudaSetDevice(device_));
    debug_delay(rx_stream_);
    CUDA_CHECK_THROW(cudaMemcpyAsync(dst, payload_slot(rank_, src, seq), bytes, cudaMemcpyDefault,
                                     rx_stream_));
    return record(false, rx_stream_);
}

void ShmWire::release(int src, std::uint64_t upto) {
    check_link(src, "release");
    const auto s = static_cast<std::size_t>(src);
    if (upto < rx_released_[s] || upto > rx_arrived_[s]) {
        throw std::logic_error("ShmWire::release: link " + std::to_string(src) + "->" +
                               std::to_string(rank_) + " release to " + std::to_string(upto) +
                               " outside [" + std::to_string(rx_released_[s]) + ", " +
                               std::to_string(rx_arrived_[s]) + "]");
    }
    store_release(released_word(rank_, src), upto);
    rx_released_[s] = upto;
}

bool ShmWire::done(OpId id) {
    auto it = ops_.find(id);
    if (it == ops_.end()) throw std::invalid_argument("ShmWire::done: unknown op " + std::to_string(id));
    Op& op = it->second;
    if (op.is_write ? !op.complete : !event_done(op.event)) return false;
    recycle_event(op.event);
    ops_.erase(it);
    return true;
}

}  // namespace nano_nccl::transport::peer
