// Multi-process tests for the EP peer exchange over ShmWire (host-staged
// shared memory). Run with mpirun, one process per rank; ranks may share a
// GPU. Uses no CUDA peer access.
//
//   peer_shm_wire_test                 correctness matrix + negative tests
//   peer_shm_wire_test --bench [--slots S] [--slot-bytes B] [--max-bytes M]
//                                      uniform exchange bandwidth
//
// Correctness mirrors the CPU loopback test: per round every rank issues
// exchange_counts (host memory) then exchangev (data in device or pinned host
// memory, sized from the received counts); up to `inflight` rounds overlap.
// Payloads and guard bytes are verified, completion-event implications are
// checked on every poll, and send buffers are overwritten as soon as they are
// reported reusable.
#include "transport/peer/peer_exchanger.h"
#include "transport/peer/shm_wire.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

namespace {

using nano_nccl::transport::peer::CompletionEvent;
using nano_nccl::transport::peer::ExchangeId;
using nano_nccl::transport::peer::ExchangeLayout;
using nano_nccl::transport::peer::PeerExchanger;
using nano_nccl::transport::peer::ShmWire;
using nano_nccl::transport::peer::ShmWireConfig;
using nano_nccl::transport::peer::exchange_counts_async;
using Clock = std::chrono::steady_clock;

constexpr std::uint32_t kDataTag = 7;
constexpr std::size_t kGuardBytes = 256;
constexpr std::byte kGuard{0xEE};
constexpr double kCaseTimeoutSeconds = 120.0;

void cuda_check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

int g_rank = 0;
int g_nranks = 0;
int g_device = 0;

enum class Memory { kDevice, kPinnedHost };

struct CaseConfig {
    int slots;
    std::size_t slot_bytes;
    int rounds;
    int inflight;
    Memory memory;
    std::uint64_t seed;
    int delay_us;  // ShmWireConfig::debug_copy_delay_us
};

std::string name_of(const CaseConfig& c) {
    return "ranks=" + std::to_string(g_nranks) + " slots=" + std::to_string(c.slots) +
           " slot_bytes=" + std::to_string(c.slot_bytes) + " rounds=" + std::to_string(c.rounds) +
           " inflight=" + std::to_string(c.inflight) +
           " memory=" + (c.memory == Memory::kDevice ? "device" : "pinned") +
           " seed=" + std::to_string(c.seed) + " delay_us=" + std::to_string(c.delay_us);
}

std::byte pattern(std::uint64_t epoch, int src, int dst, std::size_t index) {
    std::uint64_t x = epoch * 0x9E3779B97F4A7C15ull ^ (static_cast<std::uint64_t>(src) << 40) ^
                      (static_cast<std::uint64_t>(dst) << 20) ^ index;
    x ^= x >> 31;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 29;
    return static_cast<std::byte>(x & 0xff);
}

// bytes[round][src][dst], identical on every rank (same seed).
std::vector<std::vector<std::vector<std::size_t>>> make_sizes(const CaseConfig& c) {
    std::mt19937_64 rng(c.seed * 7919 + 17);
    std::vector<std::vector<std::vector<std::size_t>>> sizes(static_cast<std::size_t>(c.rounds));
    for (auto& round : sizes) {
        round.assign(static_cast<std::size_t>(g_nranks), std::vector<std::size_t>(static_cast<std::size_t>(g_nranks)));
        const auto silent = static_cast<int>(rng() % static_cast<std::uint64_t>(g_nranks));
        for (int s = 0; s < g_nranks; ++s) {
            for (int d = 0; d < g_nranks; ++d) {
                const auto pick = rng() % 8;
                std::size_t bytes;
                if (s == silent || pick == 0) {
                    bytes = 0;
                } else if (pick <= 3) {
                    bytes = 1 + rng() % c.slot_bytes;
                } else if (pick == 4) {
                    bytes = c.slot_bytes * (1 + rng() % 3);
                } else if (pick <= 6) {
                    bytes = 1 + rng() % (c.slot_bytes * 5);
                } else {
                    bytes = c.slot_bytes * 12 + rng() % c.slot_bytes;
                }
                round[static_cast<std::size_t>(s)][static_cast<std::size_t>(d)] = bytes;
            }
        }
    }
    return sizes;
}

// Device or pinned host bytes.
class Buffer {
public:
    Buffer(Memory memory, std::size_t bytes) : memory_(memory), bytes_(bytes) {
        const std::size_t alloc = std::max<std::size_t>(bytes, 1);
        if (memory == Memory::kDevice) {
            cuda_check(cudaMalloc(&ptr_, alloc), "cudaMalloc");
        } else {
            cuda_check(cudaMallocHost(&ptr_, alloc), "cudaMallocHost");
        }
    }
    ~Buffer() {
        if (memory_ == Memory::kDevice) cudaFree(ptr_); else cudaFreeHost(ptr_);
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    std::byte* data() const { return static_cast<std::byte*>(ptr_); }
    std::size_t size() const { return bytes_; }
    void upload(const std::vector<std::byte>& host) const {
        cuda_check(cudaMemcpy(ptr_, host.data(), host.size(), cudaMemcpyDefault), "upload");
    }
    std::vector<std::byte> download() const {
        std::vector<std::byte> host(bytes_);
        cuda_check(cudaMemcpy(host.data(), ptr_, bytes_, cudaMemcpyDefault), "download");
        return host;
    }
    void fill(std::byte value) const {
        if (memory_ == Memory::kDevice) {
            cuda_check(cudaMemset(ptr_, static_cast<int>(value), bytes_), "cudaMemset");
            cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(fill)");
        } else {
            std::memset(ptr_, static_cast<int>(value), bytes_);
        }
    }

private:
    Memory memory_;
    std::size_t bytes_;
    void* ptr_ = nullptr;
};

struct Round {
    int index = 0;
    std::uint64_t epoch = 0;
    std::vector<std::uint64_t> row;
    std::vector<std::uint64_t> send_counts;  // pageable host memory
    std::vector<std::uint64_t> recv_counts;
    ExchangeId counts_id = 0;
    ExchangeId data_id = 0;
    ExchangeLayout layout;
    std::size_t recv_total = 0;
    std::unique_ptr<Buffer> send;
    std::unique_ptr<Buffer> recv;  // recv_total data bytes followed by guard bytes
    bool verified = false;
    bool counts_scribbled = false;
    bool data_scribbled = false;
};

void check_implications(const PeerExchanger& ex, ExchangeId id, const std::string& what) {
    auto q = [&](CompletionEvent e) { return ex.query_completion(id, e); };
    if (q(CompletionEvent::kSendBufferReusable) && !q(CompletionEvent::kSubmitted)) {
        throw std::logic_error(what + ": send buffer reusable before submitted");
    }
    if (q(CompletionEvent::kConsumerReadable) && !q(CompletionEvent::kPayloadArrived)) {
        throw std::logic_error(what + ": consumer readable before payload arrived");
    }
    if (q(CompletionEvent::kReceiveSlotsReleased) && !q(CompletionEvent::kConsumerReadable)) {
        throw std::logic_error(what + ": slots released before consumer readable (copy-out mode)");
    }
}

bool finished(const PeerExchanger& e, ExchangeId id) {
    return e.query_completion(id, CompletionEvent::kSendBufferReusable) &&
           e.query_completion(id, CompletionEvent::kReceiveSlotsReleased);
}

std::uint64_t run_case(const CaseConfig& c) {
    ShmWire wire(MPI_COMM_WORLD, {c.slots, c.slot_bytes, g_device, c.delay_us});
    PeerExchanger e(&wire);
    const auto sizes = make_sizes(c);
    const int r = g_rank;
    const auto n = static_cast<std::size_t>(g_nranks);
    int next_round = 0;
    std::uint64_t bytes_checked = 0;
    std::deque<std::unique_ptr<Round>> active;
    const auto start = Clock::now();

    while (next_round < c.rounds || !active.empty()) {
        if (std::chrono::duration<double>(Clock::now() - start).count() > kCaseTimeoutSeconds) {
            throw std::runtime_error("deadlock: case did not finish in time (round " +
                                     std::to_string(active.empty() ? next_round : active.front()->index) + ")");
        }
        e.progress();

        // Program order: counts_k, data_k, counts_k+1, ...
        const bool previous_data_issued = active.empty() || active.back()->data_id != 0;
        if (next_round < c.rounds && previous_data_issued && static_cast<int>(active.size()) < c.inflight) {
            auto round = std::make_unique<Round>();
            round->index = next_round;
            round->epoch = 1000 + static_cast<std::uint64_t>(next_round);
            const auto& row = sizes[static_cast<std::size_t>(next_round)][static_cast<std::size_t>(r)];
            round->row.assign(row.begin(), row.end());
            round->send_counts = round->row;
            round->recv_counts.assign(n, ~0ull);
            round->counts_id = exchange_counts_async(e, round->epoch, round->send_counts.data(),
                                                     round->recv_counts.data(), 1);
            active.push_back(std::move(round));
            ++next_round;
        }

        for (auto& round_ptr : active) {
            Round& round = *round_ptr;
            const std::string where = "rank " + std::to_string(r) + " round " + std::to_string(round.index);
            check_implications(e, round.counts_id, where + " counts");
            if (!round.counts_scribbled &&
                e.query_completion(round.counts_id, CompletionEvent::kSendBufferReusable)) {
                std::fill(round.send_counts.begin(), round.send_counts.end(), 0x5A5A5A5A5A5A5A5Aull);
                round.counts_scribbled = true;
            }
            if (round.data_id == 0) {
                if (!e.query_completion(round.counts_id, CompletionEvent::kConsumerReadable)) break;
                for (int p = 0; p < g_nranks; ++p) {
                    const auto expected = sizes[static_cast<std::size_t>(round.index)][static_cast<std::size_t>(p)]
                                               [static_cast<std::size_t>(r)];
                    if (round.recv_counts[static_cast<std::size_t>(p)] != expected) {
                        throw std::runtime_error(where + ": wrong count from peer " + std::to_string(p));
                    }
                }
                ExchangeLayout& L = round.layout;
                std::size_t send_total = 0;
                for (std::size_t p = 0; p < n; ++p) {
                    L.send_offsets.push_back(send_total);
                    L.send_bytes.push_back(round.row[p]);
                    send_total += round.row[p];
                    L.recv_offsets.push_back(round.recv_total);
                    L.recv_bytes.push_back(round.recv_counts[p]);
                    round.recv_total += round.recv_counts[p];
                }
                std::vector<std::byte> host(send_total);
                for (std::size_t p = 0; p < n; ++p) {
                    for (std::size_t i = 0; i < L.send_bytes[p]; ++i) {
                        host[L.send_offsets[p] + i] = pattern(round.epoch, r, static_cast<int>(p), i);
                    }
                }
                round.send = std::make_unique<Buffer>(c.memory, send_total);
                round.send->upload(host);
                round.recv = std::make_unique<Buffer>(c.memory, round.recv_total + kGuardBytes);
                round.recv->fill(kGuard);
                round.data_id = e.exchangev_async(round.epoch, kDataTag, L, round.send->data(),
                                                  round.recv->data());
                break;  // data_k goes before counts_k+1
            }
            check_implications(e, round.data_id, where + " data");
            if (!round.data_scribbled && e.query_completion(round.data_id, CompletionEvent::kSendBufferReusable)) {
                round.send->fill(std::byte{0x5A});
                round.data_scribbled = true;
            }
            if (!round.verified && e.query_completion(round.data_id, CompletionEvent::kConsumerReadable)) {
                const std::vector<std::byte> got = round.recv->download();
                for (std::size_t p = 0; p < n; ++p) {
                    for (std::size_t i = 0; i < round.layout.recv_bytes[p]; ++i) {
                        if (got[round.layout.recv_offsets[p] + i] != pattern(round.epoch, static_cast<int>(p), r, i)) {
                            throw std::runtime_error(where + ": payload mismatch from peer " + std::to_string(p) +
                                                     " at byte " + std::to_string(i));
                        }
                    }
                }
                for (std::size_t i = round.recv_total; i < got.size(); ++i) {
                    if (got[i] != kGuard) throw std::runtime_error(where + ": guard byte overwritten");
                }
                bytes_checked += round.recv_total;
                round.verified = true;
            }
        }
        while (!active.empty()) {
            Round& front = *active.front();
            if (front.data_id == 0 || !front.verified || !finished(e, front.counts_id) ||
                !finished(e, front.data_id)) {
                break;
            }
            e.retire(front.counts_id);
            e.retire(front.data_id);
            active.pop_front();
        }
    }
    if (e.active_exchanges() != 0) throw std::runtime_error("exchanges left active");
    // Peers may still be reading this rank's region; keep it until all finish.
    MPI_Barrier(MPI_COMM_WORLD);
    return bytes_checked;
}

// Collective: true when the expected error was raised on at least one rank and
// no rank saw a different error.
bool negative(const char* name, const char* needle, void (*fn)()) {
    int saw_expected = 0, saw_other = 0;
    try {
        fn();
    } catch (const std::exception& error) {
        if (std::strstr(error.what(), needle) != nullptr) {
            saw_expected = 1;
        } else {
            saw_other = 1;
            std::printf("rank %d negative %s: wrong error: %s\n", g_rank, name, error.what());
        }
    }
    int any_expected = 0, any_other = 0;
    MPI_Allreduce(&saw_expected, &any_expected, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&saw_other, &any_other, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    const bool ok = any_expected == 1 && any_other == 0;
    if (g_rank == 0) std::printf("%s negative %s\n", ok ? "PASS" : "FAIL", name);
    return ok;
}

// Drives for a bounded time; the receiver's header check is what must fire.
void drive(PeerExchanger& e) {
    const auto start = Clock::now();
    while (std::chrono::duration<double>(Clock::now() - start).count() < 2.0) e.progress();
}

bool negative_tests() {
    bool ok = true;
    ok &= negative("geometry_mismatch", "configuration", [] {
        ShmWire wire(MPI_COMM_WORLD, {g_rank == 0 ? 2 : 4, 4096, g_device});
    });
    // Every rank sends 40 bytes to the next rank, which expects 30.
    ok &= negative("size_mismatch", "unexpected chunk", [] {
        ShmWire wire(MPI_COMM_WORLD, {2, 16, g_device});
        PeerExchanger e(&wire);
        const auto n = static_cast<std::size_t>(g_nranks);
        ExchangeLayout l{std::vector<std::size_t>(n, 0), std::vector<std::size_t>(n, 0),
                         std::vector<std::size_t>(n, 0), std::vector<std::size_t>(n, 0)};
        l.send_bytes[static_cast<std::size_t>((g_rank + 1) % g_nranks)] = 40;
        l.recv_bytes[static_cast<std::size_t>((g_rank + g_nranks - 1) % g_nranks)] = 30;
        std::vector<std::byte> s(64), d(64);
        e.exchangev_async(1, kDataTag, l, s.data(), d.data());
        try {
            drive(e);
        } catch (...) {
            MPI_Barrier(MPI_COMM_WORLD);
            throw;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    });
    // Ranks disagree on epoch for the same exchange.
    ok &= negative("stale_epoch", "unexpected chunk", [] {
        ShmWire wire(MPI_COMM_WORLD, {2, 64, g_device});
        PeerExchanger e(&wire);
        std::vector<std::uint64_t> s(static_cast<std::size_t>(g_nranks), 3), d(static_cast<std::size_t>(g_nranks));
        exchange_counts_async(e, g_rank == 0 ? 41 : 42, s.data(), d.data(), 1);
        try {
            drive(e);
        } catch (...) {
            MPI_Barrier(MPI_COMM_WORLD);
            throw;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    });
    return ok;
}

int run_tests() {
    int failed = 0, passed = 0;
    std::uint64_t bytes = 0;
    std::vector<CaseConfig> cases;
    // Slow copies: any completion reported before its copy finished is seen.
    // They run first, so a mutation check shows whether they catch a bug.
    for (int slots : {1, 2}) {
        for (int inflight : {1, 2}) {
            for (Memory memory : {Memory::kDevice, Memory::kPinnedHost}) {
                cases.push_back({slots, 4096, 6, inflight, memory, static_cast<std::uint64_t>(77 + slots * 10 + inflight),
                                 200});
            }
        }
    }
    for (int slots : {1, 2, 4}) {
        for (std::size_t slot_bytes : {std::size_t{4096}, std::size_t{256} << 10}) {
            for (int inflight : {1, 2}) {
                for (Memory memory : {Memory::kDevice, Memory::kPinnedHost}) {
                    cases.push_back({slots, slot_bytes, 8, inflight, memory,
                                     static_cast<std::uint64_t>(slots * 100 + inflight * 10) + slot_bytes, 0});
                }
            }
        }
    }
    for (const CaseConfig& c : cases) {
        const auto start = Clock::now();
        try {
            bytes += run_case(c);
            ++passed;
        } catch (const std::exception& error) {
            // Ranks cannot recover a half-finished exchange together.
            std::printf("FAIL rank %d %s\n  %s\n", g_rank, name_of(c).c_str(), error.what());
            std::fflush(stdout);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (g_rank == 0) {
            std::printf("PASS %s (%.2f s)\n", name_of(c).c_str(),
                        std::chrono::duration<double>(Clock::now() - start).count());
            std::fflush(stdout);
        }
    }
    std::uint64_t all_bytes = 0;
    MPI_Reduce(&bytes, &all_bytes, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    const bool negatives_ok = negative_tests();
    const bool ok = failed == 0 && negatives_ok;
    if (g_rank == 0) {
        std::printf("matrix: %d passed, %d failed, %llu payload bytes verified\n", passed, failed,
                    static_cast<unsigned long long>(all_bytes));
        std::printf("peer_shm_wire=%s\n", ok ? "PASS" : "FAIL");
    }
    return ok ? 0 : 1;
}

// Uniform exchange: every rank sends `bytes` to every rank (itself included).
// Time runs from issue to the exchange being fully finished on this rank; the
// reported time is the slowest rank's.
int run_bench(int slots, std::size_t slot_bytes, std::size_t max_bytes) {
    ShmWire wire(MPI_COMM_WORLD, {slots, slot_bytes, g_device});
    PeerExchanger e(&wire);
    const auto n = static_cast<std::size_t>(g_nranks);
    Buffer send(Memory::kDevice, max_bytes * n);
    Buffer recv(Memory::kDevice, max_bytes * n);
    send.fill(std::byte{1});
    constexpr int kWarmup = 5, kIters = 20;
    if (g_rank == 0) {
        std::printf("# ShmWire uniform exchange: ranks=%d slots=%d slot_bytes=%zu region=%zu MiB/rank\n",
                    g_nranks, slots, slot_bytes, wire.region_bytes() >> 20);
        std::printf("# %12s %12s %14s %14s\n", "bytes/peer", "time_us", "out_GB/s", "remote_GB/s");
    }
    std::uint64_t epoch = 1;
    for (std::size_t bytes = 256 << 10; bytes <= max_bytes; bytes *= 4) {
        const auto layout = nano_nccl::transport::peer::uniform_layout(g_nranks, bytes);
        double total = 0;
        for (int it = 0; it < kWarmup + kIters; ++it) {
            MPI_Barrier(MPI_COMM_WORLD);
            const auto t0 = Clock::now();
            const ExchangeId id = e.exchangev_async(epoch++, kDataTag, layout, send.data(), recv.data());
            while (!finished(e, id)) e.progress();
            double elapsed = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
            e.retire(id);
            double slowest = 0;
            MPI_Allreduce(&elapsed, &slowest, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (it >= kWarmup) total += slowest;
        }
        const double us = total / kIters;
        if (g_rank == 0) {
            // out: everything a rank sends; remote: excluding its self link.
            std::printf("  %12zu %12.2f %14.2f %14.2f\n", bytes, us, bytes * n / us / 1e3,
                        bytes * (n - 1) / us / 1e3);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_nranks);
    int status = 1;
    try {
        MPI_Comm node;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
        int local_rank = 0;
        MPI_Comm_rank(node, &local_rank);
        MPI_Comm_free(&node);
        int devices = 0;
        cuda_check(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
        if (devices == 0) throw std::runtime_error("no CUDA device");
        g_device = local_rank % devices;
        cuda_check(cudaSetDevice(g_device), "cudaSetDevice");

        bool bench = false;
        int slots = 4;
        std::size_t slot_bytes = std::size_t{1} << 20, max_bytes = std::size_t{64} << 20;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--bench") bench = true;
            else if (arg == "--slots" && i + 1 < argc) slots = std::atoi(argv[++i]);
            else if (arg == "--slot-bytes" && i + 1 < argc) slot_bytes = std::strtoull(argv[++i], nullptr, 0);
            else if (arg == "--max-bytes" && i + 1 < argc) max_bytes = std::strtoull(argv[++i], nullptr, 0);
            else throw std::invalid_argument("unknown argument " + arg);
        }
        status = bench ? run_bench(slots, slot_bytes, max_bytes) : run_tests();
    } catch (const std::exception& error) {
        std::printf("rank %d error: %s\n", g_rank, error.what());
        std::fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
