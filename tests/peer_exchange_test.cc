// CPU multi-rank protocol tests for the EP peer exchange (B1).
//
// Each case runs N logical ranks over the loopback fabric. Every rank issues,
// in program order, per round: exchange_counts (its row of the byte matrix)
// then exchangev (the data, sized from the received counts). Up to `inflight`
// rounds overlap. Payloads are verified byte for byte, completion-event
// implications are checked on every poll, and a step limit catches deadlock.
#include "peer_loopback_fabric.h"
#include "transport/peer/peer_exchanger.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <exception>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using nano_nccl::test::LoopbackConfig;
using nano_nccl::test::LoopbackFabric;
using nano_nccl::transport::peer::CompletionEvent;
using nano_nccl::transport::peer::ExchangeId;
using nano_nccl::transport::peer::ExchangeLayout;
using nano_nccl::transport::peer::PeerExchanger;
using nano_nccl::transport::peer::exchange_counts_async;

constexpr std::uint32_t kDataTag = 7;
constexpr std::uint64_t kStepLimit = 2'000'000;

struct CaseConfig {
    int nranks;
    int slots;
    std::size_t slot_bytes;
    int rounds;
    int inflight;
    int max_delay;
    std::uint64_t seed;
};

std::string name_of(const CaseConfig& c) {
    return "ranks=" + std::to_string(c.nranks) + " slots=" + std::to_string(c.slots) +
           " slot_bytes=" + std::to_string(c.slot_bytes) + " rounds=" + std::to_string(c.rounds) +
           " inflight=" + std::to_string(c.inflight) + " delay=" + std::to_string(c.max_delay) +
           " seed=" + std::to_string(c.seed);
}

std::byte pattern(std::uint64_t epoch, int src, int dst, std::size_t index) {
    std::uint64_t x = epoch * 0x9E3779B97F4A7C15ull ^ (static_cast<std::uint64_t>(src) << 40) ^
                      (static_cast<std::uint64_t>(dst) << 20) ^ index;
    x ^= x >> 31;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 29;
    return static_cast<std::byte>(x & 0xff);
}

// bytes[round][src][dst]: zero rows, short messages, exact slot multiples and
// messages many slots long.
std::vector<std::vector<std::vector<std::size_t>>> make_sizes(const CaseConfig& c) {
    std::mt19937_64 rng(c.seed * 7919 + 17);
    std::vector<std::vector<std::vector<std::size_t>>> sizes(c.rounds);
    for (auto& round : sizes) {
        round.assign(c.nranks, std::vector<std::size_t>(c.nranks));
        const int silent_rank = static_cast<int>(rng() % static_cast<std::uint64_t>(c.nranks));
        for (int s = 0; s < c.nranks; ++s) {
            for (int d = 0; d < c.nranks; ++d) {
                const auto pick = rng() % 8;
                std::size_t bytes;
                if (s == silent_rank || pick == 0) {
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
                round[s][d] = bytes;
            }
        }
    }
    return sizes;
}

struct Round {
    int index = 0;
    std::uint64_t epoch = 0;
    std::vector<std::uint64_t> row;          // this rank's sizes; send_counts may be scribbled
    std::vector<std::uint64_t> send_counts;
    std::vector<std::uint64_t> recv_counts;
    ExchangeId counts_id = 0;
    ExchangeId data_id = 0;
    ExchangeLayout layout;
    std::vector<std::byte> send;
    std::vector<std::byte> recv;
    bool verified = false;
    bool counts_scribbled = false;
    bool data_scribbled = false;
};

// Reusing a send buffer the moment the exchange says it may be reused: if
// kSendBufferReusable is reported too early, the receiver sees garbage.
void scribble_when_reusable(const PeerExchanger& e, ExchangeId id, void* data, std::size_t bytes,
                            bool* done) {
    if (*done || id == 0 || !e.query_completion(id, CompletionEvent::kSendBufferReusable)) return;
    std::memset(data, 0x5A, bytes);
    *done = true;
}

struct RankState {
    int next_round = 0;
    std::deque<std::unique_ptr<Round>> active;
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

struct CaseStats {
    std::uint64_t steps = 0;
    std::uint64_t chunks = 0;
};

CaseStats run_case(const CaseConfig& c) {
    LoopbackFabric fabric({c.nranks, c.slots, c.slot_bytes, c.seed, c.max_delay});
    std::vector<std::unique_ptr<PeerExchanger>> ex;
    for (int r = 0; r < c.nranks; ++r) {
        ex.push_back(std::make_unique<PeerExchanger>(&fabric.endpoint(r)));
    }
    const auto sizes = make_sizes(c);
    std::vector<RankState> state(c.nranks);
    std::vector<int> order(c.nranks);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937_64 rng(c.seed ^ 0xA5A5A5A5ull);

    auto all_done = [&] {
        for (const RankState& s : state) {
            if (s.next_round != c.rounds || !s.active.empty()) return false;
        }
        return true;
    };

    while (!all_done()) {
        if (fabric.now() > kStepLimit) {
            throw std::runtime_error("deadlock: step limit reached\n" + fabric.describe());
        }
        std::shuffle(order.begin(), order.end(), rng);
        for (int r : order) {
            PeerExchanger& e = *ex[r];
            RankState& s = state[r];
            e.progress();

            // Program order per rank: counts_k, data_k, counts_k+1, ...
            // counts_k+1 is only issued after data_k, so all ranks agree on
            // the per-link order.
            const bool previous_data_issued = s.active.empty() || s.active.back()->data_id != 0;
            if (s.next_round < c.rounds && previous_data_issued &&
                static_cast<int>(s.active.size()) < c.inflight) {
                auto round = std::make_unique<Round>();
                round->index = s.next_round;
                round->epoch = 1000 + static_cast<std::uint64_t>(s.next_round);
                round->row.assign(sizes[s.next_round][r].begin(), sizes[s.next_round][r].end());
                round->send_counts = round->row;
                round->recv_counts.assign(c.nranks, ~0ull);
                round->counts_id = exchange_counts_async(e, round->epoch, round->send_counts.data(),
                                                         round->recv_counts.data(), 1);
                s.active.push_back(std::move(round));
                ++s.next_round;
            }

            for (auto& round_ptr : s.active) {
                Round& round = *round_ptr;
                const std::string where = "rank " + std::to_string(r) + " round " + std::to_string(round.index);
                check_implications(e, round.counts_id, where + " counts");
                scribble_when_reusable(e, round.counts_id, round.send_counts.data(),
                                       round.send_counts.size() * sizeof(std::uint64_t),
                                       &round.counts_scribbled);
                if (round.data_id == 0) {
                    if (!e.query_completion(round.counts_id, CompletionEvent::kConsumerReadable)) break;
                    for (int p = 0; p < c.nranks; ++p) {
                        if (round.recv_counts[p] != sizes[round.index][p][r]) {
                            throw std::runtime_error(where + ": wrong count from peer " + std::to_string(p));
                        }
                    }
                    ExchangeLayout& L = round.layout;
                    std::size_t send_total = 0, recv_total = 0;
                    for (int p = 0; p < c.nranks; ++p) {
                        L.send_offsets.push_back(send_total);
                        L.send_bytes.push_back(round.row[p]);
                        send_total += round.row[p];
                        L.recv_offsets.push_back(recv_total);
                        L.recv_bytes.push_back(round.recv_counts[p]);
                        recv_total += round.recv_counts[p];
                    }
                    round.send.resize(send_total);
                    round.recv.assign(recv_total, std::byte{0xEE});
                    for (int p = 0; p < c.nranks; ++p) {
                        for (std::size_t i = 0; i < L.send_bytes[p]; ++i) {
                            round.send[L.send_offsets[p] + i] = pattern(round.epoch, r, p, i);
                        }
                    }
                    round.data_id = e.exchangev_async(round.epoch, kDataTag, L, round.send.data(),
                                                      round.recv.data());
                    // Data of this round must be issued before counts of the
                    // next one; stop here this pass.
                    break;
                }
                check_implications(e, round.data_id, where + " data");
                scribble_when_reusable(e, round.data_id, round.send.data(), round.send.size(),
                                       &round.data_scribbled);
                if (!round.verified && e.query_completion(round.data_id, CompletionEvent::kConsumerReadable)) {
                    for (int p = 0; p < c.nranks; ++p) {
                        for (std::size_t i = 0; i < round.layout.recv_bytes[p]; ++i) {
                            if (round.recv[round.layout.recv_offsets[p] + i] != pattern(round.epoch, p, r, i)) {
                                throw std::runtime_error(where + ": payload mismatch from peer " +
                                                         std::to_string(p) + " at byte " + std::to_string(i));
                            }
                        }
                    }
                    round.verified = true;
                }
            }
            while (!s.active.empty()) {
                Round& front = *s.active.front();
                if (front.data_id == 0 || !front.verified) break;
                const bool counts_done =
                    e.query_completion(front.counts_id, CompletionEvent::kSendBufferReusable) &&
                    e.query_completion(front.counts_id, CompletionEvent::kReceiveSlotsReleased);
                const bool data_done =
                    e.query_completion(front.data_id, CompletionEvent::kSendBufferReusable) &&
                    e.query_completion(front.data_id, CompletionEvent::kReceiveSlotsReleased);
                if (!counts_done || !data_done) break;
                e.retire(front.counts_id);
                e.retire(front.data_id);
                s.active.pop_front();
            }
        }
        fabric.step();
    }
    while (!fabric.idle()) fabric.step();
    if (!fabric.drained()) throw std::runtime_error("fabric not drained\n" + fabric.describe());
    for (const auto& e : ex) {
        if (e->active_exchanges() != 0) throw std::runtime_error("exchanges left active");
    }
    return {fabric.now(), fabric.chunks_landed()};
}

template <typename Fn>
bool expect_throw(const char* name, const char* needle, Fn fn) {
    try {
        fn();
    } catch (const std::exception& error) {
        if (std::string(error.what()).find(needle) != std::string::npos) {
            std::printf("PASS negative %s\n", name);
            return true;
        }
        std::printf("FAIL negative %s: wrong error: %s\n", name, error.what());
        return false;
    }
    std::printf("FAIL negative %s: no error raised\n", name);
    return false;
}

// Drives a fixed set of exchanges to completion or failure.
void drive(LoopbackFabric& fabric, std::vector<std::unique_ptr<PeerExchanger>>& ex) {
    for (std::uint64_t i = 0; i < 100000; ++i) {
        for (auto& e : ex) e->progress();
        fabric.step();
    }
}

bool negative_tests() {
    bool ok = true;
    ok &= expect_throw("stale_epoch", "unexpected chunk", [] {
        CaseConfig c{4, 2, 32, 3, 1, 3, 99};
        // run_case builds its own fabric; reproduce one corrupted round here.
        LoopbackFabric fabric({c.nranks, c.slots, c.slot_bytes, c.seed, c.max_delay});
        std::vector<std::unique_ptr<PeerExchanger>> ex;
        for (int r = 0; r < c.nranks; ++r) ex.push_back(std::make_unique<PeerExchanger>(&fabric.endpoint(r)));
        fabric.corrupt_next_header(2, 1, [](auto& h) { h.epoch -= 1; });
        std::vector<std::vector<std::uint64_t>> send(c.nranks, std::vector<std::uint64_t>(c.nranks, 5));
        std::vector<std::vector<std::uint64_t>> recv(c.nranks, std::vector<std::uint64_t>(c.nranks));
        for (int r = 0; r < c.nranks; ++r) exchange_counts_async(*ex[r], 42, send[r].data(), recv[r].data(), 1);
        drive(fabric, ex);
    });
    ok &= expect_throw("size_mismatch", "unexpected chunk", [] {
        LoopbackFabric fabric({2, 2, 16, 5, 2});
        std::vector<std::unique_ptr<PeerExchanger>> ex;
        for (int r = 0; r < 2; ++r) ex.push_back(std::make_unique<PeerExchanger>(&fabric.endpoint(r)));
        std::vector<std::byte> s0(100), r0(100), s1(100), r1(100);
        ExchangeLayout l0{{0, 40}, {0, 0}, {0, 0}, {0, 0}};
        ExchangeLayout l1{{0, 0}, {0, 0}, {30, 0}, {0, 0}};  // expects 30, rank 0 sends 40
        ex[0]->exchangev_async(1, kDataTag, l0, s0.data(), r0.data());
        ex[1]->exchangev_async(1, kDataTag, l1, s1.data(), r1.data());
        drive(fabric, ex);
    });
    ok &= expect_throw("issue_order_mismatch", "unexpected chunk", [] {
        LoopbackFabric fabric({2, 4, 16, 6, 1});
        std::vector<std::unique_ptr<PeerExchanger>> ex;
        for (int r = 0; r < 2; ++r) ex.push_back(std::make_unique<PeerExchanger>(&fabric.endpoint(r)));
        std::vector<std::byte> b(64);
        const ExchangeLayout l = nano_nccl::transport::peer::uniform_layout(2, 8);
        ex[0]->exchangev_async(1, 10, l, b.data(), b.data() + 32);  // rank 0: tag 10 then 11
        ex[0]->exchangev_async(1, 11, l, b.data(), b.data() + 32);
        ex[1]->exchangev_async(1, 11, l, b.data(), b.data() + 32);  // rank 1: tag 11 first
        ex[1]->exchangev_async(1, 10, l, b.data(), b.data() + 32);
        drive(fabric, ex);
    });
    ok &= expect_throw("retire_in_flight", "still in flight", [] {
        LoopbackFabric fabric({2, 2, 16, 1, 1});
        PeerExchanger e(&fabric.endpoint(0));
        std::vector<std::byte> b(64);
        const ExchangeId id = e.exchangev_async(1, kDataTag, nano_nccl::transport::peer::uniform_layout(2, 8),
                                                b.data(), b.data() + 32);
        e.retire(id);
    });
    ok &= expect_throw("overlapping_recv", "overlap", [] {
        LoopbackFabric fabric({2, 2, 16, 1, 1});
        PeerExchanger e(&fabric.endpoint(0));
        std::vector<std::byte> b(64);
        ExchangeLayout l{{8, 8}, {0, 8}, {8, 8}, {0, 4}};
        e.exchangev_async(1, kDataTag, l, b.data(), b.data() + 32);
    });
    return ok;
}

}  // namespace

int main() {
    int failed = 0, passed = 0;
    std::uint64_t total_chunks = 0;
    for (int nranks : {2, 4, 8}) {
        for (int slots : {1, 2, 4}) {
            for (int inflight : {1, 2}) {
                for (int delay : {0, 5}) {
                    for (std::uint64_t seed : {1ull, 2ull}) {
                        const CaseConfig c{nranks, slots, 32, 5, inflight, delay,
                                           seed * 1000 + static_cast<std::uint64_t>(nranks * 100 + slots * 10 + inflight)};
                        try {
                            const CaseStats st = run_case(c);
                            total_chunks += st.chunks;
                            ++passed;
                        } catch (const std::exception& error) {
                            std::printf("FAIL %s\n  %s\n", name_of(c).c_str(), error.what());
                            ++failed;
                        }
                    }
                }
            }
        }
    }
    std::printf("matrix: %d passed, %d failed, %llu chunks delivered\n", passed, failed,
                static_cast<unsigned long long>(total_chunks));
    const bool negatives_ok = negative_tests();
    const bool ok = failed == 0 && negatives_ok;
    std::printf("peer_exchange=%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
