#!/usr/bin/env python3
"""Adds in-situ copy costs to a simulator profile.

Reads nsys reports (exported to sqlite) of real multi-process EP runs
(tests/ep_moe_test over ShmWire, one GPU per rank, rank r on CUDA device r)
and measures, per rank:

1. Remote chunk copy cost against interference from the same GPU's other
   direction. A remote copy is tx (device -> host slot, D2H) or rx (host
   slot -> device, H2D). On the A40 host the cost of a copy depends on how
   much of it overlaps the same GPU's remote copy in the other direction and
   hardly on the other GPU's copies (a per-engine fit over the other three
   engines gives cross-GPU terms near zero). So, per rank and direction, a
   least-squares line
       us_per_MiB = a + b * f,   f = fraction of the copy overlapped by the
                                     same GPU's other-direction remote copies
   is written as a two-point table (f = 0 -> a, f = 1 -> a + b), with b
   clamped to >= 0.

2. Engine turnaround: for a copy that was already issued when the previous
   copy on its stream ended, the gap previous end -> start, per stream and
   pair of kinds (remote = H2D/D2H, self = D2D), as the mean of the gaps up
   to TRIM_US (the distribution has a long tail and the mean is what sets
   the engine's throughput; longer gaps are rare stalls). The tx stream is
   the one carrying D2H copies, the rx stream the one carrying H2D.

3. Optionally (--gpu-pair OVERLAPPED SERIAL), the GPU slowdown while the
   same GPU copies remote chunks: two runs with the same expert grouping,
   one overlapped and one serial, so their expert ops (permute, expert
   GEMMs, permute on the compute stream) have the same rows. Per rank,
   r = median op span overlapped / median op span serial and f = mean over
   the overlapped ops of the fraction of the op covered by the GPU's remote
   tx copies plus that covered by its remote rx copies (0..2); the slowdown
   with both engines busy for all of an op is 1 + (r - 1) * 2 / f, written
   as gpu_copy_slowdown@rank. This needs an overlapped run, so keep that
   configuration out of the validation set.

The simulator (sim_wire.cc, sim_layer.cc) uses (1) for remote chunks and (3)
for GPU ops when its contention model is on, and (2) for every queued copy,
so these costs come from the same kind of run the simulator replays rather
than from an isolated microbenchmark. Calibrate (1) and (2) from runs
without communication/compute overlap (serial ones) and validate on the
overlapped configurations.

Usage: calibrate_copies.py PROFILE_IN PROFILE_OUT REPORT.sqlite...
                           [--min-bytes N] [--gpu-pair OVERLAPPED SERIAL]
Samples of all reports are pooled.
"""
import argparse
import os
import sqlite3
import statistics
import sys

TABLES = {"tx": "copy_tx_remote_insitu_us_per_mib", "rx": "copy_rx_remote_insitu_us_per_mib"}
TURN = {"tx": "copy_tx_turnaround_us_", "rx": "copy_rx_turnaround_us_"}
COPY_HTOD, COPY_DTOH, COPY_DTOD = 1, 2, 8
PAIRS = ("rr", "rs", "sr", "ss")
TRIM_US = 100.0


def trimmed_mean(gaps):
    kept = [g for g in gaps if g <= TRIM_US]
    return sum(kept) / len(kept) if kept else statistics.median(gaps)


def turnarounds(db, min_bytes):
    """{(device, direction): {"rs": [gaps], ...}} for queued copies."""
    api_end = dict(db.execute("SELECT correlationId, end FROM CUPTI_ACTIVITY_KIND_RUNTIME"))
    streams = {}
    for dev, strm, s, e, k, b, corr in db.execute(
            "SELECT deviceId, streamId, start, end, copyKind, bytes, correlationId FROM CUPTI_ACTIVITY_KIND_MEMCPY"):
        if b >= min_bytes and k in (COPY_HTOD, COPY_DTOH, COPY_DTOD):
            streams.setdefault((dev, strm), []).append((s, e, k, api_end.get(corr)))
    out = {}
    for (dev, _), v in streams.items():
        kinds = {k for _, _, k, _ in v}
        if COPY_DTOH in kinds and COPY_HTOD not in kinds:
            direction = "tx"
        elif COPY_HTOD in kinds and COPY_DTOH not in kinds:
            direction = "rx"
        else:
            continue  # not an exchange copy stream
        v.sort()
        gaps = {}
        for i in range(1, len(v)):
            issued = v[i][3]
            if issued is None or issued >= v[i - 1][1]:
                continue
            pair = ("s" if v[i - 1][2] == COPY_DTOD else "r") + ("s" if v[i][2] == COPY_DTOD else "r")
            gaps.setdefault(pair, []).append(max(0.0, (v[i][0] - v[i - 1][1]) / 1000.0))
        prev = out.get((dev, direction))
        if gaps and (prev is None or len(v) > prev[0]):
            out[(dev, direction)] = (len(v), gaps)
    return {k: g for k, (_, g) in out.items()}


def interference(db, min_bytes):
    """{(device, direction): [(f, us_per_mib)]} for remote chunk copies."""
    copies = sorted(((dev, s, e, k, b) for dev, s, e, k, b in db.execute(
        "SELECT deviceId, start, end, copyKind, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY")
        if k in (COPY_HTOD, COPY_DTOH) and b >= min_bytes and e > s), key=lambda c: c[1])
    by_engine = {}
    for c in copies:
        by_engine.setdefault((c[0], c[3]), []).append(c)
    samples = {}
    for (dev, k), mine in by_engine.items():
        partner = by_engine.get((dev, COPY_DTOH if k == COPY_HTOD else COPY_HTOD), [])
        j = 0
        for _, s, e, _, b in mine:
            while j < len(partner) and partner[j][2] <= s:
                j += 1
            overlap, i = 0, j
            while i < len(partner) and partner[i][1] < e:
                overlap += max(0, min(e, partner[i][2]) - max(s, partner[i][1]))
                i += 1
            samples.setdefault((dev, "rx" if k == COPY_HTOD else "tx"), []).append(
                (overlap / (e - s), (e - s) / 1000.0 / (b / 2**20)))
    return samples


def expert_ops(db):
    """{device: [(start, end)]} of expert ops: permute, kernels, permute."""
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    per = {}
    for dev, s, e, n in db.execute("SELECT deviceId, start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        per.setdefault(dev, []).append((s, e, names[n]))
    ops = {}
    for dev, v in per.items():
        v.sort()
        begin = None
        for s, e, n in v:
            if n.startswith("permute_rows"):
                if begin is None:
                    begin = s
                else:
                    ops.setdefault(dev, []).append((begin, e))
                    begin = None
    return ops


def remote_cover(db, dev, s, e):
    """Fraction of [s, e) covered by dev's remote tx plus by its remote rx copies."""
    total = 0.0
    for kind in (COPY_HTOD, COPY_DTOH):
        rows = db.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE deviceId = ? AND copyKind = ? "
                          "AND bytes >= 65536 AND end > ? AND start < ?", (dev, kind, s, e)).fetchall()
        total += sum(min(e, ce) - max(s, cs) for cs, ce in rows) / (e - s)
    return min(2.0, total)


def gpu_slowdown(overlapped, serial):
    """{device: (slowdown, r, f, ops)} from a pair of reports with the same grouping."""
    a, b = sqlite3.connect(overlapped), sqlite3.connect(serial)
    oa, ob = expert_ops(a), expert_ops(b)
    out = {}
    for dev in sorted(oa):
        # The forward's bitwise check also runs a serial reference forward
        # with one big expert op: its spans are far longer, skip them.
        ref = statistics.median(e - s for s, e in ob.get(dev, [])) if ob.get(dev) else None
        if ref is None:
            sys.exit("no expert ops for device %d in %s" % (dev, serial))
        sa = [(s, e) for s, e in oa[dev] if e - s < 2 * ref]
        sb = [e - s for s, e in ob[dev] if e - s < 2 * ref]
        r = statistics.median(e - s for s, e in sa) / statistics.median(sb)
        f = sum(remote_cover(a, dev, s, e) for s, e in sa) / len(sa)
        out[dev] = (1.0 + max(0.0, r - 1.0) * 2.0 / f if f > 0.05 else 1.0, r, f, len(sa))
    return out


def fit(points):
    n = len(points)
    mf = sum(f for f, _ in points) / n
    my = sum(y for _, y in points) / n
    var = sum((f - mf) ** 2 for f, _ in points)
    b = sum((f - mf) * (y - my) for f, y in points) / var if var > 0 else 0.0
    b = max(0.0, b)
    return my - b * mf, b


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profile_in")
    ap.add_argument("profile_out")
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--min-bytes", type=int, default=1 << 19)
    ap.add_argument("--gpu-pair", nargs=2, metavar=("OVERLAPPED", "SERIAL"))
    a = ap.parse_args()
    samples, turns = {}, {}
    for report in a.reports:
        db = sqlite3.connect(report)
        for key, v in interference(db, a.min_bytes).items():
            samples.setdefault(key, []).extend(v)
        for key, gaps in turnarounds(db, 65536).items():
            for p, g in gaps.items():
                turns.setdefault(key, {}).setdefault(p, []).extend(g)
    devices = sorted({d for d, _ in samples})

    # Drop earlier in-situ entries.
    out, skip = [], False
    for line in open(a.profile_in).read().splitlines():
        if line.startswith("table ") and line.split()[1].split("@")[0] in TABLES.values():
            skip = True
            continue
        if skip:
            skip = line != "end"
            continue
        if line.startswith("meta in-situ copy costs"):
            continue
        if line.startswith("scalar ") and any(line.split()[1].startswith(t) for t in TURN.values()):
            continue
        if a.gpu_pair and line.startswith("scalar gpu_copy_slowdown@"):
            continue
        if line.startswith("meta in-situ gpu slowdown") and a.gpu_pair:
            continue
        out.append(line)
    out.append("meta in-situ copy costs from %s: remote chunk copies >= %d B, us/MiB fitted against the "
               "overlapped fraction by the same GPU's other-direction remote copies; queued-copy turnarounds "
               "(mean of gaps <= %g us)" % (" ".join(os.path.basename(r) for r in a.reports), a.min_bytes, TRIM_US))
    for direction, name in TABLES.items():
        pooled = []
        for dev in devices:
            pts = samples.get((dev, direction), [])
            if len(pts) < 20:
                sys.exit("fewer than 20 remote %s copies for device %d" % (direction, dev))
            c0, c1 = fit(pts)
            pooled += pts
            out += ["table %s@%d overlap us_per_mib" % (name, dev), "0 %.4f" % c0, "1 %.4f" % (c0 + c1), "end"]
            print("rank %d %s: %.1f us/MiB alone, +%.1f with the other direction busy (n=%d)" %
                  (dev, direction, c0, c1, len(pts)))
        c0, c1 = fit(pooled)
        out += ["table %s overlap us_per_mib" % name, "0 %.4f" % c0, "1 %.4f" % (c0 + c1), "end"]
    for direction, prefix in TURN.items():
        # A pair with few samples on one rank (e.g. back-to-back remote
        # copies in a serial run) takes the median over all ranks.
        merged = {}
        for dev in devices:
            for p, g in turns.get((dev, direction), {}).items():
                merged.setdefault(p, []).extend(g)
        for p in PAIRS:
            if len(merged.get(p, [])) < 3:
                sys.exit("fewer than 3 queued %s copies of kind %s" % (direction, p))
        overall = {p: trimmed_mean(merged[p]) for p in PAIRS}
        for dev in devices:
            gaps = turns.get((dev, direction), {})
            med = {p: trimmed_mean(gaps[p]) if len(gaps.get(p, [])) >= 3 else overall[p] for p in PAIRS}
            out += ["scalar %s%s@%d %.4f" % (prefix, p, dev, med[p]) for p in PAIRS]
            print("rank %d %s turnaround us: %s" % (dev, direction, "  ".join("%s %.1f" % (p, med[p]) for p in PAIRS)))
        out += ["scalar %s%s %.4f" % (prefix, p, overall[p]) for p in PAIRS]
    if a.gpu_pair:
        out.append("meta in-situ gpu slowdown from %s vs %s (expert-op spans)" %
                   tuple(os.path.basename(r) for r in a.gpu_pair))
        for dev, (slow, r, f, n) in gpu_slowdown(*a.gpu_pair).items():
            out.append("scalar gpu_copy_slowdown@%d %.4f" % (dev, slow))
            print("rank %d gpu: expert ops %.3fx serial with %.2f engines' remote copies on average (n=%d) "
                  "-> slowdown %.3f with both engines busy" % (dev, r, f, n, slow))
    with open(a.profile_out, "w") as f:
        f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
