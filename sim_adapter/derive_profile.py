#!/usr/bin/env python3
"""Derives a cost profile for another GPU from a measured one and public specs.

    python3 derive_profile.py BASE OUT [--compute-tflops T --mem-gbs B
        --base-compute-tflops T0 --base-mem-gbs B0 --pcie-ratio R --name NAME]

Defaults: BASE measured on an A40 (BF16 dense 149.7 TFLOPS, 696 GB/s GDDR6,
PCIe Gen4), target H100 PCIe (BF16 dense 756 TFLOPS, 2.0 TB/s HBM2e, PCIe
Gen5: R = 2). This is an estimate for the simulator, not a measurement;
the rules are:

  gpu_expert_ffn_us   roofline-efficiency transfer: at every point the base
                      time's efficiency against max(flops / peak, bytes /
                      bandwidth) is kept and applied to the target roofline
                      (flops = 6 h f per row; bytes = 6 h f weights + (4 h +
                      12 f) per row, bf16);
  other GPU tables    memory-bound kernels: t = a + (t - a) * B0 / B, a = the
                      table's first value (launch / latency part, unchanged);
  copy_*_self_us      device-memory copies: same rule;
  copy_*_remote*      host-staged copies over PCIe: the growing part / R
                      (in-situ us-per-MiB tables: / R);
  everything else     host (CPU) costs, launch latencies, interference
                      ratios: unchanged (the target host CPU is unknown).
The output keeps the base's key and adds "meta derived ..." lines.
"""
import argparse
import sys

MEM_TABLES = ("gpu_permute_us", "gpu_route_us", "gpu_count_us", "gpu_pack_us", "gpu_reduce_us",
              "gpu_expert_probe_us", "copy_tx_self_us", "copy_rx_self_us")
PCIE_PREFIXES = ("copy_tx_remote_us", "copy_rx_remote_us")


def parse(text):
    """Returns a list of entries: ("line", text) or ("table", header, [(x, y)])."""
    out, lines, i = [], text.splitlines(), 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("table "):
            pts, i = [], i + 1
            while lines[i].strip() != "end":
                x, y = lines[i].split()
                pts.append((float(x), float(y)))
                i += 1
            out.append(("table", line, pts))
        else:
            out.append(("line", line))
        i += 1
    return out


def key_value(text, name):
    for line in text.splitlines():
        if line.startswith("key "):
            for item in line.split()[1:]:
                k, _, v = item.partition("=")
                if k == name:
                    return float(v)
    raise SystemExit(f"derive_profile: key has no {name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("--compute-tflops", type=float, default=756.0)
    ap.add_argument("--mem-gbs", type=float, default=2000.0)
    ap.add_argument("--base-compute-tflops", type=float, default=149.7)
    ap.add_argument("--base-mem-gbs", type=float, default=696.0)
    ap.add_argument("--pcie-ratio", type=float, default=2.0)
    ap.add_argument("--name", default="H100 PCIe (derived)")
    a = ap.parse_args()

    text = open(a.base).read()
    h, f = key_value(text, "hidden"), key_value(text, "ffn")
    mem_ratio = a.base_mem_gbs / a.mem_gbs

    def roofline(rows, tflops, gbs):
        flops = rows * 6 * h * f
        nbytes = 6 * h * f + rows * (4 * h + 12 * f)
        return max(flops / (tflops * 1e6), nbytes / (gbs * 1e3))  # us

    def affine(pts, ratio):
        a0 = pts[0][1]
        return [(x, a0 + max(0.0, y - a0) * ratio) for x, y in pts]

    out = []
    for e in parse(text):
        if e[0] == "line":
            out.append(e[1])
            if e[1].startswith("key "):
                out += [f"meta derived {a.name} from {a.base.split('/')[-1]} by sim_adapter/derive_profile.py "
                        "(estimate, not a measurement)",
                        f"meta derived compute {a.base_compute_tflops} -> {a.compute_tflops} TFLOPS, "
                        f"memory {a.base_mem_gbs} -> {a.mem_gbs} GB/s, PCIe x{a.pcie_ratio}; "
                        "host costs unchanged"]
            continue
        _, header, pts = e
        name = header.split()[1]
        base_name = name.split("@")[0]
        if base_name == "gpu_expert_ffn_us":
            pts = [(x, y * roofline(x, a.compute_tflops, a.mem_gbs) / roofline(x, a.base_compute_tflops,
                                                                              a.base_mem_gbs))
                   for x, y in pts]
        elif base_name in MEM_TABLES:
            pts = affine(pts, mem_ratio)
        elif base_name.startswith(PCIE_PREFIXES):
            if "insitu" in base_name:
                pts = [(x, y / a.pcie_ratio) for x, y in pts]
            else:
                pts = affine(pts, 1.0 / a.pcie_ratio)
        out.append(header)
        out += [f"{x:g} {y:.6g}" for x, y in pts]
        out.append("end")
    with open(a.out, "w") as fo:
        fo.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
