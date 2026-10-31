"""PyTorch reference check for `ep_moe_test --ffn --dump DIR`.

Recomputes every rank's MoE output on one GPU from the dumped inputs, routing
and expert weights, and compares it with what the EP data plane produced:

- "matched": bf16 at the same points as ExpertFfn (GEMM outputs and the
  SwiGLU product rounded to bf16, top-k sum in fp32), with fp32 GEMM
  reductions on both sides (PyTorch's bf16 reduced-precision reduction off);
- "fp32": the same layer in fp32 from the bf16 inputs.

A third reference with the top-k weights of each token rotated must be far
off, which shows the tolerance can catch a routing or combine error.

Usage: python ep_moe_reference.py DIR [--device cuda|cpu]
Exit status 0 when both comparisons are within tolerance.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

MATCHED_REL_L2 = 1.5e-3
FP32_REL_L2 = 2e-2
WRONG_MIN_REL_L2 = 0.1


def load(path, dtype, shape):
    raw = np.fromfile(path, dtype=np.int16 if dtype == torch.bfloat16 else
                      {torch.int32: np.int32, torch.float32: np.float32}[dtype])
    t = torch.from_numpy(raw)
    if dtype == torch.bfloat16:
        t = t.view(torch.bfloat16)
    return t.reshape(shape)


def moe(x, ids, weights, w13, w2, ffn, matched):
    """x [T,H] bf16, ids/weights [T,K]; returns fp32 [T,H]."""
    tokens, top_k = ids.shape
    y = torch.zeros(tokens, top_k, x.shape[1], dtype=torch.float32, device=x.device)
    for e in range(len(w13)):
        t_idx, k_idx = torch.nonzero(ids == e, as_tuple=True)
        if t_idx.numel() == 0:
            continue
        xe = x[t_idx]
        if matched:
            h = xe @ w13[e]
            a = (F.silu(h[:, :ffn].float()) * h[:, ffn:].float()).bfloat16()
            y[t_idx, k_idx] = (a @ w2[e]).float()
        else:
            h = xe.float() @ w13[e].float()
            a = F.silu(h[:, :ffn]) * h[:, ffn:]
            y[t_idx, k_idx] = a @ w2[e].float()
    acc = torch.zeros(tokens, x.shape[1], dtype=torch.float32, device=x.device)
    for k in range(top_k):
        acc = acc + weights[:, k:k + 1] * y[:, k]
    return acc


def rel_l2(got, want):
    return (torch.linalg.vector_norm(got - want) / torch.linalg.vector_norm(want).clamp_min(1e-30)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    d = pathlib.Path(args.dir)
    meta = json.loads((d / "meta.json").read_text())
    world, tokens, experts = meta["world"], meta["tokens"], meta["experts"]
    top_k, hidden, ffn = meta["top_k"], meta["hidden"], meta["ffn"]
    dev = torch.device(args.device)
    # Its default lets cuBLAS reduce split-K partial sums in bf16 (~3e-3 noise).
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    w13 = [load(d / f"w13_{e}.bin", torch.bfloat16, (hidden, 2 * ffn)).to(dev) for e in range(experts)]
    w2 = [load(d / f"w2_{e}.bin", torch.bfloat16, (ffn, hidden)).to(dev) for e in range(experts)]

    ok = True
    for r in range(world):
        x = load(d / f"x_{r}.bin", torch.bfloat16, (tokens, hidden)).to(dev)
        ids = load(d / f"ids_{r}.bin", torch.int32, (tokens, top_k)).to(dev)
        weights = load(d / f"weights_{r}.bin", torch.float32, (tokens, top_k)).to(dev)
        got = load(d / f"out_{r}.bin", torch.bfloat16, (tokens, hidden)).to(dev).float()
        matched = moe(x, ids, weights, w13, w2, ffn, matched=True).bfloat16().float()
        golden = moe(x, ids, weights, w13, w2, ffn, matched=False)
        wrong = moe(x, ids, weights.roll(1, dims=1), w13, w2, ffn, matched=True)
        e_m, e_g, e_w = rel_l2(got, matched), rel_l2(got, golden), rel_l2(got, wrong)
        max_abs = (got - matched).abs().max().item()
        finite = bool(torch.isfinite(got).all())
        rank_ok = finite and e_m < MATCHED_REL_L2 and e_g < FP32_REL_L2 and (top_k == 1 or e_w > WRONG_MIN_REL_L2)
        ok &= rank_ok
        print(f"rank {r}: rel_l2 matched={e_m:.2e} fp32={e_g:.2e} rotated-weights={e_w:.2e} "
              f"max_abs(matched)={max_abs:.3e} ref_max={matched.abs().max().item():.3e} "
              f"{'PASS' if rank_ok else 'FAIL'}")
    print(f"thresholds: matched<{MATCHED_REL_L2} fp32<{FP32_REL_L2} rotated>{WRONG_MIN_REL_L2}")
    print(f"ep_moe_reference={'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
