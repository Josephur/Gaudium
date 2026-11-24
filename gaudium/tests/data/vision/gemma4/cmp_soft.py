#!/usr/bin/env python3
"""Compares the engine's image soft tokens (`--dump-vision-emb`, tokens x hidden BF16) with the
reference's (`make_gemma4_refs.py`, `<stem>.soft.bf16`).

    cmp_soft.py <engine.bf16> <reference.soft.bf16> <hidden>

Prints the relative error of the whole block (Frobenius), the worst row's relative error, the
mean and smallest row cosine, and the share of values that are bit-equal or one BF16 step away.
"""
import sys

import numpy as np


def load(path, hidden):
    raw = np.fromfile(path, dtype=np.uint16)
    if raw.size % hidden:
        sys.exit(f"{path}: {raw.size} values are not rows of {hidden}")
    f = (raw.astype(np.uint32) << 16).view(np.float32).reshape(-1, hidden)
    return raw.reshape(-1, hidden), f


def main():
    eng_path, ref_path, hidden = sys.argv[1], sys.argv[2], int(sys.argv[3])
    er, e = load(eng_path, hidden)
    rr, r = load(ref_path, hidden)
    if e.shape != r.shape:
        sys.exit(f"shapes differ: engine {e.shape}, reference {r.shape}")
    rel = np.linalg.norm(e - r) / max(np.linalg.norm(r), 1e-30)
    row_rel = np.linalg.norm(e - r, axis=1) / np.maximum(np.linalg.norm(r, axis=1), 1e-30)
    cos = (e * r).sum(1) / np.maximum(np.linalg.norm(e, axis=1) * np.linalg.norm(r, axis=1), 1e-30)
    equal = (er == rr).mean()
    # one BF16 step: neighbouring bit patterns of the same sign
    step = (np.abs(er.astype(np.int32) - rr.astype(np.int32)) <= 1).mean()
    print(
        f"rows {e.shape[0]} rel_err {rel:.4%} worst_row {row_rel.max():.4%} "
        f"cos_mean {cos.mean():.6f} cos_min {cos.min():.6f} equal {equal:.2%} within_one_step {step:.2%}"
    )


if __name__ == "__main__":
    main()
