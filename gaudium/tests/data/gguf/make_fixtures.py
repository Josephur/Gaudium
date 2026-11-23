#!/usr/bin/env python3
"""Regenerates the dequantisation reference vectors of the GGUF decoder tests.

Usage (numpy required; the reference package is imported from the path given, nothing is
installed):

    python -I make_fixtures.py <path to the gguf-py directory of the reference converter>

It writes one `<TYPE>.vec` file per ggml type next to itself:

    b"GQV1" | u32 type id | u32 block count | the raw blocks | the reference f32 values

The blocks are random bytes from a fixed seed, with the binary16 scale fields of each block
replaced by finite values of moderate size (a scale of inf or NaN only tests NaN arithmetic),
and, for the types the reference can quantise, also blocks quantised from random data. The
values are what the reference's pure-numpy dequantiser returns for those blocks. The Rust test
`gguf::dequant` tests compare the decoder with them bit for bit (NaN with NaN).
"""
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# (name, type id, weights per block, bytes per block, offsets of binary16 scale fields, blocks)
TYPES = [
    ("F16", 1, 1, 2, [], 64),
    ("BF16", 30, 1, 2, [], 64),
    ("Q4_0", 2, 32, 18, [0], 8),
    ("Q4_1", 3, 32, 20, [0, 2], 8),
    ("Q5_0", 6, 32, 22, [0], 8),
    ("Q5_1", 7, 32, 24, [0, 2], 8),
    ("Q8_0", 8, 32, 34, [0], 8),
    ("Q2_K", 10, 256, 84, [80, 82], 4),
    ("Q3_K", 11, 256, 110, [108], 4),
    ("Q4_K", 12, 256, 144, [0, 2], 4),
    ("Q5_K", 13, 256, 176, [0, 2], 4),
    ("Q6_K", 14, 256, 210, [208], 4),
    ("IQ4_NL", 20, 32, 18, [0], 8),
    ("IQ4_XS", 23, 256, 136, [0], 4),
    ("MXFP4", 39, 32, 17, [], 8),
    ("NVFP4", 40, 64, 36, [], 8),
    ("F8_E4M3", 43, 1, 1, [], 256),
]


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.path.insert(0, os.path.abspath(sys.argv[1]))
    from gguf import quants
    from gguf.constants import GGMLQuantizationType as Q

    rng = np.random.default_rng(20261123)
    for name, tid, bw, bb, scales, nblk in TYPES:
        qt = Q(tid)
        raw = rng.integers(0, 256, size=(nblk, bb), dtype=np.uint8)
        for off in scales:
            # finite binary16 of magnitude 2^-10 .. 2^2, either sign
            h = (rng.integers(0, 2, size=nblk) << 15) | (rng.integers(5, 17, size=nblk) << 10) | rng.integers(0, 1024, size=nblk)
            raw[:, off:off + 2] = h.astype("<u2").view(np.uint8).reshape(nblk, 2)
        if name == "F16":
            raw[:, 1] &= 0x7B  # finite
        if name == "F8_E4M3":
            raw = np.arange(256, dtype=np.uint8).reshape(256, 1)  # every code
        blocks = [raw]
        try:
            data = rng.standard_normal(size=(nblk // 2 or 1, bw), dtype=np.float32) * 0.3
            blocks.append(np.ascontiguousarray(quants.quantize(data, qt)).view(np.uint8).reshape(-1, bb))
        except (NotImplementedError, Exception):
            pass
        allb = np.concatenate(blocks, axis=0)
        if name == "F16":
            vals = allb.view("<f2").astype(np.float32).reshape(-1)
        else:
            vals = quants.dequantize(allb.reshape(-1), qt).astype(np.float32).reshape(-1)
        assert vals.size == allb.shape[0] * bw, name
        path = os.path.join(HERE, f"{name}.vec")
        with open(path, "wb") as f:
            f.write(b"GQV1" + struct.pack("<II", tid, allb.shape[0]))
            f.write(allb.tobytes())
            f.write(vals.astype("<f4").tobytes())
        print(f"{name}: {allb.shape[0]} blocks, {vals.size} values")


if __name__ == "__main__":
    main()
