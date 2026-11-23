#!/usr/bin/env python3
"""Reference values of the Inkling 2 image input on the host: the processor's patch rows, the
bilinear position-table taps, the rotary positions and tables, and (with the checkpoint) the
resampled position rows and the class row.

usage: make_fixture.py <Inkling 2 model dir> <out .json> [--with-weights]

Images: the two PNG files next to this script (decoded by the reference image library) and two
generated thin images (`synthetic(w, h)` below, the same formula as the Rust test) whose sides are
not whole tiles. For each it writes:
  "grid": [tile rows, tile cols], "patch_grid": [rows, cols] of 20 x 20 patches,
  "pixels_fnv": FNV-1a 64 (16 hex digits) of the processor's `pixel_values` cut into the tower's patch rows
      (f32 little-endian, patch-row order: tile raster, a tile's 2 x 2 patches, y, x, channel),
  "taps_fnv" / "weights_fnv": of the position-table taps (i64) and weights (f32) per patch,
  "rope_pos_fnv": of the rotary positions (i64 pairs (h, w), class row first),
  "inv_freq": the f32 bits of the inverse frequencies,
  "rope_rows": {row: [cos 64 values, sin 64 values]} for a few sequence rows,
  with --with-weights also "pos_fnv" (the resampled position rows, BF16 bits, patch-row order)
  and "cls_fnv" (class + class position, BF16 bits).

The tracked tests in `engine::inkling2::vision::tests` check everything but "pos_fnv" / "cls_fnv";
the ignored test `pos_rows_match_the_reference` checks those against the checkpoint
(GAUDIUM_INKLING2_DIR). Needs the reference model library with its image processor, CPU only.
"""
import json
import os
import struct
import sys

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("HABANA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoConfig, AutoProcessor  # noqa: E402
from transformers.models.inkling2.modeling_inkling2 import Inkling2VisionRotaryEmbedding  # noqa: E402
from transformers.vision_utils import (  # noqa: E402
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def fnv1a64(data: bytes) -> str:
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def synthetic(w: int, h: int) -> Image.Image:
    a = np.zeros((h, w, 3), np.uint8)
    for y in range(h):
        for x in range(w):
            a[y, x] = ((x * 7 + y * 3) % 256, (x * 5 + 1) % 256, (y * 11 + 2) % 256)
    return Image.fromarray(a, "RGB")


def main():
    model, out = sys.argv[1], sys.argv[2]
    with_weights = "--with-weights" in sys.argv
    proc = AutoProcessor.from_pretrained(model)
    vcfg = AutoConfig.from_pretrained(model).vision_config
    rot = Inkling2VisionRotaryEmbedding(vcfg)
    table = cls = None
    if with_weights:
        from safetensors import safe_open

        with safe_open(os.path.join(model, "model-00001-of-00001.safetensors"), "pt") as f:
            table = f.get_tensor("model.visual.backbone.embeddings.position_embedding")
            cls = (
                f.get_tensor("model.visual.backbone.embeddings.class_embedding")
                + f.get_tensor("model.visual.backbone.embeddings.cls_position_embedding")
            )
    images = [
        ("img_200x120.png", Image.open(os.path.join(HERE, "img_200x120.png")).convert("RGB")),
        ("img_448x448.png", Image.open(os.path.join(HERE, "img_448x448.png")).convert("RGB")),
        ("synthetic_41x1", synthetic(41, 1)),
        ("synthetic_7x81", synthetic(7, 81)),
    ]
    res = {}
    for name, im in images:
        enc = proc.image_processor(images=[im], return_tensors="pt")
        pv, thw = enc["pixel_values"], enc["image_grid_thw"]
        rows, cols = int(thw[0, 1]), int(thw[0, 2])
        patches = pv.reshape(-1, 2, 20, 2, 20, 3).transpose(2, 3).reshape(-1, 1200)
        pgrid = thw * torch.tensor([1, 2, 2])
        idx, wts = get_vision_interpolation_indices_and_weights(
            pgrid, num_grid_per_side=32, mode="bilinear", align_corners=False, spatial_merge_size=2
        )
        pos = get_vision_position_ids(pgrid, 2) + 1
        pos = torch.cat([torch.zeros(1, 2, dtype=pos.dtype), pos])
        cos, sin = rot(torch.zeros(1), pos)
        n = pos.shape[0]
        sample = sorted({0, 1, 2, 5, n // 2, n - 1})
        rec = {
            "grid": [rows, cols],
            "patch_grid": [rows * 2, cols * 2],
            "pixels_fnv": fnv1a64(patches.float().numpy().astype("<f4").tobytes()),
            "taps_fnv": fnv1a64(idx.numpy().astype("<i8").tobytes()),
            "weights_fnv": fnv1a64(wts.float().numpy().astype("<f4").tobytes()),
            "rope_pos_fnv": fnv1a64(pos.numpy().astype("<i8").tobytes()),
            "inv_freq": [struct.unpack("<I", struct.pack("<f", float(v)))[0] for v in rot.inv_freq],
            "rope_rows": {
                str(r): [[float(v) for v in cos[r]], [float(v) for v in sin[r]]] for r in sample
            },
        }
        if table is not None:
            pe = (table[idx] * wts[:, :, None]).sum(1).to(torch.bfloat16)
            rec["pos_fnv"] = fnv1a64(pe.view(torch.int16).numpy().astype("<i2").tobytes())
            rec["cls_fnv"] = fnv1a64(cls.to(torch.bfloat16).view(torch.int16).numpy().astype("<i2").tobytes())
        res[name] = rec
    with open(out, "w") as f:
        json.dump(res, f, sort_keys=True, separators=(",", ":"))
        f.write("\n")


if __name__ == "__main__":
    main()
