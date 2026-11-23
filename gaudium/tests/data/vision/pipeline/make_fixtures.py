#!/usr/bin/env python3
"""Fixtures of the whole-pipeline test (`vision::media::tests`): files whose path from bytes to
patch rows needs more than a decoder. Run in this directory; needs the reference image library.

- `exif_o<k>.jpg` / `.png`: pictures stored turned, with the EXIF orientation that turns them back.
- `anim_*.gif` / `.webp`: animations at least one token (32 pixels) on a side, with one delay for
  all frames, for the video path (frame sampling, temporal patches, time labels).

Their reference fingerprints come from `../make_pipeline_refs.py --fingerprints` (stills) and
`--video --fingerprints` (animations) with the checkpoint's own processors.
"""
import numpy as np
from PIL import Image, ImageDraw


def picture(w, h, k=0):
    y, x = np.mgrid[0:h, 0:w]
    a = np.stack([((x + 5 * k) * 255 // w) % 256, y * 255 // h, (x * 3 + y * 5 + 11 * k) % 256], -1)
    return Image.fromarray(a.astype(np.uint8))


def frames(w, h, n, colours):
    out = []
    for k in range(n):
        im = picture(w, h, k)
        ImageDraw.Draw(im).ellipse([2 + k * (w - 16) // max(n - 1, 1), h // 3, 14 + k * (w - 16) // max(n - 1, 1), h // 3 + 12], fill=(255, 0, 0))
        out.append(im.quantize(colours) if colours else im)
    return out


def main():
    base = picture(40, 24)
    for o, ext, kw in [(6, "jpg", {"quality": 85}), (3, "jpg", {"quality": 85}), (8, "png", {}), (5, "png", {}), (7, "webp", {"quality": 80})]:
        ex = Image.Exif()
        ex[0x0112] = o
        base.save(f"exif_o{o}.{ext}", exif=ex.tobytes(), **kw)
    # 30 frames at 10 per second: 6 are kept, and one time label is a rounding tie (1.45 -> "1.4")
    f = frames(48, 32, 30, 8)
    f[0].save("anim_30f.gif", save_all=True, append_images=f[1:], duration=100, loop=0)
    # 7 frames at 2 per second: all kept, the last temporal patch shows its last frame twice
    f = frames(64, 48, 7, 16)
    f[0].save("anim_7f.gif", save_all=True, append_images=f[1:], duration=500, loop=0)
    f = frames(64, 48, 9, 0)
    f[0].save("anim_9f.webp", save_all=True, append_images=f[1:], duration=250, loop=0, lossless=True, method=6)
    f[0].save("anim_9f_lossy.webp", save_all=True, append_images=f[1:], duration=250, loop=0, quality=60, method=4)


if __name__ == "__main__":
    main()
