#!/usr/bin/env python3
"""Regenerates the WebP fixtures of this directory and prints their reference hashes.

    python3 make_fixtures.py            # rewrite the fixtures, then print the hashes
    python3 make_fixtures.py --hashes   # only print the hashes of the files that are here

The hash of a frame is the FNV-1a 64 of the RGB bytes the reference image library returns for
it: `im.seek(k); im.convert("RGB").tobytes()`. The table this script prints is the `FIXTURES`
table of the tests in `src/vision/decode/webp.rs` (and the single hashes used in `vp8.rs` and
`vp8l.rs`).

Three kinds of fixtures:
- written by the encoder of the reference image library (needs only that library and numpy);
- written by two other encoders through their command-line tools (`magick`, `vpxenc`), for
  the loop-filter, partition and loop-filter-delta variants that encoder does not write; they
  are skipped, and the tracked files kept, when the tools are not installed;
- hand-built: animations and alpha chunks assembled here chunk by chunk, and the `*_synth_*`
  files, which a bit-level test encoder wrote to reach syntax no stock encoder emits (delta
  segment quantisers, probability updates, skipped macroblocks, unused prefix-code groups,
  predictor modes 14 and 15, palette indices past the table). The synth files are kept as
  they are; this script only prints their hashes.

An encoder of another version may write different bytes for the same picture; the hashes then
change with the files, and the constants in the tests have to be updated together with them.
"""
import io
import os
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


# ---- pictures ---------------------------------------------------------------------------
def photo(w, h, seed, alpha=None):
    """Gradients, waves, noise and saturated rectangles (sharp chroma edges)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.zeros((h, w, 3))
    img[..., 0] = 255 * x / max(w - 1, 1)
    img[..., 1] = 255 * y / max(h - 1, 1)
    img[..., 2] = 127 + 127 * np.sin(x / 5.0 + seed) * np.cos(y / 7.0)
    img += rng.normal(0, 12, (h, w, 3))
    for _ in range(4):
        x0, y0 = rng.integers(0, w), rng.integers(0, h)
        x1 = x0 + rng.integers(1, max(2, w // 2 + 1))
        y1 = y0 + rng.integers(1, max(2, h // 2 + 1))
        img[y0:y1, x0:x1] = rng.choice([0, 255], 3)
    rgb = np.clip(img, 0, 255).astype(np.uint8)
    if alpha is None:
        return Image.fromarray(rgb, "RGB")
    return Image.fromarray(np.dstack([rgb, alpha_plane(w, h, seed, alpha)]), "RGBA")


def alpha_plane(w, h, seed, kind):
    rng = np.random.default_rng(seed + 1000)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    if kind == "grad":
        a = 255 * (x + y) / max(w + h - 2, 1)
    elif kind == "holes":
        a = np.full((h, w), 255.0)
        a[h // 4: h // 2 + 1, w // 4: w // 2 + 1] = 0
        a[: max(1, h // 8), :] = 128
    elif kind == "noise":
        a = rng.integers(0, 256, (h, w)).astype(np.float64)
    else:  # mixed: a ramp, a transparent and an opaque corner
        a = 255 * (x + y) / max(w + h - 2, 1)
        a[h // 3:, : w // 3 + 1] = 0
        a[: h // 3 + 1, w // 2:] = 255
    return np.clip(a, 0, 255).astype(np.uint8)


def few(w, h, seed, colors, alpha=False):
    """Blocks and scattered pixels of a small palette."""
    rng = np.random.default_rng(seed)
    pal = rng.integers(0, 256, (colors, 4 if alpha else 3)).astype(np.uint8)
    if alpha:
        pal[0, 3] = 0
        pal[-1, 3] = 255
    idx = (np.add.outer(np.arange(h) // 3, np.arange(w) // 5) + seed) % colors
    idx = np.where(rng.random((h, w)) < 0.15, rng.integers(0, colors, (h, w)), idx)
    return Image.fromarray(pal[idx], "RGBA" if alpha else "RGB")


# ---- encoders ---------------------------------------------------------------------------
def save(im, **kw):
    b = io.BytesIO()
    im.save(b, "WEBP", **kw)
    return b.getvalue()


def magick(im, quality, *defines):
    """A lossy file from the `magick` command-line tool, or None when it is missing."""
    if not shutil.which("magick"):
        return None
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "in.png"), os.path.join(tmp, "out.webp")
        im.save(src)
        args = ["magick", src, "-quality", str(quality)]
        for d in defines:
            args += ["-define", "webp:" + d]
        subprocess.run(args + [dst], check=True, capture_output=True)
        return open(dst, "rb").read()


def vpx_key_frame(im, *opts):
    """A key frame from the `vpxenc` command-line encoder (it writes loop-filter deltas),
    wrapped as a simple WebP file, or None when the tool is missing."""
    if not shutil.which("vpxenc"):
        return None
    w, h = im.size
    ycc = np.asarray(im.convert("YCbCr"))
    pad = np.pad(ycc, ((0, h & 1), (0, w & 1), (0, 0)), mode="edge")
    with tempfile.TemporaryDirectory() as tmp:
        raw, ivf = os.path.join(tmp, "in.yuv"), os.path.join(tmp, "out.ivf")
        with open(raw, "wb") as f:
            f.write(ycc[..., 0].tobytes() + pad[::2, ::2, 1].tobytes() + pad[::2, ::2, 2].tobytes())
        subprocess.run(["vpxenc", "--codec=vp8", "--i420", f"--width={w}", f"--height={h}", "--limit=1",
                        "--ivf", "--passes=1", *opts, "-o", ivf, raw],
                       check=True, capture_output=True, stdin=subprocess.DEVNULL)
        d = open(ivf, "rb").read()
    size = struct.unpack("<I", d[32:36])[0]
    return riff(chunk(b"VP8 ", d[44:44 + size]))


# ---- container --------------------------------------------------------------------------
def chunks(data):
    out, pos = [], 12
    while pos + 8 <= len(data):
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        out.append((data[pos:pos + 4], data[pos + 8:pos + 8 + size]))
        pos += 8 + size + (size & 1)
    return out


def chunk(tag, payload):
    return tag + struct.pack("<I", len(payload)) + payload + (b"\0" if len(payload) & 1 else b"")


def riff(body):
    return b"RIFF" + struct.pack("<I", len(body) + 4) + b"WEBP" + body


def le24(v):
    return struct.pack("<I", v)[:3]


def vp8x(flags, w, h):
    return chunk(b"VP8X", bytes([flags, 0, 0, 0]) + le24(w - 1) + le24(h - 1))


def image_chunks(data):
    """The ALPH / VP8 / VP8L chunks of a still file."""
    return b"".join(chunk(t, p) for t, p in chunks(data) if t in (b"ALPH", b"VP8 ", b"VP8L"))


def anmf(x, y, w, h, duration, dispose_to_background, no_blend, payload):
    flags = (1 if dispose_to_background else 0) | (2 if no_blend else 0)
    head = le24(x // 2) + le24(y // 2) + le24(w - 1) + le24(h - 1) + le24(duration) + bytes([flags])
    return chunk(b"ANMF", head + payload)


def animation(w, h, frames, background=0):
    return riff(vp8x(0x12, w, h) + chunk(b"ANIM", struct.pack("<IH", background, 0)) + b"".join(frames))


def payload(im, kind):
    """The chunks of one animation frame: lossless, lossy, or lossy with an alpha plane."""
    if kind == "ll":
        return image_chunks(save(im, lossless=True, exact=True))
    if kind == "ly":
        return image_chunks(save(im.convert("RGB"), quality=70))
    return image_chunks(save(im, quality=70, alpha_quality=90, exact=True))


def alpha_filter(plane, f):
    """The residual of an alpha plane under prediction filter f (1 horizontal, 2 vertical,
    3 gradient): what an ALPH chunk stores."""
    p = plane.astype(np.int32)
    h, w = p.shape
    out = np.zeros_like(p)
    for y in range(h):
        for x in range(w):
            if f == 0:
                pred = 0
            elif y == 0:
                pred = p[0, x - 1] if x else 0
            elif f == 1:
                pred = p[y, x - 1] if x else p[y - 1, 0]
            elif f == 2:
                pred = p[y - 1, x]
            elif x == 0:
                pred = p[y - 1, 0]
            else:
                pred = int(np.clip(p[y, x - 1] + p[y - 1, x] - p[y - 1, x - 1], 0, 255))
            out[y, x] = (p[y, x] - pred) & 255
    return out.astype(np.uint8)


def alpha_animation(f, lossless):
    """Two frames of 12 x 9: an opaque lossy one, and a lossy one with a hand-built alpha chunk
    (raw or lossless-coded, filter f) blended over it, so that the alpha plane shows in RGB."""
    w, h = 12, 9
    plane = alpha_filter(alpha_plane(w, h, f, ["grad", "holes", "noise", "mixed"][f]), f)
    if lossless:
        green = Image.fromarray(np.dstack([np.zeros_like(plane), plane, np.zeros_like(plane)]), "RGB")
        body = dict(chunks(save(green, lossless=True)))[b"VP8L"][5:]
    else:
        body = plane.tobytes()
    alph = chunk(b"ALPH", bytes([(1 if lossless else 0) | (f << 2)]) + body)
    under = image_chunks(save(photo(w, h, 40 + f), quality=60))
    over = image_chunks(save(photo(w, h, 50 + f), quality=60))
    return animation(w, h, [anmf(0, 0, w, h, 40, False, False, under), anmf(0, 0, w, h, 60, False, False, alph + over)])


def mux_animation(kinds, seed):
    """Six frames on a 24 x 18 canvas: full and partial rectangles, every blend / dispose pair."""
    rects = [(0, 0, 24, 18), (4, 2, 12, 10), (10, 6, 14, 12), (0, 0, 24, 18), (2, 4, 9, 7), (8, 0, 16, 18)]
    flags = [(0, 0), (1, 0), (0, 0), (0, 1), (1, 1), (1, 0)]
    frames = []
    for j, ((x, y, w, h), (dispose, no_blend)) in enumerate(zip(rects, flags)):
        im = few(w, h, seed + j, 6, alpha=True) if kinds[j] == "ll" else photo(w, h, seed + j, ["mixed", "holes", "grad"][j % 3])
        frames.append(anmf(x, y, w, h, 10 * (j + 1), dispose, no_blend, payload(im, kinds[j])))
    return animation(24, 18, frames, background=0x80ff4020)


def build():
    """name -> bytes of every fixture this script can write (None: tool missing, keep the file)."""
    f = {}
    # lossy, the encoder of the reference library: normal loop filter, four segments
    f["lossy_q75_33x47"] = save(photo(33, 47, 1), quality=75, method=4)
    f["lossy_q20_17x17"] = save(photo(17, 17, 2), quality=20, method=6)
    rng = np.random.default_rng(3)
    blocks = Image.fromarray(rng.integers(0, 2, (16, 16, 3)).astype(np.uint8) * 255, "RGB")
    f["lossy_q100_edges_16x16"] = save(blocks, quality=100, method=6)
    for w, h in ((1, 1), (2, 3), (9, 1), (1, 16)):
        f[f"lossy_{w}x{h}"] = save(photo(w, h, 4 + w + h), quality=60)
    # lossy, other encoders: simple filter with sharpness and several partitions; strong filter
    # at the highest sharpness; loop-filter deltas
    f["lossy_simple_filter_64x48"] = magick(photo(64, 48, 5), 40, "filter-type=0", "filter-strength=60",
                                            "filter-sharpness=3", "segments=4", "partitions=3", "auto-filter=false")
    f["lossy_sharp7_50x50"] = magick(photo(50, 50, 6), 30, "filter-type=1", "filter-strength=100",
                                     "filter-sharpness=7", "segments=2", "partitions=2", "auto-filter=false")
    f["lossy_filter_deltas_48x40"] = vpx_key_frame(photo(48, 40, 7), "--good", "--cpu-used=3", "--min-q=20",
                                                   "--max-q=40", "--token-parts=2", "--sharpness=2")
    # lossless: transforms and colour cache on a photo, palettes of 2 / 4 / 16 / 40 colours
    # (8, 4, 2 and 1 pixels per coded pixel), backward references on a tiled picture
    f["ll_photo_40x30"] = save(photo(40, 30, 8), lossless=True, quality=75, method=4)
    f["ll_photo_alpha_31x16"] = save(photo(31, 16, 9, "mixed"), lossless=True, quality=90, method=5, exact=True)
    for colors in (2, 4, 16, 40):
        f[f"ll_pal{colors}_33x9"] = save(few(33, 9, colors, colors), lossless=True, quality=60, method=4)
    f["ll_1x1"] = save(Image.new("RGB", (1, 1), (10, 200, 30)), lossless=True)
    tile = np.random.default_rng(10).integers(0, 256, (7, 9, 3)).astype(np.uint8)
    f["ll_tiles_45x35"] = save(Image.fromarray(np.tile(tile, (5, 5, 1)), "RGB"), lossless=True, quality=90, method=5)
    # lossy with an alpha plane, and metadata
    f["la_24x18"] = save(photo(24, 18, 11, "mixed"), quality=60, alpha_quality=80, exact=True)
    exif = Image.Exif()
    exif[0x0112] = 6
    f["lossy_exif_20x12"] = save(photo(20, 12, 12), quality=60, exif=exif.tobytes())
    # an extended still whose header announces neither alpha nor EXIF: the alpha plane and the
    # EXIF chunk are there but not used, and unknown chunks are skipped
    parts = dict(chunks(f["la_24x18"]))
    f["still_unannounced_24x18"] = riff(vp8x(0, 24, 18) + chunk(b"JUNK", b"abc") + chunk(b"ALPH", parts[b"ALPH"])
                                        + chunk(b"VP8 ", parts[b"VP8 "]) + chunk(b"EXIF", exif.tobytes())
                                        + chunk(b"zzzz", b""))
    # animations from the reference library's encoder
    frames = [photo(20, 15, 20 + k, ["grad", "holes", "mixed", "noise"][k]) for k in range(4)]
    for name, kw in (("anim_ll_20x15", dict(lossless=True)), ("anim_lossy_20x15", dict(quality=50))):
        b = io.BytesIO()
        frames[0].save(b, "WEBP", save_all=True, append_images=frames[1:], duration=[30, 0, 100, 65535], **kw)
        f[name] = b.getvalue()
    # hand-built animations
    f["anim_mux_ll"] = mux_animation(["ll"] * 6, 30)
    f["anim_mux_mixed"] = mux_animation(["la", "ly", "ll", "la", "ly", "la"], 40)
    small = photo(10, 8, 13, "grad")
    f["anim_first_small"] = animation(20, 14, [anmf(6, 4, 10, 8, 77, False, False, payload(small, "ll")),
                                               anmf(0, 0, 10, 8, 5, False, False, payload(small, "la"))])
    for flt in range(4):
        f[f"anim_alpha_raw_f{flt}"] = alpha_animation(flt, False)
        f[f"anim_alpha_ll_f{flt}"] = alpha_animation(flt, True)
    return f


# ---- reference hashes -------------------------------------------------------------------
def fnv1a(data):
    h = 0xcbf29ce484222325
    for b in data:
        h = ((h ^ b) * 0x100000001b3) & 0xffffffffffffffff
    return h


def describe(path):
    im = Image.open(path)
    frames = []
    for k in range(getattr(im, "n_frames", 1)):
        im.seek(k)
        rgb = im.convert("RGB").tobytes()
        frames.append((im.info.get("duration", 0), fnv1a(rgb)))
    alpha = None
    if len(frames) == 1 and im.mode == "RGBA":
        im.seek(0)
        alpha = fnv1a(im.getchannel("A").tobytes())
    return im.size, frames, im.info.get("exif"), alpha


def main():
    if "--hashes" not in sys.argv:
        for name, data in build().items():
            path = os.path.join(HERE, name + ".webp")
            if data is None:
                print(f"// {name}: encoder tool not installed, file kept", file=sys.stderr)
                continue
            with open(path, "wb") as out:
                out.write(data)
    total = 0
    for name in sorted(os.listdir(HERE)):
        if not name.endswith(".webp"):
            continue
        path = os.path.join(HERE, name)
        total += os.path.getsize(path)
        (w, h), frames, exif, alpha = describe(path)
        rows = ", ".join(f"({d}, 0x{hsh:016x})" for d, hsh in frames)
        print(f'    fixture!("{name}", {w}, {h}, [{rows}]),')
        if exif:
            print(f"    // {name}: EXIF block of {len(exif)} bytes, FNV-1a 0x{fnv1a(exif):016x}")
        if alpha is not None:
            print(f"    // {name}: alpha plane FNV-1a 0x{alpha:016x}")
    print(f"// {total} bytes of fixtures", file=sys.stderr)


if __name__ == "__main__":
    main()
