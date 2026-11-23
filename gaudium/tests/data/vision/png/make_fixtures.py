#!/usr/bin/env python3
"""Regenerates the PNG fixtures of the decoder tests and prints their reference hashes.

Run it from any directory; it writes the files next to itself. Every line it prints is one
entry of `FIXTURES` in `src/vision/decode/png.rs`: file name, width, height, and the FNV-1a 64
hash of what the reference image library returns for

    Image.open(path).convert("RGB").tobytes()

The files are written chunk by chunk here (not by an image library), so that each one holds
exactly the colour type, bit depth, filters, interlace and chunk layout its name says. The
compressed bytes depend on the zlib build, the decoded pixels and the hashes do not.
"""
import os
import random
import struct
import warnings
import zlib

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SIGNATURE = b"\x89PNG\r\n\x1a\n"
CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
ADAM7 = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]
# The EXIF block of exif.png: a big-endian TIFF header with one entry, orientation = 6.
EXIF = b"MM\x00*\x00\x00\x00\x08\x00\x01\x01\x12\x00\x03\x00\x00\x00\x01\x00\x06\x00\x00\x00\x00\x00\x00"


def fnv1a64(data):
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def picture(w, h, color, depth, seed):
    """Rows of pixel tuples: gradients, noise, hard edges, and an alpha pattern."""
    rng = random.Random(seed)
    top = (1 << depth) - 1
    rows = []
    for y in range(h):
        row = []
        for x in range(w):
            ramp = (x * top // max(1, w - 1) + y * top // max(1, h - 1)) // 2
            edge = top if ((x // 3 + y // 4) & 1) else 0
            kind = (x * 3 + y) % 7

            def sample(k):
                if kind < 3:
                    return min(top, ramp + rng.randrange(top // 8 + 1)) if k != 1 else top - ramp
                if kind < 5:
                    return rng.randrange(top + 1)
                return edge

            alpha = [0, top, rng.randrange(top + 1), top // 2][(x // 2 + y // 2) % 4]
            if color == 0 and depth == 16:
                # mostly small values: the reference clips 16-bit greyscale at 255
                row.append((sample(0) % 640 if (x + y) % 4 else sample(0),))
            elif color == 0:
                row.append((sample(0),))
            elif color == 3:
                row.append((rng.randrange(top + 1) if kind else (x + y) & top,))
            elif color == 2:
                row.append((sample(0), sample(1), sample(2)))
            elif color == 4:
                row.append((sample(0), alpha))
            else:
                row.append((sample(0), sample(1), sample(2), alpha))
        rows.append(row)
    return rows


def pack_row(pixels, depth):
    if depth == 16:
        return b"".join(struct.pack(">H", s) for p in pixels for s in p)
    if depth == 8:
        return bytes(s for p in pixels for s in p)
    out = bytearray()
    acc = bits = 0
    for p in pixels:
        acc = (acc << depth) | p[0]
        bits += depth
        if bits == 8:
            out.append(acc)
            acc = bits = 0
    if bits:
        out.append(acc << (8 - bits))
    return bytes(out)


def paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else (b if pb <= pc else c)


def filter_row(kind, cur, prev, unit):
    out = bytearray()
    for i, x in enumerate(cur):
        a = cur[i - unit] if i >= unit else 0
        b = prev[i]
        c = prev[i - unit] if i >= unit else 0
        out.append((x - [0, a, b, (a + b) >> 1, paeth(a, b, c)][kind]) & 255)
    return bytes(out)


def scanlines(rows, w, h, color, depth, interlace, first_filter=0):
    """The filtered image data; the row filters cycle through all five types."""
    unit = max(1, CHANNELS[color] * depth // 8)
    out = bytearray()
    count = first_filter

    def emit(part):
        nonlocal count
        prev = None
        for r in part:
            cur = pack_row(r, depth)
            kind = count % 5
            count += 1
            out.append(kind)
            out.extend(filter_row(kind, cur, prev or bytes(len(cur)), unit))
            prev = cur

    if not interlace:
        emit(rows)
    else:
        for x0, y0, dx, dy in ADAM7:
            if x0 < w and y0 < h:
                emit([[rows[y][x] for x in range(x0, w, dx)] for y in range(y0, h, dy)])
    return bytes(out)


def ihdr(w, h, color, depth, interlace):
    return chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, color, 0, 0, interlace))


def png(w, h, color, depth, seed, interlace=0, before=(), after=(), pieces=None, level=6,
        strategy=zlib.Z_DEFAULT_STRATEGY, extra=b"", canvas=None):
    """One file. `before` / `after`: chunks around the image data. `pieces`: IDAT sizes, cycled.
    `extra`: bytes appended to the image data. `canvas`: the IHDR size when it is not (w, h)."""
    rows = picture(w, h, color, depth, seed)
    raw = scanlines(rows, w, h, color, depth, interlace, first_filter=seed) + extra
    packer = zlib.compressobj(level, zlib.DEFLATED, 15, 8, strategy)
    data = packer.compress(raw) + packer.flush()
    cw, ch = canvas or (w, h)
    out = SIGNATURE + ihdr(cw, ch, color, depth, interlace) + b"".join(before)
    if pieces is None:
        out += chunk(b"IDAT", data)
    else:
        at = i = 0
        while at < len(data):
            n = pieces[i % len(pieces)]
            out += chunk(b"IDAT", data[at:at + n])
            at += n
            i += 1
    return out + b"".join(after) + chunk(b"IEND", b"")


def palette(n, seed):
    rng = random.Random(seed)
    return chunk(b"PLTE", bytes(rng.randrange(256) for _ in range(n * 3)))


def frame_control(seq, w, h, x, y):
    return chunk(b"fcTL", struct.pack(">IIIIIHHBB", seq, w, h, x, y, 1, 10, 0, 0))


def frame_data(seq, w, h, color, depth, seed):
    raw = scanlines(picture(w, h, color, depth, seed), w, h, color, depth, 0)
    return chunk(b"fdAT", struct.pack(">I", seq) + zlib.compress(raw))


def fixtures():
    W, H = 17, 13
    files = {}
    # every colour type and bit depth, all five filters in each
    for color, depth, name in [(0, 1, "grey1"), (0, 2, "grey2"), (0, 4, "grey4"), (0, 8, "grey8"), (0, 16, "grey16"),
                               (2, 8, "rgb8"), (2, 16, "rgb16"), (4, 8, "greya8"), (4, 16, "greya16"),
                               (6, 8, "rgba8"), (6, 16, "rgba16")]:
        files[name] = png(W, H, color, depth, len(files))
    for depth in (1, 2, 4, 8):
        files[f"pal{depth}"] = png(W, H, 3, depth, len(files), before=[palette(1 << depth, depth)])
    # Adam7, on widths and heights that leave some passes empty or one pixel wide
    files["grey1_adam7"] = png(W, H, 0, 1, 20, interlace=1)
    files["grey16_adam7"] = png(W, H, 0, 16, 21, interlace=1)
    files["rgb8_adam7"] = png(W, H, 2, 8, 22, interlace=1)
    files["pal2_adam7"] = png(W, H, 3, 2, 23, interlace=1, before=[palette(4, 23)])
    files["pal4_adam7"] = png(W, H, 3, 4, 24, interlace=1, before=[palette(16, 24)])
    files["greya8_adam7"] = png(W, H, 4, 8, 25, interlace=1)
    files["rgba16_adam7"] = png(W, H, 6, 16, 26, interlace=1)
    files["rgb8_adam7_1x7"] = png(1, 7, 2, 8, 27, interlace=1)
    files["rgb8_adam7_3x2"] = png(3, 2, 2, 8, 28, interlace=1)
    files["grey8_1x1"] = png(1, 1, 0, 8, 29)
    files["pal1_7x1"] = png(7, 1, 3, 1, 30, before=[palette(2, 30)])
    # transparency chunks: a grey key, a colour key, palette alpha
    files["grey8_trns"] = png(W, H, 0, 8, 31, before=[chunk(b"tRNS", struct.pack(">H", 255))])
    files["rgb16_trns"] = png(W, H, 2, 16, 32, before=[chunk(b"tRNS", struct.pack(">HHH", 65535, 0, 65535))])
    files["pal8_trns"] = png(W, H, 3, 8, 33, before=[palette(256, 33), chunk(b"tRNS", bytes(range(0, 250, 5)))])
    # a palette shorter than the indices in the image
    files["pal8_short"] = png(W, H, 3, 8, 34, before=[palette(5, 34)])
    files["pal4_short"] = png(W, H, 3, 4, 35, before=[palette(3, 35)])
    # 16-bit greyscale around the value where the reference clips
    line = [(v,) for v in list(range(0, 520, 7)) + [0x7FFF, 0x8000, 0xFF00, 0xFFFF]]
    raw = b"\x00" + pack_row(line, 16)
    files["grey16_clip"] = SIGNATURE + ihdr(len(line), 1, 0, 16, 0) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    # image data layouts: many IDAT chunks (some empty), stored blocks, fixed Huffman codes,
    # data that goes on after the last row
    files["rgb8_idat_pieces"] = png(W, H, 2, 8, 36, pieces=[1, 0, 2, 3, 0, 0, 7, 50])
    files["rgba8_stored"] = png(W, H, 6, 8, 37, level=0, pieces=[100])
    files["rgb8_fixed"] = png(W, H, 2, 8, 38, strategy=zlib.Z_FIXED)
    files["grey8_long_data"] = png(W, H, 0, 8, 39, extra=bytes(range(200)))
    # ancillary chunks that must not change the pixels, and an EXIF block
    notes = [chunk(b"gAMA", struct.pack(">I", 45455)), chunk(b"sRGB", b"\x00"),
             chunk(b"cHRM", struct.pack(">8I", 31270, 32900, 64000, 33000, 30000, 60000, 15000, 6000)),
             chunk(b"iCCP", b"icc\x00\x00" + zlib.compress(b"not a profile")), chunk(b"bKGD", struct.pack(">HHH", 1, 2, 3)),
             chunk(b"pHYs", struct.pack(">IIB", 2835, 2835, 1)), chunk(b"tEXt", b"Comment\x00fixture"), chunk(b"prVt", b"private")]
    files["rgb8_ancillary"] = png(W, H, 2, 8, 40, before=notes, after=[chunk(b"tEXt", b"After\x00x")])
    files["exif"] = png(W, H, 2, 8, 41, before=[chunk(b"eXIf", EXIF)])
    # animations: frame 0 is the image data; a frame-control chunk before it gives its region
    anim = chunk(b"acTL", struct.pack(">II", 2, 0))
    files["apng_first_frame"] = png(W, H, 6, 8, 42, before=[anim, frame_control(0, W, H, 0, 0)],
                                    after=[frame_control(1, 9, 6, 3, 2), frame_data(2, 9, 6, 6, 8, 43)])
    files["apng_default_image"] = png(W, H, 2, 8, 44, before=[anim],
                                      after=[frame_control(0, 9, 6, 3, 2), frame_data(1, 9, 6, 2, 8, 45),
                                             frame_control(2, 9, 6, 0, 0), frame_data(3, 9, 6, 2, 8, 46)])
    files["apng_region_pal4"] = png(9, 6, 3, 4, 47, canvas=(W, H), before=[palette(16, 47), anim, frame_control(0, 9, 6, 5, 4)])
    files["apng_region_rgb8"] = png(9, 6, 2, 8, 48, canvas=(W, H), before=[anim, frame_control(0, 9, 6, 5, 4)])
    files["apng_region_adam7"] = png(9, 6, 2, 8, 49, canvas=(W, H), interlace=1, before=[anim, frame_control(0, 9, 6, 5, 4)])
    return files


def main():
    warnings.simplefilter("ignore")
    for name, data in fixtures().items():
        path = os.path.join(HERE, name + ".png")
        with open(path, "wb") as f:
            f.write(data)
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            print(f'    ("{name}.png", {rgb.width}, {rgb.height}, 0x{fnv1a64(rgb.tobytes()):016x}),')


if __name__ == "__main__":
    main()
