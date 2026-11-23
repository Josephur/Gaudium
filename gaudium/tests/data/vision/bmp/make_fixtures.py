#!/usr/bin/env python3
"""Regenerates the BMP fixtures of the decoder tests and prints their reference hashes.

Run it from any directory; it writes the files next to itself. Every line it prints is one
entry of `FIXTURES` in `src/vision/decode/bmp.rs`: file name, width, height, and the FNV-1a 64
hash of what the reference image library returns for

    Image.open(path).convert("RGB").tobytes()

The files are written field by field here (not by an image library), so that each one holds
exactly the header, bit depth, masks, row order and compression its name says.
"""
import os
import random
import struct
import warnings

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 13, 7


def fnv1a64(data):
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def indices(w, h, top, seed):
    """Rows (top to bottom) of palette indices up to `top`: ramps, noise and hard edges."""
    rng = random.Random(seed)
    rows = []
    for y in range(h):
        row = []
        for x in range(w):
            kind = (x * 3 + y) % 7
            if kind < 3:
                row.append((x * top // max(1, w - 1) + y * top // max(1, h - 1)) // 2)
            elif kind < 5:
                row.append(rng.randrange(top + 1))
            else:
                row.append(top if ((x // 3 + y // 2) & 1) else 0)
        rows.append(row)
    return rows


def colours(w, h, seed):
    """Rows of (blue, green, red, alpha) bytes: ramps, noise, hard edges, an alpha pattern."""
    rng = random.Random(seed)
    rows = []
    for y in range(h):
        row = []
        for x in range(w):
            kind = (x * 3 + y) % 7
            if kind < 3:
                px = [x * 255 // max(1, w - 1), y * 255 // max(1, h - 1), (x * 37 + y * 11) & 255]
            elif kind < 5:
                px = [rng.randrange(256) for _ in range(3)]
            else:
                px = [255 if ((x // 3 + y // 2) & 1) else 0] * 3
            row.append(tuple(px) + ([0, 255, rng.randrange(256), 128][(x // 2 + y // 2) % 4],))
        rows.append(row)
    return rows


def pack(rows, bits, encode=bytes):
    """Packs rows into BMP row bytes, each padded to a multiple of four bytes."""
    out = []
    for row in rows:
        b = bytearray()
        if bits <= 8:
            per = 8 // bits
            for i in range(0, len(row), per):
                acc = 0
                for k in range(per):
                    acc = (acc << bits) | (row[i + k] if i + k < len(row) else 0)
                b.append(acc)
        else:
            for px in row:
                b += encode(px)
        b += bytes(-len(b) % 4)
        out.append(bytes(b))
    return out


def palette_bytes(colors, entry=4):
    return b"".join(bytes([b, g, r]) + bytes(entry - 3) for (r, g, b) in colors)


def random_palette(n, seed):
    rng = random.Random(seed)
    return [(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(n)]


def bmp(w, h, bits, rows=(), header=40, compression=0, top_down=False, masks=None, palette=b"",
        colors=0, data=None, gap=b"", offset=None):
    """One file. `rows`: packed rows, top to bottom. `data`: the pixel data as is (run-length
    files). `masks`: four channel masks, stored in the header when it has room for them and after
    a 40-byte header otherwise. `offset`: the pixel offset field when it is not the true one."""
    if data is None:
        data = b"".join(rows if top_down else reversed(rows))
    if header == 12:
        info = struct.pack("<IHHHH", 12, w, h, 1, bits)
    else:
        info = struct.pack("<IiiHHIIiiII", header, w, -h if top_down else h, 1, bits, compression,
                           len(data), 2835, 2835, colors, 0)
        m = struct.pack("<IIII", *(masks or (0, 0, 0, 0)))
        if header >= 52:
            info += m[:header - 40]
            info += bytes(header - len(info))
        elif masks:
            info += m[:12]
    start = 14 + len(info) + len(palette) + len(gap)
    head = b"BM" + struct.pack("<IHHI", start + len(data), 0, 0, start if offset is None else offset)
    return head + info + palette + gap + data


def run_length(rows, nibbles, seed):
    """RLE8 / RLE4 data for rows of indices (top to bottom): runs, literal runs with their
    padding, skips over index 0, an end-of-line code per row and the end-of-image code."""
    rng = random.Random(seed)
    out = bytearray()
    w = len(rows[0])
    for row in rows[::-1]:
        x = 0
        while x < w:
            run = 1
            while x + run < w and row[x + run] == row[x + (run % 2 if nibbles else 0)]:
                run += 1
            zeros = 0
            while x + zeros < w and row[x + zeros] == 0:
                zeros += 1
            if zeros >= 3:
                out += bytes([0, 2, zeros, 0])
                x += zeros
            elif w - x >= 3 and (run < 3 or rng.random() < 0.25):
                n = min(w - x, rng.randrange(3, 8))
                seg = row[x:x + n]
                if nibbles:
                    seg = seg + [0]
                    body = bytes((seg[i] << 4) | seg[i + 1] for i in range(0, n, 2))
                else:
                    body = bytes(seg)
                out += bytes([0, n]) + body + bytes(len(body) % 2)
                x += n
            else:
                second = row[x + 1] if run > 1 else 0
                out += bytes([run, (row[x] << 4) | second if nibbles else row[x]])
                x += run
        out += b"\x00\x00"
    return bytes(out + b"\x00\x01")


def fixtures():
    files = {}
    bgr = lambda px: bytes(px[:3])
    bgra = lambda px: bytes(px)
    word = lambda v: struct.pack("<H", v)
    pic = colours(W, H, 1)
    rng = random.Random(2)
    words = [[rng.randrange(65536) for _ in range(W)] for _ in range(H)]

    # palette images
    files["pal1"] = bmp(W, H, 1, pack(indices(W, H, 1, 3), 1), palette=palette_bytes(random_palette(2, 3)))
    files["pal1_black_white"] = bmp(W, H, 1, pack(indices(W, H, 1, 4), 1), palette=palette_bytes([(0, 0, 0), (255, 255, 255)]))
    files["pal1_white_black"] = bmp(W, H, 1, pack(indices(W, H, 1, 5), 1), palette=palette_bytes([(255, 255, 255), (0, 0, 0)]))
    files["pal1_black_only"] = bmp(W, H, 1, pack(indices(W, H, 1, 6), 1), palette=palette_bytes([(0, 0, 0)]), colors=1)
    files["pal4"] = bmp(W, H, 4, pack(indices(W, H, 15, 7), 4), palette=palette_bytes(random_palette(16, 7)))
    files["pal4_top_down"] = bmp(W, H, 4, pack(indices(W, H, 15, 8), 4), palette=palette_bytes(random_palette(16, 8)), top_down=True)
    files["pal8"] = bmp(W, H, 8, pack(indices(W, H, 255, 9), 8), palette=palette_bytes(random_palette(256, 9)))
    files["pal8_short"] = bmp(W, H, 8, pack(indices(W, H, 255, 10), 8), palette=palette_bytes(random_palette(40, 10)), colors=40)
    files["pal8_grey_ramp"] = bmp(W, H, 8, pack(indices(W, H, 255, 11), 8), palette=palette_bytes([(i, i, i) for i in range(256)]))
    files["pal8_grey_ramp_short"] = bmp(W, H, 8, pack(indices(W, H, 255, 12), 8), palette=palette_bytes([(i, i, i) for i in range(16)]), colors=16)
    files["pal8_core_header"] = bmp(W, H, 8, pack(indices(W, H, 255, 13), 8), header=12, palette=palette_bytes(random_palette(256, 13), 3))
    files["pal4_core_header"] = bmp(W, H, 4, pack(indices(W, H, 15, 14), 4), header=12, palette=palette_bytes(random_palette(16, 14), 3))
    # where the pixels start: offset 0, an offset that points at the palette, a gap
    rows8 = pack(indices(W, H, 255, 15), 8)
    pal8 = palette_bytes(random_palette(256, 15))
    files["pal8_offset_zero"] = bmp(W, H, 8, rows8, palette=pal8, offset=0)
    files["pal8_offset_header_end"] = bmp(W, H, 8, rows8, palette=pal8, offset=54)
    files["pal8_gap"] = bmp(W, H, 8, rows8, palette=pal8, gap=b"gap")
    files["pal8_no_last_padding"] = bmp(W, H, 8, rows8, palette=pal8)[:-3]
    # 16 bits
    files["rgb555"] = bmp(W, H, 16, pack(words, 16, word))
    files["rgb565_masks"] = bmp(W, H, 16, pack(words, 16, word), compression=3, masks=(0xF800, 0x7E0, 0x1F, 0))
    files["rgb555_masks_v4"] = bmp(W, H, 16, pack(words, 16, word), header=108, compression=3, masks=(0x7C00, 0x3E0, 0x1F, 0), top_down=True)
    # 24 bits
    files["rgb24"] = bmp(W, H, 24, pack(pic, 24, bgr))
    files["rgb24_top_down"] = bmp(W, H, 24, pack(pic, 24, bgr), top_down=True)
    files["rgb24_core_header"] = bmp(W, H, 24, pack(pic, 24, bgr), header=12)
    files["rgb24_os2_header"] = bmp(W, H, 24, pack(pic, 24, bgr), header=64)
    files["rgb24_masks"] = bmp(W, H, 24, pack(pic, 24, bgr), compression=3, masks=(0xFF0000, 0xFF00, 0xFF, 0))
    files["rgb24_1x1"] = bmp(1, 1, 24, pack(colours(1, 1, 16), 24, bgr))
    # 32 bits: no masks, then every byte order the masks can name
    files["rgb32"] = bmp(W, H, 32, pack(pic, 32, bgra))
    for name, header, masks in [("bgrx_after_header", 40, (0xFF0000, 0xFF00, 0xFF, 0)),
                                ("bgrx_v2", 52, (0xFF0000, 0xFF00, 0xFF, 0)),
                                ("xbgr_v3", 56, (0xFF000000, 0xFF0000, 0xFF00, 0)),
                                ("bgxr_v4", 108, (0xFF000000, 0xFF00, 0xFF, 0)),
                                ("abgr_v4", 108, (0xFF000000, 0xFF0000, 0xFF00, 0xFF)),
                                ("rgba_v5", 124, (0xFF, 0xFF00, 0xFF0000, 0xFF000000)),
                                ("bgra_v5", 124, (0xFF0000, 0xFF00, 0xFF, 0xFF000000)),
                                ("bgar_v5", 124, (0xFF000000, 0xFF00, 0xFF, 0xFF0000)),
                                ("zero_masks_v5", 124, (0, 0, 0, 0))]:
        files["rgb32_" + name] = bmp(W, H, 32, pack(pic, 32, bgra), header=header, compression=3, masks=masks)
    # run-length data
    few = [[v // 3 * 3 if (x // 4 + y) % 3 else v for x, v in enumerate(row)] for y, row in enumerate(indices(21, 9, 40, 17))]
    data8 = run_length(few, False, 17)
    files["rle8"] = bmp(21, 9, 8, palette=palette_bytes(random_palette(256, 17)), compression=1, data=data8)
    files["rle8_top_down"] = bmp(21, 9, 8, palette=palette_bytes(random_palette(256, 18)), compression=1, data=data8, top_down=True)
    files["rle8_odd_offset"] = bmp(21, 9, 8, palette=palette_bytes(random_palette(256, 19)), compression=1,
                                   data=run_length(few, False, 19), gap=b"\x00")
    files["rle8_grey_ramp"] = bmp(21, 9, 8, palette=palette_bytes([(i, i, i) for i in range(256)]), compression=1, data=data8)
    files["rle8_black_white"] = bmp(21, 9, 8, palette=palette_bytes([(0, 0, 0), (255, 255, 255)]), colors=2, compression=1, data=data8)
    few4 = [[v // 2 * 2 if (x // 4 + y) % 3 else v for x, v in enumerate(row)] for y, row in enumerate(indices(21, 9, 15, 20))]
    files["rle4"] = bmp(21, 9, 4, palette=palette_bytes(random_palette(16, 20)), compression=2, data=run_length(few4, True, 20))
    files["rle4_short_palette"] = bmp(21, 9, 4, palette=palette_bytes(random_palette(5, 21)), colors=5, compression=2,
                                      data=run_length(few4, True, 21))
    # the edge cases of run-length data the reference library reads its own way: a run past
    # the row's end, end-of-line codes on a complete and on an empty row, a literal run that
    # continues into the next row, a delta, and no end-of-image code
    odd = bytes([9, 7, 0, 0, 0, 0, 3, 5, 0, 0, 0, 7, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 2, 2, 1, 1, 9, 0, 0, 5, 8])
    files["rle8_edge_cases"] = bmp(5, 5, 8, palette=palette_bytes(random_palette(256, 22)), compression=1, data=odd)
    return files


def main():
    warnings.simplefilter("ignore")
    for name, data in fixtures().items():
        path = os.path.join(HERE, name + ".bmp")
        with open(path, "wb") as f:
            f.write(data)
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            print(f'    ("{name}.bmp", {rgb.width}, {rgb.height}, 0x{fnv1a64(rgb.tobytes()):016x}),')


if __name__ == "__main__":
    main()
