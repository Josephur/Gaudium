#!/usr/bin/env python3
"""Regenerates the JPEG decoder fixtures in this directory and prints their reference hashes.

    python3 make_fixtures.py            # rewrite the fixtures, then print the table
    python3 make_fixtures.py --hashes   # only print the table for the files that are here

Each table line is `(file, width, height, hash)`, where `hash` is the FNV-1a 64 hash of
`PIL.Image.open(file).convert("RGB").tobytes()`. The constants in the tests of
`src/vision/decode/jpeg.rs` are these lines. A file the reference library refuses, or one the
decoder must refuse, has no hash and is listed below the table.

Needs only the reference image library (the `PIL` import). Layouts that its writer cannot
produce (4:4:0, 4:1:1, mixed sampling factors, separate scans per component, YCCK) are written
by the small encoder in this file, which uses one flat Huffman code for every table. Its
floating-point DCT may round differently on another machine; the printed table then differs
and is what the tests must use.
"""
import io
import math
import os
import struct
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def fnv1a(data):
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def picture(w, h, seed):
    """Photo-like RGB test content: smooth gradients, noise and saturated sharp edges."""
    state = seed * 2654435761 + 12345
    out = bytearray()
    for y in range(h):
        for x in range(w):
            px = [128 + 100 * math.sin(x / (w / 3 + 1) + seed) + 60 * math.cos(y / (h / 2 + 1)),
                  255 * (0.6 * x / max(w - 1, 1) + 0.4 * y / max(h - 1, 1)),
                  128 + 120 * math.sin((x + 2 * y) / 7 + seed)]
            if w > 4 and x == w // 2:
                px = [255, 0, 0]
            if h > 4 and y == h // 3:
                px = [0, 0, 255]
            if w > 8 and h > 8 and w // 5 <= x < w // 3 and h // 2 <= y < h * 3 // 4:
                px = [255, 255, 0]
            for v in px:
                state = (state * 6364136223846793005 + 1442695040888963407) & (2 ** 64 - 1)
                noise = ((state >> 33) % 49) - 24
                out.append(max(0, min(255, int(v) + noise)))
    return Image.frombytes("RGB", (w, h), bytes(out))


def pil(im, mode="RGB", **options):
    buf = io.BytesIO()
    (im if mode == "RGB" else im.convert(mode)).save(buf, "JPEG", **options)
    return buf.getvalue()


# ---------------------------------------------------------------- marker helpers

def seg(marker, body=b""):
    return bytes([0xFF, marker]) + struct.pack(">H", len(body) + 2) + bytes(body)


def split(data):
    """(marker, start, end) of every segment; a scan segment includes its entropy data."""
    out = []
    i = 2
    while i < len(data):
        m = data[i + 1]
        if m == 0xD9:
            out.append((m, i, i + 2))
            break
        end = i + 2 + struct.unpack(">H", data[i + 2:i + 4])[0]
        if m == 0xDA:
            while not (data[end] == 0xFF and data[end + 1] != 0
                       and not 0xD0 <= data[end + 1] <= 0xD7):
                end += 1
        out.append((m, i, end))
        i = end
    return out


def without(data, drop):
    """Removes the segments for which `drop(index, marker)` holds."""
    out = data[:2]
    for n, (m, s, e) in enumerate(split(data)):
        if not drop(n, m):
            out += data[s:e]
    return out


# ---------------------------------------------------------------- the small encoder

ZIGZAG = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41,
          34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30,
          37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63]
COS = [[math.cos((2 * x + 1) * u * math.pi / 16) for u in range(8)] for x in range(8)]


def forward_dct(block, quant):
    """8x8 samples (rows) to 64 quantized coefficients in zigzag order."""
    natural = []
    for u in range(8):
        for v in range(8):
            s = sum((block[y][x] - 128) * COS[x][v] * COS[y][u]
                    for y in range(8) for x in range(8))
            s *= (0.5 if u else math.sqrt(0.125)) * (0.5 if v else math.sqrt(0.125))
            natural.append(int(round(s / quant[u * 8 + v])))
    return [natural[ZIGZAG[k]] for k in range(64)]


class Bits:
    def __init__(self):
        self.out = bytearray()
        self.acc = 0
        self.n = 0

    def put(self, value, count):
        self.acc = (self.acc << count) | (value & ((1 << count) - 1))
        self.n += count
        while self.n >= 8:
            byte = (self.acc >> (self.n - 8)) & 0xFF
            self.out.append(byte)
            if byte == 0xFF:
                self.out.append(0)
            self.n -= 8
        self.acc &= (1 << self.n) - 1

    def flush(self):
        if self.n:
            self.put((1 << (8 - self.n)) - 1, 8 - self.n)


def put_number(bits, value):
    size = abs(value).bit_length()
    bits.put(value if value >= 0 else value + (1 << size) - 1, size)


def put_ac_symbol(bits, symbol):
    # 255 symbols with 9-bit codes, the last one with a 10-bit code.
    if symbol < 255:
        bits.put(symbol, 9)
    else:
        bits.put(510, 10)


def put_block(bits, zz, pred):
    diff = zz[0] - pred
    bits.put(abs(diff).bit_length(), 5)
    put_number(bits, diff)
    run = 0
    last = max([k for k in range(1, 64) if zz[k]], default=0)
    for k in range(1, last + 1):
        if zz[k] == 0:
            run += 1
            continue
        while run > 15:
            put_ac_symbol(bits, 0xF0)
            run -= 16
        put_ac_symbol(bits, (run << 4) | abs(zz[k]).bit_length())
        put_number(bits, zz[k])
        run = 0
    if last < 63:
        put_ac_symbol(bits, 0)
    return zz[0]


def encode(im_planes, width, height, factors, scans=None, head=b"", quant_step=6):
    """Sequential JPEG from full-resolution planes (lists of rows), one per component.

    `factors`: (id, h, v) per component. `scans`: lists of component indices, default one
    interleaved scan."""
    hmax = max(f[1] for f in factors)
    vmax = max(f[2] for f in factors)
    quant = [quant_step + (k % 8 + k // 8) for k in range(64)]  # row-major
    mcus_x = -(-width // (8 * hmax))
    mcus_y = -(-height // (8 * vmax))
    coefs = []
    for plane, (_, h, v) in zip(im_planes, factors):
        fx, fy = hmax // h, vmax // v
        cols, rows = mcus_x * h * 8, mcus_y * v * 8
        small = []
        for y in range(rows):
            row = []
            for x in range(cols):
                acc = 0
                for dy in range(fy):
                    for dx in range(fx):
                        acc += plane[min(height - 1, y * fy + dy)][min(width - 1, x * fx + dx)]
                row.append(acc // (fx * fy))
            small.append(row)
        blocks = {}
        for by in range(rows // 8):
            for bx in range(cols // 8):
                block = [small[by * 8 + r][bx * 8:bx * 8 + 8] for r in range(8)]
                blocks[(by, bx)] = forward_dct(block, quant)
        coefs.append(blocks)

    out = b"\xff\xd8" + head
    out += seg(0xDB, bytes([0]) + bytes(quant[ZIGZAG[k]] for k in range(64)))
    body = struct.pack(">BHHB", 8, height, width, len(factors))
    for cid, h, v in factors:
        body += bytes([cid, (h << 4) | v, 0])
    out += seg(0xC0, body)
    dc = bytes([0x00]) + bytes([0, 0, 0, 0, 16] + [0] * 11) + bytes(range(16))
    counts = [0] * 16
    counts[8], counts[9] = 255, 1
    out += seg(0xC4, dc + bytes([0x10]) + bytes(counts) + bytes(range(256)))
    for which in scans or [list(range(len(factors)))]:
        body = bytes([len(which)])
        for ci in which:
            body += bytes([factors[ci][0], 0])
        out += seg(0xDA, body + bytes([0, 63, 0]))
        bits = Bits()
        preds = {ci: 0 for ci in which}
        if len(which) == 1:
            ci = which[0]
            _, h, v = factors[ci]
            bw = -(-(-(-width * h // hmax)) // 8)
            bh = -(-(-(-height * v // vmax)) // 8)
            order = [(ci, by, bx) for by in range(bh) for bx in range(bw)]
        else:
            order = [(ci, my * factors[ci][2] + dy, mx * factors[ci][1] + dx)
                     for my in range(mcus_y) for mx in range(mcus_x) for ci in which
                     for dy in range(factors[ci][2]) for dx in range(factors[ci][1])]
        for ci, by, bx in order:
            preds[ci] = put_block(bits, coefs[ci][(by, bx)], preds[ci])
        bits.flush()
        out += bytes(bits.out)
    return out + b"\xff\xd9"


def planes_of(im, mode):
    conv = im.convert(mode)
    w, h = conv.size
    data = conv.tobytes()
    n = len(conv.getbands())
    return [[[data[(y * w + x) * n + c] for x in range(w)] for y in range(h)] for c in range(n)]


ADOBE_YCCK = seg(0xEE, b"Adobe" + struct.pack(">HHHB", 100, 0, 0, 2))
JFIF = seg(0xE0, b"JFIF\0" + bytes([1, 1, 0, 0, 1, 0, 1, 0, 0]))

# ---------------------------------------------------------------- the fixtures


def build():
    a = picture(17, 13, 1)
    b = picture(33, 49, 2)
    files = {}
    # Baseline, the three layouts of the reference writer, and extreme qualities.
    files["base_420_17x13.jpg"] = pil(a, quality=75, subsampling=2)
    files["base_422_33x49.jpg"] = pil(b, quality=90, subsampling=1, optimize=True)
    files["base_444_q100_17x13.jpg"] = pil(a, quality=100, subsampling=0)
    files["base_420_q1_33x49.jpg"] = pil(b, quality=1, subsampling=2)
    # 16-bit quantization tables (extended sequential frame).
    files["base_wide_quant_17x13.jpg"] = pil(
        a, quality=50, subsampling=2, qtables=[[300 + 9 * i for i in range(64)]] * 2)
    # Progressive: spectral selection, successive approximation, end-of-band runs.
    files["prog_420_33x49.jpg"] = pil(b, quality=85, subsampling=2, progressive=True)
    files["prog_444_restart_17x13.jpg"] = pil(a, quality=92, subsampling=0, progressive=True,
                                             restart_marker_blocks=2)
    files["gray_17x13.jpg"] = pil(a, "L", quality=80)
    files["gray_prog_33x49.jpg"] = pil(b, "L", quality=70, progressive=True)
    # Four components: CMYK from the reference writer, YCCK with subsampled chroma.
    files["cmyk_17x13.jpg"] = pil(a, "CMYK", quality=85)
    ycc = planes_of(a, "YCbCr")
    black = planes_of(a.transpose(Image.Transpose.FLIP_LEFT_RIGHT), "L")
    files["ycck_420_17x13.jpg"] = encode(
        ycc + black, 17, 13, [(1, 2, 2), (2, 1, 1), (3, 1, 1), (4, 2, 2)], head=ADOBE_YCCK)
    # RGB stored without a colour transform.
    files["rgb_17x13.jpg"] = pil(a, quality=90, keep_rgb=True)
    files["restart_420_33x49.jpg"] = pil(b, quality=80, subsampling=2, restart_marker_blocks=3)
    files["one_pixel.jpg"] = pil(picture(1, 1, 3), quality=90)
    # Chroma only 2 samples wide: replicated instead of filtered.
    files["narrow_420_3x9.jpg"] = pil(picture(3, 9, 4), quality=90, subsampling=2)
    # Layouts from the small encoder.
    ycc_b = planes_of(picture(33, 13, 5), "YCbCr")
    files["base_440_17x13.jpg"] = encode(ycc, 17, 13, [(1, 1, 2), (2, 1, 1), (3, 1, 1)],
                                         head=JFIF)
    files["base_411_33x13.jpg"] = encode(ycc_b, 33, 13, [(1, 4, 1), (2, 1, 1), (3, 1, 1)],
                                         head=JFIF)
    # Chroma planes at different resolutions, one scan per component.
    files["mixed_scans_17x13.jpg"] = encode(ycc, 17, 13, [(1, 2, 2), (2, 1, 2), (3, 1, 1)],
                                            scans=[[0], [1], [2]], head=JFIF)
    # Luma at lower resolution than chroma.
    files["luma_small_17x13.jpg"] = encode(ycc, 17, 13, [(1, 1, 1), (2, 2, 2), (3, 2, 2)],
                                           head=JFIF)
    # No Huffman tables in the file: the standard ones apply.
    plain = pil(a, quality=60, subsampling=2)
    files["no_tables_17x13.jpg"] = without(plain, lambda _, m: m == 0xC4)
    # EXIF block that holds a thumbnail (a whole JPEG inside a segment).
    thumb = pil(picture(8, 8, 6), quality=30)
    tiff = b"MM\x00\x2a\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00" + thumb
    with_exif = plain[:2] + seg(0xE1, b"Exif\0\0" + tiff) + plain[2:]
    files["exif_thumbnail_17x13.jpg"] = with_exif
    # Progressive file whose last refinement scan is missing: must be refused.
    whole = pil(a, quality=85, subsampling=0, progressive=True)
    scans = [n for n, (m, _, _) in enumerate(split(whole)) if m == 0xDA]
    files["prog_incomplete_17x13.jpg"] = without(whole, lambda n, _: n == scans[-1])
    return files


def main():
    if "--hashes" not in sys.argv:
        for name, data in build().items():
            with open(os.path.join(HERE, name), "wb") as f:
                f.write(data)
    names = sorted(n for n in os.listdir(HERE) if n.endswith(".jpg"))
    total = 0
    refused = []
    for name in names:
        path = os.path.join(HERE, name)
        total += os.path.getsize(path)
        if name.startswith("prog_incomplete"):
            refused.append(f"{name}: decoder must refuse (the reference library smooths it)")
            continue
        try:
            im = Image.open(path)
            rgb = im.convert("RGB").tobytes()
        except Exception as e:  # noqa: BLE001
            refused.append(f"{name}: reference library refuses ({e})")
            continue
        print(f'    ("{name}", {im.width}, {im.height}, 0x{fnv1a(rgb):016x}),')
        exif = im.info.get("exif")
        if exif:
            print(f"    // {name}: EXIF block of {len(exif) - 6} bytes after \"Exif\\0\\0\", "
                  f"hash 0x{fnv1a(exif[6:]):016x}")
    for line in refused:
        print("    //", line)
    print(f"    // {len(names)} files, {total} bytes")


if __name__ == "__main__":
    main()
