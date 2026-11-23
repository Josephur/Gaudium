#!/usr/bin/env python3
"""Regenerates the GIF fixtures of this directory and prints their reference hashes.

Every file is built byte by byte here (no encoder library), so the fixtures do not change with
the tools installed. The hashes are FNV-1a 64 of what the reference image library shows for
each frame:

    im = Image.open(path); im.seek(k); im.convert("RGB").tobytes(); im.info.get("duration", 0)

Run it in this directory: it rewrites the *.gif files and prints the table that
`gaudium/src/vision/decode/gif.rs` holds in its tests, followed by the hashes of the two
files that test builds in memory (they are larger than a fixture should be).
"""
import io
import os
import struct
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- content

class Lcg:
    """A fixed random sequence (the same on every machine)."""

    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def next(self, n):
        self.s = (self.s * 1664525 + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) % n


def cube(levels, entries):
    """A colour table: a levels^3 colour cube, filled up with greys to `entries`."""
    t = []
    for r in range(levels):
        for g in range(levels):
            for b in range(levels):
                t += [r * 255 // (levels - 1), g * 255 // (levels - 1), b * 255 // (levels - 1)]
    k = 0
    while len(t) < entries * 3:
        v = (k * 37 + 11) % 256
        t += [v, v, 255 - v]
        k += 1
    return bytes(t[:entries * 3])


def scene(w, h, levels, t=0, seed=1, hole=None):
    """Indices into a `cube(levels, ..)` table: gradients, noise, a box outline and a diagonal.
    With `hole`, a pattern of that index (the transparent one) is cut into it."""
    rnd = Lcg(seed * 7919 + t)
    px = []
    for y in range(h):
        for x in range(w):
            r = (x * 255 // max(w - 1, 1) + 37 * t) % 256
            g = (y * 255 // max(h - 1, 1) + 61 * t) % 256
            b = ((x + y) * 9 + 23 * t) % 256
            r = min(255, max(0, r + rnd.next(49) - 24))
            g = min(255, max(0, g + rnd.next(49) - 24))
            b = min(255, max(0, b + rnd.next(49) - 24))
            if x in (w // 5, w - 1 - w // 5) and h // 5 <= y <= h - 1 - h // 5:
                r, g, b = 255, 255, 255
            if y in (h // 5, h - 1 - h // 5) and w // 5 <= x <= w - 1 - w // 5:
                r, g, b = 0, 0, 0
            if x * h == y * w or x * h == y * w + h:
                r, g, b = 255, 0, 0
            q = lambda v: (v * (levels - 1) + 127) // 255
            i = (q(r) * levels + q(g)) * levels + q(b)
            if hole is not None and ((x // 2 + y // 2 + t) % 3 == 0 or (x + 2 * y + t) % 7 == 0):
                i = hole
            px.append(i)
    return px


def noise(count, maxv, seed):
    rnd = Lcg(seed)
    return [rnd.next(maxv) for _ in range(count)]


# ---------------------------------------------------------------- GIF writer

def pack_codes(codes):
    """codes: (value, bits) pairs, packed least significant bit first."""
    acc = n = 0
    out = bytearray()
    for value, bits in codes:
        acc |= value << n
        n += bits
        while n >= 8:
            out.append(acc & 255)
            acc >>= 8
            n -= 8
    if n:
        out.append(acc & 255)
    return bytes(out)


def lzw(pixels, bits, clear_every=None, keep_full=None, end=True):
    """GIF LZW. `clear_every`: a clear code after that many pixels. `keep_full`: how many
    codes are written with a full table before the clear code (None: clear at once)."""
    clear = 1 << bits
    out = [(clear, bits + 1)]
    size, table, nxt, full = bits + 1, {}, clear + 2, 0

    def emit(string):
        nonlocal size, nxt
        out.append((string[0] if len(string) == 1 else table[string], size))

    def grow(string):
        nonlocal size, nxt, table, full
        if nxt < 4096:
            if string is not None:
                table[string] = nxt
            if nxt == (1 << size) and size < 12:
                size += 1
            nxt += 1
        else:
            full += 1
            if keep_full is None or full > keep_full:
                reset()

    def reset():
        nonlocal size, nxt, table, full
        out.append((clear, size))
        size, table, nxt, full = bits + 1, {}, clear + 2, 0

    cur = None
    for n, px in enumerate(pixels, 1):
        if cur is None:
            cur = (px,)
        elif cur + (px,) in table:
            cur = cur + (px,)
        else:
            emit(cur)
            grow(cur + (px,))
            cur = (px,)
        if clear_every and n % clear_every == 0:
            emit(cur)
            grow(None)
            reset()
            cur = None
    if cur is not None:
        emit(cur)
        grow(None)
    if end:
        out.append((clear + 1, size))
    return pack_codes(out)


def blocks(data, size=255):
    out = bytearray()
    for i in range(0, len(data), size):
        out.append(len(data[i:i + size]))
        out += data[i:i + size]
    return bytes(out) + b"\x00"


def table_bits(table):
    n = len(table) // 3
    assert n in (2, 4, 8, 16, 32, 64, 128, 256)
    return n.bit_length() - 2


def screen(w, h, table=None, background=0, version=b"GIF89a"):
    flags = 0 if table is None else 0x80 | table_bits(table) | (table_bits(table) << 4)
    return version + struct.pack("<HHBBB", w, h, flags, background, 0) + (table or b"")


def control(delay=0, transparent=None, disposal=0):
    flags = (disposal << 2) | (transparent is not None)
    return b"\x21\xf9\x04" + struct.pack("<BHB", flags, delay, transparent or 0) + b"\x00"


def image(x, y, w, h, bits, data, table=None, interlace=False, block=255):
    flags = (0x40 if interlace else 0) | (0 if table is None else 0x80 | table_bits(table))
    return (b"\x2c" + struct.pack("<HHHHB", x, y, w, h, flags) + (table or b"") + bytes([bits])
            + (data if isinstance(data, memoryview) else blocks(data, block)))


LOOP = b"\x21\xff\x0bNETSCAPE2.0\x03\x01\x00\x00\x00"
COMMENT = b"\x21\xfe" + blocks(b"made by make_fixtures.py")
TEXT = b"\x21\x01\x0c" + struct.pack("<HHHHBBBB", 0, 0, 8, 8, 8, 8, 1, 0) + blocks(b"text")
APP = b"\x21\xff\x0bFIXTURE 1.0" + blocks(bytes(range(40)), 16)
END = b"\x3b"


# ---------------------------------------------------------------- the fixtures

def fixtures():
    c2, c8, c16 = cube(2, 2), cube(2, 8), cube(2, 16)
    c64, c256 = cube(4, 64), cube(6, 256)
    gray16 = bytes(v for i in range(16) for v in (i, i, i))
    f = {}

    # stills
    f["still_256.gif"] = (screen(31, 23, c256, version=b"GIF87a")
                          + image(0, 0, 31, 23, 8, lzw(scene(31, 23, 6), 8)) + END)
    f["still_interlaced.gif"] = (screen(29, 21, c8) + COMMENT
                                 + image(0, 0, 29, 21, 3, lzw(scene(29, 21, 2), 3), interlace=True) + END)
    f["still_two_colours.gif"] = (screen(40, 9, c2)
                                  + image(0, 0, 40, 9, 2, lzw([(x * y + x // 3) % 2 for y in range(9) for x in range(40)], 2))
                                  + END)
    f["still_1x1.gif"] = screen(1, 1, c8) + image(0, 0, 1, 1, 3, lzw([5], 3)) + END
    f["still_1x9_interlaced.gif"] = (screen(1, 9, c8)
                                     + image(0, 0, 1, 9, 3, lzw([1, 2, 3, 4, 5, 6, 7, 0, 1], 3), interlace=True) + END)
    f["still_7x1.gif"] = screen(7, 1, c8) + image(0, 0, 7, 1, 3, lzw([7, 6, 5, 4, 3, 2, 1], 3), interlace=True) + END
    f["still_local_only.gif"] = (screen(19, 14)
                                 + image(0, 0, 19, 14, 6, lzw(scene(19, 14, 4, seed=2), 6), table=c64) + END)
    f["still_no_table.gif"] = screen(16, 10) + image(0, 0, 16, 10, 8, lzw(noise(160, 256, 3), 8)) + END
    # a local grey ramp over a global colour table shows the global colours
    f["still_local_gray.gif"] = (screen(12, 8, c16)
                                 + image(0, 0, 12, 8, 4, lzw(noise(96, 16, 4), 4), table=gray16) + END)
    # frame smaller than the screen, with a transparent index: the canvas takes its colour
    f["still_offset_transparent.gif"] = (screen(24, 16, c64, background=9) + control(transparent=37)
                                         + image(7, 5, 10, 8, 6, lzw(scene(10, 8, 4, hole=37), 6)) + END)
    # frame reaching outside the screen: the canvas is 15x11
    f["still_outside_screen.gif"] = (screen(8, 8, c64, background=3)
                                     + image(3, 5, 12, 6, 6, lzw(scene(12, 6, 4, seed=5), 6)) + END)
    # the table fills (4095 is the last code), 60 codes go on with the full table, then a
    # clear code starts a new one (no colour table: grey levels)
    f["still_table_full.gif"] = (screen(66, 62)
                                 + image(0, 0, 66, 62, 8, lzw(noise(66 * 62, 256, 6), 8, keep_full=60)) + END)
    # sub-blocks of 1 and 7 bytes, clear codes inside the data, no end code, no trailer
    f["still_sub_blocks.gif"] = (screen(20, 13, c64) + b"\x00"
                                 + image(0, 0, 20, 13, 6, lzw(scene(20, 13, 4, seed=7), 6, clear_every=23, end=False), block=1)[:-1]
                                 )
    px = scene(20, 13, 4, seed=8)
    data = lzw(px, 6, clear_every=50)
    # empty sub-blocks in front of and inside the data do not end it
    chained = b"\x00\x00" + blocks(data[:31], 7)[:-1] + b"\x00" + blocks(data[31:], 7)
    f["still_empty_blocks.gif"] = screen(20, 13, c64) + image(0, 0, 20, 13, 6, memoryview(chained)) + END

    # every disposal method with transparency and offsets, a local table, a frame without a
    # control extension, the extensions that are skipped
    a = bytearray(screen(28, 20, c64, background=9) + LOOP + COMMENT)
    a += control(5, None, 1) + image(0, 0, 28, 20, 6, lzw(scene(28, 20, 4, 0), 6))
    a += control(10, 5, 2) + image(3, 2, 12, 9, 6, lzw(scene(12, 9, 4, 1, hole=5), 6))
    a += TEXT + control(0, None, 2) + image(14, 8, 10, 10, 6, lzw(scene(10, 10, 4, 2), 6))
    a += control(7, 40, 3) + APP + image(6, 11, 16, 7, 6, lzw(scene(16, 7, 4, 3, hole=40), 6))
    a += control(65535, 2, 3) + image(18, 1, 9, 12, 3, lzw(scene(9, 12, 2, 4, hole=2), 3), table=c8)
    a += image(1, 10, 8, 8, 6, lzw(scene(8, 8, 4, 5), 6), interlace=True)
    a += control(3, 17, 0) + image(10, 3, 11, 13, 6, lzw(scene(11, 13, 4, 6, hole=17), 6))
    a += control(4, 63, 1) + image(0, 0, 28, 20, 6, lzw(scene(28, 20, 4, 7, hole=63), 6), interlace=True)
    f["anim_dispose.gif"] = bytes(a) + END

    # a transparent first frame smaller than the screen, restored to the background
    a = bytearray(screen(20, 16, c16, background=6))
    a += control(2, 3, 2) + image(5, 4, 9, 7, 4, lzw(scene(9, 7, 2, 0, hole=3), 4))
    a += control(4, None, 3) + image(0, 0, 6, 6, 4, lzw(noise(36, 16, 9), 4))
    a += control(6, 1, 2) + image(10, 8, 10, 8, 4, lzw(scene(10, 8, 2, 2, hole=1), 4))
    a += control(8, 200, 0) + image(2, 9, 7, 5, 4, lzw(noise(35, 16, 10), 4))
    f["anim_first_small.gif"] = bytes(a) + END

    # "restore to previous" on a transparent first frame fills with the transparent colour
    a = bytearray(screen(14, 10, c16))
    a += control(1, 12, 3) + image(2, 1, 9, 7, 4, lzw(scene(9, 7, 2, 0, hole=12), 4))
    a += control(2, 12, 1) + image(6, 4, 8, 6, 4, lzw(scene(8, 6, 2, 1, hole=12), 4))
    a += control(3, None, 0) + image(0, 7, 5, 3, 4, lzw(noise(15, 16, 11), 4))
    f["anim_first_previous.gif"] = bytes(a) + END

    # a grey-ramp global table (so: grey levels), then a frame with colours of its own
    a = bytearray(screen(18, 12, gray16, background=11))
    a += control(1, None, 3) + image(0, 0, 18, 12, 4, lzw(noise(216, 16, 12), 4))
    a += control(2, 2, 2) + image(4, 3, 9, 6, 4, lzw(scene(9, 6, 2, 1, hole=2), 4))
    a += control(3, 7, 3) + image(8, 5, 10, 7, 4, lzw(noise(70, 16, 13), 4))
    a += control(4, 1, 2) + image(1, 1, 8, 8, 3, lzw(scene(8, 8, 2, 3, hole=1), 3), table=c8)
    a += control(5, 9, 2) + image(6, 2, 11, 9, 4, lzw(noise(99, 16, 14), 4))
    # indices past the end of an 8-entry table (the transparent one too)
    a += control(6, 13, 2) + image(0, 4, 12, 8, 4, lzw(noise(96, 16, 15), 4), table=c8)
    a += control(7, None, 1) + image(9, 0, 9, 5, 4, lzw(noise(45, 16, 16), 4), table=gray16)
    f["anim_gray.gif"] = bytes(a) + END

    # no table at all in the first frame, a local table later, with a transparent first frame:
    # the reference shows frame 0 and fails on frame 1
    a = bytearray(screen(6, 4))
    a += control(1, 2, 0) + image(0, 0, 6, 4, 2, lzw(noise(24, 4, 17), 2))
    a += control(2, None, 0) + image(0, 0, 6, 4, 2, lzw(noise(24, 4, 18), 2), table=cube(2, 4))
    f["anim_table_after_none.gif"] = bytes(a) + END

    # no global table; local tables of 4 to 256 entries with code sizes 2 to 8
    a = bytearray(screen(17, 11) + LOOP)
    for k, (levels, entries) in enumerate([(1, 4), (2, 8), (2, 16), (3, 32), (4, 64), (5, 128), (6, 256)]):
        bits = k + 2
        px = noise(17 * 11, entries, 20 + k) if levels == 1 else scene(17, 11, levels, k, seed=30)
        table = cube(max(levels, 2), entries)
        a += control(k * 3, None, k % 4) + image(0, 0, 17, 11, bits, lzw(px, bits), table=table, interlace=k % 2 == 1)
    f["anim_code_sizes.gif"] = bytes(a) + END

    # code sizes outside 2..8 as the reference reads them: 0 and 1 bit (the code size never
    # grows), 9 and 12 bits (a code is stored as its low 8 bits; with 12 bits the table takes
    # no entry and only the code 4098 names a string)
    a = bytearray(screen(8, 4, c256))
    a += control(1) + image(0, 0, 8, 4, 0, pack_codes([(1, 1)] + [(0, 1)] * 32), table=cube(2, 8)[3:] + cube(2, 8)[:3])
    a += control(2) + image(0, 0, 8, 4, 1, pack_codes([(2, 2)] + [((x + y) % 2, 2) for y in range(4) for x in range(8)]))
    a += control(3) + image(0, 0, 8, 4, 9, lzw(noise(8, 512, 21) * 4, 9))
    codes = [(4096, 13)]
    for v in noise(11, 4096, 22):
        codes += [(v, 13), (4098, 13)]
    a += control(4) + image(0, 0, 8, 4, 12, pack_codes(codes + [(v, 13) for v in noise(10, 4096, 23)]))
    f["anim_odd_code_sizes.gif"] = bytes(a) + END
    return f


def feed_case(tail_pairs):
    """A 4x4 frame whose data stops at an end code after 8 pixels, followed by `tail_pairs`
    one-byte sub-blocks of zero bits (the same bytes as `early_end` in the Rust tests)."""
    codes = [(2, 2)] + [(v, 2) for v in (1, 0, 1, 1, 0, 1, 0, 0)] + [(3, 2)]
    table = bytes([10, 20, 30, 200, 100, 50])
    return screen(4, 4, table) + image(0, 0, 4, 4, 1, pack_codes(codes)) + b"\x01\x00" * tail_pairs


# ---------------------------------------------------------------- reference hashes

def fnv(data):
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def reference(data):
    """(size, [(delay, hash)], how the reference stopped)."""
    im = Image.open(io.BytesIO(data))
    size, frames, k = im.size, [], 0
    while True:
        try:
            im.seek(k)
        except EOFError:
            return size, frames, "end"
        try:
            rgb = im.convert("RGB").tobytes()
        except Exception as e:  # the reference can not show this frame
            return size, frames, "fails on frame %d (%s)" % (k, type(e).__name__)
        assert im.size == size
        frames.append((im.info.get("duration", 0), fnv(rgb)))
        k += 1


def main():
    total = 0
    print("// name, canvas width and height, (delay_ms, FNV-1a 64 of the RGB bytes) of every frame")
    for name, data in sorted(fixtures().items()):
        with open(os.path.join(HERE, name), "wb") as out:
            out.write(data)
        total += len(data)
        size, frames, how = reference(data)
        rows = ", ".join("(%d, 0x%016X)" % f for f in frames)
        if how != "end":
            print("// the reference %s of the next file" % how)
        print('fixture!("%s", %d, %d, [%s]),' % (name, size[0], size[1], rows))
    print("// %d bytes in %d files" % (total, len(fixtures())), file=sys.stderr)
    size, frames, how = reference(feed_case(40000))
    print("// early end code, 40000 tail sub-blocks: 0x%016X (%s)" % (frames[0][1], how))
    size, frames, how = reference(feed_case(20000))
    print("// early end code, 20000 tail sub-blocks: %d frames, the reference %s" % (len(frames), how))


if __name__ == "__main__":
    main()
