#!/usr/bin/env python3
"""Regenerates the fixed images of the vision regression suite (`suite.json` beside this file).

Run it from any directory; it writes the files next to itself:

- shapes.png: a red square on the left and a blue circle on the right, on an off-white ground,
  no text (the answer needs the picture itself);
- number.png: the number 4271 in large yellow digits on a dark blue ground (reading text).

Both are 640 x 480: 20 x 15 merged patches (300 image tokens) for a Qwen3.5 / 3.8 tower, so in
the suite's two-image case the second image's span crosses the first 512-row prefill tile.
The digits are drawn with the first bold sans-serif font found (DejaVu Sans Bold on most Linux
systems, or the path given as the first argument); the files in the repository were made with it.
"""
import os
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 640, 480
FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
]


def shapes():
    img = Image.new("RGB", (W, H), (248, 248, 242))
    d = ImageDraw.Draw(img)
    d.rectangle([50, 120, 290, 360], fill=(200, 30, 30))
    d.ellipse([370, 120, 610, 360], fill=(30, 60, 200))
    return img


def number(font_path):
    img = Image.new("RGB", (W, H), (22, 40, 92))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, 150)
    text = "4271"
    box = d.textbbox((0, 0), text, font=font)
    x = (W - (box[2] - box[0])) // 2 - box[0]
    y = (H - (box[3] - box[1])) // 2 - box[1]
    d.text((x, y), text, font=font, fill=(255, 230, 60))
    return img


def main():
    font = sys.argv[1] if len(sys.argv) > 1 else next((f for f in FONTS if os.path.exists(f)), None)
    if font is None:
        sys.exit("no bold sans-serif font found: give its path as the first argument")
    shapes().save(os.path.join(HERE, "shapes.png"), optimize=True)
    number(font).save(os.path.join(HERE, "number.png"), optimize=True)


if __name__ == "__main__":
    main()
