#!/usr/bin/env python3
"""Reference records of the whole host image pipeline: file -> reference loader (EXIF transpose,
RGB) -> the checkpoint's own image processor -> patch rows.

usage: make_pipeline_refs.py <model dir> <out dir> <image file>... [--max-pixels N] [--fingerprints]
       make_pipeline_refs.py <model dir> <out dir> <animated file>... --video [--fingerprints]

For every image file it writes, into <out dir>, a copy of the file, `<name>.f32` (the processor's
`pixel_values`, little-endian f32) and `<name>.json` (`grid`: patch rows and columns). The ignored
test `vision::media::tests::reference_pipeline_records_match_bit_for_bit` compares them with this
crate's pipeline (set GAUDIUM_IMAGE_PIPELINE_REF_DIR to <out dir>). With `--fingerprints` it prints
one line per file with the grid and the FNV-1a 64 of the f32 bytes instead (the constants of the
tracked fixture test). Needs the reference tokenizer library with its image processor, on the CPU.

With `--video` every file is an animation (GIF, WebP) with one delay for all frames: its frames go
through the checkpoint's video processor at the rate the delays give, and the record is
`<name>.video.f32` (`pixel_values_videos`) with `<name>.video.json` (`grid`: temporal patches, rows,
columns; `labels`: the time label of each temporal patch as the processor writes it in the prompt).
"""
import json, os, shutil, sys

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("HABANA_VISIBLE_DEVICES", "")

from transformers import AutoImageProcessor  # noqa: E402
from transformers.image_utils import load_image  # noqa: E402


def fnv1a64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def video_records(model: str, out: str, files: list, fingerprints: bool) -> None:
    from PIL import Image
    from transformers import AutoProcessor
    from transformers.video_utils import VideoMetadata

    proc = AutoProcessor.from_pretrained(model)
    if not fingerprints:
        os.makedirs(out, exist_ok=True)
    for path in files:
        im = Image.open(path)
        frames, delays = [], []
        for k in range(getattr(im, "n_frames", 1)):
            im.seek(k)
            frames.append(im.convert("RGB"))
            delays.append(im.info.get("duration", 0))
        if len(set(delays)) != 1 or delays[0] <= 0:
            raise SystemExit(f"{path}: needs one positive delay for all frames, has {sorted(set(delays))}")
        fps = 1000.0 / delays[0]
        meta = VideoMetadata(total_num_frames=len(frames), fps=fps)
        res = proc.video_processor(videos=[frames], video_metadata=[meta], return_metadata=True, return_tensors="pt")
        rows = res["pixel_values_videos"].numpy().astype("<f4")
        grid = [int(v) for v in res["video_grid_thw"][0]]
        kept = res["video_metadata"][0]
        times = proc._calculate_timestamps(kept.frames_indices, kept.fps, proc.video_processor.temporal_patch_size)
        name = os.path.basename(path)
        if fingerprints:
            labels = ", ".join(f'"{t:.1f}"' for t in times)
            print(f'("{name}", ({grid[0]}, {grid[1]}, {grid[2]}), 0x{fnv1a64(rows.tobytes()):016x}, &[{labels}]),')
            continue
        shutil.copyfile(path, os.path.join(out, name))
        rows.tofile(os.path.join(out, name + ".video.f32"))
        with open(os.path.join(out, name + ".video.json"), "w") as f:
            json.dump({"grid": grid, "labels": [f"<{t:.1f} seconds>" for t in times], "frames": len(frames)}, f)
        print(f"{name}: {len(frames)} frames at {fps:g}/s -> grid {grid}")


def main() -> None:
    args = sys.argv[1:]
    if "--video" in args:
        args.remove("--video")
        fingerprints = "--fingerprints" in args
        if fingerprints:
            args.remove("--fingerprints")
        video_records(args[0], args[1], args[2:], fingerprints)
        return
    max_pixels = None
    fingerprints = "--fingerprints" in args
    if fingerprints:
        args.remove("--fingerprints")
    if "--max-pixels" in args:
        i = args.index("--max-pixels")
        max_pixels = int(args[i + 1])
        del args[i : i + 2]
    model, out, files = args[0], args[1], args[2:]
    proc = AutoImageProcessor.from_pretrained(model)
    size = dict(proc.size)
    if max_pixels:
        size["longest_edge"] = max_pixels
    if not fingerprints:
        os.makedirs(out, exist_ok=True)
    for path in files:
        img = load_image(path)
        res = proc(images=[img], size=size, return_tensors="pt")
        rows = res["pixel_values"].numpy().astype("<f4")
        _, gh, gw = (int(v) for v in res["image_grid_thw"][0])
        name = os.path.basename(path)
        if fingerprints:
            print(f'("{name}", ({gh}, {gw}), 0x{fnv1a64(rows.tobytes()):016x}),')
            continue
        shutil.copyfile(path, os.path.join(out, name))
        stem = os.path.join(out, name)
        rows.tofile(stem + ".f32")
        with open(stem + ".json", "w") as f:
            json.dump({"grid": [gh, gw], "size": list(img.size), "max_pixels": size["longest_edge"]}, f)
        print(f"{name}: {img.size[0]}x{img.size[1]} -> grid {gh}x{gw}")


if __name__ == "__main__":
    main()
