#!/usr/bin/env python3
"""Reference records of the Gemma 4 image path, made with the checkpoint's own processor and
vision weights on the CPU (the decoder is never loaded).

    make_gemma4_refs.py <model_dir> <out_dir> <image> [<image> ...]

For every image (PNG / JPEG / PPM, decoded to RGB8 and written out as `<stem>.ppm`, which is what
the Rust check reads) and the model at <model_dir>:

    <out>/<stem>.json        target size, patch grid, soft-token count
    <out>/<stem>.pix.f32     the processor's pixel_values rows (real rows only), float32
    <out>/<stem>.pos.i32     the processor's position ids (x, y) of those rows, int32
    <out>/<stem>.soft.bf16   the soft tokens the decoder receives ({tokens, text hidden} BF16)

Run with `TORCH_DEVICE_BACKEND_AUTOLOAD=0 HABANA_VISIBLE_DEVICES=` so the accelerator is never
opened. The tower is built as the library's loader builds it (parameters in BF16, rotation
frequencies in float32) and runs on the real rows only.
"""
import json
import os
import struct
import sys

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open


def load_rgb(path):
    img = Image.open(path)
    img = img.convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def write_ppm(path, rgb):
    h, w, _ = rgb.shape
    with open(path, "wb") as f:
        f.write(b"P6\n%d %d\n255\n" % (w, h))
        f.write(rgb.tobytes())


def tensors_with_prefix(model_dir, prefixes):
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        files = sorted(set(json.load(open(idx))["weight_map"].values()))
    else:
        files = ["model.safetensors"]
    out = {}
    for f in files:
        with safe_open(os.path.join(model_dir, f), framework="pt") as st:
            for k in st.keys():
                for p in prefixes:
                    if k.startswith(p):
                        out[k] = st.get_tensor(k)
    return out


def main():
    model_dir, out_dir, images = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(out_dir, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("REF_THREADS", "8")))
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    unified = cfg["vision_config"].get("model_type") == "gemma4_unified_vision"

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_dir)
    if unified:
        from transformers.models.gemma4_unified.image_processing_gemma4_unified import (
            Gemma4UnifiedImageProcessor as Proc,
        )
        from transformers.models.gemma4_unified.modeling_gemma4_unified import (
            Gemma4UnifiedVisionEmbedder,
        )
    else:
        from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor as Proc
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4MultimodalEmbedder,
            Gemma4VisionModel,
        )
    pcfg = json.load(open(os.path.join(model_dir, "processor_config.json")))["image_processor"]
    pcfg = {k: v for k, v in pcfg.items() if k != "image_processor_type"}
    proc = Proc(**pcfg)

    # the image path's modules, parameters in BF16 (as the loader builds them)
    torch.set_default_dtype(torch.bfloat16)
    vcfg = config.vision_config
    vcfg._attn_implementation = "sdpa"
    if unified:
        emb = Gemma4UnifiedVisionEmbedder(vcfg, config.text_config)
        sd = tensors_with_prefix(model_dir, ["model.vision_embedder.", "model.embed_vision."])
        mapped = {}
        for k, v in sd.items():
            if k.startswith("model.vision_embedder."):
                mapped[k[len("model.vision_embedder."):]] = v
            else:
                mapped["multimodal_embedder." + k[len("model.embed_vision."):]] = v
        missing, unexpected = emb.load_state_dict(mapped, strict=False)
        assert not unexpected, unexpected
        assert not [m for m in missing if "inv_freq" not in m], missing
        tower, proj = None, None
    else:
        tower = Gemma4VisionModel(vcfg)
        proj = Gemma4MultimodalEmbedder(vcfg, config.text_config)
        sd = tensors_with_prefix(model_dir, ["model.vision_tower.", "model.embed_vision."])
        tw = {k[len("model.vision_tower."):]: v for k, v in sd.items() if k.startswith("model.vision_tower.")}
        pw = {k[len("model.embed_vision."):]: v for k, v in sd.items() if k.startswith("model.embed_vision.")}
        missing, unexpected = tower.load_state_dict(tw, strict=False)
        assert not unexpected, unexpected
        assert not [m for m in missing if "inv_freq" not in m], missing
        proj.load_state_dict(pw)
        tower.eval()
        emb = None
    torch.set_default_dtype(torch.float32)

    for path in images:
        stem = os.path.splitext(os.path.basename(path))[0]
        rgb = load_rgb(path)
        write_ppm(os.path.join(out_dir, stem + ".ppm"), rgb)
        t = torch.from_numpy(rgb.copy()).permute(2, 0, 1).contiguous()
        bf = proc(images=[t], return_tensors="pt")
        pv = bf["pixel_values"][0]
        pos = bf["image_position_ids"][0]
        n_soft = int(bf["num_soft_tokens_per_image"][0])
        real = (pos != -1).all(dim=-1)
        pv_real = pv[real].to(torch.float32).contiguous()
        pos_real = pos[real].to(torch.int32).contiguous()
        pv_real.numpy().tofile(os.path.join(out_dir, stem + ".pix.f32"))
        pos_real.numpy().tofile(os.path.join(out_dir, stem + ".pos.i32"))
        with torch.no_grad():
            if unified:
                out = emb(pv[real].unsqueeze(0), pos[real].unsqueeze(0), return_dict=True)
                soft = out.pooler_output[0]
            else:
                # the real rows only (the reference with its padding rows differs by a few
                # percent, from BF16 rounding through the layers)
                h = tower(pixel_values=pv[real].unsqueeze(0), pixel_position_ids=pos[real].unsqueeze(0))
                soft = proj(inputs_embeds=h.last_hidden_state)
                if soft.dim() == 3:
                    soft = soft[0]
        soft = soft.to(torch.bfloat16).contiguous()
        soft.view(torch.int16).numpy().tofile(os.path.join(out_dir, stem + ".soft.bf16"))
        xs = pos_real[:, 0].max().item() + 1
        ys = pos_real[:, 1].max().item() + 1
        rec = {
            "image": os.path.basename(path),
            "src_size": [int(rgb.shape[1]), int(rgb.shape[0])],
            "patch_grid": [int(ys), int(xs)] if not unified else None,
            "merged_grid": [int(ys), int(xs)] if unified else None,
            "rows": int(pv_real.shape[0]),
            "row_len": int(pv_real.shape[1]),
            "soft_tokens": n_soft,
            "soft_shape": list(soft.shape),
            "unified": unified,
        }
        json.dump(rec, open(os.path.join(out_dir, stem + ".json"), "w"), indent=1)
        print(stem, rec, flush=True)


if __name__ == "__main__":
    main()
