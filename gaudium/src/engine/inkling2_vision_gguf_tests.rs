//! Host tests of the projector-file tower source: a synthetic tower written both as a
//! checkpoint (safetensors) and as projector files (GGUF) must pack the same device tensors;
//! the choice among several projector files; the spec from the metadata and its refusals; and,
//! with the real files, every device tensor of the published tower from both forms.

use super::super::{host_tables, pack_weight, vision_spec, StTower, LAYER_WEIGHTS};
use super::*;
use crate::fp8::f32_to_bf16_rne;
use crate::gguf::tests::{temp_dir, Builder};
use crate::safetensors::SafetensorsStore;

// ---------------------------------------------------------------------------
// A synthetic tower
// ---------------------------------------------------------------------------

/// hid 8, 2 heads of 4, one layer, MLP 16, 2-pixel patches merged 2 x 2, a 2 x 2 position
/// grid (4-pixel images), merger 32 -> 12 -> 6.
fn toy_dims() -> IkvDims {
    IkvDims {
        hid: 8,
        layers: 1,
        nh: 2,
        dh: 4,
        inter: 16,
        patch: 2,
        merge: 2,
        grid: 2,
        proj: 12,
        out: 6,
        channels: 3,
        eps: 1e-5,
        theta: 10_000.0,
    }
}

/// The checkpoint tensors of the toy tower (names below `model.visual.`, row-major shapes),
/// with values that BF16 holds exactly.
fn toy_tensors() -> Vec<(String, Vec<u64>, Vec<f32>)> {
    let d = toy_dims();
    let (h, i) = (d.hid, d.inter);
    let mut seed = 0u64;
    let mut t = |name: &str, shape: Vec<u64>| {
        seed += 1;
        let n: u64 = shape.iter().product();
        let v = (0..n)
            .map(|k| (((seed * 31 + k * 7) % 97) as f32 - 48.0) / 16.0)
            .collect();
        (name.to_string(), shape, v)
    };
    let mut out = vec![
        t("backbone.embeddings.class_embedding", vec![h]),
        t("backbone.embeddings.cls_position_embedding", vec![h]),
        t(
            "backbone.embeddings.patch_embedding.weight",
            vec![h, d.row_len()],
        ),
        t(
            "backbone.embeddings.position_embedding",
            vec![d.grid * d.grid, h],
        ),
        t("backbone.ln_pre.weight", vec![h]),
        t("backbone.ln_pre.bias", vec![h]),
        t("proj_in.weight", vec![d.proj, d.merged()]),
        t("proj_out.weight", vec![d.out, d.proj]),
    ];
    for (n, s) in [
        ("ln_1.weight", vec![h]),
        ("ln_1.bias", vec![h]),
        ("ln_2.weight", vec![h]),
        ("ln_2.bias", vec![h]),
        ("ls_1.gamma", vec![h]),
        ("ls_2.gamma", vec![h]),
        ("self_attn.q_proj.weight", vec![h, h]),
        ("self_attn.q_proj.bias", vec![h]),
        ("self_attn.k_proj.weight", vec![h, h]),
        ("self_attn.k_proj.bias", vec![h]),
        ("self_attn.v_proj.weight", vec![h, h]),
        ("self_attn.v_proj.bias", vec![h]),
        ("self_attn.out_proj.weight", vec![h, h]),
        ("self_attn.out_proj.bias", vec![h]),
        ("mlp.c_fc.weight", vec![i, h]),
        ("mlp.c_fc.bias", vec![i]),
        ("mlp.c_proj.weight", vec![h, i]),
        ("mlp.c_proj.bias", vec![h]),
    ] {
        out.push(t(&format!("backbone.layers.0.{n}"), s));
    }
    out
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| f32_to_bf16_rne(x).to_le_bytes())
        .collect()
}

fn f32_le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The toy tower as a checkpoint directory (one BF16 safetensors file).
fn write_checkpoint(dir: &Path) {
    let mut header = String::from("{");
    let mut data = Vec::new();
    for (k, (name, shape, v)) in toy_tensors().iter().enumerate() {
        let b = bf16_bytes(v);
        if k > 0 {
            header.push(',');
        }
        header.push_str(&format!(
            r#""model.visual.{name}":{{"dtype":"BF16","shape":{shape:?},"data_offsets":[{},{}]}}"#,
            data.len(),
            data.len() + b.len()
        ));
        data.extend_from_slice(&b);
    }
    header.push('}');
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&data);
    std::fs::write(dir.join("model.safetensors"), file).unwrap();
}

fn kv_f32(b: &mut Builder, key: &str, x: f32) {
    b.kv_raw(key, 6, &x.to_le_bytes());
}

fn kv_f32_array(b: &mut Builder, key: &str, v: &[f32]) {
    let mut raw = 6u32.to_le_bytes().to_vec();
    raw.extend_from_slice(&(v.len() as u64).to_le_bytes());
    raw.extend_from_slice(&f32_le(v));
    b.kv_raw(key, 9, &raw);
}

/// The toy tower as a projector file: matrices in `bulk` (BF16 or F32), small tensors F32,
/// the patch linear as a `[out, channel, y, x]` kernel; `proj_dim` and `ptype` as metadata.
fn projector_bytes(bulk: GgmlType, proj_dim: u32, ptype: &str) -> Vec<u8> {
    let d = toy_dims();
    let mut b = Builder::new();
    b.kv_str("general.architecture", "clip")
        .kv_str("general.type", "mmproj")
        .kv_str("clip.vision.projector_type", ptype)
        .kv_u32("clip.vision.projection_dim", proj_dim)
        .kv_u32("clip.vision.image_size", (d.grid * d.patch) as u32)
        .kv_u32("clip.vision.patch_size", d.patch as u32)
        .kv_u32("clip.vision.embedding_length", d.hid as u32)
        .kv_u32("clip.vision.feed_forward_length", d.inter as u32)
        .kv_u32("clip.vision.block_count", d.layers as u32)
        .kv_u32("clip.vision.attention.head_count", d.nh as u32)
        .kv_u32("clip.vision.spatial_merge_size", d.merge as u32);
    kv_f32(&mut b, "clip.vision.attention.layer_norm_epsilon", d.eps);
    kv_f32_array(&mut b, "clip.vision.image_mean", &[0.5, 0.5, 0.5]);
    kv_f32_array(&mut b, "clip.vision.image_std", &[0.5, 0.5, 0.5]);
    for (name, shape, v) in toy_tensors() {
        let (gname, rm, vals) = if name == PATCH_EMBEDDING {
            // [out][y][x][c] -> [out][c][y][x]
            let (o, c, p) = (d.hid as usize, 3usize, d.patch as usize);
            let row = c * p * p;
            let mut k = vec![0f32; v.len()];
            for oo in 0..o {
                for cc in 0..c {
                    for y in 0..p {
                        for x in 0..p {
                            k[oo * row + (cc * p + y) * p + x] = v[oo * row + (y * p + x) * c + cc];
                        }
                    }
                }
            }
            (
                "v.patch_embd.weight".to_string(),
                vec![d.hid, 3, d.patch, d.patch],
                k,
            )
        } else {
            (tensor_name(&name).unwrap(), shape, v)
        };
        let dims: Vec<u64> = rm.iter().rev().copied().collect();
        if rm.len() == 2 && bulk == GgmlType::BF16 && gname != "v.position_embd.weight" {
            b.tensor(&gname, &dims, GgmlType::BF16, bf16_bytes(&vals));
        } else {
            b.tensor(&gname, &dims, GgmlType::F32, f32_le(&vals));
        }
    }
    b.build()
}

/// A main model file of hidden size `hidden` whose vocabulary holds the image placeholder.
fn main_bytes(hidden: u32) -> Vec<u8> {
    let mut b = Builder::new();
    b.kv_str("general.architecture", "inkling2_loop")
        .kv_u32("inkling2_loop.embedding_length", hidden)
        .kv_str_array(
            "tokenizer.ggml.tokens",
            &["a", "<|content_image|>", "b", "<|unused_200054|>"],
        );
    b.build()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn checkpoint_names_map_to_projector_names() {
    let n = |s: &str| tensor_name(s).unwrap();
    assert_eq!(n("backbone.embeddings.class_embedding"), "v.class_embd");
    assert_eq!(n("backbone.layers.7.ls_2.gamma"), "v.blk.7.ls2.weight");
    assert_eq!(
        n("backbone.layers.0.self_attn.out_proj.bias"),
        "v.blk.0.attn_out.bias"
    );
    assert_eq!(
        n("backbone.layers.22.mlp.c_fc.weight"),
        "v.blk.22.ffn_up.weight"
    );
    assert_eq!(n("proj_out.weight"), "mm.2.weight");
    assert!(tensor_name("backbone.layers.x.ln_1.weight").is_none());
    assert!(tensor_name("backbone.layers.0.something").is_none());
    // the patch linear is not a plain rename
    assert!(tensor_name(PATCH_EMBEDDING).is_none());
}

#[test]
fn patch_rows_undo_the_kernel_layout() {
    // one output, 3 channels, 2 x 2 pixels: kernel value = 100 c + 10 y + x
    let mut k = Vec::new();
    for c in 0..3u16 {
        for y in 0..2u16 {
            for x in 0..2u16 {
                k.push(100 * c + 10 * y + x);
            }
        }
    }
    let r = patch_rows(&k, 1, 3, 2);
    // rows are y, x, channel
    assert_eq!(&r[..6], &[0, 100, 200, 1, 101, 201]);
    assert_eq!(&r[6..], &[10, 110, 210, 11, 111, 211]);
}

#[test]
fn precision_ranks_of_today() {
    // the preference (a choice pinned here; the laws bound how it is used)
    let r = |p| mmp_rank(p);
    assert!(r(MmpPrecision::MmpBf16) < r(MmpPrecision::MmpF32));
    assert!(r(MmpPrecision::MmpF32) < r(MmpPrecision::MmpF16));
    assert!(r(MmpPrecision::MmpF16) < r(MmpPrecision::MmpOther));
}

#[test]
fn both_forms_of_a_tower_pack_the_same_device_tensors() {
    let dir = temp_dir("ikv_mmproj_pack");
    write_checkpoint(&dir);
    let d = toy_dims();
    let store = SafetensorsStore::open(&dir).unwrap();
    let st = StTower(&store);
    for bulk in [GgmlType::BF16, GgmlType::F32] {
        let p = dir.join("mmproj-toy.gguf");
        std::fs::write(&p, projector_bytes(bulk, 6, "inkling2v")).unwrap();
        let g = Gguf::open(&p).unwrap();
        assert_eq!(projector_dims(&g).unwrap(), d);
        let pj = ProjectorSource::new(Arc::new(g));
        let mut names: Vec<String> = ["VW_patch_W", "VW_pre_w", "VW_pre_b", "VM_in_W", "VM_out_W"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        names.extend(LAYER_WEIGHTS.iter().map(|k| format!("VL0_{k}")));
        for n in &names {
            assert_eq!(
                pack_weight(&st, &d, n).unwrap(),
                pack_weight(&pj, &d, n).unwrap(),
                "{n} ({})",
                bulk.name()
            );
        }
        assert_eq!(host_tables(&st, &d).unwrap(), host_tables(&pj, &d).unwrap());
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_best_matching_projector_file_is_chosen() {
    let dir = temp_dir("ikv_mmproj_pick");
    let model = dir.join("model.gguf");
    std::fs::write(&model, main_bytes(6)).unwrap();
    // nothing beside the model: text only
    assert_eq!(find_projector(&model, 6).unwrap(), None);
    // a projector of another width and one of another type: refused with the reasons
    std::fs::write(
        dir.join("mmproj-a.gguf"),
        projector_bytes(GgmlType::BF16, 7, "inkling2v"),
    )
    .unwrap();
    std::fs::write(
        dir.join("mmproj-b.gguf"),
        projector_bytes(GgmlType::BF16, 6, "other"),
    )
    .unwrap();
    let e = find_projector(&model, 6).unwrap_err();
    assert!(
        e.contains("projection width 7, the model's hidden size is 6"),
        "{e}"
    );
    assert!(e.contains("projector type 'other'"), "{e}");
    // F32 matches; a BF16 one after it in name order is preferred
    std::fs::write(
        dir.join("mmproj-c-f32.gguf"),
        projector_bytes(GgmlType::F32, 6, "inkling2v"),
    )
    .unwrap();
    assert_eq!(
        find_projector(&model, 6).unwrap(),
        Some(dir.join("mmproj-c-f32.gguf"))
    );
    std::fs::write(
        dir.join("mmproj-d-bf16.gguf"),
        projector_bytes(GgmlType::BF16, 6, "inkling2v"),
    )
    .unwrap();
    std::fs::write(
        dir.join("mmproj-e-bf16.gguf"),
        projector_bytes(GgmlType::BF16, 6, "inkling2v"),
    )
    .unwrap();
    // a tie keeps the first in name order
    assert_eq!(
        find_projector(&model, 6).unwrap(),
        Some(dir.join("mmproj-d-bf16.gguf"))
    );
    // a file that is not named mmproj* is not a candidate
    assert_eq!(candidates(&model).len(), 5);

    let main = Gguf::open(&model).unwrap();
    let t = model_tower(&model, &main).unwrap().unwrap();
    assert_eq!(t.projector, dir.join("mmproj-d-bf16.gguf"));
    assert_eq!(t.dims, toy_dims());
    assert_eq!(t.spec.image_token_id, 3);
    assert_eq!(t.spec.image_block, "<|content_image|><|unused_200054|>");
    assert_eq!(t.spec.out_hidden, 6);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_projector_that_disagrees_with_its_model_or_itself_is_refused() {
    let dir = temp_dir("ikv_mmproj_refuse");
    let p = dir.join("mmproj.gguf");
    std::fs::write(&p, projector_bytes(GgmlType::BF16, 6, "inkling2v")).unwrap();
    let proj = Gguf::open(&p).unwrap();
    let m = dir.join("main.gguf");
    std::fs::write(&m, main_bytes(10)).unwrap();
    let main = Gguf::open(&m).unwrap();
    let e = tower_spec(&main, &proj, 10).unwrap_err();
    assert!(
        e.contains("projection width 6") && e.contains("hidden size is 10"),
        "{e}"
    );

    // metadata that disagrees with the tensors: one block more than the file holds
    let mut b = Builder::new();
    b.kv_str("clip.vision.projector_type", "inkling2v")
        .kv_u32("clip.vision.projection_dim", 6)
        .kv_u32("clip.vision.image_size", 4)
        .kv_u32("clip.vision.patch_size", 2)
        .kv_u32("clip.vision.embedding_length", 8)
        .kv_u32("clip.vision.feed_forward_length", 16)
        .kv_u32("clip.vision.block_count", 2)
        .kv_u32("clip.vision.attention.head_count", 2)
        .kv_u32("clip.vision.spatial_merge_size", 2);
    kv_f32(&mut b, "clip.vision.attention.layer_norm_epsilon", 1e-5);
    let full = Gguf::open(&p).unwrap();
    for t in full.tensors() {
        b.tensor(&t.name, &t.dims, t.ty, full.tensor_data(t).to_vec());
    }
    let q = dir.join("mmproj-bad.gguf");
    std::fs::write(&q, b.build()).unwrap();
    let e = projector_dims(&Gguf::open(&q).unwrap()).unwrap_err();
    assert!(
        e.contains("block_count is 2, the file holds 1 layers"),
        "{e}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// The publisher's checkpoint and the GGUF build's projector file give the same tower: the same
/// geometry and image spec, and every device tensor byte for byte. Needs both:
/// `GAUDIUM_INKLING2_DIR` (the checkpoint directory) and `GAUDIUM_INKLING2_GGUF` (the main GGUF
/// file; its projector file lies beside it).
#[test]
#[ignore = "needs GAUDIUM_INKLING2_DIR and GAUDIUM_INKLING2_GGUF (the same model in both forms)"]
fn projector_and_checkpoint_pack_the_same_tower() {
    let (Some(dir), Some(file)) = (
        std::env::var_os("GAUDIUM_INKLING2_DIR"),
        std::env::var_os("GAUDIUM_INKLING2_GGUF"),
    ) else {
        panic!("set GAUDIUM_INKLING2_DIR and GAUDIUM_INKLING2_GGUF");
    };
    let (dir, file) = (PathBuf::from(dir), PathBuf::from(file));
    let main = Gguf::open(&file).unwrap();
    let t = model_tower(&file, &main)
        .unwrap()
        .expect("a projector file beside the GGUF file");
    eprintln!("projector file: {}", t.projector.display());
    let config = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let proc = std::fs::read_to_string(dir.join("processor_config.json")).ok();
    let d = IkvDims::from_config_json(&config).unwrap();
    assert_eq!(t.dims, d, "tower geometry");
    assert_eq!(
        t.spec,
        vision_spec(&config, proc.as_deref()).unwrap(),
        "image spec"
    );
    assert_eq!(t.spec, model_vision_spec(&file).unwrap());

    let store = SafetensorsStore::open(&dir).unwrap();
    let st = StTower(&store);
    let pj = ProjectorSource::open(&t.projector).unwrap();
    let mut names: Vec<String> = ["VW_patch_W", "VW_pre_w", "VW_pre_b", "VM_in_W", "VM_out_W"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    for l in 0..d.layers {
        names.extend(LAYER_WEIGHTS.iter().map(|k| format!("VL{l}_{k}")));
    }
    let mut bad = Vec::new();
    let mut bytes = 0usize;
    for n in &names {
        let (x, y) = (
            pack_weight(&st, &d, n).unwrap(),
            pack_weight(&pj, &d, n).unwrap(),
        );
        bytes += x.len();
        if x != y {
            let diff = x.chunks(2).zip(y.chunks(2)).filter(|(p, q)| p != q).count();
            bad.push(format!(
                "{n}: {diff} of {} 16-bit words differ ({} vs {} bytes)",
                x.len() / 2,
                x.len(),
                y.len()
            ));
        }
    }
    let (ts, cs) = host_tables(&st, &d).unwrap();
    let (tg, cg) = host_tables(&pj, &d).unwrap();
    if ts != tg {
        bad.push("position table differs".into());
    }
    if cs != cg {
        bad.push("class row differs".into());
    }
    eprintln!("{} device tensors, {bytes} bytes, compared", names.len());
    assert!(bad.is_empty(), "device tensors differ:\n{}", bad.join("\n"));
}
