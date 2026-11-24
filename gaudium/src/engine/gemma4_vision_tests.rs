//! Host checks of the Gemma 4 image path (no card).

use super::graphs::{build_block_graph, build_embed_vis_graph, build_pool_graph, build_unified_graph};
use super::*;
use crate::vision::preprocess::RgbImage;

/// `config.json` of a tower size reduced to the fields the image path reads (the 31B's tower).
pub(crate) const TOWER_31B: &str = r#"{
  "architectures": ["Gemma4ForConditionalGeneration"],
  "model_type": "gemma4",
  "image_token_id": 258880, "boi_token_id": 255999, "eoi_token_id": 258882, "video_token_id": 258884,
  "vision_soft_tokens_per_image": 280,
  "text_config": {"hidden_size": 5376, "pad_token_id": 0, "use_bidirectional_attention": "vision"},
  "vision_config": {
    "model_type": "gemma4_vision", "hidden_size": 1152, "intermediate_size": 4304,
    "num_hidden_layers": 27, "num_attention_heads": 16, "num_key_value_heads": 16, "head_dim": 72,
    "patch_size": 16, "pooling_kernel_size": 3, "position_embedding_size": 10240,
    "rms_norm_eps": 1e-06, "rope_parameters": {"rope_theta": 100.0, "rope_type": "default"},
    "standardize": true, "use_clipped_linears": false, "hidden_activation": "gelu_pytorch_tanh",
    "attention_bias": false, "default_output_length": 280
  }
}"#;

/// The small sizes' tower (E2B: clamped linears, no standardisation).
pub(crate) const TOWER_E2B: &str = r#"{
  "model_type": "gemma4",
  "image_token_id": 258880, "boi_token_id": 255999, "eoi_token_id": 258882,
  "text_config": {"hidden_size": 1536},
  "vision_config": {
    "model_type": "gemma4_vision", "hidden_size": 768, "intermediate_size": 3072,
    "num_hidden_layers": 16, "num_attention_heads": 12, "num_key_value_heads": 12, "head_dim": 64,
    "patch_size": 16, "pooling_kernel_size": 3, "position_embedding_size": 10240,
    "rope_parameters": {"rope_theta": 100.0}, "standardize": false, "use_clipped_linears": true
  }
}"#;

/// The 12B's encoder-free embedder.
pub(crate) const UNIFIED_12B: &str = r#"{
  "architectures": ["Gemma4UnifiedForConditionalGeneration"],
  "model_type": "gemma4_unified",
  "image_token_id": 258880, "boi_token_id": 255999, "eoi_token_id": 258882,
  "text_config": {"hidden_size": 3840},
  "vision_config": {
    "model_type": "gemma4_unified_vision", "mm_embed_dim": 3840, "mm_posemb_size": 1120,
    "model_patch_size": 48, "num_soft_tokens": 280, "output_proj_dims": 3840,
    "patch_size": 16, "pooling_kernel_size": 3, "rms_norm_eps": 1e-06
  }
}"#;

const PROCESSOR: &str = r#"{"image_processor": {"do_normalize": false, "do_rescale": true,
  "max_soft_tokens": 280, "patch_size": 16, "pooling_kernel_size": 3, "resample": 3,
  "rescale_factor": 0.00392156862745098}}"#;

fn spec(cfg: &str) -> VisionSpec {
    vision_spec(cfg, Some(PROCESSOR), None).unwrap()
}

#[test]
fn the_two_image_paths_parse() {
    let s = spec(TOWER_31B);
    assert_eq!(s.processor, ImageProcessor::Gemma4 { merged: false });
    assert_eq!((s.image_block.as_str(), s.image_text.as_str()), ("<|image|>", "<|image><|image|><image|>"));
    assert_eq!((s.patch, s.merge, s.resize_factor(), s.patches_per_token()), (16, 3, 48, 9));
    assert_eq!(soft_budget(&s), 280);
    assert_eq!(s.video_token_id, -1);
    let d = G4vDims::from_config_json(TOWER_31B).unwrap();
    assert_eq!((d.kind, d.hid, d.nh, d.dh, d.layers, d.out), (G4vKind::Tower, 1152, 16, 72, 27, 5376));
    assert!(d.standardize && !d.clipped);
    assert_eq!(d.row_len(), 768);
    let e = G4vDims::from_config_json(TOWER_E2B).unwrap();
    assert!(e.clipped && !e.standardize);
    let u = G4vDims::from_config_json(UNIFIED_12B).unwrap();
    assert_eq!((u.kind, u.hid, u.row_len(), u.npos), (G4vKind::Unified, 3840, 6912, 1120));
    assert_eq!(row_len(&spec(UNIFIED_12B)), 6912);
    assert!(is_gemma4_vision(TOWER_31B) && is_gemma4_vision(UNIFIED_12B));
    // a normalising processor or another resampler is refused, not run differently
    assert!(vision_spec(TOWER_31B, Some(r#"{"image_processor": {"do_normalize": true}}"#), None).is_err());
    assert!(vision_spec(TOWER_31B, Some(r#"{"image_processor": {"resample": 2}}"#), None).is_err());
    let grouped = TOWER_31B.replace("\"num_key_value_heads\": 16", "\"num_key_value_heads\": 4");
    assert!(G4vDims::from_config_json(&grouped).is_err());
    // the unified embedder's merged patch and projection must agree with its graph
    for bad in [
        UNIFIED_12B.replace("\"model_patch_size\": 48", "\"model_patch_size\": 32"),
        UNIFIED_12B.replace("\"output_proj_dims\": 3840", "\"output_proj_dims\": 2560"),
    ] {
        assert!(G4vDims::from_config_json(&bad).is_err());
    }
    let unstated = UNIFIED_12B
        .replace("\"model_patch_size\": 48, ", "")
        .replace("\"output_proj_dims\": 3840, ", "");
    assert!(G4vDims::from_config_json(&unstated).is_ok());
}

#[test]
fn the_host_processor_takes_one_rgb_frame_and_a_budget_of_a_token() {
    // today's processor: one frame per patch, 3 channels, at least one 48x48 square of budget
    let s = spec(TOWER_31B);
    assert!(s.validate().is_ok());
    let side = s.resize_factor();
    for (what, v) in [
        ("two frames", VisionSpec { temporal: 2, ..s.clone() }),
        ("one channel", VisionSpec { in_channels: 1, ..s.clone() }),
        ("no patch", VisionSpec { patch: 0, ..s.clone() }),
        ("no pooling", VisionSpec { merge: 0, ..s.clone() }),
        ("a budget below one token", VisionSpec { max_pixels: side * side - 1, ..s.clone() }),
        ("sizes that overflow", VisionSpec { patch: 1 << 40, merge: 1 << 30, ..s.clone() }),
    ] {
        assert!(v.validate().is_err(), "{what}");
    }
    assert!(VisionSpec { max_pixels: side * side, ..s.clone() }.validate().is_ok());
}

/// Resized sizes the reference processor gave for the fixed image set (patch grids from its
/// dumps), and the edge cases of its rule.
#[test]
fn target_sizes_match_the_reference_processor() {
    let s = spec(TOWER_31B);
    for ((w, h), (gh, gw)) in [
        ((672, 448), (39, 60)),
        ((256, 256), (48, 48)),
        ((448, 320), (42, 57)),
        ((1770, 1180), (39, 60)),
        ((17, 13), (42, 57)),
        ((1000, 10), (3, 501)),
        ((203, 301), (60, 39)),
    ] {
        let (rh, rw) = target_size(w, h, &s).unwrap();
        assert_eq!((rh / 16, rw / 16), (gh, gw), "{w}x{h}");
        assert!(image_tokens(rh, rw, &s) <= 280);
    }
    // a strip too thin for one square on its short side: one square, the long side by the ratio
    let (rh, rw) = target_size(100_000, 10, &s).unwrap();
    assert_eq!(rh, 48);
    assert_eq!(rw, 280 * 48);
    assert!(target_size(0, 10, &s).is_err());
    // a lower budget (a smaller planned tower) shrinks the image with it
    let mut small = s.clone();
    small.max_pixels = 70 * 48 * 48;
    let (rh, rw) = target_size(672, 448, &small).unwrap();
    assert!(image_tokens(rh, rw, &small) <= 70);
}

#[test]
fn the_position_tables_refuse_grids_past_their_rows() {
    // a table of 4 rows per axis: a grid of 4 columns or rows fits, a fifth is refused on
    // either axis (a fifth column of the tower's table would read the rows axis)
    let (npos, hid) = (4usize, 2usize);
    let tower: Vec<u16> = (0..2 * npos * hid).map(|v| v as u16).collect();
    assert!(tower_positions(&tower, npos, hid, 3, npos, 1).is_ok());
    assert!(tower_positions(&tower, npos, hid, npos, 3, 1).is_ok());
    assert!(tower_positions(&tower, npos, hid, 3, npos + 1, 1).is_err());
    assert!(tower_positions(&tower, npos, hid, npos + 1, 3, 1).is_err());
    let merged: Vec<u16> = (0..npos * 2 * hid).map(|v| v as u16).collect();
    assert!(merged_positions(&merged, npos, hid, 3, npos).is_ok());
    assert!(merged_positions(&merged, npos, hid, npos, 3).is_ok());
    assert!(merged_positions(&merged, npos, hid, 3, npos + 1).is_err());
    assert!(merged_positions(&merged, npos, hid, npos + 1, 3).is_err());
}

#[test]
fn the_pooling_weight_sums_each_channel_of_a_token_s_patches() {
    // entry (o, j h + c) is one exactly when c == o: output channel o sums channel o of each of
    // the token's patches j, nothing else
    for (h, p2) in [(1u64, 1u64), (4, 9), (7, 4), (128, 9)] {
        let w = super::run::pool_weight(h, p2, 1);
        assert_eq!(w.len() as u64, h * h * p2);
        for o in 0..h {
            for j in 0..p2 {
                for c in 0..h {
                    assert_eq!(w[(o * h * p2 + j * h + c) as usize], u16::from(c == o), "h {h} p2 {p2} o {o} j {j} c {c}");
                }
            }
        }
    }
}

#[test]
fn pooling_order_is_a_permutation_of_whole_blocks() {
    for (gh, gw) in [(3, 3), (39, 60), (48, 48), (3, 501), (60, 39)] {
        let o = pooling_order(gh, gw, 3);
        assert_eq!(o.len(), gh * gw);
        let mut seen = vec![false; gh * gw];
        for (i, &(y, x)) in o.iter().enumerate() {
            assert!(!seen[y * gw + x]);
            seen[y * gw + x] = true;
            // the 9 patches of output token i / 9 are one 3 x 3 block, blocks in reading order
            let blk = i / 9;
            assert_eq!((y / 3, x / 3), (blk / (gw / 3), blk % (gw / 3)));
            // and the row is recovered from the patch's position
            let row = crate::verified::g4v_pool_row(y as u64, x as u64, (gw / 3) as u64, 3);
            assert_eq!(row, i as u64);
        }
    }
}

#[test]
fn prepared_rows_hold_one_token_per_square() {
    let s = spec(TOWER_31B);
    let img = RgbImage::from_raw(96, 48, (0..96 * 48 * 3).map(|i| (i % 251) as u8).collect()).unwrap();
    let p = prepare_image(&img, &s).unwrap();
    assert_eq!(p.n_tokens() as u64, image_tokens(p.height, p.width, &s));
    assert_eq!(p.pixels.len(), p.n_patches() * 768);
    // value of byte v at the tower's input: bf16(2 (v / 255 - 0.5)); byte 0 is exactly -1
    let (_, _, f) = pixel_rows_f32(&img, &s).unwrap();
    assert_eq!(f.len(), p.pixels.len());
    let u = spec(UNIFIED_12B);
    let pu = prepare_image(&img, &u).unwrap();
    assert_eq!(pu.row_len, 6912);
    assert_eq!(pu.pixels.len(), pu.n_tokens() * 6912);
}

#[test]
fn graphs_have_the_planned_shapes() {
    let d = G4vDims::from_config_json(TOWER_E2B).unwrap();
    let n = 2520;
    let blk = build_block_graph(&d, n, &[]);
    let specs = blk.persistent_tensor_specs();
    let size = |name: &str| specs.iter().find(|s| s.0 == name).map(|s| s.3);
    // f32 scores of every head at the largest image
    assert_eq!(size("vb_scores"), Some(n * n * 12 * 4));
    // a clamped size binds four bounds per projection
    assert_eq!(size("vb_q_imin"), Some(256));
    assert_eq!(size("vb_down_omax"), Some(256));
    let pool = build_pool_graph(&d, n, &[]);
    let ps = pool.persistent_tensor_specs();
    assert_eq!(ps.iter().find(|s| s.0 == "v_feat").map(|s| s.3), Some(1536 * 280 * 2));
    let u = G4vDims::from_config_json(UNIFIED_12B).unwrap();
    let ug = build_unified_graph(&u, 280, &[]);
    let us = ug.persistent_tensor_specs();
    assert_eq!(us.iter().find(|s| s.0 == "u_px").map(|s| s.3), Some(6912 * 280 * 2));
}

#[test]
fn the_embedding_stage_reads_the_feature_buffer() {
    let spec = crate::engine::gemma4::tests::e2b();
    let d = crate::engine::gemma4::G4Dims::from_spec(&spec, 64, crate::engine::gemma4::G4NormMode::Tile).unwrap();
    let gb = build_embed_vis_graph(&d, 1 + 560, 64, &[]);
    let specs = gb.persistent_tensor_specs();
    let size = |name: &str| specs.iter().find(|s| s.0 == name).map(|s| s.3);
    assert_eq!(size("vis_feat"), Some(d.hid * 561 * 2));
    assert_eq!(size("vis_fidx"), Some(64 * 4));
    assert!(size("ple_out").is_some());
}

/// Checks the host processor against reference dumps (`make_gemma4_refs.py`): set
/// `GAUDIUM_G4V_REF_DIR` to a dump directory and `GAUDIUM_G4V_MODEL` to its model directory;
/// run with `--ignored`. The processor's pixel rows must match bit for bit.
#[test]
#[ignore = "needs GAUDIUM_G4V_REF_DIR and GAUDIUM_G4V_MODEL"]
fn reference_pixel_rows_match_bit_for_bit() {
    let dir = std::path::PathBuf::from(std::env::var("GAUDIUM_G4V_REF_DIR").expect("GAUDIUM_G4V_REF_DIR"));
    let model = std::path::PathBuf::from(std::env::var("GAUDIUM_G4V_MODEL").expect("GAUDIUM_G4V_MODEL"));
    let vs = VisionSpec::from_model_dir(&model).unwrap();
    let mut checked = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("ppm") {
            continue;
        }
        let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
        let rec = crate::json::parse_json(&std::fs::read_to_string(dir.join(format!("{stem}.json"))).unwrap()).unwrap();
        let img = RgbImage::load_ppm(&path).unwrap();
        let (gh, gw, rows) = pixel_rows_f32(&img, &vs).unwrap();
        let want = std::fs::read(dir.join(format!("{stem}.pix.f32"))).unwrap();
        let got: Vec<u8> = rows.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(got.len(), want.len(), "{stem}: row bytes");
        let diff = got.chunks(4).zip(want.chunks(4)).filter(|(a, b)| a != b).count();
        assert_eq!(diff, 0, "{stem}: {diff} values differ from the reference");
        let tokens = rec.get("soft_tokens").and_then(JsonValue::as_u64).unwrap();
        let p = prepare_image(&img, &vs).unwrap();
        assert_eq!(p.n_tokens() as u64, tokens, "{stem}: soft tokens ({gh} x {gw} patches)");
        checked += 1;
    }
    assert!(checked > 0, "no reference records in {}", dir.display());
}
