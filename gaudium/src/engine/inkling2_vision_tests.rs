//! Host tests of the Inkling 2 image input: the processor, the position-table taps and the
//! rotary tables against the reference values of `tests/data/vision/inkling2/fixture.json`
//! (written by `make_fixture.py` next to it), the folded weights, the bucket ladder and the plan.

use super::*;
use crate::model_spec::ModelSpec;

/// `config.json` of the checkpoint reduced to what the vision code reads (and the text model's
/// geometry for the plan).
const CONFIG: &str = r#"{"architectures":["Inkling2LoopForConditionalGeneration"],"model_type":"inkling2_loop","eos_token_id":200006,
"text_config":{"model_type":"inkling2_loop_text","hidden_size":2304,"num_hidden_layers":24,"vocab_size":201024,
"num_attention_heads":24,"num_key_value_heads":8,"head_dim":128,"d_rel":16,"rel_extent":1024,"log_scaling_n_floor":8192,
"log_scaling_alpha":0.1,"rms_norm_eps":1e-06,"qk_l2_clip_threshold":22.627416997969522,"rel_l2_clip_threshold":8.0,
"local_layer_ids":[0,1,2,3,4,6,7,8,9,10,12,13,14,15,16,18,19,20,21,22],"sconv_kernel_size":4,"unpadded_vocab_size":200058,
"final_logit_softcapping":30.0,"sliding_window_size":1024,"n_layers_per_attn_res_block":6,"n_routed_experts":128,
"num_experts_per_tok":8,"intermediate_size":1152,"mlp_gate_softcap":4.0,"mlp_up_softcap":25.0,"route_scale":8.0,
"loop_config":{"core_start":6,"core_end":18,"num_passes":3,"gate_rank":64}},
"vision_config":{"model_type":"inkling2_vision","text_hidden_size":2304,"temporal_patch_size":2,"native_patch_size":20,
"adapter_merge_size":2,"projection_hidden_dim":4096,"projector_bias":false,"hidden_size":1024,"num_hidden_layers":23,
"num_attention_heads":16,"mlp_ratio":4.0,"posemb_grid_size":32,"layer_norm_eps":1e-05}}"#;

const PROCESSOR: &str = r#"{"image_processor":{"do_convert_rgb":true,"do_normalize":true,"do_rescale":true,
"image_mean":[0.5,0.5,0.5],"image_processor_type":"Inkling2ImageProcessor","image_std":[0.5,0.5,0.5],
"rescale_factor":0.00392156862745098,"size":{"height":40,"width":40}},"image_token":"<|unused_200054|>",
"processor_class":"Inkling2Processor"}"#;

fn fixture_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/vision/inkling2")
}

fn fixture() -> JsonValue {
    let text = std::fs::read_to_string(fixture_dir().join("fixture.json")).unwrap();
    parse_json(&text).unwrap()
}

fn fnv(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The generated thin images of the fixture (the same formula as `make_fixture.py`).
fn synthetic(w: usize, h: usize) -> RgbImage {
    let mut data = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            data.extend([
                ((x * 7 + y * 3) % 256) as u8,
                ((x * 5 + 1) % 256) as u8,
                ((y * 11 + 2) % 256) as u8,
            ]);
        }
    }
    RgbImage::from_raw(w, h, data).unwrap()
}

fn fixture_images() -> Vec<(String, RgbImage)> {
    let limits = crate::vision::decode::DecodeLimits {
        max_pixels: 64_000_000,
    };
    let png = |n: &str| {
        let bytes = std::fs::read(fixture_dir().join(n)).unwrap();
        crate::vision::decode::decode_image(&bytes, &limits).unwrap()
    };
    vec![
        ("img_200x120.png".into(), png("img_200x120.png")),
        ("img_448x448.png".into(), png("img_448x448.png")),
        ("synthetic_41x1".into(), synthetic(41, 1)),
        ("synthetic_7x81".into(), synthetic(7, 81)),
    ]
}

fn spec() -> VisionSpec {
    vision_spec(CONFIG, Some(PROCESSOR)).unwrap()
}

/// A fingerprint of the fixture (16 hex digits).
fn u64_of(v: &JsonValue, k: &str) -> u64 {
    let s = v
        .get(k)
        .and_then(JsonValue::as_str)
        .unwrap_or_else(|| panic!("fixture key {k}"));
    u64::from_str_radix(s, 16).unwrap()
}

#[test]
fn spec_of_the_checkpoint() {
    assert!(is_inkling2_vision(CONFIG));
    assert!(!is_inkling2_vision(
        r#"{"vision_config":{"model_type":"other"}}"#
    ));
    let s = spec();
    assert_eq!(s.processor, ImageProcessor::Tiles);
    assert_eq!(s.image_block, "<|content_image|><|unused_200054|>");
    assert_eq!(s.image_token_id, 200_054);
    assert_eq!(
        (s.patch_row(), s.patches_per_token(), s.pos_side()),
        (1200, 4, 32)
    );
    assert_eq!(
        (s.depth, s.hidden, s.out_hidden, s.video),
        (23, 1024, 2304, None)
    );
    let d = IkvDims::from_config_json(CONFIG).unwrap();
    assert_eq!(
        (d.dh, d.inter, d.proj, d.tile(), d.merged()),
        (64, 4096, 4096, 40, 4096)
    );
    assert_eq!((d.eps, d.theta), (1e-5, 10_000.0));
    // the server-side spec is the same through `VisionSpec::from_json`
    assert_eq!(VisionSpec::from_json(CONFIG, Some(PROCESSOR)).unwrap(), s);
    let bad = PROCESSOR.replace("\"height\":40", "\"height\":32");
    assert!(vision_spec(CONFIG, Some(&bad)).is_err());
}

#[test]
fn tile_rule_pads_bottom_right() {
    let s = spec();
    assert_eq!(padded_size(200, 120, &s).unwrap(), (120, 200));
    assert_eq!(padded_size(448, 448, &s).unwrap(), (480, 480));
    assert_eq!(padded_size(41, 1, &s).unwrap(), (40, 80));
    assert_eq!(padded_size(40, 40, &s).unwrap(), (40, 40));
    assert!(padded_size(0, 5, &s).is_err());
    // the media pipeline's size path agrees and counts one token per tile
    let (rh, rw) = crate::vision::preprocess::image_target_size(448, 448, &s).unwrap();
    let sized = PreparedImage::sized((448, 448), rw, rh, &s);
    assert_eq!(
        (sized.grid_h, sized.grid_w, sized.n_tokens()),
        (24, 24, 144)
    );
    // pad pixels are -1 on the 0..255 scale
    let img = synthetic(41, 1);
    let (gh, gw, v) = tile_rows_f32(&img, &s).unwrap();
    assert_eq!((gh, gw, v.len()), (2, 4, 8 * 1200));
    let pad = (-1.0f32 - 127.5) / 127.5;
    assert_eq!(
        v[1200 * 2 + 3],
        pad,
        "bottom-left patch of the first tile is padding"
    );
    assert_eq!(v[0], (f32::from(img.data[0]) - 127.5) / 127.5);
}

#[test]
fn pixels_taps_and_rope_match_the_reference() {
    let s = spec();
    let d = IkvDims::from_config_json(CONFIG).unwrap();
    let fx = fixture();
    for (name, img) in fixture_images() {
        let rec = fx
            .get(&name)
            .unwrap_or_else(|| panic!("fixture has no {name}"));
        let grid: Vec<u64> = rec
            .get("patch_grid")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        let (gh, gw, rows) = tile_rows_f32(&img, &s).unwrap();
        assert_eq!(vec![gh as u64, gw as u64], grid, "{name} patch grid");
        let bytes: Vec<u8> = rows.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(
            fnv(&bytes),
            u64_of(rec, "pixels_fnv"),
            "{name}: patch rows differ from the processor's"
        );
        let prep = prepare_tiles(&img, &s).unwrap();
        assert_eq!((prep.grid_h, prep.grid_w), (gh, gw));
        assert!(prep
            .pixels
            .iter()
            .zip(&rows)
            .all(|(&b, &f)| b == f32_to_bf16_rne(f)));

        let taps = interp_taps(gh, gw, d.grid as usize, d.merge as usize);
        let idx: Vec<u8> = taps
            .iter()
            .flat_map(|(i, _)| i.iter().flat_map(|&x| (x as i64).to_le_bytes()))
            .collect();
        let wts: Vec<u8> = taps
            .iter()
            .flat_map(|(_, w)| w.iter().flat_map(|x| x.to_le_bytes()))
            .collect();
        assert_eq!(
            fnv(&idx),
            u64_of(rec, "taps_fnv"),
            "{name}: position-table taps"
        );
        assert_eq!(
            fnv(&wts),
            u64_of(rec, "weights_fnv"),
            "{name}: position-table weights"
        );

        let pos = rope_positions(gh, gw, d.merge as usize);
        let pb: Vec<u8> = pos
            .iter()
            .flat_map(|&(h, w)| [i64::from(h), i64::from(w)])
            .flat_map(i64::to_le_bytes)
            .collect();
        assert_eq!(
            fnv(&pb),
            u64_of(rec, "rope_pos_fnv"),
            "{name}: rotary positions"
        );
        let inv = inv_freq(d.dh, d.theta);
        let want: Vec<u32> = rec
            .get("inv_freq")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let got: Vec<u32> = inv.iter().map(|v| v.to_bits()).collect();
        for (g, w) in got.iter().zip(&want) {
            assert!(
                g.abs_diff(*w) <= 1,
                "{name}: inverse frequency {g:#x} vs {w:#x}"
            );
        }
        let JsonValue::Object(sample) = rec.get("rope_rows").unwrap() else {
            panic!("rope_rows")
        };
        for (r, cs) in sample.iter() {
            let r: usize = r.parse().unwrap();
            let (c, sn) = rope_row(pos[r], &inv);
            let pair = cs.as_array().unwrap();
            for (got, want) in [(&c, &pair[0]), (&sn, &pair[1])] {
                for (g, w) in got.iter().zip(want.as_array().unwrap()) {
                    let w = w.as_f64().unwrap() as f32;
                    assert!(
                        (g - w).abs() <= 2.0 * f32::EPSILON,
                        "{name} row {r}: rotary {g} vs {w}"
                    );
                }
            }
        }
    }
}

/// Needs the checkpoint (`GAUDIUM_INKLING2_DIR`): the resampled position rows and the class
/// row bit for bit.
#[test]
#[ignore = "needs GAUDIUM_INKLING2_DIR with the checkpoint"]
fn pos_rows_match_the_reference() {
    let dir = std::env::var("GAUDIUM_INKLING2_DIR").expect("GAUDIUM_INKLING2_DIR is not set");
    let store = SafetensorsStore::open(&dir).unwrap();
    let d = IkvDims::from_config_json(
        &std::fs::read_to_string(std::path::Path::new(&dir).join("config.json")).unwrap(),
    )
    .unwrap();
    let (table, cls) = host_tables(&store, &d).unwrap();
    let fx = fixture();
    let s = spec();
    for (name, img) in fixture_images() {
        let rec = fx.get(&name).unwrap();
        let (gh, gw, _) = tile_rows_f32(&img, &s).unwrap();
        let taps = interp_taps(gh, gw, d.grid as usize, d.merge as usize);
        let rows = pos_rows(&table, d.hid as usize, &taps);
        assert_eq!(
            fnv(&u16_bytes(&rows)),
            u64_of(rec, "pos_fnv"),
            "{name}: position rows"
        );
        assert_eq!(fnv(&u16_bytes(&cls)), u64_of(rec, "cls_fnv"), "class row");
    }
}

#[test]
fn folded_rotation_and_scale_are_exact() {
    // rows [out][in] = [[1, 2], [3, 4], [5, 6], [7, 8]] in BF16
    let v: Vec<u16> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        .iter()
        .map(|&x| f32_to_bf16_rne(x))
        .collect();
    let r: Vec<f32> = rotate_pairs(&v, 2)
        .iter()
        .map(|&x| bf16_to_f32(x))
        .collect();
    assert_eq!(r, vec![-3.0, -4.0, 1.0, 2.0, -7.0, -8.0, 5.0, 6.0]);
    let s: Vec<f32> = scale_pow2(&v, 0.125)
        .iter()
        .map(|&x| bf16_to_f32(x))
        .collect();
    assert_eq!(s, vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]);
}

#[test]
fn bucket_ladder_of_today() {
    // the rungs (a tuning choice, pinned here, bounded by the verified laws)
    let rungs: Vec<u64> = [1, 15, 16, 17, 24, 25, 144, 193, 576, 1000, 1024, 1025]
        .iter()
        .map(|&n| ikv_round(n))
        .collect();
    assert_eq!(
        rungs,
        vec![16, 16, 16, 24, 24, 32, 192, 256, 768, 1024, 1024, 1536]
    );
    assert_eq!(ikv_bucket(1000, ikv_round(1000), 1024), 1024);
    assert_eq!(ikv_bucket(900, ikv_round(900), 960), 960);
    assert_eq!(ikv_seq_rows(16, 4), 65);
    assert!(ikv_key_live(64, 15, 4) == false && ikv_key_live(60, 15, 4));
}

#[test]
fn plan_of_the_largest_image() {
    let spec_text = ModelSpec::from_config_json(CONFIG).unwrap();
    let text = IkDims::from_spec(&spec_text, crate::engine::inkling2::IK_MAX_PREFILL_TILE).unwrap();
    let d = IkvDims::from_config_json(CONFIG).unwrap();
    let p = IkvPlan::new(
        d,
        &text,
        4096,
        16_384,
        crate::engine::inkling2::IK_MAX_PREFILL_TILE,
    )
    .unwrap();
    assert_eq!(
        (p.cap, p.max_rows, p.feat_cols),
        (1024, 4097, 1 + 16_384 + 1024)
    );
    // every layer's weights, once each
    assert_eq!(p.weights.len(), 1 + 2 + 2 + 23 * LAYER_WEIGHTS.len());
    let layer: u64 = LAYER_WEIGHTS
        .iter()
        .map(|k| p.weights[&format!("VL0_{k}")])
        .sum();
    // q, qr, k, kr, v, o (1024^2) + fc1, fc2 (1024 x 4096) in BF16, plus norms / biases / scales
    assert!(layer > (6 * 1024 * 1024 + 2 * 4096 * 1024) * 2);
    // the f32 scores of all heads dominate the activations
    assert_eq!(p.acts["vb_scores"], 4097 * 4097 * 16 * 4);
    assert!(p.act_bytes() < 5 << 30, "{} bytes", p.act_bytes());
    eprintln!(
        "tower plan at 4096 patches / 16384 tokens: weights {} bytes, activations {} bytes",
        p.weights_bytes(),
        p.act_bytes()
    );
    assert!(IkvPlan::new(d, &text, 3, 16, 256).is_err());
}

#[test]
fn requests_are_checked_against_the_plan() {
    let spec_text = ModelSpec::from_config_json(CONFIG).unwrap();
    let text = IkDims::from_spec(&spec_text, 256).unwrap();
    let d = IkvDims::from_config_json(CONFIG).unwrap();
    let mut v = IkVision {
        spec: spec(),
        plan: IkvPlan::new(d, &text, 4 * 144, 200, 256).unwrap(),
        text_hid: 2304,
        image_token_id: 200_054,
        precompile: Vec::new(),
        dump_dir: None,
        sets: HashMap::new(),
        text_embed: HashMap::new(),
        stage: None,
        weights_addr: BTreeMap::new(),
        act_addr: BTreeMap::new(),
        xbuf: [0; 2],
        feat: 0,
        pos_table: Vec::new(),
        cls_row: Vec::new(),
        pending: None,
        active: None,
        last_report: None,
        last_tokens: 0,
    };
    let s = spec();
    let small = prepare_tiles(&synthetic(200, 120), &s).unwrap(); // 15 tokens
    let big = prepare_tiles(&synthetic(448, 448), &s).unwrap(); // 144 tokens
    let wide = prepare_tiles(&synthetic(481, 440), &s).unwrap(); // 11 x 13 = 143
    assert_eq!(wide.n_tokens(), 143);
    v.set_request_images(vec![wide]).unwrap();
    let over = prepare_tiles(&synthetic(481, 481), &s).unwrap(); // 13 x 13 = 169
    let e = v.set_request_images(vec![over]).unwrap_err();
    assert!(
        e.contains("169 image tokens") && e.contains("at most 144"),
        "{e}"
    );
    // two big images: 288 > 200 tokens per request
    let e = v
        .set_request_images(vec![big.clone(), big.clone()])
        .unwrap_err();
    assert!(e.contains("288") || e.contains("200"), "{e}");
    v.set_request_images(vec![small.clone(), big]).unwrap();
    // placeholders must match the tokens
    let mut ids = vec![1, 2];
    ids.extend(std::iter::repeat_n(200_054, 15));
    ids.push(3);
    assert!(v
        .begin_request(&ids, false)
        .unwrap_err()
        .contains("15 image placeholder rows"));
    assert!(v.active.is_none() && v.pending.is_none());
    v.set_request_images(vec![small]).unwrap();
    assert!(v
        .begin_request(&ids, true)
        .unwrap_err()
        .contains("STEP_PREFILL"));
    // a text request after an image request has no images
    v.begin_request(&[1, 2, 3], false).unwrap();
    assert!(v.active.is_none() && v.last_report().is_none());
}

#[test]
fn text_embed_variant_binds_the_feature_buffer() {
    let spec_text = ModelSpec::from_config_json(CONFIG).unwrap();
    let text = IkDims::from_spec(&spec_text, 256).unwrap();
    let gb = build_text_embed_graph(&text, 1000, 64, &[]);
    let names: Vec<String> = gb
        .persistent_tensor_specs()
        .into_iter()
        .map(|(n, ..)| n)
        .collect();
    for want in [
        "ids", "embed", "w_embed", "x0g", "x0", "vis_feat", "vis_fidx", "vis_keep", "vis_xn",
        "vis_fg", "vis_xk",
    ] {
        assert!(names.iter().any(|n| n == want), "missing {want}");
    }
    let feat = gb
        .persistent_tensor_specs()
        .into_iter()
        .find(|(n, ..)| n == "vis_feat")
        .unwrap();
    assert_eq!(feat.2, vec![2304, 1000]);
}

fn data_url(bytes: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::from("data:image/png;base64,");
    for c in bytes.chunks(3) {
        let n = (u32::from(c[0]) << 16)
            | (u32::from(*c.get(1).unwrap_or(&0)) << 8)
            | u32::from(*c.get(2).unwrap_or(&0));
        for i in 0..4 {
            if i <= c.len() {
                s.push(A[((n >> (18 - 6 * i)) & 63) as usize] as char);
            } else {
                s.push('=');
            }
        }
    }
    s
}

#[test]
fn server_media_path_expands_one_placeholder_per_tile() {
    use crate::core::vision::MediaLimits;
    use crate::vision::media::{MediaRef, RequestMedia, VisionServe};
    let serve = VisionServe::new(spec(), 4096, 16_384, MediaLimits::default()).unwrap();
    let png = std::fs::read(fixture_dir().join("img_448x448.png")).unwrap();
    let refs = vec![MediaRef::Image(data_url(&png))];
    for with_pixels in [false, true] {
        let media = RequestMedia::prepare(&refs, &serve, with_pixels).unwrap();
        let c = media.counts();
        assert_eq!((c.images, c.patches, c.tokens), (1, 576, 144));
        let text = "<|message_user|><|content_image|><|unused_200054|><|end_message|><|message_user|><|content_text|>Describe the image.<|end_message|>";
        assert_eq!(media.expand_text(text).unwrap(), text);
        assert!(media
            .expand_text("<|message_user|><|content_text|>no image<|end_message|>")
            .is_err());
        let ids = [200_000, 200_005, 200_054, 200_010, 9];
        let out = media.expand_ids(&ids, &serve.spec).unwrap();
        assert_eq!(out.len(), 4 + 144);
        assert_eq!(out.iter().filter(|&&t| t == 200_054).count(), 144);
    }
    // the reference prompt: the placeholder run of the image is its 144 tokens
    let small = VisionServe::new(spec(), 4 * 100, 16_384, MediaLimits::default()).unwrap();
    let e = RequestMedia::prepare(&refs, &small, false).unwrap_err();
    assert!(e.contains("576 patches") && e.contains("400"), "{e}");
}
