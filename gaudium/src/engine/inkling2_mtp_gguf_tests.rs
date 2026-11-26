//! Host tests of the drafter-file source: which file beside a GGUF model is chosen (synthetic
//! drafter files), the contract from a file's metadata and its refusals, and, with the real
//! files, every device tensor of the published drafter from the checkpoint and from both
//! drafter files.

use super::*;
use crate::gguf::tests::{temp_dir, Builder};
use crate::safetensors::SafetensorsStore;

// ---------------------------------------------------------------------------
// Synthetic drafter files
// ---------------------------------------------------------------------------

/// The toy drafter: hidden 128, one head of 128, MLP 128, rank 128, 3 layers, block 4.
const HID: u64 = 128;
const RANK: u64 = 128;
const BLOCK: u64 = 4;

fn kv_f32(b: &mut Builder, key: &str, x: f32) {
    b.kv_raw(key, 6, &x.to_le_bytes());
}

fn kv_array(b: &mut Builder, key: &str, elem: u32, items: &[Vec<u8>]) {
    let mut raw = elem.to_le_bytes().to_vec();
    raw.extend_from_slice(&(items.len() as u64).to_le_bytes());
    for it in items {
        raw.extend_from_slice(it);
    }
    b.kv_raw(key, 9, &raw);
}

/// What a synthetic drafter file states.
struct Toy {
    arch: &'static str,
    folded: bool,
    hid: u64,
    vocab: u64,
    layers_present: u64,
    window: u32,
    mask_rows: u64,
    bulk: GgmlType,
    /// Rows of the token table `markov_w1` of a folded file; 0: no table.
    w1_rows: u64,
}

impl Default for Toy {
    fn default() -> Self {
        Self {
            arch: ARCH,
            folded: false,
            hid: HID,
            vocab: 6,
            layers_present: 3,
            window: 16,
            mask_rows: BLOCK - 1,
            bulk: GgmlType::BF16,
            w1_rows: 0,
        }
    }
}

fn zeros(n: u64, ty: GgmlType) -> Vec<u8> {
    let bytes = match ty {
        GgmlType::Q8_0 => n / 32 * 34,
        GgmlType::F32 => n * 4,
        _ => n * 2,
    };
    vec![0u8; bytes as usize]
}

fn toy_bytes(t: &Toy) -> Vec<u8> {
    let mut b = Builder::new();
    let a = |k: &str| format!("{ARCH}.{k}");
    b.kv_str("general.architecture", t.arch)
        .kv_u32(&a("embedding_length"), t.hid as u32)
        .kv_u32(&a("block_count"), 3)
        .kv_u32(&a("feed_forward_length"), 128)
        .kv_u32(&a("attention.head_count"), 1)
        .kv_u32(&a("attention.head_count_kv"), 1)
        .kv_u32(&a("attention.key_length"), 128)
        .kv_u32(&a("attention.sliding_window"), t.window)
        .kv_u32(&a("block_size"), BLOCK as u32)
        .kv_u32(&a("vocab_size_unpadded"), t.vocab as u32);
    kv_f32(&mut b, &a("rope.freq_base"), 1.0e6);
    kv_f32(&mut b, &a("attention.layer_norm_rms_epsilon"), 1e-6);
    kv_array(
        &mut b,
        &a("target_layers"),
        4,
        &[5u32, 23, 41, 48].map(|x| x.to_le_bytes().to_vec()),
    );
    kv_array(
        &mut b,
        &a("attention.sliding_window_pattern"),
        7,
        &[vec![1], vec![0], vec![1]],
    );
    // the recurrent head [state | previous | row | context] -> [gate | candidate | latent]
    let cols = 2 * RANK + 2 * t.hid;
    b.tensor(
        "joint_proj.weight",
        &[cols, 3 * RANK],
        t.bulk,
        zeros(cols * 3 * RANK, t.bulk),
    );
    for l in 0..t.layers_present {
        b.tensor(
            &format!("blk.{l}.attn_q.weight"),
            &[t.hid, 128],
            t.bulk,
            zeros(t.hid * 128, t.bulk),
        );
    }
    if t.folded {
        b.tensor(
            "blk.0.ctx_mix.weight",
            &[4],
            GgmlType::F32,
            zeros(4, GgmlType::F32),
        )
        .tensor(
            "markov_w2.weight",
            &[RANK, t.vocab],
            GgmlType::BF16,
            zeros(RANK * t.vocab, GgmlType::BF16),
        )
        .tensor(
            "mask_embd.weight",
            &[t.hid, t.mask_rows],
            GgmlType::F32,
            zeros(t.hid * t.mask_rows, GgmlType::F32),
        );
        if t.w1_rows > 0 {
            b.tensor(
                "markov_w1.weight",
                &[RANK, t.w1_rows],
                GgmlType::BF16,
                zeros(RANK * t.w1_rows, GgmlType::BF16),
            );
        }
    } else {
        b.tensor(
            "dspark_fuse.weight",
            &[4, 4],
            GgmlType::F32,
            zeros(16, GgmlType::F32),
        );
    }
    b.build()
}

fn target() -> Target {
    Target { hid: HID, vocab: 6 }
}

/// A model directory holding `model.gguf` (a file of another architecture: the drafter search
/// never opens it as a drafter) and the given files.
fn model_dir(tag: &str, files: &[(&str, Vec<u8>)]) -> (PathBuf, PathBuf) {
    let dir = temp_dir(tag);
    let mut main = Builder::new();
    main.kv_str("general.architecture", "inkling2_loop");
    let model = dir.join("model.gguf");
    std::fs::write(&model, main.build()).unwrap();
    for (name, bytes) in files {
        let p = dir.join(name);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, bytes).unwrap();
    }
    (dir, model)
}

fn name_of(p: &Path) -> String {
    let parent = p
        .parent()
        .and_then(|d| d.file_name())
        .unwrap()
        .to_string_lossy();
    format!("{parent}/{}", p.file_name().unwrap().to_string_lossy())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn checkpoint_names_map_to_drafter_file_names() {
    let p = |s: &str| tensor_name(Layout::Plain, s);
    let f = |s: &str| tensor_name(Layout::Folded, s);
    assert_eq!(p("fuse_weight").unwrap(), "dspark_fuse.weight");
    assert_eq!(f("fuse_weight"), None);
    assert_eq!(p("mask_embed").unwrap(), "mask_embd.weight");
    assert_eq!(f("mask_embed"), None);
    assert_eq!(f("markov_head.hidden_out.weight"), None);
    assert_eq!(
        p("layers.2.dflare_norm.weight").unwrap(),
        "blk.2.dflare_norm.weight"
    );
    assert_eq!(
        f("layers.2.dflare_norm.weight").unwrap(),
        "blk.2.ctx_norm.weight"
    );
    assert_eq!(
        f("markov_head.dflare_norm.weight").unwrap(),
        "markov_ctx_norm.weight"
    );
    assert_eq!(
        p("layers.1.self_attn.k_norm_ctx.weight").unwrap(),
        "blk.1.attn_k_ctx_norm.weight"
    );
    assert_eq!(
        p("layers.0.self_attn.o_proj.weight").unwrap(),
        "blk.0.attn_output.weight"
    );
    assert_eq!(
        p("layers.0.mlp.down_proj.weight").unwrap(),
        "blk.0.ffn_down.weight"
    );
    assert_eq!(p("markov_head.rnn_proj.bias").unwrap(), "joint_proj.bias");
    assert_eq!(p("layers.x.mlp.down_proj.weight"), None);
}

/// Today's preference among matching drafter files: BF16 (bit-exact to the publisher's
/// weights) before F32 before Q8_0 before anything else. A tuning choice, pinned here.
#[test]
fn bf16_drafter_files_are_preferred() {
    use MtpfPrecision::*;
    let r: Vec<u64> = [MtpfBf16, MtpfF32, MtpfQ8, MtpfOther]
        .into_iter()
        .map(mtpf_rank)
        .collect();
    assert!(r.windows(2).all(|w| w[0] < w[1]), "{r:?}");
}

#[test]
fn a_token_table_is_padded_to_its_device_rows_and_an_unsizable_one_refused() {
    let bytes = toy_bytes(&Toy {
        folded: true,
        w1_rows: 4,
        ..Toy::default()
    });
    let (dir, _) = model_dir("mtpf_w1", &[("drafter-BF16.gguf", bytes)]);
    let f = GgufMtp::open(&dir.join("drafter-BF16.gguf")).unwrap();
    // the file's 4 rows, then zero rows up to the device tensor's 8
    let t = f.table("markov_w1.weight", RANK, 4, 8).unwrap();
    assert_eq!(t.len(), (8 * RANK * 2) as usize);
    // a device tensor whose bytes overflow u64 (its rows come from the model's config) is an
    // error, not a wrapped size
    let e = f
        .table("markov_w1.weight", RANK, 4, u64::MAX / 2)
        .unwrap_err();
    assert!(e.contains("overflows"), "{e}");
}

#[test]
fn the_best_matching_file_is_chosen_beside_the_model_or_in_dspark() {
    let q8 = toy_bytes(&Toy {
        folded: true,
        bulk: GgmlType::Q8_0,
        ..Toy::default()
    });
    let bf = toy_bytes(&Toy::default());
    // a Q8_0 file beside the model, the BF16 one in dspark/: the BF16 file wins
    let (dir, model) = model_dir(
        "mtpf_pick",
        &[
            ("drafter-Q8_0.gguf", q8.clone()),
            ("dspark/drafter-BF16.gguf", bf.clone()),
            ("mmproj-BF16.gguf", bf.clone()),
        ],
    );
    let c = candidates(&model);
    // the model's own file and the projector file are no candidates
    assert_eq!(
        c.iter().map(|p| name_of(p)).collect::<Vec<_>>(),
        vec![
            format!(
                "{}/drafter-Q8_0.gguf",
                dir.file_name().unwrap().to_string_lossy()
            ),
            "dspark/drafter-BF16.gguf".to_string()
        ]
    );
    let got = find_drafter(&model, target()).unwrap();
    assert_eq!(name_of(&got), "dspark/drafter-BF16.gguf");
    assert_eq!(
        precision(&Gguf::open(&got).unwrap()),
        MtpfPrecision::MtpfBf16
    );
    // only the Q8_0 file: it is used
    std::fs::remove_dir_all(dir.join("dspark")).unwrap();
    let got = find_drafter(&model, target()).unwrap();
    assert!(got.ends_with("drafter-Q8_0.gguf"));
    assert_eq!(precision(&Gguf::open(&got).unwrap()), MtpfPrecision::MtpfQ8);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn files_of_another_model_are_never_chosen() {
    let wide = toy_bytes(&Toy {
        hid: 256,
        ..Toy::default()
    });
    let vocab = toy_bytes(&Toy {
        vocab: 7,
        ..Toy::default()
    });
    let other = toy_bytes(&Toy {
        arch: "qwen3",
        ..Toy::default()
    });
    let (dir, model) = model_dir(
        "mtpf_other",
        &[
            ("a.gguf", wide),
            ("b.gguf", vocab),
            ("dspark/c.gguf", other),
        ],
    );
    let e = find_drafter(&model, target()).unwrap_err();
    assert!(e.contains("hidden size 256, the model's is 128"), "{e}");
    assert!(
        e.contains("vocabulary of 7 tokens, the model's has 6"),
        "{e}"
    );
    assert!(e.contains("architecture 'qwen3'"), "{e}");
    assert!(
        e.contains(&dir.join(DRAFTER_DIR).display().to_string()),
        "{e}"
    );
    std::fs::remove_dir_all(&dir).ok();
    // nothing beside the model: the refusal names where it looked
    let (dir, model) = model_dir("mtpf_none", &[]);
    let e = find_drafter(&model, target()).unwrap_err();
    assert!(e.contains("no drafter file"), "{e}");
    assert!(e.contains(&dir.display().to_string()), "{e}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_contract_comes_from_the_metadata() {
    let (dir, model) = model_dir(
        "mtpf_cfg",
        &[
            ("plain.gguf", toy_bytes(&Toy::default())),
            (
                "folded.gguf",
                toy_bytes(&Toy {
                    folded: true,
                    window: 17,
                    ..Toy::default()
                }),
            ),
        ],
    );
    let _ = model;
    let cfg = |name: &str| {
        let p = dir.join(name);
        drafter_config(&Gguf::open(&p).unwrap(), &p, target())
    };
    let c = cfg("plain.gguf").unwrap();
    assert_eq!(
        (c.hid, c.nh, c.nkv, c.ff, c.block, c.rank, c.window),
        (128, 1, 1, 128, BLOCK, RANK, 16)
    );
    assert_eq!(c.local, vec![true, false, true]);
    assert_eq!(c.taps, vec![5, 23, 41, 48]);
    assert_eq!(c.weights, MtpWeights::Gguf(dir.join("plain.gguf")));
    // the folded layout's window counts the anchor: one more than the positions read
    assert_eq!(cfg("folded.gguf").unwrap().window, 16);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_file_that_disagrees_is_refused_with_both_numbers() {
    let (dir, _) = model_dir(
        "mtpf_bad",
        &[
            (
                "hid.gguf",
                toy_bytes(&Toy {
                    hid: 256,
                    ..Toy::default()
                }),
            ),
            (
                "voc.gguf",
                toy_bytes(&Toy {
                    vocab: 9,
                    ..Toy::default()
                }),
            ),
            (
                "layers.gguf",
                toy_bytes(&Toy {
                    layers_present: 2,
                    ..Toy::default()
                }),
            ),
            (
                "mask.gguf",
                toy_bytes(&Toy {
                    folded: true,
                    mask_rows: 2,
                    ..Toy::default()
                }),
            ),
        ],
    );
    let cfg = |name: &str| {
        let p = dir.join(name);
        drafter_config(&Gguf::open(&p).unwrap(), &p, target()).unwrap_err()
    };
    assert!(cfg("hid.gguf").contains("hidden size 256, the model's is 128"));
    assert!(cfg("voc.gguf").contains("vocabulary of 9 tokens, the model's has 6"));
    assert!(cfg("layers.gguf").contains("block_count is 3, the file holds 2 layers"));
    let e = cfg("mask.gguf");
    assert!(
        e.contains("block_size 4 needs 3 mask rows") && e.contains("[2, 128]"),
        "{e}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

// ---------------------------------------------------------------------------
// The real files
// ---------------------------------------------------------------------------

fn bf16_vals(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<2>()
        .0
        .iter()
        .map(|x| bf16_to_f32(*x))
        .collect()
}

fn f32_vals(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|x| f32::from_le_bytes(*x))
        .collect()
}

/// Largest difference relative to the tensor's largest value, and the count of values that
/// differ.
fn rel_diff(a: &[f32], b: &[f32]) -> (f64, usize) {
    let top = b
        .iter()
        .fold(0f64, |m, &x| m.max((x as f64).abs()))
        .max(1e-30);
    let mut worst = 0f64;
    let mut n = 0;
    for (&x, &y) in a.iter().zip(b) {
        if x.to_bits() != y.to_bits() {
            n += 1;
            worst = worst.max(((x - y) as f64).abs() / top);
        }
    }
    (worst, n)
}

/// Drafter weights that are f32 on the card.
fn is_f32_weight(name: &str) -> bool {
    matches!(
        name,
        "mtw_alpha" | "mtw_rnnb" | "mtw_rope_hi" | "mtw_rope_lo"
    )
}

/// The publisher's checkpoint and the GGUF build's drafter files give the same drafter: the same
/// contract, every device tensor byte for byte from the BF16 file, and (reported) the Q8_0 file's
/// distance per device tensor. Needs both forms of the model: `GAUDIUM_INKLING2_DIR` (the
/// checkpoint directory) and `GAUDIUM_INKLING2_GGUF` (the main GGUF file; its drafter files lie
/// beside it and in `dspark/`).
#[test]
#[ignore = "needs GAUDIUM_INKLING2_DIR and GAUDIUM_INKLING2_GGUF (the same model in both forms)"]
fn drafter_files_and_checkpoint_pack_the_same_drafter() {
    let (Some(dir), Some(file)) = (
        std::env::var_os("GAUDIUM_INKLING2_DIR"),
        std::env::var_os("GAUDIUM_INKLING2_GGUF"),
    ) else {
        panic!("set GAUDIUM_INKLING2_DIR and GAUDIUM_INKLING2_GGUF");
    };
    let (dir, file) = (PathBuf::from(dir), PathBuf::from(file));
    let main = Gguf::open(&file).unwrap();
    let cfg_json = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let ck = MtpConfig::from_config_json(&cfg_json).unwrap();
    let spec = crate::model_spec::ModelSpec::from_config_json(&cfg_json).unwrap();
    let d = IkDims::from_spec(&spec, 256).unwrap();

    // the BF16 file is chosen, and both files give the checkpoint's contract
    let chosen = model_drafter(&file, &main).unwrap();
    let MtpWeights::Gguf(bf_path) = chosen.weights.clone() else {
        panic!("a GGUF drafter")
    };
    eprintln!("drafter file: {}", bf_path.display());
    let bf_g = Gguf::open(&bf_path).unwrap();
    assert_eq!(precision(&bf_g), MtpfPrecision::MtpfBf16);
    let t = target_of(&main).unwrap();
    let q8_path = candidates(&file)
        .into_iter()
        .find(|p| Gguf::open(p).is_ok_and(|g| layout(&g) == Some(Layout::Folded)))
        .expect("the Q8_0 drafter file beside the model");
    let q8_cfg = drafter_config(&Gguf::open(&q8_path).unwrap(), &q8_path, t).unwrap();
    for c in [&chosen, &q8_cfg] {
        let mut c = c.clone();
        c.weights = MtpWeights::Checkpoint;
        assert_eq!(c, ck, "contract");
    }

    let g = MtpGeom::new(ck.clone(), &d, 8192, 256).unwrap();
    let plan = MtpPlan::build(&g);
    let store = SafetensorsStore::open_file(dir.join(MTP_FILE)).unwrap();
    let st = StMtp(&store);
    let bf = GgufMtp::open(&bf_path).unwrap();
    let q8 = GgufMtp::open(&q8_path).unwrap();
    assert!(bf.folds_on_card() && !q8.folds_on_card());
    let (pk_st, pk_bf, pk_q8) = (
        MtpPacker::new(&st, &g),
        MtpPacker::new(&bf, &g),
        MtpPacker::new(&q8, &g),
    );
    let mut bytes = 0usize;
    let mut report = Vec::new();
    for name in plan.weights.keys() {
        let x = pk_st.pack(name).unwrap();
        let y = pk_bf.pack(name).unwrap();
        assert!(x == y, "BF16 drafter file: device tensor '{name}' differs");
        bytes += x.as_ref().map_or(0, Vec::len);
        let z = pk_q8.pack(name).unwrap();
        if name == "mtw_w1" {
            // the rows a token id can reach are the checkpoint's; only the padding rows are zero
            let keep = (g.vocab_out * g.c.rank * 2) as usize;
            let (x, z) = (x.as_ref().unwrap(), z.as_ref().unwrap());
            assert!(
                x[..keep] == z[..keep],
                "Q8_0 drafter file: mtw_w1 rows below the vocabulary"
            );
            assert!(z[keep..].iter().all(|&b| b == 0));
        }
        match (&x, &z) {
            (Some(x), Some(z)) => {
                assert_eq!(x.len(), z.len(), "{name}");
                let (a, b) = if is_f32_weight(name) {
                    (f32_vals(z), f32_vals(x))
                } else {
                    (bf16_vals(z), bf16_vals(x))
                };
                let (worst, n) = rel_diff(&a, &b);
                report.push(format!(
                    "{name:14} {worst:.3e} ({n} of {} values differ)",
                    a.len()
                ));
            }
            (None, Some(_)) => report.push(format!("{name:14} stored in the Q8_0 file")),
            (_, None) => report.push(format!("{name:14} not uploaded from the Q8_0 file")),
        }
    }
    eprintln!(
        "BF16 drafter file: {} device tensors, {bytes} bytes, all identical",
        plan.weights.len()
    );

    // the fold the card computes from the checkpoint against the Q8_0 file's stored one, on a
    // sample of vocabulary rows (host f64, rounded to BF16 as the card stores it)
    let main_st = SafetensorsStore::open(&dir).unwrap();
    let un = main_st
        .names()
        .into_iter()
        .find(|n| n.ends_with("unembed.weight"))
        .map(str::to_string)
        .unwrap();
    let u = main_st.tensor_bytes(&un).unwrap().unwrap();
    let ho = bf16_vals(
        &st.bf16("markov_head.hidden_out.weight", &[g.c.hid, g.c.rank])
            .unwrap(),
    );
    let hb = bf16_vals(&st.bf16("markov_head.hidden_out.bias", &[g.c.hid]).unwrap());
    let (fold, fb) = (
        bf16_vals(&pk_q8.pack("mtw_fold").unwrap().unwrap()),
        bf16_vals(&pk_q8.pack("mtw_bf").unwrap().unwrap()),
    );
    let (h, rk) = (g.c.hid as usize, g.c.rank as usize);
    let (mut ref_w, mut got_w, mut ref_b, mut got_b) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for v in (0..g.vocab_out as usize).step_by(97) {
        let row = bf16_vals(&u[v * h * 2..(v + 1) * h * 2]);
        for j in 0..rk {
            let s: f64 = (0..h).map(|i| row[i] as f64 * ho[i * rk + j] as f64).sum();
            ref_w.push(crate::fp8::bf16_to_f32(crate::fp8::f32_to_bf16_rne(
                s as f32,
            )));
            got_w.push(fold[v * rk + j]);
        }
        let s: f64 = (0..h).map(|i| row[i] as f64 * hb[i] as f64).sum();
        ref_b.push(crate::fp8::bf16_to_f32(crate::fp8::f32_to_bf16_rne(
            s as f32,
        )));
        got_b.push(fb[v]);
    }
    for (name, a, b) in [("mtw_fold", &got_w, &ref_w), ("mtw_bf", &got_b, &ref_b)] {
        let (worst, n) = rel_diff(a, b);
        report.push(format!(
            "{name:14} {worst:.3e} ({n} of {} sampled values differ from the fold of the checkpoint)",
            a.len()
        ));
    }
    eprintln!(
        "Q8_0 drafter file against the checkpoint (largest difference / largest value):\n  {}",
        report.join("\n  ")
    );
}
