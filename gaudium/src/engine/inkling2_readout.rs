//! What the readout learns from the Inkling 2 engine: the speculative runtime it was loaded with
//! ([`SpecInfo`]) and the byte model of its launches for the live record's `profile` block.
//!
//! A child module of `engine::inkling2`, so it reads that module's tensor-role rules. Host-only
//! bookkeeping: each launch's graph description is rebuilt with the builder its recipe was
//! compiled from (same shapes, no kernel hashes: only the names and sizes matter here) and is
//! never compiled, launched or fingerprinted. The live reporter caches each result by key, so a
//! model is built once per recipe shape.
//!
//! What is charged ([`Traffic`]): every persistent tensor a node reads or writes. The experts run
//! dense-masked, so every expert matrix of every layer application is charged on every pass (no
//! cap by the experts a row selects), and a looped layer's matrices are charged once per
//! application. A global layer's cache is read over the chunks of its recipe's bucket, a sliding
//! layer's ring in full; a pass writes its own rows of either. The short convolutions' history
//! rings are charged in full on every pass (an upper bound: the kernels touch a few positions of
//! them; together they are well under 1% of a step).
//!
//! Phases: in the decode step (the verify pass) the head (logits and arg-max) is `readout` and
//! every other node is the verify trunk; every node of the drafter's two launches (its block,
//! whose own readout drafts the tokens, and its context pass over the verified rows) is `drafts`.
//! Not modelled: the vision tower, an image tile's embedding, the host's sampling.

use super::mtp::{build_block_graph, build_ctx_graph, MtpGeom};
use super::*;
use crate::engine::qwen3::spec::readout::SpecInfo;
use crate::live_profile::{Kind, Phase, Traffic, TrafficRules};

/// The data kind of a tensor of any Inkling 2 graph, the drafter's included.
fn kind_of_name(name: &str) -> Kind {
    if is_kv_name(name) || name.starts_with("mts_") {
        // global caches, sliding rings, the drafter's context keys / values
        Kind::Kv
    } else if is_hist_name(name) {
        // the short convolutions' history rings
        Kind::State
    } else if is_weight_name(name) || name.starts_with("mtw_") {
        Kind::Weight
    } else {
        Kind::Act
    }
}

/// One launch of `gb`: `rows` rows per pass, global caches of `nch` chunks.
fn charge(gb: &GraphBuilder, rows: u64, nch: u64) -> Traffic {
    Traffic::from_graph_kinds(
        gb,
        TrafficRules {
            rows,
            kv_positions: IK_CHUNK * nch,
            experts_per_row: 0,
        },
        &kind_of_name,
    )
}

/// Moves the bytes and FLOPs of every phase except `keep` and `to` into `to`.
fn fold_phases(mut tr: Traffic, keep: Option<Phase>, to: Phase) -> Traffic {
    let t = to as usize;
    for p in 0..4 {
        if p == t || keep.is_some_and(|k| k as usize == p) {
            continue;
        }
        for k in 0..4 {
            tr.bytes[t][k] += tr.bytes[p][k];
            tr.bytes[p][k] = 0.0;
        }
        tr.flops[t] += tr.flops[p];
        tr.flops[p] = 0.0;
    }
    tr
}

/// One launch of the decode step recipe of `rows` rows over `bucket` chunks (a one-row step, or
/// the verify pass of `rows - 1` drafts).
pub(super) fn verify(d: &IkDims, lmax: u64, rows: u64, bucket: u64) -> Traffic {
    let gb = build_step_graph(d, lmax, bucket, rows, &[]);
    fold_phases(
        charge(&gb, rows, bucket),
        Some(Phase::Readout),
        Phase::Verify,
    )
}

/// One launch of the drafter's block recipe whose global layer reads `nch` chunks.
pub(super) fn draft_block(g: &MtpGeom, nch: u64) -> Traffic {
    let gb = build_block_graph(g, nch, &[]);
    fold_phases(charge(&gb, g.c.block, nch), None, Phase::Draft)
}

/// One launch of the drafter's context recipe over `rows` rows.
pub(super) fn draft_ctx(g: &MtpGeom, rows: u64) -> Traffic {
    let gb = build_ctx_graph(g, rows, &[]);
    fold_phases(charge(&gb, rows, g.max_chunks), None, Phase::Draft)
}

/// The launches one `step` of the decode loop consumes: the verify pass of `rows` rows over
/// `bucket` chunks and, with the block drafter (`drafter`: its geometry and the chunks its block
/// reads), the drafter's block before it and its context pass over the same rows after it.
pub(super) fn spec_step(
    d: &IkDims,
    lmax: u64,
    rows: u64,
    bucket: u64,
    drafter: Option<(&MtpGeom, u64)>,
) -> Traffic {
    let mut tr = verify(d, lmax, rows, bucket);
    if let Some((g, nch)) = drafter {
        tr.add(&draft_block(g, nch), 1.0);
        tr.add(&draft_ctx(g, rows), 1.0);
    }
    tr
}

/// Key of [`spec_step`]'s launches in the live reporter's cache.
pub(super) fn step_key(rows: u64, bucket: u64, draft_chunks: Option<u64>) -> u64 {
    ((rows & 0xffff) << 32)
        | ((bucket & 0xffff) << 16)
        | draft_chunks.map_or(0, |n| (n + 1) & 0xffff)
}

/// The launches of one prefill tile of `t` rows whose global layers read `nch` chunks: the
/// embedding, every layer application and, with the block drafter, the final mix and the
/// drafter's context pass over the tile. The `last` tile also carries the readout of the prompt's
/// last row (and, without the drafter, the final mix it reads).
pub(super) fn prefill_tile(
    d: &IkDims,
    lmax: u64,
    t: u64,
    nch: u64,
    drafter: Option<&MtpGeom>,
    last: bool,
) -> Traffic {
    let mut tr = charge(&build_embed_graph(d, t, &[]), t, nch);
    let mut kinds: Vec<(IkKind, f64)> = Vec::new();
    for app in d.run_apps() {
        let k = IkKind {
            sliding: app.sliding,
            msgs: app.msgs,
        };
        match kinds.iter_mut().find(|(c, _)| *c == k) {
            Some(e) => e.1 += 1.0,
            None => kinds.push((k, 1.0)),
        }
    }
    for (k, n) in kinds {
        tr.add(
            &charge(&build_app_graph(d, k, t, lmax, nch, &[]), t, nch),
            n,
        );
    }
    let fin = charge(&build_final_graph(d, t, &[]), t, nch);
    if let Some(g) = drafter {
        tr.add(&fin, 1.0);
        tr.add(&draft_ctx(g, t), 1.0);
    }
    if last {
        if drafter.is_none() {
            tr.add(&fin, 1.0);
        }
        tr.add(&charge(&build_head_graph(d, &[]), 1, nch), 1.0);
    }
    tr
}

/// Launches of [`prefill_tile`] (the verified count: `verified/inkling2_readout.bend`).
pub(super) fn prefill_tile_launches(d: &IkDims, drafter: bool, last: bool) -> u64 {
    crate::verified::ik_tile_launches(d.n_apps as u64, drafter, last)
}

/// Key of [`prefill_tile`] in the live reporter's cache.
pub(super) fn tile_key(t: u64, nch: u64, drafter: bool, last: bool) -> u64 {
    (t & 0xffff) | ((nch & 0xffff) << 16) | (u64::from(drafter) << 32) | (u64::from(last) << 33)
}

/// Data type of the block drafter's own projection matrices (its first layer's context
/// projection), read off its context graph.
fn drafter_matrix(g: &MtpGeom) -> Option<SynDataType> {
    build_ctx_graph(g, 1, &[])
        .persistent_tensor_specs()
        .into_iter()
        .find(|t| t.0 == "mtw_L0_Wkvc")
        .map(|t| t.1)
}

/// [`Inkling2Engine::spec_info`] of an engine loaded with `sc`, verifying `spec_rows` rows.
fn spec_info_of(sc: &SpecConfig, spec_rows: u64, drafter_matrix: Option<SynDataType>) -> SpecInfo {
    SpecInfo {
        method: sc.method_name(),
        draft_k: vec![spec_rows.saturating_sub(1)],
        // the verify pass samples each row at its own position (`decode_spec`)
        samples: true,
        tree: None,
        drafter_matrix,
    }
}

/// The live record's method label of a request that runs the drafter `sc` (`None`: `K = 0`),
/// in the label form the other engines use: `<method> K=<drafts> <precision> <sampling>`.
fn live_label_of(sc: Option<&SpecConfig>, spec_rows: u64, temperature: f32) -> String {
    let sampling = if temperature > 0.0 {
        format!("T={temperature:.2}")
    } else {
        "greedy".to_string()
    };
    match sc {
        Some(sc) => format!(
            "{} K={} BF16 {sampling}",
            sc.method_name(),
            spec_rows.saturating_sub(1)
        ),
        None => format!("K=0 BF16 {sampling}"),
    }
}

impl Inkling2Engine {
    /// The speculative runtime this engine was loaded with (`None`: a `K = 0` engine). The
    /// block drafter runs a plain chain of drafts (`mtp`) and verifies sampled requests too.
    pub fn spec_info(&self) -> Option<SpecInfo> {
        let sc = self.draft.as_ref()?;
        let matrix = self.mtp.as_ref().and_then(|m| drafter_matrix(&m.g));
        Some(spec_info_of(sc, self.spec_rows, matrix))
    }

    /// The live record's method label of a request: what its decode runs (a request that asks
    /// for the plain loop runs `K = 0` on an engine with a drafter).
    pub(super) fn live_method_label(&self, params: &GenParams) -> String {
        let sc = self.draft.as_ref().filter(|_| !params.non_speculative);
        live_label_of(sc, self.spec_rows, params.temperature)
    }
}

#[cfg(test)]
mod tests {
    use super::super::mtp::MtpConfig;
    use super::*;

    const MTP_JSON: &str = r#"{"mtp_config": {"hidden_size": 2304, "num_hidden_layers": 3,
        "num_attention_heads": 24, "num_key_value_heads": 8, "head_dim": 128,
        "intermediate_size": 9216, "q_bias": false, "o_bias": false, "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0, "local_layer_ids": [0, 2], "sliding_window_size": 1024,
        "block_size": 8, "markov_rank": 256, "markov_head_type": "rnn_hidden",
        "target_layer_ids": [5, 23, 41, 48]}}"#;

    const CTX: u64 = 262144;

    fn dims(rows: u64) -> IkDims {
        IkDims::from_spec(&super::super::tests::mini(), rows).unwrap()
    }

    fn geom(d: &IkDims) -> MtpGeom {
        MtpGeom::new(MtpConfig::from_config_json(MTP_JSON).unwrap(), d, CTX, 256).unwrap()
    }

    const GB: f64 = 1e9;
    const W: usize = Kind::Weight as usize;

    #[test]
    fn names_take_their_kinds() {
        assert_eq!(kind_of_name("L7_W_gate"), Kind::Weight);
        assert_eq!(kind_of_name("A_ctl"), Kind::Weight);
        assert_eq!(kind_of_name("W_head"), Kind::Weight);
        assert_eq!(kind_of_name("mtw_L0_Wkvc"), Kind::Weight);
        assert_eq!(kind_of_name("A5_kcache"), Kind::Kv);
        assert_eq!(kind_of_name("A_vring"), Kind::Kv);
        assert_eq!(kind_of_name("mts_L1_k"), Kind::Kv);
        assert_eq!(kind_of_name("A3_mraw"), Kind::State);
        assert_eq!(kind_of_name("A3_scores"), Kind::Act);
        assert_eq!(kind_of_name("mti_tap0"), Kind::Act);
    }

    #[test]
    fn a_decode_step_reads_every_expert_of_every_application() {
        let d = dims(8);
        let one = verify(&d, CTX, 1, 1);
        // 48 applications of 128 experts x 3 matrices of 2304 x 1152 BF16: about 98 GB
        let experts = 48.0 * 128.0 * 3.0 * 2304.0 * 1152.0 * 2.0;
        let w = one.bytes[Phase::Verify as usize][W];
        assert!(
            w > experts && w < 1.06 * experts,
            "verify weights {:.2} GB",
            w / GB
        );
        // the readout is the head matrix (and its logits), nothing else
        let head = 2304.0 * 200064.0 * 2.0;
        let ro = one.by_phase(Phase::Readout as usize);
        assert!(ro > head && ro < 1.01 * head, "readout {:.3} GB", ro / GB);
        assert_eq!(one.by_phase(Phase::Draft as usize), 0.0);
        assert_eq!(one.by_phase(Phase::Other as usize), 0.0);
        // 98 GB in the 51.5 ms one-row step measured on the card stays under the HBM peak
        let gbps = one.total() / 0.0515 / GB;
        assert!(gbps < crate::live_profile::HBM_PEAK_GBPS, "{gbps:.0} GB/s");
    }

    #[test]
    fn wider_passes_and_longer_contexts_add_attention_not_weights() {
        let d = dims(8);
        let one = verify(&d, CTX, 1, 1);
        let wide = verify(&d, CTX, 8, 1);
        let far = verify(&d, CTX, 1, 16);
        let w = |t: &Traffic| t.by_kind(W);
        assert!((w(&wide) / w(&one) - 1.0).abs() < 1e-3);
        assert!((w(&far) / w(&one) - 1.0).abs() < 1e-3);
        // the 8 global applications read their caches over the bucket: 16 chunks of K and V
        let kv = |t: &Traffic| t.by_kind(Kind::Kv as usize);
        let cache_16 = 8.0 * 2.0 * 128.0 * 4096.0 * 8.0 * 16.0 * 2.0;
        assert!(kv(&far) - kv(&one) > 0.9 * cache_16 * 15.0 / 16.0);
        assert!(wide.by_kind(Kind::Act as usize) > one.by_kind(Kind::Act as usize));
        // a cache write is the pass's rows, not the cache
        assert!(kv(&wide) - kv(&one) < 0.01 * kv(&one));
    }

    #[test]
    fn the_drafter_is_charged_as_drafts_and_costs_a_few_percent() {
        let d = dims(8);
        let g = geom(&d);
        let plain = spec_step(&d, CTX, 8, 1, None);
        let st = spec_step(&d, CTX, 8, 1, Some((&g, 1)));
        let dr = st.by_phase(Phase::Draft as usize);
        assert_eq!(plain.by_phase(Phase::Draft as usize), 0.0);
        assert!(
            (st.by_phase(Phase::Verify as usize) - plain.by_phase(Phase::Verify as usize)).abs()
                < 1.0
        );
        // the block reads the head matrix and the drafter's own (about 0.75 GB of them)
        let head = 2304.0 * 200064.0 * 2.0;
        assert!(
            dr > head + 0.5e9 && dr < 0.05 * st.total(),
            "drafts {:.3} GB",
            dr / GB
        );
        // the context pass writes the verify's rows of the drafter's caches, not the caches
        let ctx = draft_ctx(&g, 8);
        assert!(
            ctx.by_kind(Kind::Kv as usize) < 1e6,
            "{}",
            ctx.by_kind(Kind::Kv as usize)
        );
        assert_ne!(step_key(8, 1, Some(1)), step_key(8, 1, None));
        assert_ne!(step_key(8, 1, Some(1)), step_key(8, 2, Some(1)));
        assert_ne!(step_key(8, 1, Some(0)), step_key(8, 1, None));
    }

    #[test]
    fn a_prefill_tile_reads_the_weights_once_for_its_rows() {
        let d = dims(256);
        let g = geom(&d);
        let mid = prefill_tile(&d, CTX, 256, 1, None, false);
        let last = prefill_tile(&d, CTX, 256, 1, None, true);
        let one = verify(&dims(1), CTX, 1, 1);
        assert!(
            (mid.by_kind(W) + 2304.0 * 200064.0 * 2.0 - one.by_kind(W)).abs()
                < 0.02 * one.by_kind(W)
        );
        assert!(last.total() > mid.total());
        assert!(mid.total_flops() > 200.0 * one.total_flops());
        let with = prefill_tile(&d, CTX, 256, 1, Some(&g), false);
        assert!(with.total() > mid.total());
        assert_eq!(prefill_tile_launches(&d, false, false), 49);
        assert_eq!(prefill_tile_launches(&d, true, true), 52);
        // today's plan, pinned here rather than sealed (a later fusion may change it): the last
        // tile and a drafter add launches, never remove them
        for drafter in [false, true] {
            assert!(
                prefill_tile_launches(&d, drafter, true)
                    >= prefill_tile_launches(&d, drafter, false)
            );
        }
        for last in [false, true] {
            assert!(
                prefill_tile_launches(&d, true, last) >= prefill_tile_launches(&d, false, last)
            );
        }
        assert_ne!(
            tile_key(256, 1, false, true),
            tile_key(256, 1, false, false)
        );
        assert_ne!(
            tile_key(256, 1, true, false),
            tile_key(256, 1, false, false)
        );
    }

    #[test]
    fn the_block_drafter_reads_as_a_chain_of_its_drafts() {
        let sc = SpecConfig {
            method: SpecMethod::Mtp {
                tree: None,
                reference: Vec::new(),
                draft_vocab: 0,
            },
            draft_k: vec![7],
        };
        let info = spec_info_of(&sc, 8, Some(SynDataType::Bf16));
        assert_eq!((info.method, info.draft_k.as_slice()), ("mtp", &[7u64][..]));
        assert!(info.samples && info.tree.is_none());
        assert_eq!(live_label_of(Some(&sc), 8, 0.0), "mtp K=7 BF16 greedy");
        assert_eq!(live_label_of(Some(&sc), 8, 0.7), "mtp K=7 BF16 T=0.70");
        assert_eq!(live_label_of(None, 8, 0.0), "K=0 BF16 greedy");
        assert_eq!(live_label_of(None, 1, 1.0), "K=0 BF16 T=1.00");
    }

    #[test]
    fn the_drafter_matrices_are_bf16() {
        let d = dims(8);
        assert_eq!(drafter_matrix(&geom(&d)), Some(SynDataType::Bf16));
    }
}
