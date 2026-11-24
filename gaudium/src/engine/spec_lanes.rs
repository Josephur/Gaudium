//! Batched speculative steps of the lanes (`--parallel N` with a DSpine drafter).
//!
//! This is a child module of `engine::qwen3::spec` (declared there with `#[path]`).
//!
//! One launch of `g_spec_ds{fam}_B{b}_T{t}_C{nch}` does, for `b` lanes in slots `0..b`, what
//! `b` single-stream steps of `t = K + 1` rows do:
//! - **verify**: the trunk (norms, projections, MLPs, readout GEMM) runs once over all `b * t`
//!   rows, so each weight is read once for every lane; per slot, on its own rows, the attention
//!   core over the slot's lane cache at the slot's cursor and the DeltaNet recurrence from
//!   checkpoint slot `a_in` of the lane's state ring, one checkpoint per row into the other ring;
//! - **accept**: per slot, the longest draft prefix that equals the verify tokens;
//! - **draft**: per slot, the drafter's context update for the slot's verified rows (into the
//!   lane's own context caches) and its next block.
//!
//! Every slot runs exactly the nodes of the single-stream step on its own rows (same shapes, same
//! kernels): its tokens are the verify argmax (or sampled) chain of its own request, so each
//! stream equals its single-stream `K = 0` stream whatever the drafts are.

use super::*;
use crate::engine::qwen3::lanes_rt::slot_pfx;

/// Name of a batched speculative step recipe of slots of `ts` rows (widest first; `r`: the
/// replay form of the DeltaNet verify rows): `..._B3_T8_C1` when every slot has the same rows,
/// `..._B3_T16x1_8x2_C1` otherwise.
pub fn batched_spec_name(ts: &[u64], nch: u64, sampled: bool, replay: bool) -> String {
    let b = ts.len();
    let mut runs: Vec<(u64, usize)> = Vec::new();
    for &t in ts {
        match runs.last_mut() {
            Some((r, n)) if *r == t => *n += 1,
            _ => runs.push((t, 1)),
        }
    }
    let rows = if runs.len() == 1 {
        runs[0].0.to_string()
    } else {
        runs.iter()
            .map(|(t, n)| format!("{t}x{n}"))
            .collect::<Vec<_>>()
            .join("_")
    };
    format!(
        "g_spec_ds{}{}_B{b}_T{rows}_C{nch}",
        if sampled { "s" } else { "" },
        if replay { "r" } else { "" }
    )
}

/// The slots' column groups of `x {d, sum ts}` (`ts[j]` columns of slot `j`) as tensors
/// `{d, ts[j]}` (intermediates).
fn split_groups(
    gb: &mut GraphBuilder,
    x: TensorId,
    dt: SynDataType,
    d: u64,
    ts: &[u64],
    name: &str,
) -> Vec<TensorId> {
    let parts: Vec<TensorId> = ts
        .iter()
        .enumerate()
        .map(|(j, &t)| gb.add_intermediate(format!("{name}_g{j}"), dt, &[d, t]))
        .collect();
    gb.op_split(format!("{name}_split"), x, &parts, 1);
    parts
}

/// The batched speculative step over `nch` KV chunks of one slot per entry of `ts` (`ts[j]` =
/// the verify rows of slot `j`, widest first) with a DSpine drafter (`ns` = rows of the DeltaNet
/// state rings and update buffers, at least every slot's). `elf_hashes` must include the
/// drafter's kernels (and the sampler's for `sampled`); they also select the form of the
/// DeltaNet verify rows (checkpoint or replay).
#[allow(clippy::too_many_arguments)]
pub fn build_batched_spec_step_graph(
    spec: &ModelSpec,
    lmax: u64,
    nch: u64,
    ts: &[u64],
    ns: u64,
    dc: &dspine::DSpineCfg,
    elf_hashes: &[(String, String)],
    sampled: bool,
) -> GraphBuilder {
    let gdn_fast = elf_hashes.iter().any(|(k, _)| k == GDNV_ELF);
    // the replay form of the DeltaNet verify rows (no per-row state checkpoints) when the step
    // family carries its kernel, as in the single-stream steps
    let gdn_replay = replay::enabled(elf_hashes);
    let mut gb = GraphBuilder::new(batched_spec_name(ts, nch, sampled, gdn_replay));
    gb.set_fp8_weights(spec.fp8_weights);
    for (k, h) in elf_hashes {
        gb.record_kernel_elf_hash(k, h);
    }
    let b = ts.len() as u64;
    let t_max = ts.iter().copied().max().unwrap_or(1);
    let hid = spec.hidden;
    let rows: u64 = ts.iter().sum();
    let table =
        gb.get_or_add_persistent_tensor("embed_table", SynDataType::Bf16, &[hid, spec.vocab]);
    let rope_c = gb.get_or_add_persistent_tensor("rope_c", SynDataType::Float, &[128, lmax]);
    let rope_s = gb.get_or_add_persistent_tensor("rope_s", SynDataType::Float, &[128, lmax]);
    // per slot: its ids, cursor and ring slot (the lane's step buffers of this parity)
    let ids: Vec<TensorId> = (0..b)
        .map(|j| gb.add_persistent_tensor(format!("{}ids", slot_pfx(j)), SynDataType::Int32, &[ts[j as usize]]))
        .collect();
    let cursors: Vec<TensorId> = (0..b)
        .map(|j| {
            gb.add_persistent_tensor(format!("{}cursor", slot_pfx(j)), SynDataType::Int32, &[1])
        })
        .collect();
    let a_in: Vec<TensorId> = (0..b)
        .map(|j| gb.add_persistent_tensor(format!("{}a_in", slot_pfx(j)), SynDataType::Int32, &[1]))
        .collect();
    // replay form: per slot, its lane's first-step cursor (the first step replays nothing)
    let rp_first: Vec<TensorId> = if gdn_replay {
        (0..b)
            .map(|j| gb.add_persistent_tensor(format!("{}rp_first", slot_pfx(j)), SynDataType::Int32, &[1]))
            .collect()
    } else {
        Vec::new()
    };
    let all_ids = gb.add_intermediate("ids_all", SynDataType::Int32, &[rows]);
    gb.op_concat("ids_cat", &ids, all_ids, 0);
    let mut x = gb.add_persistent_tensor("x0", SynDataType::Bf16, &[hid, rows]);
    gb.op_gather_bf16("embed", table, all_ids, x, 1);

    let mut h_pre: Option<TensorId> = None;
    let mut layer_out: Vec<TensorId> = Vec::with_capacity(spec.n_layers as usize);
    let n_layers = spec.n_layers as usize;
    for l in 0..n_layers {
        let lpfx = format!("L{l}_");
        let (next_w, next_h) = if l + 1 < n_layers {
            (format!("L{}_w_in", l + 1), format!("L{}_h", l + 1))
        } else {
            ("w_final".to_string(), "h_final".to_string())
        };
        let attn = if spec.is_full_attention_layer(l) {
            let (nh, dh) = (spec.n_heads, spec.head_dim);
            let (qg, k, v) = append_attn_proj(&mut gb, spec, x, h_pre, &lpfx, rows);
            let qgs = split_groups(&mut gb, qg, SynDataType::Bf16, spec.qg_dim(), ts, &format!("{lpfx}qg"));
            let ks = split_groups(&mut gb, k, SynDataType::Bf16, spec.kv_dim(), ts, &format!("{lpfx}k"));
            let vs = split_groups(&mut gb, v, SynDataType::Bf16, spec.kv_dim(), ts, &format!("{lpfx}v"));
            // every slot on its own rows, cursor and caches, stage by stage across the slots
            let slots: Vec<AttnSlot> = (0..b as usize)
                .map(|j| AttnSlot {
                    qg: qgs[j],
                    k: ks[j],
                    v: vs[j],
                    cursor: cursors[j],
                    spfx: format!("{}{lpfx}", slot_pfx(j as u64)),
                    t: ts[j],
                })
                .collect();
            let cores = append_attn_core_slots(&mut gb, spec, &slots, rope_c, rope_s, &lpfx, nch);
            let core = gb.add_intermediate(format!("{lpfx}core"), SynDataType::Bf16, &[nh * dh, rows]);
            gb.op_concat(format!("{lpfx}core_cat"), &cores, core, 1);
            add_linear_bf16(&mut gb, core, &format!("{lpfx}W_o"), nh * dh, hid, rows, &format!("{lpfx}attn"))
        } else {
            let (hv, dl) = (spec.gdn_v_heads, spec.gdn_v_dim);
            let (qkvz, ba) = append_linear_proj(&mut gb, spec, x, h_pre, &lpfx, rows);
            let spfx: Vec<String> = (0..b).map(|j| format!("{}{lpfx}", slot_pfx(j))).collect();
            let widths = ts.to_vec();
            let core = if gdn_replay {
                replay::append_gdn_replay_slots(
                    &mut gb, spec, qkvz, ba, &a_in, &cursors, &rp_first, &lpfx, &spfx, &widths, ns,
                )
            } else {
                append_gdn_ckpt_slots(
                    &mut gb, spec, qkvz, ba, &a_in, &lpfx, &spfx, &widths, ns, gdn_fast,
                )
            };
            add_linear_bf16(&mut gb, core, &format!("{lpfx}W_o"), hv * dl, hid, rows, &format!("{lpfx}attn"))
        };
        let (out_x, next) = append_post_and_mlp(
            &mut gb,
            spec,
            x,
            attn,
            &lpfx,
            rows,
            true,
            Some((next_w.as_str(), next_h.as_str())),
        );
        x = out_x;
        h_pre = next;
        layer_out.push(out_x);
    }

    let h_final = h_pre.expect("fused final norm produced by last layer");
    let lm_w = if spec.tie_word_embeddings {
        "embed_table"
    } else {
        "W_lm"
    };
    let logits = add_linear_bf16(&mut gb, h_final, lm_w, hid, spec.vocab, rows, "logits");
    // per slot: its verify tokens
    let tokens: Vec<TensorId> = if sampled {
        let cols = split_groups(&mut gb, logits, SynDataType::Bf16, spec.vocab, ts, "logits");
        (0..b as usize)
            .map(|j| {
                let sp = slot_pfx(j as u64);
                let at = crate::engine::sampler::SamplerAt {
                    base: cursors[j],
                    off: 1,
                    role: 0,
                };
                crate::engine::sampler::append_sampler_as(
                    &mut gb,
                    cols[j],
                    spec.vocab,
                    ts[j],
                    &at,
                    &format!("{sp}tokens"),
                    &format!("{sp}samp_f"),
                    &format!("{sp}samp_i"),
                    None,
                )
            })
            .collect()
    } else {
        let all = gb.add_intermediate("tokens_all", SynDataType::Int32, &[1, rows]);
        gb.op_argmax_bf16("argmax", logits, all, 0);
        let parts: Vec<TensorId> = (0..b)
            .map(|j| {
                gb.add_persistent_tensor(format!("{}tokens", slot_pfx(j)), SynDataType::Int32, &[1, ts[j as usize]])
            })
            .collect();
        gb.op_split("tokens_split", all, &parts, 1);
        parts
    };

    // the chain tables, at the widest slot's rows (a narrower slot's kernels read their first rows)
    let parent = gb.get_or_add_persistent_tensor("sp_parent", SynDataType::Int32, &[t_max]);
    let depth = gb.get_or_add_persistent_tensor("sp_depth", SynDataType::Int32, &[t_max]);
    let mut draft_slots = Vec::with_capacity(b as usize);
    for j in 0..b as usize {
        let sp = slot_pfx(j as u64);
        let a_out = gb.add_persistent_tensor(format!("{sp}a_out"), SynDataType::Int32, &[1]);
        let cursor_out = gb.add_persistent_tensor(format!("{sp}cursor_out"), SynDataType::Int32, &[1]);
        let emitted = gb.add_persistent_tensor(format!("{sp}emitted"), SynDataType::Int32, &[ts[j]]);
        let fix_out = gb.add_persistent_tensor(format!("{sp}fix_out"), SynDataType::Int32, &[2]);
        gb.add_node(
            "gaudium_accept_t",
            format!("{sp}accept"),
            &[ids[j], tokens[j], cursors[j], parent, depth],
            &[a_out, cursor_out, emitted, fix_out],
            &[],
        );
        draft_slots.push(dspine::StepSlot {
            samp: (format!("{sp}samp_f"), format!("{sp}samp_i")),
            pfx: sp,
            t: ts[j],
            cursor: cursors[j],
            tokens: tokens[j],
            a_out,
            cursor_out,
        });
    }
    // the drafter over every slot at once: its context pass over all verify rows and the next
    // blocks side by side (each drafter weight read once per step); per slot only its context
    // caches, its block attention and its walk
    let taps: Vec<TensorId> = dc.taps.iter().map(|&l| layer_out[l]).collect();
    dspine::append_step_drafter_slots(&mut gb, "", spec, dc, &taps, &draft_slots, depth, lmax, sampled);
    gb
}

/// What the batched speculative steps of a loaded engine are built from (cloneable, so recipes
/// can be described on several threads).
#[derive(Debug, Clone)]
pub struct SpecLaneGraph {
    pub lmax: u64,
    /// Rows of the DeltaNet state rings and update buffers: at least `wide`.
    pub ns: u64,
    /// Verify rows of a lane at its narrow width (every lane may step at it).
    pub narrow: u64,
    /// Verify rows of a lane at its wide width (`narrow` when lanes have one width).
    pub wide: u64,
    pub dc: dspine::DSpineCfg,
    pub hashes: Vec<(String, String)>,
    pub samp_hashes: Vec<(String, String)>,
}

impl SpecLaneGraph {
    /// The step of slots of `ts` rows (widest first).
    pub fn build(&self, spec: &ModelSpec, ts: &[u64], nch: u64, sampled: bool) -> GraphBuilder {
        let hashes = if sampled {
            &self.samp_hashes
        } else {
            &self.hashes
        };
        build_batched_spec_step_graph(spec, self.lmax, nch, ts, self.ns, &self.dc, hashes, sampled)
    }

    /// The slot rows of every step of `b` slots: `k` wide slots, then `b - k` narrow ones.
    pub fn slot_rows(&self, b: u64) -> Vec<Vec<u64>> {
        if self.wide == self.narrow {
            return vec![vec![self.narrow; b as usize]];
        }
        (0..=b)
            .map(|k| {
                let mut ts = vec![self.wide; k as usize];
                ts.resize(b as usize, self.narrow);
                ts
            })
            .collect()
    }
}

/// A tensor of a batched speculative step bound to the lane in one of its slots: `(slot, the
/// lane's own name of the buffer)`. Beside the K=0 step's (caches, states, cursor, sampler
/// parameters): the step buffers that alternate by parity, the drafter's context caches and
/// its context bound.
pub fn spec_slot_bound(name: &str) -> Option<(usize, &str)> {
    let rest = name.strip_prefix('s')?;
    let sep = rest.find('_')?;
    let j: usize = rest[..sep].parse().ok()?;
    let own = &rest[sep + 1..];
    let lane_owned = matches!(
        own,
        "cursor" | "samp_f" | "samp_i" | "ids" | "next_ids" | "cursor_out" | "a_in" | "a_out" | "emitted" | "df_nofix"
    ) || is_kv_tensor(own)
        || own.ends_with("_S_in")
        || own.ends_with("_S_ckpt")
        || own.ends_with("_cs_in")
        || own.ends_with("_cs_ckpt")
        || own == "rp_first"
        || replay::is_parity_name(own);
    lane_owned.then_some((j, own))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target_27b() -> ModelSpec {
        ModelSpec::from_config_json(
            r#"{"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
              "text_config": {"model_type": "qwen3_5_text", "hidden_size": 5120,
                "intermediate_size": 17408, "num_hidden_layers": 64, "num_attention_heads": 24,
                "num_key_value_heads": 4, "head_dim": 256, "vocab_size": 248320,
                "rms_norm_eps": 1e-6, "full_attention_interval": 4, "linear_num_key_heads": 16,
                "linear_num_value_heads": 48, "linear_key_head_dim": 128,
                "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
                "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.25}}}"#,
        )
        .unwrap()
    }

    #[test]
    fn batched_spec_step_gives_each_slot_its_verify_accept_and_drafter() {
        let spec = target_27b();
        let dc = dspine::DSpineCfg::from_json(dspine::tests::CFG_27B).unwrap();
        let g = build_batched_spec_step_graph(&spec, 16384, 1, &[8, 8, 8], 16, &dc, &[], false);
        assert_eq!(g.recipe_name(), "g_spec_ds_B3_T8_C1");
        let solo = build_spec_step_graph(&spec, 16384, 1, 8, 16, Some(&dc), &[]);
        let views = g.node_io_views();
        let solo_views = solo.node_io_views();
        let count = |v: &[crate::synapse::graph::NodeIoView<'_>], guid: &str| {
            v.iter().filter(|n| n.guid == guid).count()
        };
        // every per-row kernel of the solo step, three times; the trunk GEMMs once
        for guid in [
            "gaudium_accept_t", "gaudium_attn_prep", "gaudium_gdn_f32", "gaudium_ds_score",
            "gaudium_df_next", "gaudium_df_attn_prep", "gaudium_ds_codes", "gaudium_ds_msg",
            "gaudium_ds_finish", "gaudium_ds_base", "gaudium_df_walk", "gaudium_df_ids",
        ] {
            assert_eq!(count(&views, guid), 3 * count(&solo_views, guid), "{guid}");
        }
        assert_eq!(count(&views, "argmax_fwd_bf16"), 1);
        // the DeltaNet gate and output norm compute each row on its own: once for all slots
        for guid in ["gaudium_gdn_gate", "gaudium_gdn_post"] {
            assert_eq!(count(&views, guid), count(&solo_views, guid), "{guid}");
        }
        assert_eq!(count(&views, "gaudium_gdn_prep"), 3 * count(&solo_views, "gaudium_gdn_prep"));
        // the drafter's weights are read as often as in the solo step: once per launch for all
        // three slots (its context pass, its block projections, MLPs, bridge and readout)
        let readers = |v: &[crate::synapse::graph::NodeIoView<'_>], w: &str| {
            v.iter().filter(|n| n.inputs.iter().any(|t| t.name == w)).count()
        };
        let drafter_weights: Vec<String> = solo
            .persistent_tensor_specs()
            .into_iter()
            .map(|s| s.0)
            .filter(|n| dspine::is_drafter_weight(n) && (n.contains("_W_") || n.starts_with("ds_W") || n.starts_with("ds_R") || n == "ds_PW"))
            .collect();
        assert!(drafter_weights.len() >= 5 * 7, "{drafter_weights:?}");
        for w in &drafter_weights {
            assert_eq!(readers(&views, w), readers(&solo_views, w), "{w}");
        }
        // and so is the draft readout
        assert_eq!(count(&views, "gaudium_df_topk1"), count(&solo_views, "gaudium_df_topk1"));
        let specs = g.persistent_tensor_specs();
        let dims = |n: &str| specs.iter().find(|s| s.0 == n).map(|s| s.2.clone());
        assert_eq!(dims("L3_qg"), Some(vec![spec.qg_dim(), 24]));
        for j in 0..3 {
            assert_eq!(dims(&format!("s{j}_ids")), Some(vec![8]));
            assert_eq!(dims(&format!("s{j}_next_ids")), Some(vec![dc.block]));
            assert_eq!(dims(&format!("s{j}_L0_S_in")), Some(vec![64, 256, 48, 16]));
            assert!(dims(&format!("s{j}_D0_kcache")).is_some());
            assert_eq!(spec_slot_bound(&format!("s{j}_D0_kcache")), Some((j, "D0_kcache")));
            assert_eq!(spec_slot_bound(&format!("s{j}_emitted")), Some((j, "emitted")));
        }
        assert_eq!(spec_slot_bound("s1_blk_scores"), None);
        // the output norm in calls of at most 32 rows: 8 slots of 8 rows are two calls a layer
        let g8 = build_batched_spec_step_graph(&spec, 16384, 1, &[8; 8], 16, &dc, &[], false);
        assert_eq!(count(&g8.node_io_views(), "gaudium_gdn_post"), 2 * 48);
        for n in g8.node_io_views().iter().filter(|n| n.guid == "gaudium_gdn_post") {
            assert_eq!(n.inputs[0].dims[2], 32, "{}", n.name);
        }
        // the replay form: per slot its lane's single states, update buffers and first cursor
        let rp: Vec<(String, String)> = vec![
            ("GAUDIUM_ACCEPTT_ELF".into(), "00".into()),
            (super::replay::GDNR_ELF.into(), "11".into()),
        ];
        let r = build_batched_spec_step_graph(&spec, 16384, 1, &[8, 8, 8], 16, &dc, &rp, false);
        assert_eq!(r.recipe_name(), "g_spec_dsr_B3_T8_C1");
        let rspecs = r.persistent_tensor_specs();
        let rdims = |n: &str| rspecs.iter().find(|s| s.0 == n).map(|s| s.2.clone());
        for j in 0..3 {
            assert_eq!(rdims(&format!("s{j}_L0_S_in")), Some(vec![64, 256, 48]));
            assert_eq!(rdims(&format!("s{j}_L0_S_base")), Some(vec![64, 256, 48]));
            assert_eq!(rdims(&format!("s{j}_L0_rpk_in")), Some(vec![128, 48, 16]));
            assert_eq!(rdims(&format!("s{j}_L0_rpd_out")), Some(vec![64, 8, 48, 8]));
            assert_eq!(rdims(&format!("s{j}_rp_first")), Some(vec![1]));
            for n in ["L0_S_base", "L0_rpk_in", "L0_rpk_out", "L0_rpd_in", "L0_rpd_out", "L0_rpg_in", "L0_rpg_out", "rp_first"] {
                assert_eq!(spec_slot_bound(&format!("s{j}_{n}")), Some((j, n)), "{n}");
            }
        }
        assert!(!rspecs.iter().any(|s| s.0.ends_with("_S_ckpt")));
        let rviews = r.node_io_views();
        assert_eq!(count(&rviews, "gaudium_gdn_f32r"), 3 * 48);
        assert_eq!(count(&rviews, "gaudium_gdn_post"), 48);
        // slots of their own widths: the trunk over every slot's rows, each slot on its own
        let m = build_batched_spec_step_graph(&spec, 16384, 1, &[16, 8, 8], 16, &dc, &rp, false);
        assert_eq!(m.recipe_name(), "g_spec_dsr_B3_T16x1_8x2_C1");
        let mspecs = m.persistent_tensor_specs();
        let mdims = |n: &str| mspecs.iter().find(|s| s.0 == n).map(|s| s.2.clone());
        assert_eq!(mdims("L3_qg"), Some(vec![spec.qg_dim(), 32]));
        assert_eq!(mdims("s0_ids"), Some(vec![16]));
        assert_eq!(mdims("s1_ids"), Some(vec![8]));
        assert_eq!(mdims("s0_emitted"), Some(vec![16]));
        assert_eq!(mdims("s2_emitted"), Some(vec![8]));
        assert_eq!(mdims("s0_L0_rpd_out"), Some(vec![64, 8, 48, 16]));
        assert_eq!(mdims("s1_L0_rpd_out"), Some(vec![64, 8, 48, 8]));
        assert_eq!(mdims("sp_depth"), Some(vec![16]));
        // the output norm: 16 + 8 + 8 rows in one call
        assert_eq!(count(&m.node_io_views(), "gaudium_gdn_post"), 48);
        assert_eq!(batched_spec_name(&[16, 16, 8], 2, true, false), "g_spec_dss_B3_T16x2_8x1_C2");
        let lg = SpecLaneGraph {
            lmax: 16384,
            ns: 16,
            narrow: 8,
            wide: 16,
            dc: dc.clone(),
            hashes: Vec::new(),
            samp_hashes: Vec::new(),
        };
        assert_eq!(lg.slot_rows(2), vec![vec![8, 8], vec![16, 8], vec![16, 16]]);
        let one = SpecLaneGraph { wide: 8, ns: 8, ..lg.clone() };
        assert_eq!(one.slot_rows(3), vec![vec![8, 8, 8]]);
        // the drafter's weights are the solo step's names
        for (name, ..) in &specs {
            if dspine::is_drafter_weight(name) && !name.ends_with("df_nofix") {
                assert!(solo.find_persistent_tensor(name).is_some(), "{name}");
            }
        }
    }
}
