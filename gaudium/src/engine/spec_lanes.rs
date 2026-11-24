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

/// Name of a batched speculative step recipe.
pub fn batched_spec_name(b: u64, t: u64, nch: u64, sampled: bool) -> String {
    format!(
        "g_spec_ds{}_B{b}_T{t}_C{nch}",
        if sampled { "s" } else { "" }
    )
}

/// `b` groups of `t` columns of `x {d, b * t}` as `b` tensors `{d, t}` (intermediates).
fn split_groups(
    gb: &mut GraphBuilder,
    x: TensorId,
    dt: SynDataType,
    d: u64,
    b: u64,
    t: u64,
    name: &str,
) -> Vec<TensorId> {
    let parts: Vec<TensorId> = (0..b)
        .map(|j| gb.add_intermediate(format!("{name}_g{j}"), dt, &[d, t]))
        .collect();
    gb.op_split(format!("{name}_split"), x, &parts, 1);
    parts
}

/// The batched speculative step of `b` slots of `t` rows over `nch` KV chunks with a DSpine
/// drafter (`ns` = slots of the DeltaNet state rings). `elf_hashes` must include the drafter's
/// kernels (and the sampler's for `sampled`). The checkpoint form of the DeltaNet verify rows.
#[allow(clippy::too_many_arguments)]
pub fn build_batched_spec_step_graph(
    spec: &ModelSpec,
    lmax: u64,
    nch: u64,
    t: u64,
    ns: u64,
    b: u64,
    dc: &dspine::DSpineCfg,
    elf_hashes: &[(String, String)],
    sampled: bool,
) -> GraphBuilder {
    let gdn_fast = elf_hashes.iter().any(|(k, _)| k == GDNV_ELF);
    let mut gb = GraphBuilder::new(batched_spec_name(b, t, nch, sampled));
    gb.set_fp8_weights(spec.fp8_weights);
    for (k, h) in elf_hashes {
        gb.record_kernel_elf_hash(k, h);
    }
    let hid = spec.hidden;
    let rows = b * t;
    let table =
        gb.get_or_add_persistent_tensor("embed_table", SynDataType::Bf16, &[hid, spec.vocab]);
    let rope_c = gb.get_or_add_persistent_tensor("rope_c", SynDataType::Float, &[128, lmax]);
    let rope_s = gb.get_or_add_persistent_tensor("rope_s", SynDataType::Float, &[128, lmax]);
    // per slot: its ids, cursor and ring slot (the lane's step buffers of this parity)
    let ids: Vec<TensorId> = (0..b)
        .map(|j| gb.add_persistent_tensor(format!("{}ids", slot_pfx(j)), SynDataType::Int32, &[t]))
        .collect();
    let cursors: Vec<TensorId> = (0..b)
        .map(|j| {
            gb.add_persistent_tensor(format!("{}cursor", slot_pfx(j)), SynDataType::Int32, &[1])
        })
        .collect();
    let a_in: Vec<TensorId> = (0..b)
        .map(|j| gb.add_persistent_tensor(format!("{}a_in", slot_pfx(j)), SynDataType::Int32, &[1]))
        .collect();
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
            let qgs = split_groups(&mut gb, qg, SynDataType::Bf16, spec.qg_dim(), b, t, &format!("{lpfx}qg"));
            let ks = split_groups(&mut gb, k, SynDataType::Bf16, spec.kv_dim(), b, t, &format!("{lpfx}k"));
            let vs = split_groups(&mut gb, v, SynDataType::Bf16, spec.kv_dim(), b, t, &format!("{lpfx}v"));
            let cores: Vec<TensorId> = (0..b as usize)
                .map(|j| {
                    let spfx = format!("{}{lpfx}", slot_pfx(j as u64));
                    append_attn_core(
                        &mut gb, spec, qgs[j], ks[j], vs[j], cursors[j], rope_c, rope_s, &lpfx,
                        &spfx, t, nch,
                    )
                })
                .collect();
            let core = gb.add_intermediate(format!("{lpfx}core"), SynDataType::Bf16, &[nh * dh, rows]);
            gb.op_concat(format!("{lpfx}core_cat"), &cores, core, 1);
            add_linear_bf16(&mut gb, core, &format!("{lpfx}W_o"), nh * dh, hid, rows, &format!("{lpfx}attn"))
        } else {
            let (hv, dl) = (spec.gdn_v_heads, spec.gdn_v_dim);
            let (qkvz, ba) = append_linear_proj(&mut gb, spec, x, h_pre, &lpfx, rows);
            let qs = split_groups(&mut gb, qkvz, SynDataType::Bf16, spec.gdn_qkvz_dim(), b, t, &format!("{lpfx}qkvz"));
            let bs = split_groups(&mut gb, ba, SynDataType::Bf16, spec.gdn_ba_dim(), b, t, &format!("{lpfx}ba"));
            let cores: Vec<TensorId> = (0..b as usize)
                .map(|j| {
                    let spfx = format!("{}{lpfx}", slot_pfx(j as u64));
                    append_gdn_ckpt_core(
                        &mut gb, spec, qs[j], bs[j], a_in[j], &lpfx, &spfx, t, ns, gdn_fast,
                    )
                })
                .collect();
            let core = gb.add_intermediate(format!("{lpfx}core"), SynDataType::Bf16, &[hv * dl, rows]);
            gb.op_concat(format!("{lpfx}core_cat"), &cores, core, 1);
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
        let cols = split_groups(&mut gb, logits, SynDataType::Bf16, spec.vocab, b, t, "logits");
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
                    t,
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
                gb.add_persistent_tensor(format!("{}tokens", slot_pfx(j)), SynDataType::Int32, &[1, t])
            })
            .collect();
        gb.op_split("tokens_split", all, &parts, 1);
        parts
    };

    let parent = gb.get_or_add_persistent_tensor("sp_parent", SynDataType::Int32, &[t]);
    let depth = gb.get_or_add_persistent_tensor("sp_depth", SynDataType::Int32, &[t]);
    let taps: Vec<Vec<TensorId>> = dc
        .taps
        .iter()
        .map(|&l| split_groups(&mut gb, layer_out[l], SynDataType::Bf16, hid, b, t, &format!("tap{l}")))
        .collect();
    for j in 0..b as usize {
        let sp = slot_pfx(j as u64);
        let a_out = gb.add_persistent_tensor(format!("{sp}a_out"), SynDataType::Int32, &[1]);
        let cursor_out = gb.add_persistent_tensor(format!("{sp}cursor_out"), SynDataType::Int32, &[1]);
        let emitted = gb.add_persistent_tensor(format!("{sp}emitted"), SynDataType::Int32, &[t]);
        let fix_out = gb.add_persistent_tensor(format!("{sp}fix_out"), SynDataType::Int32, &[2]);
        gb.add_node(
            "gaudium_accept_t",
            format!("{sp}accept"),
            &[ids[j], tokens[j], cursors[j], parent, depth],
            &[a_out, cursor_out, emitted, fix_out],
            &[],
        );
        let slot_taps: Vec<TensorId> = taps.iter().map(|g| g[j]).collect();
        dspine::append_step_drafter_p(
            &mut gb,
            &sp,
            spec,
            dc,
            &slot_taps,
            t,
            cursors[j],
            depth,
            tokens[j],
            a_out,
            cursor_out,
            lmax,
            sampled,
            (&format!("{sp}samp_f"), &format!("{sp}samp_i")),
        );
    }
    gb
}

/// What the batched speculative steps of a loaded engine are built from (cloneable, so recipes
/// can be described on several threads).
#[derive(Debug, Clone)]
pub struct SpecLaneGraph {
    pub lmax: u64,
    pub ns: u64,
    /// Verify rows per slot.
    pub t: u64,
    pub dc: dspine::DSpineCfg,
    pub hashes: Vec<(String, String)>,
    pub samp_hashes: Vec<(String, String)>,
}

impl SpecLaneGraph {
    pub fn build(&self, spec: &ModelSpec, b: u64, nch: u64, sampled: bool) -> GraphBuilder {
        let hashes = if sampled {
            &self.samp_hashes
        } else {
            &self.hashes
        };
        build_batched_spec_step_graph(
            spec, self.lmax, nch, self.t, self.ns, b, &self.dc, hashes, sampled,
        )
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
        || own.ends_with("_cs_ckpt");
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
        let g = build_batched_spec_step_graph(&spec, 16384, 1, 8, 16, 3, &dc, &[], false);
        assert_eq!(g.recipe_name(), "g_spec_ds_B3_T8_C1");
        let solo = build_spec_step_graph(&spec, 16384, 1, 8, 16, Some(&dc), &[]);
        let views = g.node_io_views();
        let solo_views = solo.node_io_views();
        let count = |v: &[crate::synapse::graph::NodeIoView<'_>], guid: &str| {
            v.iter().filter(|n| n.guid == guid).count()
        };
        // every per-row kernel of the solo step, three times; the trunk GEMMs once
        for guid in ["gaudium_accept_t", "gaudium_attn_prep", "gaudium_gdn_f32", "gaudium_ds_score", "gaudium_df_next"] {
            assert_eq!(count(&views, guid), 3 * count(&solo_views, guid), "{guid}");
        }
        assert_eq!(count(&views, "argmax_fwd_bf16"), 1);
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
        // the drafter's weights are the solo step's names
        for (name, ..) in &specs {
            if dspine::is_drafter_weight(name) && !name.ends_with("df_nofix") {
                assert!(solo.find_persistent_tensor(name).is_some(), "{name}");
            }
        }
    }
}
