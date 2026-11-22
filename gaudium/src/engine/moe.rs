//! Mixture-of-experts MLP block of the Qwen3.5 MoE family on Gaudi 2, in its dense-masked form.
//!
//! For the post-attention-normed rows `h`:
//!
//! ```text
//! router   top-K of W_r h, weights renormalised over the K
//! routed   r = sum over the row's experts e of w_e * W_down_e (silu(W_gate_e h) * (W_up_e h))
//! shared   s = W_down_s (silu(W_gate_s h) * (W_up_s h))
//! out      r + sigmoid(w_sg . h) * s
//! ```
//!
//! Dense-masked execution: a layer's `NE` routed experts and its shared expert are stored as ONE
//! wide MLP of `(NE + 1) * EF` channels (block `e < NE` = routed expert `e`, block `NE` = the
//! shared expert). The gate and up GEMMs compute every block for every row on the MME;
//! `gaudium_moe_route` picks each row's experts and writes one weight table;
//! `gaudium_moe_act` writes `silu(g) * u * w` for the selected blocks (the shared block's weight
//! is `sigmoid` of the shared gate) and exact zeros elsewhere; the down GEMM is then the
//! reference's weighted sum over the experts, accumulated once by the MME. No host is in the loop
//! and the cost does not depend on which experts the rows select, so the same nodes serve one
//! decode row and a whole prefill tile.
//!
//! The wide matrices go through the common linear builder, so they are FP8 GEMMs when the graph
//! runs FP8 weights (`engine/fp8_graph.rs`); the router and the shared gate stay BF16.
//!
//! Routed execution (`ModelSpec::moe_routed`): the device MLP of a layer is the shared expert
//! alone; the routed experts are stored as two stacked BF16 tensors per layer and run through the
//! vendor expert-select node, which reads only the experts the device-written routing table
//! names. Three rules of that form live here:
//! - **Row groups.** A pass of a decode step with more than [`MOE_GROUP_ROWS`] rows runs one
//!   node per group of rows (views of the rows and of the routing tables in, a concat out): the
//!   node's result for a row depends on the node's row count above that size (measured on Gaudi 2
//!   with the legacy engine: a 9-row node left the `K = 0` stream, 8 + 1 rows did not), and the
//!   speculative verify must reproduce the one-row step bit for bit.
//! - **Shared expert views.** Every node of a graph that reads the same stacked tensor shares one
//!   set of per-expert views (the verify rows, the MTP catch-up and each MTP draft pass read
//!   the same experts).
//! - **Prefill.** The per-layer prefill recipes stop after the gated shared expert; the routed
//!   experts and the layer tail of a tile run in one generic recipe
//!   ([`build_prefill_routed_graph`]) that does not depend on the layer kind or on the KV chunk
//!   count, at a small set of padded row counts ([`routed_prefill_rows`]).

use std::collections::BTreeMap;

use crate::engine::fp8_graph::is_fp8_aux_suffix;
use crate::engine::qwen3::{add_fused_add_norm, add_linear_bf16};
use crate::model_spec::{ModelFamily, ModelSpec};
use crate::synapse::ffi::SynDataType;
use crate::synapse::graph::{GraphBuilder, TensorId};
use crate::synapse::ops::u32_params;

/// Kernel ELFs of the Qwen3.5 MoE shape (`DH=256, NKV=2, GQ=8, HK=16, HV=32`) plus the router and
/// the routed activation.
pub(crate) const QWEN35_MOE_ELFS: &[&str] = &[
    "GAUDIUM_APREP_G2Q8_ELF",
    "GAUDIUM_APOSTF_G2Q8_ELF",
    "GAUDIUM_ACOMB_ELF",
    "GAUDIUM_ASOFTC_ELF",
    "GAUDIUM_ANORM_ELF",
    "GAUDIUM_PREPS_K16R2_ELF",
    "GAUDIUM_PREPSD_K16R2_ELF",
    "GAUDIUM_GATE_ELF",
    "GAUDIUM_GDNS_ELF",
    "GAUDIUM_POST_K16R2_ELF",
    "GAUDIUM_POSTD_K16R2_ELF",
    "GAUDIUM_MOEROUTESG_ELF",
    "GAUDIUM_MOEACT_ELF",
];

/// The MoE shape the built kernels cover: the hybrid attention shape above, at most 512 experts
/// (the router kernel reads a row's logits as up to 8 vectors of 64), at most 12 experts per row,
/// and an expert width that is a multiple of the 128-channel activation block.
pub(crate) fn supported_shape(spec: &ModelSpec) -> bool {
    spec.family == ModelFamily::Qwen35Moe
        && spec.is_moe()
        && spec.head_dim == 256
        && spec.n_kv_heads == 2
        && spec.n_heads == 16
        && spec.gdn_k_heads == 16
        && spec.gdn_v_heads == 32
        && spec.n_experts.is_multiple_of(128)
        && spec.n_experts <= 512
        && (1..=12).contains(&spec.experts_per_tok)
        && spec.moe_intermediate > 0
        && spec.moe_intermediate.is_multiple_of(128)
}

/// Width of a layer's wide MLP: every routed expert block plus the shared expert's block.
pub fn wide_inter(spec: &ModelSpec) -> u64 {
    (spec.n_experts + 1) * spec.moe_intermediate
}

/// Width of the MLP a layer's dense GEMMs run on the device: the wide MLP (dense-masked form) or
/// the shared expert's block alone (routed form).
pub fn device_inter(spec: &ModelSpec) -> u64 {
    if spec.moe_routed {
        spec.moe_intermediate
    } else {
        wide_inter(spec)
    }
}

/// Name suffix of the activation that feeds the down GEMM: `m` (the masked wide product) or `ms`
/// (routed form: the gated shared expert's product). Distinct names keep the two FP8 cast sites
/// apart in one calibration table.
pub fn down_site(spec: &ModelSpec) -> &'static str {
    if spec.moe_routed {
        "ms"
    } else {
        "m"
    }
}

/// Layer weight suffixes (the part after `L<l>_`) that this module and the FP8 builder add to a
/// layer: the router, the shared-expert gate, the routed form's stacked expert tensors,
/// per-channel weight scales and site cast scales.
pub(crate) fn is_ext_layer_weight(suffix: &str) -> bool {
    matches!(suffix, "W_router" | "W_sgate" | "EW12" | "EW3") || is_fp8_aux_suffix(suffix)
}

/// `ns_MoeKernel::ParamsV4` of the vendor `moe` node (SynapseAI 1.24 `perf_lib_layer_params.h`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct MoeParamsV4 {
    activation: u32,
    experts_min: u32,
    experts_max: u32,
    flags: u32,
    block_size: u32,
    total_experts: u32,
    chunk_size: u32,
}
const _: () = assert!(std::mem::size_of::<MoeParamsV4>() == 28);
const MOE_ACTIVATION_MODE_SILU: u32 = 3;
const MOE_FLAGS_FUSED_GEMM: u32 = 1 << 0;
const MOE_FLAGS_PERMUTED_WEIGHTS: u32 = 1 << 1;

/// Most rows one expert-select node carries in a decode-step pass (see the module docs).
pub const MOE_GROUP_ROWS: u64 = 8;

/// The `ne` experts of one stacked persistent tensor `{d0, d1, ne}` as per-expert views
/// `{d0, d1}`: a split along the expert axis and a reshape each (workspace intermediates the
/// graph compiler resolves to views of the stacked tensor: a launch binds one tensor per layer
/// and kind, not one per expert). The views are created once per graph and stacked tensor.
fn stacked_views(
    gb: &mut GraphBuilder,
    lpfx: &str,
    key: &str,
    d0: u64,
    d1: u64,
    ne: u64,
) -> Vec<TensorId> {
    if let Some(v0) = gb.find_tensor(&format!("{lpfx}{key}_v0")) {
        let mut views = vec![v0];
        for e in 1..ne {
            views.push(
                gb.find_tensor(&format!("{lpfx}{key}_v{e}"))
                    .expect("expert views are created together"),
            );
        }
        return views;
    }
    let st =
        gb.get_or_add_persistent_tensor(format!("{lpfx}{key}"), SynDataType::Bf16, &[d0, d1, ne]);
    let parts: Vec<TensorId> = (0..ne)
        .map(|e| gb.add_intermediate(format!("{lpfx}{key}_p{e}"), SynDataType::Bf16, &[d0, d1, 1]))
        .collect();
    gb.op_split(format!("{lpfx}{key}_split"), st, &parts, 2);
    parts
        .iter()
        .enumerate()
        .map(|(e, &p)| {
            let v = gb.add_intermediate(format!("{lpfx}{key}_v{e}"), SynDataType::Bf16, &[d0, d1]);
            gb.op_reshape(format!("{lpfx}{key}_rs{e}"), p, v);
            v
        })
        .collect()
}

/// Appends the routed experts of `t` rows: `r {H, t} = sum_k w_k expert_k(h2)` per row, from the
/// stacked BF16 expert tensors `<wpfx>EW12` / `<wpfx>EW3` (EW12 = per expert the gate rows then
/// the up rows, `[NE][2 EF][H]`; EW3 = `[NE][H][EF]`) and the routing tables `rid` (ids) and
/// `rwb` (weights). `group_rows` > 0 runs one node per group of at most that many rows.
/// Node and intermediate names start with `apfx`; `r` is the caller's output tensor.
#[allow(clippy::too_many_arguments)]
fn append_routed_experts(
    gb: &mut GraphBuilder,
    spec: &ModelSpec,
    h2: TensorId,
    rid: TensorId,
    rwb: TensorId,
    r: TensorId,
    wpfx: &str,
    apfx: &str,
    t: u64,
    group_rows: u64,
) {
    let hid = spec.hidden;
    let ne = spec.n_experts;
    let ek = spec.experts_per_tok;
    let ef = spec.moe_intermediate;
    let mut experts = stacked_views(gb, wpfx, "EW12", hid, 2 * ef, ne);
    experts.extend(stacked_views(gb, wpfx, "EW3", ef, hid, ne));
    let mp = MoeParamsV4 {
        activation: MOE_ACTIVATION_MODE_SILU,
        experts_min: 0,
        experts_max: ne as u32 - 1,
        flags: MOE_FLAGS_FUSED_GEMM | MOE_FLAGS_PERMUTED_WEIGHTS,
        ..MoeParamsV4::default()
    };
    let params = crate::synapse::ops::as_param_bytes(&mp);
    if group_rows == 0 || t <= group_rows {
        let mut inputs = vec![h2, rid, rwb];
        inputs.extend_from_slice(&experts);
        gb.add_node("moe_bf16", format!("{apfx}moe"), &inputs, &[r], params);
        return;
    }
    let ng = t.div_ceil(group_rows);
    let rows_of = |g: u64| group_rows.min(t - g * group_rows);
    let part = |gb: &mut GraphBuilder, what: &str, dt: SynDataType, d0: u64| -> Vec<TensorId> {
        (0..ng)
            .map(|g| gb.add_intermediate(format!("{apfx}mg{g}_{what}"), dt, &[d0, rows_of(g)]))
            .collect()
    };
    let hp = part(gb, "h", SynDataType::Bf16, hid);
    let ip = part(gb, "i", SynDataType::Int32, ek);
    let wp = part(gb, "w", SynDataType::Bf16, ek);
    let rp = part(gb, "r", SynDataType::Bf16, hid);
    gb.op_split(format!("{apfx}mg_hs"), h2, &hp, 1);
    gb.op_split(format!("{apfx}mg_is"), rid, &ip, 1);
    gb.op_split(format!("{apfx}mg_ws"), rwb, &wp, 1);
    for g in 0..ng as usize {
        let mut inputs = vec![hp[g], ip[g], wp[g]];
        inputs.extend_from_slice(&experts);
        gb.add_node(
            "moe_bf16",
            format!("{apfx}mg{g}_moe"),
            &inputs,
            &[rp[g]],
            params,
        );
    }
    gb.op_concat(format!("{apfx}mg_cat"), &rp, r, 1);
}

/// Row counts the routed-experts prefill recipe is built for: a tile of `tile_len` rows runs at
/// the smallest listed count that holds it (the rows past the tile are computed and never read),
/// so a new prompt-tail length never compiles a new expert-select recipe.
pub fn routed_prefill_rows(tile_len: u64, prefill_tile: u64) -> u64 {
    let mut t = MOE_GROUP_ROWS;
    while t < tile_len {
        t *= 2;
    }
    t.min(prefill_tile.max(tile_len))
}

/// Generic names (`L_<suffix>`) of the per-layer prefill tensors the routed-experts recipe reads
/// from the layer recipe that ran before it, with the names it binds them under.
pub(crate) const ROUTED_PREFILL_INPUTS: [(&str, &str); 5] = [
    ("rx_h2", "L_h2"),
    ("rx_rid", "L_rid"),
    ("rx_rwb", "L_rwb"),
    ("rx_x2", "L_x2"),
    ("rx_mo", "L_mo"),
];

/// Builds the routed-experts prefill recipe `g_moe_rx_T{t}` of one layer and tile: the routed
/// experts of `t` rows (one node: prefill rows are the same rows in every run of a request) and
/// the layer tail `L_out = x2 + (mo + r)`. Inputs are the layer recipe's post-attention residual
/// `x2`, its normed rows `h2`, the routing tables and the gated shared expert's output `mo`.
pub fn build_prefill_routed_graph(spec: &ModelSpec, t: u64) -> GraphBuilder {
    let mut gb = GraphBuilder::new(format!("g_moe_rx_T{t}"));
    let hid = spec.hidden;
    let ek = spec.experts_per_tok;
    let h2 = gb.add_persistent_tensor("rx_h2", SynDataType::Bf16, &[hid, t]);
    let rid = gb.add_persistent_tensor("rx_rid", SynDataType::Int32, &[ek, t]);
    let rwb = gb.add_persistent_tensor("rx_rwb", SynDataType::Bf16, &[ek, t]);
    let x2 = gb.add_persistent_tensor("rx_x2", SynDataType::Bf16, &[hid, t]);
    let mo = gb.add_persistent_tensor("rx_mo", SynDataType::Bf16, &[hid, t]);
    let r = gb.add_intermediate("rx_r", SynDataType::Bf16, &[hid, t]);
    append_routed_experts(&mut gb, spec, h2, rid, rwb, r, "L_", "rx_", t, 0);
    let sr = gb.add_intermediate("rx_sr", SynDataType::Bf16, &[hid, t]);
    gb.op_add_bf16("rx_sr_add", mo, r, sr);
    let out = gb.add_persistent_tensor("L_out", SynDataType::Bf16, &[hid, t]);
    gb.op_add_bf16("rx_out_add", x2, sr, out);
    gb
}

/// Binds layer `l`'s extra weights (see [`is_ext_layer_weight`]) under their generic per-layer
/// prefill names: `L<l>_<suffix>` -> `L_<suffix>`.
pub(crate) fn bind_ext_layer_weights(
    weights_addr: &BTreeMap<String, u64>,
    l: usize,
    pool: &mut BTreeMap<String, u64>,
) {
    let pfx = format!("L{l}_");
    // every key of layer l sorts inside [pfx, "L<l>`"): '`' is the byte after '_'
    let end = format!("L{l}`");
    let bounds = (
        std::ops::Bound::Included(pfx.as_str()),
        std::ops::Bound::Excluded(end.as_str()),
    );
    for (name, &addr) in weights_addr.range::<str, _>(bounds) {
        let suffix = &name[pfx.len()..];
        if is_ext_layer_weight(suffix) {
            pool.insert(format!("L_{suffix}"), addr);
        }
    }
}

/// Appends the MoE MLP of one layer after the post-attention norm (`x2` = the residual, `h2` =
/// its normed rows) and the layer tail: the residual add, fused with the next layer's input norm
/// when the caller asks for it. Returns `(layer output, next layer's normed input)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn append_moe_mlp(
    gb: &mut GraphBuilder,
    spec: &ModelSpec,
    x2: TensorId,
    h2: TensorId,
    lpfx: &str,
    t: u64,
    fused_anorm: bool,
    wnext_norm: Option<(&str, &str)>,
) -> (TensorId, Option<TensorId>) {
    append_moe_mlp_as(gb, spec, x2, h2, lpfx, lpfx, t, fused_anorm, wnext_norm)
}

/// [`append_moe_mlp`] with the layer's weights named `<wpfx><weight>` and every tensor and node
/// the pass creates named `<lpfx><name>`: several passes of one graph may run the same layer
/// (the MTP head's catch-up rows and its draft passes).
///
/// Routed form: the generic per-layer prefill recipes (`wpfx == "L_"`) end with the gated shared
/// expert's output `<lpfx>mo` and return it; [`build_prefill_routed_graph`] is the rest of the
/// layer. Every other graph is a decode-step pass and groups its rows ([`MOE_GROUP_ROWS`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn append_moe_mlp_as(
    gb: &mut GraphBuilder,
    spec: &ModelSpec,
    x2: TensorId,
    h2: TensorId,
    wpfx: &str,
    lpfx: &str,
    t: u64,
    fused_anorm: bool,
    wnext_norm: Option<(&str, &str)>,
) -> (TensorId, Option<TensorId>) {
    let hid = spec.hidden;
    let ne = spec.n_experts;
    let ek = spec.experts_per_tok;
    let ef = spec.moe_intermediate;
    let routed = spec.moe_routed;
    let inter = device_inter(spec);

    let gate = add_linear_bf16(
        gb,
        h2,
        &format!("{wpfx}W_gate"),
        hid,
        inter,
        t,
        &format!("{lpfx}gate"),
    );
    let up = add_linear_bf16(
        gb,
        h2,
        &format!("{wpfx}W_up"),
        hid,
        inter,
        t,
        &format!("{lpfx}up"),
    );

    // Router and shared-expert gate logits, in the model dtype, from the unquantised rows.
    let rl = add_linear_bf16(
        gb,
        h2,
        &format!("{wpfx}W_router"),
        hid,
        ne,
        t,
        &format!("{lpfx}rlogit"),
    );
    let sgl = add_linear_bf16(
        gb,
        h2,
        &format!("{wpfx}W_sgate"),
        hid,
        1,
        t,
        &format!("{lpfx}sgl"),
    );
    let rid = gb.add_persistent_tensor(format!("{lpfx}rid"), SynDataType::Int32, &[ek, t]);
    let rw = gb.add_persistent_tensor(format!("{lpfx}rw"), SynDataType::Float, &[ek, t]);
    let rwd = gb.add_persistent_tensor(format!("{lpfx}rwd"), SynDataType::Bf16, &[ne, t]);
    let rw2 = gb.add_persistent_tensor(
        format!("{lpfx}rw2"),
        SynDataType::Float,
        &[ne / 2 + 64, 2, t],
    );
    let rwb = gb.add_persistent_tensor(format!("{lpfx}rwb"), SynDataType::Bf16, &[ek, t]);
    gb.add_node(
        "gaudium_moe_route",
        format!("{lpfx}route"),
        &[rl, sgl],
        &[rid, rw, rwd, rw2, rwb],
        &[],
    );

    // m = silu(g) * u * w per selected block, zeros elsewhere. Dense-masked: block 0 = expert 0.
    // Routed: the one block is the shared expert, whose weight is table slot NE (the shared gate).
    let m = gb.add_persistent_tensor(
        format!("{lpfx}{}", down_site(spec)),
        SynDataType::Bf16,
        &[inter, t],
    );
    let first_block = if routed { ne as u32 } else { 0u32 };
    gb.add_node(
        "gaudium_moe_act",
        format!("{lpfx}moe_act"),
        &[gate, up, rw2],
        &[m],
        &u32_params(&[ef as u32, first_block]),
    );

    let mut mo = add_linear_bf16(
        gb,
        m,
        &format!("{wpfx}W_down"),
        inter,
        hid,
        t,
        &format!("{lpfx}mo"),
    );

    if routed {
        if wpfx == "L_" {
            // generic per-layer prefill recipe: the routed experts and the tail run in the
            // routed-experts recipe of the tile (module docs)
            return (mo, None);
        }
        let r = gb.add_persistent_tensor(format!("{lpfx}mlp_r"), SynDataType::Bf16, &[hid, t]);
        append_routed_experts(gb, spec, h2, rid, rwb, r, wpfx, lpfx, t, MOE_GROUP_ROWS);
        let sr = gb.add_persistent_tensor(format!("{lpfx}mlp_sr"), SynDataType::Bf16, &[hid, t]);
        gb.op_add_bf16(format!("{lpfx}mlp_sr_add"), mo, r, sr);
        mo = sr;
    }

    if fused_anorm {
        if let Some((next_w, next_h)) = wnext_norm {
            let (out, h_next) =
                add_fused_add_norm(gb, spec, x2, mo, next_w, &format!("{lpfx}out"), next_h, t);
            return (out, Some(h_next));
        }
    }
    let out = gb.add_persistent_tensor(format!("{lpfx}out"), SynDataType::Bf16, &[hid, t]);
    gb.op_add_bf16(format!("{lpfx}out_add"), x2, mo, out);
    (out, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::qwen3::{
        build_decode_step_graph, build_prefill_attn_layer_graph, build_prefill_linear_layer_graph,
    };
    use crate::synapse::graph::GraphBuilder;

    pub(crate) fn qwen36_moe_spec() -> ModelSpec {
        let cfg = r#"{
          "architectures": ["Qwen3_5MoeForConditionalGeneration"],
          "model_type": "qwen3_5_moe",
          "quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]},
          "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "vocab_size": 248320,
            "rms_norm_eps": 1e-6,
            "eos_token_id": 248044,
            "full_attention_interval": 4,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25}
          }
        }"#;
        ModelSpec::from_config_json(cfg).unwrap()
    }

    #[test]
    fn moe_shape_is_supported_and_wide_mlp_is_257_blocks() {
        let spec = qwen36_moe_spec();
        spec.validate().unwrap();
        assert!(supported_shape(&spec));
        assert_eq!(wide_inter(&spec), 131_584);
        assert_eq!(spec.n_full_attention_layers(), 10);
        assert_eq!(spec.n_linear_layers(), 30);
        assert_eq!(spec.gdn_qkv_dim(), 8192);
        assert_eq!(spec.gdn_qkvz_dim(), 12288);
        assert_eq!(spec.qg_dim(), 8192);
        assert_eq!(spec.eos_token_ids, vec![248_044]);
    }

    #[test]
    fn bf16_and_fp8_moe_step_graphs_differ_only_by_fp8_tensors() {
        let mut spec = qwen36_moe_spec();
        let elfs = vec![("GAUDIUM_MOEACT_ELF".to_string(), "abcd".to_string())];
        let bf = build_decode_step_graph(&spec, 4096, 1, &elfs);
        spec.fp8_weights = true;
        let f8 = build_decode_step_graph(&spec, 4096, 1, &elfs);
        assert_ne!(bf.fingerprint("1.24.2"), f8.fingerprint("1.24.2"));

        let bfs = bf.persistent_tensor_specs();
        let f8s = f8.persistent_tensor_specs();
        let get = |v: &[(String, SynDataType, Vec<u64>, u64)], n: &str| {
            v.iter().find(|s| s.0 == n).cloned()
        };
        // wide MLP weights: bf16 vs fp8 codes, HF [out, in] layout as {K, N}
        let wg_bf = get(&bfs, "L0_W_gate").unwrap();
        let wg_f8 = get(&f8s, "L0_W_gate").unwrap();
        assert_eq!(wg_bf.2, vec![2048, 131_584]);
        assert_eq!(wg_bf.1, SynDataType::Bf16);
        assert_eq!(wg_f8.1, SynDataType::Fp8_143);
        assert_eq!(wg_bf.3, 2 * wg_f8.3);
        let wd = get(&f8s, "L0_W_down").unwrap();
        assert_eq!(wd.2, vec![131_584, 2048]);
        // router + shared gate stay bf16 in both
        for specs in [&bfs, &f8s] {
            assert_eq!(
                get(specs, "L0_W_router").unwrap().1,
                SynDataType::Bf16,
                "router dtype"
            );
            assert_eq!(get(specs, "L0_W_router").unwrap().2, vec![2048, 256]);
            assert_eq!(get(specs, "L0_W_sgate").unwrap().2, vec![2048, 1]);
            assert_eq!(get(specs, "L0_rw2").unwrap().2, vec![192, 2, 1]);
            assert_eq!(get(specs, "L0_rid").unwrap().2, vec![8, 1]);
            assert_eq!(get(specs, "L0_m").unwrap().2, vec![131_584, 1]);
        }
        // every FP8 graph tensor that is not in the BF16 graph is a cast, a cast scale or a weight scale
        let bf_names: std::collections::BTreeSet<&str> = bfs.iter().map(|s| s.0.as_str()).collect();
        let mut sites = 0usize;
        for s in &f8s {
            if bf_names.contains(s.0.as_str()) {
                continue;
            }
            let (_, suffix) = crate::engine::fp8_graph::split_layer_name(&s.0).unwrap();
            assert!(
                suffix.ends_with("_xq") || is_fp8_aux_suffix(suffix),
                "unexpected FP8-only tensor {}",
                s.0
            );
            if suffix.ends_with("_cs") {
                sites += 1;
            }
        }
        // 4 GEMM input sites per layer: h, core, h2, m
        assert_eq!(sites, 4 * 40);
        // W_ba, the lm head and the embedding are never quantised
        assert_eq!(get(&f8s, "L0_W_ba").unwrap().1, SynDataType::Bf16);
        assert_eq!(get(&f8s, "W_lm").unwrap().1, SynDataType::Bf16);
        assert!(get(&f8s, "L0_W_ba_scale").is_none());
        assert!(get(&f8s, "h_final_cs").is_none());
    }

    #[test]
    fn routed_form_keeps_the_shared_expert_dense_and_stacks_the_experts() {
        let mut spec = qwen36_moe_spec();
        spec.moe_routed = true;
        spec.fp8_weights = true;
        assert_eq!(device_inter(&spec), 512);
        assert_eq!(down_site(&spec), "ms");
        let elfs: Vec<(String, String)> = Vec::new();
        let lin = build_prefill_linear_layer_graph(&spec, 4, &elfs);
        let specs = lin.persistent_tensor_specs();
        let get = |n: &str| specs.iter().find(|s| s.0 == n).cloned();
        // shared expert: an FP8 MLP of one block; its product is the `ms` cast site
        assert_eq!(get("L_W_gate").unwrap().2, vec![2048, 512]);
        assert_eq!(get("L_W_gate").unwrap().1, SynDataType::Fp8_143);
        assert_eq!(get("L_W_down").unwrap().2, vec![512, 2048]);
        assert_eq!(get("L_ms").unwrap().2, vec![512, 4]);
        assert!(get("L_ms_cs").is_some());
        assert!(get("L_m").is_none());
        // the layer recipe stops after the gated shared expert: no experts, no layer output
        assert!(get("L_EW12").is_none() && get("L_out").is_none() && get("L_mlp_r").is_none());
        for (_, src) in ROUTED_PREFILL_INPUTS {
            assert!(
                get(src).is_some(),
                "layer recipe must leave {src} for the experts recipe"
            );
        }
        assert_eq!(lin.declared_intermediate_bytes(), 0);
        let att = build_prefill_attn_layer_graph(&spec, 4, 8192, 2, &elfs);
        let anames: Vec<String> = att
            .persistent_tensor_specs()
            .into_iter()
            .map(|s| s.0)
            .collect();
        assert!(!anames.iter().any(|n| n == "L_EW12" || n == "L_out"));

        // the routed-experts recipe: stacked BF16 experts (never quantised), the tail, 7 bindings
        let rx = build_prefill_routed_graph(&spec, 4);
        let rspecs = rx.persistent_tensor_specs();
        let rget = |n: &str| rspecs.iter().find(|s| s.0 == n).cloned();
        let ew12 = rget("L_EW12").unwrap();
        assert_eq!(
            (ew12.1, ew12.2.clone()),
            (SynDataType::Bf16, vec![2048, 1024, 256])
        );
        assert_eq!(rget("L_EW3").unwrap().2, vec![512, 2048, 256]);
        assert!(rget("L_EW12_scale").is_none());
        assert_eq!(rget("L_out").unwrap().2, vec![2048, 4]);
        assert_eq!(rget("rx_rid").unwrap().2, vec![8, 4]);
        assert_eq!(rspecs.len(), 8);
        assert!(is_ext_layer_weight("EW12") && is_ext_layer_weight("EW3"));
        // the per-expert views and the two sums are workspace intermediates, not launch bindings
        assert!(rget("L_EW12_v0").is_none());
        assert_eq!(
            rx.declared_intermediate_bytes(),
            2 * 256 * (2048 * 1024 * 2 + 512 * 2048 * 2) + 2 * 2048 * 4 * 2
        );
        // one recipe whatever the layer kind or chunk count; another per padded row count
        assert_eq!(
            rx.fingerprint("1.24.2"),
            build_prefill_routed_graph(&spec, 4).fingerprint("1.24.2")
        );
        assert_ne!(
            rx.fingerprint("1.24.2"),
            build_prefill_routed_graph(&spec, 8).fingerprint("1.24.2")
        );
        let dense = {
            let mut d = qwen36_moe_spec();
            d.fp8_weights = true;
            build_prefill_linear_layer_graph(&d, 4, &elfs)
        };
        assert_ne!(dense.fingerprint("1.24.2"), lin.fingerprint("1.24.2"));
        assert!(dense
            .persistent_tensor_specs()
            .iter()
            .any(|s| s.0 == "L_out"));
    }

    #[test]
    fn routed_prefill_rows_are_a_small_padded_set() {
        assert_eq!(routed_prefill_rows(1, 256), 8);
        assert_eq!(routed_prefill_rows(8, 256), 8);
        assert_eq!(routed_prefill_rows(9, 256), 16);
        assert_eq!(routed_prefill_rows(46, 256), 64);
        assert_eq!(routed_prefill_rows(129, 256), 256);
        assert_eq!(routed_prefill_rows(256, 256), 256);
        // a tile length that is not a power of two caps the set
        assert_eq!(routed_prefill_rows(150, 200), 200);
        assert_eq!(routed_prefill_rows(200, 200), 200);
        let set: std::collections::BTreeSet<u64> =
            (1..=256).map(|t| routed_prefill_rows(t, 256)).collect();
        assert_eq!(
            set.into_iter().collect::<Vec<_>>(),
            vec![8, 16, 32, 64, 128, 256]
        );
        for t in 1..=256u64 {
            assert!(routed_prefill_rows(t, 256) >= t);
        }
    }

    #[test]
    fn routed_step_groups_rows_and_shares_the_expert_views() {
        use crate::engine::qwen3::spec::build_spec_step_graph;
        let mut spec = qwen36_moe_spec();
        spec.moe_routed = true;
        let count = |gb: &GraphBuilder, guid: &str, pfx: &str| {
            gb.node_io_views()
                .iter()
                .filter(|n| n.guid == guid && n.name.starts_with(pfx))
                .count()
        };
        // 8 verify rows: one node per layer; 9 rows: 8 + 1; 16 rows: 8 + 8
        for (t, groups) in [(1u64, 1usize), (8, 1), (9, 2), (16, 2), (17, 3)] {
            let gb = build_spec_step_graph(&spec, 4096, 1, t, t, None, &[]);
            assert_eq!(count(&gb, "moe_bf16", "L0_"), groups, "T={t}");
            assert_eq!(count(&gb, "moe_bf16", "L"), groups * 40, "T={t}");
            // one split per stacked tensor and layer whatever the group count
            assert_eq!(count(&gb, "split", "L0_EW"), 2, "T={t}");
            let views = gb.node_io_views();
            let nodes: Vec<_> = views
                .iter()
                .filter(|n| n.guid == "moe_bf16" && n.name.starts_with("L0_"))
                .collect();
            let rows: u64 = nodes.iter().map(|n| n.inputs[0].dims[1]).sum();
            assert_eq!(rows, t);
            for n in &nodes {
                assert!(n.inputs[0].dims[1] <= MOE_GROUP_ROWS);
                assert_eq!(n.inputs.len(), 3 + 2 * 256);
                // every group of the layer reads the same expert views
                assert_eq!(n.inputs[3].id, nodes[0].inputs[3].id);
                assert_eq!(n.inputs[514].id, nodes[0].inputs[514].id);
            }
        }
        // the one-row step keeps the graph (and so the cached recipe) of the ungrouped builder
        let k0 = build_decode_step_graph(&spec, 4096, 1, &[]);
        assert_eq!(count(&k0, "moe_bf16", "L"), 40);
        assert_eq!(count(&k0, "split", "L"), 80);
    }

    #[test]
    fn moe_mtp_head_runs_its_own_experts_in_every_pass() {
        use crate::engine::qwen3::spec::mtp;
        let dims = |gb: &GraphBuilder, n: &str| {
            gb.persistent_tensor_specs()
                .iter()
                .find(|t| t.0 == n)
                .map(|t| (t.1, t.2.clone()))
        };
        let count = |gb: &GraphBuilder, guid: &str, pfx: &str| {
            gb.node_io_views()
                .iter()
                .filter(|n| n.guid == guid && n.name.starts_with(pfx))
                .count()
        };
        // dense-masked form, FP8 trunk: the head's wide MLP stays BF16
        let mut spec = qwen36_moe_spec();
        spec.fp8_weights = true;
        assert!(mtp::Mtp::new(&spec, 4, 0, None).is_ok());
        assert_eq!(mtp::mtp_elfs(&spec)[0], "GAUDIUM_APREPF_G2Q8_ELF");
        let g = mtp::build_chain_step_graph(&spec, 4096, 1, 5, 5, 0, &[]);
        assert_eq!(
            dims(&g, "M_W_gate"),
            Some((SynDataType::Bf16, vec![2048, 131_584]))
        );
        assert_eq!(dims(&g, "L0_W_gate").unwrap().0, SynDataType::Fp8_143);
        assert_eq!(dims(&g, "M_W_router").unwrap().1, vec![2048, 256]);
        assert_eq!(dims(&g, "M_rid").unwrap().1, vec![8, 5]);
        assert_eq!(dims(&g, "d2_rid").unwrap().1, vec![8, 1]);
        assert_eq!(dims(&g, "d4_m").unwrap().1, vec![131_584, 1]);
        assert!(dims(&g, "d5_m").is_none() && dims(&g, "d2_act").is_none());
        assert_eq!(dims(&g, "next_ids").unwrap().1, vec![5]);
        for w in ["M_W_router", "M_W_sgate", "M_EW12", "M_EW3", "M_W_gate"] {
            assert!(mtp::is_mtp_weight(w), "{w}");
        }
        for a in ["M_rid", "M_rw2", "M_m", "d2_ms", "M_mlp_r"] {
            assert!(!mtp::is_mtp_weight(a), "{a}");
        }

        // routed form: catch-up node (5 rows) + 3 one-row draft passes on ONE set of views
        spec.moe_routed = true;
        let g = mtp::build_chain_step_graph(&spec, 4096, 1, 5, 5, 0, &[]);
        assert_eq!(dims(&g, "M_W_gate").unwrap().1, vec![2048, 512]);
        assert_eq!(dims(&g, "M_EW12").unwrap().1, vec![2048, 1024, 256]);
        assert_eq!(count(&g, "moe_bf16", "M_"), 1);
        assert_eq!(count(&g, "moe_bf16", "d"), 3);
        assert_eq!(count(&g, "split", "M_EW"), 2);
        assert_eq!(dims(&g, "d3_mlp_r").unwrap().1, vec![2048, 1]);
        // a step's draft count is chosen apart from its verify rows (several verify widths)
        let wide = mtp::build_chain_step_graph_kd(&spec, 4096, 1, 3, 6, 7, 0, &[], false);
        assert_eq!(dims(&wide, "tokens").unwrap().1, vec![1, 3]);
        assert_eq!(dims(&wide, "M_rid").unwrap().1, vec![8, 3]);
        assert_eq!(dims(&wide, "next_ids").unwrap().1, vec![7]);
        assert!(dims(&wide, "d6_tok").is_some() && dims(&wide, "d7_tok").is_none());
        let fixed = mtp::build_chain_step_graph_kd(&spec, 4096, 1, 3, 2, 7, 0, &[], false);
        assert_ne!(wide.fingerprint("x"), fixed.fingerprint("x"));
        assert_eq!(
            fixed.fingerprint("x"),
            mtp::build_chain_step_graph(&spec, 4096, 1, 3, 7, 0, &[]).fingerprint("x")
        );
        // the first block of a request and the prompt's MTP rows
        let first = mtp::build_first_graph(&spec, 4096, 1, 6, 0, None, &[]);
        assert_eq!(dims(&first, "mtp_first_ids").unwrap().1, vec![7]);
        assert_eq!(count(&first, "moe_bf16", ""), 6);
        let ctx = mtp::build_ctx_graph(&spec, 256, 4096, 1, &[]);
        assert_eq!(count(&ctx, "moe_bf16", ""), 0);
    }

    #[test]
    fn prefill_layer_graphs_name_ext_weights_generically() {
        let mut spec = qwen36_moe_spec();
        spec.fp8_weights = true;
        let elfs: Vec<(String, String)> = Vec::new();
        let lin = build_prefill_linear_layer_graph(&spec, 16, &elfs);
        let att = build_prefill_attn_layer_graph(&spec, 16, 4096, 1, &elfs);
        for gb in [&lin, &att] {
            let names: Vec<String> = gb
                .persistent_tensor_specs()
                .into_iter()
                .map(|s| s.0)
                .collect();
            for n in [
                "L_W_router",
                "L_W_sgate",
                "L_W_gate_scale",
                "L_W_down_scale",
                "L_h_cs",
                "L_h2_sa",
                "L_m_cs",
                "L_core_cs",
                "L_h2_xq",
            ] {
                assert!(names.iter().any(|x| x == n), "missing {n}");
            }
        }

        let mut weights = BTreeMap::new();
        for (i, n) in [
            "L1_W_router",
            "L1_W_sgate",
            "L1_W_gate_scale",
            "L1_h2_cs",
            "L1_h2_sa",
            "L1_W_gate",
            "L10_W_router",
            "L1_w_in",
        ]
        .iter()
        .enumerate()
        {
            weights.insert((*n).to_string(), 1000 + i as u64);
        }
        let mut pool = BTreeMap::new();
        bind_ext_layer_weights(&weights, 1, &mut pool);
        assert_eq!(pool.get("L_W_router"), Some(&1000));
        assert_eq!(pool.get("L_W_sgate"), Some(&1001));
        assert_eq!(pool.get("L_W_gate_scale"), Some(&1002));
        assert_eq!(pool.get("L_h2_cs"), Some(&1003));
        assert_eq!(pool.get("L_h2_sa"), Some(&1004));
        // plain weights are bound by the engine's own suffix lists; layer 10 is another layer
        assert_eq!(pool.len(), 5);
    }
}
