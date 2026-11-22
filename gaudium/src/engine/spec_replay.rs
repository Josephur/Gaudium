//! Checkpoint-free verify rows of the Gated DeltaNet layers (`--gdn-verify replay`).
//!
//! A speculative step runs `T` rows through every DeltaNet layer but only the accepted prefix
//! may reach the state. The checkpoint form stores the 128 x 128 fp32 state of every value head
//! after every row (3.1 MB per row and layer on the 48-head model) and lets the next step load
//! the accepted row's copy. Here a row keeps only what its update was made of: its key row
//! (`rpk`), its decay (`rpg`) and the eight delta vectors of the update (`rpd`, 2 KB per head).
//! The next step loads the state the previous step *started* from, applies the accepted rows'
//! updates to it again (`gaudium_gdn_f32r`: the row kernel's own expression on the row kernel's
//! own operands, in row order, so the result is the accepted row's state to the bit), stores it
//! once, and runs its own rows. Per step and layer: one state load and one state store, whatever
//! the verify width.
//!
//! The request's first step has no previous rows: the kernel replays nothing when the step's
//! `cursor` equals `sp_rp_first`, which the host writes with the first step's cursor.
//!
//! Buffers alternate by step parity like the rest of the step state: a step reads the update
//! buffers the previous step wrote (`*_in`) and writes the other set (`*_out`); the state is two
//! single buffers per layer instead of two rings of `T_max` slots. The conv state keeps its
//! per-row checkpoint ring (123 KB per row and layer).
//!
//! A step family is built in this form when its kernel list carries [`GDNR_ELF`]; draft trees
//! keep the checkpoint form.

use super::*;

/// Kernel of the replay recurrence (served by the step glue library).
pub const GDNR_ELF: &str = "GAUDIUM_GDNR_ELF";

/// Is the step family with these kernels built in the replay form?
pub fn enabled(elf_hashes: &[(String, String)]) -> bool {
    elf_hashes.iter().any(|(k, _)| k == GDNR_ELF)
}

/// Step tensors of the replay form bound by step parity (never by their own name).
pub fn is_parity_name(name: &str) -> bool {
    [
        "_S_base", "_rpk_in", "_rpk_out", "_rpd_in", "_rpd_out", "_rpg_in", "_rpg_out",
    ]
    .iter()
    .any(|s| name.ends_with(s))
}

/// The module buffers of the replay form for rings of `ns` rows: `(name, bytes)`. Two sets
/// (`sp_rp{k,d,g}{0,1}_L<l>`) per DeltaNet layer, and the first-step cursor.
pub fn buffer_specs(spec: &ModelSpec, ns: u64) -> Vec<(String, u64)> {
    let (hv, dl, hvp) = (spec.gdn_v_heads, spec.gdn_v_dim, spec.gdn_hvp());
    let mut out = vec![("sp_rp_first".to_string(), 4)];
    for l in 0..spec.n_layers as usize {
        if spec.is_full_attention_layer(l) {
            continue;
        }
        for p in 0..2 {
            out.push((format!("sp_rpk{p}_L{l}"), dl * hv * ns * 4));
            out.push((format!("sp_rpd{p}_L{l}"), 64 * 8 * hv * ns * 4));
            out.push((format!("sp_rpg{p}_L{l}"), hvp * ns * 4));
        }
    }
    out
}

/// Adds the replay bindings of a step of parity `p` (reads set `p`, writes set `1 - p`).
pub(super) fn bind(
    pool: &mut BTreeMap<String, u64>,
    addr: &BTreeMap<String, u64>,
    gdn_state: &BTreeMap<usize, GdnLayerState>,
    p: usize,
) {
    let q = 1 - p;
    for (&l, st) in gdn_state {
        pool.insert(format!("L{l}_S_in"), st.s[p]);
        pool.insert(format!("L{l}_S_base"), st.s[q]);
        for what in ["rpk", "rpd", "rpg"] {
            pool.insert(
                format!("L{l}_{what}_in"),
                addr[&format!("sp_{what}{p}_L{l}")],
            );
            pool.insert(
                format!("L{l}_{what}_out"),
                addr[&format!("sp_{what}{q}_L{l}")],
            );
        }
    }
}

/// Gated DeltaNet layer over `t` verify rows in the replay form (see the module text). `slot` =
/// the previous step's accepted row, `cursor` = this step's cursor, `ns` = rows of the update
/// buffers and slots of the conv-state ring.
#[allow(clippy::too_many_arguments)]
pub(super) fn append_linear_layer_replay(
    gb: &mut GraphBuilder,
    spec: &ModelSpec,
    x: TensorId,
    h_pre: Option<TensorId>,
    slot: TensorId,
    cursor: TensorId,
    lpfx: &str,
    t: u64,
    ns: u64,
    wnext_norm: Option<(&str, &str)>,
) -> (TensorId, Option<TensorId>) {
    let hid = spec.hidden;
    let hv = spec.gdn_v_heads;
    let dl = spec.gdn_v_dim;
    let qkv = spec.gdn_qkv_dim();
    let qkvz_dim = spec.gdn_qkvz_dim();
    let hvp = spec.gdn_hvp();
    let ba_dim = spec.gdn_ba_dim();
    let f32t = SynDataType::Float;

    let h = match h_pre {
        Some(h_in) => h_in,
        None => add_rmsnorm(gb, spec, x, &format!("{lpfx}w_in"), &format!("{lpfx}h"), t),
    };
    let qkvz = add_linear_bf16(
        gb,
        h,
        &format!("{lpfx}W_qkvz"),
        hid,
        qkvz_dim,
        t,
        &format!("{lpfx}qkvz"),
    );
    let ba = add_linear_bf16(
        gb,
        h,
        &format!("{lpfx}W_ba"),
        hid,
        ba_dim,
        t,
        &format!("{lpfx}ba"),
    );

    // conv state: the per-row checkpoint ring, as in the checkpoint form
    let conv_w = gb.get_or_add_persistent_tensor(format!("{lpfx}conv_w"), f32t, &[qkv, 4]);
    let conv_b = gb.get_or_add_persistent_tensor(format!("{lpfx}conv_b"), f32t, &[qkv]);
    let cs_in = gb.add_persistent_tensor(format!("{lpfx}cs_in"), f32t, &[qkv, 3, ns]);
    let cs_out = gb.add_persistent_tensor(format!("{lpfx}cs_out"), f32t, &[qkv, 3]);
    let cs_ckpt = gb.add_persistent_tensor(format!("{lpfx}cs_ckpt"), f32t, &[qkv, 3, t]);
    let q = gb.add_persistent_tensor(format!("{lpfx}q"), f32t, &[dl, hv, t]);
    let k = gb.add_persistent_tensor(format!("{lpfx}rpk_out"), f32t, &[dl, hv, t]);
    let v = gb.add_persistent_tensor(format!("{lpfx}v"), f32t, &[dl, hv, t]);
    gb.add_node(
        "gaudium_gdn_prep",
        format!("{lpfx}gdn_prep"),
        &[qkvz, conv_w, cs_in, conv_b, slot],
        &[q, k, v, cs_out, cs_ckpt],
        &u32_params(&[2 | 16]),
    );

    let a_log = gb.get_or_add_persistent_tensor(format!("{lpfx}A_log"), f32t, &[hvp]);
    let dt_bias = gb.get_or_add_persistent_tensor(format!("{lpfx}dt_bias"), f32t, &[hvp]);
    let eg = gb.add_persistent_tensor(format!("{lpfx}rpg_out"), f32t, &[hvp, t]);
    let beta = gb.add_persistent_tensor(format!("{lpfx}beta"), f32t, &[hvp, t]);
    gb.add_node(
        "gaudium_gdn_gate",
        format!("{lpfx}gdn_gate"),
        &[ba, a_log, dt_bias],
        &[eg, beta],
        &[],
    );

    let s_in = gb.add_persistent_tensor(format!("{lpfx}S_in"), f32t, &[64, 256, hv]);
    let s_base = gb.add_persistent_tensor(format!("{lpfx}S_base"), f32t, &[64, 256, hv]);
    let pk = gb.add_persistent_tensor(format!("{lpfx}rpk_in"), f32t, &[dl, hv, ns]);
    let pd = gb.add_persistent_tensor(format!("{lpfx}rpd_in"), f32t, &[64, 8, hv, ns]);
    let pg = gb.add_persistent_tensor(format!("{lpfx}rpg_in"), f32t, &[hvp, ns]);
    let dr = gb.add_persistent_tensor(format!("{lpfx}rpd_out"), f32t, &[64, 8, hv, t]);
    let cur0 = gb.get_or_add_persistent_tensor("sp_rp_first", SynDataType::Int32, &[1]);
    let o = gb.add_persistent_tensor(format!("{lpfx}o"), f32t, &[dl, hv, t]);
    gb.add_node(
        "gaudium_gdn_f32r",
        format!("{lpfx}gdn"),
        &[s_in, q, k, v, eg, beta, pk, pd, pg, slot, cursor, cur0],
        &[o, s_base, dr],
        &u32_params(&[0u32]),
    );

    let norm_w = gb.get_or_add_persistent_tensor(format!("{lpfx}norm_w"), f32t, &[dl]);
    let core = gb.add_persistent_tensor(format!("{lpfx}core"), SynDataType::Bf16, &[hv * dl, t]);
    gb.add_node(
        "gaudium_gdn_post",
        format!("{lpfx}gdn_post"),
        &[o, qkvz, norm_w],
        &[core],
        &u32_params(&[0u32]),
    );

    let attn = add_linear_bf16(
        gb,
        core,
        &format!("{lpfx}W_o"),
        hv * dl,
        hid,
        t,
        &format!("{lpfx}attn"),
    );
    append_post_and_mlp(gb, spec, x, attn, lpfx, t, true, wnext_norm)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hybrid_spec() -> ModelSpec {
        let cfg = r#"{
          "architectures": ["Qwen3_5ForConditionalGeneration"],
          "model_type": "qwen3_5",
          "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "vocab_size": 248320,
            "rms_norm_eps": 1e-6,
            "full_attention_interval": 4,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.25}
          }
        }"#;
        ModelSpec::from_config_json(cfg).unwrap()
    }

    #[test]
    fn replay_family_has_single_states_and_update_buffers() {
        let spec = hybrid_spec();
        let ck = vec![("GAUDIUM_ACCEPTT_ELF".to_string(), "00".to_string())];
        let mut rp = ck.clone();
        rp.push((GDNR_ELF.to_string(), "11".to_string()));
        assert!(!enabled(&ck) && enabled(&rp));
        let g = build_spec_step_graph(&spec, 4096, 1, 8, 16, None, &rp);
        let specs = g.persistent_tensor_specs();
        let dims = |n: &str| specs.iter().find(|t| t.0 == n).map(|t| t.2.clone());
        // one state in, one state out: no slot dimension, no per-row state checkpoints
        assert_eq!(dims("L0_S_in"), Some(vec![64, 256, 48]));
        assert_eq!(dims("L0_S_base"), Some(vec![64, 256, 48]));
        assert!(!specs.iter().any(|t| t.0.ends_with("_S_ckpt")));
        // this step's rows are written narrow, the previous step's are read at ring width
        assert_eq!(dims("L0_rpk_out"), Some(vec![128, 48, 8]));
        assert_eq!(dims("L0_rpk_in"), Some(vec![128, 48, 16]));
        assert_eq!(dims("L0_rpd_out"), Some(vec![64, 8, 48, 8]));
        assert_eq!(dims("L0_rpd_in"), Some(vec![64, 8, 48, 16]));
        assert_eq!(dims("L0_rpg_in"), Some(vec![64, 16]));
        assert_eq!(dims("sp_rp_first"), Some(vec![1]));
        // the conv state keeps its checkpoint ring
        assert_eq!(dims("L0_cs_in"), Some(vec![10240, 3, 16]));
        assert_eq!(dims("L0_cs_ckpt"), Some(vec![10240, 3, 8]));
        // the checkpoint family is a different graph
        let c = build_spec_step_graph(&spec, 4096, 1, 8, 16, None, &ck);
        assert_ne!(g.fingerprint("x"), c.fingerprint("x"));
        assert!(c
            .persistent_tensor_specs()
            .iter()
            .any(|t| t.0 == "L0_S_ckpt"));
    }

    #[test]
    fn checkpoint_family_takes_the_vector_scalar_row_kernel_when_listed() {
        let spec = hybrid_spec();
        let ck = vec![("GAUDIUM_ACCEPTT_ELF".to_string(), "00".to_string())];
        let mut fast = ck.clone();
        fast.push((super::super::GDNV_ELF.to_string(), "22".to_string()));
        let g0 = build_spec_step_graph(&spec, 4096, 1, 8, 16, None, &ck);
        let g1 = build_spec_step_graph(&spec, 4096, 1, 8, 16, None, &fast);
        let guids = |g: &GraphBuilder, name: &str| {
            g.node_io_views().iter().filter(|n| n.guid == name).count()
        };
        assert_eq!(guids(&g0, "gaudium_gdn_f32"), 48);
        assert_eq!(guids(&g0, "gaudium_gdn_f32v"), 0);
        assert_eq!(guids(&g1, "gaudium_gdn_f32"), 0);
        assert_eq!(guids(&g1, "gaudium_gdn_f32v"), 48);
        // same tensors, same bindings, same node list but for the kernel
        assert_eq!(g0.persistent_tensor_specs(), g1.persistent_tensor_specs());
        let io = |g: &GraphBuilder| -> Vec<(String, usize, usize)> {
            g.node_io_views()
                .iter()
                .map(|n| (n.name.to_string(), n.inputs.len(), n.outputs.len()))
                .collect()
        };
        assert_eq!(io(&g0), io(&g1));
        assert_ne!(g0.fingerprint("x"), g1.fingerprint("x"));
    }

    #[test]
    fn buffers_cover_every_linear_layer_and_bind_by_parity() {
        let spec = hybrid_spec();
        let specs = buffer_specs(&spec, 16);
        // 48 DeltaNet layers x 2 parities x 3 buffers + the first-step cursor
        assert_eq!(specs.len(), 48 * 2 * 3 + 1);
        let bytes = |n: &str| specs.iter().find(|s| s.0 == n).map(|s| s.1);
        assert_eq!(bytes("sp_rpk0_L0"), Some(128 * 48 * 16 * 4));
        assert_eq!(bytes("sp_rpd1_L0"), Some(64 * 8 * 48 * 16 * 4));
        assert_eq!(bytes("sp_rpg0_L1"), Some(64 * 16 * 4));
        assert_eq!(bytes("sp_rpk0_L3"), None); // layer 3 is a full-attention layer
        for n in ["L4_S_base", "L4_rpk_in", "L4_rpd_out", "L4_rpg_in"] {
            assert!(is_parity_name(n), "{n}");
        }
        for n in ["L4_S_in", "D0_bpk", "L4_k", "blk_cand"] {
            assert!(!is_parity_name(n), "{n}");
        }
    }
}
