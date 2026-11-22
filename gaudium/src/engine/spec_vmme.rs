//! DeltaNet verify rows through the MME (`--gdn-verify mme`): an opt-in speed mode that is NOT
//! token-identical to `K = 0`.
//!
//! With `S0` the state a step starts from, every verify row is linear in `S0` plus rank-1 terms
//! of the rows before it:
//!
//! ```text
//! S_t     = E_t S0 + sum_{i <= t} e(t,i) k_i (x) d_i          E_t = product of the decays up to t
//! g_t a_t = E_t (S0^T k_t) + sum_{i < t} e(t,i) (k_i . k_t) d_i      d_t = (v_t - g_t a_t) beta_t
//! o_t     = E_t (S0^T q_t) + sum_{i < t} e(t,i) (k_i . q_t) d_i + (k_t . q_t) d_t
//! ```
//!
//! The two things that touch 128-wide data against the state or against each other are batched
//! GEMMs (`PR = [k ; q] . S0`, `G = [k ; q] . [k ; q]^T`), which cost the MME the same for 2 rows
//! or 32; `gaudium_gdn_vrec` then runs one MAC chain per (row, earlier row) with no state
//! traffic. After the accept, `gaudium_gdn_vcommit` and one more GEMM per layer apply the
//! accepted rows in one closed form: `S' = E_a S0 + [k ; q]^T . dlc`. The row kernel instead
//! sweeps the 64 KB state of every head twice per row (measured on the card: 2.3 ms of a 16-row
//! step in the recurrence alone, 0.14 ms per row).
//!
//! **Exactness.** Same algebra, another summation order in fp32, and the committed state
//! depends on where the step boundaries fell (which rows were committed together). A stream
//! therefore leaves the `K = 0` stream at near-ties of the BF16 logits, and the position depends
//! on the verify width. That is why this is a switch and never a default: use `--compare-k0` to
//! see where a request forks.
//!
//! The state lives in the MME's matrix layout `{128, 128, HV, 1}` while a request decodes
//! (`L<l>_mS_in` -> `L<l>_mS_out`, alternating between the layer's two state buffers, which have
//! the same size as the row kernel's blobs). The prefill is untouched: one launch of
//! [`build_conv_graph`] per request converts its blobs.
//!
//! A step family is built in this form when its kernel list carries [`VREC_ELF`]; draft trees
//! keep the checkpoint form.

use super::*;

/// Kernels of the MME verify rows (all served by the base glue library).
pub const VREC_ELF: &str = "GAUDIUM_VREC_ELF";
pub const VCOMMIT_ELF: &str = "GAUDIUM_VCOMMIT_ELF";
pub const RELAY_ELF: &str = "GAUDIUM_RELAY_ELF";

/// Is the step family with these kernels built with the MME verify rows?
pub fn enabled(elf_hashes: &[(String, String)]) -> bool {
    elf_hashes.iter().any(|(k, _)| k == VREC_ELF)
}

/// Step tensors of this form bound by step parity (never by their own name).
pub fn is_parity_name(name: &str) -> bool {
    name.ends_with("_mS_in") || name.ends_with("_mS_out")
}

/// What a layer's commit needs from its verify rows.
pub(super) struct VmLayer {
    lpfx: String,
    s0: TensorId,
    kq: TensorId,
    dl: TensorId,
    eg: TensorId,
}

/// Gated DeltaNet layer over `t` verify rows on the MME (see the module text). `ns` = slots of
/// the conv-state ring. Pushes the layer's commit operands to `vls`.
#[allow(clippy::too_many_arguments)]
pub(super) fn append_linear_layer_vmme(
    gb: &mut GraphBuilder,
    spec: &ModelSpec,
    x: TensorId,
    h_pre: Option<TensorId>,
    slot: TensorId,
    lpfx: &str,
    t: u64,
    ns: u64,
    wnext_norm: Option<(&str, &str)>,
    vls: &mut Vec<VmLayer>,
) -> (TensorId, Option<TensorId>) {
    let hid = spec.hidden;
    let hv = spec.gdn_v_heads;
    let dl_dim = spec.gdn_v_dim;
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

    let conv_w = gb.get_or_add_persistent_tensor(format!("{lpfx}conv_w"), f32t, &[qkv, 4]);
    let conv_b = gb.get_or_add_persistent_tensor(format!("{lpfx}conv_b"), f32t, &[qkv]);
    let cs_in = gb.add_persistent_tensor(format!("{lpfx}cs_in"), f32t, &[qkv, 3, ns]);
    let cs_out = gb.add_persistent_tensor(format!("{lpfx}cs_out"), f32t, &[qkv, 3]);
    let cs_ckpt = gb.add_persistent_tensor(format!("{lpfx}cs_ckpt"), f32t, &[qkv, 3, t]);
    // k and q in the GEMM operand layout: rows 0..t = k, rows t..2t = q
    let kq = gb.add_persistent_tensor(format!("{lpfx}kq"), f32t, &[dl_dim, 2 * t, hv, 1]);
    let kx = gb.add_persistent_tensor(format!("{lpfx}k"), f32t, &[dl_dim, hv, t]);
    let v = gb.add_persistent_tensor(format!("{lpfx}v"), f32t, &[dl_dim, hv, t]);
    gb.add_node(
        "gaudium_gdn_prep",
        format!("{lpfx}gdn_prep"),
        &[qkvz, conv_w, cs_in, conv_b, slot],
        &[kq, kx, v, cs_out, cs_ckpt],
        // per-row conv checkpoints, no cs_out, k and q in the GEMM operand layout
        &u32_params(&[2 | 16 | 128]),
    );

    let a_log = gb.get_or_add_persistent_tensor(format!("{lpfx}A_log"), f32t, &[hvp]);
    let dt_bias = gb.get_or_add_persistent_tensor(format!("{lpfx}dt_bias"), f32t, &[hvp]);
    let eg = gb.add_persistent_tensor(format!("{lpfx}eg"), f32t, &[hvp, t, 1]);
    let beta = gb.add_persistent_tensor(format!("{lpfx}beta"), f32t, &[hvp, t]);
    gb.add_node(
        "gaudium_gdn_gate",
        format!("{lpfx}gdn_gate"),
        &[ba, a_log, dt_bias],
        &[eg, beta],
        &[],
    );

    let s0 = gb.add_persistent_tensor(format!("{lpfx}mS_in"), f32t, &[dl_dim, dl_dim, hv, 1]);
    let g = gb.add_persistent_tensor(format!("{lpfx}G"), f32t, &[2 * t, 2 * t, hv, 1]);
    let pr = gb.add_persistent_tensor(format!("{lpfx}PR"), f32t, &[dl_dim, 2 * t, hv, 1]);
    gb.op_batch_gemm(format!("{lpfx}G"), kq, kq, g, false, true);
    gb.op_batch_gemm(format!("{lpfx}PR"), kq, s0, pr, false, false);
    let o = gb.add_persistent_tensor(format!("{lpfx}o"), f32t, &[dl_dim, hv, t]);
    let dl = gb.add_persistent_tensor(format!("{lpfx}dl"), f32t, &[dl_dim, t, hv, 1]);
    gb.add_node(
        "gaudium_gdn_vrec",
        format!("{lpfx}gdn_vrec"),
        &[pr, g, v, eg, beta],
        &[o, dl],
        // a chain of `t` rows (no leaves)
        &u32_params(&[0, 0]),
    );

    let norm_w = gb.get_or_add_persistent_tensor(format!("{lpfx}norm_w"), f32t, &[dl_dim]);
    let core =
        gb.add_persistent_tensor(format!("{lpfx}core"), SynDataType::Bf16, &[hv * dl_dim, t]);
    gb.add_node(
        "gaudium_gdn_post",
        format!("{lpfx}gdn_post"),
        &[o, qkvz, norm_w],
        &[core],
        &u32_params(&[0u32]),
    );
    vls.push(VmLayer {
        lpfx: lpfx.to_string(),
        s0,
        kq,
        dl,
        eg,
    });

    let attn = add_linear_bf16(
        gb,
        core,
        &format!("{lpfx}W_o"),
        hv * dl_dim,
        hid,
        t,
        &format!("{lpfx}attn"),
    );
    append_post_and_mlp(gb, spec, x, attn, lpfx, t, true, wnext_norm)
}

/// The step's one state update per DeltaNet layer, after the accept wrote `a_out` (the last
/// accepted row): `S' = E_a S0 + [k ; q]^T . dlc` -> `L<l>_mS_out`.
pub(super) fn append_commits(
    gb: &mut GraphBuilder,
    spec: &ModelSpec,
    vls: &[VmLayer],
    a_out: TensorId,
    t: u64,
) {
    let (hv, dl_dim) = (spec.gdn_v_heads, spec.gdn_v_dim);
    let f32t = SynDataType::Float;
    for vl in vls {
        let l = &vl.lpfx;
        let dlc = gb.add_persistent_tensor(format!("{l}dlc"), f32t, &[dl_dim, 2 * t, hv, 1]);
        let sd = gb.add_persistent_tensor(format!("{l}mSd"), f32t, &[dl_dim, dl_dim, hv, 1]);
        let sn = gb.add_persistent_tensor(format!("{l}mS_out"), f32t, &[dl_dim, dl_dim, hv, 1]);
        gb.add_node(
            "gaudium_gdn_vcommit",
            format!("{l}gdn_vcommit"),
            &[vl.dl, vl.eg, a_out, vl.s0],
            &[dlc, sd],
            &u32_params(&[0, 0]),
        );
        let p = crate::synapse::ops::GemmParams::new(true, false);
        gb.add_node(
            "batch_gemm",
            format!("{l}Supd"),
            &[vl.kq, dlc, sd],
            &[sn],
            crate::synapse::ops::as_param_bytes(&p),
        );
    }
}

/// The prefill's state blobs -> the matrix states the first step reads (`g_spec_vmconv`).
pub fn build_conv_graph(spec: &ModelSpec, elf_hashes: &[(String, String)]) -> GraphBuilder {
    let mut gb = GraphBuilder::new("g_spec_vmconv");
    for (k, h) in elf_hashes {
        gb.record_kernel_elf_hash(k, h);
    }
    let (hv, dl_dim) = (spec.gdn_v_heads, spec.gdn_v_dim);
    for l in 0..spec.n_layers as usize {
        if spec.is_full_attention_layer(l) {
            continue;
        }
        let blob =
            gb.add_persistent_tensor(format!("L{l}_S_in"), SynDataType::Float, &[64, 256, hv]);
        let m = gb.add_persistent_tensor(
            format!("L{l}_mS_out"),
            SynDataType::Float,
            &[dl_dim, dl_dim, hv, 1],
        );
        gb.add_node(
            "gaudium_gdn_relayout",
            format!("L{l}_vmc_relay"),
            &[blob],
            &[m],
            // mode 0: blob -> matrix
            &u32_params(&[0u32]),
        );
    }
    gb
}

/// The conversion recipe of a loaded engine with its launches for both parities.
pub(super) struct VmConv {
    pub recipe: CompiledRecipe,
    /// `[p]`: reads the blobs of state buffer `p`, writes the matrices of buffer `1 - p`.
    pub launches: Option<[PreparedLaunch; 2]>,
}

impl VmConv {
    pub fn finish(
        &mut self,
        dev: &HpuDevice,
        gdn_state: &BTreeMap<usize, GdnLayerState>,
    ) -> Result<(), String> {
        self.recipe
            .bind_shared_workspace(dev)
            .map_err(|e| format!("bind {} ws: {e}", self.recipe.name))?;
        let prep = |p: usize| -> Result<PreparedLaunch, String> {
            let mut pool: BTreeMap<String, u64> = BTreeMap::new();
            for (&l, st) in gdn_state {
                pool.insert(format!("L{l}_S_in"), st.s[p]);
                pool.insert(format!("L{l}_mS_out"), st.s[1 - p]);
            }
            let exact = select_exact_bindings(&self.recipe, &pool)?;
            self.recipe
                .prepare_launch(&exact)
                .map_err(|e| format!("prepare {} (parity {p}): {e}", self.recipe.name))
        };
        self.launches = Some([prep(0)?, prep(1)?]);
        Ok(())
    }

    /// Converts the prompt's states (blobs in buffer `p`) and waits for it.
    pub fn launch(&self, dev: &HpuDevice, p: usize) -> Result<(), String> {
        let launches = self
            .launches
            .as_ref()
            .ok_or("state conversion not prepared")?;
        self.recipe
            .launch_prepared(dev, &launches[p])
            .map_err(|e| format!("launch {}: {e}", self.recipe.name))?;
        unsafe { dev.sync_stream(dev.compute_stream()) }
            .map_err(|e| format!("sync {}: {e}", self.recipe.name))
    }
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
    fn mme_family_reads_and_commits_one_matrix_state_per_layer() {
        let spec = hybrid_spec();
        let hashes = vec![
            ("GAUDIUM_ACCEPTT_ELF".to_string(), "00".to_string()),
            (VREC_ELF.to_string(), "11".to_string()),
        ];
        assert!(enabled(&hashes));
        let g = build_spec_step_graph(&spec, 4096, 1, 8, 16, None, &hashes);
        let specs = g.persistent_tensor_specs();
        let dims = |n: &str| specs.iter().find(|t| t.0 == n).map(|t| t.2.clone());
        assert_eq!(dims("L0_mS_in"), Some(vec![128, 128, 48, 1]));
        assert_eq!(dims("L0_mS_out"), Some(vec![128, 128, 48, 1]));
        assert_eq!(dims("L0_kq"), Some(vec![128, 16, 48, 1]));
        assert_eq!(dims("L0_G"), Some(vec![16, 16, 48, 1]));
        assert_eq!(dims("L0_PR"), Some(vec![128, 16, 48, 1]));
        assert_eq!(dims("L0_dl"), Some(vec![128, 8, 48, 1]));
        assert_eq!(dims("L0_dlc"), Some(vec![128, 16, 48, 1]));
        assert_eq!(dims("L0_eg"), Some(vec![64, 8, 1]));
        // no blob states, no per-row state checkpoints; the conv ring stays
        assert!(!specs
            .iter()
            .any(|t| t.0.ends_with("_S_in") || t.0.ends_with("_S_ckpt")));
        assert_eq!(dims("L0_cs_in"), Some(vec![10240, 3, 16]));
        let nodes = g.node_io_views();
        let count = |guid: &str| nodes.iter().filter(|n| n.guid == guid).count();
        assert_eq!(count("gaudium_gdn_vrec"), 48);
        assert_eq!(count("gaudium_gdn_vcommit"), 48);
        assert_eq!(count("gaudium_gdn_f32"), 0);
        // per layer: G, PR and the commit GEMM (the attention layers add their own)
        assert_eq!(
            nodes
                .iter()
                .filter(|n| n.guid == "batch_gemm" && n.name.ends_with("_Supd"))
                .count(),
            48
        );
        let supd = nodes.iter().find(|n| n.name == "L0_Supd").unwrap();
        assert_eq!(supd.inputs.len(), 3);
        assert_eq!(supd.outputs[0].name, "L0_mS_out");
        // the commit reads the accept's row
        let vc = nodes.iter().find(|n| n.name == "L0_gdn_vcommit").unwrap();
        assert_eq!(vc.inputs[2].name, "a_out");
        assert!(is_parity_name("L7_mS_in") && is_parity_name("L7_mS_out"));
        assert!(!is_parity_name("L7_mSd") && !is_parity_name("L7_S_in"));
    }

    #[test]
    fn conversion_graph_relays_every_linear_layer() {
        let spec = hybrid_spec();
        let g = build_conv_graph(&spec, &[]);
        let nodes = g.node_io_views();
        assert_eq!(nodes.len(), 48);
        assert!(nodes.iter().all(|n| n.guid == "gaudium_gdn_relayout"));
        assert_eq!(nodes[0].inputs[0].dims, &[64, 256, 48]);
        assert_eq!(nodes[0].outputs[0].dims, &[128, 128, 48, 1]);
    }
}
