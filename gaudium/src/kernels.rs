//! Gaudi 2 TPC-C kernel ELF and Synapse Graph Compiler glue registry.
//!
//! Configures `GC_KERNEL_PATH` and every `GAUDIUM_*_ELF` environment variable consumed by
//! `libgaudium_tpc.so` and `libgaudium_ga.so` from a single kernel build directory (`--kernels` or
//! `GAUDIUM_KERNELS`), and computes build-path-independent kernel ELF hashes (via
//! [`crate::synapse::recipe_cache::hash_tpc_elf_normalized`]) for recipe cache fingerprinting.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::synapse::recipe_cache::{hash_tpc_elf_normalized, sha256_hex};

/// Default Intel Gaudi SDK built-in TPC kernel library path.
pub const DEFAULT_SDK_TPC_LIB: &str = "/usr/lib/habanalabs/libtpc_kernels.so";
/// Default Intel Gaudi SDK SCAL firmware path.
pub const DEFAULT_SCAL_BIN_PATH: &str = "/opt/habanalabs/engines_fw";
/// Default Intel Gaudi SDK plugins path.
pub const DEFAULT_PLUGINS_LIB_PATH: &str = "/usr/lib/habanatools/habana_plugins";

/// Static mapping from `GAUDIUM_*_ELF` environment variable name to `<file>.o` in the kernels dir.
const STATIC_ELF_VARS: &[(&str, &str)] = &[
    // Core Qwen3 / Qwen3.5 attention & fused add+norm
    ("GAUDIUM_A128COMB_ELF", "attn128_combine_gaudi2.o"),
    ("GAUDIUM_A128COMBX_ELF", "attn128_combinex_gaudi2.o"),
    ("GAUDIUM_ANORM_ELF", "add_norm_gaudi2.o"),
    ("GAUDIUM_ANORMP_ELF", "add_normp_gaudi2.o"),
    ("GAUDIUM_ZERO_ELF", "zero_gaudi2.o"),
    // Default-shape (NKV=4, GQ=6, DH=256) attention kernels
    ("GAUDIUM_APREP_ELF", "attn_prep_gaudi2.o"),
    ("GAUDIUM_APREPF_ELF", "attn_prepf_gaudi2.o"),
    ("GAUDIUM_APREPT_ELF", "attn_prept_gaudi2.o"),
    ("GAUDIUM_ASOFT_ELF", "attn_softmax_gaudi2.o"),
    ("GAUDIUM_ASOFTC_ELF", "attn_softmax_chunk_gaudi2.o"),
    ("GAUDIUM_ASOFTCT_ELF", "attn_softmax_chunkt_gaudi2.o"),
    ("GAUDIUM_ASOFTCX_ELF", "attn_softmax_chunkx_gaudi2.o"),
    ("GAUDIUM_ASOFTCX4_ELF", "attn_softmax_chunkx4_gaudi2.o"),
    ("GAUDIUM_ASOFTCX5_ELF", "attn_softmax_chunkx5_gaudi2.o"),
    ("GAUDIUM_ASOFTF_ELF", "attn_softmax_fresh_gaudi2.o"),
    ("GAUDIUM_ASOFT16_ELF", "attn_softmax16_gaudi2.o"),
    ("GAUDIUM_ASOFT16P8_ELF", "attn_softmax16p8_gaudi2.o"),
    ("GAUDIUM_AVNAT_ELF", "attn_vnat_gaudi2.o"),
    ("GAUDIUM_AQS16_ELF", "attn_qs16_gaudi2.o"),
    ("GAUDIUM_ACOMB_ELF", "attn_combine_gaudi2.o"),
    ("GAUDIUM_ACOMBF_ELF", "attn_combinef_gaudi2.o"),
    ("GAUDIUM_ACOMBX_ELF", "attn_combinex_gaudi2.o"),
    ("GAUDIUM_APOST_ELF", "attn_post_gaudi2.o"),
    ("GAUDIUM_APOSTF_ELF", "attn_post_f32_gaudi2.o"),
    ("GAUDIUM_KVCOMMIT_ELF", "kv_commit_gaudi2.o"),
    ("GAUDIUM_FPOS_ELF", "fpos_gaudi2.o"),
    // FP8 KV attention kernels (-DKV8)
    ("GAUDIUM_APREP8_ELF", "attn_prep8_gaudi2.o"),
    ("GAUDIUM_APREPF8_ELF", "attn_prepf8_gaudi2.o"),
    ("GAUDIUM_APREPT8_ELF", "attn_prept8_gaudi2.o"),
    ("GAUDIUM_ASOFTC8_ELF", "attn_softmax_chunk8_gaudi2.o"),
    ("GAUDIUM_ASOFTCT8_ELF", "attn_softmax_chunkt8_gaudi2.o"),
    ("GAUDIUM_ASOFTCX8_ELF", "attn_softmax_chunkx8_gaudi2.o"),
    ("GAUDIUM_ASOFTF8_ELF", "attn_softmax_fresh8_gaudi2.o"),
    ("GAUDIUM_KVCOMMIT8_ELF", "kv_commit8_gaudi2.o"),
    // On-card Gumbel-max sampler kernels
    ("GAUDIUM_SBMAX_ELF", "samp_bmax_gaudi2.o"),
    ("GAUDIUM_SINIT_ELF", "samp_init_gaudi2.o"),
    ("GAUDIUM_SPART_ELF", "samp_part_gaudi2.o"),
    ("GAUDIUM_SRED_ELF", "samp_red_gaudi2.o"),
    ("GAUDIUM_SGUMBEL_ELF", "samp_gumbel_gaudi2.o"),
    ("GAUDIUM_SFINAL_ELF", "samp_final_gaudi2.o"),
    // Gated DeltaNet (GDN) default-shape (HK=16, REP=3) kernels
    ("GAUDIUM_GDN_ELF", "gdn_f32_gaudi2.o"),
    ("GAUDIUM_GDNS_ELF", "gdn_f32s_gaudi2.o"),
    ("GAUDIUM_GDNT_ELF", "gdn_f32t_gaudi2.o"),
    // replay recurrence of the verify rows (served by the step glue library)
    ("GAUDIUM_GDNR_ELF", "gdn_f32r_gaudi2.o"),
    // slotted row recurrence with its gate scalars as vector loads (served by the step glue library)
    ("GAUDIUM_GDNV_ELF", "gdn_f32v_gaudi2.o"),
    // a block drafter's context as rings of its sliding window (served by the step glue library)
    ("GAUDIUM_DSCTXR_ELF", "ds_ctxr_gaudi2.o"),
    ("GAUDIUM_DSSOFTR_ELF", "ds_softr_gaudi2.o"),
    ("GAUDIUM_PREP_ELF", "gdn_prep_gaudi2.o"),
    ("GAUDIUM_PREPS_ELF", "gdn_preps_gaudi2.o"),
    ("GAUDIUM_PREPD_ELF", "gdn_prepd_gaudi2.o"),
    ("GAUDIUM_PREPSD_ELF", "gdn_prepsd_gaudi2.o"),
    ("GAUDIUM_PREPT_ELF", "gdn_prept_gaudi2.o"),
    ("GAUDIUM_PREP2_ELF", "gdn_prep2_gaudi2.o"),
    ("GAUDIUM_PREPS2_ELF", "gdn_prep2s_gaudi2.o"),
    ("GAUDIUM_PREPA_ELF", "gdn_prepa_gaudi2.o"),
    ("GAUDIUM_PREPB_ELF", "gdn_prepb_gaudi2.o"),
    ("GAUDIUM_GATE_ELF", "gdn_gate_gaudi2.o"),
    ("GAUDIUM_GATEC_ELF", "gdn_gatec_gaudi2.o"),
    ("GAUDIUM_POST_ELF", "gdn_post_gaudi2.o"),
    ("GAUDIUM_POSTD_ELF", "gdn_postd_gaudi2.o"),
    ("GAUDIUM_POST2_ELF", "gdn_post2_gaudi2.o"),
    ("GAUDIUM_POSTA_ELF", "gdn_posta_gaudi2.o"),
    ("GAUDIUM_POSTB_ELF", "gdn_postb_gaudi2.o"),
    ("GAUDIUM_CHUNK_ELF", "gdn_chunk_gaudi2.o"),
    ("GAUDIUM_CHUNKA_ELF", "gdn_chunka_gaudi2.o"),
    ("GAUDIUM_SOLVE_ELF", "gdn_solve_gaudi2.o"),
    ("GAUDIUM_RELAY_ELF", "gdn_relayout_gaudi2.o"),
    ("GAUDIUM_RELAYS_ELF", "gdn_relayouts_gaudi2.o"),
    ("GAUDIUM_VREC_ELF", "gdn_vrec_gaudi2.o"),
    ("GAUDIUM_VCOMMIT_ELF", "gdn_vcommit_gaudi2.o"),
    // Speculative acceptance / tree / MTP / DFlash2 / DSpine / cert / PLD
    ("GAUDIUM_ACCEPT_ELF", "accept_e_gaudi2.o"),
    ("GAUDIUM_ACCEPTG_ELF", "accept_eg_gaudi2.o"),
    ("GAUDIUM_ACCEPTT_ELF", "accept_t_gaudi2.o"),
    ("GAUDIUM_ACCEPTTR_ELF", "accept_tree_gaudi2.o"),
    ("GAUDIUM_ACCEPTTRR_ELF", "accept_treer_gaudi2.o"),
    ("GAUDIUM_SEL_ELF", "sel_gaudi2.o"),
    ("GAUDIUM_NEXTIDS_ELF", "next_ids_gaudi2.o"),
    ("GAUDIUM_TREEIDS_ELF", "tree_ids_gaudi2.o"),
    ("GAUDIUM_TREECAND_ELF", "tree_cand_gaudi2.o"),
    ("GAUDIUM_MTOPK1_ELF", "mtp_topk1_gaudi2.o"),
    ("GAUDIUM_MTOPK2_ELF", "mtp_topk2_gaudi2.o"),
    ("GAUDIUM_TREEPLAN_ELF", "tree_plan_gaudi2.o"),
    ("GAUDIUM_PLDSCAN_ELF", "pld_scan_gaudi2.o"),
    ("GAUDIUM_PLDTAIL_ELF", "pld_tail_gaudi2.o"),
    ("GAUDIUM_PLDTAILG_ELF", "pld_tailg_gaudi2.o"),
    ("GAUDIUM_CERTHQ_ELF", "cert_hq_gaudi2.o"),
    ("GAUDIUM_CERTMARK_ELF", "cert_mark_gaudi2.o"),
    ("GAUDIUM_CERTCOMPACT_ELF", "cert_compact_gaudi2.o"),
    ("GAUDIUM_CERTFINAL_ELF", "cert_final_gaudi2.o"),
    ("GAUDIUM_DFCONV_ELF", "df_conv_gaudi2.o"),
    ("GAUDIUM_DFAPREP_ELF", "df_attn_prep_gaudi2.o"),
    ("GAUDIUM_DFCTX_ELF", "df_ctx_gaudi2.o"),
    ("GAUDIUM_DFSOFT_ELF", "df_softmax_gaudi2.o"),
    ("GAUDIUM_DFPOST_ELF", "df_post_gaudi2.o"),
    ("GAUDIUM_DFTOPK1_ELF", "df_topk1_gaudi2.o"),
    ("GAUDIUM_DFTOPK2_ELF", "df_topk2_gaudi2.o"),
    ("GAUDIUM_DFSCORE_ELF", "df_score_gaudi2.o"),
    ("GAUDIUM_DFWALK_ELF", "df_walk_gaudi2.o"),
    ("GAUDIUM_DFIDS_ELF", "df_ids_gaudi2.o"),
    ("GAUDIUM_DFNEXT_ELF", "df_next_gaudi2.o"),
    ("GAUDIUM_DSSOFT_ELF", "ds_softmax_gaudi2.o"),
    ("GAUDIUM_GADFSOFTR_ELF", "ga_dfsoftr_gaudi2.o"),
    ("GAUDIUM_GADFCTXR_ELF", "ga_dfctxr_gaudi2.o"),
    ("GAUDIUM_DSREAD_ELF", "ds_read_gaudi2.o"),
    ("GAUDIUM_DSCODES_ELF", "ds_codes_gaudi2.o"),
    ("GAUDIUM_DSMSG_ELF", "ds_msg_gaudi2.o"),
    ("GAUDIUM_DSFINISH_ELF", "ds_finish_gaudi2.o"),
    ("GAUDIUM_DSBASE_ELF", "ds_base_gaudi2.o"),
    ("GAUDIUM_DSSCORE_ELF", "ds_score_gaudi2.o"),
    ("GAUDIUM_DSCWALK_ELF", "ds_cwalk_gaudi2.o"),
    // MoE kernels (dense-masked, W4L/W8 TPC, MME prefill, hybrid miss gate)
    ("GAUDIUM_MOEROUTE_ELF", "moe_route_gaudi2.o"),
    ("GAUDIUM_MOEROUTESG_ELF", "moe_routesg_gaudi2.o"),
    ("GAUDIUM_MOEACT_ELF", "moe_act_gaudi2.o"),
    ("GAUDIUM_W4GATE_ELF", "moe_w4_gate_gaudi2.o"),
    ("GAUDIUM_W4GATE0_ELF", "moe_w4_gate0_gaudi2.o"),
    ("GAUDIUM_W4PLAN_ELF", "moe_w4_plan_gaudi2.o"),
    ("GAUDIUM_W4Q8_ELF", "moe_w4_q8_gaudi2.o"),
    ("GAUDIUM_W4GU_ELF", "moe_w4_gu_gaudi2.o"),
    ("GAUDIUM_W4GU8_ELF", "moe_w4_gu8_gaudi2.o"),
    ("GAUDIUM_W4DN_ELF", "moe_w4_dn_gaudi2.o"),
    ("GAUDIUM_W4DN8_ELF", "moe_w4_dn8_gaudi2.o"),
    ("GAUDIUM_W4COMB_ELF", "moe_w4_comb_gaudi2.o"),
    ("GAUDIUM_W4COMBA_ELF", "moe_w4_comba_gaudi2.o"),
    ("GAUDIUM_MDQ13_ELF", "moe_mdq13_gaudi2.o"),
    ("GAUDIUM_MDQ13W8_ELF", "moe_mdq13w8_gaudi2.o"),
    ("GAUDIUM_MDQ2_ELF", "moe_mdq2_gaudi2.o"),
    ("GAUDIUM_MDQ2W8_ELF", "moe_mdq2w8_gaudi2.o"),
    ("GAUDIUM_MSWIGLU_ELF", "moe_mswiglu_gaudi2.o"),
    ("GAUDIUM_MGATHER_ELF", "moe_mgather_gaudi2.o"),
    ("GAUDIUM_MCOMB_ELF", "moe_mcomb_gaudi2.o"),
    // QSA & hyper-connection / PLE
    ("GAUDIUM_HCNORM_ELF", "hc_norm_gaudi2.o"),
    ("GAUDIUM_HCMIX_ELF", "hc_mix_gaudi2.o"),
    ("GAUDIUM_HCCOMB_ELF", "hc_combine_gaudi2.o"),
    ("GAUDIUM_PLEGATE_ELF", "ple_gate_gaudi2.o"),
    ("GAUDIUM_PLECONV_ELF", "ple_conv_gaudi2.o"),
    ("GAUDIUM_PLESYNC_ELF", "ple_sync_gaudi2.o"),
    ("GAUDIUM_QSAPREP_ELF", "qsa_prep_gaudi2.o"),
    ("GAUDIUM_QSAPREPF_ELF", "qsa_prepf_gaudi2.o"),
    ("GAUDIUM_QSACOMMIT_ELF", "qsa_commit_gaudi2.o"),
    ("GAUDIUM_QSASCORE_ELF", "qsa_score_gaudi2.o"),
    ("GAUDIUM_QSASEL_ELF", "qsa_select_gaudi2.o"),
    ("GAUDIUM_QSASELB_ELF", "qsa_selectb_gaudi2.o"),
    ("GAUDIUM_QSAGATHER_ELF", "qsa_gather_gaudi2.o"),
    ("GAUDIUM_QSASPSOFT_ELF", "qsa_sp_softmax_gaudi2.o"),
    ("GAUDIUM_QSASPOUT_ELF", "qsa_sp_out_gaudi2.o"),
    ("GAUDIUM_QSASPOUTD_ELF", "qsa_sp_outd_gaudi2.o"),
    ("GAUDIUM_ASOFTCQ_ELF", "attn_softmax_chunkq_gaudi2.o"),
    ("GAUDIUM_ASOFTCXQ_ELF", "attn_softmax_chunkxq_gaudi2.o"),
    ("GAUDIUM_ASOFTCX4Q_ELF", "attn_softmax_chunkx4q_gaudi2.o"),
    ("GAUDIUM_ASOFTC8Q_ELF", "attn_softmax_chunk8q_gaudi2.o"),
    ("GAUDIUM_ASOFTCX8Q_ELF", "attn_softmax_chunkx8q_gaudi2.o"),
    // Inkling 2 kernels (libgaudium_ik.so)
    ("GAUDIUM_IKMIX_ELF", "ik_mix_gaudi2.o"),
    ("GAUDIUM_IKMIXF_ELF", "ik_mixf_gaudi2.o"),
    ("GAUDIUM_IKGIN_ELF", "ik_gin_gaudi2.o"),
    ("GAUDIUM_IKMGATE_ELF", "ik_mgate_gaudi2.o"),
    ("GAUDIUM_IKMGATEF_ELF", "ik_mgatef_gaudi2.o"),
    ("GAUDIUM_IKNORM_ELF", "ik_norm_gaudi2.o"),
    ("GAUDIUM_IKSAVE_ELF", "ik_save_gaudi2.o"),
    ("GAUDIUM_IKCONVA_ELF", "ik_conva_gaudi2.o"),
    ("GAUDIUM_IKCONVM_ELF", "ik_convm_gaudi2.o"),
    ("GAUDIUM_IKQPREP_ELF", "ik_qprep_gaudi2.o"),
    ("GAUDIUM_IKKVPREP_ELF", "ik_kvprep_gaudi2.o"),
    ("GAUDIUM_IKSOFT_ELF", "ik_softmax_gaudi2.o"),
    ("GAUDIUM_IKCOMB_ELF", "ik_combine_gaudi2.o"),
    ("GAUDIUM_IKROUTE_ELF", "ik_route_gaudi2.o"),
    ("GAUDIUM_IKACT_ELF", "ik_moe_act_gaudi2.o"),
    ("GAUDIUM_IKARGMAX_ELF", "ik_argmax_gaudi2.o"),
    // Inkling 2 block drafter kernels (libgaudium_ik.so)
    ("GAUDIUM_IKDMIX_ELF", "ik_dmix_gaudi2.o"),
    ("GAUDIUM_IKDKV_ELF", "ik_dkv_gaudi2.o"),
    ("GAUDIUM_IKDQPREP_ELF", "ik_dqprep_gaudi2.o"),
    ("GAUDIUM_IKDSOFT_ELF", "ik_dsoft_gaudi2.o"),
    ("GAUDIUM_IKDCOMB_ELF", "ik_dcomb_gaudi2.o"),
    ("GAUDIUM_IKDSWIGLU_ELF", "ik_dswiglu_gaudi2.o"),
    ("GAUDIUM_IKMSTEP_ELF", "ik_mstep_gaudi2.o"),
    ("GAUDIUM_IKMARGMAX_ELF", "ik_margmax_gaudi2.o"),
    // Inkling 2 routed experts & softcap kernels (libgaudium_ik.so)
    ("GAUDIUM_IKROUTER_ELF", "ik_route_r_gaudi2.o"),
    ("GAUDIUM_IKPLAN_ELF", "ik_w4_plan_gaudi2.o"),
    ("GAUDIUM_IKW4Q8_ELF", "ik_w4_q8_gaudi2.o"),
    ("GAUDIUM_IKW8Q8_ELF", "ik_w8_q8_gaudi2.o"),
    ("GAUDIUM_IKW4GU_ELF", "ik_w4_gu_gaudi2.o"),
    ("GAUDIUM_IKW4DN_ELF", "ik_w4_dn_gaudi2.o"),
    ("GAUDIUM_IKW8GU_ELF", "ik_w8_gu_gaudi2.o"),
    ("GAUDIUM_IKW8DN_ELF", "ik_w8_dn_gaudi2.o"),
    ("GAUDIUM_IKW4COMB_ELF", "ik_w4_comb_gaudi2.o"),
    ("GAUDIUM_IKSOFTCAP_ELF", "ik_softcap_gaudi2.o"),
    ("GAUDIUM_IKMCAP_ELF", "ik_mcap_gaudi2.o"),
    // Gemma 4 kernels (libgaudium_ga.so)
    ("GAUDIUM_GAPREP_ELF", "ga_prep_gaudi2.o"),
    ("GAUDIUM_GAPREPK_ELF", "ga_prepk_gaudi2.o"),
    ("GAUDIUM_GAQPREP_ELF", "ga_qprep_gaudi2.o"),
    ("GAUDIUM_GAPREPV_ELF", "ga_prepv_gaudi2.o"),
    ("GAUDIUM_GAKV_ELF", "ga_kv_gaudi2.o"),
    ("GAUDIUM_GASOFTR_ELF", "ga_softmax_ring_gaudi2.o"),
    ("GAUDIUM_GASOFTRX_ELF", "ga_softmax_ringx_gaudi2.o"),
    ("GAUDIUM_GACOMB4_ELF", "ga_combine4_gaudi2.o"),
    ("GAUDIUM_GACOMB8_ELF", "ga_combine8_gaudi2.o"),
    ("GAUDIUM_GAPOST_ELF", "ga_post_gaudi2.o"),
    ("GAUDIUM_GAGELU_ELF", "ga_gelu_gaudi2.o"),
    ("GAUDIUM_GASOFTCAP_ELF", "ga_softcap_gaudi2.o"),
    ("GAUDIUM_GACOPY_ELF", "ga_copy_gaudi2.o"),
    ("GAUDIUM_GACLAMP_ELF", "ga_clamp_gaudi2.o"),
    ("GAUDIUM_GARELATTN_ELF", "ga_relattn_gaudi2.o"),
    ("GAUDIUM_GAACT_ELF", "ga_act_gaudi2.o"),
    ("GAUDIUM_GADWCONV_ELF", "ga_dwconv_gaudi2.o"),
    ("GAUDIUM_GALNRELU_ELF", "ga_lnrelu_gaudi2.o"),
    ("GAUDIUM_GAARGMAX_ELF", "ga_argmax_gaudi2.o"),
    ("GAUDIUM_GALNORM_ELF", "ga_lnorm_gaudi2.o"),
    ("GAUDIUM_GARMSNORM_ELF", "ga_rmsnorm_gaudi2.o"),
    // Misc utility / test kernels
    ("GAUDIUM_RMW_ELF", "rmw_add_gaudi2.o"),
    ("GAUDIUM_VTEST_ELF", "vtest_f32_gaudi2.o"),
    ("GAUDIUM_VTESTBF_ELF", "vtest_bf16_gaudi2.o"),
];

/// Returns the complete `(ENV_VAR, filename)` list including all shape-specialized variants
/// (`k16r2`, `k16r3s`, `g2q8`, `g2q12`, `g8q4`).
pub fn all_elf_mappings() -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = STATIC_ELF_VARS
        .iter()
        .map(|&(k, v)| (k.to_string(), v.to_string()))
        .collect();

    // 128-channel attention head variants (plain Qwen3: Qwen3-8B = g8q4)
    for shape in ["g8q4"] {
        let tag = shape.to_ascii_uppercase();
        for &(var_pfx, file_pfx) in &[
            ("A128PREP", "attn128_prep"),
            ("A128PREPT", "attn128_prept"),
            ("A128POST", "attn128_post"),
        ] {
            out.push((
                format!("GAUDIUM_{var_pfx}_{tag}_ELF"),
                format!("{file_pfx}_{shape}_gaudi2.o"),
            ));
        }
    }

    // 256-channel attention head variants (Qwen3.6-35B-A3B = g2q8, etc.)
    for shape in ["g2q8", "g2q12"] {
        let tag = shape.to_ascii_uppercase();
        for &(var_pfx, file_pfx) in &[
            ("APREP", "attn_prep"),
            ("APREPF", "attn_prepf"),
            ("APREPT", "attn_prept"),
            ("APREP8", "attn_prep8"),
            ("APREPF8", "attn_prepf8"),
            ("APREPT8", "attn_prept8"),
            ("APOST", "attn_post"),
            ("APOSTF", "attn_post_f32"),
            ("ASOFT", "attn_softmax"),
        ] {
            out.push((
                format!("GAUDIUM_{var_pfx}_{tag}_ELF"),
                format!("{file_pfx}_{shape}_gaudi2.o"),
            ));
        }
        let g_only = tag.split('Q').next().unwrap_or(&tag);
        out.push((
            format!("GAUDIUM_KVCOMMIT_{g_only}_ELF"),
            format!("kv_commit_{shape}_gaudi2.o"),
        ));
        out.push((
            format!("GAUDIUM_KVCOMMIT8_{g_only}_ELF"),
            format!("kv_commit8_{shape}_gaudi2.o"),
        ));
    }

    // GDN shape variants (k16r2 for 32-value-head MoE models, k16r3s for sigmoid gate)
    for shape in ["k16r2"] {
        let tag = shape.to_ascii_uppercase();
        for &(var_pfx, file_pfx) in &[
            ("PREP", "gdn_prep"),
            ("PREPS", "gdn_preps"),
            ("PREPD", "gdn_prepd"),
            ("PREPSD", "gdn_prepsd"),
            ("PREPT", "gdn_prept"),
            ("PREP2", "gdn_prep2"),
            ("PREPS2", "gdn_prep2s"),
            ("PREPA", "gdn_prepa"),
            ("PREPB", "gdn_prepb"),
            ("POST", "gdn_post"),
            ("POSTD", "gdn_postd"),
            ("POST2", "gdn_post2"),
            ("POSTB", "gdn_postb"),
        ] {
            out.push((
                format!("GAUDIUM_{var_pfx}_{tag}_ELF"),
                format!("{file_pfx}_{shape}_gaudi2.o"),
            ));
        }
    }
    for shape in ["k16r3s"] {
        let tag = shape.to_ascii_uppercase();
        for &(var_pfx, file_pfx) in &[
            ("POST", "gdn_post"),
            ("POSTD", "gdn_postd"),
            ("POST2", "gdn_post2"),
            ("POSTB", "gdn_postb"),
        ] {
            out.push((
                format!("GAUDIUM_{var_pfx}_{tag}_ELF"),
                format!("{file_pfx}_{shape}_gaudi2.o"),
            ));
        }
    }

    out
}

/// Glue plugin of the step kernels added after the first kernel set (`kernels/glue/st_glue.cpp`).
pub const STEP_GLUE: &str = "libgaudium_st.so";

/// Glue plugin of the Inkling 2 kernels (`kernels/glue/ik_glue.cpp`).
pub const INKLING2_GLUE: &str = "libgaudium_ik.so";

/// Resolved kernel & glue environment for a Gaudium run.
#[derive(Debug, Clone)]
pub struct KernelEnv {
    pub kernels_dir: PathBuf,
    pub gc_kernel_path: String,
    pub present_elfs: BTreeMap<String, PathBuf>,
}

impl KernelEnv {
    /// Configures process environment variables (`GC_KERNEL_PATH`, `HABANA_*`, `GAUDIUM_*_ELF`)
    /// from `kernels_dir`. MUST be called **before** `SynapseApi::load()` / `synInitialize()`
    /// while the process is single-threaded.
    ///
    /// Existing explicitly-set `GAUDIUM_*_ELF` variables are preserved so A/B overrides still work.
    pub fn configure_from_dir(kernels_dir: &Path) -> Result<Self, String> {
        if !kernels_dir.is_dir() {
            return Err(format!(
                "kernels directory does not exist: {}",
                kernels_dir.display()
            ));
        }

        let tpc_glue = kernels_dir.join("libgaudium_tpc.so");
        let ga_glue = kernels_dir.join("libgaudium_ga.so");
        if !tpc_glue.is_file() {
            return Err(format!(
                "missing Synapse GC glue library: {} (build with `make -C kernels qwen3` or `make -C kernels`)",
                tpc_glue.display()
            ));
        }

        let mut gc_parts: Vec<String> = Vec::new();
        gc_parts.push(tpc_glue.to_string_lossy().into_owned());
        if ga_glue.is_file() {
            gc_parts.push(ga_glue.to_string_lossy().into_owned());
        }
        let st_glue = kernels_dir.join(STEP_GLUE);
        if st_glue.is_file() {
            gc_parts.push(st_glue.to_string_lossy().into_owned());
        }
        let ik_glue = kernels_dir.join(INKLING2_GLUE);
        if ik_glue.is_file() {
            gc_parts.push(ik_glue.to_string_lossy().into_owned());
        }
        if let Ok(existing) = std::env::var("GC_KERNEL_PATH") {
            for part in existing.split(':').filter(|s| !s.is_empty()) {
                if !gc_parts.iter().any(|p| p == part) {
                    gc_parts.push(part.to_string());
                }
            }
        }
        if !gc_parts.iter().any(|p| p == DEFAULT_SDK_TPC_LIB) {
            gc_parts.push(DEFAULT_SDK_TPC_LIB.to_string());
        }
        let gc_kernel_path = gc_parts.join(":");

        // SAFETY: Called during single-threaded CLI/engine initialization before Synapse spawns threads.
        unsafe {
            std::env::set_var("GC_KERNEL_PATH", &gc_kernel_path);
            if std::env::var_os("HABANA_SCAL_BIN_PATH").is_none()
                && Path::new(DEFAULT_SCAL_BIN_PATH).is_dir()
            {
                std::env::set_var("HABANA_SCAL_BIN_PATH", DEFAULT_SCAL_BIN_PATH);
            }
            if std::env::var_os("HABANA_PLUGINS_LIB_PATH").is_none()
                && Path::new(DEFAULT_PLUGINS_LIB_PATH).is_dir()
            {
                std::env::set_var("HABANA_PLUGINS_LIB_PATH", DEFAULT_PLUGINS_LIB_PATH);
            }
        }

        let mut present_elfs = BTreeMap::new();
        for (env_var, filename) in all_elf_mappings() {
            if let Ok(existing) = std::env::var(&env_var) {
                if !existing.is_empty() {
                    present_elfs.insert(env_var, PathBuf::from(existing));
                    continue;
                }
            }
            let p = kernels_dir.join(&filename);
            if p.is_file() {
                unsafe {
                    std::env::set_var(&env_var, &p);
                }
                present_elfs.insert(env_var, p);
            }
        }

        Ok(Self {
            kernels_dir: kernels_dir.to_path_buf(),
            gc_kernel_path,
            present_elfs,
        })
    }

    /// Hash of the step glue library, for the fingerprints of the recipes that use its kernels
    /// (`None`: not in the kernels directory). It is not part of [`Self::fingerprint_hashes`], so
    /// recipes without those kernels keep their keys whether or not the library is installed.
    pub fn step_glue_hash(&self) -> Option<Result<(String, String), String>> {
        let p = self.kernels_dir.join(STEP_GLUE);
        p.is_file().then(|| {
            std::fs::read(&p)
                .map(|raw| (STEP_GLUE.to_string(), sha256_hex(&raw)))
                .map_err(|e| format!("read {}: {e}", p.display()))
        })
    }

    /// Hash of one glue library of the kernels directory, for the fingerprints of the recipes that
    /// use its kernels. A family's own library is not part of [`Self::fingerprint_hashes`], so the
    /// other families' recipes keep their keys when it changes.
    pub fn glue_hash(&self, name: &str) -> Result<(String, String), String> {
        let p = self.kernels_dir.join(name);
        std::fs::read(&p)
            .map(|raw| (name.to_string(), sha256_hex(&raw)))
            .map_err(|e| format!("read {}: {e}", p.display()))
    }

    /// Computes normalized SHA-256 hashes for the glue `.so`(s) and every requested `GAUDIUM_*_ELF`
    /// variable so recipe cache fingerprints automatically invalidate if any kernel changes.
    pub fn fingerprint_hashes(
        &self,
        required_env_vars: &[&str],
    ) -> Result<Vec<(String, String)>, String> {
        let mut out = Vec::new();
        for glue in ["libgaudium_tpc.so", "libgaudium_ga.so"] {
            let p = self.kernels_dir.join(glue);
            if p.is_file() {
                let raw = std::fs::read(&p).map_err(|e| format!("read {}: {e}", p.display()))?;
                out.push((glue.to_string(), sha256_hex(&raw)));
            }
        }
        for &var in required_env_vars {
            let p = self.present_elfs.get(var).ok_or_else(|| {
                format!(
                    "required kernel ELF {var} not found in {}",
                    self.kernels_dir.display()
                )
            })?;
            let raw = std::fs::read(p).map_err(|e| format!("read {}: {e}", p.display()))?;
            let h = hash_tpc_elf_normalized(&raw);
            out.push((var.to_string(), h));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elf_mappings_include_qwen3_g8q4_and_qwen35_and_gemma4() {
        let m: BTreeMap<String, String> = all_elf_mappings().into_iter().collect();
        assert_eq!(
            m.get("GAUDIUM_A128PREP_G8Q4_ELF").map(String::as_str),
            Some("attn128_prep_g8q4_gaudi2.o")
        );
        assert_eq!(
            m.get("GAUDIUM_A128POST_G8Q4_ELF").map(String::as_str),
            Some("attn128_post_g8q4_gaudi2.o")
        );
        assert_eq!(
            m.get("GAUDIUM_A128COMB_ELF").map(String::as_str),
            Some("attn128_combine_gaudi2.o")
        );
        assert_eq!(
            m.get("GAUDIUM_ANORMP_ELF").map(String::as_str),
            Some("add_normp_gaudi2.o")
        );
        assert_eq!(
            m.get("GAUDIUM_ASOFTC_ELF").map(String::as_str),
            Some("attn_softmax_chunk_gaudi2.o")
        );
        assert_eq!(
            m.get("GAUDIUM_GDN_ELF").map(String::as_str),
            Some("gdn_f32_gaudi2.o")
        );
        assert_eq!(
            m.get("GAUDIUM_GAPREP_ELF").map(String::as_str),
            Some("ga_prep_gaudi2.o")
        );
    }
}
