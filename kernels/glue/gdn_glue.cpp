// Gaudium TPC kernel library (GC kernel-DB plugin). GUIDs:
//   gaudium_gdn_f32    gated delta rule recurrence, T tokens, state blob {64,256,HV}      ELF: GAUDIUM_GDN_ELF
//   gaudium_gdn_prep   causal conv + silu + l2norm + GQA repeat + relayout                ELF: GAUDIUM_PREP_ELF
//   gaudium_gdn_gate   beta / exp(g) per token-head                                       ELF: GAUDIUM_GATE_ELF
//   gaudium_gdn_post   gated RMSNorm per head -> bf16 row                                  ELF: GAUDIUM_POST_ELF
//   gaudium_attn_prep / gaudium_attn_softmax / gaudium_attn_post   full-attention front end, softmax, output gate (P14 path)
//   gaudium_attn_softmax_chunk  chunked two-pass masked softmax + per-chunk (m, s)     ELF: GAUDIUM_ASOFTC_ELF   (idea F, C = 4096);
//                               node param 4 = 1 selects the -DFASTEXP build              ELF: GAUDIUM_ASOFTCX_ELF  (prefill recipes)
//   gaudium_attn_combine        merge chunk partials: sum w_ch ao_ch / sum w_ch s_ch    ELF: GAUDIUM_ACOMB_ELF
//   gaudium_attn_post_f32       output gate reading the combine's f32 output            ELF: GAUDIUM_APOSTF_ELF
//   gaudium_attn_prep with 10 inputs (kf_in, vf_in) and 4 outputs (qb, qb_f, kf_out, vf_out) is the FRESH-WINDOW variant of the
//   engine's chained MTP passes (writes a 256-row window instead of the big cache)        ELF: GAUDIUM_APREPF_ELF
//   gaudium_attn_softmax_fresh  table-masked softmax over the fresh window                ELF: GAUDIUM_ASOFTF_ELF
//   gaudium_attn_combine with 4 inputs merges the fresh window as one more chunk          ELF: GAUDIUM_ACOMBF_ELF
//   gaudium_kv_commit           accepted window rows -> big chunked cache (step start)    ELF: GAUDIUM_KVCOMMIT_ELF
//   feature/fp8-kv (GAUDIUM_KV=fp8|fp8e5): the FP8 KV builds (-DKV8) are chosen by the tensors' fp8 dtypes (fp8_143 or fp8_152; the
//   format is a scalar of the kernel) and take one more input, kv8 f32 {32} (per-kv-group q / K / V / P multipliers, [16] the score scale,
//   [17] the stats scale), after the others; the QK / PV GEMMs are plain batch_gemm on fp8 operands (raw sums: the softmax scales them):
//     gaudium_attn_prep 9 inputs (plain) / 11 (fresh: kf_in, vf_in fp8; tree: depth, fix i32), fp8 qb / caches / windows
//                                                                                       ELF: GAUDIUM_APREP8_ELF / _APREPF8_ / _APREPT8_
//     gaudium_attn_softmax_chunk with an fp8 P: 3 inputs (+ kv8) / 4 (+ anc, kv8: tree) / FASTEXP node param   ELF: GAUDIUM_ASOFTC8_ELF / _ASOFTCT8_ / _ASOFTCX8_
//     gaudium_attn_softmax_fresh with an fp8 P: 4 inputs (+ kv8)                         ELF: GAUDIUM_ASOFTF8_ELF
//     gaudium_kv_commit with fp8 window / cache (raw row copy)                           ELF: GAUDIUM_KVCOMMIT8_ELF
//   gaudium_pld_scan   prompt-lookup n-gram search over the device token history ELF: GAUDIUM_PLDSCAN_ELF
//   gaudium_pld_tail   longest/most recent match -> L tail ids after the MTP chain, history advance; vision sentinel cut  ELF: GAUDIUM_PLDTAIL_ELF
//   gaudium_fpos                window slot -> position tables (catch-up / draft views)   ELF: GAUDIUM_FPOS_ELF
//   gaudium_pld_scan   prompt-lookup n-gram search over the device token history (NM members) ELF: GAUDIUM_PLDSCAN_ELF
//   gaudium_pld_tail   longest/most recent match -> L tail ids after the MTP chain; history advance ELF: GAUDIUM_PLDTAIL_ELF (-DGATE: _PLDTAILG_)
//   gaudium_accept_t / gaudium_df_*   DFlash2 drafter + tree acceptance (docs/notes/dflash2-spec.md)   ELF: GAUDIUM_ACCEPTT_ELF, GAUDIUM_DF*_ELF
//   gaudium_ds_*      DSpine drafter (docs/notes/dspine-spec.md): causal block softmax, code bridge (read / codes /
//                     msg / finish), serial-table tail (base / score)                                    ELF: GAUDIUM_DS{SOFT,READ,CODES,MSG,FINISH,BASE,SCORE}_ELF
//   gaudium_attn_prep / gaudium_attn_softmax_chunk tree variants (depth/fix, anc inputs)   ELF: GAUDIUM_APREPT_ELF / GAUDIUM_ASOFTCT_ELF
//   gaudium_cert_hq / _mark / _compact / _final   certified screening readout (docs/notes/engine-mtp-certified.md)  ELF: GAUDIUM_CERT{HQ,MARK,COMPACT,FINAL}_ELF
//   gaudium_vtest_f32  ISA micro-test                                                     ELF: GAUDIUM_VTEST_ELF
//   gaudium_gdn_f32 with 7 inputs (S_in {64,256,HV,NS} + slot) and gaudium_gdn_prep with 5 inputs (cs_in {10240,3,NS} + slot)
//   are the slot-indexed engine variants                                                 ELF: GAUDIUM_GDNS_ELF / GAUDIUM_PREPS_ELF
//   gaudium_accept_e   device-side speculative acceptance (a, cursor_out, emitted)        ELF: GAUDIUM_ACCEPT_ELF
//   gaudium_sel        out{M} = tok{1,N}[0, idx + j]                                      ELF: GAUDIUM_SEL_ELF
//   gaudium_next_ids   next verify ids from tokens, a, tokens_mtp, recursive drafts        ELF: GAUDIUM_NEXTIDS_ELF
//   MME prefill path (Idea A, gdn_chunk.hpp): gaudium_gdn_gate with ONE output (gb {128,Tp}) ELF: GAUDIUM_GATEC_ELF;
//   gaudium_gdn_chunk (coefficients per head-chunk)                                       ELF: GAUDIUM_CHUNK_ELF;
//   gaudium_gdn_relayout (state blob <-> matrix; 2 inputs = slotted)                       ELF: GAUDIUM_RELAY_ELF / GAUDIUM_RELAYS_ELF;
//   gaudium_gdn_post takes a second scalar (flags; bit5 = read the chunked X tensor)
//   flags bit6 on gaudium_gdn_prep / gaudium_gdn_post = the row-parallel, LUT-phase-batched prefill builds
//   (gdn_prep2_gaudi2.c, gdn_post2_gaudi2.c; index space {blocks|heads, T/64})   ELF: GAUDIUM_PREP2_ELF / GAUDIUM_PREPS2_ELF / GAUDIUM_POST2_ELF
//   gaudium_samp_{bmax,init,part,red,gumbel,final}  on-card sampler (samp_gaudi2.c, CPU twin samp_ref.hpp)
//                                                   ELF: GAUDIUM_SBMAX_ELF / SINIT / SPART / SRED / SGUMBEL / SFINAL
// feature/mtp-tree (draft-tree verify with the MTP drafter):
//   gaudium_gdn_f32 with 8 inputs (+ parent) / gaudium_gdn_prep with 6 inputs (+ parent): tree rows (dflash2 -DTREE)  ELF: GAUDIUM_GDNT_ELF / GAUDIUM_PREPT_ELF
//   gaudium_attn_prep with 10 inputs whose 9th is int32 (+ depth, fix), 3 outputs: tree slots / positions / path moves  ELF: GAUDIUM_APREPT_ELF
//   gaudium_attn_softmax_chunk with 3 inputs (+ anc): ancestor-masked tree rows                                         ELF: GAUDIUM_ASOFTCT_ELF
//   gaudium_accept_tree  longest accepted root path of any tree: a, cursor', emitted, ncommit, fix, d, catch-up ids/rows  ELF: GAUDIUM_ACCEPTTR_ELF
//   gaudium_tree_ids     verify ids of a static shape from the candidate table                                          ELF: GAUDIUM_TREEIDS_ELF
//   gaudium_tree_cand    candidate table for the next tree from the MTP argmaxes + top-k                                ELF: GAUDIUM_TREECAND_ELF
//   gaudium_mtp_topk1/2  top-KTOP (4) per row of bf16 logits + their softmax probabilities, two stages                  ELF: GAUDIUM_MTOPK1_ELF / GAUDIUM_MTOPK2_ELF
//   gaudium_tree_plan    device-planned caterpillar tree (N rows) from the drafter's candidate probabilities             ELF: GAUDIUM_TREEPLAN_ELF
//   gaudium_accept_tree with 6 inputs (+ rank): also reports the planned spine length and the accepted leaf's rank       ELF: GAUDIUM_ACCEPTTRR_ELF
// feature/flash-next-qsa (Qwen3.8-Flash-Next sparse attention, docs/notes/engine-qsa.md):
//   gaudium_qsa_prep     indexer q norm/rope, raw keys -> ikcache, block keys -> ibcache (7 inputs: the FRESH window build)  ELF: GAUDIUM_QSAPREP_ELF / _QSAPREPF_
//   gaudium_qsa_commit   MTP step start: accepted raw-key window rows -> ikcache + block keys                              ELF: GAUDIUM_QSACOMMIT_ELF
//   gaudium_qsa_score    per-block relu-sum of the 4 head scores                                                           ELF: GAUDIUM_QSASCORE_ELF
//   gaudium_qsa_select   exact top-K blocks per row -> mask words                                                         ELF: GAUDIUM_QSASEL_ELF
//                        with a 2nd output qbm i32 {64, NBW, T}: + the selected-block bitmap (gathered sparse core)        ELF: GAUDIUM_QSASELB_ELF
//   gaudium_qsa_gather   per-row gather of the attended blocks + tail into kg / vg slots, q -> {256, GQ, NKV*T}       ELF: GAUDIUM_QSAGATHER_ELF
//   gaudium_qsa_sp_softmax  softmax + stats over the gathered slots                                                  ELF: GAUDIUM_QSASPSOFT_ELF
//   gaudium_qsa_sp_out   1/s normalisation into the combine layout (+ ao_d: the dense first-chunk rows copied)         ELF: GAUDIUM_QSASPOUT_ELF / _QSASPOUTD_
//   (docs/notes/flash-next-qsa-sparse.md; attn_prep also accepts a one-chunk qb {256, GQ T, NKV} beside a multi-chunk cache there)
//   gaudium_attn_softmax_chunk with a 3-D int32 input after the cursor: the -DQSA builds (+ node param 6 = qfull)
//                        ELF: GAUDIUM_ASOFTCQ_ELF / _ASOFTCXQ_ / _ASOFTCX4Q_ (bf16 P), GAUDIUM_ASOFTC8Q_ELF / _ASOFTCX8Q_ (fp8 P)
// GC requires GetShapeInference and GetLibVersion entry points (KERNEL_DB log) or the library is skipped.
//   gaudium_ple_sync    qwen4_exp, GAUDIUM_PLE_GATE=1: hold the n-gram rows of a decode step until the host's flag, then copy them   ELF: GAUDIUM_PLESYNC_ELF
#include "tpc_kernel_lib_interface.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <functional>

using namespace tpc_lib_api;
struct GdnParams { int write_ckpt; };

// ELF cache. The engine compiles on GAUDIUM_COMPILE_THREADS threads (default 6), and a --serve start compiles the 12 g_zero_p* reset
// recipes at once, so the graph compiler calls this glue concurrently. Unlocked, two first uses of one ELF filled the same map entry together
// (synStatus 26 / heap corruption at serve start, 2026-11-09, tests/ref/ptpc_int/README.md). One mutex covers EVERY access to the cache: the
// lookup / insert, the file read, and put_elf's size check and copy (a failed read leaves the entry empty and the next call refills it, so a
// reader outside the lock could see it change). The copy is at most a few hundred KB, once per node instantiation: no contention cost.
static std::mutex& elf_mu() { static std::mutex mu; return mu; }
static std::vector<unsigned char>& elf_locked(const char* env) {   // caller holds elf_mu()
    static std::map<std::string, std::vector<unsigned char>> cache;   // keyed by variable AND path (an A/B harness may repoint the variable)
    const char* p = getenv(env);
    std::vector<unsigned char>& e = cache[std::string(env) + "=" + (p ? p : "")];
    if (e.empty()) {
        FILE* f = p ? fopen(p, "rb") : nullptr;
        if (!f) { fprintf(stderr, "gaudium glue: cannot open %s=%s\n", env, p ? p : "(null)"); return e; }
        fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
        e.resize(n); if (fread(e.data(), 1, n, f) != (size_t)n) e.clear(); fclose(f);
    }
    return e;
}
#include "elf_record.inc"   // which ELFs this library handed to the graph compiler: the recipe cache keys each recipe on them
static GlueCodeReturn put_elf(HabanaKernelInstantiation* out, const char* env) {
    elf_note(env);
    std::lock_guard<std::mutex> lk(elf_mu());
    std::vector<unsigned char>& e = elf_locked(env);
    if (e.empty()) return GLUE_FAILED;
    uint32_t given = out->kernel.elfSize; out->kernel.elfSize = (uint32_t)e.size();
    if (given < e.size()) return GLUE_INSUFFICIENT_ELF_BUFFER;
    memcpy(out->kernel.kernelElf, e.data(), e.size()); return GLUE_SUCCESS;
}
// ---- kernel shape variants (issue #7) ----
// The DeltaNet and attention kernels are compiled per head-count shape (src/kernels/tpc/gdn_shape.h, attn_shape.h). The glue reads the
// shape from the node's tensors and picks the ELF: the default shape (Qwen3.8-27B: 16 key heads x 3 value heads each; 4 kv groups x 6
// query heads) uses GAUDIUM_<K>_ELF, any other shape GAUDIUM_<K>_<TAG>_ELF with TAG = K<hk>R<rep> or G<nkv>Q<gq> (env.sh exports the
// variants kernels.mk builds). A shape without a built variant fails here, naming the missing variable.
static std::string gdn_tag(uint64_t hk, uint64_t rep) { return (hk == 16 && rep == 3) ? std::string() : "K" + std::to_string(hk) + "R" + std::to_string(rep); }
static std::string att_tag(uint64_t nkv, uint64_t gq) {   // gq = 0: a kernel whose code depends on the kv groups only (kv_commit): tag G<nkv>
    if (gq == 0) return nkv == 4 ? std::string() : "G" + std::to_string(nkv);
    return (nkv == 4 && gq == 6) ? std::string() : "G" + std::to_string(nkv) + "Q" + std::to_string(gq); }
static GlueCodeReturn put_elf_tag(HabanaKernelInstantiation* out, const char* env, const std::string& tag) {
    if (tag.empty()) return put_elf(out, env);
    std::string e = env; e.insert(e.size() - 4, "_" + tag);   // GAUDIUM_PREP_ELF -> GAUDIUM_PREP_K16R2_ELF
    return put_elf(out, e.c_str()); }
// linear-attention shape from the block count NBLK = 2 HK + HV (128-channel blocks of q | k | v) and HV; false when inconsistent
static bool gdn_shape(uint64_t nblk, uint64_t hv, uint64_t& hk, uint64_t& rep) {
    if (hv == 0 || nblk <= hv || (nblk - hv) % 2) return false; hk = (nblk - hv) / 2; if (hk == 0 || hv % hk) return false; rep = hv / hk; return true; }
// tensor dim `split_dim` maps 1:1 onto index-space dim 0 (with `scale` elements per index point); other dims whole.
static void pattern(TensorAccessPattern& ap, const Tensor& t, int split_dim, int scale = 1) {
    memset(&ap, 0, sizeof(ap));
    ap.allRequired = 1;
    for (uint32_t d = 0; d < t.geometry.dims; ++d) {
        ap.mapping[d].indexSpaceDim = 0;
        if ((int)d == split_dim) { ap.mapping[d].a = (float)scale; ap.mapping[d].start_b = 0; ap.mapping[d].end_b = (float)(scale - 1); }
        else { ap.mapping[d].a = 0; ap.mapping[d].start_b = 0; ap.mapping[d].end_b = (float)(t.geometry.maxSizes[d] - 1); ap.mapping[d].allRequired = true; }
    }
}
// tensor dim `d` follows index-space dim `isdim`: elements [a*i + sb, a*i + eb] (real mapping, top-level allRequired off). Idea F's probe
// glue declared these on the chunk kernels' row/chunk dims so the slicer could stage P in SRAM; that bought nothing (F2, H2 rejected) and
// in the engine, at prefill size (T = 256 rows, 8 chunks), the compiler sliced the kernels' tensors and the kernels' row % T / batch % nch
// arithmetic saw slice-relative coordinates: non-finite output. Production uses allRequired patterns (GAUDIUM_CHUNK_REAL_MAPPINGS off).
#ifdef GAUDIUM_CHUNK_REAL_MAPPINGS
static void ap_dim(TensorAccessPattern& ap, int d, int isdim, float a, float sb, float eb) {
    ap.allRequired = 0;
    ap.mapping[d].indexSpaceDim = isdim; ap.mapping[d].a = a; ap.mapping[d].start_b = sb; ap.mapping[d].end_b = eb; ap.mapping[d].allRequired = false;
}
#else
static void ap_dim(TensorAccessPattern&, int, int, float, float, float) {}
#endif
static bool all_dtype(const HabanaKernelParams* p, const TensorDataType* in, const TensorDataType* outt) {
    bool ok = true;
    for (uint32_t i = 0; i < p->inputTensorNr; ++i) if (p->inputTensors[i].geometry.dataType != in[i]) { p->inputTensors[i].geometry.dataType = in[i]; ok = false; }
    for (uint32_t i = 0; i < p->outputTensorNr; ++i) if (p->outputTensors[i].geometry.dataType != outt[i]) { p->outputTensors[i].geometry.dataType = outt[i]; ok = false; }
    return ok;
}
static bool is_f8(TensorDataType t) { return t == DATA_F8_143 || t == DATA_F8_152; }   // the KV8 builds (GAUDIUM_KV=fp8|fp8e5)
static uint32_t node_u32(const HabanaKernelParams* p, int i) {
    return (p->nodeParams.nodeParams && p->nodeParams.nodeParamsSize >= (uint32_t)(4 * (i + 1))) ? ((uint32_t*)p->nodeParams.nodeParams)[i] : 0;
}

extern "C" {
uint64_t GetLibVersion() { return 78813002322618477ULL; }
GlueCodeReturn GetShapeInference(_IN_ DeviceId, _IN_ const ShapeInferenceParams*, _OUT_ ShapeInferenceOutput*) { return GLUE_SUCCESS; }

#include "moe_w4_glue.inc"
static const char* GUIDS[] = {"gaudium_gdn_f32", "gaudium_gdn_prep", "gaudium_gdn_gate", "gaudium_gdn_post", "gaudium_attn_prep", "gaudium_attn_softmax", "gaudium_attn_post", "gaudium_attn_softmax_chunk", "gaudium_attn_softmax16", "gaudium_attn_vnat", "gaudium_attn_qs16", "gaudium_attn_combine", "gaudium_attn_post_f32", "gaudium_vtest_f32", "gaudium_vtest_bf16", "gaudium_rmw_add", "gaudium_accept_e", "gaudium_sel", "gaudium_next_ids", "gaudium_attn_softmax_fresh", "gaudium_kv_commit", "gaudium_fpos", "gaudium_pld_scan", "gaudium_pld_tail", "gaudium_cert_hq", "gaudium_cert_mark", "gaudium_cert_compact", "gaudium_cert_final",
    "gaudium_accept_t", "gaudium_df_conv", "gaudium_df_attn_prep", "gaudium_df_ctx", "gaudium_df_softmax", "gaudium_df_post", "gaudium_df_topk1", "gaudium_df_topk2", "gaudium_df_score", "gaudium_df_walk", "gaudium_df_ids", "gaudium_df_next",
    "gaudium_ds_softmax", "gaudium_ds_read", "gaudium_ds_codes", "gaudium_ds_msg", "gaudium_ds_finish", "gaudium_ds_base", "gaudium_ds_score", "gaudium_ds_cwalk",
    "gaudium_gdn_chunk", "gaudium_gdn_relayout", "gaudium_gdn_solve", "gaudium_gdn_prepa", "gaudium_gdn_prepb", "gaudium_gdn_posta", "gaudium_gdn_postb",
    "gaudium_samp_bmax", "gaudium_samp_init", "gaudium_samp_part", "gaudium_samp_red", "gaudium_samp_gumbel", "gaudium_samp_final",
    "gaudium_accept_tree", "gaudium_tree_ids", "gaudium_tree_cand", "gaudium_mtp_topk1", "gaudium_mtp_topk2", "gaudium_tree_plan", "gaudium_zero", "gaudium_fused_add_norm", "gaudium_gdn_vrec", "gaudium_gdn_vcommit",
    "gaudium_moe_route", "gaudium_moe_act", MOE_W4_GUIDS, "gaudium_qsa_prep", "gaudium_qsa_commit", "gaudium_qsa_score", "gaudium_qsa_select", "gaudium_qsa_gather", "gaudium_qsa_sp_softmax", "gaudium_qsa_sp_out", "gaudium_attn128_prep", "gaudium_attn128_post", "gaudium_attn128_combine",
    "gaudium_hc_norm", "gaudium_hc_mix", "gaudium_hc_combine", "gaudium_ple_gate", "gaudium_ple_conv", "gaudium_ple_sync"};   // qwen4_exp (Qwen3.8-Flash-Next)
GlueCodeReturn GetKernelGuids(_IN_ DeviceId deviceId, _INOUT_ uint32_t* kernelCount, _OUT_ GuidInfo* guids) {
    const uint32_t n = sizeof(GUIDS) / sizeof(GUIDS[0]);
    if (deviceId != DEVICE_ID_GAUDI2) { if (kernelCount) *kernelCount = 0; return GLUE_SUCCESS; }
    if (guids) { memset(guids, 0, n * sizeof(GuidInfo)); for (uint32_t i = 0; i < n; ++i) strcpy(guids[i].name, GUIDS[i]); }
    if (kernelCount) *kernelCount = n;
    return GLUE_SUCCESS;
}

GlueCodeReturn InstantiateTpcKernel(_IN_ HabanaKernelParams* p, _INOUT_ HabanaKernelInstantiation* out) {
    const std::string g = p->guid.name;
    if (g == "gaudium_gdn_f32") {
        const bool slotted = p->inputTensorNr >= 7;   // engine variant: S_in {64,256,HV,NS} read at device slot; 8 inputs: + parent table (tree verify)
        const bool tree = p->inputTensorNr == 8;
        if (p->inputTensorNr != 6 && p->inputTensorNr != 7 && p->inputTensorNr != 8) { p->inputTensorNr = 6; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 3) { p->outputTensorNr = 3; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[8] = {DATA_F32, DATA_F32, DATA_F32, DATA_F32, DATA_F32, DATA_F32, DATA_I32, DATA_I32}, ot[3] = {DATA_F32, DATA_F32, DATA_F32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& S = p->inputTensors[0]; const Tensor& q = p->inputTensors[1];
        if (S.geometry.dims != (slotted ? 4u : 3u) || S.geometry.maxSizes[0] != 64 || S.geometry.maxSizes[1] != 256) return GLUE_INCOMPATIBLE_INPUT_SIZE;   // blob layout
        uint64_t HV = S.geometry.maxSizes[2], T = q.geometry.maxSizes[2];
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = HV;
        pattern(out->inputTensorAccessPattern[0], p->inputTensors[0], 2);
        pattern(out->inputTensorAccessPattern[1], p->inputTensors[1], 1);
        pattern(out->inputTensorAccessPattern[2], p->inputTensors[2], 1);
        pattern(out->inputTensorAccessPattern[3], p->inputTensors[3], 1);
        pattern(out->inputTensorAccessPattern[4], p->inputTensors[4], 0);
        pattern(out->inputTensorAccessPattern[5], p->inputTensors[5], 0);
        if (slotted) pattern(out->inputTensorAccessPattern[6], p->inputTensors[6], -1);
        if (tree) pattern(out->inputTensorAccessPattern[7], p->inputTensors[7], -1);
        pattern(out->outputTensorAccessPattern[0], p->outputTensors[0], 1);
        pattern(out->outputTensorAccessPattern[1], p->outputTensors[1], 2);
        pattern(out->outputTensorAccessPattern[2], p->outputTensors[2], 2);
        out->kernel.paramsNr = 2; out->kernel.scalarParams[0] = (uint32_t)T; out->kernel.scalarParams[1] = node_u32(p, 0);
        return put_elf(out, tree ? "GAUDIUM_GDNT_ELF" : slotted ? "GAUDIUM_GDNS_ELF" : "GAUDIUM_GDN_ELF");
    }
    if (g == "gaudium_gdn_prep") {
        const bool slotted = p->inputTensorNr >= 5;   // engine variant: cs_in {10240,3,NS} read at device slot; 6 inputs: + parent table (tree verify)
        const bool tree = p->inputTensorNr == 6;
        if (p->inputTensorNr != 4 && p->inputTensorNr != 5 && p->inputTensorNr != 6) { p->inputTensorNr = 4; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 5) { p->outputTensorNr = 5; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[6] = {DATA_BF16, DATA_F32, DATA_F32, DATA_F32, DATA_I32, DATA_I32}, ot[5] = {DATA_F32, DATA_F32, DATA_F32, DATA_F32, DATA_F32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (p->inputTensors[2].geometry.dims != (slotted ? 3u : 2u)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        uint64_t T = p->inputTensors[0].geometry.maxSizes[1];
        const uint32_t pflags = node_u32(p, 0);
        // shape: NBLK from conv_w {128 NBLK, 4}, HV from v {128, HV, T} (chunked layout {128, C, HV, NCH})
        const Tensor& vt = p->outputTensors[2]; const uint64_t NBLK = p->inputTensors[1].geometry.maxSizes[0] / 128, HVs = vt.geometry.maxSizes[(pflags & 32) ? 2 : 1]; uint64_t HKs = 0, REPs = 0;
        if (p->inputTensors[1].geometry.maxSizes[0] % 128 || !gdn_shape(NBLK, HVs, HKs, REPs)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const std::string tag = gdn_tag(HKs, REPs);
        if (pflags & 64) {   // row-parallel prefill build (gdn_prep2_gaudi2.c): {80 blocks, row groups of RG rows}; chunked layout only, no per-row checkpoints
            if (!(pflags & 32) || (pflags & 2)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
            const uint32_t RG = 64;
            out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = NBLK; out->indexSpaceGeometry[1] = (T + RG - 1) / RG;
            for (uint32_t i = 0; i < p->inputTensorNr; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
            for (int i = 0; i < 5; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
            out->kernel.paramsNr = 3; out->kernel.scalarParams[0] = (uint32_t)T; out->kernel.scalarParams[1] = pflags; out->kernel.scalarParams[2] = RG;
            return put_elf_tag(out, slotted ? "GAUDIUM_PREPS2_ELF" : "GAUDIUM_PREP2_ELF", tag);
        }
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = NBLK;          // 128-channel blocks of the 10240 conv channels
        // decode builds (gdn_prep_dec): index point -> block is a permutation that interleaves the 32 q/k blocks (l2-norm, 3 GQA copies)
        // with the 48 v blocks, so the patterns are all-required; the prefill builds keep the affine block patterns
        const bool decode_build = tree || (!(pflags & 32) && T <= 32);   // the gdn_prep_dec builds (PREPD / PREPSD / PREPT)
        const int bp = decode_build ? -1 : 0;
        pattern(out->inputTensorAccessPattern[0], p->inputTensors[0], bp, 128);
        pattern(out->inputTensorAccessPattern[1], p->inputTensors[1], bp, 128);
        pattern(out->inputTensorAccessPattern[2], p->inputTensors[2], bp, 128);
        pattern(out->inputTensorAccessPattern[3], p->inputTensors[3], bp, 128);
        if (slotted) pattern(out->inputTensorAccessPattern[4], p->inputTensors[4], -1);
        if (tree) pattern(out->inputTensorAccessPattern[5], p->inputTensors[5], -1);
        for (int i = 0; i < 3; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
        pattern(out->outputTensorAccessPattern[3], p->outputTensors[3], bp, 128);
        pattern(out->outputTensorAccessPattern[4], p->outputTensors[4], bp, 128);
        out->kernel.paramsNr = 2; out->kernel.scalarParams[0] = (uint32_t)T; out->kernel.scalarParams[1] = node_u32(p, 0);
        const bool dec = !(pflags & 32) && T <= 32;   // decode / verify rows: the -DDECODE build (no chunked-layout code, no unroll)
        return put_elf_tag(out, tree ? "GAUDIUM_PREPT_ELF" : slotted ? (dec ? "GAUDIUM_PREPSD_ELF" : "GAUDIUM_PREPS_ELF") : (dec ? "GAUDIUM_PREPD_ELF" : "GAUDIUM_PREP_ELF"), tag);
    }
    if (g == "gaudium_gdn_vrec") {   // PR {128,2T,HV,1}, G {2T,2T,HV,1}, v {128,HV,T}, eg {64,T,1}, beta {64,T} -> o {128,HV,T}, dl {128,T,HV,1}; params NC, L
        if (p->inputTensorNr != 5) { p->inputTensorNr = 5; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 2) { p->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[5] = {DATA_F32, DATA_F32, DATA_F32, DATA_F32, DATA_F32}, ot[2] = {DATA_F32, DATA_F32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& v = p->inputTensors[2];
        uint64_t HV = v.geometry.maxSizes[1], T = v.geometry.maxSizes[2];
        if (T > 32 || p->inputTensors[0].geometry.maxSizes[1] != 2 * T || p->inputTensors[1].geometry.maxSizes[0] != 2 * T || p->inputTensors[1].geometry.maxSizes[1] != 2 * T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = HV;
        for (int i = 0; i < 5; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        for (int i = 0; i < 2; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
        out->kernel.paramsNr = 3; out->kernel.scalarParams[0] = (uint32_t)T; out->kernel.scalarParams[1] = node_u32(p, 0) ? node_u32(p, 0) : (uint32_t)T; out->kernel.scalarParams[2] = node_u32(p, 1);
        return put_elf(out, "GAUDIUM_VREC_ELF");
    }
    if (g == "gaudium_gdn_vcommit") {   // dl {128,T,HV,1}, eg {64,T,1}, a {1}, S0 {128,128,HV,1} -> dlc {128,T,HV,1}, Sd {128,128,HV,1}; params NC, L
        if (p->inputTensorNr != 4) { p->inputTensorNr = 4; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 2) { p->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[4] = {DATA_F32, DATA_F32, DATA_I32, DATA_F32}, ot[2] = {DATA_F32, DATA_F32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& dl = p->inputTensors[0];
        uint64_t T = dl.geometry.maxSizes[1], HV = dl.geometry.maxSizes[2];
        if (T > 32 || p->inputTensors[3].geometry.maxSizes[0] != 128 || p->inputTensors[3].geometry.maxSizes[1] != 128) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = HV;
        for (int i = 0; i < 4; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        for (int i = 0; i < 2; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
        out->kernel.paramsNr = 3; out->kernel.scalarParams[0] = (uint32_t)T; out->kernel.scalarParams[1] = node_u32(p, 0) ? node_u32(p, 0) : (uint32_t)T; out->kernel.scalarParams[2] = node_u32(p, 1);
        return put_elf(out, "GAUDIUM_VCOMMIT_ELF");
    }
    if (g == "gaudium_gdn_gate") {
        const bool chunk = p->outputTensorNr == 1;   // MME prefill variant: gb {128, Tp} = G (chunk-local cumsum) | beta; node param 0 = C
        if (p->inputTensorNr != 3) { p->inputTensorNr = 3; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 2 && p->outputTensorNr != 1) { p->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[3] = {DATA_BF16, DATA_F32, DATA_F32}, ot[2] = {DATA_F32, DATA_F32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        uint64_t T = p->inputTensors[0].geometry.maxSizes[1];
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = 1;
        if (chunk && node_u32(p, 0)) out->indexSpaceGeometry[0] = p->outputTensors[0].geometry.maxSizes[1] / node_u32(p, 0);   // one point per chunk (Tp / C)
        for (int i = 0; i < 3; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        for (uint32_t i = 0; i < p->outputTensorNr; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
        out->kernel.paramsNr = chunk ? 2 : 1; out->kernel.scalarParams[0] = (uint32_t)T; out->kernel.scalarParams[1] = node_u32(p, 0);
        return put_elf(out, chunk ? "GAUDIUM_GATEC_ELF" : "GAUDIUM_GATE_ELF");
    }
    if (g == "gaudium_gdn_post") {
        if (p->inputTensorNr != 3) { p->inputTensorNr = 3; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 1) { p->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[3] = {DATA_F32, DATA_BF16, DATA_F32}, ot[1] = {DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const bool chunkX = (node_u32(p, 0) & 32) != 0;
        uint64_t HV = p->inputTensors[0].geometry.maxSizes[chunkX ? 2 : 1], T = chunkX ? p->outputTensors[0].geometry.maxSizes[1] : p->inputTensors[0].geometry.maxSizes[2];
        // shape: the gate z starts at channel QKV = QKVZ - 128 HV of qkvz {QKVZ, T}
        const uint64_t QKVZs = p->inputTensors[1].geometry.maxSizes[0]; uint64_t HKs = 0, REPs = 0;
        if (QKVZs % 128 || QKVZs / 128 <= HV || !gdn_shape(QKVZs / 128 - HV, HV, HKs, REPs)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        // flags bit 10 (qwen4_exp): the sigmoid-gate build, tag K<hk>R<rep>S (kernels.mk GDN_SIG_SHAPES)
        const std::string tag = (node_u32(p, 0) & 1024) ? "K" + std::to_string(HKs) + "R" + std::to_string(REPs) + "S" : gdn_tag(HKs, REPs);
        if (node_u32(p, 0) & 64) {   // row-parallel prefill build (gdn_post2_gaudi2.c): {HV heads, row groups of RG rows}; chunked X only
            if (!chunkX) return GLUE_INCOMPATIBLE_INPUT_SIZE;
            const uint32_t RG = 64;
            out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = HV; out->indexSpaceGeometry[1] = (T + RG - 1) / RG;
            for (uint32_t i = 0; i < 3; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
            pattern(out->outputTensorAccessPattern[0], p->outputTensors[0], -1);
            out->kernel.paramsNr = 3; out->kernel.scalarParams[0] = (uint32_t)T; out->kernel.scalarParams[1] = node_u32(p, 0); out->kernel.scalarParams[2] = RG;
            return put_elf_tag(out, "GAUDIUM_POST2_ELF", tag);
        }
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = HV;
        pattern(out->inputTensorAccessPattern[0], p->inputTensors[0], chunkX ? 2 : 1);
        pattern(out->inputTensorAccessPattern[1], p->inputTensors[1], -1);
        pattern(out->inputTensorAccessPattern[2], p->inputTensors[2], -1);
        pattern(out->outputTensorAccessPattern[0], p->outputTensors[0], 0, 128);
        out->kernel.paramsNr = 2; out->kernel.scalarParams[0] = (uint32_t)T; out->kernel.scalarParams[1] = node_u32(p, 0);   // flags (bit5: chunked X input)
        return put_elf_tag(out, (!chunkX && T <= 32) ? "GAUDIUM_POSTD_ELF" : "GAUDIUM_POST_ELF", tag);   // decode / verify rows: the -DDECODE build
    }
    if (g == "gaudium_attn_prep" && (p->inputTensorNr == 9 || p->inputTensorNr == 11)) {   // KV8 builds (+ kv8 f32 {16} last)
        const bool tree = p->inputTensorNr == 11 && p->inputTensors[8].geometry.dataType == DATA_I32, fresh = p->inputTensorNr == 11 && !tree;
        const uint32_t nout = fresh ? 4 : 3, nin = p->inputTensorNr;
        if (p->outputTensorNr != nout) { p->outputTensorNr = nout; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const Tensor& qb = p->outputTensors[0]; const Tensor& kc = p->outputTensors[fresh ? 2 : 1]; const Tensor& vc = p->outputTensors[fresh ? 3 : 2];
        const TensorDataType f8 = kc.geometry.dataType == DATA_F8_152 ? DATA_F8_152 : DATA_F8_143;
        TensorDataType in[11] = {DATA_BF16, DATA_BF16, DATA_BF16, DATA_F32, DATA_F32, DATA_F32, DATA_F32, DATA_I32, DATA_F32, DATA_F32, DATA_F32}, ot[4] = {f8, f8, f8, f8};
        if (tree) { in[8] = DATA_I32; in[9] = DATA_I32; } else if (fresh) { in[8] = f8; in[9] = f8; }
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& k8 = p->inputTensors[nin - 1]; if (k8.geometry.dims != 1 || k8.geometry.maxSizes[0] < 16) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        uint64_t T = p->inputTensors[0].geometry.maxSizes[1];
        // shape: NH from qg {2 x 256 NH, T} (q and the output gate), NKV from k {256 NKV, T}
        const uint64_t NHs = p->inputTensors[0].geometry.maxSizes[0] / 512, NKVs = p->inputTensors[1].geometry.maxSizes[0] / 256;
        if (NHs == 0 || NKVs == 0 || NHs % NKVs || p->inputTensors[0].geometry.maxSizes[0] % 512 || p->inputTensors[1].geometry.maxSizes[0] % 256) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const uint64_t GQs = NHs / NKVs; const std::string tag = att_tag(NKVs, GQs);
        if (qb.geometry.dims != 3 || kc.geometry.dims != 3 || vc.geometry.dims != 3 || qb.geometry.maxSizes[2] % NKVs || qb.geometry.maxSizes[1] != GQs * T) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        uint64_t nch = qb.geometry.maxSizes[2] / NKVs, C = kc.geometry.maxSizes[1];
        // QSA gathered sparse core: a one-chunk q block (nch 1: q written once) beside a multi-chunk cache; the kernel uses nch only for the q copies
        const bool q1 = !fresh && !tree && nch == 1 && kc.geometry.maxSizes[2] % NKVs == 0 && kc.geometry.maxSizes[2] > NKVs;
        if (fresh) {
            if (nch == 0 || C % 256 || kc.geometry.maxSizes[2] != NKVs || vc.geometry.maxSizes[1] != C || vc.geometry.maxSizes[2] != NKVs) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
            const Tensor& qf = p->outputTensors[1]; if (qf.geometry.dims != 3 || qf.geometry.maxSizes[1] != GQs * T || qf.geometry.maxSizes[2] != NKVs) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
            const Tensor& ki = p->inputTensors[8]; if (ki.geometry.dims != 3 || ki.geometry.maxSizes[1] != C || ki.geometry.maxSizes[2] != NKVs) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        } else if (nch == 0 || C % 256 || kc.geometry.maxSizes[2] != NKVs * nch || vc.geometry.maxSizes[1] != C || vc.geometry.maxSizes[2] != NKVs * nch) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = NHs + NKVs;
        for (uint32_t i = 0; i < nin; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        for (uint32_t i = 0; i < nout; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
        const uint32_t fmt = f8 == DATA_F8_152 ? 1u : 0u;
        out->kernel.scalarParams[0] = (uint32_t)T; out->kernel.scalarParams[1] = node_u32(p, 0); out->kernel.scalarParams[2] = (uint32_t)nch;
        if (fresh) { out->kernel.paramsNr = 7; out->kernel.scalarParams[3] = node_u32(p, 2); out->kernel.scalarParams[4] = node_u32(p, 1); out->kernel.scalarParams[5] = (uint32_t)C; out->kernel.scalarParams[6] = fmt; }
        else { out->kernel.paramsNr = 5; out->kernel.scalarParams[3] = (uint32_t)C; out->kernel.scalarParams[4] = fmt; }
        return put_elf_tag(out, fresh ? "GAUDIUM_APREPF8_ELF" : tree ? "GAUDIUM_APREPT8_ELF" : "GAUDIUM_APREP8_ELF", tag);
    }
    if (g == "gaudium_attn_prep") {
        // 10 inputs: the fresh-window variant (+ kf_in, vf_in bf16; outputs qb, qb_f, kf_out, vf_out) or the tree variant (+ depth, fix int32;
        // outputs qb, kcache, vcache: DFlash2 draft-tree verify on the chunked cache)
        const bool tree = p->inputTensorNr == 10 && p->inputTensors[8].geometry.dataType == DATA_I32;
        const bool fresh = p->inputTensorNr == 10 && !tree;
        if (p->inputTensorNr != 8 && p->inputTensorNr != 10) { p->inputTensorNr = 8; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        const uint32_t nout = fresh ? 4 : 3;
        if (p->outputTensorNr != nout) { p->outputTensorNr = nout; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        TensorDataType in[10] = {DATA_BF16, DATA_BF16, DATA_BF16, DATA_F32, DATA_F32, DATA_F32, DATA_F32, DATA_I32, DATA_BF16, DATA_BF16}, ot[4] = {DATA_BF16, DATA_BF16, DATA_BF16, DATA_BF16};
        if (tree) { in[8] = DATA_I32; in[9] = DATA_I32; }
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        uint64_t T = p->inputTensors[0].geometry.maxSizes[1];
        // shape: NH from qg {2 x 256 NH, T} (q and the output gate), NKV from k {256 NKV, T}
        const uint64_t NHs = p->inputTensors[0].geometry.maxSizes[0] / 512, NKVs = p->inputTensors[1].geometry.maxSizes[0] / 256;
        if (NHs == 0 || NKVs == 0 || NHs % NKVs || p->inputTensors[0].geometry.maxSizes[0] % 512 || p->inputTensors[1].geometry.maxSizes[0] % 256) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const uint64_t GQs = NHs / NKVs; const std::string tag = att_tag(NKVs, GQs);
        // chunked caches: qb {256, 6T, 4*nch}, kcache/vcache {256, C, 4*nch} (chunk-major batch); nch = 1 and C = Lmax is the unchunked layout
        const Tensor& qb = p->outputTensors[0]; const Tensor& kc = p->outputTensors[fresh ? 2 : 1]; const Tensor& vc = p->outputTensors[fresh ? 3 : 2];
        if (qb.geometry.dims != 3 || kc.geometry.dims != 3 || vc.geometry.dims != 3 || qb.geometry.maxSizes[2] % NKVs || qb.geometry.maxSizes[1] != GQs * T) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        uint64_t nch = qb.geometry.maxSizes[2] / NKVs, C = kc.geometry.maxSizes[1];
        // QSA gathered sparse core: a one-chunk q block (nch 1: q written once) beside a multi-chunk cache; the kernel uses nch only for the q copies
        const bool q1 = !fresh && !tree && nch == 1 && kc.geometry.maxSizes[2] % NKVs == 0 && kc.geometry.maxSizes[2] > NKVs;
        if (fresh) {   // kc/vc are the window {256, NF, 4}: C = NF, 4 batches
            if (nch == 0 || C % 128 || kc.geometry.maxSizes[2] != NKVs || vc.geometry.maxSizes[1] != C || vc.geometry.maxSizes[2] != NKVs) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
            const Tensor& qf = p->outputTensors[1]; if (qf.geometry.dims != 3 || qf.geometry.maxSizes[1] != GQs * T || qf.geometry.maxSizes[2] != NKVs) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
            const Tensor& ki = p->inputTensors[8]; if (ki.geometry.dims != 3 || ki.geometry.maxSizes[1] != C || ki.geometry.maxSizes[2] != NKVs) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        } else if (nch == 0 || C % 128 || vc.geometry.maxSizes[1] != C || (q1 ? vc.geometry.maxSizes[2] != kc.geometry.maxSizes[2] : (kc.geometry.maxSizes[2] != NKVs * nch || vc.geometry.maxSizes[2] != NKVs * nch))) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = NHs + NKVs;      // 24 q heads + 4 kv heads
        for (uint32_t i = 0; i < p->inputTensorNr; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        for (uint32_t i = 0; i < nout; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
        // params: T, coff (node param 0; 0 when none), nch, C [, sbase (node param 1), NF]
        out->kernel.paramsNr = fresh ? 6 : 4; out->kernel.scalarParams[0] = (uint32_t)T; out->kernel.scalarParams[1] = node_u32(p, 0);
        if (fresh) { // the big-cache chunk geometry comes from the qb shape (nch) and the second node param (C of the big cache)
            out->kernel.scalarParams[2] = (uint32_t)nch; out->kernel.scalarParams[3] = node_u32(p, 2); out->kernel.scalarParams[4] = node_u32(p, 1); out->kernel.scalarParams[5] = (uint32_t)C; }
        else { out->kernel.scalarParams[2] = (uint32_t)nch; out->kernel.scalarParams[3] = (uint32_t)C; }
        return put_elf_tag(out, fresh ? "GAUDIUM_APREPF_ELF" : tree ? "GAUDIUM_APREPT_ELF" : "GAUDIUM_APREP_ELF", tag);
    }
    // ---- 128-channel heads (plain Qwen3, attn_shape.h ATT_DH=128): own GUIDs, so a head width is never guessed from a tensor width ----
    if (g == "gaudium_attn128_prep") {   // qg {128 NH, T} (q only), k / v {128 NKV, T}, qn_w / kn_w f32 {128}, rope_c / rope_s f32 {128, L}, cursor i32 {1}
        //                                  [+ depth, fix i32: tree / kvfix] -> qb {128, GQ T, NKV nch}, kcache / vcache {128, C, NKV nch}; params T, coff, nch, C
        const bool tree = p->inputTensorNr == 10;
        if (p->inputTensorNr != 8 && p->inputTensorNr != 10) { p->inputTensorNr = 8; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 3) { p->outputTensorNr = 3; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[10] = {DATA_BF16, DATA_BF16, DATA_BF16, DATA_F32, DATA_F32, DATA_F32, DATA_F32, DATA_I32, DATA_I32, DATA_I32}, ot[3] = {DATA_BF16, DATA_BF16, DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const uint64_t T = p->inputTensors[0].geometry.maxSizes[1];
        const uint64_t NHs = p->inputTensors[0].geometry.maxSizes[0] / 128, NKVs = p->inputTensors[1].geometry.maxSizes[0] / 128;
        if (NHs == 0 || NKVs == 0 || NHs % NKVs || p->inputTensors[0].geometry.maxSizes[0] % 128 || p->inputTensors[1].geometry.maxSizes[0] % 128) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (p->inputTensors[3].geometry.maxSizes[0] != 128 || p->inputTensors[4].geometry.maxSizes[0] != 128 || p->inputTensors[5].geometry.maxSizes[0] != 128) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const uint64_t GQs = NHs / NKVs;
        const Tensor& qb = p->outputTensors[0]; const Tensor& kc = p->outputTensors[1]; const Tensor& vc = p->outputTensors[2];
        if (qb.geometry.dims != 3 || kc.geometry.dims != 3 || vc.geometry.dims != 3 || qb.geometry.maxSizes[0] != 128 || kc.geometry.maxSizes[0] != 128 || vc.geometry.maxSizes[0] != 128) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (qb.geometry.maxSizes[2] % NKVs || qb.geometry.maxSizes[1] != GQs * T) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        const uint64_t nch = qb.geometry.maxSizes[2] / NKVs, C = kc.geometry.maxSizes[1];
        if (nch == 0 || C % 128 || kc.geometry.maxSizes[2] != NKVs * nch || vc.geometry.maxSizes[1] != C || vc.geometry.maxSizes[2] != NKVs * nch) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = NHs + NKVs;   // q heads, then kv heads
        for (uint32_t i = 0; i < p->inputTensorNr; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        for (uint32_t i = 0; i < 3; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
        out->kernel.paramsNr = 4; out->kernel.scalarParams[0] = (uint32_t)T; out->kernel.scalarParams[1] = node_u32(p, 0);
        out->kernel.scalarParams[2] = (uint32_t)nch; out->kernel.scalarParams[3] = (uint32_t)C;
        return put_elf_tag(out, tree ? "GAUDIUM_A128PREPT_ELF" : "GAUDIUM_A128PREP_ELF", att_tag(NKVs, GQs));
    }
    if (g == "gaudium_attn128_post") {   // ao f32 {128, GQ T, NKV} (lane order), qg bf16 {128 NH, T} (unused: no gate) -> core bf16 {128 NH, T}; param T
        if (p->inputTensorNr != 2) { p->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 1) { p->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[2] = {DATA_F32, DATA_BF16}, ot[1] = {DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const uint64_t T = p->inputTensors[1].geometry.maxSizes[1];
        const uint64_t NHs = p->inputTensors[1].geometry.maxSizes[0] / 128, NKVs = p->inputTensors[0].geometry.maxSizes[2];
        if (p->inputTensors[0].geometry.maxSizes[0] != 128 || NHs == 0 || NKVs == 0 || NHs % NKVs || p->inputTensors[0].geometry.maxSizes[1] != (NHs / NKVs) * T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (p->outputTensors[0].geometry.maxSizes[0] != 128 * NHs || p->outputTensors[0].geometry.maxSizes[1] != T) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = NHs;
        pattern(out->inputTensorAccessPattern[0], p->inputTensors[0], -1); pattern(out->inputTensorAccessPattern[1], p->inputTensors[1], -1);
        pattern(out->outputTensorAccessPattern[0], p->outputTensors[0], 0, 128);
        out->kernel.paramsNr = 1; out->kernel.scalarParams[0] = (uint32_t)T;
        return put_elf_tag(out, "GAUDIUM_A128POST_ELF", att_tag(NKVs, NHs / NKVs));
    }
    if (g == "gaudium_attn128_combine") {   // ao_part f32 {128,R,B}, stats f32 {64,2,R,B} -> ao f32 {128,R,G}; params: nch [, unroll]
        if (p->inputTensorNr != 2) { p->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 1) { p->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[2] = {DATA_F32, DATA_F32}, ot[1] = {DATA_F32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& A = p->inputTensors[0]; const Tensor& st = p->inputTensors[1]; const Tensor& O = p->outputTensors[0];
        const uint32_t nch = node_u32(p, 0);
        if (O.geometry.dims != 3 || O.geometry.maxSizes[0] != 128) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        const uint64_t R = O.geometry.maxSizes[1], G = O.geometry.maxSizes[2];
        if (A.geometry.dims != 3 || A.geometry.maxSizes[0] != 128 || A.geometry.maxSizes[1] != R || nch == 0 || A.geometry.maxSizes[2] != G * nch) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (st.geometry.dims != 4 || st.geometry.maxSizes[2] != R || st.geometry.maxSizes[3] != G * nch) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = R; out->indexSpaceGeometry[1] = G;
        for (uint32_t i = 0; i < 2; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        pattern(out->outputTensorAccessPattern[0], O, -1);
        out->kernel.paramsNr = 1; out->kernel.scalarParams[0] = nch;
        return put_elf(out, node_u32(p, 1) ? "GAUDIUM_A128COMBX_ELF" : "GAUDIUM_A128COMB_ELF");
    }
    if (g == "gaudium_attn_softmax") {
        if (p->inputTensorNr != 2) { p->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 1) { p->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[2] = {DATA_F32, DATA_I32}, ot[1] = {DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        uint64_t Lmax = p->inputTensors[0].geometry.maxSizes[0], R = p->inputTensors[0].geometry.maxSizes[1], G = p->inputTensors[0].geometry.maxSizes[2];
        uint32_t T = node_u32(p, 0); if (T == 0 || R % T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = R * G;
        pattern(out->inputTensorAccessPattern[0], p->inputTensors[0], -1); pattern(out->inputTensorAccessPattern[1], p->inputTensors[1], -1);
        pattern(out->outputTensorAccessPattern[0], p->outputTensors[0], -1);
        out->kernel.paramsNr = 3; out->kernel.scalarParams[0] = T; out->kernel.scalarParams[1] = (uint32_t)Lmax; out->kernel.scalarParams[2] = node_u32(p, 1);   // cursor offset
        return put_elf_tag(out, "GAUDIUM_ASOFT_ELF", att_tag(G, R / T));
    }
    if (g == "gaudium_attn_post") {
        if (p->inputTensorNr != 2) { p->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 1) { p->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[2] = {DATA_BF16, DATA_BF16}, ot[1] = {DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        uint64_t T = p->inputTensors[1].geometry.maxSizes[1];
        const uint64_t NHs = p->inputTensors[1].geometry.maxSizes[0] / 512, NKVs = p->inputTensors[0].geometry.maxSizes[2];   // shape: qg {2 x 256 NH, T}, ao {256, GQ T, NKV}
        if (NHs == 0 || NKVs == 0 || NHs % NKVs || p->inputTensors[0].geometry.maxSizes[1] != (NHs / NKVs) * T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = NHs;
        pattern(out->inputTensorAccessPattern[0], p->inputTensors[0], -1); pattern(out->inputTensorAccessPattern[1], p->inputTensors[1], -1);
        pattern(out->outputTensorAccessPattern[0], p->outputTensors[0], 0, 256);
        out->kernel.paramsNr = 1; out->kernel.scalarParams[0] = (uint32_t)T;
        return put_elf_tag(out, "GAUDIUM_APOST_ELF", att_tag(NKVs, NHs / NKVs));
    }
    // QSA builds: scores f32 {C,R,B}, cursor i32 {1}, qsel i32 {64, NW, T} [, kv8 f32 (fp8 P)] -> P, stats; node params T, nch, coff, mode,
    // fast, pbase, qfull (docs/notes/engine-qsa.md). Chunk bases must be multiples of 2048 (C % 2048 == 0 or one chunk per node).
    if (g == "gaudium_attn_softmax_chunk" && p->inputTensorNr >= 3 && p->inputTensors[2].geometry.dataType == DATA_I32 && p->inputTensors[2].geometry.dims == 3) {
        const bool f8 = p->outputTensorNr == 2 && is_f8(p->outputTensors[0].geometry.dataType);
        if (p->inputTensorNr != (f8 ? 4u : 3u)) { p->inputTensorNr = f8 ? 4 : 3; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 2) { p->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType pt = f8 ? p->outputTensors[0].geometry.dataType : DATA_BF16;
        const TensorDataType in[4] = {DATA_F32, DATA_I32, DATA_I32, DATA_F32}, ot[2] = {pt, DATA_F32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& S = p->inputTensors[0]; const Tensor& st = p->outputTensors[1]; const Tensor& Q = p->inputTensors[2];
        if (S.geometry.dims != 3) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        uint64_t C = S.geometry.maxSizes[0], R = S.geometry.maxSizes[1], B = S.geometry.maxSizes[2];
        uint32_t T = node_u32(p, 0), nch = node_u32(p, 1); const uint32_t fast = node_u32(p, 4);
        if (T == 0 || R % T || nch == 0 || B % nch || C % 256) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (fast == 0 && C > 4096) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (nch > 1 && C % 2048) return GLUE_INCOMPATIBLE_INPUT_SIZE;                                   // chunk bases on 2048-position words
        if (Q.geometry.maxSizes[0] != 64 || Q.geometry.maxSizes[2] != T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (st.geometry.dims != 4 || st.geometry.maxSizes[0] != 64 || st.geometry.maxSizes[1] != 2 || st.geometry.maxSizes[2] != R || st.geometry.maxSizes[3] != B) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (f8 && (p->inputTensors[3].geometry.dims != 1 || p->inputTensors[3].geometry.maxSizes[0] < 18)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const bool flat = !fast;
        if (flat) { out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = R * B; }
        else { out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = R; out->indexSpaceGeometry[1] = B; }
        for (uint32_t i = 0; i < p->inputTensorNr; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        for (uint32_t i = 0; i < 2; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
        uint32_t* sp = out->kernel.scalarParams; uint32_t n = 0;
        sp[n++] = T; sp[n++] = nch; sp[n++] = (uint32_t)C; sp[n++] = node_u32(p, 2); sp[n++] = node_u32(p, 3);
        if (fast) sp[n++] = node_u32(p, 5);                                                             // pbase
        if (f8) sp[n++] = pt == DATA_F8_152 ? 1u : 0u;                                                 // fmt
        sp[n++] = node_u32(p, 6);                                                                       // qfull (0: every row reads its mask)
        out->kernel.paramsNr = n;
        if (f8) return put_elf(out, fast ? "GAUDIUM_ASOFTCX8Q_ELF" : "GAUDIUM_ASOFTC8Q_ELF");
        return put_elf(out, fast == 2 ? "GAUDIUM_ASOFTCX4Q_ELF" : fast ? "GAUDIUM_ASOFTCXQ_ELF" : "GAUDIUM_ASOFTCQ_ELF");
    }
    if (g == "gaudium_attn_softmax_chunk" && p->outputTensorNr == 2 && is_f8(p->outputTensors[0].geometry.dataType)) {   // fp8 P (+ kv8 f32 {16} last)
        const bool tree = p->inputTensorNr == 4;
        if (p->inputTensorNr != 3 && p->inputTensorNr != 4) { p->inputTensorNr = 3; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        const TensorDataType f8 = p->outputTensors[0].geometry.dataType;
        const TensorDataType in[4] = {DATA_F32, DATA_I32, tree ? DATA_I32 : DATA_F32, DATA_F32}, ot[2] = {f8, DATA_F32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& S = p->inputTensors[0]; const Tensor& st = p->outputTensors[1]; const Tensor& k8 = p->inputTensors[p->inputTensorNr - 1];
        if (S.geometry.dims != 3 || k8.geometry.dims != 1 || k8.geometry.maxSizes[0] < 18) return GLUE_INCOMPATIBLE_INPUT_SIZE;   // [16] score scale, [17] stats scale
        uint64_t C = S.geometry.maxSizes[0], R = S.geometry.maxSizes[1], B = S.geometry.maxSizes[2];
        uint32_t T = node_u32(p, 0), nch = node_u32(p, 1);
        if (T == 0 || R % T || nch == 0 || B % nch || C % 256) return GLUE_INCOMPATIBLE_INPUT_SIZE;   // B = kv groups x chunks (the kernel reads the group count from the tensor)
        if (node_u32(p, 4) == 0 && C > 4096) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (st.geometry.dims != 4 || st.geometry.maxSizes[0] != 64 || st.geometry.maxSizes[1] != 2 || st.geometry.maxSizes[2] != R || st.geometry.maxSizes[3] != B) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        const bool flat = tree || !node_u32(p, 4);
        if (flat) { out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = R * B; }
        else { out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = R; out->indexSpaceGeometry[1] = B; }
        for (uint32_t i = 0; i < p->inputTensorNr; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        if (tree) { if (p->inputTensors[2].geometry.maxSizes[0] < T || T > 32) return GLUE_INCOMPATIBLE_INPUT_SIZE; }
        for (uint32_t i = 0; i < 2; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
        const uint32_t fmt = f8 == DATA_F8_152 ? 1u : 0u;
        out->kernel.paramsNr = 6; out->kernel.scalarParams[0] = T; out->kernel.scalarParams[1] = nch; out->kernel.scalarParams[2] = (uint32_t)C;
        out->kernel.scalarParams[3] = node_u32(p, 2); out->kernel.scalarParams[4] = node_u32(p, 3); out->kernel.scalarParams[5] = fmt;
        if (!tree && node_u32(p, 4)) { out->kernel.paramsNr = 7; out->kernel.scalarParams[5] = node_u32(p, 5); out->kernel.scalarParams[6] = fmt; }   // FASTEXP: pbase (key position of the group's first chunk), then fmt
        return put_elf(out, tree ? "GAUDIUM_ASOFTCT8_ELF" : node_u32(p, 4) ? "GAUDIUM_ASOFTCX8_ELF" : "GAUDIUM_ASOFTC8_ELF");
    }
    if (g == "gaudium_attn_softmax_chunk") {   // scores f32 {C,R,B}, cursor i32 {1} [, anc i32 {T}: tree variant] -> P bf16 {C,R,B}, stats f32 {64,2,R,B}; params: T, nch, coff, mode
        const bool tree = p->inputTensorNr == 3;
        if (p->inputTensorNr != 2 && p->inputTensorNr != 3) { p->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 2) { p->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[3] = {DATA_F32, DATA_I32, DATA_I32}, ot[2] = {DATA_BF16, DATA_F32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& S = p->inputTensors[0]; const Tensor& st = p->outputTensors[1];
        if (S.geometry.dims != 3) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        uint64_t C = S.geometry.maxSizes[0], R = S.geometry.maxSizes[1], B = S.geometry.maxSizes[2];
        uint32_t T = node_u32(p, 0), nch = node_u32(p, 1);
        if (T == 0 || R % T || nch == 0 || B % nch || C % 256) return GLUE_INCOMPATIBLE_INPUT_SIZE;   // B = kv groups x chunks (the kernel reads the group count from the tensor)
        if (node_u32(p, 4) == 0 && C > 4096) return GLUE_INCOMPATIBLE_INPUT_SIZE;   // decode / tree builds stage a row-chunk in 16 KB of VLM
        if (st.geometry.dims != 4 || st.geometry.maxSizes[0] != 64 || st.geometry.maxSizes[1] != 2 || st.geometry.maxSizes[2] != R || st.geometry.maxSizes[3] != B) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        const bool flat = tree || !node_u32(p, 4);   // decode / tree builds: 1-D index space of R * B row-chunks (balanced over the TPCs)
        if (flat) { out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = R * B; }
        else { out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = R; out->indexSpaceGeometry[1] = B; }
        pattern(out->inputTensorAccessPattern[0], S, -1); if (!flat) { ap_dim(out->inputTensorAccessPattern[0], 1, 0, 1, 0, 0); ap_dim(out->inputTensorAccessPattern[0], 2, 1, 1, 0, 0); }
        pattern(out->inputTensorAccessPattern[1], p->inputTensors[1], -1);
        if (tree) { if (p->inputTensors[2].geometry.maxSizes[0] < T || T > 32) return GLUE_INCOMPATIBLE_INPUT_SIZE; pattern(out->inputTensorAccessPattern[2], p->inputTensors[2], -1); }
        pattern(out->outputTensorAccessPattern[0], p->outputTensors[0], -1); if (!flat) { ap_dim(out->outputTensorAccessPattern[0], 1, 0, 1, 0, 0); ap_dim(out->outputTensorAccessPattern[0], 2, 1, 1, 0, 0); }
        pattern(out->outputTensorAccessPattern[1], st, -1); if (!flat) { ap_dim(out->outputTensorAccessPattern[1], 2, 0, 1, 0, 0); ap_dim(out->outputTensorAccessPattern[1], 3, 1, 1, 0, 0); }
        out->kernel.paramsNr = 5; out->kernel.scalarParams[0] = T; out->kernel.scalarParams[1] = nch; out->kernel.scalarParams[2] = (uint32_t)C;
        out->kernel.scalarParams[3] = node_u32(p, 2); out->kernel.scalarParams[4] = node_u32(p, 3);   // coff, mode (0 when absent)
        if (!tree && node_u32(p, 4)) { out->kernel.paramsNr = 6; out->kernel.scalarParams[5] = node_u32(p, 5); }   // FASTEXP build: + ch0 (node param 5: first chunk of a chunk group)
        return put_elf(out, tree ? "GAUDIUM_ASOFTCT_ELF" : node_u32(p, 4) == 3 ? "GAUDIUM_ASOFTCX5_ELF" : node_u32(p, 4) == 2 ? "GAUDIUM_ASOFTCX4_ELF" : node_u32(p, 4) ? "GAUDIUM_ASOFTCX_ELF" : "GAUDIUM_ASOFTC_ELF");   // node param 4: 1 = the -DFASTEXP build (prefill recipes), 2 = + the bf16-grade exp
    }
    if (g == "gaudium_attn_softmax16") {   // half-width prefill core: scores bf16 {C,R,B} (1/512 octaves), cursor i32 {1} -> P bf16 {C,R,B} (natural key order), stats f32 {64,2,R,B}; params: T, nch, coff, pbase
        if (p->inputTensorNr != 2) { p->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 2) { p->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const bool p8 = p->outputTensors[0].geometry.dataType == DATA_F8_143;   // the -DPF8 build: P as FP8 (E4M3), 256 x exp
        const TensorDataType in[2] = {DATA_BF16, DATA_I32}, ot[2] = {p8 ? DATA_F8_143 : DATA_BF16, DATA_F32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& S = p->inputTensors[0]; const Tensor& st = p->outputTensors[1];
        if (S.geometry.dims != 3) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        uint64_t C = S.geometry.maxSizes[0], R = S.geometry.maxSizes[1], B = S.geometry.maxSizes[2];
        uint32_t T = node_u32(p, 0), nch = node_u32(p, 1);
        if (T == 0 || R % T || nch == 0 || B % nch) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (C == 0 || C % 512 || C > 4096) return GLUE_INCOMPATIBLE_INPUT_SIZE;   // four 128-lane vectors per iteration; two rows staged in 16 KB of VLM
        if (st.geometry.dims != 4 || st.geometry.maxSizes[0] != 64 || st.geometry.maxSizes[1] != 2 || st.geometry.maxSizes[2] != R || st.geometry.maxSizes[3] != B) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = R; out->indexSpaceGeometry[1] = B;
        pattern(out->inputTensorAccessPattern[0], S, -1); pattern(out->inputTensorAccessPattern[1], p->inputTensors[1], -1);
        pattern(out->outputTensorAccessPattern[0], p->outputTensors[0], -1); pattern(out->outputTensorAccessPattern[1], st, -1);
        out->kernel.paramsNr = 4; out->kernel.scalarParams[0] = T; out->kernel.scalarParams[1] = nch; out->kernel.scalarParams[2] = node_u32(p, 2); out->kernel.scalarParams[3] = node_u32(p, 3);
        return put_elf(out, p8 ? "GAUDIUM_ASOFT16P8_ELF" : "GAUDIUM_ASOFT16_ELF");
    }
    if (g == "gaudium_attn_qs16") {   // queries bf16 {DH,R,B} x 512 log2 e (the score unit of gaudium_attn_softmax16)
        if (p->inputTensorNr != 1) { p->inputTensorNr = 1; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 1) { p->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[1] = {DATA_BF16}, ot[1] = {DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& Q = p->inputTensors[0]; const Tensor& O = p->outputTensors[0];
        if (Q.geometry.dims != 3 || O.geometry.dims != 3) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        for (int d = 0; d < 3; ++d) if (Q.geometry.maxSizes[d] != O.geometry.maxSizes[d]) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (Q.geometry.maxSizes[0] % 128) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = Q.geometry.maxSizes[1]; out->indexSpaceGeometry[1] = Q.geometry.maxSizes[2];
        pattern(out->inputTensorAccessPattern[0], Q, -1); pattern(out->outputTensorAccessPattern[0], O, -1);
        out->kernel.paramsNr = 0;
        return put_elf(out, "GAUDIUM_AQS16_ELF");
    }
    if (g == "gaudium_attn_vnat") {   // V chunks bf16 {DH,C,B} in cache row order -> the same in natural key order (loads and stores only)
        if (p->inputTensorNr != 1) { p->inputTensorNr = 1; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 1) { p->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[1] = {DATA_BF16}, ot[1] = {DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& V = p->inputTensors[0]; const Tensor& O = p->outputTensors[0];
        if (V.geometry.dims != 3 || O.geometry.dims != 3) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        for (int d = 0; d < 3; ++d) if (V.geometry.maxSizes[d] != O.geometry.maxSizes[d]) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (V.geometry.maxSizes[0] % 128 || V.geometry.maxSizes[1] % 128) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = V.geometry.maxSizes[1] / 128; out->indexSpaceGeometry[1] = V.geometry.maxSizes[2];
        pattern(out->inputTensorAccessPattern[0], V, -1); pattern(out->outputTensorAccessPattern[0], O, -1);
        if (node_u32(p, 0) && V.geometry.maxSizes[1] % 256) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->kernel.paramsNr = 1; out->kernel.scalarParams[0] = node_u32(p, 0);   // key order: 0 natural, 1 the FP8 P's byte order
        return put_elf(out, "GAUDIUM_AVNAT_ELF");
    }
    if (g == "gaudium_attn_combine") {   // ao_part f32 {256,R,B}, stats f32 {64,2,R,B} [, ao_f {256,R,G}, stats_f {64,2,R,G}] -> ao f32 {256,R,G}; params: nch
        const bool fresh = p->inputTensorNr == 4;
        if (p->inputTensorNr != 2 && p->inputTensorNr != 4) { p->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 1) { p->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[4] = {DATA_F32, DATA_F32, DATA_F32, DATA_F32}, ot[1] = {DATA_F32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& A = p->inputTensors[0]; const Tensor& st = p->inputTensors[1]; const Tensor& O = p->outputTensors[0];
        uint32_t nch = node_u32(p, 0);
        if (O.geometry.dims != 3 || O.geometry.maxSizes[0] != 256) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        uint64_t R = O.geometry.maxSizes[1], G = O.geometry.maxSizes[2];
        if (A.geometry.dims != 3 || A.geometry.maxSizes[0] != 256 || A.geometry.maxSizes[1] != R || nch == 0 || A.geometry.maxSizes[2] != G * nch) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (st.geometry.dims != 4 || st.geometry.maxSizes[2] != R || st.geometry.maxSizes[3] != G * nch) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (fresh) { const Tensor& Af = p->inputTensors[2]; const Tensor& sf = p->inputTensors[3];
            if (Af.geometry.dims != 3 || Af.geometry.maxSizes[0] != 256 || Af.geometry.maxSizes[1] != R || Af.geometry.maxSizes[2] != G) return GLUE_INCOMPATIBLE_INPUT_SIZE;
            if (sf.geometry.dims != 4 || sf.geometry.maxSizes[2] != R || sf.geometry.maxSizes[3] != G) return GLUE_INCOMPATIBLE_INPUT_SIZE; }
        out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = R; out->indexSpaceGeometry[1] = G;
        for (uint32_t i = 0; i < p->inputTensorNr; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        pattern(out->outputTensorAccessPattern[0], O, -1);
        out->kernel.paramsNr = 1; out->kernel.scalarParams[0] = nch;
        if (!fresh && node_u32(p, 1)) return put_elf(out, "GAUDIUM_ACOMBX_ELF");   // node param 1 = 1: the -DUNROLL prefill build (bit-identical)
        return put_elf(out, fresh ? "GAUDIUM_ACOMBF_ELF" : "GAUDIUM_ACOMB_ELF");
    }
    if (g == "gaudium_attn_post_f32") {   // ao f32 {256, 6T, 4} (lane order), qg bf16 {12288, T} -> core bf16 {6144, T}
        if (p->inputTensorNr != 2) { p->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 1) { p->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[2] = {DATA_F32, DATA_BF16}, ot[1] = {DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        uint64_t T = p->inputTensors[1].geometry.maxSizes[1];
        if (p->inputTensors[0].geometry.dims != 3 || p->inputTensors[0].geometry.maxSizes[0] != 256 || p->inputTensors[0].geometry.maxSizes[2] == 0) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const uint64_t NHs = p->inputTensors[1].geometry.maxSizes[0] / 512, NKVs = p->inputTensors[0].geometry.maxSizes[2];   // shape: qg {2 x 256 NH, T}, ao {256, GQ T, NKV}
        if (NHs == 0 || NHs % NKVs || p->inputTensors[0].geometry.maxSizes[1] != (NHs / NKVs) * T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = NHs;
        pattern(out->inputTensorAccessPattern[0], p->inputTensors[0], -1); pattern(out->inputTensorAccessPattern[1], p->inputTensors[1], -1);
        pattern(out->outputTensorAccessPattern[0], p->outputTensors[0], 0, 256);
        out->kernel.paramsNr = 1; out->kernel.scalarParams[0] = (uint32_t)T;
        return put_elf_tag(out, "GAUDIUM_APOSTF_ELF", att_tag(NKVs, NHs / NKVs));
    }
    if (g == "gaudium_vtest_f32") {
        if (p->inputTensorNr != 1 || p->outputTensorNr != 1) return GLUE_INCOMPATIBLE_INPUT_COUNT;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = 1;
        pattern(out->inputTensorAccessPattern[0], p->inputTensors[0], -1);
        pattern(out->outputTensorAccessPattern[0], p->outputTensors[0], -1);
        out->kernel.paramsNr = 1; out->kernel.scalarParams[0] = node_u32(p, 0);
        return put_elf(out, "GAUDIUM_VTEST_ELF");
    }
    if (g == "gaudium_vtest_bf16") {
        if (p->inputTensorNr != 2 || p->outputTensorNr != 2) return GLUE_INCOMPATIBLE_INPUT_COUNT;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = 1;
        for (int i = 0; i < 2; ++i) { pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1); pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1); }
        out->kernel.paramsNr = 0;
        return put_elf(out, "GAUDIUM_VTESTBF_ELF");
    }
    if (g == "gaudium_fused_add_norm") {   // x {HID,T} bf16, res {HID,T} bf16, w {HID} bf16 -> sum {HID,T} bf16, y {HID,T} bf16; scalar: T
        //   (opt/fused-add-rmsnorm, ported in review/anq-on-main): sum = bf16(x + res), y = rmsnorm(sum) * (1 + w), add_norm_gaudi2.c
        if (p->inputTensorNr != 3) { p->inputTensorNr = 3; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 2) { p->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[3] = {DATA_BF16, DATA_BF16, DATA_BF16}, ot[2] = {DATA_BF16, DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& X = p->inputTensors[0];
        // HID % 256 == 0 and HID <= 6144: the kernel keeps the row's rounded sums in a 24-block static VLM array
        if (X.geometry.dims != 2 || X.geometry.maxSizes[0] % 256 || X.geometry.maxSizes[0] > 6144) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        uint64_t HID = X.geometry.maxSizes[0], T = X.geometry.maxSizes[1];
        if (p->inputTensors[2].geometry.dims != 1 || p->inputTensors[2].geometry.maxSizes[0] != HID) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (p->inputTensors[1].geometry.maxSizes[0] != HID || p->inputTensors[1].geometry.maxSizes[1] != T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        for (int i = 0; i < 2; ++i) if (p->outputTensors[i].geometry.maxSizes[0] != HID || p->outputTensors[i].geometry.maxSizes[1] != T) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = T;
        pattern(out->inputTensorAccessPattern[0], p->inputTensors[0], 1);
        pattern(out->inputTensorAccessPattern[1], p->inputTensors[1], 1);
        pattern(out->inputTensorAccessPattern[2], p->inputTensors[2], -1);
        pattern(out->outputTensorAccessPattern[0], p->outputTensors[0], 1);
        pattern(out->outputTensorAccessPattern[1], p->outputTensors[1], 1);
        out->kernel.paramsNr = 1; out->kernel.scalarParams[0] = node_u32(p, 0);
        return put_elf(out, node_u32(p, 1) ? "GAUDIUM_ANORMP_ELF" : "GAUDIUM_ANORM_ELF");   // node param 1 = 1: plain weights (Qwen3 norms, -DPLAIN_W)
    }
    if (g == "gaudium_zero") {   // -> out {64, R} f32, R % 1024 == 0 (the engine's state-block reset, zero_gaudi2.c); no inputs
        if (p->inputTensorNr != 0) { p->inputTensorNr = 0; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 1) { p->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const Tensor& o = p->outputTensors[0];
        if (o.geometry.dataType != DATA_F32) { p->outputTensors[0].geometry.dataType = DATA_F32; return GLUE_INCOMPATIBLE_DATA_TYPE; }
        if (o.geometry.dims != 2 || o.geometry.maxSizes[0] != 64 || o.geometry.maxSizes[1] % 1024) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = o.geometry.maxSizes[1] / 1024;
        pattern(out->outputTensorAccessPattern[0], o, 1, 1024);
        out->kernel.paramsNr = 0;
        return put_elf(out, "GAUDIUM_ZERO_ELF");
    }
    if (g == "gaudium_rmw_add") {  // x, b {C,R} -> y {C,R} (RMW section): y = b, then y (op)= x twice via ST_TNSR RMW; params: mode, op
        if (p->inputTensorNr != 2 || p->outputTensorNr != 1) return GLUE_INCOMPATIBLE_INPUT_COUNT;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = p->outputTensors[0].geometry.maxSizes[1];
        pattern(out->inputTensorAccessPattern[0], p->inputTensors[0], 1);
        pattern(out->inputTensorAccessPattern[1], p->inputTensors[1], 1);
        pattern(out->outputTensorAccessPattern[0], p->outputTensors[0], 1);
        out->kernel.paramsNr = 2; out->kernel.scalarParams[0] = node_u32(p, 0); out->kernel.scalarParams[1] = node_u32(p, 1);
        return put_elf(out, "GAUDIUM_RMW_ELF");
    }
    if (g == "gaudium_accept_e") {   // ids {K+1}, tokens {1,K+1}, cursor {1} [, a_in {2} gated] -> a {1}, cursor_out {1}, emitted {K+1}, ncommit {1}
        const bool nv = p->inputTensorNr == 4;   // gated variant (prompt-lookup pg rows): + a_in {2} = [a, valid rows]
        if (p->inputTensorNr != 3 && p->inputTensorNr != 4) { p->inputTensorNr = 3; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 4) { p->outputTensorNr = 4; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[4] = {DATA_I32, DATA_I32, DATA_I32, DATA_I32}, ot[4] = {DATA_I32, DATA_I32, DATA_I32, DATA_I32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = 1;
        for (uint32_t i = 0; i < p->inputTensorNr; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        for (int i = 0; i < 4; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
        out->kernel.paramsNr = 0;
        return put_elf(out, nv ? "GAUDIUM_ACCEPTG_ELF" : "GAUDIUM_ACCEPT_ELF");
    }
    if (g == "gaudium_sel") {   // tok {1,N}, idx {1} -> out {M}
        if (p->inputTensorNr != 2) { p->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 1) { p->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[2] = {DATA_I32, DATA_I32}, ot[1] = {DATA_I32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = 1;
        for (int i = 0; i < 2; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        pattern(out->outputTensorAccessPattern[0], p->outputTensors[0], -1);
        out->kernel.paramsNr = 0;
        return put_elf(out, "GAUDIUM_SEL_ELF");
    }
    if (g == "gaudium_next_ids") {   // tokens {1,K+1}, a {1}, tokens_mtp {1,K+1}, drafts_rec {1,K-1} -> next_ids {K+1}
        if (p->inputTensorNr != 4) { p->inputTensorNr = 4; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 1) { p->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[4] = {DATA_I32, DATA_I32, DATA_I32, DATA_I32}, ot[1] = {DATA_I32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = 1;
        for (int i = 0; i < 4; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        pattern(out->outputTensorAccessPattern[0], p->outputTensors[0], -1);
        out->kernel.paramsNr = 0;
        return put_elf(out, "GAUDIUM_NEXTIDS_ELF");
    }
    if (g == "gaudium_attn_softmax_fresh" && p->inputTensorNr == 4) {   // fp8 P (+ kv8 f32 {16} last)
        if (p->outputTensorNr != 2) { p->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType f8 = p->outputTensors[0].geometry.dataType == DATA_F8_152 ? DATA_F8_152 : DATA_F8_143;
        const TensorDataType in[4] = {DATA_F32, DATA_I32, DATA_I32, DATA_F32}, ot[2] = {f8, DATA_F32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& S = p->inputTensors[0]; const Tensor& st = p->outputTensors[1];
        if (S.geometry.dims != 3 || p->inputTensors[3].geometry.maxSizes[0] < 18) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        uint64_t NF = S.geometry.maxSizes[0], R = S.geometry.maxSizes[1], G = S.geometry.maxSizes[2];
        uint32_t T = node_u32(p, 0);
        if (T == 0 || R % T || NF % 256 || p->inputTensors[1].geometry.maxSizes[0] != NF) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (st.geometry.dims != 4 || st.geometry.maxSizes[0] != 64 || st.geometry.maxSizes[1] != 2 || st.geometry.maxSizes[2] != R || st.geometry.maxSizes[3] != G) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = R; out->indexSpaceGeometry[1] = G;
        for (int i = 0; i < 4; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        for (int i = 0; i < 2; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
        out->kernel.paramsNr = 4; out->kernel.scalarParams[0] = T; out->kernel.scalarParams[1] = (uint32_t)NF; out->kernel.scalarParams[2] = node_u32(p, 1); out->kernel.scalarParams[3] = f8 == DATA_F8_152 ? 1u : 0u;
        return put_elf(out, "GAUDIUM_ASOFTF8_ELF");
    }
    if (g == "gaudium_attn_softmax_fresh") {   // scores f32 {NF,R,4}, fpos i32 {NF}, cursor i32 {1} -> P bf16 {NF,R,4}, stats f32 {64,2,R,4}; params: T, coff
        if (p->inputTensorNr != 3) { p->inputTensorNr = 3; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 2) { p->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[3] = {DATA_F32, DATA_I32, DATA_I32}, ot[2] = {DATA_BF16, DATA_F32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& S = p->inputTensors[0]; const Tensor& st = p->outputTensors[1];
        if (S.geometry.dims != 3) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        uint64_t NF = S.geometry.maxSizes[0], R = S.geometry.maxSizes[1], G = S.geometry.maxSizes[2];
        uint32_t T = node_u32(p, 0);
        if (T == 0 || R % T || NF % 256 || p->inputTensors[1].geometry.maxSizes[0] != NF) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (st.geometry.dims != 4 || st.geometry.maxSizes[0] != 64 || st.geometry.maxSizes[1] != 2 || st.geometry.maxSizes[2] != R || st.geometry.maxSizes[3] != G) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = R; out->indexSpaceGeometry[1] = G;
        for (int i = 0; i < 3; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        for (int i = 0; i < 2; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
        out->kernel.paramsNr = 3; out->kernel.scalarParams[0] = T; out->kernel.scalarParams[1] = (uint32_t)NF; out->kernel.scalarParams[2] = node_u32(p, 1);
        return put_elf(out, "GAUDIUM_ASOFTF_ELF");
    }
    if (g == "gaudium_qsa_prep") {   // iqk bf16 {640,T}, iqn_w, ikn_w f32 {128}, rope_c, rope_s f32 {128,Lmax}, cursor i32 {1} [, ikf_in bf16 {128,NF}]
        const bool fresh = p->inputTensorNr == 7;
        if (p->inputTensorNr != 6 && p->inputTensorNr != 7) { p->inputTensorNr = 6; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != (fresh ? 2u : 3u)) { p->outputTensorNr = fresh ? 2 : 3; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[7] = {DATA_BF16, DATA_F32, DATA_F32, DATA_F32, DATA_F32, DATA_I32, DATA_BF16}, ot[3] = {DATA_BF16, DATA_BF16, DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& X = p->inputTensors[0]; const Tensor& Q = p->outputTensors[0];
        const uint64_t T = X.geometry.maxSizes[1];
        if (X.geometry.maxSizes[0] != 640 || Q.geometry.maxSizes[0] != 128 || Q.geometry.maxSizes[1] != 4 * T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (p->inputTensors[3].geometry.maxSizes[0] != 128) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const Tensor& KC = p->outputTensors[1];
        if (KC.geometry.maxSizes[0] != 128) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (!fresh && (p->outputTensors[2].geometry.maxSizes[0] != 128 || p->outputTensors[2].geometry.maxSizes[1] * 4 < KC.geometry.maxSizes[1])) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (fresh && (p->inputTensors[6].geometry.maxSizes[1] != KC.geometry.maxSizes[1] || T > KC.geometry.maxSizes[1])) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = fresh ? 8 : T;
        for (uint32_t i = 0; i < p->inputTensorNr; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        for (uint32_t i = 0; i < p->outputTensorNr; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
        out->kernel.paramsNr = fresh ? 2 : 1; out->kernel.scalarParams[0] = node_u32(p, 0); out->kernel.scalarParams[1] = node_u32(p, 1);   // coff [, sbase]
        return put_elf(out, fresh ? "GAUDIUM_QSAPREPF_ELF" : "GAUDIUM_QSAPREP_ELF");
    }
    if (g == "gaudium_qsa_commit") {   // ikf bf16 {128,NF}, ncommit i32 {1}, cursor i32 {1}, ikn_w f32 {128}, rope_c, rope_s -> ikcache, ibcache
        if (p->inputTensorNr != 6) { p->inputTensorNr = 6; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 2) { p->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[6] = {DATA_BF16, DATA_I32, DATA_I32, DATA_F32, DATA_F32, DATA_F32}, ot[2] = {DATA_BF16, DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (p->inputTensors[0].geometry.maxSizes[0] != 128 || p->outputTensors[0].geometry.maxSizes[0] != 128 || p->outputTensors[1].geometry.maxSizes[0] != 128) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = 16;
        for (uint32_t i = 0; i < 6; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        for (uint32_t i = 0; i < 2; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
        out->kernel.paramsNr = 0;
        return put_elf(out, "GAUDIUM_QSACOMMIT_ELF");
    }
    if (g == "gaudium_qsa_score" || g == "gaudium_qsa_select") {   // score: isc f32 {NB,4T} -> iss f32 {NB,T}; select: iss -> qsel i32 {64,NW,T}; + cursor, cursor_big i32 {1}
        const bool sel = g == "gaudium_qsa_select";
        const bool bmp = sel && p->outputTensorNr == 2;   // + qbm i32 {64, NBW, T}: the -DBITMAP build
        if (p->inputTensorNr != 3) { p->inputTensorNr = 3; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != (bmp ? 2u : 1u)) { p->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[3] = {DATA_F32, DATA_I32, DATA_I32}, ot[2] = {sel ? DATA_I32 : DATA_F32, DATA_I32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& X = p->inputTensors[0]; const Tensor& O = p->outputTensors[0];
        const uint64_t NB = X.geometry.maxSizes[0];
        const uint64_t T = sel ? X.geometry.maxSizes[1] : O.geometry.maxSizes[1];
        if (X.geometry.dims != 2 || T == 0) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (!sel && (X.geometry.maxSizes[1] != 4 * T || O.geometry.maxSizes[0] != NB)) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (sel && (O.geometry.dims != 3 || O.geometry.maxSizes[0] != 64 || O.geometry.maxSizes[2] != T || O.geometry.maxSizes[1] * 512 < NB)) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (sel) { out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = T; }
        else { out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = (NB + 1023) / 1024; out->indexSpaceGeometry[1] = T; }
        for (uint32_t i = 0; i < 3; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        pattern(out->outputTensorAccessPattern[0], O, -1);
        if (bmp) { const Tensor& B = p->outputTensors[1];
            if (B.geometry.dims != 3 || B.geometry.maxSizes[0] != 64 || B.geometry.maxSizes[2] != T || B.geometry.maxSizes[1] * 2048 < NB) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
            pattern(out->outputTensorAccessPattern[1], B, -1); }
        out->kernel.paramsNr = 3; out->kernel.scalarParams[0] = node_u32(p, 0); out->kernel.scalarParams[1] = node_u32(p, 1);
        out->kernel.scalarParams[2] = node_u32(p, 2) ? node_u32(p, 2) : 512u;   // coff, mode, nblk (default 512)
        return put_elf(out, bmp ? "GAUDIUM_QSASELB_ELF" : sel ? "GAUDIUM_QSASEL_ELF" : "GAUDIUM_QSASCORE_ELF");
    }
    if (g == "gaudium_qsa_gather") {   // qb, kcache, vcache bf16, qbm i32, cursor i32 -> q2, kg, vg bf16; node params: coff, nblk
        if (p->inputTensorNr != 5) { p->inputTensorNr = 5; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 3) { p->outputTensorNr = 3; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[5] = {DATA_BF16, DATA_BF16, DATA_BF16, DATA_I32, DATA_I32}, ot[3] = {DATA_BF16, DATA_BF16, DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& qb = p->inputTensors[0]; const Tensor& kc = p->inputTensors[1]; const Tensor& vc = p->inputTensors[2]; const Tensor& bm = p->inputTensors[3];
        const Tensor& q2 = p->outputTensors[0]; const Tensor& kg = p->outputTensors[1]; const Tensor& vg = p->outputTensors[2];
        const uint64_t NKVs = qb.geometry.maxSizes[2], GQs = q2.geometry.maxSizes[1], T = bm.geometry.maxSizes[2], C = kc.geometry.maxSizes[1], G = kg.geometry.maxSizes[1];
        const uint32_t nblk = node_u32(p, 1) ? node_u32(p, 1) : 512u;
        if (qb.geometry.dims != 3 || qb.geometry.maxSizes[0] != 256 || NKVs == 0 || GQs == 0 || T == 0 || qb.geometry.maxSizes[1] != GQs * T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (kc.geometry.dims != 3 || kc.geometry.maxSizes[0] != 256 || (C & (C - 1)) || C < 128 || kc.geometry.maxSizes[2] % NKVs || vc.geometry.maxSizes[1] != C || vc.geometry.maxSizes[2] != kc.geometry.maxSizes[2]) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (bm.geometry.dims != 3 || bm.geometry.maxSizes[0] != 64) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (q2.geometry.dims != 3 || q2.geometry.maxSizes[0] != 256 || q2.geometry.maxSizes[2] != NKVs * T) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (kg.geometry.dims != 3 || kg.geometry.maxSizes[0] != 256 || G % 128 || G < 4ull * nblk + 4 || kg.geometry.maxSizes[2] != NKVs * T || vg.geometry.maxSizes[1] != G || vg.geometry.maxSizes[2] != NKVs * T) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        uint32_t lc = 0; while ((1ull << lc) < C) ++lc;
        uint64_t NS = 48 / T; NS = NS < 1 ? 1 : NS > 24 ? 24 : NS;   // rank slices: >= ~24-48 index points for decode-sized passes
        out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = T; out->indexSpaceGeometry[1] = NS;
        for (uint32_t i = 0; i < 5; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        for (uint32_t i = 0; i < 3; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
        out->kernel.paramsNr = 5; out->kernel.scalarParams[0] = (uint32_t)T; out->kernel.scalarParams[1] = node_u32(p, 0);
        out->kernel.scalarParams[2] = nblk; out->kernel.scalarParams[3] = (uint32_t)NS; out->kernel.scalarParams[4] = lc;
        return put_elf(out, "GAUDIUM_QSAGATHER_ELF");
    }
    if (g == "gaudium_qsa_sp_softmax") {   // scores f32 {G, GQ, BT}, cursor i32 -> P bf16 {G, GQ, BT}, stats f32 {64, 2, GQ, BT}; node params: T, coff, nblk
        if (p->inputTensorNr != 2) { p->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 2) { p->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[2] = {DATA_F32, DATA_I32}, ot[2] = {DATA_BF16, DATA_F32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& X = p->inputTensors[0]; const Tensor& S = p->outputTensors[1];
        const uint64_t G = X.geometry.maxSizes[0], GQs = X.geometry.maxSizes[1], BT = X.geometry.maxSizes[2];
        if (X.geometry.dims != 3 || G % 128 || node_u32(p, 0) == 0 || BT % node_u32(p, 0)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (S.geometry.dims != 4 || S.geometry.maxSizes[0] != 64 || S.geometry.maxSizes[1] != 2 || S.geometry.maxSizes[2] != GQs || S.geometry.maxSizes[3] != BT) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = GQs * BT;
        for (uint32_t i = 0; i < 2; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        for (uint32_t i = 0; i < 2; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
        out->kernel.paramsNr = 3; out->kernel.scalarParams[0] = node_u32(p, 0); out->kernel.scalarParams[1] = node_u32(p, 1);
        out->kernel.scalarParams[2] = node_u32(p, 2) ? node_u32(p, 2) : 512u;
        return put_elf(out, "GAUDIUM_QSASPSOFT_ELF");
    }
    if (g == "gaudium_qsa_sp_out") {   // ao_part f32 {256, GQ, BT}, stats f32, cursor i32 [, ao_d f32 {256, GQ T, NKV}] -> ao f32 {256, GQ T, NKV}; node params: T, coff, dlim
        const bool d0 = p->inputTensorNr == 4;
        if (p->inputTensorNr != 3 && !d0) { p->inputTensorNr = 3; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 1) { p->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[4] = {DATA_F32, DATA_F32, DATA_I32, DATA_F32}, ot[1] = {DATA_F32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& A = p->inputTensors[0]; const Tensor& O = p->outputTensors[0];
        const uint64_t T = node_u32(p, 0), GQs = A.geometry.maxSizes[1];
        if (A.geometry.dims != 3 || T == 0 || A.geometry.maxSizes[2] % T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const uint64_t NKVs = A.geometry.maxSizes[2] / T;
        if (O.geometry.dims != 3 || O.geometry.maxSizes[1] != GQs * T || O.geometry.maxSizes[2] != NKVs) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (d0 && (p->inputTensors[3].geometry.maxSizes[1] != GQs * T || p->inputTensors[3].geometry.maxSizes[2] != NKVs)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = GQs * T; out->indexSpaceGeometry[1] = NKVs;
        for (uint32_t i = 0; i < p->inputTensorNr; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        pattern(out->outputTensorAccessPattern[0], O, -1);
        out->kernel.paramsNr = 3; out->kernel.scalarParams[0] = (uint32_t)T; out->kernel.scalarParams[1] = node_u32(p, 1); out->kernel.scalarParams[2] = node_u32(p, 2);
        return put_elf(out, d0 ? "GAUDIUM_QSASPOUTD_ELF" : "GAUDIUM_QSASPOUT_ELF");
    }
    if (g == "gaudium_kv_commit" && p->outputTensorNr == 2 && is_f8(p->outputTensors[0].geometry.dataType)) {   // fp8 window -> fp8 cache
        if (p->inputTensorNr != 4) { p->inputTensorNr = 4; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        const TensorDataType f8 = p->outputTensors[0].geometry.dataType;
        const TensorDataType in[4] = {f8, f8, DATA_I32, DATA_I32}, ot[2] = {f8, f8};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& kc = p->outputTensors[0];
        if (kc.geometry.dims != 3 || kc.geometry.maxSizes[0] != 256 || kc.geometry.maxSizes[1] % 256) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        const uint64_t NKVs = p->inputTensors[0].geometry.maxSizes[2];   // shape: kv groups of the window kf {256, NF, NKV}
        if (NKVs == 0 || kc.geometry.maxSizes[2] % NKVs) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = NKVs;
        for (int i = 0; i < 4; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        for (int i = 0; i < 2; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
        out->kernel.paramsNr = 1; out->kernel.scalarParams[0] = (uint32_t)kc.geometry.maxSizes[1];
        return put_elf_tag(out, "GAUDIUM_KVCOMMIT8_ELF", att_tag(NKVs, 0));   // kv_commit has no GQ-dependent code: tagged by NKV only
    }
    if (g == "gaudium_kv_commit") {   // kf, vf bf16 {256,NF,4}, ncommit i32 {1}, cursor i32 {1} -> kcache, vcache bf16 {256,C,4*nch}
        if (p->inputTensorNr != 4) { p->inputTensorNr = 4; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 2) { p->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[4] = {DATA_BF16, DATA_BF16, DATA_I32, DATA_I32}, ot[2] = {DATA_BF16, DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& kc = p->outputTensors[0];
        if (kc.geometry.dims != 3 || kc.geometry.maxSizes[0] != 256 || kc.geometry.maxSizes[1] % 128) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        const uint64_t NKVs = p->inputTensors[0].geometry.maxSizes[2];   // shape: kv groups of the window kf {256, NF, NKV}
        if (NKVs == 0 || kc.geometry.maxSizes[2] % NKVs) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = NKVs;
        for (int i = 0; i < 4; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        for (int i = 0; i < 2; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
        out->kernel.paramsNr = 1; out->kernel.scalarParams[0] = (uint32_t)kc.geometry.maxSizes[1];
        return put_elf_tag(out, "GAUDIUM_KVCOMMIT_ELF", att_tag(NKVs, 0));   // kv_commit has no GQ-dependent code: tagged by NKV only
    }
    if (g == "gaudium_fpos") {   // cursor i32 {1}, a i32 {1} -> fpos_c, fpos_d i32 {NF}; params: K, Kd
        if (p->inputTensorNr != 2) { p->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 2) { p->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[2] = {DATA_I32, DATA_I32}, ot[2] = {DATA_I32, DATA_I32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = 1;
        for (int i = 0; i < 2; ++i) { pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1); pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1); }
        out->kernel.paramsNr = 3; out->kernel.scalarParams[0] = node_u32(p, 0); out->kernel.scalarParams[1] = node_u32(p, 1); out->kernel.scalarParams[2] = (uint32_t)p->outputTensors[0].geometry.maxSizes[0];
        return put_elf(out, "GAUDIUM_FPOS_ELF");
    }
    if (g == "gaudium_pld_scan") {   // hist_in {Lmax}, cursor {1}, a {1}, ids {Tv}, chain {K+1} -> part {4, NM}; one index point per member
        if (p->inputTensorNr != 5) { p->inputTensorNr = 5; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 1) { p->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[5] = {DATA_I32, DATA_I32, DATA_I32, DATA_I32, DATA_I32}, ot[1] = {DATA_I32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = p->outputTensors[0].geometry.maxSizes[1];
        for (int i = 0; i < 5; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        pattern(out->outputTensorAccessPattern[0], p->outputTensors[0], -1);
        out->kernel.paramsNr = 0;
        return put_elf(out, "GAUDIUM_PLDSCAN_ELF");
    }
    if (g == "gaudium_pld_tail") {   // part {4,NM}, hist_in {Lmax}, cursor, a, ids {Tv}, chain {K+1} -> hist_out {Lmax}, next_ids {K+1+L}
        if (p->inputTensorNr != 6) { p->inputTensorNr = 6; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        const bool gate = p->outputTensorNr == 3;   // gated variant: + a_next {2}
        if (p->outputTensorNr != 2 && p->outputTensorNr != 3) { p->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[6] = {DATA_I32, DATA_I32, DATA_I32, DATA_I32, DATA_I32, DATA_I32}, ot[3] = {DATA_I32, DATA_I32, DATA_I32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = 1;
        for (int i = 0; i < 6; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        for (uint32_t i = 0; i < p->outputTensorNr; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
        out->kernel.paramsNr = 0;
        return put_elf(out, gate ? "GAUDIUM_PLDTAILG_ELF" : "GAUDIUM_PLDTAIL_ELF");
    }
    if (g == "gaudium_cert_hq") {   // h bf16 {HID,M} -> hq bf16 {HID,2M}, coef f32 {64,8,M}; params {cs_bits}
        if (p->inputTensorNr != 1) { p->inputTensorNr = 1; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 2) { p->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[1] = {DATA_BF16}, ot[2] = {DATA_BF16, DATA_F32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const uint64_t M = p->inputTensors[0].geometry.maxSizes[1];
        if (p->outputTensors[0].geometry.maxSizes[1] != 2 * M || p->inputTensors[0].geometry.maxSizes[0] % 128) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = M;
        pattern(out->inputTensorAccessPattern[0], p->inputTensors[0], -1);
        for (int i = 0; i < 2; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
        out->kernel.paramsNr = 2; out->kernel.scalarParams[0] = node_u32(p, 0); out->kernel.scalarParams[1] = (uint32_t)M;
        return put_elf(out, "GAUDIUM_CERTHQ_ELF");
    }
    if (g == "gaudium_cert_mark") {   // S f32 {V,2M}, nv f32 {V}, ev f32 {V}, coef f32 {64,8,M}, La f32 {2M,M}, nA f32 {1,2M} -> f f32 {V}, vf f32 {64,V/64}
        if (p->inputTensorNr != 6) { p->inputTensorNr = 6; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 2) { p->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[6] = {DATA_F32, DATA_F32, DATA_F32, DATA_F32, DATA_F32, DATA_F32}, ot[2] = {DATA_F32, DATA_F32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const uint64_t M = p->inputTensors[3].geometry.maxSizes[2];
        if (M > 18 || p->inputTensors[0].geometry.maxSizes[1] != 2 * M || p->inputTensors[0].geometry.maxSizes[0] % 64) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const uint32_t NM = 24;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = NM;
        for (int i = 0; i < 6; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        for (int i = 0; i < 2; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
        out->kernel.paramsNr = 2; out->kernel.scalarParams[0] = (uint32_t)M; out->kernel.scalarParams[1] = NM;
        return put_elf(out, "GAUDIUM_CERTMARK_ELF");
    }
    if (g == "gaudium_cert_compact") {   // f f32 {V}, vf f32 {64,NV} -> cidx i32 {BS*NM}, ccnt i32 {NM}
        if (p->inputTensorNr != 2) { p->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 2) { p->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[2] = {DATA_F32, DATA_F32}, ot[2] = {DATA_I32, DATA_I32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const uint32_t NM = (uint32_t)p->outputTensors[1].geometry.maxSizes[0];
        if (p->outputTensors[0].geometry.maxSizes[0] % NM) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = NM;
        for (int i = 0; i < 2; ++i) { pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1); pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1); }
        out->kernel.paramsNr = 1; out->kernel.scalarParams[0] = NM;
        return put_elf(out, "GAUDIUM_CERTCOMPACT_ELF");
    }
    if (g == "gaudium_cert_final") {   // Lc bf16 {B,M}, cidx i32 {B}, ccnt i32 {NM} -> tok i32 {1,M}, diag i32 {4,M}
        if (p->inputTensorNr != 3) { p->inputTensorNr = 3; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 2) { p->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[3] = {DATA_BF16, DATA_I32, DATA_I32}, ot[2] = {DATA_I32, DATA_I32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const uint64_t M = p->inputTensors[0].geometry.maxSizes[1];
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = M;
        for (int i = 0; i < 3; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        for (int i = 0; i < 2; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
        out->kernel.paramsNr = 0;
        return put_elf(out, "GAUDIUM_CERTFINAL_ELF");
    }
    // ---------------- tree verify + DFlash2 drafter ----------------
    // ---------------- feature/mtp-tree: tree manager kernels (model-agnostic) + MTP top-k ----------------
    auto simple = [&](int nin, int nout, const TensorDataType* in, const TensorDataType* ot, uint64_t isg, const char* env, std::vector<uint32_t> prm) -> GlueCodeReturn {
        if ((int)p->inputTensorNr != nin) { p->inputTensorNr = nin; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if ((int)p->outputTensorNr != nout) { p->outputTensorNr = nout; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = isg;
        for (int i = 0; i < nin; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        for (int i = 0; i < nout; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
        out->kernel.paramsNr = (uint32_t)prm.size(); for (size_t i = 0; i < prm.size(); ++i) out->kernel.scalarParams[i] = prm[i];
        return put_elf(out, env); };
    const TensorDataType I3[3] = {DATA_I32, DATA_I32, DATA_I32}, I5[5] = {DATA_I32, DATA_I32, DATA_I32, DATA_I32, DATA_I32};
    if (g == "gaudium_accept_t") {   // ids {N}, tokens {1,N}, cursor, parent {N}, depth {N} -> a, cursor_out, emitted {N}, fix {2}
        const TensorDataType ot[4] = {DATA_I32, DATA_I32, DATA_I32, DATA_I32};
        return simple(5, 4, I5, ot, 1, "GAUDIUM_ACCEPTT_ELF", {});
    }
    if (g == "gaudium_df_conv") {   // x bf16 {H,T}, coef f32 {H/4,T}, base f32 {H,2} -> y bf16; params: foff [, toff (default 320: H = 5120)]
        const TensorDataType in[3] = {DATA_BF16, DATA_F32, DATA_F32}, ot[1] = {DATA_BF16};
        uint64_t T = p->inputTensors[0].geometry.maxSizes[1]; const uint32_t toff = node_u32(p, 1) ? node_u32(p, 1) : 320u;
        return simple(3, 1, in, ot, p->inputTensors[0].geometry.maxSizes[0] / 128, "GAUDIUM_DFCONV_ELF", {(uint32_t)T, node_u32(p, 0), toff});
    }
    if (g == "gaudium_df_attn_prep") {   // q, k, v bf16, qn_w, kn_w, rope_c, rope_s f32, cur i32 -> qb, kblk, vblk bf16
        const TensorDataType in[8] = {DATA_BF16, DATA_BF16, DATA_BF16, DATA_F32, DATA_F32, DATA_F32, DATA_F32, DATA_I32}, ot[3] = {DATA_BF16, DATA_BF16, DATA_BF16};
        uint64_t T = p->inputTensors[0].geometry.maxSizes[1];
        return simple(8, 3, in, ot, 40, "GAUDIUM_DFAPREP_ELF", {(uint32_t)T});
    }
    if (g == "gaudium_df_ctx") {   // kc, vc bf16 {1024,T}, kn_w, rope_c, rope_s f32, cursor, depth, fix i32 -> kcache, vcache bf16 {128,Lmax,8}
        const TensorDataType in[8] = {DATA_BF16, DATA_BF16, DATA_F32, DATA_F32, DATA_F32, DATA_I32, DATA_I32, DATA_I32}, ot[2] = {DATA_BF16, DATA_BF16};
        uint64_t T = p->inputTensors[0].geometry.maxSizes[1];
        return simple(8, 2, in, ot, 8, "GAUDIUM_DFCTX_ELF", {(uint32_t)T});
    }
    if (g == "gaudium_df_softmax") {   // sc f32 {Lmax,R,8}, sb f32 {128,R,8}, cur, fix -> Pc, Pb bf16; params: T, W
        const TensorDataType in[4] = {DATA_F32, DATA_F32, DATA_I32, DATA_I32}, ot[2] = {DATA_BF16, DATA_BF16};
        if (p->inputTensorNr != 4) { p->inputTensorNr = 4; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        uint64_t Lmax = p->inputTensors[0].geometry.maxSizes[0], R = p->inputTensors[0].geometry.maxSizes[1], G = p->inputTensors[0].geometry.maxSizes[2];
        uint32_t T = node_u32(p, 0); if (T == 0 || R % T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        return simple(4, 2, in, ot, R * G, "GAUDIUM_DFSOFT_ELF", {T, (uint32_t)Lmax, node_u32(p, 1)});
    }
    if (g == "gaudium_df_post") {   // ao bf16 {128,4T,8} -> core bf16 {4096,T}
        const TensorDataType in[1] = {DATA_BF16}, ot[1] = {DATA_BF16};
        uint64_t T = p->outputTensors[0].geometry.maxSizes[1];
        return simple(1, 1, in, ot, 32, "GAUDIUM_DFPOST_ELF", {(uint32_t)T});
    }
    if (g == "gaudium_df_topk1") {   // logits bf16 {V,T} -> pv f32 {64,16,CH,T}, pi i32; params: none (CH from the output)
        const TensorDataType in[1] = {DATA_BF16}, ot[2] = {DATA_F32, DATA_I32};
        if (p->outputTensorNr != 2) { p->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        uint64_t V = p->inputTensors[0].geometry.maxSizes[0], T = p->inputTensors[0].geometry.maxSizes[1], CH = p->outputTensors[0].geometry.maxSizes[2];
        if (V % 128) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        return simple(1, 2, in, ot, CH * T, "GAUDIUM_DFTOPK1_ELF", {(uint32_t)CH, (uint32_t)(V / 128)});
    }
    if (g == "gaudium_df_topk2") {   // pv, pi -> ids i32 {16,T}, vals f32 {16,T}
        const TensorDataType in[2] = {DATA_F32, DATA_I32}, ot[2] = {DATA_I32, DATA_F32};
        if (p->inputTensorNr != 2) { p->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        uint64_t CH = p->inputTensors[0].geometry.maxSizes[2], T = p->inputTensors[0].geometry.maxSizes[3];
        return simple(2, 2, in, ot, T, "GAUDIUM_DFTOPK2_ELF", {(uint32_t)CH});
    }
    if (g == "gaudium_df_score") {   // cand i32 {16,T}, unary f32 {16,T}, hproj bf16 {256,T}, pred, succ bf16 {256,V}, ids_blk i32 {T} -> scores f32 {16,16,S}
        const TensorDataType in[6] = {DATA_I32, DATA_F32, DATA_BF16, DATA_BF16, DATA_BF16, DATA_I32}, ot[1] = {DATA_F32};
        if (p->outputTensorNr != 1) { p->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        return simple(6, 1, in, ot, p->outputTensors[0].geometry.maxSizes[2], "GAUDIUM_DFSCORE_ELF", {});
    }
    if (g == "gaudium_df_walk") {   // scores f32 {16,16,S}, cand i32 {16,T} -> drafts, alts, alts2 i32 {S}
        const TensorDataType in[2] = {DATA_F32, DATA_I32};
        return simple(2, 3, in, I3, 1, "GAUDIUM_DFWALK_ELF", {});
    }
    if (g == "gaudium_df_ids") {   // tokens {1,N}, a {1} -> ids_blk {T}; params: mask id
        const TensorDataType in[2] = {DATA_I32, DATA_I32}, ot[1] = {DATA_I32};
        return simple(2, 1, in, ot, 1, "GAUDIUM_DFIDS_ELF", {node_u32(p, 0)});
    }
    if (g == "gaudium_df_next") {   // tokens {1,N}, a, drafts {S}, alts {S}, alts2 {S} -> next_ids {N}
        const TensorDataType ot[1] = {DATA_I32};
        return simple(5, 1, I5, ot, 1, "GAUDIUM_DFNEXT_ELF", {node_u32(p, 0)});   // node param 0: chain length (0 = the drafts' size)
    }
    // ---- DSpine ----
    if (g == "gaudium_ds_softmax") {   // as gaudium_df_softmax, causal inside the block (-DCAUSAL build)
        const TensorDataType in[4] = {DATA_F32, DATA_F32, DATA_I32, DATA_I32}, ot[2] = {DATA_BF16, DATA_BF16};
        if (p->inputTensorNr != 4) { p->inputTensorNr = 4; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        uint64_t Lmax = p->inputTensors[0].geometry.maxSizes[0], R = p->inputTensors[0].geometry.maxSizes[1], G = p->inputTensors[0].geometry.maxSizes[2];
        uint32_t T = node_u32(p, 0); if (T == 0 || R % T || T > 64) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        return simple(4, 2, in, ot, R * G, "GAUDIUM_DSSOFT_ELF", {T, (uint32_t)Lmax, node_u32(p, 1)});
    }
    if (g == "gaudium_ds_read") {   // x bf16 {H,T} -> n bf16 {H,T}, rms f32 {64,T}
        const TensorDataType in[1] = {DATA_BF16}, ot[2] = {DATA_BF16, DATA_F32};
        uint64_t H = p->inputTensors[0].geometry.maxSizes[0], T = p->inputTensors[0].geometry.maxSizes[1];
        if (H % 128) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        return simple(1, 2, in, ot, T, "GAUDIUM_DSREAD_ELF", {(uint32_t)H});
    }
    if (g == "gaudium_ds_codes") {   // z bf16 {C,T}, codebook bf16 {C,V}, ids_blk i32 {T} -> codes bf16 {C,T}
        const TensorDataType in[3] = {DATA_BF16, DATA_BF16, DATA_I32}, ot[1] = {DATA_BF16};
        uint64_t C = p->inputTensors[0].geometry.maxSizes[0], T = p->inputTensors[0].geometry.maxSizes[1];
        if (C % 128) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const float sq = sqrtf((float)C); uint32_t sqb; memcpy(&sqb, &sq, 4);   // sqrt(code_size) as an f32 scalar
        return simple(3, 1, in, ot, T, "GAUDIUM_DSCODES_ELF", {(uint32_t)C, sqb});
    }
    if (g == "gaudium_ds_msg") {   // packed bf16 {3R,T}, gate bias f32 {R} -> msg bf16 {R,T}
        const TensorDataType in[2] = {DATA_BF16, DATA_F32}, ot[1] = {DATA_BF16};
        uint64_t R3 = p->inputTensors[0].geometry.maxSizes[0], T = p->inputTensors[0].geometry.maxSizes[1];
        if (R3 % 384) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        return simple(2, 1, in, ot, T, "GAUDIUM_DSMSG_ELF", {(uint32_t)(R3 / 3)});
    }
    if (g == "gaudium_ds_finish") {   // raw, delta bf16 {H,T}, rms f32 {64,T}, w f32 {H} -> out, normed bf16 {H,T}
        const TensorDataType in[4] = {DATA_BF16, DATA_BF16, DATA_F32, DATA_F32}, ot[2] = {DATA_BF16, DATA_BF16};
        uint64_t H = p->inputTensors[0].geometry.maxSizes[0], T = p->inputTensors[0].geometry.maxSizes[1];
        if (H % 128) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        return simple(4, 2, in, ot, T, "GAUDIUM_DSFINISH_ELF", {(uint32_t)H});
    }
    if (g == "gaudium_ds_base") {   // raw bf16 {H,T}, w_norm f32 {H}, cand i32 {16,T}, W_lm bf16 {H,V} -> base f32 {16,T}
        const TensorDataType in[4] = {DATA_BF16, DATA_F32, DATA_I32, DATA_BF16}, ot[1] = {DATA_F32};
        uint64_t H = p->inputTensors[0].geometry.maxSizes[0], T = p->inputTensors[0].geometry.maxSizes[1];
        if (H % 128 || T < 2) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        return simple(4, 1, in, ot, T - 1, "GAUDIUM_DSBASE_ELF", {(uint32_t)H});
    }
    if (g == "gaudium_ds_score") {   // cand i32 {16,T}, base f32 {16,T}, packed bf16 {3R,T}, rms f32 {64,T}, gb f32 {R}, TSV bf16 {2R,V}, TG bf16 {R,V}, ids i32 {T} -> scores f32 {16,16,S}
        const TensorDataType in[8] = {DATA_I32, DATA_F32, DATA_BF16, DATA_F32, DATA_F32, DATA_BF16, DATA_BF16, DATA_I32}, ot[1] = {DATA_F32};
        if (p->inputTensorNr != 8) { p->inputTensorNr = 8; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        uint64_t R = p->inputTensors[6].geometry.maxSizes[0], S = p->outputTensors[0].geometry.maxSizes[2];
        if (R % 128 || R > 512) return GLUE_INCOMPATIBLE_INPUT_SIZE;   // the kernel keeps up to 4 x 128 rank channels in registers
        return simple(8, 1, in, ot, 16 * S, "GAUDIUM_DSSCORE_ELF", {(uint32_t)R});
    }
    if (g == "gaudium_ds_cwalk") {   // scores f32 {16,16,S}, cand i32 {16,T}, rms f32 {64,T}, spf f32 {8}, spi i32 {8}, base i32 {1}
                                     //   -> drafts, alts, alts2 i32 {S}, cs f32 {64,S}; node param 0: alpha (f32 bits)
        const TensorDataType in[6] = {DATA_F32, DATA_I32, DATA_F32, DATA_F32, DATA_I32, DATA_I32}, ot[4] = {DATA_I32, DATA_I32, DATA_I32, DATA_F32};
        return simple(6, 4, in, ot, 1, "GAUDIUM_DSCWALK_ELF", {node_u32(p, 0)});
    }
    if (g == "gaudium_gdn_prepa") {   // qkvz, conv_w, cs_in {10240,3,NS}, conv_b, slot -> y f32 {128,T,32}, v {128,C,48,NCH}, cs_out, cs_ckpt; params: flags (C = flags >> 16)
        if (p->inputTensorNr != 5) { p->inputTensorNr = 5; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 4) { p->outputTensorNr = 4; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[5] = {DATA_BF16, DATA_F32, DATA_F32, DATA_F32, DATA_I32}, ot[4] = {DATA_F32, DATA_F32, DATA_F32, DATA_F32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (p->inputTensors[2].geometry.dims != 3) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const uint64_t T = p->inputTensors[0].geometry.maxSizes[1]; const uint32_t pflags = node_u32(p, 0), RG = 64;
        if (!(pflags & 32) || p->outputTensors[0].geometry.dims != 3 || p->outputTensors[0].geometry.maxSizes[1] != T || p->outputTensors[0].geometry.maxSizes[2] % 2) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        const uint64_t HKs = p->outputTensors[0].geometry.maxSizes[2] / 2, HVs = p->outputTensors[1].geometry.maxSizes[2];   // shape: y {128, T, 2 HK}, v {128, C, HV, NCH}
        if (HKs == 0 || HVs == 0 || HVs % HKs) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = 2 * HKs + HVs; out->indexSpaceGeometry[1] = (T + RG - 1) / RG;
        for (uint32_t i = 0; i < 5; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        for (int i = 0; i < 4; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
        out->kernel.paramsNr = 3; out->kernel.scalarParams[0] = (uint32_t)T; out->kernel.scalarParams[1] = pflags; out->kernel.scalarParams[2] = RG;
        return put_elf_tag(out, "GAUDIUM_PREPA_ELF", gdn_tag(HKs, HVs / HKs));
    }
    if (g == "gaudium_gdn_prepb") {   // y f32 {128,T,32} -> q, k {128,C,48,NCH}; params: C
        if (p->inputTensorNr != 1) { p->inputTensorNr = 1; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 2) { p->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[1] = {DATA_F32}, ot[2] = {DATA_F32, DATA_F32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const uint64_t T = p->inputTensors[0].geometry.maxSizes[1]; const uint32_t C = node_u32(p, 0), RG = 64;
        if (p->inputTensors[0].geometry.dims != 3 || p->inputTensors[0].geometry.maxSizes[2] % 2 || p->inputTensors[0].geometry.maxSizes[2] == 0 || C == 0 || p->outputTensors[0].geometry.maxSizes[1] != C) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const uint64_t HKs = p->inputTensors[0].geometry.maxSizes[2] / 2, HVs = p->outputTensors[0].geometry.maxSizes[2];   // shape: y {128, T, 2 HK} -> q, k {128, C, HV, NCH}
        if (HVs == 0 || HVs % HKs) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = 2 * HKs; out->indexSpaceGeometry[1] = (T + RG - 1) / RG;
        pattern(out->inputTensorAccessPattern[0], p->inputTensors[0], -1);
        for (int i = 0; i < 2; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
        out->kernel.paramsNr = 3; out->kernel.scalarParams[0] = (uint32_t)T; out->kernel.scalarParams[1] = C; out->kernel.scalarParams[2] = RG;
        return put_elf_tag(out, "GAUDIUM_PREPB_ELF", gdn_tag(HKs, HVs / HKs));
    }
    if (g == "gaudium_gdn_posta" || g == "gaudium_gdn_postb") {   // posta: o {128,2C,48,NCH} -> r {64,T,48}; postb: o, qkvz, norm_w, r -> core bf16 {6144,T}; params: C
        const bool a = g == "gaudium_gdn_posta"; const uint32_t nin = a ? 1 : 4;
        if (p->inputTensorNr != nin) { p->inputTensorNr = nin; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 1) { p->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[4] = {DATA_F32, DATA_BF16, DATA_F32, DATA_F32}, ota[1] = {DATA_F32}, otb[1] = {DATA_BF16};
        if (!all_dtype(p, in, a ? ota : otb)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& o = p->inputTensors[0]; const bool sig = (node_u32(p, 0) >> 30) & 1;   // node param bit 30 (postb, qwen4_exp): the sigmoid-gate build
        const uint32_t C = node_u32(p, 0) & ~(1u << 30), RG = 64;
        if (o.geometry.dims != 4 || C == 0 || o.geometry.maxSizes[1] != 2 * C || o.geometry.maxSizes[2] == 0) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const uint64_t HVs = o.geometry.maxSizes[2]; std::string tag;   // posta has no shape-dependent code; postb reads the gate z at channel QKV of qkvz
        if (!a) { const uint64_t QKVZs = p->inputTensors[1].geometry.maxSizes[0]; uint64_t HKs = 0, REPs = 0;
            if (QKVZs % 128 || QKVZs / 128 <= HVs || !gdn_shape(QKVZs / 128 - HVs, HVs, HKs, REPs)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
            tag = sig ? "K" + std::to_string(HKs) + "R" + std::to_string(REPs) + "S" : gdn_tag(HKs, REPs); }
        const uint64_t T = a ? p->outputTensors[0].geometry.maxSizes[1] : p->inputTensors[1].geometry.maxSizes[1];
        out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = HVs; out->indexSpaceGeometry[1] = (T + RG - 1) / RG;
        for (uint32_t i = 0; i < nin; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        pattern(out->outputTensorAccessPattern[0], p->outputTensors[0], -1);
        out->kernel.paramsNr = 3; out->kernel.scalarParams[0] = (uint32_t)T; out->kernel.scalarParams[1] = C; out->kernel.scalarParams[2] = RG;
        return put_elf_tag(out, a ? "GAUDIUM_POSTA_ELF" : "GAUDIUM_POSTB_ELF", tag);
    }
    if (g == "gaudium_gdn_solve") {   // An {C,C,HV,NCH}, gb2 {C,2,HV,NCH} -> D0n, Dn, Dg, Lp, Id {C,C,HV,NCH} (block form, MME finishes the inverse)
        if (p->inputTensorNr != 2) { p->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 5) { p->outputTensorNr = 5; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[2] = {DATA_F32, DATA_F32}, ot[5] = {DATA_F32, DATA_F32, DATA_F32, DATA_F32, DATA_F32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& Mw = p->outputTensors[0];
        if (Mw.geometry.dims != 4 || Mw.geometry.maxSizes[0] != Mw.geometry.maxSizes[1] || Mw.geometry.maxSizes[0] > 64) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        uint64_t C = Mw.geometry.maxSizes[0], HV = Mw.geometry.maxSizes[2], NCH = Mw.geometry.maxSizes[3];
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = HV * NCH;
        for (int i = 0; i < 2; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        for (int i = 0; i < 5; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
        out->kernel.paramsNr = 2; out->kernel.scalarParams[0] = (uint32_t)C; out->kernel.scalarParams[1] = (uint32_t)HV;
        return put_elf(out, "GAUDIUM_SOLVE_ELF");
    }
    if (g == "gaudium_gdn_chunk") {   // gbT {Tp,128}, KK, QK {C,C,HV,NCH}, q, k {128,C,HV,NCH} -> M3 {C,2C,HV,NCH}, An {C,C,HV,NCH}, gb2 {C,2,HV,NCH}, GQ0, AK {128,2C,HV,NCH}, gL {1,1,HV,NCH}
        if (p->inputTensorNr != 5) { p->inputTensorNr = 5; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 6 && p->outputTensorNr != 7) { p->outputTensorNr = 6; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }   // 7: + gLI (chain mode, -DCHAIN build)
        const TensorDataType in[5] = {DATA_F32, DATA_F32, DATA_F32, DATA_F32, DATA_F32}, ot[7] = {DATA_F32, DATA_F32, DATA_F32, DATA_F32, DATA_F32, DATA_F32, DATA_F32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& Mw = p->outputTensors[1];   // An: {C, C, HV, NCH}
        if (Mw.geometry.dims != 4 || Mw.geometry.maxSizes[0] != Mw.geometry.maxSizes[1] || Mw.geometry.maxSizes[0] > 64) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        uint64_t C = Mw.geometry.maxSizes[0], HV = Mw.geometry.maxSizes[2], NCH = Mw.geometry.maxSizes[3];
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = HV * NCH;
        for (int i = 0; i < 5; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        for (uint32_t i = 0; i < p->outputTensorNr; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
        out->kernel.paramsNr = 2; out->kernel.scalarParams[0] = (uint32_t)C; out->kernel.scalarParams[1] = (uint32_t)HV;
        return put_elf(out, p->outputTensorNr == 7 ? "GAUDIUM_CHUNKA_ELF" : "GAUDIUM_CHUNK_ELF");
    }
    if (g == "gaudium_gdn_relayout") {   // mode 0: blob {64,256,HV[,NS]} [+ slot] -> matrix {128,128,HV}; mode 1: matrix -> blob
        const bool slotted = p->inputTensorNr == 2;
        if (p->inputTensorNr != 1 && p->inputTensorNr != 2) { p->inputTensorNr = 1; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 1) { p->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[2] = {DATA_F32, DATA_I32}, ot[1] = {DATA_F32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        uint64_t HV = p->outputTensors[0].geometry.maxSizes[2];
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = HV;
        pattern(out->inputTensorAccessPattern[0], p->inputTensors[0], 2);
        if (slotted) pattern(out->inputTensorAccessPattern[1], p->inputTensors[1], -1);
        pattern(out->outputTensorAccessPattern[0], p->outputTensors[0], 2);
        out->kernel.paramsNr = 1; out->kernel.scalarParams[0] = node_u32(p, 0);
        return put_elf(out, slotted ? "GAUDIUM_RELAYS_ELF" : "GAUDIUM_RELAY_ELF");
    }
    // ---- on-card sampler (samp_gaudi2.c): all-required patterns, index space {NB, R} or {R} ----
    if (g.rfind("gaudium_samp_", 0) == 0) {
        struct S { const char* g; uint32_t nin, nout; const char* env; int is2d; } tab[] = {
            {"gaudium_samp_bmax", 2, 1, "GAUDIUM_SBMAX_ELF", 1}, {"gaudium_samp_init", 1, 1, "GAUDIUM_SINIT_ELF", 0},
            {"gaudium_samp_part", 6, 1, "GAUDIUM_SPART_ELF", 1}, {"gaudium_samp_red", 4, 1, "GAUDIUM_SRED_ELF", 0},
            {"gaudium_samp_gumbel", 7, 1, "GAUDIUM_SGUMBEL_ELF", 1}, {"gaudium_samp_final", 2, 1, "GAUDIUM_SFINAL_ELF", 0}};
        const S* k = nullptr; for (auto& t : tab) if (g == t.g) k = &t;
        if (!k) return GLUE_NODE_NOT_FOUND;
        if (p->inputTensorNr != k->nin) { p->inputTensorNr = k->nin; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != k->nout) { p->outputTensorNr = k->nout; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        TensorDataType in[8], ot[1];
        const TensorDataType BF = DATA_BF16, F = DATA_F32, I = DATA_I32;
        if (g == "gaudium_samp_bmax")   { TensorDataType a[] = {BF, I}; memcpy(in, a, sizeof a); ot[0] = F; }
        if (g == "gaudium_samp_init")   { in[0] = F; ot[0] = F; }
        if (g == "gaudium_samp_part")   { TensorDataType a[] = {BF, I, F, I, F, F}; memcpy(in, a, sizeof a); ot[0] = F; }
        if (g == "gaudium_samp_red")    { TensorDataType a[] = {F, F, I, F}; memcpy(in, a, sizeof a); ot[0] = F; }
        if (g == "gaudium_samp_gumbel") { TensorDataType a[] = {BF, I, I, F, I, F, F}; memcpy(in, a, sizeof a); ot[0] = F; }
        if (g == "gaudium_samp_final")  { TensorDataType a[] = {F, I}; memcpy(in, a, sizeof a); ot[0] = I; }
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        // R = rows processed (from the output), NB = vocab blocks of 1024 (V must be a multiple of 128)
        const Tensor& o = p->outputTensors[0];
        uint64_t R, NB;
        if (g == "gaudium_samp_final") { R = p->inputTensors[0].geometry.maxSizes[3]; NB = p->inputTensors[0].geometry.maxSizes[2]; }
        else if (g == "gaudium_samp_gumbel") { R = o.geometry.maxSizes[3]; NB = o.geometry.maxSizes[2]; }
        else { R = o.geometry.maxSizes[2]; NB = o.geometry.maxSizes[1]; }
        if (g == "gaudium_samp_bmax" || g == "gaudium_samp_part" || g == "gaudium_samp_gumbel") {
            const Tensor& L = p->inputTensors[0]; if (L.geometry.maxSizes[0] % 128 || NB != (L.geometry.maxSizes[0] + 1023) / 1024) return GLUE_INCOMPATIBLE_INPUT_SIZE; }
        if (k->is2d) { out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = NB; out->indexSpaceGeometry[1] = R; }
        else { out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = R; }
        for (uint32_t i = 0; i < k->nin; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        pattern(out->outputTensorAccessPattern[0], o, -1);
        out->kernel.paramsNr = 0;
        if (g == "gaudium_samp_part" || g == "gaudium_samp_red" || g == "gaudium_samp_gumbel") {
            out->kernel.paramsNr = 2; out->kernel.scalarParams[0] = node_u32(p, 0); out->kernel.scalarParams[1] = node_u32(p, 1);
            if (g == "gaudium_samp_gumbel") { out->kernel.paramsNr = 3; out->kernel.scalarParams[2] = p->nodeParams.nodeParamsSize >= 12 ? node_u32(p, 2) : 1u; } }
        return put_elf(out, k->env);
    }
    const TensorDataType I8[8] = {DATA_I32, DATA_I32, DATA_I32, DATA_I32, DATA_I32, DATA_I32, DATA_I32, DATA_I32};
    if (g == "gaudium_accept_tree") {   // ids {N}, tokens {1,N}, cursor, parent {N}, depth {N} [, rank {N}] -> a, cursor_out, emitted {EM}, ncommit, fix {2F}, dep, c_ids {Tc}, pidx {Tc}
        const bool rk = p->inputTensorNr == 6;
        if ((p->inputTensorNr == 5 || rk) && p->inputTensors[0].geometry.maxSizes[0] > 32) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        return simple(rk ? 6 : 5, 8, I8, I8, 1, rk ? "GAUDIUM_ACCEPTTRR_ELF" : "GAUDIUM_ACCEPTTR_ELF", {});
    }
    if (g == "gaudium_tree_plan") {   // cprob f32 {KT,CM} -> parent, depth, rank, anc i32 {N}; params: Kd
        const TensorDataType in[1] = {DATA_F32};
        if (p->outputTensorNr == 4 && p->outputTensors[0].geometry.maxSizes[0] > 32) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        return simple(1, 4, in, I8, 1, "GAUDIUM_TREEPLAN_ELF", {node_u32(p, 0)});
    }
    if (g == "gaudium_tree_ids") {   // cand {KT,CM}, depth {N}, rank {N} -> ids {N}
        return simple(3, 1, I8, I8, 1, "GAUDIUM_TREEIDS_ELF", {});
    }
    if (g == "gaudium_tree_cand") {   // tokens, a, dep, c_tok, c_top, d_tok, d_top, c_prob f32, d_prob f32 -> cand i32 {KT,CM}, cprob f32 {KT,CM}; params: Kd
        const TensorDataType in[9] = {DATA_I32, DATA_I32, DATA_I32, DATA_I32, DATA_I32, DATA_I32, DATA_I32, DATA_F32, DATA_F32}, ot[2] = {DATA_I32, DATA_F32};
        return simple(9, 2, in, ot, 1, "GAUDIUM_TREECAND_ELF", {node_u32(p, 0)});
    }
    if (g == "gaudium_mtp_topk1") {   // logits bf16 {V,T} -> pv f32 {64,KT,CH,T}, pi i32 {64,KT,CH,T}, ps f32 {64,CH,T} (CH from the output)
        const TensorDataType in[1] = {DATA_BF16}, ot[3] = {DATA_F32, DATA_I32, DATA_F32};
        if (p->inputTensorNr != 1) { p->inputTensorNr = 1; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 3) { p->outputTensorNr = 3; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        uint64_t V = p->inputTensors[0].geometry.maxSizes[0], T = p->inputTensors[0].geometry.maxSizes[1], CH = p->outputTensors[0].geometry.maxSizes[2];
        if (V % 128 || p->outputTensors[0].geometry.maxSizes[1] != 4) return GLUE_INCOMPATIBLE_INPUT_SIZE;   // KTOP = 4 build
        return simple(1, 3, in, ot, CH * T, "GAUDIUM_MTOPK1_ELF", {(uint32_t)CH, (uint32_t)(V / 128)});
    }
    if (g == "gaudium_mtp_topk2") {   // pv, pi, ps -> ids i32 {KT,T}, vals f32 {KT,T}, probs f32 {KT,T}
        const TensorDataType in[3] = {DATA_F32, DATA_I32, DATA_F32}, ot[3] = {DATA_I32, DATA_F32, DATA_F32};
        if (p->inputTensorNr != 3) { p->inputTensorNr = 3; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        uint64_t CH = p->inputTensors[0].geometry.maxSizes[2], T = p->inputTensors[0].geometry.maxSizes[3];
        return simple(3, 3, in, ot, T, "GAUDIUM_MTOPK2_ELF", {(uint32_t)CH});
    }
    // ---------------- MoE (issue #8) ----------------
    // ---- qwen4_exp (Qwen3.8-Flash-Next): hyper-connections and the n-gram (PLE) layer; every access pattern all-required ----
    if (g == "gaudium_hc_norm") {   // x bf16 {S*HID, T}, w bf16 {HID, S} -> y bf16 {S*HID, T}; index {S, T}; scalar HID
        if (p->inputTensorNr != 2) { p->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 1) { p->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[2] = {DATA_BF16, DATA_BF16}, ot[1] = {DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& x = p->inputTensors[0]; const Tensor& w = p->inputTensors[1];
        const uint64_t HID = w.geometry.maxSizes[0], S = w.geometry.dims >= 2 ? w.geometry.maxSizes[1] : 1, T = x.geometry.maxSizes[1];
        if (HID % 256 || x.geometry.maxSizes[0] != S * HID || p->outputTensors[0].geometry.maxSizes[0] != S * HID || p->outputTensors[0].geometry.maxSizes[1] != T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = S; out->indexSpaceGeometry[1] = T;
        for (int i = 0; i < 2; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        pattern(out->outputTensorAccessPattern[0], p->outputTensors[0], -1);
        out->kernel.paramsNr = 1; out->kernel.scalarParams[0] = (uint32_t)HID;
        return put_elf(out, "GAUDIUM_HCNORM_ELF");
    }
    if (g == "gaudium_hc_mix") {   // xn, g bf16 {S*HID, T} -> x bf16 {HID, T}; index {HID/256, T}; scalars HID, S
        if (p->inputTensorNr != 2) { p->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 1) { p->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[2] = {DATA_BF16, DATA_BF16}, ot[1] = {DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& y = p->outputTensors[0]; const uint64_t HID = y.geometry.maxSizes[0], T = y.geometry.maxSizes[1], RW = p->inputTensors[0].geometry.maxSizes[0];
        if (HID % 256 || RW % HID || p->inputTensors[1].geometry.maxSizes[0] != RW || p->inputTensors[0].geometry.maxSizes[1] != T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = HID / 256; out->indexSpaceGeometry[1] = T;
        for (int i = 0; i < 2; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        pattern(out->outputTensorAccessPattern[0], p->outputTensors[0], -1);
        out->kernel.paramsNr = 2; out->kernel.scalarParams[0] = (uint32_t)HID; out->kernel.scalarParams[1] = (uint32_t)(RW / HID);
        return put_elf(out, "GAUDIUM_HCMIX_ELF");
    }
    if (g == "gaudium_hc_combine") {   // R bf16 {S*HID, T}, blk bf16 {HID, T}, inj bf16 {S, T} -> R' bf16 {S*HID, T}; index {HID/256, T}; scalars HID, S
        if (p->inputTensorNr != 3) { p->inputTensorNr = 3; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 1) { p->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[3] = {DATA_BF16, DATA_BF16, DATA_BF16}, ot[1] = {DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const uint64_t HID = p->inputTensors[1].geometry.maxSizes[0], T = p->inputTensors[1].geometry.maxSizes[1], S = p->inputTensors[2].geometry.maxSizes[0];
        if (HID % 256 || S < 1 || S > 4 || p->inputTensors[0].geometry.maxSizes[0] != S * HID || p->outputTensors[0].geometry.maxSizes[0] != S * HID || p->inputTensors[2].geometry.maxSizes[1] != T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = HID / 256; out->indexSpaceGeometry[1] = T;
        for (int i = 0; i < 3; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        pattern(out->outputTensorAccessPattern[0], p->outputTensors[0], -1);
        out->kernel.paramsNr = 2; out->kernel.scalarParams[0] = (uint32_t)HID; out->kernel.scalarParams[1] = (uint32_t)S;
        return put_elf(out, "GAUDIUM_HCCOMB_ELF");
    }
    if (g == "gaudium_ple_gate") {   // kn, qn bf16 {S*HID, T}, val bf16 {HID, T} -> gv bf16 {S*HID, T}; index {S, T}; scalars HID, 1/sqrt(HID)
        if (p->inputTensorNr != 3) { p->inputTensorNr = 3; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 1) { p->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[3] = {DATA_BF16, DATA_BF16, DATA_BF16}, ot[1] = {DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const uint64_t HID = p->inputTensors[2].geometry.maxSizes[0], T = p->inputTensors[2].geometry.maxSizes[1], RW = p->inputTensors[0].geometry.maxSizes[0];
        if (HID % 256 || RW % HID || p->inputTensors[1].geometry.maxSizes[0] != RW || p->outputTensors[0].geometry.maxSizes[0] != RW) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = RW / HID; out->indexSpaceGeometry[1] = T;
        for (int i = 0; i < 3; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        pattern(out->outputTensorAccessPattern[0], p->outputTensors[0], -1);
        const float rs = 1.0f / std::sqrt((float)HID); uint32_t rb; memcpy(&rb, &rs, 4);
        out->kernel.paramsNr = 2; out->kernel.scalarParams[0] = (uint32_t)HID; out->kernel.scalarParams[1] = rb;
        return put_elf(out, "GAUDIUM_PLEGATE_ELF");
    }
    if (g == "gaudium_ple_conv") {   // cn, gv, R bf16 {RW,T}, ps_in f32 {RW,9,NS}, w f32 {RW,4}, slot i32 {1} -> Ro bf16 {RW,T}, ps_ck f32 {RW,9,T|1}, ps_out f32 {RW,9}; node param: flags
        if (p->inputTensorNr != 6) { p->inputTensorNr = 6; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 3) { p->outputTensorNr = 3; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[6] = {DATA_BF16, DATA_BF16, DATA_BF16, DATA_F32, DATA_F32, DATA_I32}, ot[3] = {DATA_BF16, DATA_F32, DATA_F32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const uint64_t RW = p->inputTensors[0].geometry.maxSizes[0], T = p->inputTensors[0].geometry.maxSizes[1];
        if (RW % 128 || p->inputTensors[3].geometry.maxSizes[0] != RW || p->inputTensors[3].geometry.maxSizes[1] != 9 || p->inputTensors[4].geometry.maxSizes[0] != RW || p->inputTensors[4].geometry.maxSizes[1] != 4) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (p->outputTensors[1].geometry.maxSizes[0] != RW || p->outputTensors[1].geometry.maxSizes[1] != 9 || p->outputTensors[2].geometry.maxSizes[0] != RW) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        const uint32_t flags = node_u32(p, 0); if ((flags & 1) && p->outputTensors[1].geometry.maxSizes[2] < T) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = RW / 128;
        for (int i = 0; i < 6; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        for (int i = 0; i < 3; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
        out->kernel.paramsNr = 2; out->kernel.scalarParams[0] = (uint32_t)T; out->kernel.scalarParams[1] = flags;
        return put_elf(out, "GAUDIUM_PLECONV_ELF");
    }
    if (g == "gaudium_ple_sync") {   // e bf16 {PE, T}, flag i32 {64} -> eo bf16 {PE, T}, st i32 {64}; one index point; node param: max polls; scalars PE, T, max polls
        if (p->inputTensorNr != 2) { p->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 2) { p->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[2] = {DATA_BF16, DATA_I32}, ot[2] = {DATA_BF16, DATA_I32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const uint64_t PE = p->inputTensors[0].geometry.maxSizes[0], T = p->inputTensors[0].geometry.maxSizes[1];
        if (PE == 0 || PE % 128 || T == 0 || p->inputTensors[1].geometry.maxSizes[0] != 64) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (p->outputTensors[0].geometry.maxSizes[0] != PE || p->outputTensors[0].geometry.maxSizes[1] != T || p->outputTensors[1].geometry.maxSizes[0] != 64) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = 1;
        for (int i = 0; i < 2; ++i) { pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1); pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1); }
        out->kernel.paramsNr = 3; out->kernel.scalarParams[0] = (uint32_t)PE; out->kernel.scalarParams[1] = (uint32_t)T; out->kernel.scalarParams[2] = node_u32(p, 0);
        return put_elf(out, "GAUDIUM_PLESYNC_ELF");
    }
    if (g == "gaudium_moe_route") {   // logits bf16 {E, T} [, sgl bf16 {1, T}] -> ids i32 {K, T} (ascending), w f32 {K, T}, wd bf16 {E, T}, wd2 f32 {E/2 + 64, 2, T}, wb bf16 {K, T}; param: K
        const bool sg = p->inputTensorNr == 2;
        const TensorDataType in[2] = {DATA_BF16, DATA_BF16}, ot[5] = {DATA_I32, DATA_F32, DATA_BF16, DATA_F32, DATA_BF16};
        if (p->inputTensorNr != 1 && p->inputTensorNr != 2) { p->inputTensorNr = 1; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 5) { p->outputTensorNr = 5; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const uint64_t E = p->inputTensors[0].geometry.maxSizes[0], T = p->inputTensors[0].geometry.maxSizes[1], K = p->outputTensors[0].geometry.maxSizes[0];
        if (E == 0 || E % 128 || E > 512 || K == 0 || K > 12 || K > E) return GLUE_INCOMPATIBLE_INPUT_SIZE;     // MAXV = 8 f32 vectors, MAXK = 12 in the kernel
        if (sg && (p->inputTensors[1].geometry.maxSizes[0] != 1 || p->inputTensors[1].geometry.maxSizes[1] != T)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const Tensor& w2 = p->outputTensors[3];
        if (p->outputTensors[1].geometry.maxSizes[0] != K || p->outputTensors[2].geometry.maxSizes[0] != E || w2.geometry.dims != 3 || w2.geometry.maxSizes[0] != E / 2 + 64 || w2.geometry.maxSizes[1] != 2 || w2.geometry.maxSizes[2] != T
            || p->outputTensors[4].geometry.maxSizes[0] != K) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        return simple(sg ? 2 : 1, 5, in, ot, T, sg ? "GAUDIUM_MOEROUTESG_ELF" : "GAUDIUM_MOEROUTE_ELF", {(uint32_t)K});
    }
    if (g == "gaudium_moe_act") {   // g, u bf16 {NB F, T}, wd2 f32 {NE/2 + 64, 2, T} -> a bf16 {NB F, T}; node param: F (channels per block); index space {NB, T}
        const TensorDataType in[3] = {DATA_BF16, DATA_BF16, DATA_F32}, ot[1] = {DATA_BF16};
        if (p->inputTensorNr != 3) { p->inputTensorNr = 3; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 1) { p->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const uint64_t N = p->inputTensors[0].geometry.maxSizes[0], T = p->inputTensors[0].geometry.maxSizes[1]; const uint32_t F = node_u32(p, 0), first = node_u32(p, 1);   // first: the expert id of block 0
        const Tensor& w2 = p->inputTensors[2];
        if (F == 0 || F % 128 || N % F || p->inputTensors[1].geometry.maxSizes[0] != N || p->inputTensors[1].geometry.maxSizes[1] != T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const uint64_t NB = N / F;     // NE + 1 blocks; the weight table has a slot for each (and 64 slots of padding)
        if (w2.geometry.dims != 3 || w2.geometry.maxSizes[1] != 2 || w2.geometry.maxSizes[2] != T || 2 * w2.geometry.maxSizes[0] < first + NB) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (p->outputTensors[0].geometry.maxSizes[0] != N || p->outputTensors[0].geometry.maxSizes[1] != T) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = NB; out->indexSpaceGeometry[1] = T;
        for (int i = 0; i < 3; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        pattern(out->outputTensorAccessPattern[0], p->outputTensors[0], -1);
        out->kernel.paramsNr = 2; out->kernel.scalarParams[0] = F; out->kernel.scalarParams[1] = first;
        return put_elf(out, "GAUDIUM_MOEACT_ELF");
    }
    { bool handled = false; GlueCodeReturn r = moe_w4_glue(g, p, out, handled); if (handled) return r; }   // W4L / W8 experts (moe_w4_glue.inc)
    return GLUE_NODE_NOT_FOUND;
}
}
