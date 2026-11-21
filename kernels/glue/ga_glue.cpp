// Graph-compiler glue of the Gemma 4 kernels (docs/notes/engine-gemma4.md): its own plugin library, libgaudium_ga.so, beside
// libgaudium_tpc.so in GC_KERNEL_PATH. A model family's kernels and their glue are a separate library so that adding or changing them
// cannot change what the graph compiler builds for another family: that library stays the same binary.
//   gaudium_ga_prep          q / k norms + weights, the unweighted V norm, RoPE, cache write (ring or chunk-major), the q block
//                            7 inputs: the k_eq_v build (V from the k rows)                       ELF: GAUDIUM_GAPREP_ELF / GAUDIUM_GAPREPK_ELF
//   gaudium_ga_qprep         the query half alone (a pass with no keys of its own: the drafter)   ELF: GAUDIUM_GAQPREP_ELF
//   gaudium_ga_kv            the key / value half alone, V without a norm (a block drafter's context rows)   ELF: GAUDIUM_GAKV_ELF
//   gaudium_ga_prep, node param 2 set: V rows without a norm (a block drafter's layers)             ELF: GAUDIUM_GAPREPV_ELF
//   gaudium_ga_softmax_ring  masked softmax of a sliding layer over its K / V ring                 ELF: GAUDIUM_GASOFTR_ELF
//   gaudium_ga_softmax_ringx the same with a per-row extension (image blocks attend both ways)     ELF: GAUDIUM_GASOFTRX_ELF
//   gaudium_ga_combine       merge of the chunk partials, 256- or 512-channel heads                ELF: GAUDIUM_GACOMB4_ELF / GAUDIUM_GACOMB8_ELF
//   gaudium_ga_post          f32 attention output -> the bf16 row of the o projection              ELF: GAUDIUM_GAPOST_ELF
//   gaudium_ga_gelu          bf16(bf16(gelu_tanh(gate)) * up)                                       ELF: GAUDIUM_GAGELU_ELF
//   gaudium_ga_softcap       bf16(bf16(tanh(bf16(x / cap))) * cap)                                  ELF: GAUDIUM_GASOFTCAP_ELF
//   gaudium_ga_copy          y = x (bf16): a rounding point the graph compiler cannot fuse away     ELF: GAUDIUM_GACOPY_ELF
//   gaudium_ga_clamp         y = min(max(x, lo), hi) (bf16), the bounds as tensors                  ELF: GAUDIUM_GACLAMP_ELF
//   gaudium_ga_argmax        index of each column's maximum, the lowest among equal ones, any sign   ELF: GAUDIUM_GAARGMAX_ELF
// The audio encoder of the family's small sizes (docs/notes/engine-gemma4.md, section 14):
//   gaudium_ga_relattn       banded attention with a distance term, soft-capped, f32                ELF: GAUDIUM_GARELATTN_ELF
//   gaudium_ga_act           SiLU (node param 0) or GLU of the two halves of the rows (1)           ELF: GAUDIUM_GAACT_ELF
//   gaudium_ga_dwconv        causal depthwise convolution over time                                 ELF: GAUDIUM_GADWCONV_ELF
//   gaudium_ga_lnrelu        LayerNorm over the channels (a weight, no bias), then ReLU             ELF: GAUDIUM_GALNRELU_ELF
//   gaudium_ga_lnorm         LayerNorm over a row of any width (a weight and a bias)                ELF: GAUDIUM_GALNORM_ELF
//   gaudium_ga_rmsnorm       the family's RMS norm with a weight, one row per index-space member    ELF: GAUDIUM_GARMSNORM_ELF
// A DFlash block drafter on a target of the family (heads of 128 channels, 32 / 8; docs/notes/engine-gemma4.md, section 19): the layers
// with a sliding window keep their context keys in a ring. The other kernels of that drafter are libgaudium_tpc.so's (gaudium_df_*).
//   gaudium_ga_dfctxr        context K / V rows into rings (the ring form of gaudium_df_ctx)        ELF: GAUDIUM_GADFCTXR_ELF
//   gaudium_ga_dfsoftr       block softmax over a context ring, causal block (the ring form of gaudium_ds_softmax)   ELF: GAUDIUM_GADFSOFTR_ELF
// The head counts and widths are scalars read from the tensors: one ELF serves every layer kind and every size of the family.
// The graph compiler needs the GetShapeInference and GetLibVersion entry points or it skips the library.
#include "tpc_kernel_lib_interface.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

using namespace tpc_lib_api;

// ELF cache: the engine compiles on several threads, so the graph compiler calls this glue concurrently; one mutex covers every access
// (lookup / insert, the file read, the size check and the copy), as in libgaudium_tpc.so.
static std::mutex& elf_mu() { static std::mutex mu; return mu; }
static std::vector<unsigned char>& elf_locked(const char* env) {   // caller holds elf_mu()
    static std::map<std::string, std::vector<unsigned char>> cache;   // keyed by variable AND path
    const char* p = getenv(env);
    std::vector<unsigned char>& e = cache[std::string(env) + "=" + (p ? p : "")];
    if (e.empty()) {
        FILE* f = p ? fopen(p, "rb") : nullptr;
        if (!f) { fprintf(stderr, "gaudium glue (gemma4): cannot open %s=%s\n", env, p ? p : "(null)"); return e; }
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
// tensor dim `split_dim` maps 1:1 onto index-space dim 0 (with `scale` elements per index point); other dims whole (-1: every dim whole).
static void pattern(TensorAccessPattern& ap, const Tensor& t, int split_dim, int scale = 1) {
    memset(&ap, 0, sizeof(ap));
    ap.allRequired = 1;
    for (uint32_t d = 0; d < t.geometry.dims; ++d) {
        ap.mapping[d].indexSpaceDim = 0;
        if ((int)d == split_dim) { ap.mapping[d].a = (float)scale; ap.mapping[d].start_b = 0; ap.mapping[d].end_b = (float)(scale - 1); }
        else { ap.mapping[d].a = 0; ap.mapping[d].start_b = 0; ap.mapping[d].end_b = (float)(t.geometry.maxSizes[d] - 1); ap.mapping[d].allRequired = true; }
    }
}
static bool all_dtype(const HabanaKernelParams* p, const TensorDataType* in, const TensorDataType* outt) {
    bool ok = true;
    for (uint32_t i = 0; i < p->inputTensorNr; ++i) if (p->inputTensors[i].geometry.dataType != in[i]) { p->inputTensors[i].geometry.dataType = in[i]; ok = false; }
    for (uint32_t i = 0; i < p->outputTensorNr; ++i) if (p->outputTensors[i].geometry.dataType != outt[i]) { p->outputTensors[i].geometry.dataType = outt[i]; ok = false; }
    return ok;
}
static uint32_t node_u32(const HabanaKernelParams* p, int i) {
    return (p->nodeParams.nodeParams && p->nodeParams.nodeParamsSize >= (uint32_t)(4 * (i + 1))) ? ((uint32_t*)p->nodeParams.nodeParams)[i] : 0;
}

extern "C" {
uint64_t GetLibVersion() { return 78813002322618477ULL; }   // the interface version libgaudium_tpc.so reports
GlueCodeReturn GetShapeInference(_IN_ DeviceId, _IN_ const ShapeInferenceParams*, _OUT_ ShapeInferenceOutput*) { return GLUE_SUCCESS; }

static const char* GUIDS[] = {"gaudium_ga_prep", "gaudium_ga_softmax_ring", "gaudium_ga_combine", "gaudium_ga_post", "gaudium_ga_gelu", "gaudium_ga_softcap", "gaudium_ga_qprep", "gaudium_ga_kv", "gaudium_ga_softmax_ringx", "gaudium_ga_copy", "gaudium_ga_clamp", "gaudium_ga_relattn", "gaudium_ga_act", "gaudium_ga_dwconv", "gaudium_ga_lnrelu", "gaudium_ga_argmax", "gaudium_ga_lnorm", "gaudium_ga_rmsnorm", "gaudium_ga_dfctxr", "gaudium_ga_dfsoftr"};
GlueCodeReturn GetKernelGuids(_IN_ DeviceId deviceId, _INOUT_ uint32_t* kernelCount, _OUT_ GuidInfo* guids) {
    const uint32_t n = sizeof(GUIDS) / sizeof(GUIDS[0]);
    if (deviceId != DEVICE_ID_GAUDI2) { if (kernelCount) *kernelCount = 0; return GLUE_SUCCESS; }
    if (guids) { memset(guids, 0, n * sizeof(GuidInfo)); for (uint32_t i = 0; i < n; ++i) strcpy(guids[i].name, GUIDS[i]); }
    if (kernelCount) *kernelCount = n;
    return GLUE_SUCCESS;
}

GlueCodeReturn InstantiateTpcKernel(_IN_ HabanaKernelParams* p, _INOUT_ HabanaKernelInstantiation* out) {
    const std::string g = p->guid.name;
    if (g == "gaudium_ga_dfctxr") {   // (ga_dfctxr_gaudi2.c) kc, vc bf16 {1024, T}, kn_w f32 {128}, rope_c / rope_s f32 {128, L}, cursor i32 {1}, depth i32 {>= T}, fix i32 {2} -> kring, vring bf16 {128, C, 8}
        if (p->inputTensorNr != 8) { p->inputTensorNr = 8; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 2) { p->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[8] = {DATA_BF16, DATA_BF16, DATA_F32, DATA_F32, DATA_F32, DATA_I32, DATA_I32, DATA_I32}, ot[2] = {DATA_BF16, DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& k = p->inputTensors[0]; const Tensor& kc = p->outputTensors[0]; const Tensor& vc = p->outputTensors[1];
        const uint64_t T = k.geometry.maxSizes[1], C = kc.geometry.maxSizes[1];
        if (k.geometry.maxSizes[0] != 1024 || p->inputTensors[1].geometry.maxSizes[0] != 1024 || p->inputTensors[2].geometry.maxSizes[0] != 128 || p->inputTensors[6].geometry.maxSizes[0] < T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (kc.geometry.dims != 3 || vc.geometry.dims != 3 || kc.geometry.maxSizes[0] != 128 || kc.geometry.maxSizes[2] != 8 || C < 128 || (C & (C - 1)) || vc.geometry.maxSizes[1] != C) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = 8;   // the kv heads
        for (uint32_t i = 0; i < 8; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        for (uint32_t i = 0; i < 2; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
        out->kernel.paramsNr = 1; out->kernel.scalarParams[0] = (uint32_t)T;
        return put_elf(out, "GAUDIUM_GADFCTXR_ELF");
    }
    if (g == "gaudium_ga_dfsoftr") {   // (ga_dfsoftr_gaudi2.c) sc f32 {RING, R, 8}, sb f32 {128, R, 8}, cur i32 {1}, fix i32 {2} (not read) -> Pc bf16 {RING, R, 8}, Pb bf16 {128, R, 8}; node params: T, W
        if (p->inputTensorNr != 4) { p->inputTensorNr = 4; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 2) { p->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[4] = {DATA_F32, DATA_F32, DATA_I32, DATA_I32}, ot[2] = {DATA_BF16, DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const uint64_t RG = p->inputTensors[0].geometry.maxSizes[0], R = p->inputTensors[0].geometry.maxSizes[1], G = p->inputTensors[0].geometry.maxSizes[2];
        const uint32_t T = node_u32(p, 0), W = node_u32(p, 1);
        if (T == 0 || T > 64 || R % T || RG < 128 || (RG & (RG - 1)) || RG + 1 < (uint64_t)W + T) return GLUE_INCOMPATIBLE_INPUT_SIZE;   // a ring holds a window and the rows of a pass
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = R * G;
        for (uint32_t i = 0; i < 4; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        for (uint32_t i = 0; i < 2; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
        out->kernel.paramsNr = 3; out->kernel.scalarParams[0] = T; out->kernel.scalarParams[1] = (uint32_t)RG; out->kernel.scalarParams[2] = W;
        return put_elf(out, "GAUDIUM_GADFSOFTR_ELF");
    }
    if (g == "gaudium_ga_kv") {   // k, v {128 NV NKV, T}, kn_w f32 {128 NV}, rope_c / rope_s f32 {128 RV, L}, cursor i32 {1} -> kcache / vcache {128 NV, C, NKV nch}; node params: coff, ring
        if (p->inputTensorNr != 6) { p->inputTensorNr = 6; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 2) { p->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[6] = {DATA_BF16, DATA_BF16, DATA_F32, DATA_F32, DATA_F32, DATA_I32}, ot[2] = {DATA_BF16, DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& k = p->inputTensors[0]; const Tensor& v = p->inputTensors[1]; const Tensor& kw = p->inputTensors[2]; const Tensor& rc = p->inputTensors[3]; const Tensor& rs = p->inputTensors[4];
        const Tensor& kc = p->outputTensors[0]; const Tensor& vc = p->outputTensors[1];
        const uint64_t DH = kw.geometry.maxSizes[0], T = k.geometry.maxSizes[1];
        if ((DH != 256 && DH != 512) || k.geometry.maxSizes[0] % DH || v.geometry.maxSizes[0] != k.geometry.maxSizes[0] || v.geometry.maxSizes[1] != T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const uint64_t NV = DH / 128, NKVs = k.geometry.maxSizes[0] / DH, RV = rc.geometry.maxSizes[0] / 128;
        if (NKVs == 0 || rc.geometry.dims != 2 || rs.geometry.dims != 2 || rc.geometry.maxSizes[0] % 128 || RV == 0 || RV > NV / 2 || rs.geometry.maxSizes[0] != rc.geometry.maxSizes[0] || rs.geometry.maxSizes[1] != rc.geometry.maxSizes[1]) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (kc.geometry.dims != 3 || vc.geometry.dims != 3 || kc.geometry.maxSizes[0] != DH || vc.geometry.maxSizes[0] != DH || kc.geometry.maxSizes[2] % NKVs) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        const uint64_t C = kc.geometry.maxSizes[1], nch = kc.geometry.maxSizes[2] / NKVs; const uint32_t ring = node_u32(p, 1);
        if (nch == 0 || C % 128 || vc.geometry.maxSizes[1] != C || vc.geometry.maxSizes[2] != NKVs * nch) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (ring && (nch != 1 || (C & (C - 1)))) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = NKVs;
        for (uint32_t i = 0; i < 6; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        for (uint32_t i = 0; i < 2; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
        out->kernel.paramsNr = 7; out->kernel.scalarParams[0] = (uint32_t)T; out->kernel.scalarParams[1] = node_u32(p, 0); out->kernel.scalarParams[2] = (uint32_t)C;
        out->kernel.scalarParams[3] = (uint32_t)NKVs; out->kernel.scalarParams[4] = (uint32_t)NV; out->kernel.scalarParams[5] = (uint32_t)RV; out->kernel.scalarParams[6] = ring ? 1u : 0u;
        return put_elf(out, "GAUDIUM_GAKV_ELF");
    }
    if (g == "gaudium_ga_qprep") {   // q {128 NV NH, T}, qn_w f32 {128 NV}, rope_c / rope_s f32 {128 RV, L}, cursor i32 {1} -> qb {128 NV, GQ T, NKV nch}; node params: coff, same
        if (p->inputTensorNr != 5) { p->inputTensorNr = 5; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 1) { p->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[5] = {DATA_BF16, DATA_F32, DATA_F32, DATA_F32, DATA_I32}, ot[1] = {DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& q = p->inputTensors[0]; const Tensor& qw = p->inputTensors[1]; const Tensor& rc = p->inputTensors[2]; const Tensor& rs = p->inputTensors[3];
        const Tensor& qb = p->outputTensors[0];
        const uint64_t DH = qw.geometry.maxSizes[0], T = q.geometry.maxSizes[1];
        if ((DH != 256 && DH != 512) || q.geometry.maxSizes[0] % DH || T == 0) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const uint64_t NV = DH / 128, NHs = q.geometry.maxSizes[0] / DH, RV = rc.geometry.maxSizes[0] / 128;
        if (rc.geometry.dims != 2 || rs.geometry.dims != 2 || rc.geometry.maxSizes[0] % 128 || RV == 0 || RV > NV / 2 || rs.geometry.maxSizes[0] != rc.geometry.maxSizes[0] || rs.geometry.maxSizes[1] != rc.geometry.maxSizes[1]) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (qb.geometry.dims != 3 || qb.geometry.maxSizes[0] != DH || qb.geometry.maxSizes[1] % T) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        const uint64_t GQs = qb.geometry.maxSizes[1] / T;
        if (GQs == 0 || NHs % GQs) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        const uint64_t NKVs = NHs / GQs;
        if (qb.geometry.maxSizes[2] % NKVs) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        const uint64_t nch = qb.geometry.maxSizes[2] / NKVs;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = NHs;
        for (uint32_t i = 0; i < 5; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        pattern(out->outputTensorAccessPattern[0], p->outputTensors[0], -1);
        out->kernel.paramsNr = 8; out->kernel.scalarParams[0] = (uint32_t)T; out->kernel.scalarParams[1] = node_u32(p, 0); out->kernel.scalarParams[2] = (uint32_t)nch;
        out->kernel.scalarParams[3] = (uint32_t)NKVs; out->kernel.scalarParams[4] = (uint32_t)GQs; out->kernel.scalarParams[5] = (uint32_t)NV;
        out->kernel.scalarParams[6] = (uint32_t)RV; out->kernel.scalarParams[7] = node_u32(p, 1) ? 1u : 0u;
        return put_elf(out, "GAUDIUM_GAQPREP_ELF");
    }
    if (g == "gaudium_ga_prep") {   // q {128 NV NH, T}, k {128 NV NKV, T} [, v {128 NV NKV, T}], qn_w / kn_w f32 {128 NV}, rope_c / rope_s f32 {128 RV, L}, cursor i32 {1}
        //                             -> qb {128 NV, GQ T, NKV nch}, kcache / vcache {128 NV, C, NKV nch}; node params: coff, ring. 7 inputs: the k_eq_v build (no v)
        const bool keqv = p->inputTensorNr == 7; const uint32_t nin = p->inputTensorNr;
        if (nin != 7 && nin != 8) { p->inputTensorNr = 8; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 3) { p->outputTensorNr = 3; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        TensorDataType in[8] = {DATA_BF16, DATA_BF16, DATA_BF16, DATA_F32, DATA_F32, DATA_F32, DATA_F32, DATA_I32}, ot[3] = {DATA_BF16, DATA_BF16, DATA_BF16};
        if (keqv) { in[2] = DATA_F32; in[3] = DATA_F32; in[4] = DATA_F32; in[5] = DATA_F32; in[6] = DATA_I32; }
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& q = p->inputTensors[0]; const Tensor& k = p->inputTensors[1]; const Tensor& qw = p->inputTensors[keqv ? 2 : 3]; const Tensor& kw = p->inputTensors[keqv ? 3 : 4];
        const Tensor& rc = p->inputTensors[keqv ? 4 : 5]; const Tensor& rs = p->inputTensors[keqv ? 5 : 6];
        const Tensor& qb = p->outputTensors[0]; const Tensor& kc = p->outputTensors[1]; const Tensor& vc = p->outputTensors[2];
        const uint64_t DH = qw.geometry.maxSizes[0], T = q.geometry.maxSizes[1];
        if ((DH != 256 && DH != 512) || kw.geometry.maxSizes[0] != DH || q.geometry.maxSizes[0] % DH || k.geometry.maxSizes[0] % DH || k.geometry.maxSizes[1] != T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (!keqv && (p->inputTensors[2].geometry.maxSizes[0] != k.geometry.maxSizes[0] || p->inputTensors[2].geometry.maxSizes[1] != T)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const uint64_t NV = DH / 128, NHs = q.geometry.maxSizes[0] / DH, NKVs = k.geometry.maxSizes[0] / DH;
        if (NHs == 0 || NKVs == 0 || NHs % NKVs) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const uint64_t GQs = NHs / NKVs, RV = rc.geometry.maxSizes[0] / 128;
        if (rc.geometry.dims != 2 || rs.geometry.dims != 2 || rc.geometry.maxSizes[0] % 128 || RV == 0 || RV > NV / 2 || rs.geometry.maxSizes[0] != rc.geometry.maxSizes[0] || rs.geometry.maxSizes[1] != rc.geometry.maxSizes[1]) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (qb.geometry.dims != 3 || kc.geometry.dims != 3 || vc.geometry.dims != 3 || qb.geometry.maxSizes[0] != DH || kc.geometry.maxSizes[0] != DH || vc.geometry.maxSizes[0] != DH) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (qb.geometry.maxSizes[2] % NKVs || qb.geometry.maxSizes[1] != GQs * T) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        const uint64_t nch = qb.geometry.maxSizes[2] / NKVs, C = kc.geometry.maxSizes[1]; const uint32_t ring = node_u32(p, 1);
        if (nch == 0 || C % 128 || kc.geometry.maxSizes[2] != NKVs * nch || vc.geometry.maxSizes[1] != C || vc.geometry.maxSizes[2] != NKVs * nch) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (ring && (nch != 1 || (C & (C - 1)))) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;   // a ring is one chunk of a power-of-two slot count
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = NHs + NKVs;   // q heads, then kv heads
        for (uint32_t i = 0; i < nin; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        for (uint32_t i = 0; i < 3; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
        out->kernel.paramsNr = 9; out->kernel.scalarParams[0] = (uint32_t)T; out->kernel.scalarParams[1] = node_u32(p, 0); out->kernel.scalarParams[2] = (uint32_t)nch;
        out->kernel.scalarParams[3] = (uint32_t)C; out->kernel.scalarParams[4] = (uint32_t)NKVs; out->kernel.scalarParams[5] = (uint32_t)GQs; out->kernel.scalarParams[6] = (uint32_t)NV;
        out->kernel.scalarParams[7] = (uint32_t)RV; out->kernel.scalarParams[8] = ring ? 1u : 0u;
        if (node_u32(p, 2)) { if (keqv) return GLUE_FAILED; return put_elf(out, "GAUDIUM_GAPREPV_ELF"); }   // node param 2: V rows without a norm (a block drafter's layers)
        return put_elf(out, keqv ? "GAUDIUM_GAPREPK_ELF" : "GAUDIUM_GAPREP_ELF");
    }
    if (g == "gaudium_ga_softmax_ring" || g == "gaudium_ga_softmax_ringx") {   // scores f32 {RING, R, G}, cursor i32 {1} [, ext i32 {T}: ringx] -> P bf16 {RING, R, G}, stats f32 {64, 2, R, G}; node params: T, coff, W, mode
        const bool ext = g == "gaudium_ga_softmax_ringx"; const uint32_t ni = ext ? 3 : 2;   // ringx: row t also sees the ext[t] positions after its own (an image block)
        if (p->inputTensorNr != ni) { p->inputTensorNr = ni; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 2) { p->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[3] = {DATA_F32, DATA_I32, DATA_I32}, ot[2] = {DATA_BF16, DATA_F32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& sc = p->inputTensors[0]; const Tensor& P = p->outputTensors[0]; const Tensor& st = p->outputTensors[1];
        const uint64_t RING = sc.geometry.maxSizes[0], R = sc.geometry.maxSizes[1], G = sc.geometry.maxSizes[2]; const uint32_t T = node_u32(p, 0), W = node_u32(p, 2);
        if (sc.geometry.dims != 3 || RING % 256 || RING > 4096 || (RING & (RING - 1)) || T == 0 || R % T || W == 0 || W > RING) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (ext && p->inputTensors[2].geometry.maxSizes[0] != T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (P.geometry.dims != 3 || P.geometry.maxSizes[0] != RING || P.geometry.maxSizes[1] != R || P.geometry.maxSizes[2] != G) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (st.geometry.dims != 4 || st.geometry.maxSizes[0] != 64 || st.geometry.maxSizes[1] != 2 || st.geometry.maxSizes[2] != R || st.geometry.maxSizes[3] != G) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = R * G;
        for (uint32_t i = 0; i < ni; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        for (uint32_t i = 0; i < 2; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
        out->kernel.paramsNr = 4; out->kernel.scalarParams[0] = T; out->kernel.scalarParams[1] = node_u32(p, 1); out->kernel.scalarParams[2] = W; out->kernel.scalarParams[3] = node_u32(p, 3);
        return put_elf(out, ext ? "GAUDIUM_GASOFTRX_ELF" : "GAUDIUM_GASOFTR_ELF");
    }
    if (g == "gaudium_ga_combine") {   // ao_part f32 {DH, R, G nch}, stats f32 {64, 2, R, G nch} -> ao f32 {DH, R, G}; node param: nch
        if (p->inputTensorNr != 2) { p->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 1) { p->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[2] = {DATA_F32, DATA_F32}, ot[1] = {DATA_F32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& A = p->inputTensors[0]; const Tensor& st = p->inputTensors[1]; const Tensor& O = p->outputTensors[0]; const uint32_t nch = node_u32(p, 0);
        if (O.geometry.dims != 3 || (O.geometry.maxSizes[0] != 256 && O.geometry.maxSizes[0] != 512)) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        const uint64_t DH = O.geometry.maxSizes[0], R = O.geometry.maxSizes[1], G = O.geometry.maxSizes[2];
        if (A.geometry.dims != 3 || A.geometry.maxSizes[0] != DH || A.geometry.maxSizes[1] != R || nch == 0 || A.geometry.maxSizes[2] != G * nch) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (st.geometry.dims != 4 || st.geometry.maxSizes[2] != R || st.geometry.maxSizes[3] != G * nch) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = R; out->indexSpaceGeometry[1] = G;
        for (uint32_t i = 0; i < 2; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        pattern(out->outputTensorAccessPattern[0], O, -1);
        out->kernel.paramsNr = 1; out->kernel.scalarParams[0] = nch;
        return put_elf(out, DH == 512 ? "GAUDIUM_GACOMB8_ELF" : "GAUDIUM_GACOMB4_ELF");
    }
    if (g == "gaudium_ga_post") {   // ao f32 {DH, GQ T, NKV} -> core bf16 {DH NH, T}
        if (p->inputTensorNr != 1) { p->inputTensorNr = 1; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 1) { p->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[1] = {DATA_F32}, ot[1] = {DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& A = p->inputTensors[0]; const Tensor& O = p->outputTensors[0];
        if (A.geometry.dims != 3 || (A.geometry.maxSizes[0] != 256 && A.geometry.maxSizes[0] != 512)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const uint64_t DH = A.geometry.maxSizes[0], NKVs = A.geometry.maxSizes[2], T = O.geometry.maxSizes[1];
        if (T == 0 || A.geometry.maxSizes[1] % T || O.geometry.maxSizes[0] % DH) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        const uint64_t GQs = A.geometry.maxSizes[1] / T, NHs = O.geometry.maxSizes[0] / DH;
        if (NHs != GQs * NKVs) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = NHs;
        pattern(out->inputTensorAccessPattern[0], A, -1); pattern(out->outputTensorAccessPattern[0], O, -1);
        out->kernel.paramsNr = 3; out->kernel.scalarParams[0] = (uint32_t)T; out->kernel.scalarParams[1] = (uint32_t)GQs; out->kernel.scalarParams[2] = (uint32_t)(DH / 128);
        return put_elf(out, "GAUDIUM_GAPOST_ELF");
    }
    if (g == "gaudium_ga_clamp") {   // x bf16 {C, cols}, lo / hi bf16 {128} -> y bf16 {C, cols}; index {ceil(C / 128), cols}
        if (p->inputTensorNr != 3) { p->inputTensorNr = 3; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 1) { p->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[3] = {DATA_BF16, DATA_BF16, DATA_BF16}, ot[1] = {DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& x = p->inputTensors[0]; const Tensor& lo = p->inputTensors[1]; const Tensor& hi = p->inputTensors[2]; const Tensor& y = p->outputTensors[0];
        if (x.geometry.dims != 2 || lo.geometry.dims != 1 || hi.geometry.dims != 1 || lo.geometry.maxSizes[0] != 128 || hi.geometry.maxSizes[0] != 128) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (y.geometry.dims != 2 || y.geometry.maxSizes[0] != x.geometry.maxSizes[0] || y.geometry.maxSizes[1] != x.geometry.maxSizes[1]) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = (x.geometry.maxSizes[0] + 127) / 128; out->indexSpaceGeometry[1] = x.geometry.maxSizes[1];
        pattern(out->inputTensorAccessPattern[0], x, -1); pattern(out->inputTensorAccessPattern[1], lo, -1); pattern(out->inputTensorAccessPattern[2], hi, -1); pattern(out->outputTensorAccessPattern[0], y, -1);
        out->kernel.paramsNr = 0; return put_elf(out, "GAUDIUM_GACLAMP_ELF");
    }
    if (g == "gaudium_ga_argmax") {   // x bf16 {V, T} -> idx int32 {1, T}; index {T}
        if (p->inputTensorNr != 1) { p->inputTensorNr = 1; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 1) { p->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[1] = {DATA_BF16}, ot[1] = {DATA_I32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& x = p->inputTensors[0]; const Tensor& y = p->outputTensors[0];
        if (x.geometry.dims != 2 || x.geometry.maxSizes[0] < 1 || x.geometry.maxSizes[0] >= (1u << 24)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (y.geometry.dims != 2 || y.geometry.maxSizes[0] != 1 || y.geometry.maxSizes[1] != x.geometry.maxSizes[1]) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = x.geometry.maxSizes[1];
        pattern(out->inputTensorAccessPattern[0], x, -1); pattern(out->outputTensorAccessPattern[0], y, -1);
        out->kernel.paramsNr = 0; return put_elf(out, "GAUDIUM_GAARGMAX_ELF");
    }
    if (g == "gaudium_ga_relattn") {   // q, k, v bf16 {NH 128, T}, r f32 {NH 128, W}, qs f32 {128} -> o bf16 {NH 128, T}; index {NH, T}; node params: cap, key scale (f32 bits)
        if (p->inputTensorNr != 5) { p->inputTensorNr = 5; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 1) { p->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[5] = {DATA_BF16, DATA_BF16, DATA_BF16, DATA_F32, DATA_F32}, ot[1] = {DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& q = p->inputTensors[0]; const Tensor& r = p->inputTensors[3]; const Tensor& qs = p->inputTensors[4]; const Tensor& o = p->outputTensors[0];
        const uint64_t C = q.geometry.maxSizes[0], T = q.geometry.maxSizes[1];
        if (q.geometry.dims != 2 || C % 128) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        for (int i = 1; i < 3; ++i) if (p->inputTensors[i].geometry.dims != 2 || p->inputTensors[i].geometry.maxSizes[0] != C || p->inputTensors[i].geometry.maxSizes[1] != T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (r.geometry.dims != 2 || r.geometry.maxSizes[0] != C || r.geometry.maxSizes[1] < 1 || r.geometry.maxSizes[1] > 16) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (qs.geometry.dims != 1 || qs.geometry.maxSizes[0] != 128) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (o.geometry.dims != 2 || o.geometry.maxSizes[0] != C || o.geometry.maxSizes[1] != T) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (p->nodeParams.nodeParamsSize < 8) return GLUE_FAILED;
        out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = C / 128; out->indexSpaceGeometry[1] = T;
        for (int i = 0; i < 5; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        pattern(out->outputTensorAccessPattern[0], o, -1);
        out->kernel.paramsNr = 2; out->kernel.scalarParams[0] = node_u32(p, 0); out->kernel.scalarParams[1] = node_u32(p, 1);
        return put_elf(out, "GAUDIUM_GARELATTN_ELF");
    }
    if (g == "gaudium_ga_act") {   // x bf16 {C, T} (SiLU) or {2 C, T} (GLU) -> y bf16 {C, T}; index {ceil(C / 128), T}; node param: 0 SiLU, 1 GLU
        if (p->inputTensorNr != 1) { p->inputTensorNr = 1; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 1) { p->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[1] = {DATA_BF16}, ot[1] = {DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& x = p->inputTensors[0]; const Tensor& y = p->outputTensors[0]; const uint32_t mode = node_u32(p, 0);
        if (mode > 1 || p->nodeParams.nodeParamsSize < 4) return GLUE_FAILED;
        const uint64_t C = y.geometry.maxSizes[0], T = y.geometry.maxSizes[1];
        if (y.geometry.dims != 2 || (mode == 1 && C % 128)) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (x.geometry.dims != 2 || x.geometry.maxSizes[0] != (mode ? 2 * C : C) || x.geometry.maxSizes[1] != T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = (C + 127) / 128; out->indexSpaceGeometry[1] = T;
        pattern(out->inputTensorAccessPattern[0], x, -1); pattern(out->outputTensorAccessPattern[0], y, -1);
        out->kernel.paramsNr = 1; out->kernel.scalarParams[0] = mode;
        return put_elf(out, "GAUDIUM_GAACT_ELF");
    }
    if (g == "gaudium_ga_dwconv") {   // x bf16 {C, T}, w bf16 {C, KW} -> y bf16 {C, T}; index {C / 128, T}
        if (p->inputTensorNr != 2) { p->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 1) { p->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[2] = {DATA_BF16, DATA_BF16}, ot[1] = {DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& x = p->inputTensors[0]; const Tensor& w = p->inputTensors[1]; const Tensor& y = p->outputTensors[0];
        const uint64_t C = x.geometry.maxSizes[0], T = x.geometry.maxSizes[1];
        if (x.geometry.dims != 2 || C % 128 || w.geometry.dims != 2 || w.geometry.maxSizes[0] != C || w.geometry.maxSizes[1] < 1) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (y.geometry.dims != 2 || y.geometry.maxSizes[0] != C || y.geometry.maxSizes[1] != T) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = C / 128; out->indexSpaceGeometry[1] = T;
        pattern(out->inputTensorAccessPattern[0], x, -1); pattern(out->inputTensorAccessPattern[1], w, -1); pattern(out->outputTensorAccessPattern[0], y, -1);
        out->kernel.paramsNr = 0; return put_elf(out, "GAUDIUM_GADWCONV_ELF");
    }
    if (g == "gaudium_ga_lnrelu") {   // x bf16 {C, P}, g f32 {128} -> y bf16 {C, P}; C <= 128 and even; index {P}; node param: eps (f32 bits)
        if (p->inputTensorNr != 2) { p->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 1) { p->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[2] = {DATA_BF16, DATA_F32}, ot[1] = {DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& x = p->inputTensors[0]; const Tensor& w = p->inputTensors[1]; const Tensor& y = p->outputTensors[0];
        const uint64_t C = x.geometry.maxSizes[0], P = x.geometry.maxSizes[1];
        if (x.geometry.dims != 2 || C > 128 || (C & 1) || w.geometry.dims != 1 || w.geometry.maxSizes[0] != 128) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (y.geometry.dims != 2 || y.geometry.maxSizes[0] != C || y.geometry.maxSizes[1] != P) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (p->nodeParams.nodeParamsSize < 4) return GLUE_FAILED;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = P;
        pattern(out->inputTensorAccessPattern[0], x, -1); pattern(out->inputTensorAccessPattern[1], w, -1); pattern(out->outputTensorAccessPattern[0], y, -1);
        out->kernel.paramsNr = 1; out->kernel.scalarParams[0] = node_u32(p, 0);
        return put_elf(out, "GAUDIUM_GALNRELU_ELF");
    }
    if (g == "gaudium_ga_lnorm") {   // x bf16 {C, P}, g f32 {C}, b f32 {C} -> y bf16 {C, P}; C a multiple of 128; index {P}; node param: eps (f32 bits)
        if (p->inputTensorNr != 3) { p->inputTensorNr = 3; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 1) { p->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[3] = {DATA_BF16, DATA_F32, DATA_F32}, ot[1] = {DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& x = p->inputTensors[0]; const Tensor& w = p->inputTensors[1]; const Tensor& bb = p->inputTensors[2]; const Tensor& y = p->outputTensors[0];
        const uint64_t C = x.geometry.maxSizes[0], P = x.geometry.maxSizes[1];
        if (x.geometry.dims != 2 || C == 0 || C % 128 || w.geometry.dims != 1 || w.geometry.maxSizes[0] != C || bb.geometry.dims != 1 || bb.geometry.maxSizes[0] != C) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (y.geometry.dims != 2 || y.geometry.maxSizes[0] != C || y.geometry.maxSizes[1] != P) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (p->nodeParams.nodeParamsSize < 4) return GLUE_FAILED;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = P;
        pattern(out->inputTensorAccessPattern[0], x, -1); pattern(out->inputTensorAccessPattern[1], w, -1); pattern(out->inputTensorAccessPattern[2], bb, -1); pattern(out->outputTensorAccessPattern[0], y, -1);
        out->kernel.paramsNr = 1; out->kernel.scalarParams[0] = node_u32(p, 0);
        return put_elf(out, "GAUDIUM_GALNORM_ELF");
    }
    if (g == "gaudium_ga_rmsnorm") {   // x bf16 {C, P}, w bf16 {C} -> y bf16 {C, P}; C a multiple of 128; index {P}: a row is one member, whatever P is; node param: eps (f32 bits)
        if (p->inputTensorNr != 2) { p->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 1) { p->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[2] = {DATA_BF16, DATA_BF16}, ot[1] = {DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& x = p->inputTensors[0]; const Tensor& w = p->inputTensors[1]; const Tensor& y = p->outputTensors[0];
        const uint64_t C = x.geometry.maxSizes[0], P = x.geometry.maxSizes[1];
        if (x.geometry.dims != 2 || C == 0 || C % 128 || w.geometry.dims != 1 || w.geometry.maxSizes[0] != C) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (y.geometry.dims != 2 || y.geometry.maxSizes[0] != C || y.geometry.maxSizes[1] != P) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (p->nodeParams.nodeParamsSize < 4) return GLUE_FAILED;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = P;
        pattern(out->inputTensorAccessPattern[0], x, -1); pattern(out->inputTensorAccessPattern[1], w, -1); pattern(out->outputTensorAccessPattern[0], y, -1);
        out->kernel.paramsNr = 1; out->kernel.scalarParams[0] = node_u32(p, 0);
        return put_elf(out, "GAUDIUM_GARMSNORM_ELF");
    }
    if (g == "gaudium_ga_copy") {   // x bf16 {C, cols} -> y bf16 {C, cols}; index {ceil(C / 128), cols}
        if (p->inputTensorNr != 1) { p->inputTensorNr = 1; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 1) { p->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[1] = {DATA_BF16}, ot[1] = {DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& x = p->inputTensors[0]; const Tensor& y = p->outputTensors[0];
        if (x.geometry.dims != 2) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (y.geometry.dims != 2 || y.geometry.maxSizes[0] != x.geometry.maxSizes[0] || y.geometry.maxSizes[1] != x.geometry.maxSizes[1]) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = (x.geometry.maxSizes[0] + 127) / 128; out->indexSpaceGeometry[1] = x.geometry.maxSizes[1];
        pattern(out->inputTensorAccessPattern[0], x, -1); pattern(out->outputTensorAccessPattern[0], y, -1);
        out->kernel.paramsNr = 0; return put_elf(out, "GAUDIUM_GACOPY_ELF");
    }
    if (g == "gaudium_ga_gelu" || g == "gaudium_ga_softcap") {   // elementwise over {F, T} bf16; index {ceil(F / 256), T}. The soft cap: F a multiple of 256. The activation: any F
        // (a last partial block of 256 loads padding beyond F and stores nothing there: the vision tower's MLP is 4304 wide)
        const bool gelu = g == "gaudium_ga_gelu"; const uint32_t nin = gelu ? 2 : 1;
        if (p->inputTensorNr != nin) { p->inputTensorNr = nin; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 1) { p->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[2] = {DATA_BF16, DATA_BF16}, ot[1] = {DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& y = p->outputTensors[0]; const uint64_t F = y.geometry.maxSizes[0], T = y.geometry.maxSizes[1];
        if (y.geometry.dims != 2 || (!gelu && F % 256)) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        for (uint32_t i = 0; i < nin; ++i) if (p->inputTensors[i].geometry.dims != 2 || p->inputTensors[i].geometry.maxSizes[0] != F || p->inputTensors[i].geometry.maxSizes[1] != T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = (F + 255) / 256; out->indexSpaceGeometry[1] = T;
        for (uint32_t i = 0; i < nin; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        pattern(out->outputTensorAccessPattern[0], y, -1);
        if (gelu) { out->kernel.paramsNr = 0; return put_elf(out, "GAUDIUM_GAGELU_ELF"); }
        if (p->nodeParams.nodeParamsSize < 4) return GLUE_FAILED;   // the cap (f32 bits)
        out->kernel.paramsNr = 1; out->kernel.scalarParams[0] = node_u32(p, 0);
        return put_elf(out, "GAUDIUM_GASOFTCAP_ELF");
    }
    return GLUE_NODE_NOT_FOUND;
}
}
