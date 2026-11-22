// Graph-compiler glue of the step kernels added after the first kernel set: its own plugin library, libgaudium_st.so, beside
// libgaudium_tpc.so in GC_KERNEL_PATH. A separate library keeps libgaudium_tpc.so the same binary, so recipes that use none of
// these kernels keep their recipe-cache keys.
//   gaudium_gdn_f32v   gaudium_gdn_f32 with a slot input (state checkpoint per row), the per-row gate scalars taken as vector
//                      loads: bit-identical outputs (gdn_f32v_gaudi2.c)                                                 ELF: GAUDIUM_GDNV_ELF
//   gaudium_gdn_f32r   gated DeltaNet recurrence of a verify step without per-row state checkpoints: the accepted rows of the
//                      previous step are applied to the state again from their stored updates (gdn_f32r_gaudi2.c)   ELF: GAUDIUM_GDNR_ELF
//   gaudium_ds_ctxr    context K / V rows of a block drafter's layer into rings of any length (ds_ctxr_gaudi2.c)       ELF: GAUDIUM_DSCTXR_ELF
//   gaudium_ds_softr   block softmax over a context ring and the causal block, sweeping only the slots a row can see
//                      (ds_softr_gaudi2.c)                                                                          ELF: GAUDIUM_DSSOFTR_ELF
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

// ELF cache: the graph compiler may call this glue from several threads; one mutex covers every access (lookup / insert, the
// file read, the size check and the copy), as in libgaudium_tpc.so.
static std::mutex& elf_mu() { static std::mutex mu; return mu; }
static std::vector<unsigned char>& elf_locked(const char* env) {   // caller holds elf_mu()
    static std::map<std::string, std::vector<unsigned char>> cache;   // keyed by variable AND path
    const char* p = getenv(env);
    std::vector<unsigned char>& e = cache[std::string(env) + "=" + (p ? p : "")];
    if (e.empty()) {
        FILE* f = p ? fopen(p, "rb") : nullptr;
        if (!f) { fprintf(stderr, "gaudium glue (step): cannot open %s=%s\n", env, p ? p : "(null)"); return e; }
        fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
        e.resize(n); if (fread(e.data(), 1, n, f) != (size_t)n) e.clear(); fclose(f);
    }
    return e;
}
#include "elf_record.inc"   // which ELFs this library handed to the graph compiler
static GlueCodeReturn put_elf(HabanaKernelInstantiation* out, const char* env) {
    elf_note(env);
    std::lock_guard<std::mutex> lk(elf_mu());
    std::vector<unsigned char>& e = elf_locked(env);
    if (e.empty()) return GLUE_FAILED;
    uint32_t given = out->kernel.elfSize; out->kernel.elfSize = (uint32_t)e.size();
    if (given < e.size()) return GLUE_INSUFFICIENT_ELF_BUFFER;
    memcpy(out->kernel.kernelElf, e.data(), e.size()); return GLUE_SUCCESS;
}
// tensor dim `split_dim` maps 1:1 onto index-space dim 0; other dims whole (-1: every dim whole).
static void pattern(TensorAccessPattern& ap, const Tensor& t, int split_dim) {
    memset(&ap, 0, sizeof(ap));
    ap.allRequired = 1;
    for (uint32_t d = 0; d < t.geometry.dims; ++d) {
        ap.mapping[d].indexSpaceDim = 0;
        if ((int)d == split_dim) { ap.mapping[d].a = 1.f; ap.mapping[d].start_b = 0; ap.mapping[d].end_b = 0; }
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

static const char* GUIDS[] = {"gaudium_gdn_f32r", "gaudium_gdn_f32v", "gaudium_ds_ctxr", "gaudium_ds_softr"};
GlueCodeReturn GetKernelGuids(_IN_ DeviceId deviceId, _INOUT_ uint32_t* kernelCount, _OUT_ GuidInfo* guids) {
    const uint32_t n = sizeof(GUIDS) / sizeof(GUIDS[0]);
    if (deviceId != DEVICE_ID_GAUDI2) { if (kernelCount) *kernelCount = 0; return GLUE_SUCCESS; }
    if (guids) { memset(guids, 0, n * sizeof(GuidInfo)); for (uint32_t i = 0; i < n; ++i) strcpy(guids[i].name, GUIDS[i]); }
    if (kernelCount) *kernelCount = n;
    return GLUE_SUCCESS;
}

GlueCodeReturn InstantiateTpcKernel(_IN_ HabanaKernelParams* p, _INOUT_ HabanaKernelInstantiation* out) {
    const std::string g = p->guid.name;
    if (g == "gaudium_gdn_f32r") {
        // S_in f32 {64, 256, HV}; q, k, v f32 {128, HV, T}; eg, beta f32 {64, T}; pk f32 {128, HV, TP}; pd f32 {64, 8, HV, TP};
        // peg f32 {64, TP}; slot, cursor, cur0 i32 {1} -> o f32 {128, HV, T}, S_out f32 {64, 256, HV}, dr f32 {64, 8, HV, T}
        if (p->inputTensorNr != 12) { p->inputTensorNr = 12; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 3) { p->outputTensorNr = 3; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[12] = {DATA_F32, DATA_F32, DATA_F32, DATA_F32, DATA_F32, DATA_F32, DATA_F32, DATA_F32, DATA_F32, DATA_I32, DATA_I32, DATA_I32},
                             ot[3] = {DATA_F32, DATA_F32, DATA_F32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& S = p->inputTensors[0]; const Tensor& q = p->inputTensors[1]; const Tensor& pk = p->inputTensors[6]; const Tensor& pd = p->inputTensors[7];
        const Tensor& So = p->outputTensors[1]; const Tensor& dr = p->outputTensors[2];
        if (S.geometry.dims != 3 || S.geometry.maxSizes[0] != 64 || S.geometry.maxSizes[1] != 256) return GLUE_INCOMPATIBLE_INPUT_SIZE;   // blob layout
        const uint64_t HV = S.geometry.maxSizes[2], T = q.geometry.maxSizes[2];
        if (q.geometry.dims != 3 || q.geometry.maxSizes[0] != 128 || q.geometry.maxSizes[1] != HV || T < 1 || T > 32) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (pk.geometry.dims != 3 || pk.geometry.maxSizes[0] != 128 || pk.geometry.maxSizes[1] != HV) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (pd.geometry.dims != 4 || pd.geometry.maxSizes[0] != 64 || pd.geometry.maxSizes[1] != 8 || pd.geometry.maxSizes[2] != HV || pd.geometry.maxSizes[3] != pk.geometry.maxSizes[2]) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (So.geometry.dims != 3 || So.geometry.maxSizes[0] != 64 || So.geometry.maxSizes[1] != 256 || So.geometry.maxSizes[2] != HV) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (dr.geometry.dims != 4 || dr.geometry.maxSizes[0] != 64 || dr.geometry.maxSizes[1] != 8 || dr.geometry.maxSizes[2] != HV || dr.geometry.maxSizes[3] != T) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = HV;   // one value head per index-space point
        pattern(out->inputTensorAccessPattern[0], p->inputTensors[0], 2);
        pattern(out->inputTensorAccessPattern[1], p->inputTensors[1], 1);
        pattern(out->inputTensorAccessPattern[2], p->inputTensors[2], 1);
        pattern(out->inputTensorAccessPattern[3], p->inputTensors[3], 1);
        pattern(out->inputTensorAccessPattern[4], p->inputTensors[4], -1);   // vector loads reach past a head's own element
        pattern(out->inputTensorAccessPattern[5], p->inputTensors[5], -1);
        pattern(out->inputTensorAccessPattern[6], p->inputTensors[6], 1);
        pattern(out->inputTensorAccessPattern[7], p->inputTensors[7], 2);
        pattern(out->inputTensorAccessPattern[8], p->inputTensors[8], -1);
        pattern(out->inputTensorAccessPattern[9], p->inputTensors[9], -1);
        pattern(out->inputTensorAccessPattern[10], p->inputTensors[10], -1);
        pattern(out->inputTensorAccessPattern[11], p->inputTensors[11], -1);
        pattern(out->outputTensorAccessPattern[0], p->outputTensors[0], 1);
        pattern(out->outputTensorAccessPattern[1], p->outputTensors[1], 2);
        pattern(out->outputTensorAccessPattern[2], p->outputTensors[2], 2);
        out->kernel.paramsNr = 2; out->kernel.scalarParams[0] = (uint32_t)T; out->kernel.scalarParams[1] = node_u32(p, 0);
        return put_elf(out, "GAUDIUM_GDNR_ELF");
    }
    if (g == "gaudium_gdn_f32v") {
        // S_in f32 {64, 256, HV, NS}; q, k, v f32 {128, HV, T}; eg, beta f32 {64, T}; slot i32 {1|2}
        // -> o f32 {128, HV, T}, S_out f32 {64, 256, HV}, S_ckpt f32 {64, 256, HV, T}; node param 0: flags (gdn_f32_gaudi2.c)
        if (p->inputTensorNr != 7) { p->inputTensorNr = 7; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 3) { p->outputTensorNr = 3; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[7] = {DATA_F32, DATA_F32, DATA_F32, DATA_F32, DATA_F32, DATA_F32, DATA_I32}, ot[3] = {DATA_F32, DATA_F32, DATA_F32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& S = p->inputTensors[0]; const Tensor& q = p->inputTensors[1];
        if (S.geometry.dims != 4 || S.geometry.maxSizes[0] != 64 || S.geometry.maxSizes[1] != 256) return GLUE_INCOMPATIBLE_INPUT_SIZE;   // blob layout, slotted
        const uint64_t HV = S.geometry.maxSizes[2], T = q.geometry.maxSizes[2];
        if (q.geometry.dims != 3 || q.geometry.maxSizes[0] != 128 || q.geometry.maxSizes[1] != HV || T < 1 || T > 32) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = HV;   // one value head per index-space point
        pattern(out->inputTensorAccessPattern[0], p->inputTensors[0], 2);
        pattern(out->inputTensorAccessPattern[1], p->inputTensors[1], 1);
        pattern(out->inputTensorAccessPattern[2], p->inputTensors[2], 1);
        pattern(out->inputTensorAccessPattern[3], p->inputTensors[3], 1);
        pattern(out->inputTensorAccessPattern[4], p->inputTensors[4], -1);   // vector loads reach past a head's own element
        pattern(out->inputTensorAccessPattern[5], p->inputTensors[5], -1);
        pattern(out->inputTensorAccessPattern[6], p->inputTensors[6], -1);
        pattern(out->outputTensorAccessPattern[0], p->outputTensors[0], 1);
        pattern(out->outputTensorAccessPattern[1], p->outputTensors[1], 2);
        pattern(out->outputTensorAccessPattern[2], p->outputTensors[2], 2);
        out->kernel.paramsNr = 2; out->kernel.scalarParams[0] = (uint32_t)T; out->kernel.scalarParams[1] = node_u32(p, 0);
        return put_elf(out, "GAUDIUM_GDNV_ELF");
    }
    if (g == "gaudium_ds_ctxr") {
        // kc, vc bf16 {1024, T}; kn_w f32 {128}; rope_c, rope_s f32 {128, L}; cursor i32 {1}; depth i32 {>= T}; lim i32 {2} (not read)
        // -> kring, vring bf16 {128, RG, 8}
        if (p->inputTensorNr != 8) { p->inputTensorNr = 8; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 2) { p->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[8] = {DATA_BF16, DATA_BF16, DATA_F32, DATA_F32, DATA_F32, DATA_I32, DATA_I32, DATA_I32}, ot[2] = {DATA_BF16, DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& k = p->inputTensors[0]; const Tensor& kr = p->outputTensors[0]; const Tensor& vr = p->outputTensors[1];
        const uint64_t T = k.geometry.maxSizes[1], RG = kr.geometry.maxSizes[1];
        if (k.geometry.maxSizes[0] != 1024 || p->inputTensors[1].geometry.maxSizes[0] != 1024 || p->inputTensors[1].geometry.maxSizes[1] != T
            || p->inputTensors[2].geometry.maxSizes[0] != 128 || p->inputTensors[6].geometry.maxSizes[0] < T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        // a ring is whole 128-row blocks (the V row permutation) and no pass writes a slot twice
        if (kr.geometry.dims != 3 || vr.geometry.dims != 3 || kr.geometry.maxSizes[0] != 128 || kr.geometry.maxSizes[2] != 8
            || vr.geometry.maxSizes[0] != 128 || vr.geometry.maxSizes[2] != 8 || vr.geometry.maxSizes[1] != RG
            || RG < 128 || RG % 128 || T < 1 || T > RG) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = 8;   // the kv heads
        for (uint32_t i = 0; i < 8; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        for (uint32_t i = 0; i < 2; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
        out->kernel.paramsNr = 2; out->kernel.scalarParams[0] = (uint32_t)T; out->kernel.scalarParams[1] = (uint32_t)RG;
        return put_elf(out, "GAUDIUM_DSCTXR_ELF");
    }
    if (g == "gaudium_ds_softr") {
        // sc f32 {RG, R, 8}; sb f32 {128, R, 8}; cur i32 {1}; lim i32 {2} -> Pc bf16 {RG, R, 8}, Pb bf16 {128, R, 8}; node params: T, W
        if (p->inputTensorNr != 4) { p->inputTensorNr = 4; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != 2) { p->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[4] = {DATA_F32, DATA_F32, DATA_I32, DATA_I32}, ot[2] = {DATA_BF16, DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& sc = p->inputTensors[0]; const Tensor& sb = p->inputTensors[1];
        const uint64_t RG = sc.geometry.maxSizes[0], R = sc.geometry.maxSizes[1], G = sc.geometry.maxSizes[2];
        const uint32_t T = node_u32(p, 0), W = node_u32(p, 1);
        // RG >= W + 256: the aligned sweeps of a wrapped window's two slot ranges never meet (ds_softr_gaudi2.c)
        if (sc.geometry.dims != 3 || T == 0 || T > 64 || R % T || W < 2 || RG < 128 || RG % 128 || RG < (uint64_t)W + 256) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (sb.geometry.maxSizes[0] != 128 || sb.geometry.maxSizes[1] != R || sb.geometry.maxSizes[2] != G) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        for (uint32_t i = 0; i < 2; ++i) {
            const Tensor& o = p->outputTensors[i];
            if (o.geometry.dims != 3 || o.geometry.maxSizes[0] != (i ? 128 : RG) || o.geometry.maxSizes[1] != R || o.geometry.maxSizes[2] != G) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        }
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = R * G;   // one (query row, kv group) per index-space point
        for (uint32_t i = 0; i < 4; ++i) pattern(out->inputTensorAccessPattern[i], p->inputTensors[i], -1);
        for (uint32_t i = 0; i < 2; ++i) pattern(out->outputTensorAccessPattern[i], p->outputTensors[i], -1);
        out->kernel.paramsNr = 3; out->kernel.scalarParams[0] = T; out->kernel.scalarParams[1] = (uint32_t)RG; out->kernel.scalarParams[2] = W;
        return put_elf(out, "GAUDIUM_DSSOFTR_ELF");
    }
    return GLUE_NODE_NOT_FOUND;
}
}
