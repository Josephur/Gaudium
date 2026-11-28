// Graph-compiler glue of the Inkling 2 kernels: its own plugin library, libgaudium_ik.so, beside the other glue libraries in
// GC_KERNEL_PATH, so that adding or changing this family's kernels cannot change what the graph compiler builds for another family.
//   gaudium_ik_mix      block attention residual: softmax mix of the block states and the partial state by learned queries
//                       (the partial state bf16: GAUDIUM_IKMIX_ELF; f32: GAUDIUM_IKMIXF_ELF)
//   gaudium_ik_gin      the two normalised halves of a message gate's input                              GAUDIUM_IKGIN_ELF
//   gaudium_ik_mgate    (1 - tanh(f)) h + tanh(g) m (the gate rows bf16: GAUDIUM_IKMGATE_ELF; f32: GAUDIUM_IKMGATEF_ELF)
//   gaudium_ik_norm     weighted RMS norm of an f32 row                                                  GAUDIUM_IKNORM_ELF
//   gaudium_ik_save     the rows of a pass into a ring by position                                       GAUDIUM_IKSAVE_ELF
//   gaudium_ik_conv     short causal convolution + residual after the attention / (-DMLP) after the experts
//                                                                                    GAUDIUM_IKCONVA_ELF / GAUDIUM_IKCONVM_ELF
//   gaudium_ik_qprep    query heads and relative states: scale, clip, position scale                     GAUDIUM_IKQPREP_ELF
//   gaudium_ik_kvprep   key / value heads: short convolution, clip / norm, ring or cache write           GAUDIUM_IKKVPREP_ELF
//   gaudium_ik_softmax  masked softmax numerator with the relative-position bias, ring or chunked cache  GAUDIUM_IKSOFT_ELF
//   gaudium_ik_combine  merge of the chunk partials with the attention sink, output gate                 GAUDIUM_IKCOMB_ELF
//   gaudium_ik_route    router: top-K of sigmoid scores, dense weight table                              GAUDIUM_IKROUTE_ELF
//   gaudium_ik_moe_act  soft-capped SwiGLU of the dense-masked experts                                   GAUDIUM_IKACT_ELF
//   gaudium_ik_argmax   arg-max over the first NV rows                                                   GAUDIUM_IKARGMAX_ELF
// The block drafter (mtp.safetensors):
//   gaudium_ik_dmix     the four softmax-weighted, normalised mixes of the main model's taps                GAUDIUM_IKDMIX_ELF
//   gaudium_ik_dkv      context keys (norm, rotary position) and values into a ring or chunked cache       GAUDIUM_IKDKV_ELF
//   gaudium_ik_dqprep   the block's query / key / value heads (norm, rotary position)                      GAUDIUM_IKDQPREP_ELF
//   gaudium_ik_dsoft    softmax numerator over the context state and the block's keys                       GAUDIUM_IKDSOFT_ELF
//   gaudium_ik_dcomb    merge of the partials, normalisation                                                 GAUDIUM_IKDCOMB_ELF
//   gaudium_ik_dswiglu  plain SwiGLU                                                                         GAUDIUM_IKDSWIGLU_ELF
//   gaudium_ik_mstep    one step of the recurrent head (gate, candidate, state, latent)                     GAUDIUM_IKMSTEP_ELF
//   gaudium_ik_margmax  arg-max of a row's logits plus the head's vocabulary bias                           GAUDIUM_IKMARGMAX_ELF
// The head counts and widths are scalars read from the tensors or node parameters: one ELF serves every size of the family.
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

// ELF cache: the engine compiles on several threads, so the graph compiler calls this glue concurrently; one mutex covers every access.
static std::mutex& elf_mu() { static std::mutex mu; return mu; }
static std::vector<unsigned char>& elf_locked(const char* env) {   // caller holds elf_mu()
    static std::map<std::string, std::vector<unsigned char>> cache;   // keyed by variable AND path
    const char* p = getenv(env);
    std::vector<unsigned char>& e = cache[std::string(env) + "=" + (p ? p : "")];
    if (e.empty()) {
        FILE* f = p ? fopen(p, "rb") : nullptr;
        if (!f) { fprintf(stderr, "gaudium glue (inkling2): cannot open %s=%s\n", env, p ? p : "(null)"); return e; }
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
// every dim of every tensor whole: the kernels address their tensors by coordinates of their own
static void whole(TensorAccessPattern& ap, const Tensor& t) {
    memset(&ap, 0, sizeof(ap));
    ap.allRequired = 1;
    for (uint32_t d = 0; d < t.geometry.dims; ++d) {
        ap.mapping[d].indexSpaceDim = 0; ap.mapping[d].a = 0; ap.mapping[d].start_b = 0;
        ap.mapping[d].end_b = (float)(t.geometry.maxSizes[d] - 1); ap.mapping[d].allRequired = true;
    }
}
static void all_whole(const HabanaKernelParams* p, HabanaKernelInstantiation* out) {
    for (uint32_t i = 0; i < p->inputTensorNr; ++i) whole(out->inputTensorAccessPattern[i], p->inputTensors[i]);
    for (uint32_t i = 0; i < p->outputTensorNr; ++i) whole(out->outputTensorAccessPattern[i], p->outputTensors[i]);
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
static bool has_params(const HabanaKernelParams* p, int n) { return p->nodeParams.nodeParams && p->nodeParams.nodeParamsSize >= (uint32_t)(4 * n); }
static uint64_t dim(const Tensor& t, int d) { return t.geometry.maxSizes[d]; }
#define NIN(n) if (p->inputTensorNr != (n)) { p->inputTensorNr = (n); return GLUE_INCOMPATIBLE_INPUT_COUNT; }
#define NOUT(n) if (p->outputTensorNr != (n)) { p->outputTensorNr = (n); return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
#define IN(i) (p->inputTensors[i])
#define OUT(i) (p->outputTensors[i])

extern "C" {
uint64_t GetLibVersion() { return 78813002322618477ULL; }   // the interface version the other glue libraries report
GlueCodeReturn GetShapeInference(_IN_ DeviceId, _IN_ const ShapeInferenceParams*, _OUT_ ShapeInferenceOutput*) { return GLUE_SUCCESS; }

static const char* GUIDS[] = {"gaudium_ik_mix", "gaudium_ik_gin", "gaudium_ik_mgate", "gaudium_ik_norm", "gaudium_ik_save", "gaudium_ik_conv", "gaudium_ik_qprep", "gaudium_ik_kvprep", "gaudium_ik_softmax", "gaudium_ik_combine", "gaudium_ik_route", "gaudium_ik_moe_act", "gaudium_ik_argmax", "gaudium_ik_dmix", "gaudium_ik_dkv", "gaudium_ik_dqprep", "gaudium_ik_dsoft", "gaudium_ik_dcomb", "gaudium_ik_dswiglu", "gaudium_ik_mstep", "gaudium_ik_margmax", "gaudium_ik_route_r", "gaudium_ik_w4_plan", "gaudium_ik_w4_q8", "gaudium_ik_w8_q8", "gaudium_ik_w4_gu", "gaudium_ik_w4_dn", "gaudium_ik_w8_gu", "gaudium_ik_w8_dn", "gaudium_ik_w4_comb", "gaudium_ik_softcap", "gaudium_ik_mcap"};
GlueCodeReturn GetKernelGuids(_IN_ DeviceId deviceId, _INOUT_ uint32_t* kernelCount, _OUT_ GuidInfo* guids) {
    const uint32_t n = sizeof(GUIDS) / sizeof(GUIDS[0]);
    if (deviceId != DEVICE_ID_GAUDI2) { if (kernelCount) *kernelCount = 0; return GLUE_SUCCESS; }
    if (guids) { memset(guids, 0, n * sizeof(GuidInfo)); for (uint32_t i = 0; i < n; ++i) strcpy(guids[i].name, GUIDS[i]); }
    if (kernelCount) *kernelCount = n;
    return GLUE_SUCCESS;
}

GlueCodeReturn InstantiateTpcKernel(_IN_ HabanaKernelParams* p, _INOUT_ HabanaKernelInstantiation* out) {
    const std::string g = p->guid.name;
    if (g == "gaudium_ik_mix") {
        // 7 inputs: s0..s3 bf16 {H, T}, part bf16 {H, T}, query f32 {H, NQ}, ctl i32 {8} -> out f32 {H, T, NQ}
        // 8 inputs: the same with part f32 {H, T} and one more input xs bf16 {H, T} (the layer's input).   node params: eps, scale (f32 bits)
        const bool pf32 = p->inputTensorNr == 8;
        if (p->inputTensorNr != 7 && p->inputTensorNr != 8) { p->inputTensorNr = 7; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        NOUT(1)
        const TensorDataType in[8] = {DATA_BF16, DATA_BF16, DATA_BF16, DATA_BF16, pf32 ? DATA_F32 : DATA_BF16, DATA_F32, DATA_I32, DATA_BF16}, ot[1] = {DATA_F32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const uint64_t H = dim(OUT(0), 0), T = dim(OUT(0), 1), NQ = dim(OUT(0), 2);
        if (OUT(0).geometry.dims != 3 || H == 0 || H % 128 || NQ == 0 || NQ > 4) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        for (int i = 0; i < 5; ++i) if (IN(i).geometry.dims != 2 || dim(IN(i), 0) != H || dim(IN(i), 1) != T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (IN(5).geometry.dims != 2 || dim(IN(5), 0) != H || dim(IN(5), 1) != NQ || dim(IN(6), 0) < 3) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (pf32 && (IN(7).geometry.dims != 2 || dim(IN(7), 0) != H || dim(IN(7), 1) != T)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (!has_params(p, 2)) return GLUE_FAILED;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = T;
        all_whole(p, out);
        out->kernel.paramsNr = 2; out->kernel.scalarParams[0] = node_u32(p, 0); out->kernel.scalarParams[1] = node_u32(p, 1);
        return put_elf(out, pf32 ? "GAUDIUM_IKMIXF_ELF" : "GAUDIUM_IKMIX_ELF");
    }
    if (g == "gaudium_ik_gin") {   // h f32 {H, T, NQH}, m f32 {H, T, NQM} -> out bf16 {2 H, T}; node params: jh, jm, eps (f32 bits)
        NIN(2) NOUT(1)
        const TensorDataType in[2] = {DATA_F32, DATA_F32}, ot[1] = {DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const uint64_t H = dim(IN(0), 0), T = dim(IN(0), 1);
        if (!has_params(p, 3)) return GLUE_FAILED;
        if (IN(0).geometry.dims != 3 || IN(1).geometry.dims != 3 || H == 0 || H % 128 || dim(IN(1), 0) != H || dim(IN(1), 1) != T || node_u32(p, 0) >= dim(IN(0), 2) || node_u32(p, 1) >= dim(IN(1), 2)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (OUT(0).geometry.dims != 2 || dim(OUT(0), 0) != 2 * H || dim(OUT(0), 1) != T) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = T;
        all_whole(p, out);
        out->kernel.paramsNr = 3; for (int i = 0; i < 3; ++i) out->kernel.scalarParams[i] = node_u32(p, i);
        return put_elf(out, "GAUDIUM_IKGIN_ELF");
    }
    if (g == "gaudium_ik_mgate") {   // h f32 {H, T, NQH}, m f32 {H, T, NQM}, fg bf16 | f32 {2 H, T} -> out f32 {H, T} or {H, T, 1}; node params: jh, jm
        NIN(3) NOUT(1)
        const bool f32 = IN(2).geometry.dataType == DATA_F32;
        const TensorDataType in[3] = {DATA_F32, DATA_F32, f32 ? DATA_F32 : DATA_BF16}, ot[1] = {DATA_F32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const uint64_t H = dim(OUT(0), 0), T = dim(OUT(0), 1);
        if (!has_params(p, 2)) return GLUE_FAILED;
        if ((OUT(0).geometry.dims != 2 && !(OUT(0).geometry.dims == 3 && dim(OUT(0), 2) == 1)) || H == 0 || H % 128) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (IN(0).geometry.dims != 3 || IN(1).geometry.dims != 3 || dim(IN(0), 0) != H || dim(IN(0), 1) != T || dim(IN(1), 0) != H || dim(IN(1), 1) != T || node_u32(p, 0) >= dim(IN(0), 2) || node_u32(p, 1) >= dim(IN(1), 2)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (IN(2).geometry.dims != 2 || dim(IN(2), 0) != 2 * H || dim(IN(2), 1) != T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = H / 128; out->indexSpaceGeometry[1] = T;
        all_whole(p, out);
        out->kernel.paramsNr = 2; out->kernel.scalarParams[0] = node_u32(p, 0); out->kernel.scalarParams[1] = node_u32(p, 1);
        return put_elf(out, f32 ? "GAUDIUM_IKMGATEF_ELF" : "GAUDIUM_IKMGATE_ELF");
    }
    if (g == "gaudium_ik_norm") {   // x f32 {H, T, NQ}, w bf16 {H} -> y bf16 {H, T}; node params: j, eps (f32 bits)
        NIN(2) NOUT(1)
        const TensorDataType in[2] = {DATA_F32, DATA_BF16}, ot[1] = {DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const uint64_t H = dim(OUT(0), 0), T = dim(OUT(0), 1);
        if (!has_params(p, 2)) return GLUE_FAILED;
        if (OUT(0).geometry.dims != 2 || H == 0 || H % 128) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (IN(0).geometry.dims != 3 || dim(IN(0), 0) != H || dim(IN(0), 1) != T || node_u32(p, 0) >= dim(IN(0), 2) || IN(1).geometry.dims != 1 || dim(IN(1), 0) != H) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = T;
        all_whole(p, out);
        out->kernel.paramsNr = 2; out->kernel.scalarParams[0] = node_u32(p, 0); out->kernel.scalarParams[1] = node_u32(p, 1);
        return put_elf(out, "GAUDIUM_IKNORM_ELF");
    }
    if (g == "gaudium_ik_save") {   // x bf16 {>= C, T}, cursor i32 {1} -> ring bf16 {C, RC}; node param: coff
        NIN(2) NOUT(1)
        const TensorDataType in[2] = {DATA_BF16, DATA_I32}, ot[1] = {DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const uint64_t C = dim(OUT(0), 0), RC = dim(OUT(0), 1), T = dim(IN(0), 1);
        if (OUT(0).geometry.dims != 2 || C == 0 || C % 128 || RC == 0 || (RC & (RC - 1)) || RC < T) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (IN(0).geometry.dims != 2 || dim(IN(0), 0) < C) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = C / 128; out->indexSpaceGeometry[1] = T;
        all_whole(p, out);
        out->kernel.paramsNr = 1; out->kernel.scalarParams[0] = node_u32(p, 0);
        return put_elf(out, "GAUDIUM_IKSAVE_ELF");
    }
    if (g == "gaudium_ik_conv") {
        // 5 inputs: ring bf16 {H, RC}, cw f32 {H, KW}, xin bf16 {H, T}, ctl i32 {8}, cursor i32 {1} -> out f32 {H, T}
        // 4 inputs: ring, cw, res f32 {H, T}, cursor -> out bf16 {H, T}.   node params: coff, isd (f32 bits)
        const bool mlp = p->inputTensorNr == 4;
        if (p->inputTensorNr != 4 && p->inputTensorNr != 5) { p->inputTensorNr = 5; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        NOUT(1)
        const TensorDataType in_a[5] = {DATA_BF16, DATA_F32, DATA_BF16, DATA_I32, DATA_I32}, in_m[4] = {DATA_BF16, DATA_F32, DATA_F32, DATA_I32};
        const TensorDataType ot[1] = {mlp ? DATA_BF16 : DATA_F32};
        if (!all_dtype(p, mlp ? in_m : in_a, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const uint64_t H = dim(OUT(0), 0), T = dim(OUT(0), 1), RC = dim(IN(0), 1), KW = dim(IN(1), 1);
        if (!has_params(p, 2)) return GLUE_FAILED;
        if (OUT(0).geometry.dims != 2 || H == 0 || H % 128) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (IN(0).geometry.dims != 2 || dim(IN(0), 0) != H || (RC & (RC - 1)) || RC + 1 < T + KW) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (IN(1).geometry.dims != 2 || dim(IN(1), 0) != H || KW == 0 || KW > 16) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (IN(2).geometry.dims != 2 || dim(IN(2), 0) != H || dim(IN(2), 1) != T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (!mlp && dim(IN(3), 0) < 3) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = H / 128; out->indexSpaceGeometry[1] = T;
        all_whole(p, out);
        out->kernel.paramsNr = 2; out->kernel.scalarParams[0] = node_u32(p, 0); out->kernel.scalarParams[1] = node_u32(p, 1);
        return put_elf(out, mlp ? "GAUDIUM_IKCONVM_ELF" : "GAUDIUM_IKCONVA_ELF");
    }
    if (g == "gaudium_ik_qprep") {
        // qx bf16 {QW, T}, qsc f32 {128, NH}, rsc f32 {128, NH / 8}, tau f32 {L}, cursor i32 {1} -> qb bf16 {128, GQ T, NKV nch}, rs bf16 {128, (NH / 8) T}
        // node params: coff, NKV, roff, logscale, thr_q (f32 bits), thr_r (f32 bits)
        NIN(5) NOUT(2)
        const TensorDataType in[5] = {DATA_BF16, DATA_F32, DATA_F32, DATA_F32, DATA_I32}, ot[2] = {DATA_BF16, DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (!has_params(p, 6)) return GLUE_FAILED;
        const uint64_t T = dim(IN(0), 1), NH = dim(IN(1), 1), NKV = node_u32(p, 1), roff = node_u32(p, 2);
        if (IN(0).geometry.dims != 2 || IN(1).geometry.dims != 2 || IN(2).geometry.dims != 2 || dim(IN(1), 0) != 128 || dim(IN(2), 0) != 128) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (NKV == 0 || NH == 0 || NH % NKV || NH % 8 || dim(IN(2), 1) != NH / 8 || roff < 128 * NH || dim(IN(0), 0) < roff + 16 * NH) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const uint64_t GQ = NH / NKV;
        if (OUT(0).geometry.dims != 3 || dim(OUT(0), 0) != 128 || dim(OUT(0), 1) != GQ * T || dim(OUT(0), 2) == 0 || dim(OUT(0), 2) % NKV) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (OUT(1).geometry.dims != 2 || dim(OUT(1), 0) != 128 || dim(OUT(1), 1) != (NH / 8) * T) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = NH + NH / 8;
        all_whole(p, out);
        out->kernel.paramsNr = 9;
        out->kernel.scalarParams[0] = (uint32_t)T; out->kernel.scalarParams[1] = node_u32(p, 0); out->kernel.scalarParams[2] = (uint32_t)(dim(OUT(0), 2) / NKV);
        out->kernel.scalarParams[3] = (uint32_t)NKV; out->kernel.scalarParams[4] = (uint32_t)GQ; out->kernel.scalarParams[5] = (uint32_t)roff;
        out->kernel.scalarParams[6] = node_u32(p, 3) ? 1u : 0u; out->kernel.scalarParams[7] = node_u32(p, 4); out->kernel.scalarParams[8] = node_u32(p, 5);
        return put_elf(out, "GAUDIUM_IKQPREP_ELF");
    }
    if (g == "gaudium_ik_kvprep") {
        // kraw, vraw bf16 {128 NKV, RC}, kx, vx bf16 {KW_, T}, kcw, vcw f32 {128 NKV, KW}, cursor i32 {1} -> kcache, vcache bf16 {128, C, NKV nch}
        // node params: coff, ring, dyn, doff, isd (f32 bits), thr (f32 bits)
        NIN(7) NOUT(2)
        const TensorDataType in[7] = {DATA_BF16, DATA_BF16, DATA_BF16, DATA_BF16, DATA_F32, DATA_F32, DATA_I32}, ot[2] = {DATA_BF16, DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (!has_params(p, 6)) return GLUE_FAILED;
        const uint64_t KC = dim(IN(0), 0), RC = dim(IN(0), 1), T = dim(IN(2), 1), KW = dim(IN(4), 1), C = dim(OUT(0), 1);
        const uint32_t ring = node_u32(p, 1), dyn = node_u32(p, 2), doff = node_u32(p, 3);
        if (KC == 0 || KC % 128 || (RC & (RC - 1)) || RC + 1 < T + KW || KW == 0 || KW > 16) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const uint64_t NKV = KC / 128;
        for (int i = 0; i < 2; ++i) if (IN(i).geometry.dims != 2 || dim(IN(i), 0) != KC || dim(IN(i), 1) != RC) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        for (int i = 2; i < 4; ++i) if (IN(i).geometry.dims != 2 || dim(IN(i), 1) != T || (dyn && dim(IN(i), 0) < doff + KW * NKV)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        for (int i = 4; i < 6; ++i) if (IN(i).geometry.dims != 2 || dim(IN(i), 0) != KC || dim(IN(i), 1) != KW) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        for (int i = 0; i < 2; ++i) if (OUT(i).geometry.dims != 3 || dim(OUT(i), 0) != 128 || dim(OUT(i), 1) != C || dim(OUT(i), 2) == 0 || dim(OUT(i), 2) % NKV || dim(OUT(i), 2) != dim(OUT(0), 2)) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (C % 128 || (ring && (dim(OUT(0), 2) != NKV || (C & (C - 1))))) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = 2 * NKV;
        all_whole(p, out);
        out->kernel.paramsNr = 9;
        out->kernel.scalarParams[0] = (uint32_t)T; out->kernel.scalarParams[1] = node_u32(p, 0); out->kernel.scalarParams[2] = (uint32_t)C;
        out->kernel.scalarParams[3] = (uint32_t)NKV; out->kernel.scalarParams[4] = ring ? 1u : 0u; out->kernel.scalarParams[5] = dyn ? 1u : 0u;
        out->kernel.scalarParams[6] = doff; out->kernel.scalarParams[7] = node_u32(p, 4); out->kernel.scalarParams[8] = node_u32(p, 5);
        return put_elf(out, "GAUDIUM_IKKVPREP_ELF");
    }
    if (g == "gaudium_ik_softmax") {
        // scores f32 {C, R, B}, bias f32 {8 BW, (NH / 8) T}, cursor i32 {1} -> P bf16 {C, R, B}, stats f32 {64, 2, R, B}
        // node params: T, coff, W, ring, nch, GQ, BOFF, scale (f32 bits)
        NIN(3) NOUT(2)
        const TensorDataType in[3] = {DATA_F32, DATA_F32, DATA_I32}, ot[2] = {DATA_BF16, DATA_F32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (!has_params(p, 8)) return GLUE_FAILED;
        const Tensor& sc = IN(0); const Tensor& P = OUT(0); const Tensor& st = OUT(1);
        const uint64_t C = dim(sc, 0), R = dim(sc, 1), B = dim(sc, 2);
        const uint32_t T = node_u32(p, 0), W = node_u32(p, 2), ring = node_u32(p, 3), nch = node_u32(p, 4), GQ = node_u32(p, 5), BOFF = node_u32(p, 6);
        if (sc.geometry.dims != 3 || C == 0 || C % 128 || C > 4096 || T == 0 || GQ == 0 || R != (uint64_t)GQ * T || nch == 0 || B % nch) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (ring && (nch != 1 || (C & (C - 1)) || W == 0 || C + 1 < (uint64_t)W + T || C < (uint64_t)W + 64)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const uint64_t NH = (B / nch) * GQ;
        if (NH % 8 || IN(1).geometry.dims != 2 || dim(IN(1), 0) != 8ull * (BOFF + 65) || dim(IN(1), 1) != (NH / 8) * T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (P.geometry.dims != 3 || dim(P, 0) != C || dim(P, 1) != R || dim(P, 2) != B) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (st.geometry.dims != 4 || dim(st, 0) != 64 || dim(st, 1) != 2 || dim(st, 2) != R || dim(st, 3) != B) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = R * B;
        all_whole(p, out);
        out->kernel.paramsNr = 8; for (int i = 0; i < 8; ++i) out->kernel.scalarParams[i] = node_u32(p, i);
        return put_elf(out, "GAUDIUM_IKSOFT_ELF");
    }
    if (g == "gaudium_ik_combine") {
        // ao_part f32 {128, R, B}, stats f32 {64, 2, R, B}, qx bf16 {QW, T}, sink bf16 {NH, T} -> core bf16 {128 NH, T}; node params: nch, goff
        NIN(4) NOUT(1)
        const TensorDataType in[4] = {DATA_F32, DATA_F32, DATA_BF16, DATA_BF16}, ot[1] = {DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (!has_params(p, 2)) return GLUE_FAILED;
        const Tensor& A = IN(0); const Tensor& st = IN(1); const Tensor& O = OUT(0);
        const uint32_t nch = node_u32(p, 0), goff = node_u32(p, 1);
        const uint64_t R = dim(A, 1), B = dim(A, 2), T = dim(O, 1), NH = dim(IN(3), 0);
        if (A.geometry.dims != 3 || dim(A, 0) != 128 || nch == 0 || B % nch || T == 0 || R % T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const uint64_t G = B / nch, GQ = R / T;
        if (NH != G * GQ || st.geometry.dims != 4 || dim(st, 2) != R || dim(st, 3) != B) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (IN(2).geometry.dims != 2 || dim(IN(2), 1) != T || dim(IN(2), 0) < goff + 128 * NH || IN(3).geometry.dims != 2 || dim(IN(3), 1) != T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (O.geometry.dims != 2 || dim(O, 0) != 128 * NH) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = NH;
        all_whole(p, out);
        out->kernel.paramsNr = 4; out->kernel.scalarParams[0] = (uint32_t)T; out->kernel.scalarParams[1] = nch; out->kernel.scalarParams[2] = (uint32_t)GQ; out->kernel.scalarParams[3] = goff;
        return put_elf(out, "GAUDIUM_IKCOMB_ELF");
    }
    if (g == "gaudium_ik_route") {   // logits bf16 {E, T}, rb f32 {E} -> wd2 f32 {E / 2, 2, T}; node params: K, route_scale (f32 bits)
        NIN(2) NOUT(1)
        const TensorDataType in[2] = {DATA_BF16, DATA_F32}, ot[1] = {DATA_F32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (!has_params(p, 2)) return GLUE_FAILED;
        const uint64_t E = dim(IN(0), 0), T = dim(IN(0), 1); const uint32_t K = node_u32(p, 0);
        if (IN(0).geometry.dims != 2 || E == 0 || E % 128 || E > 512 || K == 0 || K > 12 || K > E || IN(1).geometry.dims != 1 || dim(IN(1), 0) != E) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (OUT(0).geometry.dims != 3 || dim(OUT(0), 0) != E / 2 || dim(OUT(0), 1) != 2 || dim(OUT(0), 2) != T) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = T;
        all_whole(p, out);
        out->kernel.paramsNr = 2; out->kernel.scalarParams[0] = K; out->kernel.scalarParams[1] = node_u32(p, 1);
        return put_elf(out, "GAUDIUM_IKROUTE_ELF");
    }
    if (g == "gaudium_ik_moe_act") {   // g, u bf16 {NE F, T}, wd2 f32 {NE / 2, 2, T} -> a bf16 {NE F, T}; node params: F, cg (f32 bits), cu (f32 bits)
        NIN(3) NOUT(1)
        const TensorDataType in[3] = {DATA_BF16, DATA_BF16, DATA_F32}, ot[1] = {DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (!has_params(p, 3)) return GLUE_FAILED;
        const uint32_t F = node_u32(p, 0);
        const uint64_t W = dim(OUT(0), 0), T = dim(OUT(0), 1);
        if (OUT(0).geometry.dims != 2 || F == 0 || F % 128 || W % F) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        const uint64_t NE = W / F;
        for (int i = 0; i < 2; ++i) if (IN(i).geometry.dims != 2 || dim(IN(i), 0) != W || dim(IN(i), 1) != T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (IN(2).geometry.dims != 3 || NE % 2 || dim(IN(2), 0) != NE / 2 || dim(IN(2), 1) != 2 || dim(IN(2), 2) != T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = NE; out->indexSpaceGeometry[1] = T;
        all_whole(p, out);
        out->kernel.paramsNr = 3; for (int i = 0; i < 3; ++i) out->kernel.scalarParams[i] = node_u32(p, i);
        return put_elf(out, "GAUDIUM_IKACT_ELF");
    }
    if (g == "gaudium_ik_argmax") {   // x bf16 {V, T} -> idx int32 {1, T}; node param: NV (0: all of V); index {T}
        NIN(1) NOUT(1)
        const TensorDataType in[1] = {DATA_BF16}, ot[1] = {DATA_I32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const Tensor& x = IN(0); const Tensor& y = OUT(0);
        if (x.geometry.dims != 2 || dim(x, 0) < 1 || dim(x, 0) >= (1u << 24) || node_u32(p, 0) > dim(x, 0)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (y.geometry.dims != 2 || dim(y, 0) != 1 || dim(y, 1) != dim(x, 1)) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = dim(x, 1);
        all_whole(p, out);
        out->kernel.paramsNr = 1; out->kernel.scalarParams[0] = node_u32(p, 0);
        return put_elf(out, "GAUDIUM_IKARGMAX_ELF");
    }
    if (g == "gaudium_ik_dmix") {   // x0..x3 bf16 {H, T}, w bf16 {H, 4}, alpha f32 {16} -> c0..c3 bf16 {H, T}; node param: eps (f32 bits)
        NIN(6) NOUT(4)
        const TensorDataType in[6] = {DATA_BF16, DATA_BF16, DATA_BF16, DATA_BF16, DATA_BF16, DATA_F32}, ot[4] = {DATA_BF16, DATA_BF16, DATA_BF16, DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (!has_params(p, 1)) return GLUE_FAILED;
        const uint64_t H = dim(OUT(0), 0), T = dim(OUT(0), 1);
        if (H == 0 || H % 128 || T == 0) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        for (int i = 0; i < 4; ++i) {
            if (IN(i).geometry.dims != 2 || dim(IN(i), 0) != H || dim(IN(i), 1) != T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
            if (OUT(i).geometry.dims != 2 || dim(OUT(i), 0) != H || dim(OUT(i), 1) != T) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        }
        if (IN(4).geometry.dims != 2 || dim(IN(4), 0) != H || dim(IN(4), 1) != 4 || dim(IN(5), 0) != 16) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = T;
        all_whole(p, out);
        out->kernel.paramsNr = 1; out->kernel.scalarParams[0] = node_u32(p, 0);
        return put_elf(out, "GAUDIUM_IKDMIX_ELF");
    }
    if (g == "gaudium_ik_dkv") {   // kv bf16 {256 NKV, T}, w bf16 {128}, rope_hi f32 {128, 2, NHI}, rope_lo f32 {128, 2, 256}, cursor i32 {1} -> kst, vst bf16 {128, C, B}; node params: ring, eps (f32 bits)
        NIN(5) NOUT(2)
        const TensorDataType in[5] = {DATA_BF16, DATA_BF16, DATA_F32, DATA_F32, DATA_I32}, ot[2] = {DATA_BF16, DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (!has_params(p, 2)) return GLUE_FAILED;
        const uint64_t KW = dim(IN(0), 0), T = dim(IN(0), 1), C = dim(OUT(0), 1), B = dim(OUT(0), 2); const uint32_t ring = node_u32(p, 0);
        if (IN(0).geometry.dims != 2 || KW == 0 || KW % 256 || T == 0 || dim(IN(1), 0) != 128) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (IN(2).geometry.dims != 3 || dim(IN(2), 0) != 128 || dim(IN(2), 1) != 2 || IN(3).geometry.dims != 3 || dim(IN(3), 0) != 128 || dim(IN(3), 1) != 2 || dim(IN(3), 2) != 256) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const uint64_t NKV = KW / 256;
        for (int i = 0; i < 2; ++i) if (OUT(i).geometry.dims != 3 || dim(OUT(i), 0) != 128 || dim(OUT(i), 1) != C || dim(OUT(i), 2) != B) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (C == 0 || C % 128 || B == 0 || B % NKV || (ring && (B != NKV || (C & (C - 1))))) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = NKV;
        all_whole(p, out);
        out->kernel.paramsNr = 3; out->kernel.scalarParams[0] = (uint32_t)T; out->kernel.scalarParams[1] = ring ? 1u : 0u; out->kernel.scalarParams[2] = node_u32(p, 1);
        return put_elf(out, "GAUDIUM_IKDKV_ELF");
    }
    if (g == "gaudium_ik_dqprep") {   // q bf16 {128 NH, T}, kv bf16 {256 NKV, T}, qn_w, kn_w bf16 {128}, rope_hi, rope_lo, cursor -> qb {128, GQ T, NKV nch}, qbb {128, GQ T, NKV}, kb, vb {128, 128, NKV}; node param: eps
        NIN(7) NOUT(4)
        const TensorDataType in[7] = {DATA_BF16, DATA_BF16, DATA_BF16, DATA_BF16, DATA_F32, DATA_F32, DATA_I32}, ot[4] = {DATA_BF16, DATA_BF16, DATA_BF16, DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (!has_params(p, 1)) return GLUE_FAILED;
        const uint64_t QW = dim(IN(0), 0), T = dim(IN(0), 1), KW = dim(IN(1), 0);
        if (IN(0).geometry.dims != 2 || QW == 0 || QW % 128 || T == 0 || T > 128 || IN(1).geometry.dims != 2 || KW == 0 || KW % 256 || dim(IN(1), 1) != T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const uint64_t NH = QW / 128, NKV = KW / 256;
        if (NH % NKV || dim(IN(2), 0) != 128 || dim(IN(3), 0) != 128) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (IN(4).geometry.dims != 3 || dim(IN(4), 0) != 128 || dim(IN(4), 1) != 2 || IN(5).geometry.dims != 3 || dim(IN(5), 0) != 128 || dim(IN(5), 1) != 2 || dim(IN(5), 2) != 256) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const uint64_t GQ = NH / NKV;
        if (OUT(0).geometry.dims != 3 || dim(OUT(0), 0) != 128 || dim(OUT(0), 1) != GQ * T || dim(OUT(0), 2) == 0 || dim(OUT(0), 2) % NKV) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (OUT(1).geometry.dims != 3 || dim(OUT(1), 0) != 128 || dim(OUT(1), 1) != GQ * T || dim(OUT(1), 2) != NKV) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        for (int i = 2; i < 4; ++i) if (OUT(i).geometry.dims != 3 || dim(OUT(i), 0) != 128 || dim(OUT(i), 1) != 128 || dim(OUT(i), 2) != NKV) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = NH + NKV;
        all_whole(p, out);
        out->kernel.paramsNr = 3; out->kernel.scalarParams[0] = (uint32_t)T; out->kernel.scalarParams[1] = (uint32_t)GQ; out->kernel.scalarParams[2] = node_u32(p, 0);
        return put_elf(out, "GAUDIUM_IKDQPREP_ELF");
    }
    if (g == "gaudium_ik_dsoft") {   // sc f32 {C, R, B}, sb f32 {128, R, G}, ctl i32 {>= 3} -> Pc bf16 {C, R, B}, Pb bf16 {128, R, G}, stats f32 {64, 2, R, B + G}; node params: T, which, ring, scale (f32 bits)
        NIN(3) NOUT(3)
        const TensorDataType in[3] = {DATA_F32, DATA_F32, DATA_I32}, ot[3] = {DATA_BF16, DATA_BF16, DATA_F32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (!has_params(p, 4)) return GLUE_FAILED;
        const uint64_t C = dim(IN(0), 0), R = dim(IN(0), 1), B = dim(IN(0), 2), G = dim(IN(1), 2); const uint32_t T = node_u32(p, 0), ring = node_u32(p, 2);
        if (IN(0).geometry.dims != 3 || C == 0 || C % 128 || C > 4096 || R == 0 || G == 0 || B % G) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (IN(1).geometry.dims != 3 || dim(IN(1), 0) != 128 || dim(IN(1), 1) != R || T == 0 || T > 128 || R % T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if ((ring && (B != G || (C & (C - 1)))) || dim(IN(2), 0) < 2 + node_u32(p, 1)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (OUT(0).geometry.dims != 3 || dim(OUT(0), 0) != C || dim(OUT(0), 1) != R || dim(OUT(0), 2) != B) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (OUT(1).geometry.dims != 3 || dim(OUT(1), 0) != 128 || dim(OUT(1), 1) != R || dim(OUT(1), 2) != G) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (OUT(2).geometry.dims != 4 || dim(OUT(2), 0) != 64 || dim(OUT(2), 1) != 2 || dim(OUT(2), 2) != R || dim(OUT(2), 3) != B + G) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = R * (B + G);
        all_whole(p, out);
        out->kernel.paramsNr = 4; for (int i = 0; i < 4; ++i) out->kernel.scalarParams[i] = node_u32(p, i);
        return put_elf(out, "GAUDIUM_IKDSOFT_ELF");
    }
    if (g == "gaudium_ik_dcomb") {   // aoc f32 {128, R, B}, aob f32 {128, R, G}, stats f32 {64, 2, R, B + G} -> core bf16 {128 NH, T}; node param: T
        NIN(3) NOUT(1)
        const TensorDataType in[3] = {DATA_F32, DATA_F32, DATA_F32}, ot[1] = {DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (!has_params(p, 1)) return GLUE_FAILED;
        const uint64_t R = dim(IN(0), 1), B = dim(IN(0), 2), G = dim(IN(1), 2), T = node_u32(p, 0), NH = dim(OUT(0), 0) / 128;
        if (IN(0).geometry.dims != 3 || dim(IN(0), 0) != 128 || G == 0 || B % G || T == 0 || R % T || IN(1).geometry.dims != 3 || dim(IN(1), 0) != 128 || dim(IN(1), 1) != R) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (IN(2).geometry.dims != 4 || dim(IN(2), 2) != R || dim(IN(2), 3) != B + G) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const uint64_t GQ = R / T;
        if (OUT(0).geometry.dims != 2 || dim(OUT(0), 0) % 128 || NH != G * GQ || dim(OUT(0), 1) != T) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = NH;
        all_whole(p, out);
        out->kernel.paramsNr = 2; out->kernel.scalarParams[0] = (uint32_t)T; out->kernel.scalarParams[1] = (uint32_t)GQ;
        return put_elf(out, "GAUDIUM_IKDCOMB_ELF");
    }
    if (g == "gaudium_ik_dswiglu") {   // gu bf16 {2 F, T} -> a bf16 {F, T}
        NIN(1) NOUT(1)
        const TensorDataType in[1] = {DATA_BF16}, ot[1] = {DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const uint64_t F = dim(OUT(0), 0), T = dim(OUT(0), 1);
        if (OUT(0).geometry.dims != 2 || F == 0 || F % 128) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (IN(0).geometry.dims != 2 || dim(IN(0), 0) != 2 * F || dim(IN(0), 1) != T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = F / 128; out->indexSpaceGeometry[1] = T;
        all_whole(p, out);
        out->kernel.paramsNr = 0;
        return put_elf(out, "GAUDIUM_IKDSWIGLU_ELF");
    }
    if (g == "gaudium_ik_mstep") {   // ctxp f32 {3 RK, TR}, swp f32 {3 RK, 1}, rb f32 {3 RK}, s bf16 {RK, 1} -> s' bf16 {RK, 1}, lat bf16 {RK, 1}; node param: col
        NIN(4) NOUT(2)
        const TensorDataType in[4] = {DATA_F32, DATA_F32, DATA_F32, DATA_BF16}, ot[2] = {DATA_BF16, DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (!has_params(p, 1)) return GLUE_FAILED;
        const uint64_t RK = dim(IN(3), 0);
        if (RK == 0 || RK % 128 || dim(IN(0), 0) != 3 * RK || node_u32(p, 0) >= dim(IN(0), 1) || dim(IN(1), 0) != 3 * RK || dim(IN(2), 0) != 3 * RK) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        for (int i = 0; i < 2; ++i) if (dim(OUT(i), 0) != RK) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = RK / 128;
        all_whole(p, out);
        out->kernel.paramsNr = 1; out->kernel.scalarParams[0] = node_u32(p, 0);
        return put_elf(out, "GAUDIUM_IKMSTEP_ELF");
    }
    if (g == "gaudium_ik_margmax") {   // base bf16 {V, TR}, bias bf16 {V, 1}, bf bf16 {V} -> idx int32 {1, 1}; node params: col, NV
        NIN(3) NOUT(1)
        const TensorDataType in[3] = {DATA_BF16, DATA_BF16, DATA_BF16}, ot[1] = {DATA_I32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (!has_params(p, 2)) return GLUE_FAILED;
        const uint64_t V = dim(IN(0), 0);
        if (IN(0).geometry.dims != 2 || V == 0 || V % 128 || V >= (1u << 24) || node_u32(p, 0) >= dim(IN(0), 1) || node_u32(p, 1) > V) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (dim(IN(1), 0) != V || dim(IN(2), 0) != V) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (OUT(0).geometry.dims != 2 || dim(OUT(0), 0) != 1 || dim(OUT(0), 1) != 1) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = 1;
        all_whole(p, out);
        out->kernel.paramsNr = 2; out->kernel.scalarParams[0] = node_u32(p, 0); out->kernel.scalarParams[1] = node_u32(p, 1);
        return put_elf(out, "GAUDIUM_IKMARGMAX_ELF");
    }
    if (g == "gaudium_ik_route_r") {   // logits bf16 {E, T}, rb f32 {E} -> ids i32 {K, T}, w f32 {K, T}; node params: K, route_scale (f32 bits)
        NIN(2) NOUT(2)
        const TensorDataType in[2] = {DATA_BF16, DATA_F32}, ot[2] = {DATA_I32, DATA_F32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (!has_params(p, 2)) return GLUE_FAILED;
        const uint64_t E = dim(IN(0), 0), T = dim(IN(0), 1); const uint32_t K = node_u32(p, 0);
        if (IN(0).geometry.dims != 2 || E == 0 || E % 128 || E > 512 || K == 0 || K > 12 || K > E || IN(1).geometry.dims != 1 || dim(IN(1), 0) != E) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (OUT(0).geometry.dims != 2 || dim(OUT(0), 0) != K || dim(OUT(0), 1) != T) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (OUT(1).geometry.dims != 2 || dim(OUT(1), 0) != K || dim(OUT(1), 1) != T) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = T;
        all_whole(p, out);
        out->kernel.paramsNr = 2; out->kernel.scalarParams[0] = K; out->kernel.scalarParams[1] = node_u32(p, 1);
        return put_elf(out, "GAUDIUM_IKROUTER_ELF");
    }
    if (g == "gaudium_ik_w4_plan") {   // ids i32 {K, T} -> plan i32 {>= 1 + 5 K T}; node param: E
        NIN(1) NOUT(1)
        const TensorDataType in[1] = {DATA_I32}, ot[1] = {DATA_I32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (!has_params(p, 1)) return GLUE_FAILED;
        const uint64_t K = dim(IN(0), 0), T = dim(IN(0), 1); const uint32_t E = node_u32(p, 0);
        if (IN(0).geometry.dims != 2 || K == 0 || K > 12 || T == 0 || E == 0 || E > 512) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (OUT(0).geometry.dims != 1 || dim(OUT(0), 0) < 1 + 5 * K * T) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = 1;
        all_whole(p, out);
        out->kernel.paramsNr = 3; out->kernel.scalarParams[0] = (uint32_t)K; out->kernel.scalarParams[1] = (uint32_t)T; out->kernel.scalarParams[2] = E;
        return put_elf(out, "GAUDIUM_IKPLAN_ELF");
    }
    if (g == "gaudium_ik_w4_q8" || g == "gaudium_ik_w8_q8") {   // x bf16 {H, T} -> xq i8 {256, H/32, T}, xs f32 {64, H/blk, T}
        NIN(1) NOUT(2)
        const bool w8 = g == "gaudium_ik_w8_q8"; const uint64_t blk = w8 ? 32 : 16;
        const TensorDataType in[1] = {DATA_BF16}, ot[2] = {DATA_I8, DATA_F32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        const uint64_t H = dim(IN(0), 0), T = dim(IN(0), 1);
        if (IN(0).geometry.dims != 2 || H == 0 || H % 128 || T == 0) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (OUT(0).geometry.dims != 3 || dim(OUT(0), 0) != 256 || dim(OUT(0), 1) != H / 32 || dim(OUT(0), 2) != T) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (OUT(1).geometry.dims != 3 || dim(OUT(1), 0) != 64 || dim(OUT(1), 1) != H / blk || dim(OUT(1), 2) != T) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = H / 128; out->indexSpaceGeometry[1] = T;
        all_whole(p, out);
        out->kernel.paramsNr = 0;
        return put_elf(out, w8 ? "GAUDIUM_IKW8Q8_ELF" : "GAUDIUM_IKW4Q8_ELF");
    }
    if (g == "gaudium_ik_w4_gu" || g == "gaudium_ik_w4_dn" || g == "gaudium_ik_w8_gu" || g == "gaudium_ik_w8_dn") {
        const bool gu = (g == "gaudium_ik_w4_gu" || g == "gaudium_ik_w8_gu");
        const bool w8 = (g == "gaudium_ik_w8_gu" || g == "gaudium_ik_w8_dn");
        const uint32_t nin = w8 ? 5 : 6, nout = gu ? 2 : 1;
        if (p->inputTensorNr != nin) { p->inputTensorNr = nin; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
        if (p->outputTensorNr != nout) { p->outputTensorNr = nout; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
        const TensorDataType in[6] = {DATA_I8, DATA_F32, DATA_I32, w8 ? DATA_I8 : DATA_U8, DATA_F16, DATA_I8};
        const TensorDataType ot[2] = {gu ? DATA_I8 : DATA_F32, DATA_F32};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (gu && !has_params(p, 2)) return GLUE_FAILED;
        const uint64_t blk = w8 ? 32 : 16;
        const uint64_t NBLK = dim(IN(4), 1), NT = dim(IN(3), 2), E = dim(IN(3), 3);
        const uint64_t IN_C = NBLK * blk;
        if (IN_C == 0 || NT == 0 || E == 0) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (IN(3).geometry.dims != 4 || dim(IN(3), 0) != 256 || dim(IN(3), 1) != (w8 ? IN_C : IN_C / 2)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (IN(4).geometry.dims != 4 || dim(IN(4), 0) != 256 || dim(IN(4), 2) != NT || dim(IN(4), 3) != E) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (!w8 && (IN(5).geometry.dims != 1 || dim(IN(5), 0) != 256)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        uint64_t K = 0, T = 0;
        if (gu) {
            if (OUT(0).geometry.dims != 4 || OUT(1).geometry.dims != 4) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
            const uint64_t EF = dim(OUT(0), 1) * 16;
            K = dim(OUT(0), 2); T = dim(OUT(0), 3);
            if (dim(OUT(0), 0) != 256 || EF == 0 || EF % 128 || EF / 128 != NT) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
            if (dim(OUT(1), 0) != 64 || dim(OUT(1), 1) != EF / blk || dim(OUT(1), 2) != K || dim(OUT(1), 3) != T) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
            if (IN(0).geometry.dims != 3 || dim(IN(0), 0) != 256 || dim(IN(0), 1) != IN_C / 32 || dim(IN(0), 2) != T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
            if (IN(1).geometry.dims != 3 || dim(IN(1), 0) != 64 || dim(IN(1), 1) != NBLK || dim(IN(1), 2) != T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        } else {
            if (OUT(0).geometry.dims != 3) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
            const uint64_t H = dim(OUT(0), 0); K = dim(OUT(0), 1); T = dim(OUT(0), 2);
            if (H == 0 || H % 256 || H / 256 != NT) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
            if (IN(0).geometry.dims != 4 || dim(IN(0), 0) != 256 || dim(IN(0), 1) != IN_C / 16 || dim(IN(0), 2) != K || dim(IN(0), 3) != T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
            if (IN(1).geometry.dims != 4 || dim(IN(1), 0) != 64 || dim(IN(1), 1) != NBLK || dim(IN(1), 2) != K || dim(IN(1), 3) != T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        }
        if (K == 0 || K > 12 || T == 0 || IN(2).geometry.dims != 1 || dim(IN(2), 0) < 1 + 5 * K * T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        uint64_t M = K * T; if (M > 240) M = 240; if (M == 0) M = 1;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = NT * M;
        all_whole(p, out);
        out->kernel.paramsNr = gu ? 5 : 3;
        out->kernel.scalarParams[0] = (uint32_t)NT; out->kernel.scalarParams[1] = (uint32_t)M; out->kernel.scalarParams[2] = (uint32_t)(K * T);
        if (gu) { out->kernel.scalarParams[3] = node_u32(p, 0); out->kernel.scalarParams[4] = node_u32(p, 1); }
        const char* env = w8 ? (gu ? "GAUDIUM_IKW8GU_ELF" : "GAUDIUM_IKW8DN_ELF") : (gu ? "GAUDIUM_IKW4GU_ELF" : "GAUDIUM_IKW4DN_ELF");
        return put_elf(out, env);
    }
    if (g == "gaudium_ik_w4_comb") {   // y f32 {H, K, T}, w f32 {K, T} -> out bf16 {H, T}; node param: K
        NIN(2) NOUT(1)
        const TensorDataType in[2] = {DATA_F32, DATA_F32}, ot[1] = {DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (!has_params(p, 1)) return GLUE_FAILED;
        const uint64_t H = dim(OUT(0), 0), T = dim(OUT(0), 1); const uint32_t K = node_u32(p, 0);
        if (OUT(0).geometry.dims != 2 || H == 0 || H % 128 || T == 0 || K == 0 || K > 12) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if (IN(0).geometry.dims != 3 || dim(IN(0), 0) != H || dim(IN(0), 1) != K || dim(IN(0), 2) != T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (IN(1).geometry.dims != 2 || dim(IN(1), 0) != K || dim(IN(1), 1) != T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = H / 128; out->indexSpaceGeometry[1] = T;
        all_whole(p, out);
        out->kernel.paramsNr = 1; out->kernel.scalarParams[0] = K;
        return put_elf(out, "GAUDIUM_IKW4COMB_ELF");
    }
    if (g == "gaudium_ik_softcap") {   // x bf16 {V, T} -> y bf16 {V, T}; node params: cap_bits, NV
        NIN(1) NOUT(1)
        const TensorDataType in[1] = {DATA_BF16}, ot[1] = {DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (!has_params(p, 2)) return GLUE_FAILED;
        const uint64_t V = dim(IN(0), 0), T = dim(IN(0), 1); const uint32_t NV = node_u32(p, 1);
        if (IN(0).geometry.dims != 2 || V == 0 || V % 128 || T == 0 || NV > V) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (OUT(0).geometry.dims != 2 || dim(OUT(0), 0) != V || dim(OUT(0), 1) != T) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = V / 128; out->indexSpaceGeometry[1] = T;
        all_whole(p, out);
        out->kernel.paramsNr = 2; out->kernel.scalarParams[0] = node_u32(p, 0); out->kernel.scalarParams[1] = NV;
        return put_elf(out, "GAUDIUM_IKSOFTCAP_ELF");
    }
    if (g == "gaudium_ik_mcap") {   // base bf16 {V, TR}, bias bf16 {V, 1}, bf bf16 {V} -> y bf16 {V, 1}; node params: col, NV, cap_bits
        NIN(3) NOUT(1)
        const TensorDataType in[3] = {DATA_BF16, DATA_BF16, DATA_BF16}, ot[1] = {DATA_BF16};
        if (!all_dtype(p, in, ot)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (!has_params(p, 3)) return GLUE_FAILED;
        const uint64_t V = dim(IN(0), 0); const uint32_t col = node_u32(p, 0), NV = node_u32(p, 1);
        if (IN(0).geometry.dims != 2 || V == 0 || V % 128 || col >= dim(IN(0), 1) || NV > V) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (IN(1).geometry.dims != 2 || dim(IN(1), 0) != V || dim(IN(1), 1) != 1 || IN(2).geometry.dims != 1 || dim(IN(2), 0) != V) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (OUT(0).geometry.dims != 2 || dim(OUT(0), 0) != V || dim(OUT(0), 1) != 1) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = V / 128;
        all_whole(p, out);
        out->kernel.paramsNr = 3; for (int i = 0; i < 3; ++i) out->kernel.scalarParams[i] = node_u32(p, i);
        return put_elf(out, "GAUDIUM_IKMCAP_ELF");
    }
    return GLUE_NODE_NOT_FOUND;
}
}
