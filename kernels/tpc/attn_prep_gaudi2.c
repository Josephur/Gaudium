#include "attn_shape.h"   // ATT_NKV, ATT_GQ, ATT_NH, ATT_NPT, ATT_GATE: compile-time head counts (issue #7); the defaults are Qwen3.8-27B's
// gaudium_attn_prep: Qwen3.5 full-attention front end for T new tokens at device cursor (+ coff).
//   q heads (24): per-head RMSNorm (w) -> partial NeoX RoPE on channels 0..63 -> * 1/sqrt(256) -> bf16 Q rows for batch_gemm
//   kv heads (4): per-head RMSNorm (w) -> RoPE -> K cache row ; V -> V cache row perm(.)
//  in0 qg bf16 {12288, T} (rows 256h.. = q of head h; 6144 + 256h.. = gate)   in1 k bf16 {1024, T}   in2 v bf16 {1024, T}
//  in3 qn_w f32 {256} (lane order)   in4 kn_w f32 {256} (lane order)
//  in5 rope_c f32 {128, Lmax}: lanes [0..15] cos(pos*f_2j), [16..31] same, [32..63] 1; lanes 64.. the same for the odd frequencies f_2j+1
//  in6 rope_s f32 {128, Lmax}: lanes [0..15] -sin(pos*f_2j), [16..31] +sin(pos*f_2j), [32..63] 0; lanes 64.. odd frequencies
//  in7 cursor int32 {1}
//  out0 qb bf16 {256, 6T, 4*nch}: row hl*T + t of batch ch*4 + g holds q head 6g + hl (written once per chunk ch)
//  out1 kcache bf16 {256, C, 4*nch}   out2 vcache bf16 {256, C, 4*nch}: the bucket of Lmax = nch*C positions viewed per chunk,
//      CHUNK-MAJOR batch b = ch*4 + g, position pos -> {pos % C, (pos / C)*4 + g}. A smaller bucket (fewer chunks) of the same
//      cache is a memory PREFIX of the larger one, so prefill tiles bind the layer's full cache with a bucket-sized shape.
//      nch = 1, C = Lmax is the unchunked layout.
//  Lane order: bf16 loads deinterleave (v1 = even channels, v2 = odd). RoPE pairs (i, i+32), i < 32, both even or both odd,
//  so inside each of the two f32 vectors of channels 0..127 the pair sits in lanes (j, j+16) of dual groups 0 and 1.
//  V cache rows are permuted inside each 128-row block so that the softmax kernel's interleaving f32->bf16 conversion of
//  P keeps P[m] paired with the right V row: perm(j) = j < 64 ? 2j : 2(j-64)+1 (C multiple of 128 keeps it inside a chunk).
//  scalars: T, coff (positions are cursor[0] + coff + t: lets a draft pass run at cursor_out + i without a host-written cursor), nch, C
//  -DFRESH build (engine, MTP passes of a chained step recipe): the pass does NOT write the big cache. Two extra inputs
//  kf_in / vf_in {256, NF, 4} (the fresh window of the previous pass of the same step) are copied into the outputs
//  kf_out / vf_out {256, NF, 4} (rows outside [sbase, sbase+T)) and the T new rows go to window slots sbase + t (K row = slot,
//  V row = perm(slot)); qb_f {256, 6T, 4} is the q block for the window GEMM. Outputs: qb, qb_f, kf_out, vf_out.
//  scalars: T, coff, nch, C, sbase, NF.  Every pass of the chain reads its predecessor's window and writes its own: no aliasing.
//  -DTREE build (draft-tree verify; ports feature/dflash2's attn_prep -DTREE to the chunk-major cache):
//  + depth int32 {>= T}: row t (tree node t) sits at KV SLOT cursor + coff + t but its RoPE POSITION is cursor + coff + depth[t];
//  + fix int32 {2F}: F (src, dst) slot pairs, src < 0 = none, applied in order BEFORE the new rows are written: the previous step's
//    accepted path nodes move from their tree slots to their path slots (gaudium_accept_tree). Pair i's destination is never a
//    later pair's source (path slots ascend, a path node's index >= its depth), so in-order copying in one instance is safe; each
//    kv head is one index point, so no other TPC touches its rows. Loads of fix sources precede every store of this instance to the
//    same rows (no load-after-store, hardware-learnings section 5). Single writer of the cache inside the recipe.
//  (integration: one build for both tree users: DFlash2 passes F = 1 (fix {2}), the MTP tree up to 4 path moves (fix {8}).)
//  -DATT_DH=128 builds (plain Qwen3, GUID gaudium_attn128_prep): a head is ONE 128-channel row (one bfloat128 load = {v1 even, v2 odd}
//  channels), q_proj carries q only (no output gate), RoPE covers all 128 channels (NeoX pairs (i, i+64): lanes (j, j+32) of the same vector,
//  dual groups 0<->2 and 1<->3, as gaudium_df_ctx's ROPE128), q is scaled by 1/sqrt(128) in f32 before the bf16 store. Tables: rope_c / rope_s
//  {128, Lmax} lanes [0..31] cos / -sin of the even frequencies f_2j, [32..63] cos / +sin of the same, lanes 64.. the odd frequencies
//  (qw::Packer::rope_rows with ROT = 128). Plain and -DTREE builds only (no FRESH: the MTP passes; no KV8).
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define REDUCE64(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = _a + _r; _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = _a + _r; \
    _a = _a + v_f32_shuffle_b(_a, P8, 0, _a); _a = _a + v_f32_shuffle_b(_a, P4, 0, _a); \
    _a = _a + v_f32_shuffle_b(_a, P2, 0, _a); _a = _a + v_f32_shuffle_b(_a, P1, 0, _a); _a; })
static inline uchar256 xor_pat(uchar256 l16, int s) { uchar256 src = l16 ^ (unsigned char)s; return ((src & 7) | ((src >> 3) << 5)) | 0x80; }
#define ROPE(x, C, S) ({ float64 _x = (x); float64 _sw = MOVDG_ALL(_x, 1, 0, 2, 3, _x); _x * (C) + _sw * (S); })
#define VPERM(j) (((j) & ~127) + ((((j) & 127) < 64) ? 2 * ((j) & 127) : 2 * (((j) & 127) - 64) + 1))
#ifdef KV8
//  -DKV8 builds (GAUDIUM_KV=fp8|fp8e5): the caches, the fresh windows and the q blocks are FP8 (fmt 0 = fp8_143 E4M3,
//  1 = fp8_152 E5M2; scalar after the others), so the QK / PV GEMMs run as fp8_gemm_f32 (the Gaudi 2 MME takes no bf16 x fp8 operand pair,
//  research/probes/p24_fp8_kv.cpp). One more input after the others: kv8 f32 {16} = per kv group g: [g] q multiplier (1/16 and the q / k
//  scales folded in), [4+g] K multiplier, [8+g] V multiplier, [12+g] P multiplier (read by the softmax kernels). Conversions round half to
//  even and saturate (SW_CLIP_FP). A 256-channel row is ONE fp8 vector: byte 4l + j <- f32 vector j lane l (tests/sim/sim_kv8.cpp), the
//  vectors being the two deinterleaved halves {a.v1, a.v2, b.v1, b.v2} for q and k (they share that channel order: the dot product does
//  not see it); V's vectors are regrouped per 128-position half (below) and the loader's W_v row order matches (model_qwen35.hpp vrow256). V rows are permuted inside 256-row blocks to
//  match the fp8 P bytes of the softmax: VPERM8(p) = 4 (p % 64) + (p / 64) % 4.
#define VPERM8(j) (((j) & ~255) + 4 * ((j) & 63) + (((j) >> 6) & 3))
#define VPERM8_INV(q) (((q) & ~255) + 64 * ((q) & 3) + (((q) & 255) >> 2))
#define ST8(c_, t_, x0, x1, x2, x3) do { float256 _f; _f.v1 = (x0); _f.v2 = (x1); _f.v3 = (x2); _f.v4 = (x3); \
    if (fmt) v_h8_st_tnsr((c_), t_, v_convert_f32_to_h8_all_b(_f, SW_RHNE | SW_CLIP_FP)); \
    else     v_f8_st_tnsr((c_), t_, v_convert_f32_to_f8_all_b(_f, SW_RHNE | SW_CLIP_FP)); } while (0)
#define KV8_LD(i_) ({ int5 _kc = {(i_), 0, 0, 0, 0}; s_f32_ld_g(gen_addr(_kc, kv8)); })
#endif

#if defined(KV8) && defined(FRESH)
void main(tensor qg, tensor k, tensor v, tensor qn_w, tensor kn_w, tensor rope_c, tensor rope_s, tensor cursor, tensor kf_in, tensor vf_in, tensor kv8,
          tensor qb, tensor qb_f, tensor kf_out, tensor vf_out, int T, int coff, int nch, int C, int sbase, int NF, int fmt)
#elif defined(KV8) && defined(TREE)
void main(tensor qg, tensor k, tensor v, tensor qn_w, tensor kn_w, tensor rope_c, tensor rope_s, tensor cursor, tensor depth, tensor fix, tensor kv8,
          tensor qb, tensor kcache, tensor vcache, int T, int coff, int nch, int C, int fmt)
#elif defined(KV8)
void main(tensor qg, tensor k, tensor v, tensor qn_w, tensor kn_w, tensor rope_c, tensor rope_s, tensor cursor, tensor kv8,
          tensor qb, tensor kcache, tensor vcache, int T, int coff, int nch, int C, int fmt)
#elif defined(FRESH)
void main(tensor qg, tensor k, tensor v, tensor qn_w, tensor kn_w, tensor rope_c, tensor rope_s, tensor cursor, tensor kf_in, tensor vf_in,
          tensor qb, tensor qb_f, tensor kf_out, tensor vf_out, int T, int coff, int nch, int C, int sbase, int NF)
#elif defined(TREE)
void main(tensor qg, tensor k, tensor v, tensor qn_w, tensor kn_w, tensor rope_c, tensor rope_s, tensor cursor, tensor depth, tensor fix,
          tensor qb, tensor kcache, tensor vcache, int T, int coff, int nch, int C)
#else
void main(tensor qg, tensor k, tensor v, tensor qn_w, tensor kn_w, tensor rope_c, tensor rope_s, tensor cursor,
          tensor qb, tensor kcache, tensor vcache, int T, int coff, int nch, int C)
#endif
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 l16 = (lid8 >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const float eps = 1e-6f, qscale = 0.0625f, inv_n = 1.0f / 256.0f;
    int5 zc = {0, 0, 0, 0, 0};
    const int cur = s_i32_ld_g(gen_addr(zc, cursor)) + coff;
#ifdef FRESH
    // window copy: every row of the previous pass's window that this pass does not rewrite, spread over the 28 index points.
    //   K row p holds slot p; V row p holds slot VPERM^-1(p) (V rows are permuted). A row is copied iff the SLOT it holds is
    //   outside [sbase, sbase+T). (fix/fresh-vperm: the copy used to skip rows [sbase, sbase+T) for V as well, so the new V rows
    //   at VPERM(slot) were ALSO copied from the old window by another index point (two TPCs storing the same row: whichever
    //   landed last won, a timing race), and V rows [sbase, sbase+T) that hold OTHER slots were never copied (stale rows from the
    //   buffer's previous use). Each output row now has exactly one writer.)
    for (int i = idx[0]; i < idx[0] + isz[0]; ++i) {
        for (int p = i; p < NF; p += ATT_NPT) {
            const int pin = p & 127, vslot = (p & ~127) + ((pin & 1) ? 64 + (pin >> 1) : (pin >> 1));   // VPERM^-1(p)
            const int kcopy = !(p >= sbase && p < sbase + T), vcopy = !(vslot >= sbase && vslot < sbase + T);
#ifdef KV8
            const int vslot8 = VPERM8_INV(p), vcopy8 = !(vslot8 >= sbase && vslot8 < sbase + T);   // V row p holds slot VPERM8_INV(p)
            for (int g = 0; g < ATT_NKV; ++g) {   // one 256-byte row per K / V row
                int5 kc = {0, p, g, 0, 0};
                if (kcopy) v_i8_st_tnsr(kc, kf_out, v_i8_ld_tnsr_b(kc, kf_in));
                if (vcopy8) v_i8_st_tnsr(kc, vf_out, v_i8_ld_tnsr_b(kc, vf_in));
            }
            (void)vcopy;
#else
            for (int g = 0; g < ATT_NKV; ++g) {
                int5 kc = {0, p, g, 0, 0};
                if (kcopy) {
                    bfloat128 k1 = v_bf16_ld_tnsr_b(kc, kf_in); kc[0] = 128; bfloat128 k2 = v_bf16_ld_tnsr_b(kc, kf_in);
                    v_bf16_st_tnsr(kc, kf_out, k2); kc[0] = 0; v_bf16_st_tnsr(kc, kf_out, k1);
                }
                if (vcopy) {
                    bfloat128 v1 = v_bf16_ld_tnsr_b(kc, vf_in); kc[0] = 128; bfloat128 v2 = v_bf16_ld_tnsr_b(kc, vf_in);
                    v_bf16_st_tnsr(kc, vf_out, v2); kc[0] = 0; v_bf16_st_tnsr(kc, vf_out, v1);
                }
            }
#endif
        }
    }
#endif
#ifdef TREE
    {   // accepted-path moves of the previous step (kv-head points only)
        const int F = get_dim_size(fix, 0) / 2;
        for (int i = idx[0]; i < idx[0] + isz[0]; ++i) {
            if (i < ATT_NH) continue;
            const int h = i - ATT_NH;
            for (int f = 0; f < F; ++f) {
                int5 fc = {2 * f, 0, 0, 0, 0};
                const int src = s_i32_ld_g(gen_addr(fc, fix)); fc[0] = 2 * f + 1; const int dst = s_i32_ld_g(gen_addr(fc, fix));
                if (src < 0) continue;
                const int cs = src / C, ps = src - cs * C, cd = dst / C, pd = dst - cd * C;
                int5 sk = {0, ps, cs * ATT_NKV + h, 0, 0}, dk = {0, pd, cd * ATT_NKV + h, 0, 0};
#ifdef KV8
                int5 sv8 = {0, VPERM8(ps), cs * ATT_NKV + h, 0, 0}, dv8 = {0, VPERM8(pd), cd * ATT_NKV + h, 0, 0};
                char256 k8 = v_i8_ld_tnsr_b(sk, kcache), v8 = v_i8_ld_tnsr_b(sv8, vcache);
                v_i8_st_tnsr(dk, kcache, k8); v_i8_st_tnsr(dv8, vcache, v8);
                continue;
#endif
#if ATT_DH == 128
                { int5 sv = {0, VPERM(ps), cs * ATT_NKV + h, 0, 0}, dv = {0, VPERM(pd), cd * ATT_NKV + h, 0, 0};
                  bfloat128 k1 = v_bf16_ld_tnsr_b(sk, kcache), v1 = v_bf16_ld_tnsr_b(sv, vcache);
                  v_bf16_st_tnsr(dk, kcache, k1); v_bf16_st_tnsr(dv, vcache, v1); }
                continue;
#endif
                bfloat128 k1 = v_bf16_ld_tnsr_b(sk, kcache); sk[0] = 128; bfloat128 k2 = v_bf16_ld_tnsr_b(sk, kcache);
                int5 sv = {0, VPERM(ps), cs * ATT_NKV + h, 0, 0}, dv = {0, VPERM(pd), cd * ATT_NKV + h, 0, 0};
                bfloat128 v1 = v_bf16_ld_tnsr_b(sv, vcache); sv[0] = 128; bfloat128 v2 = v_bf16_ld_tnsr_b(sv, vcache);
                v_bf16_st_tnsr(dk, kcache, k1); dk[0] = 128; v_bf16_st_tnsr(dk, kcache, k2);
                v_bf16_st_tnsr(dv, vcache, v1); dv[0] = 128; v_bf16_st_tnsr(dv, vcache, v2);
            }
        }
    }
#endif

#if ATT_DH == 128
#if defined(FRESH) || defined(KV8)
#error "attn_prep: the 128-channel builds are plain / TREE only"
#endif
#define ROPE128(x, C, S) ({ float64 _x = (x); float64 _sw = MOVDG_ALL(_x, 2, 3, 0, 1, _x); _x * (C) + _sw * (S); })
    (void)qscale; (void)inv_n;
    const float qs128 = 0.08838834764831845f, inv128 = 1.0f / 128.0f;   // 1/sqrt(128); 1/DH
    for (int i = idx[0]; i < idx[0] + isz[0]; ++i) {
        const int is_q = (i < ATT_NH);
        const int h = is_q ? i : i - ATT_NH;
        int5 wc = {0, 0, 0, 0, 0};
        float64 w1, w2;
        if (is_q) { w1 = v_f32_ld_tnsr_b(wc, qn_w); wc[0] = 64; w2 = v_f32_ld_tnsr_b(wc, qn_w); }
        else      { w1 = v_f32_ld_tnsr_b(wc, kn_w); wc[0] = 64; w2 = v_f32_ld_tnsr_b(wc, kn_w); }
        for (int t = 0; t < T; ++t) {
#ifdef TREE
            int5 dpc = {t, 0, 0, 0, 0};
            const int pos = cur + s_i32_ld_g(gen_addr(dpc, depth));
            const int kslot = cur + t;
#else
            const int pos = cur + t, kslot = pos;
#endif
            int5 xc = {128 * h, t, 0, 0, 0};
            float128 a;
            if (is_q) a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(xc, qg));
            else      a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(xc, k));
            float64 ss = REDUCE64(a.v1 * a.v1 + a.v2 * a.v2, P8, P4, P2, P1);
            float64 r = v_rsqrt_f32(ss * inv128 + eps);
            a.v1 = a.v1 * r * w1; a.v2 = a.v2 * r * w2;
            int5 rc = {0, pos, 0, 0, 0};
            float64 C1 = v_f32_ld_tnsr_b(rc, rope_c), S1 = v_f32_ld_tnsr_b(rc, rope_s);
            rc[0] = 64;
            float64 C2 = v_f32_ld_tnsr_b(rc, rope_c), S2 = v_f32_ld_tnsr_b(rc, rope_s);
            a.v1 = ROPE128(a.v1, C1, S1); a.v2 = ROPE128(a.v2, C2, S2);
            if (is_q) {
                a.v1 = a.v1 * qs128; a.v2 = a.v2 * qs128;
                bfloat128 qa = v_convert_f32_to_bf16_all_b(a);
                for (int ch = 0; ch < nch; ++ch) { int5 oc = {0, (h % ATT_GQ) * T + t, ch * ATT_NKV + (h / ATT_GQ), 0, 0}; v_bf16_st_tnsr(oc, qb, qa); }
            } else {
                int5 vc = {128 * h, t, 0, 0, 0};
                bfloat128 v1 = v_bf16_ld_tnsr_b(vc, v);
                const int chk = kslot / C, pin = kslot - chk * C;
                int5 kc = {0, pin, chk * ATT_NKV + h, 0, 0}; v_bf16_st_tnsr(kc, kcache, v_convert_f32_to_bf16_all_b(a));
                int5 vo = {0, VPERM(pin), chk * ATT_NKV + h, 0, 0}; v_bf16_st_tnsr(vo, vcache, v1);
            }
        }
    }
#else
    for (int i = idx[0]; i < idx[0] + isz[0]; ++i) {
        const int is_q = (i < ATT_NH);
        const int h = is_q ? i : i - ATT_NH;
        int5 wc = {0, 0, 0, 0, 0};
        float64 wa1, wa2, wb1, wb2;
#ifdef KV8
        const int g8 = is_q ? h / ATT_GQ : h;
        const float qm8 = KV8_LD(g8), km8 = KV8_LD(4 + g8), vm8 = KV8_LD(8 + g8);
#endif
        if (is_q) { wa1 = v_f32_ld_tnsr_b(wc, qn_w); wc[0] = 64; wa2 = v_f32_ld_tnsr_b(wc, qn_w); wc[0] = 128; wb1 = v_f32_ld_tnsr_b(wc, qn_w); wc[0] = 192; wb2 = v_f32_ld_tnsr_b(wc, qn_w); }
        else      { wa1 = v_f32_ld_tnsr_b(wc, kn_w); wc[0] = 64; wa2 = v_f32_ld_tnsr_b(wc, kn_w); wc[0] = 128; wb1 = v_f32_ld_tnsr_b(wc, kn_w); wc[0] = 192; wb2 = v_f32_ld_tnsr_b(wc, kn_w); }
        for (int t = 0; t < T; ++t) {
#ifdef TREE
            int5 dpc = {t, 0, 0, 0, 0};
            const int pos = cur + s_i32_ld_g(gen_addr(dpc, depth));   // RoPE position; the KV slot stays cur + t
            const int kslot = cur + t;
#else
            const int pos = cur + t, kslot = pos;
#endif
            int5 xc = {256 * h, t, 0, 0, 0};
            float128 a, b;
            if (is_q) { a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(xc, qg)); xc[0] = 256 * h + 128; b = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(xc, qg)); }
            else      { a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(xc, k));  xc[0] = 256 * h + 128; b = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(xc, k)); }
            float64 ss = REDUCE64(a.v1 * a.v1 + a.v2 * a.v2 + b.v1 * b.v1 + b.v2 * b.v2, P8, P4, P2, P1);
            float64 r = v_rsqrt_f32(ss * inv_n + eps);
            a.v1 = a.v1 * r * wa1; a.v2 = a.v2 * r * wa2; b.v1 = b.v1 * r * wb1; b.v2 = b.v2 * r * wb2;
            int5 rc = {0, pos, 0, 0, 0};
            float64 C1 = v_f32_ld_tnsr_b(rc, rope_c), S1 = v_f32_ld_tnsr_b(rc, rope_s);
            rc[0] = 64;
            float64 C2 = v_f32_ld_tnsr_b(rc, rope_c), S2 = v_f32_ld_tnsr_b(rc, rope_s);
            a.v1 = ROPE(a.v1, C1, S1); a.v2 = ROPE(a.v2, C2, S2);
#ifdef KV8
            if (is_q) {
                a.v1 = a.v1 * qm8; a.v2 = a.v2 * qm8; b.v1 = b.v1 * qm8; b.v2 = b.v2 * qm8;
                for (int ch = 0; ch < nch; ++ch) { int5 oc = {0, (h % ATT_GQ) * T + t, ch * ATT_NKV + (h / ATT_GQ), 0, 0}; ST8(oc, qb, a.v1, a.v2, b.v1, b.v2); }
#ifdef FRESH
                int5 of = {0, (h % ATT_GQ) * T + t, h / ATT_GQ, 0, 0}; ST8(of, qb_f, a.v1, a.v2, b.v1, b.v2);
#endif
            } else {
                int5 vc = {256 * h, t, 0, 0, 0};
                float128 va = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(vc, v)); vc[0] = 256 * h + 128; float128 vb = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(vc, v));
                a.v1 = a.v1 * km8; a.v2 = a.v2 * km8; b.v1 = b.v1 * km8; b.v2 = b.v2 * km8;
                va.v1 = va.v1 * vm8; va.v2 = va.v2 * vm8; vb.v1 = vb.v1 * vm8; vb.v2 = vb.v2 * vm8;
                // V byte order stays inside each 128-position half (the block-exact FP8 checkpoint's W_v row blocks): w0 / w1 = lanes 0..31 of
                // the even / odd vectors of both halves, w2 / w3 = lanes 32..63 (dual groups 2, 3), so byte 4i + j of half h holds position
                // 128h + kv8_vpos(4i + j) = 128h + (j < 2 ? 2i + j : 64 + 2i + j - 2) (qw::vrow256 packs W_v to match)
                float64 w0 = v_f32_mov_dual_group_all_b(vb.v1, 0xFFFFFFFF, 0, 1, 0, 1, MkWrA(0b00, 0b00, 0b11, 0b11), va.v1);
                float64 w1 = v_f32_mov_dual_group_all_b(vb.v2, 0xFFFFFFFF, 0, 1, 0, 1, MkWrA(0b00, 0b00, 0b11, 0b11), va.v2);
                float64 w2 = v_f32_mov_dual_group_all_b(va.v1, 0xFFFFFFFF, 2, 3, 2, 3, MkWrA(0b11, 0b11, 0b00, 0b00), vb.v1);
                float64 w3 = v_f32_mov_dual_group_all_b(va.v2, 0xFFFFFFFF, 2, 3, 2, 3, MkWrA(0b11, 0b11, 0b00, 0b00), vb.v2);
#ifdef FRESH
                const int slot = sbase + t;
                int5 kc = {0, slot, h, 0, 0}; ST8(kc, kf_out, a.v1, a.v2, b.v1, b.v2);
                int5 vo = {0, VPERM8(slot), h, 0, 0}; ST8(vo, vf_out, w0, w1, w2, w3);
#else
                const int chk = kslot / C, pin = kslot - chk * C;
                int5 kc = {0, pin, chk * ATT_NKV + h, 0, 0}; ST8(kc, kcache, a.v1, a.v2, b.v1, b.v2);
                int5 vo = {0, VPERM8(pin), chk * ATT_NKV + h, 0, 0}; ST8(vo, vcache, w0, w1, w2, w3);
#endif
            }
            (void)qscale;
#else
            if (is_q) {
                a.v1 = a.v1 * qscale; a.v2 = a.v2 * qscale; b.v1 = b.v1 * qscale; b.v2 = b.v2 * qscale;
                bfloat128 qa = v_convert_f32_to_bf16_all_b(a), qb2 = v_convert_f32_to_bf16_all_b(b);
                for (int ch = 0; ch < nch; ++ch) {
                    int5 oc = {0, (h % ATT_GQ) * T + t, ch * ATT_NKV + (h / ATT_GQ), 0, 0};
                    v_bf16_st_tnsr(oc, qb, qa); oc[0] = 128; v_bf16_st_tnsr(oc, qb, qb2);
                }
#ifdef FRESH
                int5 of = {0, (h % ATT_GQ) * T + t, h / ATT_GQ, 0, 0};
                v_bf16_st_tnsr(of, qb_f, qa); of[0] = 128; v_bf16_st_tnsr(of, qb_f, qb2);
#endif
            } else {
                int5 vc = {256 * h, t, 0, 0, 0};
                bfloat128 v1 = v_bf16_ld_tnsr_b(vc, v); vc[0] = 256 * h + 128; bfloat128 v2 = v_bf16_ld_tnsr_b(vc, v);
#ifdef FRESH
                const int slot = sbase + t;
                int5 kc = {0, slot, h, 0, 0};
                v_bf16_st_tnsr(kc, kf_out, v_convert_f32_to_bf16_all_b(a)); kc[0] = 128; v_bf16_st_tnsr(kc, kf_out, v_convert_f32_to_bf16_all_b(b));
                int5 vo = {0, VPERM(slot), h, 0, 0};
                v_bf16_st_tnsr(vo, vf_out, v1); vo[0] = 128; v_bf16_st_tnsr(vo, vf_out, v2);
#else
                const int chk = kslot / C, pin = kslot - chk * C;
                int5 kc = {0, pin, chk * ATT_NKV + h, 0, 0};
                v_bf16_st_tnsr(kc, kcache, v_convert_f32_to_bf16_all_b(a)); kc[0] = 128; v_bf16_st_tnsr(kc, kcache, v_convert_f32_to_bf16_all_b(b));
                int5 vo = {0, VPERM(pin), chk * ATT_NKV + h, 0, 0};
                v_bf16_st_tnsr(vo, vcache, v1); vo[0] = 128; v_bf16_st_tnsr(vo, vcache, v2);
#endif
            }
#endif
        }
    }
#endif
}
