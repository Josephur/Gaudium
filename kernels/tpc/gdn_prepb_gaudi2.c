#include "gdn_shape.h"   // GDN_HK, GDN_REP, GDN_HV, GDN_QKV: compile-time head counts (issue #7); the defaults are Qwen3.8-27B's
// gaudium_gdn_prepb: phase B of the prefill conv prep: l2norm of the conv+SiLU output of the 32 q/k blocks, the
// q scale, the GQA repeat (x3) into the chunked layout, and the zero pad rows of q / k. rsqrt table only (see gdn_prepa_gaudi2.c).
// Same arithmetic as gdn_prep2_gaudi2.c phase B: bit-identical.
//  in0 y f32 {128, T, 32}  out0 q f32 {128, C, 48, NCH}  out1 k f32 {128, C, 48, NCH}   scalars: T, C, RG.  Index space {32 blocks, ceil(T/RG)}.
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define REDUCE64(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = _a + _r; _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = _a + _r; \
    _a = _a + v_f32_shuffle_b(_a, P8, 0, _a); _a = _a + v_f32_shuffle_b(_a, P4, 0, _a); \
    _a = _a + v_f32_shuffle_b(_a, P2, 0, _a); _a = _a + v_f32_shuffle_b(_a, P1, 0, _a); _a; })
static inline uchar256 xor_pat(uchar256 l16, int s) { uchar256 src = l16 ^ (unsigned char)s; return ((src & 7) | ((src >> 3) << 5)) | 0x80; }

void main(tensor y, tensor q, tensor k, int T, int CH, int RG)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 l16 = (lid8 >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const float eps = 1e-6f, qscale = 0.08838834764831845f;   // 1/sqrt(128)
    const int Tp = get_dim_size(q, 1) * get_dim_size(q, 3);
    const int NRG = (T + RG - 1) / RG;
    for (int j = idx[0]; j < idx[0] + isz[0]; ++j) {
      const int kind = j < GDN_HK ? 0 : 1, head = kind == 0 ? j : j - GDN_HK;
      for (int rg = idx[1]; rg < idx[1] + isz[1]; ++rg) {
        const int t0 = rg * RG, t1 = (t0 + RG < T) ? t0 + RG : T;
#pragma unroll 4
        for (int t = t0; t < t1; ++t) {
            int5 yc = {0, t, j, 0, 0};
            float64 ya = v_f32_ld_tnsr_b(yc, y); yc[0] = 64; float64 yb = v_f32_ld_tnsr_b(yc, y);
            float64 ss = REDUCE64(ya * ya + yb * yb, P8, P4, P2, P1);
            float64 r = v_rsqrt_f32(ss + eps);
            if (kind == 0) r = r * qscale;
            ya = ya * r; yb = yb * r;
            int5 oc = {0, t % CH, GDN_REP * head, t / CH, 0};
            for (int rep = 0; rep < GDN_REP; ++rep) {
                oc[2] = GDN_REP * head + rep;
                if (kind == 0) { oc[0] = 0; v_f32_st_tnsr(oc, q, ya); oc[0] = 64; v_f32_st_tnsr(oc, q, yb); }
                else           { oc[0] = 0; v_f32_st_tnsr(oc, k, ya); oc[0] = 64; v_f32_st_tnsr(oc, k, yb); }
            }
        }
        if (rg != NRG - 1) continue;
        for (int t = T; t < Tp; ++t) {        // zero the pad rows of the last chunk
            const float64 z = 0.f; const int tr = t % CH, tc = t / CH;
            for (int rep = 0; rep < GDN_REP; ++rep) { int5 oc = {0, tr, GDN_REP * head + rep, tc, 0};
                if (kind == 0) { v_f32_st_tnsr(oc, q, z); oc[0] = 64; v_f32_st_tnsr(oc, q, z); } else { v_f32_st_tnsr(oc, k, z); oc[0] = 64; v_f32_st_tnsr(oc, k, z); } }
        }
      }
    }
}
