#include "attn_shape.h"   // ATT_DH: 256 (default) or 128 (gaudium_attn128_combine: two f32 vectors per row)
// gaudium_attn_combine: merge the per-chunk partial outputs of the chunked attention core (gaudium_attn_softmax_chunk + PV GEMM).
//  in0 ao_part f32 {256, R, B}  unnormalised partial outputs, B = G*nch, CHUNK-MAJOR b = ch*G + g (G = 4 kv groups)
//  in1 stats   f32 {64, 2, R, B}  per (row, chunk) max m and sum s, 64 equal lanes each
//  -DFRESH build: in2 ao_f f32 {256, R, G}, in3 stats_f f32 {64, 2, R, G}: one more "chunk" per group, the fresh window of a
//      chained MTP pass (gaudium_attn_softmax_fresh + its PV GEMM), merged with the same weights.
//  out0 ao f32 {256, R, G}: ao[:, r, g] = sum_ch w_ch ao_part[:, r, ch*G+g] / sum_ch w_ch s_ch,   w_ch = exp(m_ch - max_ch m_ch)
//  Channel order as stored by the PV GEMM (the V cache's lane order: attn_post_f32 reads it directly). nch = 1 is the plain 1/s normalisation.
//  index space 2-D: dim0 = R, dim1 = G.   scalars: nch
//  -DUNROLL build (prefill recipes, node param 1 = 1): the chunk loops unrolled by 4 so the exp and loads of
//  independent chunks overlap; the accumulation order is unchanged (bit-identical output).
// Decode exp (research/exp_bug): the vendor v_exp_f32 is off by a factor of ~2 at exactly 42 inputs in [-104, 0]
// (one per odd k in -121..-1, x log2 e just below k; x = -0.693147182 gives 1.0; exhaustive card scan == libtpcsim). DEXP is the
// FASTEXP formula of the prefill build (EXPD below: max 1 ulp on [-87, 0], none of those errors, 13 VPU ops instead of 29);
// Every build (plain, -DFRESH, the -DUNROLL prefill build: still bit-identical to the plain one); -DVENDOR_EXP restores v_exp_f32 (A/B builds).
#define EXPD(d_) ({ float64 _d = v_f32_max_b((d_), -87.0f); \
    float64 _z = v_f32_nearbyint_b(_d * 1.44269504088896341f, SW_RHNE, 0, 1, 0); \
    float64 _r = v_f32_mac_b(_z, -0.693359375f, _d, 0, 1, 0); _r = v_f32_mac_b(_z, 2.12194440e-4f, _r, 0, 1, 0); \
    float64 _p = v_f32_mac_b(_r, 0.0013836704893037677f, (float64)0.008374780416488647f, 0, 1, 0); \
    _p = v_f32_mac_b(_p, _r, (float64)0.04166822507977486f, 0, 1, 0); _p = v_f32_mac_b(_p, _r, (float64)0.1666642129421234f, 0, 1, 0); \
    _p = v_f32_mac_b(_p, _r, (float64)0.49999991059303284f, 0, 1, 0); _p = v_f32_mac_b(_p, _r, (float64)1.0f, 0, 1, 0); \
    _p = v_f32_mac_b(_p, _r, (float64)1.0f, 0, 1, 0); \
    int64 _n = v_i32_shl_b(v_convert_f32_to_i32_b(_z, SW_RHNE, 0, 1, 0), 23, 0, 0, 1, 0); \
    int64 _b = v_i32_add_b(*(int64*)&_p, _n, 0, 0, 1, 0); *(float64*)&_b; })
#ifdef VENDOR_EXP
#define DEXP(x) v_exp_f32(x)
#else
#define DEXP(x) EXPD(x)
#endif
#ifdef FRESH
void main(tensor ao_part, tensor stats, tensor ao_f, tensor stats_f, tensor ao, int nch)
#else
void main(tensor ao_part, tensor stats, tensor ao, int nch)
#endif
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const int G = get_dim_size(ao, 2);
    const float NEG = -1e30f;
    for (int g = idx[1]; g < idx[1] + isz[1]; ++g) {
        for (int row = idx[0]; row < idx[0] + isz[0]; ++row) {
            float64 M = NEG;
#ifdef UNROLL
#pragma unroll 4
#endif
            for (int ch = 0; ch < nch; ++ch) { int5 sc = {0, 0, row, ch * G + g, 0}; M = v_f32_max_b(M, v_f32_ld_tnsr_b(sc, stats)); }
#ifdef FRESH
            { int5 sc = {0, 0, row, g, 0}; M = v_f32_max_b(M, v_f32_ld_tnsr_b(sc, stats_f)); }
#endif
            float64 S = 0.f, o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
#ifdef UNROLL
#pragma unroll 4
#endif
            for (int ch = 0; ch < nch; ++ch) {
                const int b = ch * G + g;
                int5 sc = {0, 0, row, b, 0};
                float64 m = v_f32_ld_tnsr_b(sc, stats); sc[1] = 1; float64 s = v_f32_ld_tnsr_b(sc, stats);
                float64 w = DEXP(m - M);
                S = S + w * s;
                int5 ac = {0, row, b, 0, 0};
                o0 = o0 + w * v_f32_ld_tnsr_b(ac, ao_part); ac[0] = 64;
                o1 = o1 + w * v_f32_ld_tnsr_b(ac, ao_part);
#if ATT_DH == 256
                ac[0] = 128; o2 = o2 + w * v_f32_ld_tnsr_b(ac, ao_part); ac[0] = 192;
                o3 = o3 + w * v_f32_ld_tnsr_b(ac, ao_part);
#endif
            }
#ifdef FRESH
            {
                int5 sc = {0, 0, row, g, 0};
                float64 m = v_f32_ld_tnsr_b(sc, stats_f); sc[1] = 1; float64 s = v_f32_ld_tnsr_b(sc, stats_f);
                float64 w = DEXP(m - M);
                S = S + w * s;
                int5 ac = {0, row, g, 0, 0};
                o0 = o0 + w * v_f32_ld_tnsr_b(ac, ao_f); ac[0] = 64;
                o1 = o1 + w * v_f32_ld_tnsr_b(ac, ao_f);
#if ATT_DH == 256
                ac[0] = 128; o2 = o2 + w * v_f32_ld_tnsr_b(ac, ao_f); ac[0] = 192;
                o3 = o3 + w * v_f32_ld_tnsr_b(ac, ao_f);
#endif
            }
#endif
            float64 inv = v_reciprocal_f32(S);
            int5 oc = {0, row, g, 0, 0};
            v_f32_st_tnsr(oc, ao, o0 * inv); oc[0] = 64;
            v_f32_st_tnsr(oc, ao, o1 * inv);
#if ATT_DH == 256
            oc[0] = 128; v_f32_st_tnsr(oc, ao, o2 * inv); oc[0] = 192;
            v_f32_st_tnsr(oc, ao, o3 * inv);
#else
            (void)o2; (void)o3;
#endif
        }
    }
}
