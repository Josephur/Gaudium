// gaudium_moe_combine: out bf16 {H, T} = bf16_rne(add + sum over k = 0 .. K-1 of p_k), p_k = (w[k, t] == 0) ? 0 : w[k, t] * y_k,
// y_k = dev[k, t] ? yd[:, k, t] : yh[:, k, t]; f32, separate multiply and add, k ascending (= ascending expert id: the router's ids are
// sorted), one rounding (docs/notes/moe-w4l.md 1.2). y is deinterleaved per 128 channels (gaudium_moe_dn), so a 128-channel chunk is one
// even and one odd vector and the f32 -> bf16 "all" conversion writes natural order. -DADD: a 5th input add bf16 {H, T} starts the sum.
// Index space {H/128, T}. Scalar: K (<= 16).
#include "moe_w4_common.h"
#ifdef ADD
void main(tensor yd, tensor yh, tensor w, tensor dev, tensor add, tensor out, int K)
#else
void main(tensor yd, tensor yh, tensor w, tensor dev, tensor out, int K)
#endif
{
    const int5 off = get_index_space_offset(), sz = get_index_space_size();
    for (int t = off[1]; t < off[1] + sz[1]; ++t) {
        int5 wc = {0, t, 0, 0, 0};
        const float64 wr = MOVDG_F(v_f32_ld_tnsr_b(wc, w), 0, 0, 0, 0);        // lanes 0..K-1 of dual group 0, copied to all four
        const int64 dr = MOVDG_I(v_i32_ld_tnsr_b(wc, dev), 0, 0, 0, 0);
        for (int c = off[0]; c < off[0] + sz[0]; ++c) {
#ifdef ADD
            int5 ac = {128 * c, t, 0, 0, 0};
            const float128 a0 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(ac, add));
            float64 ae = a0.v1, ao = a0.v2;
#else
            float64 ae = 0.0f, ao = 0.0f;
#endif
            for (int k = 0; k < K; ++k) {
                const uchar256 pk = (unsigned char)(FPAT(k));
                const float64 wk = v_f32_shuffle_b(wr, pk, 0, wr);
                const int64 dk = v_i32_shuffle_b(dr, pk, 0, dr);
                int5 yc = {128 * c, k, t, 0, 0};
                const float64 de = v_f32_ld_tnsr_b(yc, yd), he = v_f32_ld_tnsr_b(yc, yh); yc[0] += 64;
                const float64 dO = v_f32_ld_tnsr_b(yc, yd), hO = v_f32_ld_tnsr_b(yc, yh);
                const float64 ye = v_f32_sel_eq_i32_b(dk, 0, he, de), yo = v_f32_sel_eq_i32_b(dk, 0, hO, dO);
                float64 pe = wk * ye, po = wk * yo;
                pe = v_f32_sel_eq_f32_b(wk, 0.0f, 0.0f, pe); po = v_f32_sel_eq_f32_b(wk, 0.0f, 0.0f, po);
                ae = ae + pe; ao = ao + po;
            }
            float128 r; r.v1 = ae; r.v2 = ao;
            int5 oc = {128 * c, t, 0, 0, 0};
            v_bf16_st_tnsr(oc, out, v_convert_f32_to_bf16_all_b(r, SW_RHNE));
        }
    }
}
