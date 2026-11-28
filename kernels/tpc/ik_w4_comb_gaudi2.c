// gaudium_ik_w4_comb: weighted sum of the K routed expert outputs into BF16 {H, T} in ascending k (= expert id) order.
//   in0 y   f32  {H, K, T}   deinterleaved per 128 channels (gaudium_ik_w4_dn / _w8_dn)
//   in1 w   f32  {K, T}      router weights (gaudium_ik_route_r)
//   out0 out bf16 {H, T}
//   index space {H / 128, T}.   scalar: K (<= 12)
#include "moe_w4_common.h"

void main(tensor y, tensor w, tensor out, int K)
{
    const int5 off = get_index_space_offset(), sz = get_index_space_size();
    for (int t = off[1]; t < off[1] + sz[1]; ++t) {
        int5 wc = {0, t, 0, 0, 0};
        const float64 wr = MOVDG_F(v_f32_ld_tnsr_b(wc, w), 0, 0, 0, 0);
        for (int c = off[0]; c < off[0] + sz[0]; ++c) {
            float64 ae = 0.0f, ao = 0.0f;
            for (int k = 0; k < K; ++k) {
                const uchar256 pk = (unsigned char)(FPAT(k));
                const float64 wk = v_f32_shuffle_b(wr, pk, 0, wr);
                int5 yc = {128 * c, k, t, 0, 0};
                const float64 ye = v_f32_ld_tnsr_b(yc, y); yc[0] += 64;
                const float64 yo = v_f32_ld_tnsr_b(yc, y);
                float64 pe = wk * ye, po = wk * yo;
                pe = v_f32_sel_eq_f32_b(wk, 0.0f, 0.0f, pe);
                po = v_f32_sel_eq_f32_b(wk, 0.0f, 0.0f, po);
                ae = ae + pe; ao = ao + po;
            }
            float128 r; r.v1 = ae; r.v2 = ao;
            int5 oc = {128 * c, t, 0, 0, 0};
            v_bf16_st_tnsr(oc, out, v_convert_f32_to_bf16_all_b(r, SW_RHNE));
        }
    }
}
