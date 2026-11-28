// gaudium_ik_w4_q8 / gaudium_ik_w8_q8: Inkling 2 routed expert input quantization to symmetric int8.
// Default build (block-16, NVFP4 layers): index space {H/128, T} -> xq i8 {256, H/32, T}, xs f32 {64, H/16, T}.
// -DBLK32 build (block-32, W8 layers):     index space {H/128, T} -> xq i8 {256, H/32, T}, xs f32 {64, H/32, T}.
#include "moe_w4_common.h"

void main(tensor x, tensor xq, tensor xs)
{
    const int5 off = get_index_space_offset(), sz = get_index_space_size();
    const uchar256 P8 = xor_pat16(8), P4 = xor_pat16(4), P2 = xor_pat16(2), P1 = xor_pat16(1);
    (void)P8;
    for (int t = off[1]; t < off[1] + sz[1]; ++t)
    for (int g2 = off[0]; g2 < off[0] + sz[0]; ++g2) {
        int5 c = {128 * g2, t, 0, 0, 0};
        const float128 f = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c, x));
        float64 e1 = f.v1, o1 = f.v2;
        e1 = v_f32_sel_less_f32_b(v_f32_abs_b(e1), 1.17549435e-38f, 0.0f, e1);
        o1 = v_f32_sel_less_f32_b(v_f32_abs_b(o1), 1.17549435e-38f, 0.0f, o1);
        float64 a = v_f32_max_b(v_f32_abs_b(e1), v_f32_abs_b(o1));
#ifdef BLK32
        a = DGMAX(a, P8, P4, P2, P1);
#else
        // Each 16-lane dual group holds 32 channels (lanes 0..7 = ch 0..15, lanes 8..15 = ch 16..31).
        // Skip P8 so lanes 0..7 hold block (2g) amax and lanes 8..15 hold block (2g+1) amax.
        a = v_f32_max_b(a, v_f32_shuffle_b(a, P4, 0, a));
        a = v_f32_max_b(a, v_f32_shuffle_b(a, P2, 0, a));
        a = v_f32_max_b(a, v_f32_shuffle_b(a, P1, 0, a));
#endif
        float64 qe, qo, d, d2;
        Q8(e1, a, qe, d); Q8(o1, a, qo, d2);
        (void)d2;
#ifdef BLK32
        int5 sc = {0, 4 * g2, t, 0, 0};
        v_f32_st_tnsr(sc, xs, MOVDG_F(d, 0, 0, 0, 0)); sc[1] += 1;
        v_f32_st_tnsr(sc, xs, MOVDG_F(d, 1, 1, 1, 1)); sc[1] += 1;
        v_f32_st_tnsr(sc, xs, MOVDG_F(d, 2, 2, 2, 2)); sc[1] += 1;
        v_f32_st_tnsr(sc, xs, MOVDG_F(d, 3, 3, 3, 3));
#else
        int5 sc = {0, 8 * g2, t, 0, 0};
        const float64 d0 = MOVDG_F(d, 0, 0, 0, 0);
        v_f32_st_tnsr(sc, xs, v_f32_shuffle_b(d0, (uchar256)FPAT(0), 0, d0)); sc[1] += 1;
        v_f32_st_tnsr(sc, xs, v_f32_shuffle_b(d0, (uchar256)FPAT(8), 0, d0)); sc[1] += 1;
        const float64 d1 = MOVDG_F(d, 1, 1, 1, 1);
        v_f32_st_tnsr(sc, xs, v_f32_shuffle_b(d1, (uchar256)FPAT(0), 0, d1)); sc[1] += 1;
        v_f32_st_tnsr(sc, xs, v_f32_shuffle_b(d1, (uchar256)FPAT(8), 0, d1)); sc[1] += 1;
        const float64 d2v = MOVDG_F(d, 2, 2, 2, 2);
        v_f32_st_tnsr(sc, xs, v_f32_shuffle_b(d2v, (uchar256)FPAT(0), 0, d2v)); sc[1] += 1;
        v_f32_st_tnsr(sc, xs, v_f32_shuffle_b(d2v, (uchar256)FPAT(8), 0, d2v)); sc[1] += 1;
        const float64 d3 = MOVDG_F(d, 3, 3, 3, 3);
        v_f32_st_tnsr(sc, xs, v_f32_shuffle_b(d3, (uchar256)FPAT(0), 0, d3)); sc[1] += 1;
        v_f32_st_tnsr(sc, xs, v_f32_shuffle_b(d3, (uchar256)FPAT(8), 0, d3));
#endif
        float256 cq; cq.v1 = qe; cq.v2 = qo; cq.v3 = qe; cq.v4 = qo;
        const char256 C = v_convert_f32_to_i8_all_b(cq, SW_RHNE);
        int5 qc = {0, 4 * g2, t, 0, 0};
        v_i8_st_tnsr(qc, xq, MOVDG_B(C, 0, 0, 0, 0)); qc[1] += 1;
        v_i8_st_tnsr(qc, xq, MOVDG_B(C, 1, 1, 1, 1)); qc[1] += 1;
        v_i8_st_tnsr(qc, xq, MOVDG_B(C, 2, 2, 2, 2)); qc[1] += 1;
        v_i8_st_tnsr(qc, xq, MOVDG_B(C, 3, 3, 3, 3));
    }
}
