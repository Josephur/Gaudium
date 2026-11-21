// gaudium_moe_q8: the MoE block input quantised for the int8 expert MACs (docs/notes/moe-w4l.md 3.2). Index space {H/128, T}: one
// point = 128 columns of one row = 4 blocks of 32.
//   in0  x  bf16 {H, T}
//   out0 xq i8  {256, H/32, T}   block b of row t: the same 64 bytes in each of the four dual groups (one UNIFORM u8 shuffle broadcasts
//                                 any column to all 256 lanes); column 32b + c' at byte 4 (c' >> 1) + (c' & 1) of each dual group
//   out1 xs f32 {64, H/32, T}    block b of row t: its scale in all 64 lanes
// Per block of 32 columns (w4::ref::q8): denormal inputs -> 0; a = max(max |x|, 2^-100); q = clamp(RNE(x * (RECIP(a) * 127)), +-127);
// scale = a * f32(1/127).
#include "moe_w4_common.h"
void main(tensor x, tensor xq, tensor xs)
{
    const int5 off = get_index_space_offset(), sz = get_index_space_size();
    const uchar256 P8 = xor_pat16(8), P4 = xor_pat16(4), P2 = xor_pat16(2), P1 = xor_pat16(1);
    for (int t = off[1]; t < off[1] + sz[1]; ++t)
    for (int g2 = off[0]; g2 < off[0] + sz[0]; ++g2) {
        int5 c = {128 * g2, t, 0, 0, 0};
        const float128 f = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c, x));    // v1 = even columns, v2 = odd columns
        float64 e1 = f.v1, o1 = f.v2;
        e1 = v_f32_sel_less_f32_b(v_f32_abs_b(e1), 1.17549435e-38f, 0.0f, e1);    // denormal (and zero) inputs -> +0
        o1 = v_f32_sel_less_f32_b(v_f32_abs_b(o1), 1.17549435e-38f, 0.0f, o1);
        float64 a = v_f32_max_b(v_f32_abs_b(e1), v_f32_abs_b(o1));
        a = DGMAX(a, P8, P4, P2, P1);                                               // dual group d = block d of these 128 columns
        float64 qe, qo, d, d2;
        Q8(e1, a, qe, d); Q8(o1, a, qo, d2);
        (void)d2;
        int5 sc = {0, 4 * g2, t, 0, 0};
        v_f32_st_tnsr(sc, xs, MOVDG_F(d, 0, 0, 0, 0)); sc[1] += 1; v_f32_st_tnsr(sc, xs, MOVDG_F(d, 1, 1, 1, 1));
        sc[1] += 1; v_f32_st_tnsr(sc, xs, MOVDG_F(d, 2, 2, 2, 2)); sc[1] += 1; v_f32_st_tnsr(sc, xs, MOVDG_F(d, 3, 3, 3, 3));
        float256 cq; cq.v1 = qe; cq.v2 = qo; cq.v3 = qe; cq.v4 = qo;
        const char256 C = v_convert_f32_to_i8_all_b(cq, SW_RHNE);   // dual group d: block d, column c' at byte 4 (c' >> 1) + (c' & 1)
        int5 qc = {0, 4 * g2, t, 0, 0};
        v_i8_st_tnsr(qc, xq, MOVDG_B(C, 0, 0, 0, 0)); qc[1] += 1; v_i8_st_tnsr(qc, xq, MOVDG_B(C, 1, 1, 1, 1)); qc[1] += 1;
        v_i8_st_tnsr(qc, xq, MOVDG_B(C, 2, 2, 2, 2)); qc[1] += 1; v_i8_st_tnsr(qc, xq, MOVDG_B(C, 3, 3, 3, 3));
    }
}
