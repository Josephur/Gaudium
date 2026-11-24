// gaudium_ik_moe_act: the soft-capped SwiGLU of the Inkling 2 experts in their dense-masked form. A layer's NE experts are stored as
// one wide MLP of NE blocks of F channels; the gate and up GEMMs compute every block for every row and this kernel applies the routing:
//   a[e F + i, t] = bf16( cap(g, cg) sigmoid(g) cap(u, cu) w(e, t) ),   cap(x, c) = c tanh(x / c);   exactly zero for an expert the
//   row did not select (the sigmoid sees the gate itself; the caps bound the linear gate factor and the up value)
// The down GEMM then accumulates the weighted sum over the row's experts. Rows cost the same as in a dense MLP (the matrix engine
// reads each weight once whatever the row count), so a verify pass of several rows costs little more than one row.
//  in0 g bf16 {NE F, T}   in1 u bf16 {NE F, T}   in2 wd2 f32 {NE / 2, 2, T} (gaudium_ik_route)   out0 a bf16 {NE F, T}
//  index space {NE, T}.   scalars: F (a multiple of 128), cg (f32 bits), cu (f32 bits)
#include "ik_common.h"

void main(tensor g, tensor u, tensor wd2, tensor a, int F, int cg_bits, int cu_bits)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const float cg = *(float*)&cg_bits, cu = *(float*)&cu_bits, icg = 1.0f / cg, icu = 1.0f / cu;
    for (int t = idx[1]; t < idx[1] + isz[1]; ++t) {
        for (int e = idx[0]; e < idx[0] + isz[0]; ++e) {
            int5 wc = {e >> 1, e & 1, t, 0, 0};
            const float w = s_f32_ld_g(gen_addr(wc, wd2));
            if (w == 0.f) {
                float128 z; z.v1 = 0.f; z.v2 = 0.f; const bfloat128 zb = v_convert_f32_to_bf16_all_b(z);
                for (int i = 0; i < F; i += 128) { int5 c = {e * F + i, t, 0, 0, 0}; v_bf16_st_tnsr(c, a, zb); }
            } else {
                const float64 wg = w * cg * cu;
                for (int i = 0; i < F; i += 128) {
                    int5 c = {e * F + i, t, 0, 0, 0};
                    const float128 gg = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c, g));
                    const float128 uu = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c, u));
                    float128 r;
                    r.v1 = TANH(gg.v1 * icg) * v_sigmoid_f32(gg.v1) * TANH(uu.v1 * icu) * wg;
                    r.v2 = TANH(gg.v2 * icg) * v_sigmoid_f32(gg.v2) * TANH(uu.v2 * icu) * wg;
                    v_bf16_st_tnsr(c, a, v_convert_f32_to_bf16_all_b(r));
                }
            }
        }
    }
}
