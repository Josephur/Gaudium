// The row movers of the MME prefill path (docs/notes/flash-next-prefill.md). Builds: -DGATHER (gaudium_moe_mgather), default
// (gaudium_moe_mcomb). pos i32 {K, T}: pos[k, t] = the row (b C + c) of the expert batch that holds assignment (k, t) (planned on the
// host); pos[0, t] < 0 marks a row with nothing to do (beyond the valid rows of the super-tile).
//
// GATHER:  in0 x bf16 {H, T}, in1 pos -> out0 xg bf16 {H, R}: xg[:, pos[k, t]] = x[:, t] for every k
// COMBINE: in0 yb bf16 {H, R}, in1 pos, in2 w f32 {K, T}, in3 add bf16 {H, T} -> out0 out bf16 {H, T}:
//          out[:, t] = bf16_rne(add + sum over k ascending of p_k), p_k = (w[k, t] == 0) ? 0 : w[k, t] * f32(yb[:, pos[k, t]]),
//          separate multiply and add (gaudium_moe_combine's order; the router's ids are ascending per row)
// Index space {T}: one point = one row (K scalar loads of pos, then vector work over H). Scalar: K (<= 16).
#include "moe_w4_common.h"
#define LDI2(tn, c0, c1) s_i32_ld_g(gen_addr(((int5){(c0), (c1), 0, 0, 0}), tn))
__local__ int PK[16];
#ifdef GATHER
void main(tensor x, tensor pos, tensor xg, int K)
{
    const int5 off = get_index_space_offset(), sz = get_index_space_size();
    const int NC = get_dim_size(x, 0) / 128;
    for (int t = off[0]; t < off[0] + sz[0]; ++t) {
        for (int k = 0; k < K; ++k) PK[k] = LDI2(pos, k, t);
        if (PK[0] < 0) continue;
        for (int c = 0; c < NC; ++c) {
            const bfloat128 v = v_bf16_ld_tnsr_b(((int5){128 * c, t, 0, 0, 0}), x);
            for (int k = 0; k < K; ++k) v_bf16_st_tnsr(((int5){128 * c, PK[k], 0, 0, 0}), xg, v);
        }
    }
}
#else
void main(tensor yb, tensor pos, tensor w, tensor add, tensor out, int K)
{
    const int5 off = get_index_space_offset(), sz = get_index_space_size();
    const int NC = get_dim_size(out, 0) / 128;
    for (int t = off[0]; t < off[0] + sz[0]; ++t) {
        for (int k = 0; k < K; ++k) PK[k] = LDI2(pos, k, t);
        if (PK[0] < 0) continue;
        const float64 wr = MOVDG_F(v_f32_ld_tnsr_b(((int5){0, t, 0, 0, 0}), w), 0, 0, 0, 0);   // lanes 0..K-1 of dual group 0, in all four
        for (int c = 0; c < NC; ++c) {
            const float128 a0 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(((int5){128 * c, t, 0, 0, 0}), add));
            float64 ae = a0.v1, ao = a0.v2;
            for (int k = 0; k < K; ++k) {
                const uchar256 pk = (unsigned char)(FPAT(k));
                const float64 wk = v_f32_shuffle_b(wr, pk, 0, wr);
                const float128 y = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(((int5){128 * c, PK[k], 0, 0, 0}), yb));
                float64 pe = wk * y.v1, po = wk * y.v2;
                pe = v_f32_sel_eq_f32_b(wk, 0.0f, 0.0f, pe); po = v_f32_sel_eq_f32_b(wk, 0.0f, 0.0f, po);
                ae = ae + pe; ao = ao + po;
            }
            float128 r; r.v1 = ae; r.v2 = ao;
            v_bf16_st_tnsr(((int5){128 * c, t, 0, 0, 0}), out, v_convert_f32_to_bf16_all_b(r, SW_RHNE));
        }
    }
}
#endif
