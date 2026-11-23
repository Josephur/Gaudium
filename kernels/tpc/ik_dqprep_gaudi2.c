// gaudium_ik_dqprep: the block drafter's query, key and value heads of the T block rows (row t at position cursor + t):
//   q, k = rope( bf16( x rsqrt(mean(x^2) + eps) w ), pos )      v = the projection (split channel order, as in gaudium_ik_dkv)
// The block attends its own keys without a causal mask, beside the layer's context state.
//  in0 q bf16 {128 NH, T}   in1 kv bf16 {256 NKV, T} (key heads, then value heads)   in2 qn_w bf16 {128}   in3 kn_w bf16 {128}
//  in4 rope_hi   in5 rope_lo (ik_mtp.h)   in6 cursor i32 {1}
//  out0 qb  bf16 {128, GQ T, NKV nch}: row hl T + t of batch ch NKV + g = query head GQ g + hl of row t, the same rows in every chunk
//  out1 qbb bf16 {128, GQ T, NKV}: the same rows once (for the block keys)
//  out2 kb  bf16 {128, 128, NKV}: key of row t at row t, zero beyond T
//  out3 vb  bf16 {128, 128, NKV}: value of row t at row VPERM(t), zero elsewhere
//  index space {NH + NKV}: the query heads, then the key / value heads.   scalars: T, GQ, eps (f32 bits)
#include "ik_common.h"
#include "ik_mtp.h"

void main(tensor q, tensor kv, tensor qn_w, tensor kn_w, tensor rope_hi, tensor rope_lo, tensor cursor,
          tensor qb, tensor qbb, tensor kb, tensor vb, int T, int GQ, int eps_bits)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    IK_PATS
    const int64 lane = (int64)read_lane_id_4b_b();
    const float eps = *(float*)&eps_bits;
    const int NKV = get_dim_size(kb, 2), NH = NKV * GQ, NCH = get_dim_size(qb, 2) / NKV;
    int5 zc = {0, 0, 0, 0, 0};
    const int cur = s_i32_ld_g(gen_addr(zc, cursor));
    float128 z; z.v1 = 0.f; z.v2 = 0.f;
    for (int i = idx[0]; i < idx[0] + isz[0]; ++i) {
        if (i < NH) {
            const int g = i / GQ, hl = i - g * GQ;
            for (int t = 0; t < T; ++t) {
                const float128 x = IK_HEADNORM(IK_LDH(q, 128 * i, t, 0), qn_w, eps);
                const float128 r = IK_ROPE(x, rope_hi, rope_lo, cur + t, lane);
                const bfloat128 rb = v_convert_f32_to_bf16_all_b(r);
                const int row = hl * T + t;
                int5 oc = {0, row, g, 0, 0};
                v_bf16_st_tnsr(oc, qbb, rb);
                for (int ch = 0; ch < NCH; ++ch) { oc[2] = ch * NKV + g; v_bf16_st_tnsr(oc, qb, rb); }
            }
        } else {
            const int g = i - NH;
            for (int r = 0; r < 128; ++r) {
                // key rows in order; value row r holds key VPERM^-1(r) = r even ? r / 2 : 64 + (r - 1) / 2
                const int kv_t = (r & 1) ? 64 + (r >> 1) : (r >> 1);
                if (r < T) {
                    const float128 x = IK_HEADNORM(IK_LDH(kv, 128 * g, r, 0), kn_w, eps);
                    const float128 rr = IK_ROPE(x, rope_hi, rope_lo, cur + r, lane);
                    IK_STH(kb, 0, r, g, rr)
                } else {
                    IK_STH(kb, 0, r, g, z)
                }
                int5 oc = {0, r, g, 0, 0};
                if (kv_t < T) { int5 vc = {128 * (NKV + g), kv_t, 0, 0, 0}; v_bf16_st_tnsr(oc, vb, v_bf16_ld_tnsr_b(vc, kv)); }
                else v_bf16_st_tnsr(oc, vb, v_convert_f32_to_bf16_all_b(z));
            }
        }
    }
}
