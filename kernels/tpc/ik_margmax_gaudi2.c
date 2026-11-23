// gaudium_ik_margmax: the greedy token of one row of the block drafter's head: the arg-max over the first NV ids of
//   z = bf16( base[:, col] + bias + bf )          (f32 sum, one rounding; the logit soft cap is monotonic and left out)
// the LOWEST id among equal maxima (the rule of gaudium_ik_argmax). A value that is not a number never wins.
//  in0 base bf16 {V, TR}: the block rows' logits over the unembedding (column col)   in1 bias bf16 {V, 1}: the fold of the head's
//  latent through the unembedding   in2 bf bf16 {V}: the fold of the latent map's bias      out0 idx int32 {1, 1}
//  index space {1}.   scalars: col, NV (0: every row)
#include "ik_common.h"
#define REDUCE_MIN64(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = v_f32_min_b(_a, _r); _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = v_f32_min_b(_a, _r); \
    _a = v_f32_min_b(_a, v_f32_shuffle_b(_a, P8, 0, _a)); _a = v_f32_min_b(_a, v_f32_shuffle_b(_a, P4, 0, _a)); \
    _a = v_f32_min_b(_a, v_f32_shuffle_b(_a, P2, 0, _a)); _a = v_f32_min_b(_a, v_f32_shuffle_b(_a, P1, 0, _a)); _a; })

void main(tensor base, tensor bias, tensor bf, tensor idx_out, int col, int NV)
{
    IK_PATS
    const int64 lane = (int64)read_lane_id_4b_b();
    const float64 lane2 = v_convert_i32_to_f32_b(lane, SW_RHNE) * 2.0f;
    const int V = (NV > 0 && NV < get_dim_size(base, 0)) ? NV : get_dim_size(base, 0), NB = (V + 127) / 128, rem = V - (NB - 1) * 128;
    const float64 NEGV = -3.4e38f, BIG = 1.0e9f;
    float64 b1 = NEGV, b2 = NEGV, j1 = 0.f, j2 = 0.f;
    for (int j = 0; j < NB; ++j) {
        int5 c = {j * 128, col, 0, 0, 0}, c0 = {j * 128, 0, 0, 0, 0};
        const float128 x = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c, base));
        const float128 y = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c0, bias));
        const float128 f = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c0, bf));
        float128 a = RBF(x.v1 + y.v1 + f.v1, x.v2 + y.v2 + f.v2);
        if (j == NB - 1 && rem < 128) { a.v1 = v_f32_sel_less_i32_b(lane, (rem + 1) / 2, a.v1, NEGV); a.v2 = v_f32_sel_less_i32_b(lane, rem / 2, a.v2, NEGV); }
        const float64 jf = (float)j;
        j1 = v_f32_sel_grt_f32_b(a.v1, b1, jf, j1); b1 = v_f32_sel_grt_f32_b(a.v1, b1, a.v1, b1);
        j2 = v_f32_sel_grt_f32_b(a.v2, b2, jf, j2); b2 = v_f32_sel_grt_f32_b(a.v2, b2, a.v2, b2);
    }
    const float64 m = REDUCE_MAX64(v_f32_max_b(b1, b2), P8, P4, P2, P1);
    const float64 i1 = v_f32_sel_eq_f32_b(b1, m, j1 * 128.0f + lane2, BIG);
    const float64 i2 = v_f32_sel_eq_f32_b(b2, m, j2 * 128.0f + lane2 + 1.0f, BIG);
    const float64 best = REDUCE_MIN64(v_f32_min_b(i1, i2), P8, P4, P2, P1);
    int5 oc = {0, 0, 0, 0, 0};
    v_i32_st_tnsr(oc, idx_out, v_convert_f32_to_i32_b(best, SW_RHNE, 0, 1, 0));
}
