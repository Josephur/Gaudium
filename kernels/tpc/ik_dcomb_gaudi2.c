// gaudium_ik_dcomb: the back end of a block-drafter layer's attention: merge the context chunks' partials and the block's, normalise.
//   M = max m;   S = sum exp(m - M) s;   core = bf16( sum exp(m - M) ao / S )
//  in0 aoc f32 {128, R, B} (B = G nch, chunk-major)   in1 aob f32 {128, R, G}   in2 stats f32 {64, 2, R, B + G} (gaudium_ik_dsoft)
//      the partial outputs are in the split layout (the loader orders the value projection's channels so)
//  out0 core bf16 {128 NH, T}, the checkpoint's channel order.   index space {NH}.   scalars: T, GQ
#include "ik_common.h"

void main(tensor aoc, tensor aob, tensor stats, tensor core, int T, int GQ)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const int B = get_dim_size(aoc, 2), G = get_dim_size(aob, 2), nch = B / G;
    for (int h = idx[0]; h < idx[0] + isz[0]; ++h) {
        const int g = h / GQ, hl = h - g * GQ;
        for (int t = 0; t < T; ++t) {
            const int row = hl * T + t;
            int5 sb = {0, 0, row, B + g, 0};
            float64 M = v_f32_ld_tnsr_b(sb, stats);
            for (int ch = 0; ch < nch; ++ch) { int5 sc = {0, 0, row, ch * G + g, 0}; M = v_f32_max_b(M, v_f32_ld_tnsr_b(sc, stats)); }
            float64 S = 0.f, o0 = 0.f, o1 = 0.f;
            for (int ch = 0; ch <= nch; ++ch) {
                const int blk = ch == nch;
                const int b = blk ? B + g : ch * G + g;
                int5 sc = {0, 0, row, b, 0};
                const float64 m = v_f32_ld_tnsr_b(sc, stats); sc[1] = 1;
                const float64 s = v_f32_ld_tnsr_b(sc, stats);
                const float64 w = EXPD(m - M);
                S = S + w * s;
                int5 ac = {0, row, blk ? g : b, 0, 0};
                if (blk) { o0 = o0 + w * v_f32_ld_tnsr_b(ac, aob); ac[0] = 64; o1 = o1 + w * v_f32_ld_tnsr_b(ac, aob); }
                else     { o0 = o0 + w * v_f32_ld_tnsr_b(ac, aoc); ac[0] = 64; o1 = o1 + w * v_f32_ld_tnsr_b(ac, aoc); }
            }
            const float64 inv = RECIP(S);
            float128 o; o.v1 = o0 * inv; o.v2 = o1 * inv;
            int5 oc = {128 * h, t, 0, 0, 0};
            v_bf16_st_tnsr(oc, core, v_convert_f32_to_bf16_all_b(o));
        }
    }
}
