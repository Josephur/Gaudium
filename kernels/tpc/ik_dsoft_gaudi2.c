// gaudium_ik_dsoft: the softmax numerator of a block-drafter layer over two key sets: the layer's context state and the block's own
// keys. Every block row sees the same keys (the anchor's context, no causal mask inside the block):
//   x[key] = scores[key] scale;   P[key] = exp(x - m) at the keys the row sees, 0 elsewhere;   stats = (m, sum of the exp)
// gaudium_ik_dcomb merges the partials and normalises.
//  in0 sc f32 {C, R, B} context scores, B = G kv groups x nch chunks, chunk-major b = ch G + g;  R = GQ T rows per group
//  in1 sb f32 {128, R, G} block scores (keys 0 .. T - 1 are the block rows)
//  in2 ctl i32 {>= 3}: ctl[0] = p, the anchor's position; ctl[1 + which] = n, the context positions the rows see, counting back
//      from p - 1 (the host's verified window rule): positions lo = p - n .. p - 1, never the anchor's own or a later one
//  out0 Pc bf16 {C, R, B}   out1 Pb bf16 {128, R, G}: interleaved inside each block of 128 keys as gaudium_ik_softmax's P
//  out2 stats f32 {64, 2, R, B + G}: [lane][0] = m (64 equal lanes), [lane][1] = the sum of the exp; batches B .. B + G - 1: the block
//  ring == 1 (nch = 1): slot s holds position p - 1 - ((p - 1 - s) & (C - 1)); the row sees the nv = p - lo latest positions (the
//                       host checks that the ring holds them beside every row a pass writes ahead of the anchor)
//  ring == 0: chunk ch covers positions [ch C, ch C + C); the row sees its keys at positions lo .. p - 1
//  A set a row sees nothing of gets m = -1e30, s = 0, P = 0.
//  index space {R (B + G)}: point pp = b R + row.   scalars: T, which, ring, scale (f32 bits).   C <= 4096, a multiple of 128.
#include "ik_common.h"
#define NEG (-1.0e30f)
__local__ float64 XV[64];   // one row's masked logits

void main(tensor sc, tensor sb, tensor ctl, tensor Pc, tensor Pb, tensor stats, int T, int which, int ring, int scale_bits)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    IK_PATS
    const int64 lane = (int64)read_lane_id_4b_b();
    const float scale = *(float*)&scale_bits;
    int5 zc = {0, 0, 0, 0, 0};
    const int p = s_i32_ld_g(gen_addr(zc, ctl));
    zc[0] = 1 + which;
    int seen = s_i32_ld_g(gen_addr(zc, ctl));
    if (seen < 0) seen = 0;
    if (seen > p) seen = p;
    const int lo = p - seen;
    const int CC = get_dim_size(sc, 0), RR = get_dim_size(sc, 1), B = get_dim_size(sc, 2), G = get_dim_size(sb, 2);
    for (int pp = idx[0]; pp < idx[0] + isz[0]; ++pp) {
        const int bb = pp / RR, row = pp - bb * RR;
        const int blk = bb >= B;
        const int b = blk ? bb - B : bb;
        const int C = blk ? 128 : CC, M1 = C - 1;
        // keys the row sees: ring: the nv latest slots back from pk; otherwise the key indices [k0, nv) of the chunk
        int nv, pk = 0, k0 = 0;
        if (blk) nv = T;
        else if (ring) { nv = p - lo; if (nv > C) nv = C; pk = (p - 1) & M1; }
        else { const int ch = b / G; nv = p - ch * C; if (nv > C) nv = C; k0 = lo - ch * C; if (k0 < 0) k0 = 0; if (k0 >= nv) nv = 0; }
        int5 stc = {0, 0, row, bb, 0};
        if (nv <= 0) {
            float128 zz; zz.v1 = 0.f; zz.v2 = 0.f;
            const bfloat128 zb = v_convert_f32_to_bf16_all_b(zz);
            for (int j = 0; j < C; j += 128) { int5 oc = {j, row, b, 0, 0}; if (blk) v_bf16_st_tnsr(oc, Pb, zb); else v_bf16_st_tnsr(oc, Pc, zb); }
            float64 mneg = NEG, zero = 0.f;
            v_f32_st_tnsr(stc, stats, mneg); stc[1] = 1; v_f32_st_tnsr(stc, stats, zero);
            continue;
        }
        const int64 pv = (int64)pk;
        const int rmode = ring && !blk;
        float64 m0 = NEG;
        for (int j = 0; j < C; j += 64) {
            int5 c = {j, row, b, 0, 0};
            float64 x = (blk ? v_f32_ld_tnsr_b(c, sb) : v_f32_ld_tnsr_b(c, sc)) * scale;
            if (rmode) x = v_f32_sel_less_i32_b(v_i32_and_b(pv - (lane + j), M1, 0, 0, 1, 0), nv, x, NEG);
            else     { x = v_f32_sel_less_i32_b(lane + j, nv, x, NEG); x = v_f32_sel_less_i32_b(lane + j, k0, NEG, x); }
            XV[j >> 6] = x;
            m0 = v_f32_max_b(m0, x);
        }
        const float64 m = REDUCE_MAX64(m0, P8, P4, P2, P1);
        float64 s0 = 0.f;
        for (int j = 0; j < C; j += 128) {
            const int q = j >> 6;
            float64 e0 = EXPD(XV[q] - m), e1 = EXPD(XV[q + 1] - m);
            if (rmode) {
                e0 = v_f32_sel_less_i32_b(v_i32_and_b(pv - (lane + j), M1, 0, 0, 1, 0), nv, e0, 0.f);
                e1 = v_f32_sel_less_i32_b(v_i32_and_b(pv - (lane + (j + 64)), M1, 0, 0, 1, 0), nv, e1, 0.f);
            } else {
                e0 = v_f32_sel_less_i32_b(lane + j, nv, e0, 0.f); e0 = v_f32_sel_less_i32_b(lane + j, k0, 0.f, e0);
                e1 = v_f32_sel_less_i32_b(lane + (j + 64), nv, e1, 0.f); e1 = v_f32_sel_less_i32_b(lane + (j + 64), k0, 0.f, e1);
            }
            s0 = s0 + e0 + e1;
            float128 pa; pa.v1 = e0; pa.v2 = e1;
            int5 oc = {j, row, b, 0, 0};
            if (blk) v_bf16_st_tnsr(oc, Pb, v_convert_f32_to_bf16_all_b(pa));
            else     v_bf16_st_tnsr(oc, Pc, v_convert_f32_to_bf16_all_b(pa));
        }
        const float64 s = REDUCE64(s0, P8, P4, P2, P1);
        v_f32_st_tnsr(stc, stats, m); stc[1] = 1; v_f32_st_tnsr(stc, stats, s);
    }
}
