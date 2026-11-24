// gaudium_ik_softmax: the masked softmax numerator of an Inkling 2 attention layer over its K / V ring or its chunked cache, with the
// layer's relative-position bias:
//   x[key] = scores[key] * scale + bias[head, pos - keypos]        the bias is zero beyond the bias extent (1024 positions back)
//   P[key] = exp(x - m) at the keys the row sees, 0 elsewhere;  stats = (m, sum of the exp)
// The attention sink (one more logit per row and head with a zero value) and the normalisation are applied by gaudium_ik_combine.
//  in0 scores f32 {C, R, B}   B = G kv groups x nch chunks, chunk-major b = ch G + g;  R = GQ T rows per group, row = hl T + t
//  in1 bias   f32 {8 BW, (NH / 8) T}: row grp T + t, channels [BW h', BW (h' + 1)) = head 8 grp + h' of row t, REVERSED and padded:
//             channel BW h' + x holds the bias at distance d = BOFF - x (zero where d is outside [0, extent)); BW = BOFF + 65.
//             The 64 keys from key j0 on are at distances dd, dd - 1, .., so their biases are the 64 channels from BOFF - dd on:
//             one plain vector load, no lane reversal. (The loader builds the reversed, padded projection.)
//  in2 cursor i32 {1}
//  out0 P bf16 {C, R, B}: interleaved inside each block of 128 keys (lane m <- key m even ? m / 2 : 64 + (m - 1) / 2), matched by the
//       value rows' order (VPERM)
//  out1 stats f32 {64, 2, R, B}: [lane][0] = m (the row's max over the chunk, 64 equal lanes), [lane][1] = the sum of the exp
//  ring == 1: one chunk of C = RING slots, slot = position & (RING - 1); a row at position p sees the W positions (p - W, p].
//             RING >= W + T - 1 and RING >= W + 64 keep the rows ahead of a row and the wrapped lanes outside its window.
//  ring == 0: chunk ch covers positions [ch C, ch C + C); a row at p sees every position up to p.
//  A chunk a row sees nothing of gets m = -1e30, s = 0, P = 0.
//  index space {R B}: point pp = b R + row.   scalars: T, coff, W, ring, nch, GQ, BOFF, scale (f32 bits).   C <= 4096, a multiple of 128.
#include "ik_common.h"
#define NEG (-1.0e30f)
__local__ float64 XV[64];   // one row's masked logits

void main(tensor scores, tensor bias, tensor cursor, tensor P, tensor stats,
          int T, int coff, int W, int ring, int nch, int GQ, int BOFF, int scale_bits)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    IK_PATS
    const int64 lane = (int64)read_lane_id_4b_b();
    const float scale = *(float*)&scale_bits;
    int5 zc = {0, 0, 0, 0, 0};
    const int cur = s_i32_ld_g(gen_addr(zc, cursor)) + coff;
    const int C = get_dim_size(scores, 0), RR = get_dim_size(scores, 1), B = get_dim_size(scores, 2), G = B / nch, M1 = C - 1;
    const int BW = BOFF + 65;
    for (int pp = idx[0]; pp < idx[0] + isz[0]; ++pp) {
        const int b = pp / RR, row = pp - b * RR;
        const int ch = b / G, g = b - ch * G;
        const int hl = row / T, t = row - hl * T;
        const int h = GQ * g + hl, brow = (h >> 3) * T + t, bbase = BW * (h & 7);
        const int p = cur + t;
        // nv: the number of keys the row sees in this chunk; pk: the key index (ring: slot) of the row's own position
        int nv, pk;
        if (ring) { nv = (p + 1 < W) ? p + 1 : W; pk = p & M1; }
        else { pk = p - ch * C; nv = pk + 1; if (nv > C) nv = C; }
        int5 sc = {0, 0, row, b, 0};
        if (nv <= 0) {
            float128 zz; zz.v1 = 0.f; zz.v2 = 0.f;
            const bfloat128 zb = v_convert_f32_to_bf16_all_b(zz);
            for (int j = 0; j < C; j += 128) { int5 oc = {j, row, b, 0, 0}; v_bf16_st_tnsr(oc, P, zb); }
            float64 mneg = NEG, zero = 0.f;
            v_f32_st_tnsr(sc, stats, mneg); sc[1] = 1; v_f32_st_tnsr(sc, stats, zero);
            continue;
        }
        const int64 pv = (int64)pk;
        float64 m0 = NEG;
        for (int j = 0; j < C; j += 64) {
            int5 c = {j, row, b, 0, 0};
            float64 x = v_f32_ld_tnsr_b(c, scores) * scale;
            // dd: the distance of the block's first key from the row (ring: modulo the ring)
            const int dd = ring ? ((pk - j) & M1) : (pk - j);
            if (dd >= 0 && dd <= BOFF) {
                int5 bc = {bbase + BOFF - dd, brow, 0, 0, 0};
                x = x + v_f32_ld_tnsr_b(bc, bias);
            }
            if (ring) x = v_f32_sel_less_i32_b(v_i32_and_b(pv - (lane + j), M1, 0, 0, 1, 0), nv, x, NEG);
            else      x = v_f32_sel_less_i32_b(lane + j, nv, x, NEG);
            XV[j >> 6] = x;
            m0 = v_f32_max_b(m0, x);
        }
        const float64 m = REDUCE_MAX64(m0, P8, P4, P2, P1);
        float64 s0 = 0.f;
        for (int j = 0; j < C; j += 128) {
            const int q = j >> 6;
            float64 e0 = EXPD(XV[q] - m), e1 = EXPD(XV[q + 1] - m);
            if (ring) {
                e0 = v_f32_sel_less_i32_b(v_i32_and_b(pv - (lane + j), M1, 0, 0, 1, 0), nv, e0, 0.f);
                e1 = v_f32_sel_less_i32_b(v_i32_and_b(pv - (lane + (j + 64)), M1, 0, 0, 1, 0), nv, e1, 0.f);
            } else {
                e0 = v_f32_sel_less_i32_b(lane + j, nv, e0, 0.f);
                e1 = v_f32_sel_less_i32_b(lane + (j + 64), nv, e1, 0.f);
            }
            s0 = s0 + e0 + e1;
            float128 pa; pa.v1 = e0; pa.v2 = e1;
            int5 oc = {j, row, b, 0, 0};
            v_bf16_st_tnsr(oc, P, v_convert_f32_to_bf16_all_b(pa));
        }
        const float64 s = REDUCE64(s0, P8, P4, P2, P1);
        v_f32_st_tnsr(sc, stats, m); sc[1] = 1; v_f32_st_tnsr(sc, stats, s);
    }
}
