// gaudium_ik_kvprep: the key / value side of an Inkling 2 attention layer for T rows: the short causal convolution over the projected
// rows, then the key clip or the value norm, and the write into the layer's ring or cache.
//   c[t] = bf16( ( sum over lag < KW of (w[lag] + d[t, lag]) x[pos - lag]  +  x[pos] ) * isd )     rows before position 0 are zero
//          w: a tap per channel; d: a tap per head and row, projected from the row (global layers only; isd = 1 / sqrt(2 KW + 1)
//          there, 1 / sqrt(KW + 1) in the sliding layers)
//   K = bf16( c min(1, thr / (|c| + 1e-6)) )          V = bf16( c rsqrt(mean(c^2) + 1e-6) )         per head of 128 channels
// The projected rows come from rings by position (gaudium_ik_save).
//  in0 kraw bf16 {128 NKV, RC}   in1 vraw bf16 {128 NKV, RC}
//  in2 kx bf16 {KW_, T}, in3 vx bf16 {KW_, T}: channels [doff, doff + KW NKV) hold the row's taps d (head g, lag l at doff + KW g + l);
//      read only when dyn is set
//  in4 kcw f32 {128 NKV, KW} split layout   in5 vcw f32 {128 NKV, KW} split layout (column = lag)   in6 cursor i32 {1}
//  out0 kcache bf16 {128, C, NKV nch}   out1 vcache bf16 {128, C, NKV nch}
//      ring == 0: nch chunks of C positions, chunk-major batch b = ch NKV + g, position pos -> {pos % C, (pos / C) NKV + g}
//      ring == 1: one chunk of C slots (a power of two), position pos -> slot pos & (C - 1)
//      The value rows are stored in the interleaved row order VPERM inside each block of 128 rows (the softmax kernel's P pairs them so).
//  index space {2 NKV}: the key heads, then the value heads.
//  scalars: T, coff, C, NKV, ring, dyn, doff, isd (f32 bits), thr (f32 bits)
#include "ik_common.h"

void main(tensor kraw, tensor vraw, tensor kx, tensor vx, tensor kcw, tensor vcw, tensor cursor, tensor kcache, tensor vcache,
          int T, int coff, int C, int NKV, int ring, int dyn, int doff, int isd_bits, int thr_bits)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    IK_PATS
    const float isd = *(float*)&isd_bits, thr = *(float*)&thr_bits;
    const int M1 = get_dim_size(kraw, 1) - 1, KW = get_dim_size(kcw, 1);
    int5 zc = {0, 0, 0, 0, 0};
    const int cur = s_i32_ld_g(gen_addr(zc, cursor)) + coff;
    for (int i = idx[0]; i < idx[0] + isz[0]; ++i) {
        const int is_k = (i < NKV);
        const int g = is_k ? i : i - NKV;
        for (int t = 0; t < T; ++t) {
            const int pos = cur + t;
            float64 a1 = 0.f, a2 = 0.f, x1 = 0.f, x2 = 0.f;
            for (int lag = 0; lag < KW; ++lag) {
                const int pp = pos - lag;
                if (pp < 0) break;
                int5 rc = {128 * g, pp & M1, 0, 0, 0}, wc = {128 * g, lag, 0, 0, 0};
                float128 xv; float64 w1, w2;
                if (is_k) { xv = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(rc, kraw)); w1 = v_f32_ld_tnsr_b(wc, kcw); wc[0] += 64; w2 = v_f32_ld_tnsr_b(wc, kcw); }
                else      { xv = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(rc, vraw)); w1 = v_f32_ld_tnsr_b(wc, vcw); wc[0] += 64; w2 = v_f32_ld_tnsr_b(wc, vcw); }
                if (dyn) {
                    int5 dc = {doff + KW * g + lag, t, 0, 0, 0};
                    float d;
                    if (is_k) d = s_convert_bf16_to_f32(s_bf16_ld_g(gen_addr(dc, kx)));
                    else      d = s_convert_bf16_to_f32(s_bf16_ld_g(gen_addr(dc, vx)));
                    const float64 dv = d;
                    w1 = w1 + dv; w2 = w2 + dv;
                }
                a1 = a1 + xv.v1 * w1; a2 = a2 + xv.v2 * w2;
                if (lag == 0) { x1 = xv.v1; x2 = xv.v2; }
            }
            const float128 c = RBF((a1 + x1) * isd, (a2 + x2) * isd);
            const float64 n2 = REDUCE64(c.v1 * c.v1 + c.v2 * c.v2, P8, P4, P2, P1);
            int pin, batch;
            if (ring) { pin = pos & (C - 1); batch = g; }
            else { const int chk = pos / C; pin = pos - chk * C; batch = chk * NKV + g; }
            float128 o;
            if (is_k) {
                const float64 f = CLIPF(n2, thr);
                o.v1 = c.v1 * f; o.v2 = c.v2 * f;
                int5 oc = {0, pin, batch, 0, 0};
                v_bf16_st_tnsr(oc, kcache, v_convert_f32_to_bf16_all_b(o));
            } else {
                const float64 r = v_rsqrt_f32(n2 * (1.0f / 128.0f) + 1.0e-6f);
                o.v1 = c.v1 * r; o.v2 = c.v2 * r;
                int5 oc = {0, VPERM(pin), batch, 0, 0};
                v_bf16_st_tnsr(oc, vcache, v_convert_f32_to_bf16_all_b(o));
            }
        }
    }
}
