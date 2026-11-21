// gaudium_moe_mdq13 / gaudium_moe_mdq2: dequantise slots of a W4L / W8 expert pool to bf16 for the MME prefill path
// (docs/notes/flash-next-prefill.md). Builds: -DGU (gate+up pool), default (down pool), -DW8 (int8 codes, no level table).
//
//   in0 codes  u8 {256, in/2, NT, S} (W4L) / i8 {256, in, NT, S} (W8)   (the pool, docs/notes/moe-w4l.md 2.2)
//   in1 scales f16 {256, in/32, NT, S}
//   in2 lut    i8 {256}                                                  (W4L only)
//   in2/3 src  i32 {B}       pool slot of batch entry b, -1 = unused (nothing written)
//   out0 W     bf16 {256 NT, in, B}: N (output channel position) innermost, the GEMM's input (K) outer, one matrix per entry b
//
// value = bf16_rne(level(code) * f32(scale)) (the f32 product is exact: |level| <= 127, an fp16 scale has 11 significant bits).
// N order (both kinds): lane i = 4 l + j of pool tile tt goes to position 256 tt + 128 (j >> 1) + 2 l + (j & 1)
//   (one f32 -> bf16 "all" conversion of the accumulator pairs (0, 1) and (2, 3)). For the down pool that is the natural channel
//   order; for gate+up it is: positions 256 tt + [0, 128) = gate channels 128 tt + 64 h + l at 2 l + h, + 128 = the up channels alike.
// K order: gate+up: the natural input column. Down: column c (an h channel) goes to K position kpos(c) = the position at which
//   gaudium_moe_mswiglu writes h channel c (see there): c = 128 t + 64 h + l, m = 2 l + h, kpos = 128 t + 2 (m & 63) + (m >> 6).
// Index space {NT, B}: one point = one 256-channel tile of one entry (all its columns).
#include "moe_w4_common.h"

#ifndef W_UNROLL
#define W_UNROLL 4   // column pairs per loop iteration (independent registers hide the convert / multiply latency)
#endif
#define LDI(tn, c0) s_i32_ld_g(gen_addr(((int5){(c0), 0, 0, 0, 0}), tn))

#ifdef GU
static inline int kpos(int c) { return c; }
#else
static inline int kpos(int c) { const int r = c & 127, m = 2 * (r & 63) + (r >> 6); return (c & ~127) + 2 * (m & 63) + (m >> 6); }
#endif

#ifdef W8
void main(tensor codes, tensor scales, tensor src, tensor W)
#else
void main(tensor codes, tensor scales, tensor lut, tensor src, tensor W)
#endif
{
    const int5 off = get_index_space_offset(), sz = get_index_space_size();
    const int NBLK = get_dim_size(scales, 1);
#ifndef W8
    const char256 LUT = v_i8_ld_tnsr_b(((int5){0, 0, 0, 0, 0}), lut);
#endif
    const char256 ONES = 1;
    for (int b = off[1]; b < off[1] + sz[1]; ++b) {
        const int s = LDI(src, b);
        if (s < 0) continue;
        for (int tt = off[0]; tt < off[0] + sz[0]; ++tt) {
            for (int blk = 0; blk < NBLK; ++blk) {
                int5 sc = {0, blk, tt, s, 0};
                const ushort128 s0 = v_u16_ld_tnsr_b(sc, scales); sc[0] = 128;
                const ushort128 s1 = v_u16_ld_tnsr_b(sc, scales);
                const float128 w01 = v_convert_f16_to_f32_all_b(*(half128*)&s0), w23 = v_convert_f16_to_f32_all_b(*(half128*)&s1);
                #pragma unroll W_UNROLL
                for (int q = 0; q < 16; ++q) {
                    char256 lv[2];
#ifdef W8
                    int5 cc = {0, 32 * blk + 2 * q, tt, s, 0};
                    lv[0] = v_i8_ld_tnsr_b(cc, codes); cc[1] += 1;
                    lv[1] = v_i8_ld_tnsr_b(cc, codes);
#else
                    int5 cc = {0, 16 * blk + q, tt, s, 0};
                    const uchar256 u0 = v_u8_ld_tnsr_b(cc, codes, SW_UNPACK | SW_UNPCK_4_TO_8); cc[0] = 128;
                    const uchar256 u1 = v_u8_ld_tnsr_b(cc, codes, SW_UNPACK | SW_UNPCK_4_TO_8);
                    const uchar256 i0 = u0 | 0x80, i1 = u1 | 0x80;
                    lv[0] = v_i8_shuffle_b(LUT, i0, 0, *(const char256*)&i0); lv[1] = v_i8_shuffle_b(LUT, i1, 0, *(const char256*)&i1);
#endif
                    #pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = 32 * blk + 2 * q + h;
#ifdef W_CVT_LANE   // the lane-selected up-conversion (libtpcsim: byte lane 4 l + j -> vector j lane l)
                        const float64 f0 = v_convert_i8_to_f32_b(lv[h], SW_LANE_0), f1 = v_convert_i8_to_f32_b(lv[h], SW_LANE_1);
                        const float64 f2 = v_convert_i8_to_f32_b(lv[h], SW_LANE_2), f3 = v_convert_i8_to_f32_b(lv[h], SW_LANE_3);
#else               // levels * 1 through the int8 MAC (byte lane 4 l + j -> accumulator j lane l: measured on the card, moe-w4l.md 2.4), exact
                        int256 acc; acc.v1 = 0; acc.v2 = 0; acc.v3 = 0; acc.v4 = 0; acc = v_i8_mac_b(lv[h], ONES, acc);
                        const float64 f0 = v_convert_i32_to_f32_b(acc.v1, SW_RHNE), f1 = v_convert_i32_to_f32_b(acc.v2, SW_RHNE);
                        const float64 f2 = v_convert_i32_to_f32_b(acc.v3, SW_RHNE), f3 = v_convert_i32_to_f32_b(acc.v4, SW_RHNE);
#endif
                        float128 lo, hi; lo.v1 = f0 * w01.v1; lo.v2 = f1 * w01.v2; hi.v1 = f2 * w23.v1; hi.v2 = f3 * w23.v2;
                        int5 oc = {256 * tt, kpos(col), b, 0, 0};
                        v_bf16_st_tnsr(oc, W, v_convert_f32_to_bf16_all_b(lo, SW_RHNE)); oc[0] += 128;
                        v_bf16_st_tnsr(oc, W, v_convert_f32_to_bf16_all_b(hi, SW_RHNE));
                    }
                }
            }
        }
    }
}
