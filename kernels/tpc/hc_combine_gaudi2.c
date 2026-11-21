// gaudium_hc_combine: the hyper-connection write-back (qwen4_exp; HF Qwen4ExpTextDecoderLayer: hyper_input + injection).
//   in0 R   bf16 {S*HID, T}  the residual before the block
//   in1 blk bf16 {HID, T}    the block output (token mixer or MoE)
//   in2 inj bf16 {S, T}      block_inject_weight(xn) / S: the inject GEMM output (the 1 / S is folded into the packed weight; exact, pow2)
//   out0 R' bf16 {S*HID, T}
//   scalar: HID (multiple of 256), S (<= 4)
//   w_s = bf16(2 * bf16(sigmoid(inj[s, t])));  R'[s*HID + c, t] = bf16( R + bf16(blk[c, t] * w_s) )
//   index space {HID / 256, T}. The S weights of a row come from one vector load of inj (lanes 0..S-1 after the bf16 -> f32 split:
//   v1 = even channels, v2 = odd) and are broadcast to all 64 lanes by a dual-group move + an in-group shuffle (REDUCE64's idioms).
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
// f32 lane L (0..15) of dual group 0 to every lane: all four dual groups <- group 0, then every lane <- element L of its group
#define BCAST(v, L) ({ float64 _t = MOVDG_ALL((v), 0, 0, 0, 0, (v)); const uchar256 _p = (unsigned char)(((L) & 7) | (((L) >> 3) << 5) | 0x80); \
    v_f32_shuffle_b(_t, _p, 0, _t); })
#define RBF(a, b) ({ float128 _p; _p.v1 = (a); _p.v2 = (b); v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(_p)); })

void main(tensor R, tensor blk, tensor inj, tensor Ro, int HID, int S)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const float64 two = 2.f;
    for (int t = idx[1]; t < idx[1] + isz[1]; ++t) {
        int5 ic = {0, t, 0, 0, 0};
        float128 iv = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(ic, inj));   // v1 lane j = stream 2j, v2 lane j = stream 2j + 1
        float128 sg = RBF(v_sigmoid_f32(iv.v1), v_sigmoid_f32(iv.v2));
        const float64 we = sg.v1 * two, wo = sg.v2 * two;                       // exact (pow2)
        float64 w[4];
        w[0] = BCAST(we, 0); w[1] = BCAST(wo, 0); w[2] = BCAST(we, 1); w[3] = BCAST(wo, 1);
        for (int b = idx[0]; b < idx[0] + isz[0]; ++b) {
            int5 bc = {b * 256, t, 0, 0, 0};
            bfloat128 bb0 = v_bf16_ld_tnsr_b(bc, blk); bc[0] += 128; bfloat128 bb1 = v_bf16_ld_tnsr_b(bc, blk);
            float128 k0 = v_convert_bf16_to_f32_all_b(bb0), k1 = v_convert_bf16_to_f32_all_b(bb1);
            for (int s = 0; s < S; ++s) {
                int5 rc = {s * HID + b * 256, t, 0, 0, 0};
                bfloat128 rb0 = v_bf16_ld_tnsr_b(rc, R); rc[0] += 128; bfloat128 rb1 = v_bf16_ld_tnsr_b(rc, R);
                float128 r0 = v_convert_bf16_to_f32_all_b(rb0), r1 = v_convert_bf16_to_f32_all_b(rb1);
                const float64 ws = w[s];
                float128 j0 = RBF(k0.v1 * ws, k0.v2 * ws), j1 = RBF(k1.v1 * ws, k1.v2 * ws);
                float128 o0; o0.v1 = r0.v1 + j0.v1; o0.v2 = r0.v2 + j0.v2;
                float128 o1; o1.v1 = r1.v1 + j1.v1; o1.v2 = r1.v2 + j1.v2;
                int5 oc = {s * HID + b * 256, t, 0, 0, 0};
                v_bf16_st_tnsr(oc, Ro, v_convert_f32_to_bf16_all_b(o0)); oc[0] += 128;
                v_bf16_st_tnsr(oc, Ro, v_convert_f32_to_bf16_all_b(o1));
            }
        }
    }
}
