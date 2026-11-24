// gaudium_ik_dkv: the block drafter's context keys and values of T committed rows, into a layer's state by position:
//   K = rope( bf16( k rsqrt(mean(k^2) + eps) w ), pos )        V = v (the projection; the loader orders its channels so that the
//   value product lands in the split layout)                    per head of 128 channels; row t is at position pos = cursor + t
//  in0 kv bf16 {256 NKV, T}: the key heads, then the value heads   in1 w bf16 {128} (the context key norm)
//  in2 rope_hi f32 {128, 2, NHI}   in3 rope_lo f32 {128, 2, 256} (ik_mtp.h)   in4 cursor i32 {1}
//  out0 kst bf16 {128, C, B}   out1 vst bf16 {128, C, B}
//     ring == 1: B = NKV, slot = pos & (C - 1)          ring == 0: chunk-major cache, pos -> {pos % C, (pos / C) NKV + g}
//     a value row goes to VPERM(slot) (the softmax's P pairs the keys so); a row whose chunk is beyond B is not written
//  index space {NKV}: one member writes every row of one head.   scalars: T, ring, eps (f32 bits)
#include "ik_common.h"
#include "ik_mtp.h"

void main(tensor kv, tensor w, tensor rope_hi, tensor rope_lo, tensor cursor, tensor kst, tensor vst, int T, int ring, int eps_bits)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    IK_PATS
    const int64 lane = (int64)read_lane_id_4b_b();
    const float eps = *(float*)&eps_bits;
    const int C = get_dim_size(kst, 1), B = get_dim_size(kst, 2), NKV = get_dim_size(kv, 0) / 256;
    int5 zc = {0, 0, 0, 0, 0};
    const int cur = s_i32_ld_g(gen_addr(zc, cursor));
    for (int g = idx[0]; g < idx[0] + isz[0]; ++g) {
        for (int t = 0; t < T; ++t) {
            const int pos = cur + t;
            int slot, b;
            if (ring) { slot = pos & (C - 1); b = g; }
            else { const int ch = pos / C; slot = pos - ch * C; b = ch * NKV + g; }
            if (b >= B) continue;
            const float128 k = IK_HEADNORM(IK_LDH(kv, 128 * g, t, 0), w, eps);
            const float128 kr = IK_ROPE(k, rope_hi, rope_lo, pos, lane);
            IK_STH(kst, 0, slot, b, kr)
            int5 vc = {128 * (NKV + g), t, 0, 0, 0}, oc = {0, VPERM(slot), b, 0, 0};
            v_bf16_st_tnsr(oc, vst, v_bf16_ld_tnsr_b(vc, kv));
        }
    }
}
