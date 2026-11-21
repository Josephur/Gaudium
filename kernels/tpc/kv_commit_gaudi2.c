#include "attn_shape.h"   // ATT_NKV, ATT_GQ, ATT_NH, ATT_NPT, ATT_GATE: compile-time head counts (issue #7); the defaults are Qwen3.8-27B's
// gaudium_kv_commit: move the accepted rows of the previous step's fresh window into the big (chunked) KV cache of the MTP layer.
//  Runs first in a step recipe, as the ONLY producer of the big cache in that recipe (every MTP pass then reads it through the
//  GEMMs); the rows it does not touch keep their values (persistent memory, partial write), like the trunk layers' caches.
//  in0 kf   bf16 {256, NF, 4}   previous step's catch-up window: slot j = K row of position cursor - ncommit + j
//  in1 vf   bf16 {256, NF, 4}   same, V rows at perm(j) inside the window's 128-blocks
//  in2 ncommit int32 {1}        rows to commit = accepted drafts + 1 of the previous step (0 for the first step after the prefill)
//  in3 cursor  int32 {1}        this step's cursor; the committed positions are [cursor - ncommit, cursor)
//  out0 kcache bf16 {256, C, 4*nch}   out1 vcache bf16 {256, C, 4*nch}   chunk-major (position p -> {p % C, (p / C)*4 + g}),
//        V rows permuted inside 128-blocks like gaudium_attn_prep.
//  index space 1-D: 4 (one point per kv group).   scalars: C
#define VPERM(j) (((j) & ~127) + ((((j) & 127) < 64) ? 2 * ((j) & 127) : 2 * (((j) & 127) - 64) + 1))
//  -DKV8 build: window and cache are fp8 (either format: a raw 256-byte copy per row), V rows permuted inside 256-row
//  blocks (attn_prep VPERM8) in both.
#define VPERM8(j) (((j) & ~255) + 4 * ((j) & 63) + (((j) >> 6) & 3))
void main(tensor kf, tensor vf, tensor ncommit, tensor cursor, tensor kcache, tensor vcache, int C)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    int5 zc = {0, 0, 0, 0, 0};
    const int n = s_i32_ld_g(gen_addr(zc, ncommit));
    const int cur = s_i32_ld_g(gen_addr(zc, cursor));
    for (int g = idx[0]; g < idx[0] + isz[0]; ++g) {
        for (int j = 0; j < n; ++j) {
            const int p = cur - n + j, chk = p / C, pin = p - chk * C;
            int5 sk = {0, j, g, 0, 0};
#ifdef KV8
            { int5 dk = {0, pin, chk * ATT_NKV + g, 0, 0}; v_i8_st_tnsr(dk, kcache, v_i8_ld_tnsr_b(sk, kf));
              int5 sv = {0, VPERM8(j), g, 0, 0}, dv = {0, VPERM8(pin), chk * ATT_NKV + g, 0, 0}; v_i8_st_tnsr(dv, vcache, v_i8_ld_tnsr_b(sv, vf)); }
            continue;
#endif
            bfloat128 k1 = v_bf16_ld_tnsr_b(sk, kf); sk[0] = 128; bfloat128 k2 = v_bf16_ld_tnsr_b(sk, kf);
            int5 dk = {0, pin, chk * ATT_NKV + g, 0, 0};
            v_bf16_st_tnsr(dk, kcache, k1); dk[0] = 128; v_bf16_st_tnsr(dk, kcache, k2);
            int5 sv = {0, VPERM(j), g, 0, 0};
            bfloat128 v1 = v_bf16_ld_tnsr_b(sv, vf); sv[0] = 128; bfloat128 v2 = v_bf16_ld_tnsr_b(sv, vf);
            int5 dv = {0, VPERM(pin), chk * ATT_NKV + g, 0, 0};
            v_bf16_st_tnsr(dv, vcache, v1); dv[0] = 128; v_bf16_st_tnsr(dv, vcache, v2);
        }
    }
}
