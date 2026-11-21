// gaudium_gdn_posta: phase A of the row-parallel prefill gated RMSNorm as its own node (rsqrt table only; see
// gdn_prepa_gaudi2.c for why): r[t, h] = rsqrt(mean(o[t, h, :]^2) + eps), stored as 64 equal lanes. gaudium_gdn_postb applies it.
// Same arithmetic as gdn_post2_gaudi2.c phase A: bit-identical.
//  in0 o f32 {128, 2C, 48, NCH} (row t at {., t % C, h, t / C})  out0 r f32 {64, T, 48}   scalars: T, C, RG.  Index space {48 heads, ceil(T/RG)}.
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define REDUCE64(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = _a + _r; _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = _a + _r; \
    _a = _a + v_f32_shuffle_b(_a, P8, 0, _a); _a = _a + v_f32_shuffle_b(_a, P4, 0, _a); \
    _a = _a + v_f32_shuffle_b(_a, P2, 0, _a); _a = _a + v_f32_shuffle_b(_a, P1, 0, _a); _a; })
static inline uchar256 xor_pat(uchar256 l16, int s) { uchar256 src = l16 ^ (unsigned char)s; return ((src & 7) | ((src >> 3) << 5)) | 0x80; }

void main(tensor o, tensor r, int T, int CH, int RG)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 l16 = (lid8 >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const float eps = 1e-6f, inv_n = 1.0f / 128.0f;
    for (int h = idx[0]; h < idx[0] + isz[0]; ++h) {
      for (int rg = idx[1]; rg < idx[1] + isz[1]; ++rg) {
        const int t0 = rg * RG, t1 = (t0 + RG < T) ? t0 + RG : T;
#pragma unroll 4
        for (int t = t0; t < t1; ++t) {
            int5 oc = {0, t % CH, h, t / CH, 0};
            float64 oa = v_f32_ld_tnsr_b(oc, o); oc[0] = 64; float64 ob = v_f32_ld_tnsr_b(oc, o);
            float64 ss = REDUCE64(oa * oa + ob * ob, P8, P4, P2, P1);
            int5 rc = {0, t, h, 0, 0};
            v_f32_st_tnsr(rc, r, v_rsqrt_f32(ss * inv_n + eps));
        }
      }
    }
}
