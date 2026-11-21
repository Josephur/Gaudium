// gaudium_df_attn_prep: DFlash2 draft attention front end for the T rows of one block at positions cur + t.
//   32 q heads x 128: per-head RMSNorm(w) -> full NeoX RoPE (pairs c, c+64) -> * 1/sqrt(128) -> bf16 rows for batch_gemm
//   8 kv heads: K: RMSNorm(w) -> RoPE -> kblk row t ; V -> vblk row perm(t)
//  in0 q bf16 {4096, T}  in1 k bf16 {1024, T}  in2 v bf16 {1024, T}  in3 qn_w f32 {128} (lane order)  in4 kn_w f32 {128}
//  in5 rope_c f32 {128, Lmax}, in6 rope_s: lanes [0..31] cos(pos f_2j), [32..63] same; lanes 64.. the odd frequencies;
//      sin: [0..31] -sin, [32..63] +sin (RoPE pair (c, c+64) sits in lanes (j, j+32) of v1 (even c) or v2 (odd c))
//  in7 cur int32 {1}
//  out0 qb bf16 {128, 4T, 8}: row (h%4)*T + t of group h/4 = q head h   out1 kblk bf16 {128, 128, 8}  out2 vblk bf16 {128, 128, 8}
//  index space: 40 (32 q heads + 8 kv heads).  scalars: T
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define REDUCE64(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = _a + _r; _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = _a + _r; \
    _a = _a + v_f32_shuffle_b(_a, P8, 0, _a); _a = _a + v_f32_shuffle_b(_a, P4, 0, _a); \
    _a = _a + v_f32_shuffle_b(_a, P2, 0, _a); _a = _a + v_f32_shuffle_b(_a, P1, 0, _a); _a; })
static inline uchar256 xor_pat(uchar256 l16, int s) { uchar256 src = l16 ^ (unsigned char)s; return ((src & 7) | ((src >> 3) << 5)) | 0x80; }
#define ROPE128(x, C, S) ({ float64 _x = (x); float64 _sw = MOVDG_ALL(_x, 2, 3, 0, 1, _x); _x * (C) + _sw * (S); })

void main(tensor q, tensor k, tensor v, tensor qn_w, tensor kn_w, tensor rope_c, tensor rope_s, tensor cursor,
          tensor qb, tensor kblk, tensor vblk, int T)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 l16 = (lid8 >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const float eps = 1e-6f, qscale = 0.08838834764831845f, inv_n = 1.0f / 128.0f;
    int5 zc = {0, 0, 0, 0, 0};
    const int cur = s_i32_ld_g(gen_addr(zc, cursor));
    for (int i = idx[0]; i < idx[0] + isz[0]; ++i) {
        const int is_q = (i < 32);
        const int h = is_q ? i : i - 32;
        int5 wc = {0, 0, 0, 0, 0};
        float64 wa, wb;
        if (is_q) { wa = v_f32_ld_tnsr_b(wc, qn_w); wc[0] = 64; wb = v_f32_ld_tnsr_b(wc, qn_w); }
        else      { wa = v_f32_ld_tnsr_b(wc, kn_w); wc[0] = 64; wb = v_f32_ld_tnsr_b(wc, kn_w); }
        for (int t = 0; t < T; ++t) {
            const int pos = cur + t;
            int5 xc = {128 * h, t, 0, 0, 0};
            float128 a = v_convert_bf16_to_f32_all_b(is_q ? v_bf16_ld_tnsr_b(xc, q) : v_bf16_ld_tnsr_b(xc, k));
            float64 ss = REDUCE64(a.v1 * a.v1 + a.v2 * a.v2, P8, P4, P2, P1);
            float64 r = v_rsqrt_f32(ss * inv_n + eps);
            a.v1 = a.v1 * r * wa; a.v2 = a.v2 * r * wb;
            int5 rc = {0, pos, 0, 0, 0};
            float64 C1 = v_f32_ld_tnsr_b(rc, rope_c), S1 = v_f32_ld_tnsr_b(rc, rope_s);
            rc[0] = 64;
            float64 C2 = v_f32_ld_tnsr_b(rc, rope_c), S2 = v_f32_ld_tnsr_b(rc, rope_s);
            a.v1 = ROPE128(a.v1, C1, S1); a.v2 = ROPE128(a.v2, C2, S2);
            if (is_q) {
                a.v1 = a.v1 * qscale; a.v2 = a.v2 * qscale;
                int5 oc = {0, (h % 4) * T + t, h / 4, 0, 0};
                v_bf16_st_tnsr(oc, qb, v_convert_f32_to_bf16_all_b(a));
            } else {
                int5 kc = {0, t, h, 0, 0};
                v_bf16_st_tnsr(kc, kblk, v_convert_f32_to_bf16_all_b(a));
                const int vrow = (t < 64) ? 2 * t : 2 * (t - 64) + 1;
                int5 vc = {128 * h, t, 0, 0, 0};
                bfloat128 vv = v_bf16_ld_tnsr_b(vc, v);
                int5 vo = {0, vrow, h, 0, 0};
                v_bf16_st_tnsr(vo, vblk, vv);
            }
        }
    }
}
