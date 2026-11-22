// gaudium_ds_ctxr: context K / V rows of a block drafter's layer into RINGS of any length.
// A drafter layer that looks back a bounded window (the checkpoint's sliding window) keeps a bounded state, whatever the
// context: the newest RG rows, slot = position mod RG. RG = the rings' second dimension, a multiple of 128 (V rows are stored
// permuted inside 128-row blocks, as the P x V GEMM of gaudium_ds_softr expects them).
// For T verified rows: the layer's K / V projections of hidden_norm(fc(target features)) -> k_norm -> RoPE at position
// cur + depth[t] -> K ring slot (cur + t) mod RG, V ring row perm(slot). T <= RG, so every slot has one writer per launch.
// A chain only: there is no slot move (the linear-cache kernel gaudium_df_ctx has one for tree leaves).
//  in0 kc bf16 {1024, T}  in1 vc bf16 {1024, T}  in2 kn_w f32 {128} (lane order)  in3 rope_c f32 {128, Lpos}  in4 rope_s
//  in5 cursor int32 {1}  in6 depth int32 {>= T}  in7 lim int32 {2} (not read; the softmax's bound)
//  out0 kring bf16 {128, RG, 8}  out1 vring bf16 {128, RG, 8}.   index space: 8 kv heads.  scalars: T, RG
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define REDUCE64(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = _a + _r; _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = _a + _r; \
    _a = _a + v_f32_shuffle_b(_a, P8, 0, _a); _a = _a + v_f32_shuffle_b(_a, P4, 0, _a); \
    _a = _a + v_f32_shuffle_b(_a, P2, 0, _a); _a = _a + v_f32_shuffle_b(_a, P1, 0, _a); _a; })
static inline uchar256 xor_pat(uchar256 l16, int s) { uchar256 src = l16 ^ (unsigned char)s; return ((src & 7) | ((src >> 3) << 5)) | 0x80; }
#define ROPE128(x, C, S) ({ float64 _x = (x); float64 _sw = MOVDG_ALL(_x, 2, 3, 0, 1, _x); _x * (C) + _sw * (S); })
static inline int perm(int p) { const int j = p & 127; return (p & ~127) + ((j < 64) ? 2 * j : 2 * (j - 64) + 1); }

void main(tensor kc, tensor vc, tensor kn_w, tensor rope_c, tensor rope_s, tensor cursor, tensor depth, tensor lim,
          tensor kring, tensor vring, int T, int RG)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 l16 = (lid8 >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    const float eps = 1e-6f, inv_n = 1.0f / 128.0f;
    int5 zc = {0, 0, 0, 0, 0};
    const int cur = s_i32_ld_g(gen_addr(zc, cursor));
    const int s0 = cur % RG;   // ring slot of the first row
    for (int h = idx[0]; h < idx[0] + isz[0]; ++h) {
        int5 wc = {0, 0, 0, 0, 0};
        float64 wa = v_f32_ld_tnsr_b(wc, kn_w); wc[0] = 64; float64 wb = v_f32_ld_tnsr_b(wc, kn_w);
        for (int t = 0; t < T; ++t) {
            int5 dc = {t, 0, 0, 0, 0};
            const int pos = cur + s_i32_ld_g(gen_addr(dc, depth));
            int slot = s0 + t; slot = (slot >= RG) ? slot - RG : slot;
            int5 xc = {128 * h, t, 0, 0, 0};
            float128 a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(xc, kc));
            float64 ss = REDUCE64(a.v1 * a.v1 + a.v2 * a.v2, P8, P4, P2, P1);
            float64 r = v_rsqrt_f32(ss * inv_n + eps);
            a.v1 = a.v1 * r * wa; a.v2 = a.v2 * r * wb;
            int5 rc = {0, pos, 0, 0, 0};
            float64 C1 = v_f32_ld_tnsr_b(rc, rope_c), S1 = v_f32_ld_tnsr_b(rc, rope_s);
            rc[0] = 64;
            float64 C2 = v_f32_ld_tnsr_b(rc, rope_c), S2 = v_f32_ld_tnsr_b(rc, rope_s);
            a.v1 = ROPE128(a.v1, C1, S1); a.v2 = ROPE128(a.v2, C2, S2);
            int5 kk = {0, slot, h, 0, 0};
            v_bf16_st_tnsr(kk, kring, v_convert_f32_to_bf16_all_b(a));
            int5 vo = {0, perm(slot), h, 0, 0};
            v_bf16_st_tnsr(vo, vring, v_bf16_ld_tnsr_b(xc, vc));
        }
    }
}
