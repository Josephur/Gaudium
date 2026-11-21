// gaudium_qsa_prep: QSA indexer front end (Qwen3.8-Flash-Next, docs/notes/engine-qsa.md). One index point per row t.
//  in0 iqk bf16 {640, T}: rows 0..511 the 4 indexer q heads (head h = rows 128h..128h+127), rows 512..639 the raw indexer key
//  in1 iqn_w f32 {128}, in2 ikn_w f32 {128}: 1 + w of the indexer q / k RMSNorm, LANE ORDER (index j < 64: channel 2j, else 2(j-64)+1)
//  in3 rope_c f32 {128, Lmax}, in4 rope_s f32 {128, Lmax}: the attention's RoPE tables (row = KV slot; see attn_prep_gaudi2.c)
//  in5 cursor int32 {1}: row t sits at KV slot cursor + coff + t (chain rows)
//  out0 qi bf16 {128, 4T}: column 4t + h = bf16(rope_slot(rmsnorm(q_h) * (1 + w_q)))   (f32 from the bf16 input, one rounding)
//  out1 ikcache bf16 {128, Lmax}: raw key of slot s in column s (a copy of the projection's bf16 row)
//  out2 ibcache bf16 {128, Lmax/4}: when slot % 4 == 3 the block key of block b = slot / 4:
//       bf16(rope_{4b}(rmsnorm(bf16(((k0 + k1) + (k2 + k3)) * 0.25)) * (1 + w_k)))   k_j = raw key of slot 4b + j (f32 of bf16)
//       slots of the block inside this pass come from in0 (rows t-3..t-1), older ones from ikcache (written by earlier launches:
//       this launch never reads a column it writes, hardware-learnings section 5).
//  scalars: coff.   Index space {T}.
//  -DFRESH (MTP passes, mode 1): + in6 ikf_in bf16 {128, NF}; outputs qi, ikf_out {128, NF}: the window copy of ikf_in except slots
//       [sbase, sbase + T), which get this pass's raw keys (like attn_prep -DFRESH's kf). No block keys. scalars: coff, sbase.
//       Index space {8}: point i copies window rows i, i+8, ... and does rows t = i, i+8, ...
//  -DCOMMIT (gaudium_qsa_commit, MTP step start): in0 ikf bf16 {128, NF} (previous step's window: slot j = position cursor - n + j),
//       in1 ncommit int32 {1} (n), in2 cursor int32 {1}, in3 ikn_w, in4 rope_c, in5 rope_s -> out0 ikcache, out1 ibcache:
//       raw rows j < n to slots cursor - n + j, and the block key of every block whose last slot is one of them. Index space {16}.
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define REDUCE64(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = _a + _r; _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = _a + _r; \
    _a = _a + v_f32_shuffle_b(_a, P8, 0, _a); _a = _a + v_f32_shuffle_b(_a, P4, 0, _a); \
    _a = _a + v_f32_shuffle_b(_a, P2, 0, _a); _a = _a + v_f32_shuffle_b(_a, P1, 0, _a); _a; })
static inline uchar256 xor_pat(uchar256 l16, int s) { uchar256 src = l16 ^ (unsigned char)s; return ((src & 7) | ((src >> 3) << 5)) | 0x80; }
#define ROPE(x, C, S) ({ float64 _x = (x); float64 _sw = MOVDG_ALL(_x, 1, 0, 2, 3, _x); _x * (C) + _sw * (S); })
// RMSNorm(128) * w, then RoPE at table row `slot`, of a float128 in lane order (v1 = even channels, v2 = odd); returns bf16
#define NORM_ROPE(a_, w1_, w2_, slot_) ({ float128 _nr = (a_); \
    float64 _ss = REDUCE64(_nr.v1 * _nr.v1 + _nr.v2 * _nr.v2, P8, P4, P2, P1); \
    float64 _r = v_rsqrt_f32(_ss * 0.0078125f + 1e-6f); \
    _nr.v1 = _nr.v1 * _r * (w1_); _nr.v2 = _nr.v2 * _r * (w2_); \
    int5 _rc = {0, (slot_), 0, 0, 0}; float64 _C1 = v_f32_ld_tnsr_b(_rc, rope_c), _S1 = v_f32_ld_tnsr_b(_rc, rope_s); \
    _rc[0] = 64; float64 _C2 = v_f32_ld_tnsr_b(_rc, rope_c), _S2 = v_f32_ld_tnsr_b(_rc, rope_s); \
    _nr.v1 = ROPE(_nr.v1, _C1, _S1); _nr.v2 = ROPE(_nr.v2, _C2, _S2); v_convert_f32_to_bf16_all_b(_nr); })
#define F32(x_) v_convert_bf16_to_f32_all_b(x_)
// block key from the 4 raw bf16 keys (f32 mean rounded to bf16 as the reference does, then norm + rope at the block's first slot)
#define BLOCK_KEY(k0_, k1_, k2_, k3_, s0_) ({ float128 _f0 = F32(k0_), _f1 = F32(k1_), _f2 = F32(k2_), _f3 = F32(k3_); float128 _m; \
    _m.v1 = ((_f0.v1 + _f1.v1) + (_f2.v1 + _f3.v1)) * 0.25f; _m.v2 = ((_f0.v2 + _f1.v2) + (_f2.v2 + _f3.v2)) * 0.25f; \
    float128 _mb = F32(v_convert_f32_to_bf16_all_b(_m)); NORM_ROPE(_mb, kw1, kw2, (s0_)); })

#if defined(COMMIT)
void main(tensor ikf, tensor ncommit, tensor cursor, tensor ikn_w, tensor rope_c, tensor rope_s, tensor ikcache, tensor ibcache)
#elif defined(FRESH)
void main(tensor iqk, tensor iqn_w, tensor ikn_w, tensor rope_c, tensor rope_s, tensor cursor, tensor ikf_in, tensor qi, tensor ikf_out, int coff, int sbase)
#else
void main(tensor iqk, tensor iqn_w, tensor ikn_w, tensor rope_c, tensor rope_s, tensor cursor, tensor qi, tensor ikcache, tensor ibcache, int coff)
#endif
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 l16 = (lid8 >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    int5 zc = {0, 0, 0, 0, 0};
    float64 kw1 = v_f32_ld_tnsr_b(zc, ikn_w); zc[0] = 64; float64 kw2 = v_f32_ld_tnsr_b(zc, ikn_w); zc[0] = 0;
#if defined(COMMIT)
    const int n = s_i32_ld_g(gen_addr(zc, ncommit));
    const int cur = s_i32_ld_g(gen_addr(zc, cursor));
    const int first = cur - n;                     // slot of window row 0
    for (int i = idx[0]; i < idx[0] + isz[0]; ++i) {
        for (int j = i; j < n; j += 16) {
            const int slot = first + j;
            int5 wc = {0, j, 0, 0, 0}; bfloat128 kr = v_bf16_ld_tnsr_b(wc, ikf);
            int5 oc = {0, slot, 0, 0, 0}; v_bf16_st_tnsr(oc, ikcache, kr);
            if ((slot & 3) == 3) {
                const int s0 = slot - 3;
#define LDK(u_) ({ const int _su = s0 + (u_); int5 _c2 = {0, _su >= first ? _su - first : _su, 0, 0, 0}; \
                   _su >= first ? v_bf16_ld_tnsr_b(_c2, ikf) : v_bf16_ld_tnsr_b(_c2, ikcache); })
                bfloat128 k0 = LDK(0), k1 = LDK(1), k2 = LDK(2);
#undef LDK
                bfloat128 bk = BLOCK_KEY(k0, k1, k2, kr, s0);
                int5 bc = {0, slot >> 2, 0, 0, 0}; v_bf16_st_tnsr(bc, ibcache, bk);
            }
        }
    }
#else
    zc[0] = 0; float64 qw1 = v_f32_ld_tnsr_b(zc, iqn_w); zc[0] = 64; float64 qw2 = v_f32_ld_tnsr_b(zc, iqn_w); zc[0] = 0;
    const int cur = s_i32_ld_g(gen_addr(zc, cursor)) + coff;
    const int T = get_dim_size(qi, 1) >> 2;
#ifdef FRESH
    const int NF = get_dim_size(ikf_out, 1), NP = 8;
    for (int i = idx[0]; i < idx[0] + isz[0]; ++i) {
        for (int p = i; p < NF; p += NP) {   // window copy: each output row has exactly one writer
            if (p >= sbase && p < sbase + T) continue;
            int5 wc = {0, p, 0, 0, 0}; v_bf16_st_tnsr(wc, ikf_out, v_bf16_ld_tnsr_b(wc, ikf_in));
        }
        for (int t = i; t < T; t += NP) {
#else
    for (int i = idx[0]; i < idx[0] + isz[0]; ++i) {
        {
            const int t = i;
#endif
            const int slot = cur + t;
            for (int h = 0; h < 4; ++h) {
                int5 xc = {128 * h, t, 0, 0, 0};
                bfloat128 q = NORM_ROPE(F32(v_bf16_ld_tnsr_b(xc, iqk)), qw1, qw2, slot);
                int5 oc = {0, 4 * t + h, 0, 0, 0}; v_bf16_st_tnsr(oc, qi, q);
            }
            int5 kc = {512, t, 0, 0, 0}; bfloat128 kr = v_bf16_ld_tnsr_b(kc, iqk);
#ifdef FRESH
            int5 wo = {0, sbase + t, 0, 0, 0}; v_bf16_st_tnsr(wo, ikf_out, kr);
#else
            int5 oc = {0, slot, 0, 0, 0}; v_bf16_st_tnsr(oc, ikcache, kr);
            if ((slot & 3) == 3) {
                const int s0 = slot - 3;
                // a slot of this pass: its row of the projection; an earlier slot: ikcache (written by an earlier launch)
#define LDK(u_) ({ const int _su = s0 + (u_); int5 _ca = {512, t - 3 + (u_), 0, 0, 0}, _cb = {0, _su, 0, 0, 0}; \
                   _su >= cur ? v_bf16_ld_tnsr_b(_ca, iqk) : v_bf16_ld_tnsr_b(_cb, ikcache); })
                bfloat128 k0 = LDK(0), k1 = LDK(1), k2 = LDK(2);
#undef LDK
                bfloat128 bk = BLOCK_KEY(k0, k1, k2, kr, s0);
                int5 bc = {0, slot >> 2, 0, 0, 0}; v_bf16_st_tnsr(bc, ibcache, bk);
            }
#endif
        }
    }
#endif
}
