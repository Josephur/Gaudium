// gaudium_attn_softmax_fresh: masked two-pass softmax over the FRESH WINDOW of a chained MTP pass (engine step recipe).
//  The window holds the K/V rows written in this step so far (catch-up rows, a zero row, recursive drafts) in NF slots whose
//  absolute positions are given by a device table, so the validity of a slot is a table lookup instead of a position range:
//  in0 scores f32 {NF, R, 4}   (R = 6T rows per kv group, row = hl*T + t; NF multiple of 256)
//  in1 fpos  int32 {NF}        absolute position of slot j, or a huge value (DEAD) for an unused / shadowed slot
//  in2 cursor int32 {1}        query positions are cursor + coff + t
//  out0 P bf16 {NF, R, 4}      exp(x - m) where fpos[j] <= cursor + coff + t, 0 elsewhere (interleaved order as the chunk kernel)
//  out1 stats f32 {64, 2, R, 4}  per row: m and s, exactly like gaudium_attn_softmax_chunk (one extra "chunk" for the combine)
//  index space 2-D: dim0 = R, dim1 = 4.   scalars: T, NF, coff
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define REDUCE_MAX64(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = v_f32_max_b(_a, _r); _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = v_f32_max_b(_a, _r); \
    _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P8, 0, _a)); _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P4, 0, _a)); \
    _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P2, 0, _a)); _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P1, 0, _a)); _a; })
#define REDUCE64(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = _a + _r; _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = _a + _r; \
    _a = _a + v_f32_shuffle_b(_a, P8, 0, _a); _a = _a + v_f32_shuffle_b(_a, P4, 0, _a); \
    _a = _a + v_f32_shuffle_b(_a, P2, 0, _a); _a = _a + v_f32_shuffle_b(_a, P1, 0, _a); _a; })
// Decode exp (research/exp_bug): the vendor v_exp_f32 is off by a factor of ~2 at exactly 42 inputs in [-104, 0]
// (one per odd k in -121..-1, x log2 e just below k; x = -0.693147182 gives 1.0; exhaustive card scan == libtpcsim). DEXP is the
// FASTEXP formula of the prefill build (EXPD below: max 1 ulp on [-87, 0], none of those errors, 13 VPU ops instead of 29);
// -DVENDOR_EXP restores v_exp_f32 (A/B builds).
#define EXPD(d_) ({ float64 _d = v_f32_max_b((d_), -87.0f); \
    float64 _z = v_f32_nearbyint_b(_d * 1.44269504088896341f, SW_RHNE, 0, 1, 0); \
    float64 _r = v_f32_mac_b(_z, -0.693359375f, _d, 0, 1, 0); _r = v_f32_mac_b(_z, 2.12194440e-4f, _r, 0, 1, 0); \
    float64 _p = v_f32_mac_b(_r, 0.0013836704893037677f, (float64)0.008374780416488647f, 0, 1, 0); \
    _p = v_f32_mac_b(_p, _r, (float64)0.04166822507977486f, 0, 1, 0); _p = v_f32_mac_b(_p, _r, (float64)0.1666642129421234f, 0, 1, 0); \
    _p = v_f32_mac_b(_p, _r, (float64)0.49999991059303284f, 0, 1, 0); _p = v_f32_mac_b(_p, _r, (float64)1.0f, 0, 1, 0); \
    _p = v_f32_mac_b(_p, _r, (float64)1.0f, 0, 1, 0); \
    int64 _n = v_i32_shl_b(v_convert_f32_to_i32_b(_z, SW_RHNE, 0, 1, 0), 23, 0, 0, 1, 0); \
    int64 _b = v_i32_add_b(*(int64*)&_p, _n, 0, 0, 1, 0); *(float64*)&_b; })
#ifdef VENDOR_EXP
#define DEXP(x) v_exp_f32(x)
#else
#define DEXP(x) EXPD(x)
#endif
static inline uchar256 xor_pat(uchar256 l16, int s) { uchar256 src = l16 ^ (unsigned char)s; return ((src & 7) | ((src >> 3) << 5)) | 0x80; }

// -DKV8 build: raw fp8 GEMM scores and P fp8 (fmt 0 = fp8_143, 1 = fp8_152) for the fp8 PV GEMM over the fp8 window, as the
// chunk kernel's KV8 build: d = (x - m) * kv8[16], P x kv8[12 + g], stats m in true units and s * kv8[17] (kv8 f32 {32} input after cursor).
#ifdef KV8
void main(tensor scores, tensor fpos, tensor cursor, tensor kv8, tensor P, tensor stats, int T, int NF, int coff, int fmt)
#else
void main(tensor scores, tensor fpos, tensor cursor, tensor P, tensor stats, int T, int NF, int coff)
#endif
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const uchar256 lid8 = read_lane_id_1b_b();
    const uchar256 l16 = (lid8 >> 2) & 15;
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
    int5 zc = {0, 0, 0, 0, 0};
    const int cur = s_i32_ld_g(gen_addr(zc, cursor)) + coff;
    const float NEG = -1e30f;

    for (int g = idx[1]; g < idx[1] + isz[1]; ++g) {
        for (int row = idx[0]; row < idx[0] + isz[0]; ++row) {
            const int t = row % T;
#ifdef KV8
            int5 pmc = {12 + g, 0, 0, 0, 0}; const float pm8 = s_f32_ld_g(gen_addr(pmc, kv8));
            pmc[0] = 16; const float sqk8 = s_f32_ld_g(gen_addr(pmc, kv8)); pmc[0] = 17; const float sm8 = s_f32_ld_g(gen_addr(pmc, kv8));
#define SQK(d) ((d) * sqk8)
#else
#define SQK(d) (d)
#endif
            const int qlim = cur + t + 1;                   // slot valid iff fpos[j] < qlim
            int5 c = {0, row, g, 0, 0};
            int5 fc = {0, 0, 0, 0, 0};
            int5 sc = {0, 0, row, g, 0};
            float64 m0 = NEG, m1 = NEG, m2 = NEG, m3 = NEG;
            for (int j = 0; j < NF; j += 256) {
                fc[0] = j;       int64 f0 = v_i32_ld_tnsr_b(fc, fpos);
                fc[0] = j + 64;  int64 f1 = v_i32_ld_tnsr_b(fc, fpos);
                fc[0] = j + 128; int64 f2 = v_i32_ld_tnsr_b(fc, fpos);
                fc[0] = j + 192; int64 f3 = v_i32_ld_tnsr_b(fc, fpos);
                c[0] = j;       float64 x0 = v_f32_ld_tnsr_b(c, scores);
                c[0] = j + 64;  float64 x1 = v_f32_ld_tnsr_b(c, scores);
                c[0] = j + 128; float64 x2 = v_f32_ld_tnsr_b(c, scores);
                c[0] = j + 192; float64 x3 = v_f32_ld_tnsr_b(c, scores);
                m0 = v_f32_max_b(m0, v_f32_sel_less_i32_b(f0, qlim, x0, NEG));
                m1 = v_f32_max_b(m1, v_f32_sel_less_i32_b(f1, qlim, x1, NEG));
                m2 = v_f32_max_b(m2, v_f32_sel_less_i32_b(f2, qlim, x2, NEG));
                m3 = v_f32_max_b(m3, v_f32_sel_less_i32_b(f3, qlim, x3, NEG));
            }
            float64 m = REDUCE_MAX64(v_f32_max_b(v_f32_max_b(m0, m1), v_f32_max_b(m2, m3)), P8, P4, P2, P1);
            // a row with no valid slot (cannot happen in the engine: every pass sees at least its own row) would give m = -1e30
            float64 s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
            for (int j = 0; j < NF; j += 256) {
                fc[0] = j;       int64 f0 = v_i32_ld_tnsr_b(fc, fpos);
                fc[0] = j + 64;  int64 f1 = v_i32_ld_tnsr_b(fc, fpos);
                fc[0] = j + 128; int64 f2 = v_i32_ld_tnsr_b(fc, fpos);
                fc[0] = j + 192; int64 f3 = v_i32_ld_tnsr_b(fc, fpos);
                c[0] = j;       float64 x0 = v_f32_ld_tnsr_b(c, scores);
                c[0] = j + 64;  float64 x1 = v_f32_ld_tnsr_b(c, scores);
                c[0] = j + 128; float64 x2 = v_f32_ld_tnsr_b(c, scores);
                c[0] = j + 192; float64 x3 = v_f32_ld_tnsr_b(c, scores);
                float64 e0 = v_f32_sel_less_i32_b(f0, qlim, DEXP(SQK(x0 - m)), 0.f);
                float64 e1 = v_f32_sel_less_i32_b(f1, qlim, DEXP(SQK(x1 - m)), 0.f);
                float64 e2 = v_f32_sel_less_i32_b(f2, qlim, DEXP(SQK(x2 - m)), 0.f);
                float64 e3 = v_f32_sel_less_i32_b(f3, qlim, DEXP(SQK(x3 - m)), 0.f);
                s0 = s0 + e0; s1 = s1 + e1; s2 = s2 + e2; s3 = s3 + e3;
#ifdef KV8
                { float256 f8; f8.v1 = e0 * pm8; f8.v2 = e1 * pm8; f8.v3 = e2 * pm8; f8.v4 = e3 * pm8; int5 oc = {j, row, g, 0, 0};
                  if (fmt) v_h8_st_tnsr(oc, P, v_convert_f32_to_h8_all_b(f8, SW_RHNE | SW_CLIP_FP)); else v_f8_st_tnsr(oc, P, v_convert_f32_to_f8_all_b(f8, SW_RHNE | SW_CLIP_FP)); }
#else
                float128 pa; pa.v1 = e0; pa.v2 = e1;
                float128 pb; pb.v1 = e2; pb.v2 = e3;
                int5 oc = {j, row, g, 0, 0};
                v_bf16_st_tnsr(oc, P, v_convert_f32_to_bf16_all_b(pa));
                oc[0] = j + 128;
                v_bf16_st_tnsr(oc, P, v_convert_f32_to_bf16_all_b(pb));
#endif
            }
            float64 s = REDUCE64((s0 + s1) + (s2 + s3), P8, P4, P2, P1);
#ifdef KV8
            v_f32_st_tnsr(sc, stats, m * sqk8); sc[1] = 1; v_f32_st_tnsr(sc, stats, s * sm8);
#else
            v_f32_st_tnsr(sc, stats, m); sc[1] = 1; v_f32_st_tnsr(sc, stats, s);
#endif
        }
    }
}
