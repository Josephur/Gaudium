// ik_common.h: shared pieces of the Inkling 2 kernels (ik_*_gaudi2.c).
//
// Layout conventions of the family (every kernel and the weight packer follow them):
//  - bf16 tensors hold their channels in the checkpoint's order unless a kernel says otherwise. A bf16 load of a block of 128 channels
//    converts to two f32 vectors: v1 = the even positions, v2 = the odd positions; the converting store puts them back.
//  - f32 activation tensors are stored "split": lanes [0, 64) of a block of 128 hold the block's even channels, lanes [64, 128) the odd
//    ones. A kernel therefore loads / stores them as the same (v1, v2) pair without touching a lane. Per-channel f32 tables are packed
//    the same way by the loader.
//  - Positions are cursor[0] + coff + t for row t of a pass.
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
// sum / max over all 64 lanes, broadcast to every lane (two dual-group moves, then xor butterflies inside each 16-lane dual group)
#define REDUCE64(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = _a + _r; _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = _a + _r; \
    _a = _a + v_f32_shuffle_b(_a, P8, 0, _a); _a = _a + v_f32_shuffle_b(_a, P4, 0, _a); \
    _a = _a + v_f32_shuffle_b(_a, P2, 0, _a); _a = _a + v_f32_shuffle_b(_a, P1, 0, _a); _a; })
#define REDUCE_MAX64(a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = v_f32_max_b(_a, _r); _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = v_f32_max_b(_a, _r); \
    _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P8, 0, _a)); _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P4, 0, _a)); \
    _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P2, 0, _a)); _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P1, 0, _a)); _a; })
// sum over each aligned group of 8 lanes, broadcast to its 8 lanes (the xor butterflies 4, 2, 1 stay inside such a group)
#define REDUCE8(a, P4, P2, P1) ({ float64 _a = (a); \
    _a = _a + v_f32_shuffle_b(_a, P4, 0, _a); _a = _a + v_f32_shuffle_b(_a, P2, 0, _a); _a = _a + v_f32_shuffle_b(_a, P1, 0, _a); _a; })
static inline uchar256 xor_pat(uchar256 l16, int s) { uchar256 src = l16 ^ (unsigned char)s; return ((src & 7) | ((src >> 3) << 5)) | 0x80; }
#define IK_PATS \
    const uchar256 lid8 = read_lane_id_1b_b(); const uchar256 l16 = (lid8 >> 2) & 15; \
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
// e^d for d <= 0: Cody-Waite reduction + a degree-6 polynomial (max 1 ulp on [-87, 0]); inputs below -87 give e^-87
#define EXPD(d_) ({ float64 _d = v_f32_max_b((d_), -87.0f); \
    float64 _z = v_f32_nearbyint_b(_d * 1.44269504088896341f, SW_RHNE, 0, 1, 0); \
    float64 _r = v_f32_mac_b(_z, -0.693359375f, _d, 0, 1, 0); _r = v_f32_mac_b(_z, 2.12194440e-4f, _r, 0, 1, 0); \
    float64 _p = v_f32_mac_b(_r, 0.0013836704893037677f, (float64)0.008374780416488647f, 0, 1, 0); \
    _p = v_f32_mac_b(_p, _r, (float64)0.04166822507977486f, 0, 1, 0); _p = v_f32_mac_b(_p, _r, (float64)0.1666642129421234f, 0, 1, 0); \
    _p = v_f32_mac_b(_p, _r, (float64)0.49999991059303284f, 0, 1, 0); _p = v_f32_mac_b(_p, _r, (float64)1.0f, 0, 1, 0); \
    _p = v_f32_mac_b(_p, _r, (float64)1.0f, 0, 1, 0); \
    int64 _n = v_i32_shl_b(v_convert_f32_to_i32_b(_z, SW_RHNE, 0, 1, 0), 23, 0, 0, 1, 0); \
    int64 _b = v_i32_add_b(*(int64*)&_p, _n, 0, 0, 1, 0); *(float64*)&_b; })
// 1 / a for a > 0: the unit's estimate and one Newton step
#define RECIP(a_) ({ float64 _a = (a_); float64 _q = v_reciprocal_f32(_a); _q + _q * (1.0f - _a * _q); })
// tanh(d) to f32 accuracy: (1 - t) / (1 + t) with t = e^(-2 |d|) for |d| >= 1/64, the odd series below that; the sign of d
#define TANH(d_) ({ float64 _d = (d_); float64 _a = v_f32_abs_b(_d, 0, 0, 1, 0); float64 _t = EXPD(_a * -2.0f); \
    float64 _big = (1.0f - _t) * RECIP(_t + 1.0f); float64 _d2 = _a * _a; \
    float64 _small = _a * ((_d2 * 0.13333333333f - 0.33333333333f) * _d2 + 1.0f); \
    float64 _m = v_f32_sel_less_f32_b(_a, 0.015625f, _small, _big, 0, 0, 1, 0); \
    v_f32_sel_less_f32_b(_d, 0.0f, _m * -1.0f, _m, 0, 0, 1, 0); })
// the bf16 value nearest to each lane of (a, b), back in f32: a rounding point of the reference
#define RBF(a, b) ({ float128 _p; _p.v1 = (a); _p.v2 = (b); v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(_p)); })
// the factor of an L2 clip at `thr` for a squared norm n2 (every lane holds the head's sum): min(1, thr / (|x| + 1e-6))
#define CLIPF(n2_, thr_) ({ float64 _n2 = v_f32_max_b((n2_), 1.0e-30f); float64 _nr = _n2 * v_rsqrt_f32(_n2); \
    v_f32_min_b((float64)(thr_) * RECIP(_nr + 1.0e-6f), 1.0f); })
// row `pin` of a block of 128 key rows as the V caches hold it: the softmax kernels' converting store of P pairs key m of the first
// 64 with key 64 + m, so the value rows are stored in that interleaved order
#define VPERM(j) (((j) & ~127) + ((((j) & 127) < 64) ? 2 * ((j) & 127) : 2 * (((j) & 127) - 64) + 1))
