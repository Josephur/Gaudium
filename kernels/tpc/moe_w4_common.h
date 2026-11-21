// moe_w4_common.h: shared pieces of the W4L / W8 routed-expert kernels (moe_w4_*_gaudi2.c, docs/notes/moe-w4l.md).
// Every f32 formula here is the one of src/engine/host/moe_w4.hpp (w4::ref), operation for operation: separate multiplies and adds
// (build with -ffp-contract=off), no special-function unit, no FMA. The TPC flushes f32 denormals (results and inputs); the host
// twin runs with FTZ + DAZ.
#pragma once
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_F(a, s0, s1, s2, s3) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (a))
#define MOVDG_I(a, s0, s1, s2, s3) v_i32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (a))
#define W4_L2E 1.44269504088896341f
#define W4_C1 0.693359375f
#define W4_C2 2.12194440e-4f
#define W4_AMIN 7.88860905e-31f      // 2^-100
#define W4_INV127 7.87401572e-3f     // f32(1/127)

// f32 shuffle pattern: the byte (elem & 7) | (group << 5) | 0x80 of source lane src (0..15) of the 16-lane dual group, in all 4 bytes
// of the f32 lane (measured: libtpcsim, uniform patterns). Built in the i32 domain from the 4-byte lane id: per-lane shifts of uchar256
// vectors gave wrong patterns (bits crossing byte lanes), so no u8 arithmetic here. FPAT is for scalars.
#define FPAT(src) ((((src) & 7) | (((src) >> 3) << 5)) | 0x80)
// i32 lane value v (0..255) replicated into the 4 bytes of the lane
#define REP4(v) ({ int64 _v = (v); uint64 _u = *(uint64*)&_v; _u = _u | (_u << 8); _u = _u | (_u << 16); *(uchar256*)&_u; })
// pattern vector selecting source lane src (int64, per lane, 0..15); en: int64 0 or 1 (0 = keep the income)
#define FPATV(src, en) REP4((((src) & 7) | (((src) >> 3) << 5)) | ((en) << 7))
static inline int64 lane16() { uint64 l = read_lane_id_4b_b(); int64 li = *(int64*)&l; return li & 15; }
static inline uchar256 xor_pat16(int s) { int64 one = 1; return FPATV(lane16() ^ s, one); }
// NOTE (measured in libtpcsim, 2026-11-13): v_f32_shuffle with a NON-uniform pattern that is not an xor of the lane (e.g. lane + r)
// does not follow the (elem, group) model: the source group flips depending on the destination lane. Uniform patterns and xor patterns
// (the router's butterflies) are fine. Lane rearrangements are therefore done on int8 data with v_u8_shuffle (non-uniform patterns
// verified: the LUT lookups) after a dual-group move.
#define MOVDG_B(a, s0, s1, s2, s3) ({ char256 _c = (a); int64 _i = *(int64*)&_c; _i = MOVDG_I(_i, s0, s1, s2, s3); *(char256*)&_i; })
// max over each 16-lane dual group, broadcast to its 16 lanes (xor butterfly)
#define DGMAX(a, P8, P4, P2, P1) ({ float64 _a = (a); \
    _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P8, 0, _a)); _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P4, 0, _a)); \
    _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P2, 0, _a)); _a = v_f32_max_b(_a, v_f32_shuffle_b(_a, P1, 0, _a)); _a; })
// 1/d for positive normal d: magic estimate + 3 Newton steps (w4::ref::recip)
#define RECIP(d_) ({ float64 _d = (d_); int64 _ri = (int64)0x7EF311C3 - *(int64*)&_d; float64 _r = *(float64*)&_ri; float64 _t; \
    _t = _d * _r; _t = 2.0f - _t; _r = _r * _t; _t = _d * _r; _t = 2.0f - _t; _r = _r * _t; _t = _d * _r; _t = 2.0f - _t; _r = _r * _t; _r; })
// e^x for x in [-87, 80] (w4::ref::expm)
#define EXPM(x_) ({ float64 _x = (x_); float64 _z = v_f32_nearbyint_b(_x * W4_L2E, SW_RHNE, 0, 1, 0); \
    float64 _r = _x - _z * W4_C1; _r = _r + _z * W4_C2; \
    float64 _p = _r * 1.3836704893037677e-3f; _p = _p + 8.374780416488647e-3f; \
    _p = _p * _r; _p = _p + 4.166822507977486e-2f; _p = _p * _r; _p = _p + 0.1666642129421234f; _p = _p * _r; _p = _p + 0.49999991059303284f; \
    _p = _p * _r; _p = _p + 1.0f; _p = _p * _r; _p = _p + 1.0f; \
    int64 _n = v_i32_shl_b(v_convert_f32_to_i32_b(_z, SW_RHNE, 0, 1, 0), 23, 0, 0, 1, 0); \
    int64 _b = *(int64*)&_p + _n; *(float64*)&_b; })
// silu(g) * u (w4::ref::swiglu)
#define SWIGLU(g_, u_) ({ float64 _g = (g_); float64 _nx = -_g; _nx = v_f32_max_b(_nx, -87.0f); _nx = v_f32_min_b(_nx, 80.0f); \
    float64 _e = EXPM(_nx); float64 _den = 1.0f + _e; float64 _rr = RECIP(_den); float64 _sg = _g * _rr; _sg * (u_); })
// Q8 of v given its block amax broadcast in a (any lane grouping): q (integral f32 in [-127, 127]) and the block scale d
#define Q8(v_, a_, q_, d_) { float64 _a = v_f32_max_b((a_), W4_AMIN); float64 _inv = RECIP(_a) * 127.0f; float64 _t = (v_) * _inv; \
    _t = v_f32_nearbyint_b(_t, SW_RHNE, 0, 1, 0); _t = v_f32_max_b(_t, -127.0f); q_ = v_f32_min_b(_t, 127.0f); d_ = _a * W4_INV127; }
