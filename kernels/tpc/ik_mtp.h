// ik_mtp.h: pieces shared by the kernels of the Inkling 2 block drafter (ik_d*_gaudi2.c, ik_m*_gaudi2.c), after ik_common.h.
//
// Rotary positions, half-split form over the 128 channels of a head: channel c < 64 and channel c + 64 turn together by the angle
// pos f_c (f_c = theta^(-2 c / 128)). A bf16 load of a head converts to v1 = the even channels and v2 = the odd ones; channel 2j of v1
// (j < 32) pairs with lane j + 32 of v1, and likewise in v2, so a lane's partner is the swap of the two halves of its vector.
// Tables (f32, built by the loader in f64), two levels so that they stay small for any context: hi {128, 2, NHI} holds cos / sin of
// (256 hi) f, lo {128, 2, 256} cos / sin of lo f, for pos = 256 hi + lo. Lane L < 64 of a column serves lane L of v1 (frequency index
// 2 (L % 32)), lane 64 + L serves lane L of v2 (frequency index 2 (L % 32) + 1). The sum formulas give cos and sin of the angle; the
// sine is negated in lanes 0 .. 31 (the first channel of a pair takes -sin).
#define IK_TAB(tab, ch, which, col) ({ int5 _c = {(ch), (which), (col), 0, 0}; v_f32_ld_tnsr_b(_c, tab); })
#define IK_ROPE_HALF(x_, hi_t, lo_t, H_, L_, off_, lane_) ({ \
    const float64 _ch = IK_TAB(hi_t, off_, 0, H_), _sh = IK_TAB(hi_t, off_, 1, H_); \
    const float64 _cl = IK_TAB(lo_t, off_, 0, L_), _sl = IK_TAB(lo_t, off_, 1, L_); \
    const float64 _co = _ch * _cl - _sh * _sl; float64 _si = _sh * _cl + _ch * _sl; \
    _si = v_f32_sel_less_i32_b((lane_), 32, _si * -1.0f, _si); \
    const float64 _x = (x_); const float64 _sw = MOVDG_ALL(_x, 2, 3, 0, 1, _x); _x * _co + _sw * _si; })
// a head (v1, v2) turned to position pos
#define IK_ROPE(xv_, hi_t, lo_t, pos_, lane_) ({ const int _hh = (pos_) >> 8, _ll = (pos_) & 255; float128 _o; \
    _o.v1 = IK_ROPE_HALF((xv_).v1, hi_t, lo_t, _hh, _ll, 0, lane_); _o.v2 = IK_ROPE_HALF((xv_).v2, hi_t, lo_t, _hh, _ll, 64, lane_); _o; })
// a head (v1, v2) normalised over its 128 channels with a bf16 weight w {128}, rounded to bf16 (a rounding point of the reference)
#define IK_HEADNORM(xv_, w_, eps_) ({ const float128 _x = (xv_); \
    const float64 _n2 = REDUCE64(_x.v1 * _x.v1 + _x.v2 * _x.v2, P8, P4, P2, P1); \
    const float64 _r = v_rsqrt_f32(_n2 * (1.0f / 128.0f) + (eps_)); \
    int5 _wc = {0, 0, 0, 0, 0}; const float128 _w = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(_wc, w_)); \
    RBF(_x.v1 * _r * _w.v1, _x.v2 * _r * _w.v2); })
// a bf16 head row {128} at (c0, c1, c2) of a tensor, converted
#define IK_LDH(t_, c0_, c1_, c2_) ({ int5 _c = {(c0_), (c1_), (c2_), 0, 0}; v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(_c, t_)); })
#define IK_STH(t_, c0_, c1_, c2_, v_) { int5 _c = {(c0_), (c1_), (c2_), 0, 0}; v_bf16_st_tnsr(_c, t_, v_convert_f32_to_bf16_all_b(v_)); }
