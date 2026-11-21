// samp_noise.h: the on-card sampler's position-keyed Gumbel noise and its exp / log polynomials, shared by the sampler kernels
// (samp_gaudi2.c) and DSpine's coupled walk (ds_cwalk_gaudi2.c), so a coupled draft draws exactly the noise its verify row will.
// CPU twin: src/engine/samp_ref.hpp (fmix32 / row_key / elem_hash / u01 / mlog / mexp / gumbel), operation for operation;
// build with -ffp-contract=off semantics (no FMA: the expressions below never form a*b+c in one instruction).
#pragma once
static inline unsigned s_fmix32(unsigned h) { h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16; return h; }
static inline unsigned row_key(unsigned slo, unsigned shi, unsigned n) { return s_fmix32(s_fmix32(slo ^ s_fmix32(shi ^ 0x5bd1e995u)) + n * 0x9E3779B9u); }

// ---- vector math, twins of samp_ref.hpp mlog / mexp ----
#define VLOG(y) ({ float64 _y = (y); int64 _b = as_int64(_y); int64 _e = (_b >> 23) - 127; int64 _mb = (_b & 0x7FFFFF) | 0x3F800000; \
    int64 _mh = _mb - 0x00800000; int64 _eh = _e + 1; \
    _mb = v_i32_sel_grt_i32_b(_mb, 0x3FB504F3, _mh, _mb); _e = v_i32_sel_grt_i32_b(_b & 0x7FFFFF | 0x3F800000, 0x3FB504F3, _eh, _e); \
    float64 _t = as_float64(_mb) - 1.0f; \
    float64 _q = 0.08743945509195328f; \
    _q = _q * _t; _q = _q + -0.14377330243587494f; _q = _q * _t; _q = _q + 0.14949095249176025f; _q = _q * _t; _q = _q + -0.16560696065425873f; \
    _q = _q * _t; _q = _q + 0.19956977665424347f; _q = _q * _t; _q = _q + -0.2500215470790863f; _q = _q * _t; _q = _q + 0.3333418369293213f; \
    _q = _q * _t; _q = _q + -0.49999988079071045f; _q = _q * _t; _q = _q + 1.0f; \
    float64 _p = _q * _t; float64 _ef = v_convert_i32_to_f32_b(_e, SW_RHNE); float64 _r = _ef * 0.693147182464599609375f; _r + _p; })
#define VEXP(x) ({ float64 _x = (x); float64 _y = _x * 1.44269502162933349609375f; float64 _n = _y + 12582912.0f; _n = _n - 12582912.0f; \
    float64 _r = _n * 0.693145751953125f; _r = _x - _r; float64 _r2 = _n * 1.42860682030941723212e-06f; _r = _r - _r2; \
    float64 _q = 0.0013941110810264945f; _q = _q * _r; _q = _q + 0.008369150571525097f; _q = _q * _r; _q = _q + 0.04166635125875473f; \
    _q = _q * _r; _q = _q + 0.16666504740715027f; _q = _q * _r; _q = _q + 0.5f; _q = _q * _r; _q = _q + 1.0f; \
    float64 _p = _q * _r; _p = _p + 1.0f; int64 _ni = v_convert_f32_to_i32_b(_n, SW_RHNE); float64 _sc = as_float64((_ni + 127) << 23); _p * _sc; })
#define VFMIX(h) ({ uint64 _h = (h); _h = _h ^ (_h >> 16); _h = v_u32_mul_b(_h, 0x85ebca6bu).v1; _h = _h ^ (_h >> 13); \
    _h = v_u32_mul_b(_h, 0xc2b2ae35u).v1; _h = _h ^ (_h >> 16); _h; })
// g(rk, v) for a vector of token ids iv (int64): u = ((h >> 9) + 0.5) 2^-23, g = -log(-log(u))
#define VGUMBEL(rk, iv) ({ uint64 _hh = VFMIX(((uint64)(rk)) ^ v_u32_mul_b(as_uint64(iv), 0x27D4EB2Fu).v1); \
    float64 _u = v_convert_i32_to_f32_b(as_int64(_hh >> 9), SW_RHNE) + 0.5f; _u = _u * 1.1920928955078125e-07f; \
    float64 _E = VLOG(_u); _E = -_E; float64 _g = VLOG(_E); _g = -_g; _g; })
