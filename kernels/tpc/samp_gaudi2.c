// Gaudium on-card sampler (Gumbel-max over an exactly searched top-k / top-p / min-p support). One source, six kernels
// selected by -D; the CPU mirror is tools/probes/samp_ref.hpp (read it first: it states the math and the support).
// Every float operation below has a twin in samp_ref.hpp in the same order (no FMA: build with -ffp-contract=off).
//   SAMP_BMAX   gaudium_samp_bmax    {NB, R}  logits bf16 {V, RL}, rsel {1}                    -> bm f32 {64, NB, R} (block max, all lanes)
//   SAMP_INIT   gaudium_samp_init    {R}      bm                                               -> st f32 {64, 8, R}   (field 4 = m)
//   SAMP_PART   gaudium_samp_part    {NB, R}  logits, rsel, spf, spi, st_prev, bm              -> part f32 {64, NB, R} (lanes 0..15)
//   SAMP_RED    gaudium_samp_red     {R}      part, spf, spi, st_prev                          -> st f32 {64, 8, R}
//   SAMP_GUMBEL gaudium_samp_gumbel  {NB, R}  logits, rsel, base, spf, spi, st_final, bm       -> gp f32 {64, 2, NB, R}
//   SAMP_FINAL  gaudium_samp_final   {R}      gp, rsel                                         -> tokens i32 {1, RL}  (column rsel + r)
// Row r of a node reads logits row rsel[0] + r. Positions: n = base[0] + off + rsel[0] + r (the sequence index of the
// token the row predicts). State fields (f32, all lanes equal): 0 lo key, 1 hi key, 2 done, 3 top-p target, 4 m.
// scalars (glue): PART/RED: iter, role; GUMBEL: off, role, searched (0 = no search rounds in this recipe).   role 0 = target row (verify), 1 = draft row (MTP).
#include "samp_noise.h"   // s_fmix32 / row_key / VLOG / VEXP / VFMIX / VGUMBEL (shared with ds_cwalk_gaudi2.c)
#define BLK 1024
#define NTH 16
#define NIT 4
#define NEGF (-1e30f)
#define PINF 3.0e38f
#define F_TOPP 1
#define F_MINP 2
#define F_COUPLED 4
#define F_DTRUNC 8
#define F_GREEDY 16   // issue #11: a greedy lane in a sampled recipe: no noise, no cuts, support { l == m }, smallest id (= argmax_fwd_bf16)

#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define RED_OP(OPX, a, P8, P4, P2, P1) ({ float64 _a = (a); float64 _r; \
    _r = MOVDG_ALL(_a, 1, 2, 3, 0, _a); _a = OPX(_a, _r); _r = MOVDG_ALL(_a, 2, 3, 0, 1, _a); _a = OPX(_a, _r); \
    _a = OPX(_a, v_f32_shuffle_b(_a, P8, 0, _a)); _a = OPX(_a, v_f32_shuffle_b(_a, P4, 0, _a)); \
    _a = OPX(_a, v_f32_shuffle_b(_a, P2, 0, _a)); _a = OPX(_a, v_f32_shuffle_b(_a, P1, 0, _a)); _a; })
#define ADDX(x, y) ((x) + (y))
#define MAXX(x, y) v_f32_max_b((x), (y))
#define MINX(x, y) v_f32_min_b((x), (y))
#define REDUCE_SUM(a) RED_OP(ADDX, a, P8, P4, P2, P1)
#define REDUCE_MAX(a) RED_OP(MAXX, a, P8, P4, P2, P1)
#define REDUCE_MIN(a) RED_OP(MINX, a, P8, P4, P2, P1)
#define LANE_PATS const uchar256 lid8 = read_lane_id_1b_b(); const uchar256 l16 = (lid8 >> 2) & 15; \
    const uchar256 P8 = xor_pat(l16, 8), P4 = xor_pat(l16, 4), P2 = xor_pat(l16, 2), P1 = xor_pat(l16, 1);
static inline uchar256 xor_pat(uchar256 l16, int s) { uchar256 src = l16 ^ (unsigned char)s; return ((src & 7) | ((src >> 3) << 5)) | 0x80; }

// ---- scalar helpers (bf16 order keys) ----
typedef union { float f; int i; unsigned u; } fu_t;
static inline unsigned s_fbits(float f) { fu_t x; x.f = f; return x.u; }
static inline float s_bitsf(unsigned u) { fu_t x; x.u = u; return x.f; }
static inline int key_of_bits(unsigned b) { return (b & 0x8000) ? (int)((~b) & 0xFFFF) : (int)(b | 0x8000); }
static inline unsigned bits_of_key(int k) { return (k & 0x8000) ? (unsigned)(k & 0x7FFF) : (unsigned)((~k) & 0xFFFF); }
static inline float val_of_key(int k) { return s_bitsf(bits_of_key(k) << 16); }
static inline int key_of(float f) { return key_of_bits(s_fbits(f) >> 16); }
static inline int key_ceil(float f) { int k = key_of(f); if (val_of_key(k) < f) k = k + 1; return k; }
static inline float ldf(tensor t, int a, int b, int c, int d) { int5 x = {a, b, c, d, 0}; return s_f32_ld_g(gen_addr(x, t)); }
static inline int ldi(tensor t, int a) { int5 x = {a, 0, 0, 0, 0}; return s_i32_ld_g(gen_addr(x, t)); }

// bracket of search round `it` for one row (shared by PART and RED); returns done
#define BRACKET(it, st, r, m, spf, spi, tact, lo, hi, done) do { \
    if ((it) == 0) { lo = key_ceil(m - ldf(spf, 1, 0, 0, 0)); hi = key_of(m); done = !(ldi(spi, 0) > 0) || !(tact); } \
    else if ((it) == NIT) { lo = (int)ldf(st, 0, 0, r, 0); hi = key_of(m); done = !(ldi(spi, 3) & F_TOPP) || !(tact); } \
    else { lo = (int)ldf(st, 0, 0, r, 0); hi = (int)ldf(st, 0, 1, r, 0); done = ldf(st, 0, 2, r, 0) != 0.0f; } } while (0)
// fl = spi[3], loaded once per kernel (a scalar global load costs ~0.2 us on the card)
#define TACTIVE(role, fl) (!((fl) & F_GREEDY) && ((role) == 0 || (((fl) & F_COUPLED) && ((fl) & F_DTRUNC))))

#if defined(SAMP_BMAX)
void main(tensor logits, tensor rsel, tensor bm)
{
    const int5 idx = get_index_space_offset(), isz = get_index_space_size();
    LANE_PATS
    const int rs = ldi(rsel, 0); const int V = get_dim_size(logits, 0);
    for (int r = idx[1]; r < idx[1] + isz[1]; ++r)
        for (int b = idx[0]; b < idx[0] + isz[0]; ++b) {
            int nv = (V - b * BLK) / 128; if (nv > 8) nv = 8;
            int5 c = {b * BLK, rs + r, 0, 0, 0};
            bfloat128 mx = v_bf16_ld_tnsr_b(c, logits);
            for (int q = 1; q < nv; ++q) { c[0] = b * BLK + 128 * q; mx = v_bf16_max_b(mx, v_bf16_ld_tnsr_b(c, logits)); }
            float128 f = v_convert_bf16_to_f32_all_b(mx);
            float64 m = REDUCE_MAX(v_f32_max_b(f.v1, f.v2));
            int5 o = {0, b, r, 0, 0}; v_f32_st_tnsr(o, bm, m);
        }
}
#elif defined(SAMP_INIT)
void main(tensor bm, tensor st)
{
    const int5 idx = get_index_space_offset(), isz = get_index_space_size();
    const int NB = get_dim_size(bm, 1);
    for (int r = idx[0]; r < idx[0] + isz[0]; ++r) {
        float64 m = NEGF;
        for (int b = 0; b < NB; ++b) { int5 c = {0, b, r, 0, 0}; m = v_f32_max_b(m, v_f32_ld_tnsr_b(c, bm)); }
        float64 z = 0.0f;
        for (int f = 0; f < 8; ++f) { int5 o = {0, f, r, 0, 0}; v_f32_st_tnsr(o, st, f == 4 ? m : z); }
    }
}
#elif defined(SAMP_PART)
void main(tensor logits, tensor rsel, tensor spf, tensor spi, tensor st, tensor bm, tensor part, int it, int role)
{
    const int5 idx = get_index_space_offset(), isz = get_index_space_size();
    const int64 lane = (int64)read_lane_id_4b_b();
    LANE_PATS
    const int rs = ldi(rsel, 0); const int V = get_dim_size(logits, 0);
    const float invT = ldf(spf, 0, 0, 0, 0);
    const int fl = ldi(spi, 3); const int tact = TACTIVE(role, fl);
    const int phaseB = it >= NIT;
    for (int r = idx[1]; r < idx[1] + isz[1]; ++r) {
        const float m = ldf(st, 0, 4, r, 0);
        int lo, hi, done; BRACKET(it, st, r, m, spf, spi, tact, lo, hi, done);
        float tv[NTH];
        { const int n = hi - lo + 1; const int s = (n + NTH - 1) / NTH;
          for (int j = 0; j < NTH; ++j) { if (n <= NTH) tv[j] = j < n ? val_of_key(lo + j) : PINF; else { int t = lo + j * s; if (t > 65535) t = 65535; tv[j] = val_of_key(t); } } }
        for (int b = idx[0]; b < idx[0] + isz[0]; ++b) {
            int5 o = {0, b, r, 0, 0};
            if (done || ldf(bm, 0, b, r, 0) < tv[0]) { float64 z = 0.0f; v_f32_st_tnsr(o, part, z); continue; }
            int nv = (V - b * BLK) / 128; if (nv > 8) nv = 8;
            float64 a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
            float64 a8 = 0.0f, a9 = 0.0f, a10 = 0.0f, a11 = 0.0f, a12 = 0.0f, a13 = 0.0f, a14 = 0.0f, a15 = 0.0f;
            for (int q = 0; q < nv; ++q) {
                int5 c = {b * BLK + 128 * q, rs + r, 0, 0, 0};
                float128 f = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c, logits));
                for (int h = 0; h < 2; ++h) {
                    float64 l = h ? f.v2 : f.v1;
                    float64 w = 1.0f;
                    if (phaseB) { float64 x = l - m; x = x * invT; x = v_f32_max_b(x, -30.0f); w = VEXP(x); }
#define ACC(j, a) a = a + v_f32_sel_geq_f32_b(l, tv[j], w, 0.0f);
                    ACC(0, a0) ACC(1, a1) ACC(2, a2) ACC(3, a3) ACC(4, a4) ACC(5, a5) ACC(6, a6) ACC(7, a7)
                    ACC(8, a8) ACC(9, a9) ACC(10, a10) ACC(11, a11) ACC(12, a12) ACC(13, a13) ACC(14, a14) ACC(15, a15)
                }
            }
            float64 out = 0.0f;
#define PUT(j, a) out = v_f32_sel_eq_i32_b(lane, j, REDUCE_SUM(a), out);
            PUT(0, a0) PUT(1, a1) PUT(2, a2) PUT(3, a3) PUT(4, a4) PUT(5, a5) PUT(6, a6) PUT(7, a7)
            PUT(8, a8) PUT(9, a9) PUT(10, a10) PUT(11, a11) PUT(12, a12) PUT(13, a13) PUT(14, a14) PUT(15, a15)
            v_f32_st_tnsr(o, part, out);
        }
    }
}
#elif defined(SAMP_RED)
void main(tensor part, tensor spf, tensor spi, tensor st, tensor st_out, int it, int role)
{
    const int5 idx = get_index_space_offset(), isz = get_index_space_size();
    const int64 lane = (int64)read_lane_id_4b_b();
    const float64 lanef = v_convert_i32_to_f32_b(lane, SW_RHNE);
    LANE_PATS
    const int NB = get_dim_size(part, 1);
    const int fl = ldi(spi, 3); const int tact = TACTIVE(role, fl);
    const int phaseB = it >= NIT;
    for (int r = idx[0]; r < idx[0] + isz[0]; ++r) {
        const float m = ldf(st, 0, 4, r, 0);
        int lo, hi, done; BRACKET(it, st, r, m, spf, spi, tact, lo, hi, done);
        float64 vlo = (float)lo, vhi = (float)hi, vdone = 1.0f, vtarget = phaseB && it > NIT ? ldf(st, 0, 3, r, 0) : 0.0f;
        if (!done) {
            const int n = hi - lo + 1; const int s = (n + NTH - 1) / NTH;
            float64 tot = 0.0f;
            for (int b = 0; b < NB; ++b) { int5 c = {0, b, r, 0, 0}; tot = tot + v_f32_ld_tnsr_b(c, part); }
            float64 thr;
            if (!phaseB) thr = (float)ldi(spi, 0);
            else if (it == NIT) { vtarget = ldf(spf, 2, 0, 0, 0) * REDUCE_SUM(v_f32_sel_eq_i32_b(lane, 0, tot, 0.0f)); thr = vtarget; }
            else thr = vtarget;
            float64 cnt = REDUCE_SUM(v_f32_sel_geq_f32_b(tot, thr, 1.0f, 0.0f));
            float64 J = cnt - 1.0f;
            float64 tvk = n <= NTH ? lanef + (float)lo : v_f32_min_b(lanef * (float)s + (float)lo, 65535.0f);
            float64 tJ = REDUCE_SUM(v_f32_sel_eq_f32_b(lanef, J, tvk, 0.0f));
            float64 tJ1 = REDUCE_SUM(v_f32_sel_eq_f32_b(lanef, J + 1.0f, tvk, 0.0f));
            float64 neg = v_f32_sel_less_f32_b(J, 0.0f, 1.0f, 0.0f);            // J < 0: |C| < k, the bracket start stands
            vlo = v_f32_sel_eq_f32_b(neg, 1.0f, (float)lo, tJ);
            if (n <= NTH) { vhi = vlo; vdone = 1.0f; }
            else {
                float64 h1 = v_f32_sel_less_f32_b(J, (float)(NTH - 1), tJ1 - 1.0f, (float)hi);
                vhi = v_f32_sel_eq_f32_b(neg, 1.0f, (float)lo, h1);
                vdone = v_f32_sel_eq_f32_b(vlo, vhi, 1.0f, 0.0f);
            }
            if (!phaseB) vtarget = 0.0f;
        }
        float64 vm = m, z = 0.0f;
        int5 o = {0, 0, r, 0, 0};
        v_f32_st_tnsr(o, st_out, vlo); o[1] = 1; v_f32_st_tnsr(o, st_out, vhi); o[1] = 2; v_f32_st_tnsr(o, st_out, vdone);
        o[1] = 3; v_f32_st_tnsr(o, st_out, vtarget); o[1] = 4; v_f32_st_tnsr(o, st_out, vm);
        o[1] = 5; v_f32_st_tnsr(o, st_out, z); o[1] = 6; v_f32_st_tnsr(o, st_out, z); o[1] = 7; v_f32_st_tnsr(o, st_out, z);
    }
}
#elif defined(SAMP_GUMBEL)
void main(tensor logits, tensor rsel, tensor base, tensor spf, tensor spi, tensor st, tensor bm, tensor gp, int off, int role, int searched)
{
    const int5 idx = get_index_space_offset(), isz = get_index_space_size();
    const int64 lane = (int64)read_lane_id_4b_b();
    LANE_PATS
    const int rs = ldi(rsel, 0); const int V = get_dim_size(logits, 0);
    const float invT = ldf(spf, 0, 0, 0, 0);
    const int flags = ldi(spi, 3);
    const int noise = !(flags & F_GREEDY) && (role == 0 || (flags & F_COUPLED));
    const int tact = TACTIVE(role, flags);
    const unsigned slo = (unsigned)ldi(spi, 1), shi = (unsigned)ldi(spi, 2);
    const int pos0 = ldi(base, 0) + off + rs;
    for (int r = idx[1]; r < idx[1] + isz[1]; ++r) {
        const float m = ldf(st, 0, 4, r, 0);
        int tau;   // searched = 0: the recipe has no search rounds (no top-k / top-p): the support is C, then min-p
        if (!noise) tau = key_of(m);
        else { tau = (tact && searched) ? (int)ldf(st, 0, 0, r, 0) : key_ceil(m - ldf(spf, 1, 0, 0, 0));
               if (tact && (flags & F_MINP)) { int km = key_ceil(m + ldf(spf, 3, 0, 0, 0)); if (km > tau) tau = km; } }
        const float thr = val_of_key(tau);
        const unsigned rk = row_key(slo, shi, (unsigned)(pos0 + r));
        for (int b = idx[0]; b < idx[0] + isz[0]; ++b) {
            float64 bz = NEGF, bi = PINF;
            if (!(ldf(bm, 0, b, r, 0) < thr)) {
                int nv = (V - b * BLK) / 128; if (nv > 8) nv = 8;
                for (int q = 0; q < nv; ++q) {
                    int5 c = {b * BLK + 128 * q, rs + r, 0, 0, 0};
                    float128 f = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c, logits));
                    const int64 ie = lane * 2 + (b * BLK + 128 * q);
                    for (int h = 0; h < 2; ++h) {
                        float64 l = h ? f.v2 : f.v1;
                        int64 iv = ie + h;
                        float64 x = l - m; x = x * invT;
                        float64 zz = x;
                        if (noise) {
                            float64 g = VGUMBEL(rk, iv);
                            zz = x + g;
                        }
                        zz = v_f32_sel_geq_f32_b(l, thr, zz, NEGF);
                        float64 ivf = v_convert_i32_to_f32_b(iv, SW_RHNE);
                        bi = v_f32_sel_grt_f32_b(zz, bz, ivf, bi);
                        bz = v_f32_max_b(zz, bz);
                    }
                }
            }
            float64 zm = REDUCE_MAX(bz);
            float64 im = REDUCE_MIN(v_f32_sel_eq_f32_b(bz, zm, bi, PINF));
            int5 o = {0, 0, b, r, 0}; v_f32_st_tnsr(o, gp, zm); o[1] = 1; v_f32_st_tnsr(o, gp, im);
        }
    }
}
#elif defined(SAMP_FINAL)
void main(tensor gp, tensor rsel, tensor tokens)
{
    const int5 idx = get_index_space_offset(), isz = get_index_space_size();
    const int rs = ldi(rsel, 0); const int NB = get_dim_size(gp, 2);
    for (int r = idx[0]; r < idx[0] + isz[0]; ++r) {
        float64 best = NEGF, bi = 0.0f;
        for (int b = 0; b < NB; ++b) {
            int5 c = {0, 0, b, r, 0}; float64 z = v_f32_ld_tnsr_b(c, gp); c[1] = 1; float64 i = v_f32_ld_tnsr_b(c, gp);
            bi = v_f32_sel_grt_f32_b(z, best, i, bi); best = v_f32_max_b(z, best);
        }
        int5 o = {0, rs + r, 0, 0, 0};
        v_i32_st_tnsr(o, tokens, v_convert_f32_to_i32_b(bi, SW_RHNE));
    }
}
#endif
