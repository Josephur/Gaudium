// gaudium_qsa_gather: gathered sparse attention core of QSA (Qwen3.8-Flash-Next, docs/notes/flash-next-qsa-sparse.md), step 1 of 4:
// copy each query row's attended keys / values (its selected 4-token blocks + its tail) out of the layer's caches into a fixed-size,
// per-row buffer the MME can read with plain batch GEMMs.
//  in0 qb bf16 {256, GQ*T, NKV}: the pass's q block (attn_prep, one chunk: row hl*T + t of batch g)
//  in1 kcache bf16 {256, C, NKV*nch}, in2 vcache bf16 {256, C, NKV*nch}: the layer's caches, chunk-major (position pos of group g at
//      {pos % C, (pos / C) * NKV + g}); V rows permuted inside 128-row blocks (VPERM, attn_prep)
//  in3 qbm int32 {64, NBW, T}: the selected-block bitmap of gaudium_qsa_select -DBITMAP (lane i / 32 + i of vector s: blocks
//      2048 s + 64 i + k / 2048 s + 64 i + 32 + k, bit k)
//  in4 cursor int32 {1}
//  out0 q2 bf16 {256, GQ, NKV*T}: q of row t, head hl, group g at {., hl, g*T + t} (the GEMM's batch is one (row, group))
//  out1 kg bf16 {256, G, NKV*T}, out2 vg bf16 {256, G, NKV*T}: SLOTS. Row t at p = cursor + coff + t sees nvis = p + 1 positions: nb = nvis / 4
//      complete blocks, a tail of ntail = nvis - 4 nb (< 4). nsel = min(nb, nblk).
//        slots [0, 4 nsel): the attended blocks in ascending block order (sparse row, nb > nblk: the bitmap's nblk blocks; dense row:
//                           blocks 0 .. nb - 1), slot 4r + i = position 4 b_r + i
//        slots [4 nsel, 4 nblk): zero (dense rows only)
//        slots [4 nblk, 4 nblk + ntail): the tail positions 4 nb + i;  slots [4 nblk + ntail, G): zero
//      kg rows in slot order; vg rows permuted like the V cache (VPERM inside 128-slot blocks: G % 128 == 0) so the softmax's
//      interleaving bf16 conversion of P pairs with them. Zero slots are written every launch (no stale data reaches the PV GEMM).
//  scalars: T, coff, nblk, NS, lc (log2 C).   Index space {T, NS}: dim1 slices the nblk + 1 slot blocks (rank ranges) so a decode
//  step with one or three rows still spreads over the TPCs; a point copies q (slice 0), K and V of every kv group for its ranks, so the
//  row's bitmap is scanned once per slice (not once per kv group and K / V: P26, the scan was the gather's cost at long context).
//  The bitmap is read with scalar loads (~0.1 us each when issued one after another, P26): words are loaded 16 at a time, independent
//  loads issued back to back before any is used, and a batch whose popcount ends before the slice's first rank is skipped whole.
#define VPERM(j) (((j) & ~127) + ((((j) & 127) < 64) ? 2 * ((j) & 127) : 2 * (((j) & 127) - 64) + 1))

void main(tensor qb, tensor kcache, tensor vcache, tensor qbm, tensor cursor, tensor q2, tensor kg, tensor vg,
          int T, int coff, int nblk, int NS, int lc)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const int NKV = get_dim_size(qb, 2), GQ = get_dim_size(q2, 1), G = get_dim_size(kg, 1);
    const int RS = (nblk + 1 + NS - 1) / NS;   // slot blocks (ranks) per slice
    int5 zc = {0, 0, 0, 0, 0};
    const int cur = s_i32_ld_g(gen_addr(zc, cursor)) + coff;
    const bfloat128 zb = 0;
    for (int t = idx[0]; t < idx[0] + isz[0]; ++t) {
        const int p = cur + t, nb = (p + 1) >> 2, ntail = p + 1 - 4 * nb;
        const int nsel = nb < nblk ? nb : nblk;
        for (int s = idx[1]; s < idx[1] + isz[1]; ++s) {
            const int r0 = s * RS, r1 = (r0 + RS) < (nblk + 1) ? (r0 + RS) : (nblk + 1);
            if (r0 >= r1) continue;
            if (s == 0)   // q of this row: GQ heads of every group, rows hl*T + t of batch g -> {., hl, g*T + t}
                for (int g = 0; g < NKV; ++g) for (int h = 0; h < GQ; ++h) {
                    int5 a = {0, h * T + t, g, 0, 0}, o = {0, h, g * T + t, 0, 0};
                    bfloat128 x0 = v_bf16_ld_tnsr_b(a, qb); a[0] = 128; bfloat128 x1 = v_bf16_ld_tnsr_b(a, qb);
                    v_bf16_st_tnsr(o, q2, x0); o[0] = 128; v_bf16_st_tnsr(o, q2, x1);
                }
            // one attended block b at rank r: K rows (4 consecutive cache rows) and V rows (VPERM'd) of every kv group
#define EMIT(b_, r_) do { const int _q = 4 * (b_), _ch = _q >> lc, _pin = _q - (_ch << lc), _d = 4 * (r_); \
        for (int _g = 0; _g < NKV; ++_g) { const int _bat = _ch * NKV + _g, _bt = _g * T + t; \
            int5 _a = {0, _pin, _bat, 0, 0}; \
            bfloat128 _k0 = v_bf16_ld_tnsr_b(_a, kcache); _a[0] = 128; bfloat128 _k1 = v_bf16_ld_tnsr_b(_a, kcache); _a[0] = 0; _a[1] = _pin + 1; \
            bfloat128 _k2 = v_bf16_ld_tnsr_b(_a, kcache); _a[0] = 128; bfloat128 _k3 = v_bf16_ld_tnsr_b(_a, kcache); _a[0] = 0; _a[1] = _pin + 2; \
            bfloat128 _k4 = v_bf16_ld_tnsr_b(_a, kcache); _a[0] = 128; bfloat128 _k5 = v_bf16_ld_tnsr_b(_a, kcache); _a[0] = 0; _a[1] = _pin + 3; \
            bfloat128 _k6 = v_bf16_ld_tnsr_b(_a, kcache); _a[0] = 128; bfloat128 _k7 = v_bf16_ld_tnsr_b(_a, kcache); \
            int5 _v = {0, VPERM(_pin), _bat, 0, 0}; \
            bfloat128 _v0 = v_bf16_ld_tnsr_b(_v, vcache); _v[0] = 128; bfloat128 _v1 = v_bf16_ld_tnsr_b(_v, vcache); _v[0] = 0; _v[1] = VPERM(_pin + 1); \
            bfloat128 _v2 = v_bf16_ld_tnsr_b(_v, vcache); _v[0] = 128; bfloat128 _v3 = v_bf16_ld_tnsr_b(_v, vcache); _v[0] = 0; _v[1] = VPERM(_pin + 2); \
            bfloat128 _v4 = v_bf16_ld_tnsr_b(_v, vcache); _v[0] = 128; bfloat128 _v5 = v_bf16_ld_tnsr_b(_v, vcache); _v[0] = 0; _v[1] = VPERM(_pin + 3); \
            bfloat128 _v6 = v_bf16_ld_tnsr_b(_v, vcache); _v[0] = 128; bfloat128 _v7 = v_bf16_ld_tnsr_b(_v, vcache); \
            int5 _o = {0, _d, _bt, 0, 0}; v_bf16_st_tnsr(_o, kg, _k0); _o[0] = 128; v_bf16_st_tnsr(_o, kg, _k1); _o[0] = 0; _o[1] += 1; \
            v_bf16_st_tnsr(_o, kg, _k2); _o[0] = 128; v_bf16_st_tnsr(_o, kg, _k3); _o[0] = 0; _o[1] += 1; \
            v_bf16_st_tnsr(_o, kg, _k4); _o[0] = 128; v_bf16_st_tnsr(_o, kg, _k5); _o[0] = 0; _o[1] += 1; \
            v_bf16_st_tnsr(_o, kg, _k6); _o[0] = 128; v_bf16_st_tnsr(_o, kg, _k7); \
            int5 _w = {0, VPERM(_d), _bt, 0, 0}; v_bf16_st_tnsr(_w, vg, _v0); _w[0] = 128; v_bf16_st_tnsr(_w, vg, _v1); \
            _w[0] = 0; _w[1] = VPERM(_d + 1); v_bf16_st_tnsr(_w, vg, _v2); _w[0] = 128; v_bf16_st_tnsr(_w, vg, _v3); \
            _w[0] = 0; _w[1] = VPERM(_d + 2); v_bf16_st_tnsr(_w, vg, _v4); _w[0] = 128; v_bf16_st_tnsr(_w, vg, _v5); \
            _w[0] = 0; _w[1] = VPERM(_d + 3); v_bf16_st_tnsr(_w, vg, _v6); _w[0] = 128; v_bf16_st_tnsr(_w, vg, _v7); } } while (0)
#define ZSLOT(j_) do { const int _j = (j_); for (int _g = 0; _g < NKV; ++_g) { const int _bt = _g * T + t; \
        int5 _o = {0, _j, _bt, 0, 0}; v_bf16_st_tnsr(_o, kg, zb); _o[0] = 128; v_bf16_st_tnsr(_o, kg, zb); \
        int5 _w = {0, VPERM(_j), _bt, 0, 0}; v_bf16_st_tnsr(_w, vg, zb); _w[0] = 128; v_bf16_st_tnsr(_w, vg, zb); } } while (0)
            const int re = r1 < nsel ? r1 : nsel;   // ranks [r0, re) are attended blocks
            if (r0 < re) {
                if (nb <= nblk) { for (int r = r0; r < re; ++r) EMIT(r, r); }
                else {
                    const int ngs = (nb + 2047) >> 11;
                    for (int gs = 0; gs < ngs; ++gs) { int5 pc0 = {0, gs, t, 0, 0}; prefetch((__global__ void*)gen_addr(pc0, qbm)); pc0[0] = 32; prefetch((__global__ void*)gen_addr(pc0, qbm)); }
                    int cnt = 0;
                    for (int gs = 0; gs < ngs && cnt < re; ++gs) {
                        for (int ib = 0; ib < 32 && cnt < re; ib += 8) {
                            // 16 words = blocks 2048 gs + 64 ib .. + 511 in ascending order: lane ib + u (even), 32 + ib + u (odd)
                            unsigned w[16];
#pragma unroll
                            for (int u = 0; u < 8; ++u) { int5 wa = {ib + u, gs, t, 0, 0}, wb = {32 + ib + u, gs, t, 0, 0};
                                w[2 * u] = s_u32_ld_g(gen_addr(wa, qbm)); w[2 * u + 1] = s_u32_ld_g(gen_addr(wb, qbm)); }
                            int pcs = 0;
#pragma unroll
                            for (int u = 0; u < 16; ++u) pcs += (int)s_u32_popcnt(w[u], SW_COUNT_ONES);
                            if (cnt + pcs <= r0) { cnt += pcs; continue; }
                            for (int u = 0; u < 16 && cnt < re; ++u) {
                                unsigned x = w[u];
                                const int pc = (int)s_u32_popcnt(x, SW_COUNT_ONES);
                                if (cnt + pc <= r0) { cnt += pc; continue; }
                                const int base = 2048 * gs + 64 * (ib + (u >> 1)) + 32 * (u & 1);
                                while (x != 0 && cnt < re) {
                                    const int k = (int)s_u32_find_first(x, SW_FIND_ONE | SW_LSB);
                                    x = x & (x - 1u);
                                    if (cnt >= r0) EMIT(base + k, cnt);
                                    ++cnt;
                                }
                            }
                        }
                    }
                }
            }
            // zero slot blocks of the dense rows' unused ranks
            { const int z0 = r0 > nsel ? r0 : nsel, z1 = r1 < nblk ? r1 : nblk;
              for (int r = z0; r < z1; ++r) { ZSLOT(4 * r); ZSLOT(4 * r + 1); ZSLOT(4 * r + 2); ZSLOT(4 * r + 3); } }
            // the tail (rank nblk) and the pad slots
            if (r1 == nblk + 1) {
                for (int i = 0; i < 4; ++i) {
                    const int j = 4 * nblk + i;
                    if (i < ntail) {
                        const int q = 4 * nb + i, ch = q >> lc, pin = q - (ch << lc);
                        for (int g = 0; g < NKV; ++g) { const int bat = ch * NKV + g, bt = g * T + t;
                            int5 a = {0, pin, bat, 0, 0}, o = {0, j, bt, 0, 0};
                            bfloat128 x0 = v_bf16_ld_tnsr_b(a, kcache); a[0] = 128; bfloat128 x1 = v_bf16_ld_tnsr_b(a, kcache);
                            v_bf16_st_tnsr(o, kg, x0); o[0] = 128; v_bf16_st_tnsr(o, kg, x1);
                            int5 av = {0, VPERM(pin), bat, 0, 0}, ov = {0, VPERM(j), bt, 0, 0};
                            bfloat128 y0 = v_bf16_ld_tnsr_b(av, vcache); av[0] = 128; bfloat128 y1 = v_bf16_ld_tnsr_b(av, vcache);
                            v_bf16_st_tnsr(ov, vg, y0); ov[0] = 128; v_bf16_st_tnsr(ov, vg, y1); }
                    } else ZSLOT(j);
                }
                for (int j = 4 * nblk + 4; j < G; ++j) ZSLOT(j);
            }
        }
    }
}
