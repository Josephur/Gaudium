// gaudium_moe_plan: the work list of one routed-expert call (docs/notes/moe-w4l.md 1.2, 3). One index point (a scalar program:
// the TPC cannot turn vector lanes into addresses, so the lists the expert kernels read with scalar loads are built here).
//   in0 ids     i32 {K, T}   the router's ids (ascending per row)
//   in1 slot_of i32 {E}      device slot of each expert, -1 = on the host
//   out0 plan   i32 {1 + 5 K T}   [0] U = number of entries; entry j (ascending expert id, device experts only): [1 + 4j + 0] slot,
//                            [+1] n rows, [+2] start in the row list, [+3] expert id; row list at [1 + 4 K T + r] = t * 16 + k
//                            (each expert's rows in ascending (t, k) order)
//   out1 dev    i32 {K, T}   1 when ids[k, t] is on the card, 0 when on the host
//   scalars: K (<= 16), T, E (<= 512)
// Cost model (card, 2026-11-13): a scalar global access costs ~0.2 us and they do not overlap, so every id is loaded ONCE (cached in
// SLM), each selected expert's slot once, and all outputs are assembled in vector registers (one lane per value) and written with
// vector stores. KT > 1024 (prefill tiles above 102 rows at K = 10) uses SLM-free scalar code (slow, correct).
__local__ int CNT[512];
__local__ int FIL[512];
__local__ int SEL[512];
__local__ int IR[1024];       // the id of assignment a
__local__ int RW[1024];       // row-list entry r (t * 16 + k)
// decode fast path (T <= W4_PLAN_TMERGE rows): the heads of the T-way merge, the entries and the device flags
#define W4_PLAN_TMERGE 4
#define W4_PLAN_BIG 0x40000000
__local__ int HP[W4_PLAN_TMERGE];                   // row r: index k of its head
__local__ int HV[W4_PLAN_TMERGE];                   // row r: id at its head (W4_PLAN_BIG: the row is used up)
__local__ int EN[4 * 16 * W4_PLAN_TMERGE];          // entry j: slot, n, start, id
__local__ int DF[16 * W4_PLAN_TMERGE];              // assignment a = t K + k: 1 = on the card
#define LDI(tn, c0, c1) s_i32_ld_g(gen_addr(((int5){(c0), (c1), 0, 0, 0}), tn))
#define STI(tn, c0, c1, v) s_i32_st_g(gen_addr(((int5){(c0), (c1), 0, 0, 0}), tn), (v))
// a vector-assembled stream of i32 values into a 1-D tensor at ascending element indices
#define PUT(tn, idx, val) { const int _i = (idx); if ((_i >> 6) != cur) { if (cur >= 0) v_i32_st_tnsr(((int5){cur << 6, 0, 0, 0, 0}), tn, vec); cur = _i >> 6; } \
    vec = v_i32_sel_eq_i32_b(lane, _i & 63, (int64)(val), vec); }
#define FLUSH(tn) { if (cur >= 0) v_i32_st_tnsr(((int5){cur << 6, 0, 0, 0, 0}), tn, vec); cur = -1; }
void main(tensor ids, tensor slot_of, tensor plan, tensor dev, int K, int T, int E)
{
    const int KT = K * T;
    const uint64 lu = read_lane_id_4b_b(); const int64 lane = *(int64*)&lu;
    if (T <= W4_PLAN_TMERGE) {
        // DECODE (T = 1 + drafts): the router's ids are ascending per row, so the ascending expert list is a T-way merge of the rows:
        // no pass over the E experts (the two E-loops below are ~2/3 of this node's time at T = 1, and this node is one index point
        // on the critical path route -> plan -> gu). Same plan and flags as the general path, element for element.
        int k = 0, t = 0;
        for (int a = 0; a < KT; ++a) { IR[a] = LDI(ids, k, t); if (++k == K) { k = 0; ++t; } }
        for (int r = 0; r < T; ++r) { HP[r] = 0; HV[r] = IR[r * K]; }
        int U = 0, rs = 0;
        for (;;) {
            int v = W4_PLAN_BIG;
            for (int r = 0; r < T; ++r) { const int h = HV[r]; if (h < v) v = h; }
            if (v == W4_PLAN_BIG) break;
            const int s = LDI(slot_of, v, 0); const int on = s >= 0 ? 1 : 0; int n = 0;
            for (int r = 0; r < T; ++r) {
                if (HV[r] == v) {   // rows ascending: the expert's row list is in (t, k) order
                    const int kk = HP[r]; RW[rs + n] = r * 16 + kk; DF[r * K + kk] = on; ++n;
                    HP[r] = kk + 1; HV[r] = kk + 1 < K ? IR[r * K + kk + 1] : W4_PLAN_BIG;
                }
            }
            if (on) { EN[4 * U] = s; EN[4 * U + 1] = n; EN[4 * U + 2] = rs; EN[4 * U + 3] = v; rs += n; ++U; }   // a host expert's rows are overwritten
        }
        // one ascending vector stream: U, the entries, the row list (see the note on scalar stores below)
        int64 vec = 0; int cur = -1;
        PUT(plan, 0, U);
        const int ne = 4 * U;
        for (int i = 0; i < ne; ++i) PUT(plan, 1 + i, EN[i]);
        const int rbase = 1 + 4 * KT;
        for (int r = 0; r < rs; ++r) PUT(plan, rbase + r, RW[r]);
        FLUSH(plan);
        k = 0; t = 0; int64 dv = 0;
        for (int a = 0; a < KT; ++a) {
            dv = v_i32_sel_eq_i32_b(lane, k, (int64)DF[a], dv);
            if (++k == K) { v_i32_st_tnsr(((int5){0, t, 0, 0, 0}), dev, dv); k = 0; ++t; }
        }
        return;
    }
    for (int e = 0; e < E; ++e) CNT[e] = 0;
    const bool slm = KT <= 1024;
    // pass 1: ids (one scalar load each), counts
    int k = 0, t = 0;
    for (int a = 0; a < KT; ++a) {
        const int v = LDI(ids, k, t);
        if (slm) IR[a] = v;
        CNT[v] = CNT[v] + 1;
        if (++k == K) { k = 0; ++t; }
    }
    // pass 2: selected experts ascending, their slots (one load each), entries
    int ns = 0;
    for (int e = 0; e < E; ++e) if (CNT[e] > 0) { SEL[ns] = e; ++ns; }
    for (int i = 0; i < ns; ++i) FIL[SEL[i]] = LDI(slot_of, SEL[i], 0);
    int64 vec = 0; int cur = -1;
    // U first: element 0 goes out in the same vector stream as the entries (a scalar store to an element of a chunk that a vector store
    // also writes is not ordered against it on the card: the scalar U was lost there, 2026-11-13)
    int U0 = 0; for (int i = 0; i < ns; ++i) U0 += FIL[SEL[i]] >= 0 ? 1 : 0;
    if (slm) PUT(plan, 0, U0);
    int U = 0, rs = 0;
    for (int i = 0; i < ns; ++i) {
        const int e = SEL[i], s = FIL[e];
        if (s >= 0) {
            if (slm) { const int b = 1 + 4 * U; PUT(plan, b, s); PUT(plan, b + 1, CNT[e]); PUT(plan, b + 2, rs); PUT(plan, b + 3, e); }
            else { const int b = 1 + 4 * U; STI(plan, b, 0, s); STI(plan, b + 1, 0, CNT[e]); STI(plan, b + 2, 0, rs); STI(plan, b + 3, 0, e); }
            FIL[e] = rs; rs += CNT[e]; ++U;
        } else FIL[e] = -1;
    }
    if (slm) {
        FLUSH(plan);
        // pass 3: row lists into SLM, device flags per row (vector stores, one per row: a 2-D store is clipped at the row end)
        k = 0; t = 0; int64 dv = 0;
        for (int a = 0; a < KT; ++a) {
            const int v = IR[a]; const int f = FIL[v];
            if (f >= 0) { RW[f] = t * 16 + k; FIL[v] = f + 1; }
            dv = v_i32_sel_eq_i32_b(lane, k, (int64)(f >= 0 ? 1 : 0), dv);
            if (++k == K) { v_i32_st_tnsr(((int5){0, t, 0, 0, 0}), dev, dv); k = 0; ++t; }
        }
        // the row list, ascending (its first chunk may be the entries' last one: vec still holds that chunk, U included)
        const int rbase = 1 + 4 * KT;
        for (int r = 0; r < rs; ++r) PUT(plan, rbase + r, RW[r]);
        FLUSH(plan);
        return;
    }
    // KT > 2048: scalar row lists and flags
    const int rbase = 1 + 4 * KT;
    k = 0; t = 0;
    for (int a = 0; a < KT; ++a) {
        const int v = LDI(ids, k, t); const int f = FIL[v];
        if (f >= 0) { STI(plan, rbase + f, 0, t * 16 + k); FIL[v] = f + 1; STI(dev, k, t, 1); }
        else STI(dev, k, t, 0);
        if (++k == K) { k = 0; ++t; }
    }
    STI(plan, 0, 0, U);
}
