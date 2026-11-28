// gaudium_ik_w4_plan: routed-expert dispatch plan when all experts reside on-card (no slot_of / dev).
//   in0  ids   i32 {K, T}            router expert ids (ascending per row)
//   out0 plan  i32 {1 + 5 K T}       [0] = U unique experts; entries j (ascending id): [1+4j+{0..3}] = {e, n, start, e};
//                                    row list [1+4 K T + r] = t * 16 + k (each expert's rows in ascending (t, k))
//   scalars: K (<= 12), T, E (<= 512)
#define IK_PLAN_TMERGE 8
#define IK_PLAN_BIG 0x40000000
__local__ int CNT[512];
__local__ int FIL[512];
__local__ int SEL[512];
__local__ int IR[1024];
__local__ int RW[1024];
__local__ int HP[IK_PLAN_TMERGE];
__local__ int HV[IK_PLAN_TMERGE];
__local__ int EN[4 * 12 * IK_PLAN_TMERGE];
#define LDI(tn, c0, c1) s_i32_ld_g(gen_addr(((int5){(c0), (c1), 0, 0, 0}), tn))
#define STI(tn, c0, c1, v) s_i32_st_g(gen_addr(((int5){(c0), (c1), 0, 0, 0}), tn), (v))
#define PUT(tn, idx, val) { const int _i = (idx); if ((_i >> 6) != cur) { if (cur >= 0) v_i32_st_tnsr(((int5){cur << 6, 0, 0, 0, 0}), tn, vec); cur = _i >> 6; } \
    vec = v_i32_sel_eq_i32_b(lane, _i & 63, (int64)(val), vec); }
#define FLUSH(tn) { if (cur >= 0) v_i32_st_tnsr(((int5){cur << 6, 0, 0, 0, 0}), tn, vec); cur = -1; }

void main(tensor ids, tensor plan, int K, int T, int E)
{
    const int KT = K * T;
    const uint64 lu = read_lane_id_4b_b(); const int64 lane = *(int64*)&lu;
    if (T <= IK_PLAN_TMERGE) {
        int k = 0, t = 0;
        for (int a = 0; a < KT; ++a) { IR[a] = LDI(ids, k, t); if (++k == K) { k = 0; ++t; } }
        for (int r = 0; r < T; ++r) { HP[r] = 0; HV[r] = IR[r * K]; }
        int U = 0, rs = 0;
        for (;;) {
            int v = IK_PLAN_BIG;
            for (int r = 0; r < T; ++r) { const int h = HV[r]; if (h < v) v = h; }
            if (v == IK_PLAN_BIG) break;
            int n = 0;
            for (int r = 0; r < T; ++r) {
                if (HV[r] == v) {
                    const int kk = HP[r]; RW[rs + n] = r * 16 + kk; ++n;
                    HP[r] = kk + 1; HV[r] = kk + 1 < K ? IR[r * K + kk + 1] : IK_PLAN_BIG;
                }
            }
            EN[4 * U] = v; EN[4 * U + 1] = n; EN[4 * U + 2] = rs; EN[4 * U + 3] = v; rs += n; ++U;
        }
        int64 vec = 0; int cur = -1;
        PUT(plan, 0, U);
        const int ne = 4 * U;
        for (int i = 0; i < ne; ++i) PUT(plan, 1 + i, EN[i]);
        const int rbase = 1 + 4 * KT;
        for (int r = 0; r < rs; ++r) PUT(plan, rbase + r, RW[r]);
        FLUSH(plan);
        return;
    }
    for (int e = 0; e < E; ++e) CNT[e] = 0;
    const bool slm = KT <= 1024;
    int k = 0, t = 0;
    for (int a = 0; a < KT; ++a) {
        const int v = LDI(ids, k, t);
        if (slm) IR[a] = v;
        CNT[v] = CNT[v] + 1;
        if (++k == K) { k = 0; ++t; }
    }
    int ns = 0;
    for (int e = 0; e < E; ++e) if (CNT[e] > 0) { SEL[ns] = e; ++ns; }
    int64 vec = 0; int cur = -1;
    if (slm) PUT(plan, 0, ns);
    int rs = 0;
    for (int i = 0; i < ns; ++i) {
        const int e = SEL[i], n = CNT[e];
        const int b = 1 + 4 * i;
        if (slm) { PUT(plan, b, e); PUT(plan, b + 1, n); PUT(plan, b + 2, rs); PUT(plan, b + 3, e); }
        else { STI(plan, b, 0, e); STI(plan, b + 1, 0, n); STI(plan, b + 2, 0, rs); STI(plan, b + 3, 0, e); }
        FIL[e] = rs; rs += n;
    }
    if (slm) {
        FLUSH(plan);
        k = 0; t = 0;
        for (int a = 0; a < KT; ++a) {
            const int v = IR[a]; const int f = FIL[v];
            RW[f] = t * 16 + k; FIL[v] = f + 1;
            if (++k == K) { k = 0; ++t; }
        }
        const int rbase = 1 + 4 * KT;
        for (int r = 0; r < rs; ++r) PUT(plan, rbase + r, RW[r]);
        FLUSH(plan);
        return;
    }
    const int rbase = 1 + 4 * KT;
    k = 0; t = 0;
    for (int a = 0; a < KT; ++a) {
        const int v = LDI(ids, k, t); const int f = FIL[v];
        STI(plan, rbase + f, 0, t * 16 + k); FIL[v] = f + 1;
        if (++k == K) { k = 0; ++t; }
    }
    STI(plan, 0, 0, ns);
}
