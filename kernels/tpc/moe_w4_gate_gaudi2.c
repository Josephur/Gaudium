// gaudium_moe_gate: the device-side miss gate of a hybrid routed-expert layer in a decode step (expert cache,
// docs/notes/moe-expert-cache.md 3). The step is ONE recipe, launched ahead. A layer whose routed experts are all in device slots
// passes here in three scalar loads; a layer with an expert that is not on the card stops here, tells the host, and waits until the
// host has served the miss and acknowledged: it copies the expert into a free slot and rewrites this layer's plan to use it, or it
// puts the expert's output rows (the host twin, bit-identical) into the layer's yh buffer.
//   in0  dev    i32 {K, T}        gaudium_moe_plan's flags: 1 = the expert of assignment (k, t) is in a device slot, 0 = it is not
//   in1  plan   i32 {1 + 5 K T}   gaudium_moe_plan's work list (moe_w4_plan_gaudi2.c): [0] U entries; entry j = slot, n rows, start, id;
//                                 the row list from [1 + 4 K T]. The assignments on the card are the rows of the list: start + n of
//                                 the last entry, so K T minus that is the number of assignments that are NOT (three scalar loads)
//   in2  cursor i32 {1}           the step's cursor: the nonce of this step's requests (it grows with every step of a request)
//   in3  ack    i32 {64}          element G (this gate's index, a scalar): the host writes the nonce there when it has served the layer;
//                                 0x7FFFFFFF releases every gate (the host is going down: the step's tokens are dropped). READ ONLY
//   in4  acc_in i32 {64}          the running totals of the gates before this one in the step (absent in the -DFIRST build)
//   out0 devo   i32 {K, T}        dev and plan as they are AFTER the wait: the expert kernels and the combine read these copies, so
//   out1 plano  i32 {1 + 5 K T}   they run after this node. While the gate waits the host may have uploaded missing experts into
//                                 free slots and rewritten plan and dev to use them (and put the others' rows into the layer's yh)
//   out2 st     i32 {64}          written ONLY when the layer misses: [0] 1 while waiting, then 3 = acknowledged, -1 = timed out,
//                                 -2 = released; [1] the nonce; [2] polls waited; [3] T; [4] assignments not on the card
//   out3 acc    i32 {64}          acc_in + {1 if this layer waited, 1 if it timed out or was released, polls, assignments not on the card}
//   scalars: K, T, G, maxp (polls before giving up; a poll is one scalar global load, ~0.2 us on the card)
// Every poll invalidates the scalar data cache first (cache_invalidate d): scalar global loads are served from that cache (vendor
// TPC user guide, "TPC Coherency"), so a second load of an address would return the first value and never see the host's write;
// the invalidate also keeps the load inside the loop (see ple_sync_gaudi2.c). The copies are vector loads, which have no cache: they
// see what the host wrote during the wait. The kernel makes no scalar store: the status and the totals are assembled in vector
// registers and written with vector stores (a scalar store into a chunk that a vector store also writes is lost on the card).
// Check the tracked .asm after any change here. Index space: one point.
#define MOE_GATE_RELEASE 0x7FFFFFFF
#define LDI(tn, c0, c1) s_i32_ld_g(gen_addr(((int5){(c0), (c1), 0, 0, 0}), tn))
#define SET(vec, idx, val) vec = v_i32_sel_eq_i32_b(lane, (idx), (int64)(val), vec)
#ifdef FIRST
void main(tensor dev, tensor plan, tensor cursor, tensor ack, tensor devo, tensor plano, tensor st, tensor acc, int K, int T, int G, int maxp)
#else
void main(tensor dev, tensor plan, tensor cursor, tensor ack, tensor acc_in, tensor devo, tensor plano, tensor st, tensor acc, int K, int T, int G, int maxp)
#endif
{
    const uint64 lu = read_lane_id_4b_b(); const int64 lane = *(int64*)&lu;
    const int5 z = {0, 0, 0, 0, 0};
    const int KT = K * T;
    cache_invalidate(SW_D);
    const int U = LDI(plan, 0, 0);
    int on = 0;
    if (U > 0) { const int b = 1 + 4 * (U - 1); on = LDI(plan, b + 2, 0) + LDI(plan, b + 1, 0); }
    const int nmiss = KT - on;
#ifdef FIRST
    int64 a = 0;
#else
    int64 a = v_i32_ld_tnsr_b(z, acc_in);
#endif
    if (nmiss > 0) {
        const int nonce = s_i32_ld_g(gen_addr(z, cursor));
        int64 sv = 0;
        SET(sv, 0, 1); SET(sv, 1, nonce); SET(sv, 3, T); SET(sv, 4, nmiss);
        v_i32_st_tnsr(z, st, sv);                       // the request: the host polls this
        int5 ac = {G, 0, 0, 0, 0};
        int polls = 0;
        cache_invalidate(SW_D);
        int f = s_i32_ld_g(gen_addr(ac, ack));
        while (f != nonce && f != MOE_GATE_RELEASE && polls < maxp) { polls = polls + 1; cache_invalidate(SW_D); f = s_i32_ld_g(gen_addr(ac, ack)); }
        const int state = (f == nonce) ? 3 : ((f == MOE_GATE_RELEASE) ? -2 : -1);
        SET(sv, 0, state); SET(sv, 2, polls);
        v_i32_st_tnsr(z, st, sv);
        int64 d = 0;
        SET(d, 0, 1); SET(d, 1, state == 3 ? 0 : 1); SET(d, 2, polls); SET(d, 3, nmiss);
        a = a + d;
    }
    for (int t = 0; t < T; ++t) {                       // after the wait: what the host left in dev and plan
        int5 c = {0, t, 0, 0, 0};
        v_i32_st_tnsr(c, devo, v_i32_ld_tnsr_b(c, dev));
    }
    const int NP = 1 + 5 * KT;
    for (int c0 = 0; c0 < NP; c0 += 64) {
        int5 c = {c0, 0, 0, 0, 0};
        v_i32_st_tnsr(c, plano, v_i32_ld_tnsr_b(c, plan));
    }
    v_i32_st_tnsr(z, acc, a);
}
