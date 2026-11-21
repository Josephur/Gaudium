// gaudium_ple_sync: the device-side gate of the n-gram (PLE) rows of a decode step (qwen4_exp; docs/notes/flash-next.md 4.3.1).
//   The step is launched before the host has gathered its n-gram rows. This node waits until the host says they are on the card,
//   then hands them to the n-gram layer; every node that does not depend on the rows runs meanwhile.
//   in0  e    bf16 {PE, T}  the host-uploaded table rows of this step's verify tokens (PE a multiple of 128)
//   in1  flag i32  {64}     element 0: 0 = the rows are not there yet, non-zero = they are. READ ONLY (the host writes it by DMA
//                           after the rows, on the same stream); the engine binds another slot of a ring per step
//   out0 eo   bf16 {PE, T}  a copy of e, taken after the flag was seen
//   out1 st   i32  {64}     every lane = the number of polls that found the flag clear, or -1 when maxp polls passed (the copy is
//                           still made: the host treats -1 as fatal). ONE vector store: a scalar store into a chunk that a vector
//                           store also writes is lost on the card
//   scalars: PE, T, maxp (polls before giving up; a scalar global load is ~0.2 us on the card)
//   Every poll invalidates the scalar data cache first (cache_invalidate d: no flush, and this kernel makes no scalar store): scalar
//   global loads are served from that cache (vendor TPC user guide, "TPC Coherency"), so a second load of the same address would
//   return the first value and never see the host's write. The invalidate also keeps the load inside the loop: without it tpc-clang
//   hoists the load (one ld_g, then an empty counting loop: seen in the disassembly). Check the tracked .asm after any change here.
//   index space: one point.
void main(tensor e, tensor flag, tensor eo, tensor st, int PE, int T, int maxp)
{
    int5 fc = {0, 0, 0, 0, 0};
    int polls = 0;
    cache_invalidate(SW_D);
    int f = s_i32_ld_g(gen_addr(fc, flag));
    while (f == 0 && polls < maxp) { polls = polls + 1; cache_invalidate(SW_D); f = s_i32_ld_g(gen_addr(fc, flag)); }
    for (int t = 0; t < T; ++t)
        for (int c0 = 0; c0 < PE; c0 += 128) {
            int5 c = {c0, t, 0, 0, 0};
            bfloat128 v = v_bf16_ld_tnsr_b(c, e);
            v_bf16_st_tnsr(c, eo, v);
        }
    const int status = (f == 0) ? -1 : polls;
    int64 sv = status;
    int5 sc = {0, 0, 0, 0, 0};
    v_i32_st_tnsr(sc, st, sv);
}
