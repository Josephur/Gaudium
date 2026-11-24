// gaudium_ik_save: the rows of a pass into a short ring by position, so that the short causal convolutions of an Inkling 2 layer
// read the rows before a pass from the same place as the rows of the pass:
//   ring[:, (cursor + coff + t) & (RC - 1)] = x[:, t]         for the first C channels (C = the ring's width)
//  in0 x bf16 {>= C, T}   in1 cursor i32 {1}   out0 ring bf16 {C, RC}, RC a power of two of at least T + taps - 1 rows
//  index space {C / 128, T}.   scalar: coff
void main(tensor x, tensor cursor, tensor ring, int coff)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    int5 zc = {0, 0, 0, 0, 0};
    const int cur = s_i32_ld_g(gen_addr(zc, cursor)) + coff;
    const int M1 = get_dim_size(ring, 1) - 1;
    for (int b = idx[0]; b < idx[0] + isz[0]; ++b)
    for (int t = idx[1]; t < idx[1] + isz[1]; ++t) {
        int5 c = {128 * b, t, 0, 0, 0}, oc = {128 * b, (cur + t) & M1, 0, 0, 0};
        v_bf16_st_tnsr(oc, ring, v_bf16_ld_tnsr_b(c, x));
    }
}
