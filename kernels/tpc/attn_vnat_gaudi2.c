// gaudium_attn_vnat: the V chunks of a prefill attention layer in the key order of the half-width core's P (GAUDIUM_PREFILL_S16). The V
// cache holds key s of a 128-block at row VPERM(s), the order the f32 softmax's interleaving bf16 conversion gives its P.
// Loads and stores only (no VPU work): 2 MB per kv group and chunk.
//  in0  vcache view bf16 {DH, C, B}   row p of a 128-block holds key VPERM^-1(p)
//  out0 vnat        bf16 {DH, C, B}   ord 0: row s holds key s (gaudium_attn_softmax16 stores bf16 P with lane = key)
//                                     ord 1: V / 2 in the byte order of the FP8 P (CONVERT of two bf16 vectors): rows 2i and 2i + 1 of
//                                            a 256-block hold keys i and 128 + i. The half keeps the FP8 cast of the result (the
//                                            vendor's convert_to_fp8_bf16 node, max 240) clear of V's calibrated maxima (122.5).
// index space 2-D: dim0 = C / 128 (128-key blocks), dim1 = B.   scalar: ord
void main(tensor vc, tensor vn, int ord)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const int DHv = get_dim_size(vc, 0);
    for (int b = idx[1]; b < idx[1] + isz[1]; ++b)
        for (int blk = idx[0]; blk < idx[0] + isz[0]; ++blk)
            for (int s = 0; s < 64; ++s) {   // keys k0 = 128 blk + s and k1 = k0 + 64 sit at cache rows 128 blk + 2s and + 2s + 1
                const int k0 = blk * 128 + s, k1 = k0 + 64;
#define VORD8(k_) (((k_) & ~255) + 2 * ((k_) & 127) + (((k_) >> 7) & 1))
                const int r0 = ord ? VORD8(k0) : k0, r1 = ord ? VORD8(k1) : k1;
                int5 i0 = {0, blk * 128 + 2 * s, b, 0, 0}, i1 = {0, blk * 128 + 2 * s + 1, b, 0, 0};
                int5 o0 = {0, r0, b, 0, 0}, o1 = {0, r1, b, 0, 0};
                for (int d = 0; d < DHv; d += 128) {
                    i0[0] = d; i1[0] = d; o0[0] = d; o1[0] = d;
                    bfloat128 x0 = v_bf16_ld_tnsr_b(i0, vc), x1 = v_bf16_ld_tnsr_b(i1, vc);
                    if (ord) { x0 = v_bf16_mul_b(x0, (bfloat128)0.5f); x1 = v_bf16_mul_b(x1, (bfloat128)0.5f); }
                    v_bf16_st_tnsr(o0, vn, x0); v_bf16_st_tnsr(o1, vn, x1);
                }
            }
}
