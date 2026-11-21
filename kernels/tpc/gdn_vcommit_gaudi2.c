// gaudium_gdn_vcommit: the TPC half of a verify step's ONE DeltaNet state update per linear layer (see
// gdn_vrec_gaudi2.c). After the accept kernel named the last accepted row a, the state the next step starts from is
//     S' = E_a S0 + sum_{i in path(a)} e(a,i) k_i (x) d_i          (path = a and its ancestors up to the root, row 0)
// E_a = prod of the decays g over the path, e(a,i) = prod of g over the path rows after i. This kernel writes dlc (row i = e(a,i) d_i
// on the path, 0 elsewhere; 2T rows, the last T zero, so that the GEMM can take kq = [k ; q] whole: a split view of its k rows cost a
// 4 us copy per layer) and the decayed state Sd = E_a S0; one batched GEMM on the MME then does S' = Sd + kq^T . dlc, whatever the
// number of accepted rows.
//
//  in0 dl f32 {128, T, HV, 1}     in1 eg f32 {64, T, 1}     in2 a int32 {1}     in3 S0 f32 {128, 128, HV, 1} (row c, value lanes)
//  out0 dlc f32 {128, 2T, HV, 1}  out1 Sd f32 {128, 128, HV, 1}
//  scalars: T, NC, L (the row tree, as gaudium_gdn_vrec). One index-space point = one value head. One scalar global load (a).
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define BC0(x) ({ float64 _x = (x); _x = v_f32_shuffle_b(_x, P0, 0, _x); MOVDG_ALL(_x, 0, 0, 0, 0, _x); })   // lane 0 -> every lane
#define PARENT(t) ((t) < NC ? (t) - 1 : ((t) - NC < L ? (t) - NC : (t) - NC - L))

__local__ float64 CF[32];

void main(tensor dl, tensor eg, tensor a, tensor S0, tensor dlc, tensor Sd, int T, int NC, int L)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const unsigned char pb = 0x80; const uchar256 P0 = pb;
    const int5 zc = {0, 0, 0, 0, 0};
    int last = s_i32_ld_g(gen_addr(zc, a));
    if (last < 0) last = 0;
    if (last > T - 1) last = T - 1;

    for (int h = idx[0]; h < idx[0] + isz[0]; ++h) {
        const float64 zero = 0.f;
        for (int t = 0; t < T; ++t) CF[t] = zero;
        float64 e = zero + 1.f;
        for (int i = last; i >= 0; i = PARENT(i)) { int5 gc = {h, i, 0, 0, 0}; CF[i] = e; e = e * BC0(v_f32_ld_tnsr_b(gc, eg)); }
        for (int t = 0; t < T; ++t) {
            const float64 cf = CF[t];
            int5 c = {0, t, h, 0, 0};
            const float64 d0 = v_f32_ld_tnsr_b(c, dl); c[0] = 64; const float64 d1 = v_f32_ld_tnsr_b(c, dl);
            v_f32_st_tnsr(c, dlc, d1 * cf); c[0] = 0; v_f32_st_tnsr(c, dlc, d0 * cf);
            c[1] = T + t; v_f32_st_tnsr(c, dlc, zero); c[0] = 64; v_f32_st_tnsr(c, dlc, zero);   // the q rows of kq take no part in the update
        }
        for (int r = 0; r < 128; r += 4) {   // Sd = E_a S0: 4 rows (8 vectors) per iteration, loads before stores
            int5 s0 = {0, r, h, 0, 0}, s1 = {0, r + 1, h, 0, 0}, s2 = {0, r + 2, h, 0, 0}, s3 = {0, r + 3, h, 0, 0};
            float64 x0 = v_f32_ld_tnsr_b(s0, S0), x2 = v_f32_ld_tnsr_b(s1, S0), x4 = v_f32_ld_tnsr_b(s2, S0), x6 = v_f32_ld_tnsr_b(s3, S0);
            s0[0] = 64; s1[0] = 64; s2[0] = 64; s3[0] = 64;
            float64 x1 = v_f32_ld_tnsr_b(s0, S0), x3 = v_f32_ld_tnsr_b(s1, S0), x5 = v_f32_ld_tnsr_b(s2, S0), x7 = v_f32_ld_tnsr_b(s3, S0);
            v_f32_st_tnsr(s0, Sd, x1 * e); v_f32_st_tnsr(s1, Sd, x3 * e); v_f32_st_tnsr(s2, Sd, x5 * e); v_f32_st_tnsr(s3, Sd, x7 * e);
            s0[0] = 0; s1[0] = 0; s2[0] = 0; s3[0] = 0;
            v_f32_st_tnsr(s0, Sd, x0 * e); v_f32_st_tnsr(s1, Sd, x2 * e); v_f32_st_tnsr(s2, Sd, x4 * e); v_f32_st_tnsr(s3, Sd, x6 * e);
        }
    }
}
