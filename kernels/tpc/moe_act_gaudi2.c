// gaudium_moe_act: the routed SwiGLU activation of a MoE layer in its DENSE-MASKED form (issue #8).
//
// The layer's routed experts and its shared expert are stored as one wide MLP: NB = NE + 1 blocks of F channels (block e < NE = routed
// expert e, block NE = the shared expert). The gate and up GEMMs compute every block for every row; this kernel applies the routing:
//
//  in0 g   bf16 {NB * F, T}     W_gate h        in1 u  bf16 {NB * F, T}   W_up h
//  in2 wd2 f32  {NE/2 + 64, 2, T}   the block weights from gaudium_moe_route: w(e, t) = wd2[e >> 1, e & 1, t] (the shared block NE reads
//                                    slot [NE/2, 0]: sigmoid of the shared-expert gate)
//  out0 a  bf16 {NB * F, T}     a[e F + i, t] = bf16(silu(g) * u * w(e, t)); exactly zero for an expert the row did not select
//  index space {NB, T}: one point = one block of one row.   scalars: F (a multiple of 128), first
//  first: the block weight of block b is w(first + b, t). 0 when the wide MLP holds every routed expert and the shared one (NB = NE + 1);
//  NE when the layer's routed experts live on the host and the device holds the shared expert only (NB = 1): hybrid placement.
//
// The down GEMM then gives  sum_e w_e W_down_e (silu(W_gate_e h) * (W_up_e h)) + sg * W_down_s (...)  in one pass: the weighted
// combine of the reference is the GEMM's own accumulation. Rows cost the same as in a dense MLP (the MME reads each weight once
// whatever the row count), so this form is the right one for many rows (verify rows x batch) and needs no host in the loop.
void main(tensor g, tensor u, tensor wd2, tensor a, int F, int first)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    for (int t = idx[1]; t < idx[1] + isz[1]; ++t) {
        for (int e = idx[0]; e < idx[0] + isz[0]; ++e) {
            const int eid = first + e; int5 wc = {eid >> 1, eid & 1, t, 0, 0};
            const float w = s_f32_ld_g(gen_addr(wc, wd2));
            if (w == 0.f) {
                float128 z; z.v1 = 0.f; z.v2 = 0.f; const bfloat128 zb = v_convert_f32_to_bf16_all_b(z);
                for (int i = 0; i < F; i += 128) { int5 c = {e * F + i, t, 0, 0, 0}; v_bf16_st_tnsr(c, a, zb); }
            } else {
                const float64 wv = w;
                for (int i = 0; i < F; i += 128) {
                    int5 c = {e * F + i, t, 0, 0, 0};
                    float128 gg = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c, g));
                    float128 uu = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c, u));
                    float128 r;
                    r.v1 = (gg.v1 * v_sigmoid_f32(gg.v1)) * uu.v1 * wv;
                    r.v2 = (gg.v2 * v_sigmoid_f32(gg.v2)) * uu.v2 * wv;
                    v_bf16_st_tnsr(c, a, v_convert_f32_to_bf16_all_b(r));
                }
            }
        }
    }
}
