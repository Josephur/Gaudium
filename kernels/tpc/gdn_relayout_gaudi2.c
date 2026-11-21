// gaudium_gdn_relayout: DeltaNet state between the gdn_f32 kernel's group-interleaved blob {64, 256, HV} and the plain
// row-major matrix {128(v), 128(c), HV} the MME chunk path multiplies (S[c][v], c = k index, v = value lane, both in the
// deinterleaved lane order). One index-space point = one value head; the 64 KB per head go through VLM once.
//   blob vector n = (p*8 + vb)*2 + half, lane l = 16*g + i  holds  S[c][v] with c = 64*half + 16*g + p, v = 16*vb + i
//   matrix row c, vector hv (= v >> 6), lane i' = v & 63
//   mode 0: blob -> matrix   (in0 S_in {64, 256, HV[, NS]} [+ in1 slot {1} int32 in the -DSLOT build], out0 {128, 128, HV})
//   mode 1: matrix -> blob   (in0 {128, 128, HV}, out0 {64, 256, HV})
#define WRA MkWrA(0b11, 0b11, 0b11, 0b11)
#define MOVDG_ALL(a, s0, s1, s2, s3, inc) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, s0, s1, s2, s3, WRA, (inc))
#define REP_GROUP(a, gs) MOVDG_ALL((a), gs, gs, gs, gs, (a))
#define REP_ANY(dst, a, gs) do { switch (gs) { case 0: dst = REP_GROUP(a, 0); break; case 1: dst = REP_GROUP(a, 1); break; case 2: dst = REP_GROUP(a, 2); break; default: dst = REP_GROUP(a, 3); } } while (0)

__local__ float64 BUF[256];

#ifdef SLOT
void main(tensor src, tensor slot, tensor dst, int mode)
#else
void main(tensor src, tensor dst, int mode)
#endif
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const int64 lane = (int64)read_lane_id_4b_b();
    const int64 lg = lane >> 4;                       // dual group of the lane
#ifdef SLOT
    const int5 zc = {0, 0, 0, 0, 0};
    const int slot_in = s_i32_ld_g(gen_addr(zc, slot));
#else
    const int slot_in = 0;
#endif
    for (int h = idx[0]; h < idx[0] + isz[0]; ++h) {
        if (mode == 0) {
            int5 sc = {0, 0, h, slot_in, 0};
            for (int n = 0; n < 256; n += 4) { sc[1] = n; BUF[n] = v_f32_ld_tnsr_b(sc, src); sc[1] = n + 1; BUF[n + 1] = v_f32_ld_tnsr_b(sc, src); sc[1] = n + 2; BUF[n + 2] = v_f32_ld_tnsr_b(sc, src); sc[1] = n + 3; BUF[n + 3] = v_f32_ld_tnsr_b(sc, src); }
            // c = 64*hf + 16*g + p: the four g of one (hf, p, hv) share the blob vectors n = (p*8 + 4*hv + qd)*2 + hf, so each is read
            // once and replicated with CONSTANT group selectors (feature/prefill: the runtime switch on g cost ~440 cycles per
            // element, 0.5 ms per layer at T = 256 in the profile; pure data movement, output bit-identical)
            for (int hf = 0; hf < 2; ++hf) for (int p = 0; p < 16; ++p) for (int hv = 0; hv < 2; ++hv) {
                float64 o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
#pragma unroll
                for (int qd = 0; qd < 4; ++qd) {
                    const float64 src_v = BUF[(p * 8 + 4 * hv + qd) * 2 + hf];
                    o0 = v_f32_sel_eq_i32_b(lg, qd, REP_GROUP(src_v, 0), o0);
                    o1 = v_f32_sel_eq_i32_b(lg, qd, REP_GROUP(src_v, 1), o1);
                    o2 = v_f32_sel_eq_i32_b(lg, qd, REP_GROUP(src_v, 2), o2);
                    o3 = v_f32_sel_eq_i32_b(lg, qd, REP_GROUP(src_v, 3), o3);
                }
                int5 dc = {64 * hv, 64 * hf + p, h, 0, 0};
                v_f32_st_tnsr(dc, dst, o0); dc[1] += 16; v_f32_st_tnsr(dc, dst, o1); dc[1] += 16; v_f32_st_tnsr(dc, dst, o2); dc[1] += 16; v_f32_st_tnsr(dc, dst, o3);
            }
        } else {
            int5 sc = {0, 0, h, 0, 0};
            for (int c = 0; c < 128; ++c) { sc[1] = c; sc[0] = 0; BUF[2 * c] = v_f32_ld_tnsr_b(sc, src); sc[0] = 64; BUF[2 * c + 1] = v_f32_ld_tnsr_b(sc, src); }
            for (int n = 0; n < 256; ++n) {
                const int hf = n & 1, m = n >> 1, vb = m & 7, p = m >> 3;
                float64 out = 0.f;
                for (int g = 0; g < 4; ++g) {
                    const int c = 64 * hf + 16 * g + p;
                    float64 rep; REP_ANY(rep, BUF[2 * c + (vb >> 2)], vb & 3);
                    out = v_f32_sel_eq_i32_b(lg, g, rep, out);
                }
                int5 dc = {0, n, h, 0, 0};
                v_f32_st_tnsr(dc, dst, out);
            }
        }
    }
}
