// gaudium_vtest_f32: empirical semantics of SHUFFLE and MOV_DUAL_GROUP_ALL on f32 vectors.
// in: x {64} ; out: y {64, 6}:  row0 = shuffle(x, PAT(p=5)) ; row1 = mov_dg_all(x, src dual group 2 everywhere)
//                              row2 = mov_dg_all(x, rotate 1,2,3,0) ; row3 = byte lane ids as f32 (first 64 bytes -> lanes)
//                              row4 = shuffle(x, PAT(p)) with p from scalar param ; row5 = x
void main(tensor x, tensor y, int p)
{
    int5 c = {0, 0, 0, 0, 0};
    float64 v = v_f32_ld_tnsr_b(c, x);
    uchar256 lid8 = read_lane_id_1b_b();
    uchar256 pat5 = ((lid8 & 3) + 4 * 5) | 0x80;
    uchar256 patp = ((lid8 & 3) + 4 * p) | 0x80;
    float64 zero = 0.f;
    float64 r0 = v_f32_shuffle_b(v, pat5, 0, zero);
    float64 r1 = v_f32_mov_dual_group_all_b(v, 0xFFFFFFFF, 2, 2, 2, 2, MkWrA(0b11, 0b11, 0b11, 0b11), zero);
    float64 r2 = v_f32_mov_dual_group_all_b(v, 0xFFFFFFFF, 1, 2, 3, 0, MkWrA(0b11, 0b11, 0b11, 0b11), zero);
    uint64 lid4 = read_lane_id_4b_b();
    float64 r3 = v_convert_u32_to_f32_b(lid4, 0, zero);
    float64 r4 = v_f32_shuffle_b(v, patp, 0, zero);
    c[1] = 0; v_f32_st_tnsr(c, y, r0);
    c[1] = 1; v_f32_st_tnsr(c, y, r1);
    c[1] = 2; v_f32_st_tnsr(c, y, r2);
    c[1] = 3; v_f32_st_tnsr(c, y, r3);
    c[1] = 4; v_f32_st_tnsr(c, y, r4);
    c[1] = 5; v_f32_st_tnsr(c, y, v);
}
