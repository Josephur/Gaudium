// gaudium_vtest_f32 with the vtest_f8 ELF (tests/sim/sim_kv8.cpp only): lane order and rounding of the Gaudi 2
// f32 -> fp8 conversions the FP8 KV cache uses.
// in: x f32 {64, 4} (four float64 vectors v0..v3) ; out: y u8 {256, 6}
//   row0 = convert_f32_to_f8_all(float256 {v0, v1, v2, v3}), RHNE                (fp8_143)
//   row1 = convert_f32_to_f8_b(vj, lane j) for j = 0..3 into one minifloat256    (fp8_143)
//   row2 = convert_f32_to_h8_all(float256 {v0..v3}), RHNE                        (fp8_152)
//   row3 = row0 with SW_CLIP_FP (saturate to the largest finite instead of Inf)
//   row4 = row2 with SW_CLIP_FP
//   row5 = convert_bf16_to_f8_all(bfloat256 {bf16(v0, v1), bf16(v2, v3)}) where bfloat128 = convert_f32_to_bf16_all(float128 {va, vb})
void main(tensor x, tensor y, int p)
{
    int5 c = {0, 0, 0, 0, 0};
    float256 a;
    c[1] = 0; a.v1 = v_f32_ld_tnsr_b(c, x); c[1] = 1; a.v2 = v_f32_ld_tnsr_b(c, x);
    c[1] = 2; a.v3 = v_f32_ld_tnsr_b(c, x); c[1] = 3; a.v4 = v_f32_ld_tnsr_b(c, x);
    minifloat256 r0 = v_convert_f32_to_f8_all_b(a, SW_RHNE);
    minifloat256 r1 = 0;
    r1 = v_convert_f32_to_f8_b(a.v1, 0, SW_RHNE, r1); r1 = v_convert_f32_to_f8_b(a.v2, 1, SW_RHNE, r1);
    r1 = v_convert_f32_to_f8_b(a.v3, 2, SW_RHNE, r1); r1 = v_convert_f32_to_f8_b(a.v4, 3, SW_RHNE, r1);
    minihalf256 r2 = v_convert_f32_to_h8_all_b(a, SW_RHNE);
    minifloat256 r3 = v_convert_f32_to_f8_all_b(a, SW_RHNE | SW_CLIP_FP);
    minihalf256 r4 = v_convert_f32_to_h8_all_b(a, SW_RHNE | SW_CLIP_FP);
    float128 fa; fa.v1 = a.v1; fa.v2 = a.v2; float128 fb; fb.v1 = a.v3; fb.v2 = a.v4;
    bfloat256 bb; bb.v1 = v_convert_f32_to_bf16_all_b(fa); bb.v2 = v_convert_f32_to_bf16_all_b(fb);
    minifloat256 r5 = v_convert_bf16_to_f8_all_b(bb, SW_RHNE);
    c[0] = 0;
    c[1] = 0; v_f8_st_tnsr(c, y, r0);
    c[1] = 1; v_f8_st_tnsr(c, y, r1);
    c[1] = 2; v_h8_st_tnsr(c, y, r2);
    c[1] = 3; v_f8_st_tnsr(c, y, r3);
    c[1] = 4; v_h8_st_tnsr(c, y, r4);
    c[1] = 5; v_f8_st_tnsr(c, y, r5);
}
