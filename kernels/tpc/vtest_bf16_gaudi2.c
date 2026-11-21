// gaudium_vtest_bf16: lane order of the bf16<->f32 conversions.
// in0 x bf16 {128} ; in1 f f32 {128} ; out0 y f32 {64, 4}: row0 = convert_all(x).v1, row1 = .v2, row2 = convert_b(x) (single, default), row3 = convert_b(x, lane=1 via switches? no: same with income)
// out1 z bf16 {128, 2}: row0 = convert_f32_to_bf16_all({f[0..63], f[64..127]}) ; row1 = convert_f32_to_bf16_b(f[0..63], lane 0) then convert_f32_to_bf16_b(f[64..127], lane 1, income)
void main(tensor x, tensor f, tensor y, tensor z)
{
    int5 c = {0, 0, 0, 0, 0};
    bfloat128 xb = v_bf16_ld_tnsr_b(c, x);
    float128 xf = v_convert_bf16_to_f32_all_b(xb);
    c[1] = 0; v_f32_st_tnsr(c, y, xf.v1);
    c[1] = 1; v_f32_st_tnsr(c, y, xf.v2);
    float64 x0 = v_convert_bf16_to_f32_b(xb);
    c[1] = 2; v_f32_st_tnsr(c, y, x0);
    float64 x1 = v_convert_bf16_to_f32_b(xb, 1 << 16);   // probe: a switch bit selecting the upper half? (SW_LANE_SEL?)
    c[1] = 3; v_f32_st_tnsr(c, y, x1);
    int5 fc = {0, 0, 0, 0, 0};
    float128 ff; ff.v1 = v_f32_ld_tnsr_b(fc, f); fc[0] = 64; ff.v2 = v_f32_ld_tnsr_b(fc, f);
    int5 zc = {0, 0, 0, 0, 0};
    v_bf16_st_tnsr(zc, z, v_convert_f32_to_bf16_all_b(ff));
    bfloat128 zero = 0;
    bfloat128 h = v_convert_f32_to_bf16_b(ff.v1, 0, 0, zero);
    h = v_convert_f32_to_bf16_b(ff.v2, 1, 0, h);
    zc[1] = 1; v_bf16_st_tnsr(zc, z, h);
}
