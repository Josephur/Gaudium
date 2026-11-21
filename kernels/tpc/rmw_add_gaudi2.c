// gaudium_rmw_add: probe of ST_TNSR with RMW (fabric-side read-modify-write accumulate into an RMW section).
// inputs x, b; output y (must live in a Synapse RMW section). Per row r:
//   y[r] = b[r]  (plain store)  ;  y[r] (op)= x[r]  twice via st_tnsr_rmw  ->  expect b (op) x (op) x
// mode 0: f32 {64,R}; mode 1: bf16 {128,R}. op: 0 add, 1 sub, 2 min, 3 max (RMW_OP_*).
void main(tensor x, tensor b, tensor y, int mode, int op)
{
    const int5 is = get_index_space_offset();
    const int5 ie = get_index_space_size() + is;
    int rmw_op = (op == 1) ? RMW_OP_SUB : (op == 2) ? RMW_OP_MIN : (op == 3) ? RMW_OP_MAX : RMW_OP_ADD;
    int5 c = {0, 0, 0, 0, 0};
    if (mode == 0) {
        int rmw = RMW_DT_FP32 | rmw_op | RMW_SET;
        for (int r = is[0]; r < ie[0]; ++r) {
            c[1] = r;
            float64 v = v_f32_ld_tnsr_b(c, x);
            float64 base = v_f32_ld_tnsr_b(c, b);
            v_f32_st_tnsr(c, y, base);
            v_f32_st_tnsr_rmw(c, y, v, rmw);
            v_f32_st_tnsr_rmw(c, y, v, rmw);
        }
    } else {
        int rmw = RMW_DT_BF16 | rmw_op | RMW_SET;
        for (int r = is[0]; r < ie[0]; ++r) {
            c[1] = r;
            bfloat128 v = v_bf16_ld_tnsr_b(c, x);
            bfloat128 base = v_bf16_ld_tnsr_b(c, b);
            v_bf16_st_tnsr(c, y, base);
            v_bf16_st_tnsr_rmw(c, y, v, rmw);
            v_bf16_st_tnsr_rmw(c, y, v, rmw);
        }
    }
}
