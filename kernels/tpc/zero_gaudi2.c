// gaudium_zero: out0 f32 {64, R} := 0: the per-request reset of the engine's state block on the 24 TPCs.
// synMemsetD32Async ran at ~62 GB/s (one DMA engine: 444 ms for the 27.7 GB block at Lmax 135168, paid by back-to-back serving
// requests); vector stores from every TPC are HBM-bound. Index space {R / 1024}: point i stores rows 1024 i .. 1024 i + 1023.
void main(tensor out)
{
    const int5 idx = get_index_space_offset();
    const int5 isz = get_index_space_size();
    const float64 z = 0.f;
    for (int i = idx[0]; i < idx[0] + isz[0]; ++i) {
        int5 c = {0, 1024 * i, 0, 0, 0};
#pragma unroll 8
        for (int r = 0; r < 1024; ++r) { v_f32_st_tnsr(c, out, z); c[1] += 1; }
    }
}
