//! Host-side tables of the DSpine drafter: its 1-D RoPE tables and the serial-refinement tables.
//!
//! The serial tables are a fixed function of the checkpoint and the target's `lm_head`:
//! - `TG[v]  = bf16( bf16(lm_head[v] * w_norm) @ W_last )` (`W_last` = the last layer's write
//!   projection `[H, R]`),
//! - `TSV[v] = bf16( bf16(sqrt(C) * codebook[v]) @ [G_p; A]^T )` (`[2R, C]`),
//!
//! accumulated in f32 in a fixed order, so every host produces the same bytes.

use super::bf16_round;
use crate::fp8::{bf16_to_f32, f32_to_bf16_rne};

/// `{128, n}` f32 RoPE tables for 128-channel heads with full rotary pairs `(c, c + 64)`:
/// lanes `[0..32)` and `[32..64)` carry `cos(pos * f_2j)`, lanes `[64..96)` / `[96..128)` the odd
/// frequencies; the sine table holds `-sin` in the first half of each group and `+sin` in the second.
pub fn rope128_tables(theta: f64, n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut c = vec![1.0f32; 128 * n];
    let mut s = vec![0.0f32; 128 * n];
    let freq: Vec<(f64, f64)> = (0..32)
        .map(|j| {
            (
                theta.powf(-2.0 * (2 * j) as f64 / 128.0),
                theta.powf(-2.0 * (2 * j + 1) as f64 / 128.0),
            )
        })
        .collect();
    for pos in 0..n {
        let p = pos as f64;
        let (cc, ss) = (
            &mut c[pos * 128..(pos + 1) * 128],
            &mut s[pos * 128..(pos + 1) * 128],
        );
        for (j, &(fe, fo)) in freq.iter().enumerate() {
            let (se, ce) = (p * fe).sin_cos();
            let (so, co) = (p * fo).sin_cos();
            cc[j] = ce as f32;
            cc[32 + j] = ce as f32;
            ss[j] = -(se as f32);
            ss[32 + j] = se as f32;
            cc[64 + j] = co as f32;
            cc[96 + j] = co as f32;
            ss[64 + j] = -(so as f32);
            ss[96 + j] = so as f32;
        }
    }
    (c, s)
}

/// Rows processed together so each weight row is read once per group of table rows.
const GROUP: usize = 16;

/// `y[g] += x[g][k] * w[k]` for every `k`, with `w` = `[kdim][n]` row-major and `x` = `GROUP`
/// input rows of `kdim` values. Element-wise multiply-then-add in ascending `k`: the result does
/// not depend on the vector width the compiler picks.
#[inline(always)]
fn accumulate(x: &[f32], kdim: usize, rows: usize, w: &[f32], n: usize, y: &mut [f32]) {
    for k in 0..kdim {
        let wr = &w[k * n..(k + 1) * n];
        for g in 0..rows {
            let xk = x[g * kdim + k];
            let yr = &mut y[g * n..(g + 1) * n];
            for (yy, &ww) in yr.iter_mut().zip(wr) {
                *yy += xk * ww;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn accumulate_avx2(x: &[f32], kdim: usize, rows: usize, w: &[f32], n: usize, y: &mut [f32]) {
    accumulate(x, kdim, rows, w, n, y)
}

fn accumulate_dispatch(x: &[f32], kdim: usize, rows: usize, w: &[f32], n: usize, y: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: the feature was detected at run time; the body is the portable loop.
            unsafe { accumulate_avx2(x, kdim, rows, w, n, y) };
            return;
        }
    }
    accumulate(x, kdim, rows, w, n, y)
}

/// `out[v] = bf16( bf16(src[v] * scale) @ w )` for every row `v`: `src` = `[rows][kdim]` bf16
/// bytes, `scale` = per-input-channel factors (or one value for all), `w` = `[kdim][n]` f32.
fn table(src: &[u8], rows: usize, kdim: usize, scale: &[f32], w: &[f32], n: usize) -> Vec<u8> {
    let mut out = vec![0u8; rows * n * 2];
    let threads = std::thread::available_parallelism()
        .map_or(1, |p| p.get())
        .min(32);
    let rows_per = rows.div_ceil(threads).next_multiple_of(GROUP).max(GROUP);
    std::thread::scope(|sc| {
        for (ci, chunk) in out.chunks_mut(rows_per * n * 2).enumerate() {
            let v0 = ci * rows_per;
            sc.spawn(move || {
                let mut x = vec![0f32; GROUP * kdim];
                let mut y = vec![0f32; GROUP * n];
                let nrows = chunk.len() / (n * 2);
                let mut done = 0usize;
                while done < nrows {
                    let g = GROUP.min(nrows - done);
                    for gi in 0..g {
                        let row =
                            &src[(v0 + done + gi) * kdim * 2..(v0 + done + gi + 1) * kdim * 2];
                        for (k, c) in row.as_chunks::<2>().0.iter().enumerate() {
                            let s = if scale.len() == 1 { scale[0] } else { scale[k] };
                            x[gi * kdim + k] = bf16_round(bf16_to_f32(u16::from_le_bytes(*c)) * s);
                        }
                    }
                    y[..g * n].fill(0.0);
                    accumulate_dispatch(&x, kdim, g, w, n, &mut y);
                    let dst = &mut chunk[done * n * 2..(done + g) * n * 2];
                    for (o, &val) in dst.as_chunks_mut::<2>().0.iter_mut().zip(&y[..g * n]) {
                        *o = f32_to_bf16_rne(val).to_le_bytes();
                    }
                    done += g;
                }
            });
        }
    });
    out
}

/// Builds `(TG [V][R], TSV [V][2R])` as bf16 bytes.
///
/// `lm` = `lm_head` `[V][H]` bf16 bytes, `nw` = final norm weight `[H]`, `w_last` = `[H][R]`,
/// `cb` = codebook `[V][C]` bf16 bytes, `pv` = `[predecessor_gate; prev]` `[2R][C]`.
#[allow(clippy::too_many_arguments)]
pub fn build_serial_tables(
    lm: &[u8],
    nw: &[f32],
    w_last: &[f32],
    cb: &[u8],
    pv: &[f32],
    vocab: usize,
    hid: usize,
    rank: usize,
    code: usize,
) -> (Vec<u8>, Vec<u8>) {
    assert_eq!(lm.len(), vocab * hid * 2);
    assert_eq!(cb.len(), vocab * code * 2);
    assert_eq!(w_last.len(), hid * rank);
    assert_eq!(pv.len(), 2 * rank * code);
    let tg = table(lm, vocab, hid, nw, w_last, rank);
    // [2R][C] -> [C][2R] so the code channel is the contraction index of the row-major kernel
    let mut pv_t = vec![0f32; code * 2 * rank];
    for j in 0..2 * rank {
        for c in 0..code {
            pv_t[c * 2 * rank + j] = pv[j * code + c];
        }
    }
    let tsv = table(cb, vocab, code, &[(code as f32).sqrt()], &pv_t, 2 * rank);
    (tg, tsv)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf(x: f32) -> [u8; 2] {
        f32_to_bf16_rne(x).to_le_bytes()
    }

    #[test]
    fn rope128_pairs_rotate_by_the_position_angle() {
        let (c, s) = rope128_tables(1e7, 8);
        // position 0: identity rotation
        assert!(c[..128].iter().all(|&x| x == 1.0) && s[..128].iter().all(|&x| x == 0.0));
        let pos = 5usize;
        for j in [0usize, 7, 31] {
            let fe = 1e7f64.powf(-2.0 * (2 * j) as f64 / 128.0);
            let fo = 1e7f64.powf(-2.0 * (2 * j + 1) as f64 / 128.0);
            let b = pos * 128;
            assert_eq!(c[b + j], (pos as f64 * fe).cos() as f32);
            assert_eq!(c[b + 32 + j], c[b + j]);
            assert_eq!(s[b + j], -((pos as f64 * fe).sin() as f32));
            assert_eq!(s[b + 32 + j], -s[b + j]);
            assert_eq!(c[b + 64 + j], (pos as f64 * fo).cos() as f32);
            assert_eq!(s[b + 96 + j], (pos as f64 * fo).sin() as f32);
        }
    }

    #[test]
    fn serial_tables_match_the_scalar_definition() {
        // more rows than one accumulation group and not a multiple of it
        let (vocab, hid, rank, code) = (37usize, 6usize, 4usize, 4usize);
        let val = |i: usize| ((i * 2654435761usize) % 1009) as f32 / 251.0 - 2.0;
        let lm: Vec<u8> = (0..vocab * hid).flat_map(|i| bf(val(i))).collect();
        let cb: Vec<u8> = (0..vocab * code)
            .flat_map(|i| bf(val(i + 7) * 0.03))
            .collect();
        let nw: Vec<f32> = (0..hid).map(|i| bf16_round(val(i + 3))).collect();
        let w_last: Vec<f32> = (0..hid * rank).map(|i| bf16_round(val(i + 11))).collect();
        let pv: Vec<f32> = (0..2 * rank * code)
            .map(|i| bf16_round(val(i + 5)))
            .collect();
        let (tg, tsv) = build_serial_tables(&lm, &nw, &w_last, &cb, &pv, vocab, hid, rank, code);
        assert_eq!(tg.len(), vocab * rank * 2);
        assert_eq!(tsv.len(), vocab * 2 * rank * 2);

        let rd = |b: &[u8], i: usize| bf16_to_f32(u16::from_le_bytes([b[2 * i], b[2 * i + 1]]));
        for v in 0..vocab {
            for r in 0..rank {
                let mut acc = 0f32;
                for h in 0..hid {
                    acc += bf16_round(rd(&lm, v * hid + h) * nw[h]) * w_last[h * rank + r];
                }
                assert_eq!(rd(&tg, v * rank + r), bf16_round(acc), "TG[{v}][{r}]");
            }
            for j in 0..2 * rank {
                let mut acc = 0f32;
                for c in 0..code {
                    acc += bf16_round(rd(&cb, v * code + c) * 2.0) * pv[j * code + c];
                }
                assert_eq!(rd(&tsv, v * 2 * rank + j), bf16_round(acc), "TSV[{v}][{j}]");
            }
        }
    }
}
