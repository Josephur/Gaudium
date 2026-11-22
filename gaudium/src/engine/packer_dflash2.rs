//! Host-side tables and layouts of the DFlash2 drafter: its 1-D RoPE tables, the per-channel
//! parameters of the block kernels, and the base kernels of the two-tap dynamic convolutions.
//!
//! Matrices need no packing: a checkpoint `[out, in]` weight is the `{in, out}` operand of a
//! transposed-B GEMM as it is, and the selector codebooks `[vocab, rank]` are `{rank, vocab}`.

use crate::fp8::bf16_to_f32;
use crate::verified::lane256;

/// Channels one vector of the block kernels covers.
const BLOCK: usize = 128;

/// `{128, n}` f32 RoPE rows for 128-channel heads with full rotary pairs `(c, c + 64)`, one row
/// per entry of `positions`: lanes `[0..32)` and `[32..64)` carry `cos(pos * f_2j)`, lanes
/// `[64..96)` / `[96..128)` the odd frequencies; the sine row holds `-sin` in the first half of
/// each group and `+sin` in the second.
///
/// A row is addressed by KV slot and holds the rotation of that slot's *position*. For a text
/// request the two are equal ([`rope128_tables`]); after an image they differ, and the rows of the
/// slots behind it are rewritten with the positions the target assigned to them.
pub fn rope128_rows(theta: f64, positions: impl Iterator<Item = u64>) -> (Vec<f32>, Vec<f32>) {
    let freq: Vec<(f64, f64)> = (0..32)
        .map(|j| {
            (
                theta.powf(-2.0 * (2 * j) as f64 / 128.0),
                theta.powf(-2.0 * (2 * j + 1) as f64 / 128.0),
            )
        })
        .collect();
    let (mut c, mut s) = (Vec::new(), Vec::new());
    for pos in positions {
        let p = pos as f64;
        let base = c.len();
        c.resize(base + BLOCK, 1.0f32);
        s.resize(base + BLOCK, 0.0f32);
        let (cc, ss) = (&mut c[base..], &mut s[base..]);
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

/// The tables of a text request: slot `i` sits at position `i`.
pub fn rope128_tables(theta: f64, n: usize) -> (Vec<f32>, Vec<f32>) {
    rope128_rows(theta, 0..n as u64)
}

/// Little-endian bytes of an f32 table.
pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bf16_at(bytes: &[u8], i: usize) -> f32 {
    bf16_to_f32(u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]))
}

/// A per-channel bf16 parameter as f32 in the deinterleaved lane order of each 128-channel block
/// (the order a bf16 -> f32 vector conversion leaves the channels in).
pub fn lane_f32(bf16: &[u8]) -> Vec<u8> {
    let n = bf16.len() / 2;
    let mut out = vec![0f32; n];
    for i in 0..n {
        out[lane256(i as u64) as usize] = bf16_at(bf16, i);
    }
    f32_bytes(&out)
}

/// One side of a convolution's base kernel as the conv kernel reads it: f32 `{hid, 2}`, tap major,
/// channels in lane order. `base` = the checkpoint's bf16 `[2 sides, 2 taps, hid]`.
pub fn conv_base(base: &[u8], side: usize, hid: usize) -> Vec<u8> {
    assert!(side < 2 && base.len() == 2 * 2 * hid * 2);
    let mut out = vec![0f32; 2 * hid];
    for tap in 0..2 {
        for c in 0..hid {
            out[tap * hid + lane256(c as u64) as usize] = bf16_at(base, (side * 2 + tap) * hid + c);
        }
    }
    f32_bytes(&out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fp8::f32_to_bf16_rne;

    fn f32s(b: &[u8]) -> Vec<f32> {
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|&c| f32::from_le_bytes(c))
            .collect()
    }

    #[test]
    fn rope_rows_follow_positions_not_slots() {
        let (c, s) = rope128_tables(1e7, 4);
        assert_eq!(c.len(), 4 * 128);
        // position 0: no rotation
        assert!(c[..128].iter().all(|&x| x == 1.0) && s[..128].iter().all(|&x| x == 0.0));
        // the pair layout of one frequency: cos twice, -sin then +sin
        let (j, pos) = (5usize, 3usize);
        let fe = 1e7f64.powf(-2.0 * (2 * j) as f64 / 128.0);
        let row = pos * 128;
        assert_eq!(c[row + j], (3.0 * fe).cos() as f32);
        assert_eq!(c[row + 32 + j], c[row + j]);
        assert_eq!(s[row + j], -((3.0 * fe).sin() as f32));
        assert_eq!(s[row + 32 + j], -s[row + j]);
        // a slot behind an image carries the rotation of its position
        let (c2, s2) = rope128_rows(1e7, [0u64, 1, 3, 3].into_iter());
        assert_eq!(&c2[2 * 128..3 * 128], &c[3 * 128..4 * 128]);
        assert_eq!(&s2[3 * 128..4 * 128], &s[3 * 128..4 * 128]);
    }

    #[test]
    fn conv_base_is_tap_major_in_lane_order() {
        let hid = 256usize;
        // bf16-exact values that tell side, tap and channel apart
        let val = |side: usize, tap: usize, c: usize| {
            (c + 1) as f32 * [1.0, 1024.0][side] * [1.0, -1.0][tap]
        };
        let mut base = Vec::new();
        for side in 0..2 {
            for tap in 0..2 {
                for c in 0..hid {
                    base.extend_from_slice(&f32_to_bf16_rne(val(side, tap, c)).to_le_bytes());
                }
            }
        }
        for side in 0..2 {
            let out = f32s(&conv_base(&base, side, hid));
            assert_eq!(out.len(), 2 * hid);
            for tap in 0..2 {
                for c in [0usize, 1, 2, 127, 128, 129, 255] {
                    assert_eq!(
                        out[tap * hid + lane256(c as u64) as usize],
                        val(side, tap, c)
                    );
                }
            }
        }
        // even channels fill the first half of a block, odd channels the second
        let out = f32s(&conv_base(&base, 0, hid));
        assert_eq!(&out[..3], &[1.0, 3.0, 5.0]);
        assert_eq!(&out[64..67], &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn lane_f32_permutes_inside_each_block() {
        let n = 256usize;
        let bytes: Vec<u8> = (0..n)
            .flat_map(|i| f32_to_bf16_rne(i as f32).to_le_bytes())
            .collect();
        let out = f32s(&lane_f32(&bytes));
        for i in 0..n {
            assert_eq!(out[lane256(i as u64) as usize], i as f32);
        }
        assert!(out[128..].iter().all(|&x| x >= 128.0));
    }
}
