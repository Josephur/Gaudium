//! Vision tower weights and per-image host tables.
//!
//! Device tensor names (the tower's graphs use the same names):
//! - `V_patch_W` `{row, H}` / `V_patch_b`: the patch convolution as one GEMM (stride = kernel).
//! - `V<l>_n1_w|b`, `V<l>_n2_w|b` (F32): the block's two LayerNorms.
//! - `V<l>_q|k|v_W|b`: the three row blocks of the fused `qkv` projection.
//! - `V<l>_qr|kr_W|b`: `rotate_half` of q's / k's rows per head (a row permutation with a sign
//!   flip, exact in BF16), so the rotary product needs no on-card lane shuffle.
//! - `V<l>_o_W|b`, `V<l>_fc1_W|b`, `V<l>_fc2_W|b`.
//! - `V_m_n_w|b` (F32), `V_m_fc1_W|b`, `V_m_fc2_W|b`: the merger.
//!
//! Host tables per image grid: bilinear position rows and the tower's 2-D rotary tables, both
//! in the processor's merge order (2x2 patch blocks adjacent).

use std::borrow::Cow;

use crate::fp8::{bf16_to_f32, f32_to_bf16_rne};
use crate::safetensors::SafetensorsStore;
use crate::vision::mrope::f32s_to_bytes;
use crate::vision::VisionSpec;

/// Safetensors prefix of the vision tower.
pub const VISION_PREFIX: &str = "model.visual.";

pub struct VisionPacker<'a> {
    store: &'a SafetensorsStore,
    vs: &'a VisionSpec,
}

impl<'a> VisionPacker<'a> {
    pub fn new(store: &'a SafetensorsStore, vs: &'a VisionSpec) -> Self {
        Self { store, vs }
    }

    fn raw(&self, suffix: &str, shape: &[u64]) -> Result<&'a [u8], String> {
        let full = format!("{VISION_PREFIX}{suffix}");
        let t = self
            .store
            .get(&full)
            .map_err(|e| format!("missing vision weight '{full}': {e}"))?;
        if t.dtype != "BF16" {
            return Err(format!(
                "vision weight '{full}' has dtype {}, expected BF16",
                t.dtype
            ));
        }
        let elems: u64 = shape.iter().product();
        let have: u64 = t.shape.iter().product();
        if have != elems {
            return Err(format!(
                "vision weight '{full}' has shape {:?}, expected {elems} elements {shape:?}",
                t.shape
            ));
        }
        self.store
            .tensor_bytes(&full)
            .map_err(|e| format!("read '{full}': {e}"))?
            .ok_or_else(|| format!("vision weight '{full}' is not memory-mapped"))
    }

    fn as_f32(&self, suffix: &str, len: u64) -> Result<Cow<'a, [u8]>, String> {
        let src = self.raw(suffix, &[len])?;
        let v: Vec<f32> = src
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| bf16_to_f32(u16::from_le_bytes(c)))
            .collect();
        Ok(Cow::Owned(f32s_to_bytes(&v)))
    }

    /// Rows `[r0, r0 + rows)` of a `[_, cols]` BF16 matrix (borrowed from the mmap).
    fn rows(src: &'a [u8], r0: u64, rows: u64, cols: u64) -> Cow<'a, [u8]> {
        let a = (r0 * cols * 2) as usize;
        let b = ((r0 + rows) * cols * 2) as usize;
        Cow::Borrowed(&src[a..b])
    }

    /// `rotate_half` of rows `[r0, r0 + hidden)` per head: row `d < half` takes `-row[d + half]`,
    /// row `d >= half` takes `row[d - half]`.
    fn rot_rows(&self, src: &[u8], r0: u64, cols: u64) -> Cow<'a, [u8]> {
        let (nh, dh) = (self.vs.n_heads as usize, self.vs.head_dim() as usize);
        let half = dh / 2;
        let row_bytes = (cols as usize) * 2;
        let base = (r0 as usize) * row_bytes;
        let mut out = vec![0u8; nh * dh * row_bytes];
        for h in 0..nh {
            for d in 0..dh {
                let lo = d < half;
                let s = base + (h * dh + if lo { d + half } else { d - half }) * row_bytes;
                let o = (h * dh + d) * row_bytes;
                out[o..o + row_bytes].copy_from_slice(&src[s..s + row_bytes]);
                if lo {
                    // flip the BF16 sign bit (high byte, little endian) of every element
                    for e in out[o..o + row_bytes].iter_mut().skip(1).step_by(2) {
                        *e ^= 0x80;
                    }
                }
            }
        }
        Cow::Owned(out)
    }

    /// Device bytes of the named vision weight.
    pub fn pack(&self, name: &str) -> Result<Cow<'a, [u8]>, String> {
        let vs = self.vs;
        let (h, inter, row, mw, out) = (
            vs.hidden,
            vs.intermediate,
            vs.patch_row(),
            vs.merged_width(),
            vs.out_hidden,
        );
        match name {
            "V_patch_W" => {
                return Ok(Cow::Borrowed(
                    self.raw("patch_embed.proj.weight", &[h, row])?,
                ))
            }
            "V_patch_b" => return Ok(Cow::Borrowed(self.raw("patch_embed.proj.bias", &[h])?)),
            "V_m_n_w" => return self.as_f32("merger.norm.weight", h),
            "V_m_n_b" => return self.as_f32("merger.norm.bias", h),
            "V_m_fc1_W" => {
                return Ok(Cow::Borrowed(
                    self.raw("merger.linear_fc1.weight", &[mw, mw])?,
                ))
            }
            "V_m_fc1_b" => return Ok(Cow::Borrowed(self.raw("merger.linear_fc1.bias", &[mw])?)),
            "V_m_fc2_W" => {
                return Ok(Cow::Borrowed(
                    self.raw("merger.linear_fc2.weight", &[out, mw])?,
                ))
            }
            "V_m_fc2_b" => return Ok(Cow::Borrowed(self.raw("merger.linear_fc2.bias", &[out])?)),
            _ => {}
        }
        let (layer, what) = parse_block_weight(name)
            .ok_or_else(|| format!("unknown vision weight tensor '{name}'"))?;
        if layer >= vs.depth {
            return Err(format!(
                "vision weight '{name}': block {layer} >= depth {}",
                vs.depth
            ));
        }
        let b = format!("blocks.{layer}.");
        match what {
            "n1_w" => return self.as_f32(&format!("{b}norm1.weight"), h),
            "n1_b" => return self.as_f32(&format!("{b}norm1.bias"), h),
            "n2_w" => return self.as_f32(&format!("{b}norm2.weight"), h),
            "n2_b" => return self.as_f32(&format!("{b}norm2.bias"), h),
            "o_W" => {
                return Ok(Cow::Borrowed(
                    self.raw(&format!("{b}attn.proj.weight"), &[h, h])?,
                ))
            }
            "o_b" => {
                return Ok(Cow::Borrowed(
                    self.raw(&format!("{b}attn.proj.bias"), &[h])?,
                ))
            }
            "fc1_W" => {
                return Ok(Cow::Borrowed(
                    self.raw(&format!("{b}mlp.linear_fc1.weight"), &[inter, h])?,
                ))
            }
            "fc1_b" => {
                return Ok(Cow::Borrowed(
                    self.raw(&format!("{b}mlp.linear_fc1.bias"), &[inter])?,
                ))
            }
            "fc2_W" => {
                return Ok(Cow::Borrowed(
                    self.raw(&format!("{b}mlp.linear_fc2.weight"), &[h, inter])?,
                ))
            }
            "fc2_b" => {
                return Ok(Cow::Borrowed(
                    self.raw(&format!("{b}mlp.linear_fc2.bias"), &[h])?,
                ))
            }
            _ => {}
        }
        let qw = self.raw(&format!("{b}attn.qkv.weight"), &[3 * h, h])?;
        let qb = self.raw(&format!("{b}attn.qkv.bias"), &[3 * h])?;
        Ok(match what {
            "q_W" => Self::rows(qw, 0, h, h),
            "k_W" => Self::rows(qw, h, h, h),
            "v_W" => Self::rows(qw, 2 * h, h, h),
            "q_b" => Self::rows(qb, 0, h, 1),
            "k_b" => Self::rows(qb, h, h, 1),
            "v_b" => Self::rows(qb, 2 * h, h, 1),
            "qr_W" => self.rot_rows(qw, 0, h),
            "kr_W" => self.rot_rows(qw, h, h),
            "qr_b" => self.rot_rows(qb, 0, 1),
            "kr_b" => self.rot_rows(qb, h, 1),
            _ => return Err(format!("unknown vision block weight '{name}'")),
        })
    }

    /// Host copy of the learned position table `[n_pos][hidden]` BF16.
    pub fn pos_table(&self) -> Result<Vec<u16>, String> {
        let src = self.raw("pos_embed.weight", &[self.vs.n_pos, self.vs.hidden])?;
        Ok(src
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| u16::from_le_bytes(c))
            .collect())
    }
}

/// Splits `V<l>_<what>` into the block index and the weight kind.
pub fn parse_block_weight(name: &str) -> Option<(u64, &str)> {
    let rest = name.strip_prefix('V')?;
    let sep = rest.find('_')?;
    if sep == 0 || !rest[..sep].bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some((rest[..sep].parse().ok()?, &rest[sep + 1..]))
}

/// Patch `(row, column)` of every tower row in merge order: 2x2 blocks row-major, the block's
/// four patches row-major (the order `preprocess` emits patch rows in).
pub fn merge_order(grid_h: usize, grid_w: usize, merge: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(grid_h * grid_w);
    for bh in 0..grid_h / merge {
        for bw in 0..grid_w / merge {
            for ih in 0..merge {
                for iw in 0..merge {
                    out.push((bh * merge + ih, bw * merge + iw));
                }
            }
        }
    }
    out
}

/// `n` points from 0 to `side - 1` in f32, computed from both ends (the symmetric linspace of
/// the reference implementation).
fn linspace_f32(n: usize, side: usize) -> Vec<f32> {
    let last = (side - 1) as f32;
    let step = if n > 1 { last / (n - 1) as f32 } else { 0.0 };
    (0..n)
        .map(|i| {
            if i < n / 2 {
                0.0 + step * i as f32
            } else {
                last - step * (n - 1 - i) as f32
            }
        })
        .collect()
}

/// Position rows of an image grid: the learned `side x side` table bilinearly interpolated to
/// `grid_h x grid_w` (4 BF16 rows x F32 weights summed in F32, rounded to BF16), rows in merge
/// order. Returns `[n_patches][hidden]` BF16.
pub fn pos_embed(vs: &VisionSpec, table: &[u16], grid_h: usize, grid_w: usize) -> Vec<u16> {
    let side = vs.pos_side() as usize;
    let hid = vs.hidden as usize;
    debug_assert_eq!(table.len(), side * side * hid);
    let hg = linspace_f32(grid_h, side);
    let wg = linspace_f32(grid_w, side);
    let mut out = Vec::with_capacity(grid_h * grid_w * hid);
    for (y, x) in merge_order(grid_h, grid_w, vs.merge as usize) {
        let (hf, wf) = (hg[y] as usize, wg[x] as usize);
        let (hc, wc) = ((hf + 1).min(side - 1), (wf + 1).min(side - 1));
        let (fh, fw) = (hg[y] - hf as f32, wg[x] - wf as f32);
        let idx = [
            hf * side + wf,
            hf * side + wc,
            hc * side + wf,
            hc * side + wc,
        ];
        let wt = [
            (1.0 - fh) * (1.0 - fw),
            (1.0 - fh) * fw,
            fh * (1.0 - fw),
            fh * fw,
        ];
        let rows = [
            &table[idx[0] * hid..(idx[0] + 1) * hid],
            &table[idx[1] * hid..(idx[1] + 1) * hid],
            &table[idx[2] * hid..(idx[2] + 1) * hid],
            &table[idx[3] * hid..(idx[3] + 1) * hid],
        ];
        for c in 0..hid {
            let mut s = 0.0f32;
            for (r, &w) in rows.iter().zip(&wt) {
                s += bf16_to_f32(r[c]) * w;
            }
            out.push(f32_to_bf16_rne(s));
        }
    }
    out
}

/// The tower's rotary tables for one grid, `[n_patches][hidden]` F32 each, rows in merge order,
/// identical for every head: feature `d` of a head rotates by `(d % half < quarter ? row : col)
/// * inv_freq[d % quarter]`. q's tables carry the attention scale `1 / sqrt(head_dim)`.
pub struct VitRopeTables {
    pub cos_q: Vec<f32>,
    pub sin_q: Vec<f32>,
    pub cos_k: Vec<f32>,
    pub sin_k: Vec<f32>,
}

pub fn vit_rope_tables(vs: &VisionSpec, grid_h: usize, grid_w: usize) -> VitRopeTables {
    let (nh, dh) = (vs.n_heads as usize, vs.head_dim() as usize);
    let half = dh / 2;
    let quarter = dh / 4;
    let hid = nh * dh;
    let qscale = 1.0f32 / (dh as f32).sqrt();
    let inv: Vec<f32> = (0..quarter)
        .map(|i| 1.0f32 / 10000.0f32.powf((2 * i) as f32 / half as f32))
        .collect();
    let n = grid_h * grid_w;
    let mut t = VitRopeTables {
        cos_q: Vec::with_capacity(n * hid),
        sin_q: Vec::with_capacity(n * hid),
        cos_k: Vec::with_capacity(n * hid),
        sin_k: Vec::with_capacity(n * hid),
    };
    let mut head_c = vec![0f32; dh];
    let mut head_s = vec![0f32; dh];
    for (y, x) in merge_order(grid_h, grid_w, vs.merge as usize) {
        let (row, col) = (y as f32, x as f32);
        for d in 0..dh {
            let j = d % half;
            let ang = (if j < quarter { row } else { col }) * inv[j % quarter];
            head_c[d] = ang.cos();
            head_s[d] = ang.sin();
        }
        for _ in 0..nh {
            t.cos_k.extend_from_slice(&head_c);
            t.sin_k.extend_from_slice(&head_s);
            t.cos_q.extend(head_c.iter().map(|&c| c * qscale));
            t.sin_q.extend(head_s.iter().map(|&s| s * qscale));
        }
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::test_util::qwen38_vision_spec;

    #[test]
    fn block_weight_names_parse() {
        assert_eq!(parse_block_weight("V0_q_W"), Some((0, "q_W")));
        assert_eq!(parse_block_weight("V26_fc2_b"), Some((26, "fc2_b")));
        assert_eq!(parse_block_weight("V_patch_W"), None);
        assert_eq!(parse_block_weight("vb_q_W"), None);
        assert_eq!(parse_block_weight("Vx_q_W"), None);
    }

    #[test]
    fn merge_order_keeps_two_by_two_blocks_adjacent() {
        assert_eq!(
            merge_order(2, 4, 2),
            vec![
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
                (0, 2),
                (0, 3),
                (1, 2),
                (1, 3)
            ]
        );
        assert_eq!(merge_order(16, 28, 2).len(), 448);
    }

    #[test]
    fn linspace_is_symmetric_and_hits_both_ends() {
        let v = linspace_f32(16, 48);
        assert_eq!(v[0], 0.0);
        assert_eq!(v[15], 47.0);
        for i in 0..8 {
            assert_eq!(v[i] + v[15 - i], 47.0);
        }
        assert_eq!(linspace_f32(48, 48)[17], 17.0);
    }

    #[test]
    fn pos_embed_of_the_native_grid_returns_the_table_rows_in_merge_order() {
        let mut vs = qwen38_vision_spec();
        vs.n_pos = 16; // 4 x 4 table
        vs.hidden = 16;
        vs.n_heads = 2;
        // table row r holds the value r (exact in BF16)
        let table: Vec<u16> = (0..16)
            .flat_map(|r| std::iter::repeat_n(f32_to_bf16_rne(r as f32), 16))
            .collect();
        let out = pos_embed(&vs, &table, 4, 4);
        let got: Vec<f32> = (0..16).map(|r| bf16_to_f32(out[r * 16])).collect();
        let want: Vec<f32> = merge_order(4, 4, 2)
            .iter()
            .map(|&(y, x)| (y * 4 + x) as f32)
            .collect();
        assert_eq!(got, want);
        // a 2 x 2 grid samples the four corners
        let out2 = pos_embed(&vs, &table, 2, 2);
        let got2: Vec<f32> = (0..4).map(|r| bf16_to_f32(out2[r * 16])).collect();
        assert_eq!(got2, vec![0.0, 3.0, 12.0, 15.0]);
    }

    #[test]
    fn vit_rope_tables_rotate_rows_then_columns_and_scale_q() {
        let vs = qwen38_vision_spec();
        let t = vit_rope_tables(&vs, 2, 4);
        assert_eq!(t.cos_k.len(), 8 * 1152);
        // tower row 5 is patch (0, 3) in merge order
        let r = 5 * 1152;
        // d < 18: row angle (row 0 -> cos 1, sin 0)
        assert_eq!(t.cos_k[r], 1.0);
        assert_eq!(t.sin_k[r + 17], 0.0);
        // 18 <= d < 36: column angle, frequency d - 18
        assert_eq!(t.cos_k[r + 18], 3.0f32.cos());
        assert_eq!(t.sin_k[r + 18], 3.0f32.sin());
        let inv1 = 1.0f32 / 10000.0f32.powf(2.0 / 36.0);
        assert_eq!(t.sin_k[r + 19], (3.0f32 * inv1).sin());
        // second half of the head repeats the first, every head is the same
        assert_eq!(t.cos_k[r + 36..r + 72], t.cos_k[r..r + 36]);
        assert_eq!(t.sin_k[r + 72 * 15..r + 72 * 16], t.sin_k[r..r + 72]);
        // q carries 1 / sqrt(72)
        let qs = 1.0f32 / 72.0f32.sqrt();
        assert_eq!(t.cos_q[r + 18], 3.0f32.cos() * qs);
        assert_eq!(t.sin_q[r + 18], 3.0f32.sin() * qs);
    }

    #[test]
    fn rotate_half_rows_permute_and_negate() {
        let mut vs = qwen38_vision_spec();
        vs.hidden = 8;
        vs.n_heads = 2; // head width 4, half 2
        let store = SafetensorsStore::empty();
        let p = VisionPacker::new(&store, &vs);
        // 8 rows x 1 col: value r + 1
        let src: Vec<u8> = (0..8)
            .flat_map(|r| f32_to_bf16_rne((r + 1) as f32).to_le_bytes())
            .collect();
        let out = p.rot_rows(&src, 0, 1);
        let got: Vec<f32> = out
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| bf16_to_f32(u16::from_le_bytes(c)))
            .collect();
        // head 0 rows [1, 2, 3, 4] -> [-3, -4, 1, 2]; head 1 rows [5, 6, 7, 8] -> [-7, -8, 5, 6]
        assert_eq!(got, vec![-3.0, -4.0, 1.0, 2.0, -7.0, -8.0, 5.0, 6.0]);
    }
}
