//! GGUF tensors presented through the weight store the engines read ([`SafetensorsStore`]),
//! so a packer keeps asking for tensors by their checkpoint names.
//!
//! A caller gives a name map: for every GGUF tensor, the checkpoint tensors it becomes (none to
//! skip it, several to split it), each with a [`ViewSpec`]: a run of the tensor's rows (one
//! expert of a stacked `[E, R, C]` tensor, one half of a fused projection), the row-major shape
//! to register, the output type (BF16 or F32), a scale (one value, or one per outer index such
//! as the per-expert `.scale` of an NVFP4 tensor), a value transform, row or column maps, and a
//! real 2-D transpose. Each becomes a lazy tensor ([`LazyRows`]) of the store.
//!
//! Decoded tensors are NOT cached, neither in RAM nor on disk: every read decodes from the
//! mapped file. A quantised 50 GB model therefore never sits decoded in anonymous memory, and
//! there is no second copy of it on disk; the earlier engine measured that host decoding on a
//! handful of threads outruns the copy into the card's staging buffers, so a cache would not
//! shorten a load. The exception is a transposed view, which needs every source row for each
//! output row: it is decoded once, on first use, and kept (meant for small tensors only).
//!
//! GGUF lists dims innermost first; the default registered shape is the dims reversed, which is
//! the safetensors `[out, in]` convention for the same bytes (no data moves).

use std::sync::{Arc, OnceLock};

use super::dequant::{rows_bf16 as gg_rows_bf16, rows_f32 as gg_rows_f32};
use super::{err, GgmlType, Gguf, GgufError, TensorInfo, MAX_ELEMS};
use crate::fp8::f32_to_bf16_rne;
use crate::safetensors::{LazyRows, SafetensorsStore};
use crate::verified as v;

/// Element type a view is registered as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutType {
    #[default]
    Bf16,
    F32,
}

impl OutType {
    pub fn dtype(self) -> &'static str {
        match self {
            OutType::Bf16 => "BF16",
            OutType::F32 => "F32",
        }
    }
}

/// A multiplier applied to every decoded value (in `f32`, before rounding to BF16).
#[derive(Debug, Clone, Default, PartialEq)]
pub enum Scale {
    #[default]
    None,
    /// One factor for the whole view.
    All(f32),
    /// One factor per index of the registered shape's first axis (e.g. per expert).
    PerOuter(Arc<[f32]>),
}

/// A value transform applied to each decoded row (in `f32`, after the scale).
pub type ValueFn = Arc<dyn Fn(&mut [f32]) + Send + Sync>;

/// How one GGUF tensor (or a part of it) becomes one checkpoint tensor.
#[derive(Clone, Default)]
pub struct ViewSpec {
    /// `(first row, row count)` of the GGUF tensor's rows (all axes but the innermost,
    /// flattened); `None` = every row.
    pub rows: Option<(u64, u64)>,
    /// Registered row-major shape; `None` = the GGUF dims reversed (whole tensor) or
    /// `[rows, row_len]` (a row range). Its first axis indexes whole GGUF rows.
    pub shape: Option<Vec<u64>>,
    pub out: OutType,
    pub scale: Scale,
    /// Element transform on each decoded row.
    pub value_fn: Option<ValueFn>,
    /// View row `r` reads GGUF row `first + row_map[r]` (length = row count).
    pub row_map: Option<Arc<[u64]>>,
    /// Output column `c` takes GGUF column `col_map[c]` (length = row length).
    pub col_map: Option<Arc<[u32]>>,
    /// Register the 2-D transpose (`[row_len, rows]`); decoded once on first use.
    pub transpose: bool,
}

impl ViewSpec {
    /// The whole tensor, BF16, shape = dims reversed.
    pub fn whole() -> Self {
        Self::default()
    }
    /// GGUF rows `[first, first + count)` as a `[count, row_len]` matrix.
    pub fn rows(first: u64, count: u64) -> Self {
        Self {
            rows: Some((first, count)),
            ..Self::default()
        }
    }
    /// Expert `e` of a stacked `[E, R, C]` tensor (`rows_per` = R) as `[R, C]`.
    pub fn expert(e: u64, rows_per: u64) -> Self {
        Self::rows(v::gg_elem_offset(e, rows_per), rows_per)
    }
    pub fn f32(mut self) -> Self {
        self.out = OutType::F32;
        self
    }
    pub fn scaled(mut self, s: f32) -> Self {
        self.scale = Scale::All(s);
        self
    }
    pub fn per_outer_scale(mut self, s: Arc<[f32]>) -> Self {
        self.scale = Scale::PerOuter(s);
        self
    }
    pub fn with_shape(mut self, shape: Vec<u64>) -> Self {
        self.shape = Some(shape);
        self
    }
    pub fn map_values(mut self, f: ValueFn) -> Self {
        self.value_fn = Some(f);
        self
    }
}

/// The checkpoint view of (part of) a GGUF tensor: a [`LazyRows`] producer.
pub struct GgufView {
    g: Arc<Gguf>,
    t: TensorInfo,
    first: u64,
    count: u64,
    shape: Vec<u64>,
    /// Values per registered row.
    rest: u64,
    spec: ViewSpec,
    transposed: OnceLock<Result<Vec<f32>, String>>,
}

impl GgufView {
    /// Builds and checks a view of `t`.
    pub fn new(g: Arc<Gguf>, t: &TensorInfo, spec: ViewSpec) -> Result<Self, GgufError> {
        let file = g.files[t.shard].path.display().to_string();
        let bad = |m: String| err(&file, format!("tensor {}: {m}", t.name));
        if !t.ty.info().decodable {
            return Err(bad(format!(
                "no host decoder for ggml type {} yet",
                t.ty.name()
            )));
        }
        let rows = t.rows();
        let cols = t.row_len();
        let (first, count) = spec.rows.unwrap_or((0, rows));
        if !v::gg_span_ok(first, count, rows) || !v::gg_limit_ok(1, count) {
            return Err(bad(format!(
                "rows [{first}, +{count}) outside its {rows} rows"
            )));
        }
        let shape = match (&spec.shape, spec.transpose, spec.rows) {
            (Some(s), _, _) => s.clone(),
            (None, true, _) => vec![cols, count],
            (None, false, None) => t.shape_row_major(),
            (None, false, Some(_)) => vec![count, cols],
        };
        let mut n: u64 = 1;
        for &d in &shape {
            if !v::gg_dim_ok(n, d, MAX_ELEMS) {
                return Err(bad(format!(
                    "shape {shape:?} has a zero or too large dimension"
                )));
            }
            n = v::gg_dim_step(n, d);
        }
        if !v::gg_count_fits(count, cols, n)
            || !v::gg_count_fits(n, 1, v::gg_elem_offset(count, cols))
        {
            return Err(bad(format!(
                "shape {shape:?} does not hold {count} rows of {cols}"
            )));
        }
        let rest = if spec.transpose {
            if shape.len() != 2 || shape[0] != cols || shape[1] != count {
                return Err(bad(format!(
                    "a transposed view has shape [{cols}, {count}], not {shape:?}"
                )));
            }
            count
        } else {
            // the registered rows split the values evenly
            if !v::gg_blocks_ok(n, shape[0]) {
                return Err(bad(format!(
                    "shape {shape:?} does not split into {} rows",
                    shape[0]
                )));
            }
            n / shape[0]
        };
        if let Scale::PerOuter(s) = &spec.scale {
            if spec.transpose {
                return Err(bad(
                    "a transposed view takes one scale, not one per outer index".into(),
                ));
            }
            if s.len() as u64 != shape[0] {
                return Err(bad(format!(
                    "{} scales for an outer axis of {}",
                    s.len(),
                    shape[0]
                )));
            }
        }
        if let Some(m) = &spec.row_map {
            if m.len() as u64 != count || m.iter().any(|&r| !v::gg_span_ok(r, 1, count)) {
                return Err(bad("row map does not permute the view's rows".into()));
            }
        }
        if let Some(m) = &spec.col_map {
            if m.len() as u64 != cols || m.iter().any(|&c| !v::gg_span_ok(c as u64, 1, cols)) {
                return Err(bad("column map does not permute the row's columns".into()));
            }
        }
        Ok(Self {
            g,
            t: t.clone(),
            first,
            count,
            shape,
            rest,
            spec,
            transposed: OnceLock::new(),
        })
    }

    pub fn shape(&self) -> &[u64] {
        &self.shape
    }
    pub fn out_type(&self) -> OutType {
        self.spec.out
    }
    /// Values per registered row.
    fn rest(&self) -> u64 {
        self.rest
    }
    fn plain(&self) -> bool {
        self.spec.value_fn.is_none() && self.spec.row_map.is_none() && self.spec.col_map.is_none()
    }
    fn scale_of(&self, outer: u64) -> Option<f32> {
        match &self.spec.scale {
            Scale::None => None,
            Scale::All(s) => Some(*s),
            Scale::PerOuter(s) => Some(s[outer as usize]),
        }
    }
    fn check(&self, r0: u64, r1: u64, have: usize) -> Result<(), String> {
        let rest = self.rest();
        if !v::gg_rows_ok(r0, r1, 1, self.shape[0])
            || !v::gg_count_fits(r1 - r0, rest, have as u64)
            || have as u64 != v::gg_elem_offset(r1 - r0, rest)
        {
            return Err(format!(
                "{}: rows [{r0}, {r1}) of {} into {have} values",
                self.t.name, self.shape[0]
            ));
        }
        Ok(())
    }

    /// One GGUF row of the view (index inside the view) as `f32`, with the maps, the scale
    /// (`scale`, or per value from its registered row when `first_value` gives the row's first
    /// value index) and the value function applied.
    fn gguf_row_f32(
        &self,
        vr: u64,
        scale: Option<f32>,
        out: &mut [f32],
        tmp: &mut Vec<f32>,
        first_value: Option<usize>,
    ) -> Result<(), String> {
        let gr = self.first + self.spec.row_map.as_ref().map_or(vr, |m| m[vr as usize]);
        let e = |e: GgufError| e.to_string();
        match &self.spec.col_map {
            None => gg_rows_f32(&self.g, &self.t, gr, gr + 1, out).map_err(e)?,
            Some(cm) => {
                tmp.resize(out.len(), 0.0);
                gg_rows_f32(&self.g, &self.t, gr, gr + 1, tmp).map_err(e)?;
                for (o, &c) in out.iter_mut().zip(cm.iter()) {
                    *o = tmp[c as usize];
                }
            }
        }
        match (scale, first_value, &self.spec.scale) {
            (Some(s), _, _) => out.iter_mut().for_each(|x| *x *= s),
            (None, Some(v0), Scale::All(s)) => {
                let _ = v0;
                out.iter_mut().for_each(|x| *x *= *s)
            }
            (None, Some(v0), Scale::PerOuter(s)) => {
                let rest = self.rest as usize;
                out.iter_mut()
                    .enumerate()
                    .for_each(|(i, x)| *x *= s[(v0 + i) / rest])
            }
            _ => {}
        }
        if let Some(f) = &self.spec.value_fn {
            f(out);
        }
        Ok(())
    }

    fn transposed(&self) -> Result<&[f32], String> {
        self.transposed
            .get_or_init(|| {
                let cols = self.t.row_len() as usize;
                let n = self.count as usize;
                let mut src = vec![0f32; cols];
                let mut tmp = Vec::new();
                let mut out = vec![0f32; cols * n];
                let scale = self.scale_of(0);
                for r in 0..n {
                    self.gguf_row_f32(r as u64, scale, &mut src, &mut tmp, None)?;
                    for c in 0..cols {
                        out[c * n + r] = src[c];
                    }
                }
                Ok(out)
            })
            .as_ref()
            .map(Vec::as_slice)
            .map_err(Clone::clone)
    }

    /// Registered rows `[r0, r1)` as `f32`.
    pub fn try_f32(&self, r0: u64, r1: u64, out: &mut [f32]) -> Result<(), String> {
        self.check(r0, r1, out.len())?;
        let rest = self.rest() as usize;
        if self.spec.transpose {
            out.copy_from_slice(&self.transposed()?[r0 as usize * rest..r1 as usize * rest]);
            return Ok(());
        }
        // the GGUF rows (of the view) that hold values [e0, e1)
        let cols = self.t.row_len() as usize;
        let e0 = r0 as usize * rest;
        let e1 = r1 as usize * rest;
        let (g0, g1) = (e0 / cols, e1.div_ceil(cols));
        let mut row = vec![0f32; cols];
        let mut tmp = Vec::new();
        for gr in g0..g1 {
            self.gguf_row_f32(gr as u64, None, &mut row, &mut tmp, Some(gr * cols))?;
            let lo = (gr * cols).max(e0);
            let hi = ((gr + 1) * cols).min(e1);
            out[lo - e0..hi - e0].copy_from_slice(&row[lo - gr * cols..hi - gr * cols]);
        }
        Ok(())
    }

    /// Registered rows `[r0, r1)` as BF16 (round to nearest even from the `f32` value).
    pub fn try_bf16(&self, r0: u64, r1: u64, out: &mut [u16]) -> Result<(), String> {
        self.check(r0, r1, out.len())?;
        let cols = self.t.row_len();
        if !self.spec.transpose && self.plain() && self.rest.is_multiple_of(cols) {
            // straight from the decoder: one run of whole GGUF rows per registered row
            let per = self.rest / cols;
            for (i, o) in out.chunks_exact_mut(self.rest as usize).enumerate() {
                let outer = r0 + i as u64;
                let g0 = self.first + v::gg_elem_offset(outer, per);
                gg_rows_bf16(&self.g, &self.t, g0, g0 + per, self.scale_of(outer), o)
                    .map_err(|e| e.to_string())?;
            }
            return Ok(());
        }
        let mut f = vec![0f32; out.len()];
        self.try_f32(r0, r1, &mut f)?;
        out.iter_mut()
            .zip(&f)
            .for_each(|(o, &x)| *o = f32_to_bf16_rne(x));
        Ok(())
    }

    /// The stored bytes when the view is exactly a BF16 tensor's bytes in place (whole rows,
    /// no transform, BF16 output): readers may use them without decoding.
    pub fn bytes_in_place(&self) -> Option<&[u8]> {
        if self.t.ty != GgmlType::BF16
            || self.spec.out != OutType::Bf16
            || !self.plain()
            || self.spec.transpose
            || self.spec.scale != Scale::None
        {
            return None;
        }
        let rb = self.t.row_bytes();
        let data = self.g.tensor_data(&self.t);
        let b0 = v::gg_elem_offset(self.first, rb) as usize;
        let b1 = v::gg_elem_offset(self.first + self.count, rb) as usize;
        Some(&data[b0..b1])
    }
}

impl LazyRows for GgufView {
    fn rows_bf16(&self, r0: u64, r1: u64, out: &mut [u16]) {
        if let Err(e) = self.try_bf16(r0, r1, out) {
            panic!("gguf view: {e}");
        }
    }
    fn try_rows_bf16(&self, r0: u64, r1: u64, out: &mut [u16]) -> Result<(), String> {
        self.try_bf16(r0, r1, out)
    }
    fn try_rows_f32(&self, r0: u64, r1: u64, out: &mut [f32]) -> Result<(), String> {
        self.try_f32(r0, r1, out)
    }
    fn mapped_bytes(&self) -> Option<&[u8]> {
        self.bytes_in_place()
    }
}

/// What [`register`] did.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Registered {
    /// Checkpoint names registered.
    pub names: Vec<String>,
    /// GGUF tensors the map skipped.
    pub skipped: Vec<String>,
}

/// Registers the views `map` asks for into `store`. `map` sees every GGUF tensor once and
/// returns the checkpoint tensors it becomes (an empty list skips it). A name registered twice
/// is refused.
pub fn register<F>(
    store: &mut SafetensorsStore,
    g: &Arc<Gguf>,
    mut map: F,
) -> Result<Registered, GgufError>
where
    F: FnMut(&Gguf, &TensorInfo) -> Result<Vec<(String, ViewSpec)>, GgufError>,
{
    let mut done = Registered::default();
    let src = g.source.display().to_string();
    for t in g.tensors() {
        let views = map(g, t)?;
        if views.is_empty() {
            done.skipped.push(t.name.clone());
        }
        for (name, spec) in views {
            if store.find(&name).is_some() {
                return Err(err(
                    &src,
                    format!(
                        "checkpoint tensor {name} is produced twice (from {})",
                        t.name
                    ),
                ));
            }
            let view = GgufView::new(Arc::clone(g), t, spec)?;
            let dtype = view.out_type().dtype().to_owned();
            let shape = view.shape().to_vec();
            store.add_lazy(name.clone(), dtype, shape, Arc::new(view));
            done.names.push(name);
        }
    }
    Ok(done)
}

/// Every GGUF tensor under its own name, BF16, shape = dims reversed (tensors of a type with
/// no host decoder are skipped).
pub fn register_identity(
    store: &mut SafetensorsStore,
    g: &Arc<Gguf>,
) -> Result<Registered, GgufError> {
    register(store, g, |_, t| {
        Ok(if t.ty.info().decodable {
            vec![(t.name.clone(), ViewSpec::whole())]
        } else {
            Vec::new()
        })
    })
}

/// The per-tensor or per-expert scale stored beside a block-scaled tensor (NVFP4):
/// `<name without .weight>.scale`, F32, one value per outer index. `None` when absent.
///
/// How it combines (reference loader and graph, read for facts): the converter stores the
/// checkpoint's second-level scale (`weight_scale_2`) of each expert in this tensor and keeps
/// the per-16 block scales (UE4M3) inside the NVFP4 blocks; the reference multiplies the
/// product of the decoded weight and the activations by `scale[e]` per expert, so the
/// effective weight is `W[e, r, c] = fp4(q) * ue4m3(d) * scale[e]`. The decoder gives
/// `fp4(q) * ue4m3(d)` exactly in `f32` (`FP4[q] * ue4m3_half(d)`: at most 8 significant bits);
/// a view with `Scale::PerOuter(scale)` multiplies by `scale[e]` in `f32` (one rounding) and
/// rounds the product to BF16 (nearest even). The reference applies the factor to the matmul
/// output instead; the two are equal up to that one `f32` rounding.
pub fn companion_scale(g: &Gguf, weight: &str) -> Result<Option<Arc<[f32]>>, GgufError> {
    let base = weight.strip_suffix(".weight").unwrap_or(weight);
    let name = format!("{base}.scale");
    if g.tensor(&name).is_none() {
        return Ok(None);
    }
    Ok(Some(g.read_f32(&name)?.into()))
}
