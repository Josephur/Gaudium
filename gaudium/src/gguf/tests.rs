//! Host tests of the GGUF reader: synthetic files built here, truncated and hostile inputs,
//! split models, the store adapter.

use super::adapter::{self, Scale, ViewSpec};
use super::dequant;
use super::*;
use crate::safetensors::SafetensorsStore;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Builds GGUF files in memory.
pub(crate) struct Builder {
    pub version: u32,
    pub alignment: Option<u64>,
    kv: Vec<u8>,
    nkv: u64,
    tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)>,
}

pub(crate) fn put_str(out: &mut Vec<u8>, s: &[u8]) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s);
}

impl Builder {
    pub fn new() -> Self {
        Self {
            version: 3,
            alignment: None,
            kv: Vec::new(),
            nkv: 0,
            tensors: Vec::new(),
        }
    }
    /// A key with a raw value of type `t`.
    pub fn kv_raw(&mut self, key: &str, t: u32, val: &[u8]) -> &mut Self {
        put_str(&mut self.kv, key.as_bytes());
        self.kv.extend_from_slice(&t.to_le_bytes());
        self.kv.extend_from_slice(val);
        self.nkv += 1;
        self
    }
    pub fn kv_u32(&mut self, key: &str, x: u32) -> &mut Self {
        self.kv_raw(key, 4, &x.to_le_bytes())
    }
    pub fn kv_str(&mut self, key: &str, s: &str) -> &mut Self {
        let mut v = Vec::new();
        put_str(&mut v, s.as_bytes());
        self.kv_raw(key, 8, &v)
    }
    pub fn kv_str_array(&mut self, key: &str, items: &[&str]) -> &mut Self {
        let mut v = 8u32.to_le_bytes().to_vec();
        v.extend_from_slice(&(items.len() as u64).to_le_bytes());
        for s in items {
            put_str(&mut v, s.as_bytes());
        }
        self.kv_raw(key, 9, &v)
    }
    pub fn tensor(&mut self, name: &str, dims: &[u64], ty: GgmlType, data: Vec<u8>) -> &mut Self {
        self.tensors
            .push((name.to_owned(), dims.to_vec(), ty.id(), data));
        self
    }
    pub fn build(&self) -> Vec<u8> {
        let align = self.alignment.unwrap_or(32) as usize;
        let mut out = b"GGUF".to_vec();
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        let extra = self.alignment.is_some() as u64;
        out.extend_from_slice(&(self.nkv + extra).to_le_bytes());
        out.extend_from_slice(&self.kv);
        if let Some(a) = self.alignment {
            put_str(&mut out, b"general.alignment");
            out.extend_from_slice(&4u32.to_le_bytes());
            out.extend_from_slice(&(a as u32).to_le_bytes());
        }
        let mut off = 0usize;
        for (name, dims, ty, data) in &self.tensors {
            put_str(&mut out, name.as_bytes());
            out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in dims {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&ty.to_le_bytes());
            out.extend_from_slice(&(off as u64).to_le_bytes());
            off = (off + data.len()).next_multiple_of(align);
        }
        if !self.tensors.is_empty() {
            out.resize(out.len().next_multiple_of(align), 0);
        }
        for (_, _, _, data) in &self.tensors {
            out.extend_from_slice(data);
            out.resize(out.len().next_multiple_of(align), 0);
        }
        out
    }
}

pub(crate) fn temp_dir(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let p = std::env::temp_dir().join(format!("gaudium_gguf_{tag}_{nanos}_{}", std::process::id()));
    fs::create_dir_all(&p).unwrap();
    p
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn sample() -> Builder {
    let mut b = Builder::new();
    b.kv_str("general.architecture", "toy")
        .kv_u32("toy.block_count", 2)
        .kv_raw("toy.u8", 0, &[7])
        .kv_raw("toy.i8", 1, &[0xFE])
        .kv_raw("toy.u16", 2, &513u16.to_le_bytes())
        .kv_raw("toy.i16", 3, &(-3i16).to_le_bytes())
        .kv_raw("toy.i32", 5, &(-70000i32).to_le_bytes())
        .kv_raw("toy.f32", 6, &1.5f32.to_le_bytes())
        .kv_raw("toy.bool", 7, &[1])
        .kv_raw("toy.u64", 10, &(1u64 << 40).to_le_bytes())
        .kv_raw("toy.i64", 11, &(-5i64).to_le_bytes())
        .kv_raw("toy.f64", 12, &0.25f64.to_le_bytes())
        .kv_str("toy.empty", "")
        .kv_str_array("tokenizer.ggml.tokens", &["a", "bc", "", "déf"]);
    // array of i32
    let mut arr = 5u32.to_le_bytes().to_vec();
    arr.extend_from_slice(&3u64.to_le_bytes());
    for x in [1i32, -2, 3] {
        arr.extend_from_slice(&x.to_le_bytes());
    }
    b.kv_raw("toy.sections", 9, &arr);
    // nested array: [[u8; 2]]
    let mut nest = 9u32.to_le_bytes().to_vec();
    nest.extend_from_slice(&1u64.to_le_bytes());
    nest.extend_from_slice(&0u32.to_le_bytes());
    nest.extend_from_slice(&2u64.to_le_bytes());
    nest.extend_from_slice(&[4, 5]);
    b.kv_raw("toy.nested", 9, &nest);
    let w: Vec<f32> = (0..12).map(|i| i as f32 * 0.5 - 2.0).collect();
    b.tensor("w.f32", &[4, 3], GgmlType::F32, f32_bytes(&w));
    let bf: Vec<u8> = w
        .iter()
        .flat_map(|x| crate::fp8::f32_to_bf16_rne(*x).to_le_bytes())
        .collect();
    b.tensor("w.bf16", &[4, 3], GgmlType::BF16, bf);
    // one Q8_0 row: d = 0.5 (f16 0x3800), q = -16..16
    let mut q = vec![0x00, 0x38];
    q.extend((0..32).map(|i| (i as i8 - 16) as u8));
    b.tensor("w.q8", &[32], GgmlType::Q8_0, q);
    b
}

#[test]
fn reads_metadata_tensors_and_rows() {
    for align in [None, Some(64)] {
        let mut b = sample();
        b.alignment = align;
        let bytes = b.build();
        let (version, alignment, data_start, kvs, tensors) = parse(&bytes, "mem").unwrap();
        assert_eq!(version, 3);
        assert_eq!(alignment, align.unwrap_or(32));
        assert_eq!(data_start % alignment, 0);
        assert!(kvs.iter().any(|(k, _)| k == "toy.nested"));
        assert_eq!(tensors.len(), 3);
        assert_eq!(tensors[0].dims, vec![4, 3]);
        assert_eq!(tensors[0].nbytes, 48);
        assert_eq!(tensors[2].nbytes, 34);
        assert_eq!(tensors[1].offset % alignment, 0);
    }
    let dir = temp_dir("read");
    let p = dir.join("toy.gguf");
    fs::write(&p, sample().build()).unwrap();
    let g = Gguf::open(&p).unwrap();
    assert_eq!(g.arch(), "toy");
    assert_eq!(g.arch_kv("block_count").unwrap().as_u64(), Some(2));
    assert_eq!(g.kv("toy.u8").unwrap().as_u64(), Some(7));
    assert_eq!(g.kv("toy.i8").unwrap().as_i64(), Some(-2));
    assert_eq!(g.kv("toy.u16").unwrap().as_u64(), Some(513));
    assert_eq!(g.kv("toy.i16").unwrap().as_i64(), Some(-3));
    assert_eq!(g.kv("toy.i32").unwrap().as_i64(), Some(-70000));
    assert_eq!(g.kv("toy.f32").unwrap().as_f64(), Some(1.5));
    assert_eq!(g.kv("toy.bool").unwrap().as_bool(), Some(true));
    assert_eq!(g.kv("toy.u64").unwrap().as_u64(), Some(1 << 40));
    assert_eq!(g.kv("toy.i64").unwrap().as_i64(), Some(-5));
    assert_eq!(g.kv("toy.f64").unwrap().as_f64(), Some(0.25));
    assert_eq!(g.get_str("toy.empty"), Some(""));
    let toks = g.kv("tokenizer.ggml.tokens").unwrap().as_array().unwrap();
    assert_eq!(toks.len(), 4);
    let all: Vec<_> = toks.strings().collect();
    assert_eq!(all, vec![Some("a"), Some("bc"), Some(""), Some("déf")]);
    assert_eq!(toks.string_at(1), Some(&b"bc"[..]));
    assert_eq!(toks.string_at(4), None);
    let sec = g.kv("toy.sections").unwrap().as_array().unwrap();
    assert_eq!(sec.to_i64(), Some(vec![1, -2, 3]));
    assert_eq!(sec.get(3), None);
    assert_eq!(
        g.kv("toy.nested").unwrap().as_array().unwrap().elem_type(),
        ValueType::Array
    );

    let t = g.get("w.f32").unwrap();
    assert_eq!(t.shape_row_major(), vec![3, 4]);
    assert_eq!((t.rows(), t.row_len()), (3, 4));
    let mut out = vec![0f32; 8];
    dequant::rows_f32(&g, t, 1, 3, &mut out).unwrap();
    assert_eq!(
        out,
        (4..12).map(|i| i as f32 * 0.5 - 2.0).collect::<Vec<_>>()
    );
    assert!(dequant::rows_f32(&g, t, 2, 4, &mut vec![0f32; 8]).is_err());
    assert!(dequant::rows_f32(&g, t, 2, 1, &mut []).is_err());
    assert!(dequant::rows_f32(&g, t, 0, 1, &mut vec![0f32; 3]).is_err());
    let mut bf = vec![0u16; 12];
    dequant::rows_bf16(&g, g.get("w.bf16").unwrap(), 0, 3, None, &mut bf).unwrap();
    let mut bf2 = vec![0u16; 12];
    dequant::rows_bf16(&g, t, 0, 3, None, &mut bf2).unwrap();
    assert_eq!(bf, bf2);
    let mut q = vec![0f32; 32];
    dequant::rows_f32(&g, g.get("w.q8").unwrap(), 0, 1, &mut q).unwrap();
    assert_eq!(
        q,
        (0..32).map(|i| (i as f32 - 16.0) * 0.5).collect::<Vec<_>>()
    );
    assert_eq!(g.read_f32("w.f32").unwrap().len(), 12);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn every_truncation_is_refused() {
    let bytes = sample().build();
    // the data section ends with the last tensor's padding: every shorter prefix that cuts the
    // header, the index or a tensor's bytes must be refused
    let (_, _, data_start, _, tensors) = parse(&bytes, "mem").unwrap();
    let last_end = tensors
        .iter()
        .map(|t| data_start + t.offset + t.nbytes)
        .max()
        .unwrap();
    for n in 0..last_end as usize {
        assert!(
            parse(&bytes[..n], "mem").is_err(),
            "prefix of {n} bytes was accepted"
        );
    }
}

fn header(nt: u64, nkv: u64) -> Vec<u8> {
    let mut b = b"GGUF".to_vec();
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&nt.to_le_bytes());
    b.extend_from_slice(&nkv.to_le_bytes());
    b
}

fn tensor_entry(name: &str, dims: &[u64], ty: u32, off: u64) -> Vec<u8> {
    let mut b = Vec::new();
    put_str(&mut b, name.as_bytes());
    b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for d in dims {
        b.extend_from_slice(&d.to_le_bytes());
    }
    b.extend_from_slice(&ty.to_le_bytes());
    b.extend_from_slice(&off.to_le_bytes());
    b
}

fn refused(bytes: &[u8], needle: &str) {
    match parse(bytes, "mem") {
        Ok(_) => panic!("accepted a file that should name '{needle}'"),
        Err(e) => assert!(
            e.to_string().contains(needle),
            "{e} does not name '{needle}'"
        ),
    }
}

#[test]
fn hostile_inputs_are_refused_with_the_culprit_named() {
    refused(b"GGUX\x03\0\0\0", "magic");
    let mut v1 = header(0, 0);
    v1[4] = 1;
    refused(&v1, "version 1");
    let mut be = header(0, 0);
    be[4..8].copy_from_slice(&0x0300_0000u32.to_le_bytes());
    refused(&be, "big-endian");
    refused(&header(u64::MAX, 0), "cannot fit");
    refused(&header(0, u64::MAX / 2), "cannot fit");

    // a key whose string length runs past the end
    let mut b = header(0, 1);
    b.extend_from_slice(&u64::MAX.to_le_bytes());
    b.extend_from_slice(&[0u8; 32]);
    refused(&b, "key 0");
    // a string value whose length runs past the end
    let mut b = header(0, 1);
    put_str(&mut b, b"k.str");
    b.extend_from_slice(&8u32.to_le_bytes());
    b.extend_from_slice(&(u64::MAX - 3).to_le_bytes());
    refused(&b, "k.str");
    // a numeric array whose count times element size overflows
    let mut b = header(0, 1);
    put_str(&mut b, b"k.arr");
    b.extend_from_slice(&9u32.to_le_bytes());
    b.extend_from_slice(&10u32.to_le_bytes());
    b.extend_from_slice(&(u64::MAX / 4).to_le_bytes());
    refused(&b, "k.arr");
    // a string array with a huge count
    let mut b = header(0, 1);
    put_str(&mut b, b"k.toks");
    b.extend_from_slice(&9u32.to_le_bytes());
    b.extend_from_slice(&8u32.to_le_bytes());
    b.extend_from_slice(&u64::MAX.to_le_bytes());
    refused(&b, "k.toks");
    // nesting too deep
    let mut b = header(0, 1);
    put_str(&mut b, b"k.deep");
    b.extend_from_slice(&9u32.to_le_bytes());
    for _ in 0..8 {
        b.extend_from_slice(&9u32.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
    }
    b.extend_from_slice(&[0u8; 64]);
    refused(&b, "nested too deeply");
    // bad value type
    let mut b = header(0, 1);
    put_str(&mut b, b"k.bad");
    b.extend_from_slice(&13u32.to_le_bytes());
    b.extend_from_slice(&[0u8; 16]);
    refused(&b, "k.bad");
    // a duplicate key
    let mut d = Builder::new();
    d.kv_u32("k", 1).kv_u32("k", 2);
    refused(&d.build(), "appears twice");

    // bad alignments
    for a in [0u64, 3, 48] {
        let mut b = sample();
        b.alignment = Some(a.max(1));
        let mut bytes = b.build();
        if a == 0 {
            // the builder cannot lay out with 0: patch a valid file's value
            let mut ok = sample();
            ok.alignment = Some(32);
            bytes = ok.build();
            let pos = bytes
                .windows(17)
                .position(|w| w == b"general.alignment")
                .unwrap()
                + 17
                + 4;
            bytes[pos..pos + 4].copy_from_slice(&0u32.to_le_bytes());
        }
        refused(&bytes, "general.alignment");
    }

    let tail = |entry: Vec<u8>, data: usize| {
        let mut b = header(1, 0);
        b.extend_from_slice(&entry);
        b.resize(b.len().next_multiple_of(32) + data, 0);
        b
    };
    // dims whose product overflows
    refused(
        &tail(tensor_entry("t.big", &[1 << 32, 1 << 32, 4], 0, 0), 64),
        "t.big",
    );
    // a zero dim
    refused(&tail(tensor_entry("t.zero", &[4, 0], 0, 0), 64), "t.zero");
    // rank 5 and rank 0
    refused(
        &tail(tensor_entry("t.rank", &[1, 1, 1, 1, 1], 0, 0), 64),
        "rank 5",
    );
    refused(&tail(tensor_entry("t.rank0", &[], 0, 0), 64), "rank 0");
    // unknown type
    refused(
        &tail(tensor_entry("t.ty", &[4], 4, 0), 64),
        "unknown ggml type 4",
    );
    // innermost dim not a multiple of the block
    refused(
        &tail(tensor_entry("t.blk", &[33], 8, 0), 64),
        "not a multiple of the Q8_0 block",
    );
    // data past the end
    refused(&tail(tensor_entry("t.end", &[16], 0, 32), 64), "t.end");
    refused(
        &tail(tensor_entry("t.huge", &[16], 0, u64::MAX - 8), 64),
        "t.huge",
    );
    // data larger than the section
    refused(
        &tail(tensor_entry("t.size", &[1 << 40], 0, 0), 64),
        "t.size",
    );
    // misaligned offset
    refused(
        &tail(tensor_entry("t.mis", &[4], 0, 4), 64),
        "not a multiple of the alignment",
    );
}

#[test]
fn damaged_copies_never_panic() {
    let good = sample().build();
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    for _ in 0..3000 {
        let mut b = good.clone();
        for _ in 0..1 + next() % 4 {
            let at = (next() % b.len() as u64) as usize;
            match next() % 3 {
                0 => b[at] = next() as u8,
                1 => b[at] ^= 0x80,
                _ => {
                    let end = (at + 8).min(b.len());
                    b[at..end].iter_mut().for_each(|x| *x = 0xFF);
                }
            }
        }
        let cut = if next() % 4 == 0 {
            (next() % b.len() as u64) as usize
        } else {
            b.len()
        };
        if let Ok((_, _, _, kvs, tensors)) = parse(&b[..cut], "mem") {
            // anything accepted must read back inside the buffer
            for t in &tensors {
                assert!(t.abs + t.nbytes <= cut as u64);
            }
            assert!(kvs.len() <= b.len());
        }
    }
}

#[test]
fn split_models_and_directories() {
    let dir = temp_dir("split");
    let mk = |no: u16, count: u16, total: i32, name: &str, arch: bool| {
        let mut b = Builder::new();
        if arch {
            b.kv_str("general.architecture", "toy");
        }
        b.kv_raw("split.no", 2, &no.to_le_bytes())
            .kv_raw("split.count", 2, &count.to_le_bytes())
            .kv_raw("split.tensors.count", 5, &total.to_le_bytes())
            .tensor(name, &[4], GgmlType::F32, f32_bytes(&[no as f32; 4]));
        b.build()
    };
    fs::write(dir.join("m-00001-of-00002.gguf"), mk(0, 2, 2, "a", true)).unwrap();
    fs::write(dir.join("m-00002-of-00002.gguf"), mk(1, 2, 2, "b", false)).unwrap();
    let g = Gguf::open(dir.join("m-00002-of-00002.gguf")).unwrap();
    assert_eq!(g.files.len(), 2);
    assert_eq!(g.arch(), "toy");
    assert_eq!(g.tensor_count(), 2);
    assert_eq!(g.read_f32("b").unwrap(), vec![1.0; 4]);
    assert_eq!(g.get("b").unwrap().shard, 1);
    // the directory: the same model; a projector beside it is set aside
    let mut proj = Builder::new();
    proj.kv_str("general.architecture", "clip");
    fs::write(dir.join("mmproj.gguf"), proj.build()).unwrap();
    let gd = Gguf::open(&dir).unwrap();
    assert_eq!(gd.tensor_count(), 2);
    assert_eq!(gd.side_files.len(), 1);
    // a block drafter beside it is set aside too (its model's loader finds it there)
    let mut drafter = Builder::new();
    drafter.kv_str("general.architecture", "dflash");
    fs::write(dir.join("drafter.gguf"), drafter.build()).unwrap();
    let gd = Gguf::open(&dir).unwrap();
    assert_eq!(gd.tensor_count(), 2);
    assert_eq!(gd.side_files.len(), 2);
    fs::remove_file(dir.join("drafter.gguf")).unwrap();
    assert!(looks_like_gguf(&dir));
    // a wrong total, a missing shard, a duplicate name, two unrelated models
    fs::write(dir.join("m-00001-of-00002.gguf"), mk(0, 2, 3, "a", true)).unwrap();
    assert!(Gguf::open(dir.join("m-00001-of-00002.gguf"))
        .unwrap_err()
        .msg
        .contains("split.tensors.count"));
    fs::write(dir.join("m-00001-of-00002.gguf"), mk(0, 2, 2, "a", true)).unwrap();
    fs::write(dir.join("m-00002-of-00002.gguf"), mk(1, 2, 2, "a", false)).unwrap();
    assert!(Gguf::open(dir.join("m-00001-of-00002.gguf"))
        .unwrap_err()
        .msg
        .contains("appears twice"));
    fs::write(dir.join("m-00002-of-00002.gguf"), mk(0, 2, 2, "b", false)).unwrap();
    assert!(Gguf::open(dir.join("m-00001-of-00002.gguf"))
        .unwrap_err()
        .msg
        .contains("one set"));
    fs::remove_file(dir.join("m-00002-of-00002.gguf")).unwrap();
    assert!(Gguf::open(dir.join("m-00001-of-00002.gguf")).is_err());
    assert!(Gguf::open(&dir).unwrap_err().msg.contains("1 of 2 shards"));
    fs::remove_file(dir.join("m-00001-of-00002.gguf")).unwrap();
    fs::write(dir.join("x.gguf"), sample().build()).unwrap();
    fs::write(dir.join("y.gguf"), sample().build()).unwrap();
    assert!(Gguf::open(&dir).unwrap_err().msg.contains("unrelated"));
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn adapter_registers_views_by_checkpoint_name() {
    let dir = temp_dir("adapter");
    let p = dir.join("m.gguf");
    // stacked [E=2, R=3, C=4] F32 experts, its per-expert scale, a 2x2 norm stored as 1 + w
    let w: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let mut b = Builder::new();
    b.kv_str("general.architecture", "toy")
        .tensor(
            "blk.0.ffn_up_exps.weight",
            &[4, 3, 2],
            GgmlType::F32,
            f32_bytes(&w),
        )
        .tensor(
            "blk.0.ffn_up_exps.scale",
            &[2],
            GgmlType::F32,
            f32_bytes(&[0.5, 2.0]),
        )
        .tensor(
            "norm",
            &[2, 2],
            GgmlType::F32,
            f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
        )
        .tensor(
            "emb",
            &[4, 3],
            GgmlType::BF16,
            (0..12u16)
                .flat_map(|x| (0x3F80 + x).to_le_bytes())
                .collect(),
        );
    fs::write(&p, b.build()).unwrap();
    let g = Arc::new(Gguf::open(&p).unwrap());
    let scale = adapter::companion_scale(&g, "blk.0.ffn_up_exps.weight")
        .unwrap()
        .unwrap();
    assert_eq!(&scale[..], &[0.5, 2.0]);
    assert!(adapter::companion_scale(&g, "norm").unwrap().is_none());

    let mut store = SafetensorsStore::empty();
    let sc = scale.clone();
    let done = adapter::register(&mut store, &g, |_, t| {
        Ok(match t.name.as_str() {
            "blk.0.ffn_up_exps.weight" => vec![
                (
                    "experts.up".into(),
                    ViewSpec::whole().per_outer_scale(sc.clone()),
                ),
                (
                    "experts.1.up".into(),
                    ViewSpec::expert(1, 3).scaled(sc[1]).f32(),
                ),
            ],
            "norm" => vec![
                (
                    "norm.w".into(),
                    ViewSpec::whole().map_values(Arc::new(|r: &mut [f32]| {
                        r.iter_mut().for_each(|x| *x -= 1.0)
                    })),
                ),
                (
                    "norm.t".into(),
                    ViewSpec {
                        transpose: true,
                        out: adapter::OutType::F32,
                        ..ViewSpec::whole()
                    },
                ),
                (
                    "norm.cols".into(),
                    ViewSpec {
                        col_map: Some(Arc::from(vec![1u32, 0])),
                        row_map: Some(Arc::from(vec![1u64, 0])),
                        ..ViewSpec::whole().f32()
                    },
                ),
            ],
            "emb" => vec![("emb".into(), ViewSpec::whole())],
            _ => vec![],
        })
    })
    .unwrap();
    assert_eq!(done.skipped, vec!["blk.0.ffn_up_exps.scale".to_owned()]);
    let m = store.get("experts.up").unwrap();
    assert_eq!((m.dtype.as_str(), m.shape.clone()), ("BF16", vec![2, 3, 4]));
    let all = store.tensor_data("experts.up").unwrap();
    let vals: Vec<f32> = all
        .chunks_exact(2)
        .map(|c| crate::fp8::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let want: Vec<f32> = (0..24)
        .map(|i| i as f32 * if i < 12 { 0.5 } else { 2.0 })
        .collect();
    assert_eq!(vals, want);
    let e1 = store.get("experts.1.up").unwrap();
    assert_eq!((e1.dtype.as_str(), e1.shape.clone()), ("F32", vec![3, 4]));
    let mut row = vec![0f32; 4];
    e1.lazy
        .as_ref()
        .unwrap()
        .try_rows_f32(2, 3, &mut row)
        .unwrap();
    assert_eq!(row, vec![40.0, 42.0, 44.0, 46.0]);
    assert!(e1
        .lazy
        .as_ref()
        .unwrap()
        .try_rows_f32(2, 4, &mut vec![0f32; 8])
        .is_err());
    let f = |n: &str| -> Vec<f32> {
        store
            .tensor_data(n)
            .unwrap()
            .chunks_exact(if store.get(n).unwrap().dtype == "F32" {
                4
            } else {
                2
            })
            .map(|c| {
                if c.len() == 4 {
                    f32::from_le_bytes(c.try_into().unwrap())
                } else {
                    crate::fp8::bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))
                }
            })
            .collect()
    };
    assert_eq!(f("norm.w"), vec![0.0, 1.0, 2.0, 3.0]);
    assert_eq!(f("norm.t"), vec![1.0, 3.0, 2.0, 4.0]);
    assert_eq!(f("norm.cols"), vec![4.0, 3.0, 2.0, 1.0]);
    // a plain BF16 tensor is served in place, without decoding
    assert!(matches!(
        store.tensor_data("emb").unwrap(),
        std::borrow::Cow::Borrowed(_)
    ));
    assert!(matches!(
        store.tensor_data("norm.w").unwrap(),
        std::borrow::Cow::Owned(_)
    ));

    // views that do not fit are refused when they are registered
    let t = g.get("blk.0.ffn_up_exps.weight").unwrap();
    assert!(adapter::GgufView::new(g.clone(), t, ViewSpec::expert(2, 3)).is_err());
    assert!(
        adapter::GgufView::new(g.clone(), t, ViewSpec::whole().with_shape(vec![5, 4])).is_err()
    );
    // registered rows need not be whole GGUF rows: [4, 6] over rows of 4
    let odd = adapter::GgufView::new(g.clone(), t, ViewSpec::whole().with_shape(vec![4, 6]).f32())
        .unwrap();
    let mut r = vec![0f32; 6];
    odd.try_f32(1, 2, &mut r).unwrap();
    assert_eq!(r, (6..12).map(|i| i as f32).collect::<Vec<_>>());
    assert!(adapter::GgufView::new(
        g.clone(),
        t,
        ViewSpec {
            scale: Scale::PerOuter(Arc::from(vec![1.0f32; 3])),
            ..ViewSpec::whole()
        }
    )
    .is_err());
    assert!(adapter::GgufView::new(g.clone(), t, ViewSpec::rows(u64::MAX, 2)).is_err());
    let mut s2 = SafetensorsStore::empty();
    assert!(adapter::register(&mut s2, &g, |_, t| Ok(vec![
        (
            "same".into(),
            ViewSpec::whole()
        );
        (t.name == "norm") as usize * 2
    ]))
    .is_err());
    let idn = adapter::register_identity(&mut SafetensorsStore::empty(), &g).unwrap();
    assert_eq!(idn.names.len(), 4);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn parallel_decode_matches_single_thread() {
    let dir = temp_dir("par");
    let p = dir.join("m.gguf");
    let mut q = Vec::new();
    for r in 0..37u32 {
        q.extend_from_slice(&[0x00, 0x3C]); // d = 1.0
        q.extend((0..32).map(|i| ((r * 7 + i) % 251) as u8));
    }
    let mut b = Builder::new();
    b.tensor("q", &[32, 37], GgmlType::Q8_0, q);
    fs::write(&p, b.build()).unwrap();
    let g = Gguf::open(&p).unwrap();
    let t = g.get("q").unwrap();
    let mut one = vec![0u16; 32 * 35];
    dequant::rows_bf16(&g, t, 1, 36, Some(0.25), &mut one).unwrap();
    for threads in [1, 2, 3, 8, 64] {
        let mut par = vec![0u16; 32 * 35];
        dequant::rows_bf16_par(&g, t, 1, 36, Some(0.25), threads, &mut par).unwrap();
        assert_eq!(par, one, "threads={threads}");
    }
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn verified_padding_lands_on_the_alignment() {
    // pins the padding formula the reader uses (the law only bounds it)
    for a in [1u64, 2, 8, 32, 64, 4096] {
        for x in 0..300u64 {
            let p = v::gg_pad(x, a);
            assert!(p < a);
            assert_eq!((x + p) % a, 0, "x={x} a={a}");
        }
    }
    assert!(v::gg_alignment_ok(32, MAX_ALIGNMENT));
    assert!(!v::gg_alignment_ok(48, MAX_ALIGNMENT));
    assert!(!v::gg_alignment_ok(MAX_ALIGNMENT * 2, MAX_ALIGNMENT));
    assert_eq!(v::gg_part_chunk(10, 3), 4);
    assert_eq!(
        (0..4).map(|i| v::gg_part_lo(i, 10, 4)).collect::<Vec<_>>(),
        vec![0, 4, 8, 10]
    );
}

/// The published Inkling 2 Mini GGUF build, when present (`GAUDIUM_GGUF_REF_DIR` = its
/// directory; run with `--ignored`): every file there opens, the main model has the expected
/// tensor table, every NVFP4 tensor has its per-expert scale, and the first row of one tensor
/// of every type decodes to finite numbers.
#[test]
#[ignore]
fn reference_gguf_files_open_and_decode() {
    let Some(dir) = std::env::var_os("GAUDIUM_GGUF_REF_DIR").map(PathBuf::from) else {
        eprintln!("GAUDIUM_GGUF_REF_DIR is not set: nothing to check");
        return;
    };
    let mut files = list_gguf(&dir).unwrap();
    if dir.join("dspark").is_dir() {
        files.extend(list_gguf(&dir.join("dspark")).unwrap());
    }
    assert!(!files.is_empty(), "no .gguf file in {}", dir.display());
    for p in files {
        let g = Gguf::open(&p).unwrap_or_else(|e| panic!("{e}"));
        let counts = super::inspect::type_counts(&g);
        eprintln!(
            "{}: arch {:?}, {} tensors, {counts:?}",
            p.display(),
            g.arch(),
            g.tensor_count()
        );
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        if name.starts_with("Inkling-2-Mini-") && !name.starts_with("mmproj") {
            assert_eq!(g.arch(), "inkling2_loop");
            assert_eq!(g.tensor_count(), 744);
            assert!(counts.get("NVFP4").is_some_and(|c| c.params > 0));
        }
        let mut done = std::collections::BTreeSet::new();
        for t in g.tensors() {
            if t.ty == GgmlType::NVFP4 {
                let s = adapter::companion_scale(&g, &t.name).unwrap();
                let s = s.unwrap_or_else(|| panic!("{}: no .scale", t.name));
                assert_eq!(s.len() as u64, *t.dims.last().unwrap(), "{}", t.name);
                assert!(s.iter().all(|x| x.is_finite() && *x > 0.0), "{}", t.name);
                if t.name.starts_with("blk.0.") {
                    let mut row = vec![0f32; t.row_len() as usize];
                    dequant::rows_f32(&g, t, 0, 1, &mut row).unwrap();
                    let amax = row.iter().fold(0f32, |m, x| m.max(x.abs()));
                    eprintln!(
                        "  {} dims {:?}: scale[0..3] {:?}, row 0 |max| {amax} before the scale",
                        t.name,
                        t.dims,
                        &s[..3]
                    );
                }
            }
            if t.ty.info().decodable && done.insert(t.ty) {
                let mut row = vec![0f32; t.row_len() as usize];
                dequant::rows_f32(&g, t, 0, 1, &mut row).unwrap();
                assert!(
                    row.iter().all(|x| x.is_finite()),
                    "{} ({})",
                    t.name,
                    t.ty.name()
                );
            }
        }
    }
}

#[test]
fn inspect_reports_and_refuses() {
    let dir = temp_dir("inspect");
    let p = dir.join("toy.gguf");
    fs::write(&p, sample().build()).unwrap();
    let r = super::inspect::inspect(&p).unwrap();
    assert!(r.contains("architecture: \"toy\""), "{r}");
    assert!(r.contains("toy.block_count = 2"), "{r}");
    assert!(r.contains("tokenizer: ? (4 tokens)"), "{r}");
    assert!(
        r.contains("Q8_0") && r.contains("total         3 tensors"),
        "{r}"
    );
    let mut bad = sample().build();
    bad.truncate(bad.len() - 40);
    fs::write(&p, &bad).unwrap();
    let e = super::inspect::inspect(&p).unwrap_err().to_string();
    assert!(e.contains("w.q8") || e.contains("w.bf16"), "{e}");
    fs::remove_dir_all(&dir).unwrap();
}

/// A tiny synthetic `qwen35` GGUF maps to every name the Qwen BF16 packer reads, with the
/// converter's transforms undone.
#[test]
fn qwen35_map_gives_the_packer_names() {
    use super::qwen35;
    let dir = temp_dir("qwen35");
    let p = dir.join("q.gguf");
    // H = 4, 2 key heads, 4 value heads (r = 2) of dim 1, key dim 1: qk = 4, qkv rows = 8
    let (h, hk, hv) = (4u64, 2u64, 4u64);
    let seq = |n: u64, k: f32| f32_bytes(&(0..n).map(|i| i as f32 + k).collect::<Vec<_>>());
    let mut b = Builder::new();
    b.kv_str("general.architecture", "qwen35")
        .kv_u32("qwen35.block_count", 3)
        .kv_u32("qwen35.nextn_predict_layers", 1)
        .kv_u32("qwen35.embedding_length", h as u32)
        .kv_u32("qwen35.ssm.group_count", hk as u32)
        .kv_u32("qwen35.ssm.time_step_rank", hv as u32)
        .kv_u32("qwen35.ssm.state_size", 1)
        .kv_u32("qwen35.ssm.inner_size", hv as u32);
    let m = |b: &mut Builder, name: &str, cols: u64, rows: u64| {
        b.tensor(name, &[cols, rows], GgmlType::F32, seq(cols * rows, 0.0));
    };
    let v = |b: &mut Builder, name: &str, n: u64| {
        b.tensor(name, &[n], GgmlType::F32, seq(n, 1.0));
    };
    m(&mut b, "token_embd.weight", h, 16);
    m(&mut b, "output.weight", h, 16);
    v(&mut b, "output_norm.weight", h);
    for l in 0..3 {
        let x = |r: &str| format!("blk.{l}.{r}");
        v(&mut b, &x("attn_norm.weight"), h);
        v(&mut b, &x("post_attention_norm.weight"), h);
        m(&mut b, &x("ffn_gate.weight"), h, 8);
        m(&mut b, &x("ffn_up.weight"), h, 8);
        m(&mut b, &x("ffn_down.weight"), 8, h);
        if l == 0 {
            m(&mut b, &x("attn_qkv.weight"), h, 8);
            m(&mut b, &x("attn_gate.weight"), h, hv);
            m(&mut b, &x("ssm_beta.weight"), h, hv);
            m(&mut b, &x("ssm_alpha.weight"), h, hv);
            m(&mut b, &x("ssm_out.weight"), hv, h);
            m(&mut b, &x("ssm_conv1d.weight"), 4, 8);
            // A_log stored as -exp(A_log)
            let a: Vec<f32> = (0..hv).map(|i| -((i as f32 * 0.5).exp())).collect();
            b.tensor(&x("ssm_a"), &[hv], GgmlType::F32, f32_bytes(&a));
            v(&mut b, &x("ssm_dt.bias"), hv);
            v(&mut b, &x("ssm_norm.weight"), 1);
        } else {
            m(&mut b, &x("attn_q.weight"), h, 8);
            m(&mut b, &x("attn_k.weight"), h, 4);
            m(&mut b, &x("attn_v.weight"), h, 4);
            m(&mut b, &x("attn_output.weight"), 4, h);
            v(&mut b, &x("attn_q_norm.weight"), 4);
            v(&mut b, &x("attn_k_norm.weight"), 4);
        }
        if l == 2 {
            m(&mut b, &x("nextn.eh_proj.weight"), 2 * h, h);
            v(&mut b, &x("nextn.enorm.weight"), h);
            v(&mut b, &x("nextn.hnorm.weight"), h);
            v(&mut b, &x("nextn.shared_head_norm.weight"), h);
        }
    }
    fs::write(&p, b.build()).unwrap();
    let g = Arc::new(Gguf::open(&p).unwrap());
    let mut store = SafetensorsStore::empty();
    let done = qwen35::register_qwen35(&mut store, &g).unwrap();
    assert!(done.skipped.is_empty(), "{:?}", done.skipped);
    let lp = "model.language_model.";
    let mut want = vec![
        format!("{lp}embed_tokens.weight"),
        "lm_head.weight".into(),
        format!("{lp}norm.weight"),
        "mtp.fc.weight".into(),
        "mtp.pre_fc_norm_embedding.weight".into(),
        "mtp.pre_fc_norm_hidden.weight".into(),
        "mtp.norm.weight".into(),
    ];
    for (pre, lin) in [
        (format!("{lp}layers.0."), true),
        (format!("{lp}layers.1."), false),
        ("mtp.layers.0.".to_owned(), false),
    ] {
        for r in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            want.push(format!("{pre}{r}"));
        }
        let roles: &[&str] = if lin {
            &[
                "linear_attn.in_proj_qkv.weight",
                "linear_attn.in_proj_z.weight",
                "linear_attn.in_proj_b.weight",
                "linear_attn.in_proj_a.weight",
                "linear_attn.conv1d.weight",
                "linear_attn.A_log",
                "linear_attn.dt_bias",
                "linear_attn.norm.weight",
                "linear_attn.out_proj.weight",
            ]
        } else {
            &[
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
            ]
        };
        want.extend(roles.iter().map(|r| format!("{pre}{r}")));
    }
    for n in &want {
        let t = store.get(n).unwrap_or_else(|_| panic!("missing {n}"));
        assert_eq!(t.dtype, "BF16", "{n}");
    }
    assert_eq!(store.len(), want.len());
    let vals = |n: &str| -> Vec<f32> {
        store
            .tensor_data(n)
            .unwrap()
            .chunks_exact(2)
            .map(|c| crate::fp8::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()
    };
    // norms come back as w (stored 1 + w = 1, 2, 3, 4)
    assert_eq!(vals(&format!("{lp}norm.weight")), vec![0.0, 1.0, 2.0, 3.0]);
    assert_eq!(
        vals(&format!("{lp}layers.0.linear_attn.norm.weight")),
        vec![1.0]
    );
    // A_log = ln(-x)
    let a = vals(&format!("{lp}layers.0.linear_attn.A_log"));
    let want_a: Vec<f32> = [0.0f32, 2.0, 1.0, 3.0]
        .iter()
        .map(|i| crate::fp8::bf16_to_f32(crate::fp8::f32_to_bf16_rne(i * 0.5)))
        .collect();
    assert_eq!(a, want_a);
    // value heads: checkpoint (k, r) -> stored (r, k): with 2 x 2 heads of dim 1, entries 1 and 2 swap
    assert_eq!(
        vals(&format!("{lp}layers.0.linear_attn.dt_bias")),
        vec![1.0, 3.0, 2.0, 4.0]
    );
    let qkv = vals(&format!("{lp}layers.0.linear_attn.in_proj_qkv.weight"));
    let row = |k: usize| qkv[4 * k];
    assert_eq!(
        (0..8).map(row).collect::<Vec<_>>(),
        vec![0.0, 4.0, 8.0, 12.0, 16.0, 24.0, 20.0, 28.0]
    );
    let out = vals(&format!("{lp}layers.0.linear_attn.out_proj.weight"));
    assert_eq!(&out[..4], &[0.0, 2.0, 1.0, 3.0]);
    assert_eq!(
        store
            .get(&format!("{lp}layers.0.linear_attn.conv1d.weight"))
            .unwrap()
            .shape,
        vec![8, 1, 4]
    );
    // a model of another family, or inconsistent metadata, is refused
    let mut o = Builder::new();
    o.kv_str("general.architecture", "llama");
    fs::write(&p, o.build()).unwrap();
    assert!(qwen35::dims(&Gguf::open(&p).unwrap()).is_err());
    // an inner size of 0, or none, leaves the value heads empty, so it is refused up front
    for inner in [Some(0u32), None] {
        let mut z = Builder::new();
        z.kv_str("general.architecture", "qwen35")
            .kv_u32("qwen35.block_count", 3)
            .kv_u32("qwen35.nextn_predict_layers", 1)
            .kv_u32("qwen35.embedding_length", h as u32)
            .kv_u32("qwen35.ssm.group_count", hk as u32)
            .kv_u32("qwen35.ssm.time_step_rank", hv as u32)
            .kv_u32("qwen35.ssm.state_size", 1);
        if let Some(i) = inner {
            z.kv_u32("qwen35.ssm.inner_size", i);
        }
        fs::write(&p, z.build()).unwrap();
        assert!(
            qwen35::dims(&Gguf::open(&p).unwrap()).is_err(),
            "inner size {inner:?}"
        );
    }
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn split_names_are_read_on_bytes() {
    // a shard name, and names that only look like one
    let p = |s: &str| split_name(Path::new(s));
    assert_eq!(p("m-00002-of-00003.gguf"), Some(("m".to_string(), 2, 3)));
    assert_eq!(p("dir/é-00001-of-00001.gguf"), Some(("dir/é".to_string(), 1, 1)));
    assert_eq!(p("m-0000x-of-00003.gguf"), None);
    assert_eq!(p("m-+0001-of-00003.gguf"), None);
    // the last 20 bytes start with '-' but cut the 'é' characters: an unsplit file, no panic
    assert_eq!(p("-aaaaééabcdef.gguf"), None);
    assert_eq!(p("short.gguf"), None);
}
