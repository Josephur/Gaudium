//! Zero-copy memory-mapped Safetensors multi-shard reader (`safetensors.hpp`).
//!
//! Reads either `model.safetensors.index.json` (multi-shard) or a single `model.safetensors`,
//! validates every JSON header and `[begin, end]` byte range strictly, and exposes zero-copy
//! `&[u8]` tensor views.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::fmt;
use std::fs::{self, File};
use std::io;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::json::{parse_json_bytes, JsonError, JsonValue};

/// Optional lazy row producer for quantised/transformed external containers (e.g. GGUF).
pub trait LazyRows: Send + Sync {
    /// Materialises rows `[r0, r1)` of the `[shape[0], rest]` matrix as row-major BF16.
    fn rows_bf16(&self, r0: u64, r1: u64, out: &mut [u16]);
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SafetensorsError {
    Io(String),
    Json(JsonError),
    Malformed(String),
    OutOfRange {
        tensor: String,
        beg: u64,
        end: u64,
        avail: u64,
    },
    NotFound(String),
}

impl fmt::Display for SafetensorsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(m) => write!(f, "safetensors io error: {m}"),
            Self::Json(e) => write!(f, "safetensors {e}"),
            Self::Malformed(m) => write!(f, "safetensors malformed: {m}"),
            Self::OutOfRange {
                tensor,
                beg,
                end,
                avail,
            } => {
                write!(
                    f,
                    "safetensors offsets [{beg}, {end}) out of range (avail {avail}) for {tensor}"
                )
            }
            Self::NotFound(name) => write!(f, "safetensors tensor not found: {name}"),
        }
    }
}

impl std::error::Error for SafetensorsError {}

impl From<JsonError> for SafetensorsError {
    fn from(e: JsonError) -> Self {
        Self::Json(e)
    }
}

impl From<io::Error> for SafetensorsError {
    fn from(e: io::Error) -> Self {
        Self::Io(e.to_string())
    }
}

/// Safe RAII wrapper over a read-only POSIX `mmap` shard.
pub struct MappedShard {
    ptr: *const u8,
    len: usize,
    _file: File,
}

// SAFETY: Mapping is immutable (`PROT_READ`, `MAP_PRIVATE`) and outlives any borrowed slice tied to `&Store`.
unsafe impl Send for MappedShard {}
unsafe impl Sync for MappedShard {}

#[cfg(unix)]
unsafe extern "C" {
    fn mmap(
        addr: *mut c_void,
        length: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, length: usize) -> i32;
}

const PROT_READ: i32 = 1;
const MAP_PRIVATE: i32 = 2;

impl MappedShard {
    pub fn open(path: &Path) -> Result<Self, SafetensorsError> {
        let file = File::open(path)
            .map_err(|e| SafetensorsError::Io(format!("{}: {e}", path.display())))?;
        let meta = file
            .metadata()
            .map_err(|e| SafetensorsError::Io(format!("stat {}: {e}", path.display())))?;
        let len = usize::try_from(meta.len()).map_err(|_| {
            SafetensorsError::Malformed(format!("file too large: {}", path.display()))
        })?;
        if len < 8 {
            return Err(SafetensorsError::Malformed(format!(
                "file too small (< 8 bytes): {}",
                path.display()
            )));
        }
        // SAFETY: Valid read-only fd, non-zero len, checked against MAP_FAILED (-1isize).
        let raw = unsafe {
            mmap(
                std::ptr::null_mut(),
                len,
                PROT_READ,
                MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if raw as isize == -1 {
            return Err(SafetensorsError::Io(format!(
                "mmap failed for {}: {}",
                path.display(),
                io::Error::last_os_error()
            )));
        }
        Ok(Self {
            ptr: raw as *const u8,
            len,
            _file: file,
        })
    }

    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: `self.ptr` points to `self.len` valid read-only bytes for lifetime of `self`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for MappedShard {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.len > 0 {
            // SAFETY: Matches `mmap` allocation in `MappedShard::open`.
            unsafe {
                munmap(self.ptr as *mut c_void, self.len);
            }
        }
    }
}

#[derive(Clone)]
pub struct TensorMeta {
    pub dtype: String,
    pub shape: Vec<u64>,
    pub shard_idx: Option<usize>,
    pub byte_offset: usize,
    pub nbytes: usize,
    pub shard_name: String,
    pub lazy: Option<Arc<dyn LazyRows>>,
}

impl fmt::Debug for TensorMeta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorMeta")
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .field("nbytes", &self.nbytes)
            .field("shard_name", &self.shard_name)
            .field("is_lazy", &self.lazy.is_some())
            .finish()
    }
}

/// Multi-shard zero-copy Safetensors checkpoint store.
pub struct SafetensorsStore {
    dir: PathBuf,
    shards: Vec<MappedShard>,
    tensors: BTreeMap<String, TensorMeta>,
}

impl SafetensorsStore {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, SafetensorsError> {
        let dir = dir.as_ref().to_path_buf();
        let mut store = Self {
            dir: dir.clone(),
            shards: Vec::new(),
            tensors: BTreeMap::new(),
        };
        let index_path = dir.join("model.safetensors.index.json");
        if index_path.is_file() {
            let raw = fs::read(&index_path)?;
            let parsed = parse_json_bytes(&raw)?;
            let wm = parsed
                .get("weight_map")
                .and_then(JsonValue::as_object)
                .ok_or_else(|| {
                    SafetensorsError::Malformed("index has no weight_map object".into())
                })?;
            let mut shard_names = Vec::new();
            for (_, v) in wm {
                let s = v.as_str().ok_or_else(|| {
                    SafetensorsError::Malformed("weight_map entry is not a string".into())
                })?;
                if !shard_names.iter().any(|x| x == s) {
                    shard_names.push(s.to_owned());
                }
            }
            for s in shard_names {
                store.load_shard(&s)?;
            }
        } else {
            store.load_shard("model.safetensors")?;
        }
        Ok(store)
    }

    /// Creates an empty store for programmatic / GGUF lazy tensor population.
    pub fn empty() -> Self {
        Self {
            dir: PathBuf::new(),
            shards: Vec::new(),
            tensors: BTreeMap::new(),
        }
    }

    pub fn add_lazy(
        &mut self,
        name: String,
        dtype: String,
        shape: Vec<u64>,
        lazy: Arc<dyn LazyRows>,
    ) {
        self.tensors.insert(
            name,
            TensorMeta {
                dtype,
                shape,
                shard_idx: None,
                byte_offset: 0,
                nbytes: 0,
                shard_name: String::new(),
                lazy: Some(lazy),
            },
        );
    }

    fn load_shard(&mut self, file_name: &str) -> Result<(), SafetensorsError> {
        let path = self.dir.join(file_name);
        let mapped = MappedShard::open(&path)?;
        let bytes = mapped.as_bytes();
        let hdr_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let avail_after_len = (bytes.len() - 8) as u64;
        if hdr_len > avail_after_len {
            return Err(SafetensorsError::Malformed(format!(
                "bad header length {hdr_len} > {avail_after_len} in {}",
                path.display()
            )));
        }
        let data0 = 8usize + (hdr_len as usize);
        let avail = (bytes.len() - data0) as u64;
        let header = parse_json_bytes(&bytes[8..data0])?;
        let obj = header
            .as_object()
            .ok_or_else(|| SafetensorsError::Malformed("header is not a JSON object".into()))?;
        let shard_idx = self.shards.len();

        for (name, val) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = val
                .get("dtype")
                .and_then(JsonValue::as_str)
                .ok_or_else(|| SafetensorsError::Malformed(format!("missing dtype for {name}")))?;
            let shape_arr = val
                .get("shape")
                .and_then(JsonValue::as_array)
                .ok_or_else(|| SafetensorsError::Malformed(format!("missing shape for {name}")))?;
            let off_arr = val
                .get("data_offsets")
                .and_then(JsonValue::as_array)
                .filter(|a| a.len() == 2)
                .ok_or_else(|| {
                    SafetensorsError::Malformed(format!("bad data_offsets for {name}"))
                })?;
            let beg = off_arr[0].as_u64().ok_or_else(|| {
                SafetensorsError::Malformed(format!("non-u64 begin offset for {name}"))
            })?;
            let end = off_arr[1].as_u64().ok_or_else(|| {
                SafetensorsError::Malformed(format!("non-u64 end offset for {name}"))
            })?;
            if beg > end || end > avail {
                return Err(SafetensorsError::OutOfRange {
                    tensor: name.clone(),
                    beg,
                    end,
                    avail,
                });
            }
            let mut shape = Vec::with_capacity(shape_arr.len());
            for d in shape_arr {
                shape
                    .push(d.as_u64().ok_or_else(|| {
                        SafetensorsError::Malformed(format!("bad dim in {name}"))
                    })?);
            }
            self.tensors.insert(
                name.clone(),
                TensorMeta {
                    dtype: dtype.to_owned(),
                    shape,
                    shard_idx: Some(shard_idx),
                    byte_offset: data0 + (beg as usize),
                    nbytes: (end - beg) as usize,
                    shard_name: file_name.to_owned(),
                    lazy: None,
                },
            );
        }
        self.shards.push(mapped);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn names(&self) -> Vec<&str> {
        self.tensors.keys().map(String::as_str).collect()
    }

    pub fn find(&self, name: &str) -> Option<&TensorMeta> {
        self.tensors.get(name)
    }

    pub fn get(&self, name: &str) -> Result<&TensorMeta, SafetensorsError> {
        self.find(name)
            .ok_or_else(|| SafetensorsError::NotFound(name.to_owned()))
    }

    /// Returns the zero-copy byte slice of a mapped tensor (`None` for lazy tensors).
    pub fn tensor_bytes(&self, name: &str) -> Result<Option<&[u8]>, SafetensorsError> {
        let meta = self.get(name)?;
        let Some(idx) = meta.shard_idx else {
            return Ok(None);
        };
        let shard = &self.shards[idx];
        Ok(Some(
            &shard.as_bytes()[meta.byte_offset..meta.byte_offset + meta.nbytes],
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn make_safetensors_bytes(header_json: &str, payload: &[u8]) -> Vec<u8> {
        let hb = header_json.as_bytes();
        let mut out = Vec::with_capacity(8 + hb.len() + payload.len());
        out.extend_from_slice(&(hb.len() as u64).to_le_bytes());
        out.extend_from_slice(hb);
        out.extend_from_slice(payload);
        out
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!("gaudium_st_{tag}_{nanos}"));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn reads_multi_shard_safetensors_and_validates_bounds() {
        let dir = temp_dir("multi");
        let idx_json =
            r#"{"weight_map":{"a.weight":"s1.safetensors","b.weight":"s2.safetensors"}}"#;
        fs::write(dir.join("model.safetensors.index.json"), idx_json).unwrap();

        let s1 = make_safetensors_bytes(
            r#"{"__metadata__":{"format":"pt"},"a.weight":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]}}"#,
            &[1, 2, 3, 4, 5, 6, 7, 8],
        );
        let s2 = make_safetensors_bytes(
            r#"{"b.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            &42.0f32.to_le_bytes(),
        );
        fs::write(dir.join("s1.safetensors"), s1).unwrap();
        fs::write(dir.join("s2.safetensors"), s2).unwrap();

        let store = SafetensorsStore::open(&dir).unwrap();
        assert_eq!(store.len(), 2);
        assert_eq!(store.names(), vec!["a.weight", "b.weight"]);
        let a = store.get("a.weight").unwrap();
        assert_eq!(a.dtype, "BF16");
        assert_eq!(a.shape, vec![2, 2]);
        assert_eq!(
            store.tensor_bytes("a.weight").unwrap().unwrap(),
            &[1, 2, 3, 4, 5, 6, 7, 8]
        );
        let b_bytes = store.tensor_bytes("b.weight").unwrap().unwrap();
        assert_eq!(f32::from_le_bytes(b_bytes.try_into().unwrap()), 42.0);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn rejects_out_of_range_tensor_offsets() {
        let dir = temp_dir("oob");
        let bad = make_safetensors_bytes(
            r#"{"bad.weight":{"dtype":"BF16","shape":[8],"data_offsets":[0,16]}}"#,
            &[0u8; 4], // only 4 bytes available!
        );
        fs::write(dir.join("model.safetensors"), bad).unwrap();
        let err = SafetensorsStore::open(&dir).err().unwrap();
        assert!(matches!(err, SafetensorsError::OutOfRange { .. }));
        fs::remove_dir_all(&dir).unwrap();
    }
}
