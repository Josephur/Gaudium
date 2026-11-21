//! Compiled-recipe cache, zero-dependency SHA-256, build-path-independent TPC ELF hashing,
//! and recipe placement / binding guards (`recipe_cache.hpp`).

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

// ============================================================================
// 1. Pure Rust SHA-256 (FIPS 180-4, zero external dependencies)
// ============================================================================

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

#[derive(Clone)]
pub struct Sha256 {
    h: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    total_bits: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    pub fn new() -> Self {
        Self {
            h: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buf: [0; 64],
            buf_len: 0,
            total_bits: 0,
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.total_bits = self
            .total_bits
            .wrapping_add((data.len() as u64).wrapping_mul(8));
        if self.buf_len > 0 {
            let take = (64 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                let blk = self.buf;
                self.compress(&blk);
                self.buf_len = 0;
            }
        }
        while data.len() >= 64 {
            let blk: [u8; 64] = data[..64].try_into().unwrap();
            self.compress(&blk);
            data = &data[64..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
    }

    pub fn update_u64(&mut self, v: u64) {
        self.update(&v.to_le_bytes());
    }

    pub fn update_str(&mut self, s: &str) {
        self.update_u64(s.len() as u64);
        self.update(s.as_bytes());
    }

    pub fn finalize(mut self) -> [u8; 32] {
        self.buf[self.buf_len] = 0x80;
        self.buf_len += 1;
        if self.buf_len > 56 {
            self.buf[self.buf_len..64].fill(0);
            let blk = self.buf;
            self.compress(&blk);
            self.buf_len = 0;
        }
        self.buf[self.buf_len..56].fill(0);
        self.buf[56..64].copy_from_slice(&self.total_bits.to_be_bytes());
        let blk = self.buf;
        self.compress(&blk);

        let mut out = [0u8; 32];
        for (i, &w) in self.h.iter().enumerate() {
            out[4 * i..4 * i + 4].copy_from_slice(&w.to_be_bytes());
        }
        out
    }

    pub fn finalize_hex(self) -> String {
        let d = self.finalize();
        let mut s = String::with_capacity(64);
        for b in d {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(block[4 * i..4 * i + 4].try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        self.h[0] = self.h[0].wrapping_add(a);
        self.h[1] = self.h[1].wrapping_add(b);
        self.h[2] = self.h[2].wrapping_add(c);
        self.h[3] = self.h[3].wrapping_add(d);
        self.h[4] = self.h[4].wrapping_add(e);
        self.h[5] = self.h[5].wrapping_add(f);
        self.h[6] = self.h[6].wrapping_add(g);
        self.h[7] = self.h[7].wrapping_add(h);
    }
}

pub fn sha256_hex(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    h.finalize_hex()
}

// ============================================================================
// 2. Build-Path-Independent TPC-C Kernel ELF Hashing
// ============================================================================

/// Hashes a Gaudi 2 TPC-C kernel ELF64 binary while zeroing out `.tpc_compiler` section bytes
/// (which carry `tpc-clang` build-directory command lines), so identical kernel sources across
/// worktrees share 100% of compiled Synapse recipes (`AGENTS.md` §8).
pub fn hash_tpc_elf_normalized(elf: &[u8]) -> String {
    if elf.len() < 64 || &elf[0..4] != b"\x7fELF" || elf[4] != 2 || elf[5] != 1 {
        // Non-ELF64-LE: fall back to raw SHA-256
        return sha256_hex(elf);
    }
    let e_shoff = u64::from_le_bytes(elf[40..48].try_into().unwrap()) as usize;
    let e_shentsize = u16::from_le_bytes(elf[58..60].try_into().unwrap()) as usize;
    let e_shnum = u16::from_le_bytes(elf[60..62].try_into().unwrap()) as usize;
    let e_shstrndx = u16::from_le_bytes(elf[62..64].try_into().unwrap()) as usize;

    if e_shentsize < 64 || e_shnum == 0 || e_shstrndx >= e_shnum {
        return sha256_hex(elf);
    }
    let sh_table_end = e_shoff.saturating_add(e_shentsize.saturating_mul(e_shnum));
    if sh_table_end > elf.len() {
        return sha256_hex(elf);
    }

    let str_hdr = e_shoff + e_shstrndx * e_shentsize;
    let str_off = u64::from_le_bytes(elf[str_hdr + 24..str_hdr + 32].try_into().unwrap()) as usize;
    let str_sz = u64::from_le_bytes(elf[str_hdr + 32..str_hdr + 40].try_into().unwrap()) as usize;
    if str_off.saturating_add(str_sz) > elf.len() {
        return sha256_hex(elf);
    }
    let shstrtab = &elf[str_off..str_off + str_sz];

    let mut normalized = elf.to_vec();
    for i in 0..e_shnum {
        let hdr = e_shoff + i * e_shentsize;
        let name_off = u32::from_le_bytes(elf[hdr..hdr + 4].try_into().unwrap()) as usize;
        if name_off >= shstrtab.len() {
            continue;
        }
        let name_end = shstrtab[name_off..]
            .iter()
            .position(|&b| b == 0)
            .map(|p| name_off + p)
            .unwrap_or(shstrtab.len());
        let name = &shstrtab[name_off..name_end];
        if name == b".tpc_compiler" || name == b".comment" {
            let sec_off = u64::from_le_bytes(elf[hdr + 24..hdr + 32].try_into().unwrap()) as usize;
            let sec_sz = u64::from_le_bytes(elf[hdr + 32..hdr + 40].try_into().unwrap()) as usize;
            if sec_off.saturating_add(sec_sz) <= normalized.len() {
                normalized[sec_off..sec_off + sec_sz].fill(0);
            }
            // Also zero the section size field in the header so differing path lengths normalize identically
            normalized[hdr + 32..hdr + 40].fill(0);
        }
    }
    sha256_hex(&normalized)
}

// ============================================================================
// 3. Placement & Launch Binding Guard
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetrievedTensorDesc {
    pub name: String,
    pub tensor_id: u64,
    pub section_id: u32,
    pub offset_in_section: u64,
    pub is_input: bool,
    pub dtype: u32,
    pub dims: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindingViolation {
    UnboundTensor(String),
    MissingWorkspace {
        needed: u64,
    },
    DuplicatePersistentSectionAddress {
        addr: u64,
        first_section: u32,
        second_section: u32,
        tensor_name: String,
    },
    UnexpectedExtraBinding(String),
}

/// Pure host-testable checked-launch validator (`AGENTS.md` §8):
///   - Refuses any missing binding for a recipe's persistent tensors
///   - Refuses missing workspace when `ws_size > 0`
///   - Refuses binding one HBM address to two distinct persistent sections (`synLaunch` status 30)
pub fn verify_launch_bindings(
    recipe_tensors: &[RetrievedTensorDesc],
    bindings: &BTreeMap<String, u64>,
    ws_size: u64,
    ws_addr: u64,
) -> Result<(), BindingViolation> {
    if ws_size > 0 && ws_addr == 0 {
        return Err(BindingViolation::MissingWorkspace { needed: ws_size });
    }
    let mut section_by_addr: BTreeMap<u64, u32> = BTreeMap::new();
    let mut known_names: BTreeSet<&str> = BTreeSet::new();

    for t in recipe_tensors {
        known_names.insert(t.name.as_str());
        let Some(&addr) = bindings.get(&t.name) else {
            return Err(BindingViolation::UnboundTensor(t.name.clone()));
        };
        if addr == 0 {
            return Err(BindingViolation::UnboundTensor(t.name.clone()));
        }
        // Compute base section address (`addr - offset_in_section`) to detect aliasing across distinct sections
        let sec_base = addr.saturating_sub(t.offset_in_section);
        if let Some(&prev_sec) = section_by_addr.get(&sec_base) {
            if prev_sec != t.section_id {
                return Err(BindingViolation::DuplicatePersistentSectionAddress {
                    addr: sec_base,
                    first_section: prev_sec,
                    second_section: t.section_id,
                    tensor_name: t.name.clone(),
                });
            }
        } else {
            section_by_addr.insert(sec_base, t.section_id);
        }
    }

    for bound_name in bindings.keys() {
        if !known_names.contains(bound_name.as_str()) {
            return Err(BindingViolation::UnexpectedExtraBinding(bound_name.clone()));
        }
    }
    Ok(())
}

// ============================================================================
// 4. On-Disk Compiled-Recipe Cache Directory
// ============================================================================

#[derive(Debug, Clone)]
pub struct RecipeCache {
    root: Option<PathBuf>,
}

impl RecipeCache {
    pub fn from_env() -> Self {
        match std::env::var("GAUDIUM_RECIPE_CACHE") {
            Ok(v) if v.eq_ignore_ascii_case("off") || v.trim().is_empty() => Self { root: None },
            Ok(v) => Self {
                root: Some(PathBuf::from(v)),
            },
            Err(_) => Self { root: None },
        }
    }

    pub fn at_dir(dir: impl AsRef<Path>) -> Self {
        Self {
            root: Some(dir.as_ref().to_path_buf()),
        }
    }

    pub fn disabled() -> Self {
        Self { root: None }
    }

    pub fn is_enabled(&self) -> bool {
        self.root.is_some()
    }

    pub fn recipe_path(&self, name: &str, fingerprint_hex: &str) -> Option<PathBuf> {
        let root = self.root.as_ref()?;
        let short = &fingerprint_hex[..fingerprint_hex.len().min(16)];
        Some(root.join(format!("{name}-{short}.recipe")))
    }

    pub fn meta_path(&self, name: &str, fingerprint_hex: &str) -> Option<PathBuf> {
        let root = self.root.as_ref()?;
        let short = &fingerprint_hex[..fingerprint_hex.len().min(16)];
        Some(root.join(format!("{name}-{short}.sha256")))
    }

    /// Returns cached recipe path iff the `.recipe` file exists and matches its stored `.sha256`.
    pub fn lookup_verified(&self, name: &str, fingerprint_hex: &str) -> Option<PathBuf> {
        let rpath = self.recipe_path(name, fingerprint_hex)?;
        let mpath = self.meta_path(name, fingerprint_hex)?;
        let expected_sha = fs::read_to_string(&mpath).ok()?.trim().to_owned();
        let bytes = fs::read(&rpath).ok()?;
        if sha256_hex(&bytes) == expected_sha {
            Some(rpath)
        } else {
            None
        }
    }

    /// Records the SHA-256 sidecar after `synRecipeSerialize` writes `rpath`.
    pub fn record_sha256(&self, name: &str, fingerprint_hex: &str) -> io::Result<()> {
        let (Some(rpath), Some(mpath)) = (
            self.recipe_path(name, fingerprint_hex),
            self.meta_path(name, fingerprint_hex),
        ) else {
            return Ok(());
        };
        let bytes = fs::read(&rpath)?;
        let digest = sha256_hex(&bytes);
        let tmp = mpath.with_extension("tmp");
        fs::write(&tmp, format!("{digest}\n"))?;
        fs::rename(&tmp, &mpath)
    }

    pub fn ensure_dir(&self) -> io::Result<()> {
        if let Some(root) = &self.root {
            fs::create_dir_all(root)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_nist_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn launch_binding_guard_catches_unbound_workspace_and_section_aliasing() {
        let t1 = RetrievedTensorDesc {
            name: "in_a".into(),
            tensor_id: 1,
            section_id: 10,
            offset_in_section: 0,
            is_input: true,
            dtype: 2,
            dims: vec![128],
        };
        let t2 = RetrievedTensorDesc {
            name: "in_b".into(),
            tensor_id: 2,
            section_id: 11,
            offset_in_section: 0,
            is_input: true,
            dtype: 2,
            dims: vec![128],
        };
        let t3 = RetrievedTensorDesc {
            name: "arena_slice".into(),
            tensor_id: 3,
            section_id: 10, // same section as t1 at different offset => valid!
            offset_in_section: 4096,
            is_input: true,
            dtype: 2,
            dims: vec![128],
        };

        let mut ok_map = BTreeMap::new();
        ok_map.insert("in_a".into(), 0x1000_0000);
        ok_map.insert("in_b".into(), 0x2000_0000);
        ok_map.insert("arena_slice".into(), 0x1000_1000);
        verify_launch_bindings(
            &[t1.clone(), t2.clone(), t3.clone()],
            &ok_map,
            4096,
            0x9000_0000,
        )
        .unwrap();

        // Missing workspace fails
        assert_eq!(
            verify_launch_bindings(&[t1.clone(), t2.clone(), t3.clone()], &ok_map, 4096, 0),
            Err(BindingViolation::MissingWorkspace { needed: 4096 })
        );

        // Unbound tensor fails
        let mut missing = ok_map.clone();
        missing.remove("in_b");
        assert_eq!(
            verify_launch_bindings(&[t1.clone(), t2.clone(), t3.clone()], &missing, 0, 0),
            Err(BindingViolation::UnboundTensor("in_b".into()))
        );

        // Two distinct persistent sections pointing at the same HBM base fails (synLaunch status 30 guard!)
        let mut aliased = ok_map.clone();
        aliased.insert("in_b".into(), 0x1000_0000);
        assert!(matches!(
            verify_launch_bindings(&[t1, t2, t3], &aliased, 0, 0),
            Err(BindingViolation::DuplicatePersistentSectionAddress { .. })
        ));
    }
}
