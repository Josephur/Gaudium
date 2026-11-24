//! Compiled-recipe cache, zero-dependency SHA-256, build-path-independent TPC ELF hashing,
//! and recipe placement / binding guards (`recipe_cache.hpp`).

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::SystemTime;

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

/// Sections whose contents depend on where and with which compiler build a kernel was built, not on
/// the kernel: `.tpc_compiler` carries the `tpc-clang` command line with absolute build paths, and
/// `.comment` the compiler identification string.
const ELF_BUILD_METADATA_SECTIONS: [&[u8]; 2] = [b".tpc_compiler", b".comment"];

/// Fingerprints a TPC-C kernel ELF so that the same kernel built in another directory (another
/// worktree, checkout or CI runner) gets the same hash and shares its compiled Synapse recipes
/// (`AGENTS.md` §8), while any change to its code, data, symbols or other sections changes it.
///
/// `tpc-clang` emits ELF32 relocatable objects; ELF64 is handled the same way. The hash covers the
/// ELF header (without the section-table offset), then every section in order: its name, type,
/// flags, address, link, info, alignment, entry size, and its size and bytes. File offsets are left
/// out, because a longer build path in `.tpc_compiler` moves every section after it. The build
/// metadata sections contribute only their header fields, and a symbol defined in one of them has
/// its size field zeroed. Anything this parser does not understand (not ELF, big-endian, program
/// headers, out-of-range tables) falls back to a SHA-256 of the raw bytes: stricter, never looser.
pub fn hash_tpc_elf_normalized(elf: &[u8]) -> String {
    normalized_elf_digest(elf).unwrap_or_else(|| sha256_hex(elf))
}

fn normalized_elf_digest(elf: &[u8]) -> Option<String> {
    if elf.len() < 16 || &elf[0..4] != b"\x7fELF" || elf[5] != 1 {
        return None; // not ELF, or not little-endian
    }
    let wide = match elf[4] {
        1 => false,
        2 => true,
        _ => return None,
    };
    let rd = |off: usize, n: usize| -> Option<u64> {
        let b = elf.get(off..off.checked_add(n)?)?;
        let mut v = [0u8; 8];
        v[..n].copy_from_slice(b);
        Some(u64::from_le_bytes(v))
    };
    // Word-sized e_entry, e_phoff and e_shoff, then e_flags, then the 16-bit fields from e_ehsize.
    let w = if wide { 8 } else { 4 };
    let (entry_off, shoff_off) = (24, 24 + 2 * w);
    let flags_off = shoff_off + w;
    let half_off = flags_off + 4;
    let e_type = rd(16, 2)?;
    let e_machine = rd(18, 2)?;
    let e_version = rd(20, 4)?;
    let e_entry = rd(entry_off, w)?;
    let e_shoff = usize::try_from(rd(shoff_off, w)?).ok()?;
    let e_flags = rd(flags_off, 4)?;
    let e_phnum = rd(half_off + 4, 2)?;
    let e_shentsize = rd(half_off + 6, 2)? as usize;
    let e_shnum = rd(half_off + 8, 2)? as usize;
    let e_shstrndx = rd(half_off + 10, 2)? as usize;
    let min_shentsize = if wide { 64 } else { 40 };
    if e_phnum != 0 || e_shentsize < min_shentsize || e_shnum == 0 || e_shstrndx >= e_shnum {
        return None;
    }

    struct Sec {
        name: u64,
        ty: u64,
        flags: u64,
        addr: u64,
        off: usize,
        size: usize,
        link: u64,
        info: u64,
        align: u64,
        entsize: u64,
    }
    let mut secs = Vec::with_capacity(e_shnum);
    for i in 0..e_shnum {
        let h = e_shoff.checked_add(i.checked_mul(e_shentsize)?)?;
        // Field order is the same in ELF32 and ELF64; flags, addr, offset, size, addralign and
        // entsize are word-sized, the others 32-bit.
        let flags = h + 8;
        let addr = flags + w;
        let off = addr + w;
        let size = off + w;
        let link = size + w;
        let align = link + 8;
        secs.push(Sec {
            name: rd(h, 4)?,
            ty: rd(h + 4, 4)?,
            flags: rd(flags, w)?,
            addr: rd(addr, w)?,
            off: usize::try_from(rd(off, w)?).ok()?,
            size: usize::try_from(rd(size, w)?).ok()?,
            link: rd(link, 4)?,
            info: rd(link + 4, 4)?,
            align: rd(align, w)?,
            entsize: rd(align + w, w)?,
        });
    }
    const SHT_SYMTAB: u64 = 2;
    const SHT_NOBITS: u64 = 8;
    let bytes_of = |s: &Sec| -> Option<&[u8]> {
        if s.ty == SHT_NOBITS {
            return Some(&[]);
        }
        elf.get(s.off..s.off.checked_add(s.size)?)
    };
    let shstrtab = bytes_of(&secs[e_shstrndx])?;
    let name_of = |s: &Sec| -> Option<&[u8]> {
        let rest = shstrtab.get(usize::try_from(s.name).ok()?..)?;
        Some(&rest[..rest.iter().position(|&b| b == 0)?])
    };
    let mut metadata = vec![false; secs.len()];
    for (i, s) in secs.iter().enumerate() {
        metadata[i] = ELF_BUILD_METADATA_SECTIONS.contains(&name_of(s)?);
    }

    let mut h = Sha256::new();
    h.update_str("gaudium-tpc-elf-v2");
    h.update(&elf[0..16]); // e_ident: class, byte order, version, ABI
    for v in [
        e_type,
        e_machine,
        e_version,
        e_entry,
        e_flags,
        e_shnum as u64,
        e_shstrndx as u64,
    ] {
        h.update_u64(v);
    }
    for (i, s) in secs.iter().enumerate() {
        h.update(name_of(s)?);
        h.update_u64(u64::MAX); // name terminator that no length can equal
        for v in [s.ty, s.flags, s.addr, s.link, s.info, s.align, s.entsize] {
            h.update_u64(v);
        }
        if metadata[i] {
            continue;
        }
        let data = bytes_of(s)?;
        h.update_u64(s.size as u64);
        if s.ty == SHT_SYMTAB {
            // Elf32_Sym: name, value, size, info, other, shndx(16) ; Elf64_Sym: name, info,
            // other, shndx(16), value, size.
            let (ent, size_at, size_len, shndx_at) =
                if wide { (24, 16, 8, 6) } else { (16, 8, 4, 14) };
            if s.entsize != ent as u64 || data.len() % ent != 0 {
                return None;
            }
            let mut sym = data.to_vec();
            for e in sym.chunks_exact_mut(ent) {
                let shndx = u16::from_le_bytes([e[shndx_at], e[shndx_at + 1]]) as usize;
                if shndx < metadata.len() && metadata[shndx] {
                    e[size_at..size_at + size_len].fill(0);
                }
            }
            h.update(&sym);
        } else {
            h.update(data);
        }
    }
    Some(h.finalize_hex())
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

/// Extension of a cached recipe file.
pub const RECIPE_EXT: &str = ".recipe";
/// Extension of a cached recipe's SHA-256 sidecar.
pub const SHA256_EXT: &str = ".sha256";

/// Entries (`<name>-<hash>`) this process has loaded, looked up or written: a trim run by this
/// process never removes them (`recipe_trim`), and the server refreshes their recency before
/// each trim so they still count as recent after a restart.
pub(crate) fn in_use_registry() -> &'static Mutex<BTreeSet<String>> {
    static IN_USE: OnceLock<Mutex<BTreeSet<String>>> = OnceLock::new();
    IN_USE.get_or_init(|| Mutex::new(BTreeSet::new()))
}

fn note_in_use(stem: &str) {
    let mut set = in_use_registry().lock().unwrap_or_else(|p| p.into_inner());
    if !set.contains(stem) {
        set.insert(stem.to_owned());
    }
}

/// Sets a file's modification time to now. Recency of cache entries is kept this way rather than
/// in access times, which `noatime` / `relatime` mounts do not keep.
fn touch(path: &Path) -> io::Result<()> {
    fs::File::open(path)?.set_modified(SystemTime::now())
}

/// Refreshes the sidecar time of every entry this process holds that is in `dir` (best effort:
/// an entry another process removed is skipped).
pub fn refresh_recency(dir: &Path) {
    let stems: Vec<String> = in_use_registry()
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .iter()
        .cloned()
        .collect();
    for stem in stems {
        let _ = touch(&dir.join(format!("{stem}{SHA256_EXT}")));
    }
}

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

    /// The cache directory, when the cache is on.
    pub fn root(&self) -> Option<&Path> {
        self.root.as_deref()
    }

    fn stem(name: &str, fingerprint_hex: &str) -> String {
        let short = &fingerprint_hex[..fingerprint_hex.len().min(16)];
        format!("{name}-{short}")
    }

    pub fn recipe_path(&self, name: &str, fingerprint_hex: &str) -> Option<PathBuf> {
        let root = self.root.as_ref()?;
        Some(root.join(format!("{}{RECIPE_EXT}", Self::stem(name, fingerprint_hex))))
    }

    pub fn meta_path(&self, name: &str, fingerprint_hex: &str) -> Option<PathBuf> {
        let root = self.root.as_ref()?;
        Some(root.join(format!("{}{SHA256_EXT}", Self::stem(name, fingerprint_hex))))
    }

    /// Marks the entry as held by this process before it is written, so a trim in this process
    /// leaves a recipe that is being serialized alone.
    pub fn mark_in_use(&self, name: &str, fingerprint_hex: &str) {
        if self.root.is_some() {
            note_in_use(&Self::stem(name, fingerprint_hex));
        }
    }

    /// Returns cached recipe path iff the `.recipe` file exists and matches its stored `.sha256`.
    ///
    /// A hit sets the sidecar's modification time to now (the entry's recency for trims). The
    /// entry is marked as held by this process before anything is read, so a trim in this
    /// process either removed it before (a miss) or keeps it. A trim in another process may
    /// remove the files at any moment: a file that is gone fails its read (a miss), and the bytes
    /// that were read are checked against the sidecar, so a hit is never a partial file. The
    /// caller loads the recipe again by its path; a file removed after this check fails that
    /// load, which callers take as a miss as well.
    pub fn lookup_verified(&self, name: &str, fingerprint_hex: &str) -> Option<PathBuf> {
        let rpath = self.recipe_path(name, fingerprint_hex)?;
        let mpath = self.meta_path(name, fingerprint_hex)?;
        note_in_use(&Self::stem(name, fingerprint_hex));
        let expected_sha = fs::read_to_string(&mpath).ok()?.trim().to_owned();
        let bytes = fs::read(&rpath).ok()?;
        if sha256_hex(&bytes) == expected_sha {
            let _ = touch(&mpath);
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
        note_in_use(&Self::stem(name, fingerprint_hex));
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

    /// A minimal little-endian relocatable ELF: `.text`, `.tpc_compiler` (`path` as its bytes),
    /// `.comment`, `.symtab` (a symbol in `.text` of size `text_sym_size` and one in
    /// `.tpc_compiler` of the section's length) and `.shstrtab`; contents laid out in that order
    /// and the section table at the end, like `tpc-clang` output.
    fn synth_elf(wide: bool, text: &[u8], path: &[u8], text_sym_size: u64) -> Vec<u8> {
        let w = if wide { 8 } else { 4 };
        let put = |v: &mut Vec<u8>, x: u64, n: usize| v.extend_from_slice(&x.to_le_bytes()[..n]);
        let sym = |name: u64, size: u64, shndx: u64| {
            let mut e = Vec::new();
            if wide {
                put(&mut e, name, 4);
                put(&mut e, 0x11, 1);
                put(&mut e, 0, 1);
                put(&mut e, shndx, 2);
                put(&mut e, 0, 8);
                put(&mut e, size, 8);
            } else {
                put(&mut e, name, 4);
                put(&mut e, 0, 4);
                put(&mut e, size, 4);
                put(&mut e, 0x11, 1);
                put(&mut e, 0, 1);
                put(&mut e, shndx, 2);
            }
            e
        };
        let ent = if wide { 24 } else { 16 };
        let mut symtab = vec![0u8; ent];
        symtab.extend(sym(0, text_sym_size, 1));
        symtab.extend(sym(0, path.len() as u64, 2));
        let shstr = b"\0.text\0.tpc_compiler\0.comment\0.symtab\0.shstrtab\0".to_vec();
        // (name offset, type, flags, content, link, entsize)
        let secs: Vec<(u64, u64, u64, Vec<u8>, u64, u64)> = vec![
            (1, 1, 6, text.to_vec(), 0, 0),
            (7, 1, 2, path.to_vec(), 0, 0),
            (21, 1, 0x30, b"clang version x\0".to_vec(), 0, 1),
            (30, 2, 0, symtab, 5, ent as u64),
            (38, 3, 0, shstr, 0, 0),
        ];
        let ehsize = if wide { 64 } else { 52 };
        let mut body = Vec::new();
        let mut offs = Vec::new();
        for s in &secs {
            while (ehsize + body.len()) % 4 != 0 {
                body.push(0);
            }
            offs.push(ehsize + body.len());
            body.extend_from_slice(&s.3);
        }
        let shoff = ehsize + body.len();
        let mut f = vec![0x7f, b'E', b'L', b'F', if wide { 2 } else { 1 }, 1, 1, 0];
        f.resize(16, 0);
        put(&mut f, 1, 2); // ET_REL
        put(&mut f, 3, 2); // e_machine as tpc-clang writes it
        put(&mut f, 1, 4);
        put(&mut f, 0, w); // e_entry
        put(&mut f, 0, w); // e_phoff
        put(&mut f, shoff as u64, w);
        put(&mut f, 0, 4); // e_flags
        put(&mut f, ehsize as u64, 2);
        put(&mut f, 0, 2); // e_phentsize
        put(&mut f, 0, 2); // e_phnum
        put(&mut f, if wide { 64 } else { 40 }, 2);
        put(&mut f, secs.len() as u64 + 1, 2);
        put(&mut f, secs.len() as u64, 2); // e_shstrndx: last section
        assert_eq!(f.len(), ehsize);
        f.extend(body);
        f.extend(vec![0u8; if wide { 64 } else { 40 }]); // SHN_UNDEF
        for (s, off) in secs.iter().zip(&offs) {
            put(&mut f, s.0, 4);
            put(&mut f, s.1, 4);
            put(&mut f, s.2, w);
            put(&mut f, 0, w); // addr
            put(&mut f, *off as u64, w);
            put(&mut f, s.3.len() as u64, w);
            put(&mut f, s.4, 4);
            put(&mut f, 1, 4); // info
            put(&mut f, 4, w); // addralign
            put(&mut f, s.5, w);
        }
        f
    }

    #[test]
    fn kernel_hash_ignores_build_path_of_any_length() {
        for wide in [false, true] {
            let a = synth_elf(wide, b"KERNELCODE", b"-o /tmp/a/k.o", 10);
            let b = synth_elf(
                wide,
                b"KERNELCODE",
                b"-o /home/runner/work/a/much/longer/k.o",
                10,
            );
            assert_ne!(a, b);
            assert_ne!(
                a.len(),
                b.len(),
                "later sections must move for this test to mean anything"
            );
            assert_eq!(
                hash_tpc_elf_normalized(&a),
                hash_tpc_elf_normalized(&b),
                "wide={wide}"
            );
            assert_ne!(
                hash_tpc_elf_normalized(&a),
                sha256_hex(&a),
                "must not fall back to raw"
            );
        }
    }

    #[test]
    fn kernel_hash_changes_with_code_and_symbols() {
        for wide in [false, true] {
            let base = hash_tpc_elf_normalized(&synth_elf(wide, b"KERNELCODE", b"p", 10));
            let code = hash_tpc_elf_normalized(&synth_elf(wide, b"KERNELCODF", b"p", 10));
            let sym = hash_tpc_elf_normalized(&synth_elf(wide, b"KERNELCODE", b"p", 9));
            assert_ne!(base, code, "wide={wide}");
            assert_ne!(base, sym, "wide={wide}");
        }
        let narrow = hash_tpc_elf_normalized(&synth_elf(false, b"KERNELCODE", b"p", 10));
        let wide = hash_tpc_elf_normalized(&synth_elf(true, b"KERNELCODE", b"p", 10));
        assert_ne!(narrow, wide);
    }

    #[test]
    fn kernel_hash_falls_back_to_raw_bytes_when_unparsable() {
        let not_elf = b"not an elf file at all".to_vec();
        assert_eq!(hash_tpc_elf_normalized(&not_elf), sha256_hex(&not_elf));
        let mut truncated = synth_elf(false, b"KERNELCODE", b"p", 10);
        truncated.truncate(truncated.len() - 8);
        assert_eq!(hash_tpc_elf_normalized(&truncated), sha256_hex(&truncated));
        let mut big_endian = synth_elf(false, b"KERNELCODE", b"p", 10);
        big_endian[5] = 2;
        assert_eq!(
            hash_tpc_elf_normalized(&big_endian),
            sha256_hex(&big_endian)
        );
    }

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
