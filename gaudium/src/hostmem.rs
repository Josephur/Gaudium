//! Host memory for the engine's own large buffers: anonymous mappings that go back to the system
//! when they are dropped, and what the host and its control group can still give.
//!
//! `AnonMap` is used for buffers that are large, live for a while and must leave no trace when
//! they go: the segments of the host-RAM prefix cache (`cache_ram.rs`) and the transfer ring of
//! its copies (`synapse/xfer.rs`). A mapping is a whole number of 2 MiB pages and asks for
//! transparent huge pages; it is not touched here, so its pages exist only once written (or once
//! pinned for the device, which then faults them in as memory that never has to move again).

use std::ffi::c_void;

/// Size every mapping is rounded up to (the transparent huge page of x86-64 and of aarch64 with
/// 4 KiB base pages).
pub const HUGE_PAGE: u64 = 2 * 1024 * 1024;

const PROT_READ: i32 = 1;
const PROT_WRITE: i32 = 2;
const MAP_PRIVATE: i32 = 0x02;
const MAP_ANONYMOUS: i32 = 0x20;
const MADV_HUGEPAGE: i32 = 14;

unsafe extern "C" {
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64)
        -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
    fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
}

/// One anonymous private mapping, unmapped on drop.
#[derive(Debug)]
pub struct AnonMap {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: the mapping is plain memory owned by this value; nothing in it is tied to a thread.
unsafe impl Send for AnonMap {}
unsafe impl Sync for AnonMap {}

impl AnonMap {
    /// Bytes a mapping for `bytes` takes: `bytes` rounded up to whole huge pages (the verified
    /// `align_up`), at least one.
    pub fn mapped_len(bytes: u64) -> u64 {
        crate::verified::align_up(bytes.max(1), HUGE_PAGE)
    }

    /// Maps room for `bytes` (see [`Self::mapped_len`]). The pages are not touched.
    pub fn new(bytes: u64) -> Result<Self, String> {
        let len = usize::try_from(Self::mapped_len(bytes))
            .map_err(|_| format!("a mapping of {bytes} bytes does not fit this address space"))?;
        // Over-map by one huge page so that the part kept can start on a huge-page boundary:
        // only an aligned range can be backed by huge pages.
        let span = len
            .checked_add(HUGE_PAGE as usize)
            .ok_or_else(|| format!("a mapping of {bytes} bytes overflows"))?;
        // SAFETY: a fresh anonymous mapping at an address the kernel picks.
        let raw = unsafe {
            mmap(
                std::ptr::null_mut(),
                span,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if raw as isize == -1 || raw.is_null() {
            return Err(format!(
                "host memory: mapping {len} bytes failed: {}",
                std::io::Error::last_os_error()
            ));
        }
        let base = raw as usize;
        let start = base.next_multiple_of(HUGE_PAGE as usize);
        let (head, tail) = (start - base, base + span - (start + len));
        // SAFETY: both ranges lie inside the mapping made above and are not used after this.
        unsafe {
            if head > 0 {
                munmap(base as *mut c_void, head);
            }
            if tail > 0 {
                munmap((start + len) as *mut c_void, tail);
            }
            // advice only: a kernel without transparent huge pages keeps 4 KiB pages
            madvise(start as *mut c_void, len, MADV_HUGEPAGE);
        }
        Ok(Self {
            ptr: start as *mut u8,
            len,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: the mapping is readable for `len` bytes while `self` lives.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: the mapping is uniquely owned and writable for `len` bytes.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AnonMap {
    fn drop(&mut self) {
        // SAFETY: exactly the range kept by `new`.
        unsafe {
            munmap(self.ptr as *mut c_void, self.len);
        }
    }
}

/// What the host can still give this process, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostMemory {
    /// `MemAvailable` of the host (or of the container's view of it).
    pub host_available: u64,
    /// Limit of this process's control group (`None`: no limit).
    pub cgroup_limit: Option<u64>,
    /// Memory the control group has in use that it cannot give back by dropping file pages.
    pub cgroup_in_use: Option<u64>,
}

impl HostMemory {
    /// Reads `/proc/meminfo` and the control group v2 files of this process. A file that is
    /// missing or unreadable leaves its part unknown (no limit).
    pub fn read() -> Self {
        let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
        let (limit, in_use) = match cgroup_dir() {
            Some(dir) => {
                let max = std::fs::read_to_string(dir.join("memory.max")).unwrap_or_default();
                let cur = std::fs::read_to_string(dir.join("memory.current")).unwrap_or_default();
                let stat = std::fs::read_to_string(dir.join("memory.stat")).unwrap_or_default();
                parse_cgroup(&max, &cur, &stat)
            }
            None => (None, None),
        };
        Self {
            host_available: parse_meminfo_available(&meminfo).unwrap_or(u64::MAX),
            cgroup_limit: limit,
            cgroup_in_use: in_use,
        }
    }

    /// Bytes this process can still take: the smaller of what the host has available and what
    /// its control group has left (the verified `hc_available`).
    pub fn available(&self) -> u64 {
        crate::verified::hc_available(
            self.host_available,
            self.cgroup_limit.is_some(),
            self.cgroup_limit.unwrap_or(0),
            self.cgroup_in_use.unwrap_or(0),
        )
    }

    /// The numbers behind [`Self::available`], for a message.
    pub fn describe(&self) -> String {
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        match (self.cgroup_limit, self.cgroup_in_use) {
            (Some(l), Some(u)) => format!(
                "host available {:.2} GiB; control group limit {:.2} GiB with {:.2} GiB in use",
                gib(self.host_available),
                gib(l),
                gib(u)
            ),
            _ => format!(
                "host available {:.2} GiB; no control group limit",
                gib(self.host_available)
            ),
        }
    }
}

/// The control group v2 directory of this process (`None`: not found, or not version 2).
fn cgroup_dir() -> Option<std::path::PathBuf> {
    let text = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let rel = text.lines().find_map(|l| l.strip_prefix("0::"))?;
    let dir = std::path::Path::new("/sys/fs/cgroup").join(rel.trim_start_matches('/'));
    dir.join("memory.max").exists().then_some(dir)
}

/// `MemAvailable` of a `/proc/meminfo` text, in bytes.
pub fn parse_meminfo_available(text: &str) -> Option<u64> {
    let line = text.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb.saturating_mul(1024))
}

/// `(limit, in use)` of a control group from its `memory.max`, `memory.current` and
/// `memory.stat` texts. File pages count as free: the kernel drops them before it refuses an
/// allocation (a mapped checkpoint is such pages).
pub fn parse_cgroup(max: &str, current: &str, stat: &str) -> (Option<u64>, Option<u64>) {
    let limit = max.trim().parse::<u64>().ok();
    let cur = current.trim().parse::<u64>().ok();
    let field = |name: &str| -> u64 {
        stat.lines()
            .find_map(|l| {
                let mut it = l.split_whitespace();
                (it.next() == Some(name)).then(|| it.next()?.parse::<u64>().ok())?
            })
            .unwrap_or(0)
    };
    let droppable = field("inactive_file").saturating_add(field("active_file"));
    (limit, cur.map(|c| c.saturating_sub(droppable)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_mapping_is_whole_huge_pages_aligned_and_writable() {
        assert_eq!(AnonMap::mapped_len(1), HUGE_PAGE);
        assert_eq!(AnonMap::mapped_len(HUGE_PAGE), HUGE_PAGE);
        assert_eq!(AnonMap::mapped_len(HUGE_PAGE + 1), 2 * HUGE_PAGE);
        let mut m = AnonMap::new(3 * 1024 * 1024).unwrap();
        assert_eq!(m.len() as u64, 2 * HUGE_PAGE);
        assert_eq!(m.as_ptr() as u64 % HUGE_PAGE, 0);
        // fresh anonymous memory reads as zeros and takes writes over its whole length
        assert!(m.as_slice().iter().all(|&b| b == 0));
        let n = m.len();
        m.as_mut_slice()[n - 1] = 7;
        m.as_mut_slice()[0] = 9;
        assert_eq!((m.as_slice()[0], m.as_slice()[n - 1]), (9, 7));
    }

    /// What mapping, writing and unmapping segments costs on this machine (`cargo test --release
    /// -- --ignored segment_costs --nocapture`).
    #[test]
    #[ignore]
    fn segment_costs() {
        use std::time::Instant;
        for (n, bytes) in [(96usize, 32u64 << 20), (20, 150 << 20)] {
            let t0 = Instant::now();
            let mut maps: Vec<AnonMap> = (0..n).map(|_| AnonMap::new(bytes).unwrap()).collect();
            let t_map = t0.elapsed();
            let t1 = Instant::now();
            for m in &mut maps {
                m.as_mut_slice().fill(7);
            }
            let t_fill = t1.elapsed();
            let t2 = Instant::now();
            for m in &mut maps {
                m.as_mut_slice().fill(9);
            }
            let t_again = t2.elapsed();
            let total = n as u64 * bytes;
            let t3 = Instant::now();
            drop(maps);
            let t_unmap = t3.elapsed();
            let gib = total as f64 / (1u64 << 30) as f64;
            println!(
                "{n} segments of {} MiB ({gib:.2} GiB): map {:.1} ms, first write {:.0} ms ({:.1} GiB/s), second write {:.0} ms ({:.1} GiB/s), unmap {:.1} ms ({:.1} GiB/s)",
                bytes >> 20,
                t_map.as_secs_f64() * 1e3,
                t_fill.as_secs_f64() * 1e3,
                gib / t_fill.as_secs_f64(),
                t_again.as_secs_f64() * 1e3,
                gib / t_again.as_secs_f64(),
                t_unmap.as_secs_f64() * 1e3,
                gib / t_unmap.as_secs_f64(),
            );
        }
    }

    #[test]
    fn meminfo_and_cgroup_texts_are_read_in_bytes() {
        let mi =
            "MemTotal:       115000000 kB\nMemFree:        1000 kB\nMemAvailable:   90000000 kB\n";
        assert_eq!(parse_meminfo_available(mi), Some(90_000_000 * 1024));
        assert_eq!(parse_meminfo_available("MemTotal: 1 kB\n"), None);
        let stat = "anon 100\nfile 5000\ninactive_file 3000\nactive_file 1000\n";
        assert_eq!(
            parse_cgroup("92233720368\n", "10000\n", stat),
            (Some(92_233_720_368), Some(6000))
        );
        // no limit
        assert_eq!(parse_cgroup("max\n", "10000\n", stat), (None, Some(6000)));
        assert_eq!(parse_cgroup("", "", ""), (None, None));
    }

    #[test]
    fn available_is_the_smaller_of_host_and_control_group() {
        let gib = 1u64 << 30;
        let m = HostMemory {
            host_available: 80 * gib,
            cgroup_limit: Some(86 * gib),
            cgroup_in_use: Some(40 * gib),
        };
        assert_eq!(m.available(), 46 * gib);
        let m = HostMemory {
            host_available: 20 * gib,
            ..m
        };
        assert_eq!(m.available(), 20 * gib);
        let m = HostMemory {
            host_available: 20 * gib,
            cgroup_limit: None,
            cgroup_in_use: None,
        };
        assert_eq!(m.available(), 20 * gib);
        // a group over its limit has nothing left
        let m = HostMemory {
            host_available: 20 * gib,
            cgroup_limit: Some(10 * gib),
            cgroup_in_use: Some(12 * gib),
        };
        assert_eq!(m.available(), 0);
    }
}
