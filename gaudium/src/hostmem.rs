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

/// One control group of this process's chain: its own group, or a group above it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Group {
    /// Limit of the group (`None`: no limit, or its files cannot be read).
    pub limit: Option<u64>,
    /// Memory the group has in use that it cannot give back by dropping file pages (its own
    /// and that of every group below it).
    pub in_use: Option<u64>,
}

/// What the host can still give this process, in bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostMemory {
    /// `MemAvailable` of the host (or of the container's view of it).
    pub host_available: u64,
    /// The control groups this process is in, its own first, then each group above it up to
    /// the top of what it can see. A limit on any of them bounds the process.
    pub groups: Vec<Group>,
}

impl HostMemory {
    /// Reads `/proc/meminfo` and the control group v2 files of this process and of every
    /// group above its own. A file that is missing or unreadable leaves its part unknown (no
    /// limit).
    pub fn read() -> Self {
        let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
        let groups = std::fs::read_to_string("/proc/self/cgroup")
            .ok()
            .and_then(|text| {
                text.lines()
                    .find_map(|l| l.strip_prefix("0::").map(str::to_owned))
            })
            .map(|rel| read_groups(std::path::Path::new("/sys/fs/cgroup"), &rel))
            .unwrap_or_default();
        Self {
            host_available: parse_meminfo_available(&meminfo).unwrap_or(u64::MAX),
            groups,
        }
    }

    /// A host with no control group limit (tests, and messages about a plain host).
    pub fn of_host(host_available: u64) -> Self {
        Self {
            host_available,
            groups: Vec::new(),
        }
    }

    /// Bytes this process can still take: what the host has available, and no more than any
    /// group of its chain has left. Each group is taken into account by the verified
    /// `hc_chain_step`; the walk over the chain is all that happens here.
    pub fn available(&self) -> u64 {
        self.groups.iter().fold(self.host_available, |left, g| {
            crate::verified::hc_chain_step(
                left,
                g.limit.is_some(),
                g.limit.unwrap_or(0),
                g.in_use.unwrap_or(0),
            )
        })
    }

    /// The numbers behind [`Self::available`], for a message: every group that has a limit,
    /// with how far above the process's own group it is.
    pub fn describe(&self) -> String {
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        let limits: Vec<String> = self
            .groups
            .iter()
            .enumerate()
            .filter_map(|(up, g)| {
                let limit = g.limit?;
                Some(format!(
                    "{:.2} GiB with {:.2} GiB in use ({})",
                    gib(limit),
                    gib(g.in_use.unwrap_or(0)),
                    match up {
                        0 => "its own group".to_string(),
                        1 => "the group above".to_string(),
                        n => format!("{n} groups above"),
                    }
                ))
            })
            .collect();
        if limits.is_empty() {
            format!(
                "host available {:.2} GiB; no control group limit",
                gib(self.host_available)
            )
        } else {
            format!(
                "host available {:.2} GiB; control group limit {}",
                gib(self.host_available),
                limits.join(", ")
            )
        }
    }
}

/// The control groups of a process whose group is `rel` (the path after `0::` in its
/// `/proc/<pid>/cgroup`) below the control group mount `root`: that group first, then each
/// group above it, the mount's own directory last. A directory without readable memory files
/// is a group without a limit.
pub fn read_groups(root: &std::path::Path, rel: &str) -> Vec<Group> {
    let rel = std::path::Path::new(rel.trim().trim_start_matches('/'));
    rel.ancestors()
        .map(|up| {
            let dir = root.join(up);
            let read = |name: &str| std::fs::read_to_string(dir.join(name)).unwrap_or_default();
            let (limit, in_use) = parse_cgroup(
                &read("memory.max"),
                &read("memory.current"),
                &read("memory.stat"),
            );
            Group { limit, in_use }
        })
        .collect()
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
        let own = |limit: Option<u64>, in_use: Option<u64>, host: u64| HostMemory {
            host_available: host * gib,
            groups: vec![Group {
                limit: limit.map(|l| l * gib),
                in_use: in_use.map(|u| u * gib),
            }],
        };
        assert_eq!(own(Some(86), Some(40), 80).available(), 46 * gib);
        assert_eq!(own(Some(86), Some(40), 20).available(), 20 * gib);
        assert_eq!(own(None, None, 20).available(), 20 * gib);
        assert_eq!(HostMemory::of_host(20 * gib).available(), 20 * gib);
        // a group over its limit has nothing left
        assert_eq!(own(Some(10), Some(12), 20).available(), 0);
    }

    /// A made-up control group tree under a directory of its own: `groups` are `(path below
    /// the mount, memory.max, memory.current)`, each with a `memory.stat` of 1 GiB of file
    /// pages that can be dropped.
    fn cgroup_tree(name: &str, groups: &[(&str, &str, u64)]) -> std::path::PathBuf {
        let root =
            std::env::temp_dir().join(format!("gaudium-cgroup-test-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        for (rel, max, current) in groups {
            let dir = root.join(rel);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("memory.max"), format!("{max}\n")).unwrap();
            std::fs::write(dir.join("memory.current"), format!("{current}\n")).unwrap();
            std::fs::write(
                dir.join("memory.stat"),
                format!("anon 0\ninactive_file {}\nactive_file 0\n", 1u64 << 30),
            )
            .unwrap();
        }
        root
    }

    #[test]
    fn a_limit_on_any_group_above_bounds_what_is_available() {
        let gib = 1u64 << 30;
        let g = |n: u64| (n * gib).to_string();
        // the process is in a/b/c; 64 GiB are available on the host; every group has 11 GiB
        // charged, 1 GiB of it file pages
        let available = |name: &str, a: &str, b: &str, c: &str| {
            let root = cgroup_tree(
                name,
                &[
                    ("", "max", 11 * gib),
                    ("a", a, 11 * gib),
                    ("a/b", b, 11 * gib),
                    ("a/b/c", c, 11 * gib),
                ],
            );
            let mem = HostMemory {
                host_available: 64 * gib,
                groups: read_groups(&root, "/a/b/c"),
            };
            let _ = std::fs::remove_dir_all(&root);
            (mem.available(), mem.groups.len(), mem.describe())
        };
        // no limit anywhere: the host's figure
        let (left, n, text) = available("none", "max", "max", "max");
        assert_eq!((left, n), (64 * gib, 4));
        assert!(text.ends_with("no control group limit"), "{text}");
        // only on the process's own group
        assert_eq!(available("own", "max", "max", &g(30)).0, 20 * gib);
        // only on the parent: the group itself says `max`, and is bounded all the same
        let (left, _, text) = available("parent", "max", &g(30), "max");
        assert_eq!(left, 20 * gib);
        assert!(
            text.contains("30.00 GiB with 10.00 GiB in use (the group above)"),
            "{text}"
        );
        // only on the grandparent
        let (left, _, text) = available("grandparent", &g(26), "max", "max");
        assert_eq!(left, 16 * gib);
        assert!(text.contains("(2 groups above)"), "{text}");
        // on both, the parent tighter
        assert_eq!(
            available("parent-tighter", &g(50), &g(22), "max").0,
            12 * gib
        );
        // on both, the parent looser: the grandparent's is the one that counts
        assert_eq!(available("parent-looser", &g(18), &g(50), "max").0, 8 * gib);
        // a limit above what the host has available changes nothing
        assert_eq!(available("loose", &g(500), &g(400), &g(300)).0, 64 * gib);
        // a group that is already over its limit leaves nothing, whatever the others say
        assert_eq!(available("over", &g(9), &g(50), &g(40)).0, 0);
    }

    #[test]
    fn groups_are_read_from_the_process_s_own_up_to_the_mount() {
        let gib = 1u64 << 30;
        let root = cgroup_tree(
            "walk",
            &[
                ("", "max", gib),
                ("svc", &(8 * gib).to_string(), 3 * gib),
                ("svc/unit", "max", 2 * gib),
            ],
        );
        let groups = read_groups(&root, "/svc/unit\n");
        assert_eq!(
            groups,
            vec![
                Group {
                    limit: None,
                    in_use: Some(gib)
                },
                Group {
                    limit: Some(8 * gib),
                    in_use: Some(2 * gib)
                },
                Group {
                    limit: None,
                    in_use: Some(0)
                },
            ]
        );
        // a process in the mount's own group has that one group
        assert_eq!(read_groups(&root, "/").len(), 1);
        // a group whose files cannot be read has no limit
        assert_eq!(
            read_groups(&root, "/svc/gone"),
            vec![
                Group {
                    limit: None,
                    in_use: None
                },
                groups[1],
                groups[2],
            ]
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}
