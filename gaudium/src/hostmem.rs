//! Host memory for the engine's own large buffers: anonymous mappings that go back to the system
//! when they are dropped, what the host and its control groups can still give, and the one
//! admission every such buffer passes before it is mapped or pinned.
//!
//! `AnonMap` is used for buffers that are large, live for a while and must leave no trace when
//! they go: the segments of the host-RAM prefix cache (`cache_ram.rs`) and the transfer ring of
//! its copies (`synapse/xfer.rs`). A mapping is a whole number of 2 MiB pages and asks for
//! transparent huge pages; it is not touched here, so its pages exist only once written (or once
//! pinned for the device, which then faults them in as memory that never has to move again).
//!
//! Because the pages come later, a mapping that the host cannot back does not fail when it is
//! made: the kernel stops the process when the pages are written. So nothing here is mapped
//! without an [`Admitted`] leave: bytes that were asked for against what the host and every
//! control group above the process had left at that moment (the verified `hm_admit`).

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

/// Leave to take host memory now: so many bytes, asked for against what was available at that
/// moment. The one way this code maps or pins host memory of its own ([`AnonMap::new`] and the
/// pinned transfer ring draw from one); nothing but [`Admitted::ask`] makes one.
///
/// A leave is for memory that is written, or pinned, before the next one is asked for: what it
/// admitted is then part of the figures the next asking reads.
#[derive(Debug)]
pub struct Admitted {
    left: u64,
}

impl Admitted {
    /// Asks for `need` bytes with `available` bytes available at this moment ([`HostMemory`]
    /// read now, or the store's reading of it): the verified `hm_admit`. `None`: refused.
    pub(crate) fn ask(need: u64, available: u64) -> Option<Self> {
        crate::verified::hm_admit(need, available).then_some(Self { left: need })
    }

    /// Bytes not yet taken out of the leave.
    pub fn left(&self) -> u64 {
        self.left
    }

    /// Takes `bytes` out of the leave for one mapping or one pinned allocation (the verified
    /// `hm_covers`, `hm_left`). A leave that has less is an error and stays as it is.
    pub(crate) fn draw(&mut self, bytes: u64) -> Result<(), String> {
        if !crate::verified::hm_covers(self.left, bytes) {
            return Err(format!(
                "host memory: {bytes} bytes were not admitted ({} bytes of the leave are left)",
                self.left
            ));
        }
        self.left = crate::verified::hm_left(self.left, bytes);
        Ok(())
    }
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

    /// Maps room for `bytes` (see [`Self::mapped_len`]), taken out of `leave`. The pages are
    /// not touched.
    pub fn new(bytes: u64, leave: &mut Admitted) -> Result<Self, String> {
        let len = usize::try_from(Self::mapped_len(bytes))
            .map_err(|_| format!("a mapping of {bytes} bytes does not fit this address space"))?;
        leave.draw(len as u64)?;
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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Group {
    /// Limit of the group (`None`: it has none).
    pub limit: Option<u64>,
    /// The size above which the kernel slows the group down and reclaims from it (`memory.high`
    /// of a version 2 group; `None`: not set). A cache of anonymous memory cannot be reclaimed,
    /// so a process that grows past it is throttled at every page it takes: for a server that
    /// is a limit too.
    pub high: Option<u64>,
    /// Memory the group has in use that it cannot give back by dropping file pages (its own
    /// and that of every group below it).
    pub in_use: Option<u64>,
}

/// What the host can still give this process, in bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostMemory {
    /// `MemAvailable` of the host (or of the container's view of it).
    pub host_available: u64,
    /// The memory-controlled groups this process is in, its own first, then each group above
    /// it up to the top of what it can see. A limit on any of them bounds the process.
    pub groups: Vec<Group>,
    /// Why the limits of this process are not established, when they are not: what the host
    /// has available could not be read, or the process is in a memory-controlled group whose
    /// limit could not be. `available()` is then only what could be read.
    pub unread: Option<String>,
}

/// Which control group hierarchy holds the memory controller of a process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryHierarchy {
    /// The unified hierarchy (version 2): `memory.max`, `memory.high`, `memory.current`.
    Unified,
    /// A version 1 hierarchy with the memory controller: `memory.limit_in_bytes`,
    /// `memory.usage_in_bytes`, `memory.use_hierarchy`. Also the case on a hybrid host, where
    /// the unified hierarchy exists beside it and controls no memory.
    V1,
}

/// A version 1 limit at or above this is no limit (the kernel shows the largest page count it
/// can hold, in bytes).
const V1_NO_LIMIT: u64 = 1 << 62;

impl HostMemory {
    /// Reads what the host has available and the memory control groups of this process.
    pub fn read() -> Self {
        Self::read_at(std::path::Path::new("/"))
    }

    /// [`Self::read`] with every file looked up below `root` instead of `/` (tests give a
    /// made-up tree): `proc/meminfo`, `proc/self/cgroup`, `proc/self/mountinfo` and the mount
    /// points the last one names.
    pub fn read_at(root: &std::path::Path) -> Self {
        let text = |rel: &str| std::fs::read_to_string(root.join(rel));
        let (host_available, mut unread) = match text("proc/meminfo")
            .ok()
            .as_deref()
            .and_then(parse_meminfo_available)
        {
            Some(b) => (b, None),
            None => (
                u64::MAX,
                Some("what the host has available cannot be read from /proc/meminfo".to_owned()),
            ),
        };
        let mut groups = Vec::new();
        // A kernel without control groups has no such file: nothing bounds the process there.
        if let Ok(cgroup) = text("proc/self/cgroup") {
            if let Some((hierarchy, path)) = memory_group(&cgroup) {
                let mountinfo = text("proc/self/mountinfo").unwrap_or_default();
                match group_dirs(root, hierarchy, &path, &mountinfo) {
                    Ok((dirs, root_last)) => {
                        let (read, why) = read_chain(hierarchy, &dirs, root_last);
                        groups = read;
                        unread = unread.or(why);
                    }
                    Err(why) => unread = unread.or(Some(why)),
                }
            }
        }
        Self {
            host_available,
            groups,
            unread,
        }
    }

    /// A host with no control group limit (tests, and messages about a plain host).
    pub fn of_host(host_available: u64) -> Self {
        Self {
            host_available,
            groups: Vec::new(),
            unread: None,
        }
    }

    /// Are the limits of this process established: what the host has available was read, and
    /// so was the limit of every memory-controlled group the process is in?
    pub fn limits_known(&self) -> bool {
        self.unread.is_none()
    }

    /// Bytes this process can still take: what the host has available, and no more than any
    /// group of its chain has left before its limit or before the kernel slows it down. Each
    /// bound is taken into account by the verified `hc_chain_step`; the walk over the chain is
    /// all that happens here.
    pub fn available(&self) -> u64 {
        self.groups.iter().fold(self.host_available, |left, g| {
            [g.limit, g.high].into_iter().fold(left, |left, bound| {
                crate::verified::hc_chain_step(
                    left,
                    bound.is_some(),
                    bound.unwrap_or(0),
                    g.in_use.unwrap_or(0),
                )
            })
        })
    }

    /// Leave to take `need` bytes for `what` now, against this reading ([`Admitted::ask`]). An
    /// error says what was asked for and what there is.
    pub fn admit(&self, what: &str, need: u64) -> Result<Admitted, String> {
        let available = self.available();
        Admitted::ask(need, available).ok_or_else(|| {
            let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
            format!(
                "{what} needs {:.2} GiB of host memory, and {:.2} GiB are available now ({})",
                gib(need),
                gib(available),
                self.describe()
            )
        })
    }

    /// The numbers behind [`Self::available`], for a message: every group that has a limit,
    /// with how far above the process's own group it is, and what could not be read.
    pub fn describe(&self) -> String {
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        let limits: Vec<String> = self
            .groups
            .iter()
            .enumerate()
            .flat_map(|(up, g)| {
                let place = match up {
                    0 => "its own group".to_string(),
                    1 => "the group above".to_string(),
                    n => format!("{n} groups above"),
                };
                let in_use = gib(g.in_use.unwrap_or(0));
                [
                    g.limit
                        .map(|l| format!("{:.2} GiB with {in_use:.2} GiB in use ({place})", gib(l))),
                    g.high.map(|h| {
                        format!(
                            "{:.2} GiB before the kernel slows the group down, with {in_use:.2} GiB in use ({place})",
                            gib(h)
                        )
                    }),
                ]
            })
            .flatten()
            .collect();
        let host = if self.host_available == u64::MAX {
            "host available unknown".to_owned()
        } else {
            format!("host available {:.2} GiB", gib(self.host_available))
        };
        let groups = if limits.is_empty() {
            "no control group limit".to_owned()
        } else {
            format!("control group limit {}", limits.join(", "))
        };
        match &self.unread {
            Some(why) => format!("{host}; {groups}; not established: {why}"),
            None => format!("{host}; {groups}"),
        }
    }
}

/// The memory control group of a process from its `/proc/<pid>/cgroup` text: the hierarchy
/// that holds the memory controller and the group's path in it. A version 1 line that names
/// `memory` wins (a controller is in one hierarchy only, so on a hybrid host the unified line
/// then controls no memory); else the unified line `0::`. `None`: neither is there.
pub fn memory_group(cgroup: &str) -> Option<(MemoryHierarchy, String)> {
    let line = |l: &str| -> Option<(String, String, String)> {
        let mut it = l.splitn(3, ':');
        Some((
            it.next()?.to_owned(),
            it.next()?.to_owned(),
            it.next()?.trim().to_owned(),
        ))
    };
    let lines: Vec<_> = cgroup.lines().filter_map(line).collect();
    lines
        .iter()
        .find(|(_, ctl, _)| ctl.split(',').any(|c| c == "memory"))
        .map(|(_, _, path)| (MemoryHierarchy::V1, path.clone()))
        .or_else(|| {
            lines
                .iter()
                .find(|(id, ctl, _)| id == "0" && ctl.is_empty())
                .map(|(_, _, path)| (MemoryHierarchy::Unified, path.clone()))
        })
}

/// Where `hierarchy` is mounted, from a `/proc/<pid>/mountinfo` text: `(mount point, the path
/// inside the hierarchy that is mounted there)` of every such mount.
pub fn memory_mounts(mountinfo: &str, hierarchy: MemoryHierarchy) -> Vec<(String, String)> {
    // "36 25 0:31 / /sys/fs/cgroup/memory rw,nosuid - cgroup cgroup rw,memory": the fields
    // before " - " are id, parent, device, root, mount point, options, ...; after it the file
    // system type, the source and the options of the file system itself.
    let unescape = |s: &str| {
        s.replace("\\040", " ")
            .replace("\\011", "\t")
            .replace("\\012", "\n")
            .replace("\\134", "\\")
    };
    mountinfo
        .lines()
        .filter_map(|l| {
            let (left, right) = l.split_once(" - ")?;
            let mut r = right.split_whitespace();
            let (fstype, _source, options) = (r.next()?, r.next()?, r.next().unwrap_or(""));
            let ours = match hierarchy {
                MemoryHierarchy::Unified => fstype == "cgroup2",
                MemoryHierarchy::V1 => {
                    fstype == "cgroup" && options.split(',').any(|o| o == "memory")
                }
            };
            let mut f = left.split_whitespace();
            let (root, point) = (f.nth(3)?, f.next()?);
            ours.then(|| (unescape(point), unescape(root)))
        })
        .collect()
}

/// The directories of the group `path` of `hierarchy` and of every group above it, the
/// process's own first and the top of the mounted hierarchy last, looked up below `root`;
/// and whether that last one is the root group of the hierarchy (a mount can show a part of
/// a hierarchy, and its top is then a group like any other). An error says why they cannot be
/// found (the hierarchy is mounted nowhere this process can read, or the mount does not show
/// the group).
pub fn group_dirs(
    root: &std::path::Path,
    hierarchy: MemoryHierarchy,
    path: &str,
    mountinfo: &str,
) -> Result<(Vec<std::path::PathBuf>, bool), String> {
    use std::path::{Path, PathBuf};
    let below = |abs: &str| root.join(abs.trim_start_matches('/'));
    let group = Path::new(path);
    // the mount that shows the group: the one whose mounted part is the longest prefix of it
    let mut mounts = memory_mounts(mountinfo, hierarchy);
    mounts.sort_by_key(|(_, shown)| std::cmp::Reverse(shown.len()));
    let outside = || {
        format!(
            "the memory control group of this process ({path}) is not inside what its control group mount shows"
        )
    };
    let (top, rel, root_last): (PathBuf, PathBuf, bool) =
        match mounts.iter().find_map(|(point, shown)| {
            let rel = group.strip_prefix(shown).ok()?.to_owned();
            Some((below(point), rel, shown == "/"))
        }) {
            Some(found) => found,
            None if mounts.is_empty() => {
                // no mount table (or none for this hierarchy): the usual places
                let usual = match hierarchy {
                    MemoryHierarchy::Unified => "/sys/fs/cgroup",
                    MemoryHierarchy::V1 => "/sys/fs/cgroup/memory",
                };
                (
                    below(usual),
                    group.strip_prefix("/").unwrap_or(group).to_owned(),
                    true,
                )
            }
            None => return Err(outside()),
        };
    // a group above what the process can see (a path that climbs out of its own view)
    if rel
        .components()
        .any(|c| !matches!(c, std::path::Component::Normal(_)))
    {
        return Err(outside());
    }
    if !top.is_dir() {
        return Err(format!(
            "the control group hierarchy that limits the memory of this process is not mounted where it can be read ({})",
            top.strip_prefix(root).map_or_else(|_| top.display().to_string(), |p| format!("/{}", p.display()))
        ));
    }
    Ok((rel.ancestors().map(|up| top.join(up)).collect(), root_last))
}

/// Reads the groups of a chain of directories ([`group_dirs`]): the groups, and why the
/// limits are not established when one of a memory-controlled group could not be read (the
/// groups that could be read are returned all the same).
///
/// `root_last`: the last directory is the root group of the hierarchy, which has no limit
/// and, in the unified hierarchy, no limit file. Any other group whose limit file is missing
/// has no limit only when its memory is not controlled at that level (a version 2 group lists
/// its controllers); every other missing or unreadable file leaves the limit unknown.
pub fn read_chain(
    hierarchy: MemoryHierarchy,
    dirs: &[std::path::PathBuf],
    root_last: bool,
) -> (Vec<Group>, Option<String>) {
    use std::io::ErrorKind::NotFound;
    let mut groups = Vec::new();
    let mut unread: Option<String> = None;
    let mut note = |why: String| {
        if unread.is_none() {
            unread = Some(why);
        }
    };
    for (i, dir) in dirs.iter().enumerate() {
        let is_root = root_last && i + 1 == dirs.len();
        let place = match i {
            0 => "its own group".to_string(),
            1 => "the group above".to_string(),
            n => format!("{n} groups above"),
        };
        let read = |name: &str| std::fs::read_to_string(dir.join(name));
        let number = |name: &str| read(name).ok().and_then(|t| t.trim().parse::<u64>().ok());
        // the bounds of the group first; what it has in use is read only for a group that has
        // one (its `memory.stat` is the costly file, and most groups of a chain have no bound)
        let (limit, high) = match hierarchy {
            MemoryHierarchy::Unified => {
                // `Some(limit)`: read (`None` inside: "max"); `None`: not established
                let bound = |name: &str| -> Option<Option<u64>> {
                    match read(name) {
                        Ok(t) if t.trim() == "max" => Some(None),
                        Ok(t) => t.trim().parse::<u64>().ok().map(Some),
                        Err(_) => None,
                    }
                };
                let limit = match read("memory.max") {
                    Ok(_) => {
                        let read = bound("memory.max");
                        if read.is_none() {
                            note(format!("memory.max of {place} is not a number"));
                        }
                        read
                    }
                    Err(e) if e.kind() == NotFound && is_root => Some(None),
                    Err(e) if e.kind() == NotFound => match read("cgroup.controllers") {
                        // memory is not controlled at this level: the groups above bound it
                        Ok(c) if !c.split_whitespace().any(|w| w == "memory") => Some(None),
                        Ok(_) => {
                            note(format!("{place} controls memory and has no memory.max"));
                            None
                        }
                        Err(e) => {
                            note(format!("the directory of {place} cannot be read ({e})"));
                            None
                        }
                    },
                    Err(e) => {
                        note(format!("memory.max of {place} cannot be read ({e})"));
                        None
                    }
                };
                // a missing `memory.high` is one that is not set
                (limit.flatten(), bound("memory.high").flatten())
            }
            MemoryHierarchy::V1 => {
                let limit = match read("memory.limit_in_bytes") {
                    Ok(t) => match t.trim().parse::<u64>() {
                        Ok(l) => (l < V1_NO_LIMIT).then_some(l),
                        Err(_) => {
                            note(format!("memory.limit_in_bytes of {place} is not a number"));
                            None
                        }
                    },
                    Err(e) if e.kind() == NotFound && is_root => None,
                    Err(e) => {
                        note(format!(
                            "memory.limit_in_bytes of {place} cannot be read ({e})"
                        ));
                        None
                    }
                };
                (limit, None)
            }
        };
        let bounded = limit.is_some() || high.is_some();
        let in_use = match hierarchy {
            _ if !bounded => None,
            MemoryHierarchy::Unified => {
                let stat = read("memory.stat").unwrap_or_default();
                parse_cgroup("", &read("memory.current").unwrap_or_default(), &stat).1
            }
            MemoryHierarchy::V1 => {
                let stat = read("memory.stat").unwrap_or_default();
                number("memory.usage_in_bytes").map(|used| used.saturating_sub(v1_droppable(&stat)))
            }
        };
        let group = Group {
            limit,
            high,
            in_use,
        };
        // a bound without the bytes in use says nothing about what is left
        if bounded && group.in_use.is_none() {
            note(format!("the memory in use of {place} cannot be read"));
        }
        groups.push(group);
        // Version 1: the group above accounts for this one only when it is hierarchical. A
        // file that cannot be read is taken as hierarchical (every kernel since 5.11 is).
        if hierarchy == MemoryHierarchy::V1 {
            if let Some(above) = dirs.get(i + 1) {
                let flat = std::fs::read_to_string(above.join("memory.use_hierarchy"))
                    .is_ok_and(|t| t.trim() == "0");
                if flat {
                    break;
                }
            }
        }
    }
    (groups, unread)
}

/// File pages of a version 1 group and of the groups below it (`total_...` of its
/// `memory.stat`; a group that is not hierarchical has only its own).
fn v1_droppable(stat: &str) -> u64 {
    let field = |name: &str| -> Option<u64> {
        stat.lines().find_map(|l| {
            let mut it = l.split_whitespace();
            (it.next() == Some(name)).then(|| it.next()?.parse::<u64>().ok())?
        })
    };
    let pages = |kind: &str| {
        field(&format!("total_{kind}"))
            .or_else(|| field(kind))
            .unwrap_or(0)
    };
    pages("inactive_file").saturating_add(pages("active_file"))
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
        let mut leave = Admitted::ask(2 * HUGE_PAGE, 2 * HUGE_PAGE).unwrap();
        let mut m = AnonMap::new(3 * 1024 * 1024, &mut leave).unwrap();
        assert_eq!(leave.left(), 0);
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
            let mut leave = Admitted::ask(n as u64 * AnonMap::mapped_len(bytes), u64::MAX).unwrap();
            let mut maps: Vec<AnonMap> = (0..n)
                .map(|_| AnonMap::new(bytes, &mut leave).unwrap())
                .collect();
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

    /// What one reading of the host and the control groups costs on this machine (`cargo test
    /// --release -- --ignored reading_cost --nocapture`): every admission makes one.
    #[test]
    #[ignore]
    fn reading_cost() {
        let t0 = std::time::Instant::now();
        let n = 2000;
        let mut last = HostMemory::of_host(0);
        for _ in 0..n {
            last = HostMemory::read();
        }
        println!(
            "{:.1} us per reading; {} group(s); {}",
            t0.elapsed().as_secs_f64() * 1e6 / n as f64,
            last.groups.len(),
            last.describe()
        );
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
                high: None,
                in_use: in_use.map(|u| u * gib),
            }],
            unread: None,
        };
        assert_eq!(own(Some(86), Some(40), 80).available(), 46 * gib);
        assert_eq!(own(Some(86), Some(40), 20).available(), 20 * gib);
        assert_eq!(own(None, None, 20).available(), 20 * gib);
        assert_eq!(HostMemory::of_host(20 * gib).available(), 20 * gib);
        // a group over its limit has nothing left
        assert_eq!(own(Some(10), Some(12), 20).available(), 0);
    }

    /// The groups of the unified hierarchy mounted at `mount` for a process in `rel`.
    fn read_groups(mount: &std::path::Path, rel: &str) -> (Vec<Group>, Option<String>) {
        let rel = std::path::Path::new(rel.trim().trim_start_matches('/'));
        let dirs: Vec<_> = rel.ancestors().map(|up| mount.join(up)).collect();
        read_chain(MemoryHierarchy::Unified, &dirs, true)
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
            let (groups, unread) = read_groups(&root, "/a/b/c");
            let mem = HostMemory {
                host_available: 64 * gib,
                groups,
                unread,
            };
            assert!(mem.limits_known());
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
        let (groups, unread) = read_groups(&root, "/svc/unit\n");
        assert_eq!(unread, None);
        let group = |limit: Option<u64>, in_use: Option<u64>| Group {
            limit,
            high: None,
            in_use,
        };
        // what a group has in use is read only where a bound makes it matter
        assert_eq!(
            groups,
            vec![
                group(None, None),
                group(Some(8 * gib), Some(2 * gib)),
                group(None, None),
            ]
        );
        // a process in the mount's own group has that one group
        assert_eq!(read_groups(&root, "/").0.len(), 1);
        // a group whose directory cannot be read: the groups above are read all the same, and
        // the limits are not established
        let (gone, unread) = read_groups(&root, "/svc/gone");
        assert_eq!(gone, vec![group(None, None), groups[1], groups[2]]);
        assert!(
            unread
                .as_deref()
                .is_some_and(|w| w.contains("its own group")),
            "{unread:?}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A made-up system below a directory of its own, removed when it is dropped: the files
    /// given, as `(path below the root, content)`. A path that ends in `/` is a directory.
    struct Tree(std::path::PathBuf);

    impl Tree {
        fn new(name: &str, files: &[(&str, String)]) -> Self {
            let root = std::env::temp_dir().join(format!(
                "gaudium-hostmem-test-{}-{name}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&root);
            for (path, content) in files {
                let at = root.join(path);
                if path.ends_with('/') {
                    std::fs::create_dir_all(&at).unwrap();
                } else {
                    std::fs::create_dir_all(at.parent().unwrap()).unwrap();
                    std::fs::write(&at, content).unwrap();
                }
            }
            Self(root)
        }

        fn read(&self) -> HostMemory {
            HostMemory::read_at(&self.0)
        }
    }

    impl Drop for Tree {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    const GIB: u64 = 1 << 30;

    /// `proc/meminfo` of a host with 64 GiB available.
    fn meminfo() -> (&'static str, String) {
        (
            "proc/meminfo",
            format!(
                "MemTotal: 134217728 kB\nMemAvailable: {} kB\n",
                64 * GIB / 1024
            ),
        )
    }

    fn file(path: &'static str, content: impl ToString) -> (&'static str, String) {
        (path, content.to_string())
    }

    const UNIFIED_MOUNT: &str =
        "35 27 0:29 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime shared:8 - cgroup2 cgroup2 rw,nsdelegate\n";
    const V1_MOUNT: &str =
        "40 30 0:35 / /sys/fs/cgroup/memory rw,nosuid,nodev,noexec,relatime shared:15 - cgroup cgroup rw,memory\n";
    /// What the kernel shows for a version 1 group without a limit.
    const V1_UNLIMITED: u64 = 9_223_372_036_854_771_712;

    #[test]
    fn a_unified_hierarchy_is_read_with_its_limits_and_its_throttle_size() {
        // the process is in /svc/unit; /svc has a limit; /mid controls no memory; the root
        // group of the hierarchy has no limit file
        let stat = format!("anon 0\ninactive_file {GIB}\nactive_file 0\n");
        let tree = |name: &str, unit_high: &str| {
            Tree::new(
                name,
                &[
                    meminfo(),
                    file("proc/self/cgroup", "0::/svc/mid/unit\n"),
                    file("proc/self/mountinfo", UNIFIED_MOUNT),
                    file("sys/fs/cgroup/cgroup.controllers", "cpu memory pids\n"),
                    file("sys/fs/cgroup/svc/cgroup.controllers", "cpu memory pids\n"),
                    file("sys/fs/cgroup/svc/memory.max", 30 * GIB),
                    file("sys/fs/cgroup/svc/memory.high", "max\n"),
                    file("sys/fs/cgroup/svc/memory.current", 11 * GIB),
                    file("sys/fs/cgroup/svc/memory.stat", &stat),
                    // memory is not controlled at this level: no memory files at all
                    file("sys/fs/cgroup/svc/mid/cgroup.controllers", "cpu pids\n"),
                    file(
                        "sys/fs/cgroup/svc/mid/unit/cgroup.controllers",
                        "cpu pids\n",
                    ),
                    file("sys/fs/cgroup/svc/mid/unit/memory.max", "max\n"),
                    file("sys/fs/cgroup/svc/mid/unit/memory.high", unit_high),
                    file("sys/fs/cgroup/svc/mid/unit/memory.current", 5 * GIB),
                    file("sys/fs/cgroup/svc/mid/unit/memory.stat", &stat),
                ],
            )
            .read()
        };
        let mem = tree("unified", "max\n");
        assert_eq!(mem.unread, None, "{mem:?}");
        assert_eq!(mem.groups.len(), 4);
        assert_eq!(mem.groups[2].limit, Some(30 * GIB));
        assert_eq!(mem.groups[1], Group::default());
        // the limit two groups above: 30 GiB, 11 GiB charged, 1 GiB of it file pages
        assert_eq!(mem.available(), 20 * GIB);
        assert!(
            mem.describe()
                .contains("30.00 GiB with 10.00 GiB in use (2 groups above)"),
            "{}",
            mem.describe()
        );
        // the size above which the kernel slows the group down bounds it like a limit
        let mem = tree("unified-high", &(12 * GIB).to_string());
        assert_eq!(mem.unread, None);
        assert_eq!(mem.groups[0].high, Some(12 * GIB));
        assert_eq!(mem.available(), 8 * GIB);
        assert!(
            mem.describe()
                .contains("12.00 GiB before the kernel slows the group down"),
            "{}",
            mem.describe()
        );
    }

    #[test]
    fn a_version_1_memory_hierarchy_is_read_with_the_groups_above() {
        let cgroup = "12:memory:/pods/abc\n11:cpu,cpuacct:/pods/abc\n1:name=systemd:/pods/abc\n";
        // 2 GiB charged below /pods/abc, 1 GiB of it file pages of the whole subtree (the
        // group's own are fewer)
        let stat = format!(
            "cache 0\ninactive_file 4096\nactive_file 0\ntotal_inactive_file {}\ntotal_active_file {}\n",
            GIB / 2,
            GIB / 2
        );
        let tree = |name: &str, own: u64, above: u64, above_hierarchical: &str| {
            Tree::new(
                name,
                &[
                    meminfo(),
                    file("proc/self/cgroup", cgroup),
                    file("proc/self/mountinfo", V1_MOUNT),
                    file("sys/fs/cgroup/memory/memory.limit_in_bytes", V1_UNLIMITED),
                    file("sys/fs/cgroup/memory/memory.usage_in_bytes", 40 * GIB),
                    file("sys/fs/cgroup/memory/memory.use_hierarchy", "1\n"),
                    file("sys/fs/cgroup/memory/pods/memory.limit_in_bytes", above),
                    file("sys/fs/cgroup/memory/pods/memory.usage_in_bytes", 9 * GIB),
                    file("sys/fs/cgroup/memory/pods/memory.stat", &stat),
                    file(
                        "sys/fs/cgroup/memory/pods/memory.use_hierarchy",
                        above_hierarchical,
                    ),
                    file("sys/fs/cgroup/memory/pods/abc/memory.limit_in_bytes", own),
                    file(
                        "sys/fs/cgroup/memory/pods/abc/memory.usage_in_bytes",
                        2 * GIB,
                    ),
                    file("sys/fs/cgroup/memory/pods/abc/memory.stat", &stat),
                    file("sys/fs/cgroup/memory/pods/abc/memory.use_hierarchy", "1\n"),
                ],
            )
            .read()
        };
        // a limit on the process's own group: 6 GiB, 2 GiB charged, 1 GiB of it file pages
        let mem = tree("v1-own", 6 * GIB, V1_UNLIMITED, "1\n");
        assert_eq!(mem.unread, None, "{mem:?}");
        assert_eq!(mem.groups.len(), 3);
        assert_eq!(
            mem.groups[0],
            Group {
                limit: Some(6 * GIB),
                high: None,
                in_use: Some(GIB)
            }
        );
        // "no limit" as the kernel shows it is no limit
        assert_eq!((mem.groups[1].limit, mem.groups[2].limit), (None, None));
        assert_eq!(mem.available(), 5 * GIB);
        // a limit on the group above only: it bounds the process through the hierarchy
        let mem = tree("v1-above", V1_UNLIMITED, 12 * GIB, "1\n");
        assert_eq!(mem.unread, None);
        assert_eq!(mem.available(), 4 * GIB);
        assert!(
            mem.describe().contains("(the group above)"),
            "{}",
            mem.describe()
        );
        // ... unless that group does not account for the groups below it
        let mem = tree("v1-flat", V1_UNLIMITED, 12 * GIB, "0\n");
        assert_eq!(mem.unread, None);
        assert_eq!(mem.groups.len(), 1);
        assert_eq!(mem.available(), 64 * GIB);
    }

    #[test]
    fn a_hybrid_host_is_read_where_its_memory_is_controlled() {
        // the unified hierarchy exists and names the process, but controls no memory: the
        // limit is in the version 1 hierarchy
        let cgroup = "5:memory:/user.slice/x.scope\n1:name=systemd:/user.slice/x.scope\n0::/user.slice/x.scope\n";
        let mountinfo = format!(
            "36 27 0:30 / /sys/fs/cgroup/unified rw,nosuid - cgroup2 cgroup2 rw,nsdelegate\n{V1_MOUNT}"
        );
        let mem = Tree::new(
            "hybrid",
            &[
                meminfo(),
                file("proc/self/cgroup", cgroup),
                file("proc/self/mountinfo", &mountinfo),
                // what a reader of the unified hierarchy alone would find: nothing
                file("sys/fs/cgroup/unified/cgroup.controllers", "\n"),
                file("sys/fs/cgroup/unified/user.slice/cgroup.controllers", "\n"),
                file(
                    "sys/fs/cgroup/unified/user.slice/x.scope/cgroup.controllers",
                    "\n",
                ),
                file("sys/fs/cgroup/memory/memory.limit_in_bytes", V1_UNLIMITED),
                file(
                    "sys/fs/cgroup/memory/user.slice/memory.limit_in_bytes",
                    V1_UNLIMITED,
                ),
                file(
                    "sys/fs/cgroup/memory/user.slice/x.scope/memory.limit_in_bytes",
                    3 * GIB,
                ),
                file(
                    "sys/fs/cgroup/memory/user.slice/x.scope/memory.usage_in_bytes",
                    GIB,
                ),
            ],
        )
        .read();
        assert_eq!(
            memory_group(cgroup),
            Some((MemoryHierarchy::V1, "/user.slice/x.scope".to_owned()))
        );
        assert_eq!(mem.unread, None, "{mem:?}");
        assert_eq!(mem.groups[0].limit, Some(3 * GIB));
        assert_eq!(mem.available(), 2 * GIB);
        // a hybrid host whose memory controller is in the unified hierarchy is read there
        assert_eq!(
            memory_group("1:name=systemd:/a\n0::/b\n"),
            Some((MemoryHierarchy::Unified, "/b".to_owned()))
        );
        assert_eq!(memory_group("garbage\n"), None);
    }

    #[test]
    fn a_mount_that_shows_part_of_a_hierarchy_is_followed_and_one_that_hides_the_group_is_not() {
        // a container that sees the hierarchy from its own group down: the group's files are
        // at the mount point
        let cgroup = "9:memory:/pods/abc\n";
        let files = |mountinfo: &str| {
            vec![
                meminfo(),
                file("proc/self/cgroup", cgroup),
                ("proc/self/mountinfo", mountinfo.to_owned()),
                file("sys/fs/cgroup/memory/memory.limit_in_bytes", 5 * GIB),
                file("sys/fs/cgroup/memory/memory.usage_in_bytes", GIB),
            ]
        };
        let shown =
            "40 30 0:35 /pods/abc /sys/fs/cgroup/memory ro,nosuid - cgroup cgroup rw,memory\n";
        let mem = Tree::new("part", &files(shown)).read();
        assert_eq!(mem.unread, None, "{mem:?}");
        assert_eq!((mem.groups.len(), mem.available()), (1, 4 * GIB));
        // the top of such a mount is a group like any other: without its limit file the
        // limit is unknown (the root group of a hierarchy may lack one)
        let mut bare = files(shown);
        bare.remove(3);
        let mem = Tree::new("part-bare", &bare).read();
        assert!(
            mem.unread.as_deref().is_some_and(
                |w| w.contains("memory.limit_in_bytes of its own group cannot be read")
            ),
            "{mem:?}"
        );
        // a group above what the process can see
        let above = Tree::new(
            "above",
            &[
                meminfo(),
                file("proc/self/cgroup", "0::/../outside\n"),
                file("proc/self/mountinfo", UNIFIED_MOUNT),
                file("sys/fs/cgroup/cgroup.controllers", "memory\n"),
            ],
        )
        .read();
        assert!(!above.limits_known(), "{above:?}");
        // a mount that shows another part of the hierarchy: the group's limit cannot be found
        let other =
            "40 30 0:35 /pods/zzz /sys/fs/cgroup/memory ro,nosuid - cgroup cgroup rw,memory\n";
        let mem = Tree::new("hidden", &files(other)).read();
        assert!(!mem.limits_known());
        assert!(
            mem.unread
                .as_deref()
                .unwrap()
                .contains("not inside what its control group mount shows"),
            "{:?}",
            mem.unread
        );
        // a mount point with a space in it is unescaped
        assert_eq!(
            memory_mounts(
                "40 30 0:35 / /mnt/my\\040groups rw - cgroup2 cgroup2 rw\n",
                MemoryHierarchy::Unified
            ),
            vec![("/mnt/my groups".to_owned(), "/".to_owned())]
        );
    }

    #[test]
    fn limits_that_cannot_be_read_are_not_established() {
        let unified = |name: &str, files: &[(&'static str, String)]| {
            let mut all = vec![
                meminfo(),
                file("proc/self/cgroup", "0::/svc\n"),
                file("proc/self/mountinfo", UNIFIED_MOUNT),
                file("sys/fs/cgroup/cgroup.controllers", "memory pids\n"),
            ];
            all.extend_from_slice(files);
            Tree::new(name, &all).read()
        };
        let why = |mem: &HostMemory| mem.unread.clone().unwrap_or_default();
        // everything readable: established
        let mem = unified(
            "ok",
            &[
                file("sys/fs/cgroup/svc/memory.max", 8 * GIB),
                file("sys/fs/cgroup/svc/memory.current", GIB),
            ],
        );
        assert!(mem.limits_known(), "{mem:?}");
        assert_eq!(mem.available(), 7 * GIB);
        // a group that controls memory and has no limit file
        let mem = unified(
            "no-max",
            &[file(
                "sys/fs/cgroup/svc/cgroup.controllers",
                "memory pids\n",
            )],
        );
        assert!(
            why(&mem).contains("controls memory and has no memory.max"),
            "{mem:?}"
        );
        // a limit file that cannot be read (here: a directory in its place)
        let mem = unified("unreadable", &[file("sys/fs/cgroup/svc/memory.max/", "")]);
        assert!(
            why(&mem).contains("memory.max of its own group cannot be read"),
            "{mem:?}"
        );
        // a limit file with something else than a size in it
        let mem = unified("garbage", &[file("sys/fs/cgroup/svc/memory.max", "lots\n")]);
        assert!(why(&mem).contains("is not a number"), "{mem:?}");
        // a limit without the bytes in use
        let mem = unified(
            "no-current",
            &[file("sys/fs/cgroup/svc/memory.max", 8 * GIB)],
        );
        assert!(
            why(&mem).contains("memory in use of its own group cannot be read"),
            "{mem:?}"
        );
        // what could be read still bounds the process
        assert_eq!(mem.groups[0].limit, Some(8 * GIB));
        // the group's directory is not there
        let mem = unified("no-dir", &[]);
        assert!(
            why(&mem).contains("directory of its own group cannot be read"),
            "{mem:?}"
        );
        assert!(
            mem.describe().contains("not established"),
            "{}",
            mem.describe()
        );

        // version 1: the group has no limit file
        let v1 = |name: &str, files: &[(&'static str, String)]| {
            let mut all = vec![meminfo(), file("proc/self/cgroup", "7:memory:/svc\n")];
            all.extend_from_slice(files);
            Tree::new(name, &all).read()
        };
        let mem = v1(
            "v1-no-limit-file",
            &[
                file("proc/self/mountinfo", V1_MOUNT),
                file("sys/fs/cgroup/memory/memory.limit_in_bytes", V1_UNLIMITED),
                file("sys/fs/cgroup/memory/svc/memory.usage_in_bytes", GIB),
            ],
        );
        assert!(
            why(&mem).contains("memory.limit_in_bytes of its own group cannot be read"),
            "{mem:?}"
        );
        // version 1: the hierarchy is mounted nowhere this process can read
        let mem = v1("v1-no-mount", &[file("proc/self/mountinfo", UNIFIED_MOUNT)]);
        assert!(
            why(&mem).contains("not mounted where it can be read"),
            "{mem:?}"
        );
        assert!(why(&mem).contains("/sys/fs/cgroup/memory"), "{mem:?}");

        // what the host has available cannot be read at all
        let mem = Tree::new("no-meminfo", &[file("proc/self/cgroup", "0::/\n")]).read();
        assert!(why(&mem).contains("/proc/meminfo"), "{mem:?}");
        assert_eq!(mem.host_available, u64::MAX);
        assert!(
            mem.describe().starts_with("host available unknown"),
            "{}",
            mem.describe()
        );

        // a kernel without control groups: nothing bounds the process, and that is known
        let mem = Tree::new("no-cgroups", &[meminfo()]).read();
        assert!(mem.limits_known());
        assert_eq!((mem.groups.len(), mem.available()), (0, 64 * GIB));
    }

    #[test]
    fn nothing_is_mapped_beyond_what_was_admitted() {
        // a leave is refused beyond what is available at that moment
        assert!(Admitted::ask(4 * HUGE_PAGE, 4 * HUGE_PAGE).is_some());
        assert!(Admitted::ask(4 * HUGE_PAGE, 4 * HUGE_PAGE - 1).is_none());
        assert!(Admitted::ask(1, 0).is_none());
        // a reading admits what it has left, and says what is missing otherwise
        let mem = HostMemory {
            host_available: 64 * GIB,
            groups: vec![
                Group::default(),
                Group {
                    limit: Some(3 * GIB),
                    high: None,
                    in_use: Some(GIB),
                },
            ],
            unread: None,
        };
        assert_eq!(mem.admit("a buffer", 2 * GIB).unwrap().left(), 2 * GIB);
        let e = mem.admit("a buffer", 2 * GIB + 1).unwrap_err();
        assert!(
            e.contains("a buffer needs 2.00 GiB of host memory, and 2.00 GiB are available now")
                && e.contains("3.00 GiB with 1.00 GiB in use (the group above)"),
            "{e}"
        );
        // mappings are taken out of the leave, and one it does not cover is not made
        let mut leave = Admitted::ask(3 * HUGE_PAGE, u64::MAX).unwrap();
        let a = AnonMap::new(HUGE_PAGE + 1, &mut leave).unwrap();
        assert_eq!((a.len() as u64, leave.left()), (2 * HUGE_PAGE, HUGE_PAGE));
        let e = AnonMap::new(HUGE_PAGE + 1, &mut leave).unwrap_err();
        assert!(e.contains("were not admitted"), "{e}");
        assert_eq!(leave.left(), HUGE_PAGE);
        let _b = AnonMap::new(HUGE_PAGE, &mut leave).unwrap();
        assert_eq!(leave.left(), 0);
        assert!(AnonMap::new(1, &mut leave).is_err());
    }
}
