//! Pipelined host-to-device weight upload.
//!
//! The link moves pinned host memory faster than one thread can copy a checkpoint into it, and a
//! copy-then-wait loop leaves the DMA idle during every copy and the host idle during every
//! transfer. [`H2dStreamer`] keeps both busy: the bytes are copied into a small ring of pinned
//! buffers by several threads, each filled piece is handed to the DMA at once without waiting for
//! it, and the only waits are for a ring buffer that is about to be reused while its last transfer
//! is still in flight (one device event per buffer).
//!
//! Bytes that are not in the page cache take a second path. Faulting a mapped checkpoint in
//! through the page cache reads the drive at roughly half of what it delivers to direct I/O, so a
//! large piece whose pages are mostly absent is read from its file with `O_DIRECT`, block-aligned,
//! by the same threads, straight into the pinned ring: no page cache, no second copy. Whether a
//! piece is in memory is asked of the kernel (`mincore`), and its file and offset come from this
//! process's own mapping table; anything that does not fit (no file behind the bytes, an
//! unsupported file system, a failed read) falls back to the copy from the mapping.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use crate::synapse::device::{DeviceEvent, HpuDevice, PinnedBuf};
use crate::synapse::ffi::SynapseError;

/// Default ring: four buffers of 64 MiB.
pub const RING_BUFS: usize = 4;
pub const RING_BUF_BYTES: usize = 64 * 1024 * 1024;
/// Pieces below this are copied on the calling thread.
const PAR_MIN_BYTES: usize = 4 * 1024 * 1024;
/// Pieces at least this large are candidates for a direct read.
const DIRECT_MIN_BYTES: usize = 8 * 1024 * 1024;
/// A piece is read directly when less than this share of its pages is in memory.
const DIRECT_RESIDENT_MAX: f64 = 0.5;
/// Alignment direct I/O needs for buffer address, file offset and length.
const BLOCK: usize = 4096;
/// Environment variable: `0` disables the direct-read path.
pub const DIRECT_ENV: &str = "GAUDIUM_UPLOAD_DIRECT";

/// Counters of one upload.
#[derive(Debug, Clone, Copy, Default)]
pub struct UploadStats {
    pub bytes: u64,
    pub transfers: u64,
    /// Host time copying into the pinned ring.
    pub copy_s: f64,
    /// Host time waiting for a ring buffer's transfer (incl. the final drain).
    pub wait_s: f64,
    /// Bytes read from their files with direct I/O instead of copied from the mapping.
    pub direct_bytes: u64,
    /// Host time in those direct reads.
    pub direct_s: f64,
}

pub struct H2dStreamer<'d> {
    dev: &'d HpuDevice,
    ring: Vec<PinnedBuf>,
    /// Completion marker of the last transfer queued from each ring buffer.
    events: Vec<DeviceEvent>,
    busy: Vec<bool>,
    cur: usize,
    fill: usize,
    threads: usize,
    stats: UploadStats,
    /// Direct reads are possible: enabled, and the ring is block-aligned.
    direct: bool,
    direct_files: HashMap<PathBuf, Option<Arc<std::fs::File>>>,
}

impl<'d> H2dStreamer<'d> {
    pub fn new(
        dev: &'d HpuDevice,
        bufs: usize,
        buf_bytes: usize,
        threads: usize,
    ) -> Result<Self, SynapseError> {
        let bufs = bufs.max(2);
        let mut ring = Vec::with_capacity(bufs);
        let mut events = Vec::with_capacity(bufs);
        for _ in 0..bufs {
            ring.push(dev.alloc_pinned(buf_bytes.max(4096))?);
            events.push(dev.create_event(false)?);
        }
        let aligned = ring.iter().all(|b| {
            (b.as_slice().as_ptr() as usize).is_multiple_of(BLOCK) && b.len() % BLOCK == 0
        });
        let enabled = std::env::var(DIRECT_ENV).map_or(true, |v| v.trim() != "0");
        Ok(Self {
            dev,
            ring,
            events,
            busy: vec![false; bufs],
            cur: 0,
            fill: 0,
            threads: threads.clamp(1, 16),
            stats: UploadStats::default(),
            direct: aligned && enabled && O_DIRECT.is_some(),
            direct_files: HashMap::new(),
        })
    }

    /// Copy threads for a host with `cores` logical CPUs: the staging copy is bound by memory
    /// traffic, so a few threads are enough and more only contend.
    pub fn default_threads() -> usize {
        thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .clamp(1, 4)
    }

    /// Queues `src` for `dst_hbm`. The bytes are staged before this returns (the caller may drop
    /// `src`); the transfers complete by [`Self::finish`].
    pub fn put(&mut self, src: &[u8], dst_hbm: u64) -> Result<(), SynapseError> {
        if self.direct
            && src.len() >= DIRECT_MIN_BYTES
            && resident_fraction(src) < DIRECT_RESIDENT_MAX
        {
            if let Some((path, file_off)) = mapped_file_region(src) {
                if let Some(file) = self.direct_file(&path) {
                    return self.put_direct(&file, file_off, src, dst_hbm);
                }
            }
        }
        if src.len() >= DIRECT_MIN_BYTES {
            // read through the page cache: let the kernel queue the whole piece at once
            crate::safetensors::advise_willneed(src);
        }
        self.put_copy(src, dst_hbm)
    }

    /// The file behind `path` opened for direct I/O (`None`: not possible; remembered).
    fn direct_file(&mut self, path: &Path) -> Option<Arc<std::fs::File>> {
        if let Some(known) = self.direct_files.get(path) {
            return known.clone();
        }
        let opened = open_direct(path).map(Arc::new);
        self.direct_files.insert(path.to_path_buf(), opened.clone());
        opened
    }

    /// `src` = the mapping of `[file_off, file_off + src.len())` of `file`: the whole blocks are
    /// read from the file into the ring, the partial block at either end is copied from `src`.
    fn put_direct(
        &mut self,
        file: &std::fs::File,
        file_off: u64,
        src: &[u8],
        dst_hbm: u64,
    ) -> Result<(), SynapseError> {
        let head = ((BLOCK as u64 - file_off % BLOCK as u64) % BLOCK as u64) as usize;
        let head = head.min(src.len());
        self.put_copy(&src[..head], dst_hbm)?;
        let body = (src.len() - head) / BLOCK * BLOCK;
        let mut done = 0usize;
        while done < body {
            let cap = self.ring[self.cur].len();
            let fill = self.fill.next_multiple_of(BLOCK);
            if fill >= cap {
                self.advance()?;
                continue;
            }
            let n = (body - done).min(cap - fill);
            let at = head + done;
            let t0 = Instant::now();
            let threads = self.threads;
            let dst = &mut self.ring[self.cur].as_mut_slice()[fill..fill + n];
            if read_file_parallel(file, file_off + at as u64, dst, threads) {
                self.stats.direct_s += t0.elapsed().as_secs_f64();
                self.stats.direct_bytes += n as u64;
            } else {
                // the read failed: this piece comes from the mapping, and so does the rest
                copy_parallel(&src[at..at + n], dst, threads);
                self.stats.copy_s += t0.elapsed().as_secs_f64();
                self.direct = false;
            }
            self.dev
                .enqueue_h2d_pinned(&self.ring[self.cur], fill, n, dst_hbm + at as u64)?;
            self.busy[self.cur] = true;
            self.stats.transfers += 1;
            self.stats.bytes += n as u64;
            self.fill = fill + n;
            done += n;
            if !self.direct {
                return self.put_copy(&src[at + n..], dst_hbm + (at + n) as u64);
            }
        }
        let tail = head + body;
        self.put_copy(&src[tail..], dst_hbm + tail as u64)
    }

    /// Copies `src` into the ring and queues its transfers.
    fn put_copy(&mut self, src: &[u8], dst_hbm: u64) -> Result<(), SynapseError> {
        let cap = self.ring[self.cur].len();
        let mut off = 0usize;
        while off < src.len() {
            if self.fill == cap {
                self.advance()?;
            }
            let n = (src.len() - off).min(cap - self.fill);
            let t0 = Instant::now();
            let fill = self.fill;
            let dst = &mut self.ring[self.cur].as_mut_slice()[fill..fill + n];
            copy_parallel(&src[off..off + n], dst, self.threads);
            self.stats.copy_s += t0.elapsed().as_secs_f64();
            self.dev
                .enqueue_h2d_pinned(&self.ring[self.cur], fill, n, dst_hbm + off as u64)?;
            self.busy[self.cur] = true;
            self.stats.transfers += 1;
            self.stats.bytes += n as u64;
            self.fill += n;
            off += n;
        }
        Ok(())
    }

    /// Closes the current ring buffer (marks its last transfer) and moves to the next one,
    /// waiting first if that one still has a transfer in flight.
    fn advance(&mut self) -> Result<(), SynapseError> {
        if self.busy[self.cur] {
            // SAFETY: the H2D stream belongs to `self.dev`.
            unsafe { self.events[self.cur].record(self.dev.h2d_stream())? };
        }
        self.cur = (self.cur + 1) % self.ring.len();
        self.fill = 0;
        if self.busy[self.cur] {
            let t0 = Instant::now();
            self.events[self.cur].synchronize()?;
            self.stats.wait_s += t0.elapsed().as_secs_f64();
            self.busy[self.cur] = false;
        }
        Ok(())
    }

    /// Waits for every queued transfer and returns the counters.
    pub fn finish(mut self) -> Result<UploadStats, SynapseError> {
        self.drain()?;
        Ok(self.stats)
    }

    fn drain(&mut self) -> Result<(), SynapseError> {
        let t0 = Instant::now();
        // SAFETY: the H2D stream belongs to `self.dev`.
        let res = unsafe { self.dev.sync_stream(self.dev.h2d_stream()) };
        self.stats.wait_s += t0.elapsed().as_secs_f64();
        self.busy.iter_mut().for_each(|b| *b = false);
        res
    }
}

impl Drop for H2dStreamer<'_> {
    fn drop(&mut self) {
        // Never free a pinned buffer a queued transfer still reads (error paths included).
        if self.busy.iter().any(|&b| b) {
            let _ = self.drain();
        }
    }
}

/// `O_DIRECT` of this target (`None`: no direct-read path).
#[cfg(all(target_os = "linux", any(target_arch = "x86_64", target_arch = "x86")))]
const O_DIRECT: Option<i32> = Some(0x4000);
#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
const O_DIRECT: Option<i32> = Some(0x10000);
#[cfg(not(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")
)))]
const O_DIRECT: Option<i32> = None;

fn open_direct(path: &Path) -> Option<std::fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(O_DIRECT?)
        .open(path)
        .ok()
}

#[cfg(unix)]
unsafe extern "C" {
    fn mincore(addr: *mut std::ffi::c_void, length: usize, vec: *mut u8) -> i32;
}

/// Share of the pages of `bytes` that are in memory now; `1.0` when the kernel cannot tell (so
/// the caller keeps the plain copy).
pub fn resident_fraction(bytes: &[u8]) -> f64 {
    if bytes.is_empty() {
        return 1.0;
    }
    let start = (bytes.as_ptr() as usize) & !(BLOCK - 1);
    let end = (bytes.as_ptr() as usize + bytes.len()).next_multiple_of(BLOCK);
    let pages = (end - start) / BLOCK;
    let mut vec = vec![0u8; pages];
    // SAFETY: `[start, end)` is the page-rounded range of a live slice, and `vec` has one byte
    // per page of it; `mincore` only writes `vec`.
    let rc = unsafe {
        mincore(
            start as *mut std::ffi::c_void,
            end - start,
            vec.as_mut_ptr(),
        )
    };
    if rc != 0 {
        return 1.0;
    }
    vec.iter().filter(|&&b| b & 1 != 0).count() as f64 / pages as f64
}

/// One line of `/proc/self/maps`: `(start, end, file offset, path)` of a file-backed mapping.
fn parse_maps_line(line: &str) -> Option<(usize, usize, u64, &str)> {
    let mut it = line.splitn(6, ' ');
    let (range, _perms, offset, _dev, inode) =
        (it.next()?, it.next()?, it.next()?, it.next()?, it.next()?);
    let path = it.next()?.trim_start();
    if inode == "0" || !path.starts_with('/') {
        return None;
    }
    let (a, b) = range.split_once('-')?;
    Some((
        usize::from_str_radix(a, 16).ok()?,
        usize::from_str_radix(b, 16).ok()?,
        u64::from_str_radix(offset, 16).ok()?,
        path,
    ))
}

/// The file region a mapped slice shows: `(path, file offset of bytes[0])`. `None` when the
/// slice is not wholly inside one file-backed mapping of this process.
pub fn mapped_file_region(bytes: &[u8]) -> Option<(PathBuf, u64)> {
    let (lo, hi) = (
        bytes.as_ptr() as usize,
        bytes.as_ptr() as usize + bytes.len(),
    );
    let maps = std::fs::read_to_string("/proc/self/maps").ok()?;
    maps.lines()
        .filter_map(parse_maps_line)
        .find_map(|(a, b, off, path)| {
            (a <= lo && hi <= b).then(|| (PathBuf::from(path), off + (lo - a) as u64))
        })
}

/// Fills `dst` from `file` at `offset` with `threads` positional readers (`dst.len()`, `offset`
/// and the buffer are block-aligned for a file opened with `O_DIRECT`). `false`: a read failed
/// or came up short, and `dst` must be filled another way.
pub fn read_file_parallel(
    file: &std::fs::File,
    offset: u64,
    dst: &mut [u8],
    threads: usize,
) -> bool {
    use std::os::unix::fs::FileExt;
    let fill = |part: &mut [u8], at: u64| -> bool {
        let mut got = 0usize;
        while got < part.len() {
            match file.read_at(&mut part[got..], at + got as u64) {
                Ok(0) | Err(_) => return false,
                Ok(n) => got += n,
            }
        }
        true
    };
    let n = dst.len();
    if threads <= 1 || n < PAR_MIN_BYTES {
        return fill(dst, offset);
    }
    let per = n.div_ceil(threads).next_multiple_of(BLOCK);
    thread::scope(|s| {
        let handles: Vec<_> = dst
            .chunks_mut(per)
            .enumerate()
            .map(|(i, part)| {
                let fill = &fill;
                s.spawn(move || fill(part, offset + (i * per) as u64))
            })
            .collect();
        handles.into_iter().all(|h| h.join().unwrap_or(false))
    })
}

/// `dst.copy_from_slice(src)` split over `threads` scoped threads for large pieces.
pub fn copy_parallel(src: &[u8], dst: &mut [u8], threads: usize) {
    debug_assert_eq!(src.len(), dst.len());
    let n = src.len();
    if threads <= 1 || n < PAR_MIN_BYTES {
        dst.copy_from_slice(src);
        return;
    }
    // whole pages per thread
    let per = n.div_ceil(threads).next_multiple_of(4096);
    thread::scope(|s| {
        for (d, c) in dst.chunks_mut(per).zip(src.chunks(per)) {
            s.spawn(move || d.copy_from_slice(c));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_lines_give_file_mappings_only() {
        let l = "7f10a0000000-7f10a4000000 r--p 00001000 103:02 5242881                   /data/model/L0_W_up.w8";
        assert_eq!(
            parse_maps_line(l),
            Some((
                0x7f10_a000_0000,
                0x7f10_a400_0000,
                0x1000,
                "/data/model/L0_W_up.w8"
            ))
        );
        // anonymous memory, special regions and malformed lines are not files
        for l in [
            "7f10a4000000-7f10a4021000 rw-p 00000000 00:00 0 ",
            "7ffc1a2b3000-7ffc1a2d4000 rw-p 00000000 00:00 0                          [stack]",
            "garbage",
        ] {
            assert_eq!(parse_maps_line(l), None, "{l}");
        }
    }

    #[test]
    fn a_mapped_file_slice_is_found_with_its_offset_and_read_back_in_parallel() {
        let dir = std::env::temp_dir().join(format!("gaudium-direct-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("blob.bin");
        let n = PAR_MIN_BYTES + 3 * BLOCK;
        let data: Vec<u8> = (0..n).map(|i| (i * 7 % 253) as u8).collect();
        std::fs::write(&path, &data).unwrap();
        let map = crate::safetensors::MappedShard::open(&path).unwrap();
        let bytes = map.as_bytes();
        // a slice that starts inside the file keeps its file offset
        let (found, off) = mapped_file_region(&bytes[BLOCK + 5..]).expect("file-backed");
        assert_eq!(found.file_name(), path.file_name());
        assert_eq!(off, (BLOCK + 5) as u64);
        // heap memory has no file behind it; touched pages are resident
        let heap = vec![1u8; 3 * BLOCK];
        assert!(mapped_file_region(&heap).is_none());
        assert_eq!(resident_fraction(&heap), 1.0);
        // positional parallel reads return exactly the file's bytes (plain open: the test
        // directory may not support direct I/O)
        let file = std::fs::File::open(&path).unwrap();
        for threads in [1usize, 3, 4] {
            let mut out = vec![0u8; PAR_MIN_BYTES];
            assert!(read_file_parallel(&file, BLOCK as u64, &mut out, threads));
            assert!(
                out == data[BLOCK..BLOCK + PAR_MIN_BYTES],
                "threads {threads}"
            );
        }
        // a read past the end of the file reports failure instead of leaving a hole
        let mut past = vec![0u8; 2 * BLOCK];
        assert!(!read_file_parallel(&file, (n - BLOCK) as u64, &mut past, 1));
        drop(map);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parallel_copy_equals_a_plain_copy_for_every_split() {
        let n = PAR_MIN_BYTES + 12_345;
        let src: Vec<u8> = (0..n).map(|i| (i * 31 % 251) as u8).collect();
        for threads in [1usize, 2, 3, 4, 7] {
            let mut dst = vec![0u8; n];
            copy_parallel(&src, &mut dst, threads);
            assert!(dst == src, "threads {threads}");
        }
        // small pieces take the single-thread path
        let mut small = vec![0u8; 100];
        copy_parallel(&src[..100], &mut small, 4);
        assert_eq!(small, &src[..100]);
    }
}
