//! Trimming the on-disk compiled-recipe cache to a byte budget.
//!
//! An entry is one compiled recipe: `<name>-<hash>.recipe` and its `<name>-<hash>.sha256`
//! sidecar (`<hash>` = the 16 lower-case hex digits `RecipeCache` names files with). They are
//! removed together, sidecar first, so an entry stops being a cache hit the moment its first file
//! goes. Any other file in the directory (FP8 tables, temporary files, files of other programs,
//! links, directories) is never touched and does not count against the budget.
//!
//! Recency: every cache hit sets the sidecar's modification time to now
//! (`RecipeCache::lookup_verified`), so recency does not depend on access times, which a
//! `noatime` or `relatime` mount does not keep. An entry's last use is the later of its two
//! files' modification times.
//!
//! The decisions are the verified core's (`verified/recipe_trim.bend`): what is protected (an
//! entry this process holds, one used within the keep window, one dated ahead of the clock), the
//! eviction order (least recently used first), when the walk removes the next entry (only while
//! the cache is over its budget) and the count of the bytes still held. This file scans, reads
//! times and removes files.
//!
//! Running beside a server on the same directory is safe: a trim that removes an entry a load is
//! reading makes that load a cache miss. The load reads the sidecar, then the recipe, and hashes
//! the bytes it read: a file that is gone fails the read, and a file that was read whole matches
//! its hash (a removed file's bytes stay readable to whoever has it open; nothing truncates them).

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::synapse::recipe_cache::{in_use_registry, refresh_recency, RECIPE_EXT, SHA256_EXT};
use crate::verified::{rt_after, rt_evict_before, rt_last_used, rt_protected, rt_should_evict};

/// Default keep window of `--cache-keep-days`, in days.
pub const DEFAULT_KEEP_DAYS: f64 = 1.0;
/// Default period of the server's trim (`--cache-trim-every`), in minutes.
pub const DEFAULT_TRIM_EVERY_MIN: u64 = 60;

/// What a trim aims for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrimOptions {
    /// The budget: bytes of cache entries the directory may hold afterwards.
    pub target_bytes: u64,
    /// Entries used less than this many seconds ago are kept whatever the budget.
    pub keep_secs: u64,
    /// Report what would be removed, remove nothing.
    pub dry_run: bool,
}

impl TrimOptions {
    /// From the command-line units: GiB (decimals allowed) and days (decimals allowed).
    pub fn from_gib_days(max_gib: f64, keep_days: f64, dry_run: bool) -> Result<Self, String> {
        if !max_gib.is_finite() || max_gib < 0.0 {
            return Err(format!(
                "invalid cache budget {max_gib} GiB (a number >= 0)"
            ));
        }
        if !keep_days.is_finite() || keep_days < 0.0 {
            return Err(format!(
                "invalid keep window {keep_days} days (a number >= 0)"
            ));
        }
        Ok(Self {
            target_bytes: (max_gib * (1u64 << 30) as f64) as u64,
            keep_secs: (keep_days * 86_400.0) as u64,
            dry_run,
        })
    }
}

/// What a trim found and did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TrimReport {
    pub entries: u64,
    pub bytes_before: u64,
    /// Entries removed (in a dry run: that would be removed).
    pub evicted: u64,
    pub freed_bytes: u64,
    /// Bytes of cache entries left afterwards.
    pub bytes_after: u64,
    /// Entries kept although the cache was over budget when the walk reached them, because this
    /// process holds them or they were used within the keep window.
    pub protected_over_budget: u64,
    /// Files and directories that are not cache entries, left alone.
    pub unknown_files: u64,
    pub unknown_bytes: u64,
    /// Removals that failed (the entry stays and its bytes stay counted).
    pub errors: Vec<String>,
    pub dry_run: bool,
}

/// Bytes in the largest binary unit that keeps the number at 1 or more.
fn size(b: u64) -> String {
    const UNITS: [(&str, u32); 3] = [("GiB", 30), ("MiB", 20), ("KiB", 10)];
    for (unit, shift) in UNITS {
        if b >= 1u64 << shift {
            return format!("{:.2} {unit}", b as f64 / (1u64 << shift) as f64);
        }
    }
    format!("{b} B")
}

impl TrimReport {
    /// One line for a log or a terminal.
    pub fn summary(&self, dir: &Path, opts: &TrimOptions) -> String {
        let verb = if self.dry_run { "would free" } else { "freed" };
        let mut s = format!(
            "recipe cache {}: {} entries, {}; {verb} {} in {} entries, {} left \
             (budget {}, keep {:.2} days)",
            dir.display(),
            self.entries,
            size(self.bytes_before),
            size(self.freed_bytes),
            self.evicted,
            size(self.bytes_after),
            size(opts.target_bytes),
            opts.keep_secs as f64 / 86_400.0,
        );
        if self.protected_over_budget > 0 {
            s.push_str(&format!(
                "; {} recent or in-use entries kept over budget",
                self.protected_over_budget
            ));
        }
        if self.unknown_files > 0 {
            s.push_str(&format!(
                "; {} other files ({}) untouched",
                self.unknown_files,
                size(self.unknown_bytes)
            ));
        }
        if !self.errors.is_empty() {
            s.push_str(&format!("; {} removals failed", self.errors.len()));
        }
        s
    }
}

/// Which file of an entry a name is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryFile {
    Recipe,
    Sidecar,
}

/// `<name>-<16 lower-case hex>.recipe` or `.sha256` -> (`<name>-<hex>`, which file). Anything
/// else is not a cache entry.
pub fn parse_entry_file(file_name: &str) -> Option<(&str, EntryFile)> {
    let (stem, kind) = if let Some(s) = file_name.strip_suffix(RECIPE_EXT) {
        (s, EntryFile::Recipe)
    } else if let Some(s) = file_name.strip_suffix(SHA256_EXT) {
        (s, EntryFile::Sidecar)
    } else {
        return None;
    };
    let (name, hash) = stem.rsplit_once('-')?;
    let hex = |b: u8| b.is_ascii_digit() || (b'a'..=b'f').contains(&b);
    (!name.is_empty() && hash.len() == 16 && hash.bytes().all(hex)).then_some((stem, kind))
}

/// Size and modification time (seconds since the epoch) of one file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileStat {
    bytes: u64,
    mtime: u64,
}

#[derive(Debug, Clone, Default)]
struct Entry {
    recipe: Option<FileStat>,
    sidecar: Option<FileStat>,
}

impl Entry {
    fn last_used(&self) -> u64 {
        rt_last_used(
            self.recipe.map_or(0, |f| f.mtime),
            self.sidecar.map_or(0, |f| f.mtime),
        )
    }
}

fn secs_since_epoch(t: SystemTime) -> u64 {
    t.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// A regular file's size and time; `None` for anything else (gone, a link, a directory).
fn stat_file(path: &Path) -> Option<FileStat> {
    let m = fs::symlink_metadata(path).ok()?;
    m.file_type().is_file().then(|| FileStat {
        bytes: m.len(),
        mtime: m.modified().map_or(0, secs_since_epoch),
    })
}

struct Scan {
    entries: BTreeMap<String, Entry>,
    unknown_files: u64,
    unknown_bytes: u64,
}

fn scan(dir: &Path) -> io::Result<Scan> {
    let mut out = Scan {
        entries: BTreeMap::new(),
        unknown_files: 0,
        unknown_bytes: 0,
    };
    for item in fs::read_dir(dir)? {
        let item = item?;
        let name = item.file_name();
        let parsed = name.to_str().and_then(parse_entry_file);
        let stat = stat_file(&item.path());
        match (parsed, stat) {
            (Some((stem, kind)), Some(st)) => {
                let e = out.entries.entry(stem.to_owned()).or_default();
                match kind {
                    EntryFile::Recipe => e.recipe = Some(st),
                    EntryFile::Sidecar => e.sidecar = Some(st),
                }
            }
            (_, st) => {
                // gone since the listing, a link, a directory, or a name that is not an entry's
                if fs::symlink_metadata(item.path()).is_ok() {
                    out.unknown_files += 1;
                    out.unknown_bytes += st.map_or(0, |s| s.bytes);
                }
            }
        }
    }
    Ok(out)
}

/// Removes one file of an entry: `Ok(true)` when it is gone afterwards (removed now, or already
/// removed by someone else), `Err` when it is still there.
fn remove_file(path: &Path) -> io::Result<bool> {
    match fs::remove_file(path) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(true),
        Err(e) => Err(e),
    }
}

/// Trims `dir` to `opts`, honouring the entries this process holds.
pub fn trim(dir: &Path, opts: &TrimOptions) -> io::Result<TrimReport> {
    trim_at(dir, opts, secs_since_epoch(SystemTime::now()), true)
}

/// [`trim`] at the clock time `now`; `honour_in_use = false` trims as another process would
/// (one that holds none of this process's entries).
pub(crate) fn trim_at(
    dir: &Path,
    opts: &TrimOptions,
    now: u64,
    honour_in_use: bool,
) -> io::Result<TrimReport> {
    let scan = scan(dir)?;
    let mut rep = TrimReport {
        entries: scan.entries.len() as u64,
        unknown_files: scan.unknown_files,
        unknown_bytes: scan.unknown_bytes,
        dry_run: opts.dry_run,
        ..TrimReport::default()
    };
    let mut order: Vec<(&String, &Entry, u64)> = scan
        .entries
        .iter()
        .map(|(stem, e)| (stem, e, e.last_used()))
        .collect();
    // least recently used first; the name only makes the order of equal times repeatable
    order.sort_by(|a, b| {
        if rt_evict_before(a.2, b.2) {
            std::cmp::Ordering::Less
        } else if rt_evict_before(b.2, a.2) {
            std::cmp::Ordering::Greater
        } else {
            a.0.cmp(b.0)
        }
    });
    let size = |e: &Entry| e.recipe.map_or(0, |f| f.bytes) + e.sidecar.map_or(0, |f| f.bytes);
    let mut held: u64 = scan.entries.values().map(size).sum();
    rep.bytes_before = held;

    let registry = in_use_registry();
    for (stem, seen, used) in order {
        if !rt_should_evict(held, opts.target_bytes, false) {
            break; // within budget: nothing after this entry goes either
        }
        let recipe_path = dir.join(format!("{stem}{RECIPE_EXT}"));
        let sidecar_path = dir.join(format!("{stem}{SHA256_EXT}"));
        // Held across the check and the removal, so a load in this process that marks the entry
        // first keeps it, and one that marks it after finds it gone (a miss).
        let held_set = registry.lock().unwrap_or_else(|p| p.into_inner());
        let in_use = honour_in_use && held_set.contains(stem.as_str());
        // the times again: a hit since the scan refreshed the sidecar
        let fresh = Entry {
            recipe: stat_file(&recipe_path),
            sidecar: stat_file(&sidecar_path),
        };
        let used_now = if fresh.recipe.is_none() && fresh.sidecar.is_none() {
            used
        } else {
            fresh.last_used()
        };
        let protected = rt_protected(used_now, now, opts.keep_secs, in_use);
        if !rt_should_evict(held, opts.target_bytes, protected) {
            rep.protected_over_budget += 1;
            continue;
        }
        if opts.dry_run {
            let bytes = size(seen);
            held = rt_after(held, bytes, true);
            rep.evicted += 1;
            rep.freed_bytes += bytes;
            continue;
        }
        // sidecar first: from here on the entry is not a hit for anyone
        let mut freed = 0u64;
        let mut whole = true;
        for (path, stat) in [(&sidecar_path, seen.sidecar), (&recipe_path, seen.recipe)] {
            let Some(stat) = stat else { continue };
            match remove_file(path) {
                Ok(gone) => {
                    held = rt_after(held, stat.bytes, gone);
                    freed += stat.bytes;
                }
                Err(e) => {
                    held = rt_after(held, stat.bytes, false);
                    rep.errors.push(format!("remove {}: {e}", path.display()));
                    whole = false;
                    break; // never remove the recipe of a sidecar that is still there
                }
            }
        }
        drop(held_set);
        rep.freed_bytes += freed;
        if whole {
            rep.evicted += 1;
        }
    }
    rep.bytes_after = held;
    Ok(rep)
}

/// The server's trim settings (`--cache-max-gb` with `--serve`).
#[derive(Debug, Clone, PartialEq)]
pub struct ServerTrim {
    pub dir: PathBuf,
    pub opts: TrimOptions,
    pub every: Duration,
}

/// Trims in a thread of its own: once when `start` fires (or its sender is dropped), then every
/// `every`. Before each trim it refreshes the recency of the entries this process holds, so they
/// still count as recent after a restart. Request handling never waits for it.
pub fn spawn_server_trim(
    cfg: ServerTrim,
    start: std::sync::mpsc::Receiver<()>,
) -> io::Result<std::thread::JoinHandle<()>> {
    std::thread::Builder::new()
        .name("gaudium-cache-trim".into())
        .spawn(move || {
            let _ = start.recv();
            let mut first = true;
            loop {
                if crate::signals::term_requested() {
                    break;
                }
                refresh_recency(&cfg.dir);
                match trim(&cfg.dir, &cfg.opts) {
                    // the first trim always says what it found, later ones only when they act
                    Ok(rep) if first || rep.evicted > 0 || !rep.errors.is_empty() => {
                        eprintln!("[server] {}", rep.summary(&cfg.dir, &cfg.opts))
                    }
                    Ok(_) => {}
                    Err(e) => eprintln!(
                        "[server] recipe cache trim of {} failed: {e}",
                        cfg.dir.display()
                    ),
                }
                first = false;
                // sleep in short steps so a stop request is not held up
                let until = std::time::Instant::now() + cfg.every;
                while std::time::Instant::now() < until {
                    if crate::signals::term_requested() {
                        return;
                    }
                    std::thread::sleep(Duration::from_secs(1).min(cfg.every));
                }
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synapse::recipe_cache::{sha256_hex, RecipeCache};
    use std::fs::File;

    const H1: &str = "0123456789abcdef";
    const H2: &str = "fedcba9876543210";
    const H3: &str = "00112233445566aa";
    const DAY: u64 = 86_400;
    const NOW: u64 = 1_800_000_000;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("gaudium-trim-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn set_time(path: &Path, secs: u64) {
        File::open(path)
            .unwrap()
            .set_modified(UNIX_EPOCH + Duration::from_secs(secs))
            .unwrap();
    }

    /// An entry of `bytes` recipe bytes whose files were last touched at `used`.
    fn entry(dir: &Path, name: &str, hash: &str, bytes: usize, used: u64) {
        let r = dir.join(format!("{name}-{hash}.recipe"));
        let m = dir.join(format!("{name}-{hash}.sha256"));
        let body = vec![b'x'; bytes];
        fs::write(&r, &body).unwrap();
        fs::write(&m, format!("{}\n", sha256_hex(&body))).unwrap();
        set_time(&r, used);
        set_time(&m, used);
    }

    fn exists(dir: &Path, f: &str) -> bool {
        dir.join(f).exists()
    }

    fn opts(target: u64, keep: u64) -> TrimOptions {
        TrimOptions {
            target_bytes: target,
            keep_secs: keep,
            dry_run: false,
        }
    }

    #[test]
    fn names_that_are_entries() {
        assert_eq!(
            parse_entry_file("qwen-step-0123456789abcdef.recipe"),
            Some(("qwen-step-0123456789abcdef", EntryFile::Recipe))
        );
        assert_eq!(
            parse_entry_file("a-0123456789abcdef.sha256"),
            Some(("a-0123456789abcdef", EntryFile::Sidecar))
        );
        for other in [
            "dspine-tables-0123456789abcdef.bin",
            "a-0123456789abcdef.tmp",
            "a-0123456789ABCDEF.recipe",
            "a-0123456789abcde.recipe",
            "-0123456789abcdef.recipe",
            "0123456789abcdef.recipe",
            "a-0123456789abcdefg.recipe",
            "notes.recipe",
        ] {
            assert_eq!(parse_entry_file(other), None, "{other}");
        }
    }

    #[test]
    fn evicts_least_recently_used_first_until_within_budget() {
        let d = tmp("lru");
        let sha = 65; // sidecar: 64 hex digits and a newline
        entry(&d, "old", H1, 1000, NOW - 30 * DAY);
        entry(&d, "mid", H2, 1000, NOW - 20 * DAY);
        entry(&d, "new", H3, 1000, NOW - 10 * DAY);
        let rep = trim_at(&d, &opts(2 * (1000 + sha), DAY), NOW, false).unwrap();
        assert_eq!(rep.entries, 3);
        assert_eq!(rep.evicted, 1);
        assert_eq!(rep.freed_bytes, 1000 + sha);
        assert_eq!(rep.bytes_after, 2 * (1000 + sha));
        assert!(!exists(&d, &format!("old-{H1}.recipe")));
        assert!(!exists(&d, &format!("old-{H1}.sha256")));
        assert!(exists(&d, &format!("mid-{H2}.recipe")));
        assert!(exists(&d, &format!("new-{H3}.recipe")));
        // a hit refreshes the sidecar alone: that entry is now the most recent one
        set_time(&d.join(format!("mid-{H2}.sha256")), NOW - 5 * DAY);
        let rep = trim_at(&d, &opts(1000 + sha, DAY), NOW, false).unwrap();
        assert_eq!(rep.evicted, 1);
        assert!(exists(&d, &format!("mid-{H2}.recipe")));
        assert!(!exists(&d, &format!("new-{H3}.recipe")));
        // within budget: nothing goes
        let rep = trim_at(&d, &opts(1 << 30, 0), NOW, false).unwrap();
        assert_eq!((rep.evicted, rep.freed_bytes), (0, 0));
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn removes_pairs_and_orphans_and_leaves_other_files() {
        let d = tmp("pairs");
        entry(&d, "a", H1, 500, NOW - 9 * DAY);
        fs::write(d.join(format!("lone-{H2}.recipe")), b"r").unwrap();
        set_time(&d.join(format!("lone-{H2}.recipe")), NOW - 9 * DAY);
        fs::write(d.join(format!("side-{H3}.sha256")), b"s").unwrap();
        set_time(&d.join(format!("side-{H3}.sha256")), NOW - 9 * DAY);
        let others = [
            "dspine-tables-0123456789abcdef.bin",
            "a-0123456789abcdef.tmp",
            "README",
            "x-0123456789ABCDEF.recipe",
        ];
        for f in others {
            fs::write(d.join(f), vec![0u8; 4096]).unwrap();
            set_time(&d.join(f), NOW - 400 * DAY);
        }
        fs::create_dir(d.join(format!("dir-{H1}.recipe"))).unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(d.join("README"), d.join(format!("link-{H2}.recipe"))).unwrap();
        let rep = trim_at(&d, &opts(0, 0), NOW, false).unwrap();
        assert_eq!(rep.entries, 3);
        assert_eq!(rep.evicted, 3);
        assert_eq!(rep.bytes_after, 0);
        assert!(rep.errors.is_empty(), "{:?}", rep.errors);
        assert!(!exists(&d, &format!("a-{H1}.recipe")) && !exists(&d, &format!("a-{H1}.sha256")));
        assert!(!exists(&d, &format!("lone-{H2}.recipe")));
        assert!(!exists(&d, &format!("side-{H3}.sha256")));
        for f in others {
            assert!(exists(&d, f), "{f} must stay");
        }
        assert!(d.join(format!("dir-{H1}.recipe")).is_dir());
        #[cfg(unix)]
        {
            assert!(fs::symlink_metadata(d.join(format!("link-{H2}.recipe"))).is_ok());
            assert_eq!(rep.unknown_files, 6);
        }
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn keep_window_and_clock_skew_protect() {
        let d = tmp("keep");
        entry(&d, "recent", H1, 100, NOW - 3600);
        entry(&d, "ahead", H2, 100, NOW + DAY);
        entry(&d, "stale", H3, 100, NOW - 3 * DAY);
        let rep = trim_at(&d, &opts(0, DAY), NOW, false).unwrap();
        assert_eq!(rep.evicted, 1);
        assert_eq!(rep.protected_over_budget, 2);
        assert!(rep.bytes_after > 0, "over budget, and says so");
        assert!(exists(&d, &format!("recent-{H1}.recipe")));
        assert!(exists(&d, &format!("ahead-{H2}.recipe")));
        assert!(!exists(&d, &format!("stale-{H3}.recipe")));
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn dry_run_removes_nothing_and_reports_the_same() {
        let d = tmp("dry");
        entry(&d, "a", H1, 300, NOW - 9 * DAY);
        entry(&d, "b", H2, 300, NOW - 8 * DAY);
        entry(&d, "c", H3, 300, NOW - 1);
        let mut o = opts(0, DAY);
        o.dry_run = true;
        let dry = trim_at(&d, &o, NOW, false).unwrap();
        for s in ["a", "b", "c"] {
            let h = [H1, H2, H3][(s.as_bytes()[0] - b'a') as usize];
            assert!(
                exists(&d, &format!("{s}-{h}.recipe")) && exists(&d, &format!("{s}-{h}.sha256"))
            );
        }
        assert!(dry.summary(&d, &o).contains("would free"));
        o.dry_run = false;
        let real = trim_at(&d, &o, NOW, false).unwrap();
        assert_eq!(
            (
                dry.evicted,
                dry.freed_bytes,
                dry.bytes_after,
                dry.protected_over_budget
            ),
            (
                real.evicted,
                real.freed_bytes,
                real.bytes_after,
                real.protected_over_budget
            )
        );
        assert_eq!(real.evicted, 2);
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn entries_this_process_holds_are_kept() {
        let d = tmp("inuse");
        let name = "trim-inuse-test";
        entry(&d, name, H1, 100, NOW - 9 * DAY);
        let cache = RecipeCache::at_dir(&d);
        let fp = format!("{H1}{}", "0".repeat(48));
        assert!(cache.lookup_verified(name, &fp).is_some());
        // the hit refreshed the sidecar's time: protected by the window as well as held
        let rep = trim_at(&d, &opts(0, 0), NOW + 100 * DAY, true).unwrap();
        assert_eq!((rep.evicted, rep.protected_over_budget), (0, 1));
        // another process does not hold it
        let rep = trim_at(&d, &opts(0, 0), NOW + 100 * DAY, false).unwrap();
        assert_eq!(rep.evicted, 1);
        assert!(cache.lookup_verified(name, &fp).is_none());
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn hit_refreshes_recency_without_atime() {
        let d = tmp("touch");
        let name = "trim-touch-test";
        entry(&d, name, H2, 10, NOW - 50 * DAY);
        let cache = RecipeCache::at_dir(&d);
        let fp = format!("{H2}{}", "1".repeat(48));
        let before = SystemTime::now() - Duration::from_secs(5);
        assert!(cache.lookup_verified(name, &fp).is_some());
        let m = fs::metadata(d.join(format!("{name}-{H2}.sha256"))).unwrap();
        assert!(m.modified().unwrap() >= before);
        let _ = fs::remove_dir_all(&d);
    }

    /// A trim in another process running against loads: every load is a verified hit or a clean
    /// miss, and the bytes of a hit are whole. Runs until both outcomes were seen.
    #[test]
    fn load_racing_a_trim_misses_cleanly() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let d = tmp("race");
        let cache = RecipeCache::at_dir(&d);
        let names: Vec<String> = (0..8).map(|i| format!("race{i}")).collect();
        let fp = format!("{H3}{}", "2".repeat(48));
        let stop = AtomicBool::new(false);
        let (mut hits, mut misses, mut rounds) = (0u64, 0u64, 0usize);
        std::thread::scope(|s| {
            s.spawn(|| {
                while !stop.load(Ordering::Relaxed) {
                    trim_at(&d, &opts(0, 0), u64::MAX / 2, false).unwrap();
                    std::thread::sleep(Duration::from_millis(1));
                }
            });
            while rounds < 20_000 && (hits < 20 || misses < 20) {
                let n = &names[rounds % names.len()];
                let body = vec![(rounds % 251) as u8; 2048];
                rounds += 1;
                // written the way a compile writes: the recipe, then its sidecar
                let _ = fs::write(d.join(format!("{n}-{H3}.recipe")), &body);
                let _ = cache.record_sha256(n, &fp);
                match cache.lookup_verified(n, &fp) {
                    // the load reads the file again by its path: gone is a miss
                    Some(p) => match fs::read(&p) {
                        Ok(b) => {
                            assert_eq!(b, body, "a hit is the whole recipe");
                            hits += 1;
                        }
                        Err(e) => {
                            assert_eq!(e.kind(), io::ErrorKind::NotFound);
                            misses += 1;
                        }
                    },
                    None => misses += 1,
                }
            }
            stop.store(true, Ordering::Relaxed);
        });
        assert!(
            hits > 0 && misses > 0,
            "{hits} hits, {misses} misses in {rounds} rounds"
        );
        let _ = fs::remove_dir_all(&d);
    }

    /// Each point where a trim can remove an entry under a load: before the sidecar is read,
    /// between the sidecar and the recipe, and after the check (the load by path).
    #[test]
    fn a_removed_entry_is_a_miss_at_every_step_of_a_load() {
        let d = tmp("steps");
        let cache = RecipeCache::at_dir(&d);
        let fp = format!("{H1}{}", "3".repeat(48));
        let (r, m) = (
            d.join(format!("s-{H1}.recipe")),
            d.join(format!("s-{H1}.sha256")),
        );
        entry(&d, "s", H1, 64, NOW);
        fs::remove_file(&m).unwrap(); // sidecar gone first, as a trim removes it
        assert!(cache.lookup_verified("s", &fp).is_none());
        entry(&d, "s", H1, 64, NOW);
        fs::remove_file(&r).unwrap(); // the recipe gone after the sidecar was read
        assert!(cache.lookup_verified("s", &fp).is_none());
        entry(&d, "s", H1, 64, NOW);
        let p = cache.lookup_verified("s", &fp).expect("hit");
        trim_at(&d, &opts(0, 0), NOW + DAY, false).unwrap();
        assert_eq!(fs::read(p).unwrap_err().kind(), io::ErrorKind::NotFound);
        assert!(cache.lookup_verified("s", &fp).is_none());
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn budget_units() {
        let o = TrimOptions::from_gib_days(1.5, 0.5, false).unwrap();
        assert_eq!(o.target_bytes, 3 << 29);
        assert_eq!(o.keep_secs, 43_200);
        assert!(TrimOptions::from_gib_days(-1.0, 1.0, false).is_err());
        assert!(TrimOptions::from_gib_days(1.0, f64::NAN, false).is_err());
        // today's defaults (tuning, not laws)
        assert_eq!(DEFAULT_KEEP_DAYS, 1.0);
        assert_eq!(DEFAULT_TRIM_EVERY_MIN, 60);
        assert_eq!(size(512), "512 B");
        assert_eq!(size(3 << 29), "1.50 GiB");
    }
}
