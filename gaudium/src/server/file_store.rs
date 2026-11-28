//! The server's file store: the files of the Files API (`/v1/files`, `files.rs`), kept in one
//! directory with a total quota and a per-file cap (`--files-dir`, `--files-quota-gb`,
//! `--files-max-mb`).
//!
//! Each file is two entries of the directory: `<id>.data` (its bytes) and `<id>.json` (its
//! metadata: name, size, time, purpose, media type, owner). A file exists once its metadata is
//! written; a restart reads the metadata back. `next` holds the id counter, so the id of a
//! deleted file is never handed out again, also across restarts.
//!
//! Which upload is stored, what the quota holds and which id a file gets are decided by the
//! verified ledger (`verified/file_store.bend`):
//! - `fs_restore` counts the files found at start and `fs_start` refuses to start when they hold
//!   more than the configured quota (a quota lowered since);
//! - `fs_hold` reserves an upload's room before its bytes are read (its declared size, or the
//!   per-file cap, at most the free bytes), so uploads in progress never stage more than the
//!   quota; the reservation counts as used and its ticket is given back once (`fs_delete`, then
//!   `fs_kill`), at commit or when the upload is dropped;
//! - `fs_commit` decides a staged upload: refused past its reservation or the id limit, else as
//!   `fs_upload` decides it against the per-file cap and the free bytes, handing out the id;
//! - `fs_delete` gives a deleted file's bytes back, and the file leaves the index;
//! - `fs_visible` decides who sees a file;
//! - the file count bounds the files whatever their size (`--files-max-count`): `fs_count_take`
//!   gives an upload its slot before its room is reserved, or refuses it when the store is full,
//!   and `fs_take_count` is the count kept after it (one that counts the slot within the bound);
//!   the slot passes to the stored file and is given back once (`fs_count_give`) when the upload
//!   is refused or dropped, or the file is deleted; `fs_count_restore` counts every file found at
//!   start and `fs_count_fits` refuses a start with more files than the bound.
//!
//! This module streams the bytes, writes and reads the directory, and keeps the ledger in memory
//! behind a lock.
//!
//! Ownership: with an API key configured, a file belongs to the key that stored it ([`Owner`]);
//! another key (or none) finds no such file. Without a key every client shares the store. Files
//! of a key the server no longer has are not served and still count against the quota: the start
//! reports them, and `docs/files-api.md` says how to remove them.
//!
//! The internal API other parts of the server use (the batch APIs read their input here):
//! [`FileStore::create`] stores bytes, [`FileStore::open_content`] opens a file for reading.

use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::json::{parse_json, JsonValue};
use crate::verified::{
    fs_commit, fs_count_fits, fs_count_give, fs_count_live, fs_count_open, fs_count_restore,
    fs_count_take, fs_delete, fs_free, fs_hold, fs_hold_ledger, fs_hold_ticket, fs_kill, fs_live,
    fs_next, fs_open, fs_quota, fs_restore, fs_size, fs_start, fs_take_count, fs_upload_ledger,
    fs_used, fs_visible, FsCount, FsEntry, FsLedger, FsStart, FsTake, FsUpload,
};

use super::readout::jq;

/// The default total quota of the store (`--files-quota-gb`), in bytes.
pub const DEFAULT_QUOTA_BYTES: u64 = 16 << 30;
/// The default largest file (`--files-max-mb`), in bytes.
pub const DEFAULT_MAX_FILE_BYTES: u64 = 512 << 20;
/// The default most files the store holds (`--files-max-count`), uploads in progress included.
pub const DEFAULT_MAX_FILES: u64 = 10_000;
/// Ids are 12 hexadecimal digits of the counter: values from `ID_LIMIT - 1` on are not handed out
/// (`fs_commit`'s id limit) and not read back.
const ID_LIMIT: u64 = 1 << 48;
/// The longest file name kept, in characters.
const MAX_FILENAME_CHARS: usize = 255;
/// Bytes of a file's start kept to recognise its media type.
const SNIFF_BYTES: usize = 512;
/// The name of the id-counter file in the store's directory.
const NEXT_FILE: &str = "next";

/// The store's settings (`--files-dir` and its limits).
#[derive(Debug, Clone)]
pub struct FileStoreOptions {
    /// The store's directory; `None`: the Files API is off.
    pub dir: Option<PathBuf>,
    /// The total bytes the store may hold.
    pub quota_bytes: u64,
    /// The largest single file, in bytes.
    pub max_file_bytes: u64,
    /// The most files the store holds, whatever their size (uploads in progress included).
    pub max_files: u64,
}

impl Default for FileStoreOptions {
    fn default() -> Self {
        FileStoreOptions {
            dir: None,
            quota_bytes: DEFAULT_QUOTA_BYTES,
            max_file_bytes: DEFAULT_MAX_FILE_BYTES,
            max_files: DEFAULT_MAX_FILES,
        }
    }
}

/// Who a file belongs to: the API key that stored it (as a digest, never the key itself), or
/// everyone on a server without keys.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Owner(String);

impl Owner {
    /// Every client of a server without API keys.
    pub fn everyone() -> Owner {
        Owner(String::new())
    }

    /// The holder of `key`.
    pub fn of_key(key: &str) -> Owner {
        let digest = crate::synapse::recipe_cache::sha256_hex(key.as_bytes());
        Owner(format!("key:{}", &digest[..32]))
    }

    /// The owner of a server's authorised clients: the holder of its API key, or everyone.
    pub fn of_config(api_key: Option<&str>) -> Owner {
        api_key.map_or_else(Owner::everyone, Owner::of_key)
    }

    /// Whether this is the owner recorded for a file (a digest comparison; what it decides is
    /// `fs_visible`'s).
    fn is(&self, file_owner: &str) -> bool {
        self.0 == file_owner
    }
}

/// A stored file's metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileMeta {
    pub id: String,
    pub filename: String,
    pub bytes: u64,
    /// Unix seconds.
    pub created_at: u64,
    /// The OpenAI purpose (`user_data` for files stored through the Messages API shape).
    pub purpose: String,
    pub mime_type: String,
    /// Whether the Messages API shape lets clients download the content (files the server
    /// wrote itself; uploads are not, as that API has it).
    pub downloadable: bool,
    owner: String,
    num: u64,
}

/// Why a store operation failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileError {
    /// No such file for this owner.
    NotFound(String),
    /// The file is larger than the per-file cap (`size` is at least `cap + 1`: the upload stops
    /// reading there).
    TooLarge { cap: u64 },
    /// The file does not fit the free bytes of the quota. `partial`: the upload ran past the
    /// room it could reserve (`free`), so it was not read to its end and `size` is a lower bound.
    OverQuota {
        size: u64,
        used: u64,
        free: u64,
        quota: u64,
        partial: bool,
    },
    /// The store holds its most files (`live` of `max`, uploads in progress included).
    TooManyFiles { live: u64, max: u64 },
    /// Every id has been handed out.
    IdsExhausted,
    /// The upload could not be read from the client.
    BadUpload(String),
    /// The disk refused: what failed.
    Io(String),
}

impl FileError {
    /// The HTTP status of the failure.
    pub fn status(&self) -> u16 {
        match self {
            FileError::NotFound(_) => 404,
            FileError::TooLarge { .. } => 413,
            FileError::OverQuota { .. }
            | FileError::TooManyFiles { .. }
            | FileError::BadUpload(_) => 400,
            FileError::IdsExhausted | FileError::Io(_) => 500,
        }
    }

    pub fn message(&self) -> String {
        match self {
            FileError::NotFound(id) => format!("file '{id}' not found"),
            FileError::TooLarge { cap } => format!(
                "the file is larger than this server's limit of {cap} bytes per file \
                 (--files-max-mb)"
            ),
            FileError::OverQuota {
                size,
                used,
                free,
                quota,
                partial,
            } => {
                let file = if *partial {
                    format!("the file (more than {free} bytes)")
                } else {
                    format!("the file of {size} bytes")
                };
                format!(
                    "{file} does not fit the file store: {used} of {quota} bytes are in use or \
                     reserved by uploads in progress, and {free} are free (delete files or raise \
                     --files-quota-gb)"
                )
            }
            FileError::TooManyFiles { live, max } => format!(
                "the file store holds {live} files (uploads in progress included), its limit of \
                 {max} (delete files or raise --files-max-count)"
            ),
            FileError::IdsExhausted => "the file store has handed out every file id".into(),
            FileError::BadUpload(e) => format!("the upload could not be read: {e}"),
            FileError::Io(e) => format!("the file store could not write to its directory: {e}"),
        }
    }
}

fn io_err(what: &str, e: io::Error) -> FileError {
    FileError::Io(format!("{what}: {e}"))
}

/// An upload streamed to a staging file, not yet a file of the store, and the reservation that
/// holds its room in the ledger. Dropped uncommitted, it removes its staging file and gives the
/// reservation back.
pub struct Staged<'a> {
    store: &'a FileStore,
    /// The reservation (`fs_hold`): live until given back, at commit or at drop.
    ticket: FsEntry,
    /// Whether the upload still holds its slot of the file count (`fs_count_take`): it passes to
    /// the file at commit, else it is given back at drop.
    slot: bool,
    path: PathBuf,
    /// Bytes the upload has: those written to the staging file, plus one when the source had
    /// more than the reservation (then they were not read further).
    size: u64,
    /// Whether the source had more bytes than the reservation.
    partial: bool,
    head: Vec<u8>,
}

impl Staged<'_> {
    /// Bytes the upload has (one past the reservation when it ran past it).
    pub fn size(&self) -> u64 {
        self.size
    }
}

impl Drop for Staged<'_> {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
        // After a commit the ticket is no longer live and this changes nothing (fs_delete_killed);
        // a slot passed to the stored file is no longer held here (fs_give_once).
        let mut g = self.store.lock();
        g.ledger = fs_delete(g.ledger, self.ticket);
        g.count = fs_count_give(g.count, self.slot);
    }
}

struct Held {
    meta: FileMeta,
    /// The ledger's view of the file. A delete gives its bytes back (`fs_delete`) and its slot
    /// of the file count (`fs_count_give`), and removes it from the index, so a repeated delete
    /// finds nothing to free.
    entry: FsEntry,
}

impl Held {
    /// Whether `owner` asking for `id` (or for every file, `id == None`) sees this file.
    fn visible_to(&self, owner: &Owner, id: Option<&str>) -> bool {
        fs_visible(
            self.entry,
            owner.is(&self.meta.owner),
            id.is_none_or(|id| self.meta.id == id),
        )
    }
}

struct Inner {
    ledger: FsLedger,
    /// The file count: stored files, files found at start (served or not) and uploads in
    /// progress, against `--files-max-count`.
    count: FsCount,
    /// The files by their counter value (upload order).
    files: BTreeMap<u64, Held>,
}

pub struct FileStore {
    dir: PathBuf,
    quota: u64,
    cap: u64,
    max_files: u64,
    inner: Mutex<Inner>,
    staging: AtomicU64,
    /// Files found at start whose metadata could not be read: (count, bytes). Their bytes stay
    /// counted; reported at start.
    unreadable: (u64, u64),
}

/// The counter value of an id `file-<12 hex digits><12 hex digits>`.
fn id_num(id: &str) -> Option<u64> {
    let rest = id.strip_prefix("file-")?;
    if rest.len() != 24 || !rest.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    u64::from_str_radix(&rest[..12], 16).ok()
}

/// Random bits for the second half of an id (so ids are not guessed from the counter).
fn id_tail() -> u64 {
    let mut b = [0u8; 8];
    if fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut b))
        .is_err()
    {
        let t = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64);
        b = (t ^ u64::from(std::process::id()).rotate_left(32)).to_le_bytes();
    }
    u64::from_le_bytes(b) & ((1 << 48) - 1)
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// The name a file is kept under: the last path component, without control characters, at
/// most 255 characters; `unnamed` when nothing is left.
pub fn clean_filename(raw: &str) -> String {
    let last = raw.rsplit(['/', '\\']).next().unwrap_or("");
    let name: String = last
        .chars()
        .filter(|c| !c.is_control())
        .take(MAX_FILENAME_CHARS)
        .collect();
    let name = name.trim();
    if name.is_empty() || name == "." || name == ".." {
        "unnamed".into()
    } else {
        name.to_string()
    }
}

/// A file's media type: what its first bytes show (images, PDF), else the type the client gave,
/// else what its name's extension says, else `text/plain` for UTF-8 text, else
/// `application/octet-stream`.
pub fn detect_mime(hint: Option<&str>, filename: &str, head: &[u8]) -> String {
    let magic: &[(&[u8], &str)] = &[
        (b"\x89PNG\r\n\x1a\n", "image/png"),
        (b"\xff\xd8\xff", "image/jpeg"),
        (b"GIF87a", "image/gif"),
        (b"GIF89a", "image/gif"),
        (b"%PDF-", "application/pdf"),
    ];
    for (m, t) in magic {
        if head.starts_with(m) {
            return (*t).into();
        }
    }
    if head.len() >= 12 && &head[..4] == b"RIFF" && &head[8..12] == b"WEBP" {
        return "image/webp".into();
    }
    if let Some(h) = hint {
        let t = h
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        if is_media_type(&t) && t != "application/octet-stream" {
            return t;
        }
    }
    let ext = filename
        .rsplit_once('.')
        .map(|(_, e)| e.to_ascii_lowercase())
        .unwrap_or_default();
    let by_ext = match ext.as_str() {
        "txt" | "text" | "log" => Some("text/plain"),
        "md" | "markdown" => Some("text/markdown"),
        "csv" => Some("text/csv"),
        "html" | "htm" => Some("text/html"),
        "xml" => Some("application/xml"),
        "json" => Some("application/json"),
        "jsonl" | "ndjson" => Some("application/jsonl"),
        "pdf" => Some("application/pdf"),
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        _ => None,
    };
    if let Some(t) = by_ext {
        return t.into();
    }
    let text = match std::str::from_utf8(head) {
        Ok(_) => true,
        // a character cut at the end of the sniffed bytes
        Err(e) => e.error_len().is_none() && head.len() - e.valid_up_to() < 4,
    };
    if text && !head.contains(&0) {
        "text/plain".into()
    } else {
        "application/octet-stream".into()
    }
}

/// Whether `t` is a media type `type/subtype` of HTTP token characters (RFC 9110), at most 255
/// bytes: nothing that could end a header line or break the header it is sent in.
pub fn is_media_type(t: &str) -> bool {
    let token = |s: &str| {
        !s.is_empty()
            && s.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b))
    };
    t.len() <= 255 && t.split_once('/').is_some_and(|(a, b)| token(a) && token(b))
}

/// Writes `bytes` to `path` through a temporary file and a rename.
fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let tmp = path.with_extension("tmp");
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, path)
}

fn meta_json(m: &FileMeta) -> String {
    format!(
        "{{\"id\":{},\"filename\":{},\"bytes\":{},\"created_at\":{},\"purpose\":{},\"mime_type\":{},\"downloadable\":{},\"owner\":{}}}\n",
        jq(&m.id),
        jq(&m.filename),
        m.bytes,
        m.created_at,
        jq(&m.purpose),
        jq(&m.mime_type),
        m.downloadable,
        jq(&m.owner)
    )
}

fn meta_from_json(text: &str, num: u64) -> Option<FileMeta> {
    let v: JsonValue = parse_json(text).ok()?;
    let s = |k: &str| v.get(k).and_then(JsonValue::as_str).map(str::to_string);
    Some(FileMeta {
        id: s("id")?,
        filename: s("filename")?,
        bytes: v.get("bytes")?.as_u64()?,
        created_at: v.get("created_at")?.as_u64()?,
        purpose: s("purpose")?,
        mime_type: s("mime_type")?,
        downloadable: v.get("downloadable")?.as_bool()?,
        owner: s("owner")?,
        num,
    })
}

impl FileStore {
    /// Opens (or creates) the store in `dir` and counts the files it holds. Refused, with the
    /// numbers, when they hold more than `quota` bytes or are more than `max_files`.
    pub fn open(dir: &Path, quota: u64, cap: u64, max_files: u64) -> Result<FileStore, String> {
        fs::create_dir_all(dir)
            .map_err(|e| format!("--files-dir {}: cannot create it: {e}", dir.display()))?;
        let saved_next = match fs::read_to_string(dir.join(NEXT_FILE)) {
            Ok(t) => t
                .trim()
                .parse::<u64>()
                .ok()
                .filter(|&n| n < ID_LIMIT)
                .ok_or_else(|| {
                    format!(
                        "--files-dir {}: the id counter file '{NEXT_FILE}' is damaged",
                        dir.display()
                    )
                })?,
            Err(e) if e.kind() == io::ErrorKind::NotFound => 0,
            Err(e) => return Err(format!("--files-dir {}: {e}", dir.display())),
        };
        let mut ledger = fs_open(quota, saved_next);
        let mut count = fs_count_open(max_files);
        let mut files = BTreeMap::new();
        let mut unreadable = (0u64, 0u64);
        let entries =
            fs::read_dir(dir).map_err(|e| format!("--files-dir {}: {e}", dir.display()))?;
        let mut data_ids = Vec::new();
        for ent in entries.flatten() {
            let name = ent.file_name().to_string_lossy().into_owned();
            if name.starts_with(".staging-") || name.ends_with(".tmp") {
                // an upload or a write that a stop cut short
                let _ = fs::remove_file(ent.path());
                continue;
            }
            if let Some(id) = name.strip_suffix(".data") {
                if id_num(id).is_some() {
                    data_ids.push(id.to_string());
                }
                continue;
            }
            let Some(id) = name.strip_suffix(".json") else {
                continue;
            };
            let Some(num) = id_num(id).filter(|&n| n < ID_LIMIT - 1) else {
                continue;
            };
            let data = dir.join(format!("{id}.data"));
            let Ok(size) = fs::metadata(&data).map(|m| m.len()) else {
                // a file whose data was removed by hand: nothing to serve or count
                eprintln!("[files] {id}: metadata without data; removing the metadata");
                let _ = fs::remove_file(ent.path());
                continue;
            };
            // the bytes on disk, and the file, are counted whether or not it can be served
            ledger = fs_restore(ledger, num, size);
            count = fs_count_restore(count);
            let meta = fs::read_to_string(ent.path())
                .ok()
                .and_then(|t| meta_from_json(&t, num))
                .filter(|m| m.id == id);
            match meta {
                Some(mut m) => {
                    if m.bytes != size {
                        eprintln!(
                            "[files] {id}: metadata says {} bytes, the data holds {size}; serving the data",
                            m.bytes
                        );
                        m.bytes = size;
                    }
                    files.insert(
                        num,
                        Held {
                            meta: m,
                            entry: FsEntry::FsEntry(size, true),
                        },
                    );
                }
                None => {
                    eprintln!(
                        "[files] {id}: unreadable metadata; its {size} bytes stay counted, the file is not served"
                    );
                    unreadable.0 += 1;
                    unreadable.1 += size;
                }
            }
        }
        // data whose metadata was never written: an upload a stop cut short (its id is retired
        // by the saved counter)
        for id in data_ids {
            if !dir.join(format!("{id}.json")).exists() {
                let _ = fs::remove_file(dir.join(format!("{id}.data")));
            }
        }
        let ledger = match fs_start(ledger, quota) {
            FsStart::FsReady(l) => l,
            FsStart::FsOverfull => {
                return Err(format!(
                    "--files-dir {}: its files hold {} bytes, more than --files-quota-gb allows \
                     ({quota} bytes): raise the quota, or stop the server and remove files from the \
                     directory (docs/files-api.md)",
                    dir.display(),
                    fs_used(ledger)
                ))
            }
        };
        if !fs_count_fits(count) {
            return Err(format!(
                "--files-dir {}: it holds {} files, more than --files-max-count allows ({max_files}): \
                 raise the limit, or stop the server and remove files from the directory \
                 (docs/files-api.md)",
                dir.display(),
                fs_count_live(count)
            ));
        }
        Ok(FileStore {
            dir: dir.to_path_buf(),
            quota,
            cap,
            max_files,
            inner: Mutex::new(Inner {
                ledger,
                count,
                files,
            }),
            staging: AtomicU64::new(0),
            unreadable,
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The largest file, in bytes.
    pub fn cap(&self) -> u64 {
        self.cap
    }

    /// `(used, free, quota)` bytes; `used` counts the stored files and the reservations of
    /// uploads in progress.
    pub fn usage(&self) -> (u64, u64, u64) {
        let l = self.lock().ledger;
        (fs_used(l), fs_free(l), self.quota)
    }

    /// `(files, max)`: the files the store holds (uploads in progress and files not served
    /// included) and its most (`--files-max-count`).
    pub fn file_count(&self) -> (u64, u64) {
        (fs_count_live(self.lock().count), self.max_files)
    }

    /// Live files `owner` does not see (stored under another API key, or under none when the
    /// server now has one), and files whose metadata could not be read: `(count, bytes)`. Their
    /// bytes count against the quota; reported at start.
    pub fn unreachable(&self, owner: &Owner) -> (u64, u64) {
        let g = self.lock();
        let (mut n, mut bytes) = self.unreadable;
        for h in g.files.values() {
            if fs_live(h.entry) && !h.visible_to(owner, None) {
                n += 1;
                bytes += h.meta.bytes;
            }
        }
        (n, bytes)
    }

    /// Takes a slot of the file count (refused when the store holds its most files), reserves
    /// room for an upload of `want` bytes (`None`: the size is not known; the per-file cap is
    /// reserved), at most the free bytes, and streams `src` to a staging file: up to the
    /// reservation, plus one byte to tell that the source has more (that byte is not written).
    pub fn stage(&self, want: Option<u64>, src: &mut dyn Read) -> Result<Staged<'_>, FileError> {
        let ticket = {
            let mut g = self.lock();
            // the slot first: a given slot is counted within the bound (fs_take_count_counts); a
            // refusal leaves the count (fs_take_refused_keeps) and the ledger
            let t = fs_count_take(g.count);
            g.count = fs_take_count(g.count, t);
            if let FsTake::FsTooMany(live, max) = t {
                return Err(FileError::TooManyFiles { live, max });
            }
            let h = fs_hold(g.ledger, self.cap, want.unwrap_or(self.cap));
            g.ledger = fs_hold_ledger(h);
            fs_hold_ticket(h)
        };
        let k = self.staging.fetch_add(1, Ordering::Relaxed);
        let path = self
            .dir
            .join(format!(".staging-{}-{k}", std::process::id()));
        // from here a drop gives the reservation and the slot back
        let mut staged = Staged {
            store: self,
            ticket,
            slot: true,
            path,
            size: 0,
            partial: false,
            head: Vec::new(),
        };
        let room = fs_size(ticket);
        let mut f = fs::File::create(&staged.path).map_err(|e| io_err("staging file", e))?;
        let mut limited = src.take(room.saturating_add(1));
        let mut buf = vec![0u8; 64 * 1024];
        loop {
            let n = match limited.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(FileError::BadUpload(e.to_string())),
            };
            if staged.head.len() < SNIFF_BYTES {
                let take = (SNIFF_BYTES - staged.head.len()).min(n);
                staged.head.extend_from_slice(&buf[..take]);
            }
            // the bytes within the reservation go to disk; one past it only tells there are more
            let keep = n.min((room - staged.size.min(room)) as usize);
            f.write_all(&buf[..keep])
                .map_err(|e| io_err("staging file", e))?;
            staged.partial |= keep < n;
            staged.size += n as u64;
        }
        f.sync_all().map_err(|e| io_err("staging file", e))?;
        Ok(staged)
    }

    fn refusal(&self, r: FsUpload, l: FsLedger, staged: &Staged) -> FileError {
        match r {
            FsUpload::FsTooLarge => FileError::TooLarge { cap: self.cap },
            FsUpload::FsOverQuota(free) => FileError::OverQuota {
                size: staged.size,
                used: fs_used(l),
                free,
                quota: fs_quota(l),
                partial: staged.partial,
            },
            FsUpload::FsNoIds => FileError::IdsExhausted,
            FsUpload::FsStored(..) => FileError::Io("an upload stored without a commit".into()),
        }
    }

    /// Whether the ledger would store the staged upload now (the decision of
    /// [`commit`](Self::commit), taken early so a refused upload is not read further).
    pub fn precheck(&self, staged: &Staged) -> Result<(), FileError> {
        let l = fs_delete(self.lock().ledger, staged.ticket);
        match fs_commit(l, self.cap, ID_LIMIT - 1, staged.ticket, staged.size) {
            FsUpload::FsStored(..) => Ok(()),
            r => Err(self.refusal(r, l, staged)),
        }
    }

    /// Makes a staged upload a file of the store, if the ledger stores it.
    pub fn commit(
        &self,
        mut staged: Staged,
        owner: &Owner,
        filename: &str,
        mime_hint: Option<&str>,
        purpose: &str,
        downloadable: bool,
    ) -> Result<FileMeta, FileError> {
        let filename = clean_filename(filename);
        let mime_type = detect_mime(mime_hint, &filename, &staged.head);
        let size = staged.size;
        // `g` is a local, so it is released before `staged` (a parameter) drops and locks again.
        let mut g = self.lock();
        // the reservation is given back, once: the decision is taken against the ledger without it
        let released = fs_delete(g.ledger, staged.ticket);
        let decided = fs_commit(released, self.cap, ID_LIMIT - 1, staged.ticket, size);
        let after = fs_upload_ledger(released, decided);
        g.ledger = after;
        staged.ticket = fs_kill(staged.ticket);
        let num = match decided {
            FsUpload::FsStored(num, _) => num,
            // a refused upload gives its slot back when it drops
            r => return Err(self.refusal(r, released, &staged)),
        };
        // the slot passes to the file
        staged.slot = false;
        let entry = FsEntry::FsEntry(size, true);
        // From here the id is taken: the ledger keeps the counter even if the disk refuses
        // (the bytes are then given back through the verified delete).
        let id = format!("file-{num:012x}{:012x}", id_tail());
        let meta = FileMeta {
            id: id.clone(),
            filename,
            bytes: size,
            created_at: now_secs(),
            purpose: purpose.to_string(),
            mime_type,
            downloadable,
            owner: owner.0.clone(),
            num,
        };
        let written = (|| {
            write_atomic(
                &self.dir.join(NEXT_FILE),
                format!("{}\n", fs_next(after)).as_bytes(),
            )
            .map_err(|e| io_err("id counter", e))?;
            let data = self.dir.join(format!("{id}.data"));
            fs::rename(&staged.path, &data).map_err(|e| io_err("file data", e))?;
            if let Err(e) = write_atomic(
                &self.dir.join(format!("{id}.json")),
                meta_json(&meta).as_bytes(),
            ) {
                let _ = fs::remove_file(&data);
                return Err(io_err("file metadata", e));
            }
            Ok(())
        })();
        if let Err(e) = written {
            g.ledger = fs_delete(g.ledger, entry);
            g.count = fs_count_give(g.count, true);
            return Err(e);
        }
        g.files.insert(
            num,
            Held {
                meta: meta.clone(),
                entry,
            },
        );
        Ok(meta)
    }

    /// Stores `bytes` as a file (the server's own files, such as batch results).
    pub fn create(
        &self,
        owner: &Owner,
        filename: &str,
        mime_type: Option<&str>,
        purpose: &str,
        downloadable: bool,
        bytes: &[u8],
    ) -> Result<FileMeta, FileError> {
        let staged = self.stage(Some(bytes.len() as u64), &mut &bytes[..])?;
        self.commit(staged, owner, filename, mime_type, purpose, downloadable)
    }

    /// The metadata of `id`, if `owner` may use it.
    pub fn get(&self, owner: &Owner, id: &str) -> Result<FileMeta, FileError> {
        let g = self.lock();
        id_num(id)
            .and_then(|n| g.files.get(&n))
            .filter(|h| h.visible_to(owner, Some(id)))
            .map(|h| h.meta.clone())
            .ok_or_else(|| FileError::NotFound(id.to_string()))
    }

    /// The files `owner` may use, oldest first.
    pub fn list(&self, owner: &Owner) -> Vec<FileMeta> {
        self.lock()
            .files
            .values()
            .filter(|h| h.visible_to(owner, None))
            .map(|h| h.meta.clone())
            .collect()
    }

    /// Opens `id`'s content for reading.
    pub fn open_content(&self, owner: &Owner, id: &str) -> Result<(FileMeta, fs::File), FileError> {
        let meta = self.get(owner, id)?;
        let f = fs::File::open(self.dir.join(format!("{}.data", meta.id)))
            .map_err(|_| FileError::NotFound(id.to_string()))?;
        Ok((meta, f))
    }

    /// `id`'s content in memory: `Ok(Err(size))` when it is larger than `limit` bytes.
    pub fn read_limited(
        &self,
        owner: &Owner,
        id: &str,
        limit: u64,
    ) -> Result<Result<(FileMeta, Vec<u8>), u64>, FileError> {
        let (meta, f) = self.open_content(owner, id)?;
        if meta.bytes > limit {
            return Ok(Err(meta.bytes));
        }
        let mut out = Vec::with_capacity(meta.bytes as usize);
        f.take(limit.saturating_add(1))
            .read_to_end(&mut out)
            .map_err(|e| FileError::Io(format!("reading {id}: {e}")))?;
        if out.len() as u64 > limit {
            return Ok(Err(out.len() as u64));
        }
        Ok(Ok((meta, out)))
    }

    /// Deletes `id`: its metadata first (the file is gone from then on), then its data; the
    /// ledger gives its bytes back, and the file leaves the index (its id stays retired by the
    /// saved counter).
    pub fn delete(&self, owner: &Owner, id: &str) -> Result<FileMeta, FileError> {
        let mut g = self.lock();
        let num = id_num(id)
            .filter(|n| {
                g.files
                    .get(n)
                    .is_some_and(|h| h.visible_to(owner, Some(id)))
            })
            .ok_or_else(|| FileError::NotFound(id.to_string()))?;
        let meta_path = self.dir.join(format!("{id}.json"));
        match fs::remove_file(&meta_path) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(io_err("file metadata", e)),
        }
        let _ = fs::remove_file(self.dir.join(format!("{id}.data")));
        let held = g.files.remove(&num).expect("checked above");
        g.ledger = fs_delete(g.ledger, held.entry);
        g.count = fs_count_give(g.count, true);
        Ok(held.meta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("gaudium-files-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn files_are_stored_counted_and_kept_across_a_restart() {
        let d = tmp("store");
        let s = FileStore::open(&d, 100, 60, DEFAULT_MAX_FILES).unwrap();
        let me = Owner::of_key("k1");
        let a = s
            .create(&me, "dir/a.txt", None, "user_data", false, b"hello")
            .unwrap();
        assert!(a.id.starts_with("file-") && a.id.len() == 29, "{}", a.id);
        assert_eq!(
            (a.filename.as_str(), a.bytes, a.mime_type.as_str()),
            ("a.txt", 5, "text/plain")
        );
        let b = s
            .create(
                &me,
                "b.png",
                None,
                "vision",
                false,
                b"\x89PNG\r\n\x1a\nrest",
            )
            .unwrap();
        assert_eq!(b.mime_type, "image/png");
        assert_eq!(s.usage(), (5 + 12, 100 - 17, 100));
        // ownership: another key finds nothing
        let other = Owner::of_key("k2");
        assert!(matches!(s.get(&other, &a.id), Err(FileError::NotFound(_))));
        assert!(s.list(&other).is_empty());
        assert!(matches!(
            s.delete(&other, &a.id),
            Err(FileError::NotFound(_))
        ));
        // per-file cap and quota, refused with their numbers
        assert_eq!(
            s.create(&me, "big", None, "batch", false, &[b'x'; 61]),
            Err(FileError::TooLarge { cap: 60 })
        );
        s.create(&me, "c", None, "batch", false, &[b'x'; 60])
            .unwrap();
        // the upload can reserve only the 23 free bytes: it is refused past them, unread
        match s.create(&me, "d", None, "batch", false, &[b'x'; 30]) {
            Err(FileError::OverQuota {
                size,
                used,
                free,
                quota,
                partial,
            }) => {
                assert_eq!((size, used, free, quota, partial), (24, 77, 23, 100, true))
            }
            r => panic!("{r:?}"),
        }
        assert_eq!(s.usage(), (77, 23, 100));
        // a delete gives the bytes back once
        s.delete(&me, &a.id).unwrap();
        assert_eq!(s.usage().0, 72);
        assert!(matches!(s.delete(&me, &a.id), Err(FileError::NotFound(_))));
        assert_eq!(s.usage().0, 72);
        // a deleted file leaves the index
        assert_eq!(s.lock().files.len(), 2);
        drop(s);
        // a restart reads the files back, and never hands out a deleted id again
        let s = FileStore::open(&d, 100, 60, DEFAULT_MAX_FILES).unwrap();
        assert_eq!(s.usage().0, 72);
        assert_eq!(s.list(&me).len(), 2);
        assert_eq!(s.get(&me, &b.id).unwrap(), b);
        let e = s.create(&me, "e", None, "batch", false, b"e").unwrap();
        assert!(id_num(&e.id).unwrap() > id_num(&a.id).unwrap().max(id_num(&b.id).unwrap()) + 1);
        let mut content = String::new();
        s.open_content(&me, &b.id)
            .unwrap()
            .1
            .read_to_string(&mut content)
            .ok();
        let (_, bytes) = s.read_limited(&me, &b.id, 100).unwrap().unwrap();
        assert_eq!(bytes, b"\x89PNG\r\n\x1a\nrest");
        assert_eq!(s.read_limited(&me, &b.id, 4).unwrap(), Err(12));
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn a_counter_saved_past_the_last_file_is_kept() {
        let d = tmp("counter");
        let s = FileStore::open(&d, 100, 100, DEFAULT_MAX_FILES).unwrap();
        let me = Owner::everyone();
        let a = s.create(&me, "a", None, "batch", false, b"a").unwrap();
        let b = s.create(&me, "b", None, "batch", false, b"b").unwrap();
        s.delete(&me, &b.id).unwrap();
        drop(s);
        let s = FileStore::open(&d, 100, 100, DEFAULT_MAX_FILES).unwrap();
        let c = s.create(&me, "c", None, "batch", false, b"c").unwrap();
        assert!(id_num(&c.id).unwrap() > id_num(&b.id).unwrap());
        assert_eq!(
            s.list(&me).iter().map(|m| m.id.clone()).collect::<Vec<_>>(),
            [a.id, c.id]
        );
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn a_quota_lowered_below_the_files_kept_is_refused_at_start() {
        // the review's scenario: quota 100, files of 60 + 40 bytes, restart at 50. Before the
        // start check, deleting the 60-byte file freed 60 bytes against the quota of 50 and a
        // 60-byte upload was then stored (100 bytes held against 50).
        let d = tmp("lowered");
        let me = Owner::everyone();
        let s = FileStore::open(&d, 100, 100, DEFAULT_MAX_FILES).unwrap();
        let big = s
            .create(&me, "a", None, "batch", false, &[b'a'; 60])
            .unwrap();
        s.create(&me, "b", None, "batch", false, &[b'b'; 40])
            .unwrap();
        drop(s);
        let e = FileStore::open(&d, 50, 100, DEFAULT_MAX_FILES).err().expect("refused");
        assert!(
            e.contains("hold 100 bytes") && e.contains("(50 bytes)"),
            "{e}"
        );
        // at the old quota it starts; with the 60-byte file gone the lower quota is kept
        let s = FileStore::open(&d, 100, 100, DEFAULT_MAX_FILES).unwrap();
        s.delete(&me, &big.id).unwrap();
        drop(s);
        let s = FileStore::open(&d, 50, 100, DEFAULT_MAX_FILES).unwrap();
        assert_eq!(s.usage(), (40, 10, 50));
        assert!(matches!(
            s.create(&me, "c", None, "batch", false, &[b'c'; 60]),
            Err(FileError::OverQuota { .. })
        ));
        assert_eq!(s.usage(), (40, 10, 50));
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn uploads_in_progress_reserve_their_room_and_give_it_back_once() {
        let d = tmp("reserve");
        let me = Owner::everyone();
        let s = FileStore::open(&d, 100, 60, DEFAULT_MAX_FILES).unwrap();
        // a declared 60 bytes are reserved before any byte is read
        let a = s.stage(Some(60), &mut &[b'a'; 60][..]).unwrap();
        assert_eq!(s.usage(), (60, 40, 100));
        // an upload of unknown size reserves the cap, at most the free bytes: it stages 40
        // bytes and is refused past them
        let b = s.stage(None, &mut &[b'b'; 50][..]).unwrap();
        assert_eq!(s.usage(), (100, 0, 100));
        assert_eq!(b.size(), 41);
        assert!(matches!(
            s.precheck(&b),
            Err(FileError::OverQuota {
                free: 40,
                partial: true,
                ..
            })
        ));
        assert_eq!(fs::metadata(&b.path).unwrap().len(), 40);
        drop(b);
        assert_eq!(s.usage(), (60, 40, 100));
        // a commit gives the reservation back and takes the file's bytes, once
        let m = s.commit(a, &me, "a", None, "batch", false).unwrap();
        assert_eq!((m.bytes, s.usage()), (60, (60, 40, 100)));
        let c = s.stage(None, &mut &b"small"[..]).unwrap();
        assert_eq!(s.usage(), (100, 0, 100));
        s.commit(c, &me, "c", None, "batch", false).unwrap();
        assert_eq!(s.usage(), (65, 35, 100));
        // a refused commit gives the reservation back too
        let e = s.stage(Some(10), &mut &[b'e'; 20][..]).unwrap();
        assert!(s.commit(e, &me, "e", None, "batch", false).is_err());
        assert_eq!(s.usage(), (65, 35, 100));
        let leftovers: Vec<_> = fs::read_dir(&d)
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with(".staging-"))
            .collect();
        assert!(leftovers.is_empty());
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn files_of_no_bytes_are_bounded_by_the_file_count() {
        // the review's scenario: with no free bytes, uploads of no bytes were each stored, so
        // the index and the directory grew without bound; the file count refuses them
        let d = tmp("count");
        let me = Owner::everyone();
        let s = FileStore::open(&d, 10, 10, 50).unwrap();
        s.create(&me, "full", None, "batch", false, &[b'f'; 10])
            .unwrap();
        assert_eq!(s.usage(), (10, 0, 10));
        let mut ids = Vec::new();
        for i in 0..49 {
            ids.push(s.create(&me, &format!("e{i}"), None, "batch", false, b"").unwrap().id);
        }
        assert_eq!(s.file_count(), (50, 50));
        for _ in 0..200 {
            assert_eq!(
                s.create(&me, "more", None, "batch", false, b""),
                Err(FileError::TooManyFiles { live: 50, max: 50 })
            );
        }
        // a refusal leaves the count, the bytes and the directory as they were
        assert_eq!((s.file_count(), s.usage()), ((50, 50), (10, 0, 10)));
        assert_eq!(fs::read_dir(&d).unwrap().count(), 2 * 50 + 1);
        // an upload in progress holds a slot; dropped, it gives it back
        s.delete(&me, &ids.pop().unwrap()).unwrap();
        assert_eq!(s.file_count(), (49, 50));
        let staged = s.stage(Some(0), &mut &b""[..]).unwrap();
        assert_eq!(s.file_count(), (50, 50));
        assert!(matches!(
            s.stage(Some(0), &mut &b""[..]),
            Err(FileError::TooManyFiles { .. })
        ));
        drop(staged);
        assert_eq!(s.file_count(), (49, 50));
        // a refused commit gives its slot back (once, at its drop)
        let big = s.stage(Some(5), &mut &[b'b'; 5][..]).unwrap();
        assert!(matches!(
            s.commit(big, &me, "b", None, "batch", false),
            Err(FileError::OverQuota { .. })
        ));
        assert_eq!(s.file_count(), (49, 50));
        s.create(&me, "last", None, "batch", false, b"").unwrap();
        assert_eq!(s.file_count(), (50, 50));
        drop(s);
        // a restart counts every file found, and refuses more than the bound
        let s = FileStore::open(&d, 10, 10, 50).unwrap();
        assert_eq!(s.file_count(), (50, 50));
        assert!(matches!(
            s.create(&me, "more", None, "batch", false, b""),
            Err(FileError::TooManyFiles { .. })
        ));
        drop(s);
        let e = FileStore::open(&d, 10, 10, 49).err().expect("refused");
        assert!(e.contains("holds 50 files") && e.contains("(49)"), "{e}");
        // files whose metadata cannot be read are counted too
        fs::write(d.join(format!("{}.json", ids[0])), b"not json").unwrap();
        let s = FileStore::open(&d, 10, 10, 50).unwrap();
        assert_eq!(s.file_count(), (50, 50));
        assert_eq!(s.unreachable(&me), (1, 0));
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn the_default_file_count() {
        assert_eq!(DEFAULT_MAX_FILES, 10_000);
        assert_eq!(FileStoreOptions::default().max_files, DEFAULT_MAX_FILES);
    }

    #[test]
    fn no_upload_is_stored_once_the_ids_run_out() {
        let d = tmp("ids");
        fs::create_dir_all(&d).unwrap();
        fs::write(d.join(NEXT_FILE), format!("{}\n", ID_LIMIT - 2)).unwrap();
        let me = Owner::everyone();
        let s = FileStore::open(&d, 100, 100, DEFAULT_MAX_FILES).unwrap();
        let last = s.create(&me, "a", None, "batch", false, b"a").unwrap();
        assert_eq!(id_num(&last.id), Some(ID_LIMIT - 2));
        assert_eq!(
            s.create(&me, "b", None, "batch", false, b"b"),
            Err(FileError::IdsExhausted)
        );
        assert_eq!(s.usage(), (1, 99, 100));
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn files_of_another_key_are_hidden_and_reported() {
        let d = tmp("foreign");
        let s = FileStore::open(&d, 100, 100, DEFAULT_MAX_FILES).unwrap();
        let old = Owner::of_key("old");
        let f = s.create(&old, "a", None, "batch", false, b"abc").unwrap();
        drop(s);
        // the key changed: the file is not served to the new one, and counted at start
        let s = FileStore::open(&d, 100, 100, DEFAULT_MAX_FILES).unwrap();
        let new = Owner::of_config(Some("new"));
        assert!(s.list(&new).is_empty());
        assert!(matches!(s.get(&new, &f.id), Err(FileError::NotFound(_))));
        assert!(matches!(s.delete(&new, &f.id), Err(FileError::NotFound(_))));
        assert_eq!(s.unreachable(&new), (1, 3));
        assert_eq!(s.unreachable(&old), (0, 0));
        assert_eq!(s.get(&old, &f.id).unwrap().id, f.id);
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn names_and_media_types() {
        assert_eq!(
            detect_mime(Some("text/html\n\n<script>alert(1)</script>"), "a", b"x"),
            "text/plain"
        );
        assert_eq!(
            detect_mime(Some("text/html\r\nX-Other: 1"), "a.png", b"x"),
            "image/png"
        );
        assert!(is_media_type("application/vnd.api+json"));
        assert!(!is_media_type("text/html x"));
        assert!(!is_media_type("text/"));
        assert!(!is_media_type("text/a/b"));
        assert_eq!(clean_filename("C:\\x\\y.pdf"), "y.pdf");
        assert_eq!(clean_filename("../"), "unnamed");
        assert_eq!(clean_filename("a\u{0}b"), "ab");
        assert_eq!(
            detect_mime(Some("image/x-portable-pixmap"), "a", b"P6\n"),
            "image/x-portable-pixmap"
        );
        assert_eq!(
            detect_mime(Some("text/plain"), "a", b"%PDF-1.7"),
            "application/pdf"
        );
        assert_eq!(detect_mime(None, "in.jsonl", b"{}"), "application/jsonl");
        assert_eq!(
            detect_mime(Some("application/octet-stream"), "x", b"plain words"),
            "text/plain"
        );
        assert_eq!(
            detect_mime(None, "x", b"\x00\x01\x02"),
            "application/octet-stream"
        );
        assert_eq!(detect_mime(None, "x", "caf\u{e9}".as_bytes()), "text/plain");
        assert_eq!(id_num("file-00000000002a0123456789ab"), Some(42));
        assert_eq!(id_num("file-xyz"), None);
        assert_eq!(id_num("file_00000000002a0123456789ab"), None);
    }
}
