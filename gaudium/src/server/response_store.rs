//! The in-memory store of past responses (`POST /v1/responses` with `store`): what
//! `GET /v1/responses/{id}`, `GET /v1/responses/{id}/input_items` and `previous_response_id` read
//! back, and what `DELETE /v1/responses/{id}` removes.
//!
//! The store is bounded by a number of entries and a number of bytes (`--response-store`,
//! `--response-store-mb`). Whether a new entry is taken, waits for the oldest entry to leave or
//! is refused, and the account of what the store holds, are the verified decision
//! (`verified/response_store.bend`: `rs_put`, `rs_add`, `rs_remove`); this file keeps the entries
//! themselves, oldest first, and measures each one's size. An entry holds its texts as
//! serialised JSON (parsed again when it is read), so the size it is counted with is what it
//! holds: its texts, its id as the store's index keeps it, and a fixed overhead per entry.
//!
//! The store also holds the request context of each response still being generated (`begin` /
//! `end`). Those are counted in an account of their own, bounded by the store's byte bound plus
//! the request body limit (the verified `rs_admit`), since none of them can be evicted. Whether a
//! response that ends is offered to the store at all (its request asked for it and no `DELETE`
//! reached it while it ran) is the verified `rs_keep`.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crate::json::{parse_json, JsonValue};
use crate::verified::{rs_add, rs_admit, rs_empty, rs_keep, rs_put, rs_remove, RsAcct, RsPut};

use super::responses::Head;

/// Default bound on the entries the store holds.
pub const DEFAULT_ENTRIES: u64 = 1024;
/// Default bound on the bytes the store holds (MiB).
pub const DEFAULT_MIB: u64 = 256;

/// Bytes counted for one entry beyond its texts: its structs, the `Arc` it is shared through,
/// and its slots in the id map and the oldest-first list (under 200 bytes on a 64-bit build; a
/// test checks the bound).
pub const ENTRY_OVERHEAD: u64 = 256;

/// The bounds of the store; `entries` 0 turns it off.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoreLimits {
    pub entries: u64,
    pub bytes: u64,
}

impl Default for StoreLimits {
    fn default() -> Self {
        Self {
            entries: DEFAULT_ENTRIES,
            bytes: DEFAULT_MIB << 20,
        }
    }
}

impl StoreLimits {
    /// The bound on the request contexts of the responses being generated at once: one request
    /// may carry a body up to the request limit and continue a stored conversation up to the
    /// store's byte bound, so one such request always fits when it is alone.
    pub fn live_bytes(&self) -> u64 {
        self.bytes
            .saturating_add(super::MAX_REQUEST_BODY_BYTES as u64)
    }
}

/// One stored response.
#[derive(Debug)]
pub struct StoredResponse {
    pub id: String,
    /// The response object as `GET /v1/responses/{id}` returns it (its `output` holds the
    /// output items).
    pub response: String,
    /// The conversation's input items as one JSON array, oldest first: those of the responses it
    /// continues, then its own (each with an `id`). Instructions are not items and are not kept.
    pub input: String,
    /// Where its own input items begin in `input`.
    pub own_from: usize,
}

/// The items of a JSON array text (empty for anything else; the store only holds arrays it wrote).
fn array_items(text: &str) -> Vec<JsonValue> {
    match parse_json(text) {
        Ok(JsonValue::Array(items)) => items,
        _ => Vec::new(),
    }
}

impl StoredResponse {
    /// The bytes the store counts for this entry: its texts, its id three times (here, as the
    /// map key and in the oldest-first list) and [`ENTRY_OVERHEAD`].
    pub fn size(&self) -> u64 {
        let texts = 3 * self.id.capacity() + self.response.capacity() + self.input.capacity();
        (texts as u64).saturating_add(ENTRY_OVERHEAD)
    }

    /// The conversation's input items (parsed from the stored text).
    pub fn input_items(&self) -> Vec<JsonValue> {
        array_items(&self.input)
    }

    /// The response's output items (the `output` of the stored response object).
    pub fn output_items(&self) -> Vec<JsonValue> {
        match parse_json(&self.response) {
            Ok(r) => match r.get("output") {
                Some(JsonValue::Array(items)) => items.clone(),
                _ => Vec::new(),
            },
            Err(_) => Vec::new(),
        }
    }
}

struct Entry {
    resp: Arc<StoredResponse>,
    size: u64,
}

/// Where a live response's reply lands, for a cancel waiting on it.
#[derive(Default)]
pub struct Watch {
    /// The reply, once the generation has it; `ended`: the request returned (with or without).
    state: Mutex<(Option<Arc<String>>, bool)>,
    changed: Condvar,
}

impl Watch {
    /// Waits up to `timeout` for the reply. `None`: the request ended without one (it failed),
    /// or the time ran out.
    pub fn wait(&self, timeout: Duration) -> Option<Arc<String>> {
        let deadline = Instant::now() + timeout;
        let mut g = self.state.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(r) = &g.0 {
                return Some(Arc::clone(r));
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if g.1 || left.is_zero() {
                return None;
            }
            g = self
                .changed
                .wait_timeout(g, left)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }

    fn set(&self, reply: Option<Arc<String>>, ended: bool) {
        let mut g = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if reply.is_some() {
            g.0 = reply;
        }
        g.1 |= ended;
        self.changed.notify_all();
    }
}

/// A response still being generated.
struct Live {
    head: Arc<Head>,
    /// The bytes its context is counted with in the in-flight account.
    size: u64,
    /// A `DELETE` reached it: it is not kept when it ends.
    deleted: bool,
    /// Its reply, for a cancel waiting on it.
    watch: Arc<Watch>,
}

struct Inner {
    acct: RsAcct,
    /// Ids, oldest first.
    order: VecDeque<String>,
    entries: HashMap<String, Entry>,
    /// Responses still being generated, and the account of their contexts.
    live: HashMap<String, Live>,
    live_acct: RsAcct,
}

/// What `DELETE /v1/responses/{id}` found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Deleted {
    /// A stored response, now removed.
    Stored,
    /// A response still being generated: it will not be kept (the caller cancels it).
    Running,
    /// No response of that id.
    None,
}

pub struct ResponseStore {
    limits: StoreLimits,
    inner: Mutex<Inner>,
}

impl ResponseStore {
    pub fn new(limits: StoreLimits) -> Self {
        Self {
            limits,
            inner: Mutex::new(Inner {
                acct: rs_empty(),
                order: VecDeque::new(),
                entries: HashMap::new(),
                live: HashMap::new(),
                live_acct: rs_empty(),
            }),
        }
    }

    pub fn limits(&self) -> StoreLimits {
        self.limits
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn deleted(g: &Inner, id: &str) -> bool {
        g.live.get(id).is_some_and(|l| l.deleted)
    }

    /// Offers the response `id` that has just ended to the store. `want`: its request asked for
    /// it to be kept; `make` builds the entry (only when it is offered). `Ok(true)`: kept;
    /// `Ok(false)`: not offered (the request said `store: false`, or a `DELETE` reached it while
    /// it ran); `Err`: the store refused it (it is off, or the entry alone is larger than its
    /// byte bound).
    pub fn put(
        &self,
        id: &str,
        want: bool,
        make: impl FnOnce() -> StoredResponse,
    ) -> Result<bool, String> {
        if !rs_keep(want, Self::deleted(&self.lock(), id)) {
            return Ok(false);
        }
        let mut r = make();
        r.response.shrink_to_fit();
        r.input.shrink_to_fit();
        r.id.shrink_to_fit();
        let size = r.size();
        let mut g = self.lock();
        // a DELETE may have reached it while the entry was being built
        if !rs_keep(want, Self::deleted(&g, id)) {
            return Ok(false);
        }
        // an id is kept once (a response id is never reused; this keeps the account exact)
        if let Some(old) = g.entries.remove(&r.id) {
            g.acct = rs_remove(g.acct, old.size);
            g.order.retain(|i| *i != r.id);
        }
        loop {
            match rs_put(g.acct, self.limits.entries, self.limits.bytes, size) {
                RsPut::RsPutStore => {
                    g.acct = rs_add(g.acct, size);
                    g.order.push_back(r.id.clone());
                    g.entries.insert(
                        r.id.clone(),
                        Entry {
                            resp: Arc::new(r),
                            size,
                        },
                    );
                    return Ok(true);
                }
                RsPut::RsPutEvict => {
                    let Some(oldest) = g.order.pop_front() else {
                        // the account holds an entry the list does not: never evict blindly
                        return Err("the response store's account and its entries disagree".into());
                    };
                    if let Some(e) = g.entries.remove(&oldest) {
                        g.acct = rs_remove(g.acct, e.size);
                    }
                }
                RsPut::RsPutRefuse => {
                    return Err(if self.limits.entries == 0 {
                        "the response store is off (--response-store 0)".to_string()
                    } else {
                        format!(
                            "the response ({size} bytes) is larger than the response store ({} bytes, --response-store-mb)",
                            self.limits.bytes
                        )
                    });
                }
            }
        }
    }

    pub fn get(&self, id: &str) -> Option<Arc<StoredResponse>> {
        self.lock().entries.get(id).map(|e| Arc::clone(&e.resp))
    }

    /// Removes a stored response, or marks one still being generated so that it is not kept.
    pub fn delete(&self, id: &str) -> Deleted {
        let mut g = self.lock();
        if let Some(e) = g.entries.remove(id) {
            g.acct = rs_remove(g.acct, e.size);
            g.order.retain(|i| i != id);
            return Deleted::Stored;
        }
        match g.live.get_mut(id) {
            Some(l) if !l.deleted => {
                l.deleted = true;
                Deleted::Running
            }
            _ => Deleted::None,
        }
    }

    /// Entries and bytes held.
    #[cfg(test)]
    pub fn held(&self) -> (u64, u64) {
        let RsAcct::RsAcct(n, b) = self.lock().acct;
        (n, b)
    }

    /// Contexts and bytes held for the responses being generated.
    #[cfg(test)]
    pub fn live_held(&self) -> (u64, u64) {
        let RsAcct::RsAcct(n, b) = self.lock().live_acct;
        (n, b)
    }

    /// A response begins: its request context, for its reply and for the store. `Err`: the
    /// contexts of the responses being generated already hold what the server allows.
    pub fn begin(&self, id: &str, head: Head) -> Result<(), String> {
        let size = head.size();
        let bound = self.limits.live_bytes();
        let mut g = self.lock();
        if let Some(old) = g.live.remove(id) {
            g.live_acct = rs_remove(g.live_acct, old.size);
        }
        if !rs_admit(g.live_acct, bound, size) {
            let RsAcct::RsAcct(n, held) = g.live_acct;
            return Err(format!(
                "the server is generating {n} responses whose request contexts hold {held} bytes; \
                 this one ({size} bytes) would pass the bound of {bound} bytes \
                 (--response-store-mb plus the request body limit); retry when one has finished"
            ));
        }
        g.live_acct = rs_add(g.live_acct, size);
        g.live.insert(
            id.to_string(),
            Live {
                head: Arc::new(head),
                size,
                deleted: false,
                watch: Arc::default(),
            },
        );
        Ok(())
    }

    /// The request context of a response still being generated (also once it was deleted).
    pub fn head(&self, id: &str) -> Option<Arc<Head>> {
        self.lock().live.get(id).map(|l| Arc::clone(&l.head))
    }

    /// A watch on the reply of a response being generated that was not deleted (taken before
    /// the cancel is signalled, so a reply that comes at once is not missed).
    pub fn watch(&self, id: &str) -> Option<Arc<Watch>> {
        self.lock()
            .live
            .get(id)
            .filter(|l| !l.deleted)
            .map(|l| Arc::clone(&l.watch))
    }

    /// The response object a live response's request answers with.
    pub fn settle(&self, id: &str, reply: &str) {
        let w = self.lock().live.get(id).map(|l| Arc::clone(&l.watch));
        if let Some(w) = w {
            w.set(Some(Arc::new(reply.to_string())), false);
        }
    }

    /// The response's request has returned (whatever the way out).
    pub fn end(&self, id: &str) {
        let live = {
            let mut g = self.lock();
            let live = g.live.remove(id);
            if let Some(l) = &live {
                g.live_acct = rs_remove(g.live_acct, l.size);
            }
            live
        };
        if let Some(l) = live {
            l.watch.set(None, true);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: &str, pad: usize) -> StoredResponse {
        StoredResponse {
            id: id.into(),
            response: "x".repeat(pad),
            input: String::new(),
            own_from: 0,
        }
    }

    /// The size of an entry with an id of one byte and `pad` bytes of text.
    fn sz(pad: u64) -> u64 {
        3 + pad + ENTRY_OVERHEAD
    }

    fn put(s: &ResponseStore, r: StoredResponse) -> Result<bool, String> {
        let id = r.id.clone();
        s.put(&id, true, move || r)
    }

    #[test]
    fn the_entry_overhead_covers_what_an_entry_holds_besides_its_texts() {
        use std::mem::size_of;
        // the structs, the Arc counts, the map slot (key, value, control byte) and the list slot
        let fixed = size_of::<StoredResponse>()
            + 2 * size_of::<usize>()
            + size_of::<String>()
            + size_of::<Entry>()
            + 1
            + size_of::<String>();
        assert!((fixed as u64) <= ENTRY_OVERHEAD, "{fixed}");
    }

    #[test]
    fn the_oldest_entry_leaves_when_the_count_bound_is_reached() {
        let s = ResponseStore::new(StoreLimits {
            entries: 2,
            bytes: 1 << 20,
        });
        for id in ["a", "b", "c"] {
            assert!(put(&s, entry(id, 10)).unwrap());
        }
        assert!(s.get("a").is_none());
        assert!(s.get("b").is_some() && s.get("c").is_some());
        assert_eq!(s.held(), (2, 2 * sz(10)));
    }

    #[test]
    fn a_store_of_one_entry_replaces_it_with_the_newest() {
        // evict-oldest is today's policy for a full store (the laws leave refusing open): a
        // store bounded to one entry forgets the old one as soon as a new one comes
        let s = ResponseStore::new(StoreLimits {
            entries: 1,
            bytes: 1 << 20,
        });
        assert!(put(&s, entry("a", 10)).unwrap());
        assert!(put(&s, entry("b", 10)).unwrap());
        assert!(s.get("a").is_none() && s.get("b").is_some());
        assert_eq!(s.held(), (1, sz(10)));
        // and the same with the byte bound: room for one entry only
        let s = ResponseStore::new(StoreLimits {
            entries: 10,
            bytes: sz(10) + 5,
        });
        assert!(put(&s, entry("a", 10)).unwrap());
        assert!(put(&s, entry("b", 10)).unwrap());
        assert!(s.get("a").is_none() && s.get("b").is_some());
        assert_eq!(s.held(), (1, sz(10)));
    }

    #[test]
    fn the_byte_bound_evicts_as_many_as_needed_and_never_more() {
        let s = ResponseStore::new(StoreLimits {
            entries: 100,
            bytes: 5 * sz(20),
        });
        for id in ["a", "b", "c", "d"] {
            assert!(put(&s, entry(id, 20)).unwrap());
        }
        // four held; an entry the size of two and a half needs two of them gone
        let big = 2 * sz(20) + sz(20) / 2 - sz(0);
        assert!(put(&s, entry("e", big as usize)).unwrap());
        assert!(s.get("a").is_none() && s.get("b").is_none());
        assert!(s.get("c").is_some() && s.get("d").is_some() && s.get("e").is_some());
        assert_eq!(s.held(), (3, 2 * sz(20) + sz(big)));
    }

    #[test]
    fn an_entry_that_fits_beside_the_others_evicts_nothing() {
        let s = ResponseStore::new(StoreLimits {
            entries: 3,
            bytes: 3 * sz(10),
        });
        for id in ["a", "b", "c"] {
            assert!(put(&s, entry(id, 10)).unwrap());
        }
        assert_eq!(s.held(), (3, 3 * sz(10)));
    }

    #[test]
    fn an_entry_larger_than_the_store_is_refused_and_evicts_nothing() {
        let s = ResponseStore::new(StoreLimits {
            entries: 10,
            bytes: 1000,
        });
        assert!(put(&s, entry("a", 9)).unwrap());
        assert!(put(&s, entry("big", 2000)).is_err());
        assert!(s.get("a").is_some());
        assert_eq!(s.held(), (1, sz(9)));
    }

    #[test]
    fn an_off_store_keeps_nothing_and_delete_conserves_the_account() {
        let off = ResponseStore::new(StoreLimits {
            entries: 0,
            bytes: 100,
        });
        assert!(put(&off, entry("a", 1)).unwrap_err().contains("off"));
        let s = ResponseStore::new(StoreLimits::default());
        put(&s, entry("a", 5)).unwrap();
        put(&s, entry("b", 7)).unwrap();
        assert_eq!(s.delete("a"), Deleted::Stored);
        assert_eq!(s.delete("a"), Deleted::None);
        assert_eq!(s.held(), (1, sz(7)));
        // the same id again replaces the entry
        put(&s, entry("b", 1)).unwrap();
        assert_eq!(s.held(), (1, sz(1)));
    }

    #[test]
    fn store_false_is_never_offered_and_never_built() {
        let s = ResponseStore::new(StoreLimits::default());
        assert!(!s.put("a", false, || panic!("built")).unwrap());
        assert!(s.get("a").is_none());
        assert_eq!(s.held(), (0, 0));
    }

    #[test]
    fn a_response_deleted_while_it_runs_is_not_kept() {
        let s = ResponseStore::new(StoreLimits::default());
        s.begin("a", Head::default()).unwrap();
        assert!(s.watch("a").is_some());
        assert_eq!(s.delete("a"), Deleted::Running);
        // gone for every reader at once, and a second DELETE finds nothing
        assert!(s.watch("a").is_none());
        assert_eq!(s.delete("a"), Deleted::None);
        // the generation still has its context for its own reply
        assert!(s.head("a").is_some());
        assert!(!put(&s, entry("a", 5)).unwrap());
        s.end("a");
        assert!(s.get("a").is_none());
        assert_eq!(s.held(), (0, 0));
        assert_eq!(s.live_held(), (0, 0));
    }

    #[test]
    fn the_contexts_of_running_responses_are_bounded_and_released() {
        let s = ResponseStore::new(StoreLimits {
            entries: 10,
            bytes: 0,
        });
        // contexts of 60 MiB each (reserved, never touched)
        let big = || Head {
            input: String::with_capacity(60 << 20),
            ..Head::default()
        };
        let bound = s.limits().live_bytes();
        let one = big().size();
        let fit = bound / one;
        assert!(fit >= 2);
        for i in 0..fit {
            s.begin(&format!("r{i}"), big()).unwrap();
        }
        assert_eq!(s.live_held(), (fit, fit * one));
        let e = s.begin("over", big()).unwrap_err();
        assert!(e.contains("retry"), "{e}");
        s.end("r0");
        s.begin("over", big()).unwrap();
        for i in 1..fit {
            s.end(&format!("r{i}"));
        }
        s.end("over");
        assert_eq!(s.live_held(), (0, 0));
    }

    #[test]
    fn a_cancel_waits_for_the_reply_of_the_running_response() {
        let s = ResponseStore::new(StoreLimits::default());
        let long = Duration::from_secs(10);
        s.begin("a", Head::default()).unwrap();
        let w = s.watch("a").unwrap();
        let t = std::thread::spawn({
            let w = Arc::clone(&w);
            move || w.wait(long)
        });
        std::thread::sleep(Duration::from_millis(20));
        s.settle("a", "{\"status\":\"cancelled\"}");
        let got = t.join().unwrap();
        assert_eq!(
            got.as_deref().map(String::as_str),
            Some("{\"status\":\"cancelled\"}")
        );
        // a reply that came and a request that ended before the wait are not missed
        s.end("a");
        assert!(w.wait(long).is_some());
        // a response that ends without a reply wakes its waiter with none
        s.begin("b", Head::default()).unwrap();
        let w = s.watch("b").unwrap();
        let t = std::thread::spawn({
            let w = Arc::clone(&w);
            move || w.wait(long)
        });
        std::thread::sleep(Duration::from_millis(20));
        s.end("b");
        assert!(t.join().unwrap().is_none());
        assert!(s.watch("none").is_none());
    }

    #[test]
    fn a_stored_entry_reads_its_items_back() {
        let r = StoredResponse {
            id: "resp_1".into(),
            response: r#"{"id":"resp_1","output":[{"type":"message","id":"m"}]}"#.into(),
            input: r#"[{"type":"message","id":"i0"},{"type":"message","id":"i1"}]"#.into(),
            own_from: 1,
        };
        assert_eq!(r.input_items().len(), 2);
        assert_eq!(r.output_items()[0].get("id").unwrap().as_str(), Some("m"));
    }
}
