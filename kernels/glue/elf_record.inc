// elf_record.inc: which kernel ELFs this glue library handed to the graph compiler, for the engine's recipe cache
// (src/engine/recipe_cache.hpp, docs/notes/recipe-cache.md). Included by gdn_glue.cpp and ga_glue.cpp after their ELF cache
// (elf_mu, elf_locked); each library has its own copy, so each answers for its own kernels.
//
// A compiled recipe embeds the kernels the glue gave the compiler, so the cache keys every recipe on exactly those ELFs. Only
// the glue knows which they are: it picks the variable from the node's guid, shapes, dtypes and parameters. So it keeps a
// record:
//   GaudiumElfTraceBegin()        opens a log on the calling thread (the engine calls it right before synGraphCompile);
//   GaudiumElfTraceEnd(buf, cap)  closes it: the variables put_elf was asked for on this thread since Begin, one per line,
//                                 each once, in first-use order. Returns the bytes the list needs (with its final 0); a
//                                 smaller `cap` gets nothing written (the engine then keys that recipe on every kernel ELF);
//   GaudiumElfTraceOrphans()      the number of put_elf calls made on a thread with no open log since the library was
//                                 loaded. The engine reads it before and after a compile: a change means the compiler
//                                 called the glue on a thread of its own, the log is incomplete, and the engine then keys
//                                 that recipe on every kernel ELF instead;
//   GaudiumElfGet(env, &p, &n)    the bytes put_elf hands out for a variable (read on first use, exactly as put_elf does,
//                                 and kept for the life of the process). The engine hashes THESE bytes, so a file that
//                                 changes on disk while the process runs cannot give a key that names other bytes than the
//                                 compiler got. Returns 1, or 0 when the variable is unset or the file cannot be read.
// The record costs one thread-local push per kernel instantiation and nothing when no log is open.
#include <atomic>
struct ElfLog { bool on = false; std::vector<std::string> names; };
static ElfLog& elf_log() { static thread_local ElfLog l; return l; }
static std::atomic<uint64_t>& elf_orphans() { static std::atomic<uint64_t> n{0}; return n; }
static void elf_note(const char* env) {
    ElfLog& l = elf_log();
    if (!l.on) { elf_orphans().fetch_add(1, std::memory_order_relaxed); return; }
    for (const std::string& s : l.names) if (s == env) return;
    l.names.emplace_back(env);
}
extern "C" {
void GaudiumElfTraceBegin() { ElfLog& l = elf_log(); l.on = true; l.names.clear(); }
uint32_t GaudiumElfTraceEnd(char* buf, uint32_t cap) {
    ElfLog& l = elf_log(); l.on = false;
    std::string s; for (const std::string& n : l.names) { s += n; s += '\n'; }
    const uint32_t need = (uint32_t)s.size() + 1;
    if (buf && cap >= need) memcpy(buf, s.c_str(), need);
    return need;
}
uint64_t GaudiumElfTraceOrphans() { return elf_orphans().load(std::memory_order_relaxed); }
int GaudiumElfGet(const char* env, const unsigned char** data, uint64_t* size) {
    std::lock_guard<std::mutex> lk(elf_mu());
    std::vector<unsigned char>& e = elf_locked(env);
    if (e.empty()) return 0;
    if (data) *data = e.data(); if (size) *size = e.size();
    return 1;
}
}
