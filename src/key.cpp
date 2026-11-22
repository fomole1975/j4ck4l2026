#include "key.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <new>
#include <thread>

namespace shopping {
namespace {
uint64_t hash_bytes(std::string_view s) {
    constexpr uint64_t mul = UINT64_C(0x9e3779b97f4a7c15);
    uint64_t h = s.size() * UINT64_C(0xd6e8feb86659fd93);
    const char* p = s.data();
    size_t n = s.size();
    for (; n >= 8; p += 8, n -= 8) {
        uint64_t word;
        std::memcpy(&word, p, 8);
        h = (h ^ word) * mul;
        h ^= h >> 29;
    }
    if (n) {
        uint64_t word = 0;
        std::memcpy(&word, p, n);
        h = (h ^ word) * mul;
        h ^= h >> 29;
    }
    h *= UINT64_C(0xbf58476d1ce4e5b9);
    return h ^ (h >> 31);
}

// Each shard is an insert-only open-addressing table over entries carved from
// a private arena. Lookups read the published table without the lock; misses
// re-check and insert under it. Replaced tables and entries are never freed:
// the process exits right after writing its output.
struct Table {
    size_t mask;
    std::atomic<const Key::Interned*> slots[1];
};
Table* make_table(size_t size) {
    auto* t = static_cast<Table*>(std::calloc(1, sizeof(Table) + (size - 1) * sizeof(Table::slots[0])));
    if (!t) throw std::bad_alloc();
    t->mask = size - 1;
    return t;
}
bool same(const Key::Interned* e, uint64_t h, std::string_view s) {
    return e->hash == h && e->size == s.size() && std::memcmp(e->data(), s.data(), s.size()) == 0;
}
const Key::Interned* find(const Table* t, uint64_t h, std::string_view s, size_t& at) {
    size_t i = h & t->mask;
    for (;; i = (i + 1) & t->mask) {
        const auto* e = t->slots[i].load(std::memory_order_acquire);
        if (!e) {
            at = i;
            return nullptr;
        }
        if (same(e, h, s)) return e;
    }
}
struct Shard {
    std::atomic_flag busy;
    std::atomic<Table*> table{nullptr};
    size_t used = 0;
    char* arena = nullptr;
    size_t left = 0;

    void lock() {
        while (busy.test_and_set(std::memory_order_acquire))
            while (busy.test(std::memory_order_relaxed))
                std::this_thread::yield();
    }
    void unlock() { busy.clear(std::memory_order_release); }
    const Key::Interned* lookup(uint64_t h, std::string_view s) const {
        size_t at;
        const Table* t = table.load(std::memory_order_acquire);
        return t ? find(t, h, s, at) : nullptr;
    }
    void grow() {
        Table* old = table.load(std::memory_order_relaxed);
        Table* fresh = make_table(old ? 2 * (old->mask + 1) : 256);
        for (size_t i = 0; old && i <= old->mask; ++i) {
            if (const auto* e = old->slots[i].load(std::memory_order_relaxed)) {
                size_t j = e->hash & fresh->mask;
                while (fresh->slots[j].load(std::memory_order_relaxed)) j = (j + 1) & fresh->mask;
                fresh->slots[j].store(e, std::memory_order_relaxed);
            }
        }
        table.store(fresh, std::memory_order_release);
    }
    const Key::Interned* insert(uint64_t h, std::string_view s) {
        if (!table.load(std::memory_order_relaxed)) grow();
        Table* t = table.load(std::memory_order_relaxed);
        size_t i;
        if (const auto* e = find(t, h, s, i)) return e;
        const size_t need = (sizeof(Key::Interned) + s.size() + 7) & ~size_t{7};
        if (left < need) {
            left = std::max<size_t>(need, 64 * 1024);
            arena = static_cast<char*>(std::malloc(left));
            if (!arena) throw std::bad_alloc();
        }
        auto* entry = new (arena) Key::Interned{h, static_cast<uint32_t>(s.size())};
        std::memcpy(const_cast<char*>(entry->data()), s.data(), s.size());
        arena += need;
        left -= need;
        t->slots[i].store(entry, std::memory_order_release);
        if (++used * 2 > t->mask + 1) grow();
        return entry;
    }
};

constexpr unsigned shard_bits = 10;
Shard* shards() {
    static Shard* const all = new Shard[size_t{1} << shard_bits];
    return all;
}

// Repeated IDs (a user's many rows) are answered without taking a lock.
struct Cached {
    uint64_t hash;
    const Key::Interned* entry;
};
constexpr size_t cache_size = 1024;
}

uint64_t Key::intern(std::string_view bytes) {
    const uint64_t h = hash_bytes(bytes);
    thread_local Cached* cache = static_cast<Cached*>(std::calloc(cache_size, sizeof(Cached)));
    Cached* slot = cache ? &cache[h & (cache_size - 1)] : nullptr;
    if (slot && slot->hash == h && slot->entry && slot->entry->size == bytes.size() &&
        std::memcmp(slot->entry->data(), bytes.data(), bytes.size()) == 0)
        return tag | reinterpret_cast<uintptr_t>(slot->entry);
    Shard& shard = shards()[h >> (64 - shard_bits)];
    if (const Interned* known = shard.lookup(h, bytes)) {
        if (slot) *slot = {h, known};
        return tag | reinterpret_cast<uintptr_t>(known);
    }
    shard.lock();
    const Interned* entry;
    try {
        entry = shard.insert(h, bytes);
    } catch (...) {
        shard.unlock();
        throw;
    }
    shard.unlock();
    if (slot) *slot = {h, entry};
    return tag | reinterpret_cast<uintptr_t>(entry);
}
}
