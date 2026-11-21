#include "engine.hpp"
#include "catalog.hpp"
#include "input.hpp"
#include "sessions.hpp"
#include "time_sort32.hpp"

#include <span>
#include <algorithm>
#include <array>
#include <type_traits>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>

namespace shopping {
namespace {
constexpr int64_t day_ms = 86'400'000, week_ms = 7 * day_ms, session_gap = 1'800'000;
constexpr uint32_t absent = std::numeric_limits<uint32_t>::max();

int32_t epoch_day(int64_t ts) {
    return static_cast<int32_t>(ts / day_ms - (ts % day_ms < 0));
}

template <class General, class Basic> struct EventSink {
    General general;
    Basic basic;
    void operator()(const EventInput &row) { general(row); }
};

template <class General, class Indexed> struct ClickSink {
    General general;
    Indexed indexed;
    void operator()(const ClickInput &row) { general(row); }
};
template <class Fn> void parallel_jobs(unsigned n, size_t count, Fn&& work) {
    if (!count)
        return;
    const unsigned threads = static_cast<unsigned>(std::min<size_t>(n, count));
    if (threads <= 1) {
        for (size_t i = 0; i < count; ++i)
            work(0, i);
        return;
    }
    std::atomic<size_t> next{0};
    std::mutex failure_lock;
    std::exception_ptr failure;
    auto loop = [&](unsigned worker) {
        try {
            for (;;) {
                size_t i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= count)
                    break;
                work(worker, i);
            }
        } catch (...) {
            next.store(count, std::memory_order_relaxed);
            std::lock_guard guard(failure_lock);
            if (!failure)
                failure = std::current_exception();
        }
    };
    std::vector<std::jthread> pool;
    for (unsigned i = 1; i < threads; ++i)
        pool.emplace_back(loop, i);
    loop(0);
    pool.clear();
    if (failure)
        std::rethrow_exception(failure);
}

// Open addressing is sufficient for our immutable dictionaries and append-only
// component/group indices. All keys entering the table are nonempty.
template <class Value> class KeyTable {
    struct Slot {
        Key key;
        Value value{};
    };
    std::vector<Slot> slots_;
    size_t used_ = 0;
    size_t find_slot(Key key) const {
        size_t mask = slots_.size() - 1;
        size_t i = key.hash() & mask;
        while (slots_[i].key.bits() && !(slots_[i].key == key))
            i = (i + 1) & mask;
        return i;
    }
    void grow(size_t n) {
        auto old = std::move(slots_);
        slots_ = std::vector<Slot>(n);
        for (const auto& s : old)
            if (s.key.bits())
                slots_[find_slot(s.key)] = s;
    }

  public:
    explicit KeyTable(size_t estimate = 0) : slots_(std::bit_ceil(std::max<size_t>(8, estimate + estimate / 2))) {}
    template <class F> Value add(Key k, F&& make) {
        size_t i = find_slot(k);
        if (slots_[i].key.bits())
            return slots_[i].value;
        if ((used_ + 1) * 4 > slots_.size() * 3) {
            grow(slots_.size() * 2);
            i = find_slot(k);
        }
        ++used_;
        slots_[i] = {k, make()};
        return slots_[i].value;
    }
    // Reserve before taking references to the two endpoints of a link.
    void ensure_pair_capacity() {
        if ((used_ + 2) * 4 > slots_.size() * 3)
            grow(slots_.size() * 2);
    }
    void prefetch(Key key) const {
        __builtin_prefetch(&slots_[key.hash() & (slots_.size() - 1)], 1, 1);
    }
    Value& entry(Key key, Value initial) {
        auto& slot = slots_[find_slot(key)];
        if (!slot.key.bits()) {
            slot = {key, initial};
            ++used_;
        }
        return slot.value;
    }
    Value get(Key k, Value missing) const {
        const auto& slot = slots_[find_slot(k)];
        return slot.key.bits() ? slot.value : missing;
    }
    template <class F> void transform(F&& f) {
        for (auto& s : slots_)
            if (s.key.bits())
                f(s.value);
    }
    void set(Key k, Value v) {
        add(k, [&] { return v; });
    }
};

struct Dimensions {
    CatalogIndex<false, KeyTable<uint32_t>> product_category{50'000};
    CatalogIndex<true, KeyTable<uint32_t>> campaign_index{2'000};
    std::vector<std::string_view> categories{"unknown"}, campaigns{"organic"}, channels{"organic"};
    Dimensions(const InputFiles& input) {
        std::unordered_map<std::string_view, uint32_t> categories_by_name{{"unknown", 0}};
        for (auto part : input.products)
            for_pairs(part, [&](std::string_view id, std::string_view name) {
                auto [it, fresh] = categories_by_name.try_emplace(name, categories.size());
                if (fresh)
                    categories.push_back(name);
                product_category.set(id, it->second);
            });
        for (auto part : input.campaigns)
            for_pairs(part, [&](std::string_view id, std::string_view channel) {
                campaign_index.set(id, static_cast<uint32_t>(campaigns.size()));
                campaigns.push_back(id);
                channels.push_back(channel);
            });
    }
};

// Number only linked components. After union-find, a raw ID indexes the dense
// component table, whose entry holds its bytewise smallest ID.
class Aliases {
    KeyTable<uint32_t> groups_;
    std::vector<Key> names_;

  public:
    explicit Aliases(size_t estimate) : groups_(estimate) {}
    void ensure_pair_capacity() { groups_.ensure_pair_capacity(); }
    void prefetch(Key key) const { groups_.prefetch(key); }
    uint32_t &component(Key key) { return groups_.entry(key, absent); }
    uint32_t add_person(Key name) {
        const auto i = static_cast<uint32_t>(names_.size());
        names_.push_back(name);
        return i;
    }
    template <class F> void freeze(F &&resolved) {
        groups_.transform([&](uint32_t &i) { i = resolved(i); });
    }
    uint32_t person(Key raw) const { return groups_.get(raw, absent); }
    Key name(uint32_t i) const { return names_[i]; }
    size_t size() const { return names_.size(); }
};
Aliases identities(const InputFiles &input) {
    struct Component {
        uint32_t parent;
        union { uint32_t size, person; };
        Key least;
    };
    size_t bytes = 0;
    for (auto part : input.links)
        bytes += part.size();
    Aliases ids(bytes / 16);
    std::vector<Component> components;
    components.reserve(bytes / 32);
    auto root = [&](uint32_t i) {
        while (components[i].parent != i) {
            components[i].parent = components[components[i].parent].parent;
            i = components[i].parent;
        }
        return i;
    };
    auto join = [&](Key left, Key right) {
        ids.ensure_pair_capacity();
        auto& a = ids.component(left);
        auto& b = ids.component(right);
        // Allocate a forest node only when a link starts a new component.
        // A previously unseen endpoint can inherit its neighbor's component.
        if (a == absent && b == absent) {
            auto n = static_cast<uint32_t>(components.size());
            components.push_back({n, 1, std::min(left, right)});
            a = b = n;
        } else if (a == absent) {
            a = b = root(b);
            components[a].least = std::min(components[a].least, left);
        } else if (b == absent) {
            a = b = root(a);
            components[b].least = std::min(components[b].least, right);
        } else if (a != b) {
            uint32_t x = root(a), y = root(b);
            if (x != y) {
                if (components[x].size < components[y].size)
                    std::swap(x, y);
                components[y].parent = x;
                components[x].size += components[y].size;
                components[x].least = std::min(components[x].least, components[y].least);
            }
            a = b = x;
        }
    };
    std::array<std::pair<Key, Key>, 32> pending;
    size_t used = 0;
    auto flush = [&] {
        for (size_t i = 0; i < used; ++i)
            join(pending[i].first, pending[i].second);
        used = 0;
    };
    for (auto part : input.links)
        for_pairs(part, [&](std::string_view a, std::string_view b) {
            const Key left(a), right(b);
            ids.prefetch(left);
            ids.prefetch(right);
            pending[used++] = {left, right};
            if (used == pending.size())
                flush();
        });
    flush();
    // A root's union size is dead once links have been processed. Reuse the
    // word to propagate a dense ordinal without allocating a second map.
    for (uint32_t i = 0; i < components.size(); ++i) {
        if (components[i].parent == i)
            components[i].person = ids.add_person(components[i].least);
    }
    for (uint32_t i = 0; i < components.size(); ++i)
        components[i].person = components[root(i)].person;
    ids.freeze([&](uint32_t i) { return components[i].person; });
    return ids;
}

struct Purchase {
    uint32_t price, category;
};
// The common 14-day window fits in 32-bit milliseconds; contest prices fit
// in 24 bits. Rare wider offsets or category indices go through an exact
// side table. Only purchases and these exceptions carry an amount word.
struct PackedData {
    uint32_t offset, amount;
};
constexpr uint32_t external = 0x00ffffff;
struct Candidate {
    Key id;
    int64_t ingest;
    uint64_t location;
};
// A shared prefix and up to eight nonzero bytes have an injective, zero-padded
// representation. Separate partitions keep these values disjoint from arbitrary Keys.
constexpr unsigned literal_shard_bits = 9;
constexpr size_t literal_shards = size_t{1} << literal_shard_bits;
uint64_t event_literal(std::string_view id) {
    if (id.size() < 2 || id.size() > 9 || id.front() != 'e')
        return 0;
    const size_t size = id.size() - 1;
    if (size == 1)
        return static_cast<unsigned char>(id[1]);
    // This is the first field of a structurally valid seven-field row. With
    // two or more suffix bytes, the six separators guarantee an eight-byte load.
    uint64_t bits;
    std::memcpy(&bits, id.data() + 1, sizeof(bits));
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    bits = __builtin_bswap64(bits);
#endif
    const uint64_t mask = UINT64_MAX >> ((8 - size) * 8);
    const uint64_t word = bits | ~mask;
    constexpr uint64_t ones = UINT64_C(0x0101010101010101);
    if ((word - ones) & ~word & (ones << 7))
        return 0;
    return bits & mask;
}
uint64_t literal_hash(uint64_t id) {
#if defined(__SSE4_2__)
    return _mm_crc32_u64(0, id);
#else
    uint64_t hash = (id ^ (id >> 32)) * UINT64_C(0xd6e8feb86659fd93);
    return hash ^ (hash >> 32);
#endif
}
struct LiteralCandidate {
    uint64_t id;
    uint32_t seconds, location;
};
static_assert(sizeof(LiteralCandidate) == 16);
struct WideLiteralCandidate {
    uint64_t id;
    int64_t ingest;
    uint64_t location;
};
struct Event {
    // Keeping the key word-aligned makes the record exactly 12 bytes, including
    // interned names. After lookup, words[0] is the exact raw group index.
    std::array<uint32_t, 2> words;
    uint32_t offset;
    static Event make(Key user, uint32_t offset) {
        return {std::bit_cast<decltype(words)>(user), offset};
    }
    Key user() const { return std::bit_cast<Key>(words); }
};
struct RawPurchase {
    uint32_t row, amount;
};
static_assert(sizeof(Candidate) == 24 && sizeof(Event) == 12);
// Once the window is known, every retained click either has a 32-bit
// lookback-relative offset or carries an index into a worker's wide times.
struct Click {
    union {
        Key user;
        uint64_t group;
    };
    Key id;
    uint32_t time, campaign;
};
static_assert(sizeof(Click) == 24);
struct TimedEvent {
    int64_t ts;
    Purchase purchase;
};
struct TimedClick {
    int64_t ts;
    Key id;
    uint32_t campaign;
};
// Common windows fit both event offsets and the seven-day click lookback into 32 bits.
struct ShortClick {
    uint32_t offset, campaign;
    Key id;
};
inline int64_t event_time(const TimedEvent& event, int64_t) { return event.ts; }
inline int64_t event_time(uint32_t offset, int64_t base) { return base + offset; }
inline int64_t event_time(uint64_t event, int64_t base) { return base + static_cast<int64_t>(event >> 32); }
inline Purchase event_purchase(const TimedEvent& event) { return event.purchase; }
inline Purchase event_purchase(uint64_t event) {
    auto amount = static_cast<uint32_t>(event);
    return {amount & 0x00ffffff, amount >> 24};
}
inline int64_t click_time(const TimedClick& click, int64_t) { return click.ts; }
inline int64_t click_time(const ShortClick& click, int64_t base) { return base - week_ms + click.offset; }

template <class T> using Partitions = std::vector<std::vector<T>>;
struct Local {
    Partitions<Candidate> candidates;
    Partitions<LiteralCandidate> literals;
    Partitions<WideLiteralCandidate> wide_literals;
    Partitions<Event> events;
    Partitions<RawPurchase> purchases;
    Partitions<Click> clicks;
    std::vector<TimedEvent> event_extras;
    std::vector<int64_t> wide_click_times;

    Local(size_t n) : candidates(n), literals(literal_shards), events(n), purchases(n), clicks(n) {}

    uint32_t store_click_time(int64_t ts, int64_t base, bool compact) {
        if (compact)
            return static_cast<uint32_t>(ts - base);
        const auto index = static_cast<uint32_t>(wide_click_times.size());
        wide_click_times.push_back(ts);
        return index;
    }
    int64_t click_timestamp(uint32_t time, int64_t base) const {
        return wide_click_times.empty() ? base + time : wide_click_times[time];
    }

    PackedData pack(int64_t base, int64_t ts, Purchase purchase) {
        uint64_t diff = static_cast<uint64_t>(ts) - static_cast<uint64_t>(base);
        if (diff <= UINT32_MAX && purchase.category <= UINT8_MAX)
            return {static_cast<uint32_t>(diff), purchase.price | (purchase.category << 24)};
        auto i = static_cast<uint32_t>(event_extras.size());
        event_extras.push_back({ts, purchase});
        return {i, external};
    }
    void append_event(size_t shard, Key user, PackedData data) {
        auto& rows = events[shard];
        if (data.amount)
            purchases[shard].push_back({static_cast<uint32_t>(rows.size()), data.amount});
        rows.push_back(Event::make(user, data.offset));
    }
    TimedEvent unpack(int64_t base, PackedData data) const {
        if (data.amount == external)
            return event_extras[data.offset];
        return {base + data.offset, {data.amount & 0x00ffffff, data.amount >> 24}};
    }
};

// A dedup candidate can point directly at its event despite vector reallocation.
// Each ID shard owns all decisions for one ID, so only one task writes its losers.
struct EventLocations {
    static constexpr uint64_t invalid = UINT64_MAX;
    unsigned worker_bits, shard_bits;
    size_t worker_mask, shard_mask;
    EventLocations(unsigned workers, size_t shards)
        : worker_bits(std::bit_width(workers - 1)), shard_bits(std::countr_zero(shards)),
          worker_mask((size_t{1} << worker_bits) - 1), shard_mask(shards - 1) {}
    uint64_t encode(size_t worker, size_t shard, size_t row) const {
        return (((uint64_t{row} << shard_bits) | shard) << worker_bits) | worker;
    }
    void discard(std::vector<Local>& locals, uint64_t address) const {
        if (address == invalid)
            return;
        auto& local = locals[address & worker_mask];
        const auto position = address >> worker_bits;
        local.events[position & shard_mask][position >> shard_bits].words = {};
    }
};

// Ingestion timestamps are F1 (whole seconds). Only shards with a timestamp
// or address outside the compact range need a full-width winner table.
void add_literal(Local& local, uint64_t id, int64_t ingest, uint64_t location) {
    const size_t shard = literal_hash(id) & (literal_shards - 1);
    if (static_cast<uint64_t>(ingest) <= uint64_t{UINT32_MAX} * 1000 &&
        (location < UINT32_MAX || location == EventLocations::invalid)) {
        local.literals[shard].push_back({id, static_cast<uint32_t>(ingest / 1000), static_cast<uint32_t>(location)});
    } else {
        if (local.wide_literals.empty())
            local.wide_literals.resize(literal_shards);
        local.wide_literals[shard].push_back({id, ingest, location});
    }
}
WideLiteralCandidate expand(LiteralCandidate row) {
    return {row.id, int64_t{row.seconds} * 1000,
            row.location == UINT32_MAX ? EventLocations::invalid : uint64_t{row.location}};
}
template <class Row> void select_literal_winners(std::vector<Local>& locals, size_t shard,
                                                  size_t count, const EventLocations& addresses) {
    std::vector<Row> seen(std::bit_ceil(count + count / 2 + 1));
    const size_t mask = seen.size() - 1;
    auto select = [&](Row row) {
        size_t i = (literal_hash(row.id) >> literal_shard_bits) & mask;
        while (seen[i].id && seen[i].id != row.id)
            i = (i + 1) & mask;
        auto& previous = seen[i];
        if (!previous.id) {
            previous = row;
            return;
        }
        if constexpr (std::is_same_v<Row, LiteralCandidate>) {
            if (row.seconds < previous.seconds)
                std::swap(row, previous);
            if (row.location != UINT32_MAX)
                addresses.discard(locals, row.location);
        } else {
            if (row.ingest < previous.ingest)
                std::swap(row, previous);
            addresses.discard(locals, row.location);
        }
    };
    for (auto& local : locals) {
        for (auto row : local.literals[shard]) {
            if constexpr (std::is_same_v<Row, LiteralCandidate>)
                select(row);
            else
                select(expand(row));
        }
        std::vector<LiteralCandidate>().swap(local.literals[shard]);
        if constexpr (std::is_same_v<Row, WideLiteralCandidate>) {
            if (!local.wide_literals.empty()) {
                for (auto row : local.wide_literals[shard])
                    select(row);
                std::vector<WideLiteralCandidate>().swap(local.wide_literals[shard]);
            }
        }
    }
}
// IDs in this shard already share the low hash bits. A two-bit state for the
// remaining hash prefix proves singleton IDs cannot lose deduplication. Only
// repeated prefixes need full IDs and ingest times; a prefix collision does
// not identify an event.
void select_repeated_literals(std::vector<Local>& locals, size_t shard, size_t count,
                              const EventLocations& addresses) {
    struct Marks {
        uint64_t present = 0, repeated = 0;
    };
    const size_t words = std::bit_ceil((count + 3) / 4);
    const size_t word_mask = words - 1;
    std::vector<Marks> marks(words);
    for (const auto& local : locals) {
        for (const auto& row : local.literals[shard]) {
            const auto hash = literal_hash(row.id) >> literal_shard_bits;
            const uint64_t bit = uint64_t{1} << (hash & 63);
            auto& word = marks[(hash >> 6) & word_mask];
            word.repeated |= word.present & bit;
            word.present |= bit;
        }
    }
    // Subtract singleton prefixes to obtain the exact number of records for
    // the second pass, including distinct IDs with the same prefix.
    size_t suspects = count;
    for (const auto& word : marks)
        suspects -= std::popcount(word.present ^ word.repeated);
    std::vector<LiteralCandidate> exact(std::bit_ceil(suspects + suspects / 2 + 1));
    const size_t mask = exact.size() - 1;
    for (auto& local : locals) {
        for (auto row : local.literals[shard]) {
            const auto hash = literal_hash(row.id) >> literal_shard_bits;
            if (!((marks[(hash >> 6) & word_mask].repeated >> (hash & 63)) & 1))
                continue;
            // The prefix is shared by different original IDs, so choose
            // placement in this smaller exact table with the full key.
            size_t i = ((row.id ^ (row.id >> 23)) * UINT64_C(0x9e3779b97f4a7c15) >> 23) & mask;
            while (exact[i].id && exact[i].id != row.id)
                i = (i + 1) & mask;
            auto& winner = exact[i];
            if (!winner.id) {
                winner = row;
            } else {
                if (row.seconds < winner.seconds)
                    std::swap(row, winner);
                if (row.location != UINT32_MAX)
                    addresses.discard(locals, row.location);
            }
        }
        std::vector<LiteralCandidate>().swap(local.literals[shard]);
    }
}

void deduplicate_literals(std::vector<Local>& locals, size_t shard, const EventLocations& addresses) {
    size_t count = 0, wide_count = 0;
    for (const auto& local : locals) {
        count += local.literals[shard].size();
        if (!local.wide_literals.empty())
            wide_count += local.wide_literals[shard].size();
    }
    if (wide_count)
        select_literal_winners<WideLiteralCandidate>(locals, shard, count + wide_count, addresses);
    else if (count >= 256)
        select_repeated_literals(locals, shard, count, addresses);
    else if (count)
        select_literal_winners<LiteralCandidate>(locals, shard, count, addresses);
}

void exclude_duplicate_rows(std::vector<Local>& locals, size_t shard, const EventLocations& addresses) {
    size_t count = 0;
    for (auto& local : locals)
        count += local.candidates[shard].size();
    if (!count)
        return;
    struct Seen {
        Key id;
        const Candidate* earliest;
    };
    std::vector<Seen> seen(std::bit_ceil(count + count / 2 + 1));
    const size_t mask = seen.size() - 1;
    for (auto& local : locals) {
        for (const auto& row : local.candidates[shard]) {
            size_t i = (row.id.hash() >> 32) & mask;
            while (seen[i].id.bits() && !(seen[i].id == row.id))
                i = (i + 1) & mask;
            auto& previous = seen[i];
            if (!previous.id.bits())
                previous = {row.id, &row};
            else if (row.ingest < previous.earliest->ingest) {
                addresses.discard(locals, previous.earliest->location);
                previous.earliest = &row;
            } else {
                addresses.discard(locals, row.location);
            }
        }
    }
    // The table above refers to candidates from earlier workers in this shard.
    for (auto& local : locals)
        std::vector<Candidate>().swap(local.candidates[shard]);
}

struct Totals {
    int64_t revenue = 0;
    uint32_t sessions = 0, purchases = 0, assists = 0;
};

struct CampaignTotals {
    int32_t first;
    size_t campaigns;
    bool dense;
    std::vector<Totals> cells;
    std::unordered_map<uint64_t, Totals> extra;
    CampaignTotals(Window window, size_t n)
        : first(epoch_day(window.start)), campaigns(n),
          dense((int64_t(epoch_day(window.end - 1)) - first + 1) * n <= 1'000'000) {
        if (dense)
            cells.resize((int64_t(epoch_day(window.end - 1)) - first + 1) * n);
    }
    Totals& at(int32_t day, uint32_t campaign) {
        uint64_t index = uint64_t(int64_t(day) - first) * campaigns + campaign;
        return dense ? cells[index] : extra[index];
    }
    template <class F> void each(F&& visit) const {
        auto use = [&](uint64_t key, const Totals& x) {
            if (x.sessions || x.assists)
                visit(int32_t(first + int64_t(key / campaigns)), uint32_t(key % campaigns), x);
        };
        if (dense) {
            for (size_t i = 0; i < cells.size(); ++i)
                use(i, cells[i]);
        } else {
            for (auto& [key, x] : extra)
                use(key, x);
        }
    }
};

// A half-hour bin is always a connected piece of one session. Its earliest
// and latest event are the only timestamps needed to join neighboring bins.
struct SummarySession {
    int64_t start;
    uint32_t count;
    uint32_t first_purchase;
};
struct SummaryPurchase {
    uint32_t amount;
    // During collection this is the bin. After session reconstruction it links
    // to the next purchase in the same session.
    uint32_t at;
};
inline int64_t event_time(const SummarySession& s, int64_t) { return s.start; }
struct BucketSummary {
    static constexpr uint32_t none = UINT32_MAX;
    static constexpr size_t bins = UINT32_MAX / uint32_t(session_gap) + 1;
    struct Bucket {
        uint32_t minimum, maximum, count, session;
    };
    std::array<Bucket, bins> b{};
    std::array<uint64_t, (bins + 63) / 64> occupied{};
    std::vector<SummaryPurchase> purchases;
    std::vector<SummarySession> sessions;
    void reset() {
        purchases.clear();
        sessions.clear();
    }
    void insert(const uint32_t* rows, size_t n) {
        if (!n) return;
        for (size_t i = 0; i < n; ++i) {
            const auto ts = rows[i];
            const uint32_t k = ts / uint32_t(session_gap);
            auto& slot = b[k];
            if (slot.count++) {
                slot.minimum = std::min(slot.minimum, ts);
                slot.maximum = std::max(slot.maximum, ts);
            } else {
                occupied[k / 64] |= uint64_t{1} << (k % 64);
                slot.minimum = slot.maximum = ts;
            }
        }
        const auto* side = rows + n;
        const uint32_t count = *side++;
        for (uint32_t i = 0; i < count; ++i, side += 2)
            purchases.push_back({side[1], side[0] / uint32_t(session_gap)});
    }
    // Visiting occupied bins in time order makes session boundaries exact even
    // when source user IDs interleave or a session crosses midnight.
    void finish(int64_t base) {
        int64_t previous = -session_gap - 1;
        for (size_t i = 0; i < occupied.size(); ++i) {
            uint64_t mask = occupied[i];
            occupied[i] = 0;
            while (mask) {
                uint32_t k = 64 * i + std::countr_zero(mask);
                mask &= mask - 1;
                auto& slot = b[k];
                if (int64_t(slot.minimum) - previous > session_gap)
                    sessions.push_back({base + slot.minimum, 0, none});
                slot.session = sessions.size() - 1;
                sessions.back().count += slot.count;
                previous = slot.maximum;
                slot.count = 0;
            }
        }
        for (uint32_t j = 0; j < purchases.size(); ++j) {
            auto& session = sessions[b[purchases[j].at].session];
            purchases[j].at = session.first_purchase;
            session.first_purchase = j;
        }
    }
};

// Maintain one active entry per campaign as a person's seven-day window moves.
// Identical click copies leave both the set and the latest-click tie result
// unchanged; their increments/decrements occur at the same timestamp.
class SessionState {
    enum class ClickPath { organic, pair, sweep };
    std::vector<uint32_t> counts_, positions_, active_, categories_used_, entered_at_;
    std::vector<int64_t> category_spend_;
    const std::vector<std::string_view>& names_;

  public:
    explicit SessionState(const Dimensions& dim)
        : counts_(dim.campaigns.size()), positions_(counts_.size()), entered_at_(counts_.size()), category_spend_(dim.categories.size()),
          names_(dim.categories) {}
    template <class EventRow, class ClickRow>
    void person(Key name, const EventRow* events, size_t count, const ClickRow* clicks, size_t nclick,
                CampaignTotals& campaign, std::vector<UserDaily>& users, int64_t base,
                const BucketSummary* summary = nullptr, std::span<const uint64_t> purchases = {}) {
        if (!nclick)
            process<ClickPath::organic>(name, events, count, clicks, nclick, campaign, users, base, summary, purchases);
        else if (nclick <= 2)
            process<ClickPath::pair>(name, events, count, clicks, nclick, campaign, users, base, summary, purchases);
        else
            process<ClickPath::sweep>(name, events, count, clicks, nclick, campaign, users, base, summary, purchases);
    }

  private:
    // Zero-click histories can credit organic once per day. For one or two
    // clicks, scan two pre-ranked intervals directly. Larger histories keep
    // the rolling active-campaign set as session starts advance.
    template <ClickPath Path, class EventRow, class ClickRow>
    void process(Key name, const EventRow* events, size_t count, const ClickRow* clicks, size_t nclick,
                 CampaignTotals& campaign, std::vector<UserDaily>& users, int64_t base,
                 const BucketSummary* summary, std::span<const uint64_t> purchases) {
        size_t purchase_cursor = 0;
        const ClickRow* preferred = clicks;
        const ClickRow* alternate = nullptr;
        if constexpr (Path == ClickPath::pair) {
            if (nclick == 2) {
                alternate = clicks + 1;
                const auto first = click_time(*preferred, base), other = click_time(*alternate, base);
                if (other > first || (other == first && alternate->id < preferred->id))
                    std::swap(preferred, alternate);
            }
        }
        uint32_t organic_purchases = 0;
        // A frequently clicked campaign may assist many consecutive sessions.
        // Count that interval once when it expires or the session-start day ends.
        const bool count_intervals = Path == ClickPath::sweep && nclick >= 32;
        size_t first = 0, next = 0, e = 0;
        Key winning_id;
        int64_t winning_time = std::numeric_limits<int64_t>::min();
        uint32_t winner = 0;
        UserDaily day{};
        auto finish_day = [&] {
            if constexpr (Path == ClickPath::organic) {
                auto& organic = campaign.at(day.day, 0);
                organic.sessions += day.sessions;
                organic.purchases += organic_purchases;
                organic.revenue += day.revenue;
                organic_purchases = 0;
            }
            if (count_intervals) {
                for (uint32_t k : active_) {
                    campaign.at(day.day, k).assists += day.sessions - entered_at_[k];
                    entered_at_[k] = 0;
                }
            }
            int64_t best = 0;
            for (uint32_t k : categories_used_) {
                int64_t money = category_spend_[k];
                if (money > best || (money == best && names_[k] < names_[day.category])) {
                    day.category = static_cast<int32_t>(k);
                    best = money;
                }
                category_spend_[k] = 0;
            }
            categories_used_.clear();
            users.push_back(day);
        };
        while (e < count) {
            const int64_t start = event_time(events[e], base);
            const int32_t date = epoch_day(start);
            if (!day.sessions || day.day != date) {
                if (day.sessions)
                    finish_day();
                day = {};
                day.user = name;
                day.day = date;
                day.category = -1;
            }
            int64_t revenue = 0;
            if constexpr (std::is_same_v<EventRow, SummarySession>) {
                const auto& s = events[e++];
                day.events += s.count;
                for (uint32_t j = s.first_purchase; j != BucketSummary::none; j = summary->purchases[j].at) {
                    const uint32_t amount = summary->purchases[j].amount;
                    const uint32_t cat = amount >> 24, price = amount & 0x00ffffff;
                    if (!category_spend_[cat])
                        categories_used_.push_back(cat);
                    category_spend_[cat] += price;
                    revenue += price;
                }
            } else if constexpr (std::is_same_v<EventRow, uint32_t>) {
                const size_t first_event = e++;
                while (e < count && events[e] - events[e - 1] <= uint32_t(session_gap))
                    ++e;
                day.events += static_cast<uint32_t>(e - first_event);
                const auto last_time = events[e - 1];
                while (purchase_cursor < purchases.size() && uint32_t(purchases[purchase_cursor] >> 32) <= last_time) {
                    const uint32_t amount = static_cast<uint32_t>(purchases[purchase_cursor++]);
                    const auto category = amount >> 24, price = amount & 0x00ffffff;
                    if (!category_spend_[category])
                        categories_used_.push_back(category);
                    category_spend_[category] += price;
                    revenue += price;
                }
            } else {
                size_t session_begin = e;
                do {
                    const auto p = event_purchase(events[e++]);
                    if (p.price) {
                        if (!category_spend_[p.category])
                            categories_used_.push_back(p.category);
                        category_spend_[p.category] += p.price;
                        revenue += p.price;
                    }
                } while (e < count && event_time(events[e], base) - event_time(events[e - 1], base) <= session_gap);
                day.events += static_cast<uint32_t>(e - session_begin);
            }
            ++day.sessions;
            day.revenue += revenue;
            if constexpr (Path == ClickPath::organic) {
                organic_purchases += revenue != 0;
            } else if constexpr (Path == ClickPath::pair) {
                auto eligible = [&](const ClickRow* click) {
                    if (!click)
                        return false;
                    const auto ts = click_time(*click, base);
                    return ts <= start && ts >= start - week_ms;
                };
                const bool first_ok = eligible(preferred), second_ok = eligible(alternate);
                const auto winner = first_ok ? preferred->campaign : second_ok ? alternate->campaign : 0;
                auto& credited = campaign.at(date, winner);
                ++credited.sessions;
                credited.purchases += revenue != 0;
                credited.revenue += revenue;
                if (first_ok || second_ok)
                    ++credited.assists;
                if (first_ok && second_ok && alternate->campaign != winner)
                    ++campaign.at(date, alternate->campaign).assists;
            } else {
                while (next < nclick && click_time(clicks[next], base) <= start) {
                    const auto& c = clicks[next++];
                    if (!counts_[c.campaign]++) {
                        positions_[c.campaign] = static_cast<uint32_t>(active_.size());
                        active_.push_back(c.campaign);
                        if (count_intervals)
                            entered_at_[c.campaign] = day.sessions - 1;
                    }
                    if (click_time(c, base) != winning_time || c.id < winning_id) {
                        winner = c.campaign;
                        winning_id = c.id;
                        winning_time = click_time(c, base);
                    }
                }
                while (first < next && click_time(clicks[first], base) < start - week_ms) {
                    uint32_t k = clicks[first++].campaign;
                    if (!--counts_[k]) {
                        if (count_intervals)
                            campaign.at(date, k).assists += day.sessions - 1 - entered_at_[k];
                        size_t pos = positions_[k];
                        active_[pos] = active_.back();
                        positions_[active_[pos]] = static_cast<uint32_t>(pos);
                        active_.pop_back();
                    }
                }
                auto& credited = campaign.at(date, active_.empty() ? 0 : winner);
                ++credited.sessions;
                credited.purchases += revenue != 0;
                credited.revenue += revenue;
                if (!count_intervals) {
                    for (uint32_t k : active_)
                        ++campaign.at(date, k).assists;
                }
            }
        }
        finish_day();
        if constexpr (Path == ClickPath::sweep) {
            for (uint32_t k : active_)
                counts_[k] = 0;
            active_.clear();
        }
    }
};

struct Group {
    Key user;
    uint32_t event_end = 0, click_end = 0;
};

// Events at equal timestamps belong to the same session. Only the timestamps in
// their high 32 bits need to participate in the radix passes.
inline uint32_t sort_timestamp(uint64_t event) { return static_cast<uint32_t>(event >> 32); }
inline uint32_t sort_timestamp(const ShortClick& click) { return click.offset; }

template<unsigned bits, class Row>
void radix_by_time(Row* rows, size_t n, std::vector<Row>& scratch) {
    constexpr unsigned passes = (32 + bits - 1) / bits;
    constexpr unsigned bins = 1U << bits;
    std::array<std::array<uint32_t, bins>, passes> histogram{};
    for (size_t i = 0; i < n; ++i) {
        const uint32_t ts = sort_timestamp(rows[i]);
        for (unsigned p = 0; p < passes; ++p)
            ++histogram[p][(ts >> (bits * p)) & (bins - 1)];
    }
    for (auto& pass : histogram) {
        uint32_t sum = 0;
        for (auto& slot : pass) {
            uint32_t count = slot;
            slot = sum;
            sum += count;
        }
    }
    scratch.resize(n);
    auto* in = rows;
    auto* out = scratch.data();
    for (unsigned pass = 0; pass < histogram.size(); ++pass) {
        auto& offsets = histogram[pass];
        const unsigned shift = bits * pass;
        for (size_t i = 0; i < n; ++i) {
            const auto row = in[i];
            out[offsets[(sort_timestamp(row) >> shift) & (bins - 1)]++] = row;
        }
        std::swap(in, out);
    }
    if constexpr (passes % 2)
        std::copy(in, in + n, rows);
}

void sort_packed_events(uint64_t* events, size_t n, std::vector<uint64_t>& scratch) {
#if defined(__AVX2__)
    constexpr size_t width = event_sort::width;
    if (n >= 4 && n <= 32 * width) {
        if (n <= width) event_sort::sort_small<1>(events, n);
        else if (n <= 2 * width) event_sort::sort_small<2>(events, n);
        else if (n <= 4 * width) event_sort::sort_small<4>(events, n);
        else if (n <= 8 * width) event_sort::sort_small<8>(events, n);
        else if (n <= 16 * width) event_sort::sort_small<16>(events, n);
        else event_sort::sort_small<32>(events, n);
        return;
    }
    if (n >= 128 && n <= 4096) {
        radix_by_time<8>(events, n, scratch);
        return;
    }
#endif
    if (n < 256)
        std::sort(events, events + n);
    else
        radix_by_time<11>(events, n, scratch);
}

void sort_compact_times(uint32_t* times, size_t n) {
#if defined(__AVX2__)
    constexpr auto w = time_sort32::width;
    if (n >= 4 && n <= 32*w) {
        if (n <= w) time_sort32::sort<1>(times, n);
        else if (n <= 2*w) time_sort32::sort<2>(times, n);
        else if (n <= 4*w) time_sort32::sort<4>(times, n);
        else if (n <= 8*w) time_sort32::sort<8>(times, n);
        else if (n <= 16*w) time_sort32::sort<16>(times, n);
        else time_sort32::sort<32>(times, n);
        return;
    }
#endif
    std::sort(times, times + n);
}

void sort_compact_clicks(ShortClick* clicks, size_t n, std::vector<ShortClick>& scratch) {
#if defined(__AVX512F__)
    if (n >= 3 && n <= 64) {
        if (n <= 8) event_sort::sort_clicks<1>(clicks, n);
        else if (n <= 16) event_sort::sort_clicks<2>(clicks, n);
        else if (n <= 32) event_sort::sort_clicks<4>(clicks, n);
        else event_sort::sort_clicks<8>(clicks, n);
        return;
    }
#endif
    if (n >= 64)
        radix_by_time<8>(clicks, n, scratch);
    else
        std::sort(clicks, clicks + n, [](const ShortClick& a, const ShortClick& b) { return a.offset < b.offset; });
}

// A source shard holds contiguous records for each raw user ID. Once all
// sources are grouped, only spans of records travel to canonical person shards.
template <bool Compact> struct UserBlocks {
    using EventRow = std::conditional_t<Compact, uint32_t, TimedEvent>;
    using ClickRow = std::conditional_t<Compact, ShortClick, TimedClick>;
    std::vector<EventRow> events;
    std::vector<ClickRow> clicks;
    struct Label {
        union {
            Key name;
            uint64_t person;
        };
        Label() : name{} {}
        explicit Label(Key k) : name(k) {}
        explicit Label(uint32_t i) : person(i) {}
    };
    struct Span {
        Label label;
        EventRow* events;
        ClickRow* clicks;
        uint32_t event_count, click_count;
    };
};

template <bool Compact, class SingleFn>
void group_by_raw_id(std::vector<Local>& locals, size_t shard, unsigned shard_bits, const Aliases& aliases,
                     int64_t window_start, UserBlocks<Compact>& blocks,
                     Partitions<typename UserBlocks<Compact>::Span>& destinations, SingleFn&& on_single) {
    size_t ne = 0, nc = 0;
    for (auto& local : locals) {
        ne += local.events[shard].size();
        nc += local.clicks[shard].size();
    }
    if (!ne && !nc)
        return;
    // A raw ID with only clicks may still link to an ID with events. Both
    // streams contribute groups. Their low hash bits already chose this shard.
    // Group numbers are stable positions in the two banks. The first key
    // observed at a position retains it; exact comparisons protect arbitrary
    // ID spellings. Only double collisions enter the growing overflow table.
    const size_t primary_count = std::bit_ceil(std::max<size_t>(16, (ne + nc) / 8));
    const size_t secondary_count = std::max<size_t>(4, primary_count / 8);
    const unsigned shift = 64 - std::countr_zero(primary_count);
    const size_t fixed_count = primary_count + secondary_count;
    std::vector<Group> groups;
    groups.reserve(fixed_count + primary_count / 16);
    groups.resize(fixed_count);
    std::vector<uint32_t> purchase_ends;
    if constexpr (Compact) {
        purchase_ends.reserve(groups.capacity());
        purchase_ends.resize(fixed_count);
    }
    std::vector<uint32_t> order;
    order.reserve(primary_count / 2);
    KeyTable<uint32_t> overflow(primary_count / 128);
    auto primary_for = [&](Key user) {
        return static_cast<uint32_t>((user.bits() * UINT64_C(0x9e3779b97f4a7c15)) >> shift);
    };
    auto install = [&](uint32_t index, Key user) {
        groups[index].user = user;
        order.push_back(index);
        return index;
    };
    auto group_for = [&](Key user) {
        const uint32_t primary = primary_for(user);
        const Key leader = groups[primary].user;
        if (leader == user)
            return primary;
        if (!leader.bits())
            return install(primary, user);
        const uint32_t secondary = primary_count + ((user.hash() >> shard_bits) & (secondary_count - 1));
        const Key other = groups[secondary].user;
        if (other == user)
            return secondary;
        if (!other.bits())
            return install(secondary, user);
        return overflow.add(user, [&] {
            const auto index = static_cast<uint32_t>(groups.size());
            groups.push_back({user});
            if constexpr (Compact) purchase_ends.push_back(0);
            order.push_back(index);
            return index;
        });
    };
    for (auto& l : locals) {
        for (auto& e : l.events[shard]) {
            const auto user = e.user();
            if (!user.bits()) {
                e.words[0] = UINT32_MAX;
                continue;
            }
            uint32_t g = group_for(user);
            ++groups[g].event_end;
            e.words[0] = g;
        }
        if constexpr (Compact) {
            for (const auto pay : l.purchases[shard]) {
                const auto group = l.events[shard][pay.row].words[0];
                if (group != UINT32_MAX) ++purchase_ends[group];
            }
        }
        for (auto& c : l.clicks[shard]) {
            uint32_t g = group_for(c.user);
            ++groups[g].click_end;
            c.group = g;
        }
    }
    auto& events = blocks.events;
    auto& clicks = blocks.clicks;
    uint32_t words = 0;
    for (auto id : order) {
        words += groups[id].event_end;
        if constexpr (Compact)
            if (groups[id].event_end) words += 1 + 2 * purchase_ends[id];
    }
    events.resize(words);
    clicks.resize(nc);
    uint32_t event_offset = 0, click_offset = 0;
    for (uint32_t id : order) {
        auto& g = groups[id];
        const auto nevents = g.event_end, nclicks = g.click_end;
        g.event_end = event_offset;
        event_offset += nevents;
        if constexpr (Compact) {
            // A raw span owns N time offsets, its purchase count, and that
            // many (offset, amount) pairs. The existing span stays 32 bytes.
            const auto purchases = purchase_ends[id];
            if (nevents) events[event_offset++] = purchases;
            purchase_ends[id] = event_offset;
            event_offset += 2 * purchases;
        }
        g.click_end = click_offset;
        click_offset += nclicks;
    }
    for (auto& l : locals) {
        auto& raw = l.events[shard];
        for (auto& e : raw) {
            const auto group = e.words[0];
            if (group == UINT32_MAX) continue;
            const auto address = groups[group].event_end++;
            if constexpr (Compact) {
                events[address] = e.offset;
            } else {
                e.words[1] = address;
                events[address] = {window_start + e.offset, {0, 0}};
            }
        }
        // Lost duplicates still have their original positions in this sparse
        // stream. Their group sentinel suppresses both payload and payment.
        for (const auto pay : l.purchases[shard]) {
            const auto& e = raw[pay.row];
            const auto group = e.words[0];
            if (group == UINT32_MAX) continue;
            if constexpr (Compact) {
                auto& pos = purchase_ends[group];
                events[pos++] = e.offset;
                events[pos++] = pay.amount;
            } else {
                events[e.words[1]] = l.unpack(window_start, {e.offset, pay.amount});
            }
        }
        for (const auto& c : l.clicks[shard]) {
            uint32_t g = c.group;
            auto& row = clicks[groups[g].click_end++];
            if constexpr (Compact)
                row = {c.time, c.campaign, c.id};
            else
                row = {l.click_timestamp(c.time, window_start - week_ms), c.id, c.campaign};
        }
        std::vector<Event>().swap(l.events[shard]);
        std::vector<RawPurchase>().swap(l.purchases[shard]);
        std::vector<Click>().swap(l.clicks[shard]);
    }
    const size_t mask = destinations.size() - 1;
    uint32_t first_event = 0, first_click = 0;
    for (uint32_t id : order) {
        const auto& group = groups[id];
        const uint32_t person = aliases.person(group.user);
        using Label = typename UserBlocks<Compact>::Label;
        const auto event_count = group.event_end - first_event, click_count = group.click_end - first_click;
        if (person == absent) {
            if (event_count) {
                on_single(group.user, events.data() + first_event,
                          click_count ? clicks.data() + first_click : nullptr, event_count, click_count);
            }
        } else {
            destinations[person & mask].push_back(
                {Label(person), event_count ? events.data() + first_event : nullptr,
                 click_count ? clicks.data() + first_click : nullptr, event_count, click_count});
        }
        if constexpr (Compact) first_event = purchase_ends[id];
        else first_event = group.event_end;
        first_click = group.click_end;
    }
}

template <bool Compact>
void sessionize(std::vector<Local>& locals, const Aliases& aliases, int64_t start, unsigned workers, size_t shards,
                std::vector<SessionState>& states, std::vector<CampaignTotals>& totals, Result& result) {
    using Blocks = UserBlocks<Compact>;
    using Span = typename Blocks::Span;
    using EventRow = typename Blocks::EventRow;
    using ClickRow = typename Blocks::ClickRow;
    std::vector<Blocks> sources(shards);
    std::vector<Partitions<Span>> destinations(workers, Partitions<Span>(shards));
    struct alignas(64) Workspace {
        std::vector<uint32_t> offsets;
        std::vector<Span> spans;
        std::vector<uint64_t> event_scratch, purchases;
        BucketSummary summary;
        std::vector<ClickRow> click_scratch, merged_clicks;
        std::vector<EventRow> merged_events;
    };
    std::vector<Workspace> workspaces(workers);
    auto process_histories = [&]<bool Unlinked>(std::span<Span> spans, unsigned worker, size_t shard) {
        auto& scratch = workspaces[worker];
        auto& event_scratch = scratch.event_scratch;
        auto& summary = scratch.summary;
        auto& click_scratch = scratch.click_scratch;
        auto& merged_clicks = scratch.merged_clicks;
        auto& merged_events = scratch.merged_events;
        for (size_t first = 0; first < spans.size();) {
            const auto& group = spans[first];
            size_t end = first + 1, ne = group.event_count, nc = group.click_count;
            while (!Unlinked && end < spans.size() && spans[end].label.person == group.label.person) {
                ne += spans[end].event_count;
                nc += spans[end].click_count;
                ++end;
            }
            if (ne) {
                const Key person_name = Unlinked ? group.label.name : aliases.name(group.label.person);
                if constexpr (Compact) {
                    constexpr size_t threshold = 256;
                    // For dense histories, aggregate all raw-ID spans without copying
                    // or time-sorting their events. Sparse groups use the small sorter.
                    if (ne >= threshold) {
                        summary.reset();
                        auto* clicks = group.clicks;
                        if (end != first + 1) {
                            merged_clicks.clear();
                            merged_clicks.reserve(nc);
                        }
                        for (size_t i = first; i < end; ++i) {
                            const auto& span = spans[i];
                            summary.insert(span.events, span.event_count);
                            if (end != first + 1 && span.click_count)
                                merged_clicks.insert(merged_clicks.end(), span.clicks,
                                                     span.clicks + span.click_count);
                        }
                        summary.finish(start);
                        if (end != first + 1)
                            clicks = merged_clicks.data();
                        if (nc > 2)
                            sort_compact_clicks(clicks, nc, click_scratch);
                        states[worker].person(person_name, summary.sessions.data(), summary.sessions.size(), clicks,
                                              nc, totals[worker], result.users[shard], start, &summary);
                        first = end;
                        continue;
                    }
                }
                // Single-ID people can sort their source block directly. Multi-ID
                // people first combine their disjoint blocks into reusable vectors.
                EventRow* events = group.events;
                ClickRow* clicks = group.clicks;
                if (end != first + 1) {
                    merged_events.clear();
                    merged_events.reserve(ne);
                    merged_clicks.clear();
                    merged_clicks.reserve(nc);
                    for (size_t i = first; i < end; ++i) {
                        const auto& block = spans[i];
                        if (block.event_count)
                            merged_events.insert(merged_events.end(), block.events, block.events + block.event_count);
                        if (block.click_count)
                            merged_clicks.insert(merged_clicks.end(), block.clicks, block.clicks + block.click_count);
                    }
                    events = merged_events.data();
                    clicks = merged_clicks.data();
                }
                if constexpr (Compact) {
                    auto& purchases = scratch.purchases;
                    purchases.clear();
                    for (size_t i = first; i < end; ++i) {
                        const auto& span = spans[i];
                        if (!span.event_count) continue;
                        const auto* side = span.events + span.event_count;
                        const uint32_t count = *side++;
                        for (uint32_t j = 0; j < count; ++j, side += 2)
                            purchases.push_back((uint64_t{side[0]} << 32) | side[1]);
                    }
                    if (purchases.size() > 1)
                        sort_packed_events(purchases.data(), purchases.size(), event_scratch);
                    sort_compact_times(events, ne);
                    if (nc > 2)
                        sort_compact_clicks(clicks, nc, click_scratch);
                } else {
                    std::sort(events, events + ne, [](const EventRow& a, const EventRow& b) { return a.ts < b.ts; });
                    if (nc > 2)
                        std::sort(clicks, clicks + nc,
                                  [](const ClickRow& a, const ClickRow& b) { return a.ts < b.ts; });
                }
                if constexpr (Compact)
                    states[worker].person(person_name, events, ne, clicks, nc, totals[worker], result.users[shard],
                                          start, nullptr, scratch.purchases);
                else
                    states[worker].person(person_name, events, ne, clicks, nc, totals[worker], result.users[shard], start);
            }
            first = end;
        }
    };
    parallel_jobs(workers, shards, [&](unsigned worker, size_t shard) {
        group_by_raw_id<Compact>(
            locals, shard, std::countr_zero(shards), aliases, start, sources[shard], destinations[worker],
            [&](Key person_name, EventRow* evs, ClickRow* clks, uint32_t ne, uint32_t nc) {
                Span single{typename Blocks::Label(person_name), evs, clks, ne, nc};
                process_histories.template operator()<true>(std::span<Span>(&single, 1), worker, shard);
            });
    });
    const auto shard_bits = std::countr_zero(shards);
    parallel_jobs(workers, shards, [&](unsigned worker, size_t shard) {
        auto& offsets = workspaces[worker].offsets;
        auto& linked = workspaces[worker].spans;
        const auto person_count = aliases.size() / shards + (aliases.size() % shards > shard);
        offsets.assign(person_count, 0);
        for (const auto& writer : destinations)
            for (const auto& span : writer[shard])
                ++offsets[span.label.person >> shard_bits];
        uint32_t total = 0;
        for (auto& offset : offsets) {
            const auto count = offset;
            offset = total;
            total += count;
        }
        linked.resize(total);
        for (const auto& writer : destinations)
            for (const auto& span : writer[shard])
                linked[offsets[span.label.person >> shard_bits]++] = span;
        process_histories.template operator()<false>(linked, worker, shard);
    });
}
} // namespace

Result run_engine(const InputFiles& input, unsigned workers) {
    auto last = std::chrono::steady_clock::now();
    auto mark = [&](const char* name) {
        if (std::getenv("PROFILE")) {
            auto now = std::chrono::steady_clock::now();
            std::cerr << name << '=' << std::chrono::duration<double>(now - last).count() << "s ";
            last = now;
        }
    };
    Dimensions dim(input);
    std::optional<Aliases> aliases;
    size_t event_bytes = 0;
    for (auto part : input.events)
        event_bytes += part.size();
    // Bound the number of active write streams during parsing. Four input
    // megabytes per shard and a 512-shard cap balance cache locality and skew.
    const size_t nshards =
        std::bit_ceil(std::clamp<size_t>(event_bytes / 4'000'000, workers, std::max<unsigned>(workers, 512)));
    const size_t mask = nshards - 1;
    std::vector<Local> locals;
    for (unsigned i = 0; i < workers; ++i)
        locals.emplace_back(nshards);
    EventLocations addresses(workers, nshards);
    // Leave headroom for workers that finish more input chunks. A little extra
    // reservation avoids doubling thousands of partitions partway through parsing.
    const size_t estimate = event_bytes / (64 * nshards * workers);
    for (auto& local : locals) {
        for (auto& rows : local.literals)
            rows.reserve(estimate * nshards / literal_shards);
        for (auto& rows : local.events)
            rows.reserve(estimate);
    }
    parallel_jobs(workers, input.events.size() + 1, [&](unsigned worker, size_t task) {
        if (!task) {
            aliases.emplace(identities(input));
            return;
        }
        auto& local = locals[worker];
        auto general = [&](const EventInput &e) {
            uint64_t location = EventLocations::invalid;
            if (e.valid) {
                Key user = e.packed_user ? Key::from_packed_ascii(e.packed_user) : Key(e.user);
                const size_t shard = user.hash() & mask;
                auto& rows = local.events[shard];
                location = addresses.encode(worker, shard, rows.size());
                const auto category = e.price ? dim.product_category.get(e.product, 0) : 0;
                local.append_event(shard, user, local.pack(input.window.start, e.ts, {e.price, category}));
            }
            // Even an invalid payload can be the earliest well-formed copy.
            if (const uint64_t literal = e.literal_id ? e.literal_id : event_literal(e.id)) {
                add_literal(local, literal, e.ingest, location);
            } else {
                Key id(e.id);
                local.candidates[id.hash() & mask].push_back({id, e.ingest, location});
            }
        };
        auto basic = [&](uint64_t user_bits, uint64_t literal, int64_t time, int64_t ingest) {
            uint64_t location = EventLocations::invalid;
            if (user_bits) {
                const auto user = Key::from_packed_ascii(user_bits);
                const auto shard = user.hash() & mask;
                auto &rows = local.events[shard];
                location = addresses.encode(worker, shard, rows.size());
                local.append_event(shard, user, local.pack(input.window.start, time, {0, 0}));
            }
            add_literal(local, literal, ingest, location);
        };
        EventSink sink{general, basic};
        for_events(input.events[task - 1], input.window, sink);
    });
    mark("events+links");
    const int64_t click_start = input.window.start - week_ms;
    const bool compact_clicks = input.window.end - input.window.start <= int64_t{UINT32_MAX} - week_ms;
    // Estimate records from uniformly spaced physical chunks. This keeps
    // the first write to each click partition from repeatedly copying all of
    // its preceding records, while preserving ordinary growth for skew.
    size_t click_bytes = 0, sampled_bytes = 0, sampled_rows = 0;
    for (const auto &chunk : input.clicks) {
        click_bytes += chunk.size();
        const auto sample = chunk.raw.substr(0, 2048);
        sampled_bytes += sample.size();
        sampled_rows += std::count(sample.begin(), sample.end(), '\n');
    }
    if (sampled_rows) {
        const auto partitions = workers * nshards;
        const auto projected = static_cast<size_t>(static_cast<long double>(sampled_rows) * click_bytes /
                                                   sampled_bytes / partitions);
        // Avoid crossing a geometrically grown capacity for only a handful
        // of spare slots. Leave at least 1/24 headroom for worker imbalance.
        const auto tier = std::bit_ceil(projected + projected / 24 + 8);
        const auto capacity =
            std::min({projected + projected / 8 + 8, tier, click_bytes / (24 * partitions) + 1});
        for (auto &local : locals)
            for (auto &rows : local.clicks)
                rows.reserve(capacity);
    }
    parallel_jobs(workers, input.clicks.size() + literal_shards + nshards, [&](unsigned worker, size_t task) {
        if (task < input.clicks.size()) {
            auto& local = locals[worker];
            auto general = [&](const ClickInput &c) {
                const auto campaign = dim.campaign_index.get(c.campaign, absent);
                if (campaign == absent) return;
                Key user(c.user);
                const uint32_t time = local.store_click_time(c.ts, click_start, compact_clicks);
                local.clicks[user.hash() & mask].push_back({user, Key(c.id), time, campaign});
            };
            auto indexed = [&](uint64_t click_bits, uint64_t user_bits, uint32_t slot, int64_t ts) {
                const auto campaign = dim.campaign_index.known_slot(slot);
                if (!campaign) return false;
                if (*campaign != absent) {
                    auto user = Key::from_packed_ascii(user_bits);
                    const uint32_t time = local.store_click_time(ts, click_start, compact_clicks);
                    local.clicks[user.hash() & mask].push_back(
                        {user, Key::from_packed_ascii(click_bits), time, *campaign});
                }
                return true;
            };
            ClickSink sink{general, indexed};
            for_clicks(input.clicks[task], input.window, sink);
        } else if (task < input.clicks.size() + literal_shards) {
            deduplicate_literals(locals, task - input.clicks.size(), addresses);
        } else {
            exclude_duplicate_rows(locals, task - input.clicks.size() - literal_shards, addresses);
        }
    });
    mark("clicks+dedup");
    Result result;
    result.users.resize(nshards);
    std::vector<CampaignTotals> totals;
    std::vector<SessionState> states;
    for (unsigned i = 0; i < workers; ++i) {
        totals.emplace_back(input.window, dim.campaigns.size());
        states.emplace_back(dim);
    }
    const bool compact = compact_clicks && dim.categories.size() <= uint32_t{UINT8_MAX} + 1;
    if (compact)
        sessionize<true>(locals, *aliases, input.window.start, workers, nshards, states, totals, result);
    else
        sessionize<false>(locals, *aliases, input.window.start, workers, nshards, states, totals, result);
    CampaignTotals combined(input.window, dim.campaigns.size());
    for (auto& table : totals)
        table.each([&](int32_t d, uint32_t c, const Totals& t) {
            auto& x = combined.at(d, c);
            x.sessions += t.sessions;
            x.purchases += t.purchases;
            x.revenue += t.revenue;
            x.assists += t.assists;
        });
    combined.each([&](int32_t d, uint32_t c, const Totals& t) {
        result.campaigns.push_back({c, d, t.sessions, t.purchases, t.assists, t.revenue});
    });
    result.category_names = std::move(dim.categories);
    result.campaign_names = std::move(dim.campaigns);
    result.channel_names = std::move(dim.channels);
    mark("sessions");
    return result;
}
} // namespace shopping
