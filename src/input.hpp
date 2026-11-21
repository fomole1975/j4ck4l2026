#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string_view>
#include <vector>

#if defined(__AVX2__) || defined(__AVX512BW__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace shopping {
struct Chunk {
    std::string_view raw;
    const char* eof;
    size_t size() const { return raw.size(); }

    // Adjacent workers choose the same boundary, even when a physical chunk
    // starts exactly on an LF. The first worker removes the file header.
    std::string_view aligned() const {
        auto after_line = [&](const char* p) {
            const auto* nl = static_cast<const char*>(std::memchr(p,'\n',eof-p));
            return nl ? nl+1 : eof;
        };
        const char* first = after_line(raw.data());
        const char* last = raw.data()+raw.size();
        if (last != eof) last = after_line(last);
        return {first,static_cast<size_t>(last-first)};
    }
};

struct Window { int64_t start, end; };

class InputFiles {
public:
    explicit InputFiles(const std::filesystem::path& root);
    ~InputFiles();
    InputFiles(const InputFiles&) = delete;
    InputFiles& operator=(const InputFiles&) = delete;
    Window window{};
    std::vector<Chunk> events, clicks, links, products, campaigns;
private:
    struct Mapping;
    std::vector<Mapping> mappings;
    void add(const std::filesystem::path&, std::vector<Chunk>&);
};

struct EventInput {
    std::string_view id, user, product;
    int64_t ts, ingest;
    uint32_t price;
    bool valid;
    uint64_t packed_user = 0, literal_id = 0;
};
struct ClickInput {
    std::string_view id, user, campaign;
    int64_t ts;
};

namespace input_detail {
constexpr int64_t bad_time = std::numeric_limits<int64_t>::min();
constexpr int64_t day_ms = 86400000;
struct Masks { uint64_t delimiters, newlines, quotes; };

#if defined(__aarch64__)
// The deinterleaved load groups four adjacent input bytes per lane. Form
// one comparison nibble per lane, then pack pairs into an ordinary bitmask.
inline uint64_t eq64(uint8x16x4_t block, uint8_t byte) {
    const uint8x16_t target = vdupq_n_u8(byte);
    auto mask = vceqq_u8(block.val[3],target);
    mask = vsliq_n_u8(vceqq_u8(block.val[2],target),mask,1);
    mask = vsliq_n_u8(vceqq_u8(block.val[1],target),mask,1);
    mask = vsliq_n_u8(vceqq_u8(block.val[0],target),mask,1);
    auto nibbles = vreinterpretq_u16_u8(vandq_u8(mask,vdupq_n_u8(15)));
    auto packed = vmovn_u16(vorrq_u16(nibbles,vshrq_n_u16(nibbles,4)));
    return vget_lane_u64(vreinterpret_u64_u8(packed),0);
}
#endif

inline Masks masks64(const char* p) {
#if defined(__AVX512BW__)
    const __m512i data = _mm512_loadu_si512(p);
    uint64_t nl = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8('\n'));
    uint64_t comma = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8(','));
    uint64_t quote = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8('"'));
    return {nl | comma, nl, quote};
#elif defined(__AVX2__)
    const __m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    const __m256i hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 32));
    auto equal = [&](char c) -> uint64_t {
        const auto v = _mm256_set1_epi8(c);
        const uint32_t a = _mm256_movemask_epi8(_mm256_cmpeq_epi8(lo, v));
        const uint32_t b = _mm256_movemask_epi8(_mm256_cmpeq_epi8(hi, v));
        return (uint64_t{b} << 32) | a;
    };
    uint64_t nl = equal('\n');
    return {nl | equal(','), nl, equal('"')};
#elif defined(__aarch64__)
    const auto block = vld4q_u8(reinterpret_cast<const uint8_t*>(p));
    const uint64_t nl = eq64(block,'\n');
    return {nl | eq64(block,','), nl, eq64(block,'"')};
#else
    Masks result{};
    for (unsigned i = 0; i < 64; ++i) {
        const uint64_t bit = UINT64_C(1) << i;
        if (p[i] == ',' || p[i] == '\n') result.delimiters |= bit;
        if (p[i] == '\n') result.newlines |= bit;
        if (p[i] == '"') result.quotes |= bit;
    }
    return result;
#endif
}

// Stream structural masks across blocks. Each field checks both its quote
// count and its endpoints; line recovery always occurs at the next LF.
template<size_t N, class Callback>
void csv_fallback(std::string_view chunk, Callback&& row) {
    if (chunk.empty()) return;
    std::array<std::string_view,N> columns;
    const char* field = chunk.data();
    const char* p = field;
    const char* end = field + chunk.size();
    unsigned index = 0;
    unsigned quotes = 0;
    auto finish = [&](const char* stop, bool newline) {
        if (index < N) {
            if (quotes == 0) {
                columns[index++] = std::string_view(field, stop - field);
            } else if (quotes == 2 && stop - field >= 2 && field[0] == '"' && stop[-1] == '"') {
                columns[index++] = std::string_view(field + 1, stop - field - 2);
            } else {
                index = N + 1;
            }
        } else {
            index = N + 1;
        }
        if (newline) {
            if (index == N) row(columns);
            index = 0;
        }
        field = stop == end ? stop : stop + 1;
        quotes = 0;
    };
    while (end - p >= 64) {
        auto m = masks64(p);
        while (m.delimiters) {
            unsigned pos = std::countr_zero(m.delimiters);
            uint64_t bit = UINT64_C(1) << pos;
            uint64_t prefix = bit | (bit - 1);
            quotes += std::popcount(m.quotes & prefix);
            finish(p + pos, (m.newlines & bit) != 0);
            m.quotes &= ~prefix;
            m.delimiters &= m.delimiters - 1;
        }
        quotes += std::popcount(m.quotes);
        p += 64;
    }
    for (; p != end; ++p) {
        if (*p == ',' || *p == '\n') finish(p, *p == '\n');
        else if (*p == '"') ++quotes;
    }
    if (field != end || index != 0) finish(end, true);
}

inline uint64_t quote_parity(uint64_t bits) {
#if defined(__PCLMUL__)
    const auto product = _mm_clmulepi64_si128(_mm_cvtsi64_si128(bits),_mm_set1_epi64x(-1),0);
    return static_cast<uint64_t>(_mm_cvtsi128_si64(product));
#elif defined(__aarch64__) && defined(__ARM_FEATURE_CRYPTO)
    return static_cast<uint64_t>(vmull_p64(bits,UINT64_MAX));
#else
    bits ^= bits << 1; bits ^= bits << 2; bits ^= bits << 4;
    bits ^= bits << 8;
    bits ^= bits << 16;
    bits ^= bits << 32;
    return bits;
#endif
}

#if defined(__AVX2__) && defined(__BMI2__) && !defined(__AVX512VBMI2__)
inline uint64_t sparse_delimiters(uint64_t delimiters, uint64_t closing) {
    // Each PEXT emits one coordinate bit per separator, already in CSV order.
    // A shared two-KiB table spreads each eight-bit plane to eight bytes.
    static constexpr auto spread = [] {
        std::array<uint64_t, 256> bytes{};
        for (unsigned i = 1; i < 256; ++i)
            bytes[i] = (bytes[i >> 1] << 8) | (i & 1);
        return bytes;
    }();
    auto plane = [&](uint64_t pattern) { return spread[_pext_u64(pattern, delimiters)]; };
    return plane(closing) | (plane(UINT64_C(0xaaaaaaaaaaaaaaaa)) << 1) |
           (plane(UINT64_C(0xcccccccccccccccc)) << 2) | (plane(UINT64_C(0xf0f0f0f0f0f0f0f0)) << 3) |
           (plane(UINT64_C(0xff00ff00ff00ff00)) << 4) | (plane(UINT64_C(0xffff0000ffff0000)) << 5) |
           (plane(UINT64_C(0xffffffff00000000)) << 6);
}
#endif

// Validate quoting in whole SIMD blocks. If a block contains unusual input,
// replay this short batch with the fully general parser to recover per line.
template <size_t N, class Callback> bool csv_batch(std::string_view part, Callback &row, size_t &fault) {
    constexpr size_t capacity = 16384;
    if (part.size() + 1 >= capacity) return false;
    // Packed stores may write 31 padding positions after the logical end.
    std::array<uint16_t,capacity+31> ends;
    size_t tokens = 0;
    // A virtual LF lets the final record use exactly the same token logic.
    const size_t synthetic = part.back() != '\n';
    const size_t length = part.size() + synthetic;
#if defined(__AVX512VBMI2__)
    size_t newlines = 0;
#endif
#if defined(__AVX512VBMI2__) || (defined(__AVX2__) && defined(__BMI2__))
    uint64_t previous_quote = 0;
#else
    unsigned carry = 0;
#endif
    uint64_t prev_separator = 1, prev_parity = 0;
#if defined(__AVX2__) && defined(__BMI2__) && !defined(__AVX512VBMI2__)
    const bool sparse_cpu=__builtin_cpu_is("intel");
#endif
    for (size_t off=0; off<length; off+=64) {
        Masks m;
        if (part.size() - off >= 64) {
            m = masks64(part.data()+off);
        } else {
            char tail[64]{};
            size_t remainder = part.size()-off;
            std::memcpy(tail,part.data()+off,remainder);
            if (synthetic) tail[remainder] = '\n';
            m = masks64(tail);
        }
        uint64_t next_separator = 0;
        if (off + 64 < part.size()) {
            char c = part[off+64];
            next_separator = uint64_t(c == ',' || c == '\n') << 63;
        } else if (off + 64 == part.size() && synthetic) {
            next_separator = UINT64_C(1)<<63;
        }
        const uint64_t d = m.delimiters;
        // Quotes may touch only a field edge. Even parity at each separator
        // then guarantees every quoted field has both a start and an end.
        const uint64_t near_delim = (d<<1)|prev_separator|(d>>1)|next_separator;
        const uint64_t parity = quote_parity(m.quotes)^prev_parity;
        const auto invalid = (m.quotes & ~near_delim) | (parity & d);
        if (invalid) {
            fault = off + std::countr_zero(invalid);
            return false;
        }
        prev_separator = d >> 63;
        prev_parity = uint64_t(0) - (parity >> 63);

#if defined(__AVX512VBMI2__) || (defined(__AVX2__) && defined(__BMI2__))
        const uint64_t closing = (m.quotes << 1) | previous_quote;
        previous_quote = m.quotes >> 63;
#endif
#if defined(__AVX512VBMI2__)
        newlines += std::popcount(m.newlines);
        alignas(64) static constexpr uint16_t offsets[32] = {
             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
            16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};
        const __m512i relative = _mm512_load_si512(offsets);
        for (unsigned lane = 0; lane < 64; lane += 32) {
            const auto active = static_cast<uint32_t>(d >> lane);
            const auto quoted = static_cast<uint32_t>(closing >> lane);
            const __m512i positions = _mm512_add_epi16(relative, _mm512_set1_epi16(off+lane));
            const __m512i marked = _mm512_mask_add_epi16(positions, quoted, positions, _mm512_set1_epi16(INT16_MIN));
            _mm512_storeu_si512(ends.data()+tokens, _mm512_maskz_compress_epi16(active, marked));
            tokens += std::popcount(active);
        }
#elif defined(__AVX2__) && defined(__BMI2__)
        // Compress the LF bits into delimiter order. With N fields, every
        // Nth delimiter must be an LF, including across block boundaries.
        static constexpr auto newline_patterns = [] {
            std::array<uint64_t,N> patterns{};
            for (unsigned phase = 0; phase < N; ++phase)
                for (unsigned bit = N-1-phase; bit < 64; bit += N)
                    patterns[phase] |= UINT64_C(1) << bit;
            return patterns;
        }();
        const unsigned count = std::popcount(d);
        const auto expected = _bzhi_u64(newline_patterns[tokens % N], count);
        if (_pext_u64(m.newlines, d) != expected) {
            fault = off;
            return false;
        }
        if(sparse_cpu && count<=8) {
            const auto packed=_mm_cvtsi64_si128(sparse_delimiters(d,closing));
            auto entries = _mm_cvtepu8_epi16(packed);
#if defined(__AVX512BW__)
            entries = _mm_or_si128(_mm_srli_epi16(entries, 1), _mm_slli_epi16(entries, 15));
            entries = _mm_add_epi16(entries, _mm_set1_epi16(off));
#else
            entries = _mm_add_epi16(entries, _mm_set1_epi16(2 * off));
#endif
            _mm_storeu_si128(reinterpret_cast<__m128i *>(ends.data() + tokens), entries);
            tokens += count;
            continue;
        }
        uint64_t remaining = d;
        while (remaining) {
            const unsigned pos = std::countr_zero(remaining);
#if defined(__AVX512BW__)
            ends[tokens++] = static_cast<uint16_t>((off + pos) | (((closing >> pos) & 1) << 15));
#else
            ends[tokens++] = static_cast<uint16_t>(2 * (off + pos) + ((closing >> pos) & 1));
#endif
            remaining &= remaining-1;
        }
#else
        uint64_t unmatched = d;
        while (m.newlines) {
            uint64_t bit = m.newlines & -m.newlines;
            uint64_t prefix = bit | (bit - 1);
            if (carry + std::popcount(unmatched & prefix) != N) {
                fault = off + std::countr_zero(bit);
                return false;
            }
            carry = 0;
            unmatched &= ~prefix;
            m.newlines &= m.newlines - 1;
        }
        carry += std::popcount(unmatched);
        while (m.delimiters) {
            ends[tokens++] = static_cast<uint16_t>(off+std::countr_zero(m.delimiters));
            m.delimiters &= m.delimiters - 1;
        }
#endif
    }
    const char* base = part.data();
#if defined(__AVX512VBMI2__)
    // The count proves there are no extra LFs once every Nth separator is
    // an LF. Complete validation before callbacks so recovery is atomic.
    if (tokens != N * newlines) return false;
    for (size_t i = N-1; i < tokens; i += N) {
        const unsigned stop = ends[i] & 0x7fff;
        if (stop < part.size() && base[stop] != '\n') {
            fault = stop;
            return false;
        }
    }
#elif defined(__AVX2__) && defined(__BMI2__)
    if (tokens % N) return false;
#endif
    unsigned first=0;
    size_t i = 0;
#if defined(__AVX512BW__)
    if constexpr (requires { row.eight(base, ends.data(), first); }) {
        for (; i + 8 * N < tokens; i += 8 * N) {
            row.eight(base, ends.data() + i, first);
            first = (ends[i + 8 * N - 1] & 0x7fff) + 1;
        }
    }
#elif defined(__AVX2__) && defined(__BMI2__)
    if constexpr (requires { row.four(base, ends.data(), first); }) {
        for (; i + 4*N < tokens; i += 4*N) {
            row.four(base, ends.data() + i, first);
            first = (ends[i + 4*N - 1] >> 1) + 1;
        }
    }
#endif
    for (; i < tokens; i += N) {
        std::array<std::string_view,N> fields;
        for (unsigned j=0; j<N; ++j) {
#if defined(__AVX512BW__)
            const unsigned entry = ends[i+j];
            const unsigned last = entry & 0x7fff;
            const unsigned quoted = entry >> 15;
#elif defined(__AVX2__) && defined(__BMI2__)
            const unsigned entry = ends[i+j];
            const unsigned last = entry >> 1;
            const unsigned quoted = entry & 1;
#else
            unsigned last = ends[i+j];
            const unsigned quoted = (first < last) && (base[first] == '"');
#endif
            fields[j] = {base+first+quoted,last-first-2*quoted};
            first = last+1;
        }
        row(fields);
    }
    return true;
}

template <size_t N, class Callback>
void recover_csv(std::string_view part, Callback &row, size_t fault, unsigned retries) {
    if (part.size() < 512 || retries == 0 || part.size() >= 16383) {
        csv_fallback<N>(part, row);
        return;
    }
    fault = std::min(fault, part.size() - 1);
    const auto left = fault ? part.rfind('\n', fault - 1) : part.npos;
    const auto right = part.find('\n', fault);
    const size_t first = left == part.npos ? 0 : left + 1,
                 last = right == part.npos ? part.size() : right + 1;
    auto check = [&](std::string_view portion) {
        if (portion.empty()) return;
        size_t at = portion.size() / 2;
        if (!csv_batch<N>(portion, row, at)) recover_csv<N>(portion, row, at, retries - 1);
    };
    check(part.substr(0, first));
    csv_fallback<N>(part.substr(first, last - first), row);
    check(part.substr(last));
}

#include "streaming_csv.hpp"

template<size_t N, class Callback>
void csv(std::string_view chunk, Callback&& row) {
    while (!chunk.empty()) {
        size_t n=chunk.size();
        if (n > 8192) {
            const size_t next = chunk.find('\n',8191);
            if (next != chunk.npos) n=next+1;
        }
        auto part = chunk.substr(0,n);
#if defined(__AVX512BW__)
        if constexpr (N == 7 && requires { row.eight(part.data(), static_cast<const uint16_t*>(nullptr), 0u); }) {
            if (part.size() < 16383)
                part.remove_prefix(stream_event_fields(part, row));
        }
#endif
        size_t fault = part.size() / 2;
        if (!csv_batch<N>(part, row, fault)) recover_csv<N>(part, row, fault, 4);
        chunk.remove_prefix(n);
    }
}

inline unsigned digit(char c) { return static_cast<unsigned char>(c) - unsigned('0'); }
inline unsigned pair_value(const char* p) {
    return (static_cast<unsigned char>(p[0]) & 15u) * 10 + (static_cast<unsigned char>(p[1]) & 15u);
}

int64_t midnight(unsigned y, unsigned m, unsigned d);

inline uint64_t load8(const char* p) {
    uint64_t word;
    std::memcpy(&word,p,8);
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    word = __builtin_bswap64(word);
#endif
    return word;
}

inline bool ascii_digits8(uint64_t w) {
    // Reject non-ASCII first, then check all lanes fall in [0x30,0x39].
    constexpr uint64_t high = UINT64_C(0x8080808080808080);
    return ((w | (w + UINT64_C(0x4646464646464646)) |
            (w - UINT64_C(0x3030303030303030))) & high) == 0;
}

// Combine eight digit nibbles into four pairs, two groups of four, then
// one eight-digit integer, without a serial multiply per character.
inline uint32_t decimal8(uint64_t w) {
    w &= UINT64_C(0x0f0f0f0f0f0f0f0f);
    w = (w * 10 + (w >> 8)) & UINT64_C(0x00ff00ff00ff00ff);
    w = (w * 100 + (w >> 16)) & UINT64_C(0x0000ffff0000ffff);
    return static_cast<uint32_t>(w * 10000 + (w >> 32));
}

inline int64_t civil_day(const char* p) {
    struct Entry { uint64_t first = 0; uint16_t last = 0; int64_t value = bad_time; };
    static thread_local std::array<Entry,64> cache;
    const uint64_t first = load8(p);
    uint16_t last;
    std::memcpy(&last,p+8,2);
    // Every day of a month gets its own slot. Alternate month parity
    // avoids collisions while parsing dates near a month boundary.
    const unsigned slot = ((p[8] & 3) * 10 + (p[9] & 15)) & 31;
    auto& entry = cache[((p[6] & 1) << 5) | slot];
    if (entry.first == first && entry.last == last) return entry.value;
    int64_t value = bad_time;
    constexpr uint64_t hyphens = (UINT64_C(0xff) << 32) | (UINT64_C(0xff) << 56);
    constexpr uint64_t expected = (UINT64_C(0x2d) << 32) | (UINT64_C(0x2d) << 56);
    if ((first & hyphens) == expected && ascii_digits8(first + ((UINT64_C(3) << 32) | (UINT64_C(3) << 56))) &&
        digit(p[8]) < 10 && digit(p[9]) < 10) {
        value = midnight(pair_value(p)*100+pair_value(p+2),pair_value(p+5),pair_value(p+8));
    }
    entry = {first,last,value};
    return value;
}

template<bool validate>
inline int64_t civil(const char* p) {
    const uint64_t w = load8(p + 11);
    const unsigned h = (w & 15u)*10 + ((w >> 8)&15u);
    const unsigned m = ((w >> 24)&15u)*10 + ((w >> 32)&15u);
    const unsigned s = ((w >> 48)&15u)*10 + ((w >> 56)&15u);
    if constexpr (validate) {
        constexpr uint64_t colons = (UINT64_C(0xff) << 16) | (UINT64_C(0xff) << 40);
        constexpr uint64_t expected = (UINT64_C(0x3a) << 16) | (UINT64_C(0x3a) << 40);
        if (p[10] != 'T' || (w & colons) != expected || h >= 24 || m >= 60 || s >= 60 ||
            !ascii_digits8(w - ((UINT64_C(10) << 16) | (UINT64_C(10) << 40)))) return bad_time;
    }
    const int64_t day = civil_day(p);
    if (day == bad_time) return bad_time;
    return day + ((h * 60 + m) * 60 + s) * 1000;
}

#if defined(__AVX2__)
inline int64_t epoch13(const char* p) {
    // Two overlapping eight-byte loads stay within the 13-byte field.
    const __m128i raw = _mm_set_epi64x(load8(p+5), load8(p));
    const __m128i digits = _mm_sub_epi8(raw, _mm_set1_epi8('0'));
    const __m128i invalid = _mm_subs_epu8(digits, _mm_set1_epi8(9));
    if (!_mm_testz_si128(invalid, invalid)) return bad_time;
    const __m128i order = _mm_setr_epi8(-1,-1,-1,0,1,2,3,4,8,9,10,11,12,13,14,15);
    const __m128i padded = _mm_shuffle_epi8(digits, order);
    const __m128i pairs = _mm_maddubs_epi16(padded, _mm_set1_epi16(0x010a));
    const __m128i fours = _mm_madd_epi16(pairs, _mm_set1_epi32(0x00010064));
    const __m128i eights = _mm_madd_epi16(_mm_packs_epi32(fours,fours), _mm_set1_epi32(0x00012710));
    return int64_t(_mm_cvtsi128_si32(eights)) * 100000000 + _mm_extract_epi32(eights,1);
}

inline void paired_civil(const char* event, const char* ingest, int64_t& event_ms, int64_t& ingest_ms) {
    // Within each 64-bit lane the eight bytes are "hh:mm:ss". Subtraction
    // simultaneously converts ASCII digits and makes valid colons zero.
    const __m128i raw = _mm_set_epi64x(load8(ingest+11), load8(event+11));
    const __m128i digits = _mm_sub_epi8(raw, _mm_set1_epi64x(UINT64_C(0x30303a30303a3030)));
    const __m128i pair_weights = _mm_setr_epi8(10,1,0,10,1,0,10,1,10,1,0,10,1,0,10,1);
    const __m128i second_weights = _mm_setr_epi16(3600,60,60,1,3600,60,60,1);
    const __m128i pairs = _mm_maddubs_epi16(digits, pair_weights);
    const __m128i halves = _mm_madd_epi16(pairs, second_weights);
    const __m128i seconds = _mm_hadd_epi32(halves, halves);
    const int64_t ingest_day = civil_day(ingest);
    const int64_t event_day = std::memcmp(event, ingest, 10) == 0 ? ingest_day : civil_day(event);
    // The tens digit of minutes/seconds is at most five. With a leading
    // hour digit of two the units digit is additionally restricted to 0–3.
    const __m128i limits = _mm_setr_epi8(2,9,0,5,9,0,5,9,2,9,0,5,9,0,5,9);
    const __m128i invalid = _mm_subs_epu8(digits, limits);
    if (_mm_cvtsi128_si64(invalid) || (event[11] == '2' && event[12] > '3') ||
        event[10] != 'T' || event_day == bad_time) {
        event_ms = bad_time;
    } else {
        event_ms = event_day + int64_t(_mm_cvtsi128_si32(seconds))*1000;
    }
    // The input contract guarantees F1 ingestion times. Preserve the plain
    // decoder's behavior for out-of-contract input with malformed bytes.
    if (_mm_extract_epi64(invalid,1)) {
        ingest_ms = civil<false>(ingest);
    } else {
        ingest_ms = ingest_day == bad_time ? bad_time : ingest_day + int64_t(_mm_extract_epi32(seconds,1))*1000;
    }
}
#endif

inline int64_t timestamp(std::string_view s) {
    const char* p = s.data();
    if (s.size() == 13) {
#if defined(__AVX2__)
        return epoch13(p);
#else
        uint64_t a = load8(p), b = load8(p+5);
        if (!ascii_digits8(a) || !ascii_digits8(b)) return bad_time;
        return int64_t(decimal8(a)/1000)*100000000 + decimal8(b);
#endif
    }
    if ((s.size() == 20 || s.size() == 25) && p[4] == '-') {
        const int64_t base = civil<true>(p);
        if (s.size() == 20) return p[19] == 'Z' ? base : bad_time;
        if ((p[19] != '+' && p[19] != '-') || p[22] != ':' || digit(p[20]) >= 10 ||
            digit(p[21]) >= 10 || digit(p[23]) >= 10 || digit(p[24]) >= 10 || base == bad_time)
            return bad_time;
        const unsigned h = pair_value(p + 20), m = pair_value(p + 23);
        if (h >= 24 || m >= 60) return bad_time;
        const int64_t offset = (h * 60 + m) * 60000;
        return p[19] == '+' ? base - offset : base + offset;
    }
    if (s.empty()) return bad_time;
    uint64_t value;
    const auto result = std::from_chars(p, p + s.size(), value);
    if (result.ec != std::errc{} || result.ptr != p + s.size() || value > INT64_MAX) return bad_time;
    return static_cast<int64_t>(value);
}

inline uint32_t price(std::string_view s) {
    if (s.empty()) return 0;
    uint32_t value;
    const auto result = std::from_chars(s.data(), s.data() + s.size(), value);
    return result.ec == std::errc{} && result.ptr == s.data() + s.size() && value <= 10000000 ?
           value : 0;
}
} // namespace input_detail

namespace input_detail {
template <class Callback> struct EventDecoder {
    Window w;
    Callback &emit;
    void operator()(const std::array<std::string_view, 7> &f) {
        if (f[0].empty()) return;
        int64_t ts, ingest;
#if defined(__AVX2__)
        if (f[2].size() == 20 && f[2][19] == 'Z' && f[3].size() == 20) {
            paired_civil(f[2].data(), f[3].data(), ts, ingest);
        } else
#endif
        {
            ts = timestamp(f[2]);
            // Ingestion timestamps are guaranteed to be valid F1.
            ingest = f[3].size() == 20 ? civil<false>(f[3].data()) : bad_time;
        }
        const bool purchase = f[4] == "purchase";
        const uint32_t cents = purchase ? price(f[6]) : 0;
        bool valid = !f[1].empty() && !f[5].empty() && ts >= w.start && ts < w.end &&
                     ingest != bad_time && ingest <= ts + day_ms &&
                     (purchase ? cents != 0 : f[4] == "view" || f[4] == "cart");
        emit(EventInput{f[0],f[1],f[5],ts,ingest,cents,valid});
    }
#if defined(__AVX512BW__)
    void eight(const char *base, const uint16_t *ends, unsigned first);
    template<class Columns, class Fields>
    void project8(const char *base, Columns columns, Fields fields);
#elif defined(__AVX2__) && defined(__BMI2__)
    void four(const char *base, const uint16_t *ends, unsigned first);
    template <class Columns, class Fields> void project4(const char *base, Columns columns, Fields fields);
#endif
};
} // namespace input_detail
#include "column_times.hpp"
#include "column_time4.hpp"
#include "anchored_events.hpp"

template <class Callback> void for_events(Chunk chunk, Window w, Callback &&emit) {
    input_detail::EventDecoder<std::remove_reference_t<Callback>> decoder{w, emit};
#if defined(__AVX512BW__)
    input_detail::anchor_events(chunk.aligned(), decoder);
    return;
#elif defined(__AVX2__) && defined(__BMI2__) && !defined(__AVX512BW__)
    if (__builtin_cpu_is("intel") || __builtin_cpu_is("znver3")) {
        input_detail::anchor_events(chunk.aligned(), decoder);
        return;
    }
#endif
#if defined(__AVX512BW__) || (defined(__AVX2__) && defined(__BMI2__))
    if (!__builtin_cpu_is("intel")) {
        // EPYC performs the row path's individual scalar gathers efficiently.
        // The wrapper deliberately exposes only the scalar row callback.
        auto scalar_row = [&](const auto& fields) { decoder(fields); };
        input_detail::csv<7>(chunk.aligned(), scalar_row);
        return;
    }
#endif
    input_detail::csv<7>(chunk.aligned(), decoder);
}

namespace input_detail {
template <class Callback> struct ClickDecoder {
    Window w;
    Callback &emit;
    void operator()(const std::array<std::string_view, 4> &f) {
        if (f[0].empty() || f[1].empty() || f[2].empty()) return;
        auto time = timestamp(f[3]);
        if (time >= w.start - 7 * day_ms && time < w.end) emit(ClickInput{f[0], f[1], f[2], time});
    }
#if defined(__AVX2__) && defined(__BMI2__)
    void four(const char *, const uint16_t *, unsigned);
#if defined(__AVX512BW__)
    void eight(const char *base, const uint16_t *ends, unsigned first);
#endif
#endif
};
} // namespace input_detail
#include "column_clicks.hpp"
#include "column_click8.hpp"

template<class Callback>
void for_clicks(Chunk chunk, Window w, Callback&& emit) {
    input_detail::ClickDecoder<std::remove_reference_t<Callback>> decoder{w, emit};
    input_detail::csv<4>(chunk.aligned(), decoder);
}

template<class Callback>
void for_pairs(Chunk chunk, Callback&& emit) {
    input_detail::csv<2>(chunk.aligned(), [&](const auto& f) {
        if (!f[0].empty() && !f[1].empty()) emit(f[0],f[1]);
    });
}
}
