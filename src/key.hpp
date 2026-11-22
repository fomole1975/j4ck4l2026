#pragma once

#include <compare>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <string_view>

#if defined(__BMI2__) || defined(__SSE4_2__)
#include <immintrin.h>
#endif

namespace shopping {

// Nine nonzero ASCII bytes fit in 63 bits. Their inlined ordering is also
// their bytewise ordering. Other byte strings live in a process-wide pool.
class Key {
public:
    // A byte string that does not pack into 63 bits, stored once per process.
    struct Interned {
        uint64_t hash;
        uint32_t size;
        const char* data() const { return reinterpret_cast<const char*>(this + 1); }
    };
    static_assert(sizeof(Interned) == 16);

    Key() = default;
    explicit Key(std::string_view bytes) {
        const size_t n = bytes.size();
        if (n > 9) {
            bits_ = intern(bytes);
            return;
        }
        if (n >= 8) {
            uint64_t word;
            std::memcpy(&word, bytes.data(), 8);
            constexpr uint64_t high = UINT64_C(0x8080808080808080);
            constexpr uint64_t ones = UINT64_C(0x0101010101010101);
            const unsigned tail = n == 9 ? static_cast<unsigned char>(bytes[8]) : 0;
            if ((word & high) || ((word - ones) & ~word & high) || tail >= 128 ||
                (n == 9 && tail == 0)) {
                bits_ = intern(bytes);
                return;
            }
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
            word = __builtin_bswap64(word);
#endif
#if defined(__BMI2__)
            bits_ = (_pext_u64(word, ~high) << 7) | tail;
#else
            // Remove each high bit in three stages, merging adjacent units.
            word = (word & UINT64_C(0x007f007f007f007f)) |
                   ((word >> 1) & UINT64_C(0x3f803f803f803f80));
            word = (word & UINT64_C(0x00003fff00003fff)) |
                   ((word >> 2) & UINT64_C(0x0fffc0000fffc000));
            bits_ = (((word >> 4) & UINT64_C(0x00fffffff0000000)) |
                     (word & UINT64_C(0x000000000fffffff))) << 7 | tail;
#endif
            return;
        }
        uint64_t value = 0;
        for (unsigned char c : bytes) {
            if (c == 0 || c >= 128) {
                bits_ = intern(bytes);
                return;
            }
            value = (value << 7) | c;
        }
        bits_ = value << ((9 - n) * 7);
    }

    static Key from_packed_ascii(uint64_t bits) {
        Key key;
        key.bits_ = bits;
        return key;
    }
    uint64_t bits() const { return bits_; }
    uint64_t hash() const {
#if defined(__SSE4_2__)
        // CRC32C distributes the packed ASCII digits across shard bits.
        // Multiplication spreads it across the high table-index bits too.
        return _mm_crc32_u64(0, bits_) * UINT64_C(0xd6e8feb86659fd93) ^ bits_;
#else
        uint64_t x = bits_ ^ (bits_ >> 32);
        x *= UINT64_C(0xd6e8feb86659fd93);
        x ^= x >> 32;
        x *= UINT64_C(0xd6e8feb86659fd93);
        return x ^ (x >> 32);
#endif
    }
    bool operator==(const Key&) const = default;
    bool operator<(Key rhs) const {
        if (((bits_ | rhs.bits_) & tag) == 0) return bits_ < rhs.bits_;
        char a[9], b[9];
        return view(a) < rhs.view(b);
    }
    std::string_view view(char (&storage)[9]) const {
        if (bits_ & tag) {
            const auto* entry = reinterpret_cast<const Interned*>(bits_ & ~tag);
            return {entry->data(), entry->size};
        }
        size_t size = 0;
        uint64_t x = bits_;
        while (x != 0) {
            storage[size++] = static_cast<char>(x >> 56);
            x = (x << 7) & ~tag;
        }
        return {storage, size};
    }
    std::string str() const {
        char storage[9];
        return std::string(view(storage));
    }
private:
    static constexpr uint64_t tag = UINT64_C(1) << 63;
    static uint64_t intern(std::string_view);
    uint64_t bits_ = 0;
};

struct KeyHash {
    size_t operator()(Key k) const { return k.hash(); }
};

}

template <>
struct std::hash<shopping::Key> : shopping::KeyHash {};
