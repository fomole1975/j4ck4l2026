#pragma once

#if defined(__AVX2__)
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace shopping::event_sort {

// In a bitonic sorting stage each lane compares with its partner. Exactly
// one of the two lanes keeps the higher value.
template<unsigned group, unsigned step>
constexpr bool take_higher(unsigned lane) {
    return bool(lane & group) != bool(lane & step);
}

#if defined(__AVX512F__)
using Vector = __m512i;
constexpr size_t width = 8;

template<unsigned group, unsigned step>
inline Vector exchange(Vector values) {
    Vector partners;
    if constexpr (step == 1)
        partners = _mm512_permutex_epi64(values, _MM_SHUFFLE(2, 3, 0, 1));
    else if constexpr (step == 2)
        partners = _mm512_permutex_epi64(values, _MM_SHUFFLE(1, 0, 3, 2));
    else
        partners = _mm512_shuffle_i64x2(values, values, _MM_SHUFFLE(1, 0, 3, 2));

    constexpr __mmask8 upper = [] {
        unsigned mask = 0;
        for (unsigned i = 0; i < width; ++i)
            if (take_higher<group, step>(i)) mask |= 1U << i;
        return static_cast<__mmask8>(mask);
    }();
    return _mm512_mask_max_epu64(_mm512_min_epu64(values, partners), upper, values, partners);
}
inline Vector merge_one(Vector v) {
    return exchange<8, 1>(exchange<8, 2>(exchange<8, 4>(v)));
}
inline Vector sort_one(Vector v) {
    return merge_one(exchange<4, 1>(exchange<4, 2>(exchange<2, 1>(v))));
}
inline void exchange_pair(Vector& a, Vector& b) {
    const auto low = _mm512_min_epu64(a, b);
    b = _mm512_max_epu64(a, b);
    a = low;
}
inline Vector reverse(Vector v) {
    return _mm512_permutexvar_epi64(_mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0), v);
}
inline Vector load(const uint64_t* rows, size_t n, size_t begin) {
    if (n >= begin + width) return _mm512_loadu_si512(rows + begin);
    const auto pad = _mm512_set1_epi64(-1);
    if (n <= begin) return pad;
    return _mm512_mask_loadu_epi64(pad, (1U << (n - begin)) - 1, rows + begin);
}
inline void store(uint64_t* rows, size_t n, size_t begin, Vector values) {
    if (n >= begin + width)
        _mm512_storeu_si512(rows + begin, values);
    else if (n > begin)
        _mm512_mask_storeu_epi64(rows + begin, (1U << (n - begin)) - 1, values);
}
#else
using Vector = __m256i;
constexpr size_t width = 4;

template<unsigned group, unsigned step>
inline Vector exchange(Vector values) {
    const auto partners = _mm256_permute4x64_epi64(values, step == 1 ? _MM_SHUFFLE(2, 3, 0, 1)
                                                                   : _MM_SHUFFLE(1, 0, 3, 2));
    constexpr auto upper = [](unsigned i) -> int64_t { return take_higher<group, step>(i) ? -1 : 0; };
    const auto mask = _mm256_xor_si256(_mm256_cmpgt_epi64(values, partners),
                                       _mm256_setr_epi64x(upper(0), upper(1), upper(2), upper(3)));
    return _mm256_blendv_epi8(values, partners, mask);
}
inline Vector merge_one(Vector v) { return exchange<4, 1>(exchange<4, 2>(v)); }
inline Vector sort_one(Vector v) { return merge_one(exchange<2, 1>(v)); }
inline void exchange_pair(Vector& a, Vector& b) {
    const auto mask = _mm256_cmpgt_epi64(a, b);
    const auto low = _mm256_blendv_epi8(a, b, mask);
    b = _mm256_blendv_epi8(b, a, mask);
    a = low;
}
inline Vector reverse(Vector v) { return _mm256_permute4x64_epi64(v, _MM_SHUFFLE(0, 1, 2, 3)); }
inline Vector valid_lanes(size_t n) {
    return _mm256_cmpgt_epi64(_mm256_set1_epi64x(n), _mm256_setr_epi64x(0, 1, 2, 3));
}
inline Vector load(const uint64_t* rows, size_t n, size_t begin) {
    // AVX2 compares signed 64-bit lanes. Flip the sign bit at the boundaries.
    const auto sign = _mm256_set1_epi64x(INT64_MIN);
    if (n >= begin + width)
        return _mm256_xor_si256(sign, _mm256_loadu_si256(reinterpret_cast<const Vector*>(rows + begin)));
    const auto pad = _mm256_set1_epi64x(INT64_MAX);
    if (n <= begin) return pad;
    const auto mask = valid_lanes(n - begin);
    const auto tail = _mm256_maskload_epi64(reinterpret_cast<const long long*>(rows + begin), mask);
    return _mm256_blendv_epi8(pad, _mm256_xor_si256(sign, tail), mask);
}
inline void store(uint64_t* rows, size_t n, size_t begin, Vector values) {
    const auto unsigned_values = _mm256_xor_si256(values, _mm256_set1_epi64x(INT64_MIN));
    if (n >= begin + width)
        _mm256_storeu_si256(reinterpret_cast<Vector*>(rows + begin), unsigned_values);
    else if (n > begin)
        _mm256_maskstore_epi64(reinterpret_cast<long long*>(rows + begin), valid_lanes(n - begin), unsigned_values);
}
#endif

template<size_t registers>
inline void merge(Vector* block) {
    if constexpr (registers == 1) {
        block[0] = merge_one(block[0]);
    } else {
        constexpr size_t half = registers / 2;
        for (size_t i = 0; i < half; ++i)
            exchange_pair(block[i], block[i + half]);
        merge<half>(block);
        merge<half>(block + half);
    }
}

template<size_t registers>
inline void sort_registers(Vector* block) {
    if constexpr (registers == 1) {
        block[0] = sort_one(block[0]);
    } else {
        constexpr size_t half = registers / 2;
        sort_registers<half>(block);
        sort_registers<half>(block + half);
        // Reverse the second sorted half to create a bitonic sequence.
        for (size_t i = 0; i < half / 2; ++i)
            std::swap(block[half + i], block[registers - 1 - i]);
        for (size_t i = half; i < registers; ++i)
            block[i] = reverse(block[i]);
        merge<registers>(block);
    }
}

template<size_t registers>
inline void sort_small(uint64_t* rows, size_t n) {
    Vector block[registers];
    for (size_t i = 0; i < registers; ++i)
        block[i] = load(rows, n, width * i);
    sort_registers<registers>(block);
    for (size_t i = 0; i < registers; ++i)
        store(rows, n, width * i, block[i]);
}

#if defined(__AVX512F__)
// A compact click is two 64-bit words. Sort timestamp and original-position
// keys, then permute both original words together so the click ID survives ties.
template<size_t registers, class Click>
inline void sort_clicks(Click* rows, size_t n) {
    static_assert(sizeof(Click) == 16);
    __m512i fields[registers], ids[registers], keys[registers];
    const auto even = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    const auto odd = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
    const auto lanes = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    const auto pad = _mm512_set1_epi64(-1);
    auto load_four = [&](size_t i) {
        if (n >= i + 4) return _mm512_loadu_si512(rows + i);
        if (n <= i) return _mm512_setzero_si512();
        return _mm512_maskz_loadu_epi64((1U << (2 * (n - i))) - 1, rows + i);
    };
    for (size_t k = 0; k < registers; ++k) {
        const size_t at = 8 * k;
        const auto left = load_four(at), right = load_four(at + 4);
        fields[k] = _mm512_permutex2var_epi64(left, even, right);
        ids[k] = _mm512_permutex2var_epi64(left, odd, right);
        const auto indices = _mm512_add_epi64(lanes, _mm512_set1_epi64(at));
        const auto key = _mm512_or_si512(_mm512_slli_epi64(fields[k], 32), indices);
        if (n >= at + 8) keys[k] = key;
        else if (n > at) keys[k] = _mm512_mask_mov_epi64(pad, (1U << (n - at)) - 1, key);
        else keys[k] = pad;
    }
    sort_registers<registers>(keys);
    auto store_four = [&](size_t i, __m512i values) {
        if (n >= i + 4) _mm512_storeu_si512(rows + i, values);
        else if (n > i) _mm512_mask_storeu_epi64(rows + i, (1U << (2 * (n - i))) - 1, values);
    };
    for (size_t k = 0; k < registers && 8 * k < n; ++k) {
        const auto order = keys[k];
        __m512i data, id;
        if constexpr (registers == 1) {
            data = _mm512_permutexvar_epi64(order, fields[0]);
            id = _mm512_permutexvar_epi64(order, ids[0]);
        } else {
            data = _mm512_permutex2var_epi64(fields[0], order, fields[1]);
            id = _mm512_permutex2var_epi64(ids[0], order, ids[1]);
            for (size_t j = 2; j < registers; j += 2) {
                // A two-register permutation consumes the low four index bits.
                const auto pair = _mm512_and_si512(order, _mm512_set1_epi64((8 * registers - 1) & ~15U));
                const auto select = _mm512_cmpeq_epi64_mask(pair, _mm512_set1_epi64(8 * j));
                data = _mm512_mask_mov_epi64(data, select, _mm512_permutex2var_epi64(fields[j], order, fields[j + 1]));
                id = _mm512_mask_mov_epi64(id, select, _mm512_permutex2var_epi64(ids[j], order, ids[j + 1]));
            }
        }
        store_four(8 * k, _mm512_permutex2var_epi64(data, _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11), id));
        store_four(8 * k + 4, _mm512_permutex2var_epi64(data, _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15), id));
    }
}
#endif

} // namespace shopping::event_sort
#endif
