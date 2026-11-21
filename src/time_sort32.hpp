#pragma once
#if defined(__AVX2__)
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace shopping::time_sort32 {
#if defined(__AVX512F__)
using Vector = __m512i;
constexpr size_t width = 16;
#else
using Vector = __m256i;
constexpr size_t width = 8;
#endif

template<unsigned Group, unsigned Step>
inline Vector exchange(Vector v) {
    constexpr unsigned upper = [] {
        unsigned bits = 0;
        for (unsigned i = 0; i < width; ++i)
            if (bool(i & Group) != bool(i & Step)) bits |= 1u << i;
        return bits;
    }();
#if defined(__AVX512F__)
    Vector p;
    if constexpr (Step == 1) p = _mm512_shuffle_epi32(v, static_cast<_MM_PERM_ENUM>(_MM_SHUFFLE(2,3,0,1)));
    else if constexpr (Step == 2) p = _mm512_shuffle_epi32(v, static_cast<_MM_PERM_ENUM>(_MM_SHUFFLE(1,0,3,2)));
    else if constexpr (Step == 4) p = _mm512_shuffle_i32x4(v, v, _MM_SHUFFLE(2,3,0,1));
    else p = _mm512_shuffle_i32x4(v, v, _MM_SHUFFLE(1,0,3,2));
    return _mm512_mask_max_epu32(_mm512_min_epu32(v, p), upper, v, p);
#else
    Vector p;
    if constexpr (Step == 1) p = _mm256_shuffle_epi32(v, _MM_SHUFFLE(2,3,0,1));
    else if constexpr (Step == 2) p = _mm256_shuffle_epi32(v, _MM_SHUFFLE(1,0,3,2));
    else p = _mm256_permute2x128_si256(v, v, 1);
    return _mm256_blend_epi32(_mm256_min_epu32(v,p), _mm256_max_epu32(v,p), upper);
#endif
}
inline Vector merge_one(Vector v) {
#if defined(__AVX512F__)
    v = exchange<16,8>(v);
#endif
    return exchange<width,1>(exchange<width,2>(exchange<width,4>(v)));
}
inline Vector sort_one(Vector v) {
    v = exchange<4,1>(exchange<4,2>(exchange<2,1>(v)));
#if defined(__AVX512F__)
    v = exchange<8,1>(exchange<8,2>(exchange<8,4>(v)));
#endif
    return merge_one(v);
}
inline Vector reverse(Vector v) {
#if defined(__AVX512F__)
    return _mm512_permutexvar_epi32(_mm512_setr_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0),v);
#else
    return _mm256_permutevar8x32_epi32(v,_mm256_setr_epi32(7,6,5,4,3,2,1,0));
#endif
}
inline void split(Vector& a, Vector& b) {
#if defined(__AVX512F__)
    auto low = _mm512_min_epu32(a,b); b = _mm512_max_epu32(a,b); a = low;
#else
    auto low = _mm256_min_epu32(a,b); b = _mm256_max_epu32(a,b); a = low;
#endif
}
template<size_t N> inline void merge(Vector* v) {
    if constexpr (N == 1) v[0] = merge_one(v[0]);
    else {
        constexpr size_t half = N/2;
        for (size_t i=0;i<half;++i) split(v[i],v[i+half]);
        merge<half>(v); merge<half>(v+half);
    }
}
template<size_t N> inline void sort_registers(Vector* v) {
    if constexpr (N == 1) v[0] = sort_one(v[0]);
    else {
        constexpr size_t half = N/2;
        sort_registers<half>(v); sort_registers<half>(v+half);
        for (size_t i=0;i<half/2;++i) std::swap(v[half+i],v[N-1-i]);
        for (size_t i=half;i<N;++i) v[i]=reverse(v[i]);
        merge<N>(v);
    }
}
template<size_t N> inline void sort(uint32_t* data, size_t n) {
    Vector v[N];
#if defined(__AVX512F__)
    const auto pad = _mm512_set1_epi32(-1);
#else
    const auto pad = _mm256_set1_epi32(-1);
    const auto lanes = _mm256_setr_epi32(0,1,2,3,4,5,6,7);
#endif
    for (size_t i=0;i<N;++i) {
        const size_t at=i*width;
        if (n <= at) {v[i]=pad;continue;}
#if defined(__AVX512F__)
        if (n>=at+width) v[i]=_mm512_loadu_si512(data+at);
        else v[i]=_mm512_mask_loadu_epi32(pad, (1u<<(n-at))-1, data+at);
#else
        if (n>=at+width) v[i]=_mm256_loadu_si256(reinterpret_cast<const Vector*>(data+at));
        else {
            auto live=_mm256_cmpgt_epi32(_mm256_set1_epi32(n-at), lanes);
            v[i]=_mm256_blendv_epi8(pad,_mm256_maskload_epi32(reinterpret_cast<const int*>(data+at),live),live);
        }
#endif
    }
    sort_registers<N>(v);
    for (size_t i=0;i<N && i*width<n;++i) {
        const size_t at=i*width;
#if defined(__AVX512F__)
        if (n>=at+width) _mm512_storeu_si512(data+at,v[i]);
        else _mm512_mask_storeu_epi32(data+at,(1u<<(n-at))-1,v[i]);
#else
        if (n>=at+width) _mm256_storeu_si256(reinterpret_cast<Vector*>(data+at),v[i]);
        else _mm256_maskstore_epi32(reinterpret_cast<int*>(data+at),_mm256_cmpgt_epi32(_mm256_set1_epi32(n-at),lanes),v[i]);
#endif
    }
}
}
#endif
