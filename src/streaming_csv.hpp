// Included after the scalar and SIMD structural classifiers in input_detail.
#if defined(__AVX512BW__)
template<class Decoder>
size_t stream_event_fields(std::string_view part, Decoder& decoder) {
    constexpr unsigned columns = 7, lanes = 8, packet = columns * lanes;
    constexpr unsigned high_water = 4 * packet;
    alignas(64) std::array<uint16_t, high_water + 64 + 32> ends;
    const char* bytes = part.data();
    unsigned first = 0, buffered = 0;
    uint64_t separator_before = 1, quote_before = 0, parity_before = 0;
    static constexpr auto expected_lfs = [] {
        std::array<uint64_t, columns> pattern{};
        for (unsigned phase = 0; phase < columns; ++phase)
            for (unsigned bit = columns - phase - 1; bit < 64; bit += columns)
                pattern[phase] |= uint64_t{1} << bit;
        return pattern;
    }();
#if defined(__AVX512VBMI2__)
    static constexpr auto ranks = [] {
        std::array<uint16_t, 32> values{};
        for (unsigned i = 0; i < values.size(); ++i) values[i] = i;
        return values;
    }();
#endif
    // Keep the next byte available for a quote at the end of a vector. The
    // original decoder handles the final few rows and arbitrarily long rows.
    for (unsigned offset = 0; offset + 64 < part.size(); offset += 64) {
        const auto m = masks64(bytes + offset);
        const uint64_t d = m.delimiters;
        const unsigned count = std::popcount(d);
        const char next = bytes[offset + 64];
        const uint64_t neighbor = (d << 1) | (d >> 1) | separator_before |
                                  (uint64_t(next == ',' || next == '\n') << 63);
        const auto parity = quote_parity(m.quotes) ^ parity_before;
        if ((m.quotes & ~neighbor) | (parity & d)) break;
        if (_pext_u64(m.newlines, d) != _bzhi_u64(expected_lfs[buffered % columns], count)) break;
        separator_before = d >> 63;
        parity_before = uint64_t{0} - (parity >> 63);
        const uint64_t closing = (m.quotes << 1) | quote_before;
        quote_before = m.quotes >> 63;
#if defined(__AVX512VBMI2__)
        const auto at = _mm512_add_epi16(_mm512_loadu_si512(ranks.data()), _mm512_set1_epi16(offset));
        for (unsigned half = 0; half < 64; half += 32) {
            const auto bits = static_cast<uint32_t>(d >> half);
            const auto pos = _mm512_add_epi16(at, _mm512_set1_epi16(half));
            const auto tagged = _mm512_mask_add_epi16(pos, static_cast<uint32_t>(closing >> half), pos,
                                                      _mm512_set1_epi16(INT16_MIN));
            _mm512_storeu_si512(ends.data() + buffered, _mm512_maskz_compress_epi16(bits, tagged));
            buffered += std::popcount(bits);
        }
#else
        if (count <= 8) {
            auto words = _mm_cvtepu8_epi16(_mm_cvtsi64_si128(sparse_delimiters(d, closing)));
            words = _mm_or_si128(_mm_srli_epi16(words, 1), _mm_slli_epi16(words, 15));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(ends.data() + buffered),
                             _mm_add_epi16(words, _mm_set1_epi16(offset)));
            buffered += count;
        } else {
            for (auto bits = d; bits; bits &= bits - 1) {
                const auto pos = std::countr_zero(bits);
                ends[buffered++] = (offset + pos) | (((closing >> pos) & 1) << 15);
            }
        }
#endif
        if (buffered < high_water) continue;
        unsigned consumed = 0;
        while (consumed + packet <= buffered) {
            const unsigned last = ends[consumed + packet - 1] & 32767;
            // The projection kernel fetches a whole machine word for short
            // fields. Near the end of the mapping use the bounded row decoder.
            if (last + 32 >= part.size()) break;
            decoder.eight(bytes, ends.data() + consumed, first);
            consumed += packet;
            first = last + 1;
        }
        buffered -= consumed;
        std::memmove(ends.data(), ends.data() + consumed, buffered * sizeof(ends[0]));
        if (buffered >= high_water) break;
    }
    return first;
}
#endif
