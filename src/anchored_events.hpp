#if defined(__AVX2__) && defined(__BMI2__)
namespace input_detail {

// A stop occupies one byte: its coordinate modulo 128 and the quoted-field flag.
// A checked newline span makes each residue unambiguous relative to its row.
template <class Decoder> size_t anchor_event_batch(std::string_view part, Decoder &decoder) {
    constexpr unsigned capacity = 8192;
    if (part.size() < 512 || part.size() > capacity)
        return 0;
    alignas(64) std::array<unsigned char, capacity + 64> stops;
    alignas(64) std::array<uint16_t, capacity + 1> origins;
    origins[0] = 0;
    const char *base = part.data();
    unsigned tokens = 0, lines = 0;
    uint64_t prev_sep = 1, prev_quote = 0, prev_parity = 0;
#if defined(__AVX512VBMI2__)
    alignas(64) static constexpr auto ranks = [] {
        std::array<unsigned char, 64> v{};
        for (unsigned i = 0; i < v.size(); ++i)
            v[i] = i;
        return v;
    }();
#endif
    for (unsigned off = 0; off + 64 < part.size(); off += 64) {
        const auto m = masks64(base + off);
        const uint64_t d = m.delimiters;
        const char next = base[off + 64];
        const uint64_t near = (d << 1) | (d >> 1) | prev_sep | (uint64_t(next == ',' || next == '\n') << 63);
        const uint64_t parity = quote_parity(m.quotes) ^ prev_parity;
        if ((m.quotes & ~near) | (parity & d))
            break;
        const unsigned count = std::popcount(d);
        const uint64_t closing = (m.quotes << 1) | prev_quote;
        prev_quote = m.quotes >> 63;
        prev_sep = d >> 63;
        prev_parity = uint64_t{0} - (parity >> 63);
#if defined(__AVX512VBMI2__)
        const auto at = _mm512_add_epi8(_mm512_load_si512(ranks.data()), _mm512_set1_epi8(off & 64));
        const auto marked = _mm512_mask_add_epi8(at, closing, at, _mm512_set1_epi8(-128));
        _mm512_storeu_si512(stops.data() + tokens, _mm512_maskz_compress_epi8(d, marked));
#else
        if (count <= 8) {
            const uint64_t positions = sparse_delimiters(d, closing);
            constexpr uint64_t ones = 0x0101010101010101ULL;
            const uint64_t code =
                ((positions >> 1) & (63 * ones)) | ((positions & ones) << 7) | ((off & 64) * ones);
            std::memcpy(stops.data() + tokens, &code, 8);
        } else {
            unsigned j = 0;
            for (uint64_t bits = d; bits; bits &= bits - 1) {
                const auto pos = std::countr_zero(bits);
                stops[tokens + j++] = ((off + pos) & 127) | (((closing >> pos) & 1) << 7);
            }
        }
#endif
        tokens += count;
        auto lfs = m.newlines;
        while (lfs) {
            origins[++lines] = off + std::countr_zero(lfs) + 1;
            lfs &= lfs - 1;
        }
    }
    unsigned row = 0;
#if defined(__AVX512BW__)
    const auto spread_origin = [](__m512i positions) {
        return _mm512_shuffle_epi8(positions,
                                   _mm512_set_epi64(0x0808080808080808ULL, 0, 0x0808080808080808ULL, 0,
                                                    0x0808080808080808ULL, 0, 0x0808080808080808ULL, 0));
    };
    // In a span of at most 127 bytes, residues identify positions uniquely.
    // The first premature LF would equal the true endpoint in columns 0..5.
    // An extra field would put column 6 strictly before the true endpoint.
    // These comparisons therefore certify all eight individual field counts.
    for (; row + 8 <= lines && 7 * (row + 8) <= tokens && origins[row + 8] + 32 < part.size(); row += 8) {
        const auto origins8 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(origins.data() + row));
        const auto next8 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(origins.data() + row + 1));
        if (_mm_movemask_epi8(_mm_cmpgt_epi16(_mm_sub_epi16(next8, origins8), _mm_set1_epi16(127))))
            break;
        const auto origin = _mm512_cvtepu16_epi64(origins8);
#if defined(__AVX512VBMI2__)
        const auto codes =
            _mm512_maskz_expand_epi8(0x7f7f7f7f7f7f7f7fULL, _mm512_loadu_si512(stops.data() + 7 * row));
#else
        // Two seven-byte records occupy each independent 128-bit shuffle lane.
        const auto *bytes = stops.data() + 7 * row;
        const auto load = [&](unsigned offset) {
            return _mm_loadu_si128(reinterpret_cast<const __m128i *>(bytes + offset));
        };
        auto rows = _mm512_castsi128_si512(load(0));
        rows = _mm512_inserti32x4(rows, load(14), 1);
        rows = _mm512_inserti32x4(rows, load(28), 2);
        rows = _mm512_inserti32x4(rows, load(42), 3);
        const auto order = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, -1, 7, 8, 9, 10, 11, 12, 13, -1);
        const auto codes = _mm512_shuffle_epi8(rows, _mm512_broadcast_i32x4(order));
#endif
        const auto lf = _mm512_cvtepu16_epi64(_mm_sub_epi16(next8, _mm_set1_epi16(1)));
        const auto equal = _mm512_cmpeq_epi8_mask((codes ^ spread_origin(lf)) & _mm512_set1_epi8(127),
                                                  _mm512_setzero_si512());
        if ((equal & 0x7f7f7f7f7f7f7f7fULL) != 0x4040404040404040ULL)
            break;
        const auto ends = _mm512_sub_epi8(codes, spread_origin(origin)) & _mm512_set1_epi8(127);
        const auto quoted = _mm512_srli_epi16(codes, 7) & _mm512_set1_epi8(1);
        const auto after =
            _mm512_add_epi8(_mm512_slli_epi64(ends, 8), _mm512_set1_epi64(0x0101010101010100ULL));
        const auto begins = _mm512_add_epi8(after, quoted);
        const auto widths = _mm512_sub_epi8(_mm512_sub_epi8(ends, quoted), begins);
        alignas(64) unsigned char begin_bytes[8][8], width_bytes[8][8];
        _mm512_store_si512(begin_bytes, begins);
        _mm512_store_si512(width_bytes, widths);
        const auto fields = [&](unsigned j, unsigned k) {
            return std::string_view(base + origins[row + j] + begin_bytes[j][k], width_bytes[j][k]);
        };
        const auto spans = [&]<int C>() {
            const auto bytes = _mm512_set1_epi64(255);
            return EventField8{_mm512_add_epi64(origin, _mm512_srli_epi64(begins, 8 * C) & bytes),
                               _mm512_srli_epi64(widths, 8 * C) & bytes};
        };
        decoder.project8(base, spans, fields);
    }
#else
    const auto broadcast = [](__m256i v) {
        return _mm256_shuffle_epi8(v, _mm256_set_epi64x(0x0808080808080808ULL, 0, 0x0808080808080808ULL, 0));
    };
    const auto pattern =
        _mm256_broadcastsi128_si256(_mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, -1, 7, 8, 9, 10, 11, 12, 13, -1));
    const bool vector = __builtin_cpu_is("intel");
    for (; row + 4 <= lines && 7 * (row + 4) <= tokens; row += 4) {
        if (vector && origins[row + 4] + 32 >= part.size())
            break;
        const auto source = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(origins.data() + row));
        const auto next = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(origins.data() + row + 1));
        if (_mm_movemask_epi8(_mm_cmpgt_epi16(_mm_sub_epi16(next, source), _mm_set1_epi16(127))))
            break;
        const auto *stream = stops.data() + 7 * row;
        const auto low = _mm_loadu_si128(reinterpret_cast<const __m128i *>(stream));
        const auto high = _mm_loadu_si128(reinterpret_cast<const __m128i *>(stream + 14));
        const auto raw = _mm256_shuffle_epi8(_mm256_set_m128i(high, low), pattern);
        const auto lf = broadcast(_mm256_cvtepu16_epi64(_mm_sub_epi16(next, _mm_set1_epi16(1))));
        const auto res = _mm256_and_si256(_mm256_xor_si256(raw, lf), _mm256_set1_epi8(127));
        const unsigned match = _mm256_movemask_epi8(_mm256_cmpeq_epi8(res, _mm256_setzero_si256()));
        if ((match & 0x7f7f7f7f) != 0x40404040)
            break;
        const auto origin = broadcast(_mm256_cvtepu16_epi64(source));
        const auto ends = _mm256_and_si256(_mm256_sub_epi8(raw, origin), _mm256_set1_epi8(127));
        const auto q = _mm256_and_si256(_mm256_srli_epi16(raw, 7), _mm256_set1_epi8(1));
        const auto after =
            _mm256_add_epi8(_mm256_slli_epi64(ends, 8), _mm256_set1_epi64x(0x0101010101010100ULL));
        const auto begins = _mm256_add_epi8(after, q),
                   widths = _mm256_sub_epi8(_mm256_sub_epi8(ends, q), begins);
        alignas(32) unsigned char begin_bytes[4][8], width_bytes[4][8];
        _mm256_store_si256(reinterpret_cast<__m256i *>(begin_bytes), begins);
        _mm256_store_si256(reinterpret_cast<__m256i *>(width_bytes), widths);
        auto fields = [&](unsigned j, unsigned k) {
            return std::string_view(base + origins[row + j] + begin_bytes[j][k], width_bytes[j][k]);
        };
        if (vector) {
            const auto origin32 = _mm_cvtepu16_epi32(source);
            const auto select = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);
            auto spans = [&]<int C>() {
                const auto extract = [&](__m256i q) {
                    auto packed = _mm256_permutevar8x32_epi32(_mm256_srli_epi64(q, 8 * C), select);
                    return _mm_and_si128(_mm256_castsi256_si128(packed), _mm_set1_epi32(255));
                };
                return EventField4{_mm_add_epi32(origin32, extract(begins)), extract(widths)};
            };
            decoder.project4(base, spans, fields);
        } else {
            for (unsigned j = 0; j < 4; ++j) {
                std::array<std::string_view, 7> f;
                for (unsigned k = 0; k < 7; ++k)
                    f[k] = fields(j, k);
                decoder(f);
            }
        }
    }
#endif
    return origins[row];
}

template <class Decoder> void anchor_events(std::string_view source, Decoder &decoder) {
    auto ordinary = [&](std::string_view part) {
#if defined(__AVX512BW__)
        csv<7>(part, decoder);
#else
        auto scalar = [&](const auto &fields) { decoder(fields); };
        csv<7>(part, scalar);
#endif
    };
    // Stop on an eight- or four-row packet boundary, then include the tail in
    // the next scan window. Only an exceptional row or the file end requires
    // the ordinary decoder.
    while (source.size() >= 1024) {
        const auto consumed = anchor_event_batch(source.substr(0, 8192), decoder);
        if (consumed) {
            source.remove_prefix(consumed);
            continue;
        }
        const auto newline = source.find('\n', 511);
        const auto n = newline == source.npos ? source.size() : newline + 1;
        ordinary(source.substr(0, n));
        source.remove_prefix(n);
    }
    ordinary(source);
}
} // namespace input_detail
#endif
