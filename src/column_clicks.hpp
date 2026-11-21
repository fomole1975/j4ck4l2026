// Four independent clicks share each date, clock, ID and campaign operation.
#if defined(__AVX2__) && defined(__BMI2__)
namespace input_detail {
template <class Callback>
void ClickDecoder<Callback>::four(const char *base, const uint16_t *ends, unsigned first) {
    auto load = [&](unsigned j) { return _mm_loadl_epi64(reinterpret_cast<const __m128i *>(ends + j * 4)); };
    const auto a = _mm_unpacklo_epi16(load(0), load(1)), b = _mm_unpacklo_epi16(load(2), load(3));
    const auto ab = _mm_unpacklo_epi32(a, b), cd = _mm_unpackhi_epi32(a, b);
    const auto e0 = _mm_cvtepu16_epi32(ab), e1 = _mm_cvtepu16_epi32(_mm_srli_si128(ab, 8));
    const auto e2 = _mm_cvtepu16_epi32(cd), e3 = _mm_cvtepu16_epi32(_mm_srli_si128(cd, 8));
    const auto one = _mm_set1_epi32(1);
#if defined(__AVX512BW__)
    const auto pos = [](__m128i v) { return _mm_and_si128(v, _mm_set1_epi32(32767)); };
    const auto quoted = [](__m128i v) { return _mm_srli_epi32(v, 15); };
    const auto spos = [](unsigned e) { return e & 32767; };
    const auto sq = [](unsigned e) { return e >> 15; };
#else
    const auto pos = [](__m128i v) { return _mm_srli_epi32(v, 1); };
    const auto quoted = [&](__m128i v) { return _mm_and_si128(v, one); };
    const auto spos = [](unsigned e) { return e >> 1; };
    const auto sq = [](unsigned e) { return e & 1; };
#endif
    const auto begin = [&](__m128i prev, __m128i end) {
        return _mm_add_epi32(_mm_add_epi32(pos(prev), one), quoted(end));
    };
    const auto size = [&](__m128i end, __m128i s) {
        return _mm_sub_epi32(_mm_sub_epi32(pos(end), quoted(end)), s);
    };
    const auto get = [&](__m128i p, int extra) {
        return _mm256_i32gather_epi64(reinterpret_cast<const long long *>(base),
                                      _mm_add_epi32(p, _mm_set1_epi32(extra)), 1);
    };
    const auto t = begin(e2, e3), n = size(e3, t), iso = _mm_cmpeq_epi32(n, _mm_set1_epi32(20)),
               ep = _mm_cmpeq_epi32(n, _mm_set1_epi32(13));
    const auto safe = _mm_and_si128(_mm_or_si128(iso, ep), t);
    const auto prefix = get(safe, 0), middle = get(safe, 8), suffix = get(_mm_and_si128(iso, t), 12);
    static thread_local Calendar4 dates;
    const auto start = w.start - 7 * day_ms;
    dates.reset(start);
    Vec4 good_day, good_clock;
    const auto date = dates.day(prefix, middle, good_day);
    const auto clock =
        clock4(blend4(_mm256_slli_epi64(suffix, 8), _mm256_srli_epi64(middle, 24), v4(255)), good_clock);
    const auto known_iso = _mm256_cvtepi32_epi64(iso) & good_day;
    const auto zulu = eq4(middle & v4(0xff0000), v4(0x540000)) & eq4(_mm256_srli_epi64(suffix, 56), v4('Z'));
    auto times = blend4(v4(bad_time), _mm256_add_epi64(date, clock), known_iso & good_clock & zulu);
    const auto epoch = _mm256_cvtepi32_epi64(ep);
    if (lane4(epoch)) {
        auto head = _mm256_sub_epi8(prefix, _mm256_set1_epi8('0'));
        auto tail =
            _mm256_sub_epi8(_mm256_or_si256(_mm256_srli_epi64(prefix, 40), _mm256_slli_epi64(middle, 24)),
                            _mm256_set1_epi8('0'));
        const auto over = _mm256_subs_epu8(_mm256_max_epu8(head, tail), _mm256_set1_epi8(9));
        const auto milliseconds = _mm256_add_epi64(
            _mm256_mul_epu32(decimal4(_mm256_slli_epi64(head, 24)), v4(100000000)), decimal4(tail));
        times = blend4(times, milliseconds, epoch & eq4(over, v4(0)));
    }
    const unsigned known = lane4(known_iso | epoch),
                   in_window = lane4(~gt4(v4(start), times) & gt4(v4(w.end), times));
    alignas(32) int64_t ms[4];
    _mm256_store_si256(reinterpret_cast<Vec4 *>(ms), times);
    alignas(32) uint64_t ids[4], users[4], slots[4];
    unsigned projected = 0;
    if constexpr (requires { emit.indexed(uint64_t{}, uint64_t{}, uint32_t{}, int64_t{}); }) {
        const auto prev = _mm_insert_epi32(_mm_slli_si128(pos(e3), 4), int(first) - 1, 0);
        const auto is = _mm_add_epi32(_mm_add_epi32(prev, one), quoted(e0)), us = begin(e0, e1),
                   cs = begin(e1, e2), cn = size(e2, cs);
        auto key = [&](__m128i offset, __m128i len) {
            const auto eight = _mm_cmpeq_epi32(len, _mm_set1_epi32(8)),
                       nine = _mm_cmpeq_epi32(len, _mm_set1_epi32(9));
            const auto safe = _mm_and_si128(_mm_or_si128(eight, nine), offset);
            const auto raw = get(safe, 0), ninth = _mm256_cvtepi32_epi64(nine);
            const auto tail = ninth & _mm256_srli_epi64(get(safe, 1), 56);
            const auto ok = _mm256_cvtepi32_epi64(_mm_or_si128(eight, nine)) &
                            eq4(_mm256_cmpgt_epi8(raw, v4(0)), v4(-1)) &
                            ~(ninth & (eq4(tail, v4(0)) | gt4(tail, v4(127))));
            const auto pair = _mm256_maddubs_epi16(_mm256_set1_epi16(0x0180), raw);
            const auto four = _mm256_madd_epi16(pair, _mm256_set1_epi32(0x00014000));
            return (_mm256_slli_epi64(
                        _mm256_slli_epi64(four & v4(0x0fffffff), 28) | _mm256_srli_epi64(four, 32), 7) |
                    tail) &
                   ok;
        };
        const auto id = key(is, size(e0, is)), user = key(us, size(e1, us));
        // Only the catalog's canonical c_<decimal> spelling can enter the dense
        // lookup. A leading zero, missing entry outside that range, and every
        // other spelling retain the original bytewise sparse lookup.
        const auto ordinary =
            _mm_and_si128(_mm_cmpgt_epi32(cn, _mm_set1_epi32(2)), _mm_cmpgt_epi32(_mm_set1_epi32(9), cn));
        const auto raw = get(_mm_and_si128(ordinary, cs), 0);
        const auto clen = _mm256_cvtepi32_epi64(cn),
                   shift = _mm256_slli_epi64(_mm256_sub_epi64(v4(10), clen), 3);
        const auto digits =
            _mm256_sllv_epi64(_mm256_sub_epi8(_mm256_srli_epi64(raw, 16), _mm256_set1_epi8('0')), shift);
        const auto cgood = _mm256_cvtepi32_epi64(ordinary) & eq4(raw & v4(65535), v4(0x5f63)) &
                           eq4(_mm256_subs_epu8(digits, _mm256_set1_epi8(9)), v4(0)) &
                           (eq4(clen, v4(3)) | ~eq4(raw & v4(0xff0000), v4(0x300000)));
        projected = lane4(cgood & ~eq4(id, v4(0)) & ~eq4(user, v4(0)));
        _mm256_store_si256(reinterpret_cast<Vec4 *>(ids), id);
        _mm256_store_si256(reinterpret_cast<Vec4 *>(users), user);
        _mm256_store_si256(reinterpret_cast<Vec4 *>(slots), decimal4(digits));
    }
    for (unsigned j = 0; j < 4; ++j) {
        const unsigned bit = 1u << j;
        const auto *d = ends + 4 * j;
        unsigned row_start = first;
        first = spos(d[3]) + 1;
        if ((known & bit) && !(in_window & bit)) continue;
        auto field = [&](unsigned k) {
            auto beg = (k ? spos(d[k - 1]) + 1 : row_start) + sq(d[k]);
            return std::string_view(base + beg, spos(d[k]) - sq(d[k]) - beg);
        };
        auto ts = ms[j];
        if (!(known & bit)) {
            ts = timestamp(field(3));
            if (ts < start || ts >= w.end) continue;
        }
        if constexpr (requires { emit.indexed(uint64_t{}, uint64_t{}, uint32_t{}, int64_t{}); }) {
            if ((projected & bit) && emit.indexed(ids[j], users[j], slots[j], ts)) continue;
        }
        const auto id = field(0), u = field(1), c = field(2);
        if (!id.empty() && !u.empty() && !c.empty()) emit(ClickInput{id, u, c, ts});
    }
}
} // namespace input_detail
#endif
