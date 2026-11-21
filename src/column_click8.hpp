#if defined(__AVX512BW__)
namespace input_detail {
template <class Callback>
void ClickDecoder<Callback>::eight(const char *base, const uint16_t *ends, unsigned first) {
    // Four fields times eight clicks exactly fill a single cache line of
    // structural tokens. One 32-word permutation transposes the whole block.
    alignas(64) static constexpr auto transpose = [] {
        std::array<uint16_t, 32> r{};
        for (unsigned c = 0; c < 4; ++c)
            for (unsigned j = 0; j < 8; ++j)
                r[c * 8 + j] = j * 4 + c;
        return r;
    }();
    const auto table =
        _mm512_permutexvar_epi16(_mm512_load_si512(transpose.data()), _mm512_loadu_si512(ends));
    const auto e0 = _mm512_cvtepu16_epi64(_mm512_castsi512_si128(table)),
               e1 = _mm512_cvtepu16_epi64(_mm512_extracti32x4_epi32(table, 1));
    const auto e2 = _mm512_cvtepu16_epi64(_mm512_extracti32x4_epi32(table, 2)),
               e3 = _mm512_cvtepu16_epi64(_mm512_extracti32x4_epi32(table, 3));
    const auto v = [](int64_t x) { return _mm512_set1_epi64(x); };
    const auto pos = [&](__m512i x) { return x & v(32767); };
    const auto q = [](__m512i x) { return _mm512_srli_epi64(x, 15); };
    const auto begin = [&](__m512i prev, __m512i end) {
        return _mm512_add_epi64(_mm512_add_epi64(pos(prev), v(1)), q(end));
    };
    const auto size = [&](__m512i end, __m512i s) {
        return _mm512_sub_epi64(_mm512_sub_epi64(pos(end), q(end)), s);
    };
    const auto get = [&](__m512i p) { return _mm512_i64gather_epi64(p, base, 1); };
    const auto eq = [](__m512i x, __m512i y) { return _mm512_cmpeq_epi64_mask(x, y); };
    const auto t = begin(e2, e3), n = size(e3, t);
    const auto iso = eq(n, v(20)), tz = eq(n, v(25)), epoch = eq(n, v(13));
    const auto safe = _mm512_maskz_mov_epi64(iso | tz | epoch, t);
    const auto pref = get(safe), mid = get(_mm512_add_epi64(safe, v(8)));
    const auto suffix = get(_mm512_maskz_add_epi64(iso | tz, t, v(12)));
    const auto tw = _mm512_mask_blend_epi8(UINT64_C(0x0101010101010101), _mm512_slli_epi64(suffix, 8),
                                           _mm512_srli_epi64(mid, 24));
    static thread_local CalendarBlock dates;
    const auto start = w.start - 7 * day_ms;
    dates.reset(start);
    __mmask8 good_day, good_clock;
    const auto day = dates.day(pref, mid, good_day), clock = column_digit_clock(tw, good_clock);
    const auto known_iso = (iso | tz) & good_day;
    const auto letter = eq(mid & v(0xff0000), v(0x540000));
    const auto suffix_char = _mm512_srli_epi64(suffix, 56), ordinary = _mm512_add_epi64(day, clock);
    auto result = _mm512_mask_mov_epi64(
        v(bad_time), iso & good_day & good_clock & letter & eq(suffix_char, v('Z')), ordinary);
    result = apply_zone8(base, t, tz, good_day & good_clock & letter, suffix_char, ordinary, result);
    if (epoch) {
        const auto head = _mm512_sub_epi8(pref, _mm512_set1_epi8('0'));
        const auto tail = _mm512_sub_epi8(
            _mm512_or_si512(_mm512_srli_epi64(pref, 40), _mm512_slli_epi64(mid, 24)), _mm512_set1_epi8('0'));
        const auto over = _mm512_subs_epu8(_mm512_max_epu8(head, tail), _mm512_set1_epi8(9));
        result = _mm512_mask_mov_epi64(
            result, epoch & eq(over, v(0)),
            _mm512_add_epi64(_mm512_mul_epu32(column_decimal8(_mm512_slli_epi64(head, 24)), v(100000000)),
                             column_decimal8(tail)));
    }
    const auto known = known_iso | epoch;
    const auto active = _mm512_cmpge_epi64_mask(result, v(start)) & _mm512_cmplt_epi64_mask(result, v(w.end));
    alignas(64) int64_t ms[8];
    _mm512_store_si512(ms, result);
    alignas(64) uint64_t ids[8], users[8], slots[8];
    unsigned direct = 0;
    if constexpr (requires { emit.indexed(uint64_t{}, uint64_t{}, uint32_t{}, int64_t{}); }) {
        const auto prev = _mm512_mask_set1_epi64(_mm512_alignr_epi64(pos(e3), pos(e3), 7), 1, int(first) - 1);
        const auto is = _mm512_add_epi64(_mm512_add_epi64(prev, v(1)), q(e0)), us = begin(e0, e1),
                   cs = begin(e1, e2);
        const auto cn = size(e2, cs);
        auto pack = [&](__m512i offset, __m512i length) {
            const auto u8 = eq(length, v(8)), u9 = eq(length, v(9));
            const auto us0 = _mm512_maskz_mov_epi64(u8 | u9, offset);
            const auto uword = get(us0),
                       utail = _mm512_maskz_srli_epi64(u9, get(_mm512_add_epi64(us0, v(1))), 56);
            const auto bytes = _mm512_cmpgt_epi8_mask(uword, _mm512_setzero_si512());
            const auto valid_user = (u8 | u9) & eq(_mm512_maskz_set1_epi8(bytes, -1), v(-1)) &
                                    ~(u9 & (eq(utail, v(0)) | _mm512_cmpgt_epu64_mask(utail, v(127))));
            const auto pairs = _mm512_maddubs_epi16(_mm512_set1_epi16(0x0180), uword);
            const auto fours = _mm512_madd_epi16(pairs, _mm512_set1_epi32(0x00014000));
            const auto key =
                _mm512_slli_epi64(_mm512_slli_epi64(fours & v(0x0fffffff), 28) | _mm512_srli_epi64(fours, 32),
                                  7) |
                utail;
            return _mm512_maskz_mov_epi64(valid_user, key);
        };
        const auto id = pack(is, size(e0, is)), user = pack(us, size(e1, us));
        const auto ordinary_campaign = _mm512_cmpge_epu64_mask(cn, v(3)) & _mm512_cmple_epu64_mask(cn, v(8));
        const auto raw = get(_mm512_maskz_mov_epi64(ordinary_campaign, cs));
        const auto digits =
            _mm512_sllv_epi64(_mm512_sub_epi8(_mm512_srli_epi64(raw, 16), _mm512_set1_epi8('0')),
                              _mm512_slli_epi64(_mm512_sub_epi64(v(10), cn), 3));
        const auto valid_campaign = ordinary_campaign & eq(raw & v(65535), v(0x5f63)) &
                                    eq(_mm512_subs_epu8(digits, _mm512_set1_epi8(9)), v(0)) &
                                    (eq(cn, v(3)) | ~eq(raw & v(0xff0000), v(0x300000)));
        direct = valid_campaign & _mm512_cmpneq_epi64_mask(user, v(0)) & _mm512_cmpneq_epi64_mask(id, v(0));
        _mm512_store_si512(ids, id);
        _mm512_store_si512(users, user);
        _mm512_store_si512(slots, column_decimal8(digits));
    }
    for (unsigned j = 0; j < 8; ++j) {
        const unsigned bit = 1u << j;
        const auto *d = ends + 4 * j;
        const unsigned row_start = first;
        first = (d[3] & 32767) + 1;
        if ((known & bit) && !(active & bit)) continue;
        auto field = [&](unsigned k) {
            unsigned s = (k ? (d[k - 1] & 32767) + 1 : row_start) + (d[k] >> 15);
            return std::string_view(base + s, (d[k] & 32767) - (d[k] >> 15) - s);
        };
        auto ts = ms[j];
        if (!(known & bit)) {
            ts = timestamp(field(3));
            if (ts < start || ts >= w.end) continue;
        }
        if constexpr (requires { emit.indexed(uint64_t{}, uint64_t{}, uint32_t{}, int64_t{}); }) {
            if ((direct & bit) && emit.indexed(ids[j], users[j], slots[j], ts)) continue;
        }
        const auto id = field(0), u = field(1), c = field(2);
        if (!id.empty() && !u.empty() && !c.empty()) emit(ClickInput{id, u, c, ts});
    }
}
} // namespace input_detail
#endif
