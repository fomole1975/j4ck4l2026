// Included inside the shopping namespace after the scalar row decoder.
#if defined(__AVX512BW__)
namespace input_detail {
struct CalendarBlock {
    struct Month {
        uint64_t prefix = 0;
        int64_t preceding_midnight = 0;
        int last = 0;
    };
    Month months[2];
    int64_t window = bad_time;
    void reset(int64_t start) {
        if (window == start) return;
        window = start;
        months[0] = {};
        months[1] = {};
        if (start < -62167219200000LL || start > 253402300799999LL) return;
        using namespace std::chrono;
        auto rep = year_month_day(floor<days>(sys_time<milliseconds>{milliseconds{start}}) - days{2});
        for (auto &m : months) {
            auto ym = rep.year() / rep.month();
            rep = (ym + std::chrono::months{1}) / 1;
            if (int(ym.year()) < 0 || int(ym.year()) > 9999) continue;
            const auto before = sys_days(ym / 1) - days{1};
            char bytes[20]{};
            std::snprintf(bytes, sizeof(bytes), "%04d-%02u-", int(ym.year()), unsigned(ym.month()));
            m = {load8(bytes), duration_cast<milliseconds>(before.time_since_epoch()).count(),
                 int(unsigned((ym / last).day()))};
        }
    }
    __m512i day(__m512i prefix, __m512i day_bytes, __mmask8 &ok) const {
        const auto z = _mm512_setzero_si512(), lo = _mm512_set1_epi64(65535);
        const auto d = _mm512_and_si512(_mm512_sub_epi8(day_bytes, _mm512_set1_epi64(0x3030)), lo);
        const auto overflow = _mm512_subs_epu8(d, _mm512_set1_epi64(0x0909));
        const auto day_number = _mm512_maddubs_epi16(d, _mm512_set1_epi64(0x010a));
        const auto month0 = _mm512_cmpeq_epi64_mask(prefix, _mm512_set1_epi64(months[0].prefix));
        const auto month1 = _mm512_cmpeq_epi64_mask(prefix, _mm512_set1_epi64(months[1].prefix));
        const auto last = _mm512_mask_blend_epi64(month1, _mm512_set1_epi64(months[0].last),
                                                  _mm512_set1_epi64(months[1].last));
        const auto before = _mm512_mask_blend_epi64(month1, _mm512_set1_epi64(months[0].preceding_midnight),
                                                    _mm512_set1_epi64(months[1].preceding_midnight));
        ok = (month0 | month1) & _mm512_cmpeq_epi64_mask(overflow, z) &
             _mm512_cmpneq_epi64_mask(day_number, z) & _mm512_cmple_epu64_mask(day_number, last);
        return _mm512_add_epi64(before, _mm512_mul_epu32(day_number, _mm512_set1_epi64(day_ms)));
    }
};

inline __m512i column_digit_clock(__m512i bytes, __mmask8 &valid) {
    const __m512i d = _mm512_sub_epi8(bytes, _mm512_set1_epi64(UINT64_C(0x30303a30303a3030)));
    const __m512i limits = _mm512_set1_epi64(UINT64_C(0x0905000905000902));
    const auto excess = _mm512_subs_epu8(d, limits);
    const auto pairs = _mm512_maddubs_epi16(d, _mm512_set1_epi64(UINT64_C(0x010a00010a00010a)));
    // pairs consist of hour, minute tens * 10, minute units, second.
    const auto halves = _mm512_madd_epi16(pairs, _mm512_set1_epi64(UINT64_C(0x0001003c003c0e10)));
    valid = _mm512_cmpeq_epi64_mask(excess, _mm512_setzero_si512()) &
            _mm512_cmplt_epu64_mask(_mm512_and_si512(pairs, _mm512_set1_epi64(65535)), _mm512_set1_epi64(24));
    return _mm512_mul_epu32(_mm512_add_epi32(halves, _mm512_srli_epi64(halves, 32)), _mm512_set1_epi64(1000));
}

inline __m512i column_decimal8(__m512i digits) {
    const auto pairs = _mm512_maddubs_epi16(digits, _mm512_set1_epi16(0x010a));
    const auto fours = _mm512_madd_epi16(pairs, _mm512_set1_epi32(0x00010064));
    return _mm512_add_epi64(_mm512_mul_epu32(fours, _mm512_set1_epi64(10000)), _mm512_srli_epi64(fours, 32));
}

inline __m512i apply_zone8(const char* base,__m512i t,__mmask8 tz,__mmask8 valid,
                            __m512i suffix_char,__m512i ordinary,__m512i time_ms) {
    if (tz) {
        // Last five bytes HH:MM are present only in numeric offsets.
        const auto p = _mm512_i64gather_epi64(_mm512_maskz_add_epi64(tz, t, _mm512_set1_epi64(17)), base, 1);
        const auto d = _mm512_sub_epi8(_mm512_srli_epi64(p, 24), _mm512_set1_epi64(UINT64_C(0x30303a3030)));
        const auto overflow = _mm512_subs_epu8(d, _mm512_set1_epi64(UINT64_C(0x0905000902)));
        const auto pairs = _mm512_maddubs_epi16(d, _mm512_set1_epi64(UINT64_C(0x010a00010a)));
        const auto h = _mm512_and_si512(pairs, _mm512_set1_epi64(65535));
        const auto halves = _mm512_madd_epi16(pairs, _mm512_set1_epi64(UINT64_C(0x000000010001003c)));
        const auto minutes = _mm512_add_epi32(halves, _mm512_srli_epi64(halves, 32));
        const auto offset = _mm512_mul_epu32(minutes, _mm512_set1_epi64(60000));
        const auto plus = _mm512_cmpeq_epi64_mask(suffix_char, _mm512_set1_epi64('+'));
        const auto minus = _mm512_cmpeq_epi64_mask(suffix_char, _mm512_set1_epi64('-'));
        const auto ok = tz & valid & (plus | minus) &
                        _mm512_cmpeq_epi64_mask(overflow, _mm512_setzero_si512()) &
                        _mm512_cmplt_epu64_mask(h, _mm512_set1_epi64(24));
        const auto adjusted =
            _mm512_mask_sub_epi64(_mm512_add_epi64(ordinary, offset), plus, ordinary, offset);
        time_ms = _mm512_mask_mov_epi64(time_ms, ok, adjusted);
    }
    return time_ms;
}

struct EventField8 { __m512i start, length; };

template <class Callback>
void EventDecoder<Callback>::eight(const char *base, const uint16_t *ends, unsigned first) {
    // Transpose 8 x 7 validated separator words into named columns. Each
    // selector takes its entries from a pair of contiguous cache lines.
    const auto lo = _mm512_loadu_si512(ends), hi = _mm512_loadu_si512(ends + 32);
    const auto col = [&]<int C>() {
        alignas(64) static constexpr auto rank = [] {
            std::array<uint16_t, 32> v{};
            for (int j = 0; j < 8; ++j)
                v[j] = 7 * j + C;
            return v;
        }();
        return _mm512_cvtepu16_epi64(
            _mm512_castsi512_si128(_mm512_permutex2var_epi16(lo, _mm512_load_si512(rank.data()), hi)));
    };
    const auto prev = _mm512_and_si512(col.template operator()<6>(), _mm512_set1_epi64(32767));
    const auto starts = _mm512_add_epi64(
        _mm512_mask_set1_epi64(_mm512_alignr_epi64(prev, prev, 7), 1, int(first) - 1),
        _mm512_set1_epi64(1));
    const auto spans = [&]<int C>() {
        const auto end = col.template operator()<C>();
        const auto q = _mm512_srli_epi64(end, 15);
        const auto raw_start = [&] {
            if constexpr (C==0) return starts;
            else return _mm512_add_epi64(col.template operator()<C-1>() & _mm512_set1_epi64(32767),
                                         _mm512_set1_epi64(1));
        }();
        const auto begin = _mm512_add_epi64(raw_start,q);
        return EventField8{begin,_mm512_sub_epi64(_mm512_sub_epi64(end & _mm512_set1_epi64(32767),q),begin)};
    };
    const auto fields = [&](unsigned j, unsigned k) {
        const uint16_t *d = ends + 7 * j;
        const unsigned start = (k || j ? (d[static_cast<int>(k) - 1] & 32767) + 1 : first) + (d[k] >> 15);
        return std::string_view(base + start, (d[k] & 32767) - (d[k] >> 15) - start);
    };
    project8(base, spans, fields);
}

template <class Callback>
template <class Columns, class Fields>
void EventDecoder<Callback>::project8(const char *base, Columns col, Fields fields) {
    const auto z = _mm512_setzero_si512(), one = _mm512_set1_epi64(1);
    const auto gather = [&](__m512i p) { return _mm512_i64gather_epi64(p, base, 1); };
    const auto time = col.template operator()<2>(), ingest = col.template operator()<3>();
    const auto t = time.start, n = time.length, g = ingest.start, gn = ingest.length;
    const auto iso = _mm512_cmpeq_epi64_mask(n, _mm512_set1_epi64(20));
    const auto tz = _mm512_cmpeq_epi64_mask(n, _mm512_set1_epi64(25));
    const auto epoch = _mm512_cmpeq_epi64_mask(n, _mm512_set1_epi64(13));
    const auto good_ingest_length = _mm512_cmpeq_epi64_mask(gn, _mm512_set1_epi64(20));
    const auto gsafe = _mm512_maskz_mov_epi64(good_ingest_length, g);
    const auto gp = gather(gsafe), gm = gather(_mm512_add_epi64(gsafe, _mm512_set1_epi64(8)));
    const auto safe = _mm512_maskz_mov_epi64(iso | tz | epoch, t);
    const auto tp = gather(safe), tm = gather(_mm512_add_epi64(safe, _mm512_set1_epi64(8)));
    const auto tz_clock = gather(_mm512_maskz_add_epi64(iso | tz, t, _mm512_set1_epi64(12)));
    const auto tw = _mm512_mask_blend_epi8(UINT64_C(0x0101010101010101), _mm512_slli_epi64(tz_clock, 8),
                                           _mm512_srli_epi64(tm, 24));
    static thread_local CalendarBlock dates;
    dates.reset(w.start);
    __mmask8 good_gday, good_tday, valid_clock, valid_ingest_clock;
    const auto gday = dates.day(gp, gm, good_gday), tday = dates.day(tp, tm, good_tday);
    const auto ingest_clock =
        column_digit_clock(gather(_mm512_add_epi64(gsafe, _mm512_set1_epi64(11))), valid_ingest_clock);
    const auto event_clock = column_digit_clock(tw, valid_clock);
    const auto ingest_fast = good_gday & good_ingest_length & valid_ingest_clock;
    const auto fast_iso = (iso | tz) & good_tday;
    const auto letter_t = _mm512_cmpeq_epi64_mask(_mm512_and_si512(tm, _mm512_set1_epi64(0xff0000)),
                                                  _mm512_set1_epi64(0x540000));
    const auto suffix_char = _mm512_srli_epi64(tz_clock, 56);
    const auto zulu = _mm512_cmpeq_epi64_mask(suffix_char, _mm512_set1_epi64('Z'));
    const auto ordinary = _mm512_add_epi64(tday, event_clock);
    auto time_ms = _mm512_mask_mov_epi64(_mm512_set1_epi64(bad_time),
                                         fast_iso & iso & valid_clock & letter_t & zulu, ordinary);
    time_ms=apply_zone8(base,t,tz,good_tday & valid_clock & letter_t,suffix_char,ordinary,time_ms);
    if (epoch) {
        const auto front = _mm512_sub_epi8(tp, _mm512_set1_epi8('0'));
        const auto tail = _mm512_sub_epi8(_mm512_or_si512(_mm512_srli_epi64(tp, 40), _mm512_slli_epi64(tm, 24)), _mm512_set1_epi8('0'));
        const auto invalid = _mm512_subs_epu8(_mm512_max_epu8(front, tail), _mm512_set1_epi8(9));
        const auto front5 = column_decimal8(_mm512_slli_epi64(front, 24));
        const auto epoch_ms =
            _mm512_add_epi64(_mm512_mul_epu32(front5, _mm512_set1_epi64(100000000)), column_decimal8(tail));
        time_ms = _mm512_mask_mov_epi64(time_ms, epoch & _mm512_cmpeq_epi64_mask(invalid, z), epoch_ms);
    }
    alignas(64) int64_t ts_ms[8], ingest_ms[8];
    _mm512_store_si512(ts_ms, time_ms);
    _mm512_store_si512(ingest_ms, _mm512_add_epi64(gday, ingest_clock));

    alignas(64) uint64_t user_bits[8]{}, literals[8]{};
    const auto user_field = col.template operator()<1>();
    const auto user_start = user_field.start, un = user_field.length;
    const auto user_ok =
        _mm512_cmpeq_epi64_mask(un, _mm512_set1_epi64(8)) | _mm512_cmpeq_epi64_mask(un, _mm512_set1_epi64(9));
    const auto user_safe = _mm512_maskz_mov_epi64(user_ok, user_start);
    auto word = gather(user_safe);
    const auto zero =
        _mm512_andnot_si512(word, _mm512_sub_epi64(word, _mm512_set1_epi64(UINT64_C(0x0101010101010101))));
    auto user_ascii =
        user_ok & _mm512_cmpeq_epi64_mask(_mm512_and_si512(_mm512_or_si512(zero, word),
                                                           _mm512_set1_epi64(UINT64_C(0x8080808080808080))),
                                          z);
    const auto nine = _mm512_cmpeq_epi64_mask(un, _mm512_set1_epi64(9));
    const auto tail = _mm512_maskz_srli_epi64(nine, gather(_mm512_add_epi64(user_safe, one)), 56);
    user_ascii &=
        ~(nine & (_mm512_cmpeq_epi64_mask(tail, z) | _mm512_cmpgt_epu64_mask(tail, _mm512_set1_epi64(127))));
    const auto rev = _mm512_set1_epi64(UINT64_C(0x0001020304050607));
    // PSHUFB selects independently inside 128-bit lanes.
    word = _mm512_shuffle_epi8(
        word, _mm512_mask_add_epi8(rev, UINT64_C(0xff00ff00ff00ff00), rev, _mm512_set1_epi8(8)));
    word = _mm512_or_si512(
        _mm512_and_si512(word, _mm512_set1_epi64(UINT64_C(0x007f007f007f007f))),
        _mm512_and_si512(_mm512_srli_epi64(word, 1), _mm512_set1_epi64(UINT64_C(0x3f803f803f803f80))));
    word = _mm512_or_si512(
        _mm512_and_si512(word, _mm512_set1_epi64(UINT64_C(0x00003fff00003fff))),
        _mm512_and_si512(_mm512_srli_epi64(word, 2), _mm512_set1_epi64(UINT64_C(0x0fffc0000fffc000))));
    word = _mm512_or_si512(
        _mm512_and_si512(_mm512_srli_epi64(word, 4), _mm512_set1_epi64(UINT64_C(0x00fffffff0000000))),
        _mm512_and_si512(word, _mm512_set1_epi64(UINT64_C(0x000000000fffffff))));
    _mm512_store_si512(user_bits,
                       _mm512_maskz_mov_epi64(user_ascii, _mm512_or_si512(_mm512_slli_epi64(word, 7), tail)));

    const auto id_field = col.template operator()<0>();
    const auto id_start = id_field.start, id_len = id_field.length;
    const auto suf_len = _mm512_sub_epi64(id_len, one);
    const auto suf_mask = _mm512_srlv_epi64(
        _mm512_set1_epi64(-1), _mm512_slli_epi64(_mm512_sub_epi64(_mm512_set1_epi64(8), suf_len), 3));
    const auto suffix = gather(_mm512_add_epi64(id_start, one));
    const auto full = _mm512_or_si512(suffix, _mm512_xor_si512(suf_mask, _mm512_set1_epi64(-1)));
    const auto zeros =
        _mm512_andnot_si512(full, _mm512_sub_epi64(full, _mm512_set1_epi64(UINT64_C(0x0101010101010101))));
    const auto literal_ok =
        _mm512_cmpeq_epi64_mask(_mm512_and_si512(gather(id_start), _mm512_set1_epi64(255)),
                                _mm512_set1_epi64('e')) &
        _mm512_cmpeq_epi64_mask(_mm512_and_si512(zeros, _mm512_set1_epi64(UINT64_C(0x8080808080808080))), z);
    _mm512_store_si512(literals, _mm512_maskz_mov_epi64(literal_ok, _mm512_and_si512(suffix, suf_mask)));
    __mmask8 direct = 0, accepted = 0;
    if constexpr (requires { emit.basic(uint64_t{}, uint64_t{}, int64_t{}, int64_t{}); }) {
        const auto e4 = col.template operator()<4>(), e5 = col.template operator()<5>();
        const auto kind = e4.start, kn = e4.length;
        const auto k = _mm512_and_si512(gather(kind), _mm512_set1_epi64(UINT32_MAX));
        const auto ordinary_kind = _mm512_cmpeq_epi64_mask(kn, _mm512_set1_epi64(4)) &
                                   (_mm512_cmpeq_epi64_mask(k, _mm512_set1_epi64(0x77656976)) |
                                    _mm512_cmpeq_epi64_mask(k, _mm512_set1_epi64(0x74726163)));
        const auto has_product = _mm512_cmpneq_epi64_mask(e5.length, z);
        direct = ordinary_kind & (fast_iso | epoch) & ingest_fast &
                 _mm512_cmpneq_epi64_mask(_mm512_load_si512(user_bits), z) &
                 _mm512_cmpneq_epi64_mask(_mm512_load_si512(literals), z);
        accepted = has_product & _mm512_cmpge_epi64_mask(time_ms, _mm512_set1_epi64(w.start)) &
                   _mm512_cmplt_epi64_mask(time_ms, _mm512_set1_epi64(w.end)) &
                   _mm512_cmple_epi64_mask(_mm512_load_si512(ingest_ms),
                                           _mm512_add_epi64(time_ms, _mm512_set1_epi64(day_ms)));
    }
    for (unsigned j = 0; j < 8; ++j) {
        if constexpr (requires { emit.basic(uint64_t{}, uint64_t{}, int64_t{}, int64_t{}); }) {
            if (direct & (1 << j)) {
                emit.basic((accepted & (1 << j)) ? user_bits[j] : 0, literals[j], ts_ms[j], ingest_ms[j]);
                continue;
            }
        }
        const auto field = [&](unsigned k) { return fields(j, k); };
        const auto id = field(0), user = field(1);
        if (id.empty()) continue;
        int64_t ts = ts_ms[j], ingest = ingest_ms[j];
        if (!((fast_iso | epoch) & (1u << j))) ts = timestamp(field(2));
        if (!(ingest_fast & (1u << j))) {
            const auto f = field(3);
            ingest = f.size() == 20 ? civil<false>(f.data()) : bad_time;
        }
        const auto type = field(4), product = field(5);
        const bool purchased = type == "purchase";
        const auto cents = purchased ? price(field(6)) : 0;
        const bool valid = !user.empty() && !product.empty() && ts >= w.start && ts < w.end &&
                           ingest != bad_time && ingest <= ts + day_ms &&
                           (purchased ? cents != 0 : type == "view" || type == "cart");
        emit(EventInput{id, user, product, ts, ingest, cents, valid, user_bits[j], literals[j]});
    }
}
} // namespace input_detail
#endif
