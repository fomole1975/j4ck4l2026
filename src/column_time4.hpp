// Four independent 64-bit timestamp lanes for pre-AVX-512 machines.
#if defined(__AVX2__) && defined(__BMI2__)
namespace input_detail {
using Vec4 = __m256i;
inline Vec4 v4(int64_t x) { return _mm256_set1_epi64x(x); }
inline Vec4 eq4(Vec4 a, Vec4 b) { return _mm256_cmpeq_epi64(a, b); }
inline Vec4 gt4(Vec4 a, Vec4 b) { return _mm256_cmpgt_epi64(a, b); }
inline unsigned lane4(Vec4 mask) { return _mm256_movemask_pd(_mm256_castsi256_pd(mask)); }
inline Vec4 blend4(Vec4 no, Vec4 yes, Vec4 mask) { return _mm256_blendv_epi8(no, yes, mask); }

struct Calendar4 {
    struct Month {
        uint64_t prefix = 0;
        int64_t preceding = 0;
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
            char date[20]{};
            std::snprintf(date, sizeof(date), "%04d-%02u-", int(ym.year()), unsigned(ym.month()));
            auto prev = sys_days(ym / 1) - days{1};
            m = {load8(date), duration_cast<milliseconds>(prev.time_since_epoch()).count(),
                 int(unsigned((ym / last).day()))};
        }
    }
    Vec4 day(Vec4 prefix, Vec4 dd, Vec4 &good) const {
        const auto d = _mm256_and_si256(_mm256_sub_epi8(dd, v4(0x3030)), v4(65535));
        const auto overflow = _mm256_subs_epu8(d, v4(0x0909));
        const auto n = _mm256_maddubs_epi16(d, v4(0x010a));
        const auto m0 = eq4(prefix, v4(months[0].prefix)), m1 = eq4(prefix, v4(months[1].prefix));
        const auto lim = blend4(v4(months[0].last), v4(months[1].last), m1);
        good = (m0 | m1) & eq4(overflow, v4(0)) & gt4(n, v4(0)) & ~gt4(n, lim);
        return _mm256_add_epi64(blend4(v4(months[0].preceding), v4(months[1].preceding), m1),
                                _mm256_mul_epu32(n, v4(day_ms)));
    }
};
inline Vec4 clock4(Vec4 bytes, Vec4 &ok) {
    auto d = _mm256_sub_epi8(bytes, v4(UINT64_C(0x30303a30303a3030)));
    auto pairs = _mm256_maddubs_epi16(d, v4(UINT64_C(0x010a00010a00010a)));
    auto halves = _mm256_madd_epi16(pairs, v4(UINT64_C(0x0001003c003c0e10)));
    ok = eq4(_mm256_subs_epu8(d, v4(UINT64_C(0x0905000905000902))), v4(0)) & gt4(v4(24), pairs & v4(65535));
    return _mm256_mul_epu32(_mm256_add_epi32(halves, _mm256_srli_epi64(halves, 32)), v4(1000));
}
inline Vec4 decimal4(Vec4 d) {
    auto pairs = _mm256_maddubs_epi16(d, _mm256_set1_epi16(0x010a));
    auto fours = _mm256_madd_epi16(pairs, _mm256_set1_epi32(0x00010064));
    return _mm256_add_epi64(_mm256_mul_epu32(fours, v4(10000)), _mm256_srli_epi64(fours, 32));
}
#if !defined(__AVX512BW__)
struct EventField4 {
    __m128i start, length;
};
template <class Callback>
void EventDecoder<Callback>::four(const char *base, const uint16_t *ends, unsigned first) {
    // A four-by-four transpose of sixteen separator/quote words gives each
    // timestamp column's offsets before issuing independent gather loads.
    auto load = [&](unsigned row) {
        return _mm_loadl_epi64(reinterpret_cast<const __m128i *>(ends + 7 * row + 1));
    };
    auto a = _mm_unpacklo_epi16(load(0), load(1)), b = _mm_unpacklo_epi16(load(2), load(3));
    auto v12 = _mm_unpacklo_epi32(a, b), v34 = _mm_unpackhi_epi32(a, b);
    auto e1 = _mm_cvtepu16_epi32(v12), e2 = _mm_cvtepu16_epi32(_mm_srli_si128(v12, 8)),
         e3 = _mm_cvtepu16_epi32(v34);
    auto one = _mm_set1_epi32(1);
    auto begin = [&](__m128i prev, __m128i end) {
        return _mm_add_epi32(_mm_add_epi32(_mm_srli_epi32(prev, 1), one), _mm_and_si128(end, one));
    };
    auto size = [&](__m128i end, __m128i start) {
        return _mm_sub_epi32(_mm_sub_epi32(_mm_srli_epi32(end, 1), _mm_and_si128(end, one)), start);
    };
    auto col = [&]<int C>() {
        if constexpr (C == 0)
            return _mm_setr_epi32(ends[0], ends[7], ends[14], ends[21]);
        else if constexpr (C == 1)
            return e1;
        else if constexpr (C == 2)
            return e2;
        else if constexpr (C == 3)
            return e3;
        else if constexpr (C == 4)
            return _mm_cvtepu16_epi32(_mm_srli_si128(v34, 8));
        else
            return _mm_setr_epi32(ends[C], ends[C + 7], ends[C + 14], ends[C + 21]);
    };
    auto spans = [&]<int C>() {
        const auto e = col.template operator()<C>();
        const auto start = [&] {
            if constexpr (C != 0)
                return begin(col.template operator()<C - 1>(), e);
            else
                return _mm_add_epi32(
                    _mm_setr_epi32(first, (ends[6] >> 1) + 1, (ends[13] >> 1) + 1, (ends[20] >> 1) + 1),
                    _mm_and_si128(e, one));
        }();
        return EventField4{start, size(e, start)};
    };
    auto fields = [&](unsigned j, unsigned k) {
        const unsigned at = 7 * j + k, e = ends[at];
        const unsigned start = (at ? (ends[at - 1] >> 1) + 1 : first) + (e & 1);
        return std::string_view(base + start, (e >> 1) - (e & 1) - start);
    };
    project4(base, spans, fields);
}
template <class Callback>
template <class Columns, class Fields>
void EventDecoder<Callback>::project4(const char *base, Columns col, Fields fields) {
    const auto time_field = col.template operator()<2>(), ingest_field = col.template operator()<3>();
    const auto t = time_field.start, g = ingest_field.start, n = time_field.length, gn = ingest_field.length;
    auto iso = _mm_cmpeq_epi32(n, _mm_set1_epi32(20)), epoch = _mm_cmpeq_epi32(n, _mm_set1_epi32(13)),
         valid_g = _mm_cmpeq_epi32(gn, _mm_set1_epi32(20));
    auto safe = _mm_and_si128(_mm_or_si128(iso, epoch), t), gsafe = _mm_and_si128(valid_g, g);
    auto get = [&](__m128i index, int plus) {
        return _mm256_i32gather_epi64(reinterpret_cast<const long long *>(base),
                                      _mm_add_epi32(index, _mm_set1_epi32(plus)), 1);
    };
    const auto tp = get(safe, 0), tm = get(safe, 8), sfx = get(_mm_and_si128(iso, t), 12), gp = get(gsafe, 0),
               gm = get(gsafe, 8);
    const auto tw = blend4(_mm256_slli_epi64(sfx, 8), _mm256_srli_epi64(tm, 24), v4(255));
    static thread_local Calendar4 dates;
    dates.reset(w.start);
    Vec4 td_ok, gd_ok, tc_ok, gc_ok;
    const auto td = dates.day(tp, tm, td_ok), gd = dates.day(gp, gm, gd_ok);
    const auto tc = clock4(tw, tc_ok), gc = clock4(get(gsafe, 11), gc_ok);
    const auto fast_iso = _mm256_cvtepi32_epi64(iso) & td_ok;
    const auto tz_ok = eq4(tm & v4(0xff0000), v4(0x540000)) & eq4(_mm256_srli_epi64(sfx, 56), v4('Z'));
    auto time = blend4(v4(bad_time), _mm256_add_epi64(td, tc), fast_iso & tc_ok & tz_ok);
    const auto epoch64 = _mm256_cvtepi32_epi64(epoch);
    if (lane4(epoch64)) {
        const auto f = _mm256_sub_epi8(tp, _mm256_set1_epi8('0'));
        const auto tail = _mm256_sub_epi8(
            _mm256_or_si256(_mm256_srli_epi64(tp, 40), _mm256_slli_epi64(tm, 24)), _mm256_set1_epi8('0'));
        const auto over = _mm256_subs_epu8(_mm256_max_epu8(f, tail), _mm256_set1_epi8(9));
        const auto milliseconds = _mm256_add_epi64(
            _mm256_mul_epu32(decimal4(_mm256_slli_epi64(f, 24)), v4(100000000)), decimal4(tail));
        time = blend4(time, milliseconds, epoch64 & eq4(over, v4(0)));
    }
    alignas(32) int64_t times[4], ingests[4];
    _mm256_store_si256(reinterpret_cast<Vec4 *>(times), time);
    _mm256_store_si256(reinterpret_cast<Vec4 *>(ingests), _mm256_add_epi64(gd, gc));
    const unsigned time_done = lane4(fast_iso | epoch64),
                   ingest_done = lane4(gd_ok & gc_ok & _mm256_cvtepi32_epi64(valid_g));
    const auto u = col.template operator()<1>();
    const auto us = u.start, un = u.length;
    const auto u8 = _mm_cmpeq_epi32(un, _mm_set1_epi32(8)), u9 = _mm_cmpeq_epi32(un, _mm_set1_epi32(9));
    const auto usafe = _mm_and_si128(us, _mm_or_si128(u8, u9));
    const auto uw = get(usafe, 0);
    const auto u9mask = _mm256_cvtepi32_epi64(u9);
    const auto utail = _mm256_srli_epi64(get(usafe, 1), 56) & u9mask;
    const auto uascii = _mm256_cvtepi32_epi64(_mm_or_si128(u8, u9)) &
                        eq4(_mm256_cmpgt_epi8(uw, v4(0)), v4(-1)) &
                        ~(u9mask & (eq4(utail, v4(0)) | gt4(utail, v4(127))));
    const auto pairs = _mm256_maddubs_epi16(_mm256_set1_epi16(0x0180), uw);
    const auto fours = _mm256_madd_epi16(pairs, _mm256_set1_epi32(0x00014000));
    const auto packed = (_mm256_slli_epi64(fours & v4(0x0fffffff), 28) | _mm256_srli_epi64(fours, 32));
    const auto keys = (_mm256_slli_epi64(packed, 7) | utail) & uascii;
    const auto id = col.template operator()<0>();
    const auto ids = id.start, ilen32 = id.length;
    const auto ilen = _mm256_cvtepi32_epi64(ilen32);
    const auto suf_mask = _mm256_srlv_epi64(v4(-1), _mm256_slli_epi64(_mm256_sub_epi64(v4(9), ilen), 3));
    const auto suffix = get(ids, 1);
    const auto full = suffix | ~suf_mask;
    const auto zero = _mm256_andnot_si256(full, _mm256_sub_epi64(full, v4(UINT64_C(0x0101010101010101))));
    const auto literal = eq4(get(ids, 0) & v4(255), v4('e')) &
                         eq4(zero & v4(UINT64_C(0x8080808080808080)), v4(0)) & suffix & suf_mask;
    alignas(32) uint64_t user_keys[4], literal_ids[4];
    _mm256_store_si256(reinterpret_cast<Vec4 *>(user_keys), keys);
    _mm256_store_si256(reinterpret_cast<Vec4 *>(literal_ids), literal);
    unsigned direct = 0, accepted = 0;
    if constexpr (requires { emit.basic(uint64_t{}, uint64_t{}, int64_t{}, int64_t{}); }) {
        const auto kind = col.template operator()<4>(), prod = col.template operator()<5>();
        const auto kstart = kind.start;
        const auto kl = _mm256_cvtepi32_epi64(_mm_cmpeq_epi32(kind.length, _mm_set1_epi32(4)));
        const auto label = get(kstart, 0) & v4(UINT32_MAX);
        const auto has_product = _mm256_cvtepi32_epi64(_mm_cmpgt_epi32(prod.length, _mm_setzero_si128()));
        direct = time_done & ingest_done &
                 lane4(kl & (eq4(label, v4(0x77656976)) | eq4(label, v4(0x74726163))) & ~eq4(keys, v4(0)) &
                       ~eq4(literal, v4(0)));
        accepted = lane4(has_product & ~gt4(v4(w.start), time) & gt4(v4(w.end), time) &
                         ~gt4(_mm256_add_epi64(gd, gc), _mm256_add_epi64(time, v4(day_ms))));
    }
    for (unsigned j = 0; j < 4; ++j) {
        if constexpr (requires { emit.basic(uint64_t{}, uint64_t{}, int64_t{}, int64_t{}); }) {
            if (direct & (1u << j)) {
                emit.basic((accepted & (1u << j)) ? user_keys[j] : 0, literal_ids[j], times[j], ingests[j]);
                continue;
            }
        }
        auto field = [&](unsigned k) { return fields(j, k); };
        const auto id = field(0), user = field(1);
        if (id.empty()) continue;
        auto ts = times[j], ingest = ingests[j];
        if (!(time_done & (1u << j))) ts = timestamp(field(2));
        if (!(ingest_done & (1u << j))) {
            auto f = field(3);
            ingest = f.size() == 20 ? civil<false>(f.data()) : bad_time;
        }
        const auto kind = field(4), product = field(5);
        const bool purchased = kind == "purchase";
        const auto cents = purchased ? price(field(6)) : 0;
        const bool valid = !user.empty() && !product.empty() && ts >= w.start && ts < w.end &&
                           ingest != bad_time && ingest <= ts + day_ms &&
                           (purchased ? cents != 0 : kind == "view" || kind == "cart");
        emit(EventInput{id, user, product, ts, ingest, cents, valid, user_keys[j], literal_ids[j]});
    }
}
#endif
} // namespace input_detail
#endif
