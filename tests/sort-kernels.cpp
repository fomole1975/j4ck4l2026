#include "sessions.hpp"
#include "time_sort32.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <random>

namespace {
using namespace shopping::event_sort;

bool check_events() {
    std::mt19937_64 random(17);
    for (size_t n = 1; n <= 32 * width; ++n) {
        for (unsigned trial = 0; trial < 128; ++trial) {
            std::array<uint64_t, 32 * width + 18> actual;
            for (auto& value : actual) {
                value = random();
                if (trial % 3 == 0) value %= 5;
                if (trial % 5 == 0 && value % 3 == 1) value = UINT64_MAX;
            }

            // Compare the entire array so a store outside the requested range fails.
            const size_t start = 1 + trial % 17;
            auto expected = actual;
            std::sort(expected.begin() + start, expected.begin() + start + n);
            auto* rows = actual.data() + start;
            if (n <= width) sort_small<1>(rows, n);
            else if (n <= 2 * width) sort_small<2>(rows, n);
            else if (n <= 4 * width) sort_small<4>(rows, n);
            else if (n <= 8 * width) sort_small<8>(rows, n);
            else if (n <= 16 * width) sort_small<16>(rows, n);
            else sort_small<32>(rows, n);

            if (actual != expected) {
                std::fprintf(stderr, "event mismatch at length %zu, trial %u\n", n, trial);
                return false;
            }
        }
    }
    return true;
}

bool check_time_offsets() {
    namespace kernel = shopping::time_sort32;
    std::mt19937 random(23);
    constexpr size_t lanes = kernel::width;
    for (size_t n = 1; n <= 32 * lanes; ++n) {
        for (unsigned trial = 0; trial < 64; ++trial) {
            std::array<uint32_t, 32 * lanes + 18> actual;
            for (auto& x : actual) {
                x = random();
                if (trial % 3 == 0) x %= 7;
                if (trial % 5 == 0 && x % 3 == 1) x = UINT32_MAX;
            }
            const size_t start = 1 + trial % 17;
            auto expected = actual;
            std::sort(expected.begin() + start, expected.begin() + start + n);
            auto* rows = actual.data() + start;
            if (n <= lanes) kernel::sort<1>(rows, n);
            else if (n <= 2 * lanes) kernel::sort<2>(rows, n);
            else if (n <= 4 * lanes) kernel::sort<4>(rows, n);
            else if (n <= 8 * lanes) kernel::sort<8>(rows, n);
            else if (n <= 16 * lanes) kernel::sort<16>(rows, n);
            else kernel::sort<32>(rows, n);
            if (actual != expected) {
                std::fprintf(stderr, "timestamp offset mismatch at length %zu, trial %u\n", n, trial);
                return false;
            }
        }
    }
    return true;
}

#if defined(__AVX512F__)
struct Click {
    uint32_t offset, campaign;
    uint64_t id;
    bool operator==(const Click&) const = default;
};

bool check_clicks() {
    std::mt19937_64 random(19);
    for (size_t n = 1; n <= 64; ++n) {
        for (unsigned trial = 0; trial < 128; ++trial) {
            std::array<Click, 64 + 18> actual;
            for (auto& click : actual) {
                auto offset = static_cast<uint32_t>(random());
                if (trial % 2 == 0) offset %= 5;
                if (trial % 5 == 0 && offset % 3 == 1) offset = UINT32_MAX;
                click = {offset, static_cast<uint32_t>(random()), random()};
            }

            const size_t start = 1 + trial % 17;
            auto expected = actual;
            std::stable_sort(expected.begin() + start, expected.begin() + start + n,
                             [](const Click& a, const Click& b) { return a.offset < b.offset; });
            auto* rows = actual.data() + start;
            if (n <= 8) sort_clicks<1>(rows, n);
            else if (n <= 16) sort_clicks<2>(rows, n);
            else if (n <= 32) sort_clicks<4>(rows, n);
            else sort_clicks<8>(rows, n);

            if (actual != expected) {
                std::fprintf(stderr, "click mismatch at length %zu, trial %u\n", n, trial);
                return false;
            }
        }
    }
    return true;
}
#endif
} // namespace

int main() {
    if (!check_events() || !check_time_offsets()) return 1;
#if defined(__AVX512F__)
    if (!check_clicks()) return 1;
#endif
    return 0;
}
