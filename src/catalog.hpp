#pragma once

#include "key.hpp"
#include <charconv>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
#include <vector>

namespace shopping {

// Canonical decimal IDs can address a small array. Every other spelling,
// including a leading zero, retains its exact identity in the sparse table.
template <bool Campaign, class SparseTable>
class CatalogIndex {
    static constexpr uint32_t missing = std::numeric_limits<uint32_t>::max();
    std::vector<uint32_t> dense_;
    SparseTable sparse_;

    std::optional<uint32_t> slot(std::string_view id) const {
        constexpr std::string_view prefix = Campaign ? "c_" : "p";
        if (!id.starts_with(prefix)) return {};
        id.remove_prefix(prefix.size());
        if (id.empty() || id.size() > 6 || (id.size() > 1 && id.front() == '0')) return {};
        uint32_t index;
        const char* end = id.data() + id.size();
        auto result = std::from_chars(id.data(), end, index);
        if (result.ec != std::errc{} || result.ptr != end || index >= dense_.size()) return {};
        return index;
    }
public:
    explicit CatalogIndex(size_t expected) : dense_(expected + 1, missing) {}

    void set(std::string_view id, uint32_t value) {
        if (auto index = slot(id)) dense_[*index] = value;
        else sparse_.set(Key(id), value);
    }
    std::optional<uint32_t> known_slot(uint32_t slot) const {
        if (slot >= dense_.size()) return {};
        return dense_[slot];
    }
    uint32_t get(std::string_view id, uint32_t fallback) const {
        if (auto index = slot(id)) {
            uint32_t value = dense_[*index];
            return value == missing ? fallback : value;
        }
        return sparse_.get(Key(id), fallback);
    }
};

}
