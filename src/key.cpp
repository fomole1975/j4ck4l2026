#include "key.hpp"

#include <array>
#include <mutex>
#include <unordered_set>

namespace shopping {
uint64_t Key::intern(std::string_view bytes) {
    struct Bucket {
        std::mutex lock;
        std::unordered_set<std::string> strings;
    };
    static std::array<Bucket, 128> pool;
    auto& bucket = pool[std::hash<std::string_view>{}(bytes) % pool.size()];
    std::lock_guard guard(bucket.lock);
    const auto it = bucket.strings.emplace(bytes).first;
    return tag | reinterpret_cast<uintptr_t>(&*it);
}
}
