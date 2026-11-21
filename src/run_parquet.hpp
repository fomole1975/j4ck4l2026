#pragma once
#include "engine.hpp"
#include <cstddef>
#include <filesystem>
#include <span>
#include <vector>

namespace shopping {

using UserSpan = std::span<const UserDaily>;
inline constexpr std::size_t parquet_group_rows = 65536;

// On a page-size limit the stream has been closed and the caller rewrites
// the whole file with the general Apache writer. I/O failures propagate.
bool try_run_encoded_users(const std::filesystem::path &path, const std::vector<UserSpan> &spans,
                           const Result &result);
bool try_plain_campaigns(const std::filesystem::path &path, const Result &result);

} // namespace shopping
