#pragma once

#include <filesystem>

namespace shopping {

struct Result;
void write_output(const std::filesystem::path& directory, const Result& result, unsigned workers);

}
