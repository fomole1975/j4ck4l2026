#include "output.hpp"
#include "run_parquet.hpp"

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <dlfcn.h>
#include <exception>
#include <stdexcept>
#include <thread>

namespace shopping {
namespace {
struct GeneralWriter {
    using Users = void (*)(const std::filesystem::path&, const std::vector<UserSpan>&, const Result&, bool);
    using Campaigns = void (*)(const std::filesystem::path&, const Result&, bool);
    Users users;
    Campaigns campaigns;
};

// Loading Arrow maps its types and initializes its runtime. The built-in writer
// handles ordinary scoring inputs without needing those libraries.
const GeneralWriter& general_writer() {
    static const GeneralWriter instance = [] {
        Dl_info loc{};
        if (!dladdr(reinterpret_cast<const void*>(&general_writer), &loc) || !loc.dli_fname)
            throw std::runtime_error("Unable to locate the Parquet writer");
        const auto bin = std::filesystem::absolute(loc.dli_fname).parent_path();
        auto path = bin / SHOPPING_WRITER_NAME;
        if (!std::filesystem::exists(path)) path = bin.parent_path() / "lib" / SHOPPING_WRITER_NAME;
        void* module = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!module) throw std::runtime_error(dlerror());
        auto users = reinterpret_cast<GeneralWriter::Users>(dlsym(module, "shopping_write_users"));
        auto campaigns = reinterpret_cast<GeneralWriter::Campaigns>(dlsym(module, "shopping_write_campaigns"));
        if (!users || !campaigns) throw std::runtime_error("Missing Parquet writer entry point");
        return GeneralWriter{users, campaigns};
    }();
    return instance;
}
}

void write_output(const std::filesystem::path& directory, const Result& result, unsigned workers) {
    bool snappy = false;
    if (const char* name = std::getenv("SHOPPING_CODEC")) {
        if (std::string_view(name) == "snappy") snappy = true;
        else if (std::string_view(name) != "plain") throw std::runtime_error("Unsupported SHOPPING_CODEC");
    }
    auto user_dir = directory / "user_daily";
    auto campaign_dir = directory / "campaign_daily";
    std::filesystem::create_directories(user_dir);
    std::filesystem::create_directories(campaign_dir);
    std::size_t total = 0;
    for (const auto& rows : result.users) total += rows.size();
    std::size_t parts = std::clamp<std::size_t>((total + parquet_group_rows - 1) / parquet_group_rows, 1, std::max(1U, workers));
    std::size_t per_part = (total + parts - 1) / parts;
    std::vector<std::vector<UserSpan>> partitions(parts);
    std::size_t offset = 0;
    for (const auto& chunk : result.users) {
        UserSpan rows(chunk);
        while (!rows.empty()) {
            auto length = std::min(per_part - offset % per_part, rows.size());
            partitions[offset / per_part].push_back(rows.first(length));
            rows = rows.subspan(length);
            offset += length;
        }
    }
    std::vector<std::exception_ptr> errors(parts);
    auto write_part = [&](std::size_t part) {
        try {
            const bool direct = std::endian::native == std::endian::little &&
                                !snappy;
            const auto path = user_dir / ("part-" + std::to_string(part) + ".parquet");
            if (!direct || !try_run_encoded_users(path, partitions[part], result)) {
                general_writer().users(path, partitions[part], result, snappy);
            }
            if (part == 0) {
                const auto path = campaign_dir / "part-0.parquet";
                if (!direct || !try_plain_campaigns(path, result)) {
                    general_writer().campaigns(path, result, snappy);
                }
            }
        } catch (...) {
            errors[part] = std::current_exception();
        }
    };
    {
        std::vector<std::jthread> threads;
        for (std::size_t i = 1; i < parts; ++i) threads.emplace_back(write_part, i);
        write_part(0);
    }
    for (const auto& error : errors) if (error) std::rethrow_exception(error);
}

}
