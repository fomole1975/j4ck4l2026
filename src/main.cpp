#include "engine.hpp"
#include "input.hpp"
#include "output.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <sys/resource.h>

namespace {
const char* env_or(const char* name, const char* fallback) {
    const char* value = std::getenv(name);
    return value && *value ? value : fallback;
}

unsigned worker_count() {
    if (const char* setting = std::getenv("NCPU")) {
        std::string_view field(setting);
        unsigned workers = 0;
        const auto [end, error] = std::from_chars(field.data(), field.data() + field.size(), workers);
        if (error != std::errc{} || end != field.data() + field.size() || workers == 0)
            throw std::runtime_error("NCPU must be a positive integer");
        return workers;
    }
    return std::clamp(std::thread::hardware_concurrency(), 1u, 8u);
}
}

int main() {
    const auto wall_main = std::chrono::system_clock::now();
    try {
        const auto start = std::chrono::steady_clock::now();
        const unsigned workers = worker_count();
        shopping::InputFiles input(env_or("DATA_DIR", "/data"));
        const auto mapped = std::chrono::steady_clock::now();
        const auto results = shopping::run_engine(input, workers);
        const auto computed = std::chrono::steady_clock::now();
        shopping::write_output(env_or("OUTPUT_DIR", "/output"), results, workers);
        if (std::getenv("PROFILE")) {
            const auto end = std::chrono::steady_clock::now();
            using Seconds = std::chrono::duration<double>;
            struct rusage usage {};
            getrusage(RUSAGE_SELF, &usage);
#if defined(__APPLE__)
            const auto rss_mib = usage.ru_maxrss / (1024.0 * 1024);
#else
            const auto rss_mib = usage.ru_maxrss / 1024.0;
#endif
            std::cerr << "map=" << Seconds(mapped - start).count()
                      << " compute=" << Seconds(computed - mapped).count()
                      << " parquet=" << Seconds(end - computed).count() << " seconds rss=" << rss_mib << " MiB\n";
            const auto ns = [](auto t) { return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count(); };
            std::cerr << "wall_main=" << ns(wall_main) << " wall_done=" << ns(std::chrono::system_clock::now()) << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "shopping-sessions: " << error.what() << '\n';
        return 1;
    }
}
