#include "input.hpp"

#include <cerrno>
#include <fcntl.h>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <system_error>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace shopping {

int64_t input_detail::midnight(unsigned year, unsigned month, unsigned day) {
    constexpr unsigned days_per_month[] = {31,28,31,30,31,30,31,31,30,31,30,31};
    if (month == 0 || month > 12 || day == 0) return bad_time;
    const bool leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    if (day > days_per_month[month - 1] + (month == 2 && leap)) return bad_time;
    int y = static_cast<int>(year) - (month <= 2);
    const int era = (y >= 0 ? y : y - 399) / 400;
    const int yoe = y - era * 400;
    const int doy = (153 * (int(month) + (month > 2 ? -3 : 9)) + 2) / 5 + int(day) - 1;
    const int epoch_day = era * 146097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719468;
    return epoch_day * day_ms;
}

struct InputFiles::Mapping {
    const char* data = nullptr;
    size_t size = 0;
    explicit Mapping(const std::filesystem::path& path) {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::system_error(errno, std::generic_category(), path.string());
        struct stat st{};
        if (fstat(fd, &st) != 0) {
            auto ec = errno;
            close(fd);
            throw std::system_error(ec, std::generic_category(), path.string());
        }
        size = static_cast<size_t>(st.st_size);
        if (size) {
            auto mapped = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (mapped == MAP_FAILED) {
                auto ec = errno;
                close(fd);
                throw std::system_error(ec, std::generic_category(), path.string());
            }
            data = static_cast<const char*>(mapped);
        }
        close(fd);
    }
    Mapping(Mapping&& other) noexcept : data(std::exchange(other.data, nullptr)), size(other.size) {}
    ~Mapping() { if (data) munmap(const_cast<char*>(data),size); }
};

void InputFiles::add(const std::filesystem::path& path, std::vector<Chunk>& out) {
    const auto& file = mappings.emplace_back(path);
    if (file.size == 0) return;
    const auto all = std::string_view(file.data,file.size);
    // Schedule physical byte ranges without faulting in their pages during
    // setup. Workers align each end to a newline when consuming the range.
    constexpr size_t target = 1u << 20;
    for (size_t pos=0; pos<all.size(); pos+=target) {
        out.push_back({all.substr(pos,target),all.data()+all.size()});
    }
}

InputFiles::InputFiles(const std::filesystem::path& root) {
    std::ifstream input(root / "manifest.json");
    std::string manifest{std::istreambuf_iterator<char>(input),{}};
    auto instant = [&](std::string_view key) -> int64_t {
        auto name = manifest.find(std::string("\"") + std::string(key) + "\"");
        if (name != manifest.npos) {
            auto colon = manifest.find(':',name + key.size() + 2);
            if (colon != manifest.npos) {
                auto first = manifest.find('"',colon + 1);
                if (first != manifest.npos) {
                    auto last = manifest.find('"',first + 1);
                    if (last != manifest.npos) {
                        auto t = input_detail::timestamp(std::string_view(manifest).substr(first + 1,last - first - 1));
                        if (t != input_detail::bad_time) return t;
                    }
                }
            }
        }
        throw std::runtime_error("Invalid manifest " + std::string(key));
    };
    window = {instant("window_start"), instant("window_end")};
    auto parts = [&](const char* folder, std::vector<Chunk>& dest) {
        for (const auto& entry : std::filesystem::directory_iterator(root / folder))
            if (entry.path().extension() == ".csv" && entry.is_regular_file()) add(entry.path(),dest);
        // Preserve adjacent ranges from one file so the kernel can read ahead.
        std::stable_sort(dest.begin(), dest.end(), [](Chunk a, Chunk b) { return a.size() > b.size(); });
    };
    parts("events",events);
    parts("clicks",clicks);
    parts("links",links);
    add(root / "products.csv",products);
    add(root / "campaigns.csv",campaigns);
}
InputFiles::~InputFiles() = default;
}
