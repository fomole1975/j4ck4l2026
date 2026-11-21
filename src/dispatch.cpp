#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits.h>
#include <string>
#include <string_view>
#include <time.h>
#include <unistd.h>

int main(int, char** argv) {
    const bool avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("bmi") &&
                      __builtin_cpu_supports("bmi2") && __builtin_cpu_supports("popcnt") &&
                      __builtin_cpu_supports("pclmul");
    const bool avx512 = avx2 && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw");
    const bool vbmi2 = avx512 && __builtin_cpu_supports("avx512vbmi2");
    std::string_view variant = vbmi2 ? "vbmi2" : avx512 ? "avx512" : avx2 ? "avx2" : "generic";
    if (const char* requested = std::getenv("SHOPPING_ISA")) {
        const std::string_view name(requested);
        if (name == "generic" || (name == "avx2" && avx2) || (name == "avx512" && avx512) ||
            (name == "vbmi2" && vbmi2)) {
            variant = name;
        } else {
            std::fprintf(stderr, "Unsupported SHOPPING_ISA: %s\n", requested);
            return 1;
        }
    }
    char path[PATH_MAX];
    const auto size = readlink("/proc/self/exe", path, sizeof(path));
    if (size < 0 || size == sizeof(path)) {
        std::fprintf(stderr, "Cannot locate pipeline executable\n");
        return 1;
    }
    std::string binary(path, size);
    binary += '-';
    binary += variant;
    if (std::getenv("PROFILE")) {
        timespec now{};
        clock_gettime(CLOCK_REALTIME, &now);
        std::fprintf(stderr, "isa=%s wall_dispatch=%lld\n", variant.data(),
                     static_cast<long long>(now.tv_sec) * 1000000000LL + now.tv_nsec);
    }
    execv(binary.c_str(), argv);
    std::fprintf(stderr, "Cannot start %s: %s\n", binary.c_str(), std::strerror(errno));
    return 1;
}
