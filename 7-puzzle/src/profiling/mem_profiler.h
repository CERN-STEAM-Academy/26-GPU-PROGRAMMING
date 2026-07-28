#ifndef SRO_MEM_PROFILER
#define SRO_MEM_PROFILER

#include <thread>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

namespace sro {

// Profiling is opt-in: enabled only when GPUDECIDE_MEM_PROFILE is set.
// Off -> original per-op output, no sampler thread, zero overhead.
inline bool mem_profile_enabled() {
    static const bool on = std::getenv("GPUDECIDE_MEM_PROFILE") != nullptr;
    return on;
}

// Resident system memory in MB = MemTotal - MemAvailable (excludes reclaimable cache;
// on unified memory captures both device and host allocations, unlike cudaMemGetInfo
// which over-reports by counting page cache as "used").
inline size_t sys_used_mb() {
    FILE* f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    size_t total_kb = 0, avail_kb = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        sscanf(line, "MemTotal: %zu kB", &total_kb);
        sscanf(line, "MemAvailable: %zu kB", &avail_kb);
    }
    fclose(f);
    return (total_kb > avail_kb) ? (total_kb - avail_kb) / 1024 : 0;
}

inline std::string spill_dir_path() {
    if (const char* env = std::getenv("GPUDECIDE_SPILL_DIR")) return std::string(env);
    return "/tmp";
}

// Total MB of THIS process's spill files on disk (filtered by pid for precision).
inline size_t spilled_mb(const std::string& dir, const std::string& prefix) {
    DIR* d = opendir(dir.c_str());
    if (!d) return 0;
    size_t total = 0;
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
        if (strncmp(e->d_name, prefix.c_str(), prefix.size()) == 0) {
            std::string path = dir + "/" + e->d_name;
            struct stat st;
            if (stat(path.c_str(), &st) == 0) total += (size_t)st.st_size;
        }
    }
    closedir(d);
    return total / (1024 * 1024);
}

// Background sampler: peak resident, peak spilled, and peak total (resident+spilled
// at the same instant) across an operation. A no-op (no thread) when profiling is off.
class PeakMemSampler {
    bool                _enabled;
    std::atomic<bool>   _running{true};
    std::atomic<size_t> _used{0}, _spill{0}, _total{0};
    std::thread         _thread;
    std::string         _dir, _prefix;

    void update() {
        size_t u = sys_used_mb();
        size_t s = spilled_mb(_dir, _prefix);
        if (u     > _used.load (std::memory_order_relaxed)) _used.store (u,     std::memory_order_relaxed);
        if (s     > _spill.load(std::memory_order_relaxed)) _spill.store(s,     std::memory_order_relaxed);
        if (u + s > _total.load(std::memory_order_relaxed)) _total.store(u + s, std::memory_order_relaxed);
    }

public:
    explicit PeakMemSampler(unsigned interval_ms = 50)
        : _enabled(mem_profile_enabled()),
          _dir(spill_dir_path()),
          _prefix("gpudecide_spill_" + std::to_string(getpid()) + "_") {
        if (!_enabled) return;          // off: no thread, no overhead
        update();                       // seed so fast ops still report a value
        _thread = std::thread([this, interval_ms] {
            while (_running.load(std::memory_order_relaxed)) {
                update();
                std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
            }
        });
    }

    bool   enabled()       const { return _enabled; }
    size_t peak_used_mb()  const { return _used.load (std::memory_order_relaxed); }
    size_t peak_spill_mb() const { return _spill.load(std::memory_order_relaxed); }
    size_t peak_total_mb() const { return _total.load(std::memory_order_relaxed); }

    void stop() {
        if (!_enabled) return;          // off: nothing to stop
        if (_running.exchange(false, std::memory_order_relaxed) && _thread.joinable())
            _thread.join();
    }
    ~PeakMemSampler() { stop(); }

    PeakMemSampler(const PeakMemSampler&) = delete;
    PeakMemSampler& operator=(const PeakMemSampler&) = delete;
};

// Prints the per-op stats line. Profiling on -> peak resident/spilled/total;
// off -> the original used/free/total + irreducible line. The branch lives here,
// keeping bdd_operations.cu to a single call. `irreducible_fn` is a callable,
// evaluated ONLY in the non-profile path, so profiling never pays for it.
template <typename IrreducibleFn>
inline void print_op_stats(PeakMemSampler& sampler,
                           size_t free_bytes, size_t total_bytes,
                           uint64_t arc_nodes, float elapsed_ms,
                           uint64_t node_count, IrreducibleFn&& irreducible_fn) {
    if (sampler.enabled()) {
        sampler.stop();
        printf("PeakResident: %6zu MB, \tPeakSpilled: %7zu MB, \tPeakTotal: %7zu MB, "
               "\tArc nodes: %10lu, \tTime: %10.2fms \tNodes: %10lu\n",
               sampler.peak_used_mb(), sampler.peak_spill_mb(), sampler.peak_total_mb(),
               arc_nodes, elapsed_ms, node_count);
    } else {
        size_t used_bytes = total_bytes - free_bytes;
        printf("Used: %5zu MB, \tFree: %5zu MB, \tTotal: %5zu MB, "
               "\tArc nodes: %10lu, \tIrreducible nodes: %10lu, \tTime: %10.2fms \tNodes: %10lu\n",
               used_bytes / (1024 * 1024), free_bytes / (1024 * 1024), total_bytes / (1024 * 1024),
               arc_nodes, (uint64_t)irreducible_fn(), elapsed_ms, node_count);
    }
}

} // namespace sro
#endif // SRO_MEM_PROFILER