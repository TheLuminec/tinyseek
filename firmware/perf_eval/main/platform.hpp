// Board services for the benchmark: SD card, files, heap snapshots, timing statistics.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace perf {

constexpr const char *kMount = "/sdcard";
constexpr const char *kRoot = "/sdcard/tinyseek_eval";

bool mount_sd_card();

// Whole file into a PSRAM buffer (caller frees with free_buffer). nullptr on failure.
uint8_t *read_file(const std::string &path, size_t *size);
bool write_file(const std::string &path, const void *data, size_t size);
bool append_line(const std::string &path, const std::string &line);
bool file_exists(const std::string &path);
bool make_dir(const std::string &path);
std::vector<std::string> list_dir(const std::string &dir, const char *suffix);
std::vector<std::string> read_lines(const std::string &path);

// key=value file, as written by tinyseek.deploy.layout.write_kv. Missing file -> empty.
std::map<std::string, std::string> read_kv(const std::string &path);
int kv_int(const std::map<std::string, std::string> &kv, const char *key, int fallback);
float kv_float(const std::map<std::string, std::string> &kv, const char *key, float fallback);

// Next unused results/run_NNN, so a reboot never overwrites earlier results.
std::string next_run_dir(const std::string &results);

void *alloc_psram(size_t bytes);
void *alloc_internal(size_t bytes);
void free_buffer(void *p);

struct MemSnapshot {
  size_t internal_free, internal_min_free, internal_largest;
  size_t psram_free, psram_min_free, psram_largest;
};
MemSnapshot mem_snapshot();

inline int64_t now_us();

struct Stats {
  int n = 0;
  double mean_ms = 0, min_ms = 0, max_ms = 0, std_ms = 0;
};
Stats summarize(const std::vector<int64_t> &durations_us);

int cpu_mhz();

}  // namespace perf

#include "esp_timer.h"
inline int64_t perf::now_us() { return esp_timer_get_time(); }
