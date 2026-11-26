#include "platform.hpp"

#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <strings.h>  // strcasecmp

#include <cstdarg>

#include "esp_heap_caps.h"
#include "esp_idf_version.h"

#include "sdkconfig.h"

#ifdef ARDUINO
#include <Arduino.h>
#include <SD_MMC.h>

#include "arduino_config.h"
#else
#include "driver/sdmmc_host.h"
#include "esp_private/esp_clk.h"
#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"
#endif

#define BOARD_FREENOVE_S3_CAM
#include "board_pins.h"

namespace perf {

void logln(const char *fmt, ...) {
  char buf[256];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
#ifdef ARDUINO
  Serial.println(buf);
#else
  printf("%s\n", buf);
#endif
}

// The settings that change latency and that the two builds do not share: the ESP-IDF
// build sets them in sdkconfig.defaults, the Arduino core fixes them in its prebuilt
// libraries. Read from sdkconfig.h so each result says what it ran with.
#ifndef CONFIG_ESP32S3_DATA_CACHE_SIZE
#define CONFIG_ESP32S3_DATA_CACHE_SIZE 0
#endif
#ifndef CONFIG_ESP32S3_DATA_CACHE_LINE_SIZE
#define CONFIG_ESP32S3_DATA_CACHE_LINE_SIZE 0
#endif
#ifndef CONFIG_ESP32S3_INSTRUCTION_CACHE_SIZE
#define CONFIG_ESP32S3_INSTRUCTION_CACHE_SIZE 0
#endif
#ifndef CONFIG_SPIRAM_SPEED
#define CONFIG_SPIRAM_SPEED 0
#endif
#if defined(CONFIG_COMPILER_OPTIMIZATION_PERF)
#define TINYSEEK_SDK_OPT "perf"
#elif defined(CONFIG_COMPILER_OPTIMIZATION_SIZE)
#define TINYSEEK_SDK_OPT "size"
#elif defined(CONFIG_COMPILER_OPTIMIZATION_DEBUG)
#define TINYSEEK_SDK_OPT "debug"
#else
#define TINYSEEK_SDK_OPT "none"
#endif

std::string sdk_settings() {
  char buf[160];
  snprintf(buf, sizeof(buf), "dcache=%dKB/%dB icache=%dKB spiram=%dMHz sdk_opt=%s",
           CONFIG_ESP32S3_DATA_CACHE_SIZE / 1024, CONFIG_ESP32S3_DATA_CACHE_LINE_SIZE,
           CONFIG_ESP32S3_INSTRUCTION_CACHE_SIZE / 1024, CONFIG_SPIRAM_SPEED, TINYSEEK_SDK_OPT);
  return buf;
}

#ifdef ARDUINO

BuildInfo build_info() {
  return {"arduino-esp32 " ESP_ARDUINO_VERSION_STR, esp_get_idf_version(), TINYSEEK_ARDUINO_ESPDL,
          "bundled with core " ESP_ARDUINO_VERSION_STR};
}

bool mount_sd_card() {
  // As in firmware/capture/capture.ino: the Arduino SD_MMC library mounts the card as a FAT
  // VFS at /sdcard, so the plain fopen()/opendir() calls below work unchanged. 20 MHz
  // (SDMMC_FREQ_DEFAULT) like the ESP-IDF build, so model load times compare; the
  // library's own default is high speed (40 MHz).
  SD_MMC.setPins(SD_PIN_CLK, SD_PIN_CMD, SD_PIN_D0);
  if (!SD_MMC.begin(kMount, true /* 1-bit */, false /* never format */, SDMMC_FREQ_DEFAULT, 5)) {
    logln("SD mount failed");
    return false;
  }
  if (SD_MMC.cardType() == CARD_NONE) return false;
  logln("SD card: %llu MB", static_cast<unsigned long long>(SD_MMC.cardSize() >> 20));
  return true;
}

int cpu_mhz() { return static_cast<int>(getCpuFrequencyMhz()); }

#else  // ESP-IDF

BuildInfo build_info() {
  return {"esp-idf", esp_get_idf_version(), TINYSEEK_ESPDL_VERSION, TINYSEEK_CAMERA_VERSION};
}

int cpu_mhz() { return esp_clk_cpu_freq() / 1000000; }

bool mount_sd_card() {
  // Same wiring as the capture sketch: SD_MMC, 1-bit, pins from board_pins.h. Based on
  // ESP-IDF v5.5.6 examples/storage/sd_card/sdmmc.
  esp_vfs_fat_sdmmc_mount_config_t mount_config = {};
  mount_config.format_if_mount_failed = false;  // never wipe the card
  mount_config.max_files = 5;
  mount_config.allocation_unit_size = 16 * 1024;
  sdmmc_host_t host = SDMMC_HOST_DEFAULT();
  sdmmc_slot_config_t slot = SDMMC_SLOT_CONFIG_DEFAULT();
  slot.width = 1;
  slot.clk = static_cast<gpio_num_t>(SD_PIN_CLK);
  slot.cmd = static_cast<gpio_num_t>(SD_PIN_CMD);
  slot.d0 = static_cast<gpio_num_t>(SD_PIN_D0);
  slot.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP;
  sdmmc_card_t *card = nullptr;
  esp_err_t err = esp_vfs_fat_sdmmc_mount(kMount, &host, &slot, &mount_config, &card);
  if (err != ESP_OK) {
    logln("SD mount failed: %s", esp_err_to_name(err));
    return false;
  }
  sdmmc_card_print_info(stdout, card);
  return true;
}

#endif  // ARDUINO

void *alloc_psram(size_t bytes) { return heap_caps_aligned_alloc(16, bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT); }
void *alloc_internal(size_t bytes) {
  return heap_caps_aligned_alloc(16, bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
}
void free_buffer(void *p) { heap_caps_free(p); }

uint8_t *read_file(const std::string &path, size_t *size) {
  FILE *f = fopen(path.c_str(), "rb");
  if (!f) return nullptr;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  uint8_t *buf = n > 0 ? static_cast<uint8_t *>(alloc_psram(n)) : nullptr;
  if (buf && fread(buf, 1, n, f) != static_cast<size_t>(n)) {
    free_buffer(buf);
    buf = nullptr;
  }
  fclose(f);
  if (buf) *size = static_cast<size_t>(n);
  return buf;
}

bool write_file(const std::string &path, const void *data, size_t size) {
  FILE *f = fopen(path.c_str(), "wb");
  if (!f) return false;
  bool ok = fwrite(data, 1, size, f) == size;
  return fclose(f) == 0 && ok;
}

bool append_line(const std::string &path, const std::string &line) {
  FILE *f = fopen(path.c_str(), "a");
  if (!f) return false;
  fputs(line.c_str(), f);
  fputc('\n', f);
  return fclose(f) == 0;
}

bool file_exists(const std::string &path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0;
}

bool make_dir(const std::string &path) { return file_exists(path) || mkdir(path.c_str(), 0775) == 0; }

std::vector<std::string> list_dir(const std::string &dir, const char *suffix) {
  std::vector<std::string> names;
  DIR *d = opendir(dir.c_str());
  if (!d) return names;
  const size_t sl = strlen(suffix);
  for (struct dirent *e = readdir(d); e; e = readdir(d)) {
    std::string n = e->d_name;
    // Case-insensitive: FAT may report an 8.3 name in upper case ("00001.JPG").
    if (n.size() > sl && strcasecmp(n.c_str() + n.size() - sl, suffix) == 0) names.push_back(n);
  }
  closedir(d);
  std::sort(names.begin(), names.end());
  return names;
}

std::vector<std::string> read_lines(const std::string &path) {
  std::vector<std::string> lines;
  FILE *f = fopen(path.c_str(), "r");
  if (!f) return lines;
  char buf[256];
  while (fgets(buf, sizeof(buf), f)) {
    std::string s = buf;
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
    if (!s.empty() && s[0] != '#') lines.push_back(s);
  }
  fclose(f);
  return lines;
}

std::map<std::string, std::string> read_kv(const std::string &path) {
  std::map<std::string, std::string> kv;
  for (const std::string &line : read_lines(path)) {
    size_t eq = line.find('=');
    if (eq != std::string::npos) kv[line.substr(0, eq)] = line.substr(eq + 1);
  }
  return kv;
}

int kv_int(const std::map<std::string, std::string> &kv, const char *key, int fallback) {
  auto it = kv.find(key);
  return it == kv.end() ? fallback : static_cast<int>(strtol(it->second.c_str(), nullptr, 10));
}

float kv_float(const std::map<std::string, std::string> &kv, const char *key, float fallback) {
  auto it = kv.find(key);
  return it == kv.end() ? fallback : strtof(it->second.c_str(), nullptr);
}

std::string next_run_dir(const std::string &results) {
  make_dir(results);
  int highest = 0;
  DIR *d = opendir(results.c_str());
  if (d) {
    for (struct dirent *e = readdir(d); e; e = readdir(d)) {
      int n = 0;
      if (sscanf(e->d_name, "run_%d", &n) == 1) highest = std::max(highest, n);
    }
    closedir(d);
  }
  char name[32];
  snprintf(name, sizeof(name), "/run_%03d", highest + 1);
  return results + name;
}

MemSnapshot mem_snapshot() {
  const uint32_t in = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT, ps = MALLOC_CAP_SPIRAM;
  return {heap_caps_get_free_size(in), heap_caps_get_minimum_free_size(in), heap_caps_get_largest_free_block(in),
          heap_caps_get_free_size(ps), heap_caps_get_minimum_free_size(ps), heap_caps_get_largest_free_block(ps)};
}

Stats summarize(const std::vector<int64_t> &d) {
  Stats s;
  s.n = static_cast<int>(d.size());
  if (d.empty()) return s;
  double sum = 0, lo = 1e300, hi = 0;
  for (int64_t v : d) {
    double ms = v / 1000.0;
    sum += ms;
    lo = std::min(lo, ms);
    hi = std::max(hi, ms);
  }
  s.mean_ms = sum / s.n;
  double ss = 0;
  for (int64_t v : d) ss += (v / 1000.0 - s.mean_ms) * (v / 1000.0 - s.mean_ms);
  s.std_ms = s.n > 1 ? std::sqrt(ss / (s.n - 1)) : 0.0;  // sample standard deviation
  s.min_ms = lo;
  s.max_ms = hi;
  return s;
}

}  // namespace perf
