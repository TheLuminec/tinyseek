// Arduino-only build settings for the evaluation sketch (hand-written, not generated).
// Everything else in src/ is a generated copy of firmware/perf_eval; see README.md.
#pragma once

#include "esp_arduino_version.h"

// Where dl::Model comes from.
//   0 (default): the esp-dl that arduino-esp32 core 3.3.12 and later ship precompiled
//                (esp-dl 3.3.11 in core 3.3.12). No extra library to install.
//   1: the community "espdlx" library (Library Manager), an Arduino port of esp-dl 3.3.11,
//      for cores before 3.3.12. Do not use it on 3.3.12+: the core already links esp-dl,
//      and two copies of the same symbols may clash.
#define TINYSEEK_USE_ESPDLX 0

#if TINYSEEK_USE_ESPDLX
#define TINYSEEK_ARDUINO_ESPDL "esp-dl 3.3.11 via the espdlx library"
#else
#if ESP_ARDUINO_VERSION < ESP_ARDUINO_VERSION_VAL(3, 3, 12)
#error "This sketch needs arduino-esp32 core 3.3.12 or later (it ships esp-dl), or set TINYSEEK_USE_ESPDLX to 1"
#endif
// The core does not expose the esp-dl version; 3.3.12 bundles 3.3.11 (its versions.txt).
#define TINYSEEK_ARDUINO_ESPDL "esp-dl bundled with core " ESP_ARDUINO_VERSION_STR
#endif

// JPEG decoder for bench.jpg, the parity JPEGs and camera frames.
//   0 (default): esp32-camera's fmt2rgb888 (img_converters.h, bundled with the core), which
//                decodes with the chip's ROM TJpgDec. Not the decoder of the ESP-IDF build.
//   1: esp_new_jpeg, the same decoder as the ESP-IDF build. Core 3.3.12 links it, but this
//      sketch has not been built with it; try it if you need comparable jpeg_decode times.
#define TINYSEEK_ARDUINO_USE_ESP_NEW_JPEG 0
#define TINYSEEK_JPEG_VERSION "1.0.2 (bundled with core " ESP_ARDUINO_VERSION_STR ")"

// fmt2rgb888 byte order for JPEG input. In the esp32-camera that core 3.3.12 bundles
// (decoding through esp_jpeg since May 2025) it is R,G,B; older cameras gave B,G,R. If the
// parity check reports a large jpeg_input difference, or reds look blue, set this to 1.
#define TINYSEEK_FMT2RGB888_IS_BGR 0
#define TINYSEEK_ARDUINO_JPEG_DECODER "esp32-camera fmt2rgb888 (ROM TJpgDec)"
