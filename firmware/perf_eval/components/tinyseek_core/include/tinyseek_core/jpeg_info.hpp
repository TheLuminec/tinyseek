// JPEG frame size from the header, without decoding (portable C++, no ESP headers).
//
// Both on-board decoders write into a caller buffer sized for 800x600, and the Arduino
// one (esp32-camera's fmt2rgb888) does not report the size it decoded, so the firmware
// checks the header first. Progressive JPEGs are flagged: the on-board decoders handle
// baseline only.
#pragma once

#include <cstddef>
#include <cstdint>

namespace tinyseek {

struct JpegInfo {
  int width = 0;
  int height = 0;
  int components = 0;
  bool progressive = false;
};

// Scans markers up to the first start-of-frame. False if the data is not a JPEG or no
// frame header is found within `len` bytes.
bool jpeg_info(const uint8_t *data, size_t len, JpegInfo &info);

}  // namespace tinyseek
