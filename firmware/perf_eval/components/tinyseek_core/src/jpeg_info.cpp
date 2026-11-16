#include "tinyseek_core/jpeg_info.hpp"

namespace tinyseek {

bool jpeg_info(const uint8_t *d, size_t len, JpegInfo &info) {
  if (len < 4 || d[0] != 0xFF || d[1] != 0xD8) return false;  // SOI
  size_t i = 2;
  while (i + 4 <= len) {
    if (d[i] != 0xFF) return false;
    const uint8_t marker = d[i + 1];
    if (marker == 0xFF) {  // fill byte before a marker
      ++i;
      continue;
    }
    if (marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) {  // markers without a length
      i += 2;
      continue;
    }
    const size_t seg = (size_t(d[i + 2]) << 8) | d[i + 3];  // includes its own 2 length bytes
    if (seg < 2) return false;
    // SOF0..SOF15 except DHT (C4), JPG (C8) and DAC (CC).
    if (marker >= 0xC0 && marker <= 0xCF && marker != 0xC4 && marker != 0xC8 && marker != 0xCC) {
      if (i + 2 + seg > len || seg < 8) return false;
      info.height = (d[i + 5] << 8) | d[i + 6];
      info.width = (d[i + 7] << 8) | d[i + 8];
      info.components = d[i + 9];
      info.progressive = marker == 0xC2 || marker == 0xC6 || marker == 0xCA || marker == 0xCE;
      return true;
    }
    if (marker == 0xDA || marker == 0xD9) return false;  // scan or end before any frame header
    i += 2 + seg;
  }
  return false;
}

}  // namespace tinyseek
