// OV2640 with the frozen data-collection settings, and on-board JPEG decoding.
#pragma once

#include <cstddef>
#include <cstdint>

namespace perf {

// Same sensor settings as firmware/capture/capture.ino (frozen 2026-11-06): SVGA JPEG q10,
// no flip/mirror, AWB auto, AE level 0, aec2 off, 4x gain ceiling; frames are dropped for
// `warmup_ms` so exposure and white balance settle, as in the capture sketch.
bool camera_init(uint32_t warmup_ms);
void camera_deinit();

// JPEG -> RGB888 (R,G,B byte order) with esp_new_jpeg. `rgb` must hold w*h*3 bytes and be
// 16-byte aligned (the decoder shifts columns on the S3 otherwise). Fails if the image is
// not exactly w x h, so a wrong frame size never reaches the preprocessing plan.
bool decode_jpeg(const uint8_t *jpeg, size_t len, uint8_t *rgb, int w, int h);

}  // namespace perf
