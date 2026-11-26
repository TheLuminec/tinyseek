// Model output -> boxes in original-frame pixels (portable C++, no ESP headers).
//
// Mirrors tinyseek.train.targets.decode followed by tinyseek.eval.evaluate.predict:
// sigmoid on the heatmap logits, a cell is a peak when it equals the 3x3 max of its
// neighbourhood (ties count, as in `heat == max_pool2d(heat)`), the top `max_detections`
// peaks over all classes are kept, then those scoring >= `score_threshold`.
//   center = (cell + offset) * stride,  wh = exp(log_wh) * stride,
//   original = (letterboxed - pad) / scale.
// The output tensor is the ESP-DL int8 (or int16) tensor, NHWC [1, G, G, num_classes + 4],
// with channels [heatmap_logit..., log_w, log_h, off_x, off_y] and one power-of-two exponent.
#pragma once

#include <cstdint>
#include <vector>

// Same-directory include, so the header also works when copied into an Arduino sketch.
#include "preprocess.hpp"

namespace tinyseek {

struct Detection {
  float x, y, w, h;  // original-frame pixels, COCO xywh
  float score;
  int cls;
};

// Upper bound on log_w/log_h (in cells) before exp: e^16 cells is far beyond any frame, and
// a real model's outputs never reach it, so the clamp only removes inf from degenerate
// outputs. tinyseek.deploy.reference.MAX_LOG_WH is the same value.
constexpr float kMaxLogWH = 16.0f;

struct DecodeParams {
  int grid_h = 0, grid_w = 0;
  int num_classes = 1;
  int stride = 8;
  int exponent = 0;  // output tensor exponent: value = q * 2^exponent
  float score_threshold = 0.05f;
  int max_detections = 20;
};

// Appends detections, highest score first. The output tensor is int8, or int16 for a
// w16a16 model (`exponent` is then the int16 tensor's exponent).
void decode(const int8_t *out, const DecodeParams &p, const Letterbox &lb, std::vector<Detection> &dets);
void decode(const int16_t *out, const DecodeParams &p, const Letterbox &lb, std::vector<Detection> &dets);

}  // namespace tinyseek
