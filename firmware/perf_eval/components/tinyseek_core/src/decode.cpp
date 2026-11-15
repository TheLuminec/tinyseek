#include "tinyseek_core/decode.hpp"

#include <algorithm>
#include <cmath>

namespace tinyseek {

namespace {

struct Peak {
  float score;
  int flat;  // class * H * W + cell, the index torch.topk sees
};

}  // namespace

void decode(const int8_t *out, const DecodeParams &p, const Letterbox &lb,
            std::vector<Detection> &dets) {
  const int H = p.grid_h, W = p.grid_w, C = p.num_classes + 4;
  const float unit = std::ldexp(1.0f, p.exponent);
  auto at = [&](int y, int x, int ch) { return out[(y * W + x) * C + ch]; };

  // Peaks are tested on the int8 logits: sigmoid is strictly increasing and the 256
  // int8 levels map to distinct float32 scores, so `q >= every neighbour` is exactly
  // `heat == max_pool2d(heat, 3, 1, 1)` and the sigmoid is only needed for candidates.
  std::vector<Peak> peaks;
  for (int c = 0; c < p.num_classes; ++c) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const int8_t q = at(y, x, c);
        bool peak = true;
        for (int dy = -1; dy <= 1 && peak; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            const int yy = y + dy, xx = x + dx;
            if (yy < 0 || yy >= H || xx < 0 || xx >= W) continue;  // -inf padding
            if (at(yy, xx, c) > q) {
              peak = false;
              break;
            }
          }
        }
        if (!peak) continue;
        const float score = 1.0f / (1.0f + std::exp(-(q * unit)));
        peaks.push_back({score, c * H * W + y * W + x});
      }
    }
  }
  // topk(max_detections) over peak scores (non-peaks score 0 and never pass a positive
  // threshold), then the threshold.
  std::stable_sort(peaks.begin(), peaks.end(),
                   [](const Peak &a, const Peak &b) { return a.score > b.score; });
  if (static_cast<int>(peaks.size()) > p.max_detections) peaks.resize(p.max_detections);

  const int nc = p.num_classes;
  for (const Peak &pk : peaks) {
    if (!(pk.score >= p.score_threshold)) continue;
    const int cls = pk.flat / (H * W);
    const int cell = pk.flat % (H * W);
    const int cy = cell / W, cx = cell % W;
    const float bw = std::exp(at(cy, cx, nc + 0) * unit);
    const float bh = std::exp(at(cy, cx, nc + 1) * unit);
    const float ox = cx + at(cy, cx, nc + 2) * unit;
    const float oy = cy + at(cy, cx, nc + 3) * unit;
    // Same operation order as decode() (float32), then Letterbox.inverse.
    const float x0 = (ox - bw / 2) * p.stride, y0 = (oy - bh / 2) * p.stride;
    const float x1 = (ox + bw / 2) * p.stride, y1 = (oy + bh / 2) * p.stride;
    const float sx = static_cast<float>(lb.scale);
    const float X0 = (x0 - lb.pad_x) / sx, Y0 = (y0 - lb.pad_y) / sx;
    const float X1 = (x1 - lb.pad_x) / sx, Y1 = (y1 - lb.pad_y) / sx;
    dets.push_back({X0, Y0, X1 - X0, Y1 - Y0, pk.score, cls});
  }
}

}  // namespace tinyseek
