#include "tinyseek_core/preprocess.hpp"

#include <algorithm>
#include <cmath>

namespace tinyseek {

namespace {

// Pillow: #define PRECISION_BITS (32 - 8 - 2)
constexpr int kPrecisionBits = 32 - 8 - 2;

double bilinear_filter(double x) {
  if (x < 0.0) x = -x;
  if (x < 1.0) return 1.0 - x;
  return 0.0;
}
constexpr double kBilinearSupport = 1.0;

// Pillow's clip8: the accumulator carries 2^(PRECISION_BITS-1) for rounding, then shifts.
inline uint8_t clip8(int32_t in) {
  int32_t v = in >> kPrecisionBits;  // arithmetic shift, as in Pillow
  if (v < 0) return 0;
  if (v > 255) return 255;
  return static_cast<uint8_t>(v);
}

}  // namespace

Letterbox compute_letterbox(int src_w, int src_h, int size) {
  Letterbox lb;
  lb.scale = static_cast<double>(size) * 1.0 / std::max(src_w, src_h);
  lb.resized_w = std::max(1, static_cast<int>(std::nearbyint(src_w * lb.scale)));
  lb.resized_h = std::max(1, static_cast<int>(std::nearbyint(src_h * lb.scale)));
  lb.pad_x = static_cast<int>(std::nearbyint((size - lb.resized_w) * 0.5));
  lb.pad_y = static_cast<int>(std::nearbyint((size - lb.resized_h) * 0.5));
  return lb;
}

ResampleAxis bilinear_axis(int in_size, int out_size) {
  ResampleAxis ax;
  // Pillow passes the box edges as float; (in1 - in0) is exact for these sizes.
  const float in0 = 0.0f, in1 = static_cast<float>(in_size);
  double scale = static_cast<double>(in1 - in0) / out_size;
  double filterscale = scale < 1.0 ? 1.0 : scale;
  double support = kBilinearSupport * filterscale;
  ax.ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
  ax.bounds.assign(out_size * 2, 0);
  ax.kk.assign(static_cast<size_t>(out_size) * ax.ksize, 0);
  std::vector<double> k(ax.ksize);
  for (int xx = 0; xx < out_size; ++xx) {
    double center = in0 + (xx + 0.5) * scale;
    double ww = 0.0;
    double ss = 1.0 / filterscale;
    int xmin = static_cast<int>(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    int xmax = static_cast<int>(center + support + 0.5);
    if (xmax > in_size) xmax = in_size;
    xmax -= xmin;
    for (int x = 0; x < xmax; ++x) {
      double w = bilinear_filter((x + xmin - center + 0.5) * ss);
      k[x] = w;
      ww += w;
    }
    for (int x = 0; x < xmax; ++x) {
      if (ww != 0.0) k[x] /= ww;
    }
    for (int x = xmax; x < ax.ksize; ++x) k[x] = 0.0;
    // normalize_coeffs_8bpc: round half away from zero into fixed point.
    for (int x = 0; x < ax.ksize; ++x) {
      double v = k[x] * (1 << kPrecisionBits);
      ax.kk[static_cast<size_t>(xx) * ax.ksize + x] =
          static_cast<int32_t>(k[x] < 0 ? -0.5 + v : 0.5 + v);
    }
    ax.bounds[xx * 2 + 0] = xmin;
    ax.bounds[xx * 2 + 1] = xmax;
  }
  return ax;
}

int8_t quantize_int8(float x, int exponent) {
  // Division by a power of two is exact in float32, so ldexp gives the same value as
  // esp-ppq's x / scale; floor(t + 0.5) is its ROUND_HALF_UP.
  float t = std::ldexp(x, -exponent);
  float r = std::floor(t + 0.5f);
  if (r < -128.0f) return -128;
  if (r > 127.0f) return 127;
  return static_cast<int8_t>(r);
}

void make_plan(PreprocessPlan &plan, int src_w, int src_h, int size, int exponent,
               const NormalizeParams &norm) {
  plan.src_w = src_w;
  plan.src_h = src_h;
  plan.size = size;
  plan.exponent = exponent;
  plan.lb = compute_letterbox(src_w, src_h, size);
  plan.horiz = bilinear_axis(src_w, plan.lb.resized_w);
  plan.vert = bilinear_axis(src_h, plan.lb.resized_h);
  // Pillow runs the horizontal pass only over the rows the vertical pass will read.
  const int h = plan.lb.resized_h;
  plan.ybox_first = plan.vert.bounds[0];
  int ybox_last = plan.vert.bounds[h * 2 - 2] + plan.vert.bounds[h * 2 - 1];
  plan.ybox_rows = ybox_last - plan.ybox_first;
  for (int c = 0; c < 3; ++c) {
    for (int v = 0; v < 256; ++v) {
      // float32 throughout, the same operations and order as numpy in to_tensor.
      float x = static_cast<float>(v) / 255.0f;
      x = (x - norm.mean[c]) / norm.std[c];
      plan.lut[c][v] = quantize_int8(x, exponent);
    }
  }
}

size_t scratch_bytes(const PreprocessPlan &plan) {
  return static_cast<size_t>(plan.ybox_rows) * plan.lb.resized_w * 3;
}

void preprocess(const PreprocessPlan &plan, const uint8_t *rgb, uint8_t *scratch, int8_t *out) {
  const int nw = plan.lb.resized_w, nh = plan.lb.resized_h, size = plan.size;
  const bool need_h = nw != plan.src_w;
  const bool need_v = nh != plan.src_h;
  const int32_t half = 1 << (kPrecisionBits - 1);

  // Horizontal pass: source rows [ybox_first, ybox_first + ybox_rows) -> scratch (nw wide).
  const uint8_t *temp = rgb;
  int temp_row0 = 0;  // source row index held in temp row 0
  if (need_h) {
    const int ks = plan.horiz.ksize;
    for (int yy = 0; yy < plan.ybox_rows; ++yy) {
      const uint8_t *row = rgb + static_cast<size_t>(yy + plan.ybox_first) * plan.src_w * 3;
      uint8_t *dst = scratch + static_cast<size_t>(yy) * nw * 3;
      for (int xx = 0; xx < nw; ++xx) {
        const int xmin = plan.horiz.bounds[xx * 2], xmax = plan.horiz.bounds[xx * 2 + 1];
        const int32_t *k = &plan.horiz.kk[static_cast<size_t>(xx) * ks];
        int32_t s0 = half, s1 = half, s2 = half;
        const uint8_t *p = row + xmin * 3;
        for (int x = 0; x < xmax; ++x, p += 3) {
          s0 += p[0] * k[x];
          s1 += p[1] * k[x];
          s2 += p[2] * k[x];
        }
        dst[xx * 3 + 0] = clip8(s0);
        dst[xx * 3 + 1] = clip8(s1);
        dst[xx * 3 + 2] = clip8(s2);
      }
    }
    temp = scratch;
    temp_row0 = plan.ybox_first;
  }
  const int temp_w = need_h ? nw : plan.src_w;

  // Pad colour everywhere, then the resized image at (pad_x, pad_y).
  const int8_t pad[3] = {plan.lut[0][114], plan.lut[1][114], plan.lut[2][114]};
  for (int i = 0; i < size * size; ++i) {
    out[i * 3 + 0] = pad[0];
    out[i * 3 + 1] = pad[1];
    out[i * 3 + 2] = pad[2];
  }

  // Vertical pass straight into the quantized canvas (no second uint8 buffer).
  const int ks = plan.vert.ksize;
  for (int yy = 0; yy < nh; ++yy) {
    int8_t *dst = out + (static_cast<size_t>(yy + plan.lb.pad_y) * size + plan.lb.pad_x) * 3;
    if (!need_v) {
      const uint8_t *src = temp + static_cast<size_t>(yy - temp_row0) * temp_w * 3;
      for (int xx = 0; xx < nw; ++xx) {
        dst[xx * 3 + 0] = plan.lut[0][src[xx * 3 + 0]];
        dst[xx * 3 + 1] = plan.lut[1][src[xx * 3 + 1]];
        dst[xx * 3 + 2] = plan.lut[2][src[xx * 3 + 2]];
      }
      continue;
    }
    const int ymin = plan.vert.bounds[yy * 2] - temp_row0, ymax = plan.vert.bounds[yy * 2 + 1];
    const int32_t *k = &plan.vert.kk[static_cast<size_t>(yy) * ks];
    for (int xx = 0; xx < nw; ++xx) {
      int32_t s0 = half, s1 = half, s2 = half;
      const uint8_t *p = temp + (static_cast<size_t>(ymin) * temp_w + xx) * 3;
      const size_t step = static_cast<size_t>(temp_w) * 3;
      for (int y = 0; y < ymax; ++y, p += step) {
        s0 += p[0] * k[y];
        s1 += p[1] * k[y];
        s2 += p[2] * k[y];
      }
      dst[xx * 3 + 0] = plan.lut[0][clip8(s0)];
      dst[xx * 3 + 1] = plan.lut[1][clip8(s1)];
      dst[xx * 3 + 2] = plan.lut[2][clip8(s2)];
    }
  }
}

}  // namespace tinyseek
