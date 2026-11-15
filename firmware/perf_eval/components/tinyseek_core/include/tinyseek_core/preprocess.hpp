// Frame -> model input, bit-exact with the PC path (portable C++, no ESP headers).
//
// The PC evaluates with tinyseek.data.transforms.letterbox (Pillow Image.resize with
// BILINEAR, which on downscaling is a *filtered* resample whose support grows with the
// scale factor, not 4-tap bilinear interpolation), then tinyseek.data.dataset.to_tensor
// (float32 /255, ImageNet mean/std), then esp-ppq quantizes the input tensor with a
// power-of-two scale and ROUND_HALF_UP (floor(x + 0.5)), saturating to int8. This file
// reproduces each step exactly, including Pillow's 22-bit fixed-point coefficients and
// its uint8 intermediate after the horizontal pass, so the int8 tensor matches byte for
// byte. ESP-DL tensors are NHWC, so the output is HWC (batch 1).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tinyseek {

// Same convention as tinyseek.data.transforms.Letterbox: x' = x * scale + pad_x.
struct Letterbox {
  double scale = 1.0;
  int pad_x = 0;
  int pad_y = 0;
  int resized_w = 0;
  int resized_h = 0;
};

// letterbox(image, size) with zoom 1 and centered shift. Python's round() is
// round-half-to-even, so this uses nearbyint under the default rounding mode.
Letterbox compute_letterbox(int src_w, int src_h, int size);

// Pillow's separable resampling coefficients for one axis (precompute_coeffs +
// normalize_coeffs_8bpc in libImaging/Resample.c).
struct ResampleAxis {
  int ksize = 0;
  std::vector<int> bounds;   // per output pixel: first source index, tap count
  std::vector<int32_t> kk;   // per output pixel: ksize fixed-point weights
};

ResampleAxis bilinear_axis(int in_size, int out_size);

struct NormalizeParams {
  float mean[3] = {0.485f, 0.456f, 0.406f};
  float std[3] = {0.229f, 0.224f, 0.225f};
  uint8_t pad_value = 114;
};

// Everything that depends only on (source size, model size, input exponent): built once
// per resolution so the per-frame cost is just the resample and the table lookups.
struct PreprocessPlan {
  int src_w = 0, src_h = 0, size = 0, exponent = 0;
  Letterbox lb;
  ResampleAxis horiz, vert;
  int ybox_first = 0, ybox_rows = 0;  // source rows the vertical pass reads
  int8_t lut[3][256];                 // uint8 pixel -> quantized normalized value
};

// `exponent` is the model input tensor's power-of-two exponent (value = q * 2^exponent).
void make_plan(PreprocessPlan &plan, int src_w, int src_h, int size, int exponent,
               const NormalizeParams &norm = NormalizeParams());

// Bytes of scratch `preprocess` needs (horizontal-pass rows, uint8 RGB).
size_t scratch_bytes(const PreprocessPlan &plan);

// rgb: src_h x src_w x 3, R,G,B order, row-major. out: size x size x 3 int8 (HWC).
// scratch: at least scratch_bytes(plan) bytes. Neither buffer may alias.
void preprocess(const PreprocessPlan &plan, const uint8_t *rgb, uint8_t *scratch, int8_t *out);

// The float->int8 step on its own, mirroring esp-ppq: clamp(floor(x / 2^exp + 0.5)).
int8_t quantize_int8(float x, int exponent);

}  // namespace tinyseek
