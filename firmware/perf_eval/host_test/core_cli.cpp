// Host driver for tinyseek_core, so the Python tests can check the C++ against the PC
// pipeline on this machine (tests/test_deploy_core.py builds and runs it).
//
//   core_cli preprocess <rgb888 file> <w> <h> <size> <exponent> <out.bin> [pillow|bilinear] [int16]
//       writes size*size*3 int8 (or int16, little-endian) HWC and prints
//       "scale pad_x pad_y resized_w resized_h"
//   core_cli decode <NHWC file> <grid> <num_classes> <stride> <exponent>
//                   <scale> <pad_x> <pad_y> <score_threshold> <max_detections> [int16]
//       reads an int8 (or int16) tensor; prints one "x y w h score cls" line per detection
//   core_cli jpeginfo <file.jpg>
//       prints "width height components progressive" (or "invalid", exit 1)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

#include "tinyseek_core/decode.hpp"
#include "tinyseek_core/jpeg_info.hpp"
#include "tinyseek_core/preprocess.hpp"

static std::vector<uint8_t> read_file(const char *path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "cannot read %s\n", path);
    std::exit(2);
  }
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), {});
}

static int run_preprocess(char **a) {
  const int w = std::atoi(a[1]), h = std::atoi(a[2]), size = std::atoi(a[3]);
  const int exponent = std::atoi(a[4]);
  std::vector<uint8_t> rgb = read_file(a[0]);
  if (rgb.size() != static_cast<size_t>(w) * h * 3) {
    std::fprintf(stderr, "expected %d bytes, got %zu\n", w * h * 3, rgb.size());
    return 2;
  }
  bool bilinear = false, wide = false;
  for (int i = 6; a[i]; ++i) {
    if (!std::strcmp(a[i], "bilinear")) {
      bilinear = true;
    } else if (!std::strcmp(a[i], "int16")) {
      wide = true;
    } else if (std::strcmp(a[i], "pillow")) {
      std::fprintf(stderr, "options are pillow|bilinear and int16\n");
      return 2;
    }
  }
  static tinyseek::PreprocessPlan plan;  // 768-byte LUT: keep it off the stack
  tinyseek::make_plan(plan, w, h, size, exponent, tinyseek::NormalizeParams(),
                      bilinear ? tinyseek::Resample::kBilinear : tinyseek::Resample::kPillow);
  std::vector<uint8_t> scratch(tinyseek::scratch_bytes(plan));
  const size_t n = static_cast<size_t>(size) * size * 3;
  if (wide) {
    std::vector<int16_t> out(n);
    tinyseek::preprocess(plan, rgb.data(), scratch.data(), out.data());
    std::ofstream(a[5], std::ios::binary).write(reinterpret_cast<const char *>(out.data()), n * 2);
  } else {
    std::vector<int8_t> out(n);
    tinyseek::preprocess(plan, rgb.data(), scratch.data(), out.data());
    std::ofstream(a[5], std::ios::binary).write(reinterpret_cast<const char *>(out.data()), n);
  }
  std::printf("%.17g %d %d %d %d\n", plan.lb.scale, plan.lb.pad_x, plan.lb.pad_y,
              plan.lb.resized_w, plan.lb.resized_h);
  return 0;
}

static int run_decode(char **a) {
  std::vector<uint8_t> raw = read_file(a[0]);
  tinyseek::DecodeParams p;
  p.grid_h = p.grid_w = std::atoi(a[1]);
  p.num_classes = std::atoi(a[2]);
  p.stride = std::atoi(a[3]);
  p.exponent = std::atoi(a[4]);
  tinyseek::Letterbox lb;
  lb.scale = std::atof(a[5]);
  lb.pad_x = std::atoi(a[6]);
  lb.pad_y = std::atoi(a[7]);
  p.score_threshold = static_cast<float>(std::atof(a[8]));
  p.max_detections = std::atoi(a[9]);
  const bool wide = a[10] && !std::strcmp(a[10], "int16");
  const size_t elem = wide ? 2 : 1;
  if (raw.size() != elem * p.grid_h * p.grid_w * (p.num_classes + 4)) {
    std::fprintf(stderr, "output tensor has %zu bytes, expected %zu\n", raw.size(),
                 elem * p.grid_h * p.grid_w * (p.num_classes + 4));
    return 2;
  }
  std::vector<tinyseek::Detection> dets;
  if (wide) {
    std::vector<int16_t> q(raw.size() / 2);
    std::memcpy(q.data(), raw.data(), raw.size());
    tinyseek::decode(q.data(), p, lb, dets);
  } else {
    tinyseek::decode(reinterpret_cast<const int8_t *>(raw.data()), p, lb, dets);
  }
  for (const auto &d : dets) std::printf("%.9g %.9g %.9g %.9g %.9g %d\n", d.x, d.y, d.w, d.h, d.score, d.cls);
  return 0;
}

static int run_jpeginfo(char **a) {
  std::vector<uint8_t> raw = read_file(a[0]);
  tinyseek::JpegInfo info;
  if (!tinyseek::jpeg_info(raw.data(), raw.size(), info)) {
    std::printf("invalid\n");
    return 1;
  }
  std::printf("%d %d %d %d\n", info.width, info.height, info.components, info.progressive ? 1 : 0);
  return 0;
}

int main(int argc, char **argv) {
  if (argc >= 8 && argc <= 10 && !std::strcmp(argv[1], "preprocess")) return run_preprocess(argv + 2);
  if ((argc == 12 || argc == 13) && !std::strcmp(argv[1], "decode")) return run_decode(argv + 2);
  if (argc == 3 && !std::strcmp(argv[1], "jpeginfo")) return run_jpeginfo(argv + 2);
  std::fprintf(stderr, "usage: see the comment at the top of core_cli.cpp\n");
  return 2;
}
