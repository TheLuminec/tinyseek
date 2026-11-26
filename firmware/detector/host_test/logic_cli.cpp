// Host driver for the detector's portable logic (src/detector_logic.cpp), used by
// tests/test_deploy_detector.py. Not part of the sketch: the Arduino IDE compiles only
// the sketch folder's top level and src/.
//
//   logic_cli settings <config.txt>         prints settings_json, then one warning per line
//   logic_cli prompt <typed>                prints the normalized name and 1/0 for valid
//   logic_cli with_prompt <config.txt> <p>  prints the config text with prompt=<p>
//   logic_cli meta <model.txt>              prints the parsed sidecar as key=value
//   logic_cli frame                         prints a frame_json and a replay_json line
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "../src/detector_logic.hpp"

static std::string read_text(const char *path) {
  std::ifstream f(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(f), {});
}

// The same parsing as perf::read_kv on the board (platform.cpp): '#' lines skipped,
// trailing whitespace stripped, split at the first '='.
static det::KeyValues read_kv(const char *path) {
  det::KeyValues kv;
  std::string text = read_text(path), line;
  for (size_t pos = 0; pos <= text.size(); ++pos) {
    if (pos == text.size() || text[pos] == '\n') {
      while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
      if (!line.empty() && line[0] != '#') {
        size_t eq = line.find('=');
        if (eq != std::string::npos) kv[line.substr(0, eq)] = line.substr(eq + 1);
      }
      line.clear();
    } else {
      line += text[pos];
    }
  }
  return kv;
}

int main(int argc, char **argv) {
  if (argc == 3 && !std::strcmp(argv[1], "settings")) {
    std::vector<std::string> warnings;
    det::Settings s = det::parse_settings(read_kv(argv[2]), warnings);
    std::printf("%s\n", det::settings_json(s).c_str());
    for (const auto &w : warnings) std::printf("%s\n", w.c_str());
    return 0;
  }
  if (argc == 3 && !std::strcmp(argv[1], "prompt")) {
    std::string p = det::normalize_prompt(argv[2]);
    std::printf("%s\n%d\n", p.c_str(), det::valid_prompt(p) ? 1 : 0);
    return 0;
  }
  if (argc == 4 && !std::strcmp(argv[1], "with_prompt")) {
    std::fputs(det::with_prompt(read_text(argv[2]), argv[3]).c_str(), stdout);
    return 0;
  }
  if (argc == 3 && !std::strcmp(argv[1], "meta")) {
    det::ModelMeta m = det::parse_model_meta(read_kv(argv[2]));
    std::printf("prompt=%s\ninput_size=%d\nstride=%d\nnum_classes=%d\nhas_exponents=%d\ninput_exponent=%d\n"
                "output_exponent=%d\nmean=%g,%g,%g\nstd=%g,%g,%g\npad_value=%d\nscore_threshold=%g\n"
                "max_detections=%d\ndetect_threshold=%g\nsha256=%s\ninput_dtype=%s\noutput_dtype=%s\n",
                m.prompt.c_str(), m.input_size, m.stride, m.num_classes, m.has_exponents ? 1 : 0, m.input_exponent,
                m.output_exponent, m.mean[0], m.mean[1], m.mean[2], m.std[0], m.std[1], m.std[2], m.pad_value,
                m.score_threshold, m.max_detections, m.detect_threshold, m.sha256.c_str(),
                m.input_dtype.c_str(), m.output_dtype.c_str());
    return 0;
  }
  if (argc == 2 && !std::strcmp(argv[1], "frame")) {
    std::vector<tinyseek::Detection> dets = {{100.25f, 200.5f, 50.0f, 80.125f, 0.91f, 0},
                                             {10.0f, 20.0f, 5.0f, 6.0f, 0.07f, 0}};
    det::FrameReport r;
    r.n = 7;
    r.prompt = "back\"pack";
    r.source = "replay";
    r.file = "00001.jpg";
    r.width = 800;
    r.height = 600;
    r.t.cap = 1.25;
    r.t.dec = 431.0;
    r.t.pre = 208.5;
    r.t.inf = 448.0;
    r.t.post = 0.31;
    r.t.lat = 1089.0;
    r.t.period = 1090.0;
    r.fps = 0.92;
    r.detect_threshold = 0.3f;
    r.dets = &dets;
    std::printf("%s\n", det::frame_json(r, 0.1f).c_str());
    std::printf("%s\n", det::replay_json(r).c_str());
    r.ok = false;
    r.error = "JPEG decode failed";
    dets.clear();
    std::printf("%s\n", det::frame_json(r, 0.1f).c_str());
    return 0;
  }
  std::fprintf(stderr, "usage: see the comment at the top of logic_cli.cpp\n");
  return 2;
}
