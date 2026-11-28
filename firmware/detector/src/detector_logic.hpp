// The detector's board-independent logic: config.txt settings, prompt names, the model
// sidecar and the one-line JSON reports. Portable C++ (no ESP or Arduino headers), so
// tests/test_deploy_detector.py builds it with g++ through host_test/logic_cli.cpp.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "tinyseek_core/decode.hpp"

namespace det {

using KeyValues = std::map<std::string, std::string>;

enum class Source : uint8_t { kCamera, kReplay };
enum class WifiMode : uint8_t { kOff, kSta, kAp };
enum class LedMode : uint8_t { kBoth, kOnboard, kExternal, kOff };

// /sdcard/tinyseek/config.txt. The defaults are the measured, parity-checked path
// (full-size TJpgDec decode, Pillow-exact resize, single-core inference, no pipelining).
struct Settings {
  std::string prompt;                 // prompt: the model file is models/<prompt>.espdl
  Source source = Source::kCamera;    // source: camera | replay
  int decode_scale = 0;               // decode_scale: JPEG decoded at 1/2^s (0..3)
  bool bilinear = false;              // resize: pillow | bilinear
  bool esp_new_jpeg = false;          // decoder: tjpgd | esp_new_jpeg (full scale only)
  bool multicore = false;             // multicore: 0 | 1 (ESP-DL RUNTIME_MODE_MULTI_CORE)
  int internal_kb = 0;                // internal_kb: ESP-DL activations in internal RAM
  bool pipeline = false;              // pipeline: 0 | 1 (capture + decode on core 0)
  LedMode led = LedMode::kBoth;       // led: both | onboard | external | off
  float print_min_score = 0.1f;       // print_min_score: serial/web list detections >= this
  WifiMode wifi = WifiMode::kOff;     // wifi: off | sta | ap
  std::string wifi_ssid;              // wifi_ssid (sta)
  std::string wifi_password;          // wifi_password (sta)
  std::string ap_ssid = "tinyseek";   // ap_ssid (ap)
  std::string ap_password;            // ap_password (ap, WPA2: 8-63 characters, required)
  std::string hostname = "tinyseek";  // hostname (sta)
  int http_port = 80;                 // http_port
  int camera_warmup_ms = 4000;        // camera_warmup_ms
  std::string replay_dir = "/sdcard/tinyseek/replay";  // replay_dir
  int replay_limit = 0;               // replay_limit: 0 = every file
};

// Unknown keys and invalid values leave the default and add a message to `warnings`.
Settings parse_settings(const KeyValues &kv, std::vector<std::string> &warnings);

// The settings as a JSON object, without the Wi-Fi password or AP password.
std::string settings_json(const Settings &s);

// What a user typed -> a model name: trimmed, lower case, spaces and '-' become '_'.
std::string normalize_prompt(const std::string &typed);
// [a-z0-9_]{1,32}: safe as a file name on FAT and in a URL.
bool valid_prompt(const std::string &name);

// config.txt text with its prompt line replaced (or appended); every other line kept
// byte for byte. Prompt lines are matched by trimmed key ("prompt = cup" too), and extra
// prompt lines are dropped.
std::string with_prompt(const std::string &config_text, const std::string &prompt);

// What to write when saving a new prompt: false (do not save) when config.txt exists but
// could not be read, since writing would drop every other setting. A missing file gives
// a config holding just the prompt.
bool config_for_save(bool exists, bool read_ok, const std::string &text, const std::string &prompt,
                     std::string *out);

// At boot, after a save that was interrupted (see persist_prompt in detector_app.cpp).
enum class ConfigRecovery : uint8_t { kNone, kRestoreTmp, kDeleteTmp };
ConfigRecovery config_recovery(bool config_exists, bool tmp_exists);

// A requested prompt against the current one: switch, nothing to do, or refuse because
// a replay is running (its results file must hold one model only).
enum class PromptAction : uint8_t { kNone, kSwitch, kRefuseReplay };
PromptAction prompt_action(const std::string &pending, const std::string &current, bool replay_active);

// models/<prompt>.txt, written by tinyseek.deploy.detector_card.
struct ModelMeta {
  std::string prompt;
  // Model input in pixels, 0 = not given (the loaded model's own shape is what is used; this
  // only detects a stale card). A square model's sidecar has input_size=S, which sets both;
  // a rectangular one has input_width=W and input_height=H instead. If a file has both
  // forms, input_width/input_height win.
  int input_w = 0, input_h = 0;
  int stride = 8, num_classes = 1;
  int input_exponent = 0, output_exponent = 0;  // 0 = not given; the model's own are used
  bool has_exponents = false;
  float mean[3] = {0.485f, 0.456f, 0.406f};
  float std[3] = {0.229f, 0.224f, 0.225f};
  int pad_value = 114;
  float score_threshold = 0.05f;   // decode threshold (as in the PC evaluation, for AP)
  int max_detections = 20;
  float detect_threshold = 0.3f;   // "the object is there": LED, web highlight
  std::string input_dtype, output_dtype;  // "INT8" / "INT16"; empty = not given
  std::string sha256, run;
};
ModelMeta parse_model_meta(const KeyValues &kv);

// How the loaded model's shape is reported, so a square model's lines are what they have
// always been and a rectangular one cannot be mistaken for a square.
//   model_shape_json: `"input":192,"grid":24` (square), or
//                     `"input_w":256,"input_h":192,"grid_w":32,"grid_h":24` (rectangular);
//   model_shape_kv:   `input_size=192`, or `input_width=256\ninput_height=192` (run.txt, the
//                     same keys as the sidecar).
std::string model_shape_json(int in_w, int in_h, int grid_w, int grid_h);
std::string model_shape_kv(int in_w, int in_h);

struct StageTimes {
  double cap = 0, dec = 0, pre = 0, inf = 0, post = 0;  // ms per stage
  double lat = 0;     // capture start to boxes, ms
  double period = 0;  // time since the previous frame finished, ms
};

struct FrameReport {
  uint32_t n = 0;
  std::string prompt;
  std::string source;  // "camera" or "replay"
  std::string file;    // replay file name, else empty
  int width = 0, height = 0;  // JPEG frame size (boxes are in these pixels)
  bool ok = true;
  std::string error;
  StageTimes t;
  double fps = 0;
  float detect_threshold = 0.3f;
  const std::vector<tinyseek::Detection> *dets = nullptr;
};

// One JSON object on one line, no trailing newline.
//   serial/web: {"type":"frame",...} with detections >= min_score, coarse precision;
//   replay results: every detection, full precision (for AP on the PC).
std::string frame_json(const FrameReport &r, float min_score);
std::string replay_json(const FrameReport &r);

std::string json_escape(const std::string &s);
std::string json_string_array(const std::vector<std::string> &items);

// Running mean/min/max of one stage, for the replay summary.
struct StageStats {
  int n = 0;
  double sum = 0, min = 0, max = 0;
  void add(double ms);
  double mean() const { return n ? sum / n : 0; }
};

}  // namespace det
