// The live detector. One frame:
//   capture (camera JPEG, or a file in replay mode) -> decode (TJpgDec, optionally at 1/2,
//   1/4, 1/8) -> letterbox + normalise + quantise straight into the ESP-DL input tensor ->
//   Model::run -> CenterNet decode to boxes in 800x600 frame pixels -> LED, one JSON line
//   on Serial, the web page, and (replay) a line in the results file.
// With pipeline=1, capture and decode of the next frame run in a task on core 0 while this
// loop (core 1) runs the model on the current one. Settings: /sdcard/tinyseek/config.txt
// (README.md). Shared code (platform, camera, tinyseek_core) is generated from
// firmware/perf_eval by python -m tinyseek.deploy.arduino_sketch.
#include "detector_app.hpp"

#include <Arduino.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "camera.hpp"
#include "detector_logic.hpp"
#include "esp_camera.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "platform.hpp"
#include "tinyseek_core/decode.hpp"
#include "tinyseek_core/jpeg_info.hpp"
#include "tinyseek_core/preprocess.hpp"
#include "web.hpp"

#include "arduino_config.h"  // esp-dl source and the core version check, shared with perf_eval_arduino
#if TINYSEEK_USE_ESPDLX
#include <espdlx.h>
#else
#include "dl_model_base.hpp"  // the esp-dl 3.3.11 that arduino-esp32 3.3.12 ships
#endif

#define BOARD_FREENOVE_S3_CAM
#include "board_pins.h"

namespace {

constexpr const char *kRoot = "/sdcard/tinyseek";
constexpr int kFrameW = 800, kFrameH = 600;  // SVGA, the frozen capture size
constexpr size_t kJpegInitial = 128 * 1024;
constexpr int kInternalReserveKb = 24;  // left free when internal_kb is clamped to the heap

// --- serial output: one JSON object per line ------------------------------------------------

void emit(const std::string &json) {
  std::string line = json + "\n";
  Serial.write(reinterpret_cast<const uint8_t *>(line.data()), line.size());
}

void emit_msg(const char *type, const std::string &msg) {
  emit(std::string("{\"type\":\"") + type + "\",\"msg\":\"" + det::json_escape(msg) + "\"}");
}

std::string mem_json() {
  const perf::MemSnapshot m = perf::mem_snapshot();
  char buf[256];
  snprintf(buf, sizeof(buf),
           "{\"internal_free\":%u,\"internal_min_free\":%u,\"internal_largest\":%u,\"psram_free\":%u,"
           "\"psram_min_free\":%u,\"psram_largest\":%u}",
           unsigned(m.internal_free), unsigned(m.internal_min_free), unsigned(m.internal_largest),
           unsigned(m.psram_free), unsigned(m.psram_min_free), unsigned(m.psram_largest));
  return buf;
}

void emit_mem(const char *point) {
  emit(std::string("{\"type\":\"mem\",\"point\":\"") + point + "\",\"mem\":" + mem_json() + "}");
}

// --- state ----------------------------------------------------------------------------------

struct Model {
  std::string prompt, file;
  size_t file_bytes = 0;
  dl::Model *net = nullptr;
  dl::TensorBase *input = nullptr, *output = nullptr;
  // Input pixels and output cells, read from the model: NHWC [1, in_h, in_w, 3] and
  // [1, grid_h, grid_w, channels]. Square models have in_w == in_h.
  int in_w = 0, in_h = 0, grid_w = 0, grid_h = 0, channels = 0, in_exp = 0, out_exp = 0;
  bool in16 = false, out16 = false;  // int16 I/O tensors (a w16a16 export), read from the model
  int internal_kb = 0;  // what ESP-DL was given (after clamping)
  double load_ms = 0;
  det::ModelMeta meta;
  tinyseek::DecodeParams dp;
  tinyseek::NormalizeParams norm;
};

struct Slot {
  uint8_t *jpeg = nullptr;
  size_t jpeg_len = 0, jpeg_cap = 0;
  uint8_t *rgb = nullptr;
  size_t rgb_cap = 0;
  int w = 0, h = 0;              // decoded size
  int frame_w = 0, frame_h = 0;  // JPEG size (boxes are reported in these pixels)
  std::string file, error;
  bool ok = false, eof = false;
  int64_t t0 = 0;
  double cap_ms = 0, dec_ms = 0;
};

struct Replay {
  std::vector<std::string> files;
  size_t next = 0;
  bool active = false, started = false;
  std::string dir;  // results/run_NNN
  int processed = 0, failed = 0;
  det::StageStats cap, dec, pre, inf, post, lat;
  int64_t start_us = 0;
};

struct App {
  det::Settings s;
  perf::JpegDecoder decoder = perf::JpegDecoder::kTjpgd;
  Model model;
  tinyseek::PreprocessPlan *plan = nullptr;
  int plan_w = -1, plan_h = -1;
  uint8_t *scratch = nullptr;
  size_t scratch_cap = 0;
  std::vector<tinyseek::Detection> dets;
  Slot slots[2];
  int nslots = 1;
  QueueHandle_t free_q = nullptr, ready_q = nullptr;
  volatile bool producer_running = false;
  Replay replay;
  bool camera_ok = false, wifi_ok = false;
  std::string wifi_info = "{\"mode\":\"off\"}";
  uint32_t frames = 0;
  int64_t last_end_us = 0;
  double fps = 0;
  std::string serial_line;
  // Shared with the web server task (under mu).
  SemaphoreHandle_t mu = nullptr;
  std::string state = "starting", message, pending_prompt, model_json = "null", last_frame = "null";
  std::string shown_prompt;  // g.model.prompt as the web task may read it
  std::string settings_cache = "{}";  // settings_json(g.s), rebuilt whenever g.s changes
  std::atomic<bool> replay_busy{false};  // g.replay.active, for the web task
};

App g;

std::string model_path(const std::string &prompt, const char *ext) {
  return std::string(kRoot) + "/models/" + prompt + ext;
}

void set_state(const std::string &state, const std::string &message) {
  xSemaphoreTake(g.mu, portMAX_DELAY);
  g.state = state;
  g.message = message;
  xSemaphoreGive(g.mu);
}

std::vector<std::string> available_prompts() {
  std::vector<std::string> out;
  for (const std::string &f : perf::list_dir(std::string(kRoot) + "/models", ".espdl")) {
    out.push_back(f.substr(0, f.size() - 6));
  }
  return out;
}

void set_leds(bool on) {
  const bool onboard = g.s.led == det::LedMode::kBoth || g.s.led == det::LedMode::kOnboard;
  const bool external = g.s.led == det::LedMode::kBoth || g.s.led == det::LedMode::kExternal;
  if (LED_PIN >= 0 && onboard) digitalWrite(LED_PIN, on ? LED_ON : !LED_ON);
  if (EXT_LED_PIN >= 0 && external) digitalWrite(EXT_LED_PIN, on ? LED_ON : !LED_ON);
}

bool grow(uint8_t **buf, size_t *cap, size_t need) {
  if (need <= *cap) return true;
  void *p = heap_caps_realloc(*buf, need, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!p) return false;
  *buf = static_cast<uint8_t *>(p);
  *cap = need;
  return true;
}

// --- model ----------------------------------------------------------------------------------

void unload_model() {
  delete g.model.net;
  g.model = Model();
  g.plan_w = g.plan_h = -1;  // the plan depends on the model's size, exponent and mean/std
  xSemaphoreTake(g.mu, portMAX_DELAY);
  g.model_json = "null";
  g.shown_prompt.clear();
  xSemaphoreGive(g.mu);
}

// Loads models/<prompt>.espdl and its .txt sidecar; checks the I/O against tinyseek_core.
bool load_model(const std::string &prompt, std::string *err) {
  Model &m = g.model;
  m.prompt = prompt;
  m.file = model_path(prompt, ".espdl");
  m.meta = det::parse_model_meta(perf::read_kv(model_path(prompt, ".txt")));
  FILE *f = fopen(m.file.c_str(), "rb");
  if (!f) {
    *err = "no model file " + m.file;
    return false;
  }
  fseek(f, 0, SEEK_END);
  m.file_bytes = static_cast<size_t>(ftell(f));
  fclose(f);
  for (int c = 0; c < 3; ++c) {
    m.norm.mean[c] = m.meta.mean[c];
    m.norm.std[c] = m.meta.std[c];
  }
  m.norm.pad_value = static_cast<uint8_t>(m.meta.pad_value);

  // ESP-DL allocates max_internal_size as one internal block and fails the load when no
  // free block is that large (perf_eval README): clamp to what is free now.
  m.internal_kb = g.s.internal_kb;
  const int largest_kb = static_cast<int>(perf::mem_snapshot().internal_largest / 1024);
  if (m.internal_kb > 0 && m.internal_kb > largest_kb - kInternalReserveKb) {
    const int clamped = largest_kb > kInternalReserveKb ? largest_kb - kInternalReserveKb : 0;
    emit_msg("warning", "internal_kb=" + std::to_string(m.internal_kb) + " does not fit (largest free internal block " +
                            std::to_string(largest_kb) + " KB); using " + std::to_string(clamped));
    m.internal_kb = clamped;
  }
  emit_mem("before_load");
  const int64_t t0 = perf::now_us();
  // dl_model_base.hpp (esp-dl 3.3.11): (path, location, max_internal_size, mm_type, key,
  // param_copy), as in perf_eval, where this exact call loads these exports on the board.
  m.net = new dl::Model(m.file.c_str(), fbs::MODEL_LOCATION_IN_SDCARD, m.internal_kb * 1024,
                        dl::MEMORY_MANAGER_GREEDY, nullptr, true);
  m.load_ms = (perf::now_us() - t0) / 1000.0;
  if (!m.net->get_fbs_model() || !m.net->get_input() || !m.net->get_output()) {
    *err = "ESP-DL could not load " + m.file;
    return false;
  }
  emit_mem("after_load");
  m.input = m.net->get_input();
  m.output = m.net->get_output();
  const std::vector<int> is = m.input->get_shape();
  const std::vector<int> os = m.output->get_shape();
  const dl::dtype_t idt = m.input->get_dtype(), odt = m.output->get_dtype();
  if (is.size() != 4 || is[0] != 1 || is[1] < 1 || is[2] < 1 || is[3] != 3 ||
      (idt != dl::DATA_TYPE_INT8 && idt != dl::DATA_TYPE_INT16)) {
    *err = "model input must be int8 or int16 NHWC [1,H,W,3]";
    return false;
  }
  if (os.size() != 4 || os[1] < 1 || os[2] < 1 || os[3] < 5 ||
      (odt != dl::DATA_TYPE_INT8 && odt != dl::DATA_TYPE_INT16)) {
    *err = "model output must be int8 or int16 NHWC [1,GH,GW,C>=5]";
    return false;
  }
  // One stride for both axes: the grid has the input's aspect (always true for a square
  // input with a square grid, the only shape accepted before rectangular inputs).
  if (is[1] * os[2] != is[2] * os[1]) {
    *err = "model input and output grid differ in aspect";
    return false;
  }
  m.in_h = is[1];  // NHWC: height first
  m.in_w = is[2];
  m.grid_h = os[1];
  m.grid_w = os[2];
  m.channels = os[3];
  m.in16 = idt == dl::DATA_TYPE_INT16;  // dtypes and exponents read from the model, never assumed
  m.out16 = odt == dl::DATA_TYPE_INT16;
  m.in_exp = m.input->get_exponent();
  m.out_exp = m.output->get_exponent();
  m.dp.grid_h = m.grid_h;
  m.dp.grid_w = m.grid_w;
  m.dp.num_classes = m.channels - 4;
  m.dp.stride = m.in_h / m.grid_h;  // the same on both axes (checked above)
  m.dp.exponent = m.out_exp;
  m.dp.score_threshold = m.meta.score_threshold;
  m.dp.max_detections = m.meta.max_detections;
  if (m.meta.has_exponents && (m.meta.input_exponent != m.in_exp || m.meta.output_exponent != m.out_exp)) {
    emit_msg("warning", prompt + ": exponents differ from " + prompt + ".txt (stale card?)");
  }
  if ((!m.meta.input_dtype.empty() && m.meta.input_dtype != (m.in16 ? "INT16" : "INT8")) ||
      (!m.meta.output_dtype.empty() && m.meta.output_dtype != (m.out16 ? "INT16" : "INT8"))) {
    emit_msg("warning", prompt + ": I/O dtypes differ from " + prompt + ".txt (stale card?)");
  }
  if (m.meta.input_w && (m.meta.input_w != m.in_w || m.meta.input_h != m.in_h)) {
    emit_msg("warning", prompt + ": input size differs from " + prompt + ".txt (stale card?)");
  }
  // "input":S,"grid":G for a square model; input_w/input_h/grid_w/grid_h for a rectangular one.
  const std::string shape = det::model_shape_json(m.in_w, m.in_h, m.grid_w, m.grid_h);
  char buf[512];
  snprintf(buf, sizeof(buf),
           "{\"prompt\":\"%s\",%s,\"channels\":%d,\"in_exp\":%d,\"out_exp\":%d,"
           "\"in_dtype\":\"%s\",\"out_dtype\":\"%s\","
           "\"bytes\":%u,\"load_ms\":%.1f,\"internal_kb\":%d,\"score_threshold\":%.3f,\"detect_threshold\":%.3f,"
           "\"max_detections\":%d,\"sha256\":\"%s\"}",
           det::json_escape(prompt).c_str(), shape.c_str(), m.channels, m.in_exp, m.out_exp,
           m.in16 ? "INT16" : "INT8", m.out16 ? "INT16" : "INT8", unsigned(m.file_bytes),
           m.load_ms, m.internal_kb, m.meta.score_threshold, m.meta.detect_threshold, m.meta.max_detections,
           det::json_escape(m.meta.sha256.substr(0, 16)).c_str());
  xSemaphoreTake(g.mu, portMAX_DELAY);
  g.model_json = buf;
  g.shown_prompt = prompt;
  xSemaphoreGive(g.mu);
  emit(std::string("{\"type\":\"model\",\"model\":") + buf + "}");
  return true;
}

// Plan (resample coefficients, quantisation tables) for the decoded size; rebuilt only
// when the decoded size or the model changes.
bool ensure_plan(int w, int h) {
  if (g.plan && w == g.plan_w && h == g.plan_h) return true;
  if (!g.plan) g.plan = new tinyseek::PreprocessPlan();
  tinyseek::make_plan(*g.plan, w, h, g.model.in_w, g.model.in_h, g.model.in_exp, g.model.norm,
                      g.s.bilinear ? tinyseek::Resample::kBilinear : tinyseek::Resample::kPillow);
  const size_t need = tinyseek::scratch_bytes(*g.plan);
  if (need > g.scratch_cap) {
    perf::free_buffer(g.scratch);
    g.scratch = static_cast<uint8_t *>(perf::alloc_psram(need));
    g.scratch_cap = g.scratch ? need : 0;
    if (!g.scratch) return false;
  }
  g.plan_w = w;
  g.plan_h = h;
  return true;
}

// --- frames ---------------------------------------------------------------------------------

bool read_whole(const std::string &path, Slot &s) {
  FILE *f = fopen(path.c_str(), "rb");
  if (!f) return false;
  fseek(f, 0, SEEK_END);
  const long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  bool ok = n > 0 && grow(&s.jpeg, &s.jpeg_cap, static_cast<size_t>(n)) &&
            fread(s.jpeg, 1, static_cast<size_t>(n), f) == static_cast<size_t>(n);
  fclose(f);
  s.jpeg_len = ok ? static_cast<size_t>(n) : 0;
  return ok;
}

// Fills one slot: the JPEG (camera or the next replay file) and its decoded pixels.
// Runs on the producer task (pipeline=1) or inline on the loop.
void acquire(Slot &s) {
  s.ok = s.eof = false;
  s.error.clear();
  s.file.clear();
  s.w = s.h = s.frame_w = s.frame_h = 0;
  s.dec_ms = 0;
  s.t0 = perf::now_us();
  if (g.s.source == det::Source::kCamera) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      s.error = "no camera frame";
      s.cap_ms = (perf::now_us() - s.t0) / 1000.0;
      return;
    }
    const bool copied = grow(&s.jpeg, &s.jpeg_cap, fb->len);
    if (copied) memcpy(s.jpeg, fb->buf, fb->len);
    s.jpeg_len = copied ? fb->len : 0;
    esp_camera_fb_return(fb);
    if (!copied) {
      s.error = "out of PSRAM for the JPEG copy";
      return;
    }
  } else {
    Replay &r = g.replay;
    const size_t limit = g.s.replay_limit > 0 ? static_cast<size_t>(g.s.replay_limit) : r.files.size();
    if (r.next >= r.files.size() || r.next >= limit) {
      s.eof = true;
      return;
    }
    s.file = r.files[r.next++];
    if (!read_whole(g.s.replay_dir + "/" + s.file, s)) {
      s.error = "cannot read " + s.file;
      return;
    }
  }
  const int64_t t1 = perf::now_us();
  s.cap_ms = (t1 - s.t0) / 1000.0;
  tinyseek::JpegInfo info;
  if (!tinyseek::jpeg_info(s.jpeg, s.jpeg_len, info)) {
    s.error = "not a JPEG";
    return;
  }
  s.frame_w = info.width;
  s.frame_h = info.height;
  if (!perf::decode_jpeg_scaled(s.jpeg, s.jpeg_len, g.decoder, g.s.decode_scale, s.rgb, s.rgb_cap, &s.w, &s.h)) {
    s.error = "JPEG decode failed";
  } else {
    s.ok = true;
  }
  s.dec_ms = (perf::now_us() - t1) / 1000.0;
}

void producer_task(void *) {
  for (;;) {
    Slot *s = nullptr;
    xQueueReceive(g.free_q, &s, portMAX_DELAY);
    acquire(*s);
    const bool eof = s->eof;
    xQueueSend(g.ready_q, &s, portMAX_DELAY);
    if (eof) break;
    vTaskDelay(1);  // let IDLE0 run (task watchdog)
  }
  g.producer_running = false;
  vTaskDelete(nullptr);
}

bool start_producer() {
  g.producer_running = true;
  // Core 0, the same priority as the loop task, so it time-slices fairly with ESP-DL's
  // core-0 worker when multicore=1.
  if (xTaskCreatePinnedToCore(producer_task, "ts_capture", 8192, nullptr, 1, nullptr, 0) != pdPASS) {
    g.producer_running = false;
    return false;
  }
  return true;
}

// --- replay ---------------------------------------------------------------------------------

void write_run_txt() {
  const perf::BuildInfo bi = perf::build_info();
  std::string t;
  t += std::string("build=") + bi.build + "\nidf_version=" + bi.idf_version + "\nespdl_version=" + bi.espdl_version;
  t += "\nsdk_settings=" + perf::sdk_settings() + "\ncpu_mhz=" + std::to_string(perf::cpu_mhz());
  t += std::string("\njpeg_decoder=") + perf::jpeg_decoder_label(g.decoder);
  t += "\nprompt=" + g.model.prompt + "\nmodel_sha256=" + g.model.meta.sha256 + "\nmodel_run=" + g.model.meta.run;
  // input_size=S for a square model, input_width=W and input_height=H for a rectangular one.
  t += "\n" + det::model_shape_kv(g.model.in_w, g.model.in_h) + "\ninput_exponent=" + std::to_string(g.model.in_exp) +
       "\noutput_exponent=" + std::to_string(g.model.out_exp) + "\ninput_dtype=" +
       (g.model.in16 ? "INT16" : "INT8") + "\noutput_dtype=" + (g.model.out16 ? "INT16" : "INT8") +
       "\ninternal_kb_effective=" +
       std::to_string(g.model.internal_kb);
  char thr[64];
  snprintf(thr, sizeof(thr), "\nscore_threshold=%.4f\nmax_detections=%d", g.model.meta.score_threshold,
           g.model.meta.max_detections);
  t += thr;
  t += "\nsettings=" + det::settings_json(g.s) + "\nfiles=" + std::to_string(g.replay.files.size()) + "\n";
  perf::write_file(g.replay.dir + "/run.txt", t.data(), t.size());
}

void start_replay() {
  Replay &r = g.replay;
  r = Replay();
  r.files = perf::list_dir(g.s.replay_dir, ".jpg");
  if (r.files.empty()) {
    emit_msg("error", "no .jpg files in " + g.s.replay_dir);
    set_state("error", "replay folder empty");
    return;
  }
  if (g.s.replay_limit > 0 && r.files.size() > static_cast<size_t>(g.s.replay_limit)) r.files.resize(g.s.replay_limit);
  r.dir = perf::next_run_dir(std::string(kRoot) + "/results");
  if (!perf::make_dir(r.dir)) {
    emit_msg("error", "cannot create " + r.dir);
    set_state("error", "cannot write results");
    return;
  }
  write_run_txt();
  r.active = r.started = true;
  g.replay_busy = true;
  r.start_us = perf::now_us();
  emit("{\"type\":\"replay_start\",\"files\":" + std::to_string(r.files.size()) + ",\"results\":\"" +
       det::json_escape(r.dir) + "\"}");
  set_state("replay", std::to_string(r.files.size()) + " files");
  if (g.s.pipeline && !start_producer()) emit_msg("error", "cannot start the capture task");
}

void finish_replay() {
  Replay &r = g.replay;
  r.active = false;
  g.replay_busy = false;
  const double secs = (perf::now_us() - r.start_us) / 1e6;
  auto stat = [](const char *name, const det::StageStats &st) {
    char buf[160];
    snprintf(buf, sizeof(buf), "%s_mean_ms=%.3f\n%s_min_ms=%.3f\n%s_max_ms=%.3f\n", name, st.mean(), name, st.min,
             name, st.max);
    return std::string(buf);
  };
  std::string t = "images=" + std::to_string(r.processed) + "\nfailed=" + std::to_string(r.failed) + "\n";
  t += stat("capture", r.cap) + stat("decode", r.dec) + stat("preprocess", r.pre) + stat("inference", r.inf) +
       stat("postprocess", r.post) + stat("latency", r.lat);
  char buf[64];
  snprintf(buf, sizeof(buf), "wall_s=%.2f\n", secs);
  t += buf;
  perf::write_file(r.dir + "/summary.txt", t.data(), t.size());
  perf::append_line(r.dir + "/run.txt", "complete=1");
  char js[400];
  snprintf(js, sizeof(js),
           "{\"type\":\"replay_done\",\"results\":\"%s\",\"images\":%d,\"failed\":%d,\"mean_ms\":{\"cap\":%.1f,"
           "\"dec\":%.1f,\"pre\":%.1f,\"inf\":%.1f,\"post\":%.2f,\"lat\":%.1f},\"wall_s\":%.1f}",
           det::json_escape(r.dir).c_str(), r.processed, r.failed, r.cap.mean(), r.dec.mean(), r.pre.mean(),
           r.inf.mean(), r.post.mean(), r.lat.mean(), secs);
  emit(js);
  set_state("replay_done", r.dir);
  set_leds(false);
}

// --- one frame ------------------------------------------------------------------------------

void process(Slot &s) {
  det::FrameReport r;
  r.n = ++g.frames;
  r.prompt = g.model.prompt;
  r.source = g.s.source == det::Source::kReplay ? "replay" : "camera";
  r.file = s.file;
  r.width = s.frame_w;
  r.height = s.frame_h;
  r.detect_threshold = g.model.meta.detect_threshold;
  r.t.cap = s.cap_ms;
  r.t.dec = s.dec_ms;
  g.dets.clear();
  r.dets = &g.dets;
  r.ok = s.ok;
  r.error = s.error;
  if (s.ok && !ensure_plan(s.w, s.h)) {
    r.ok = false;
    r.error = "out of PSRAM for the preprocessing plan";
  }
  if (r.ok) {
    Model &m = g.model;
    const int64_t a = perf::now_us();
    // Straight into ESP-DL's input tensor: it shares the arena with the intermediates, so
    // it is rewritten before every run anyway (perf_eval copies into it instead).
    if (m.in16) {
      tinyseek::preprocess(*g.plan, s.rgb, g.scratch, static_cast<int16_t *>(m.input->data));
    } else {
      tinyseek::preprocess(*g.plan, s.rgb, g.scratch, static_cast<int8_t *>(m.input->data));
    }
    const int64_t b = perf::now_us();
    m.net->run(g.s.multicore ? dl::RUNTIME_MODE_MULTI_CORE : dl::RUNTIME_MODE_SINGLE_CORE);
    const int64_t c = perf::now_us();
    const tinyseek::Letterbox lb = tinyseek::frame_letterbox(g.plan->lb, g.s.decode_scale);
    if (m.out16) {
      tinyseek::decode(static_cast<const int16_t *>(m.output->data), m.dp, lb, g.dets);
    } else {
      tinyseek::decode(static_cast<const int8_t *>(m.output->data), m.dp, lb, g.dets);
    }
    const int64_t d = perf::now_us();
    r.t.pre = (b - a) / 1000.0;
    r.t.inf = (c - b) / 1000.0;
    r.t.post = (d - c) / 1000.0;
  }
  const int64_t end = perf::now_us();
  r.t.lat = (end - s.t0) / 1000.0;
  r.t.period = g.last_end_us ? (end - g.last_end_us) / 1000.0 : 0;
  g.last_end_us = end;
  if (r.t.period > 0) g.fps = g.fps > 0 ? 0.8 * g.fps + 0.2 * (1000.0 / r.t.period) : 1000.0 / r.t.period;
  r.fps = g.fps;

  bool hit = false;
  for (const auto &d : g.dets) hit |= d.score >= r.detect_threshold;
  set_leds(hit);
  const std::string line = det::frame_json(r, g.s.print_min_score);
  emit(line);
  if (g.wifi_ok) det::web_publish(s.jpeg, s.jpeg_len, line);
  xSemaphoreTake(g.mu, portMAX_DELAY);
  g.last_frame = line;
  xSemaphoreGive(g.mu);

  if (g.replay.active) {
    Replay &rp = g.replay;
    perf::append_line(rp.dir + "/dets.jsonl", det::replay_json(r));
    rp.processed++;
    if (!r.ok) {
      rp.failed++;
    } else {
      rp.cap.add(r.t.cap);
      rp.dec.add(r.t.dec);
      rp.pre.add(r.t.pre);
      rp.inf.add(r.t.inf);
      rp.post.add(r.t.post);
      rp.lat.add(r.t.lat);
    }
  }
}

// --- prompts --------------------------------------------------------------------------------

std::string prompt_reply(bool ok, const std::string &prompt, const std::string &message) {
  return std::string("{\"ok\":") + (ok ? "true" : "false") + ",\"prompt\":\"" + det::json_escape(prompt) +
         "\",\"message\":\"" + det::json_escape(message) + "\",\"available\":" +
         det::json_string_array(available_prompts()) + "}";
}

// Called from the web server task and from the serial command parser.
std::string request_prompt(const std::string &typed) {
  const std::string name = det::normalize_prompt(typed);
  if (!det::valid_prompt(name)) {
    return prompt_reply(false, name, "'" + typed + "' is not a prompt name (letters, digits and _ only)");
  }
  if (g.replay_busy) return prompt_reply(false, name, "a replay is running; switch after it finishes");
  if (!perf::file_exists(model_path(name, ".espdl"))) {
    return prompt_reply(false, name, "no model for '" + name + "' on the card");
  }
  xSemaphoreTake(g.mu, portMAX_DELAY);
  g.pending_prompt = name;
  xSemaphoreGive(g.mu);
  return prompt_reply(true, name, "switching to '" + name + "'");
}

std::string config_path() { return std::string(kRoot) + "/config.txt"; }
std::string config_tmp_path() { return std::string(kRoot) + "/config.tmp"; }

void persist_prompt(const std::string &prompt) {
  const std::string cfg = config_path(), tmp = config_tmp_path();
  const bool exists = perf::file_exists(cfg);
  std::string text, out;
  const bool read_ok = exists && perf::read_text(cfg, &text);
  if (!det::config_for_save(exists, read_ok, text, prompt, &out)) {
    emit_msg("warning", "config.txt could not be read; prompt=" + prompt + " not saved (the file is unchanged)");
    return;
  }
  // Write a complete copy, then swap it in (FAT cannot rename over a file, so the old one
  // is removed first). An interruption is repaired at the next boot (recover_config).
  bool ok = perf::write_file(tmp, out.data(), out.size());
  if (ok) {
    remove(cfg.c_str());
    ok = rename(tmp.c_str(), cfg.c_str()) == 0;
  }
  if (!ok) emit_msg("warning", "could not save prompt=" + prompt + " to config.txt");
}

// Repairs an interrupted persist_prompt before config.txt is read.
void recover_config() {
  const std::string cfg = config_path(), tmp = config_tmp_path();
  switch (det::config_recovery(perf::file_exists(cfg), perf::file_exists(tmp))) {
    case det::ConfigRecovery::kRestoreTmp:
      if (rename(tmp.c_str(), cfg.c_str()) == 0) emit_msg("warning", "config.txt was missing; restored it from config.tmp");
      else emit_msg("error", "config.txt is missing and config.tmp could not be renamed");
      break;
    case det::ConfigRecovery::kDeleteTmp:
      remove(tmp.c_str());  // left over from a save that stopped before replacing config.txt
      break;
    case det::ConfigRecovery::kNone:
      break;
  }
}

void update_settings_cache() {
  const std::string js = det::settings_json(g.s);
  xSemaphoreTake(g.mu, portMAX_DELAY);
  g.settings_cache = js;
  xSemaphoreGive(g.mu);
}

void switch_prompt(const std::string &name) {
  if (g.replay.active) {  // also refused in request_prompt; this covers a request queued before the replay began
    emit("{\"type\":\"prompt\",\"ok\":false,\"prompt\":\"" + det::json_escape(name) +
         "\",\"msg\":\"a replay is running; switch after it finishes\"}");
    return;
  }
  const std::string previous = g.model.prompt;
  set_state("switching", name);
  set_leds(false);
  unload_model();
  emit_mem("after_unload");
  std::string err;
  if (load_model(name, &err)) {
    xSemaphoreTake(g.mu, portMAX_DELAY);
    g.s.prompt = name;  // the web task reads settings only through settings_cache
    xSemaphoreGive(g.mu);
    update_settings_cache();
    persist_prompt(name);
    emit("{\"type\":\"prompt\",\"ok\":true,\"prompt\":\"" + det::json_escape(name) + "\"}");
    set_state("running", "");
    if (g.s.source == det::Source::kReplay && !g.replay.started) start_replay();
    return;
  }
  emit("{\"type\":\"prompt\",\"ok\":false,\"prompt\":\"" + det::json_escape(name) + "\",\"msg\":\"" +
       det::json_escape(err) + "\"}");
  unload_model();
  std::string err2;
  if (!previous.empty() && load_model(previous, &err2)) {
    set_state("running", "could not load '" + name + "': " + err);
  } else {
    unload_model();
    set_state("no_model", "could not load '" + name + "': " + err);
  }
}

std::string status_json() {
  xSemaphoreTake(g.mu, portMAX_DELAY);
  std::string o = "{\"state\":\"" + det::json_escape(g.state) + "\",\"message\":\"" + det::json_escape(g.message) +
                  "\",\"prompt\":\"" + det::json_escape(g.shown_prompt) + "\",\"model\":" + g.model_json +
                  ",\"last\":" + g.last_frame + ",\"settings\":" + g.settings_cache;
  xSemaphoreGive(g.mu);
  o += ",\"available\":" + det::json_string_array(available_prompts());
  o += ",\"mem\":" + mem_json();
  o += ",\"wifi\":" + (g.wifi_ok ? det::wifi_status_json() : g.wifi_info) + "}";
  return o;
}

void serial_command(const std::string &line) {
  std::string cmd = line, arg;
  const size_t sp = line.find(' ');
  if (sp != std::string::npos) {
    cmd = line.substr(0, sp);
    arg = line.substr(sp + 1);
  }
  if (cmd == "prompt" && !arg.empty()) {
    emit("{\"type\":\"reply\",\"reply\":" + request_prompt(arg) + "}");
  } else if (cmd == "list") {
    emit("{\"type\":\"reply\",\"available\":" + det::json_string_array(available_prompts()) + "}");
  } else if (cmd == "status") {
    emit("{\"type\":\"status\",\"status\":" + status_json() + "}");
  } else if (cmd == "replay") {
    if (g.s.source != det::Source::kReplay) emit_msg("reply", "source is not replay (config.txt)");
    else if (g.replay.active || g.producer_running) emit_msg("reply", "a replay is running");
    else if (!g.model.net) emit_msg("reply", "no model loaded");
    else start_replay();
  } else {
    emit_msg("reply", "commands: prompt <name> | list | status | replay");
  }
}

void poll_serial() {
  while (Serial.available() > 0) {
    const int c = Serial.read();
    if (c < 0) break;
    if (c == '\n' || c == '\r') {
      if (!g.serial_line.empty()) serial_command(g.serial_line);
      g.serial_line.clear();
    } else if (g.serial_line.size() < 96) {
      g.serial_line += static_cast<char>(c);
    }
  }
}

void halt(const std::string &why) {
  emit_msg("fatal", why);
  for (bool on = true;; on = !on) {  // blink: the board is not detecting
    if (LED_PIN >= 0) digitalWrite(LED_PIN, on ? LED_ON : !LED_ON);
    delay(250);
  }
}

}  // namespace

void detector_setup() {
  g.mu = xSemaphoreCreateMutex();
  if (LED_PIN >= 0) pinMode(LED_PIN, OUTPUT);
  if (EXT_LED_PIN >= 0) pinMode(EXT_LED_PIN, OUTPUT);
  set_leds(false);
  const perf::BuildInfo bi = perf::build_info();
  emit(std::string("{\"type\":\"boot\",\"build\":\"") + bi.build + "\",\"idf\":\"" + bi.idf_version +
       "\",\"espdl\":\"" + det::json_escape(bi.espdl_version) + "\",\"sdk\":\"" + perf::sdk_settings() +
       "\",\"cpu_mhz\":" + std::to_string(perf::cpu_mhz()) + "}");
  emit_mem("boot");
  if (!perf::mount_sd_card()) halt("no SD card (FAT32, inserted before power-on?)");
  if (!perf::file_exists(kRoot)) halt("/sdcard/tinyseek missing: copy the staged tinyseek folder to the card root");

  recover_config();
  std::vector<std::string> warnings;
  g.s = det::parse_settings(perf::read_kv(config_path()), warnings);
  g.decoder = g.s.esp_new_jpeg ? perf::JpegDecoder::kEspNewJpeg : perf::JpegDecoder::kTjpgd;
  if (!perf::jpeg_decoder_available(g.decoder)) {
    warnings.push_back(std::string(perf::jpeg_decoder_label(g.decoder)) + " is not in this build; using tjpgd");
    g.decoder = perf::JpegDecoder::kTjpgd;
    g.s.esp_new_jpeg = false;
  }
  update_settings_cache();
  emit("{\"type\":\"config\",\"settings\":" + det::settings_json(g.s) + ",\"jpeg_decoder\":\"" +
       perf::jpeg_decoder_label(g.decoder) + "\",\"warnings\":" + det::json_string_array(warnings) + "}");

  // Wi-Fi before the model: it takes internal RAM, and internal_kb is clamped to what is
  // left, instead of Wi-Fi failing after the model has taken it.
  if (g.s.wifi != det::WifiMode::kOff) {
    g.wifi_ok = det::wifi_start(g.s, &g.wifi_info);
    emit("{\"type\":\"wifi\",\"wifi\":" + g.wifi_info + "}");
    emit_mem("after_wifi");
  }
  if (g.s.source == det::Source::kCamera) {
    g.camera_ok = perf::camera_init(static_cast<uint32_t>(g.s.camera_warmup_ms));
    if (!g.camera_ok) halt("camera init failed (ribbon cable seated?)");
    emit_mem("after_camera");
  }

  // Frame slots: two when capture runs ahead in its own task.
  g.nslots = g.s.pipeline ? 2 : 1;
  const size_t rgb = size_t(kFrameW >> g.s.decode_scale) * (kFrameH >> g.s.decode_scale) * 3;
  for (int i = 0; i < g.nslots; ++i) {
    Slot &s = g.slots[i];
    s.rgb = static_cast<uint8_t *>(perf::alloc_psram(rgb));  // 16-byte aligned (esp_new_jpeg)
    s.rgb_cap = s.rgb ? rgb : 0;
    grow(&s.jpeg, &s.jpeg_cap, kJpegInitial);
    if (!s.rgb || !s.jpeg) halt("out of PSRAM for frame buffers");
  }
  if (g.s.pipeline) {
    g.free_q = xQueueCreate(g.nslots, sizeof(Slot *));
    g.ready_q = xQueueCreate(g.nslots, sizeof(Slot *));
    for (int i = 0; i < g.nslots; ++i) {
      Slot *p = &g.slots[i];
      xQueueSend(g.free_q, &p, 0);
    }
  }

  // The web server and the capture task before the model, so their stacks (internal RAM)
  // are already taken when internal_kb is clamped to what is left.
  if (g.wifi_ok) {
    det::WebHooks hooks;
    hooks.status_json = status_json;
    hooks.request_prompt = request_prompt;
    if (!det::web_start(g.s.http_port, hooks)) {
      emit_msg("error", "web server did not start");
      g.wifi_ok = false;
    } else {
      emit("{\"type\":\"web\",\"port\":" + std::to_string(g.s.http_port) + ",\"wifi\":" + g.wifi_info + "}");
    }
    emit_mem("after_web");
  }
  if (g.s.source == det::Source::kCamera && g.s.pipeline && !start_producer()) {
    halt("cannot start the capture task");
  }

  if (g.s.prompt.empty()) {
    set_state("no_model", "no prompt in config.txt");
    emit_msg("error", "config.txt has no prompt=; type: prompt <name>");
  } else {
    std::string err;
    if (load_model(g.s.prompt, &err)) {
      set_state("running", "");
    } else {
      unload_model();
      set_state("no_model", err);
      emit_msg("error", err + "; available: " + det::json_string_array(available_prompts()));
    }
  }
  if (g.s.source == det::Source::kReplay && g.model.net) start_replay();
}

void detector_loop() {
  poll_serial();
  std::string pending;
  xSemaphoreTake(g.mu, portMAX_DELAY);
  pending.swap(g.pending_prompt);
  xSemaphoreGive(g.mu);
  switch (det::prompt_action(pending, g.model.prompt, g.replay.active)) {
    case det::PromptAction::kSwitch:
      switch_prompt(pending);
      break;
    case det::PromptAction::kRefuseReplay:
      emit("{\"type\":\"prompt\",\"ok\":false,\"prompt\":\"" + det::json_escape(pending) +
           "\",\"msg\":\"a replay is running; switch after it finishes\"}");
      break;
    case det::PromptAction::kNone:
      break;
  }

  const bool replay = g.s.source == det::Source::kReplay;
  if (replay && !g.replay.active) {
    delay(20);  // done or not started: only serial and web
    return;
  }
  if (g.s.pipeline) {
    Slot *s = nullptr;
    if (xQueueReceive(g.ready_q, &s, pdMS_TO_TICKS(50)) != pdTRUE) return;
    if (s->eof) {
      xQueueSend(g.free_q, &s, portMAX_DELAY);
      finish_replay();
      return;
    }
    if (g.model.net) process(*s);  // without a model, camera frames are dropped
    xQueueSend(g.free_q, &s, portMAX_DELAY);
    return;
  }
  if (!g.model.net) {
    delay(50);
    return;
  }
  Slot &s = g.slots[0];
  acquire(s);
  if (s.eof) {
    finish_replay();
    return;
  }
  process(s);
  delay(1);  // yield (other loop-priority tasks, and the web server's copies)
}
