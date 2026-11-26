// On-device evaluation of tinyseek detectors (ESP-DL .espdl models from the SD card).
//
// For every /sdcard/tinyseek_eval/models/<m>.espdl:
//   1. load time and heap before/after load;
//   2. preprocessing (800x600 RGB -> letterboxed, normalised, int8 or int16 input), inference and
//      post-processing (decode to boxes) timed separately: warm-up, then N timed runs;
//   3. parity: every staged case through the board pipeline, from the PC-decoded .rgb and
//      from the board's own JPEG decode, tensors and detections written back to the card;
//   4. optional camera stage: capture + JPEG decode + full pipeline, end-to-end FPS.
// Results go to /sdcard/tinyseek_eval/results/run_NNN/ (bench.csv, memory.csv, parity/)
// and the serial log. Settings come from bench.txt (see README.md).
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "camera.hpp"
#include "esp_camera.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "perf_eval.hpp"
#include "platform.hpp"
#include "tinyseek_core/decode.hpp"
#include "tinyseek_core/preprocess.hpp"

#ifdef ARDUINO
#include "arduino_config.h"  // only in the sketch: which esp-dl, which JPEG decoder
#endif
#if defined(ARDUINO) && TINYSEEK_USE_ESPDLX
#include <espdlx.h>
#else
#include "dl_model_base.hpp"  // ESP-IDF component, or the copy arduino-esp32 3.3.12+ ships
#endif

namespace {

constexpr int kFrameW = 800, kFrameH = 600;  // SVGA, the frozen capture size
constexpr const char *kBoard = "freenove_esp32s3_cam_n16r8";

struct Settings {
  int warmup, runs, camera, camera_frames, parity, internal_kb, profile;
};

struct ModelCtx {
  std::string name, file;
  size_t file_bytes = 0;
  dl::Model *model = nullptr;
  dl::TensorBase *input = nullptr, *output = nullptr;
  int size = 0, grid = 0, channels = 0, in_exp = 0, out_exp = 0;
  bool in16 = false, out16 = false;  // int16 input/output tensors (w16a16 export), from the model
  tinyseek::PreprocessPlan *plan = nullptr;
  tinyseek::DecodeParams dp;
  tinyseek::NormalizeParams norm;
  size_t in_bytes() const { return size_t(size) * size * 3 * (in16 ? 2 : 1); }
  size_t out_bytes() const { return size_t(grid) * grid * channels * (out16 ? 2 : 1); }
  const char *in_dtype() const { return in16 ? "INT16" : "INT8"; }
  const char *out_dtype() const { return out16 ? "INT16" : "INT8"; }
};

// Preprocess into `dst` and decode the model output, in whichever element type the model has.
void preprocess_into(const ModelCtx &m, const uint8_t *rgb, uint8_t *scratch, void *dst) {
  if (m.in16) {
    tinyseek::preprocess(*m.plan, rgb, scratch, static_cast<int16_t *>(dst));
  } else {
    tinyseek::preprocess(*m.plan, rgb, scratch, static_cast<int8_t *>(dst));
  }
}

void decode_output(const ModelCtx &m, std::vector<tinyseek::Detection> &dets) {
  if (m.out16) {
    tinyseek::decode(static_cast<const int16_t *>(m.output->data), m.dp, m.plan->lb, dets);
  } else {
    tinyseek::decode(static_cast<const int8_t *>(m.output->data), m.dp, m.plan->lb, dets);
  }
}

std::string g_run;   // results/run_NNN
std::string g_meta;  // leading CSV columns shared by every row

void fail(const char *why) {
  perf::logln("FATAL: %s", why);
  for (;;) vTaskDelay(pdMS_TO_TICKS(1000));
}

void bench_row(const ModelCtx &m, const Settings &s, const char *stage, const perf::Stats &st, double fps = 0) {
  char buf[512];
  snprintf(buf, sizeof(buf), "%s,%s,%u,%d,%d,%d,%d,%s,%d,%d,%.3f,%.3f,%.3f,%.3f,%.2f,%s,%s", g_meta.c_str(),
           m.name.c_str(), unsigned(m.file_bytes), m.size, m.in_exp, m.out_exp, s.internal_kb, stage, st.n, s.warmup,
           st.mean_ms, st.min_ms, st.max_ms, st.std_ms, fps, m.in_dtype(), m.out_dtype());
  perf::append_line(g_run + "/bench.csv", buf);
  perf::logln("%-6s %-15s n=%-4d mean %8.3f ms  min %8.3f  max %8.3f  std %7.3f%s", m.name.c_str(), stage, st.n,
           st.mean_ms, st.min_ms, st.max_ms, st.std_ms, fps > 0 ? "  (fps below)" : "");
  if (fps > 0) perf::logln("%-6s end-to-end %.2f fps", m.name.c_str(), fps);
}

void mem_row(const std::string &model, const char *point) {
  perf::MemSnapshot m = perf::mem_snapshot();
  char buf[512];
  snprintf(buf, sizeof(buf), "%s,%s,%s,%u,%u,%u,%u,%u,%u", g_meta.c_str(), model.c_str(), point, unsigned(m.internal_free),
           unsigned(m.internal_min_free), unsigned(m.internal_largest), unsigned(m.psram_free),
           unsigned(m.psram_min_free), unsigned(m.psram_largest));
  perf::append_line(g_run + "/memory.csv", buf);
  perf::logln("mem %-6s %-16s internal free %7u min %7u largest %7u | psram free %8u min %8u largest %8u",
           model.c_str(), point, unsigned(m.internal_free), unsigned(m.internal_min_free), unsigned(m.internal_largest),
           unsigned(m.psram_free), unsigned(m.psram_min_free), unsigned(m.psram_largest));
}

Settings load_settings() {
  auto kv = perf::read_kv(std::string(perf::kRoot) + "/bench.txt");
  return {perf::kv_int(kv, "warmup", 5),      perf::kv_int(kv, "runs", 50),
          perf::kv_int(kv, "camera", 0),      perf::kv_int(kv, "camera_frames", 50),
          perf::kv_int(kv, "parity", 1),      perf::kv_int(kv, "internal_kb", 0),
          perf::kv_int(kv, "profile", 0)};
}

void parse_floats(const std::string &csv, float out[3]) {
  if (csv.empty()) return;
  sscanf(csv.c_str(), "%f,%f,%f", &out[0], &out[1], &out[2]);
}

// Loads one model and checks its I/O against what tinyseek_core implements.
bool load_model(ModelCtx &m, const Settings &s) {
  const std::string base = std::string(perf::kRoot) + "/models/" + m.name;
  m.file = base + ".espdl";
  auto meta = perf::read_kv(base + ".txt");
  parse_floats(meta["mean"], m.norm.mean);
  parse_floats(meta["std"], m.norm.std);
  m.norm.pad_value = static_cast<uint8_t>(perf::kv_int(meta, "pad_value", 114));
  FILE *f = fopen(m.file.c_str(), "rb");
  if (!f) return false;
  fseek(f, 0, SEEK_END);
  m.file_bytes = ftell(f);
  fclose(f);

  mem_row(m.name, "before_load");
  int64_t t0 = perf::now_us();
  // dl_model_base.hpp, identical in ESP-DL 3.3.11 (Arduino) and 3.3.15 (ESP-IDF):
  // (path, location, max_internal_size, mm_type, key, param_copy). max_internal_size = 0
  // keeps every activation in PSRAM; internal_kb in bench.txt moves up to that much of the
  // activation arena to internal RAM. (For SD-card models the loader forces param_copy off:
  // the parameters stay in the file buffer it read into PSRAM.)
  m.model = new dl::Model(m.file.c_str(), fbs::MODEL_LOCATION_IN_SDCARD, s.internal_kb * 1024,
                          dl::MEMORY_MANAGER_GREEDY, nullptr, true);
  int64_t t1 = perf::now_us();
  // is_built() only exists from ESP-DL 3.3.15; get_fbs_model() and the I/O tensors are in
  // both versions. TODO(board): confirm this catches a failed SD load.
  if (!m.model->get_fbs_model() || !m.model->get_input() || !m.model->get_output()) {
    perf::logln("ERROR: %s: model did not load", m.file.c_str());
    return false;
  }
  mem_row(m.name, "after_load");

  m.input = m.model->get_input();
  m.output = m.model->get_output();
  // get_shape() returns a copy in 3.3.11 and a const reference in 3.3.15; copy in both.
  const std::vector<int> is = m.input->get_shape();
  const std::vector<int> os = m.output->get_shape();
  const dl::dtype_t idt = m.input->get_dtype(), odt = m.output->get_dtype();
  if (is.size() != 4 || is[0] != 1 || is[1] != is[2] || is[3] != 3 ||
      (idt != dl::DATA_TYPE_INT8 && idt != dl::DATA_TYPE_INT16)) {
    perf::logln("ERROR: %s: input must be int8 or int16 NHWC [1,S,S,3]", m.name.c_str());
    return false;
  }
  if (os.size() != 4 || os[1] != os[2] || (odt != dl::DATA_TYPE_INT8 && odt != dl::DATA_TYPE_INT16)) {
    perf::logln("ERROR: %s: output must be int8 or int16 NHWC [1,G,G,C]", m.name.c_str());
    return false;
  }
  m.size = is[1];
  m.grid = os[1];
  m.channels = os[3];
  m.in16 = idt == dl::DATA_TYPE_INT16;  // dtypes and exponents read from the model, never assumed
  m.out16 = odt == dl::DATA_TYPE_INT16;
  m.in_exp = m.input->get_exponent();
  m.out_exp = m.output->get_exponent();
  m.dp.grid_h = m.dp.grid_w = m.grid;
  m.dp.num_classes = m.channels - 4;
  m.dp.stride = m.size / m.grid;
  m.dp.exponent = m.out_exp;
  m.dp.score_threshold = perf::kv_float(meta, "score_threshold", 0.05f);
  m.dp.max_detections = perf::kv_int(meta, "max_detections", 20);
  if (perf::kv_int(meta, "input_exponent", m.in_exp) != m.in_exp ||
      perf::kv_int(meta, "output_exponent", m.out_exp) != m.out_exp) {
    perf::logln("WARNING: %s: exponents differ from the staged model.txt (stale card?)", m.name.c_str());
  }
  if ((meta.count("input_dtype") && meta["input_dtype"] != m.in_dtype()) ||
      (meta.count("output_dtype") && meta["output_dtype"] != m.out_dtype())) {
    perf::logln("WARNING: %s: I/O dtypes differ from the staged model.txt (stale card?)", m.name.c_str());
  }
  perf::logln("%s: input [1,%d,%d,3] exp %d, output [1,%d,%d,%d] exp %d, %u bytes", m.name.c_str(), m.size, m.size,
           m.in_exp, m.grid, m.grid, m.channels, m.out_exp, unsigned(m.file_bytes));
  perf::logln("%s: input %s, output %s", m.name.c_str(), m.in_dtype(), m.out_dtype());

  // The plan (resample coefficients, quantisation tables) is per resolution and built
  // once; it is timed as its own row, not as part of per-frame preprocessing.
  m.plan = new tinyseek::PreprocessPlan();
  int64_t t2 = perf::now_us();
  tinyseek::make_plan(*m.plan, kFrameW, kFrameH, m.size, m.in_exp, m.norm);
  int64_t t3 = perf::now_us();
  bench_row(m, s, "load", perf::summarize({t1 - t0}));
  bench_row(m, s, "preprocess_plan", perf::summarize({t3 - t2}));
  return true;
}

void unload(ModelCtx &m) {
  delete m.model;
  delete m.plan;
  m.model = nullptr;
  m.plan = nullptr;
}

// Buffers shared by all models, sized for the largest input (allocated once).
struct Buffers {
  uint8_t *rgb = nullptr;      // 800x600x3, PSRAM, 16-byte aligned for the JPEG decoder
  uint8_t *scratch = nullptr;  // horizontal-pass rows, PSRAM
  void *in = nullptr;          // model input staging (int8 or int16), PSRAM
  size_t scratch_size = 0, in_size = 0;
};

bool ensure_buffers(Buffers &b, const ModelCtx &m) {
  if (!b.rgb) b.rgb = static_cast<uint8_t *>(perf::alloc_psram(kFrameW * kFrameH * 3));
  size_t need_s = tinyseek::scratch_bytes(*m.plan), need_i = m.in_bytes();
  if (need_s > b.scratch_size) {
    perf::free_buffer(b.scratch);
    b.scratch = static_cast<uint8_t *>(perf::alloc_psram(need_s));
    b.scratch_size = need_s;
  }
  if (need_i > b.in_size) {
    perf::free_buffer(b.in);
    b.in = perf::alloc_psram(need_i);
    b.in_size = need_i;
  }
  return b.rgb && b.scratch && b.in;
}

// ESP-DL shares one arena between input, intermediates and output, so the input is
// refilled before every run (outside the timed region).
void set_input(const ModelCtx &m, const void *in) {
  memcpy(m.input->data, in, m.in_bytes());
}

void run_benchmark(ModelCtx &m, const Settings &s, Buffers &b) {
  size_t jlen = 0;
  uint8_t *jpeg = perf::read_file(std::string(perf::kRoot) + "/bench.jpg", &jlen);
  if (!jpeg) fail("bench.jpg missing");
  std::vector<int64_t> t_jpeg, t_pre, t_inf, t_post;
  std::vector<tinyseek::Detection> dets;
  dets.reserve(64);
  for (int i = 0; i < s.warmup + s.runs; ++i) {
    const bool timed = i >= s.warmup;
    int64_t a = perf::now_us();
    if (!perf::decode_jpeg(jpeg, jlen, b.rgb, kFrameW, kFrameH)) fail("bench.jpg decode failed (must be 800x600)");
    int64_t c = perf::now_us();
    preprocess_into(m, b.rgb, b.scratch, b.in);
    int64_t d = perf::now_us();
    set_input(m, b.in);
    int64_t e = perf::now_us();
    m.model->run();
    int64_t f = perf::now_us();
    dets.clear();
    decode_output(m, dets);
    int64_t g = perf::now_us();
    if (timed) {
      t_jpeg.push_back(c - a);
      t_pre.push_back(d - c);
      t_inf.push_back(f - e);
      t_post.push_back(g - f);
    }
    vTaskDelay(1);  // let the idle task run (task watchdog), outside every timed region
  }
  perf::free_buffer(jpeg);
  bench_row(m, s, "jpeg_decode", perf::summarize(t_jpeg));
  bench_row(m, s, "preprocess", perf::summarize(t_pre));
  bench_row(m, s, "inference", perf::summarize(t_inf));
  bench_row(m, s, "postprocess", perf::summarize(t_post));
  perf::logln("%s: bench.jpg gives %u detections", m.name.c_str(), unsigned(dets.size()));
  mem_row(m.name, "after_inference");
}

void write_dets(const std::string &path, const ModelCtx &m, const std::string &cs, const char *src,
                const std::vector<tinyseek::Detection> &dets) {
  for (size_t r = 0; r < dets.size(); ++r) {
    const auto &d = dets[r];
    char buf[200];
    snprintf(buf, sizeof(buf), "%s,%s,%s,%u,%.4f,%.4f,%.4f,%.4f,%.7f,%d", m.name.c_str(), cs.c_str(), src, unsigned(r),
             d.x, d.y, d.w, d.h, d.score, d.cls);
    perf::append_line(path, buf);
  }
}

// One case through the pipeline from an RGB frame already in b.rgb; tensors + detections to the card.
void parity_one(const ModelCtx &m, Buffers &b, const std::string &dir, const std::string &cs, const char *src) {
  preprocess_into(m, b.rgb, b.scratch, b.in);
  perf::write_file(dir + "/" + cs + "_" + src + "_in.bin", b.in, m.in_bytes());  // int16: little-endian
  set_input(m, b.in);
  m.model->run();
  perf::write_file(dir + "/" + cs + "_" + src + "_out.bin", m.output->data, m.out_bytes());
  std::vector<tinyseek::Detection> dets;
  decode_output(m, dets);
  write_dets(dir + "/dets.csv", m, cs, src, dets);
  perf::logln("%s %s %s: %u detections", m.name.c_str(), cs.c_str(), src, unsigned(dets.size()));
}

void run_parity(const ModelCtx &m, Buffers &b) {
  const std::string pdir = std::string(perf::kRoot) + "/parity";
  const std::string out = g_run + "/parity/" + m.name;
  perf::make_dir(g_run + "/parity");
  perf::make_dir(out);
  perf::append_line(out + "/dets.csv", "model,case,path,rank,x,y,w,h,score,cls");
  for (const std::string &cs : perf::read_lines(pdir + "/cases.txt")) {
    size_t n = 0;
    uint8_t *rgb = perf::read_file(pdir + "/" + cs + ".rgb", &n);
    if (rgb && n == size_t(kFrameW) * kFrameH * 3) {
      memcpy(b.rgb, rgb, n);
      parity_one(m, b, out, cs, "rgb");
    } else {
      perf::logln("WARNING: %s.rgb missing or not 800x600", cs.c_str());
    }
    perf::free_buffer(rgb);
    uint8_t *jpg = perf::read_file(pdir + "/" + cs + ".jpg", &n);
    if (jpg && perf::decode_jpeg(jpg, n, b.rgb, kFrameW, kFrameH)) {
      parity_one(m, b, out, cs, "jpg");
    } else {
      perf::logln("WARNING: %s.jpg missing or not decodable at 800x600", cs.c_str());
    }
    perf::free_buffer(jpg);
    vTaskDelay(1);
  }
}

void run_camera(ModelCtx &m, const Settings &s, Buffers &b) {
  std::vector<int64_t> t_cap, t_jpeg, t_total;
  std::vector<tinyseek::Detection> dets;
  int64_t start = 0;
  for (int i = 0; i < s.warmup + s.camera_frames; ++i) {
    if (i == s.warmup) start = perf::now_us();
    int64_t a = perf::now_us();
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) continue;
    int64_t c = perf::now_us();
    bool ok = perf::decode_jpeg(fb->buf, fb->len, b.rgb, kFrameW, kFrameH);
    esp_camera_fb_return(fb);
    int64_t d = perf::now_us();
    if (!ok) continue;
    preprocess_into(m, b.rgb, b.scratch, b.in);
    set_input(m, b.in);
    m.model->run();
    dets.clear();
    decode_output(m, dets);
    int64_t e = perf::now_us();
    if (i >= s.warmup) {
      t_cap.push_back(c - a);
      t_jpeg.push_back(d - c);
      t_total.push_back(e - a);
    }
    vTaskDelay(1);  // counted in the FPS below: a real app has to yield too
  }
  double secs = (perf::now_us() - start) / 1e6;
  double fps = secs > 0 ? t_total.size() / secs : 0;
  bench_row(m, s, "camera_capture", perf::summarize(t_cap));
  bench_row(m, s, "camera_jpeg", perf::summarize(t_jpeg));
  bench_row(m, s, "end_to_end", perf::summarize(t_total), fps);
  mem_row(m.name, "after_camera");
}

}  // namespace

#ifndef ARDUINO
// ESP-IDF entry point; the Arduino sketch calls perf_eval_run() from setup() instead.
extern "C" void app_main(void) {
  vTaskDelay(pdMS_TO_TICKS(1000));  // let a serial monitor attach
  perf_eval_run();
}
#endif

void perf_eval_run() {
  if (!perf::mount_sd_card()) fail("no SD card (FAT32, inserted before power-on?)");
  if (!perf::file_exists(perf::kRoot)) fail("/sdcard/tinyseek_eval missing: copy the staged folder to the card root");
  const Settings s = load_settings();
  const perf::BuildInfo bi = perf::build_info();

  g_run = perf::next_run_dir(std::string(perf::kRoot) + "/results");
  if (!perf::make_dir(g_run)) fail("cannot create the results folder");
  // build and jpeg_decoder tell ESP-IDF and Arduino results apart; the comparison tools
  // read columns by name, so extra columns do not affect them.
  char meta[320];
  snprintf(meta, sizeof(meta), "%s,%s,%s,%s,%s,%s,%d", g_run.substr(g_run.rfind('/') + 1).c_str(), kBoard,
           bi.build, bi.idf_version, bi.espdl_version, perf::jpeg_decoder_name(), perf::cpu_mhz());
  g_meta = meta;
  const char *cols = "run,board,build,idf_version,espdl_version,jpeg_decoder,cpu_mhz";
  perf::append_line(g_run + "/bench.csv", std::string(cols) +
                    ",model,model_bytes,input_size,input_exponent,output_exponent,internal_kb,stage,runs,warmup,"
                    "mean_ms,min_ms,max_ms,std_ms,fps,input_dtype,output_dtype");
  perf::append_line(g_run + "/memory.csv", std::string(cols) +
                    ",model,point,internal_free,internal_min_free,internal_largest,psram_free,psram_min_free,"
                    "psram_largest");
  {
    char buf[800];
    snprintf(buf, sizeof(buf),
             "board=%s\nbuild=%s\nsdk_settings=%s\nidf_version=%s\nespdl_version=%s\ncamera_version=%s\n"
             "jpeg_decoder=%s\ncpu_mhz=%d\n"
             "warmup=%d\nruns=%d\ncamera=%d\ncamera_frames=%d\nparity=%d\ninternal_kb=%d\n"
             "psram_total=%u\ninternal_total=%u\n",
             kBoard, bi.build, perf::sdk_settings().c_str(), bi.idf_version, bi.espdl_version, bi.camera_version,
             perf::jpeg_decoder_name(),
             perf::cpu_mhz(), s.warmup, s.runs, s.camera, s.camera_frames, s.parity, s.internal_kb,
             unsigned(heap_caps_get_total_size(MALLOC_CAP_SPIRAM)),
             unsigned(heap_caps_get_total_size(MALLOC_CAP_INTERNAL)));
    perf::write_file(g_run + "/run.txt", buf, strlen(buf));
  }
  perf::logln("results in %s", g_run.c_str());
  perf::logln("build %s | IDF %s | %s | JPEG %s | %d MHz | %s", bi.build, bi.idf_version, bi.espdl_version,
              perf::jpeg_decoder_name(), perf::cpu_mhz(), perf::sdk_settings().c_str());
  mem_row("-", "boot");

  // The camera is started once, before any model, so its frame buffers are part of every
  // model's memory picture in the same way when the camera stage is on.
  if (s.camera && !perf::camera_init(4000)) fail("camera init failed (ribbon cable seated?)");

  Buffers b;
  for (const std::string &file : perf::list_dir(std::string(perf::kRoot) + "/models", ".espdl")) {
    ModelCtx m;
    m.name = file.substr(0, file.size() - 6);
    if (!load_model(m, s)) {
      perf::logln("ERROR: skipping %s", file.c_str());
      unload(m);
      continue;
    }
    if (!ensure_buffers(b, m)) fail("out of PSRAM for frame buffers");
    run_benchmark(m, s, b);
    if (s.parity) run_parity(m, b);
    if (s.camera) run_camera(m, s, b);
    if (s.profile) m.model->profile();  // per-layer latency and memory table on serial
    unload(m);
    mem_row(m.name, "after_unload");
  }
  if (s.camera) perf::camera_deinit();
  perf::logln("done: %s", g_run.c_str());
  perf::append_line(g_run + "/run.txt", "complete=1");
}
