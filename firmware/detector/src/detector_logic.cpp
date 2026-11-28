#include "detector_logic.hpp"

#include <cctype>
#include <cstdio>
#include <cstdlib>

#include "tinyseek_core/kv.hpp"

namespace det {

namespace {

bool parse_int(const std::string &v, int lo, int hi, int *out) {
  if (v.empty()) return false;
  char *end = nullptr;
  long x = std::strtol(v.c_str(), &end, 10);
  if (*end != '\0' || x < lo || x > hi) return false;
  *out = static_cast<int>(x);
  return true;
}

bool parse_float(const std::string &v, float lo, float hi, float *out) {
  if (v.empty()) return false;
  char *end = nullptr;
  float x = std::strtof(v.c_str(), &end);
  if (*end != '\0' || !(x >= lo && x <= hi)) return false;
  *out = x;
  return true;
}

bool parse_bool(const std::string &v, bool *out) {
  if (v == "1" || v == "on" || v == "true" || v == "yes") return *out = true, true;
  if (v == "0" || v == "off" || v == "false" || v == "no") return *out = false, true;
  return false;
}

void parse_floats3(const std::string &csv, float out[3]) {
  float a, b, c;
  if (std::sscanf(csv.c_str(), "%f,%f,%f", &a, &b, &c) == 3) {
    out[0] = a;
    out[1] = b;
    out[2] = c;
  }
}

std::string fmt(const char *f, double v) {
  char buf[48];
  std::snprintf(buf, sizeof(buf), f, v);
  return buf;
}

std::string times_json(const StageTimes &t) {
  return "{\"cap\":" + fmt("%.1f", t.cap) + ",\"dec\":" + fmt("%.1f", t.dec) + ",\"pre\":" + fmt("%.1f", t.pre) +
         ",\"inf\":" + fmt("%.1f", t.inf) + ",\"post\":" + fmt("%.2f", t.post) + ",\"lat\":" + fmt("%.1f", t.lat) +
         ",\"period\":" + fmt("%.1f", t.period) + "}";
}

std::string dets_json(const std::vector<tinyseek::Detection> *dets, float min_score, bool full, int *count) {
  std::string out = "[";
  int n = 0;
  if (dets) {
    for (const auto &d : *dets) {
      if (d.score < min_score) continue;
      char buf[160];
      if (full) {
        std::snprintf(buf, sizeof(buf), "{\"s\":%.6f,\"x\":%.3f,\"y\":%.3f,\"w\":%.3f,\"h\":%.3f,\"c\":%d}",
                      d.score, d.x, d.y, d.w, d.h, d.cls);
      } else {
        std::snprintf(buf, sizeof(buf), "{\"s\":%.3f,\"x\":%.1f,\"y\":%.1f,\"w\":%.1f,\"h\":%.1f}", d.score, d.x,
                      d.y, d.w, d.h);
      }
      if (n++) out += ",";
      out += buf;
    }
  }
  if (count) *count = n;
  return out + "]";
}

}  // namespace

Settings parse_settings(const KeyValues &kv, std::vector<std::string> &warnings) {
  Settings s;
  auto bad = [&](const std::string &k, const std::string &v) {
    warnings.push_back("config.txt: invalid " + k + "=" + v + ", using the default");
  };
  auto trim = [](const std::string &x) {
    size_t a = x.find_first_not_of(" \t\r"), b = x.find_last_not_of(" \t\r");
    return a == std::string::npos ? std::string() : x.substr(a, b - a + 1);
  };
  for (const auto &[raw_k, raw_v] : kv) {
    const std::string k = trim(raw_k), v = trim(raw_v);
    if (k == "prompt") {
      const std::string p = normalize_prompt(v);
      if (p.empty() || valid_prompt(p)) s.prompt = p; else bad(k, v);
    } else if (k == "source") {
      if (v == "camera") s.source = Source::kCamera;
      else if (v == "replay") s.source = Source::kReplay;
      else bad(k, v);
    } else if (k == "decode_scale") {
      if (!parse_int(v, 0, 3, &s.decode_scale)) bad(k, v);
    } else if (k == "resize") {
      if (v == "pillow") s.bilinear = false;
      else if (v == "bilinear") s.bilinear = true;
      else bad(k, v);
    } else if (k == "decoder") {
      if (v == "tjpgd") s.esp_new_jpeg = false;
      else if (v == "esp_new_jpeg") s.esp_new_jpeg = true;
      else bad(k, v);
    } else if (k == "multicore") {
      if (!parse_bool(v, &s.multicore)) bad(k, v);
    } else if (k == "internal_kb") {
      if (!parse_int(v, 0, 512, &s.internal_kb)) bad(k, v);
    } else if (k == "pipeline") {
      if (!parse_bool(v, &s.pipeline)) bad(k, v);
    } else if (k == "led") {
      if (v == "both") s.led = LedMode::kBoth;
      else if (v == "onboard") s.led = LedMode::kOnboard;
      else if (v == "external") s.led = LedMode::kExternal;
      else if (v == "off") s.led = LedMode::kOff;
      else bad(k, v);
    } else if (k == "print_min_score") {
      if (!parse_float(v, 0.0f, 1.0f, &s.print_min_score)) bad(k, v);
    } else if (k == "wifi") {
      if (v == "off") s.wifi = WifiMode::kOff;
      else if (v == "sta") s.wifi = WifiMode::kSta;
      else if (v == "ap") s.wifi = WifiMode::kAp;
      else bad(k, v);
    } else if (k == "wifi_ssid") {
      s.wifi_ssid = v;
    } else if (k == "wifi_password") {
      s.wifi_password = v;
    } else if (k == "ap_ssid") {
      if (!v.empty() && v.size() <= 32) s.ap_ssid = v; else bad(k, v);
    } else if (k == "ap_password") {
      s.ap_password = v;
    } else if (k == "hostname") {
      if (!v.empty() && v.size() <= 32) s.hostname = v; else bad(k, v);
    } else if (k == "http_port") {
      if (!parse_int(v, 1, 65535, &s.http_port)) bad(k, v);
    } else if (k == "camera_warmup_ms") {
      if (!parse_int(v, 0, 60000, &s.camera_warmup_ms)) bad(k, v);
    } else if (k == "replay_dir") {
      if (!v.empty() && v[0] == '/') s.replay_dir = v; else bad(k, v);
    } else if (k == "replay_limit") {
      if (!parse_int(v, 0, 1000000, &s.replay_limit)) bad(k, v);
    } else {
      warnings.push_back("config.txt: unknown key " + k + " (ignored)");
    }
  }
  if (s.esp_new_jpeg && s.decode_scale != 0) {
    warnings.push_back("config.txt: decoder=esp_new_jpeg decodes at full size only; using decode_scale=0");
    s.decode_scale = 0;
  }
  if (s.wifi == WifiMode::kSta && s.wifi_ssid.empty()) {
    warnings.push_back("config.txt: wifi=sta needs wifi_ssid; Wi-Fi stays off");
    s.wifi = WifiMode::kOff;
  }
  if (s.wifi == WifiMode::kAp && (s.ap_password.size() < 8 || s.ap_password.size() > 63)) {
    warnings.push_back("config.txt: wifi=ap needs ap_password (8-63 characters); Wi-Fi stays off");
    s.wifi = WifiMode::kOff;
  }
  return s;
}

std::string settings_json(const Settings &s) {
  static const char *kLed[] = {"both", "onboard", "external", "off"};
  static const char *kWifi[] = {"off", "sta", "ap"};
  std::string o = "{\"prompt\":\"" + json_escape(s.prompt) + "\"";
  o += std::string(",\"source\":\"") + (s.source == Source::kReplay ? "replay" : "camera") + "\"";
  o += ",\"decode_scale\":" + std::to_string(s.decode_scale);
  o += std::string(",\"resize\":\"") + (s.bilinear ? "bilinear" : "pillow") + "\"";
  o += std::string(",\"decoder\":\"") + (s.esp_new_jpeg ? "esp_new_jpeg" : "tjpgd") + "\"";
  o += ",\"multicore\":" + std::to_string(s.multicore ? 1 : 0);
  o += ",\"internal_kb\":" + std::to_string(s.internal_kb);
  o += ",\"pipeline\":" + std::to_string(s.pipeline ? 1 : 0);
  o += std::string(",\"led\":\"") + kLed[static_cast<int>(s.led)] + "\"";
  o += ",\"print_min_score\":" + fmt("%.3f", s.print_min_score);
  o += std::string(",\"wifi\":\"") + kWifi[static_cast<int>(s.wifi)] + "\"";
  if (s.wifi == WifiMode::kSta) o += ",\"wifi_ssid\":\"" + json_escape(s.wifi_ssid) + "\"";
  if (s.wifi == WifiMode::kAp) o += ",\"ap_ssid\":\"" + json_escape(s.ap_ssid) + "\"";
  o += ",\"http_port\":" + std::to_string(s.http_port);
  o += ",\"replay_dir\":\"" + json_escape(s.replay_dir) + "\"";
  o += ",\"replay_limit\":" + std::to_string(s.replay_limit) + "}";
  return o;
}

std::string normalize_prompt(const std::string &typed) {
  size_t a = 0, b = typed.size();
  while (a < b && std::isspace(static_cast<unsigned char>(typed[a]))) ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(typed[b - 1]))) --b;
  std::string out;
  for (size_t i = a; i < b; ++i) {
    char c = static_cast<char>(std::tolower(static_cast<unsigned char>(typed[i])));
    if (c == ' ' || c == '-') c = '_';
    out += c;
  }
  return out;
}

bool valid_prompt(const std::string &name) {
  if (name.empty() || name.size() > 32) return false;
  for (char c : name) {
    if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_')) return false;
  }
  return true;
}

std::string with_prompt(const std::string &text, const std::string &prompt) {
  // A line is a prompt line when its key, read as tinyseek::parse_kv reads it (trimmed,
  // after a leading BOM on the first line), is "prompt". The first one is replaced and any
  // others dropped, so the file cannot hold a second prompt= that wins on the next boot.
  std::string out;
  bool replaced = false;
  size_t pos = 0;
  if (text.compare(0, 3, "\xEF\xBB\xBF") == 0) {
    out += text.substr(0, 3);
    pos = 3;
  }
  while (pos < text.size()) {
    size_t end = text.find('\n', pos);
    std::string line = text.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
    pos = end == std::string::npos ? text.size() : end + 1;
    const std::string t = tinyseek::trim(line);
    const size_t eq = t.find('=');
    const bool is_prompt = !t.empty() && t[0] != '#' && eq != std::string::npos && tinyseek::trim(t.substr(0, eq)) == "prompt";
    if (is_prompt) {
      if (!replaced) out += "prompt=" + prompt + "\n";
      replaced = true;
      continue;
    }
    out += line + "\n";
  }
  if (!replaced) out += "prompt=" + prompt + "\n";
  return out;
}

bool config_for_save(bool exists, bool read_ok, const std::string &text, const std::string &prompt,
                     std::string *out) {
  if (exists && !read_ok) return false;  // never replace a config we could not read
  *out = with_prompt(exists ? text : std::string(), prompt);
  return true;
}

ConfigRecovery config_recovery(bool config_exists, bool tmp_exists) {
  if (!tmp_exists) return ConfigRecovery::kNone;
  // Saving writes config.tmp completely, removes config.txt, then renames. With both
  // present the save stopped before the remove: config.txt is intact, the copy is stale or
  // partial. With only the copy, the save stopped between remove and rename.
  return config_exists ? ConfigRecovery::kDeleteTmp : ConfigRecovery::kRestoreTmp;
}

PromptAction prompt_action(const std::string &pending, const std::string &current, bool replay_active) {
  if (pending.empty() || pending == current) return PromptAction::kNone;
  return replay_active ? PromptAction::kRefuseReplay : PromptAction::kSwitch;
}

ModelMeta parse_model_meta(const KeyValues &kv) {
  ModelMeta m;
  auto get = [&](const char *k) -> std::string {
    auto it = kv.find(k);
    return it == kv.end() ? std::string() : it->second;
  };
  int v = 0;
  float f = 0;
  m.prompt = get("prompt");
  if (parse_int(get("input_size"), 1, 4096, &v)) m.input_w = m.input_h = v;  // square
  int iw = 0, ih = 0;
  if (parse_int(get("input_width"), 1, 4096, &iw) && parse_int(get("input_height"), 1, 4096, &ih)) {
    m.input_w = iw;  // rectangular (both keys are needed)
    m.input_h = ih;
  }
  if (parse_int(get("stride"), 1, 64, &v)) m.stride = v;
  if (parse_int(get("num_classes"), 1, 1000, &v)) m.num_classes = v;
  int ie = 0, oe = 0;
  if (parse_int(get("input_exponent"), -32, 32, &ie) && parse_int(get("output_exponent"), -32, 32, &oe)) {
    m.input_exponent = ie;
    m.output_exponent = oe;
    m.has_exponents = true;
  }
  parse_floats3(get("mean"), m.mean);
  parse_floats3(get("std"), m.std);
  if (parse_int(get("pad_value"), 0, 255, &v)) m.pad_value = v;
  if (parse_float(get("score_threshold"), 0.0f, 1.0f, &f)) m.score_threshold = f;
  if (parse_int(get("max_detections"), 1, 1000, &v)) m.max_detections = v;
  if (parse_float(get("detect_threshold"), 0.0f, 1.0f, &f)) m.detect_threshold = f;
  m.sha256 = get("espdl_sha256");
  m.input_dtype = get("input_dtype");
  m.output_dtype = get("output_dtype");
  m.run = get("run");
  return m;
}

std::string model_shape_json(int in_w, int in_h, int grid_w, int grid_h) {
  if (in_w == in_h && grid_w == grid_h) {
    return "\"input\":" + std::to_string(in_w) + ",\"grid\":" + std::to_string(grid_w);
  }
  return "\"input_w\":" + std::to_string(in_w) + ",\"input_h\":" + std::to_string(in_h) +
         ",\"grid_w\":" + std::to_string(grid_w) + ",\"grid_h\":" + std::to_string(grid_h);
}

std::string model_shape_kv(int in_w, int in_h) {
  if (in_w == in_h) return "input_size=" + std::to_string(in_w);
  return "input_width=" + std::to_string(in_w) + "\ninput_height=" + std::to_string(in_h);
}

std::string frame_json(const FrameReport &r, float min_score) {
  int shown = 0;
  const std::string dets = dets_json(r.dets, min_score, false, &shown);
  bool hit = false;
  if (r.dets)
    for (const auto &d : *r.dets) hit |= d.score >= r.detect_threshold;
  std::string o = "{\"type\":\"frame\",\"n\":" + std::to_string(r.n) + ",\"prompt\":\"" + json_escape(r.prompt) +
                  "\",\"src\":\"" + json_escape(r.source) + "\"";
  if (!r.file.empty()) o += ",\"file\":\"" + json_escape(r.file) + "\"";
  o += ",\"w\":" + std::to_string(r.width) + ",\"h\":" + std::to_string(r.height);
  o += std::string(",\"ok\":") + (r.ok ? "true" : "false");
  if (!r.ok) o += ",\"error\":\"" + json_escape(r.error) + "\"";
  o += ",\"t\":" + times_json(r.t) + ",\"fps\":" + fmt("%.2f", r.fps);
  o += ",\"thr\":" + fmt("%.3f", r.detect_threshold);
  o += std::string(",\"hit\":") + (hit ? "true" : "false");
  o += ",\"ndet\":" + std::to_string(r.dets ? r.dets->size() : 0) + ",\"dets\":" + dets + "}";
  return o;
}

std::string replay_json(const FrameReport &r) {
  std::string o = "{\"file\":\"" + json_escape(r.file) + "\"";
  o += ",\"w\":" + std::to_string(r.width) + ",\"h\":" + std::to_string(r.height);
  o += std::string(",\"ok\":") + (r.ok ? "true" : "false");
  if (!r.ok) o += ",\"error\":\"" + json_escape(r.error) + "\"";
  o += ",\"t\":" + times_json(r.t) + ",\"dets\":" + dets_json(r.dets, -1.0f, true, nullptr) + "}";
  return o;
}

std::string json_escape(const std::string &s) {
  std::string o;
  for (unsigned char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          o += buf;
        } else {
          o += static_cast<char>(c);
        }
    }
  }
  return o;
}

std::string json_string_array(const std::vector<std::string> &items) {
  std::string o = "[";
  for (size_t i = 0; i < items.size(); ++i) {
    if (i) o += ",";
    o += "\"" + json_escape(items[i]) + "\"";
  }
  return o + "]";
}

void StageStats::add(double ms) {
  if (n == 0 || ms < min) min = ms;
  if (n == 0 || ms > max) max = ms;
  sum += ms;
  ++n;
}

}  // namespace det
