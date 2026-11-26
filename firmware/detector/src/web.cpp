#include "web.hpp"

#include <Arduino.h>
#include <WiFi.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "esp_heap_caps.h"
#include "esp_http_server.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "web_page.h"

// APIs as in arduino-esp32 3.3.12 (libraries/WiFi/src/WiFiSTA.h, WiFiAP.h, WiFiGeneric.h)
// and ESP-IDF v5.5.5 components/esp_http_server/include/esp_http_server.h, the IDF that
// core bundles. The CameraWebServer example of the same core uses the same server API.

namespace det {

namespace {

WifiMode g_mode = WifiMode::kOff;
httpd_handle_t g_server = nullptr;
WebHooks g_hooks;

// The latest frame, double-buffered by copy: the detector loop copies into it under the
// mutex, the frame handler copies out under the mutex and sends outside it.
SemaphoreHandle_t g_mu = nullptr;
uint8_t *g_jpeg = nullptr;
size_t g_jpeg_len = 0, g_jpeg_cap = 0;
std::string g_meta;
uint32_t g_seq = 0;  // frame number of g_jpeg (0 = none yet)

// Only the server task touches these (one request at a time).
uint8_t *g_send = nullptr;
size_t g_send_cap = 0;

bool grow(uint8_t **buf, size_t *cap, size_t need) {
  if (need <= *cap) return true;
  void *p = heap_caps_realloc(*buf, need, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!p) return false;
  *buf = static_cast<uint8_t *>(p);
  *cap = need;
  return true;
}

uint32_t frame_number(const std::string &meta) {
  const char *p = std::strstr(meta.c_str(), "\"n\":");
  return p ? static_cast<uint32_t>(std::strtoul(p + 4, nullptr, 10)) : 0;
}

esp_err_t send_json(httpd_req_t *req, const std::string &body) {
  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Cache-Control", "no-store");
  return httpd_resp_send(req, body.c_str(), static_cast<ssize_t>(body.size()));
}

esp_err_t index_handler(httpd_req_t *req) {
  httpd_resp_set_type(req, "text/html");
  return httpd_resp_send(req, kIndexHtml, static_cast<ssize_t>(sizeof(kIndexHtml) - 1));
}

esp_err_t frame_handler(httpd_req_t *req) {
  long have = -1;
  char query[48], val[16];
  if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK &&
      httpd_query_key_value(query, "have", val, sizeof(val)) == ESP_OK) {
    have = std::strtol(val, nullptr, 10);
  }
  std::string meta;
  size_t len = 0;
  bool fresh = false;
  xSemaphoreTake(g_mu, portMAX_DELAY);
  if (g_seq != 0 && static_cast<long>(g_seq) != have && grow(&g_send, &g_send_cap, g_jpeg_len)) {
    std::memcpy(g_send, g_jpeg, g_jpeg_len);
    len = g_jpeg_len;
    meta = g_meta;
    fresh = true;
  }
  xSemaphoreGive(g_mu);
  httpd_resp_set_hdr(req, "Cache-Control", "no-store");
  if (!fresh) {
    httpd_resp_set_status(req, "204 No Content");
    return httpd_resp_send(req, nullptr, 0);
  }
  httpd_resp_set_type(req, "image/jpeg");
  httpd_resp_set_hdr(req, "X-Meta", meta.c_str());  // must outlive the send: `meta` does
  return httpd_resp_send(req, reinterpret_cast<const char *>(g_send), static_cast<ssize_t>(len));
}

esp_err_t status_handler(httpd_req_t *req) {
  return send_json(req, g_hooks.status_json ? g_hooks.status_json() : std::string("{}"));
}

esp_err_t prompt_handler(httpd_req_t *req) {
  char query[96], name[64] = "";
  if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
    httpd_query_key_value(query, "name", name, sizeof(name));
  }
  // The browser URL-encodes the text box; prompts are [a-z0-9_], so decoding '+' and %20
  // as spaces (normalized to '_') and dropping other escapes is enough.
  std::string typed;
  for (size_t i = 0; name[i]; ++i) {
    if (name[i] == '+') {
      typed += ' ';
    } else if (name[i] == '%' && name[i + 1] && name[i + 2]) {
      char hex[3] = {name[i + 1], name[i + 2], 0};
      typed += static_cast<char>(std::strtol(hex, nullptr, 16));
      i += 2;
    } else {
      typed += name[i];
    }
  }
  return send_json(req, g_hooks.request_prompt ? g_hooks.request_prompt(typed) : std::string("{}"));
}

bool add(const char *uri, esp_err_t (*fn)(httpd_req_t *)) {
  httpd_uri_t u = {};
  u.uri = uri;
  u.method = HTTP_GET;
  u.handler = fn;
  u.user_ctx = nullptr;
  return httpd_register_uri_handler(g_server, &u) == ESP_OK;
}

}  // namespace

bool wifi_start(const Settings &s, std::string *info) {
  g_mode = s.wifi;
  if (s.wifi == WifiMode::kSta) {
    WiFi.setHostname(s.hostname.c_str());  // before the interface starts
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);  // modem sleep adds latency to every request
    WiFi.begin(s.wifi_ssid.c_str(), s.wifi_password.empty() ? nullptr : s.wifi_password.c_str());
    for (int i = 0; i < 80 && WiFi.status() != WL_CONNECTED; ++i) delay(250);
    if (WiFi.status() != WL_CONNECTED) {
      // Not joined yet: the core keeps retrying in the background, so the interface is up
      // and the web server starts anyway; /status.json shows when the connection comes.
      *info = "{\"mode\":\"sta\",\"ok\":true,\"connected\":false,\"ssid\":\"" + json_escape(s.wifi_ssid) +
              "\",\"hostname\":\"" + json_escape(s.hostname) + "\"}";
      return true;
    }
    *info = "{\"mode\":\"sta\",\"ok\":true,\"connected\":true,\"ip\":\"" +
            std::string(WiFi.localIP().toString().c_str()) +
            "\",\"hostname\":\"" + json_escape(s.hostname) + "\",\"rssi\":" + std::to_string(WiFi.RSSI()) + "}";
    return true;
  }
  if (s.wifi == WifiMode::kAp) {
    WiFi.mode(WIFI_AP);
    if (!WiFi.softAP(s.ap_ssid.c_str(), s.ap_password.c_str())) {
      *info = "{\"mode\":\"ap\",\"ok\":false}";
      return false;
    }
    *info = "{\"mode\":\"ap\",\"ok\":true,\"ssid\":\"" + json_escape(s.ap_ssid) + "\",\"ip\":\"" +
            std::string(WiFi.softAPIP().toString().c_str()) + "\"}";
    return true;
  }
  *info = "{\"mode\":\"off\"}";
  return false;
}

std::string wifi_status_json() {
  if (g_mode == WifiMode::kSta) {
    const bool up = WiFi.status() == WL_CONNECTED;
    return std::string("{\"mode\":\"sta\",\"connected\":") + (up ? "true" : "false") + ",\"ip\":\"" +
           (up ? std::string(WiFi.localIP().toString().c_str()) : std::string()) + "\",\"rssi\":" +
           std::to_string(up ? WiFi.RSSI() : 0) + "}";
  }
  if (g_mode == WifiMode::kAp) {
    return "{\"mode\":\"ap\",\"ip\":\"" + std::string(WiFi.softAPIP().toString().c_str()) +
           "\",\"stations\":" + std::to_string(WiFi.softAPgetStationNum()) + "}";
  }
  return "{\"mode\":\"off\"}";
}

bool web_start(int port, const WebHooks &hooks) {
  g_hooks = hooks;
  if (!g_mu) g_mu = xSemaphoreCreateMutex();
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = static_cast<uint16_t>(port);
  config.stack_size = 8192;     // std::string work in the status handler
  config.core_id = 0;           // off the detector loop's core
  config.max_uri_handlers = 8;
  config.max_open_sockets = 4;
  config.lru_purge_enable = true;  // a phone that drops off Wi-Fi must not use up the sockets
  if (httpd_start(&g_server, &config) != ESP_OK) {
    g_server = nullptr;
    return false;
  }
  bool ok = add("/", index_handler) && add("/frame.jpg", frame_handler) && add("/status.json", status_handler) &&
            add("/prompt", prompt_handler);
  return ok;
}

bool web_running() { return g_server != nullptr; }

void web_publish(const uint8_t *jpeg, size_t len, const std::string &frame_json) {
  if (!g_server || !jpeg || !len) return;
  xSemaphoreTake(g_mu, portMAX_DELAY);
  if (grow(&g_jpeg, &g_jpeg_cap, len)) {
    std::memcpy(g_jpeg, jpeg, len);
    g_jpeg_len = len;
    g_meta = frame_json;
    g_seq = frame_number(frame_json);
  }
  xSemaphoreGive(g_mu);
}

}  // namespace det
