// Optional Wi-Fi and the detector's web page (Arduino WiFi + ESP-IDF esp_http_server).
// Off unless config.txt says wifi=sta or wifi=ap; credentials come only from the card.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "detector_logic.hpp"

namespace det {

// Joins the network (sta, waits up to 20 s) or starts the access point (ap). `info` gets a
// JSON object {"mode":...,"ip":...,...} for the boot log and /status.json. False when
// Wi-Fi could not start (the detector keeps running without it).
bool wifi_start(const Settings &s, std::string *info);
// The current Wi-Fi state as JSON (ip, rssi or connected stations).
std::string wifi_status_json();

// What the server asks of the detector. Both run on the server's task (core 0) while
// the detector loop runs on core 1, so they must be thread-safe.
struct WebHooks {
  std::string (*status_json)() = nullptr;                   // the whole /status.json body
  std::string (*request_prompt)(const std::string &typed);  // the /prompt?name= reply (JSON)
};

bool web_start(int port, const WebHooks &hooks);
bool web_running();

// Hands the server the latest frame: the camera JPEG as captured and its frame JSON
// (frame_json(); becomes the X-Meta header). Copied, so the caller may reuse its buffers.
void web_publish(const uint8_t *jpeg, size_t len, const std::string &frame_json);

}  // namespace det
