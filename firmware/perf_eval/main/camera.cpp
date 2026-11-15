#include "camera.hpp"

#include "esp_camera.h"
#include "esp_jpeg_dec.h"
#include "esp_log.h"
#include "esp_timer.h"

#define BOARD_FREENOVE_S3_CAM
#include "board_pins.h"

static const char *TAG = "camera";

namespace perf {

bool camera_init(uint32_t warmup_ms) {
  camera_config_t c = {};
  c.pin_pwdn = CAM_PIN_PWDN;
  c.pin_reset = CAM_PIN_RESET;
  c.pin_xclk = CAM_PIN_XCLK;
  c.pin_sccb_sda = CAM_PIN_SIOD;
  c.pin_sccb_scl = CAM_PIN_SIOC;
  c.pin_d7 = CAM_PIN_D7;
  c.pin_d6 = CAM_PIN_D6;
  c.pin_d5 = CAM_PIN_D5;
  c.pin_d4 = CAM_PIN_D4;
  c.pin_d3 = CAM_PIN_D3;
  c.pin_d2 = CAM_PIN_D2;
  c.pin_d1 = CAM_PIN_D1;
  c.pin_d0 = CAM_PIN_D0;
  c.pin_vsync = CAM_PIN_VSYNC;
  c.pin_href = CAM_PIN_HREF;
  c.pin_pclk = CAM_PIN_PCLK;
  c.xclk_freq_hz = 20000000;
  c.ledc_timer = LEDC_TIMER_0;
  c.ledc_channel = LEDC_CHANNEL_0;
  c.pixel_format = PIXFORMAT_JPEG;
  c.frame_size = FRAMESIZE_SVGA;
  c.jpeg_quality = 10;
  c.fb_count = 2;
  c.fb_location = CAMERA_FB_IN_PSRAM;
  c.grab_mode = CAMERA_GRAB_LATEST;
  esp_err_t err = esp_camera_init(&c);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "esp_camera_init failed: 0x%x", err);
    return false;
  }
  sensor_t *s = esp_camera_sensor_get();
  s->set_vflip(s, 0);
  s->set_hmirror(s, 0);
  s->set_whitebal(s, 1);
  s->set_awb_gain(s, 1);
  s->set_wb_mode(s, 0);
  s->set_exposure_ctrl(s, 1);
  s->set_aec2(s, 0);
  s->set_ae_level(s, 0);
  s->set_gain_ctrl(s, 1);
  s->set_gainceiling(s, GAINCEILING_4X);
  s->set_bpc(s, 1);
  s->set_wpc(s, 1);
  s->set_lenc(s, 1);
  s->set_raw_gma(s, 1);
  for (int64_t start = esp_timer_get_time(); esp_timer_get_time() - start < int64_t(warmup_ms) * 1000;) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb) esp_camera_fb_return(fb);
  }
  return true;
}

void camera_deinit() { esp_camera_deinit(); }

bool decode_jpeg(const uint8_t *jpeg, size_t len, uint8_t *rgb, int w, int h) {
  // Decoder API per esp_new_jpeg 1.0.2 include/esp_jpeg_dec.h and its test_decoder.c.
  jpeg_dec_config_t config = DEFAULT_JPEG_DEC_CONFIG();
  config.output_type = JPEG_PIXEL_FORMAT_RGB888;
  jpeg_dec_handle_t dec = nullptr;
  if (jpeg_dec_open(&config, &dec) != JPEG_ERR_OK) return false;
  jpeg_dec_io_t io = {};
  jpeg_dec_header_info_t info = {};
  io.inbuf = const_cast<uint8_t *>(jpeg);
  io.inbuf_len = static_cast<int>(len);
  bool ok = jpeg_dec_parse_header(dec, &io, &info) == JPEG_ERR_OK && info.width == w && info.height == h;
  if (ok) {
    io.outbuf = rgb;
    ok = jpeg_dec_process(dec, &io) == JPEG_ERR_OK;
  } else {
    ESP_LOGE(TAG, "JPEG is %dx%d, expected %dx%d", info.width, info.height, w, h);
  }
  jpeg_dec_close(dec);
  return ok;
}

}  // namespace perf
