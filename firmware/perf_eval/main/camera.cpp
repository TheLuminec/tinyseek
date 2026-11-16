#include "camera.hpp"

#include "esp_camera.h"
#include "esp_timer.h"
#include "platform.hpp"
#include "tinyseek_core/jpeg_info.hpp"

#ifdef ARDUINO
#include "arduino_config.h"  // only in the sketch
#endif

// The ESP-IDF build always uses esp_new_jpeg; the Arduino sketch uses esp32-camera's
// fmt2rgb888 unless arduino_config.h selects esp_new_jpeg.
#if defined(ARDUINO) && !TINYSEEK_ARDUINO_USE_ESP_NEW_JPEG
#define TINYSEEK_JPEG_FMT2RGB888 1
#include "img_converters.h"
#else
#define TINYSEEK_JPEG_FMT2RGB888 0
#include "esp_jpeg_dec.h"
#endif

#define BOARD_FREENOVE_S3_CAM
#include "board_pins.h"

namespace perf {

#if TINYSEEK_JPEG_FMT2RGB888

const char *jpeg_decoder_name() { return TINYSEEK_ARDUINO_JPEG_DECODER; }

// esp32-camera's converter (img_converters.h, bundled with the Arduino core). It is a
// different decoder from the ESP-IDF build's, so jpeg_decode times and the jpg parity path
// are not comparable between the two builds; the jpeg_decoder column says which ran.
static bool decode_with_esp32_camera(const uint8_t *jpeg, size_t len, uint8_t *rgb, int w, int h) {
  if (!fmt2rgb888(jpeg, len, PIXFORMAT_JPEG, rgb)) return false;
#if TINYSEEK_FMT2RGB888_IS_BGR
  // The model wants R,G,B; swapping is part of the timed decode, as a real app would pay it.
  for (size_t i = 0, n = size_t(w) * h * 3; i < n; i += 3) {
    const uint8_t t = rgb[i];
    rgb[i] = rgb[i + 2];
    rgb[i + 2] = t;
  }
#else
  (void)w;
  (void)h;
#endif
  return true;
}

#else

const char *jpeg_decoder_name() { return "esp_new_jpeg " TINYSEEK_JPEG_VERSION; }

#endif

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
    logln("esp_camera_init failed: 0x%x", err);
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
  // Check the header first: the output buffer holds exactly w x h, and the Arduino
  // decoder does not report the size it decoded.
  tinyseek::JpegInfo info;
  if (!tinyseek::jpeg_info(jpeg, len, info) || info.width != w || info.height != h || info.progressive) {
    logln("JPEG is %dx%d%s, expected baseline %dx%d", info.width, info.height,
          info.progressive ? " progressive" : "", w, h);
    return false;
  }
#if TINYSEEK_JPEG_FMT2RGB888
  return decode_with_esp32_camera(jpeg, len, rgb, w, h);
#else
  // Decoder API per esp_new_jpeg 1.0.2 include/esp_jpeg_dec.h and its test_decoder.c.
  jpeg_dec_config_t config = DEFAULT_JPEG_DEC_CONFIG();
  config.output_type = JPEG_PIXEL_FORMAT_RGB888;  // R,G,B byte order
  jpeg_dec_handle_t dec = nullptr;
  if (jpeg_dec_open(&config, &dec) != JPEG_ERR_OK) return false;
  jpeg_dec_io_t io = {};
  jpeg_dec_header_info_t hdr = {};
  io.inbuf = const_cast<uint8_t *>(jpeg);
  io.inbuf_len = static_cast<int>(len);
  bool ok = jpeg_dec_parse_header(dec, &io, &hdr) == JPEG_ERR_OK && hdr.width == w && hdr.height == h;
  if (ok) {
    io.outbuf = rgb;
    ok = jpeg_dec_process(dec, &io) == JPEG_ERR_OK;
  }
  jpeg_dec_close(dec);
  return ok;
#endif
}

}  // namespace perf
