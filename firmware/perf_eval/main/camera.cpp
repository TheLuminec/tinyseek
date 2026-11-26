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

// For decode_jpeg_scaled: TJpgDec through esp_jpeg wherever its header is on the include
// path (the Arduino core; the ESP-IDF build would need the espressif/esp_jpeg component),
// esp_new_jpeg wherever that is (both builds).
#if defined(__has_include)
#if __has_include("jpeg_decoder.h")
#define TINYSEEK_HAVE_ESP_JPEG 1
#include "jpeg_decoder.h"
#endif
#if __has_include("esp_jpeg_dec.h")
#define TINYSEEK_HAVE_ESP_NEW_JPEG 1
#include "esp_jpeg_dec.h"
#endif
#endif
#ifndef TINYSEEK_HAVE_ESP_JPEG
#define TINYSEEK_HAVE_ESP_JPEG 0
#endif
#ifndef TINYSEEK_HAVE_ESP_NEW_JPEG
#define TINYSEEK_HAVE_ESP_NEW_JPEG 0
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

const char *jpeg_decoder_label(JpegDecoder d) {
  return d == JpegDecoder::kTjpgd ? "tjpgd (ROM TJpgDec via esp_jpeg)" : "esp_new_jpeg";
}

bool jpeg_decoder_available(JpegDecoder d) {
  return d == JpegDecoder::kTjpgd ? TINYSEEK_HAVE_ESP_JPEG : TINYSEEK_HAVE_ESP_NEW_JPEG;
}

bool decode_jpeg_scaled(const uint8_t *jpeg, size_t len, JpegDecoder decoder, int scale_shift, uint8_t *rgb,
                        size_t capacity, int *out_w, int *out_h) {
  tinyseek::JpegInfo info;
  if (!tinyseek::jpeg_info(jpeg, len, info) || info.progressive || info.components != 3) {
    logln("JPEG not decodable on the board: %dx%d, %d components%s", info.width, info.height, info.components,
          info.progressive ? ", progressive" : "");
    return false;
  }
  if (scale_shift < 0 || scale_shift > 3) return false;
  const int w = info.width >> scale_shift, h = info.height >> scale_shift;
  if (size_t(w) * h * 3 > capacity) {
    logln("JPEG %dx%d at 1/%d does not fit the %u-byte frame buffer", info.width, info.height, 1 << scale_shift,
          unsigned(capacity));
    return false;
  }
  if (decoder == JpegDecoder::kTjpgd) {
#if TINYSEEK_HAVE_ESP_JPEG
    // esp_jpeg 1.3.1 (jpeg_decoder.h): with CONFIG_JD_USE_ROM it runs the ROM TJpgDec R0.01b.
    // Our own work buffer in internal RAM, so this does not share esp32-camera's static one.
    static uint8_t work[3100];  // esp_jpeg's JPEG_WORK_BUF_SIZE
    esp_jpeg_image_cfg_t cfg = {};
    cfg.indata = const_cast<uint8_t *>(jpeg);
    cfg.indata_size = static_cast<uint32_t>(len);
    cfg.outbuf = rgb;
    cfg.outbuf_size = static_cast<uint32_t>(capacity);
    cfg.out_format = JPEG_IMAGE_FORMAT_RGB888;
    cfg.out_scale = static_cast<esp_jpeg_image_scale_t>(scale_shift);  // 0, 1/2, 1/4, 1/8 in enum order
    cfg.flags.swap_color_bytes = 0;  // TJpgDec emits R,G,B
    cfg.advanced.working_buffer = work;
    cfg.advanced.working_buffer_size = sizeof(work);
    esp_jpeg_image_output_t img = {};
    if (esp_jpeg_decode(&cfg, &img) != ESP_OK) return false;
    *out_w = img.width;
    *out_h = img.height;
    return img.width == w && img.height == h;
#else
    logln("TJpgDec (esp_jpeg) is not in this build");
    return false;
#endif
  }
#if TINYSEEK_HAVE_ESP_NEW_JPEG
  if (scale_shift != 0) {
    logln("esp_new_jpeg is used at full scale only (decode_scale=0)");
    return false;
  }
  // esp_new_jpeg 1.0.2 include/esp_jpeg_dec.h; output must be 16-byte aligned.
  jpeg_dec_config_t config = DEFAULT_JPEG_DEC_CONFIG();
  config.output_type = JPEG_PIXEL_FORMAT_RGB888;
  jpeg_dec_handle_t dec = nullptr;
  if (jpeg_dec_open(&config, &dec) != JPEG_ERR_OK) return false;
  jpeg_dec_io_t io = {};
  jpeg_dec_header_info_t hdr = {};
  io.inbuf = const_cast<uint8_t *>(jpeg);
  io.inbuf_len = static_cast<int>(len);
  bool ok = jpeg_dec_parse_header(dec, &io, &hdr) == JPEG_ERR_OK && hdr.width == info.width &&
            hdr.height == info.height;
  if (ok) {
    io.outbuf = rgb;
    ok = jpeg_dec_process(dec, &io) == JPEG_ERR_OK;
  }
  jpeg_dec_close(dec);
  *out_w = w;
  *out_h = h;
  return ok;
#else
  logln("esp_new_jpeg is not in this build");
  return false;
#endif
}

}  // namespace perf
