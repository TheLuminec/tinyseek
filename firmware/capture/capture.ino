// Interval photo capture to microSD, for building the ESP32-camera image set
// (domain LoRA training and the on-device test set).
//
// Each boot starts a new folder /tinyseek/session_NNN/ on the card holding
//   img_00001.jpg, img_00002.jpg, ...   one JPEG every CAPTURE_INTERVAL_MS
//   captures.csv                         file, ms since boot, size, bytes
//   session.txt                          the capture settings used
//
// BOOT button: pause / resume. LED: one short blink per photo, steady on while
// paused, fast blinking forever on an error (no card, camera failure, card full).
// Settings and the serial log are documented in README.md next to this file.

#define BOARD_FREENOVE_S3_CAM
#include "board_pins.h"

#include <FS.h>
#include <SD_MMC.h>
#include "esp_camera.h"

// ---- Capture settings -------------------------------------------------------
static const uint32_t CAPTURE_INTERVAL_MS = 5000;     // time between photos
static const framesize_t FRAME_SIZE = FRAMESIZE_SVGA; // 800x600; UXGA = 1600x1200 max
static const int JPEG_QUALITY = 10;                   // 0-63, lower = better quality
static const uint32_t WARMUP_MS = 4000;               // frames dropped while AEC/AWB settle
static const uint32_t MAX_IMAGES = 0;                 // stop after this many; 0 = no limit
static const bool VFLIP = true;                       // check the first photo and
static const bool HMIRROR = false;                    // flip these if it is upside down

// ---- Sensor tuning (OV2640) --------------------------------------------------
// The library's defaults (2x gain ceiling, no DSP exposure) leave indoor shots dark,
// and the auto white balance drifts green when the image is that underexposed.
// The detector firmware must use these same values so training and deployment
// images match; every value is written to session.txt.
static const gainceiling_t GAIN_CEILING = GAINCEILING_16X;  // 2X..128X; higher = brighter, noisier
static const int AE_LEVEL = 1;                              // exposure bias, -2..2
static const bool AEC_DSP = true;                           // "aec2": DSP exposure, better in low light
static const int WB_MODE = 0;  // 0 auto, 1 sunny, 2 cloudy, 3 office, 4 home (fixed presets)
static const uint64_t MIN_FREE_BYTES = 4ULL * 1024 * 1024;
static const char *ROOT_DIR = "/tinyseek";
// -----------------------------------------------------------------------------

static char sessionDir[40];
static uint32_t imageCount = 0;
static uint32_t lastCaptureMs = 0;
static bool paused = false;

static void setLed(bool on) {
  if (LED_PIN >= 0) digitalWrite(LED_PIN, on ? LED_ON : !LED_ON);
}

// Unrecoverable: report on serial and blink fast until reset.
static void fail(const char *why) {
  Serial.printf("ERROR: %s\n", why);
  for (;;) {
    setLed(true);
    delay(100);
    setLed(false);
    delay(100);
  }
}

static bool initCamera() {
  camera_config_t config = {};
  config.pin_pwdn = CAM_PIN_PWDN;
  config.pin_reset = CAM_PIN_RESET;
  config.pin_xclk = CAM_PIN_XCLK;
  config.pin_sccb_sda = CAM_PIN_SIOD;
  config.pin_sccb_scl = CAM_PIN_SIOC;
  config.pin_d7 = CAM_PIN_D7;
  config.pin_d6 = CAM_PIN_D6;
  config.pin_d5 = CAM_PIN_D5;
  config.pin_d4 = CAM_PIN_D4;
  config.pin_d3 = CAM_PIN_D3;
  config.pin_d2 = CAM_PIN_D2;
  config.pin_d1 = CAM_PIN_D1;
  config.pin_d0 = CAM_PIN_D0;
  config.pin_vsync = CAM_PIN_VSYNC;
  config.pin_href = CAM_PIN_HREF;
  config.pin_pclk = CAM_PIN_PCLK;
  config.xclk_freq_hz = 20000000;
  config.ledc_timer = LEDC_TIMER_0;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAME_SIZE;
  config.jpeg_quality = JPEG_QUALITY;
  config.fb_count = 2;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_LATEST;  // always the newest frame, never a stale one

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("esp_camera_init failed: 0x%x\n", err);
    return false;
  }
  sensor_t *s = esp_camera_sensor_get();
  s->set_vflip(s, VFLIP);
  s->set_hmirror(s, HMIRROR);
  s->set_whitebal(s, 1);
  s->set_awb_gain(s, 1);
  s->set_wb_mode(s, WB_MODE);
  s->set_exposure_ctrl(s, 1);
  s->set_aec2(s, AEC_DSP);
  s->set_ae_level(s, AE_LEVEL);
  s->set_gain_ctrl(s, 1);
  s->set_gainceiling(s, GAIN_CEILING);
  s->set_bpc(s, 1);   // black-pixel correction
  s->set_wpc(s, 1);   // white-pixel correction
  s->set_lenc(s, 1);  // lens shading correction
  s->set_raw_gma(s, 1);
  return true;
}

static bool initSdCard() {
  SD_MMC.setPins(SD_PIN_CLK, SD_PIN_CMD, SD_PIN_D0);
  if (!SD_MMC.begin("/sdcard", true /* 1-bit mode */)) return false;
  if (SD_MMC.cardType() == CARD_NONE) return false;
  Serial.printf("SD card: %llu MB, %llu MB used\n", (unsigned long long)(SD_MMC.totalBytes() >> 20),
                (unsigned long long)(SD_MMC.usedBytes() >> 20));
  return true;
}

// Next unused session number, so a reboot never overwrites earlier photos.
static int nextSessionNumber() {
  if (!SD_MMC.exists(ROOT_DIR)) SD_MMC.mkdir(ROOT_DIR);
  File root = SD_MMC.open(ROOT_DIR);
  int highest = 0;
  for (File entry = root.openNextFile(); entry; entry = root.openNextFile()) {
    int n = 0;
    const char *name = strrchr(entry.name(), '/');
    name = name ? name + 1 : entry.name();
    if (entry.isDirectory() && sscanf(name, "session_%d", &n) == 1 && n > highest) highest = n;
  }
  return highest + 1;
}

static void writeSessionInfo() {
  sensor_t *s = esp_camera_sensor_get();
  char path[64];
  snprintf(path, sizeof(path), "%s/session.txt", sessionDir);
  File f = SD_MMC.open(path, FILE_WRITE);
  if (!f) fail("cannot write session.txt");
  f.printf("sensor_pid=0x%x\n", s->id.PID);
  f.printf("frame_size=%d\n", (int)FRAME_SIZE);
  f.printf("jpeg_quality=%d\n", JPEG_QUALITY);
  f.printf("interval_ms=%lu\n", (unsigned long)CAPTURE_INTERVAL_MS);
  f.printf("warmup_ms=%lu\n", (unsigned long)WARMUP_MS);
  f.printf("vflip=%d\nhmirror=%d\n", VFLIP, HMIRROR);
  f.printf("gainceiling=%dx\n", 2 << (int)GAIN_CEILING);
  f.printf("ae_level=%d\naec_dsp=%d\nwb_mode=%d\n", AE_LEVEL, AEC_DSP, WB_MODE);
  f.printf("awb=1\nawb_gain=1\nbpc=1\nwpc=1\nlenc=1\nraw_gma=1\n");
  f.close();

  snprintf(path, sizeof(path), "%s/captures.csv", sessionDir);
  f = SD_MMC.open(path, FILE_WRITE);
  if (!f) fail("cannot write captures.csv");
  f.println("file,ms_since_boot,width,height,bytes");
  f.close();
}

static bool captureOne() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("capture failed, retrying next interval");
    return false;
  }
  char name[24], path[72];
  snprintf(name, sizeof(name), "img_%05lu.jpg", (unsigned long)(imageCount + 1));
  snprintf(path, sizeof(path), "%s/%s", sessionDir, name);

  File f = SD_MMC.open(path, FILE_WRITE);
  size_t written = f ? f.write(fb->buf, fb->len) : 0;
  if (f) f.close();
  bool ok = written == fb->len;
  if (ok) {
    snprintf(path, sizeof(path), "%s/captures.csv", sessionDir);
    File log = SD_MMC.open(path, FILE_APPEND);
    if (log) {
      log.printf("%s,%lu,%u,%u,%u\n", name, (unsigned long)millis(), (unsigned)fb->width,
                 (unsigned)fb->height, (unsigned)fb->len);
      log.close();
    }
    imageCount++;
    Serial.printf("%s  %ux%u  %u bytes\n", path, (unsigned)fb->width, (unsigned)fb->height,
                  (unsigned)fb->len);
  } else {
    Serial.printf("write failed for %s (%u of %u bytes)\n", path, (unsigned)written,
                  (unsigned)fb->len);
  }
  esp_camera_fb_return(fb);
  return ok;
}

static void pollButton() {
  if (BUTTON_PIN < 0) return;
  static bool wasDown = false;
  bool down = digitalRead(BUTTON_PIN) == LOW;
  if (down && !wasDown) {
    delay(30);  // debounce
    if (digitalRead(BUTTON_PIN) == LOW) {
      paused = !paused;
      setLed(paused);
      Serial.println(paused ? "paused" : "resumed");
    }
  }
  wasDown = down;
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  if (LED_PIN >= 0) pinMode(LED_PIN, OUTPUT);
  if (BUTTON_PIN >= 0) pinMode(BUTTON_PIN, INPUT_PULLUP);
  setLed(false);

  if (!psramFound()) fail("no PSRAM: enable OPI PSRAM in the board settings");
  if (!initSdCard()) fail("no SD card (FAT32, inserted before power-on?)");
  if (!initCamera()) fail("camera init failed (ribbon cable seated?)");

  snprintf(sessionDir, sizeof(sessionDir), "%s/session_%03d", ROOT_DIR, nextSessionNumber());
  if (!SD_MMC.mkdir(sessionDir)) fail("cannot create session folder");
  writeSessionInfo();

  // Keep frames flowing so auto exposure and white balance converge before the first photo.
  for (uint32_t start = millis(); millis() - start < WARMUP_MS;) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb) esp_camera_fb_return(fb);
  }
  Serial.printf("capturing to %s every %lu ms; BOOT button pauses\n", sessionDir,
                (unsigned long)CAPTURE_INTERVAL_MS);
  lastCaptureMs = millis() - CAPTURE_INTERVAL_MS;  // first photo right away
}

void loop() {
  pollButton();
  if (paused || millis() - lastCaptureMs < CAPTURE_INTERVAL_MS) {
    delay(10);
    return;
  }
  lastCaptureMs = millis();

  if (SD_MMC.totalBytes() - SD_MMC.usedBytes() < MIN_FREE_BYTES) fail("SD card full");
  setLed(true);
  captureOne();
  setLed(false);

  if (MAX_IMAGES && imageCount >= MAX_IMAGES) {
    Serial.printf("done: %lu images in %s\n", (unsigned long)imageCount, sessionDir);
    esp_camera_deinit();
    SD_MMC.end();
    for (;;) {
      setLed(true);
      delay(1000);
      setLed(false);
      delay(1000);
    }
  }
}
