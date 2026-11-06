// Interval photo capture to microSD, for building the ESP32-camera image set
// (domain LoRA training and the on-device test set).
//
// Each boot, and each 3 s hold of button 1, starts a new folder /tinyseek/session_NNN/
// on the card holding
//   img_00001.jpg, img_00002.jpg, ...   one JPEG every CAPTURE_INTERVAL_MS
//   captures.csv                         file, ms since boot, size, bytes, trigger
//   session.txt                          the capture settings used
//
// Button 1 (BOOT): tap to pause / resume the interval, hold 3 s for a new session
// (three quick blinks confirm). Button 2: take a photo now, paused or
// not, so pausing gives a manual-only mode. LEDs (onboard + rig): one short blink per
// photo, steady on while paused, fast blinking forever on an error (no card, camera
// failure, card full).
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
static const uint32_t MAX_IMAGES = 0;                 // per session; 0 = no limit
static const uint32_t NEW_SESSION_HOLD_MS = 3000;     // hold button 1 this long: new session
static const bool VFLIP = false;                      // camera rig v1: module upright, flex
static const bool HMIRROR = false;                    // at the bottom -> no flip needed

// ---- Sensor tuning (OV2640) --------------------------------------------------
// Close to the library defaults: on a working module they expose well. (Raising the AE
// bias, DSP exposure and a 16x gain ceiling only compensated for a faulty module; on a
// good one they washed images out and amplified purple chroma noise.) 4x gain leaves
// some headroom for dim rooms. The detector firmware must use these same values so
// training and deployment images match; every value is written to session.txt.
static const gainceiling_t GAIN_CEILING = GAINCEILING_4X;   // 2X..128X; higher = brighter, noisier
static const int AE_LEVEL = 0;                              // exposure bias, -2..2
static const bool AEC_DSP = false;                          // "aec2": DSP exposure, brighter, hazier
static const int WB_MODE = 0;  // 0 auto, 1 sunny, 2 cloudy, 3 office, 4 home (fixed presets)
static const uint64_t MIN_FREE_BYTES = 4ULL * 1024 * 1024;
static const char *ROOT_DIR = "/tinyseek";
// -----------------------------------------------------------------------------

static char sessionDir[40];
static uint32_t imageCount = 0;
static uint32_t lastCaptureMs = 0;
static bool paused = false;
static bool button1Down = false;  // interval shots wait while button 1 is held

// The onboard LED and the rig's external LED always show the same pattern.
static void setLed(bool on) {
  if (LED_PIN >= 0) digitalWrite(LED_PIN, on ? LED_ON : !LED_ON);
  if (EXT_LED_PIN >= 0) digitalWrite(EXT_LED_PIN, on ? LED_ON : !LED_ON);
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
  f.printf("started_ms_since_boot=%lu\n", (unsigned long)millis());
  f.close();

  snprintf(path, sizeof(path), "%s/captures.csv", sessionDir);
  f = SD_MMC.open(path, FILE_WRITE);
  if (!f) fail("cannot write captures.csv");
  f.println("file,ms_since_boot,width,height,bytes,trigger");
  f.close();
}

// Opens the next session folder; photo numbering restarts at img_00001.
static void startSession() {
  snprintf(sessionDir, sizeof(sessionDir), "%s/session_%03d", ROOT_DIR, nextSessionNumber());
  if (!SD_MMC.mkdir(sessionDir)) fail("cannot create session folder");
  writeSessionInfo();
  imageCount = 0;
  Serial.printf("session %s\n", sessionDir);
}

static bool captureOne(const char *trigger) {
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
      log.printf("%s,%lu,%u,%u,%u,%s\n", name, (unsigned long)millis(), (unsigned)fb->width,
                 (unsigned)fb->height, (unsigned)fb->len, trigger);
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

// True once per press of an active-low button (debounced).
static bool pressed(int pin, bool &wasDown) {
  if (pin < 0) return false;
  bool down = digitalRead(pin) == LOW;
  bool edge = false;
  if (down && !wasDown) {
    delay(30);  // debounce
    edge = digitalRead(pin) == LOW;
  }
  wasDown = down;
  return edge;
}

// Button 1: a short press (released before NEW_SESSION_HOLD_MS) toggles pause; holding it
// that long starts a new session folder, confirmed by three quick blinks. Both edges are
// debounced, and pause toggles on release so a long hold never pauses by accident.
static void handleButton1() {
  if (BUTTON_PIN < 0) return;
  bool &down = button1Down;
  static bool held = false;
  static uint32_t changedAt = 0;
  if (millis() - changedAt < 50) return;  // debounce
  bool now = digitalRead(BUTTON_PIN) == LOW;
  if (now && !down) {
    down = true;
    held = false;
    changedAt = millis();
  } else if (now && down && !held && millis() - changedAt >= NEW_SESSION_HOLD_MS) {
    held = true;
    startSession();
    for (int i = 0; i < 3; i++) {
      setLed(true);
      delay(120);
      setLed(false);
      delay(120);
    }
    setLed(paused);
    lastCaptureMs = millis();  // first interval photo one interval later: time to aim
  } else if (!now && down) {
    down = false;
    changedAt = millis();
    if (!held) {
      paused = !paused;
      setLed(paused);
      Serial.println(paused ? "paused" : "resumed");
    }
  }
}

// Takes one photo with the LED blink, then restores the paused indicator.
static void shoot(const char *trigger) {
  if (SD_MMC.totalBytes() - SD_MMC.usedBytes() < MIN_FREE_BYTES) fail("SD card full");
  setLed(true);
  captureOne(trigger);
  if (paused) delay(150);  // make the blink visible against the steady paused light
  setLed(paused);
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  if (LED_PIN >= 0) pinMode(LED_PIN, OUTPUT);
  if (EXT_LED_PIN >= 0) pinMode(EXT_LED_PIN, OUTPUT);
  if (BUTTON_PIN >= 0) pinMode(BUTTON_PIN, INPUT_PULLUP);
  if (BUTTON2_PIN >= 0) pinMode(BUTTON2_PIN, INPUT_PULLUP);
  setLed(false);

  if (!psramFound()) fail("no PSRAM: enable OPI PSRAM in the board settings");
  if (!initSdCard()) fail("no SD card (FAT32, inserted before power-on?)");
  if (!initCamera()) fail("camera init failed (ribbon cable seated?)");

  startSession();

  // Keep frames flowing so auto exposure and white balance converge before the first photo.
  for (uint32_t start = millis(); millis() - start < WARMUP_MS;) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb) esp_camera_fb_return(fb);
  }
  Serial.printf("capturing every %lu ms; button 1: tap pauses, hold 3 s = new session; "
                "button 2 shoots\n", (unsigned long)CAPTURE_INTERVAL_MS);
  lastCaptureMs = millis() - CAPTURE_INTERVAL_MS;  // first photo right away
}

void loop() {
  static bool shootWasDown = false;
  handleButton1();
  if (pressed(BUTTON2_PIN, shootWasDown)) {
    shoot("button");
  } else if (!paused && !button1Down && millis() - lastCaptureMs >= CAPTURE_INTERVAL_MS) {
    lastCaptureMs = millis();
    shoot("interval");
  } else {
    delay(10);
    return;
  }

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
