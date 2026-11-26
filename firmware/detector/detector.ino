// tinyseek live detector for the Freenove ESP32-S3 WROOM CAM (N16R8, OV2640).
//
// Detects the prompted object in the camera frames with a per-prompt ESP-DL model from
// the SD card (/sdcard/tinyseek/models/<prompt>.espdl), lights the LED while it is seen,
// prints one JSON line per frame on Serial and, if Wi-Fi is configured on the card, serves
// a page that shows the frame, its boxes and a box to type a new prompt. A replay mode
// runs the same pipeline on the JPEGs in /sdcard/tinyseek/replay/ and writes the boxes to
// the card, to measure the deployed accuracy on the PC.
//
// The logic is in src/ (detector_app.cpp); src/ also holds generated copies of the shared
// firmware code (python -m tinyseek.deploy.arduino_sketch). Settings, SD layout and the
// serial format: README.md next to this file.

#include "src/detector_app.hpp"

// ESP-DL inference runs on the loop task: give it 16 KB, as perf_eval does, instead of
// the core's default 8 KB.
SET_LOOP_TASK_STACK_SIZE(16 * 1024);

void setup() {
  // A deep buffer so printing a frame line does not stall the loop at 115200 baud
  // (HardwareSerial and HWCDC both take it before begin()).
  Serial.setTxBufferSize(8192);
  Serial.begin(115200);
  delay(1000);  // let the serial monitor attach
  detector_setup();
}

void loop() { detector_loop(); }
