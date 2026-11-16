// On-device evaluation of tinyseek detectors: latency, memory and PC parity, as an Arduino
// sketch. Same measurements, SD-card layout and result files as the ESP-IDF project in
// firmware/perf_eval; all logic lives in src/, generated from that project's sources by
//   python -m tinyseek.deploy.arduino_sketch
// Settings for this build (esp-dl source, JPEG decoder) are in src/arduino_config.h.
// Board settings and steps: README.md next to this file.

#include "src/perf_eval.hpp"

// ESP-DL inference and the benchmark run on the loop task. Give it the same 16 KB the
// ESP-IDF build gives its main task, instead of the core's default 8 KB.
SET_LOOP_TASK_STACK_SIZE(16 * 1024);

void setup() {
  Serial.begin(115200);
  delay(1000);  // let the serial monitor attach
  Serial.println("tinyseek perf_eval starting");
  perf_eval_run();  // one full pass over every model on the card, then returns
}

void loop() {
  delay(1000);  // nothing to repeat: reset the board for another pass
}
