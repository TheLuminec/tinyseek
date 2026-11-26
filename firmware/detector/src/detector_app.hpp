// The live detector: camera (or replay files) -> JPEG decode -> preprocess -> ESP-DL ->
// boxes, with per-prompt models from the SD card. detector.ino calls these.
#pragma once

void detector_setup();
void detector_loop();
