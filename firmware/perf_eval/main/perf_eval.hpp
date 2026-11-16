// Entry point of the on-device evaluation, shared by the ESP-IDF app (app_main) and the
// Arduino sketch (setup()). Runs one full pass over every model on the SD card and returns.
#pragma once

void perf_eval_run();
