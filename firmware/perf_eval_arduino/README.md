# perf_eval_arduino: the on-device evaluation as an Arduino sketch

This is the same evaluation as the ESP-IDF project in `../perf_eval`, built and flashed
from the Arduino IDE:

- per model: load time, then JPEG decode, preprocessing, inference and post-processing
  times (mean, min, max, standard deviation);
- heap snapshots;
- PC parity cases;
- an optional camera stage with end-to-end FPS.

It reads the same SD-card folder and writes the same result files, so
`tinyseek.deploy.stage` and `tinyseek.deploy.compare` work unchanged. `../perf_eval/README.md`
describes the measurements, `bench.txt` and the CSV columns. This file covers only what
is different for Arduino.

**Not yet built or run:** this sketch has not been compiled with the Arduino IDE or run on
a board. See "What is not verified" at the end.

## What is in this folder

| File | Role |
| --- | --- |
| `perf_eval_arduino.ino` | `setup()` runs one full evaluation pass. Reset the board for another pass. |
| `src/arduino_config.h` | Hand-written Arduino-only switches: where esp-dl comes from, which JPEG decoder to use, and the RGB order |
| everything else in `src/` | **Generated** copies of `../perf_eval` sources and `../capture/board_pins.h` |

Do not edit the generated files; each starts with a "GENERATED" banner. Edit the source in
`../perf_eval` (Arduino-only code goes behind `#ifdef ARDUINO` there), then regenerate:

```bash
~/.venvs/tinyseek/bin/python -m tinyseek.deploy.arduino_sketch          # rewrite the copies
~/.venvs/tinyseek/bin/python -m tinyseek.deploy.arduino_sketch --check  # 1 if any copy is stale
```

`tests/test_deploy_arduino_sketch.py` fails when a copy has drifted from its source. The
same command also regenerates the live detector sketch (`../detector`), which shares these
sources and this folder's `src/arduino_config.h`.

## Install (once)

1. **Arduino IDE 2.**
2. **Board package:** open Boards Manager and install **esp32 by Espressif Systems, version
   3.3.12** or later in the 3.3 line. The board manager URL is
   `https://espressif.github.io/arduino-esp32/package_esp32_index.json`, under
   File > Preferences.
   - Core 3.3.12 ships Espressif's esp-dl 3.3.11, precompiled, so **no extra library is
     needed**.
   - The sketch stops with a clear `#error` on an older core.
3. **Library:** none for the default setup.
   - On a core older than 3.3.12 you can instead install the community library **espdlx
     1.2.0** from the Library Manager and set `TINYSEEK_USE_ESPDLX 1` in
     `src/arduino_config.h`. This is untested; see "About espdlx" below.
   - Do not use espdlx on 3.3.12 or later: the core already links esp-dl, and a second
     copy of the same symbols may clash.

## Board settings (Tools menu)

| Setting | Value | Why |
| --- | --- | --- |
| Board | **ESP32S3 Dev Module** | generic S3 entry. The Freenove board is wired like the ESP32-S3-EYE (pins in `src/board_pins.h`) |
| Flash Size | **16MB (128Mb)** | N16R8. The default is 4 MB |
| PSRAM | **OPI PSRAM** | 8 MB octal PSRAM. Models, activations and frame buffers live there |
| Flash Mode | **QIO 80MHz** | |
| Partition Scheme | **16M Flash (3MB APP/9.9MB FATFS)** | the largest app slot on a 16 MB preset. Models are loaded from the SD card, not flash |
| CPU Frequency | **240MHz (WiFi)** | the CSVs record the actual value |
| Core Debug Level | **None** | the sketch prints through `Serial` regardless; extra core logging could only add noise to timings |
| Arduino Runs On / Events Run On | **Core 1** / **Core 1** | the default; the benchmark runs on the loop task |
| USB CDC On Boot | **Disabled** when connected through the board's USB-UART port. **Enabled** only when using the native USB port | the same rule as for `firmware/capture` |
| USB Mode | Hardware CDC and JTAG | the default |
| Upload Mode | UART0 / Hardware CDC | the default |
| Erase All Flash Before Sketch Upload | Disabled | |

The same settings as an arduino-cli FQBN, if you ever build from a shell:
`esp32:esp32:esp32s3:PSRAM=opi,FlashSize=16M,FlashMode=qio,PartitionScheme=app3M_fat9M_16MB,CPUFreq=240`

## 1. Stage the SD card (PC, unchanged)

```bash
cd ~/Projects/tinyseek
avm run "tinyseek deploy stage backpack px192+px128" -- env CUDA_VISIBLE_DEVICES= \
  ~/.venvs/tinyseek/bin/python -m tinyseek.deploy.stage \
    --model px192=runs/backpack/heldout_qwen21_turbo_far_v1_only_tone_px192/seed0 \
    --model px128=runs/backpack/heldout_qwen21_turbo_far_v1_only_tone/seed0 \
    --out runs/deploy/stage_backpack --cases 8 --runs 50 --warmup 5 --camera 0
```

Then copy the `tinyseek_eval` folder to the root of the microSD card, so the card holds
`/tinyseek_eval/bench.txt`, `/tinyseek_eval/models/...` and so on.

- **Linux:** `cp -r runs/deploy/stage_backpack/sdcard/tinyseek_eval /media/$USER/<card>/ && sync`
- **Windows:** copy `runs\deploy\stage_backpack\sdcard\tinyseek_eval` onto the card in
  Explorer, then eject the card.

The card must be FAT32 and inserted before power-on. One card can hold the folder for
both builds. Each boot adds a new `results/run_NNN`, and `run.txt` says which build wrote it.

## 2. Build and upload

1. Open `firmware/perf_eval_arduino/perf_eval_arduino.ino` in the Arduino IDE. The folder
   name must stay `perf_eval_arduino`.
2. Select the board, settings and port above.
3. Click **Upload**.
4. Open the Serial Monitor at **115200** baud and press the board's reset button.

## 3. What success looks like on serial

Numbers below are placeholders; the lines are what the sketch prints.

```
tinyseek perf_eval starting
SD card: 30436 MB
results in /sdcard/tinyseek_eval/results/run_001
build arduino-esp32 3.3.12 | IDF v5.5.5 | esp-dl bundled with core 3.3.12 | JPEG esp32-camera fmt2rgb888 (ROM TJpgDec) | 240 MHz | dcache=32KB/32B icache=16KB spiram=80MHz sdk_opt=size
mem -      boot             internal free  ...  | psram free ...
mem px128  before_load      internal free  ...
mem px128  after_load       internal free  ...
px128: input [1,128,128,3] exp -6, output [1,16,16,5] exp -4, 650768 bytes
px128  load            n=1    mean  ... ms  min ...  max ...  std ...
px128  preprocess_plan n=1    mean  ... ms ...
px128  jpeg_decode     n=50   mean  ... ms ...
px128  preprocess      n=50   mean  ... ms ...
px128  inference       n=50   mean  ... ms ...
px128  postprocess     n=50   mean  ... ms ...
px128: bench.jpg gives N detections
px128 c000 rgb: N detections
px128 c000 jpg: N detections
...                                   (the same for px192)
done: /sdcard/tinyseek_eval/results/run_001
```

**A complete pass** ends with `done: ...`, and `run.txt` gets `complete=1`.

**On failure:** the sketch prints `FATAL: <reason>` and stops. The reasons are no card,
no `/tinyseek_eval` folder, camera init failure, or out of PSRAM. A model that does not
load is reported as `ERROR: ... model did not load` and skipped.

## 4. Bring the results back and compare (PC, unchanged)

```bash
mkdir -p runs/deploy/stage_backpack/results
cp -r /media/$USER/<card>/tinyseek_eval/results/run_001 runs/deploy/stage_backpack/results/
~/.venvs/tinyseek/bin/python -m tinyseek.deploy.compare --stage runs/deploy/stage_backpack \
    --results runs/deploy/stage_backpack/results/run_001 \
    --json runs/deploy/stage_backpack/results/run_001/compare.json
```

The first output line (`board: ...`) shows `build=arduino-esp32 ...`, `sdk_settings=...`
and `jpeg_decoder=...`. Every `bench.csv` and `memory.csv` row has `build` and
`jpeg_decoder` columns.

## What differs from the ESP-IDF build, and which numbers count

| | ESP-IDF build (`../perf_eval`) | This sketch on core 3.3.12 |
| --- | --- | --- |
| ESP-IDF | v5.5.6 | v5.5.5, inside the core |
| esp-dl | 3.3.15 (component) | 3.3.11, precompiled by Espressif with the core |
| D-cache / I-cache | 64 KB, 64 B lines / 32 KB | 32 KB, 32 B lines / 16 KB (fixed by the core) |
| SDK optimisation | `-O2` (performance) | size (`-Os`); this sketch's own C++ is also `-Os` |
| JPEG decoder | esp_new_jpeg 1.0.2 | esp32-camera `fmt2rgb888` (ROM TJpgDec). `TINYSEEK_ARDUINO_USE_ESP_NEW_JPEG 1` switches to esp_new_jpeg, untested |
| SD card | IDF SDMMC driver, 20 MHz | Arduino `SD_MMC`, also set to 20 MHz |
| Model-loaded check | `get_fbs_model()` and the I/O tensors (`is_built()` exists only from esp-dl 3.3.15) | same |

`run.txt` records the cache sizes, PSRAM speed and SDK optimisation level (`sdk_settings`)
from `sdkconfig.h`, so each result says what it ran with.

**Latency will differ between the two builds**, mostly in inference (cache sizes, esp-dl
version) and in jpeg_decode (different decoders).

- **If the final detector firmware will also be an Arduino sketch,** the numbers from this
  sketch on the same core version are the ones that count. That includes `inference`,
  `preprocess`, `postprocess`, `jpeg_decode`, and `end_to_end` with the camera on. The
  ESP-IDF numbers then only show the headroom you would gain by moving to ESP-IDF.
- **If the detector ships on ESP-IDF,** use the ESP-IDF numbers.
- **Parity is unaffected by the build.** `preprocess_exact` and `decode_identical` come
  from the same C++ in both builds. The `jpg` parity path and `jpeg_input_*` describe
  whichever decoder ran.

## About espdlx

**What it is.** `espdlx` (github.com/salernosimone/arduino-espdlx, MIT, Library Manager
name `espdlx`, version 1.2.0) is a one-person port of esp-dl 3.3.11. It has 4 commits
since 2026-09-11, no issues and no CI, and states no supported core version.

**What is prebuilt.** Its only binary, `src/esp32s3/libfbs_model.a`, is Espressif's own
file (byte-identical to the one in esp-dl). The rest of esp-dl is compiled from source in
your IDE with the core's `-Os` flags. esp-dl's own build uses `-O3 -ffast-math`, so speed
may differ.

**When to use it.** Core 3.3.12 made it unnecessary for this sketch. It is kept only as a
fallback for older cores (`TINYSEEK_USE_ESPDLX 1`). That path is untested. Older cores may
also bundle an esp32-camera whose `fmt2rgb888` returns B,G,R; see the next section.

## What is not verified

Nothing here has been compiled with the real core or run on a board. Checked on the PC:

- The generated copies match their sources (pytest).
- Every quoted include resolves inside the sketch folder (pytest).
- `src/*.cpp` and the `.ino` pass a `g++ -fsyntax-only -Werror` check against stub
  headers written from the core 3.3.12 and esp-dl 3.3.11 sources. The non-default switches
  were checked the same way, except `TINYSEEK_USE_ESPDLX`. This checks the C++, not that
  the real headers match.

Open until the first build or run:

- **Including esp-dl.** That `#include "dl_model_base.hpp"` resolves and links on core
  3.3.12 without a library. The core's include and link flags list esp-dl, but no sketch
  has been built.
- **Model format.** That esp-dl 3.3.11 loads these esp-ppq 1.3.11 exports, including the
  int16 layers.
- **Load-failure check.** That the `get_fbs_model()` check catches a failed load.
- **JPEG byte order.** That `fmt2rgb888` returns R,G,B for JPEG on this core. If the jpg
  parity path shows a large `jpeg_input_mean_abs_lsb`, or detections collapse only on the
  jpg path, set `TINYSEEK_FMT2RGB888_IS_BGR 1`.
- **App size.** That the app fits the 3 MB slot of the chosen partition scheme. If the
  upload reports it is too large, none of the 16 MB presets has a bigger app slot.
- **Stack.** That 16 KB of loop-task stack is enough.
