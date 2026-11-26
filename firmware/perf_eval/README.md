# perf_eval: detector latency, memory and PC parity on the ESP32-S3

This firmware measures how exported students (`model.espdl` from `tinyseek.train.export`)
run on the Freenove ESP32-S3 WROOM CAM (N16R8, OV2640). Its first job is to decide whether
192 px inputs (71 M MACs) are usable next to 128 px (31 M MACs). Each `.espdl` it finds on
the SD card goes through these steps:

1. **Load.** It records the load time and heap snapshots before and after the load.
2. **Benchmark.** It times each stage separately over warm-up runs followed by N timed runs:
   JPEG decode of an 800x600 frame, preprocessing (frame to the letterboxed, normalised,
   int8 input), inference, and post-processing (decode to boxes). Each stage reports
   mean, min, max and standard deviation in ms.
3. **Parity.** Staged camera photos go through the board's pipeline twice: once from the
   PC-decoded pixels (`.rgb`) and once from the board's own JPEG decode. The tensors and
   detections are written back to the card, and `tinyseek.deploy.compare` checks them
   against the PC.
4. **Camera (optional, `camera=1`).** Live capture with the frozen sensor settings, timing
   capture, JPEG decode and the whole pipeline, and reporting end-to-end FPS.

Nothing here has been compiled for the ESP32-S3 or run on a board yet. See
"Verification status" at the end.

## ESP-IDF project and Arduino sketch

This ESP-IDF project is the source of truth. The same evaluation also builds as an Arduino
sketch in `../perf_eval_arduino`, for flashing from the Arduino IDE. Its `src/` folder
holds generated copies of these sources (`python -m tinyseek.deploy.arduino_sketch`). Its
README lists the board settings and how its numbers differ from these.

Why keep an ESP-IDF build at all:

- It pins every version (IDF, ESP-DL 3.3.15, cache and PSRAM settings, compiler flags),
  and the benchmark numbers depend on them.
- arduino-esp32 3.3.12 ships esp-dl 3.3.11 precompiled, with the core's smaller caches
  and size optimisation fixed.

Build differences live behind `#ifdef ARDUINO` in `main/platform.cpp` and
`main/camera.cpp`. Both builds read the same card and write the same files. The `build`
and `jpeg_decoder` columns and `run.txt` say which build produced a result. Both reuse the
capture sketch's `board_pins.h` and copy its sensor settings.

## Versions (pinned)

| Part | Version | Where |
| --- | --- | --- |
| ESP-IDF | **v5.5.6** (latest stable patch, 2026-10-16) | `main/idf_component.yml` requires `>=5.5.6,<5.6` |
| ESP-DL | **3.3.15** (`espressif/esp-dl`) | `main/idf_component.yml`; it needs IDF >= 5.3 on the ESP32-S3 |
| esp32-camera | **2.1.9** | `main/idf_component.yml` |
| esp_new_jpeg | **1.0.2** (also a dependency of ESP-DL) | `main/idf_component.yml` |
| esp-ppq (export side) | 1.3.11 | the venv used by `tinyseek.train.export` |

Every CSV row and `run.txt` records the resolved IDF and ESP-DL versions. The ESP-DL
version is read from the component manager's downloaded manifest at build time, so
`unknown` means the managed component was not found.

## Board configuration

- `sdkconfig.defaults` sets 16 MB QIO flash, 8 MB octal PSRAM at 80 MHz, a 240 MHz CPU,
  ESP-DL's esp32s3 cache settings (32 KB I-cache, 64 KB D-cache, 64 B lines), `-O2`
  (performance), long file names on FAT, and the custom partition table.
- `partitions.csv` holds nvs, phy_init and an 8 MB factory app. Models are not stored in
  flash.
- Pins come from `../capture/board_pins.h` (`BOARD_FREENOVE_S3_CAM`). The camera is wired
  like the ESP32-S3-EYE. The microSD slot uses SD_MMC in 1-bit mode (CLK 39, CMD 38,
  D0 40). For another board, add a `BOARD_*` block there and change the `#define` in
  `main/platform.cpp` and `main/camera.cpp`.
- The camera uses the settings frozen for data collection on 2026-11-06: SVGA JPEG q10, no
  flip or mirror, AWB auto, AE level 0, aec2 off, 4x gain ceiling, lens/pixel corrections
  on, and 4 s of warm-up frames.
- Models load from the SD card with `dl::Model(path, fbs::MODEL_LOCATION_IN_SDCARD, ...)`,
  so swapping a model needs no reflash. ESP-DL reads the whole file into PSRAM. Input
  size, dtype and exponents are read from the loaded model, never assumed. A model counts
  as loaded when `get_fbs_model()` and its input and output tensors are non-null. That
  check works in both ESP-DL 3.3.15 and 3.3.11; `is_built()` exists only from 3.3.15.

## 1. Stage the SD card (PC)

```bash
cd ~/Projects/tinyseek
avm run "tinyseek deploy stage backpack px192+px128" -- env CUDA_VISIBLE_DEVICES= \
  ~/.venvs/tinyseek/bin/python -m tinyseek.deploy.stage \
    --model px192=runs/backpack/heldout_qwen21_turbo_far_v1_only_tone_px192/seed0 \
    --model px128=runs/backpack/heldout_qwen21_turbo_far_v1_only_tone/seed0 \
    --out runs/deploy/stage_backpack --cases 8 --runs 50 --warmup 5 --camera 0
```

- **Cost:** CPU only, about 2 minutes per model. Most of that is rebuilding esp-ppq's
  simulated int8 graph, which gives the expected outputs.
- **Model check:** the tool stops if the rebuilt graph is not the same model as the
  deployed `model.espdl`. It compares ops, weights and exponents through `model.info`.
- **Card size:** about 13 MB for 8 cases.
- **`--no-sim`** skips the rebuild. The parity check then covers only preprocessing,
  decode and float-model agreement.

Then copy the folder to the card root. The card must be FAT32 and inserted before
power-on.

```bash
cp -r runs/deploy/stage_backpack/sdcard/tinyseek_eval /media/$USER/<card>/
sync
```

### SD-card layout

```
/tinyseek_eval/bench.txt          settings (below)
/tinyseek_eval/bench.jpg          one 800x600 camera photo used for the timing loops
/tinyseek_eval/models/px192.espdl every *.espdl here is benchmarked, in name order
/tinyseek_eval/models/px192.txt   decode settings and provenance (score_threshold 0.05,
                                  max_detections 20, mean/std, exponents, sha256, run dir)
/tinyseek_eval/parity/cases.txt   case ids, one per line
/tinyseek_eval/parity/c000.jpg    camera JPEG (from the run's held-out eval set)
/tinyseek_eval/parity/c000.rgb    the same photo decoded by Pillow, 800x600x3 RGB888
/tinyseek_eval/results/run_NNN/   written by the board, one folder per boot
```

`bench.txt` keys (key=value; missing keys use the defaults):

| Key | Default | Meaning |
| --- | --- | --- |
| `warmup` | 5 | untimed runs before timing starts (also for the camera loop) |
| `runs` | 50 | timed runs per stage |
| `parity` | 1 | run the parity cases |
| `camera` | 0 | 1: also benchmark live capture (needs the camera module connected) |
| `camera_frames` | 50 | timed frames in the camera loop |
| `internal_kb` | 0 | ESP-DL `max_internal_size`. 0 keeps every activation in PSRAM; for example 100 asks ESP-DL to put the first 100 KB of the activation arena in internal RAM. ESP-DL allocates that as one block and does not clamp it: if no free internal block is that large the model fails to load ("Failed to alloc ... internal RAM"). On the Freenove N16R8 under arduino-esp32 3.3.12 the largest free block at that point was 128 KB, so 200 failed (2026-11-16); use 100. Benchmark both. |
| `profile` | 0 | 1: also print ESP-DL's per-layer latency and memory table (`Model::profile()`) on serial |

## 2. Build and flash

Install ESP-IDF v5.5.6 once:

```bash
mkdir -p ~/esp && cd ~/esp
git clone -b v5.5.6 --recursive https://github.com/espressif/esp-idf.git esp-idf-v5.5.6
./esp-idf-v5.5.6/install.sh esp32s3
```

Build, flash and watch the log:

```bash
. ~/esp/esp-idf-v5.5.6/export.sh
cd ~/Projects/tinyseek/firmware/perf_eval
idf.py set-target esp32s3          # first time only: applies sdkconfig.defaults
idf.py build                       # downloads esp-dl, esp32-camera, esp_new_jpeg
PORT=/dev/ttyACM0                  # or /dev/ttyUSB0 on the USB-UART port
idf.py -p $PORT flash monitor      # Ctrl+] leaves the monitor
```

- `sdkconfig.defaults` only takes effect when `sdkconfig` is first created. After editing
  it, run `idf.py fullclean` and delete `sdkconfig`.
- Power the board through one USB-C port only.

## 3. Run

Insert the staged card and reset the board. Each boot does a full pass over all models,
which takes a few minutes, and creates a new `results/run_NNN`. The log prints each
timing row and memory snapshot. The pass ends with `done: /sdcard/tinyseek_eval/results/run_NNN`,
and `run.txt` gains `complete=1`; a folder without that line is from an interrupted pass.

What one boot does:

1. Mount the card, read `bench.txt`, create `results/run_NNN`, and record `run.txt`
   (board, build, `sdk_settings` (cache sizes, PSRAM speed and SDK optimisation from
   sdkconfig.h), IDF/ESP-DL/camera versions, JPEG decoder, CPU MHz, settings, PSRAM and
   internal totals).
2. If `camera=1`, start the camera and drop 4 s of frames.
3. For each model:
   1. Load it and build the preprocessing plan.
   2. Run the timing loop on `bench.jpg`.
   3. Run the parity cases.
   4. If `camera=1`, run the camera loop.
   5. Unload it.

Re-running with `internal_kb=100` (edit `bench.txt` on the card, then reset) measures
whether internal RAM for activations matters.

## 4. Compare and read the results (PC)

```bash
mkdir -p runs/deploy/stage_backpack/results
cp -r /media/$USER/<card>/tinyseek_eval/results/run_001 runs/deploy/stage_backpack/results/
~/.venvs/tinyseek/bin/python -m tinyseek.deploy.compare \
    --stage runs/deploy/stage_backpack --results runs/deploy/stage_backpack/results/run_001 \
    --json runs/deploy/stage_backpack/results/run_001/compare.json
```

### `bench.csv`

There is one row per model and stage. Each row carries these columns:

| Column(s) | Meaning |
| --- | --- |
| `run`, `board`, `build`, `idf_version`, `espdl_version`, `jpeg_decoder`, `cpu_mhz` | which run and setup produced the row. `build` is `esp-idf` or `arduino-esp32 <core version>`, and `jpeg_decoder` names the decoder used |
| `model`, `model_bytes`, `input_size` | the model |
| `input_exponent`, `output_exponent` | read from the loaded model |
| `internal_kb` | the ESP-DL internal-RAM limit used |
| `stage` | which measurement this row is (table below) |
| `runs`, `warmup` | timed and untimed run counts |
| `mean_ms`, `min_ms`, `max_ms`, `std_ms` | latency statistics; std is the sample standard deviation |
| `fps` | end-to-end rate, for `end_to_end` only |

| `stage` | What is timed |
| --- | --- |
| `load` | `dl::Model` construction: SD read, parse, parameter copy, memory planning (1 run) |
| `preprocess_plan` | one-time resample coefficients and quantisation tables for 800x600 to S (1 run) |
| `jpeg_decode` | `bench.jpg` 800x600 to RGB888, including the header size check (esp_new_jpeg here; see `jpeg_decoder`) |
| `preprocess` | RGB888 to int8 NHWC letterboxed input (Pillow-exact resample, normalise, quantise) |
| `inference` | `Model::run()` only; the input copy into the model is outside the timer |
| `postprocess` | decode to boxes (peaks, top-k, threshold, letterbox inverse) |
| `camera_capture` | `esp_camera_fb_get()` with 2 frame buffers and GRAB_LATEST, so it includes waiting for a frame |
| `camera_jpeg` | decoding the live frame |
| `end_to_end` | capture, decode, preprocess, inference, post-process and one `vTaskDelay(1)`; `fps` = frames / wall time |

Per-frame cost without the camera is `jpeg_decode + preprocess + inference + postprocess`.

### `memory.csv`

These are heap snapshots: internal and PSRAM free, minimum-ever-free, and largest free
block, in bytes. They are taken at `boot`, then for each model at `before_load`,
`after_load`, `after_inference`, `after_camera` and `after_unload`.

- `before_load - after_load` is what the model holds: its parameters, the activation arena
  and the file buffer.
- The minimum-free columns show the peak during inference.
- `Model::profile()` (`profile=1`) breaks the totals down further.

### Parity summary

`compare` prints, per model:

| Line | Meaning | Expected |
| --- | --- | --- |
| `preprocess_exact` | cases whose board input tensor (from `.rgb`) equals the PC's byte for byte | all cases |
| `model_output_exact`, `model_output_max_abs_lsb`, `model_output_within_1_lsb` | board raw output against esp-ppq's simulated int8 graph for the same input, in int8 steps of 2^output_exponent | small differences are possible: the simulator computes in float and ESP-DL in integers |
| `decode_identical` | board detections against the PC decode of the board's own output | all |
| `rgb_dets_vs_sim_f1`, `jpg_dets_vs_sim_f1` | detection agreement with the simulated graph (IoU >= 0.5 one-to-one matching) | |
| `jpeg_input_mean_abs_lsb`, `jpeg_input_max_abs_lsb` | how far the board's JPEG decoder moves the input from Pillow's | |

The per-case JSON also has `dets_vs_float`, which is agreement with the float checkpoint.
That is the quantization effect, not a deployment bug.

## Code layout

```
main/perf_eval.{hpp,cpp}     perf_eval_run(): benchmark, parity and camera loops, CSV output;
                             app_main() calls it (the Arduino sketch calls it from setup())
main/platform.{hpp,cpp}      platform shim: SD card, files, heap snapshots, timing statistics,
                             logging, build info (#ifdef ARDUINO for the sketch)
main/camera.{hpp,cpp}        OV2640 with the frozen settings; JPEG decode (esp_new_jpeg here,
                             esp32-camera's fmt2rgb888 in the sketch)
components/tinyseek_core/    portable C++ (no ESP headers): Pillow-exact letterbox and
                             quantisation, plus a fast 2-tap bilinear mode the detector can
                             select (preprocess.*), CenterNet decode (decode.*), JPEG header
                             check (jpeg_info.*)
host_test/core_cli.cpp       host driver for tinyseek_core, used by tests/test_deploy_core.py
host_test/tjpgd/             ChaN's TJpgDec R0.01b (verbatim, its licence header kept) and a
                             driver that runs it as esp_jpeg runs the ESP32-S3 ROM copy;
                             tinyseek.deploy.tjpgd builds it so the PC decodes JPEGs exactly
                             like the board (tests/test_deploy_tjpgd.py)
```

The same shared sources (platform, camera, tinyseek_core) also feed the live detector,
`../detector`. `camera.*` has a second decoder entry, `decode_jpeg_scaled()`. It runs TJpgDec
through esp_jpeg at 1/1 to 1/8 size, or esp_new_jpeg, and the detector uses it. perf_eval
itself still calls `decode_jpeg()` only, so its measurements and result files are unchanged.

The portable core is the main correctness check that can run off the board:

```bash
~/.venvs/tinyseek/bin/python -m pytest tests/test_deploy_core.py
```

This builds it with g++ and checks it against the PC code (`tinyseek.deploy.reference`,
which calls the evaluation code directly).

## Verification status and assumptions

**Verified on the PC (2026-11-15):**

- `tinyseek_core` preprocessing is byte-identical to the PC path (Pillow BILINEAR letterbox,
  ImageNet normalisation, esp-ppq quantisation). This holds for real camera photos and odd
  sizes, at 128, 160 and 192 px.
- Decode matches `tinyseek.train.targets.decode` plus `Letterbox.inverse`, including peak
  ties and top-k.
- esp-ppq's executor gives the same output for our int8 input as for the float input.
  This confirms ROUND_HALF_UP and saturation on the input.
- The JPEG header check (`jpeg_info`) reports the same size as Pillow, flags progressive
  files, and rejects non-JPEG and truncated input (2026-11-16).

**Not verified (needs a real build and the board):**

- Compiling against the real ESP-IDF/ESP-DL headers. Only a syntax check against stub
  headers was done.
- That ESP-DL 3.3.15 loads esp-ppq 1.3.11 files, including the int16 depthwise layers.
- That the `get_fbs_model()` check catches a failed load.
- Writing the input through `TensorBase::data`.
- `heap_caps_aligned_alloc` buffers working with esp_new_jpeg.
- The RGB byte order of esp_new_jpeg's output, documented as R,G,B.
- SD_MMC pull-ups on this board.
