# detector: live prompt detection on the ESP32-S3

This Arduino sketch runs the detector continuously on the Freenove ESP32-S3 WROOM CAM
(N16R8, OV2640, 160-degree lens, microSD):

```
camera frame (SVGA JPEG, frozen sensor settings)
  -> JPEG decode (ROM TJpgDec, optionally at 1/2, 1/4 or 1/8 size)
  -> letterbox + normalise + quantise into the ESP-DL input tensor
  -> ESP-DL inference (the model for the active prompt: int8, or int16 I/O for a w16a16 export)
  -> CenterNet decode to boxes in 800x600 frame pixels
  -> LED, one JSON line on Serial, the web page (optional), the results file (replay)
```

Each prompt has its own model on the SD card. You can switch prompts at run time from the
serial monitor or the web page, so there is nothing to reflash.

**Status:** not yet built or run on a board. Every API was checked against the sources
that arduino-esp32 3.3.12 ships (see "What is verified"), and the C++ passes a syntax
check against headers taken from those sources. The defaults use the same code path that
perf_eval_arduino has already run on the board.

## Contents

| File | Role |
| --- | --- |
| `detector.ino` | `setup()` and `loop()`; they call `src/detector_app.cpp` |
| `src/detector_app.cpp` | the frame loop, models, replay, prompt switching, serial commands |
| `src/detector_logic.*` | portable logic: `config.txt` keys, prompt names, JSON lines (host-tested) |
| `src/web.*`, `src/web_page.h` | Wi-Fi and the web page (esp_http_server) |
| everything else in `src/` | **generated** copies of `../perf_eval` sources (platform, camera, tinyseek_core), `../capture/board_pins.h` and `../perf_eval_arduino/src/arduino_config.h` |
| `host_test/logic_cli.cpp` | host driver for `tests/test_deploy_detector.py` (the IDE does not compile it) |

Do not edit the generated files; each one starts with a "GENERATED" banner. To change
them, edit the source and regenerate:

```bash
~/.venvs/tinyseek/bin/python -m tinyseek.deploy.arduino_sketch          # both sketches
~/.venvs/tinyseek/bin/python -m tinyseek.deploy.arduino_sketch --check
```

## 1. Arduino IDE settings

Use the same settings as `../perf_eval_arduino` (its README explains each one).

- **Core:** esp32 by Espressif **3.3.12**. No extra library is needed: ESP-DL 3.3.11,
  esp32-camera and esp_jpeg come with the core.
- **Board:** ESP32S3 Dev Module.
- **Tools settings:** Flash Size **16MB**, PSRAM **OPI PSRAM**, Partition Scheme
  **16M Flash (3MB APP/9.9MB FATFS)**, CPU **240MHz**, Core Debug Level **None**.
- **USB CDC On Boot:** **Disabled** on the USB-UART port.
- **Sketch folder:** open `firmware/detector/detector.ino`. The folder must keep the name
  `detector`.
- **Serial Monitor:** **115200** baud, line ending **Newline** (for typed commands).

## 2. Stage the SD card (PC)

```bash
cd ~/Projects/tinyseek
~/.venvs/tinyseek/bin/python -m tinyseek.deploy.detector_card \
  --model runs/backpack/heldout_qwen21_turbo_far_v1_only_tone_px192/seed0 \
  --threshold backpack=0.3 \
  --replay-ann data/esp32cam/reviewed_labels_s013_014.json --replay-images data/esp32cam/raw \
  --out runs/deploy/detector_card
```

- **Several prompts:** repeat `--model`. The prompt name comes from the run's
  `model_meta.json`; write `name=run_dir` to choose another name.
- **Active prompt:** the first model, unless you pass `--prompt`.
- **Detect threshold:** the score that lights the LED and highlights a box. It is taken from
  `--threshold` (one value for all, or `prompt=value`), else `detect_threshold` in
  `model_meta.json` or in the run config's `eval:` section, else 0.3. The detections the
  board reports and writes keep the PC evaluation's `score_threshold` (0.05) and
  `max_detections` (20), so replay AP compares directly with the PC numbers.
- **Other settings:** `--set decode_scale=1 --set resize=bilinear` and so on (keys below).
  `--source replay` makes replay the default.
- **Replay set:** the `--replay-*` options are optional. They copy the label file's photos
  that the board can decode, renamed `00000.jpg`, `00001.jpg`, ... in label-file order.
  Progressive, grayscale and larger-than-800x600 files are skipped and listed in
  `card.json`.
- **Cost:** CPU only, seconds. Without a replay set the card holds about 1 MB per model;
  227 photos add about 25 MB.

Then copy the `tinyseek` folder to the root of the FAT32 card:

- **Linux:** `cp -r runs/deploy/detector_card/sdcard/tinyseek /media/$USER/<card>/ && sync`
- **Windows:** copy `runs\deploy\detector_card\sdcard\tinyseek` onto the card.

If the card already has a `config.txt` with your Wi-Fi settings, copy only `models/` (and
`replay/`) so you keep them. The perf_eval folder (`/tinyseek_eval`) can stay on the same
card.

### SD layout

```
/tinyseek/config.txt             settings (key=value, one per line, '#' starts a comment)
/tinyseek/models/<prompt>.espdl  one ESP-DL model per prompt; the prompt is the file name
/tinyseek/models/<prompt>.txt    its sidecar: input size, stride, exponents, I/O dtypes, mean/std,
                                 score_threshold, max_detections, detect_threshold, sha256
/tinyseek/replay/00000.jpg ...   replay photos (optional)
/tinyseek/results/run_NNN/       written by the board in replay mode: run.txt, dets.jsonl,
                                 summary.txt
```

### Model input size in the sidecar

The model input may be square or rectangular (`model.input_size: [width, height]` when
training, for example 256x192 for the camera's 4:3 frame). The sidecar records it in one
of two forms:

| Model | Sidecar lines | Model file input (NHWC) |
| --- | --- | --- |
| square, e.g. 192x192 | `input_size=192` | [1, 192, 192, 3] |
| rectangular, e.g. 256 wide, 192 high | `input_width=256` and `input_height=192` (no `input_size`) | [1, 192, 256, 3] |

Cards staged before rectangular inputs existed have `input_size` only and load as they
always did. If a file has both forms, `input_width`/`input_height` are used.

The firmware takes the input and grid shape from the loaded model, not from the sidecar; it
uses the sidecar's size only to warn about a stale card ("input size differs from
<prompt>.txt"). The frame is scaled by min(input width / frame width, input height / frame
height) and centred, so an 800x600 frame fills a 256x192 input with no padding and has 24
padded rows above and below in a 192x192 one. Boxes are reported in frame pixels either way.

### `config.txt` keys

Missing keys take the default. The firmware prints a warning for unknown keys and invalid
values, then uses the default.

- **Format:** spaces and tabs around keys and values are ignored (`prompt = cup` works).
  A line whose first non-blank character is `#` is a comment. A UTF-8 byte-order mark, as
  Windows Notepad may add, is ignored.
- **Repeated keys:** if a key appears twice, the later line wins. Un-commenting the
  footer's `#   wifi=sta` therefore overrides the staged `wifi=off` above it.
- **Prompt switches:** when you switch prompts, the board rewrites the prompt line in
  `config.txt` and drops any duplicate prompt lines.
  - It writes `config.tmp` first, then swaps it in.
  - If that swap is interrupted, the next boot restores `config.txt` from `config.tmp`.
  - If `config.txt` exists but cannot be read, the board does not save the new prompt and
    warns, so your other settings are never lost.

| Key | Default | Values |
| --- | --- | --- |
| `prompt` | the first staged model | a model name, `[a-z0-9_]`, up to 32 characters |
| `source` | `camera` | `camera`, or `replay` (runs on `/tinyseek/replay/*.jpg`) |
| `decode_scale` | `0` | `0` full 800x600, `1` 400x300, `2` 200x150, `3` 100x75 (speed switch 1) |
| `resize` | `pillow` | `pillow` (the PC's exact resample) or `bilinear` (speed switch 2) |
| `decoder` | `tjpgd` | `tjpgd`, or `esp_new_jpeg` (full size only; no bit-exact PC reference) |
| `multicore` | `0` | `1`: ESP-DL splits convolutions over both cores (speed switch 3) |
| `internal_kb` | `0` | KB of activations in internal RAM (speed switch 4). It is clamped to what is free |
| `pipeline` | `0` | `1`: capture and decode run on core 0 while the model runs on core 1 (speed switch 5) |
| `led` | `both` | `both`, `onboard` (IO2), `external` (IO1, rig LED), `off` |
| `print_min_score` | `0.1` | lowest score listed in the serial/web `dets` (the count `ndet` includes all) |
| `wifi` | `off` | `off`, `sta` (join a network) or `ap` (the board is the access point) |
| `wifi_ssid`, `wifi_password` | | for `sta`. **Write these on the card only** |
| `ap_ssid`, `ap_password` | `tinyseek`, none | for `ap`. The password needs 8 to 63 characters; without one, Wi-Fi stays off |
| `hostname` | `tinyseek` | the STA hostname |
| `http_port` | `80` | |
| `camera_warmup_ms` | `4000` | frames dropped at start so exposure settles, as in the capture sketch |
| `replay_dir` | `/sdcard/tinyseek/replay` | |
| `replay_limit` | `0` | at most this many replay files; `0` means all |

Wi-Fi credentials never go into the repository. The staging tool refuses them; add them to
`config.txt` on the card by hand.

## 3. What you see on Serial

Every line the firmware writes is one JSON object with a `"type"`. A few plain-text lines
come from the shared camera code (for example a JPEG it could not decode); parsers should
skip any line that does not start with `{`. The values below are examples, not
measurements.

```
{"type":"boot","build":"arduino-esp32 3.3.12","idf":"v5.5.5","espdl":"esp-dl bundled with core 3.3.12","sdk":"dcache=32KB/32B icache=16KB spiram=80MHz sdk_opt=size","cpu_mhz":240}
{"type":"mem","point":"boot","mem":{"internal_free":...,"internal_min_free":...,"internal_largest":...,"psram_free":...,...}}
{"type":"config","settings":{"prompt":"backpack","source":"camera","decode_scale":0,...},"jpeg_decoder":"tjpgd (ROM TJpgDec via esp_jpeg)","warnings":[]}
{"type":"mem","point":"after_camera",...}
{"type":"mem","point":"before_load",...}
{"type":"mem","point":"after_load",...}
{"type":"model","model":{"prompt":"backpack","input":192,"grid":24,"channels":5,"in_exp":-6,"out_exp":-4,"in_dtype":"INT8","out_dtype":"INT8","bytes":649728,"load_ms":920.0,"internal_kb":0,"score_threshold":0.050,"detect_threshold":0.300,"max_detections":20,"sha256":"075daca68fd835e5"}}
{"type":"frame","n":1,"prompt":"backpack","src":"camera","w":800,"h":600,"ok":true,"t":{"cap":12.0,"dec":431.0,"pre":208.0,"inf":448.0,"post":0.40,"lat":1100.0,"period":0.0},"fps":0.00,"thr":0.300,"hit":true,"ndet":20,"dets":[{"s":0.812,"x":301.5,"y":222.0,"w":60.2,"h":81.0}]}
```

- **`model` fields:** for a square model `input` and `grid` are the input side in pixels
  and the output side in cells, as above. A rectangular model has
  `"input_w":256,"input_h":192,"grid_w":32,"grid_h":24` in their place.
- **`frame` fields:**
  - `n` is the frame number;
  - `t` holds the stage times in ms: `cap` (wait for and copy the camera frame, or read
    the file), `dec` (JPEG decode), `pre` (preprocessing), `inf` (`Model::run`) and `post`
    (box decode), plus `lat` (capture start to boxes) and `period` (time since the previous
    frame);
  - `fps` is a running mean of 1/`period`;
  - `hit` is true when any detection scores at least `thr` (the LED shows the same);
  - `dets` lists the detections with score at least `print_min_score`, in frame pixels
    (COCO xywh); `ndet` counts all of them.
- **Other types:** `mem` (heap at each stage), `wifi`, `web`, `prompt` (the result of a
  switch), `replay_start`, `replay_done`, `reply`, `status`, `warning`, `error` and `fatal`.
  After a `fatal` line the onboard LED blinks.

**Commands** (type in the Serial Monitor, then press Enter):

| Command | What it does |
| --- | --- |
| `prompt cup` | switches to `models/cup.espdl` if it exists. The reply lists the available prompts |
| `list` | lists the prompts on the card |
| `status` | the same JSON as the web page's `/status.json` |
| `replay` | runs the replay again into a new `results/run_NNN` (`source=replay` only) |

On a switch the board unloads the old model, loads the new one (about 1 s from the card)
and saves `prompt=` to `config.txt`. If the new model does not load, the old one is loaded
again. While a replay is running, switches are refused (its results file holds one model),
including a request that arrived just before the replay started.

## 4. The web page (optional)

Set `wifi=sta` with `wifi_ssid`/`wifi_password`, or `wifi=ap` with `ap_password`, on the
card. The `wifi` line on Serial gives the address. Open `http://<ip>/`; in AP mode that is
usually `http://192.168.4.1/`.

In STA mode the server starts even if the network has not answered within 20 s. The core
keeps reconnecting, and `"connected"` in the `wifi` line and in `/status.json` shows when it
is up. Type `status` on Serial to get the address once connected.

- **What the page shows:**
  - the latest frame, with the boxes drawn by the browser (green at or above the detect
    threshold; tick "show low scores" for the rest);
  - the frame's stage times and FPS;
  - a text box: type a prompt and press Detect. If `<prompt>.espdl` exists the board
    switches; otherwise the page says so and lists the prompts on the card as buttons;
  - the settings, the model, heap and Wi-Fi state (from `/status.json`, every 2 s).
- **Endpoints:**
  - `/frame.jpg?have=<n>` returns the camera JPEG exactly as captured (never re-encoded),
    with that frame's JSON line in the `X-Meta` header. It returns 204 if frame `n` is
    still the latest.
  - `/status.json` returns the state.
  - `/prompt?name=<typed>` switches the prompt.
- **Where it runs:** the server runs on core 0. Each frame costs the loop one copy of the
  JPEG (about 50-100 KB, PSRAM to PSRAM).

**Wi-Fi and internal RAM.** These are the measured numbers for the board without Wi-Fi
(CLAUDE.md, 2026-11-16):

- with `internal_kb=0`, internal heap never below 173 KB free;
- with `internal_kb=100`, 74 KB minimum free, largest block 32-43 KB;
- ESP-DL needs `internal_kb` as one block, and 200 failed with a 128 KB largest block.

The Wi-Fi driver, lwIP and the server task need internal RAM too:

- the core keeps its 8 static RX and 8 static TX buffers (about 1.6 KB each) in internal
  RAM for DMA;
- the web server task stack is 8 KB;
- `CONFIG_SPIRAM_TRY_ALLOCATE_WIFI_LWIP=y` in this core moves the other Wi-Fi and lwIP
  allocations to PSRAM.

Expect roughly 40-70 KB of internal RAM for Wi-Fi plus the server. This is an estimate:
the `mem` lines `after_wifi` and `after_web` give the real numbers on the first run.

- **With Wi-Fi on, keep `internal_kb=0`, or at most about 48.** With 100, the heap left
  for Wi-Fi would be near zero.
- **The order is chosen for this.** The firmware starts Wi-Fi, the camera, the web server
  and (with `pipeline=1`) the capture task before it loads the model. It then clamps
  `internal_kb` to the largest free internal block minus 24 KB and logs a warning; with
  `internal_kb=0` there is nothing to clamp and no warning. A too-large value therefore
  costs speed, not a crash or a failed load.

## 5. Replay: deployed accuracy on the held-out photos

1. Stage with `--replay-ann ... --replay-images ...` and `--source replay` (or set
   `source=replay` in `config.txt` on the card).
2. Boot the board. It runs every file in `/tinyseek/replay/` through exactly the camera
   path, with the same decode, resize and model settings. It prints a `frame` line per file
   and then `replay_done` with the mean stage times.
3. The results go to `/tinyseek/results/run_NNN/`:
   - `run.txt`: build, settings, model sha256 and input size (`input_size=192`, or
     `input_width=256` and `input_height=192` for a rectangular model); `complete=1` at the
     end;
   - `dets.jsonl`: one line per file, every detection at full precision;
   - `summary.txt`: per-stage mean/min/max.
4. Copy that folder back and score it:

   ```bash
   cp -r /media/$USER/<card>/tinyseek/results/run_001 runs/deploy/detector_card/results/
   ~/.venvs/tinyseek/bin/python -m tinyseek.deploy.replay --card runs/deploy/detector_card \
       --results runs/deploy/detector_card/results/run_001 \
       --json runs/deploy/detector_card/results/run_001/score.json
   ```

The scorer reports AP@0.5 (pycocotools, overall and by object size) and ECE, computed the
same way as `tinyseek.eval.evaluate`, plus the mean stage times. It warns if the board ran
a different model or prompt from the one staged, and it counts files without a result as
empty.

Replay also works as a benchmark of the speed switches: change one key in `config.txt`,
reset, and compare `summary.txt`. With `replay_limit=20` a run takes under a minute.

## 6. Speed switches

These are the measured numbers at 192 px with the defaults (CLAUDE.md, 2026-11-16), per
800x600 frame:

| Stage | Time |
| --- | --- |
| JPEG decode | 431 ms |
| preprocessing | 208 ms |
| inference | 448 ms (387 ms with `internal_kb=100`) |
| boxes | under 1 ms |

That is about 1.09 s per frame. The switches below are listed fastest-first to try. The
expected effects are estimates from the code, **not measurements**; replay mode (above)
measures each one.

| # | Switch | What changes | Expected | Pixels the model sees |
| --- | --- | --- | --- | --- |
| 1 | `decode_scale=1` / `2` | TJpgDec descales while decoding: each output pixel is the truncated mean of a 2x2 (4x4) block. Huffman decoding, IDCT and colour conversion still run at full size; only the output writing (esp_jpeg's per-pixel callback into PSRAM) shrinks 4x (16x) | decode perhaps 0.7-0.8x. The resize then starts from 400x300 or 200x150, so `pre` drops to roughly 1/3 (scale 1) or 1/8 (scale 2) of 208 ms | changed; bit-exact PC reference |
| 2 | `resize=bilinear` | 2-tap fixed-point bilinear instead of Pillow's wide filter; meant to follow switch 1 (from 200x150 to 192x144 it is nearly 1:1) | `pre` about 10-20 ms | changed; bit-exact PC reference |
| 3 | `multicore=1` | `Model::run(RUNTIME_MODE_MULTI_CORE)`: esp-dl 3.3.11 splits every convolution whose input is taller than 4x its kernel into a top half and a bottom half, one per core (`dl_base.hpp` ConvOpArgs; workers `dl_mc0`/`dl_mc1` at the caller's priority). Other layers stay single-core | inference perhaps 0.6-0.75x. It competes with the pipeline task, the camera driver and Wi-Fi on core 0 | unchanged (the same integer kernels) |
| 4 | `internal_kb=100` | up to 100 KB of the activation arena in internal RAM | measured 448 to 387 ms; mind the Wi-Fi note above | unchanged |
| 5 | `pipeline=1` | capture and decode of the next frame in a task on core 0 (8 KB stack, priority 1, two frame slots) while the loop runs preprocessing, inference and boxes on core 1 | frame period becomes max(capture+decode, pre+inference+boxes) instead of their sum: about 0.66 s instead of 1.09 s at the defaults. `lat` per frame stays the same or grows | unchanged |
| - | `decoder=esp_new_jpeg` | Espressif's esp_new_jpeg 1.0.2 (the ESP-IDF perf_eval build's decoder), full size only | unknown under this core; worth one replay run | changed; **no bit-exact PC reference** (closed source). Measure its accuracy with replay |

Switches 1, 2 and 5 combine: with `decode_scale=2`, `resize=bilinear` and `pipeline=1`,
the loop is bound by inference (about 390-450 ms), so roughly 2 frames per second at
192 px. Adding `multicore=1` may give more, or may lose against the core-0 capture task.
Measure it.

**PC side of the pixel-changing switches.** `tinyseek.deploy.reference.board_input(jpeg,
size, exponent, mode)` returns the int8 tensor the board computes for each mode ("full",
"half", "quarter", "eighth", each optionally with "_bilinear"), with the letterbox in frame
pixels. It decodes with the host build of the board's own decoder
(`tinyseek.deploy.tjpgd`). That decoder is bit-exact against the board on the full-size
path, which perf_eval measured on 8 photos x 2 models x 3 runs.

## What is verified, and what only the board can confirm

**Checked on the PC:**

- `pytest tests/test_deploy_detector.py`:
  - `config.txt` parsing;
  - prompt names;
  - the config rewrite;
  - the sidecar parser against what the stager writes, for square (`input_size`) and
    rectangular (`input_width`, `input_height`) models, and against hand-written old and
    new sidecars;
  - the JSON lines;
  - the replay scorer reading the firmware's own result lines.
- `tests/test_deploy_tjpgd.py` and `tests/test_deploy_core.py`:
  - host TJpgDec plus preprocessing reproduces the board's own int8 inputs from its JPEG
    decode (perf_eval runs 003, 004, 006);
  - the scaled decode equals the truncating box average of the full decode;
  - the C++ bilinear resample equals `reference.bilinear_resize` byte for byte;
  - the Pillow-exact path holds at 400x300 and 200x150;
  - rectangular inputs (256x192, 288x224 and the tall 192x256): preprocessing equals the
    PC reference byte for byte for both resamples, int8 and int16, and the decode of a
    non-square grid gives the PC's boxes; square inputs give the same bytes as before
    rectangular inputs were added (pinned hashes).
- Syntax: every `src/*.cpp` and `detector.ino` passes `g++ -fsyntax-only -Wall -Wextra
  -Werror`. Real headers from the core's esp32s3-libs-3.3.12 were used where they are
  self-contained (esp_camera.h, sensor.h, img_converters.h, jpeg_decoder.h, esp_jpeg_dec.h,
  sdkconfig.h). For esp-dl, esp_http_server, WiFi, FreeRTOS and the Arduino core, the check
  used declarations copied from those sources.

**Open until the first build and run:**

- The sketch compiles and links with the real core, including `esp_jpeg_decode` from
  `libespressif__esp_jpeg.a`. The core's `ld_libs` lists it, but no build has run.
- Scaled decode (`decode_scale` 1-3) on the board gives the same pixels as the host
  TJpgDec. Same ROM code, but only scale 0 has been compared.
- Timings of every switch, and whether `multicore=1` helps when `pipeline=1` or Wi-Fi is
  on.
- Internal heap with Wi-Fi on (the `after_wifi` and `after_web` lines).
- `esp_new_jpeg` under this core (never run).
- The web page on a phone over the board's AP.
- A rectangular model on the board: that esp-dl 3.3.11 loads and runs an export whose
  input is [1, 192, 256, 3] with output identical to the PC simulation, its inference time
  and its memory. Only square models (128 and 192 px) have run on the board.
