# capture: interval photos to microSD

Collects real photos from the target camera (Freenove ESP32-S3 WROOM CAM, OV2640). They are
used to train the Stable Diffusion domain LoRA, so generated images take on this sensor's
look, and, kept separate, as the on-device test set.

On every boot, and whenever button 1 is held for 3 s, the sketch starts a new folder on
the card (one session per object or place: note what each session number holds). It saves
one JPEG every `CAPTURE_INTERVAL_MS` (5 s by default); no interval photo is taken while
button 1 is held, and the first one in a new session comes one interval later.

```
/tinyseek/session_001/img_00001.jpg ...
/tinyseek/session_001/captures.csv     file, ms since boot, width, height, bytes, trigger
/tinyseek/session_001/session.txt      sensor id and the capture settings
```

| Control (camera rig v1) | GPIO | Behavior |
| --- | --- | --- |
| Button 1 (in parallel with BOOT) | 0 | tap: pause / resume the interval (on release); hold 3 s: start a new session folder, confirmed by three quick blinks |
| Button 2 | 47 | take a photo now, paused or not; pause first for manual-only shooting |
| Status LEDs (onboard IO2 + rig LED) | 2, 1 | short blink: photo saved; steady: paused; fast blinking: error, reason on serial (115200 baud) |

`trigger` in `captures.csv` is `interval` or `button`. Power the board through USB-C only
(BreadVolt USB-A when standalone, the PC cable when flashing), never both at once.

## Settings

At the top of `capture.ino`:

| Setting | Default | Notes |
| --- | --- | --- |
| `CAPTURE_INTERVAL_MS` | 5000 | ~720 photos per hour |
| `FRAME_SIZE` | `FRAMESIZE_SVGA` (800x600) | Enough for 512 px LoRA crops while keeping the sensor's character |
| `JPEG_QUALITY` | 10 | 0-63, lower is better; 10-12 is the usual range for the OV2640 |
| `MAX_IMAGES` | 0 (no limit) | Stop after N photos |
| `VFLIP` / `HMIRROR` | false / false | Upright in camera rig v1 (flex leaving the module's bottom edge) |
| `WARMUP_MS` | 4000 | Frames are dropped this long so exposure and white balance settle |
| `GAIN_CEILING` | `GAINCEILING_4X` | A little above the library's 2x for dim rooms; 16x amplified chroma noise |
| `AE_LEVEL` | 0 | Exposure bias, -2 to 2; +1 washed out well-lit rooms |
| `AEC_DSP` | false | The sensor's DSP exposure mode (`aec2`); brighter but hazier |
| `WB_MODE` | 0 (auto) | 1-4 lock a preset (sunny, cloudy, office, home) |

The sensor settings are part of the camera's "look": the detector firmware must use the
same values, so photos taken here match what the board sees when deployed.

## Checking color and exposure

```bash
python -m tinyseek.data.camera_check data/esp32cam/raw --sheet contact.jpg
```

prints median brightness and green/red, green/blue ratios per session. To tell a
firmware problem from a bad camera module, photograph white paper by a window in daylight:
G/R and G/B should be about 1.0 and brightness above 100. If the paper still comes out
clearly green (both ratios above 1.3) in good light, the module or its lens (missing or
poor IR-cut filter) is the likely cause and swapping the camera is worth trying.

Pins live in `board_pins.h`; another board gets its own `BOARD_*` block there.

## Build and flash

Arduino IDE 2 or `arduino-cli`, with the Espressif ESP32 core 3.x
(`https://espressif.github.io/arduino-esp32/package_esp32_index.json`). Board settings:

- Board: **ESP32S3 Dev Module**
- PSRAM: **OPI PSRAM** (the sketch refuses to run without it)
- Flash size: **16MB**
- USB CDC On Boot: **Enabled** only when using the board's native USB port; leave it disabled
  on the USB-UART port.

```bash
arduino-cli core install esp32:esp32
arduino-cli compile --fqbn esp32:esp32:esp32s3:PSRAM=opi,FlashSize=16M firmware/capture
PORT=/dev/ttyACM0   # or /dev/ttyUSB0; check `arduino-cli board list`
arduino-cli upload  --fqbn esp32:esp32:esp32s3:PSRAM=opi,FlashSize=16M -p $PORT firmware/capture
arduino-cli monitor -p $PORT -c baudrate=115200
```

The card must be FAT32 (32 GB or smaller is safest) and inserted before power-on.

## What to photograph

The LoRA should learn the camera, not a scene, so spread the photos out:

- **200-300 photos** for the LoRA, from at least 15-20 different places: kitchen, desk,
  living room, hallway, classroom, outdoors.
- Mix lighting: daylight, warm indoor lamps, dim rooms, backlit windows.
- Mix distance and angle: close-ups, across a room, looking down at a table.
- Include the target objects (person, cup, backpack) in about half the photos, at sizes you
  expect in the demo, plus ordinary clutter around them.
- Drop near-duplicates afterwards; at a 5 s interval, move the board between shots or pause
  with BOOT while repositioning.

Keep a **separate set of about 200 photos** for the deployment test set, taken in different
sessions from the LoRA photos. Those get box labels later and are never used for training.

## Copying off the card

Copy each session folder unchanged into the gitignored data directory:

```bash
mkdir -p data/esp32cam/raw
cp -rn /media/$USER/<card>/tinyseek/session_* data/esp32cam/raw/
```
