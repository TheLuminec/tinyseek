# capture: interval photos to microSD

Collects real photos from the target camera (Freenove ESP32-S3 WROOM CAM, OV2640). They are
used to train the Stable Diffusion domain LoRA, so generated images take on this sensor's
look, and, kept separate, as the on-device test set.

On every boot the sketch starts a new folder on the card and saves one JPEG every
`CAPTURE_INTERVAL_MS` (5 s by default):

```
/tinyseek/session_001/img_00001.jpg ...
/tinyseek/session_001/captures.csv     file, ms since boot, width, height, bytes
/tinyseek/session_001/session.txt      sensor id and the capture settings
```

| Control | Behavior |
| --- | --- |
| BOOT button | pause / resume |
| LED: short blink | one photo saved |
| LED: steady on | paused |
| LED: fast blinking | error; the reason is printed on serial (115200 baud) |

## Settings

At the top of `capture.ino`:

| Setting | Default | Notes |
| --- | --- | --- |
| `CAPTURE_INTERVAL_MS` | 5000 | ~720 photos per hour |
| `FRAME_SIZE` | `FRAMESIZE_SVGA` (800x600) | Enough for 512 px LoRA crops while keeping the sensor's character |
| `JPEG_QUALITY` | 10 | 0-63, lower is better; 10-12 is the usual range for the OV2640 |
| `MAX_IMAGES` | 0 (no limit) | Stop after N photos |
| `VFLIP` / `HMIRROR` | true / false | Check the first photo; flip if it is upside down |

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
