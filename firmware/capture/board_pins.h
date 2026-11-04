// Camera, SD card, LED and button pins per board. Pick one with BOARD_* in capture.ino.
#pragma once

#if defined(BOARD_FREENOVE_S3_CAM)
// Freenove ESP32-S3 WROOM CAM (N16R8, OV2640). The camera is wired like Espressif's
// ESP32-S3-EYE; the microSD slot uses SD_MMC in 1-bit mode.
#define CAM_PIN_PWDN -1
#define CAM_PIN_RESET -1
#define CAM_PIN_XCLK 15
#define CAM_PIN_SIOD 4
#define CAM_PIN_SIOC 5
#define CAM_PIN_D7 16  // Y9
#define CAM_PIN_D6 17  // Y8
#define CAM_PIN_D5 18  // Y7
#define CAM_PIN_D4 12  // Y6
#define CAM_PIN_D3 10  // Y5
#define CAM_PIN_D2 8   // Y4
#define CAM_PIN_D1 9   // Y3
#define CAM_PIN_D0 11  // Y2
#define CAM_PIN_VSYNC 6
#define CAM_PIN_HREF 7
#define CAM_PIN_PCLK 13

#define SD_PIN_CLK 39
#define SD_PIN_CMD 38
#define SD_PIN_D0 40

// Camera rig v1 wiring: button 1 is in parallel with BOOT; button 2 and the external
// LED are on the rig's control deck. GPIO48 is the onboard WS2812 (unused); GPIO21 spare.
#define LED_PIN 2        // onboard "IO2" LED; -1 to disable
#define EXT_LED_PIN 1    // rig status LED via 330 ohm to GND; -1 to disable
#define BUTTON_PIN 0     // BOOT / rig button 1: pause-resume, active low; -1 to disable
#define BUTTON2_PIN 47   // rig button 2: photo now, active low; -1 to disable
#define LED_ON HIGH

#else
#error "No board selected: define one BOARD_* at the top of capture.ino (see board_pins.h)"
#endif
