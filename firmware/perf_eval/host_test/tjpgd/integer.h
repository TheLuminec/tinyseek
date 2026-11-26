/* Integer types for TJpgDec on the host, sized as on the ESP32-S3 (ILP32 Xtensa).
 *
 * Replaces the FatFs integer.h that ships with TJpgDec R0.01b. The chip's ROM header
 * (esp_rom/esp32s3/include/esp32s3/rom/tjpgd.h in ESP-IDF v5.5.5) declares the same names
 * with `long` for LONG/DWORD, which is 32 bits on Xtensa but 64 on x86-64 Linux, so the
 * host build uses fixed-width types to keep every intermediate the same width. Build with
 * -fwrapv so any signed overflow wraps as it does on the chip.
 */
#ifndef _INTEGER
#define _INTEGER

#include <stdint.h>

typedef int INT;
typedef unsigned int UINT;

typedef char CHAR;
typedef unsigned char UCHAR;
typedef unsigned char BYTE;

typedef short SHORT;
typedef unsigned short USHORT;
typedef unsigned short WORD;
typedef unsigned short WCHAR;

typedef int32_t LONG;
typedef uint32_t ULONG;
typedef uint32_t DWORD;

#endif
