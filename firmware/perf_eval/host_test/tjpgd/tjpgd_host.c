/* The board's JPEG decoder on the host: ChaN's TJpgDec R0.01b driven exactly the way
 * esp_jpeg 1.3.1's esp_jpeg_decode() drives the ESP32-S3 ROM copy of it.
 *
 * On arduino-esp32 3.3.12, esp32-camera's fmt2rgb888() and the detector's decoder both
 * call esp_jpeg_decode() (espressif/esp_jpeg 1.3.1, CONFIG_JD_USE_ROM=y), which runs the
 * ROM's TJpgDec (jd_prepare/jd_decomp at fixed ROM addresses). The ROM header matches
 * R0.01b (JD_SZBUF 512, JD_FORMAT 0 = RGB888, JD_USE_SCALE 1, JD_TBLCLIP 1). This file
 * repeats esp_jpeg's input callback (copy from memory), its 3100-byte work buffer and its
 * output callback (each rectangle written at y * (width / div) + x, RGB888, no byte swap).
 *
 * tinyseek.deploy.tjpgd builds this into a shared library and calls it through ctypes;
 * tests/test_deploy_tjpgd.py checks it against tensors the board itself produced.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "tjpgd.h"

#define WORK_BUF_SIZE 3100 /* esp_jpeg's JPEG_WORK_BUF_SIZE (and esp32-camera's static buffer) */

typedef struct {
    const uint8_t *data;
    size_t size, read;
    uint8_t *out;
    unsigned line; /* output pixels per row: width / div */
} host_dev_t;

static UINT in_cb(JDEC *jd, BYTE *buff, UINT nbyte)
{
    host_dev_t *dev = (host_dev_t *)jd->device;
    size_t to_read = nbyte;
    if (buff) {
        if (dev->read + to_read > dev->size) to_read = dev->size - dev->read;
        memcpy(buff, dev->data + dev->read, to_read);
    }
    dev->read += to_read; /* esp_jpeg also advances by nbyte when skipping */
    return (UINT)to_read;
}

static UINT out_cb(JDEC *jd, void *bitmap, JRECT *rect)
{
    host_dev_t *dev = (host_dev_t *)jd->device;
    const uint8_t *in = (const uint8_t *)bitmap;
    for (int y = rect->top; y <= rect->bottom; y++) {
        for (int x = rect->left; x <= rect->right; x++) {
            uint8_t *dst = dev->out + ((size_t)y * dev->line + x) * 3;
            dst[0] = in[0];
            dst[1] = in[1];
            dst[2] = in[2];
            in += 3;
        }
    }
    return 1;
}

/* Output size for `scale` (0..3: 1, 1/2, 1/4, 1/8) without decoding. 0 on success, else a
 * JRESULT code. */
int tinyseek_tjpgd_size(const uint8_t *jpeg, size_t len, int scale, int *w, int *h)
{
    static BYTE work[WORK_BUF_SIZE];
    JDEC jd;
    host_dev_t dev = {jpeg, len, 0, NULL, 0};
    JRESULT rc = jd_prepare(&jd, in_cb, work, sizeof(work), &dev);
    if (rc != JDR_OK) return (int)rc;
    const unsigned div = 1u << scale;
    *w = (int)(jd.width / div);
    *h = (int)(jd.height / div);
    return 0;
}

/* Decodes into `out` (cap bytes, at least w*h*3 for the size above). 0 on success, a
 * JRESULT code on a decode error, -1 if `out` is too small. The buffer is zeroed first:
 * esp_jpeg leaves pixels of skipped (fully rounded-off) MCUs untouched. */
int tinyseek_tjpgd_decode(const uint8_t *jpeg, size_t len, int scale, uint8_t *out, size_t cap, int *w, int *h)
{
    static BYTE work[WORK_BUF_SIZE];
    JDEC jd;
    host_dev_t dev = {jpeg, len, 0, out, 0};
    if (scale < 0 || scale > 3) return (int)JDR_PAR;
    JRESULT rc = jd_prepare(&jd, in_cb, work, sizeof(work), &dev);
    if (rc != JDR_OK) return (int)rc;
    const unsigned div = 1u << scale;
    const size_t need = (size_t)(jd.height / div) * (jd.width / div) * 3;
    if (need > cap) return -1;
    memset(out, 0, need);
    dev.line = jd.width / div;
    *w = (int)(jd.width / div);
    *h = (int)(jd.height / div);
    rc = jd_decomp(&jd, out_cb, (BYTE)scale);
    return rc == JDR_OK ? 0 : (int)rc;
}
