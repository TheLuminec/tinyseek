"""The board's JPEG decoder on the PC: TJpgDec R0.01b, as the ESP32-S3 ROM runs it.

On arduino-esp32 3.3.12 every JPEG the firmware decodes (esp32-camera's ``fmt2rgb888`` in
perf_eval, ``esp_jpeg_decode`` in the detector) goes through espressif/esp_jpeg 1.3.1 with
``CONFIG_JD_USE_ROM=y``, i.e. the chip's ROM copy of ChaN's TJpgDec R0.01b. That decoder is
plain integer C, so compiling the same source on the PC reproduces it bit for bit:
decoding the eight staged parity photos here and preprocessing them gives the same int8
tensors the board wrote in runs/deploy/stage_backpack/results/run_003, run_004 and run_006
(tests/test_deploy_tjpgd.py). Scaled decoding (1/2, 1/4, 1/8) is the same code path plus a
per-MCU box average with truncation (``r >> 2*scale``), so it is covered by the same source,
but it has not yet been compared against tensors from the board.

The C sources live in firmware/perf_eval/host_test/tjpgd/; the first call compiles them
with the system C compiler into a cached shared library (``$TINYSEEK_BUILD_CACHE`` or
``runs/.build``).
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "firmware" / "perf_eval" / "host_test" / "tjpgd"
SOURCES = ("tjpgd.c", "tjpgd_host.c")
HEADERS = ("tjpgd.h", "integer.h")
JRESULT = {1: "JDR_INTR", 2: "JDR_INP", 3: "JDR_MEM1", 4: "JDR_MEM2", 5: "JDR_PAR",
           6: "JDR_FMT1 (damaged data)", 7: "JDR_FMT2 (unsupported)",
           8: "JDR_FMT3 (unsupported JPEG: progressive, grayscale or unusual sampling)"}


class TjpgdError(RuntimeError):
    pass


def compiler() -> str | None:
    for cc in (os.environ.get("CC"), "cc", "gcc", "clang"):
        if cc and shutil.which(cc):
            return cc
    return None


def available() -> bool:
    return compiler() is not None and all((SRC / f).exists() for f in SOURCES + HEADERS)


def _cache_dir() -> Path:
    return Path(os.environ.get("TINYSEEK_BUILD_CACHE", REPO / "runs" / ".build"))


@lru_cache(maxsize=1)
def _lib() -> ctypes.CDLL:
    cc = compiler()
    if cc is None:
        raise TjpgdError("no C compiler found (set CC); the board-exact JPEG decoder needs one")
    digest = hashlib.sha256()
    for name in SOURCES + HEADERS:
        digest.update((SRC / name).read_bytes())
    out = _cache_dir() / f"tjpgd_{digest.hexdigest()[:16]}.so"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
            tmp_out = Path(tmp) / out.name
            subprocess.run([cc, "-O2", "-fwrapv", "-shared", "-fPIC", "-w", f"-I{SRC}",
                            *(str(SRC / s) for s in SOURCES), "-o", str(tmp_out)], check=True)
            os.replace(tmp_out, out)  # atomic: concurrent builds cannot see a partial file
    lib = ctypes.CDLL(str(out))
    for fn in (lib.tinyseek_tjpgd_size, lib.tinyseek_tjpgd_decode):
        fn.restype = ctypes.c_int
    lib.tinyseek_tjpgd_size.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int,
                                        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    lib.tinyseek_tjpgd_decode.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p,
                                          ctypes.c_size_t, ctypes.POINTER(ctypes.c_int),
                                          ctypes.POINTER(ctypes.c_int)]
    return lib


def decode(jpeg: bytes | Path | str, scale: int = 0) -> np.ndarray:
    """(H >> scale, W >> scale, 3) uint8 RGB, exactly as the board's ROM TJpgDec decodes it.

    ``scale`` is TJpgDec's descaling factor: 0 full size, 1 half, 2 quarter, 3 eighth.
    Raises TjpgdError for files the board cannot decode either (progressive, grayscale).
    """
    data = Path(jpeg).read_bytes() if isinstance(jpeg, (str, Path)) else bytes(jpeg)
    if scale not in (0, 1, 2, 3):
        raise ValueError(f"scale must be 0..3, got {scale}")
    lib = _lib()
    w, h = ctypes.c_int(), ctypes.c_int()
    rc = lib.tinyseek_tjpgd_size(data, len(data), scale, ctypes.byref(w), ctypes.byref(h))
    if rc:
        raise TjpgdError(f"jd_prepare failed: {JRESULT.get(rc, rc)}")
    out = np.empty((h.value, w.value, 3), np.uint8)
    rc = lib.tinyseek_tjpgd_decode(data, len(data), scale, out.ctypes.data, out.nbytes,
                                   ctypes.byref(w), ctypes.byref(h))
    if rc:
        raise TjpgdError(f"jd_decomp failed: {JRESULT.get(rc, rc)}")
    return out
