"""Reference quantization, model.info parsing and staging helpers in tinyseek.deploy."""

from __future__ import annotations

import argparse

import numpy as np
import pytest

from tinyseek.data.dataset import Record
from tinyseek.deploy.layout import read_kv, write_kv
from tinyseek.deploy.reference import dequantize, quantize
from tinyseek.deploy.simulate import info_signature
from tinyseek.deploy.stage import graph_io, parse_model_arg, pick_cases

INFO = """model properties:

graph main_graph (
  %input.1[INT8, 1x192x192x3], exponents: [-6]
) initializers (
  %w0[INT8, 3]
  %w1[INT16, 2]
) {
  %a = Conv[group = 1](%input.1, %w0)
  %641 = Concat[axis = 3](%a, %w1)
  return %641
}


initializers value:
%w0, shape: [3], exponents: [-4],
value: array([1, 2, 3])
%w1, shape: [2], exponents: [-12],
value: array([5, 6])
%641[INT8, 1x24x24x5], exponents: [-4]
%a[INT8, 1x24x24x3], exponents: [-3]
"""


def test_quantize_rounds_half_up_and_saturates():
    x = np.array([0.5, -0.5, 1.5, -1.5, 2.49, 1000.0, -1000.0], np.float32)
    assert quantize(x, 0).tolist() == [1, 0, 2, -1, 2, 127, -128]
    assert quantize(np.float32([0.25 + 1 / 128]), -6).tolist() == [17]  # 16.5 -> 17
    assert np.allclose(dequantize(np.int8([-128, 64]), -6), [-2.0, 1.0])


def test_graph_io_reads_shapes_and_exponents():
    inp, out = graph_io(INFO)
    assert (inp.name, inp.shape, inp.dtype, inp.exponent) == ("input.1", (1, 192, 192, 3), "INT8", -6)
    assert (out.name, out.shape, out.exponent) == ("641", (1, 24, 24, 5), -4)


def test_info_signature_ignores_order_but_not_values():
    head, values = INFO.split("initializers value:")
    w0 = "%w0, shape: [3], exponents: [-4],\nvalue: array([1, 2, 3])\n"
    w1 = "%w1, shape: [2], exponents: [-12],\nvalue: array([5, 6])\n"
    swapped = head.replace("  %w0[INT8, 3]\n  %w1[INT16, 2]", "  %w1[INT16, 2]\n  %w0[INT8, 3]")
    swapped += "initializers value:\n" + w1 + w0 + values.split("value: array([5, 6])\n", 1)[1]
    assert info_signature(INFO) == info_signature(swapped)
    assert info_signature(INFO) != info_signature(INFO.replace("[1, 2, 3]", "[1, 2, 4]"))
    assert info_signature(INFO) != info_signature(INFO.replace("exponents: [-3]", "exponents: [-2]"))


CLIP_INFO = """graph main_graph (
  %input.1[INT8, 1x8x8x3], exponents: [-6]
) initializers (
  %PPQ_Variable_{lo}[INT8, scalar]
  %PPQ_Variable_{hi}[INT8, scalar]
) {{
  %y = Clip[quant_type = 'S8'](%input.1, %PPQ_Variable_{lo}, %PPQ_Variable_{hi})
  return %y
}}

initializers value:
%PPQ_Variable_{lo}, shape: [], exponents: [-4],
value: array([0])
%PPQ_Variable_{hi}, shape: [], exponents: [-4],
value: array([{max}])
%y[INT8, 1x8x8x3], exponents: [-4]
"""


def test_info_signature_ignores_generated_numbering():
    """Re-exports renumber esp-ppq's PPQ_Variable_N (a Clip's min and max swap numbers)."""
    a = CLIP_INFO.format(lo=1, hi=37, max=127)
    b = CLIP_INFO.format(lo=37, hi=1, max=127)
    assert info_signature(a) == info_signature(b)
    assert info_signature(a) != info_signature(CLIP_INFO.format(lo=37, hi=1, max=100))


def test_info_signature_ignores_where_trailing_section_headers_land():
    """The dump's 'values info:' and 'test ... value:' headers follow whichever block is
    last; a w16a16 re-export put a different tensor last, which must not count as a change."""
    head = INFO.split("initializers value:")[0]
    w0 = "%w0, shape: [3], exponents: [-4],\nvalue: array([1, 2, 3])\n"
    w1 = "%w1, shape: [2], exponents: [-12],\nvalue: array([5, 6])\n"
    tail = "values info:\n%641[INT8, 1x24x24x5], exponents: [-4]\n%a[INT8, 1x24x24x3], exponents: [-3]\n" \
           "test inputs value:\n\n\ntest outputs value:\n"
    tail_swapped = "values info:\n%a[INT8, 1x24x24x3], exponents: [-3]\n%641[INT8, 1x24x24x5], exponents: [-4]\n" \
                   "test inputs value:\n\n\ntest outputs value:\n"
    a = head + "initializers value:\n" + w0 + w1 + tail
    b = head + "initializers value:\n" + w1 + w0 + tail_swapped
    assert info_signature(a) == info_signature(b)
    assert info_signature(a) != info_signature(b.replace("exponents: [-3]", "exponents: [-2]"))


def test_pick_cases_prefers_images_with_boxes_and_is_spread():
    recs = [Record(f"{i}.jpg", i, np.zeros((i % 2, 4), np.float32)) for i in range(20)]
    picked = pick_cases(recs, 4)
    assert [r.image_id for r in picked] == [1, 7, 13, 19]
    assert len(pick_cases(recs, 50)) == 10


def test_parse_model_arg_validates_card_names(tmp_path):
    assert parse_model_arg(f"px192={tmp_path}") == ("px192", tmp_path)
    for bad in ("PX192=x", "px-192=x", "noequals", "averyveryverylongname=x"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_model_arg(bad)


def test_kv_roundtrip(tmp_path):
    write_kv(tmp_path / "a.txt", {"runs": 50, "mean": "0.485,0.456,0.406"})
    assert read_kv(tmp_path / "a.txt") == {"runs": "50", "mean": "0.485,0.456,0.406"}
