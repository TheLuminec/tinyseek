"""The Arduino sketches' copies of the shared firmware sources must match their sources."""

from __future__ import annotations

import shutil
from pathlib import Path

from tinyseek.deploy import arduino_sketch as sk


def test_sketch_copies_are_up_to_date():
    assert sk.stale() == [], "run: python -m tinyseek.deploy.arduino_sketch"
    assert sk.orphans() == []


def test_sketch_includes_resolve_inside_the_sketch():
    assert sk.unresolved_includes() == []


def test_sketches_have_no_other_sources():
    """Every C/C++ file in a sketch folder is either generated or listed as hand-written."""
    for sketch in sk.SKETCHES:
        folder = sk.REPO / sketch.path
        known = set(sketch.copies) | sketch.hand_written
        found = {p.relative_to(folder).as_posix() for p in folder.rglob("*")
                 if p.is_file() and p.suffix in {".c", ".cpp", ".h", ".hpp", ".ino"}}
        assert found <= known, (sketch.path, found - known)
        assert sketch.ino in sketch.hand_written and (folder / sketch.ino).exists()


def test_both_sketches_share_the_core():
    """The detector and the evaluation run the same preprocessing, decode and JPEG code."""
    for dest in ("src/preprocess.cpp", "src/decode.cpp", "src/camera.cpp", "src/platform.cpp"):
        assert sk.DETECTOR.copies[dest] == sk.PERF_EVAL.copies[dest]


def _copy_sources(root: Path) -> None:
    for sketch in sk.SKETCHES:
        for rel in sketch.copies.values():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(sk.REPO / rel, root / rel)


def test_generate_detects_and_fixes_drift(tmp_path: Path):
    root = tmp_path / "repo"
    _copy_sources(root)
    total = sum(len(s.copies) for s in sk.SKETCHES)
    assert len(sk.stale(root)) == total  # nothing generated yet
    sk.generate(root)
    assert sk.stale(root) == []
    src = root / sk.SHARED["src/decode.cpp"]
    src.write_text(src.read_text() + "// change\n")
    assert sk.stale(root) == ["firmware/perf_eval_arduino/src/decode.cpp", "firmware/detector/src/decode.cpp"]
    old = root / sk.DETECTOR.path / "src/old.cpp"
    old.write_text(sk.banner("gone.cpp") + "int x;\n")
    assert sk.orphans(root) == ["firmware/detector/src/old.cpp"]
    sk.generate(root)
    assert sk.stale(root) == [] and sk.orphans(root) == [] and not old.exists()


def test_hand_written_files_are_never_generated():
    for sketch in sk.SKETCHES:
        assert not set(sketch.copies) & sketch.hand_written
        for rel in sketch.hand_written:
            path = sk.REPO / sketch.path / rel
            assert path.exists(), path
            assert sk.MARKER not in path.read_text(errors="ignore"), path
