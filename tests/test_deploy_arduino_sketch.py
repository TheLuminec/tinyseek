"""The Arduino sketch's copies of the shared firmware sources must match their sources."""

from __future__ import annotations

import shutil
from pathlib import Path

from tinyseek.deploy import arduino_sketch as sk


def test_sketch_copies_are_up_to_date():
    assert sk.stale() == [], "run: python -m tinyseek.deploy.arduino_sketch"
    assert sk.orphans() == []


def test_sketch_includes_resolve_inside_the_sketch():
    assert sk.unresolved_includes() == []


def test_sketch_has_no_other_sources():
    """Every compiled file in the sketch is either generated or listed as hand-written."""
    sketch = sk.REPO / sk.SKETCH
    known = set(sk.COPIES) | sk.HAND_WRITTEN
    found = {p.relative_to(sketch).as_posix() for p in sketch.rglob("*")
             if p.is_file() and p.suffix in {".c", ".cpp", ".h", ".hpp", ".ino"}}
    assert found <= known, found - known


def test_generate_detects_and_fixes_drift(tmp_path: Path):
    root = tmp_path / "repo"
    for rel in sk.COPIES.values():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sk.REPO / rel, root / rel)
    assert len(sk.stale(root)) == len(sk.COPIES)  # nothing generated yet
    sk.generate(root)
    assert sk.stale(root) == []
    src = root / sk.COPIES["src/decode.cpp"]
    src.write_text(src.read_text() + "// change\n")
    assert sk.stale(root) == ["src/decode.cpp"]
    old = root / sk.SKETCH / "src/old.cpp"
    old.write_text(sk.banner("gone.cpp") + "int x;\n")
    assert sk.orphans(root) == ["src/old.cpp"]
    sk.generate(root)
    assert sk.stale(root) == [] and not old.exists()
