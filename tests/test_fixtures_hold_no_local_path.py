"""No test fixture records a path on the machine that made it.

Up to 0.7.0 `tests/data/angle_reading_bit_identity_baseline.json` kept, under a key no test reads, the absolute
path of the dataset folder on its author's machine: the sdist and the public repository showed it.
"""

import re
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent / "data"
LOCAL_PATH = re.compile(r"/Users/|/home/[A-Za-z]|[A-Za-z]:\\\\Users")
TEXT_SUFFIXES = {".json", ".csv", ".txt", ".py", ".md"}


@pytest.mark.parametrize(
    "path",
    sorted(p for p in DATA.rglob("*") if p.is_file() and p.suffix in TEXT_SUFFIXES),
    ids=lambda p: p.relative_to(DATA).as_posix(),
)
def test_a_fixture_holds_no_local_path(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    found = LOCAL_PATH.search(text)
    assert found is None, f"{path.name} holds a local path at offset {found.start()}"
