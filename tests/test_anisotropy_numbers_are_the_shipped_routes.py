"""Every public text quotes the anisotropy numbers measured on the shipped route, and none quotes the older ones.

Up to 0.7.0 the README, the changelog, the documentation and the docstrings quoted "+0.8, +1.2 and +1.2 degrees ...
against +1.8, +2.4 and +5.0", measured with a research up-sampler and another angle statistic. The numbers measured
on the shipped route (`construct_mesh_in_physical_units`, one seed, 12 simulated tissues at a real cell size) are
+0.83, +0.71 and +1.17 degrees, against +1.38, +1.97 and +4.10 for meshing the native grid.
"""

import json
import re
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
OLD = [re.compile(r"\+0\.8, \+1\.2 and \+1\.2"), re.compile(r"\+1\.8, \+2\.4 and \+5\.0")]
SHIPPED = "+0.83, +0.71 and +1.17"


def _public_texts() -> list[Path]:
    paths = [REPOSITORY / name for name in ("README.md", "CHANGELOG.md", "RELEASE_NOTES.md")]
    for folder, patterns in (("docs", ("*.md",)), ("src", ("*.py",)), ("Examples", ("*.py", "*.ipynb"))):
        for pattern in patterns:
            paths += sorted((REPOSITORY / folder).rglob(pattern))
    return [path for path in paths if path.is_file()]


def _text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".ipynb":
        raw = "\n".join("".join(cell["source"]) for cell in json.loads(raw)["cells"])
    return " ".join(raw.replace("#:", " ").replace("*", " ").split())


@pytest.mark.parametrize("path", _public_texts(), ids=lambda p: p.relative_to(REPOSITORY).as_posix())
def test_no_public_text_quotes_the_older_numbers(path):
    text = _text(path)
    for pattern in OLD:
        assert pattern.search(text) is None, f"{path.name} quotes {pattern.pattern}"


@pytest.mark.parametrize("name", ["README.md", "CHANGELOG.md"])
def test_the_readme_and_the_changelog_quote_the_shipped_route(name):
    assert SHIPPED in _text(REPOSITORY / name)
