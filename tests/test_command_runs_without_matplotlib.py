"""The `foambryo` command starts without matplotlib, as a plain `pip install foambryo` leaves it.

matplotlib comes only with the `viewing` extra, yet building the command's parser imported the viewer's figure
module for one default value (the explode coefficient of `foambryo view --scattered`), and that module imports
matplotlib: every sub-command, `--help` and `--version` included, failed with `ModuleNotFoundError` after a plain
install. The suite always has matplotlib, so this runs the command in a fresh interpreter in which importing
matplotlib raises, and checks `--help`, every sub-command's `--help` and `--version`.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

BLOCK_MATPLOTLIB = """
import sys


class _NoMatplotlib:
    def find_spec(self, name, path=None, target=None):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ModuleNotFoundError(f"No module named {name!r} (blocked by the test)")
        return None


sys.meta_path.insert(0, _NoMatplotlib())
from foambryo.cli import main

sys.exit(main(sys.argv[1:]))
"""


@pytest.mark.parametrize(
    "arguments",
    [
        ["--help"],
        ["--version"],
        ["mesh", "--help"],
        ["infer", "--help"],
        ["infer-over-seeds", "--help"],
        ["view", "--help"],
    ],
)
def test_the_command_starts_without_matplotlib(arguments: list[str]) -> None:
    done = subprocess.run(  # noqa: S603 — fixed script + trusted arguments, a test subprocess
        [sys.executable, "-c", BLOCK_MATPLOTLIB, *arguments],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert "matplotlib" not in done.stderr
