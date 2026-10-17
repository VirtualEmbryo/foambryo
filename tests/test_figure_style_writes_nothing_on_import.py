"""The figure style module must not write anywhere when it is imported.

It used to create ``<workspace root>/figures/`` at import time, and dw3d's provenance test executes
this module from the sibling checkout, so every dw3d test run recreated that folder in the
workspace root. Figures now go to the Git-ignored ``foambryo_benchmarks/outputs/figures/``, which
is created only when a figure is saved.
"""

from __future__ import annotations

import importlib.util
import pathlib
import shutil
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
STYLE_PATH = REPO_ROOT / "foambryo_benchmarks" / "figures" / "_style.py"


def _execute_style_module(monkeypatch):
    created = []
    real_mkdir = pathlib.Path.mkdir

    def recording_mkdir(self: pathlib.Path, *args: object, **kwargs: object) -> None:
        created.append(self)
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "mkdir", recording_mkdir)
    name = "_figure_style_under_test"
    spec = importlib.util.spec_from_file_location(name, STYLE_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module, created


def test_importing_the_style_module_creates_no_directory_in_the_workspace(monkeypatch) -> None:
    # matplotlib may create its own configuration folder in the home directory; that is not ours.
    workspace = REPO_ROOT.parent
    _module, created = _execute_style_module(monkeypatch)
    assert [p for p in created if p.resolve().is_relative_to(workspace)] == []


def test_figures_go_to_the_ignored_outputs_folder_inside_the_repository(monkeypatch) -> None:
    module, _created = _execute_style_module(monkeypatch)
    assert module.FIGURES_OUT == REPO_ROOT / "foambryo_benchmarks" / "outputs" / "figures"
    probe = module.FIGURES_OUT.relative_to(REPO_ROOT) / "probe.png"
    git = shutil.which("git")
    assert git is not None
    checkout = subprocess.run(  # noqa: S603 — fixed arguments + trusted paths, a test subprocess
        [git, "-C", str(REPO_ROOT), "rev-parse", "--is-inside-work-tree"],
        check=False,
        capture_output=True,
    )
    if checkout.returncode != 0:
        pytest.skip("not a Git checkout (an unpacked source distribution): nothing is ignored or tracked")
    ignored = subprocess.run(  # noqa: S603 — fixed arguments + trusted paths, a test subprocess
        [git, "-C", str(REPO_ROOT), "check-ignore", "-q", str(probe)],
        check=False,
    )
    assert ignored.returncode == 0, f"{probe} is not ignored by Git"
