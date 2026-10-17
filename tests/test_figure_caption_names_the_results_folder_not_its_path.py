"""`foambryo view --results OUTDIR`: the panel and a saved figure's caption name OUTDIR, never the path to it.

Up to 0.7.0 the caption, `Values: read from OUTDIR ...`, repeated OUTDIR as typed, so a figure made with an absolute
path showed the local folders above the results to whoever it was shared with.
"""

from __future__ import annotations

import pytest

from foambryo import _result_tables as tables
from foambryo import cli, dcel_mesh_from_file
from foambryo._viewer_figures import interfaces_on_mesh
from tests.conftest import require_substitute_mesh


@pytest.fixture
def recorded_viewer(monkeypatch):
    """The three viewer functions replaced by recorders of the command's arguments: no window."""
    pytest.importorskip("polyscope")
    import foambryo.viewing as viewing

    calls: list[tuple[str, tuple, dict]] = []
    for name in ("plot_force_inference", "plot_tension_inference", "plot_pressure_inference"):
        monkeypatch.setattr(viewing, name, lambda *args, _name=name, **kwargs: calls.append((_name, args, kwargs)))
    return calls


def _run(capsys, *arguments: object) -> tuple[int, str]:
    code = cli.main([str(argument) for argument in arguments])
    return code, capsys.readouterr().err


def test_results_of_infer_are_named_by_their_folder(capsys, tmp_path, recorded_viewer):
    mesh_path = require_substitute_mesh()
    outdir = (tmp_path / "local" / "folders" / "my-results").resolve()
    code, err = _run(capsys, "infer", mesh_path, "-o", outdir)
    assert code == cli.EXIT_OK, err
    code, err = _run(capsys, "view", mesh_path, "--results", outdir)
    assert code == cli.EXIT_OK, err
    values_from = recorded_viewer[-1][2]["values_from"]
    assert values_from.startswith("read from my-results (foambryo infer: ")
    assert str(tmp_path) not in values_from
    assert "local" not in values_from


def test_an_ensemble_is_named_by_its_folder(capsys, tmp_path, recorded_viewer):
    mesh_path = require_substitute_mesh()
    outdir = (tmp_path / "local" / "folders" / "my-ensemble").resolve()
    outdir.mkdir(parents=True)
    lines = [",".join(tables.ENSEMBLE_TENSION_COLUMNS)]
    for a, b in sorted(interfaces_on_mesh(dcel_mesh_from_file(mesh_path))):
        lines.append(f"{a},{b},1.0,0.01,8,false")
    (outdir / tables.TENSION_TABLE).write_text("\n".join(lines) + "\n", encoding="ascii")
    code, err = _run(capsys, "view", mesh_path, "--results", outdir)
    assert code == cli.EXIT_OK, err
    values_from = recorded_viewer[-1][2]["values_from"]
    assert values_from == "tensions combined over seeds, read from my-ensemble"
    assert str(tmp_path) not in values_from


def test_the_current_folder_is_named_too(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cli._folder_name(cli.Path(".")) == tmp_path.resolve().name
