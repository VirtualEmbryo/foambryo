"""`foambryo infer --no-pressures` and `solves.csv`: two outputs a front end needs to offer on the command line.

What this pins, and why:

* **Tensions only are `infer_forces`' first call, bitwise.** `--no-pressures` (and the library's
  `infer_tensions_then_pressures(..., with_pressures=False)`) runs `infer_tensions` with the same options and
  nothing else: `tensions.csv` reads back bitwise as `infer_tensions`' map and as `infer_forces`' tensions, no
  `pressures.csv` is written, and `run.json` says that the pressures were not computed rather than recording
  pressure options that were not used. A pressure option given with `--no-pressures` is refused, not ignored.
* **The solve table is the run record's `solves`.** With `--diagnostics basic` or `spectrum`, `solves.csv` holds
  one row per linear solve, and every field reads back equal to the same run's `run.json` entry (floats bitwise),
  so a reader can use either; `--overwrite` deletes it like the other outputs.

The benchmark mesh 000 and a doublet with a closed sphere (an interface with no triple line) only.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from dw3d.io import save_rec

from foambryo import (
    TensionComputationMethod,
    cli,
    dcel_mesh_from_file,
    infer_forces,
    infer_tensions,
    infer_tensions_then_pressures,
)
from foambryo.results import SOLVE_COLUMNS, SOLVE_TABLE, read_solve_table, read_tension_table, write_solve_table
from foambryo_benchmarks import phantoms as ph
from tests.test_command_line import _run, _same_floats
from tests.test_junction_graph import _combine, _uv_sphere

from .conftest import require_substitute_mesh


@pytest.fixture(scope="module")
def benchmark_mesh():
    return require_substitute_mesh()


# ---------------------------------------------------------------------------------------------
# tensions only
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("options", [[], ["--tension-method=lami", "--mean-tension=2.5"], ["--solver=dense"],
                                     ["--diagnostics=basic"]], ids=lambda o: " ".join(o) or "defaults")
def test_no_pressures_writes_infer_tensions_bitwise_and_no_pressure_table(capsys, tmp_path, benchmark_mesh, options):
    outdir = tmp_path / "results"
    code, out, err = _run(capsys, "infer", benchmark_mesh, "-o", outdir, "--no-pressures", *options)
    assert code == cli.EXIT_OK, err
    args = cli.build_parser().parse_args(["infer", "m.rec", "-o", "out", *options])
    mesh = dcel_mesh_from_file(benchmark_mesh)
    tensions = infer_tensions(mesh, mean_tension=args.mean_tension, mode=cli.TENSION_METHODS[args.tension_method],
                              solver=args.solver)
    written = read_tension_table(outdir / "tensions.csv")
    assert _same_floats(written, tensions)
    assert _same_floats(written, infer_forces(mesh, mean_tension=args.mean_tension, solver=args.solver,
                                              mode_tension=cli.TENSION_METHODS[args.tension_method])[0])
    assert not (outdir / "pressures.csv").exists()
    assert "pressures.csv" not in out
    record = json.loads((outdir / "run.json").read_text())
    assert record["status"] == "complete"
    assert record["options"]["no_pressures"] is True
    assert record["options"]["pressure_method"] is None
    assert record["options"]["base_pressure"] is None
    assert record["library_call"].startswith("infer_tensions(mesh, ")
    assert record["results"]["pressures"] is None
    assert record["results"]["undetermined_pressures"] is None
    assert "pressures.csv" not in record["outputs"]


def test_without_no_pressures_the_record_has_no_new_option(capsys, tmp_path, benchmark_mesh):
    """The record of an ordinary run is unchanged: `no_pressures` appears only when it was given."""
    outdir = tmp_path / "results"
    assert _run(capsys, "infer", benchmark_mesh, "-o", outdir)[0] == cli.EXIT_OK
    assert "no_pressures" not in json.loads((outdir / "run.json").read_text())["options"]


@pytest.mark.parametrize("option", [["--pressure-method", "laplace"], ["--pressure-method=variational"],
                                    ["--base-pressure", "0"]])
def test_a_pressure_option_with_no_pressures_is_refused_not_ignored(capsys, tmp_path, benchmark_mesh, option):
    """Even the default value, typed: the command cannot honour it, so it says so (exit 2) and writes nothing."""
    outdir = tmp_path / "results"
    code, out, err = _run(capsys, "infer", benchmark_mesh, "-o", outdir, "--no-pressures", *option)
    assert code == cli.EXIT_USAGE
    name = option[0].split("=")[0]
    assert err == f"foambryo infer: error: {name} applies to the pressures: --no-pressures infers the tensions only\n"
    assert out == ""
    assert not outdir.exists()


def test_no_pressures_on_an_interface_without_a_triple_line_is_complete(capsys, tmp_path):
    """Without the pressure solve there is nothing to refuse: the undetermined tension is nan, exit 0."""
    doublet = ph.doublet_mesh(1.0, r0=1.0, n_rings=24, n_phi=48)
    sphere = _uv_sphere(label_outside=0, label_inside=5, center=np.array([0.0, 0.0, 50.0]), radius=1.0)
    mesh_path = tmp_path / "doublet_and_a_closed_sphere.rec"
    save_rec(mesh_path, *_combine(doublet, sphere))
    outdir = tmp_path / "results"
    code, _, err = _run(capsys, "infer", mesh_path, "-o", outdir, "--no-pressures")
    assert code == cli.EXIT_OK, err
    record = json.loads((outdir / "run.json").read_text())
    assert record["status"] == "complete"
    assert record["results"]["undetermined_tensions"] == 1
    assert _same_floats(read_tension_table(outdir / "tensions.csv"), infer_tensions(dcel_mesh_from_file(mesh_path)))


@pytest.mark.parametrize("diagnostics", [False, True, "spectrum"])
def test_the_library_helper_without_pressures_is_infer_tensions(benchmark_mesh, diagnostics):
    mesh = dcel_mesh_from_file(benchmark_mesh)
    outcome = infer_tensions_then_pressures(mesh, mode_tension=TensionComputationMethod.Lami,
                                            diagnostics=diagnostics, with_pressures=False)
    expected = infer_tensions(mesh, mode=TensionComputationMethod.Lami, diagnostics=diagnostics)
    tensions = expected if diagnostics is False else expected[0]
    assert outcome.status == "complete"
    assert outcome.pressures is None
    assert _same_floats(outcome.tensions, tensions)
    if diagnostics is not False:
        assert [r.system for r in outcome.records] == [r.system for r in expected[1].records]
    assert outcome.library_call == (f"infer_tensions(mesh, mean_tension=1, mode=TensionComputationMethod.Lami, "
                                    f"solver='auto', diagnostics={diagnostics!r})")


# ---------------------------------------------------------------------------------------------
# the solve table
# ---------------------------------------------------------------------------------------------
def _same_value(a: object, b: object) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or np.float64(a).tobytes() == np.float64(b).tobytes()
    return a == b


@pytest.mark.parametrize("options", [["--diagnostics=basic"], ["--diagnostics=spectrum"],
                                     ["--diagnostics=basic", "--solver=sparse", "--tension-method=lami"],
                                     ["--diagnostics=spectrum", "--no-pressures"]], ids=" ".join)
def test_the_solve_table_is_the_run_records_solves(capsys, tmp_path, benchmark_mesh, options):
    outdir = tmp_path / "results"
    code, out, err = _run(capsys, "infer", benchmark_mesh, "-o", outdir, *options)
    assert code == cli.EXIT_OK, err
    assert f"wrote {outdir / SOLVE_TABLE}" in out
    record = json.loads((outdir / "run.json").read_text())
    assert SOLVE_TABLE in record["outputs"]
    rows = read_solve_table(outdir / SOLVE_TABLE)
    assert len(rows) == len(record["solves"]) > 0
    for index, (row, solve) in enumerate(zip(rows, record["solves"], strict=True)):
        assert row["index"] == index
        for column in SOLVE_COLUMNS[1:]:
            recorded = solve[column]
            if isinstance(recorded, str) and column in ("residual_norm", "condition_number"):
                recorded = float(recorded)  # run.json writes a non-finite number as a string
            assert _same_value(row[column], recorded), column


def test_no_solve_table_without_diagnostics(capsys, tmp_path, benchmark_mesh):
    outdir = tmp_path / "results"
    assert _run(capsys, "infer", benchmark_mesh, "-o", outdir)[0] == cli.EXIT_OK
    assert not (outdir / SOLVE_TABLE).exists()


def test_overwrite_deletes_an_earlier_solve_table(capsys, tmp_path, benchmark_mesh):
    outdir = tmp_path / "results"
    assert _run(capsys, "infer", benchmark_mesh, "-o", outdir, "--diagnostics", "basic")[0] == cli.EXIT_OK
    code, _, err = _run(capsys, "infer", benchmark_mesh, "-o", outdir)
    assert code == cli.EXIT_FILE
    assert SOLVE_TABLE in err
    assert _run(capsys, "infer", benchmark_mesh, "-o", outdir, "--overwrite")[0] == cli.EXIT_OK
    assert not (outdir / SOLVE_TABLE).exists()


def test_the_solve_table_reads_back_what_was_written(tmp_path, benchmark_mesh):
    """Every column survives: empty for a value the solve has not, exact floats, `inf` as such."""
    _, _, diagnostics = infer_forces(dcel_mesh_from_file(benchmark_mesh), diagnostics=True)
    records = diagnostics.records
    records[0].condition_number = math.inf
    write_solve_table(tmp_path / SOLVE_TABLE, records)
    rows = read_solve_table(tmp_path / SOLVE_TABLE)
    assert rows[0]["condition_number"] == math.inf
    for row, record in zip(rows, records, strict=True):
        scalars = record.scalars()
        for column in SOLVE_COLUMNS[1:]:
            expected = scalars[column]
            if expected == "inf":
                expected = math.inf
            assert _same_value(row[column], expected), column
    assert any(row["lsqr_iterations"] is None for row in rows)  # a dense solve has no lsqr count


def test_the_solve_table_refuses_another_header(tmp_path):
    (tmp_path / SOLVE_TABLE).write_text("label_1,label_2,tension\n")
    with pytest.raises(ValueError, match="expected the header index,system"):
        read_solve_table(tmp_path / SOLVE_TABLE)
