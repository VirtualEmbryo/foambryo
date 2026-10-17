"""The `foambryo` command gives the library's numbers, and says what it did.

What this pins, and why:

* **`infer` writes exactly the library's numbers.** `tensions.csv` and `pressures.csv`, read back as
  64-bit floats, are bitwise `infer_forces(dcel_mesh_from_file(MESH), ...)` with the same options,
  for every tension method, every pressure method and non-default gauge, solver and diagnostics
  options. A command whose numbers drift from the library's by a format or a default is exactly
  what a command-line wrapper tends to become: the same mesh then gives two answers.
* **No default of its own.** The parser's defaults are read off the library's signatures; a test
  checks them against those signatures, so a library default that changes moves the command too.
* **`mesh` writes the library's mesh.** The arrays of the `.rec` file have the SHA-256 of
  delaunay-watershed-3d's entry point for masks with a voxel size, `construct_mesh_in_physical_units`
  (a mask with anisotropic voxels resampled to cubic voxels first), with the crop faces as the last
  rows, for masks with and without cells that reach the image box, with cubic and anisotropic voxels;
  the file is byte for byte that entry point's own `.rec` export. With cubic voxels the arrays are
  also those of the plain reconstruction scaled by the spacing, as before the resampling route. Sums
  or counts would miss a changed mesh; hashes do not. A spacing with two different short axes is
  refused; above an anisotropy of 3 the reconstruction's warning that this was not measured is relayed.
* **The run record** has every key of schema 1, with its type; the spectra file has the library's own
  layout.
* **Exit codes and messages** for a missing file, a mask without spacing, existing outputs, results
  of another mesh, and a refused pressure solve.

Synthetic masks (four cells in a ball, built here) and the benchmark mesh 000 only.
"""

from __future__ import annotations

import configparser
import contextlib
import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest
from dw3d import get_default_mesh_reconstruction_algorithm
from dw3d.io import load_rec_appendix, load_rec_with_crop_faces, save_rec

from foambryo import (
    PressureComputationMethod,
    TensionComputationMethod,
    UndeterminedTensionError,
    cli,
    dcel_mesh_from_file,
    infer_forces,
    infer_tensions,
    save_solve_diagnostics,
)
from foambryo.segmentation_masks import segmentation_mask_report
from foambryo_benchmarks import phantoms as ph
from tests.test_junction_graph import _combine, _uv_sphere

from .conftest import require_substitute_mesh

REPOSITORY = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------------------
def _four_cells_in_a_ball(n: int = 36, centre_z: float | None = None) -> np.ndarray:
    """Four Voronoi cells filling a ball of radius 14 voxels; `centre_z` near 0 makes the image cut it."""
    z, y, x = np.indices((n, n, n))
    c = (n - 1) / 2
    cz = c if centre_z is None else centre_z
    ball = (z - cz) ** 2 + (y - c) ** 2 + (x - c) ** 2 <= 14**2
    seeds = np.array([[cz - 6, c - 5, c], [cz + 6, c - 5, c + 1], [cz, c + 7, c - 6], [cz + 1, c + 4, c + 8]])
    distances = ((np.stack([z, y, x], axis=-1)[..., None, :] - seeds) ** 2).sum(axis=-1)
    return np.where(ball, distances.argmin(axis=-1) + 1, 0).astype(np.uint16)


MASKS = {
    "four cells in a ball": lambda: _four_cells_in_a_ball(),
    "four cells cut by the image box": lambda: _four_cells_in_a_ball(centre_z=6.0),
}


@pytest.fixture(scope="module")
def benchmark_mesh() -> Path:
    return require_substitute_mesh()


def _run(capsys: pytest.CaptureFixture[str], *arguments: object) -> tuple[int, str, str]:
    code = cli.main([str(argument) for argument in arguments])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _same_floats(a: dict, b: dict) -> bool:
    """The two maps have the same keys and, per key, bitwise the same float (nan equal to nan)."""
    if set(a) != set(b):
        return False
    for key in a:
        x, y = float(a[key]), float(b[key])
        if math.isnan(x) or math.isnan(y):
            if not (math.isnan(x) and math.isnan(y)):
                return False
        elif np.float64(x).tobytes() != np.float64(y).tobytes():
            return False
    return True


def _array_sha256(array: np.ndarray, dtype: type) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(array, dtype=dtype)).tobytes()).hexdigest()


# ---------------------------------------------------------------------------------------------
# defaults and names
# ---------------------------------------------------------------------------------------------
def test_every_default_of_the_command_is_the_librarys():
    parser = cli.build_parser()
    infer = parser.parse_args(["infer", "m.rec", "-o", "out"])
    library = inspect.signature(infer_forces).parameters
    assert cli.TENSION_METHODS[infer.tension_method] is library["mode_tension"].default
    assert cli.PRESSURE_METHODS[infer.pressure_method] is library["mode_pressure"].default
    assert infer.mean_tension == library["mean_tension"].default
    assert infer.base_pressure == library["base_pressure"].default
    assert infer.solver == library["solver"].default
    assert cli.DIAGNOSTICS[infer.diagnostics] is library["diagnostics"].default
    mesh = parser.parse_args(["mesh", "m.npy", "-o", "m.rec"])
    reconstruction = inspect.signature(get_default_mesh_reconstruction_algorithm).parameters
    assert mesh.min_distance == reconstruction["min_distance"].default
    assert mesh.spacing is None  # no default spacing, on purpose


def test_every_method_has_one_command_line_name():
    assert sorted(cli.TENSION_METHODS.values(), key=lambda m: m.value) == list(TensionComputationMethod)
    assert sorted(cli.PRESSURE_METHODS.values(), key=lambda m: m.value) == list(PressureComputationMethod)
    assert cli.option_name(TensionComputationMethod.YoungDupre) == "young-dupre"
    assert cli.option_name(TensionComputationMethod.ForceBalanceMAP) == "force-balance-map"
    assert cli.option_name(PressureComputationMethod.WeightedLaplace) == "weighted-laplace"


def test_the_console_entry_point_and_the_cli_extra_are_declared():
    config = configparser.ConfigParser()
    config.read(REPOSITORY / "setup.cfg")
    scripts = [line.strip() for line in config["options.entry_points"]["console_scripts"].strip().splitlines()]
    assert scripts == ["foambryo = foambryo.cli:main"]
    extra = [line.strip() for line in config["options.extras_require"]["cli"].strip().splitlines()]
    assert [requirement.split(">")[0] for requirement in extra] == ["tifffile"]


def test_help_and_usage_exit_codes(capsys):
    for arguments in ([], ["mesh"], ["infer"], ["view"], ["infer", "m.rec"]):
        assert _run(capsys, *arguments)[0] == cli.EXIT_USAGE
    for arguments in (["--help"], ["mesh", "--help"], ["infer", "--help"], ["view", "--help"], ["--version"]):
        assert _run(capsys, *arguments)[0] == cli.EXIT_OK


# ---------------------------------------------------------------------------------------------
# infer: the library's numbers
# ---------------------------------------------------------------------------------------------
INFER_CASES = [
    *[[f"--tension-method={name}"] for name in cli.TENSION_METHODS],
    *[[f"--pressure-method={name}"] for name in cli.PRESSURE_METHODS],
    ["--mean-tension=2.5", "--base-pressure=-0.75"],
    ["--solver=dense"],
    ["--solver=sparse"],
    ["--diagnostics=basic"],
    ["--diagnostics=spectrum", "--pressure-method=weighted-laplace"],
]


def _library_options(options: list[str]) -> dict:
    args = cli.build_parser().parse_args(["infer", "m.rec", "-o", "out", *options])
    return {
        "mean_tension": args.mean_tension,
        "base_pressure": args.base_pressure,
        "mode_tension": cli.TENSION_METHODS[args.tension_method],
        "mode_pressure": cli.PRESSURE_METHODS[args.pressure_method],
        "solver": args.solver,
        "diagnostics": cli.DIAGNOSTICS[args.diagnostics],
    }


@pytest.mark.parametrize("options", INFER_CASES, ids=" ".join)
def test_infer_tables_are_bitwise_the_library_call(capsys, tmp_path, benchmark_mesh, options):
    outdir = tmp_path / "results"
    code, _, err = _run(capsys, "infer", benchmark_mesh, "-o", outdir, *options)
    kwargs = _library_options(options)
    record = json.loads((outdir / "run.json").read_text())
    refusal = None
    try:
        expected = infer_forces(dcel_mesh_from_file(benchmark_mesh), **kwargs)
    except UndeterminedTensionError:
        # The library refuses the pressures: so does the command, after writing the tensions.
        assert code == cli.EXIT_REFUSED, err
        assert record["status"] == "pressures refused"
        assert not (outdir / "pressures.csv").exists()
        kwargs.pop("base_pressure")
        kwargs.pop("mode_pressure")
        tensions = infer_tensions(dcel_mesh_from_file(benchmark_mesh), mode=kwargs.pop("mode_tension"), **kwargs)
        tensions = tensions if kwargs["diagnostics"] is False else tensions[0]
        assert _same_floats(cli.tables.read_tension_table(outdir / "tensions.csv"), tensions)
        return
    except ValueError as error:
        refusal = str(error)
    if refusal is not None:  # the library refused these options: so does the command, saying why
        assert code == cli.EXIT_REFUSED, err
        assert record["status"] == "failed"
        assert record["error"] == refusal
        return
    assert code == cli.EXIT_OK, err
    assert record["status"] == "complete"
    assert _same_floats(cli.tables.read_tension_table(outdir / "tensions.csv"), expected[0])
    assert _same_floats(cli.tables.read_pressure_table(outdir / "pressures.csv"), expected[1])
    if kwargs["diagnostics"] is not False:
        assert len(record["solves"]) == len(expected[2].records)
        assert [s["system"] for s in record["solves"]] == [r.system for r in expected[2].records]


def test_sparse_solver_on_a_method_without_a_sparse_path_is_refused(capsys, tmp_path, benchmark_mesh):
    """`--solver sparse` with the variational tensions: the library's ValueError, exit 4, run.json says so."""
    outdir = tmp_path / "results"
    code, _, err = _run(capsys, "infer", benchmark_mesh, "-o", outdir, "--tension-method", "variational",
                        "--solver", "sparse")
    assert code == cli.EXIT_REFUSED
    assert err.startswith("foambryo infer: error: failed: ")
    record = json.loads((outdir / "run.json").read_text())
    assert record["status"] == "failed"
    assert record["outputs"] == ["run.json"]


def test_spectra_file_is_the_librarys_layout(capsys, tmp_path, benchmark_mesh):
    outdir = tmp_path / "results"
    assert _run(capsys, "infer", benchmark_mesh, "-o", outdir, "--diagnostics", "spectrum")[0] == cli.EXIT_OK
    _, _, diagnostics = infer_forces(dcel_mesh_from_file(benchmark_mesh), diagnostics="spectrum")
    _, npz = save_solve_diagnostics(diagnostics, tmp_path / "library")
    with np.load(outdir / "spectra.npz") as written, np.load(npz) as library:
        assert sorted(written.files) == sorted(library.files)
        assert written.files  # a dense solve records every singular value
        for key in library.files:
            assert np.array_equal(written[key], library[key])


RUN_RECORD_KEYS = {
    "schema": int, "command": str, "arguments": list, "date": str, "elapsed_seconds": float, "versions": dict,
    "input": dict, "options": dict, "library_call": str, "mesh": dict, "results": dict, "units": dict,
    "status": str, "error": type(None), "warnings": list, "solves": type(None), "outputs": list,
}


def test_run_record_has_every_schema_1_key(capsys, tmp_path, benchmark_mesh):
    outdir = tmp_path / "results"
    assert _run(capsys, "infer", benchmark_mesh, "-o", outdir)[0] == cli.EXIT_OK
    record = json.loads((outdir / "run.json").read_text())
    assert set(record) == set(RUN_RECORD_KEYS)
    for key, kind in RUN_RECORD_KEYS.items():
        assert isinstance(record[key], kind), key
    assert record["schema"] == 1
    assert record["command"] == "foambryo infer"
    assert record["arguments"] == ["infer", str(benchmark_mesh), "-o", str(outdir)]
    assert set(record["versions"]) == {"python", "foambryo", "delaunay-watershed-3d", "numpy", "scipy"}
    assert record["input"]["sha256"] == hashlib.sha256(benchmark_mesh.read_bytes()).hexdigest()
    assert record["input"]["bytes"] == benchmark_mesh.stat().st_size
    assert set(record["input"]) == {"path", "sha256", "bytes", "metadata"}
    assert record["options"] == {"tension_method": "young-dupre", "pressure_method": "variational",
                                 "mean_tension": 1.0, "base_pressure": 0.0, "solver": "auto", "diagnostics": "none"}
    assert record["library_call"] == (
        "infer_forces(mesh, mean_tension=1.0, base_pressure=0.0, mode_tension=TensionComputationMethod.YoungDupre, "
        "mode_pressure=PressureComputationMethod.Variational, solver='auto', diagnostics=False)"
    )
    assert set(record["mesh"]) == {"points", "triangles", "materials", "crop_faces", "cut_cells"}
    assert set(record["results"]) == {"tensions", "undetermined_tensions", "pressures", "undetermined_pressures"}
    assert set(record["units"]) == {"tension", "pressure"}
    assert record["status"] == "complete"
    assert record["outputs"] == ["tensions.csv", "pressures.csv", "run.json"]


def test_existing_outputs_need_overwrite_and_overwrite_leaves_no_stale_file(capsys, tmp_path, benchmark_mesh):
    outdir = tmp_path / "results"
    assert _run(capsys, "infer", benchmark_mesh, "-o", outdir, "--diagnostics", "spectrum")[0] == cli.EXIT_OK
    (outdir / "notes.txt").write_text("mine")
    code, _, err = _run(capsys, "infer", benchmark_mesh, "-o", outdir)
    assert code == cli.EXIT_FILE
    assert err == (f"foambryo infer: error: {outdir} already holds tensions.csv, pressures.csv, run.json, "
                   "spectra.npz, solves.csv: pass --overwrite to replace them\n")
    assert _run(capsys, "infer", benchmark_mesh, "-o", outdir, "--overwrite")[0] == cli.EXIT_OK
    assert not (outdir / "spectra.npz").exists()  # the earlier run's file did not survive
    assert (outdir / "notes.txt").read_text() == "mine"


def test_a_closed_interface_refuses_the_pressures_and_keeps_the_tensions(capsys, tmp_path):
    """An interface with no triple line has the tension nan; the library refuses the pressures (exit 4)."""
    doublet = ph.doublet_mesh(1.0, r0=1.0, n_rings=24, n_phi=48)
    sphere = _uv_sphere(label_outside=0, label_inside=5, center=np.array([0.0, 0.0, 50.0]), radius=1.0)
    points, triangles, labels = _combine(doublet, sphere)
    mesh_path = tmp_path / "doublet_and_a_closed_sphere.rec"
    save_rec(mesh_path, points, triangles, labels)
    outdir = tmp_path / "results"
    code, _, err = _run(capsys, "infer", mesh_path, "-o", outdir)
    assert code == cli.EXIT_REFUSED
    assert err.splitlines()[-1].startswith("foambryo infer: error: pressures refused: ")
    assert "warning: JunctionGraphGaugeWarning" in err
    with pytest.raises(UndeterminedTensionError):
        infer_forces(dcel_mesh_from_file(mesh_path))
    tensions = cli.tables.read_tension_table(outdir / "tensions.csv")
    assert math.isnan(tensions[(0, 5)])
    assert _same_floats(tensions, infer_tensions(dcel_mesh_from_file(mesh_path)))
    assert not (outdir / "pressures.csv").exists()
    record = json.loads((outdir / "run.json").read_text())
    assert record["status"] == "pressures refused"
    assert record["results"]["undetermined_tensions"] == 1
    assert record["results"]["pressures"] is None
    assert "JunctionGraphGaugeWarning" in {w["category"] for w in record["warnings"]}
    assert record["outputs"] == ["tensions.csv", "run.json"]


# ---------------------------------------------------------------------------------------------
# mesh: the library's mesh
# ---------------------------------------------------------------------------------------------
SPACINGS = {"unit voxels": (1.0, 1.0, 1.0), "cubic voxels of 0.5": (0.5, 0.5, 0.5), "z twice as long": (1.0, 0.5, 0.5)}


@pytest.mark.parametrize("spacing_name", list(SPACINGS))
@pytest.mark.parametrize("mask_name", list(MASKS))
def test_mesh_file_holds_the_library_reconstruction(capsys, tmp_path, mask_name, spacing_name):
    spacing = SPACINGS[spacing_name]
    mask = MASKS[mask_name]()
    mask_path, mesh_path = tmp_path / "mask.npy", tmp_path / "mesh.rec"
    np.save(mask_path, mask)
    code, out, err = _run(capsys, "mesh", mask_path, "-o", mesh_path, "--spacing", *spacing)
    assert code == cli.EXIT_OK, err
    anisotropic = max(spacing) != min(spacing)
    assert ("AnisotropicVoxelSpacingWarning" in err) == anisotropic
    assert "UnmeasuredAnisotropyWarning" not in err

    algorithm = get_default_mesh_reconstruction_algorithm(3, False)
    points, triangles, labels = algorithm.construct_mesh_in_physical_units(mask, spacing, spacing_unit="um")
    report = algorithm.crop_face_report
    crop = report.crop_faces if report.meets_the_image_box else np.zeros(len(triangles), dtype=bool)
    order = np.concatenate([np.flatnonzero(~crop), np.flatnonzero(crop)])

    read_points, read_triangles, read_labels, read_crop = load_rec_with_crop_faces(mesh_path)
    assert _array_sha256(read_points, np.float64) == _array_sha256(points, np.float64)
    assert _array_sha256(read_triangles, np.int64) == _array_sha256(np.asarray(triangles)[order], np.int64)
    assert _array_sha256(read_labels, np.int64) == _array_sha256(np.asarray(labels)[order], np.int64)
    if report.meets_the_image_box:
        assert _array_sha256(read_crop, np.bool_) == _array_sha256(crop[order], np.bool_)
    else:  # recorded exactly when the reconstruction's own export records it
        assert read_crop is None
    assert (mask_name == "four cells cut by the image box") == bool(crop.any())
    # the whole file is the entry point's own export
    algorithm.save_to_rec_mesh(tmp_path / "library.rec")
    assert mesh_path.read_bytes() == (tmp_path / "library.rec").read_bytes()
    if not anisotropic:  # cubic voxels: the plain reconstruction, scaled, bitwise as before resampling existed
        plain = get_default_mesh_reconstruction_algorithm(3, False)
        plain_points = np.asarray(plain.construct_mesh_from_segmentation_mask(mask)[0], dtype=np.float64)
        assert _array_sha256(points, np.float64) == _array_sha256(plain_points * np.asarray(spacing), np.float64)

    appendix = load_rec_appendix(mesh_path)
    assert appendix["coordinate_frame"] == "physical"
    assert appendix["axis_order"] == "zyx"
    assert appendix["spacing"] == spacing
    assert appendix["spacing_unit"] == "um"
    assert appendix["source_shape"] == mask.shape
    assert appendix["resampling"] == "shape"
    assert float(appendix["resampled_spacing"]) == min(spacing)  # dw3d's reader returns this key as text
    assert ("crop_faces" in appendix) == report.meets_the_image_box
    n_cells = np.count_nonzero(np.unique(labels))
    assert out == (f"{segmentation_mask_report(mask).summary()}\n"
                   f"wrote {mesh_path}: {len(points)} points, {len(triangles)} triangles, {n_cells} cells "
                   f"({len(report.cut_cells) if report.meets_the_image_box else 0} cut by the image boundary)\n")
    # and foambryo reads it, crop faces and source voxel size included
    mesh = dcel_mesh_from_file(mesh_path)
    assert int(mesh.crop_faces.sum()) == int(crop.sum())
    assert mesh.source_spacing == spacing


def test_mesh_says_the_mask_is_resampled_and_warns_above_the_measured_anisotropy(capsys, tmp_path):
    np.save(tmp_path / "mask.npy", MASKS["four cells in a ball"]())
    code, _, err = _run(capsys, "mesh", tmp_path / "mask.npy", "-o", tmp_path / "k2.rec", "--spacing", 0.6, 0.3, 0.3)
    assert code == cli.EXIT_OK
    assert err == ("foambryo mesh: warning: AnisotropicVoxelSpacingWarning: the voxels are 2 times longer along "
                   "axis 0 (z) than along the other two: the mask is resampled to cubic voxels of 0.3 um by "
                   "shape-based interpolation before meshing (2 times the voxels; --min-distance counts cubic "
                   "voxels)\n")
    code, _, err = _run(capsys, "mesh", tmp_path / "mask.npy", "-o", tmp_path / "k3.5.rec", "--spacing", 1, 1, 3.5)
    assert code == cli.EXIT_OK
    relayed = {line.split(": ")[2]: line for line in err.splitlines()}
    assert relayed["AnisotropicVoxelSpacingWarning"].startswith(
        "foambryo mesh: warning: AnisotropicVoxelSpacingWarning: the voxels are 3.5 times longer along axis 2 (x)")
    assert relayed["UnmeasuredAnisotropyWarning"].startswith(
        "foambryo mesh: warning: UnmeasuredAnisotropyWarning: voxels 3.5 times longer along axis 2")
    assert "measured up to a ratio of 3 only" in relayed["UnmeasuredAnisotropyWarning"]
    assert load_rec_appendix(tmp_path / "k3.5.rec")["spacing"] == (1.0, 1.0, 3.5)


def test_a_spacing_with_two_different_short_axes_is_refused(capsys, tmp_path):
    mask = MASKS["four cells in a ball"]()
    np.save(tmp_path / "mask.npy", mask)
    code, out, err = _run(capsys, "mesh", tmp_path / "mask.npy", "-o", tmp_path / "m.rec", "--spacing", 2, 1, 0.5)
    assert (code, out) == (cli.EXIT_USAGE, "")
    assert err.startswith("foambryo mesh: error: --spacing: spacing (2.0, 1.0, 0.5) has two different fine axes")
    np.savez(tmp_path / "mask.npz", mask=mask, spacing=np.array([2.0, 1.0, 0.5]))
    code, out, err = _run(capsys, "mesh", tmp_path / "mask.npz", "-o", tmp_path / "m.rec")
    assert (code, out) == (cli.EXIT_FILE, "")
    assert err.startswith("foambryo mesh: error: mask.npz: spacing (2.0, 1.0, 0.5) has two different fine axes")
    assert not (tmp_path / "m.rec").exists()


def test_mesh_reads_the_spacing_an_npz_records_and_a_tiff(capsys, tmp_path):
    mask = MASKS["four cells in a ball"]()
    np.save(tmp_path / "mask.npy", mask)
    assert _run(capsys, "mesh", tmp_path / "mask.npy", "-o", tmp_path / "from_npy.rec", "--spacing", 2, 1, 1,
                "--spacing-unit", "nm")[0] == cli.EXIT_OK
    np.savez(tmp_path / "mask.npz", mask=mask, spacing=np.array([2.0, 1.0, 1.0]), spacing_unit=np.array("nm"))
    assert _run(capsys, "mesh", tmp_path / "mask.npz", "-o", tmp_path / "from_npz.rec")[0] == cli.EXIT_OK
    assert (tmp_path / "from_npz.rec").read_bytes() == (tmp_path / "from_npy.rec").read_bytes()
    # the command line wins over the file, and says so
    code, _, err = _run(capsys, "mesh", tmp_path / "mask.npz", "-o", tmp_path / "override.rec", "--spacing", 1, 1, 1)
    assert code == cli.EXIT_OK
    assert "--spacing 1 1 1 replaces the spacing mask.npz records, 2 1 1" in err
    tifffile = pytest.importorskip("tifffile")
    tifffile.imwrite(tmp_path / "mask.tif", mask)
    assert _run(capsys, "mesh", tmp_path / "mask.tif", "-o", tmp_path / "from_tif.rec", "--spacing", 2, 1, 1,
                "--spacing-unit", "nm")[0] == cli.EXIT_OK
    assert (tmp_path / "from_tif.rec").read_bytes() == (tmp_path / "from_npy.rec").read_bytes()


# ---------------------------------------------------------------------------------------------
# exit codes and messages
# ---------------------------------------------------------------------------------------------
def test_missing_inputs_exit_3_with_their_name(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _run(capsys, "mesh", "missing.npy", "-o", "m.rec", "--spacing", 1, 1, 1) == (
        cli.EXIT_FILE, "", "foambryo mesh: error: no such file: 'missing.npy'\n")
    assert _run(capsys, "infer", "missing.rec", "-o", "out") == (
        cli.EXIT_FILE, "", "foambryo infer: error: no such file: 'missing.rec'\n")
    assert _run(capsys, "view", "missing.rec", "--colour-bars", "b.png", "--no-show") == (
        cli.EXIT_FILE, "", "foambryo view: error: no such file: 'missing.rec'\n")
    assert not (tmp_path / "out").exists()


def test_a_mask_without_spacing_exits_2_and_says_how_to_give_it(capsys, tmp_path):
    np.save(tmp_path / "mask.npy", MASKS["four cells in a ball"]())
    code, out, err = _run(capsys, "mesh", tmp_path / "mask.npy", "-o", tmp_path / "m.rec")
    assert (code, out) == (cli.EXIT_USAGE, "")
    assert err == ("foambryo mesh: error: mask.npy records no voxel spacing: give it with --spacing DZ DY DX "
                   "(for example --spacing 1 1 1 to keep voxel units)\n")
    assert not (tmp_path / "m.rec").exists()


@pytest.mark.parametrize(("array", "message"), [
    (np.zeros((4, 4)), "mask.npy holds a 2-D array; a mask is 3-D"),
    (np.zeros((4, 4, 4)), "mask.npy holds float64 values; a mask holds integer labels"),
    (-np.ones((4, 4, 4), dtype=np.int32), "mask.npy holds negative labels; labels are 0 (medium) or positive"),
])
def test_a_mask_that_is_not_labels_exits_3(capsys, tmp_path, array, message):
    np.save(tmp_path / "mask.npy", array)
    assert _run(capsys, "mesh", tmp_path / "mask.npy", "-o", tmp_path / "m.rec", "--spacing", 1, 1, 1) == (
        cli.EXIT_FILE, "", f"foambryo mesh: error: {message}\n")


def test_mesh_refuses_another_suffix_and_an_existing_file(capsys, tmp_path):
    np.save(tmp_path / "mask.npy", MASKS["four cells in a ball"]())
    code, _, _ = _run(capsys, "mesh", tmp_path / "mask.npy", "-o", tmp_path / "m.vtk", "--spacing", 1, 1, 1)
    assert code == cli.EXIT_USAGE
    (tmp_path / "m.rec").write_text("mine")
    code, _, err = _run(capsys, "mesh", tmp_path / "mask.npy", "-o", tmp_path / "m.rec", "--spacing", 1, 1, 1)
    assert code == cli.EXIT_FILE
    assert err == f"foambryo mesh: error: {tmp_path / 'm.rec'} exists: pass --overwrite to replace it\n"
    assert (tmp_path / "m.rec").read_text() == "mine"


def test_bad_option_values_exit_2(capsys, tmp_path):
    np.save(tmp_path / "mask.npy", MASKS["four cells in a ball"]())
    for options in (["--spacing", 1, 0, 1], ["--spacing", 1, "nan", 1], ["--spacing", 1, 1, 1, "--spacing-unit", "µm"],
                    ["--spacing", 1, 1, 1, "--min-distance", 0]):
        assert _run(capsys, "mesh", tmp_path / "mask.npy", "-o", tmp_path / "m.rec", *options)[0] == cli.EXIT_USAGE
    assert _run(capsys, "infer", "m.rec", "-o", "out", "--tension-method", "youngdupre")[0] == cli.EXIT_USAGE
    assert _run(capsys, "infer", "m.rec", "-o", "out", "--mean-tension", "inf")[0] == cli.EXIT_USAGE


# ---------------------------------------------------------------------------------------------
# view
# ---------------------------------------------------------------------------------------------
def test_view_no_show_writes_the_colour_bars_without_polyscope(capsys, tmp_path, benchmark_mesh, monkeypatch):
    outdir = tmp_path / "results"
    assert _run(capsys, "infer", benchmark_mesh, "-o", outdir)[0] == cli.EXIT_OK
    monkeypatch.setitem(__import__("sys").modules, "polyscope", None)  # any import of Polyscope now fails
    bars = tmp_path / "bars.png"
    code, out, err = _run(capsys, "view", benchmark_mesh, "--results", outdir, "--colour-bars", bars, "--no-show")
    assert code == cli.EXIT_OK, err
    assert bars.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert out.startswith(f"wrote {bars}: tension ")


def test_view_refuses_results_of_another_mesh(capsys, tmp_path, benchmark_mesh):
    outdir = tmp_path / "results"
    assert _run(capsys, "infer", benchmark_mesh, "-o", outdir)[0] == cli.EXIT_OK
    other = tmp_path / "other.rec"
    other.write_bytes(benchmark_mesh.read_bytes() + b"\n")
    code, _, err = _run(capsys, "view", other, "--results", outdir, "--colour-bars", tmp_path / "b.png", "--no-show")
    assert code == cli.EXIT_FILE
    assert "was computed from another mesh" in err
    assert not (tmp_path / "b.png").exists()


def test_view_no_show_needs_colour_bars(capsys, benchmark_mesh):
    assert _run(capsys, "view", benchmark_mesh, "--no-show") == (
        cli.EXIT_USAGE, "", "foambryo view: error: --no-show needs --colour-bars: there would be nothing to do\n")


def test_view_opens_the_viewer_on_infers_results(capsys, tmp_path, benchmark_mesh, monkeypatch):
    """The window itself, on Polyscope's headless mock backend (skipped without Polyscope)."""
    ps = pytest.importorskip("polyscope")
    real_init, real_show = ps.init, ps.show
    try:
        real_init("openGL_mock")
    except RuntimeError as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"Polyscope's mock backend is unavailable: {error}")
    monkeypatch.setattr(ps, "init", lambda *_args, **_kwargs: real_init("openGL_mock"))
    monkeypatch.setattr(ps, "show", lambda *_args, **_kwargs: real_show(forFrames=1))
    outdir = tmp_path / "results"
    assert _run(capsys, "infer", benchmark_mesh, "-o", outdir)[0] == cli.EXIT_OK
    code, out, err = _run(capsys, "view", benchmark_mesh, "--results", outdir)
    ps.remove_all_structures()
    assert code == cli.EXIT_OK, err
    assert "Extremas of the Surface Tensions plotted" in out


# ---------------------------------------------------------------------------------------------
# foambryo infer-over-seeds
# ---------------------------------------------------------------------------------------------
def test_every_default_of_infer_over_seeds_is_the_librarys():
    from foambryo.seed_ensembles import DEFAULT_ENSEMBLE_SEEDS, infer_tensions_over_seeds

    args = cli.build_parser().parse_args(["infer-over-seeds", "m.npy", "-o", "out"])
    library = inspect.signature(infer_tensions_over_seeds).parameters
    assert list(range(args.seeds)) == list(DEFAULT_ENSEMBLE_SEEDS) == list(library["seeds"].default)
    assert args.combine == library["combine"].default
    assert args.angle_reading == library["angle_reading"].default
    assert cli.TENSION_METHODS[args.tension_method] is library["mode"].default
    assert args.max_workers == library["max_workers"].default
    assert args.min_distance == library["min_distance"].default
    assert args.spacing is None  # no default spacing, on purpose


@pytest.mark.parametrize(
    ("spacing", "options"),
    [
        ((1.0, 1.0, 1.0), []),
        ((2.0, 1.0, 1.0), []),
        ((2.0, 1.0, 1.0), ["--combine", "angles", "--angle-reading", "quadric"]),
    ],
    ids=["cubic voxels, tensions", "anisotropic voxels, tensions", "anisotropic voxels, angles with quadric"],
)
def test_infer_over_seeds_tables_are_bitwise_the_library_call(capsys, tmp_path, spacing, options):
    from foambryo.seed_ensembles import SeedEnsembleExperimentalWarning, infer_tensions_over_seeds

    mask = _four_cells_in_a_ball()
    np.save(tmp_path / "mask.npy", mask)
    code, out, _err = _run(capsys, "infer-over-seeds", tmp_path / "mask.npy", "-o", tmp_path / "out", "--seeds", 2,
                           "--spacing", *spacing, *options)
    assert code == 0, out
    combine = options[1] if options else "tensions"
    reading = options[3] if options else "chord"
    with pytest.warns(SeedEnsembleExperimentalWarning) if combine == "angles" else contextlib.nullcontext():
        library = infer_tensions_over_seeds(mask, range(2), spacing=spacing, combine=combine, angle_reading=reading)
    table = cli.tables.read_ensemble_tension_table(tmp_path / "out" / "tensions.csv")
    assert set(table) == set(library.tensions)
    assert _same_floats({k: v[0] for k, v in table.items()}, library.tensions)
    assert _same_floats({k: v[1] for k, v in table.items()}, library.tension_seed_sd)
    assert {k: v[2] for k, v in table.items()} == library.n_seeds_found
    assert {k: v[3] for k, v in table.items()} == library.found_by_fewer_than_half_the_seeds
    record = json.loads((tmp_path / "out" / "run.json").read_text())
    assert record["command"] == "foambryo infer-over-seeds"
    assert record["seeds"] == [0, 1]
    assert record["options"]["combine"] == combine
    assert record["input"]["spacing"] == list(spacing)
    assert record["status"] == "complete"
    assert record["outputs"] == ["tensions.csv", "tensions_per_seed.csv", "run.json"]
    assert [r["seed"] for r in record["reconstructions"]] == [0, 1]
    per_seed = cli.tables.read_per_seed_tension_table(tmp_path / "out" / "tensions_per_seed.csv")
    assert list(per_seed) == [0, 1]
    for seed in (0, 1):
        assert _same_floats(per_seed[seed], library.per_seed_tensions[seed])
    assert not (tmp_path / "out" / "pressures.csv").exists()


def test_infer_over_seeds_refusal_writes_run_json_and_exits_4(capsys, tmp_path):
    np.save(tmp_path / "mask.npy", _four_cells_in_a_ball())
    code, _out, err = _run(capsys, "infer-over-seeds", tmp_path / "mask.npy", "-o", tmp_path / "out", "--seeds", 1,
                           "--spacing", 1, 1, 1, "--combine", "angles", "--tension-method", "force-balance")
    assert code == cli.EXIT_REFUSED
    assert "failed" in err
    assert "combine='angles'" in err
    record = json.loads((tmp_path / "out" / "run.json").read_text())
    assert record["status"] == "failed"
    assert record["outputs"] == ["run.json"]
    assert not (tmp_path / "out" / "tensions.csv").exists()
    assert not (tmp_path / "out" / "tensions_per_seed.csv").exists()


def test_infer_over_seeds_needs_overwrite_and_a_spacing(capsys, tmp_path):
    np.save(tmp_path / "mask.npy", _four_cells_in_a_ball())
    code, _out, err = _run(capsys, "infer-over-seeds", tmp_path / "mask.npy", "-o", tmp_path / "out", "--seeds", 1)
    assert code == cli.EXIT_USAGE
    assert "--spacing" in err
    arguments = ["infer-over-seeds", tmp_path / "mask.npy", "-o", tmp_path / "out", "--seeds", 1, "--spacing", 1, 1, 1]
    assert _run(capsys, *arguments)[0] == 0
    code, _out, err = _run(capsys, *arguments)
    assert code == cli.EXIT_FILE
    assert "--overwrite" in err
    assert _run(capsys, *arguments, "--overwrite")[0] == 0
