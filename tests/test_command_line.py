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
* **The run record** has every key of schema 2, with its type; the spectra file has the library's own
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
import warnings
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
    # The pressure method's default is per dimension (None: variational in 3D, laplace in 2D), in both.
    assert infer.pressure_method is library["mode_pressure"].default is None
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
        "mode_pressure": None if args.pressure_method is None else cli.PRESSURE_METHODS[args.pressure_method],
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
    "schema": int, "dimension": int, "command": str, "arguments": list, "date": str, "elapsed_seconds": float, "versions": dict,
    "input": dict, "options": dict, "library_call": str, "mesh": dict, "results": dict, "units": dict,
    "status": str, "error": type(None), "warnings": list, "solves": type(None), "outputs": list,
}


def test_run_record_has_every_schema_2_key(capsys, tmp_path, benchmark_mesh):
    outdir = tmp_path / "results"
    assert _run(capsys, "infer", benchmark_mesh, "-o", outdir)[0] == cli.EXIT_OK
    record = json.loads((outdir / "run.json").read_text())
    assert set(record) == set(RUN_RECORD_KEYS)
    for key, kind in RUN_RECORD_KEYS.items():
        assert isinstance(record[key], kind), key
    assert record["schema"] == 2 and record["dimension"] == 3
    assert record["command"] == "foambryo infer"
    assert record["arguments"] == ["infer", str(benchmark_mesh), "-o", str(outdir)]
    assert set(record["versions"]) == {"python", "foambryo", "delaunay-watershed-3d", "delaunay-watershed-2d", "iorec",
                                       "numpy", "scipy"}
    assert record["input"]["sha256"] == hashlib.sha256(benchmark_mesh.read_bytes()).hexdigest()
    assert record["input"]["bytes"] == benchmark_mesh.stat().st_size
    assert set(record["input"]) == {"path", "sha256", "bytes", "metadata"}
    assert record["options"] == {"tension_method": "young-dupre", "pressure_method": "variational",
                                 "mean_tension": 1.0, "base_pressure": 0.0, "solver": "auto", "diagnostics": "none",
                                 "media": None}
    assert record["library_call"] == (
        "infer_forces(mesh, mean_tension=1.0, base_pressure=0.0, mode_tension=TensionComputationMethod.YoungDupre, "
        "mode_pressure=PressureComputationMethod.Variational, solver='auto', diagnostics=False)"
    )
    assert set(record["mesh"]) == {"points", "triangles", "materials", "cells", "media", "unresolved", "crop_faces",
                                   "cut_cells"}
    assert record["mesh"]["unresolved"] == []
    assert record["mesh"]["media"] == [0]
    assert record["mesh"]["cells"] == record["mesh"]["materials"] - 1
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

#: (mask, spacing) pairs on which delaunay-watershed-3d's reconstruction loses a cell, or keeps only a fragment of it.
CELL_LOST_BY_THE_WATERSHED = {
    ("four cells cut by the image box", "unit voxels"): "dw3d's watershed loses cell 1 of the cut mask (0.8.0 keeps 4 "
    "triangles of it, 0.010 of its volume)",
    ("four cells cut by the image box", "cubic voxels of 0.5"): "the same mesh as on unit voxels, scaled",
}


def _volumes_over_voxels_inside_the_box(mask, spacing, points, triangles, labels) -> dict[int, float]:
    """Each cell's mesh volume over its voxel volume inside the box of voxel centres (where a mesh can reach)."""
    points, triangles, labels = np.asarray(points, float), np.asarray(triangles, np.int64), np.asarray(labels, np.int64)
    det = np.einsum("ij,ij->i", points[triangles[:, 0]], np.cross(points[triangles[:, 1]], points[triangles[:, 2]])) / 6
    out = {}
    for cell in (int(c) for c in np.unique(mask) if c):
        volume = float(det[labels[:, 0] == cell].sum() - det[labels[:, 1] == cell].sum())
        index = np.argwhere(mask == cell)
        weight = np.ones(len(index))
        for axis, n in enumerate(mask.shape):
            weight *= np.where((index[:, axis] == 0) | (index[:, axis] == n - 1), 0.5, 1.0)
        out[cell] = volume / (weight.sum() * float(np.prod(spacing)))
    return out


@pytest.mark.parametrize("spacing_name", list(SPACINGS))
@pytest.mark.parametrize("mask_name", list(MASKS))
def test_every_cell_of_the_mask_is_meshed(request, mask_name, spacing_name):
    """Every cell of the mask is in the mesh `foambryo mesh` writes, with at least half its volume, cut or not.

    The volume is read against the cell's voxels inside the box of voxel centres, which is where the mesh can reach (a
    cut cell's voxels on the image box count for half). On cubic voxels, dw3d 0.7.1's seeded watershed loses cell 1 of
    the cut mask at min_distance 3 (it keeps only its seed tetrahedron, which mesh surgery relabels); dw3d 0.7.0 loses
    it on macOS arm64 and kept 8 of its triangles in one Linux run. dw3d 0.8.0 pads the mask's four free faces, 4 voxels
    from the ball, and keeps 4 triangles of cell 1: 0.010 of its volume, while cell 2 holds 1.25 times its own and cell
    4 0.42. On the uncut ball with z twice as long, dw3d 0.8.0's padded mesh keeps 0.064 of cell 3's volume; without
    padding the same mesh loses that cell on 2 of 20 dither seeds. The expected failures are not strict, because the
    outcome depends on the installed delaunay-watershed-3d and the machine.
    """
    if (mask_name, spacing_name) in CELL_LOST_BY_THE_WATERSHED:
        reason = CELL_LOST_BY_THE_WATERSHED[(mask_name, spacing_name)]
        request.applymarker(pytest.mark.xfail(strict=False, reason=reason))
    mask = MASKS[mask_name]()
    algorithm = get_default_mesh_reconstruction_algorithm(3, False)
    spacing = SPACINGS[spacing_name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        points, triangles, labels = algorithm.construct_mesh_in_physical_units(mask, spacing, spacing_unit="um")
    cells = {int(label) for label in np.unique(mask)} - {0}
    assert cells <= {int(label) for label in np.unique(labels)}
    ratios = _volumes_over_voxels_inside_the_box(mask, spacing, points, triangles, labels)
    assert all(ratio >= 0.5 for ratio in ratios.values()), ratios


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
    # a label of the mask absent from the mesh is named (dw3d's `unresolved`, every cause since its 0.8.0 line)
    unresolved = sorted((u.label, u.voxels) for u in getattr(algorithm.enclosed_medium_report, "unresolved", ()))
    absent = ""
    if unresolved:
        names = ", ".join(f"{label}: {voxels} voxels" for label, voxels in unresolved)
        absent = (f"; {len(unresolved)} label{'s' if len(unresolved) != 1 else ''} of the mask unresolved, absent "
                  f"from the mesh ({names})")
    assert out == (f"{segmentation_mask_report(mask).summary()}\n"
                   f"wrote {mesh_path}: {len(points)} points, {len(triangles)} triangles, {n_cells} cells "
                   f"({len(report.cut_cells) if report.meets_the_image_box else 0} cut by the image boundary)"
                   f"{absent}\n")
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
                   "voxels)\n"
                   # a resampled mask is not padded by default (delaunay-watershed 0.8.0): the ball is 4 voxels from
                   # the faces in y and x, and the meshes of cells 3 and 4 lie on two of them; the hint names the
                   # command's option, not the library's keyword; each cell's mesh volume over its voxel count shows
                   # that neither took the medium
                   + "".join(
                       f"{prefix}cell {cell} lies on the image face {face}, 4 voxels of medium away from its voxels "
                       f"(not padded: off by default on a resampled mask, pass --pad-thin-medium-margins), mesh volume "
                       f"{ratio} times its voxel count{end}"
                       for prefix, cell, face, ratio, end in (
                           ("foambryo mesh: warning: ThinMediumMarginWarning: ", 3, "axis1_high", "0.82", "; "),
                           ("", 4, "axis2_high", "0.97", (
                               ". The mesh of such a cell may include medium between it and the face, which makes its "
                               "volume larger than its voxel count (cell_volume_report lists the cells outside "
                               "[0.5, 1.5]). See dw3d.thin_medium_margins.\n")),
                       )
                   ))
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
    (np.zeros((4, 4, 4, 4), dtype=np.int32), "mask.npy holds a 4-D array; a mask is 3-D (z, y, x) or 2-D (y, x)"),
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


# ---------------------------------------------------------------------------------------------
# view --stress full|deviatoric
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ball_mesh_and_results(tmp_path_factory) -> tuple[Path, Path]:
    """The four-cell ball meshed and inferred by the command (no dataset needed: runs in the CI layout)."""
    folder = tmp_path_factory.mktemp("ball")
    np.save(folder / "mask.npy", MASKS["four cells in a ball"]())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert cli.main(["mesh", str(folder / "mask.npy"), "-o", str(folder / "ball.rec"), "--spacing", "1", "1",
                         "1"]) == cli.EXIT_OK
        assert cli.main(["infer", str(folder / "ball.rec"), "-o", str(folder / "results")]) == cli.EXIT_OK
    return folder / "ball.rec", folder / "results"


@pytest.fixture
def opened_force_views(monkeypatch):
    """`plot_force_inference` replaced by a recorder of its keywords (skipped without Polyscope)."""
    pytest.importorskip("polyscope")
    from foambryo import viewing

    calls: list = []
    monkeypatch.setattr(viewing, "plot_force_inference", lambda *args, **kwargs: calls.append(kwargs))
    return calls


@pytest.mark.parametrize("stress", [None, "full", "deviatoric"])
def test_view_passes_the_stress_to_the_force_view(capsys, ball_mesh_and_results, opened_force_views, stress):
    mesh, outdir = ball_mesh_and_results
    options = [] if stress is None else ["--stress", stress]
    code, _, err = _run(capsys, "view", mesh, "--results", outdir, *options)
    assert code == cli.EXIT_OK, err
    assert opened_force_views[-1]["stress"] == (stress or "full")  # full by default, as plot_force_inference


def test_view_refuses_an_unknown_stress_and_a_deviatoric_stress_without_pressures(capsys, ball_mesh_and_results):
    mesh, _ = ball_mesh_and_results
    assert _run(capsys, "view", mesh, "--stress", "isotropic")[0] == cli.EXIT_USAGE
    code, out, err = _run(capsys, "view", mesh, "--stress", "deviatoric", "--no-pressures")
    assert (code, out) == (cli.EXIT_USAGE, "")
    assert "--stress deviatoric is drawn by the force view: leave out --no-pressures" in err


def test_view_no_show_writes_the_deviatoric_views_bars_without_polyscope(capsys, tmp_path, ball_mesh_and_results,
                                                                        monkeypatch):
    """The tension bar and the deviatoric stress's, from the tensions alone; no pressure bar, no Polyscope."""
    from foambryo._viewer_figures import (
        DEVIATORIC_STRESS_TITLE,
        deviatoric_stress_principal_values,
        principal_stress_colour_scale,
    )

    benchmark_mesh, outdir = ball_mesh_and_results
    monkeypatch.setitem(__import__("sys").modules, "polyscope", None)  # any import of Polyscope now fails
    bars = tmp_path / "bars.png"
    code, out, err = _run(capsys, "view", benchmark_mesh, "--results", outdir, "--colour-bars", bars, "--no-show",
                          "--stress", "deviatoric")
    assert code == cli.EXIT_OK, err
    assert bars.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    mesh = cli._read_mesh(Path(benchmark_mesh), None)
    tensions = cli._read_view_results(outdir, Path(benchmark_mesh), mesh).tensions
    assert "pressure" not in out
    bar = principal_stress_colour_scale(DEVIATORIC_STRESS_TITLE, deviatoric_stress_principal_values(mesh, tensions))
    assert bar.low < 0 < bar.high
    assert out == f"wrote {bars}: tension {out.split('tension ')[1].split(',')[0]}, deviatoric stress " \
                  f"{bar.low:.6g} to {bar.high:.6g}\n"


# ---------------------------------------------------------------------------------------------
# mesh --pad-thin-medium-margins / --no-pad-thin-medium-margins
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize(("options", "expected"), [([], None), (["--pad-thin-medium-margins"], True),
                                                   (["--no-pad-thin-medium-margins"], False)])
def test_mesh_passes_the_padding_choice_to_the_reconstruction(capsys, tmp_path, monkeypatch, options, expected):
    """None (the library's default) unless the option is given."""
    calls: list = []
    real = cli.reconstruct_mask

    def recording(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(cli, "reconstruct_mask", recording)
    np.save(tmp_path / "mask.npy", MASKS["four cells in a ball"]())
    code, _, err = _run(capsys, "mesh", tmp_path / "mask.npy", "-o", tmp_path / "m.rec", "--spacing", 1, 1, 1,
                        *options)
    assert code == cli.EXIT_OK, err
    assert calls[-1]["pad_thin_medium_margins"] is expected


def test_mesh_names_its_options_in_the_thin_margin_warning(capsys, tmp_path):
    """Asked for, a resampled mask is padded and no cell is named; refused, a cubic one is not, and the hint says so."""
    np.save(tmp_path / "mask.npy", MASKS["four cells in a ball"]())
    code, _, err = _run(capsys, "mesh", tmp_path / "mask.npy", "-o", tmp_path / "k2.rec", "--spacing", 0.6, 0.3, 0.3,
                        "--pad-thin-medium-margins")
    assert code == cli.EXIT_OK, err
    assert "not padded" not in err
    cube = np.zeros((18, 18, 18), dtype=np.uint16)
    cube[3:15, 3:15, 3:15] = 1  # a 12-voxel cube 3 voxels from every face
    np.save(tmp_path / "cube.npy", cube)
    code, _, err = _run(capsys, "mesh", tmp_path / "cube.npy", "-o", tmp_path / "padded.rec", "--spacing", 1, 1, 1)
    assert code == cli.EXIT_OK, err
    assert "ThinMediumMarginWarning" not in err  # padded by default on cubic voxels
    code, _, err = _run(capsys, "mesh", tmp_path / "cube.npy", "-o", tmp_path / "unit.rec", "--spacing", 1, 1, 1,
                        "--no-pad-thin-medium-margins")
    assert code == cli.EXIT_OK, err
    line = next(line for line in err.splitlines() if "ThinMediumMarginWarning" in line)
    assert "(not padded: --no-pad-thin-medium-margins was given)" in line
    assert "pad_thin_medium_margins" not in err  # the library's keyword is not what a command-line user types


def test_reconstruct_mask_pads_as_asked_and_is_the_library_default_otherwise():
    """Bitwise delaunay-watershed's mesh: the default call, padding on and padding off (cubic and resampled)."""
    from dw3d import get_default_mesh_reconstruction_algorithm

    from foambryo.io import reconstruct_mask

    mask = MASKS["four cells in a ball"]()
    meshes = {}
    for spacing in ((1.0, 1.0, 1.0), (0.6, 0.3, 0.3)):
        for pad in (None, True, False):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ours = reconstruct_mask(mask, spacing, pad_thin_medium_margins=pad)
                algorithm = get_default_mesh_reconstruction_algorithm(None, False)
                keyword = {} if pad is None else {"pad_thin_medium_margins": pad}
                theirs = algorithm.construct_mesh_in_physical_units(mask, spacing, spacing_unit="um", axis_order="zyx",
                                                                    **keyword)
            for a, b in zip((ours.points, ours.triangles, ours.labels), theirs, strict=True):
                np.testing.assert_array_equal(a, b)
            meshes[spacing, pad] = ours.points
    # the defaults: padded on cubic voxels, not padded when resampled
    np.testing.assert_array_equal(meshes[(1.0, 1.0, 1.0), None], meshes[(1.0, 1.0, 1.0), True])
    np.testing.assert_array_equal(meshes[(0.6, 0.3, 0.3), None], meshes[(0.6, 0.3, 0.3), False])
    assert meshes[(1.0, 1.0, 1.0), True].shape != meshes[(1.0, 1.0, 1.0), False].shape or not np.array_equal(
        meshes[(1.0, 1.0, 1.0), True], meshes[(1.0, 1.0, 1.0), False])  # the choice changes the mesh


def test_a_two_dimensional_mask_takes_the_padding_choice_too(capsys, tmp_path):
    """A square 2 pixels from one side of the image: refused, it is not padded and the hint names the option."""
    mask = np.zeros((48, 48), dtype=np.uint16)
    mask[2:26, 12:36] = 1
    mask[26:46, 12:36] = 2
    np.save(tmp_path / "mask2d.npy", mask)
    code, _, err = _run(capsys, "mesh", tmp_path / "mask2d.npy", "-o", tmp_path / "off.rec", "--spacing", 1, 1,
                        "--no-pad-thin-medium-margins")
    assert code == cli.EXIT_OK, err
    assert "(not padded: --no-pad-thin-medium-margins was given)" in err
    code, _, err = _run(capsys, "mesh", tmp_path / "mask2d.npy", "-o", tmp_path / "on.rec", "--spacing", 1, 1)
    assert code == cli.EXIT_OK, err
    assert "not padded" not in err


# ---------------------------------------------------------------------------------------------
# the requirement a too-old delaunay-watershed-3d is told about
# ---------------------------------------------------------------------------------------------
def _declared_delaunay_watershed_floor() -> str:
    """The release of `delaunay-watershed>=X` in `setup.cfg`'s `install_requires`, without its pre-release part."""
    parser = configparser.ConfigParser()
    parser.read(REPOSITORY / "setup.cfg", encoding="utf-8")
    for requirement in parser["options"]["install_requires"].split():
        if requirement.startswith("delaunay-watershed>="):
            from packaging.version import Version  # pytest's own dependency

            return Version(requirement.removeprefix("delaunay-watershed>=")).base_version
    raise AssertionError("setup.cfg declares no delaunay-watershed floor")


@pytest.mark.parametrize("command", ["mesh", "infer-over-seeds"])
def test_a_reconstruction_without_physical_units_names_the_requirement_foambryo_declares(capsys, tmp_path,
                                                                                        monkeypatch, command):
    """The message names the delaunay-watershed release foambryo requires, not an older one it no longer accepts."""
    from dw3d.reconstruction_algorithm import MeshReconstructionAlgorithm

    monkeypatch.delattr(MeshReconstructionAlgorithm, "construct_mesh_in_physical_units")
    np.save(tmp_path / "mask.npy", _four_cells_in_a_ball())
    output = tmp_path / ("m.rec" if command == "mesh" else "ensemble")
    code, _, err = _run(capsys, command, tmp_path / "mask.npy", "-o", output, "--spacing", 1, 1, 1)
    assert code == cli.EXIT_FILE
    floor = _declared_delaunay_watershed_floor()
    assert f"needs delaunay-watershed {floor} or later, as foambryo requires" in err
    assert "pip install -U delaunay-watershed" in err
    assert "0.7 or later" not in err


def test_the_library_entry_points_name_the_requirement_foambryo_declares(monkeypatch):
    """`reconstruct_mask` and `dcel_mesh_from_segmentation_mask` say the same as the command."""
    from foambryo import dcel_mesh_from_segmentation_mask
    from foambryo import io as foambryo_io

    class OlderReconstruction:
        def construct_mesh_from_segmentation_mask(self, _mask: np.ndarray) -> None:
            raise AssertionError

    floor = _declared_delaunay_watershed_floor()
    monkeypatch.setattr(foambryo_io, "get_default_mesh_reconstruction_algorithm", lambda *_: OlderReconstruction())
    expected = f"needs delaunay-watershed {floor} or later, as foambryo requires"
    with pytest.raises(ValueError, match=expected):
        foambryo_io.reconstruct_mask(_four_cells_in_a_ball(), (1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match=expected):
        dcel_mesh_from_segmentation_mask(_four_cells_in_a_ball(), spacing=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match=expected):
        dcel_mesh_from_segmentation_mask(_four_cells_in_a_ball(), physical_image_faces=("axis0_low",))
