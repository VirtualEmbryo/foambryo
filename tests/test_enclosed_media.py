"""A medium other than the exterior (an enclosed lumen) is solved like a cell but reported and drawn as a medium.

delaunay-watershed-3d 0.7.1 gives an enclosed pocket of label 0 lined by two or more cells its own label and records
the media in the `.rec` file (`media = 0 L ...`); `foambryo infer --media L` names them for a mesh made elsewhere.
What this pins:

* the media come from the file's appendix (parsed by dw3d 0.7.1, or as a string by an older dw3d) and from
  `--media`; a label the mesh lacks is refused (exit 2);
* a lumen is not a cell: not in `cells`, `n_cells`, `cut_cells`, the stress tensors, nor the mask report's cell
  statistics; the orientation check can leave every medium out;
* `pressures.csv` gains `region_kind` (`exterior`, `medium`, `cell`) only when there is a medium besides 0, so that
  a file without one stays readable by foambryo 0.7.0; `run.json` records `mesh.media` and `mesh.cells`;
* a triangle between a cell and the lumen is coloured by the cell's pressure, not the lumen's.

Mesh: a ball of radius 30 voxels cut into its 8 octants (cells 1-8) around a core of radius 10 voxels labelled 9 in
the mask (as dw3d 0.7.1 labels an enclosed pocket), 6 voxels of exterior around; meshed with the installed dw3d's
default algorithm at `min_distance` 3. The core is a lumen lined by all 8 cells.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from dw3d import get_default_mesh_reconstruction_algorithm
from dw3d.io import save_rec

from foambryo import cli, dcel_mesh_from_file, infer_forces
from foambryo._colour_scales import surface_face_pressures
from foambryo.dcel import DcelData
from foambryo.results import (
    PRESSURE_COLUMNS,
    PRESSURE_COLUMNS_WITH_REGION_KIND,
    read_pressure_region_kinds,
    read_pressure_table,
    write_pressure_table,
)
from foambryo.segmentation_masks import segmentation_mask_report
from foambryo.validation.mesh_io import check_label_orientation_convention

LUMEN = 9


def _shell_with_a_labelled_core() -> np.ndarray:
    n = 2 * (30 + 6) + 1
    centre = np.full(3, n // 2) + np.array([0.31, 0.17, 0.43])
    z, y, x = np.indices((n, n, n), dtype=np.float64)
    dz, dy, dx = z - centre[0], y - centre[1], x - centre[2]
    r2 = dz**2 + dy**2 + dx**2
    mask = np.zeros((n, n, n), dtype=np.uint16)
    inside = r2 < 30**2
    mask[inside] = (1 + (dz > 0) + 2 * (dy > 0) + 4 * (dx > 0))[inside]
    mask[r2 < 10**2] = LUMEN
    return mask


@pytest.fixture(scope="module")
def lumen_mesh_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    algorithm = get_default_mesh_reconstruction_algorithm(3, False)
    points, triangles, labels = algorithm.construct_mesh_from_segmentation_mask(_shell_with_a_labelled_core())
    assert {int(a) for a, b in np.asarray(labels).tolist() if b == LUMEN} == set(range(1, 9))
    return np.asarray(points), np.asarray(triangles), np.asarray(labels)


@pytest.fixture
def rec_with_media(lumen_mesh_arrays: tuple, tmp_path: Path) -> Path:
    path = tmp_path / "lumen_with_media_key.rec"
    save_rec(path, *lumen_mesh_arrays, metadata={"media": (0, LUMEN)})
    return path


@pytest.fixture
def rec_without_media(lumen_mesh_arrays: tuple, tmp_path: Path) -> Path:
    path = tmp_path / "lumen_without_media_key.rec"
    save_rec(path, *lumen_mesh_arrays)
    return path


def test_media_default_to_the_exterior_and_exclude_the_lumen_from_the_cells(lumen_mesh_arrays: tuple) -> None:
    plain = DcelData(*lumen_mesh_arrays)
    assert plain.media == (0,)
    assert plain.n_cells == 9
    mesh = DcelData(*lumen_mesh_arrays, media=[LUMEN])
    assert mesh.media == (0, LUMEN)
    assert mesh.cells.tolist() == list(range(1, 9))
    assert mesh.n_cells == 8
    assert mesh.has_enclosed_media
    assert mesh.cut_cells == ()
    with pytest.raises(ValueError, match="no triangle carries"):
        DcelData(*lumen_mesh_arrays, media=[77])


def test_the_media_are_read_from_the_rec_appendix(rec_with_media: Path, rec_without_media: Path) -> None:
    assert dcel_mesh_from_file(rec_with_media).media == (0, LUMEN)
    assert dcel_mesh_from_file(rec_without_media).media == (0,)


def test_the_lumen_gets_its_own_pressure_and_tensions(rec_with_media: Path) -> None:
    mesh = dcel_mesh_from_file(rec_with_media)
    tensions, pressures = infer_forces(mesh)
    assert all((cell, LUMEN) in tensions for cell in range(1, 9))
    assert math.isfinite(pressures[LUMEN])
    assert pressures[0] == 0.0


def test_the_pressure_table_says_which_rows_are_media(tmp_path: Path) -> None:
    pressures = {0: 0.0, 1: 0.5, 2: 0.25, LUMEN: 0.125}
    with_lumen, plain = tmp_path / "with.csv", tmp_path / "plain.csv"
    write_pressure_table(with_lumen, pressures, media=(0, LUMEN))
    write_pressure_table(plain, pressures, media=(0,))
    assert with_lumen.read_text().splitlines()[0] == ",".join(PRESSURE_COLUMNS_WITH_REGION_KIND)
    assert with_lumen.read_text().splitlines()[1:] == ["0,0.0,exterior", "1,0.5,cell", "2,0.25,cell", "9,0.125,medium"]
    # Without a medium besides 0 the file is the 0.7.0 format, which foambryo 0.7.0's reader requires.
    assert plain.read_text() == "label,pressure\n0,0.0\n1,0.5\n2,0.25\n9,0.125\n"
    assert plain.read_text().splitlines()[0] == ",".join(PRESSURE_COLUMNS)
    write_pressure_table(tmp_path / "default.csv", pressures)
    assert (tmp_path / "default.csv").read_bytes() == plain.read_bytes()
    assert read_pressure_table(with_lumen) == read_pressure_table(plain) == pressures
    assert read_pressure_region_kinds(with_lumen) == {0: "exterior", 1: "cell", 2: "cell", LUMEN: "medium"}
    assert read_pressure_region_kinds(plain) == {0: "exterior", 1: "cell", 2: "cell", LUMEN: "cell"}


def _run(capsys: pytest.CaptureFixture[str], *arguments: object) -> tuple[int, str, str]:
    code = cli.main([str(argument) for argument in arguments])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_infer_reports_the_media_from_the_file(capsys: pytest.CaptureFixture[str], rec_with_media: Path) -> None:
    outdir = rec_with_media.parent / "from-file"
    assert _run(capsys, "infer", rec_with_media, "-o", outdir)[0] == cli.EXIT_OK
    kinds = read_pressure_region_kinds(outdir / "pressures.csv")
    assert kinds[LUMEN] == "medium"
    assert kinds[0] == "exterior"
    record = json.loads((outdir / "run.json").read_text())
    assert record["mesh"]["media"] == [0, LUMEN]
    assert record["mesh"]["cells"] == 8
    assert record["options"]["media"] is None


def test_infer_takes_media_for_a_mesh_made_elsewhere(
    capsys: pytest.CaptureFixture[str],
    rec_without_media: Path,
) -> None:
    outdir = rec_without_media.parent / "named"
    assert _run(capsys, "infer", rec_without_media, "-o", outdir, "--media", LUMEN)[0] == cli.EXIT_OK
    assert read_pressure_region_kinds(outdir / "pressures.csv")[LUMEN] == "medium"
    record = json.loads((outdir / "run.json").read_text())
    assert record["mesh"]["media"] == [0, LUMEN]
    assert record["options"]["media"] == [LUMEN]
    plain = rec_without_media.parent / "plain"
    assert _run(capsys, "infer", rec_without_media, "-o", plain)[0] == cli.EXIT_OK
    assert (plain / "pressures.csv").read_text().splitlines()[0] == "label,pressure"
    code, _, err = _run(capsys, "infer", rec_without_media, "-o", rec_without_media.parent / "bad", "--media", 77)
    assert code == cli.EXIT_USAGE
    assert "--media" in err


def test_a_face_between_a_cell_and_the_lumen_takes_the_cells_pressure(lumen_mesh_arrays: tuple) -> None:
    mesh = DcelData(*lumen_mesh_arrays, media=[LUMEN])
    pressures = {label: float(label) for label in mesh.materials}
    colours = surface_face_pressures(mesh, pressures)
    against_the_lumen = mesh.f[:, 4] == LUMEN
    assert against_the_lumen.any()
    np.testing.assert_array_equal(colours[against_the_lumen], mesh.f[against_the_lumen, 3].astype(float))
    plain = DcelData(*lumen_mesh_arrays)
    np.testing.assert_array_equal(surface_face_pressures(plain, pressures)[against_the_lumen], float(LUMEN))


def test_the_orientation_check_and_the_mask_report_can_leave_the_media_out(lumen_mesh_arrays: tuple) -> None:
    points, triangles, labels = lumen_mesh_arrays
    report = check_label_orientation_convention(points, triangles, labels, exterior=(0, LUMEN))
    assert report["ok"]
    assert LUMEN in report["volumes"]
    mask = _shell_with_a_labelled_core()
    assert segmentation_mask_report(mask).n_cells == 9
    with_media = segmentation_mask_report(mask, media=[LUMEN])
    assert with_media.n_cells == 8
    assert str(LUMEN) not in with_media.as_record()["small_labels"]


# ---------------------------------------------------------------------------------------------
# a label of the mask absent from the mesh (delaunay-watershed-3d 0.7.1 drops the seed of a region with no
# tetrahedron in its voxels, and records it as `unresolved_regions`)
# ---------------------------------------------------------------------------------------------
def test_an_unresolved_label_is_read_and_kept_out_of_the_counts(
    capsys: pytest.CaptureFixture[str],
    lumen_mesh_arrays: tuple,
    tmp_path: Path,
) -> None:
    path = tmp_path / "with_an_unresolved_cell.rec"
    save_rec(path, *lumen_mesh_arrays, metadata={"unresolved_regions": (77,), "unresolved_regions_voxels": (27,)})
    mesh = dcel_mesh_from_file(path)
    assert mesh.unresolved == (77,)
    assert mesh.n_cells == 9
    assert 77 not in mesh.cut_cells
    outdir = tmp_path / "results"
    assert _run(capsys, "infer", path, "-o", outdir)[0] == cli.EXIT_OK
    record = json.loads((outdir / "run.json").read_text())
    assert record["mesh"]["unresolved"] == [77]
    assert record["mesh"]["cells"] == 9
    with pytest.raises(ValueError, match="carries"):
        DcelData(*lumen_mesh_arrays, unresolved=[1])


def test_reconstruct_mask_counts_and_names_an_unresolved_cell(
    monkeypatch: pytest.MonkeyPatch, lumen_mesh_arrays: tuple,
) -> None:
    """A stub of delaunay-watershed-3d 0.7.1's reconstruction: label 77 of the mask is absent from the mesh."""
    from types import SimpleNamespace

    from foambryo import io as foambryo_io

    points, triangles, labels = lumen_mesh_arrays

    class Stub:
        medium_labels = (0, LUMEN)
        enclosed_medium_report = SimpleNamespace(unresolved=(SimpleNamespace(label=77, voxels=27, kind="cell"),))
        crop_face_report = None
        crop_faces = None

        def construct_mesh_in_physical_units(self, *_args: object, **_kwargs: object) -> tuple:
            return points, triangles, labels

        def save_to_rec_mesh(self, filename: Path) -> None:
            save_rec(filename, points, triangles, labels)

    monkeypatch.setattr(foambryo_io, "get_default_mesh_reconstruction_algorithm", lambda *_args: Stub())
    reconstruction = foambryo_io.reconstruct_mask(np.zeros((4, 4, 4), dtype=np.uint8), (1.0, 1.0, 1.0))
    assert reconstruction.n_cells == 8
    assert reconstruction.media == (0, LUMEN)
    assert reconstruction.unresolved == {77: 27}
    assert reconstruction.n_cut_cells == 0
    summary = reconstruction.summary()
    assert "8 cells" in summary
    assert "1 enclosed medium (label 9)" in summary
    assert "1 label of the mask unresolved, absent from the mesh (77: 27 voxels)" in summary
    mesh = reconstruction.mesh()
    assert (mesh.unresolved, mesh.media, mesh.n_cells) == ((77,), (0, LUMEN), 8)
