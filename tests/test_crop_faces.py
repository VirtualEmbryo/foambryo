"""Crop faces: kept out of every equation, absent from the tension map, cut cells withheld.

A mesh of a cropped tissue carries a mask of its crop faces (`DcelData(..., crop_faces=...)`):
triangles on the boundary of the imaged volume, labelled `(0, c)` like medium contact but not
medium contact. These tests pin what `foambryo.crop_faces` prescribes:

* a crop face, and every junction edge and vertex that touches it, enters no equation; for the
  angle-reading methods that is exactly an edge-level refusal of every junction edge with a
  vertex on a crop face (the formulation validated on plane crops of the benchmark tissues);
* an interface made only of crop faces is absent from every tension map, never a number;
* the crop and a wedge-refusal policy are two reasons to decline an edge, and combine;
* the cut cells' pressures are solved for and returned as `nan`, with a warning naming them;
* a mesh without crop faces gives what it gave before, bit for bit;
* the crop survives a `.rec` and a `.vtk` file, and a file that does not record it warns.

The cropped tissue is a ball cut in three cells and cropped through two of them, reconstructed by
dw3d's default algorithm (deterministic), so the test needs no data file.
"""

import warnings

import numpy as np
import pytest
from dw3d import get_default_mesh_reconstruction_algorithm
from dw3d.io import save_rec, save_vtk

from foambryo import (
    CutCellPressureWarning,
    PressureComputationMethod,
    TensionComputationMethod,
    UnrecordedCropFacesWarning,
    WedgeRefusal,
    WedgeRefusalPolicy,
    dcel_mesh_from_file,
    dcel_mesh_from_segmentation_mask,
    infer_force_balance,
    infer_pressures,
    infer_tensions,
    infer_tensions_by_junction_component,
    infer_tensions_with_uncertainty,
    infer_tensions_young_dupre_local,
    refusal_report,
)
from foambryo.crop_faces import border_like_faces, crop_edges
from foambryo.dcel import DcelData
from foambryo.geometry import compute_trijunction_edges
from foambryo.uncertainty import compute_inference_diagnostics
from foambryo.wedge_refusal import line_averaged_angles, refusal_with_crop

ANGLE_METHODS = (
    TensionComputationMethod.YoungDupre,
    TensionComputationMethod.ProjectionYoungDupre,
    TensionComputationMethod.Cotan,
    TensionComputationMethod.InvCotan,
    TensionComputationMethod.Lami,
    TensionComputationMethod.InvLami,
    TensionComputationMethod.LogLami,
    TensionComputationMethod.YoungDupreLocal,
)


def _ball_in_three_cells(cropped: bool, past_the_centre: int = 10) -> np.ndarray:
    """Cells 1 and 2 side by side, cell 3 below both; cropped, the image cuts cells 1 and 2.

    The crop keeps axis 0 up to `past_the_centre` slices beyond the ball's centre.
    """
    n, radius = 60, 20
    z, y, x = np.mgrid[:n, :n, :n]
    centre = n / 2
    inside = (z - centre) ** 2 + (y - centre) ** 2 + (x - centre) ** 2 <= radius**2
    mask = np.zeros((n, n, n), dtype=np.uint16)
    mask[inside & (y < centre) & (z >= centre - 8)] = 1
    mask[inside & (y >= centre) & (z >= centre - 8)] = 2
    mask[inside & (z < centre - 8)] = 3
    return np.ascontiguousarray(mask[: int(centre) + past_the_centre]) if cropped else mask


@pytest.fixture(scope="module")
def reconstruction():
    """The cropped ball as dw3d reconstructs it: points, triangles, labels and crop-face mask."""
    algorithm = get_default_mesh_reconstruction_algorithm()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        points, triangles, labels = algorithm.construct_mesh_from_segmentation_mask(_ball_in_three_cells(True))
    return (
        np.asarray(points, dtype=np.float64),
        np.asarray(triangles, dtype=np.int64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(algorithm.crop_faces),
    )


@pytest.fixture(scope="module")
def cropped(reconstruction):
    points, triangles, labels, crop = reconstruction
    return DcelData(points, triangles, labels, crop_faces=crop)


@pytest.fixture(scope="module")
def as_medium(reconstruction):
    """The same mesh with the crop read as medium contact, as before crop faces were recorded."""
    points, triangles, labels, _crop = reconstruction
    return DcelData(points, triangles, labels)


def _quiet(function, *args: object, **kwargs: object):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


# ---------------------------------------------------------------------------------------------
# the mesh
# ---------------------------------------------------------------------------------------------
def test_the_mesh_records_its_crop_faces_and_cut_cells(cropped, reconstruction):
    *_mesh, crop = reconstruction
    assert cropped.has_crop_faces
    assert np.array_equal(cropped.crop_faces, crop)
    assert cropped.cut_cells == (1, 2)
    # Cells 1 and 2 still touch the medium beside the crop, so their `(0, c)` interfaces are real.
    assert cropped.crop_interfaces == ()
    assert set(np.unique(cropped.f[cropped.crop_faces, :3])) == set(cropped.crop_vertices.tolist())


@pytest.mark.parametrize("method", list(TensionComputationMethod))
def test_an_interface_made_only_of_crop_faces_is_absent(reconstruction, method):
    """Mark every `(0, 1)` face as crop: the interface is the crop, and no method may give it a value."""
    points, triangles, labels, _crop = reconstruction
    whole = (labels.min(axis=1) == 0) & (labels.max(axis=1) == 1)
    mesh = DcelData(points, triangles, labels, crop_faces=whole)
    assert mesh.crop_interfaces == ((0, 1),)
    tensions = _quiet(infer_tensions, mesh, 1, method)
    assert (0, 1) not in tensions
    assert set(tensions) == set(mesh.compute_areas_interfaces()) - {(0, 1)}


def test_crop_faces_can_be_given_as_rows_and_are_checked(reconstruction):
    points, triangles, labels, crop = reconstruction
    by_rows = DcelData(points, triangles, labels, crop_faces=np.flatnonzero(crop))
    assert np.array_equal(by_rows.crop_faces, crop)
    with pytest.raises(ValueError, match="mask of shape"):
        DcelData(points, triangles, labels, crop_faces=crop[:-1])
    interior = int(np.flatnonzero(labels.min(axis=1) != 0)[0])
    with pytest.raises(ValueError, match="medium label 0"):
        DcelData(points, triangles, labels, crop_faces=[interior])
    without = DcelData(points, triangles, labels, crop_faces=None)
    assert not without.has_crop_faces
    assert without.cut_cells == ()
    assert without.crop_interfaces == ()


# ---------------------------------------------------------------------------------------------
# tensions
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("method", list(TensionComputationMethod))
def test_no_tension_map_holds_a_crop_interface(cropped, method):
    tensions = _quiet(infer_tensions, cropped, 1, method)
    expected = set(cropped.compute_areas_interfaces()) - set(cropped.crop_interfaces)
    assert set(tensions) == expected
    assert np.isfinite([tensions[k] for k in tensions]).any()


@pytest.mark.parametrize("method", ANGLE_METHODS)
def test_the_crop_is_an_edge_level_refusal_of_every_junction_edge_touching_it(cropped, as_medium, method):
    """The validated formulation: the same numbers as refusing those edges on the uncropped mesh object."""
    crop_vertices = set(cropped.crop_vertices.tolist())
    touching = frozenset(e.vertices for e in compute_trijunction_edges(as_medium) if set(e.vertices) & crop_vertices)
    explicit = WedgeRefusal(WedgeRefusalPolicy.ExcludeEdgeFromLineMean, edges=touching)
    reference = _quiet(infer_tensions, as_medium, 1, method, refusal=explicit)
    tensions = _quiet(infer_tensions, cropped, 1, method)
    for key, value in tensions.items():
        assert value == reference[key] or (np.isnan(value) and np.isnan(reference[key]))
    for key in cropped.crop_interfaces:
        assert np.isnan(reference[key])


def test_the_crop_changes_the_tensions_it_should(cropped, as_medium):
    """Read as medium, the crop's fake junction lines enter the system; kept out, they do not."""
    as_medium_tensions = _quiet(infer_tensions, as_medium, 1, TensionComputationMethod.YoungDupre)
    tensions = _quiet(infer_tensions, cropped, 1, TensionComputationMethod.YoungDupre)
    shared = [k for k in tensions if np.isfinite(tensions[k])]
    assert any(abs(tensions[k] - as_medium_tensions[k]) > 1e-6 for k in shared)


def test_by_junction_component_and_per_edge_results_drop_the_crop(cropped):
    solution = _quiet(infer_tensions_by_junction_component, cropped, 1, TensionComputationMethod.YoungDupre)
    assert not set(solution.dict_tensions) & set(cropped.crop_interfaces)
    assert not set(solution.component_of) & set(cropped.crop_interfaces)
    assert not set(solution.undetermined) & set(cropped.crop_interfaces)
    assert len(solution.tensions) == len(solution.dict_tensions)
    local = _quiet(infer_tensions_young_dupre_local, cropped)
    assert not set(local.dict_tensions) & set(cropped.crop_interfaces)
    assert len(local.tensions) == len(local.dict_tensions)


def test_the_crop_and_a_refusal_policy_combine(cropped):
    crop = crop_edges(cropped)
    edges = compute_trijunction_edges(cropped)
    # A line that reaches the crop and keeps edges off it.
    by_line: dict = {}
    for e in edges:
        by_line.setdefault(tuple(sorted({int(k[1]) for k in e.keys})), []).append(e)
    line, line_edges = next(
        (t, es)
        for t, es in sorted(by_line.items())
        if any(e.vertices in crop for e in es) and sum(e.vertices not in crop for e in es) >= 2
    )
    off_crop = [e.vertices for e in line_edges if e.vertices not in crop]

    # The crop alone never drops the line: only its edges on the crop go.
    angles = line_averaged_angles(cropped, refusal=WedgeRefusal(WedgeRefusalPolicy.DropLine, edges=frozenset()))
    assert any(tuple(sorted(k)) == line for k in angles.dict_rad)
    # DropLine with one edge off the crop refused drops it.
    one = WedgeRefusal(WedgeRefusalPolicy.DropLine, edges=frozenset([off_crop[0]]))
    assert not any(tuple(sorted(k)) == line for k in line_averaged_angles(cropped, refusal=one).dict_rad)
    # The report counts the two reasons apart.
    report = refusal_report(cropped, one)
    assert report.n_edges_refused == 1
    assert report.n_edges_touching_crop == sum(e.vertices in crop for e in edges)
    assert report.length_touching_crop > 0
    # Idempotent, and a no-op without crop faces.
    composed = refusal_with_crop(cropped, one)
    assert composed.crop_edges == crop
    assert refusal_with_crop(cropped, composed) is composed


def test_a_mesh_without_crop_faces_is_untouched(as_medium):
    assert refusal_with_crop(as_medium, WedgeRefusal()) == WedgeRefusal()
    assert refusal_report(as_medium, WedgeRefusal()).n_edges_touching_crop == 0


# ---------------------------------------------------------------------------------------------
# pressures
# ---------------------------------------------------------------------------------------------
def test_cut_cells_pressures_are_withheld_with_a_warning(cropped):
    tensions = _quiet(infer_tensions, cropped, 1, TensionComputationMethod.YoungDupre)
    with pytest.warns(CutCellPressureWarning, match="2 cut cell"):
        pressures = infer_pressures(cropped, tensions)
    assert pressures[0] == 0
    assert np.isnan(pressures[1])
    assert np.isnan(pressures[2])
    assert np.isfinite(pressures[3])


def test_large_cropped_pressure_systems_take_a_sparse_path_with_the_same_answer(cropped, monkeypatch):
    """A tissue-scale crop cannot afford the dense solve; the sparse normal-equations path must agree."""
    import foambryo.pressure_inference as pressure_inference

    tensions = _quiet(infer_tensions, cropped, 1, TensionComputationMethod.YoungDupre)
    dense = _quiet(infer_pressures, cropped, tensions)
    monkeypatch.setattr(pressure_inference, "DENSE_CROP_PRESSURE_MAX_ENTRIES", 0)
    sparse = _quiet(infer_pressures, cropped, tensions)
    assert set(dense) == set(sparse)
    for cell, value in dense.items():
        assert np.isnan(value) == np.isnan(sparse[cell])
        if np.isfinite(value):
            assert abs(sparse[cell] - value) <= 1e-8 * max(abs(value), 1e-12)


def test_cut_cells_withheld_from_the_force_balance_too(cropped):
    with pytest.warns(CutCellPressureWarning):
        result = infer_force_balance(cropped)
    assert not set(result.interfaces) & set(cropped.crop_interfaces)
    pressures = result.dict_pressures
    assert np.isnan(pressures[1])
    assert np.isnan(pressures[2])
    assert np.isfinite(pressures[3])


def test_methods_without_a_crop_formulation_refuse(cropped):
    tensions = _quiet(infer_tensions, cropped, 1, TensionComputationMethod.YoungDupre)
    with pytest.raises(ValueError, match="no formulation for crop faces"):
        infer_pressures(cropped, tensions, PressureComputationMethod.Laplace)
    with pytest.raises(ValueError, match="no formulation for crop faces"):
        compute_inference_diagnostics(cropped)
    with pytest.raises(ValueError, match="no formulation for crop faces"):
        infer_tensions_with_uncertainty(cropped)


# ---------------------------------------------------------------------------------------------
# files and reconstruction
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("binary_mode", [False, True])
def test_a_rec_file_keeps_the_crop(reconstruction, cropped, tmp_path, binary_mode):
    points, triangles, labels, crop = reconstruction
    path = tmp_path / "cropped.rec"
    save_rec(path, points, triangles, labels, binary_mode=binary_mode, crop_faces=crop)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnrecordedCropFacesWarning)
        mesh = dcel_mesh_from_file(path)
    assert int(mesh.crop_faces.sum()) == int(crop.sum())
    assert mesh.cut_cells == cropped.cut_cells
    assert mesh.crop_interfaces == cropped.crop_interfaces


def test_a_vtk_file_keeps_the_crop(reconstruction, cropped, tmp_path):
    points, triangles, labels, crop = reconstruction
    path = tmp_path / "cropped.vtk"
    save_vtk(path, points, triangles, labels, crop_faces=crop)
    mesh = _quiet(dcel_mesh_from_file, path)
    assert np.array_equal(mesh.crop_faces, crop)
    assert mesh.cut_cells == cropped.cut_cells


def test_a_file_that_does_not_record_the_crop_warns(tmp_path):
    """A crop through the ball's widest section, large enough to look like one (`BORDER_LIKE_MIN_PATCH_AREA`)."""
    algorithm = get_default_mesh_reconstruction_algorithm()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        points, triangles, labels = algorithm.construct_mesh_from_segmentation_mask(_ball_in_three_cells(True, 0))
    crop = np.asarray(algorithm.crop_faces)
    assert border_like_faces(points, triangles, labels).sum() > 0.5 * crop.sum()
    path = tmp_path / "old_style.rec"
    save_rec(path, points, triangles, labels)
    with pytest.warns(UnrecordedCropFacesWarning, match="records no crop faces"):
        mesh = dcel_mesh_from_file(path)
    assert not mesh.has_crop_faces


def test_a_few_flat_medium_faces_at_the_extreme_do_not_warn():
    """A uncropped tissue can have scattered flat triangles on its bounding box; only a crop-sized patch warns."""
    from foambryo.crop_faces import BORDER_LIKE_MIN_PATCH_AREA

    # Four unit triangles at the corners of the 20 x 20 face z = 0: they span the whole face but
    # cover 0.5 % of it. An apex above sets the bounding box.
    apex = [[20.0, 20.0, 10.0]]
    corners = [(0.0, 0.0), (19.0, 0.0), (0.0, 19.0), (19.0, 19.0)]
    scattered = np.array([[x + dx, y + dy, 0.0] for x, y in corners for dx, dy in ((0, 0), (1, 0), (0, 1))])
    points = np.vstack([scattered, apex])
    triangles = np.arange(12).reshape(4, 3)
    labels = np.tile([0, 1], (4, 1))
    assert not border_like_faces(points, triangles, labels).any()

    # A connected 6 x 6 square of them, in one corner: 9 % of the face, spanning 30 % of each side.
    grid = np.array([[i, j, 0.0] for i in range(7) for j in range(7)], dtype=float)
    cells = [(i * 7 + j, (i + 1) * 7 + j, i * 7 + j + 1, (i + 1) * 7 + j + 1) for i in range(6) for j in range(6)]
    square = np.array([t for a, b, c, d in cells for t in ((a, b, c), (b, d, c))])
    assert BORDER_LIKE_MIN_PATCH_AREA <= 36 / 400
    points = np.vstack([grid, apex])
    assert border_like_faces(points, square, np.tile([0, 1], (len(square), 1))).all()

    # The same 72 triangles pulled apart (no shared edge) do not form a patch.
    apart = np.array(
        [
            [float(k % 9) * 2.2 + dx, float(k // 9) * 2.2 + dy, 0.0]
            for k in range(72)
            for dx, dy in ((0, 0), (1, 0), (0, 1))
        ]
    )
    points = np.vstack([apart, apex])
    assert not border_like_faces(points, np.arange(216).reshape(72, 3), np.tile([0, 1], (72, 1))).any()


def test_an_uncropped_file_does_not_warn(tmp_path):
    algorithm = get_default_mesh_reconstruction_algorithm()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        points, triangles, labels = algorithm.construct_mesh_from_segmentation_mask(_ball_in_three_cells(False))
    path = tmp_path / "complete.rec"
    save_rec(path, points, triangles, labels)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnrecordedCropFacesWarning)
        mesh = dcel_mesh_from_file(path)
    assert not mesh.has_crop_faces


def test_reconstruction_passes_the_crop_and_a_physical_face_removes_it():
    mask = _ball_in_three_cells(True)
    mesh = _quiet(dcel_mesh_from_segmentation_mask, mask)
    assert mesh.cut_cells == (1, 2)
    physical = _quiet(dcel_mesh_from_segmentation_mask, mask, physical_image_faces=("axis0_high",))
    assert not physical.has_crop_faces
    assert np.array_equal(physical.f, mesh.f)
    with pytest.raises(ValueError, match="unknown image face"):
        _quiet(dcel_mesh_from_segmentation_mask, mask, physical_image_faces=("bottom",))


# ---------------------------------------------------------------------------------------------
# viewers
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def headless():
    ps = pytest.importorskip("polyscope")
    real_init, real_show = ps.init, ps.show
    try:
        real_init("openGL_mock")
    except RuntimeError as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"Polyscope's mock backend is unavailable: {error}")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ps, "init", lambda *_args, **_kwargs: real_init("openGL_mock"))
        patch.setattr(ps, "show", lambda *_args, **_kwargs: real_show(forFrames=2))
        yield ps
    ps.remove_all_structures()


@pytest.mark.parametrize("name", ["plot_force_inference", "plot_tension_inference", "plot_pressure_inference"])
def test_the_viewers_draw_the_crop_apart(headless, cropped, name):
    import foambryo.viewing as viewing

    _quiet(getattr(viewing, name), cropped)
    assert headless.has_surface_mesh("Crop faces")
    headless.remove_all_structures()


def test_junction_residuals_skip_the_crop(headless, reconstruction):
    """With a crop interface absent from the tension map, the residual viewer still runs, on real lines only."""
    import foambryo.viewing as viewing
    from foambryo.tension_inference import compute_residual_junctions_dict

    points, triangles, labels, _crop = reconstruction
    whole = (labels.min(axis=1) == 0) & (labels.max(axis=1) == 1)
    mesh = DcelData(points, triangles, labels, crop_faces=whole)
    tensions = _quiet(infer_tensions, mesh)
    residuals = _quiet(compute_residual_junctions_dict, mesh, tensions)
    assert residuals
    assert not any(set(line) >= {0, 1} for line in residuals)
    _quiet(viewing.plot_residual_junctions, mesh)
    headless.remove_all_structures()
