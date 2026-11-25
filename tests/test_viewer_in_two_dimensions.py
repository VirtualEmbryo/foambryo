"""The Polyscope viewer on a 2-D tissue: what is drawn, where, in which colours; and `foambryo view` on 2-D input.

What this pins, and why:

* **the cells' polygons are exact**: each cell's triangles cover its boundary polygon (outline minus holes) to
  rounding, convex or not, with an enclosed cell (a hole in its neighbour) and a pocket of medium; no triangle lies
  outside its cell. Polyscope's own polygon faces are fans, wrong for a non-convex cell;
* **the frame is not a mirror**: a mesh point `(row, column)` is drawn at Polyscope's `(column, -row)`, a rotation
  (determinant +1), so the window shows the image as napari and matplotlib's `imshow` do (row 0 at the top);
* **the colours are the library's**: each interface segment's colour is `foambryo.colour_scales`' colour of its
  tension, each cell's that of its pressure; an undetermined tension is a green curve; the segments that close a cut
  cell along the image border carry no value and are drawn in the glass's edge colour;
* the window shows the tensions first (curves over the image) and the panel's *pressures* fills the cells;
* **explode** moves each cell by `coeff x (its area centroid - the tissue's)`;
* an image under the mesh must be on the mesh's pixel grid, and only a 2-D mesh takes one;
* **`foambryo view` takes 2-D input**: a 2-D mesh (with `--image`), a 2-D mask (the 2-D mask window, its choices the
  2-D ones); `foambryo infer --angle-reading` gives the library's numbers for the reading asked, in both dimensions,
  and refuses a reading of the other dimension (exit 2).

Polyscope runs on its mock backend (no window); the drawn figures are checked by hand in the record's screenshots.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import foambryo
from foambryo import cli
from foambryo._colour_scales import (
    CROP_FACE_EDGE_COLOR,
    UNDETERMINED_COLOR,
    colour_bar_ranges,
    pressure_colour_fractions,
    pressure_colours,
    surface_pressure_colour_upper,
    tension_colour_fractions,
    tension_colours,
)
from foambryo._planar_drawing import _cell_loops, _signed_area, triangulate_polygon
from foambryo.io import dcel_mesh_from_segmentation_mask
from foambryo.plotting_utilities import planar_mesh_drawing


def _run(capsys: pytest.CaptureFixture[str], *arguments: object) -> tuple[int, str, str]:
    code = cli.main([str(argument) for argument in arguments])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _four_cells() -> np.ndarray:
    """Four cells in a disk of medium, 80 x 100 pixels, none on the image border."""
    rows, columns = np.mgrid[:80, :100]
    inside = (rows - 40) ** 2 / 34**2 + (columns - 50) ** 2 / 44**2 < 1
    labels = np.zeros((80, 100), dtype=np.int32)
    labels[inside & (rows < 40) & (columns < 50)] = 1
    labels[inside & (rows < 40) & (columns >= 50)] = 2
    labels[inside & (rows >= 40) & (columns < 55)] = 3
    labels[inside & (rows >= 40) & (columns >= 55)] = 4
    return labels


@pytest.fixture(scope="module")
def mesh():
    return dcel_mesh_from_segmentation_mask(_four_cells())


@pytest.fixture(scope="module")
def cut_mesh():
    return dcel_mesh_from_segmentation_mask(np.ascontiguousarray(_four_cells()[:, 30:]))


@pytest.fixture
def mock_polyscope(monkeypatch):
    ps = pytest.importorskip("polyscope")
    real_init, real_show = ps.init, ps.show
    try:
        real_init("openGL_mock")
    except RuntimeError as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"Polyscope's mock backend is unavailable: {error}")
    monkeypatch.setattr(ps, "init", lambda *_args, **_kwargs: real_init("openGL_mock"))
    monkeypatch.setattr(ps, "show", lambda *_args, **_kwargs: real_show(forFrames=1))
    yield ps
    ps.remove_all_structures()
    ps.set_navigation_style("turntable")


# ------------------------------------------------------------------------------------------------
# the scene: exact polygons, junctions, image-border segments
# ------------------------------------------------------------------------------------------------
def test_ear_clipping_covers_a_non_convex_polygon_exactly():
    # a comb: three teeth, every second vertex reflex
    outline = np.array([[0, 0], [0, 9], [6, 9], [6, 7], [2, 7], [2, 5], [6, 5], [6, 3], [2, 3], [2, 1], [6, 1],
                        [6, 0]], dtype=np.float64)
    loop = np.arange(len(outline))
    triangles = np.asarray(triangulate_polygon([loop], outline))
    corners = outline[triangles]
    u, v = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    areas = 0.5 * np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
    assert areas.sum() == pytest.approx(abs(_signed_area(outline)), rel=1e-12)
    from matplotlib.path import Path

    assert Path(outline).contains_points(corners.mean(axis=1)).all()


def test_cell_triangles_cover_each_cell_with_an_enclosed_cell_and_a_pocket_of_medium():
    labels = _four_cells()
    rows, columns = np.mgrid[:80, :100]
    labels[(rows - 20) ** 2 + (columns - 30) ** 2 < 7**2] = 5  # a cell inside cell 1
    labels[(rows - 55) ** 2 + (columns - 75) ** 2 < 5**2] = 0  # medium inside cell 4
    with pytest.warns(UserWarning, match="touch no side of the image"):  # the enclosed pocket of medium
        mesh = dcel_mesh_from_segmentation_mask(labels)
    scene = planar_mesh_drawing(mesh)
    for cell in scene.cells:
        rows_of_cell = np.flatnonzero((mesh.labels == cell).any(axis=1))
        loops = _cell_loops(mesh.segments[rows_of_cell], len(mesh.points))
        signed = sorted((abs(_signed_area(mesh.points[loop])) for loop in loops), reverse=True)
        assert scene.areas[cell] == pytest.approx(signed[0] - sum(signed[1:]), rel=1e-12), cell
        if cell != 5:  # the mesh of the small enclosed cell (r = 7 px, at min_distance 4) is coarser than its pixels
            assert scene.areas[cell] == pytest.approx(np.count_nonzero(labels == cell), rel=0.05), cell
    assert len(_cell_loops(mesh.segments[(mesh.labels == 1).any(axis=1)], len(mesh.points))) == 2
    assert len(_cell_loops(mesh.segments[(mesh.labels == 4).any(axis=1)], len(mesh.points))) == 2


def test_junctions_and_the_image_border_segments(mesh, cut_mesh):
    scene = planar_mesh_drawing(mesh)
    assert not scene.border.any()
    degree = np.bincount(mesh.segments.reshape(-1), minlength=len(mesh.points))
    assert set(scene.junctions) == set(np.flatnonzero(degree >= 3))
    cut = planar_mesh_drawing(cut_mesh)
    border = cut.border
    assert border.any()
    on_border = np.zeros(len(cut_mesh.points), dtype=bool)
    on_border[cut_mesh.image_border_vertices] = True
    assert on_border[cut_mesh.segments[border]].all()
    assert (cut_mesh.labels[border, 0] == 0).all()
    assert set(cut_mesh.labels[border, 1]) <= set(cut_mesh.cut_cells)


def test_the_drawn_frame_is_the_image_rotated_not_mirrored():
    from foambryo._display_frame import drawn_points
    from foambryo._planar_viewer import planar_scene_points

    rc = np.array([[0.0, 0.0], [0.0, 10.0], [5.0, 0.0]])  # top-left, top-right, lower-left pixel of an image
    drawn = drawn_points(planar_scene_points(rc))
    assert np.array_equal(drawn, [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, -5.0, 0.0]])
    # (row, column) -> (x, y) = (column, -row): the matrix [[0, 1], [-1, 0]], a rotation
    assert np.linalg.det(np.array([[0.0, 1.0], [-1.0, 0.0]])) == 1.0


def test_explode_moves_each_cell_by_its_area_centroid(mesh):
    scene = planar_mesh_drawing(mesh)
    total = sum(scene.areas.values())
    tissue = sum(scene.areas[c] * scene.centroids[c] for c in scene.cells) / total
    assert np.allclose(scene.tissue_centroid, tissue)
    moves = scene.displacements(0.3)
    for cell in scene.cells:
        assert np.allclose(moves[cell], 0.3 * (scene.centroids[cell] - tissue))


# ------------------------------------------------------------------------------------------------
# the window (Polyscope's mock backend)
# ------------------------------------------------------------------------------------------------
def _planar_window(mesh, kind="force", **kwargs: object):
    from foambryo import viewing

    options = viewing._InferenceOptions(foambryo.TensionComputationMethod.YoungDupre, None, 1.0, 0.0, "auto",
                                        dimension=2, pressures=kind != "tension")
    tensions, pressures, values_from = options.infer(mesh, kind, None, None, None)
    return viewing._window_class(2)(mesh, kind, tensions, pressures, options, values_from, **kwargs)


def _captured(monkeypatch, ps):
    """The colours handed to the interfaces' curve network and to the cells' surface, by quantity name."""
    seen: dict[tuple[str, str], np.ndarray] = {}
    for cls in (ps.CurveNetwork, ps.SurfaceMesh):
        original = cls.add_color_quantity

        def recording(self, name, values, *args: object, _original=original, **kwargs: object):
            seen[(self.get_name(), name)] = np.array(values)
            return _original(self, name, values, *args, **kwargs)

        monkeypatch.setattr(cls, "add_color_quantity", recording)
    return seen


def test_the_window_draws_the_library_s_colours(mock_polyscope, monkeypatch, mesh):
    seen = _captured(monkeypatch, mock_polyscope)
    window = _planar_window(mesh)
    window.draw()
    scene = window.scene
    ranges = colour_bar_ranges(mesh, window.tensions, window.pressures)
    expected = tension_colours(tension_colour_fractions(
        np.array([window.tensions[scene.pairs[k]] for k in scene.segment_pair]), ranges.tension))
    # the curve network's edges follow the segments (no border segment on this mesh)
    assert np.array_equal(seen[("Interfaces", "Surface Tensions")], expected)
    cells = seen[("Cells", "Pressures Cells")]
    values = np.array([window.pressures[int(c)] for c in scene.triangle_cell])
    upper = surface_pressure_colour_upper(mesh, window.pressures)
    assert upper == ranges.pressure[1]
    assert np.array_equal(cells, pressure_colours(pressure_colour_fractions(values, upper)))
    for name in ("Interfaces", "Junctions", "Cells"):
        assert mock_polyscope.has_curve_network(name) or mock_polyscope.has_point_cloud(name) or \
            mock_polyscope.has_surface_mesh(name)
    assert not mock_polyscope.get_surface_mesh("Cells").is_enabled()  # tensions first: the curves over the image
    assert window.flat_colours  # a plane facing the camera: drawn flat, the bars' colours
    window.showing = "pressure"
    window.apply_showing()
    assert mock_polyscope.get_surface_mesh("Cells").is_enabled()


def test_undetermined_tensions_are_green_and_the_border_has_no_value(mock_polyscope, monkeypatch, cut_mesh):
    seen = _captured(monkeypatch, mock_polyscope)
    window = _planar_window(cut_mesh, kind="tension")
    pair = next(p for p in window.scene.pairs if p[0] != 0)
    window.tensions = {**window.tensions, pair: float("nan")}
    window.draw()
    scene = window.scene
    interior = np.flatnonzero(~scene.border)
    colours = seen[("Interfaces", "Surface Tensions")]
    undetermined = scene.segment_pair[interior] == scene.pairs.index(pair)
    assert undetermined.any()
    assert np.array_equal(colours[undetermined], np.tile(UNDETERMINED_COLOR, (int(undetermined.sum()), 1)))
    border = mock_polyscope.get_curve_network("Image border (no value)")
    assert np.allclose(border.get_color(), CROP_FACE_EDGE_COLOR)
    assert window.scales.crop
    assert window.scales.planar


def test_explode_draws_the_cells_moved(mock_polyscope, monkeypatch, mesh):
    registered = {}
    original = mock_polyscope.register_surface_mesh

    def register(name, vertices, faces, *args: object, **kwargs: object):
        registered[name] = np.array(vertices)
        return original(name, vertices, faces, *args, **kwargs)

    monkeypatch.setattr(mock_polyscope, "register_surface_mesh", register)
    window = _planar_window(mesh)
    window.draw()
    still = registered["Cells"].copy()
    window.set_explode(True, 0.5)
    moved = registered["Cells"]
    # Polyscope's (x, y) are (column, -row): the move of a vertex of cell c is 0.5 x (centroid - tissue) rotated
    shifts = {tuple(np.round(m - s, 9)) for s, m in zip(still[:, :2], moved[:, :2], strict=True)}
    expected = {tuple(np.round([0.5 * (window.scene.centroids[c] - window.scene.tissue_centroid)[1],
                                -0.5 * (window.scene.centroids[c] - window.scene.tissue_centroid)[0]], 9))
                for c in window.scene.cells}
    assert shifts == expected


def test_an_image_must_be_on_the_mesh_s_pixel_grid(mock_polyscope, mesh):
    from foambryo import viewing

    with pytest.raises(ValueError, match="cannot be drawn under it"):
        _planar_window(mesh, image=np.zeros((10, 10)))
    window = _planar_window(mesh, image=np.random.default_rng(0).random(mesh.image_shape))
    window.draw()
    assert mock_polyscope.has_surface_mesh("Image")
    three_d = foambryo.dcel_mesh_from_segmentation_mask(np.pad(np.ones((6, 6, 6), np.int32), 2))
    with pytest.raises(ValueError, match="2-D mesh only"):
        viewing.plot_force_inference(three_d, image=np.zeros((10, 10)), show=False)


def test_plot_force_inference_writes_a_2d_figure(mock_polyscope, tmp_path, mesh):
    from foambryo import viewing

    path = tmp_path / "figure.png"
    viewing.plot_force_inference(mesh, length_unit="um", save_figure=path, show=False)
    assert path.stat().st_size > 0
    assert mock_polyscope.get_navigation_style() == "turntable"  # restored for a later 3-D window


# ------------------------------------------------------------------------------------------------
# the command line
# ------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def files(tmp_path_factory):
    folder = tmp_path_factory.mktemp("view_two_dimensional")
    np.save(folder / "mask.npy", _four_cells())
    assert cli.main(["mesh", str(folder / "mask.npy"), "-o", str(folder / "mesh.rec"), "--spacing", "0.5",
                     "0.5"]) == cli.EXIT_OK
    np.save(folder / "image.npy", np.random.default_rng(1).random((80, 100)))
    return folder


def test_view_writes_the_colour_bars_of_a_2d_mesh(capsys, files, tmp_path):
    code, out, err = _run(capsys, "view", files / "mesh.rec", "--no-show", "--colour-bars", tmp_path / "bars.png")
    assert code == cli.EXIT_OK, err
    tensions, pressures = foambryo.infer_forces(foambryo.dcel_mesh_from_file(files / "mesh.rec"))
    ranges = colour_bar_ranges(foambryo.dcel_mesh_from_file(files / "mesh.rec"), tensions, pressures)
    assert f"pressure 0 to {ranges.pressure[1]:.6g}" in out


@pytest.mark.parametrize(("arguments", "message"), [
    (("view", "{mesh}", "--no-show", "--colour-bars", "{out}/b.png", "--image", "{out}/small.npy"), "pixels, the mesh"),
    (("view", "{mesh}", "--no-show", "--colour-bars", "{out}/b.png", "--pressure-method", "variational"),
     "laplace only"),
    (("view", "{mesh}", "--no-show", "--colour-bars", "{out}/b.png", "--angle-reading", "quadric"),
     "not offered for a 2-D tissue"),
    (("infer", "{mesh}", "-o", "{out}/r", "--angle-reading", "quadric"), "not offered for a 2-D mesh"),
    (("infer", "{mesh}", "-o", "{out}/r", "--angle-reading", "chord", "--tension-method", "equilibrium-tiling",
      "--mask", "{mask}"), "reads no angle"),
])
def test_what_does_not_apply_exits_2(capsys, files, tmp_path, arguments, message):
    np.save(tmp_path / "small.npy", np.zeros((10, 10)))
    filled = [a.format(mesh=files / "mesh.rec", mask=files / "mask.npy", out=tmp_path) for a in arguments]
    code, _, err = _run(capsys, *filled)
    assert code == cli.EXIT_USAGE, err
    assert message in err


def test_infer_reads_the_angles_as_asked_in_2d(capsys, files, tmp_path):
    code, _, err = _run(capsys, "infer", files / "mesh.rec", "-o", tmp_path, "--angle-reading", "chord")
    assert code == cli.EXIT_OK, err
    mesh = foambryo.dcel_mesh_from_file(files / "mesh.rec")
    tensions, pressures = foambryo.infer_forces(mesh, angle_reading="chord")
    assert foambryo.results.read_tension_table(tmp_path / "tensions.csv") == tensions
    assert foambryo.results.read_pressure_table(tmp_path / "pressures.csv") == pressures
    assert tensions != foambryo.infer_forces(mesh)[0]  # the arc reading, the 2-D default, gives other numbers
    record = json.loads((tmp_path / "run.json").read_text())
    assert record["options"]["angle_reading"] == "chord"
    assert record["library_call"].endswith("angle_reading='chord')")


def test_infer_reads_the_angles_as_asked_in_3d_and_records_nothing_by_default(capsys, tmp_path):
    mask = np.zeros((24, 24, 24), dtype=np.int32)
    mask[4:20, 4:12, 4:20] = 1
    mask[4:20, 12:20, 4:12] = 2
    mask[4:20, 12:20, 12:20] = 3
    np.save(tmp_path / "boxes.npy", mask)
    assert cli.main(["mesh", str(tmp_path / "boxes.npy"), "-o", str(tmp_path / "boxes.rec"), "--spacing", "1", "1",
                     "1"]) == cli.EXIT_OK
    code, _, err = _run(capsys, "infer", tmp_path / "boxes.rec", "-o", tmp_path / "default")
    assert code == cli.EXIT_OK, err
    assert "angle_reading" not in json.loads((tmp_path / "default" / "run.json").read_text())["options"]
    code, _, err = _run(capsys, "infer", tmp_path / "boxes.rec", "-o", tmp_path / "arc", "--angle-reading", "arc")
    assert code == cli.EXIT_USAGE
    assert "not offered for a 3-D mesh" in err
    code, _, err = _run(capsys, "infer", tmp_path / "boxes.rec", "-o", tmp_path / "chord", "--angle-reading", "chord")
    assert code == cli.EXIT_OK, err
    mesh = foambryo.dcel_mesh_from_file(tmp_path / "boxes.rec")
    assert foambryo.results.read_tension_table(tmp_path / "chord" / "tensions.csv") == foambryo.infer_forces(mesh)[0]


def test_view_opens_the_2d_mask_window_with_the_2d_choices(capsys, files, monkeypatch):
    from foambryo import viewing

    opened = {}
    monkeypatch.setattr(viewing, "_view_mask_workflow", lambda workflow, **_: opened.update(workflow=workflow))
    code, _, err = _run(capsys, "view", files / "mask.npy", "--spacing", "0.5", "0.5", "--angle-reading", "chord")
    assert code == cli.EXIT_OK, err
    workflow = opened["workflow"]
    assert workflow.min_distance == 4
    assert workflow.estimate() is None
    choices = workflow.choices
    assert (choices.dimension, choices.pressure_method, choices.angle_reading) == (2, "laplace", "chord")
    _, kwargs = choices.one_mesh_call(None)
    assert kwargs["angle_reading"] == "chord"
    commands = workflow.commands()
    assert "--spacing 0.5 0.5" in commands[0]
    assert "--min-distance 4" in commands[0]
    assert "--angle-reading chord" in commands[1]


def test_infer_forces_reading_none_is_the_call_without_it(mesh):
    assert foambryo.infer_forces(mesh, angle_reading=None) == foambryo.infer_forces(mesh)
    outcome = foambryo.infer_tensions_then_pressures(mesh, angle_reading="fitted_tangent")
    assert (outcome.tensions, outcome.pressures) == foambryo.infer_forces(mesh, angle_reading="fitted_tangent")
    assert "angle_reading='fitted_tangent'" in outcome.library_call
