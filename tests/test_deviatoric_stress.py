"""The deviatoric stress of a cell, `sigma - (tr sigma / d) I` (`foambryo._cell_stress`, `part="deviatoric"`).

It depends on the tensions and the geometry only, so:

* **the pressures do not enter**: bitwise the same tensors when every pressure is perturbed, or when none is given,
  in 2D and 3D; and it is the deviatoric part of the full tensor (to rounding);
* **an isotropic cell has none**: a regular polygon (2D), a cube and an icosphere (3D), whose `sum l t t` or
  `sum A (I - n n)` is a multiple of `I`;
* **closed forms**: a rectangle `a x b` against the medium, `sigma_dev = t (1/b - 1/a) diag(1, -1)` in
  `(row, column)` (rows along `a`); a box `a x b x c`, `T = 2 t diag(1/b + 1/c, 1/a + 1/c, 1/a + 1/b)` minus its
  mean; the regular polygon of R212's tests, whose full tensor is `(-p + t / (R cos(pi / n))) I`, has zero;
* **R199's exact foams**: an 8-cell equilibrium foam with exact tensions and pressures; the tissue's total deviatoric
  stress, computed without the pressures, vanishes as its total full stress does, to the chord bound (the total of
  the deviatoric parts is the deviatoric part of the total, which is zero for a body free of external load), while
  each cell's is not zero;
* the default (`part="full"`) is the tensor of before, bitwise.

And the viewers draw it on request, under its own name, for a cell with an undetermined pressure too.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import foambryo
from foambryo import PlanarMesh, dcel_mesh_from_segmentation_mask
from foambryo._cell_stress import STRESS_PARTS, cell_mean_stress_tensors, deviatoric_part, split_principal_stresses
from foambryo._planar_curvature_and_stress import cell_areas_and_centroids
from foambryo.dcel import DcelData

from .exact_circular_arc_foams import (
    DEFAULT_STEP,
    _circle,
    exact_arc_mesh,
    exact_tensions_and_pressures,
    regular_polygon_cell,
)
from .test_cell_stress import _bubble, _mirror_doublet
from .test_two_dimensional_global_fit import _random_tiling
from .test_viewer_draws_curvature_and_stress_in_two_dimensions import _drawn, _structure_calls, _window
from .test_viewer_in_two_dimensions import _four_cells

WINDOW = (150.0, 250.0, 150.0, 250.0)
FULL = "Stress tensors principal directions"
DEVIATORIC = "Deviatoric stress principal directions"


def _rectangle_cell(a: float, b: float) -> PlanarMesh:
    """One cell (label 1) in the medium, the rectangle `[0, a] x [0, b]` in `(row, column)`, 4 vertices per side."""
    corners = np.array([[0.0, 0.0], [a, 0.0], [a, b], [0.0, b]])
    points = np.concatenate(
        [
            corners[k] + np.linspace(0, 1, 4, endpoint=False)[:, None] * (corners[(k + 1) % 4] - corners[k])
            for k in range(4)
        ],
    )
    n = len(points)
    segments = np.column_stack([(np.arange(n) + 1) % n, np.arange(n)])  # as `regular_polygon_cell`
    return PlanarMesh(points, segments, np.tile([0, 1], (n, 1)), image_border_vertices=[])


def _box_cell(extents: tuple[float, float, float]) -> DcelData:
    """One cell (label 1) in the medium: the box `[0, a] x [0, b] x [0, c]`, two triangles per face, wound outward.

    Built by hand: trimesh 4.0.0, the floor, builds its box with `ndarray.ptp`, which NumPy 2 removed.
    """
    corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=np.float64)
    vertices = corners * np.asarray(extents, dtype=np.float64)
    centre = vertices.mean(axis=0)
    faces = []
    for axis in range(3):
        for side in (0, 1):
            quad = [k for k in range(8) if corners[k, axis] == side]  # four corners of one face, in grid order
            a, b, c, d = quad[0], quad[1], quad[3], quad[2]  # around the face
            for triangle in ((a, b, c), (a, c, d)):
                normal = np.cross(
                    vertices[triangle[1]] - vertices[triangle[0]], vertices[triangle[2]] - vertices[triangle[0]],
                )
                outward = normal @ (vertices[list(triangle)].mean(axis=0) - centre) > 0
                faces.append(triangle if outward else (triangle[0], triangle[2], triangle[1]))
    # outward winding: the normal points from the cell (1) to the medium (0)
    return DcelData(vertices, np.asarray(faces, dtype=np.int64), np.tile([1, 0], (len(faces), 1)))


def _assert_traceless_and_symmetric(tensor: np.ndarray, scale: float) -> None:
    assert abs(np.trace(tensor)) <= 1e-12 * scale
    np.testing.assert_allclose(tensor, tensor.T, rtol=0, atol=1e-12 * scale)


@pytest.fixture(scope="module", params=(0, 1, 2))
def foam(request):
    v = _random_tiling(8, request.param)
    tensions, pressures = exact_tensions_and_pressures(v)
    mesh = exact_arc_mesh(v, WINDOW)
    r_min = min(_circle(v, a, b)[1] for a, b in {tuple(p) for p in mesh.labels.tolist()})
    return mesh, tensions, pressures, r_min


# ---------------------------------------------------------------------------------------- the part, and its checks
def test_the_parts_and_the_default():
    assert STRESS_PARTS == ("full", "deviatoric")
    tensor = np.array([[3.0, 1.0], [1.0, -1.0]])
    np.testing.assert_array_equal(deviatoric_part(tensor), [[2.0, 1.0], [1.0, -2.0]])
    with pytest.raises(ValueError, match="deviatoric"):
        cell_mean_stress_tensors(regular_polygon_cell(5.0, 8), {(0, 1): 1.0}, {1: 0.0}, {1: 1.0}, [1], part="mean")


def test_the_default_is_the_full_tensor_bitwise(foam):
    mesh, tensions, pressures, _ = foam
    areas, _ = cell_areas_and_centroids(mesh)
    default = cell_mean_stress_tensors(mesh, tensions, pressures, areas, list(areas))
    full = cell_mean_stress_tensors(mesh, tensions, pressures, areas, list(areas), part="full")
    assert all(np.array_equal(default[c], full[c]) for c in areas)
    doublet = _mirror_doublet()
    volumes = doublet.compute_volumes_cells()
    tensions_3d = {(0, 1): 1.0, (0, 2): 1.0, (1, 2): 0.7}
    default = cell_mean_stress_tensors(doublet, tensions_3d, {1: 0.3, 2: 0.3}, volumes, [1, 2])
    full = cell_mean_stress_tensors(doublet, tensions_3d, {1: 0.3, 2: 0.3}, volumes, [1, 2], part="full")
    assert all(np.array_equal(default[c], full[c]) for c in (1, 2))


# ------------------------------------------------------------------------------------ the pressures do not enter
def test_in_2d_the_pressures_do_not_enter(foam):
    mesh, tensions, pressures, _ = foam
    areas, _ = cell_areas_and_centroids(mesh)
    cells = list(areas)
    rng = np.random.default_rng(7)
    perturbed = {c: p + 10 * rng.normal() for c, p in pressures.items()}
    reference = cell_mean_stress_tensors(mesh, tensions, pressures, areas, cells, part="deviatoric")
    for other in (perturbed, None, dict.fromkeys(pressures, math.nan)):
        found = cell_mean_stress_tensors(mesh, tensions, other, areas, cells, part="deviatoric")
        assert all(np.array_equal(found[c], reference[c]) for c in cells)
    # and it is the deviatoric part of the full tensor, with any pressure
    for given in (pressures, perturbed):
        full = cell_mean_stress_tensors(mesh, tensions, given, areas, cells)
        for c in cells:
            scale = np.max(np.abs(full[c]))
            np.testing.assert_allclose(reference[c], deviatoric_part(full[c]), rtol=0, atol=1e-13 * scale)
            _assert_traceless_and_symmetric(reference[c], scale)


@pytest.mark.parametrize("shared_tension", [0.4, 1.7])
def test_in_3d_the_pressures_do_not_enter(shared_tension):
    mesh = _mirror_doublet()
    volumes = mesh.compute_volumes_cells()
    tensions = {(0, 1): 1.0, (0, 2): 1.0, (1, 2): shared_tension}
    reference = cell_mean_stress_tensors(mesh, tensions, {1: 0.3, 2: 0.3}, volumes, [1, 2], part="deviatoric")
    for other in ({1: -4.0, 2: 12.5}, None, {1: math.nan, 2: math.nan}):
        found = cell_mean_stress_tensors(mesh, tensions, other, volumes, [1, 2], part="deviatoric")
        assert all(np.array_equal(found[c], reference[c]) for c in (1, 2))
    full = cell_mean_stress_tensors(mesh, tensions, {1: 0.3, 2: -2.0}, volumes, [1, 2])
    for c in (1, 2):
        np.testing.assert_allclose(reference[c], deviatoric_part(full[c]), rtol=0, atol=1e-13)
        _assert_traceless_and_symmetric(reference[c], 1.0)
        assert np.max(np.abs(reference[c])) > 1e-2  # the doublet's cells are anisotropic
    # the principal directions are the full tensor's own
    for c in (1, 2):
        _, full_axes = np.linalg.eigh(full[c])
        _, deviatoric_axes = np.linalg.eigh(reference[c])
        np.testing.assert_allclose(np.abs(full_axes.T @ deviatoric_axes), np.identity(3), atol=1e-9)


# ------------------------------------------------------------------------------------- an isotropic cell has none
@pytest.mark.parametrize("n", [3, 8, 64])
def test_a_regular_polygon_has_none(n):
    radius, tension = 10.0, 1.3
    mesh = regular_polygon_cell(radius, n, centre=(5.0, -3.0))
    areas, _ = cell_areas_and_centroids(mesh)
    full = cell_mean_stress_tensors(mesh, {(0, 1): tension}, {1: 0.2}, areas, [1])[1]
    # R212's closed form of the full tensor, and the deviatoric part of it: zero
    np.testing.assert_allclose(full, (-0.2 + tension / (radius * math.cos(math.pi / n))) * np.identity(2), atol=1e-14)
    deviatoric = cell_mean_stress_tensors(mesh, {(0, 1): tension}, None, areas, [1], part="deviatoric")[1]
    assert np.max(np.abs(deviatoric)) <= 1e-14 * tension / radius


def test_a_cube_and_an_icosphere_have_none():
    cube = _box_cell((2.0, 2.0, 2.0))
    volumes = cube.compute_volumes_cells()
    deviatoric = cell_mean_stress_tensors(cube, {(0, 1): 1.3}, None, volumes, [1], part="deviatoric")[1]
    assert np.max(np.abs(deviatoric)) <= 1e-14
    bubble = _bubble(4, 7.5)
    volumes = bubble.compute_volumes_cells()
    deviatoric = cell_mean_stress_tensors(bubble, {(0, 1): 1.3}, None, volumes, [1], part="deviatoric")[1]
    assert np.max(np.abs(deviatoric)) <= 1e-12 * 1.3 / 7.5  # icosahedral symmetry: isotropic to rounding


# ------------------------------------------------------------------------------------------------- closed forms
@pytest.mark.parametrize(("a", "b"), [(4.0, 10.0), (7.0, 3.0), (5.0, 5.0)])
def test_a_rectangle_has_the_closed_form(a, b):
    tension = 1.3
    mesh = _rectangle_cell(a, b)
    areas, _ = cell_areas_and_centroids(mesh)
    assert areas[1] == pytest.approx(a * b, rel=1e-14)
    deviatoric = cell_mean_stress_tensors(mesh, {(0, 1): tension}, None, areas, [1], part="deviatoric")[1]
    # T = (t / ab) (2 a e_r e_r + 2 b e_c e_c) = diag(2t/b, 2t/a); minus its mean, t (1/a + 1/b)
    expected = tension * (1 / b - 1 / a) * np.diag([1.0, -1.0])
    np.testing.assert_allclose(deviatoric, expected, rtol=0, atol=1e-15)
    # the cell is pulled more along its long side: the tensile axis is the long side's
    axes = split_principal_stresses(deviatoric)
    if a != b:
        long_axis = np.array([1.0, 0.0]) if a > b else np.array([0.0, 1.0])
        tensile = axes.tension[:, np.argmax(np.linalg.norm(axes.tension, axis=0))]
        assert abs(tensile @ long_axis) == pytest.approx(np.linalg.norm(tensile))


@pytest.mark.parametrize("extents", [(2.0, 3.0, 5.0), (4.0, 1.0, 1.5)])
def test_a_box_has_the_closed_form(extents):
    a, b, c = extents
    tension = 0.8
    mesh = _box_cell(extents)
    volumes = mesh.compute_volumes_cells()
    assert volumes[1] == pytest.approx(a * b * c, rel=1e-13)
    deviatoric = cell_mean_stress_tensors(mesh, {(0, 1): tension}, None, volumes, [1], part="deviatoric")[1]
    t = 2 * tension * np.array([1 / b + 1 / c, 1 / a + 1 / c, 1 / a + 1 / b])
    np.testing.assert_allclose(deviatoric, np.diag(t - t.mean()), rtol=0, atol=1e-14)


# ------------------------------------------------------------------------------------------- R199's exact foams
def test_the_free_foam_carries_no_total_deviatoric_stress_without_pressures(foam):
    mesh, tensions, _, r_min = foam
    areas, _ = cell_areas_and_centroids(mesh)
    deviatoric = cell_mean_stress_tensors(mesh, tensions, None, areas, list(areas), part="deviatoric")
    total = sum(areas[c] * deviatoric[c] for c in areas)
    lengths = np.hypot(*(mesh.points[mesh.segments[:, 1]] - mesh.points[mesh.segments[:, 0]]).T)
    scale = float(sum(lengths[k] * tensions[tuple(pair)] for k, pair in enumerate(mesh.labels.tolist())))
    bound = (DEFAULT_STEP / r_min) ** 2 / 12  # the chord bound of the full stress's test (R212)
    assert np.max(np.abs(total)) / scale < bound
    # each cell's is not zero: the cells are anisotropic (measured: the largest is 1.9e4 to 5.5e4 times the total,
    # and the total 0.035 to 0.052 of the bound, below the full stress's 0.54 to 0.58: the chords' error is mostly
    # isotropic)
    largest = max(np.max(np.abs(areas[c] * deviatoric[c])) for c in areas)
    assert largest > 1e3 * np.max(np.abs(total))


# ------------------------------------------------------------------------------------------------- the viewers
@pytest.fixture(scope="module")
def mesh_2d():
    return dcel_mesh_from_segmentation_mask(_four_cells())


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_the_2d_window_draws_the_deviatoric_stress_on_request(mesh_2d):
    window = _window(mesh_2d, stress="deviatoric")
    recorder, _ = window.prepare(False, 0.2)
    with pytest.raises(KeyError):
        _structure_calls(recorder, FULL)
    args, _, methods = _structure_calls(recorder, DEVIATORIC)
    cells = list(window.scene.cells)
    np.testing.assert_allclose(args[1], _drawn(np.array([window.scene.centroids[c] for c in cells])), atol=1e-12)
    areas, _ = cell_areas_and_centroids(window.mesh)
    tensors = cell_mean_stress_tensors(window.mesh, window.tensions, None, areas, cells, part="deviatoric")
    axes = [split_principal_stresses(tensors[c]) for c in cells]
    vectors = {a[0]: a[1] for m, a, _ in methods if m == "add_vector_quantity"}
    for name in ("compression", "tension"):
        for k in (0, 1):
            expected = _drawn(np.array([getattr(axis, name)[:, k] for axis in axes]))
            np.testing.assert_allclose(vectors[f"{name}_axes_{k}"], expected, rtol=0, atol=1e-15)
    # the full window, by default, is unchanged: it draws the full tensors under their own name
    default_recorder, _ = _window(mesh_2d).prepare(False, 0.2)
    _structure_calls(default_recorder, FULL)
    with pytest.raises(KeyError):
        _structure_calls(default_recorder, DEVIATORIC)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_a_cell_with_an_undetermined_pressure_gets_a_deviatoric_glyph(mesh_2d):
    window = _window(mesh_2d, stress="deviatoric")
    cell = next(iter(window.scene.cells))
    window.pressures = {**window.pressures, cell: math.nan}
    args, _, _ = _structure_calls(window.prepare(False, 0.2)[0], DEVIATORIC)
    assert len(args[1]) == len(window.scene.cells)
    window.stress = "full"
    args, _, _ = _structure_calls(window.prepare(False, 0.2)[0], FULL)
    assert len(args[1]) == len(window.scene.cells) - 1


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_the_3d_window_draws_the_deviatoric_stress_on_request():
    from foambryo import viewing

    mask = np.zeros((20, 20, 34), dtype=np.int64)
    mask[3:17, 3:17, 3:17] = 1
    mask[3:17, 3:17, 17:31] = 2
    mesh = dcel_mesh_from_segmentation_mask(mask)
    options = viewing._InferenceOptions(foambryo.TensionComputationMethod.YoungDupre, None, 1.0, 0.0, "auto")
    tensions, pressures, values_from = options.infer(mesh, "force", None, None, None)
    full_calls = _structure_calls(
        viewing._ForceWindow(mesh, "force", tensions, pressures, options, values_from).prepare(False, 0.2)[0],
        FULL,
    )
    window = viewing._ForceWindow(mesh, "force", tensions, pressures, options, values_from, stress="deviatoric")
    recorder, _ = window.prepare(False, 0.2)
    with pytest.raises(KeyError):
        _structure_calls(recorder, FULL)
    args, kwargs, methods = _structure_calls(recorder, DEVIATORIC)
    # the same glyphs, sizes and colours as the full stress's, at the same points; other vectors
    assert kwargs == full_calls[1]
    np.testing.assert_array_equal(args[1], full_calls[0][1])
    assert [(m, a[0], k) for m, a, k in methods] == [(m, a[0], k) for m, a, k in full_calls[2]]
    volumes = mesh.compute_volumes_cells()
    tensors = cell_mean_stress_tensors(mesh, tensions, None, volumes, [1, 2], part="deviatoric")
    for c in (1, 2):
        _assert_traceless_and_symmetric(tensors[c], 1.0)
    with pytest.raises(ValueError, match="deviatoric"):
        viewing._ForceWindow(mesh, "force", tensions, pressures, options, values_from, stress="isotropic")


def test_plot_force_inference_refuses_an_unknown_stress_before_drawing():
    from foambryo import viewing

    with pytest.raises(ValueError, match="deviatoric"):
        viewing.plot_force_inference(
            regular_polygon_cell(5.0, 8),
            {(0, 1): 1.0},
            {0: 0.0, 1: 0.2},
            stress="mean",
            show=False,
        )


# ---------------------------------------------------------------------------------------------------------------------
# The colour bars of a view drawn with the deviatoric stress: the stress's own bar, no pressure bar
# ---------------------------------------------------------------------------------------------------------------------
def _colour_quantities(recorder) -> set[str]:
    """The names of every colour quantity the recorded scene adds, on any structure."""
    return {args[0] for target, name, args, _ in recorder.calls if target is not None and name == "add_color_quantity"}


def _two_cubes_window(stress: str):
    from foambryo import viewing

    mask = np.zeros((20, 20, 34), dtype=np.int64)
    mask[3:17, 3:17, 3:17] = 1
    mask[3:17, 3:17, 17:31] = 2
    mesh = dcel_mesh_from_segmentation_mask(mask)
    options = viewing._InferenceOptions(foambryo.TensionComputationMethod.YoungDupre, None, 1.0, 0.0, "auto")
    tensions, pressures, values_from = options.infer(mesh, "force", None, None, None)
    return viewing._ForceWindow(mesh, "force", tensions, pressures, options, values_from, stress=stress)


def _principal_values(window, tensors) -> np.ndarray:
    return np.concatenate([np.linalg.eigvalsh(t) for t in tensors.values()])


@pytest.mark.parametrize("scattered", [False, True])
def test_the_3d_deviatoric_view_has_the_stress_bar_and_no_pressure(scattered):
    from foambryo._viewer_figures import DEVIATORIC_STRESS_TITLE, PRESSURE_TITLE, TENSION_TITLE

    window = _two_cubes_window("deviatoric")
    recorder, scales = window.prepare(scattered, 0.3)
    assert [s.title for s in scales.scales] == [TENSION_TITLE, DEVIATORIC_STRESS_TITLE]
    bar = scales.scale(DEVIATORIC_STRESS_TITLE)
    assert bar.colour_function == "stress"
    volumes = window.mesh.compute_volumes_cells()
    tensors = cell_mean_stress_tensors(window.mesh, window.tensions, None, volumes, [1, 2], part="deviatoric")
    values = _principal_values(window, tensors)
    assert values.min() < 0 < values.max()  # traceless and not zero: both colours are drawn
    np.testing.assert_allclose([bar.low, bar.high], [values.min(), values.max()], rtol=1e-12)
    # no cell is coloured by pressure: neither the surface nor the graph's nodes
    assert {"Pressures Cells", "Pressure"}.isdisjoint(_colour_quantities(recorder))
    # the full view, the default, keeps its pressure bar and colours and has no stress bar
    full_recorder, full_scales = _two_cubes_window("full").prepare(scattered, 0.3)
    assert [s.title for s in full_scales.scales] == [TENSION_TITLE, PRESSURE_TITLE]
    assert "Pressure" in _colour_quantities(full_recorder)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("scattered", [False, True])
def test_the_2d_deviatoric_view_has_the_stress_bar_and_no_pressure(mesh_2d, scattered):
    from foambryo import viewing
    from foambryo._viewer_figures import DEVIATORIC_STRESS_TITLE, PRESSURE_TITLE, TENSION_TITLE

    window = _window(mesh_2d, stress="deviatoric")
    recorder, scales = window.prepare(scattered, 0.3)
    assert [s.title for s in scales.scales] == [TENSION_TITLE, DEVIATORIC_STRESS_TITLE]
    cells = list(window.scene.cells)
    areas, _ = cell_areas_and_centroids(window.mesh)
    tensors = cell_mean_stress_tensors(window.mesh, window.tensions, None, areas, cells, part="deviatoric")
    values = _principal_values(window, tensors)
    bar = scales.scale(DEVIATORIC_STRESS_TITLE)
    np.testing.assert_allclose([bar.low, bar.high], [min(values.min(), 0.0), max(values.max(), 0.0)], rtol=1e-12)
    assert viewing._PRESSURES not in _colour_quantities(recorder)
    full_scales = _window(mesh_2d).prepare(scattered, 0.3)[1]
    assert [s.title for s in full_scales.scales] == [TENSION_TITLE, PRESSURE_TITLE]


def test_the_stress_bar_has_the_glyphs_two_colours():
    from foambryo._viewer_figures import (
        COMPRESSIVE_AXIS_COLOR,
        TENSILE_AXIS_COLOR,
        principal_stress_colour_scale,
    )
    from foambryo._planar_viewer import _STRESS_COLOURS

    assert dict(_STRESS_COLOURS) == {"compression": COMPRESSIVE_AXIS_COLOR, "tension": TENSILE_AXIS_COLOR}
    bar = principal_stress_colour_scale("Deviatoric stress", [-0.3, 0.1, math.nan, 0.5])
    assert (bar.low, bar.high) == (-0.3, 0.5)
    colours = bar.colours()
    values = np.linspace(-0.3, 0.5, len(colours))
    np.testing.assert_array_equal(colours[values < 0], np.tile(COMPRESSIVE_AXIS_COLOR, ((values < 0).sum(), 1)))
    np.testing.assert_array_equal(colours[values >= 0], np.tile(TENSILE_AXIS_COLOR, ((values >= 0).sum(), 1)))
    assert principal_stress_colour_scale("Deviatoric stress", [math.nan]) is None


def test_the_saved_bars_of_a_deviatoric_view_are_its_own(tmp_path, monkeypatch):
    """`save_colour_bars` (the tension and pressure bars) is not used: the window's bars are written."""
    import matplotlib

    matplotlib.use("Agg")
    from foambryo import viewing

    window = _two_cubes_window("deviatoric")
    window.draw = lambda: None  # no Polyscope: the scales are those `prepare` computes
    window.scales = window.prepare(False, 0.2)[1]

    def refuse(*args, **kwargs):
        raise AssertionError("the tension and pressure bars were written")

    monkeypatch.setattr(viewing, "save_colour_bars", refuse)
    path = window.save_colour_bars(tmp_path / "bars.png")
    assert path.is_file() and path.stat().st_size > 0


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_the_bars_written_without_a_window_have_the_windows_range(mesh_2d):
    """`deviatoric_stress_principal_values` (for `foambryo view --colour-bars`) gives the values the windows draw."""
    from foambryo._viewer_figures import deviatoric_stress_principal_values

    for window in (_two_cubes_window("deviatoric"), _window(mesh_2d, stress="deviatoric")):
        window.prepare(False, 0.2)
        assert window._stress_principal_values  # drawn
        assert deviatoric_stress_principal_values(window.mesh, window.tensions) == window._stress_principal_values


def test_the_stress_bar_is_drawn_with_its_zero_and_a_caption_that_fits():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from foambryo._viewer_figures import _draw_bar, principal_stress_colour_scale

    bar = principal_stress_colour_scale("Deviatoric stress", [-0.05, 0.02, 0.05])
    figure, axes = plt.subplots(figsize=(4, 0.6))
    _draw_bar(axes, bar)
    assert list(axes.get_xticks()) == [-0.05, 0.0, 0.05]
    assert [t.get_text() for t in axes.get_xticklabels()] == ["-0.05", "0", "0.05"]
    assert len(f"{bar.title} ({bar.clipping})") <= len("Surface tension (centred on the mean, 1; to the farther of the "
                                                       "5% and 95% quantiles)")  # no longer than the tension bar's
    plt.close(figure)
