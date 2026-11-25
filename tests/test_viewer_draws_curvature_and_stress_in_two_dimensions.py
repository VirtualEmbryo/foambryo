"""The 2-D window draws each cell's principal stresses and the interfaces' curvatures, as the 3-D window does.

On recorded scenes (`window.prepare`: every Polyscope call with its arrays, in the drawn frame, no display):

* the force window draws a point cloud "Stress tensors principal directions" at each uncut cell's area centroid, with
  the 3-D window's eight vector quantities per axis set (compression and tension, each axis both ways) and its sizes and
  colours; each vector is the principal axis of `cell_mean_stress_tensors` scaled by its value, mapped as every point
  is, `(row, column) -> (column, -row)`; the tension and pressure windows draw none, as in 3D;
* a cell cut by the image border has none; exploded, the glyphs move with their cells;
* `scalar_quantities=True` adds "Curvature (arc)" (per segment) and "Curvature (local)" (per node) to the interfaces,
  off, in `jet` between the `alpha` quantiles, junctions green; `plot_force_inference` accepts it on a 2-D mesh.
"""

from __future__ import annotations

import numpy as np
import pytest

import foambryo
from foambryo import dcel_mesh_from_segmentation_mask
from foambryo._cell_stress import cell_mean_stress_tensors, split_principal_stresses
from foambryo._colour_scales import UNDETERMINED_COLOR
from foambryo._planar_curvature_and_stress import cell_areas_and_centroids
from foambryo.curvature import compute_curvature_interfaces, compute_curvature_vertices_cotan

from .test_viewer_in_two_dimensions import _four_cells

STRESS = "Stress tensors principal directions"


@pytest.fixture(scope="module")
def mesh():
    return dcel_mesh_from_segmentation_mask(_four_cells())


@pytest.fixture(scope="module")
def cut_mesh():
    return dcel_mesh_from_segmentation_mask(np.ascontiguousarray(_four_cells()[:, 30:]))


def _window(mesh, kind="force", **kwargs: object):
    from foambryo import viewing

    options = viewing._InferenceOptions(foambryo.TensionComputationMethod.YoungDupre, None, 1.0, 0.0, "auto",
                                        dimension=2, pressures=kind != "tension")
    tensions, pressures, values_from = options.infer(mesh, kind, None, None, None)
    return viewing._window_class(2)(mesh, kind, tensions, pressures, options, values_from, **kwargs)


def _structure_calls(recorder, name: str) -> tuple[tuple, dict, list[tuple[str, tuple, dict]]]:
    """The registration of structure `name` (its arguments) and the methods called on it."""
    for index, (target, call, args, kwargs) in enumerate(recorder.calls):
        if target is None and call.startswith("register_") and args[0] == name:
            methods = [(m, a, k) for t, m, a, k in recorder.calls if t == index]
            return args, kwargs, methods
    raise KeyError(name)


def _drawn(points: np.ndarray) -> np.ndarray:
    """`(row, column)` points or vectors as recorded in the drawn frame: `(column, -row, 0)`."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return np.column_stack([p[:, 1], -p[:, 0], np.zeros(len(p))])


def _three_dimensional_stress_calls():
    """The 3-D force window's stress glyphs on two cubes: the reference for sizes, names and colours."""
    from foambryo import viewing

    mask = np.zeros((20, 20, 34), dtype=np.int64)
    mask[3:17, 3:17, 3:17] = 1
    mask[3:17, 3:17, 17:31] = 2
    mesh_3d = dcel_mesh_from_segmentation_mask(mask)
    options = viewing._InferenceOptions(foambryo.TensionComputationMethod.YoungDupre, None, 1.0, 0.0, "auto")
    tensions, pressures, values_from = options.infer(mesh_3d, "force", None, None, None)
    window = viewing._ForceWindow(mesh_3d, "force", tensions, pressures, options, values_from)
    recorder, _ = window.prepare(False, 0.2)
    return _structure_calls(recorder, STRESS)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_the_force_window_draws_each_cells_principal_stresses_as_3d_does(mesh):
    window = _window(mesh)
    recorder, _ = window.prepare(False, 0.2)
    args, kwargs, methods = _structure_calls(recorder, STRESS)
    cells = list(window.scene.cells)
    np.testing.assert_allclose(args[1], _drawn(np.array([window.scene.centroids[c] for c in cells])), rtol=0,
                               atol=1e-12)
    areas, _ = cell_areas_and_centroids(window.mesh)
    tensors = cell_mean_stress_tensors(window.mesh, window.tensions, window.pressures, areas, cells)
    axes = [split_principal_stresses(tensors[c]) for c in cells]
    vectors = {a[0]: (a[1], k) for m, a, k in methods if m == "add_vector_quantity"}
    assert sorted(vectors) == sorted(f"{name}_axes_{k}{suffix}" for name in ("compression", "tension")
                                     for k in (0, 1) for suffix in ("", "_down"))
    for name in ("compression", "tension"):
        for k in (0, 1):
            expected = _drawn(np.array([getattr(a, name)[:, k] for a in axes]))
            np.testing.assert_allclose(vectors[f"{name}_axes_{k}"][0], expected, rtol=0, atol=1e-15)
            np.testing.assert_allclose(vectors[f"{name}_axes_{k}_down"][0], -expected, rtol=0, atol=1e-15)
    # the 3-D window's sizes and colours, quantity by quantity
    _, kwargs_3d, methods_3d = _three_dimensional_stress_calls()
    assert kwargs == kwargs_3d
    options_3d = {a[0]: k for m, a, k in methods_3d if m == "add_vector_quantity"}
    for name, (_, options) in vectors.items():
        assert options == options_3d[name]


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("kind", ["tension", "pressure"])
def test_the_tension_and_pressure_windows_draw_no_stress(mesh, kind):
    recorder, _ = _window(mesh, kind).prepare(False, 0.2)
    with pytest.raises(KeyError):
        _structure_calls(recorder, STRESS)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_a_cell_cut_by_the_image_border_has_no_stress(cut_mesh):
    window = _window(cut_mesh)
    cut = set(window.mesh.cut_cells)
    assert cut
    recorder, _ = window.prepare(False, 0.2)
    args, _, _ = _structure_calls(recorder, STRESS)
    uncut = [c for c in window.scene.cells if c not in cut]
    np.testing.assert_allclose(args[1], _drawn(np.array([window.scene.centroids[c] for c in uncut])), atol=1e-12)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_exploded_the_stresses_move_with_their_cells(mesh):
    window = _window(mesh)
    still = _structure_calls(window.prepare(False, 0.2)[0], STRESS)[0][1]
    moved = _structure_calls(window.prepare(True, 0.5)[0], STRESS)[0][1]
    moves = window.scene.displacements(0.5)
    np.testing.assert_allclose(moved - still, _drawn(np.array([moves[c] for c in window.scene.cells])), atol=1e-12)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("kind", ["force", "tension"])
def test_scalar_quantities_add_the_interfaces_curvatures(mesh, kind):
    window = _window(mesh, kind, scalar_quantities=True)
    recorder, _ = window.prepare(False, 0.2)
    args, _, methods = _structure_calls(recorder, "Interfaces")
    quantities = {a[0]: (np.asarray(a[1]), k) for m, a, k in methods if m == "add_color_quantity"}
    assert quantities["Curvature (arc)"][1] == {"defined_on": "edges", "enabled": False}
    assert quantities["Curvature (local)"][1] == {"defined_on": "nodes", "enabled": False}
    # per node: the vertex curvature; the junctions have none and are green
    nodes = args[1]
    points = window.mesh.points
    vertex_of_node = [int(np.flatnonzero(np.all(_drawn(points) == node, axis=1))[0]) for node in nodes]
    local = compute_curvature_vertices_cotan(window.mesh)[0][vertex_of_node]
    colours = quantities["Curvature (local)"][0]
    assert np.array_equal(colours[np.isnan(local)], np.tile(UNDETERMINED_COLOR, (int(np.isnan(local).sum()), 1)))
    assert np.isfinite(local).sum() > 0.8 * len(local)
    # per segment: |kappa| of the interface's fitted circle; one colour per interface
    arc = compute_curvature_interfaces(window.mesh)
    per_segment = np.array([abs(arc[window.scene.pairs[k]]) for k in window.scene.segment_pair])
    arc_colours = quantities["Curvature (arc)"][0]
    for value in np.unique(per_segment[np.isfinite(per_segment)]):
        assert len(np.unique(arc_colours[per_segment == value], axis=0)) == 1
    # off until chosen: the tensions still colour the curves
    assert any(a[0] == "Surface Tensions" and k.get("enabled") for m, a, k in methods if m == "add_color_quantity")


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_the_pressure_window_adds_no_curvature(mesh):
    recorder, _ = _window(mesh, "pressure", scalar_quantities=True).prepare(False, 0.2)
    _, _, methods = _structure_calls(recorder, "Interfaces")
    assert not [a for m, a, _ in methods if m == "add_color_quantity" and str(a[0]).startswith("Curvature")]


def test_plot_force_inference_accepts_scalar_quantities_on_a_2d_mesh(mesh, monkeypatch):
    from foambryo import viewing

    opened = {}

    class Window:
        def __init__(self, *_args: object, **kwargs: object) -> None:
            opened.update(kwargs)

        def open(self, **_kwargs: object) -> None:
            pass

    monkeypatch.setattr(viewing, "_window_class", lambda _dimension: Window)
    viewing.plot_force_inference(mesh, scalar_quantities=True, show=False)
    assert opened["scalar_quantities"] is True
