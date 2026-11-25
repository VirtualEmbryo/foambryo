"""A 2-D mesh's orientation is measured on input: either orientation gives the same numbers; a mixed one is refused.

The rule (`foambryo.planar_mesh`'s docstring): each edge is the positively oriented boundary of its first label in
`(row, column)`, the rule of the triangles in 3D; dw2d follows it since 0.8.0 and stored every edge the other way round
before. The meshes here are dw2d 0.1.5's golden outputs (`two_dimensional_reference_meshes.npz`: the former
orientation, with the image-border edges in the arbitrary order dw2d 0.1.5 left them) and the same meshes with every
edge reversed (the one rule). The test that decides a mesh's orientation is the signed area of each cell's boundary
in `(row, column)`, each edge counted as stored for its first label and reversed for its second: positive for every
cell under the one rule, negative under the former one.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

import foambryo
from foambryo import _two_dimensional as fb2
from foambryo.planar_mesh import _orientation_from_geometry

DATA = Path(__file__).resolve().parent / "data"
NAMES = sorted(json.loads((DATA / "two_dimensional_reference_tensions.json").read_text())["cases"])


def _former(name: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(DATA / "two_dimensional_reference_meshes.npz") as arrays:
        return arrays[f"{name}__vertices"], arrays[f"{name}__edges"]


def _one_rule(edges: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(edges[:, [1, 0, 2, 3]])


def _same(a: dict, b: dict) -> bool:
    """Bitwise equality of two maps of floats, `nan` equal to `nan`."""
    return a.keys() == b.keys() and all(
        a[k] == b[k] or (math.isnan(a[k]) and math.isnan(b[k])) for k in a
    )


@pytest.mark.parametrize("name", NAMES)
def test_dw2d_0_1_5_meshes_measure_as_the_former_orientation_and_reversed_as_the_one_rule(name):
    vertices, edges = _former(name)
    assert _orientation_from_geometry(vertices, edges[:, :2], edges[:, 2:]) == "boundary_of_second_label"
    reversed_edges = _one_rule(edges)
    one_rule = _orientation_from_geometry(vertices, reversed_edges[:, :2], reversed_edges[:, 2:])
    assert one_rule == "boundary_of_first_label"


@pytest.mark.parametrize("name", NAMES)
def test_tensions_and_pressures_are_bitwise_the_same_from_either_orientation(name):
    vertices, edges = _former(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for reading in fb2.ANGLE_READINGS:
            former = fb2.infer_tensions((vertices, edges), angle_reading=reading)
            one_rule = fb2.infer_tensions((vertices, _one_rule(edges)), angle_reading=reading)
            assert _same(former, one_rule), reading
        tensions = fb2.infer_tensions((vertices, edges))
        assert _same(fb2.infer_pressures((vertices, edges), tensions),
                     fb2.infer_pressures((vertices, _one_rule(edges)), tensions))
        former_mesh = foambryo.PlanarMesh(vertices, edges[:, :2], edges[:, 2:])
        one_rule_mesh = foambryo.PlanarMesh(vertices, _one_rule(edges)[:, :2], edges[:, 2:])
        (t1, p1), (t2, p2) = foambryo.infer_forces(former_mesh), foambryo.infer_forces(one_rule_mesh)
    assert _same(t1, t2)
    assert _same(p1, p2)


def test_a_planar_mesh_stores_its_segments_in_the_one_rule():
    vertices, edges = _former("c_elegans__md3")
    former = foambryo.PlanarMesh(vertices, edges[:, :2], edges[:, 2:])
    assert former._reversed_on_input
    assert np.array_equal(former.segments, edges[:, [1, 0]])
    one_rule = foambryo.PlanarMesh(vertices, edges[:, [1, 0]], edges[:, 2:])
    assert not one_rule._reversed_on_input
    assert np.array_equal(one_rule.segments, former.segments)
    # Under the one rule each cell's boundary (its edges as stored where it is the first label, reversed where it is the
    # second) runs counter-clockwise in (row, column): a positive signed area.
    start, end = vertices[former.segments[:, 0]], vertices[former.segments[:, 1]]
    cross = start[:, 0] * end[:, 1] - start[:, 1] * end[:, 0]
    for cell in former.cells:
        area = 0.5 * (cross[former.labels[:, 0] == cell].sum() - cross[former.labels[:, 1] == cell].sum())
        assert area > 0, cell


def test_a_mesh_whose_cells_disagree_is_refused():
    vertices, edges = _former("c_elegans__md3")
    mixed = _one_rule(edges)
    cell = int(max(np.unique(edges[:, 2:]), key=lambda m: int(np.count_nonzero(edges[:, 2:] == m))))
    rows = (edges[:, 2] == cell) | (edges[:, 3] == cell)
    mixed[rows, :2] = mixed[rows][:, [1, 0]]
    with pytest.raises(ValueError, match="no single orientation rule"):
        foambryo.PlanarMesh(vertices, mixed[:, :2], mixed[:, 2:])
    with pytest.raises(ValueError, match="no single orientation rule"):
        fb2.infer_tensions((vertices, mixed))


def test_a_mesh_without_a_closed_cell_is_taken_as_given():
    vertices = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    edges = np.array([[0, 1, 1, 2], [1, 2, 1, 2]])
    assert _orientation_from_geometry(vertices, edges[:, :2], edges[:, 2:]) is None
    mesh = foambryo.PlanarMesh(vertices, edges[:, :2], edges[:, 2:])
    assert not mesh._reversed_on_input
    assert np.array_equal(mesh.segments, edges[:, :2])


def test_a_rec_file_in_the_former_orientation_gives_the_same_pressures(tmp_path):
    dw2d_io = pytest.importorskip("dw2d.io")
    vertices, edges = _former("mask_Cells_3__md3")
    former_path, one_rule_path = tmp_path / "former.rec", tmp_path / "one_rule.rec"
    dw2d_io.save_rec(former_path, vertices, edges[:, :2], edges[:, 2:], metadata={"dimension": 2, "axis_order": "yx"})
    dw2d_io.save_rec(one_rule_path, vertices, edges[:, [1, 0]], edges[:, 2:],
                     metadata={"dimension": 2, "axis_order": "yx", "orientation": "boundary_of_first_label"})
    former, one_rule = foambryo.dcel_mesh_from_file(former_path), foambryo.dcel_mesh_from_file(one_rule_path)
    assert np.array_equal(former.segments, one_rule.segments)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (t1, p1), (t2, p2) = foambryo.infer_forces(former), foambryo.infer_forces(one_rule)
    assert _same(t1, t2)
    assert _same(p1, p2)


def test_a_rec_file_that_contradicts_its_recorded_orientation_is_refused(tmp_path):
    dw2d_io = pytest.importorskip("dw2d.io")
    vertices, edges = _former("mask_Cells_3__md3")
    path = tmp_path / "contradicts.rec"
    dw2d_io.save_rec(path, vertices, edges[:, :2], edges[:, 2:],
                     metadata={"dimension": 2, "axis_order": "yx", "orientation": "boundary_of_first_label"})
    with pytest.raises(ValueError, match="inconsistent"):
        foambryo.dcel_mesh_from_file(path)
    path = tmp_path / "unknown.rec"
    dw2d_io.save_rec(path, vertices, edges[:, [1, 0]], edges[:, 2:],
                     metadata={"dimension": 2, "axis_order": "yx", "orientation": "clockwise"})
    with pytest.raises(ValueError, match="unknown orientation rule"):
        foambryo.dcel_mesh_from_file(path)


def test_dw2d_s_meshes_follow_the_one_rule_and_record_it(tmp_path):
    dw2d = pytest.importorskip("dw2d")
    n = 60
    r, c = np.indices((n, n))
    disk = (((r - 29.5) ** 2 + (c - 29.5) ** 2) <= 400).astype(np.int64)
    algorithm = dw2d.get_default_mesh_reconstruction_algorithm()
    points, segments, labels = algorithm.construct_mesh_from_segmentation_mask(disk)
    assert _orientation_from_geometry(points, segments, labels) == "boundary_of_first_label"
    algorithm.save_to_rec_mesh(tmp_path / "disk.rec")
    mesh = foambryo.dcel_mesh_from_file(tmp_path / "disk.rec")
    assert not mesh._reversed_on_input
    assert np.array_equal(mesh.segments, segments)


def _fully_cropped() -> dict[str, np.ndarray]:
    """Six cells filling a 60 x 90 image, every one cut by the border, meshed by dw2d 0.1.5 and by dw2d 0.8.0.

    The mask: Voronoi cells of seeds on a 2 x 3 grid jittered with `numpy.random.default_rng(3)` (stored in the file),
    meshed at min_distance 4 by dw2d 0.1.5 run on NumPy 2 and by dw2d 0.8.0's legacy route (no relocation, no
    canonical order, no repair, no sliver relabelling, no padding).
    """
    with np.load(DATA / "fully_cropped_tissue_from_dw2d_0_1_5_and_0_8_0.npz") as arrays:
        return {key: arrays[key] for key in arrays.files}


def test_a_fully_cropped_tissue_from_dw2d_0_1_5_is_measured_along_its_cells_loops():
    """Every cell cut by the image border: the cut cells vote with their border edges taken as their loops run.

    dw2d 0.1.5 stored the image-border edges in an arbitrary order, so counted as stored they gave 4 of the 6 cells a
    counter-clockwise boundary and 2 a clockwise one, and the mesh was refused. Taken along each cell's loop, every
    cell is clockwise: the former orientation.
    """
    arrays = _fully_cropped()
    vertices, edges = arrays["dw2d_0_1_5_vertices"], arrays["dw2d_0_1_5_edges"]
    on_border = foambryo.planar_mesh._vertices_on_the_image_border(vertices)
    border_rows = on_border[edges[:, 0]] & on_border[edges[:, 1]]
    cells = [int(c) for c in np.unique(edges[:, 2:]) if c]
    assert all(np.any(border_rows & np.any(edges[:, 2:] == c, axis=1)) for c in cells)  # every cell is cut
    start, end = vertices[edges[:, 0]], vertices[edges[:, 1]]
    cross = start[:, 0] * end[:, 1] - start[:, 1] * end[:, 0]
    as_stored = [0.5 * (cross[edges[:, 2] == c].sum() - cross[edges[:, 3] == c].sum()) for c in cells]
    assert min(as_stored) < 0 < max(as_stored)  # counted as stored, the cells disagree: the test is not vacuous
    assert _orientation_from_geometry(vertices, edges[:, :2], edges[:, 2:]) == "boundary_of_second_label"
    one_rule = arrays["dw2d_0_8_0_segments"], arrays["dw2d_0_8_0_labels"]
    assert _orientation_from_geometry(arrays["dw2d_0_8_0_points"], *one_rule) == "boundary_of_first_label"


def test_a_fully_cropped_tissue_from_dw2d_0_1_5_gives_the_tensions_of_dw2d_0_8_0():
    """The same mask meshed by dw2d 0.1.5 and by dw2d 0.8.0's legacy route: bitwise the same tensions."""
    arrays = _fully_cropped()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        former = foambryo.infer_tensions((arrays["dw2d_0_1_5_vertices"], arrays["dw2d_0_1_5_edges"]))
        mesh = arrays["dw2d_0_8_0_points"], arrays["dw2d_0_8_0_segments"], arrays["dw2d_0_8_0_labels"]
        current = foambryo.infer_tensions(mesh)
    assert len(current) == 14
    assert _same(former, current)
