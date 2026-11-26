"""The public drawing of a 2-D mesh, `foambryo.plotting_utilities.planar_mesh_drawing`: what its docstring promises.

Its triangles cover each cell exactly (non-convex cells and cells with a hole included), counter-clockwise in
`(row, column)`; it stays in the mesh's frame and unit; it is the same from either orientation; it raises `TypeError`
and `ValueError` where its docstring says, and draws a cell whose boundary is open without raising. It is reachable
without Polyscope: importing `foambryo.plotting_utilities` does not import Polyscope, as the napari plugin, which does
not install Polyscope, needs.
"""

import subprocess
import sys

import numpy as np
import pytest

from foambryo import PlanarMesh, colour_scales, plotting_utilities
from foambryo.plotting_utilities import PlanarMeshDrawing, planar_mesh_drawing


def _grid_mesh(cells: np.ndarray, spacing: float | None = None) -> PlanarMesh:
    """A 2-D mesh of unit squares: `cells[r, c]` is the label of the square of corner `(r, c)` (0 the medium).

    Every side between two squares of different labels (or a square and the outside, the medium) is a segment, stored
    in the one orientation rule: its first label on the side `(-dc, dr)` of its direction `(dr, dc)`.
    """
    rows, columns = cells.shape
    padded = np.zeros((rows + 2, columns + 2), dtype=np.int64)
    padded[1:-1, 1:-1] = cells

    def label_at(point: np.ndarray) -> int:
        return int(padded[int(np.floor(point[0])) + 1, int(np.floor(point[1])) + 1])

    index = {(r, c): r * (columns + 1) + c for r in range(rows + 1) for c in range(columns + 1)}
    points = np.array(sorted(index, key=index.get), dtype=np.float64)
    segments, labels = [], []
    for (r, c), i in index.items():
        for dr, dc in ((1, 0), (0, 1)):
            if (r + dr, c + dc) not in index:
                continue
            j = index[(r + dr, c + dc)]
            middle = np.array([r + dr / 2, c + dc / 2])
            left, right = label_at(middle + 0.5 * np.array([-dc, dr])), label_at(middle - 0.5 * np.array([-dc, dr]))
            if left == right:
                continue
            segments.append((i, j) if left < right else (j, i))
            labels.append(sorted((left, right)))
    scale = 1.0 if spacing is None else spacing
    return PlanarMesh(points * scale, np.array(segments), np.array(labels), spacing=spacing)


#: An L-shaped cell 1 (three squares: non-convex) and a square cell 2 in its notch.
L_SHAPE = np.array([[1, 2], [1, 1]])
#: A ring, cell 1 (eight squares around a hole), and cell 2 filling the hole.
RING = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]])


def _signed_triangle_areas(drawing: PlanarMeshDrawing) -> np.ndarray:
    a, b, c = (drawing.points[drawing.triangles[:, k]] for k in range(3))
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


def test_it_lives_in_plotting_utilities_not_in_the_colour_scales():
    assert plotting_utilities.planar_mesh_drawing is planar_mesh_drawing
    assert plotting_utilities.PlanarMeshDrawing is PlanarMeshDrawing
    assert not {"planar_mesh_drawing", "PlanarMeshDrawing"} & set(dir(colour_scales))


@pytest.mark.parametrize("module", ["foambryo.plotting_utilities", "foambryo.colour_scales"])
def test_importing_the_drawing_modules_does_not_import_polyscope(module):
    # A fresh interpreter: in this one, another test may have imported Polyscope already.
    code = (f"import sys, {module}; from foambryo.plotting_utilities import planar_mesh_drawing; "
            "print(sorted(name for name in sys.modules if name.split('.')[0] == 'polyscope'))")
    out = subprocess.run(  # noqa: S603 — fixed script, the test's own interpreter
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
    ).stdout
    assert out.strip() == "[]"


@pytest.mark.parametrize(("cells", "areas", "centroid_of_2"), [(L_SHAPE, {1: 3.0, 2: 1.0}, (0.5, 1.5)),
                                                               (RING, {1: 8.0, 2: 1.0}, (1.5, 1.5))])
def test_the_triangles_cover_each_cell_counter_clockwise(cells, areas, centroid_of_2):
    drawing = planar_mesh_drawing(_grid_mesh(cells))
    signed = _signed_triangle_areas(drawing)
    assert np.all(signed > 0)
    for cell, area in areas.items():
        assert signed[drawing.triangle_cell == cell].sum() == pytest.approx(area, abs=1e-12)
        assert drawing.areas[cell] == pytest.approx(area, abs=1e-12)
    np.testing.assert_allclose(drawing.centroids[2], centroid_of_2, atol=1e-12)
    total = sum(drawing.areas.values())
    expected = sum(drawing.areas[c] * drawing.centroids[c] for c in drawing.cells) / total
    np.testing.assert_allclose(drawing.tissue_centroid, expected, atol=1e-12)
    assert drawing.cells == (1, 2)
    assert set(drawing.pairs) == {(0, 1), (1, 2)} | ({(0, 2)} if cells is L_SHAPE else set())


def test_it_stays_in_the_mesh_frame_and_unit():
    plain = planar_mesh_drawing(_grid_mesh(RING))
    scaled = planar_mesh_drawing(_grid_mesh(RING, spacing=0.25))
    np.testing.assert_array_equal(scaled.triangles, plain.triangles)
    np.testing.assert_allclose(scaled.points, 0.25 * plain.points, atol=0)
    for cell in plain.cells:
        assert scaled.areas[cell] == pytest.approx(0.0625 * plain.areas[cell], rel=1e-12)


def test_it_is_the_same_from_either_orientation():
    mesh = _grid_mesh(RING)
    reversed_mesh = PlanarMesh(mesh.points, mesh.segments[:, ::-1], mesh.labels)
    one, other = planar_mesh_drawing(mesh), planar_mesh_drawing(reversed_mesh)
    np.testing.assert_array_equal(one.triangles, other.triangles)
    np.testing.assert_array_equal(one.segments, other.segments)
    assert one.areas == other.areas


def test_the_accepted_forms_give_the_same_drawing():
    mesh = _grid_mesh(L_SHAPE)
    from_triple = planar_mesh_drawing((mesh.points, mesh.segments, mesh.labels))
    from_pair = planar_mesh_drawing((mesh.points, mesh.e))
    for drawing in (from_triple, from_pair):
        np.testing.assert_array_equal(drawing.triangles, planar_mesh_drawing(mesh).triangles)


def test_what_it_refuses():
    with pytest.raises(TypeError, match="not a 2-D mesh"):
        planar_mesh_drawing("mesh.rec")
    mesh = _grid_mesh(L_SHAPE)
    with pytest.raises(ValueError, match=r"points must be \(n, 2\)"):
        planar_mesh_drawing((np.c_[mesh.points, np.zeros(len(mesh.points))], mesh.segments, mesh.labels))
    with pytest.raises(ValueError, match="smaller first"):
        planar_mesh_drawing((mesh.points, mesh.segments, mesh.labels[:, ::-1]))


def test_a_cell_with_an_open_boundary_is_drawn_without_triangles():
    mesh = _grid_mesh(L_SHAPE)
    gap = int(np.flatnonzero((mesh.labels == [0, 2]).all(axis=1))[0])
    keep = np.arange(len(mesh.segments)) != gap
    drawing = planar_mesh_drawing(PlanarMesh(mesh.points, mesh.segments[keep], mesh.labels[keep]))
    assert not np.any(drawing.triangle_cell == 2)
    assert drawing.areas[2] == 0.0
    used = np.unique(drawing.segments[(drawing.labels == 2).any(axis=1)])
    np.testing.assert_allclose(drawing.centroids[2], drawing.points[used].mean(axis=0))
    assert drawing.areas[1] == pytest.approx(3.0, abs=1e-12)
