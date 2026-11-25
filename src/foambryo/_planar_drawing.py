"""What the viewers draw of a 2-D mesh: its interfaces as curves, its cells as filled polygons, its junctions as points.

No Polyscope and no napari here (numpy only), so that the drawing is tested without a display, and shared by
foambryo's Polyscope window and the napari plugin. Its public interface is `foambryo.colour_scales.planar_mesh_drawing`
and the `PlanarMeshDrawing` it returns; the rest is private.

A 2-D mesh (`foambryo.PlanarMesh`) holds points `(row, column)`, segments and the two materials each segment separates.
From it, `planar_mesh_drawing` builds:

* **the interfaces**: every segment, with the index of its label pair; the segments that only close a cut cell along
  the image border (both ends on the image border, the medium 0 on one side: delaunay-watershed-2d closes a cell the
  image cuts this way) are marked `border`: like 3-D crop faces, they carry no value;
* **the junctions**: the vertices where three or more segments meet;
* **the cells as triangles**: each cell's boundary is chained into closed loops; the largest loop is the cell's
  outline and a loop inside it a hole (an enclosed cell or pocket of medium), joined to the outline by a bridge;
  the polygon is then cut into triangles by ear clipping, which is exact for any simple polygon, convex or not.
  Polyscope's own polygon faces are triangulated as fans, which is wrong for a non-convex cell. A polygon that ear
  clipping cannot cut (a cell that touches itself at a vertex) falls back on the triangles of a Delaunay
  triangulation of its loops whose centres lie inside it (even-odd rule), which covers it up to slivers along a
  concave side;
* **each cell's area and area centroid**, from its triangles, and the displacement of the exploded view: each cell
  moved by `coeff x (its centroid - the tissue's centroid)`, the tissue's centroid being the area-weighted mean of the
  cells' centroids (the 2-D counterpart of `DcelData.compute_scattered_arrays`).

The arrays stay in the mesh's frame and unit: `(row, column)` times the pixel size. The viewers decide where to draw
them (`foambryo._planar_viewer`: `(x, y) = (column, -row)`, rows downward as in napari and matplotlib).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "PlanarMeshDrawing",
    "image_border_segments",
    "junction_vertices",
    "planar_mesh_drawing",
    "triangulate_polygon",
]


def image_border_segments(mesh: object) -> NDArray[np.bool_]:
    """The segments that close a cut cell along the image border: both ends on the border, the medium on one side.

    delaunay-watershed-2d marks the vertices on the image border (`image_border_vertices`); a mesh without that mark
    (`None`) has no such segment here.
    """
    segments = np.asarray(mesh.segments, dtype=np.int64)
    labels = np.asarray(mesh.labels, dtype=np.int64)
    border = getattr(mesh, "image_border_vertices", None)
    if border is None or not len(segments):
        return np.zeros(len(segments), dtype=bool)
    on_border = np.zeros(len(mesh.points), dtype=bool)
    on_border[np.asarray(border, dtype=np.int64)] = True
    return on_border[segments].all(axis=1) & (labels[:, 0] == 0)


def junction_vertices(segments: NDArray[np.integer], n_points: int) -> NDArray[np.int64]:
    """The vertices where three or more segments meet, ascending."""
    degree = np.bincount(np.asarray(segments, dtype=np.int64).reshape(-1), minlength=n_points)
    return np.flatnonzero(degree >= 3).astype(np.int64)


def _signed_area(polygon: NDArray[np.float64]) -> float:
    """The signed area of a closed polygon `(n, 2)` (positive counter-clockwise in its own axes)."""
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _points_in_triangle(points: NDArray[np.float64], a: NDArray, b: NDArray, c: NDArray) -> NDArray[np.bool_]:
    """Which of `points` lie inside or on the counter-clockwise triangle `a, b, c`."""
    def side(p: NDArray, q: NDArray) -> NDArray:
        return (q[0] - p[0]) * (points[:, 1] - p[1]) - (q[1] - p[1]) * (points[:, 0] - p[0])

    eps = -1e-12
    return (side(a, b) >= eps) & (side(b, c) >= eps) & (side(c, a) >= eps)


class _EarClippingFailedError(ValueError):
    """No ear found: the polygon is not simple (it touches or crosses itself)."""


def _ear_clip(polygon: NDArray[np.float64]) -> list[tuple[int, int, int]]:  # noqa: C901 - one loop, its cases
    """Triangles `(i, j, k)` (positions in `polygon`, counter-clockwise) covering a simple counter-clockwise polygon.

    Repeated positions (a hole's bridge visits two vertices twice) are allowed: a vertex at the position of a corner
    of the candidate ear does not block it.
    """
    n = len(polygon)
    if n < 3:
        return []
    remaining = list(range(n))
    triangles: list[tuple[int, int, int]] = []
    while len(remaining) > 3:
        m = len(remaining)
        found = False
        for k in range(m):
            i0, i1, i2 = remaining[k - 1], remaining[k], remaining[(k + 1) % m]
            a, b, c = polygon[i0], polygon[i1], polygon[i2]
            cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
            if abs(cross) <= 1e-12 * max(1.0, float(np.abs(polygon).max())) ** 2:
                if np.dot(b - a, c - b) >= 0:  # straight through b: b adds nothing, drop it
                    remaining.pop(k)
                    found = True
                    break
                continue
            if cross < 0:  # reflex
                continue
            others = [r for r in remaining if r not in (i0, i1, i2)]
            if others:
                points = polygon[others]
                coincident = (np.all(points == a, axis=1) | np.all(points == b, axis=1) | np.all(points == c, axis=1))
                if np.any(_points_in_triangle(points, a, b, c) & ~coincident):
                    continue
            triangles.append((i0, i1, i2))
            remaining.pop(k)
            found = True
            break
        if not found:
            message = "no ear: the polygon touches or crosses itself"
            raise _EarClippingFailedError(message)
    a, b, c = (polygon[i] for i in remaining)
    if (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) > 0:
        triangles.append(tuple(remaining))
    return triangles


def _segments_cross(p: NDArray, q: NDArray, starts: NDArray, ends: NDArray) -> NDArray[np.bool_]:
    """Whether the segment `p q` crosses each segment `starts[i] ends[i]` at an interior point of both."""
    def orient(a: NDArray, b: NDArray, c: NDArray) -> NDArray:
        return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])

    d1 = orient(starts, ends, p)
    d2 = orient(starts, ends, q)
    d3 = orient(p, q, starts)
    d4 = orient(p, q, ends)
    return (d1 * d2 < 0) & (d3 * d4 < 0)


def _bridge_holes(outline: NDArray[np.int64], holes: list[NDArray[np.int64]], points: NDArray) -> NDArray[np.int64]:
    """One vertex sequence: the outline (counter-clockwise) with each hole (clockwise) joined by a two-way bridge.

    Each hole, from the one reaching farthest along x, is joined from its rightmost vertex to the nearest vertex of the
    polygon built so far whose bridge crosses no edge of it nor of the holes left.
    """
    polygon = list(outline)
    for hole in sorted(holes, key=lambda h: -points[h, 0].max()):
        start = int(np.argmax(points[hole, 0]))
        h = hole[start]
        rest = [hh for hh in holes if hh is not hole]
        loop_edges = [np.c_[np.asarray(polygon), np.roll(np.asarray(polygon), -1)]]
        loop_edges += [np.c_[other, np.roll(other, -1)] for other in [hole, *rest]]
        edges = np.vstack(loop_edges)
        order = np.argsort(np.linalg.norm(points[polygon] - points[h], axis=1))
        chosen = None
        for position in order:
            v = polygon[position]
            touching = (edges == v).any(axis=1) | (edges == h).any(axis=1)
            starts, ends = points[edges[~touching, 0]], points[edges[~touching, 1]]
            if not _segments_cross(points[h], points[v], starts, ends).any():
                chosen = int(position)
                break
        if chosen is None:
            message = "no bridge to a hole"
            raise _EarClippingFailedError(message)
        rotated = list(np.roll(hole, -start))
        polygon = polygon[: chosen + 1] + rotated + [h] + polygon[chosen:]
    return np.asarray(polygon, dtype=np.int64)


def _inside_even_odd(points: NDArray[np.float64], loops: list[NDArray[np.float64]]) -> NDArray[np.bool_]:
    """Which points lie inside the region the loops bound, by the even-odd rule (a ray along +x)."""
    inside = np.zeros(len(points), dtype=bool)
    x, y = points[:, 0][:, None], points[:, 1][:, None]
    for loop in loops:
        a, b = loop, np.roll(loop, -1, axis=0)
        straddle = (a[:, 1] > y) != (b[:, 1] > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            at = a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1])
        inside ^= (straddle & (x < at)).sum(axis=1) % 2 == 1
    return inside


def _delaunay_fill(loops: list[NDArray[np.int64]], points: NDArray[np.float64]) -> list[tuple[int, int, int]]:
    """The triangles of a Delaunay triangulation of the loops' vertices whose centres lie inside the loops."""
    from scipy.spatial import Delaunay

    vertices = np.unique(np.concatenate(loops))
    if len(vertices) < 3:
        return []
    try:
        triangulation = Delaunay(points[vertices])
    except Exception:  # collinear or too few points: nothing to fill
        return []
    simplices = vertices[triangulation.simplices]
    centres = points[simplices].mean(axis=1)
    kept = simplices[_inside_even_odd(centres, [points[loop] for loop in loops])]
    out = []
    for i, j, k in kept:
        if _signed_area(points[[i, j, k]]) < 0:
            j, k = k, j
        out.append((int(i), int(j), int(k)))
    return out


def triangulate_polygon(loops: list[NDArray[np.int64]], points: NDArray[np.float64]) -> list[tuple[int, int, int]]:
    """Triangles (vertex indices into `points`, counter-clockwise in `points`' own axes) covering the region of `loops`.

    The largest loop is the outline; each other loop lying inside an outline is a hole of it, and one lying outside
    every outline is another piece. Ear clipping (with holes bridged) is used, the Delaunay fill when it fails.

    Args:
        loops: closed vertex sequences (the last vertex joins the first), any orientation.
        points: `(n, 2)` positions.

    Returns:
        list of `(i, j, k)` triangles.
    """
    points = np.asarray(points, dtype=np.float64)
    loops = [np.asarray(loop, dtype=np.int64) for loop in loops if len(loop) >= 3]
    loops.sort(key=lambda loop: -abs(_signed_area(points[loop])))
    pieces: list[tuple[NDArray[np.int64], list[NDArray[np.int64]]]] = []
    for loop in loops:
        probe = points[loop].mean(axis=0, keepdims=True)
        holder = next((piece for piece in pieces if _inside_even_odd(probe, [points[piece[0]]])[0]), None)
        if holder is None:
            pieces.append((loop, []))
        else:
            holder[1].append(loop)
    triangles: list[tuple[int, int, int]] = []
    for outline, holes in pieces:
        outline = outline if _signed_area(points[outline]) > 0 else outline[::-1]
        holes = [hole if _signed_area(points[hole]) < 0 else hole[::-1] for hole in holes]
        try:
            sequence = _bridge_holes(outline, holes, points) if holes else outline
            triangles += [(int(sequence[i]), int(sequence[j]), int(sequence[k]))
                          for i, j, k in _ear_clip(points[sequence])]
        except _EarClippingFailedError:
            triangles += _delaunay_fill([outline, *holes], points)
    return triangles


def _cell_loops(segments: NDArray[np.int64], n_points: int) -> list[NDArray[np.int64]]:
    """The closed loops a cell's boundary segments form (each segment used once)."""
    neighbours: dict[int, list[tuple[int, int]]] = {}
    for index, (u, v) in enumerate(segments):
        neighbours.setdefault(int(u), []).append((int(v), index))
        neighbours.setdefault(int(v), []).append((int(u), index))
    used = np.zeros(len(segments), dtype=bool)
    loops = []
    for start_index in range(len(segments)):
        if used[start_index]:
            continue
        used[start_index] = True
        first, current = (int(x) for x in segments[start_index])
        loop = [first]
        while current != first:
            loop.append(current)
            step = next(((w, i) for w, i in neighbours[current] if not used[i]), None)
            if step is None:  # an open chain (the boundary is not closed): not a loop
                loop = []
                break
            used[step[1]] = True
            current = step[0]
        if len(loop) >= 3:
            loops.append(np.asarray(loop, dtype=np.int64))
    del n_points
    return loops


@dataclass(frozen=True)
class PlanarMeshDrawing:
    """A 2-D mesh as the viewers draw it, in the mesh's frame `(row, column)` and unit (`planar_mesh_drawing`).

    Provisional, as every 2-D name: it may change in any release until it is declared stable.

    Attributes:
        points: `(n, 2)` the mesh's points (unchanged; some are on no segment).
        segments: `(m, 2)` the mesh's segments.
        labels: `(m, 2)` the two materials of each segment, smaller first.
        pairs: the interfaces, the label pairs `(label_1, label_2)` ascending.
        segment_pair: `(m,)` each segment's index into `pairs`.
        border: `(m,)` the segments that close a cut cell along the image border (no value).
        junctions: the vertices where three or more segments meet.
        cells: the cell labels (every material but 0), ascending.
        triangles: `(k, 3)` the cells' triangles, vertex indices into `points`, counter-clockwise in `(row, column)`.
        triangle_cell: `(k,)` the cell of each triangle.
        areas: each cell's area (the sum of its triangles'), in the mesh's unit squared.
        centroids: each cell's area centroid, `(row, column)`.
        tissue_centroid: the area-weighted mean of the cells' centroids.
    """

    points: NDArray[np.float64]
    segments: NDArray[np.int64]
    labels: NDArray[np.int64]
    pairs: tuple[tuple[int, int], ...]
    segment_pair: NDArray[np.int64]
    border: NDArray[np.bool_]
    junctions: NDArray[np.int64]
    cells: tuple[int, ...]
    triangles: NDArray[np.int64]
    triangle_cell: NDArray[np.int64]
    areas: dict[int, float]
    centroids: dict[int, NDArray[np.float64]]
    tissue_centroid: NDArray[np.float64]

    @property
    def extent(self) -> float:
        """The largest side of the box of the points on a segment, in the mesh's unit (1 when there is none)."""
        used = np.unique(self.segments)
        if not len(used):
            return 1.0
        box = np.ptp(self.points[used], axis=0)
        return float(box.max()) if box.max() > 0 else 1.0

    def displacements(self, coeff: float) -> dict[int, NDArray[np.float64]]:
        """Each cell's move in the exploded view: `coeff x (its centroid - the tissue's centroid)`."""
        return {cell: coeff * (centroid - self.tissue_centroid) for cell, centroid in self.centroids.items()}

    def interface_sides(self) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Each segment once per cell on either side (the medium has no side), for the exploded view.

        Returns:
            tuple: the segment index of each side, and the cell it belongs to.
        """
        first = np.flatnonzero(self.labels[:, 0] > 0)
        second = np.flatnonzero(self.labels[:, 1] > 0)
        rows = np.concatenate([first, second])
        cells = np.concatenate([self.labels[first, 0], self.labels[second, 1]])
        order = np.argsort(rows, kind="stable")
        return rows[order], cells[order]


def planar_mesh_drawing(mesh: object) -> PlanarMeshDrawing:
    """What the viewers draw of a 2-D mesh: its cells cut into triangles, its interfaces, junctions and border segments.

    Provisional, as every 2-D name: it may change in any release until it is declared stable. foambryo's Polyscope
    window and the napari plugin draw a 2-D mesh from it, so the two draw the same polygons. numpy and SciPy only: it
    needs neither Polyscope nor a display.

    Each cell's boundary segments are chained into closed loops; the loop of largest area is the cell's outline, a
    loop inside it a hole (an enclosed cell or pocket of medium, joined to the outline by a bridge), a loop outside it
    another piece of the cell. Each piece is cut into triangles by ear clipping, exact for any simple polygon, convex
    or not (Polyscope's own polygon faces are fans, wrong for a non-convex cell). The triangles of a cell cover it
    without overlap, so their areas sum to the area its boundary encloses.

    Args:
        mesh: a 2-D mesh: a `foambryo.PlanarMesh`, delaunay-watershed-2d's `DCEL_Data`, an object with `v` `(n, 2)`
            and `e` `(m, 4)`, a `(points, segments, labels)` triple or a `(vertices, edges)` pair
            (`PlanarMesh.from_any`). Its orientation is measured and, if it is the former one, every segment is
            reversed first (`foambryo.planar_mesh`); the drawing is the same from either orientation.

    Returns:
        PlanarMeshDrawing: the mesh's points and segments unchanged, and the triangles (vertex indices into `points`,
        counter-clockwise in `(row, column)`), the cell of each, the interfaces as label pairs, the image-border
        segments, the junctions, each cell's area and area centroid and the tissue's centroid. **Frame and unit**: the
        mesh's own, `(row, column)` in pixels, or in the unit of its pixel size when the mesh was scaled
        (`PlanarMesh.spacing`); areas in that unit squared. Nothing is rotated or flipped: where to draw `(row,
        column)` is the viewer's choice (foambryo's Polyscope window draws `(x, y) = (column, -row)`, napari draws
        `(row, column)` as its own axes).

    Raises:
        TypeError: `mesh` is not a 2-D mesh of an accepted form (a 3-D `DcelData` among others).
        ValueError: the arrays are malformed (shapes, an index out of range, a row of `labels` not ascending) or the
            cells' boundaries follow no single orientation (`PlanarMesh`).

    Degenerate cells do not raise; they are drawn as well as their boundary allows:

    * a boundary that touches or crosses itself (ear clipping finds no ear) is filled by the triangles of a Delaunay
      triangulation of its loops' vertices whose centres lie inside it (even-odd rule), which may leave slivers along a
      concave side uncovered;
    * segments that do not close into a loop (an open chain) are left out of the cell's triangles;
    * a cell with no closed loop has no triangle, an area of 0 and, as its centroid, the mean of its vertices.
    """
    from foambryo.planar_mesh import PlanarMesh

    mesh = PlanarMesh.from_any(mesh)
    points, segments, labels = mesh.points, mesh.segments, mesh.labels
    unique, segment_pair = np.unique(labels, axis=0, return_inverse=True) if len(labels) else (
        np.zeros((0, 2), np.int64), np.zeros(0, np.int64),
    )
    pairs = tuple((int(a), int(b)) for a, b in unique)
    cells = tuple(int(m) for m in np.unique(labels) if m != 0)
    triangles: list[tuple[int, int, int]] = []
    triangle_cell: list[int] = []
    areas: dict[int, float] = {}
    centroids: dict[int, NDArray[np.float64]] = {}
    for cell in cells:
        rows = np.flatnonzero((labels == cell).any(axis=1))
        # Each segment walked from its second vertex: the loops, hence the triangles, of the walk on the segments as
        # stored before the 2-D and 3-D orientation rules were unified (every segment reversed since), bit for bit.
        cell_triangles = triangulate_polygon(_cell_loops(segments[rows][:, ::-1], len(points)), points)
        corners = points[np.asarray(cell_triangles, dtype=np.int64).reshape(-1, 3)] if cell_triangles else np.zeros(
            (0, 3, 2))
        a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
        tri_area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        area = float(tri_area.sum())
        areas[cell] = area
        if area > 0:
            centroids[cell] = (tri_area[:, None] * corners.mean(axis=1)).sum(axis=0) / area
        else:
            used = np.unique(segments[rows])
            centroids[cell] = points[used].mean(axis=0) if len(used) else np.zeros(2)
        triangles += cell_triangles
        triangle_cell += [cell] * len(cell_triangles)
    total = sum(areas.values())
    tissue = (sum(areas[c] * centroids[c] for c in cells) / total) if total > 0 else (
        np.mean([centroids[c] for c in cells], axis=0) if cells else np.zeros(2))
    return PlanarMeshDrawing(
        points=points,
        segments=segments,
        labels=labels,
        pairs=pairs,
        segment_pair=np.asarray(segment_pair, dtype=np.int64).reshape(-1),
        border=image_border_segments(mesh),
        junctions=junction_vertices(segments, len(points)),
        cells=cells,
        triangles=np.asarray(triangles, dtype=np.int64).reshape(-1, 3),
        triangle_cell=np.asarray(triangle_cell, dtype=np.int64),
        areas=areas,
        centroids=centroids,
        tissue_centroid=np.asarray(tissue, dtype=np.float64),
    )
