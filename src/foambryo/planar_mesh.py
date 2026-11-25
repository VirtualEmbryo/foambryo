"""foambryo's 2-D mesh, `PlanarMesh`, and the rule that tells a 2-D mesh from a 3-D one.

foambryo's package-level functions (`infer_tensions`, `infer_pressures`, `infer_forces`, ...) take a 3-D mesh
(`foambryo.dcel.DcelData`, triangles) or a 2-D one (`PlanarMesh`, segments) and dispatch on its type
(`dimension_of`). A 2-D mesh is the one delaunay-watershed's 2-D reconstruction (dw2d) builds:

* `points` `(n, 2)` float64, in the mask's index order `(row, column)`, in pixels (or in the unit of an isotropic
  pixel size);
* `segments` `(m, 2)` int64, the two vertex indices of each edge;
* `labels` `(m, 2)` int64, the two materials each edge separates, smaller first (0 is the medium).

**Orientation: one rule in 2D and 3D, in index order.** Each mesh element is the positively oriented boundary of its
first label: a triangle's right-hand normal `(p1 - p0) x (p2 - p0)` in `(z, y, x)` points out of `labels[k, 0]`, and
material `labels[k, 0]` lies on the left of the edge `segments[k, 0] -> segments[k, 1]` in `(row, column)`, so that the
normal `(dc, -dr)` of an edge of tangent `(dr, dc)` points out of `labels[k, 0]` too. dw2d follows it since 0.8.0;
dw2d 0.1.5 and the development versions of 0.8.0 stored every edge the other way round. The 2-D pressures rely on it
(which side of an interface each cell is on).

foambryo does not trust a 2-D mesh's orientation, it measures it: on input (`PlanarMesh`, a mesh file, dw2d's
`DCEL_Data` or arrays given to any 2-D function), the signed area of each cell's boundary, `A = (1/2) sum (r0 c1 -
c0 r1)` over its edges (each counted as stored where the cell is the first label, reversed where it is the second), is
positive for every cell under the one rule and negative for every cell under the former one. A mesh in the former
orientation has every edge reversed, so the results are the same from either; a mesh whose cells disagree is refused
(`ValueError`). Only cells with a closed boundary vote, and the cells cut by the image border only when no other cell
can (dw2d 0.1.5 left its image-border edges in an arbitrary order: a cut cell then votes with its area taken along its
own loops, each image-border edge in the direction its loop runs); a mesh with no closed cell is taken as given.

dw2d's own `DCEL_Data`, any object with `v` `(n, 2)` and `e` `(m, 4)` (`e` = segments and labels side by side), a
`(points, segments, labels)` triple and a `(vertices, edges)` pair are accepted too, and read the same way.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

#: The orientation rule, as delaunay-watershed records it in the appendix of a 2-D `.rec` file (key `orientation`):
#: each element is the positively oriented boundary of its first label in index order (module docstring).
_ORIENTATION = "boundary_of_first_label"


def _vertices_on_the_image_border(points: NDArray[np.float64]) -> NDArray[np.bool_]:
    """The vertices on the sides of the points' bounding box, when its four corners are vertices; otherwise none.

    A dw2d mesh keeps the four image corners as vertices.
    """
    on = np.zeros(len(points), dtype=bool)
    if not len(points):
        return on
    lo, hi = points.min(axis=0), points.max(axis=0)
    corners = {(lo[0], lo[1]), (lo[0], hi[1]), (hi[0], lo[1]), (hi[0], hi[1])}
    if not corners <= {tuple(p) for p in points.tolist()}:
        return on
    return (points[:, 0] == lo[0]) | (points[:, 0] == hi[0]) | (points[:, 1] == lo[1]) | (points[:, 1] == hi[1])


def _orientation_from_geometry(
    points: NDArray[np.floating],
    segments: NDArray[np.integer],
    labels: NDArray[np.integer],
    image_border_vertices: Sequence[int] | NDArray[np.integer] | None = None,
) -> str | None:
    """Which orientation rule a 2-D mesh follows, measured from its geometry (the module docstring's test).

    Each cell's boundary signed area in `(row, column)`, each edge counted as stored for its first label and reversed
    for its second, is positive for every cell when each edge is the positive boundary of its first label (the one
    rule), negative for every cell when it is the positive boundary of its second (dw2d before 0.8.0). Voters: the cells
    (labels other than 0) whose edges close (every vertex meets an even number of them) and enclose a nonzero area;
    among them the cells with no image-border edge (both ends on the border), unless there are none: then the cut
    cells vote, each with its area taken along its own loops, the image-border edges in the direction their loop runs
    (`_area_along_loops`), since dw2d 0.1.5 stored those edges in an arbitrary order.

    Args:
        points: `(n, 2)` vertex positions, `(row, column)`.
        segments: `(m, 2)` vertex indices.
        labels: `(m, 2)` the two materials of each edge.
        image_border_vertices: the vertices on the image border (dw2d's mark); `None` locates them from the positions
            (the sides of the bounding box, when its corners are vertices).

    Returns:
        `"boundary_of_first_label"` (the one rule), `"boundary_of_second_label"` (dw2d before 0.8.0), or `None` when no
        cell can vote (no closed boundary).

    Raises:
        ValueError: when the voting cells disagree, naming some of each kind.
    """
    points = np.asarray(points, dtype=np.float64)
    segments = np.asarray(segments, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if not len(segments):
        return None
    start, end = points[segments[:, 0]], points[segments[:, 1]]
    cross = start[:, 0] * end[:, 1] - start[:, 1] * end[:, 0]
    material = np.concatenate([labels[:, 0], labels[:, 1]])
    materials, owner = np.unique(material, return_inverse=True)
    area = 0.5 * np.bincount(owner, weights=np.concatenate([cross, -cross]), minlength=len(materials))
    ends = np.concatenate([segments[:, 0], segments[:, 0], segments[:, 1], segments[:, 1]])
    key = np.concatenate([owner, owner]) * np.int64(len(points)) + ends
    incidences, counts = np.unique(key, return_counts=True)
    closed = np.ones(len(materials), dtype=bool)
    closed[np.unique(incidences[counts % 2 == 1] // len(points))] = False
    if image_border_vertices is None:
        on_border = _vertices_on_the_image_border(points)
    else:
        on_border = np.zeros(len(points), dtype=bool)
        on_border[np.asarray(image_border_vertices, dtype=np.int64).ravel()] = True
    border_rows = on_border[segments[:, 0]] & on_border[segments[:, 1]]
    cut = np.isin(materials, labels[border_rows].ravel())
    candidates = (materials != 0) & closed & (area != 0)
    voters = candidates & ~cut
    if not voters.any():
        # Only cut cells can vote. dw2d 0.1.5 stored its image-border edges in an arbitrary order, so each cut cell's
        # area is taken along its own loops: every border edge in the direction its loop runs (`_area_along_loops`).
        voters = candidates
        for index in np.flatnonzero(candidates):
            along = _area_along_loops(points, segments, labels, int(materials[index]), border_rows)
            if along is not None:
                area[index] = along
        voters = voters & (area != 0)
    if not voters.any():
        return None
    positive, negative = materials[voters & (area > 0)], materials[voters & (area < 0)]
    if not len(negative):
        return _ORIENTATION
    if not len(positive):
        return "boundary_of_second_label"
    error = (
        "the edges of this 2-D mesh follow no single orientation rule: in (row, column), the boundary of "
        f"{len(positive)} cell(s) (e.g. {', '.join(map(str, positive[:5].tolist()))}) runs counter-clockwise and "
        f"that of {len(negative)} cell(s) (e.g. {', '.join(map(str, negative[:5].tolist()))}) clockwise, each edge "
        "counted as "
        "stored for its first label and reversed for its second. dw2d's meshes follow one rule (since 0.8.0 each edge "
        "is the positively oriented boundary of its first label; before, of its second), so this mesh was edited or "
        "assembled otherwise; foambryo cannot tell which side of an interface each cell is on, and refuses it"
    )
    raise ValueError(error)


def _area_along_loops(
    points: NDArray[np.float64],
    segments: NDArray[np.int64],
    labels: NDArray[np.int64],
    cell: int,
    border_rows: NDArray[np.bool_],
) -> float | None:
    """A cut cell's boundary signed area with each image-border edge taken in the direction its loop runs.

    The cell's edges, each as stored where the cell is the first label and reversed where it is the second, are
    followed around each loop from an edge that is not on the image border, in its stored direction; the border edges,
    whose stored direction dw2d 0.1.5 left arbitrary, are taken as the loop runs. The area is the shoelace sum along
    the loops. `None` when the edges do not form simple loops (a vertex shared by more than two of them), when a loop
    has no edge off the border, or when the edges off the border disagree with their loop (then the raw area decides).
    """
    rows = np.flatnonzero((labels[:, 0] == cell) | (labels[:, 1] == cell))
    directed = segments[rows].copy()
    second = labels[rows, 1] == cell
    directed[second] = directed[second][:, ::-1]
    on_border = border_rows[rows]
    at: dict[int, list[int]] = {}
    for i, (a, b) in enumerate(directed.tolist()):
        at.setdefault(a, []).append(i)
        at.setdefault(b, []).append(i)
    if any(len(edges) != 2 for edges in at.values()):
        return None
    seen = np.zeros(len(rows), dtype=bool)
    total = 0.0
    for first in range(len(rows)):
        if seen[first] or on_border[first]:
            continue
        tail, head = (int(v) for v in directed[first])
        loop, i = [tail], first
        while not seen[i]:
            seen[i] = True
            if not on_border[i] and (int(directed[i][0]), int(directed[i][1])) != (tail, head):
                return None
            loop.append(head)
            nxt = at[head][0] if at[head][1] == i else at[head][1]
            a, b = (int(v) for v in directed[nxt])
            tail, head, i = head, (b if a == head else a), nxt
        ring = points[np.asarray(loop[:-1])]
        total += 0.5 * float(np.sum(ring[:, 0] * np.roll(ring[:, 1], -1) - ring[:, 1] * np.roll(ring[:, 0], -1)))
    if not seen.all():
        return None
    return total


def _segments_in_the_one_orientation(
    points: NDArray[np.floating],
    segments: NDArray[np.integer],
    labels: NDArray[np.integer],
    image_border_vertices: Sequence[int] | NDArray[np.integer] | None = None,
    declared: str | None = None,
) -> tuple[NDArray[np.int64], bool]:
    """`segments` in the one orientation rule, and whether they were reversed (`_orientation_from_geometry`).

    `declared` is the rule a file records (the `.rec` appendix's `orientation`), or `None`: an unknown rule is refused,
    and so is a mesh whose geometry contradicts the rule its file records.

    Raises:
        ValueError: for an inconsistent mesh, an unknown declared rule, or a geometry that contradicts it.
    """
    segments = np.asarray(segments, dtype=np.int64)
    if declared is not None and declared != _ORIENTATION:
        error = f"unknown orientation rule {declared!r} (foambryo reads {_ORIENTATION!r}, or a file that records none)"
        raise ValueError(error)
    measured = _orientation_from_geometry(points, segments, labels, image_border_vertices)
    if measured != "boundary_of_second_label":
        return segments, False
    if declared is not None:
        error = (
            f"the file records orientation = {declared!r}, but its geometry says each edge bounds its second label: "
            "the file is inconsistent"
        )
        raise ValueError(error)
    return np.ascontiguousarray(segments[:, ::-1]), True


class PlanarMesh:
    """A 2-D multimaterial polyline mesh: points, segments and the two materials each segment separates.

    Attributes:
        points: `(n, 2)` float64 vertex positions, `(row, column)`.
        segments: `(m, 2)` int64 vertex indices.
        labels: `(m, 2)` int64 materials, smaller first.
        image_border_vertices: sorted int64 indices of the vertices on the image border (dw2d's mark), or `None`
            when unknown; the 2-D inference excludes the junctions there by default.
        image_shape: `(rows, columns)` of the label image the mesh was built from, or `None`.
        spacing: the pixel size the points were scaled by (`None`: the points are in pixels).
        unit: the unit of `spacing`, or `None`.
        v, e: the points, and segments and labels side by side `(m, 4)`, under the names of dw2d's `DCEL_Data`.
    """

    def __init__(
        self,
        points: NDArray[np.floating],
        segments: NDArray[np.integer],
        labels: NDArray[np.integer],
        *,
        image_border_vertices: Sequence[int] | NDArray[np.integer] | None = None,
        image_shape: tuple[int, int] | None = None,
        spacing: float | None = None,
        unit: str | None = None,
    ) -> None:
        """Check and store the arrays (copied to float64 and int64).

        The segments are stored in the one orientation rule: given in the former one, every segment is reversed (the
        module docstring says how foambryo measures it; `_reversed_on_input` records it).

        Raises:
            ValueError: if `points` is not `(n, 2)`, `segments` and `labels` not `(m, 2)`, an index is out of range, a
                row of `labels` is not in ascending order, or the cells' boundaries follow no single orientation.
        """
        points = np.array(points, dtype=np.float64)
        segments = np.array(segments, dtype=np.int64)
        labels = np.array(labels, dtype=np.int64)
        if points.ndim != 2 or points.shape[1] != 2:
            error = f"points must be (n, 2), got {points.shape}"
            raise ValueError(error)
        if segments.ndim != 2 or segments.shape[1] != 2 or labels.shape != segments.shape:
            error = f"segments and labels must both be (m, 2), got {segments.shape} and {labels.shape}"
            raise ValueError(error)
        if len(segments) and (segments.min() < 0 or segments.max() >= len(points)):
            error = f"a segment names a vertex outside 0..{len(points) - 1}"
            raise ValueError(error)
        if len(labels) and np.any(labels[:, 0] >= labels[:, 1]):
            error = "every row of labels must hold two different materials, smaller first"
            raise ValueError(error)
        border = None if image_border_vertices is None else np.unique(np.asarray(image_border_vertices, dtype=np.int64))
        segments, self._reversed_on_input = _segments_in_the_one_orientation(points, segments, labels, border)
        self.points = points
        self.segments = segments
        self.labels = labels
        self.image_border_vertices = border
        self.image_shape = None if image_shape is None else (int(image_shape[0]), int(image_shape[1]))
        self.spacing = None if spacing is None else float(spacing)
        self.unit = unit
        self._edges = np.hstack((segments, labels))

    @property
    def v(self) -> NDArray[np.float64]:
        """The points, under the name of dw2d's `DCEL_Data`."""
        return self.points

    @property
    def e(self) -> NDArray[np.int64]:
        """`(m, 4)` rows `[v1, v2, m1, m2]`: segments and labels side by side, as dw2d's `DCEL_Data.e`."""
        return self._edges

    @property
    def materials(self) -> tuple[int, ...]:
        """Every material of the mesh, ascending (0, the medium, included when present)."""
        return tuple(int(m) for m in np.unique(self.labels))

    @property
    def cells(self) -> tuple[int, ...]:
        """The cells: every material but 0."""
        return tuple(m for m in self.materials if m != 0)

    @property
    def media(self) -> tuple[int, ...]:
        """The media: `(0,)` (enclosed media are not labelled in 2D)."""
        return (0,)

    @property
    def unresolved(self) -> tuple[int, ...]:
        """Labels of the mask absent from the mesh: not recorded on a `PlanarMesh` (empty)."""
        return ()

    @property
    def cut_cells(self) -> tuple[int, ...]:
        """The cells with a vertex on the image border (dw2d's `cut_cells`), ascending; empty when the border is not
        known (`image_border_vertices` is None)."""
        if self.image_border_vertices is None or not len(self.segments):
            return ()
        on_border = np.zeros(len(self.points), dtype=bool)
        on_border[self.image_border_vertices] = True
        touching = on_border[self.segments].any(axis=1)
        return tuple(int(m) for m in np.unique(self.labels[touching]) if m != 0)

    def __repr__(self) -> str:
        return (
            f"PlanarMesh({len(self.points)} points, {len(self.segments)} segments, {len(self.cells)} cells"
            + ("" if self.spacing is None else f", spacing {self.spacing:g} {self.unit or ''}".rstrip())
            + ")"
        )

    @classmethod
    def from_any(cls, mesh: object) -> PlanarMesh:
        """A `PlanarMesh` from any accepted 2-D mesh (see the module docstring); a `PlanarMesh` is returned as is."""
        if isinstance(mesh, PlanarMesh):
            return mesh
        if hasattr(mesh, "v") and hasattr(mesh, "e"):
            edges = np.asarray(mesh.e)
            return cls(
                mesh.v,
                edges[:, :2],
                edges[:, 2:],
                image_border_vertices=getattr(mesh, "image_border_vertices", None),
            )
        if isinstance(mesh, (tuple, list)) and len(mesh) == 3:
            return cls(*mesh)
        if isinstance(mesh, (tuple, list)) and len(mesh) == 2:
            edges = np.asarray(mesh[1])
            return cls(mesh[0], edges[:, :2], edges[:, 2:])
        error = f"not a 2-D mesh: {type(mesh).__name__}"
        raise TypeError(error)


def _shape(value: object) -> tuple[int, ...] | None:
    try:
        return np.shape(value)
    except (TypeError, ValueError):
        return None


def dimension_of(mesh: object) -> int:
    """3 for a 3-D mesh (`DcelData`), 2 for a 2-D one; the rule every package-level function dispatches on.

    * 3: a `foambryo.dcel.DcelData`, or any object with a face array `f` (the 3-D half-edge layout).
    * 2: a `PlanarMesh`; an object with `v` `(n, 2)` and `e` `(m, 4)` (dw2d's `DCEL_Data`); a `(points (n, 2),
      segments (m, 2), labels (m, 2))` triple; a `(vertices (n, 2), edges (m, 4))` pair.

    Raises:
        TypeError: for anything else, naming the accepted types.
    """
    from foambryo.dcel import DcelData

    if isinstance(mesh, DcelData):
        return 3
    if isinstance(mesh, PlanarMesh):
        return 2
    if hasattr(mesh, "f") and hasattr(mesh, "v"):
        return 3
    if hasattr(mesh, "v") and hasattr(mesh, "e"):
        v, e = _shape(mesh.v), _shape(mesh.e)
        if v is not None and e is not None and len(v) == 2 and v[1] == 2 and len(e) == 2 and e[1] == 4:
            return 2
    if isinstance(mesh, (tuple, list)) and len(mesh) in (2, 3):
        shapes = [_shape(a) for a in mesh]
        if all(s is not None and len(s) == 2 for s in shapes) and shapes[0][1] == 2:
            widths = [s[1] for s in shapes[1:]]
            if widths in ([4], [2, 2]):
                return 2
    error = (
        f"expected a mesh: a foambryo DcelData (3-D) or a 2-D mesh (foambryo.PlanarMesh, dw2d's DCEL_Data, a "
        f"(points, segments, labels) triple or a (vertices, edges) pair), got {type(mesh).__name__}"
    )
    raise TypeError(error)


def require_three_dimensions(mesh: object, function: str) -> None:
    """Refuse a 2-D mesh in a function that has no 2-D counterpart yet.

    Raises:
        ValueError: naming the function, when `mesh` is a 2-D mesh.
    """
    if not hasattr(mesh, "f") and dimension_of(mesh) == 2:
        error = f"{function} has no 2-D counterpart yet: it takes a 3-D mesh (DcelData), got a 2-D mesh"
        raise ValueError(error)
