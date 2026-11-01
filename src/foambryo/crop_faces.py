r"""Cropped tissues: what a crop face is, and what inference does with it.

The problem
-----------
A tissue imaged in part, or cropped from a larger image, has cells that the image boundary cuts.
The mesh closes them with triangles lying on the boundary: **crop faces**, labelled `(0, c)` like
contact with the medium, although nothing beyond them was observed. Read as medium contact, they
make three kinds of error:

* a tension unknown for an interface that does not exist (the crop), and a fake junction line
  `(0, a, b)` wherever the crop meets a cell-cell interface `(a, b)`, with angles set by the crop
  plane rather than by the tissue;
* force balances written at vertices on the crop, where the tissue that was cut away pulled;
* a pressure for every cut cell, from a volume that is partly the crop.

Measured on 433 plane crops of the 47 benchmark tissues (`YoungDupre`, median over the 42
equilibrium cases of the relative error of the gauge-matched tensions of the interfaces that touch
the crop, dw3d 0.7.1's meshes): 14 %, 33 % and 39 % error with 10 %, 25 % and 40 % of the tissue
cropped away when crop faces are read as medium contact, against 10 %, 9 % and 13 % when they are
kept out (13 %, 33 %, 40 % against 10 %, 9 %, 12 % on 0.7.0's meshes); every one of
the 13 tension methods improves at 25 % and 40 %, and `LogLami` stops failing on 25 of the crops.
The interfaces away from the crop are affected too: at 40 % their error was 17 points above the
uncropped tissue's, and 2 points above it with the crop kept out.

What inference does with a crop face
------------------------------------
A mesh carries its crop as a mask of faces (`DcelData(..., crop_faces=...)`), never as a reserved
label. Then, in every tension method:

* a crop face carries no tension unknown and enters no equation; an interface made only of crop
  faces (`DcelData.crop_interfaces`) is **absent** from the tension map, never a number;
* a junction **edge** with a vertex on a crop face gives no equation; the rest of a junction line
  that ends on the crop still does (the angle-reading methods exclude such edges from the line
  mean, as `foambryo.wedge_refusal.WedgeRefusalPolicy.ExcludeEdgeFromLineMean` does for a folded
  edge, and both reasons combine);
* a **vertex** on a crop face gives no force balance (the `Variational` and `ForceBalance*`
  methods drop its three rows; `Equilibrium` drops the edges that touch it);
* an interface left with no equation is `nan`, as an interface with no junction line already is.

And in the pressure methods, a **cut cell** (`DcelData.cut_cells`: a cell with a crop face or a
vertex on one) keeps its pressure as an unknown, so that its force is not pushed into anyone
else's, but the pressure is not reported: it comes back `nan`, with a `CutCellPressureWarning`
naming the cells. On the benchmark crops the cut cells' pressures read as measurements had 15 % to
46 % error; withholding them leaves the other cells' pressures as accurate as without the crop.

What has no formulation for crop faces yet refuses a cropped mesh with a `ValueError`, rather
than read the crop as medium: the two Laplace pressure methods (`PressureComputationMethod.Laplace`
and `WeightedLaplace`; `PressureComputationMethod.has_crop_face_formulation` says which methods
accept a cropped mesh), `foambryo.compute_inference_diagnostics` and
`foambryo.infer_tensions_with_uncertainty`. `infer_force_balance` follows the rules above.

On a mesh without crop faces none of this runs.

Where crop faces come from
--------------------------
`delaunay-watershed-3d` (dw3d) marks them during the reconstruction: the triangles a cell has on a
face of the image box that the cell reaches in the mask, unless that face is declared physical (a
coverslip); a mask none of whose cells reaches the image box has none. They reach foambryo
through `dcel_mesh_from_segmentation_mask`, or through a `.rec` file whose metadata appendix
records `crop_faces = <first row> <count>` (dw3d writes them as the last rows), or a `.vtk` file
with a `crop_face` cell array. A file written before crop faces were recorded carries no such
record; when its mesh has faces that look like a crop, `dcel_mesh_from_file` warns
(`UnrecordedCropFacesWarning`) instead of silently reading them as medium contact.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from foambryo.dcel import DcelData

#: Geometric edge as an ordered `(min, max)` vertex-key pair.
GeometricEdge = tuple[int, int]


class CutCellPressureWarning(UserWarning):
    """Emitted when a pressure inference withholds the pressures of cut cells, reported as `nan`.

    A cut cell has a crop face, or a vertex on one: part of its surface and volume lie outside the
    imaged volume. Its pressure is solved for, so that its force does not end up in the other
    cells' pressures, but it is not a measurement and is reported as `nan`. The other cells'
    pressures are unaffected.
    """


class UnrecordedCropFacesWarning(UserWarning):
    """Emitted when a mesh read from a file looks cropped but the file records no crop faces.

    Such a mesh has triangles `(0, c)` lying flat on a face of its own bounding box, like the faces
    a reconstruction places where the image boundary cuts a cell. Without a record of which they are, they
    are read as contact with the medium, which puts large errors on the tensions near them and on
    the pressures of the cells they close. Re-export the mesh with a current
    delaunay-watershed-3d, or pass the crop faces explicitly (`DcelData(..., crop_faces=...)`).
    """


def crop_edges(mesh: DcelData) -> frozenset[GeometricEdge]:
    """Every mesh edge with a vertex on a crop face, as `(min, max)` vertex-key pairs.

    A superset of the junction edges that touch the crop: the estimators match it against their
    own junction edges only, so the other edges it holds are never consulted.

    Args:
        mesh (DcelData): the mesh.

    Returns:
        frozenset[GeometricEdge]: empty when the mesh has no crop face.
    """
    if not mesh.has_crop_faces:
        return frozenset()
    triangles = mesh.f[:, :3]
    on_crop = np.isin(triangles, mesh.crop_vertices)
    edges: set[GeometricEdge] = set()
    for i, j in ((0, 1), (1, 2), (0, 2)):
        rows = on_crop[:, i] | on_crop[:, j]
        u, w = triangles[rows, i], triangles[rows, j]
        edges.update(zip(np.minimum(u, w).tolist(), np.maximum(u, w).tolist(), strict=True))
    return frozenset(edges)


def vertex_rows_off_the_crop(mesh: DcelData, excluded: NDArray[np.integer] | None = None) -> NDArray[np.intp]:
    """Row indices `3v, 3v + 1, 3v + 2` of every vertex on no crop face and not in `excluded`.

    The rows a vertex-wise force balance keeps on a cropped mesh: a vertex on a crop face is held
    by the tissue that was cut away, so its balance is not an observation.

    Args:
        mesh (DcelData): the mesh.
        excluded (NDArray | None, optional): further vertex keys to leave out.

    Returns:
        NDArray[np.intp]: ascending row indices into a `(3 nv, ...)` operator.
    """
    keep = np.ones(len(mesh.v), dtype=bool)
    keep[mesh.crop_vertices] = False
    if excluded is not None and len(excluded):
        keep[np.asarray(excluded, dtype=np.intp)] = False
    vertices = np.flatnonzero(keep)
    return (3 * vertices[:, None] + np.arange(3)[None, :]).ravel()


def warn_on_cut_cell_pressures(cells: tuple[int, ...], stacklevel: int = 4) -> None:
    """Emit `CutCellPressureWarning` naming the cut cells whose pressure is reported as `nan`.

    Args:
        cells (tuple[int, ...]): labels of the cut cells (nothing is emitted if empty).
        stacklevel (int, optional): passed to `warnings.warn`. Defaults to 4.
    """
    if not cells:
        return
    shown = ", ".join(str(c) for c in cells[:20])
    if len(cells) > 20:
        shown += f", ... ({len(cells)} in total)"
    message = (
        f"the pressure of {len(cells)} cut cell(s) is not reported and is nan: {shown}. These cells "
        "touch the crop faces of the mesh (the boundary of the imaged volume), so part of their "
        "surface and volume was not observed; their pressures were solved for and withheld. The "
        "other pressures are unaffected."
    )
    warnings.warn(message, CutCellPressureWarning, stacklevel=stacklevel)


def require_no_crop_faces(mesh: DcelData, what: str) -> None:
    """Refuse a cropped mesh in a computation that has no crop-aware formulation yet.

    A crop face read as contact with the medium is the error this module exists to prevent, so a
    computation that would read it that way raises instead of returning a number.

    Args:
        mesh (DcelData): the mesh.
        what (str): the computation's name, for the message.

    Raises:
        ValueError: if the mesh has crop faces.
    """
    if not mesh.has_crop_faces:
        return
    message = (
        f"{what} has no formulation for crop faces yet, and this mesh has "
        f"{int(mesh.crop_faces.sum())} of them: it would read them as contact with the medium. "
        "Use infer_tensions (any method) and infer_pressures (Variational), which keep the crop "
        "out of their equations, or pass crop_faces=None to treat the crop as medium knowingly."
    )
    raise ValueError(message)


#: Fraction of a bounding-box face's extent a planar patch must span before it looks like a crop.
BORDER_LIKE_MIN_SPAN = 0.25

#: Area of the largest edge-connected patch of flat medium-contact triangles, as a fraction of the
#: bounding-box face, before that face looks like a crop. A fraction of the face, so it does not
#: depend on the mesh's resolution. Measured on dw3d reconstructions whose faces also span a quarter
#: of the face: uncropped tissues (47 benchmark masks, 141 masks of a resolution sweep, 492 embryo
#: timepoints, a 110-cell tissue, 94 rotated benchmark masks) reach at most 0.025, cropped ones (915
#: plane, corner and oblique crops of the benchmark tissues, 12 crops of larger simulated tissues) at
#: least 0.086; the threshold is their geometric mean. It then separated, without a miss, a
#: 1 866-cell tissue and 8 complete simulated tissues (at most 0) from 207 further crops, coarse,
#: large and real (at least 0.092).
BORDER_LIKE_MIN_PATCH_AREA = 0.046


def _largest_connected_area(triangles: NDArray[np.integer], corners: NDArray[np.floating]) -> float:
    """Area of the largest set of `triangles` connected through shared edges, `corners` their vertices' positions."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(triangles)
    areas = 0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    edges = np.concatenate([np.sort(triangles[:, [a, b]], axis=1) for a, b in ((0, 1), (1, 2), (2, 0))])
    owner = np.tile(np.arange(n), 3)
    _, edge_id = np.unique(edges, axis=0, return_inverse=True)
    order = np.argsort(edge_id.ravel(), kind="stable")
    shared = edge_id.ravel()[order][1:] == edge_id.ravel()[order][:-1]
    first, second = owner[order][:-1][shared], owner[order][1:][shared]
    graph = coo_matrix((np.ones(len(first)), (first, second)), shape=(n, n))
    _, component = connected_components(graph, directed=False)
    return float(np.bincount(component, weights=areas).max())


def border_like_faces(
    points: NDArray[np.floating],
    triangles: NDArray[np.integer],
    labels: NDArray[np.integer],
) -> NDArray[np.bool_]:
    """Triangles that look like crop faces although nothing says they are.

    A crop face, as a reconstruction writes it, is a triangle between the medium and a cell lying
    flat on a face of the volume, so on a face of the mesh's own bounding box. A triangle counts
    here when its labels are `(0, c)` in either order and its three vertices have exactly the
    bounding box's minimum or maximum coordinate on one axis; and a bounding-box face counts only
    when such triangles span at least `BORDER_LIKE_MIN_SPAN` of its extent on both of its other axes
    and the largest patch of them connected through shared edges covers at least
    `BORDER_LIKE_MIN_PATCH_AREA` of the face, so that the few scattered flat triangles an uncropped
    tissue has at its extremes do not count.

    A heuristic, used only to decide whether to warn: it misses the few crop-face vertices a
    junction relocation moved off the plane, and it cannot tell a crop from a large flat contact
    that coincides with the bounding box.

    Args:
        points: `(nv, 3)` vertex positions.
        triangles: `(nf, 3)` triangles.
        labels: `(nf, 2)` the two materials of each triangle.

    Returns:
        NDArray[np.bool_]: `(nf,)`, `True` on the triangles that look like crop faces.
    """
    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    out = np.zeros(len(triangles), dtype=bool)
    if len(triangles) == 0:
        return out
    medium = (labels == 0).any(axis=1)
    low, high = points.min(axis=0), points.max(axis=0)
    extent = np.where(high > low, high - low, 1.0)
    corners = points[triangles]
    for axis in range(3):
        others = [a for a in range(3) if a != axis]
        for level in (low[axis], high[axis]):
            flat = medium & np.all(corners[:, :, axis] == level, axis=1)
            if not flat.any():
                continue
            patch = corners[flat][:, :, others].reshape(-1, 2)
            span = (patch.max(axis=0) - patch.min(axis=0)) / extent[others]
            if not np.all(span >= BORDER_LIKE_MIN_SPAN):
                continue
            face_area = float(extent[others[0]] * extent[others[1]])
            if _largest_connected_area(triangles[flat], corners[flat]) >= BORDER_LIKE_MIN_PATCH_AREA * face_area:
                out |= flat
    return out


def warn_if_crop_faces_unrecorded(
    points: NDArray[np.floating],
    triangles: NDArray[np.integer],
    labels: NDArray[np.integer],
    source: str,
    stacklevel: int = 3,
) -> None:
    """Emit `UnrecordedCropFacesWarning` when a mesh with no crop record has border-like faces.

    Args:
        points: `(nv, 3)` vertex positions, as read.
        triangles: `(nf, 3)` triangles, as read.
        labels: `(nf, 2)` the two materials of each triangle, as read.
        source (str): where the mesh came from, for the message.
        stacklevel (int, optional): passed to `warnings.warn`. Defaults to 3.
    """
    suspicious = border_like_faces(points, triangles, labels)
    if not suspicious.any():
        return
    message = (
        f"{source} records no crop faces, but {int(suspicious.sum())} of its triangles between the "
        "medium and a cell lie flat on a face of the mesh's bounding box, like the faces a "
        "reconstruction places where the image boundary cuts a cell. They are read as contact with the "
        "medium, which biases the tensions near them and the pressures of the cells they close. "
        "Re-export the mesh with a current delaunay-watershed-3d (which records crop faces), or "
        "pass them explicitly with DcelData(..., crop_faces=...)."
    )
    warnings.warn(message, UnrecordedCropFacesWarning, stacklevel=stacklevel)


def remove_crop_interfaces(solution, crop_interfaces: tuple[tuple[int, int], ...]):  # noqa: ANN001, ANN201
    """A tension solution without its crop interfaces, which must have come out undetermined.

    Works on any solution dataclass with `tensions` (aligned with `dict_tensions`' keys),
    `dict_tensions`, `component_of`, `undetermined`, `fragmented` and `n_components`
    (`junction_graph.ComponentTensionSolution`, `per_edge_tension_inference.YoungDupreLocalResult`).
    A crop interface has no equation, so it is undetermined in the solve; removing it changes no
    other value. `fragmented` is re-read without it, so a mesh is not reported fragmented only
    because its crop is.

    Args:
        solution (ComponentTensionSolution | YoungDupreLocalResult): the solution as the
            estimator returned it.
        crop_interfaces (tuple[tuple[int, int], ...]): `DcelData.crop_interfaces`.

    Returns:
        ComponentTensionSolution | YoungDupreLocalResult: a copy of `solution`
            (`dataclasses.replace`) without the crop interfaces, or `solution` itself when there
            are none.

    Raises:
        RuntimeError: if a crop interface received a finite tension, which would mean an
            equation read the crop: a defect, never a result.
    """
    import dataclasses

    if not crop_interfaces:
        return solution
    crop = set(crop_interfaces)
    keys = list(solution.dict_tensions)
    numbered = [k for k in keys if k in crop and np.isfinite(solution.dict_tensions[k])]
    if numbered:
        message = f"crop interface(s) {numbered} received a tension; no equation should read a crop face"
        raise RuntimeError(message)
    keep = np.array([k not in crop for k in keys], dtype=bool)
    undetermined = tuple(k for k in solution.undetermined if k not in crop)
    return dataclasses.replace(
        solution,
        tensions=np.asarray(solution.tensions)[keep],
        dict_tensions={k: v for k, v in solution.dict_tensions.items() if k not in crop},
        component_of={k: c for k, c in solution.component_of.items() if k not in crop},
        undetermined=undetermined,
        fragmented=solution.n_components > 1 or bool(undetermined),
    )


def remove_crop_interfaces_from_map(
    dict_tensions: dict[tuple[int, int], float],
    crop_interfaces: tuple[tuple[int, int], ...],
) -> tuple[NDArray[np.float64], dict[tuple[int, int], float]]:
    """`remove_crop_interfaces` for a method that returns a plain map: `(array, map)` without them.

    Args:
        dict_tensions: interface -> tension, in the method's own (sorted) key order.
        crop_interfaces: `DcelData.crop_interfaces`.

    Returns:
        tuple: the tensions as an array in the map's key order, and the map, both without the
        crop interfaces.

    Raises:
        RuntimeError: if a crop interface received a finite tension.
    """
    crop = set(crop_interfaces)
    numbered = [k for k in dict_tensions if k in crop and np.isfinite(dict_tensions[k])]
    if numbered:
        message = f"crop interface(s) {numbered} received a tension; no equation should read a crop face"
        raise RuntimeError(message)
    kept = {k: v for k, v in dict_tensions.items() if k not in crop}
    return np.array(list(kept.values()), dtype=np.float64), kept
