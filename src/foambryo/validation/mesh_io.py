"""Raw multimaterial mesh I/O and combinatorics, for the unified force-balance operator.

The test suite compares these functions with an independent reference implementation of the same algorithms
(`tests/reference/force_balance_reference_implementation.py`). It documents these conventions: regions are
integers; one of them (0 by default) is the external medium and carries no pressure unknown. A mesh is
`(V, F, regions)` with `V` of shape `(nv, 3)` float, `F` of shape `(nf, 3)` int vertex indices, `regions` of shape
`(nf, 2)` int, the two regions each triangle separates.

This module operates on raw `(V, F, regions)` arrays rather than `DcelData`/half-edges,
deliberately: it is the exact input format `force_balance.force_balance_operator`
expects, and keeping it independent of the half-edge structure is what makes the
unit tests in `tests/test_validation_force_balance.py` a direct, literal comparison
against the reference implementation's own outputs.

Two mesh layouts are understood:
    text   -> nv / nv lines "x y z" / nf / nf lines "v0 v1 v2 ra rb"
    binary -> int64 nv / nv*3 float64 / int64 nf / nf*(3 int64 verts + 2 int32 regions), little-endian

An `.arec` file is text. A `.rec` file may be either: the reference implementation wrote it binary, while
delaunay-watershed-3d writes it as text by default (and so does `foambryo mesh`); `read_rec` tells them apart from
the first bytes (see `_rec_is_text`). A metadata appendix after the geometry, as delaunay-watershed-3d writes it, is
ignored by both readers.
"""

from __future__ import annotations

import struct
import warnings
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def read_arec(
    path: str | Path,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Read a text multimaterial mesh (`.arec`).

    Args:
        path (str | Path): Path to the `.arec` file.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]: `(V, F, regions)`.
    """
    with Path(path).open() as f:
        nv = int(f.readline())
        v = np.array([list(map(float, f.readline().split())) for _ in range(nv)])
        nf = int(f.readline())
        rows = np.array([list(map(float, f.readline().split())) for _ in range(nf)])
    return v, rows[:, :3].astype(int), rows[:, 3:5].astype(int)


def _rec_is_text(head: bytes) -> bool:
    """Tell a text `.rec` from a binary one by its first bytes.

    A text file starts with its vertex count as ASCII digits on the first line. A binary file starts with the count
    as a little-endian int64, whose last byte is 0 for any count below 2**56, and a NUL byte never occurs in a text
    file; so a head with no NUL byte in its first 8 bytes and an integer as its first line is text.

    Args:
        head (bytes): the file's first bytes (at least the first line, or 8 bytes, whichever is longer, to decide).

    Returns:
        bool: `True` for a text `.rec`, `False` for a binary one.
    """
    if b"\x00" in head[:8]:
        return False
    return head.split(b"\n", 1)[0].strip().isdigit()


def read_rec(
    path: str | Path,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Read a multimaterial mesh (`.rec`), text or binary (told apart by `_rec_is_text`).

    A text `.rec` (delaunay-watershed-3d's default, and what `foambryo mesh` writes) is read as `read_arec` reads an
    `.arec`; a binary one in the layout of the module docstring. Before 0.7.0 only the binary layout was read.

    Args:
        path (str | Path): Path to the `.rec` file.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]: `(V, F, regions)`.
    """
    raw = Path(path).read_bytes()
    if _rec_is_text(raw[:64]):
        return read_arec(path)
    pos = 0
    nv = struct.unpack_from("<q", raw, pos)[0]
    pos += 8
    v = np.frombuffer(raw, "<f8", count=3 * nv, offset=pos).reshape(nv, 3).copy()
    pos += 3 * nv * 8
    nf = struct.unpack_from("<q", raw, pos)[0]
    pos += 8
    rec = np.frombuffer(raw, np.dtype([("v", "<3i8"), ("r", "<2i4")]), count=nf, offset=pos)
    return v, rec["v"].astype(int), rec["r"].astype(int)


def read_mesh(
    path: str | Path,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Read a mesh, choosing the reader from the extension.

    Falls back to a binary/text sniff of the first bytes when the extension is
    neither `.rec` nor `.arec`. A `.rec` is read by `read_rec`, text or binary.

    Args:
        path (str | Path): Path to the mesh file.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]: `(V, F, regions)`.
    """
    path = Path(path)
    if path.suffix == ".rec":
        return read_rec(path)
    if path.suffix == ".arec":
        return read_arec(path)
    head = path.open("rb").read(16)  # sniff: text meshes start with an integer
    try:
        int(head.split()[0])
    except ValueError:
        return read_rec(path)
    else:
        return read_arec(path)


def is_intact(v: NDArray[np.float64], f: NDArray[np.int64]) -> bool:
    """Check that every face points to an existing vertex (no dangling reference).

    Args:
        v (NDArray[np.float64]): Mesh vertices, shape `(nv, 3)`.
        f (NDArray[np.int64]): Mesh triangles (vertex indices), shape `(nf, 3)`.

    Returns:
        bool: `True` if no vertex index in `f` is out of range for `v`.
    """
    return int(f.max()) < len(v)


def _undirected_edges(f: NDArray[np.int64]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return the mesh's distinct undirected edges and, per edge occurrence, its edge index.

    Each triangle contributes its three edges, sorted so that `(u, v)` has `u < v`; the
    occurrence array is in the order `[all (v0,v1), all (v1,v2), all (v2,v0)]`, so
    occurrence `k` belongs to triangle `k % nf`.

    Args:
        f (NDArray[np.int64]): Mesh triangles (vertex indices), shape `(nf, 3)`.

    Returns:
        tuple[NDArray[np.int64], NDArray[np.int64]]: `(edges, occurrence_to_edge)` with
            `edges` of shape `(n_edges, 2)` sorted lexicographically and
            `occurrence_to_edge` of shape `(3 * nf,)` indexing into it.
    """
    corners = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    corners = np.sort(corners, axis=1)
    edges, occurrence_to_edge = np.unique(corners, axis=0, return_inverse=True)
    return edges, occurrence_to_edge.ravel()


def _count_distinct_per_edge(
    occurrence_to_edge: NDArray[np.int64],
    occurrence_value: NDArray[np.int64],
    n_edges: int,
) -> NDArray[np.int64]:
    """Count, per edge, how many distinct values of `occurrence_value` occur around it.

    Args:
        occurrence_to_edge (NDArray[np.int64]): Edge index of each occurrence.
        occurrence_value (NDArray[np.int64]): Value attached to each occurrence (a region
            label, or an interface index).
        n_edges (int): Number of distinct edges.

    Returns:
        NDArray[np.int64]: Per-edge count of distinct values, shape `(n_edges,)`.
    """
    pairs = np.unique(np.column_stack([occurrence_to_edge, occurrence_value]), axis=0)
    return np.bincount(pairs[:, 0], minlength=n_edges)


def _per_edge_quantities(
    f: NDArray[np.int64],
    regions: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Per-edge region/interface/triangle counts, shared by `mesh_integrity` and `list_junctions`.

    Both callers classify an edge by the same three quantities and must agree on how they
    are computed; this is the single implementation of that computation (see `list_junctions`
    for why having two independently-computed implementations of "is this edge a genuine
    triple line" was a bug in itself).

    Args:
        f (NDArray[np.int64]): Mesh triangles (vertex indices), shape `(nf, 3)`.
        regions (NDArray[np.int64]): Per-triangle region pair, shape `(nf, 2)`.

    Returns:
        tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
            `(edges, occurrence_to_edge, regions_per_edge, interfaces_per_edge,
            triangles_per_edge)`. `edges` has shape `(n_edges, 2)`, `u < v` per row; the four
            count arrays have shape `(n_edges,)`.
    """
    edges, occ_to_edge = _undirected_edges(f)
    n_edges = len(edges)
    triangles_per_edge = np.bincount(occ_to_edge, minlength=n_edges)

    # One occurrence carries two region labels (the pair its triangle separates), so the
    # region-label occurrence array is the edge-occurrence array repeated twice.
    occ_face = np.tile(np.arange(len(f)), 3)
    regions_per_edge = _count_distinct_per_edge(
        np.concatenate([occ_to_edge, occ_to_edge]),
        np.concatenate([regions[occ_face, 0], regions[occ_face, 1]]),
        n_edges,
    )

    interface_index = {key: i for i, key in enumerate(list_interfaces(regions))}
    face_interface = np.array([interface_index[tuple(pair)] for pair in np.sort(regions, axis=1).tolist()])
    interfaces_per_edge = _count_distinct_per_edge(occ_to_edge, face_interface[occ_face], n_edges)

    return edges, occ_to_edge, regions_per_edge, interfaces_per_edge, triangles_per_edge


def mesh_integrity(
    v: NDArray[np.float64],
    f: NDArray[np.int64],
    regions: NDArray[np.int64],
) -> dict:
    """Structural integrity report for a multimaterial mesh.

    Implements the dataset-integrity check — no dangling vertex references, and every triple
    line shared by exactly 3 interfaces — and reports the
    surrounding edge statistics that say *how* a mesh fails when it does.

    `intact` is deliberately narrow: exactly those two conditions. Everything else
    (open boundary edges, quadrijunction edges, non-manifold interface edges, degenerate
    or duplicated triangles) is reported separately, because those are mesh-quality facts
    rather than integrity failures. In particular quadrijunction edges are *expected*
    meshing artifacts, not corruption (in a dry foam exactly three films meet along an edge,
    so a valence-4 edge is an artifact of discretisation).

    Edge terminology, all keyed by the number of *distinct region labels* appearing on the
    triangles incident to an edge. This region-label count alone is only a classification,
    not a validity check: `list_junctions` uses it to name candidate triple-line edges but,
    like this function's own `n_bad_triple_lines` below, only *admits* one once it also
    carries three interfaces on exactly three incident triangles (`triple_line_ok`). A
    non-manifold edge with three region labels but a fourth incident triangle is malformed,
    and is left out of the tension system.

    - 1 region: an edge interior to a single interface's triangulation;
    - 2 regions: an interface edge;
    - 3 regions: a triple-line edge — the physical Plateau junction;
    - >= 4 regions: a quadrijunction edge.

    A triple-line edge spanning regions `{a, b, c}` must carry all three interfaces
    `(a,b)`, `(a,c)`, `(b,c)`, one triangle each. Fewer means the junction is malformed:
    the operator would then assemble a force balance for a junction that is not one.

    Args:
        v (NDArray[np.float64]): Mesh vertices, shape `(nv, 3)`.
        f (NDArray[np.int64]): Mesh triangles (vertex indices), shape `(nf, 3)`.
        regions (NDArray[np.int64]): Per-triangle region pair, shape `(nf, 2)`.

    Returns:
        dict: Integrity report. `intact` is `True` when there is no dangling vertex
            reference and every triple-line edge carries exactly its three interfaces.
    """
    nv, nf = len(v), len(f)
    dangling = not is_intact(v, f)

    n_degenerate_faces = int((f[:, 0] == f[:, 1]).sum() + (f[:, 1] == f[:, 2]).sum() + (f[:, 2] == f[:, 0]).sum())
    sorted_faces = np.sort(f, axis=1)
    n_duplicate_faces = nf - len(np.unique(sorted_faces, axis=0))
    n_unreferenced_vertices = nv - len(np.unique(f)) if nf else nv

    edges, _occ_to_edge, regions_per_edge, interfaces_per_edge, triangles_per_edge = _per_edge_quantities(f, regions)
    n_edges = len(edges)

    is_triple_line = regions_per_edge == 3
    triple_line_ok = (interfaces_per_edge == 3) & (triangles_per_edge == 3)
    n_bad_triple_lines = int((is_triple_line & ~triple_line_ok).sum())

    is_interface_edge = regions_per_edge == 2
    return {
        "n_vertices": nv,
        "n_faces": nf,
        "n_edges": n_edges,
        "intact": not dangling and n_bad_triple_lines == 0,
        "dangling_vertex_reference": dangling,
        "n_unreferenced_vertices": int(n_unreferenced_vertices),
        "n_degenerate_faces": n_degenerate_faces,
        "n_duplicate_faces": int(n_duplicate_faces),
        "n_triple_line_edges": int(is_triple_line.sum()),
        "n_malformed_triple_line_edges": n_bad_triple_lines,
        "n_quadjunction_edges": int((regions_per_edge >= 4).sum()),
        "n_boundary_edges": int((triangles_per_edge == 1).sum()),
        "n_nonmanifold_interface_edges": int((is_interface_edge & (triangles_per_edge != 2)).sum()),
        "edge_region_count_histogram": {
            int(k): int(c) for k, c in zip(*np.unique(regions_per_edge, return_counts=True), strict=True)
        },
    }


def vertex_region_counts(
    v: NDArray[np.float64],
    f: NDArray[np.int64],
    regions: NDArray[np.int64],
) -> NDArray[np.int64]:
    """Number of distinct region labels touching each vertex.

    This is the vertex analogue of the edge classification in `mesh_integrity`, and it is
    what separates the two places a multimaterial mesh can fail force balance:

    - 2 regions: the vertex lies in the *interior* of one interface. Force balance there is
      Young-Laplace, `gamma * 2H = Delta_P`, which a converged smooth interface satisfies
      pointwise to discretisation order.
    - 3 regions: the vertex lies on a triple line. Force balance there is Young-Dupré, and
      a discrete mesh cannot satisfy it exactly — this is where a *healthy* mesh's residual
      is expected to concentrate.
    - >= 4 regions: a quadruple point (or a meshing artifact).

    Args:
        v (NDArray[np.float64]): Mesh vertices, shape `(nv, 3)`.
        f (NDArray[np.int64]): Mesh triangles (vertex indices), shape `(nf, 3)`.
        regions (NDArray[np.int64]): Per-triangle region pair, shape `(nf, 2)`.

    Returns:
        NDArray[np.int64]: Per-vertex count of distinct regions, shape `(nv,)`. Vertices
            referenced by no triangle get 0.
    """
    # One (vertex, region) incidence per triangle corner per region column: 6 per triangle.
    vertex_occurrence = np.concatenate([f[:, corner] for corner in range(3)] * 2)
    region_occurrence = np.concatenate(
        [regions[:, column] for column in range(2) for _ in range(3)],
    )
    pairs = np.unique(np.column_stack([vertex_occurrence, region_occurrence]), axis=0)
    return np.bincount(pairs[:, 0], minlength=len(v))


def cell_labels(regions: NDArray[np.int64], exterior: int = 0) -> list[int]:
    """Sorted list of cell regions (everything except the external medium).

    Args:
        regions (NDArray[np.int64]): Per-triangle region pair, shape `(nf, 2)`.
        exterior (int, optional): Label of the external medium. Defaults to 0.

    Returns:
        list[int]: Sorted cell labels, excluding `exterior`.
    """
    return sorted(r for r in set(regions.flatten().tolist()) if r != exterior)


def list_interfaces(regions: NDArray[np.int64]) -> list[tuple[int, int]]:
    """List the region pairs present in the mesh; one surface-tension unknown each.

    Args:
        regions (NDArray[np.int64]): Per-triangle region pair, shape `(nf, 2)`.

    Returns:
        list[tuple[int, int]]: Sorted, sorted-pair interface keys `(a, b)` with `a < b`.
    """
    return sorted(set(map(tuple, np.sort(regions, axis=1).tolist())))


class MalformedTripleLineWarning(UserWarning):
    """Emitted when `list_junctions` excludes a candidate triple-line edge because it is malformed.

    "Malformed" means: exactly three region labels touch the edge (the classification
    `mesh_integrity` also uses), but it does not also carry three interfaces on exactly
    three incident triangles (the validity rule, `mesh_integrity`'s `triple_line_ok`) -- a
    non-manifold mesh defect, not a modelling choice. Admitted, such an edge would give an
    interface pair that no triangle carries, and `junction_graph.partition_by_junction_graph`
    would fail on it with a `KeyError`; it is dropped from the tension system instead, and this
    warning says so. Use `mesh_integrity` to get the count and locate the edges.
    """


def _count_malformed_triple_line_edges(f: NDArray[np.int64], regions: NDArray[np.int64]) -> int:
    """`mesh_integrity`'s `n_malformed_triple_line_edges`, without requiring vertex positions.

    Shared by `mesh_integrity` (which also reports vertex-based checks `list_junctions`
    doesn't need) and `warn_on_malformed_triple_lines` (called from
    `junction_graph.partition_by_junction_graph`, which is never handed a mesh's vertices).

    Args:
        f (NDArray[np.int64]): Mesh triangles (vertex indices), shape `(nf, 3)`.
        regions (NDArray[np.int64]): Per-triangle region pair, shape `(nf, 2)`.

    Returns:
        int: Number of edges with three region labels that are not a genuine triple line.
    """
    _edges, _occ_to_edge, regions_per_edge, interfaces_per_edge, triangles_per_edge = _per_edge_quantities(
        f, regions,
    )
    is_triple_line = regions_per_edge == 3
    triple_line_ok = (interfaces_per_edge == 3) & (triangles_per_edge == 3)
    return int((is_triple_line & ~triple_line_ok).sum())


def warn_on_malformed_triple_lines(
    f: NDArray[np.int64],
    regions: NDArray[np.int64],
    stacklevel: int = 2,
) -> None:
    """Emit `MalformedTripleLineWarning` if `list_junctions` would exclude any edge.

    Args:
        f (NDArray[np.int64]): Mesh triangles (vertex indices), shape `(nf, 3)`.
        regions (NDArray[np.int64]): Per-triangle region pair, shape `(nf, 2)`.
        stacklevel (int, optional): passed through to `warnings.warn`, so the warning points
            as close to the user's own line as the caller can arrange. Defaults to 2 (the
            immediate caller). Unlike `AngleReadingCoverageWarning` (raised from a single
            fixed entry point two frames from the user), this function's only production
            caller, `junction_graph.partition_by_junction_graph`, is itself reached from
            several different `foambryo.tension_inference` entry points at varying call
            depth -- no fixed `stacklevel` here points at the user's own line in every case.
    """
    n_bad = _count_malformed_triple_line_edges(f, regions)
    if n_bad:
        message = (
            f"{n_bad} triple-line edge(s) excluded from the tension system: three region "
            "labels touch the edge, but not three interfaces on exactly three incident "
            "triangles. This mesh has a non-manifold defect there (see "
            "foambryo.validation.mesh_io.mesh_integrity's n_malformed_triple_line_edges to "
            "locate it) -- the excluded edges are dropped, not guessed at or crashed on."
        )
        warnings.warn(message, MalformedTripleLineWarning, stacklevel=stacklevel)


def list_junctions(
    f: NDArray[np.int64],
    regions: NDArray[np.int64],
) -> tuple[list[tuple[int, int, int]], dict[tuple[int, int], tuple[int, int, int]]]:
    """Find triple lines: edges carrying three interfaces on exactly three incident triangles.

    An edge is admitted as a genuine triple line only under the rule `mesh_integrity`
    already documents and enforces as `triple_line_ok`: three distinct region labels
    (`regions_per_edge == 3`), spanning three distinct interfaces
    (`interfaces_per_edge == 3`), on exactly three incident triangles
    (`triangles_per_edge == 3`) -- not merely three region labels. An edge with three
    region labels but a fourth (or more) incident triangle is a *malformed* triple line
    (`mesh_integrity`'s `n_malformed_triple_line_edges`), not a genuine one, and is excluded:
    admitted, it would give a region-triple whose derived interface pair need not be any
    triangle's actual region pair in the mesh, which
    `junction_graph.partition_by_junction_graph`'s union-find has no key for (a `KeyError`).

    Args:
        f (NDArray[np.int64]): Mesh triangles (vertex indices), shape `(nf, 3)`.
        regions (NDArray[np.int64]): Per-triangle region pair, shape `(nf, 2)`.

    Returns:
        tuple[list[tuple[int, int, int]], dict[tuple[int, int], tuple[int, int, int]]]:
            - sorted region-triples, one per distinct genuine triple line;
            - map from mesh edge `(u, v)` (`u < v`) to its region-triple, genuine triple
              lines only.
    """
    edges, occ_to_edge, regions_per_edge, interfaces_per_edge, triangles_per_edge = _per_edge_quantities(f, regions)
    is_genuine_triple_line = (regions_per_edge == 3) & (interfaces_per_edge == 3) & (triangles_per_edge == 3)

    occ_face = np.tile(np.arange(len(f)), 3)
    edge_labels: dict[int, set[int]] = defaultdict(set)
    for edge_idx, label in zip(
        np.concatenate([occ_to_edge, occ_to_edge]).tolist(),
        np.concatenate([regions[occ_face, 0], regions[occ_face, 1]]).tolist(),
        strict=True,
    ):
        edge_labels[edge_idx].add(int(label))

    junctions: set[tuple[int, int, int]] = set()
    edge_to_triple: dict[tuple[int, int], tuple[int, int, int]] = {}
    for edge_idx in np.flatnonzero(is_genuine_triple_line).tolist():
        tri = tuple(sorted(edge_labels[edge_idx]))
        u, v = int(edges[edge_idx, 0]), int(edges[edge_idx, 1])
        junctions.add(tri)
        edge_to_triple[(u, v)] = tri
    return sorted(junctions), edge_to_triple


def signed_cell_volumes(
    v: NDArray[np.float64],
    f: NDArray[np.int64],
    regions: NDArray[np.int64],
) -> dict[int, float]:
    """Divergence-theorem volume per region, with `foambryo`'s sign convention.

    Accumulates `+det([v0, v1, v2]) / 6` into each triangle's first-listed region and
    `-det(...) / 6` into its second — the same rule as
    `foambryo.geometry.compute_volume_cells`, applied to raw arrays so it can be run on a
    mesh file *before* building a `DcelData`.

    The result is invariant under `DcelData`'s label-order normalisation (swapping a row's
    two labels also reverses its winding, which negates `det`, so the contribution is
    unchanged), and it was verified equal to `DcelData.compute_volumes_cells()` on
    `benchmarking-dataset/000_mesh.rec` to floating-point precision.

    A correctly-wound closed mesh gives a positive volume for every bounded cell and a
    negative one for the unbounded exterior label, whose faces are all wound inward from
    its point of view.

    Args:
        v (NDArray[np.float64]): Mesh vertices, shape `(nv, 3)`.
        f (NDArray[np.int64]): Mesh triangles (vertex indices), shape `(nf, 3)`.
        regions (NDArray[np.int64]): Per-triangle region pair, shape `(nf, 2)`.

    Returns:
        dict[int, float]: Region label -> signed volume.
    """
    det = np.linalg.det(v[f])
    volumes: dict[int, float] = {}
    for label in np.unique(regions):
        inflow = det[regions[:, 0] == label].sum()
        outflow = det[regions[:, 1] == label].sum()
        volumes[int(label)] = float((inflow - outflow) / 6.0)
    return volumes


def check_label_orientation_convention(
    v: NDArray[np.float64],
    f: NDArray[np.int64],
    regions: NDArray[np.int64],
    exterior: int | Iterable[int] = 0,
) -> dict:
    """Check that a mesh's triangle winding agrees with its label column order.

    `foambryo` requires each triangle's right-hand-rule normal to point away from its
    first-listed region and into its second (see `foambryo.dcel._normalize_label_order`).
    That tie between winding and label order cannot be checked from the labels — sorted
    label columns are neither necessary nor sufficient — so this checks its geometric
    consequence instead: every bounded cell must enclose a positive volume.

    Use this once on any mesh from a source other than `dw3d` (a simulation, a hand-written
    `.rec`, or a mesh whose label columns were sorted by a tool that did not reorient the
    triangles). Without it the failure is silent: `compute_volume_cells` simply returns
    negative volumes and everything downstream consumes them.

    Assumes a closed mesh. An open or cropped mesh will report non-positive volumes for
    the cells it truncates, which is a limitation of the check and not a defect in the
    mesh; `n_nonpositive_cells` and the returned volumes let a caller judge which case it
    is looking at.

    Args:
        v (NDArray[np.float64]): Mesh vertices, shape `(nv, 3)`.
        f (NDArray[np.int64]): Mesh triangles (vertex indices), shape `(nf, 3)`.
        regions (NDArray[np.int64]): Per-triangle region pair, shape `(nf, 2)`.
        exterior (int | Iterable[int], optional): Label of the external medium, excluded from the
            positivity requirement since its volume is legitimately negative; or the labels of
            every medium (`DcelData.media`: 0 and the enclosed media), which are then not
            counted as cells either. Defaults to 0.

    Returns:
        dict: `{"ok", "volumes", "nonpositive_cells", "n_nonpositive_cells",
            "n_rows_label_unsorted", "n_rows_same_label_both_sides"}`. `ok` is `True`
            when every bounded cell has a strictly positive volume.
    """
    volumes = signed_cell_volumes(v, f, regions)
    media = {int(exterior)} if isinstance(exterior, (int, np.integer)) else {int(label) for label in exterior}
    nonpositive = sorted(
        label for label, volume in volumes.items() if label not in media and volume <= 0
    )
    return {
        "ok": not nonpositive,
        "volumes": volumes,
        "nonpositive_cells": nonpositive,
        "n_nonpositive_cells": len(nonpositive),
        # Reported for information only: unsorted label columns are normal and valid
        # (every benchmarking-dataset mesh has them), never a failure on their own.
        "n_rows_label_unsorted": int((regions[:, 0] > regions[:, 1]).sum()),
        "n_rows_same_label_both_sides": int((regions[:, 0] == regions[:, 1]).sum()),
    }


def require_label_orientation_convention(
    v: NDArray[np.float64],
    f: NDArray[np.int64],
    regions: NDArray[np.int64],
    exterior: int | Iterable[int] = 0,
) -> None:
    """Raise if a mesh's winding does not agree with its label column order.

    The raising form of `check_label_orientation_convention`; see that function for what
    is checked, why label order alone cannot settle it, and the closed-mesh assumption.

    Args:
        v (NDArray[np.float64]): Mesh vertices, shape `(nv, 3)`.
        f (NDArray[np.int64]): Mesh triangles (vertex indices), shape `(nf, 3)`.
        regions (NDArray[np.int64]): Per-triangle region pair, shape `(nf, 2)`.
        exterior (int | Iterable[int], optional): Label of the external medium, or the labels of
            every medium (see `check_label_orientation_convention`). Defaults to 0.

    Raises:
        ValueError: If any bounded cell encloses a non-positive volume.
    """
    report = check_label_orientation_convention(v, f, regions, exterior=exterior)
    if not report["ok"]:
        offenders = ", ".join(f"{label}: {report['volumes'][label]:.6g}" for label in report["nonpositive_cells"])
        message = (
            f"{report['n_nonpositive_cells']} cell(s) enclose a non-positive volume "
            f"({offenders}). Each triangle's winding must put its right-hand-rule normal "
            f"on the side of its second label, i.e. outward from the first; a mesh whose "
            f"label columns were sorted without reorienting its triangles violates this. "
            f"If instead the mesh is open or cropped, this check does not apply to it."
        )
        raise ValueError(message)


def mean_edge_length(v: NDArray[np.float64], f: NDArray[np.int64]) -> float:
    """Average triangle-edge length; handy to express noise or tolerances.

    Args:
        v (NDArray[np.float64]): Mesh vertices, shape `(nv, 3)`.
        f (NDArray[np.int64]): Mesh triangles (vertex indices), shape `(nf, 3)`.

    Returns:
        float: Mean length of all (non-deduplicated) triangle edges.
    """
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    return float(np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1).mean())
