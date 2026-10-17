"""Geometry on a DCEL mesh: trijunction angles and lengths, areas, volumes, and their derivatives.

Computation of trijunction mean angles (in radians, under any of the angle readings of
`foambryo.second_order_angle_reading`) and lengths, interface and region areas, region volumes,
and their derivatives with respect to the vertex positions. Every quantity is in the mesh's own
length unit; interfaces are keyed `(label_1, label_2)`, `label_1 < label_2`, label 0 the exterior
medium.

Sacha Ichbiah, Sept 2021.
Matthieu Perez, 2024.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

from foambryo.second_order_angle_reading import (
    CHORD_READING,
    AngleReadingReport,
    reads_oriented_wedges,
    rodrigues_about,
    rotations_for_reading,
    warn_on_partial_coverage,
)

#: A full turn in radians. The oriented wedge reading reflects a reflex wedge through it.
_FULL_TURN = 2.0 * np.pi

if TYPE_CHECKING:
    from collections.abc import Collection

    from foambryo.dcel import DcelData, HalfEdge


def _find_key_multiplier(num_points: int) -> int:
    key_multiplier = 1
    while num_points // key_multiplier != 0:
        key_multiplier *= 10
    return key_multiplier


## LENGTHS AND DERIVATIVES


def compute_length_derivative_dict(mesh: "DcelData") -> dict[tuple[int, int, int], NDArray[np.float64]]:
    """Compute map trijunction (id reg 1, id reg 2, id reg 3) -> derivative of length wrt each point.

    Closed-form float64 replacement for the former autodiff path (moved to
    `foambryo._autodiff_derivatives`). For an edge with endpoints ``p, q`` the
    total-length gradient is the unit edge vector at ``p`` and its negative at ``q``:
    ``d||p - q|| / dp = (p - q) / ||p - q||``. Summed over every edge of the trijunction.
    """
    edges_trijunctions = extract_edges_trijunctions(mesh)
    points = mesh.v
    length_derivatives: dict[tuple[int, int, int], NDArray[np.float64]] = {}
    for key in sorted(edges_trijunctions.keys()):
        edges = edges_trijunctions[key]
        grad = np.zeros((len(points), 3))
        vec = points[edges[:, 0]] - points[edges[:, 1]]
        unit = vec / np.linalg.norm(vec, axis=1).reshape(-1, 1)
        np.add.at(grad, edges[:, 0], unit)
        np.add.at(grad, edges[:, 1], -unit)
        length_derivatives[key] = grad
    return length_derivatives


def compute_length_trijunctions(mesh: "DcelData", prints: bool = False) -> dict[tuple[int, int, int], float]:
    """Compute the total length of each trijunction of the mesh in a map (id reg 1, id reg 2, id reg 3) -> length."""
    length_trijunctions: dict[tuple[int, int, int], float] = {}
    edges_trijunctions = extract_edges_trijunctions(mesh, prints)
    for key in edges_trijunctions:
        length_trijunctions[key] = np.sum(compute_length_edges_trijunctions(mesh.v, edges_trijunctions[key]))
    return length_trijunctions


def compute_length_edges_trijunctions(
    points: NDArray[np.float64],
    edges_trijunctions: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Compute the length of each edge of a trijunction ."""
    positions = points[edges_trijunctions]
    return np.linalg.norm(positions[:, 0] - positions[:, 1], axis=1)


def extract_edges_trijunctions(mesh: "DcelData", prints: bool = False) -> dict[tuple[int, int, int], NDArray[np.int64]]:
    """Extract a dict that maps trijunctions (id reg 1, id reg 2, id reg 3) to a list of edges."""
    triangles_and_labels = mesh.f
    edges = np.vstack(
        (
            triangles_and_labels[:, [0, 1]],
            triangles_and_labels[:, [0, 2]],
            triangles_and_labels[:, [1, 2]],
        ),
    )
    edges = np.sort(edges, axis=1)
    zones = np.vstack(
        (
            triangles_and_labels[:, [3, 4]],
            triangles_and_labels[:, [3, 4]],
            triangles_and_labels[:, [3, 4]],
        ),
    )
    key_mult = _find_key_multiplier(len(mesh.v) + 1)
    keys = (edges[:, 0] + 1) + (edges[:, 1] + 1) * key_mult
    _, index_first_occurence, index_inverse, index_counts = np.unique(
        keys,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    if prints:
        print("Number of trijunctional edges :", np.sum(index_counts == 3))

    indices = np.arange(len(index_counts))
    trijunction_indices_to_edge_key = {key: [] for key in indices[index_counts == 3]}
    table = np.zeros(len(index_counts))
    table[index_counts == 3] += 1

    for i in range(len(index_inverse)):
        inverse = index_inverse[i]
        if table[inverse] == 1:
            trijunction_indices_to_edge_key[inverse].append(i)

    trijunctional_line = {}
    for key in sorted(trijunction_indices_to_edge_key.keys()):
        x = trijunction_indices_to_edge_key[key]
        regions = np.hstack((zones[x[0]], zones[x[1]], zones[x[2]]))
        u = np.unique(regions)
        if len(u) > 4:
            print("oui")
            continue
        else:
            trijunctional_line[tuple(u)] = trijunctional_line.get(tuple(u), [])
            trijunctional_line[tuple(u)].append(edges[x[0]])
            assert edges[x[0]][0] == edges[x[1]][0] == edges[x[2]][0]
            assert edges[x[0]][1] == edges[x[1]][1] == edges[x[2]][1]

    output_dict: dict[tuple[int, int, int], NDArray[np.int64]] = {}
    for key in sorted(trijunctional_line.keys()):
        output_dict[key] = np.vstack(trijunctional_line[key])
    return output_dict


## AREAS AND DERIVATIVES
def compute_areas_faces(mesh: "DcelData") -> None:
    """Compute area of every triangle in the mesh. Modify the info of the mesh."""
    positions = mesh.v[mesh.f[:, [0, 1, 2]]]
    sides = positions - positions[:, [2, 0, 1]]
    lengths_sides = np.linalg.norm(sides, axis=2)
    half_perimeters = np.sum(lengths_sides, axis=1) / 2

    diffs = np.array([half_perimeters] * 3).transpose() - lengths_sides
    areas = (half_perimeters * diffs[:, 0] * diffs[:, 1] * diffs[:, 2]) ** (0.5)
    for i, face in enumerate(mesh.faces):
        face.area = areas[i]


def compute_areas_cells(mesh: "DcelData") -> dict[int, float]:
    """Compute area of each cells."""
    areas: dict[int, float] = {key: 0 for key in mesh.materials}
    for face in mesh.faces:
        areas[face.material_1] += face.area
        areas[face.material_2] += face.area
    return areas


def compute_areas_interfaces(mesh: "DcelData") -> dict[tuple[int, int], float]:
    """Compute area of every interface (label1, label2) in mesh."""
    interfaces = {}
    for face in mesh.faces:
        materials = (face.material_1, face.material_2)
        key = (min(materials), max(materials))
        interfaces[key] = interfaces.get(key, 0) + face.area
    return interfaces


def compute_area_derivative_dict(
    mesh: "DcelData",
) -> dict[tuple[int, int], NDArray[np.float64]]:
    """Compute dict that maps interface (label1, label2) to array of change of area per point."""
    interfaces_keys: NDArray[np.int64] = np.array(sorted(compute_areas_interfaces(mesh).keys()))
    points, triangles, labels = mesh.v, mesh.f[:, :3], mesh.f[:, 3:]
    area_derivatives: dict[tuple[int, int], NDArray[np.float64]] = {
        tuple(t): np.zeros((len(points), 3)) for t in interfaces_keys
    }

    coords = points[triangles]
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    e3 = np.cross(z - y, x - z) / (np.linalg.norm(np.cross(z - y, x - z), axis=1).reshape(-1, 1))
    cross_e3_x = np.cross(e3, z - y) / 2
    cross_e3_y = np.cross(e3, x - z) / 2
    cross_e3_z = np.cross(e3, y - x) / 2

    list_indices_faces_per_vertices_x = {key: [[] for i in range(len(points))] for key in area_derivatives}
    list_indices_faces_per_vertices_y = {key: [[] for i in range(len(points))] for key in area_derivatives}
    list_indices_faces_per_vertices_z = {key: [[] for i in range(len(points))] for key in area_derivatives}

    for i in range(len(triangles)):
        i_x, i_y, i_z = triangles[i]
        a, b = labels[i]
        list_indices_faces_per_vertices_x[(a, b)][i_x].append(i)
        list_indices_faces_per_vertices_y[(a, b)][i_y].append(i)
        list_indices_faces_per_vertices_z[(a, b)][i_z].append(i)

    for key in area_derivatives:
        for iv in range(len(points)):
            area_derivatives[key][iv] = np.vstack(
                (
                    cross_e3_x[list_indices_faces_per_vertices_x[key][iv]],
                    cross_e3_y[list_indices_faces_per_vertices_y[key][iv]],
                    cross_e3_z[list_indices_faces_per_vertices_z[key][iv]],
                ),
            ).sum(axis=0)

    return area_derivatives


def _scatter_grad_triplets(
    triangles: NDArray[np.int64],
    row_index: NDArray[np.int64],
    grads: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """Build COO (row, col, val) triplets scattering per-triangle vertex gradients.

    For each triangle, its three vertices (slots 0, 1, 2) receive the 3-vectors
    ``grads[0]``, ``grads[1]``, ``grads[2]`` at the operator row ``row_index`` for that
    triangle. Vertex ``v``'s component ``c`` lands in column ``3*v + c``. Duplicate
    (row, col) pairs are summed later by ``coo_matrix`` — this is the scatter-add.
    """
    rows_list, cols_list, vals_list = [], [], []
    for slot in range(3):
        vtx = triangles[:, slot]
        grad = grads[slot]
        for comp in range(3):
            rows_list.append(row_index)
            cols_list.append(3 * vtx + comp)
            vals_list.append(grad[:, comp])
    return (
        np.concatenate(rows_list),
        np.concatenate(cols_list),
        np.concatenate(vals_list),
    )


def compute_area_derivative_sparse(
    mesh: "DcelData",
) -> tuple[list[tuple[int, int]], csr_matrix]:
    """Sparse area-gradient operator ``D`` (n_interfaces x 3*n_points), one row per interface.

    Row ``i`` is the flattened gradient of interface ``keys[i]``'s total area with respect to
    every vertex coordinate; vertex ``v``'s three components sit at columns ``3v, 3v+1, 3v+2``.
    The per-vertex contributions are identical to :func:`compute_area_derivative_dict`, but the
    operator is assembled by a single vectorised COO scatter-add over the triangle list
    (``O(n_triangles)`` non-zeros, mean support ~6 % of points per interface) — no Python loop
    over interfaces or vertices, and never a dense ``(n_interfaces, n_points, 3)`` array. This
    is the object the variational tension/pressure solve consumes.

    Returns:
        tuple[list[tuple[int, int]], csr_matrix]: the sorted interface keys (row order) and
            the sparse ``D``.
    """
    interfaces_keys = sorted(compute_areas_interfaces(mesh).keys())
    row_of_interface = {key: i for i, key in enumerate(interfaces_keys)}
    n_interfaces = len(interfaces_keys)

    points, triangles, labels = mesh.v, mesh.f[:, :3], mesh.f[:, 3:]
    coords = points[triangles]
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    normals = np.cross(z - y, x - z)
    e3 = normals / (np.linalg.norm(normals, axis=1).reshape(-1, 1))
    # Contribution of each triangle to its three vertices (slot 0/1/2), as in the dict path.
    grad_slot0 = np.cross(e3, z - y) / 2
    grad_slot1 = np.cross(e3, x - z) / 2
    grad_slot2 = np.cross(e3, y - x) / 2

    interface_row = np.array([row_of_interface[(a, b)] for a, b in labels], dtype=np.int64)
    rows, cols, vals = _scatter_grad_triplets(triangles, interface_row, (grad_slot0, grad_slot1, grad_slot2))
    matrix_d = coo_matrix((vals, (rows, cols)), shape=(n_interfaces, 3 * len(points))).tocsr()
    return interfaces_keys, matrix_d


##VOLUMES AND DERIVATIVES


def compute_volume_cells(mesh: "DcelData") -> dict[int, float]:
    """Compute map cell number -> volume."""
    volumes: dict[int, float] = {m: 0 for m in mesh.materials}
    for i, face in enumerate(mesh.faces):
        index = mesh.f[i, [0, 1, 2]]
        coords = mesh.v[index]
        inc = np.linalg.det(coords)
        volumes[face.material_1] += inc
        volumes[face.material_2] -= inc

    for key in volumes:
        volumes[key] = volumes[key] / 6
    return volumes


def compute_volume_derivative_dict(mesh: "DcelData") -> dict[int, NDArray[np.float64]]:
    """Compute map cell number -> derivative of volume wrt to each point."""
    points, triangles, labels = mesh.v, mesh.f[:, :3], mesh.f[:, 3:]
    materials = mesh.materials

    volumes_derivatives: dict[int, NDArray[np.float64]] = {key: np.zeros((len(points), 3)) for key in materials}

    coords = points[triangles]
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    cross_xy = np.cross(x, y) / 6
    cross_yz = np.cross(y, z) / 6
    cross_zx = np.cross(z, x) / 6
    # (Defect-cleanup fix.) A `faces_material` sign table used to be built here and never read. It was
    # also wrong: it compared each label column to the literal 1 instead of to the cell `n`
    # it was keying on, so every cell got the same table. The per-face signs it was meant to
    # carry are already applied below, by the pos_*/neg_* index lists (+cross for the cell on
    # the `labels[:, 0]` side, -cross for the one on the `labels[:, 1]` side).
    list_indices_faces_per_vertices_pos_x = {key: [[] for i in range(len(points))] for key in materials}
    list_indices_faces_per_vertices_pos_y = {key: [[] for i in range(len(points))] for key in materials}
    list_indices_faces_per_vertices_pos_z = {key: [[] for i in range(len(points))] for key in materials}
    list_indices_faces_per_vertices_neg_x = {key: [[] for i in range(len(points))] for key in materials}
    list_indices_faces_per_vertices_neg_y = {key: [[] for i in range(len(points))] for key in materials}
    list_indices_faces_per_vertices_neg_z = {key: [[] for i in range(len(points))] for key in materials}

    for i in range(len(triangles)):
        i_x, i_y, i_z = triangles[i]
        a, b = labels[i]
        # print(i_x,i_y,i_z)
        list_indices_faces_per_vertices_pos_x[a][i_x].append(i)
        list_indices_faces_per_vertices_pos_y[a][i_y].append(i)
        list_indices_faces_per_vertices_pos_z[a][i_z].append(i)

        list_indices_faces_per_vertices_neg_x[b][i_x].append(i)
        list_indices_faces_per_vertices_neg_y[b][i_y].append(i)
        list_indices_faces_per_vertices_neg_z[b][i_z].append(i)

    for n in materials:
        for iv in range(len(points)):
            volumes_derivatives[n][iv] = np.vstack(
                (
                    cross_yz[list_indices_faces_per_vertices_pos_x[n][iv]],
                    cross_zx[list_indices_faces_per_vertices_pos_y[n][iv]],
                    cross_xy[list_indices_faces_per_vertices_pos_z[n][iv]],
                    -cross_yz[list_indices_faces_per_vertices_neg_x[n][iv]],
                    -cross_zx[list_indices_faces_per_vertices_neg_y[n][iv]],
                    -cross_xy[list_indices_faces_per_vertices_neg_z[n][iv]],
                ),
            ).sum(axis=0)

    return volumes_derivatives


def compute_volume_derivative_sparse(
    mesh: "DcelData",
) -> tuple[NDArray[np.int64], csr_matrix]:
    """Sparse volume-gradient operator ``V`` (n_materials x 3*n_points), one row per material.

    Row ``j`` is the flattened gradient of material ``materials[j]``'s signed volume. Rows are
    ordered as ``mesh.materials`` (so ``materials[0]`` is the exterior); the variational solve
    uses rows ``[1:]``, the cells. Each triangle contributes ``+cross`` to its lower-label
    material and ``-cross`` to its higher-label one, exactly as :func:`compute_volume_derivative_dict`,
    but assembled by one vectorised COO scatter-add instead of six nested ``n_materials x n_points``
    Python lists.

    Returns:
        tuple[NDArray[np.int64], csr_matrix]: ``mesh.materials`` (row order) and the sparse ``V``.
    """
    materials = mesh.materials
    row_of_material = {int(m): i for i, m in enumerate(materials)}

    points, triangles, labels = mesh.v, mesh.f[:, :3], mesh.f[:, 3:]
    coords = points[triangles]
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    cross_slot0 = np.cross(y, z) / 6
    cross_slot1 = np.cross(z, x) / 6
    cross_slot2 = np.cross(x, y) / 6

    row_pos = np.array([row_of_material[int(a)] for a in labels[:, 0]], dtype=np.int64)
    row_neg = np.array([row_of_material[int(b)] for b in labels[:, 1]], dtype=np.int64)
    rows_p, cols_p, vals_p = _scatter_grad_triplets(triangles, row_pos, (cross_slot0, cross_slot1, cross_slot2))
    rows_n, cols_n, vals_n = _scatter_grad_triplets(triangles, row_neg, (-cross_slot0, -cross_slot1, -cross_slot2))
    rows = np.concatenate([rows_p, rows_n])
    cols = np.concatenate([cols_p, cols_n])
    vals = np.concatenate([vals_p, vals_n])
    matrix_v = coo_matrix((vals, (rows, cols)), shape=(len(materials), 3 * len(points))).tocsr()
    return materials, matrix_v


##ANGLES


def _wedges_at_half_edge(
    mesh: "DcelData",
    edge: "HalfEdge",
    rotations: dict[tuple[tuple[int, int], int], float] | None = None,
    geometric_edge: tuple[int, int] | None = None,
    *,
    oriented: bool = False,
) -> list[tuple[tuple[int, int, int], float]] | None:
    """The three trijunction wedge angles seen from one half-edge of a trijunctional edge.

    This is the single definition of `foambryo`'s trijunction angle convention. It is called
    once per *half*-edge by :func:`compute_angles_tri` and once per *geometric* edge by
    :func:`compute_trijunction_edges`, so the per-edge and the averaged views cannot drift
    apart — the convention lives in one place. That convention is: the returned angle is the **interior wedge angle** of
    the shared region `f`, it is **relabel-invariant**, and it is what every Young-Dupre
    builder consumes as `cos(angle)`. Do not change the normal-flip heuristic below without
    re-running `tests/test_angle_convention.py`.

    Quadrijunctions are skipped and this is deliberate, not a limitation: in a dry foam exactly
    three films meet along an edge (Plateau), so a valence-4 edge is a meshing artifact.

    **The second-order reading.** `rotations` is the only way the angle
    convention itself can change, and it is off by default. When it is supplied, each incident
    face's normal is first rotated about the junction edge by that `(geometric edge, face key)`'s
    angle. Rotating a normal about the edge rotates that triangle's in-plane trace by the same
    angle, so this is exactly "read the interface's fitted tangent instead of the triangle's
    chord", expressed in the one quantity this function consumes. The axis must run from the
    **lower** to the **higher** vertex key, because that is the orientation
    `second_order_angle_reading.quadric_angle_rotations` measured each angle in; the half-edge's
    own direction is irrelevant and using it would flip the sign of the correction on half the
    half-edges. A face with no entry in `rotations` — no admissible fit — keeps its chord, so a
    partly covered mesh degrades to the default reading face by face rather than failing.

    With `rotations=None`, or with every rotation zero, the returned angles are **bit-identical**
    to this function's behaviour before the second-order reading was added;
    `tests/test_second_order_angle_reading.py` pins both.

    **The oriented reading.** `arccos` has range `[0, pi]`, so a wedge whose interior angle
    exceeds 180 degrees is returned as `360 - theta` and the edge's three wedges sum to
    `360 - 2 (theta - 180)` instead of 360. With `oriented=True` the returned angle is instead
    the oriented arc from film `e|f` to film `f|g` measured *through* region `f`, in
    `[0, 2 pi)`, so the three sum to a full turn. The magnitude is still the same `arccos` of
    the same dot product, evaluated by the same expression; only the branch between `theta` and
    `2 pi - theta` is new, and it is taken on the strict sign of the oriented sine with no
    tolerance. The two readings are therefore **bitwise identical** on every non-reflex wedge —
    see `tests/test_oriented_wedge_reading.py`.

    The orienting axis runs from the lower to the higher **vertex key** of the geometric edge,
    never from the visiting half-edge's own direction, so all three half-edges of one geometric
    edge orient the same way. Its sign is then resolved from the geometry rather than asserted:
    three positive angles summing to a full turn contain at most one greater than `pi`, so under
    the reversed axis the three oriented wedges sum to two full turns, and the assignment that
    sums to one is taken. A wedge whose oriented sine is exactly zero (an interior angle of
    exactly 0 or exactly `pi`) reads as non-reflex under either axis, reproducing the unoriented
    reader there.

    Args:
        mesh (DcelData): Mesh to analyze.
        edge (HalfEdge): A half-edge with at least two twins, i.e. on a junction.
        rotations (dict[tuple[tuple[int, int], int], float] | None, optional): normal rotations
            in radians keyed by `((low vertex key, high vertex key), face key)`, from
            `foambryo.second_order_angle_reading.quadric_angle_rotations`. Defaults to `None`,
            the chord reading.
        geometric_edge (tuple[int, int] | None, optional): this half-edge's `(min, max)` vertex
            keys. Required when `rotations` is given; ignored otherwise.
        oriented (bool, optional): return the oriented wedge angle in `[0, 2 pi)` instead of
            `arccos`'s `[0, pi]`. Keyword-only. Defaults to `False`, the historical reading.

    Returns:
        list[tuple[tuple[int, int, int], float]] | None: three
            `((min(e, g), f, max(e, g)), angle_in_radians)` pairs, one per adjacent region `f`;
            `None` if the edge is a quadrijunction (or higher) and was skipped.
    """
    face = edge.incident_face
    faces = [face]
    normals = [face.normal]
    materials = [[face.material_1, face.material_2]]

    for neighbor in edge.twin:
        face_attached = mesh.half_edges[neighbor].incident_face
        faces.append(face_attached)
        materials.append([face_attached.material_1, face_attached.material_2])
        normals.append(face_attached.normal)

    regions_id = np.array(materials)
    if len(regions_id) != 3:
        ## If we fall onto a quadrijunction, we skip it.
        return None

    normals = np.array(normals).copy()

    if rotations is not None:
        normals = _rotate_normals_about_junction(mesh, normals, faces, rotations, geometric_edge)

    if regions_id[0, 0] == regions_id[1, 0] or regions_id[0, 1] == regions_id[1, 1]:
        regions_id[1] = regions_id[1][[1, 0]]
        normals[1] *= -1

    if regions_id[0, 0] == regions_id[2, 0] or regions_id[0, 1] == regions_id[2, 1]:
        regions_id[2] = regions_id[2][[1, 0]]
        normals[2] *= -1

    wedges = []
    oriented_sines = []
    for i1, i2 in ([0, 1], [1, 2], [2, 0]):
        angle = np.arccos(np.clip(np.dot(normals[i1], normals[i2]), -1, 1))

        if regions_id[i1][1] == regions_id[i2][0]:
            e, f, g = regions_id[i1][0], regions_id[i1][1], regions_id[i2][1]
            first, second = i1, i2

        elif regions_id[i1][0] == regions_id[i2][1]:
            e, f, g = regions_id[i2][0], regions_id[i2][1], regions_id[i1][1]
            first, second = i2, i1

        wedges.append(((min(e, g), f, max(e, g)), angle))
        oriented_sines.append((first, second))

    if not oriented:
        return wedges

    return _reflect_reflex_wedges(mesh, edge, normals, wedges, oriented_sines)


def _reflect_reflex_wedges(
    mesh: "DcelData",
    edge: "HalfEdge",
    normals: NDArray[np.float64],
    wedges: list[tuple[tuple[int, int, int], float]],
    film_order: list[tuple[int, int]],
) -> list[tuple[tuple[int, int, int], float]]:
    """Turn three `arccos` magnitudes into the three oriented wedge angles of one junction edge.

    Split out of :func:`_wedges_at_half_edge` so the historical reading's body stays exactly what
    it was: with `oriented=False` this function is never called, and with `oriented=True` it
    returns the *same float object* for every wedge that is not reflex. That is safeguard 3 —
    identity where `arccos` is already right — made bitwise rather than approximate.

    The oriented angle of the wedge inside region `f`, between films `e|f` and `f|g`, is the arc
    from the first film's normal to the second's measured counter-clockwise about the junction
    axis. Both normals are perpendicular to the edge (each triangle contains it), so the arc is
    well defined; `arccos` already gives its magnitude, and the sign of
    `dot(axis, cross(n_first, n_second))` says whether the arc is the short way round or the
    reflex way.

    **Resolving the axis sign.** Three positive angles summing to one full turn contain at most
    one greater than `pi`, so the reversed axis makes the three sum to two full turns instead of
    one. This function computes both and keeps the assignment nearer one full turn. A wedge whose
    oriented sine is exactly zero — an interior angle of exactly 0 or exactly `pi` — reads as
    non-reflex under either axis, because `-0.0 >= 0`; that is the determinate behaviour at a
    degenerate wedge and it reproduces the unoriented reader there.

    Failure modes: an edge of zero length has no axis, and the wedges are returned unchanged.

    Args:
        mesh (DcelData): the mesh the edge belongs to, read for its vertex coordinates.
        edge (HalfEdge): the half-edge whose geometric edge is being read.
        normals (NDArray[np.float64]): the three incident face normals, sign-normalised by
            :func:`_wedges_at_half_edge` so each runs from its face's first to its second region.
        wedges (list[tuple[tuple[int, int, int], float]]): the three
            `(wedge key, arccos magnitude in radians)` pairs, in the order they were built.
        film_order (list[tuple[int, int]]): for each wedge, the `(first, second)` indices into
            `normals` of films `e|f` and `f|g`, aligned index-wise with `wedges`.

    Returns:
        list[tuple[tuple[int, int, int], float]]: the same keys with angles in `[0, 2 pi)`.
    """
    low, high = sorted((edge.origin.key, edge.destination.key))
    axis = mesh.v[high] - mesh.v[low]
    length = float(np.linalg.norm(axis))
    if length <= 0:
        return wedges
    axis = axis / length

    sines = np.array(
        [float(np.dot(axis, np.cross(normals[first], normals[second]))) for first, second in film_order],
    )
    angles = np.array([angle for _key, angle in wedges])
    forward = np.where(sines >= 0, angles, _FULL_TURN - angles)
    reversed_ = np.where(-sines >= 0, angles, _FULL_TURN - angles)
    chosen = forward if abs(forward.sum() - _FULL_TURN) <= abs(reversed_.sum() - _FULL_TURN) else reversed_
    # `chosen[i] is angles[i]` in value and in bits wherever the wedge is not reflex.
    return [
        (key, angle if oriented_angle == angle else float(oriented_angle))
        for (key, angle), oriented_angle in zip(wedges, chosen, strict=True)
    ]


def _rotate_normals_about_junction(
    mesh: "DcelData",
    normals: NDArray[np.float64],
    faces: list,
    rotations: dict[tuple[tuple[int, int], int], float],
    geometric_edge: tuple[int, int] | None,
) -> NDArray[np.float64]:
    """Rotate each incident face's normal about the junction edge by its correction angle.

    Split out of :func:`_wedges_at_half_edge` only so the default reading's body stays exactly
    what it was. A face whose key is absent from `rotations`, or whose angle is exactly zero,
    is left untouched — that is safeguard 4 (identity at zero curvature) made bitwise rather
    than approximate, and it is also the per-face fallback to the chord reading.

    Args:
        mesh (DcelData): the mesh the edge belongs to, read for its vertex coordinates.
        normals (NDArray[np.float64]): the three incident face normals, in `faces` order.
        faces (list): the three incident `Face` objects, in the order
            :func:`_wedges_at_half_edge` collected them.
        rotations (dict[tuple[tuple[int, int], int], float]): correction angles in radians.
        geometric_edge (tuple[int, int] | None): the edge's `(min, max)` vertex keys.

    Returns:
        NDArray[np.float64]: the rotated normals.

    Raises:
        ValueError: if `geometric_edge` is `None`. A rotation map keyed by geometric edge is
            unusable without one, and silently reading chords instead is the failure the
            second-order reading exists to remove.
    """
    if geometric_edge is None:
        message = "_wedges_at_half_edge: `rotations` was given without `geometric_edge`"
        raise ValueError(message)
    axis = mesh.v[geometric_edge[1]] - mesh.v[geometric_edge[0]]
    length = float(np.linalg.norm(axis))
    if length <= 0:
        return normals
    axis = axis / length
    for position, incident_face in enumerate(faces):
        angle = rotations.get((geometric_edge, int(incident_face.key)))
        if angle:
            normals[position] = rodrigues_about(axis, normals[position], float(angle))
    return normals


def compute_angles_tri(
    mesh: "DcelData",
    unique: bool = True,
    *,
    reading: str = CHORD_READING,
) -> tuple[
    dict[tuple[int, int, int], float],
    dict[tuple[int, int, int], float],
    dict[tuple[int, int, int], float],
]:
    """Compute three maps trijunction (id reg 1, id reg 2, id reg 3) to mean angle, mean angle (deg), length.

    The keys are wedge keys: `(min(e, g), f, max(e, g))` is the angle of the wedge *inside*
    region `f` at the triple line where regions `e`, `f` and `g` meet, so one triple line
    contributes three keys whose angles sum to 360 degrees.

    **That sum holds exactly only under an oriented reading.** Under `"chord"` and `"quadric"`
    the angle is `arccos` of a dot product, whose range is `[0, 180]` degrees, so a wedge wider
    than a straight angle comes back as `360 - theta` and its edge's three angles sum to
    `360 - 2 (theta - 180)`. `"oriented_chord"` and `"oriented_quadric"` return the wedge in
    `[0, 360)` and restore the identity; they are bitwise identical to their unoriented
    counterparts on every wedge that is not reflex.

    **`unique`.** In earlier versions the parameter had no effect except to double every
    accumulator uniformly, which changes neither the mean angle nor the length *ratio* that
    every builder actually uses. It now means what its name says:

    - `unique=True` (the declared default) counts each **geometric** trijunctional edge exactly
      once per wedge key. `dict_length[key]` is then the true arclength of that triple line, and
      `sum(dict_length.values()) / 3` — the `total_length_mesh` every builder computes — is then
      the true total junction length of the mesh, which it was not before.
    - `unique=False` reproduces the historical accumulation **bit for bit**: each geometric edge
      is visited once per incident half-edge (3) and each visit appends twice, so
      `dict_length[key] = 6 * length` and `sum(...) / 3 = 6 * L_total`. Every call site inside
      `foambryo.tension_inference` passes `unique=False`, so no inferred number moves.

    The mean angles are identical either way — the over-count was, and remains, uniform on the
    benchmarking dataset (`BENCHMARKS.md`). That uniformity was verified rather than assumed: on
    all 47 of its meshes every trijunctional geometric edge has exactly 3 incident half-edges, and
    the three visits
    return **bit-identical** angles. The one construction where that edge accounting is *not*
    uniform — a triple line running off an open mesh boundary, where an edge that has lost one of
    its three films is dropped and the line's length weight is under-stated in proportion — is
    pinned in `tests/test_per_edge_young_dupre.py`.

    **`reading`.** `"chord"` (the default) is the convention described above; adding the selector
    changed none of its numbers.
    `"oriented_chord"` is the same reading with the wedge angle reported in `[0, 360)` degrees
    instead of `[0, 180]`, so a wedge wider than a straight angle is no longer reported as its
    reflection and the three wedge keys of a line really do sum to 360 degrees; it is bitwise
    identical to `"chord"` wherever no wedge is reflex. `"quadric"` replaces each triangle's
    chord by the tangent of a generic quadric fitted to that interface's own mesh vertices around
    the junction — see `foambryo.second_order_angle_reading`. It is **not** available on every face of every mesh:
    where the fit is inadmissible the chord is kept, so a `"quadric"` result can mix the two
    conventions. This function reports that mixing as an `AngleReadingCoverageWarning`; use
    :func:`compute_angles_tri_with_report` to receive the coverage numbers instead.

    Args:
        mesh (DcelData): Mesh to analyze.
        unique (bool, optional): Count each geometric edge once (True) or reproduce the
            historical 6x over-count (False). Defaults to True.
        reading (str, optional): `"chord"`, `"quadric"`, `"oriented_chord"` or
            `"oriented_quadric"`. Keyword-only. Defaults to `"chord"`, which is bit-identical to
            the historical behaviour.

    Returns:
        tuple[dict, dict, dict]: mean angle (rad), mean angle (deg) and accumulated length,
            all keyed by wedge key.
    """
    dict_mean_angles, dict_mean_angles_deg, dict_length, report = compute_angles_tri_with_report(
        mesh,
        unique,
        reading=reading,
    )
    warn_on_partial_coverage(report)
    return (dict_mean_angles, dict_mean_angles_deg, dict_length)


def _is_not_measured(
    geometric_edge: tuple[int, int],
    skip_edges: "Collection[tuple[int, int]] | None",
    seen: set[tuple[int, int]],
    *,
    unique: bool,
) -> bool:
    """Whether this visit to a geometric edge should be passed over, and record it if not.

    Two independent reasons to pass over a half-edge, kept together so
    :func:`compute_angles_tri_with_report`'s loop body stays one readable pass: the edge was
    declined by the caller (`skip_edges`), or `unique` is set and this geometric edge has already
    been visited through another of its half-edges. Mutates `seen`, which is why it is not a
    predicate in the pure sense; the mutation is the de-duplication.

    Args:
        geometric_edge (tuple[int, int]): the `(min, max)` vertex-key pair.
        skip_edges (Collection[tuple[int, int]] | None): edges the caller declined, or `None`.
        seen (set[tuple[int, int]]): geometric edges already accumulated, updated in place.
        unique (bool): whether each geometric edge is counted once. Keyword-only.

    Returns:
        bool: `True` to skip this half-edge.
    """
    if skip_edges is not None and geometric_edge in skip_edges:
        return True
    if unique:
        if geometric_edge in seen:
            return True
        seen.add(geometric_edge)
    return False


def compute_angles_tri_with_report(
    mesh: "DcelData",
    unique: bool = True,
    *,
    reading: str = CHORD_READING,
    skip_edges: "Collection[tuple[int, int]] | None" = None,
) -> tuple[
    dict[tuple[int, int, int], float],
    dict[tuple[int, int, int], float],
    dict[tuple[int, int, int], float],
    AngleReadingReport,
]:
    """:func:`compute_angles_tri`, plus the reading's coverage report.

    The reporting entry point. Use it whenever a result will be published, compared or
    benchmarked: it is the only way to know which trijunction lines a `"quadric"` reading could
    not cover and therefore read as chords.

    Args:
        mesh (DcelData): Mesh to analyze.
        unique (bool, optional): Count each geometric edge once (True) or reproduce the
            historical 6x over-count (False). Defaults to True.
        reading (str, optional): one of
            `foambryo.second_order_angle_reading.ANGLE_READINGS`. Keyword-only. Defaults to
            `"chord"`.
        skip_edges (Collection[tuple[int, int]] | None, optional): geometric edges, as
            `(min, max)` vertex-key pairs, whose wedges contribute neither an angle nor a length
            to any triple line's accumulators. Keyword-only, `None` by default, and with `None`
            (or an empty collection) this function's result is **bitwise** what it was before the
            parameter existed — the membership test is the only thing that runs. Supplied by
            `foambryo.wedge_refusal` so that an estimator can decline an observation its own
            geometry makes infeasible; an edge listed here is simply not measured, so a line that
            loses every one of its edges disappears from all three returned maps rather than
            appearing with an empty mean.

    Returns:
        tuple[dict, dict, dict, AngleReadingReport]: mean angle (rad), mean angle (deg),
            accumulated length, and the coverage report.
    """
    ##We compute the angles at each trijunctions. If we fall onto a quadrijunction, we skip it

    rotations, report = rotations_for_reading(mesh, reading)
    oriented = reads_oriented_wedges(reading)
    dict_length: dict[tuple[int, int, int], float] = {}
    dict_angles: dict[tuple[int, int, int], list[float]] = {}
    seen_geometric_edges: set[tuple[int, int]] = set()
    geometric_edge: tuple[int, int] | None = None
    needs_geometric_edge = unique or rotations is not None or skip_edges is not None
    for edge in mesh.half_edges:
        if len(edge.twin) > 1:
            if needs_geometric_edge:
                origin, destination = edge.origin.key, edge.destination.key
                geometric_edge = (min(origin, destination), max(origin, destination))
                if _is_not_measured(geometric_edge, skip_edges, seen_geometric_edges, unique=unique):
                    continue

            wedges = _wedges_at_half_edge(mesh, edge, rotations, geometric_edge, oriented=oriented)
            if wedges is None:
                continue

            for key, angle in wedges:
                dict_angles[key] = dict_angles.get(key, [])
                dict_angles[key].append(angle)
                dict_length[key] = dict_length.get(key, 0)
                dict_length[key] += edge.length
                if not unique:
                    dict_angles[key] = dict_angles.get(key, [])
                    dict_angles[key].append(angle)
                    dict_length[key] = dict_length.get(key, 0)
                    dict_length[key] += edge.length

    dict_mean_angles: dict[tuple[int, int, int], float] = {}
    dict_mean_angles_deg: dict[tuple[int, int, int], float] = {}
    for key in dict_angles:
        dict_mean_angles[key] = np.mean(dict_angles[key])
        dict_mean_angles_deg[key] = np.mean(dict_mean_angles[key] * 180 / np.pi)

    return (dict_mean_angles, dict_mean_angles_deg, dict_length, report)


@dataclass(frozen=True)
class TrijunctionEdge:
    """One **geometric** trijunctional edge, with the three wedge angles measured on it.

    The per-edge counterpart of :func:`compute_angles_tri`'s per-triple-line averages, and the
    object `TensionComputationMethod.YoungDupreLocal` assembles its rows from. One
    instance per geometric edge, not one per half-edge: `half_edge_keys` records the
    de-duplicated multiplicity so a caller can check it rather than assume it.

    Attributes:
        vertices: the edge's two vertex keys, ordered `(min, max)`.
        length: the edge's length. Not accumulated, not over-counted.
        half_edge_keys: keys of every half-edge incident on this geometric edge, sorted. Its
            length is the visit multiplicity — 3 for a trijunctional edge on a closed mesh.
        keys: the three wedge keys `(min(e, g), f, max(e, g))`, one per adjacent region `f`.
        angles: the wedge angle in radians, aligned index-wise with `keys`.
    """

    vertices: tuple[int, int]
    length: float
    half_edge_keys: tuple[int, ...]
    keys: tuple[tuple[int, int, int], ...]
    angles: tuple[float, ...]


def compute_trijunction_edges(
    mesh: "DcelData",
    *,
    reading: str = CHORD_READING,
) -> list[TrijunctionEdge]:
    """List every geometric trijunctional edge with its three wedge angles, each edge once.

    The report-dropping wrapper around :func:`compute_trijunction_edges_with_report`; it emits
    an `AngleReadingCoverageWarning` in the report's place when a `"quadric"` reading fell back
    to chords anywhere.

    Args:
        mesh (DcelData): Mesh to analyze.
        reading (str, optional): one of
            `foambryo.second_order_angle_reading.ANGLE_READINGS`. Keyword-only. Defaults to
            `"chord"`.

    Returns:
        list[TrijunctionEdge]: one record per geometric trijunctional edge, in increasing order
            of lowest incident half-edge key.
    """
    trijunction_edges, report = compute_trijunction_edges_with_report(mesh, reading=reading)
    warn_on_partial_coverage(report)
    return trijunction_edges


def compute_trijunction_edges_with_report(
    mesh: "DcelData",
    *,
    reading: str = CHORD_READING,
) -> tuple[list[TrijunctionEdge], AngleReadingReport]:
    """:func:`compute_trijunction_edges`, plus the reading's coverage report.

    Uses the same convention helper as :func:`compute_angles_tri` (:func:`_wedges_at_half_edge`),
    so averaging this function's angles over the edges of a triple line reproduces
    `compute_angles_tri`'s mean for that line to machine precision — pinned by
    `tests/test_per_edge_young_dupre.py`.

    De-duplication is by the geometric edge's `(min, max)` vertex-key pair, keeping the
    lowest-keyed incident half-edge; the choice does not matter numerically (the three visits of
    every trijunctional edge in the 47 meshes of the benchmarking dataset were measured to be
    **bit-identical**)
    but it makes the output deterministic. Quadrijunctional edges are skipped, exactly as
    `compute_angles_tri` skips them.

    `reading` selects the angle convention exactly as :func:`compute_angles_tri` does, and it
    is the route by which `TensionComputationMethod.YoungDupreLocal` reaches the second-order
    reading. The edge set, the lengths and the de-duplication do not depend on it — only the
    angles do.

    Args:
        mesh (DcelData): Mesh to analyze.
        reading (str, optional): one of
            `foambryo.second_order_angle_reading.ANGLE_READINGS`. Keyword-only. Defaults to
            `"chord"`.

    Returns:
        tuple[list[TrijunctionEdge], AngleReadingReport]: one record per geometric trijunctional
            edge, in increasing order of lowest incident half-edge key, and which trijunction
            faces the reading covered.
    """
    rotations, report = rotations_for_reading(mesh, reading)
    oriented = reads_oriented_wedges(reading)
    incident: dict[tuple[int, int], list[int]] = {}
    for edge in mesh.half_edges:
        if len(edge.twin) > 1:
            geometric_edge = (min(edge.origin.key, edge.destination.key), max(edge.origin.key, edge.destination.key))
            incident.setdefault(geometric_edge, []).append(edge.key)

    trijunction_edges = []
    for geometric_edge, half_edge_keys in incident.items():
        edge = mesh.half_edges[min(half_edge_keys)]
        wedges = _wedges_at_half_edge(mesh, edge, rotations, geometric_edge, oriented=oriented)
        if wedges is None:
            continue
        trijunction_edges.append(
            TrijunctionEdge(
                vertices=geometric_edge,
                length=edge.length,
                half_edge_keys=tuple(sorted(half_edge_keys)),
                keys=tuple(key for key, _ in wedges),
                angles=tuple(float(angle) for _, angle in wedges),
            ),
        )

    trijunction_edges.sort(key=lambda record: record.half_edge_keys[0])
    return trijunction_edges, report
