"""Analytic phantoms with known answers.

Three cases, each with a closed-form multimaterial mesh (exact vertices/triangles/region
labels, no reconstruction involved) and a vectorised point-classifier usable to voxelise
the same geometry at any resolution:

- `doublet_*`: a symmetric two-cell doublet (two spherical-cap cells sharing a flat
  interface, in an exterior medium) with a prescribed `gamma_cc / gamma_cm` ratio.
- `tetrahedral_vertex_*`: the Plateau 0-junction (quadruple point) where 4 trijunction
  lines meet, each carrying 3 films at 120 degrees, with 109.47 degrees between the
  lines at the point.
- `kelvin_*`: a small cluster of BCC-lattice Voronoi cells (Kelvin's tetrakaidecahedron,
  flat-faced), as a many-junction topology/performance stress test only — see its
  docstring for why it is *not* a mechanical-equilibrium check.

All three phantoms serve both `delaunay-watershed-3d`'s and `foambryo`'s harnesses: the
mesh functions here emit the exact analytic `(V, F, regions)` foambryo needs; the
`*_label_at` point-classifiers are the same rule a voxeliser rasterises to build the
segmentation mask `delaunay-watershed-3d` needs. `delaunay-watershed-3d/dw3d_benchmarks/phantoms.py`
mirrors the label-classifier logic (kept in sync by derivation, not by import, since the
two packages must stay independently installable) to generate masks at 5 resolutions.

Derivations and their limit checks
-----------------------------------
**Doublet contact angle.** Two identical spherical-cap cells (tension `gamma_cm` on each
outer A-M/B-M interface) meet a flat A-B interface (tension `gamma_cc`) along a common
circular triple line. By the top/bottom mirror symmetry the A-B interface must be flat
and lie in the symmetry plane; force balance on a line element of the triple line (a
2-D Neumann-triangle balance in the meridian plane, `f_gamma_cm * t_A + f_gamma_cm * t_B
+ f_gamma_cc * t_AB = 0` with unit tangents `t_A = (cos theta, sin theta)`,
`t_B = (cos theta, -sin theta)`, `t_AB = (-1, 0)`) gives

    cos(theta) = gamma_cc / (2 * gamma_cm)                                     (*)

with `theta` the angle each outer surface's tangent makes with the interface plane at
the rim. Three angles are physically meaningful at the rim (they must sum to 360
degrees, verified below by direct point-classification, independent of any inference
code): the wedge swept through the exterior medium, `2 * theta`, and the two
(by symmetry, equal) wedges swept through each cell's interior, `180 - theta` each.

*Limit check (required):* `gamma_cc / gamma_cm = 1` gives `theta = 60`, so the
through-medium wedge `2*theta = 120` degrees — this is the equal-tension Neumann
triangle (Plateau's 120-degree law), the expected sanity limit. Confirmed both from (*)
directly and by a fine-resolution point-classification sweep in
`tests/test_phantoms.py`, so it is shipped as verified.

*Convention question — resolved.* On the committed
doublet mesh, `foambryo.geometry.compute_angles_tri` reproduces **all three** wedges — the
through-medium `2*theta` *and* both through-cell `180 - theta` — converging to the true
wedge as resolution increases (not the supplement `theta`). It is **relabel-invariant**
(the same physical mesh with cells renamed gives identical angles to ~1e-13 degrees), so
the same-region normal-flip heuristic (`geometry.py:397-403`) is geometric, not
label-order-dependent; its purpose is precisely to return the wedge regardless of where the
shared region sits in the label order. The YD matrix builders consume `cos(that wedge)`,
matching Neumann physics. An *earlier draft* of this docstring recorded the through-cell
keys "converging to `theta`" — that measurement was taken with a pre-fix
`_orient_outward_from_lower_label` that wound the A-B disk (the doublet's only
bounded-bounded face) the wrong way; the fix in this same function (see its docstring,
note (2)) was applied before commit, so the committed phantom has always returned the
wedge, and reversing only the disk winding reproduces the old supplement reading on demand
(`foambryo_benchmarks/run_angle_convention.py`, `tests/test_angle_convention.py`). There is no
`compute_angles_tri` defect. `doublet_predicted_angles` still returns the analytic wedges
(the exact reference), which `compute_angles_tri` now demonstrably converges to.

**Tetrahedral vertex.** Regular tetrahedron, vertices `V1..V4` centred at the origin
(so the direction vectors `e_i = V_i / |V_i|` satisfy `sum(e_i) = 0`, hence pairwise
`dot(e_i, e_j) = -1/3` for `i != j` — expand `|sum e_i|^2 = 4 + 2*C(4,2)*d = 0` to get
`d = -1/3`). Cell `i` (`i` in 1..4) is the pyramid from the origin `O` to the
tetrahedron's face opposite vertex `i` (the triangle of the other three vertices); cells
1..4 exactly tile the tetrahedron's interior, and the ray `O -> V_m` is where the three
cells *other than* `m` meet (cell `m` itself does not touch that ray).

*Limit checks (both required, both verified):*
- Angle between any two rays `O->V_i`, `O->V_j` = `arccos(-1/3)` = 109.47 degrees, by the
  construction above — an exact algebraic identity, not a numerical coincidence.
- Along each ray, the 3 meeting interfaces are related by the regular tetrahedron's
  3-fold rotation about that ray's axis (which permutes the other 3 vertices cyclically
  and is a genuine symmetry of the solid), so they are spaced at equal angles around the
  ray, i.e. 120 degrees each. Confirmed numerically in `tests/test_phantoms.py` by
  building the mesh and reading `foambryo.geometry.compute_angles_tri`'s own output —
  here (unlike the doublet) every trijunction is between three *flat* interfaces of the
  same kind, and `compute_angles_tri` reproduces 120.000 degrees exactly for all of them,
  matching the true wedge with no supplement ambiguity.
"""

from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Voronoi

TETRAHEDRAL_LINE_ANGLE_DEG = 120.0
TETRAHEDRAL_POINT_ANGLE_DEG = float(np.degrees(np.arccos(-1.0 / 3.0)))  # 109.47122...


def _orient_outward_from_lower_label(
    v0: NDArray[np.float64],
    v1: NDArray[np.float64],
    v2: NDArray[np.float64],
    ra: int,
    rb: int,
    interior_point: dict[int, NDArray[np.float64] | None],
) -> tuple[int, int, int]:
    """Return the vertex permutation orienting a triangle outward from its lower label.

    That is `(0, 1, 2)` or `(0, 2, 1)`, whichever makes the right-hand-rule normal point
    away from the lower-labelled region and into the higher-labelled one.

    This is not a free choice: `foambryo.geometry.compute_volume_cells` accumulates
    `+det([v0,v1,v2])` into `material_1` (always the lower label, since `DcelData`
    enforces `material_1 < material_2`) and `-det(...)` into `material_2`, for the
    triangle exactly as stored. For that divergence-theorem sum to equal the true
    enclosed volume of a region `R`, every one of `R`'s faces must be oriented outward
    from `R` — so a face must be oriented outward from `material_1` specifically,
    since that is the side receiving the `+` sign. This was confirmed empirically
    two ways: (1) on `benchmarking-dataset/000_mesh.rec`, a real reconstructed mesh,
    `DcelData.compute_volumes_cells()` gives positive volumes for every real (bounded)
    cell under exactly this rule, and a negative "volume" for the unbounded exterior
    (label 0, always the lowest label, so always outward-from-here for every face
    touching it); (2) a first version of this function had the "both regions bounded"
    branch backwards (outward from the *higher* label) while the "one side is the
    unbounded exterior" branch had it right — the two phantoms exercising mostly-outer
    faces (`doublet_mesh`, `tetrahedral_vertex_mesh`) still passed their volume checks
    by coincidence, but `kelvin_mesh` (which is mostly *inner*, bounded-bounded faces)
    did not, which is what surfaced the bug.

    Args:
        v0 (NDArray[np.float64]): First triangle vertex position.
        v1 (NDArray[np.float64]): Second triangle vertex position.
        v2 (NDArray[np.float64]): Third triangle vertex position.
        ra (int): One region label on this triangle.
        rb (int): The other region label on this triangle.
        interior_point (dict[int, NDArray[np.float64] | None]): Map region label -> a
            point known to be strictly interior to that region (for a star-shaped
            region, e.g. its centroid), or `None` for the unbounded exterior (label 0).

    Returns:
        tuple[int, int, int]: `(0, 1, 2)` if the input order already satisfies the
            convention, else `(0, 2, 1)`.
    """
    lo, hi = (ra, rb) if ra < rb else (rb, ra)
    normal = np.cross(v1 - v0, v2 - v0)
    p_lo, p_hi = interior_point.get(lo), interior_point.get(hi)
    if p_lo is not None and p_hi is not None:
        flip = np.dot(normal, p_hi - p_lo) < 0
    elif p_lo is None:
        face_centroid = (v0 + v1 + v2) / 3
        flip = np.dot(normal, face_centroid - p_hi) > 0
    else:
        message = "unexpected: exterior (label 0) is always the lower label"
        raise AssertionError(message)
    return (0, 2, 1) if flip else (0, 1, 2)


# =====================================================================
# Doublet
# =====================================================================
def doublet_predicted_angles(ratio: float) -> dict[str, float]:
    """Predicted rim (triple-line) wedge angles for the symmetric doublet, in degrees.

    See the module docstring for the derivation and its limit check. Values are the
    true physical dihedral wedge angles (verified by direct point-classification in
    `tests/test_phantoms.py`), not `foambryo.geometry.compute_angles_tri`'s raw output
    (see the module docstring's convention note on the through-cell entries).

    Args:
        ratio (float): `gamma_cc / gamma_cm`, in the physical range `(0, 2)`.

    Returns:
        dict[str, float]: `{"theta_deg", "angle_through_medium_deg", "angle_through_cell_deg"}`.

    Raises:
        ValueError: If `ratio` is outside `(0, 2)` (no equilibrium doublet shape exists there).
    """
    if not (0 < ratio < 2):
        message = f"ratio={ratio} outside the physical domain (0, 2)"
        raise ValueError(message)
    theta = np.degrees(np.arccos(ratio / 2.0))
    return {
        "theta_deg": theta,
        "angle_through_medium_deg": 2 * theta,
        "angle_through_cell_deg": 180 - theta,
    }


def _doublet_geometry(ratio: float, r0: float) -> tuple[float, float, float]:
    theta = np.arccos(ratio / 2.0)
    radius = r0 / np.sin(theta)
    center_offset = r0 / np.tan(theta)
    return theta, radius, center_offset


def doublet_label_at(points: NDArray[np.float64], ratio: float, r0: float = 1.0) -> NDArray[np.int64]:
    """Vectorised point classifier for the doublet: 0 = medium, 1 = cell A, 2 = cell B.

    Args:
        points (NDArray[np.float64]): Query points, shape `(n, 3)`.
        ratio (float): `gamma_cc / gamma_cm`, in `(0, 2)`.
        r0 (float, optional): Triple-line (rim) radius. Defaults to 1.0.

    Returns:
        NDArray[np.int64]: Label per point, shape `(n,)`.
    """
    _, radius, zc = _doublet_geometry(ratio, r0)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    labels = np.zeros(len(points), dtype=np.int64)
    in_a = (z > 0) & (x**2 + y**2 + (z - zc) ** 2 <= radius**2)
    in_b = (z < 0) & (x**2 + y**2 + (z + zc) ** 2 <= radius**2)
    labels[in_a] = 1
    labels[in_b] = 2
    return labels


def doublet_mesh(
    ratio: float,
    r0: float = 1.0,
    n_rings: int = 24,
    n_phi: int = 48,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Exact analytic mesh of the symmetric doublet.

    Two spherical-cap "domes" (regions 1 and 2) sharing a flat circular interface with
    the exterior medium (region 0) and with each other, meeting along one triple line
    (the rim, radius `r0`). Both domes and the flat interface share the same rim-ring
    vertices exactly (not merely close within tolerance), so the triple line is exact.

    Args:
        ratio (float): `gamma_cc / gamma_cm`, in `(0, 2)`.
        r0 (float, optional): Triple-line (rim) radius. Defaults to 1.0.
        n_rings (int, optional): Number of latitude rings per dome. Defaults to 24.
        n_phi (int, optional): Number of points per ring. Defaults to 48.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]: `(V, F, regions)`.
    """
    _, radius, zc = _doublet_geometry(ratio, r0)
    theta = np.arccos(ratio / 2.0)
    psi_max = np.pi - theta
    phis = 2 * np.pi * np.arange(n_phi) / n_phi

    verts: list[NDArray[np.float64]] = []

    def add_point(p: NDArray[np.float64]) -> int:
        verts.append(p)
        return len(verts) - 1

    def ring(psi: float, mirror: bool) -> NDArray[np.float64]:
        z = zc + radius * np.cos(psi)
        z = -z if mirror else z
        x = radius * np.sin(psi) * np.cos(phis)
        y = radius * np.sin(psi) * np.sin(phis)
        return np.stack([x, y, np.full(n_phi, z)], axis=1)

    pole_a = add_point(np.array([0.0, 0.0, zc + radius]))
    pole_b = add_point(np.array([0.0, 0.0, -(zc + radius)]))
    disk_center = add_point(np.array([0.0, 0.0, 0.0]))

    interior_point = {
        1: np.array([0.0, 0.0, (zc + radius) / 2]),
        2: np.array([0.0, 0.0, -(zc + radius) / 2]),
    }

    rings_a, rings_b = [], []
    for i in range(1, n_rings + 1):
        psi_i = i * psi_max / n_rings
        rings_a.append([add_point(p) for p in ring(psi_i, mirror=False)])
        rings_b.append([add_point(p) for p in ring(psi_i, mirror=True)] if i < n_rings else rings_a[-1])

    faces: list[list[int]] = []
    regions: list[tuple[int, int]] = []

    def add_triangle(idx: list[int], ra: int, rb: int) -> None:
        v0, v1, v2 = verts[idx[0]], verts[idx[1]], verts[idx[2]]
        order = _orient_outward_from_lower_label(v0, v1, v2, ra, rb, interior_point)
        faces.append([idx[k] for k in order])
        regions.append((min(ra, rb), max(ra, rb)))

    def add_cap(pole_idx: int, rings_idx: list[list[int]], cell_label: int) -> None:
        first_ring = rings_idx[0]
        for j in range(n_phi):
            add_triangle([pole_idx, first_ring[j], first_ring[(j + 1) % n_phi]], 0, cell_label)
        for i in range(len(rings_idx) - 1):
            ring_i, ring_ip1 = rings_idx[i], rings_idx[i + 1]
            for j in range(n_phi):
                jn = (j + 1) % n_phi
                add_triangle([ring_i[j], ring_i[jn], ring_ip1[jn]], 0, cell_label)
                add_triangle([ring_i[j], ring_ip1[jn], ring_ip1[j]], 0, cell_label)

    add_cap(pole_a, rings_a, 1)
    add_cap(pole_b, rings_b, 2)

    rim = rings_a[-1]
    for j in range(n_phi):
        add_triangle([disk_center, rim[j], rim[(j + 1) % n_phi]], 1, 2)

    return np.array(verts), np.array(faces, dtype=np.int64), np.array(regions, dtype=np.int64)


def _doublet_gauge_factor(ratio: float, gamma_cm: float, mean_tension: float | None) -> float:
    """Factor rescaling the physical tensions so their mean equals `mean_tension`.

    Only tension *ratios* are physical — the overall scale is a gauge, which is why
    `foambryo.infer_tensions` has to fix it (it imposes a mean, by default 1). A ground
    truth stated in a different gauge than the one the inference imposes produces a
    relative error equal to the gauge mismatch, swamping the actual inference error. For
    the doublet's three interfaces the physical mean is `(2 + ratio) * gamma_cm / 3`, so
    at `ratio = 0.5` a mean-1 comparison carries a spurious 20% error and at
    `ratio = 1.5` a spurious 14.3%; only `ratio = 1` is accidentally gauge-free.

    Pressures take the *same* factor, since `Delta_P = 2 * gamma / R` is linear in
    tension, which keeps the rescaled tensions and pressures one consistent equilibrium.

    Args:
        ratio (float): `gamma_cc / gamma_cm`.
        gamma_cm (float): Outer (cell-medium) tension.
        mean_tension (float | None): Target mean over the 3 interfaces, or `None` to keep
            the physical scale set by `gamma_cm`.

    Returns:
        float: The multiplicative factor (1.0 when `mean_tension` is `None`).
    """
    if mean_tension is None:
        return 1.0
    physical_mean = (2.0 + ratio) * gamma_cm / 3.0
    return mean_tension / physical_mean


def doublet_ground_truth_tensions(
    ratio: float,
    gamma_cm: float = 1.0,
    mean_tension: float | None = None,
) -> dict[tuple[int, int], float]:
    """Ground-truth tension dict matching `doublet_mesh`'s region labelling.

    Args:
        ratio (float): `gamma_cc / gamma_cm`.
        gamma_cm (float, optional): Outer (cell-medium) tension. Defaults to 1.0.
        mean_tension (float | None, optional): If given, rescale all three tensions so
            their mean equals this, matching the gauge `foambryo.infer_tensions` imposes.
            Pass the same value to `doublet_ground_truth_pressures` to keep the two
            consistent. See `_doublet_gauge_factor` for why this matters. Defaults to
            `None`, i.e. the physical scale where the outer tension is exactly `gamma_cm`.

    Returns:
        dict[tuple[int, int], float]: `{(0, 1): gamma_cm, (0, 2): gamma_cm, (1, 2): ratio * gamma_cm}`,
            times the gauge factor.
    """
    scale = _doublet_gauge_factor(ratio, gamma_cm, mean_tension)
    return {
        (0, 1): gamma_cm * scale,
        (0, 2): gamma_cm * scale,
        (1, 2): ratio * gamma_cm * scale,
    }


def doublet_ground_truth_pressures(
    ratio: float,
    gamma_cm: float = 1.0,
    r0: float = 1.0,
    mean_tension: float | None = None,
) -> dict[int, float]:
    """Ground-truth Young-Laplace pressure per region, matching `doublet_mesh`'s labelling.

    Derivation
    ----------
    Each outer interface is a spherical cap of radius `R = r0 / sin(theta)` (see
    `_doublet_geometry`) with its cell on the concave side and the medium outside. For a
    sphere both principal radii equal `R`, so Young-Laplace gives a jump of
    `gamma_cm * (1/R + 1/R) = 2 * gamma_cm / R` with the higher pressure inside:

        p_cell - p_medium = 2 * gamma_cm / R = 2 * gamma_cm * sin(theta) / r0

    Fixing `p_medium = 0` (foambryo's convention — `infer_pressures`' `base_pressure`,
    default 0, and the `benchmarking-dataset` ground truth both put the exterior at 0)
    gives both cells that same pressure, by the doublet's top/bottom mirror symmetry.

    Consistency check, and why this phantom is a real end-to-end case
    ----------------------------------------------------------------
    The A-B interface is flat, so Young-Laplace across *it* requires
    `p_A - p_B = gamma_cc * 0 = 0`. The symmetry-derived answer above already satisfies
    that, so the three interfaces impose three conditions on two unknown pressures and
    the system closes. That is not automatic: it holds because `theta` was fixed by the
    triple-line force balance in the first place, so shape and pressures come from one
    consistent equilibrium. It is what makes the doublet a complete validation case —
    tensions *and* pressures — rather than a shape with tensions bolted on. The test
    `test_doublet_pressures_satisfy_young_laplace_on_the_flat_interface` pins it.

    Not provided for the other two phantoms, deliberately: `tetrahedral_vertex_mesh` has
    only flat interfaces, so every pressure is 0 and a *relative* pressure error would be
    undefined for every cell; `kelvin_mesh` is not a mechanical equilibrium at all (see
    `kelvin_label_at`), so it has no ground-truth pressure to state.

    Args:
        ratio (float): `gamma_cc / gamma_cm`, in the physical range `(0, 2)`.
        gamma_cm (float, optional): Outer (cell-medium) tension. Defaults to 1.0.
        r0 (float, optional): Triple-line (rim) radius. Defaults to 1.0.
        mean_tension (float | None, optional): If given, rescale by the same gauge factor
            `doublet_ground_truth_tensions` uses for this `mean_tension`, so tensions and
            pressures stay one consistent equilibrium. Pressures are linear in tension, so
            the factor is shared. See `_doublet_gauge_factor`. Defaults to `None`.

    Returns:
        dict[int, float]: `{0: 0.0, 1: p, 2: p}` with `p = 2 * gamma_cm * sin(theta) / r0`,
            times the gauge factor.

    Raises:
        ValueError: If `ratio` is outside `(0, 2)`.
    """
    if not (0 < ratio < 2):
        message = f"ratio={ratio} outside the physical domain (0, 2)"
        raise ValueError(message)
    _, radius, _ = _doublet_geometry(ratio, r0)
    pressure = 2.0 * gamma_cm * _doublet_gauge_factor(ratio, gamma_cm, mean_tension) / radius
    return {0: 0.0, 1: pressure, 2: pressure}


# =====================================================================
# Tetrahedral vertex (Plateau 0-junction)
# =====================================================================
_TETRA_DIRECTIONS = np.array(
    [
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ],
    dtype=float,
) / np.sqrt(3)


def tetrahedral_vertex_label_at(points: NDArray[np.float64], scale: float = 1.0) -> NDArray[np.int64]:
    """Vectorised point classifier for the tetrahedral vertex: 0 = exterior, 1..4 = cells.

    A point is in cell `i` when it lies inside the bounding tetrahedron (`V_i = scale *
    e_i`) and is closer, by the `argmin` rule below, to direction `e_i` than to any
    other — equivalently, closer to the centroid of face `i` than to any other face's
    centroid, i.e. inside the pyramid from the origin to face `i` (see the module
    docstring for why cell `i`'s pyramid is over the face *opposite* vertex `i`, not
    vertex `i` itself: for a regular tetrahedron centred at the origin, `sum(V_i) = 0`,
    so face `i`'s centroid `(V_j+V_k+V_l)/3 = -V_i/3` points in `-e_i`, not `+e_i`).

    Args:
        points (NDArray[np.float64]): Query points, shape `(n, 3)`.
        scale (float, optional): Distance from the origin to each tetrahedron vertex. Defaults to 1.0.

    Returns:
        NDArray[np.int64]: Label per point, shape `(n,)` (0 for outside the tetrahedron).
    """
    vertices = _TETRA_DIRECTIONS * scale
    inside = np.ones(len(points), dtype=bool)
    for i in range(4):
        others = [j for j in range(4) if j != i]
        face_centroid = vertices[others].mean(axis=0)
        outward = -vertices[i] / np.linalg.norm(vertices[i])  # face i's outward normal, from the origin
        inside &= np.dot(points - face_centroid, outward) <= 1e-9
    labels = np.zeros(len(points), dtype=np.int64)
    dots = points @ _TETRA_DIRECTIONS.T  # (n, 4)
    nearest_pyramid = np.argmin(dots, axis=1) + 1  # argmin: see docstring
    labels[inside] = nearest_pyramid[inside]
    return labels


def tetrahedral_vertex_mesh(scale: float = 1.0) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    r"""Exact analytic mesh of the Plateau tetrahedral vertex: 5 vertices, 10 triangles.

    Vertices: the origin `O` (the quadruple point) and the 4 vertices of a regular
    tetrahedron `V1..V4` centred at `O`. Cell `i` is the tetrahedron `{O, Vj, Vk, Vl}`
    (`j,k,l` = the other 3 indices); its outer face is `(Vj,Vk,Vl)` (interface with
    exterior, region 0) and its 3 inner faces are `(O,Vj,Vk)`, `(O,Vj,Vl)`, `(O,Vk,Vl)`
    — each shared with exactly one other cell (see the module docstring for the
    complement rule: triangle `(O,Va,Vb)` borders cells `{c,d} = {1,2,3,4} \\ {a,b}`).

    Args:
        scale (float, optional): Distance from the origin to each tetrahedron vertex. Defaults to 1.0.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]: `(V, F, regions)`,
            with `V[0]` the origin (the quadruple point) and `V[1:5]` the 4 rays' endpoints.
    """
    vertices = _TETRA_DIRECTIONS * scale
    origin = np.zeros(3)
    v = np.vstack([origin[None, :], vertices])  # index 0 = O, 1..4 = V1..V4
    idx = [1, 2, 3, 4]

    interior_point = {i: (origin + sum(v[j] for j in idx if j != i)) / 4 for i in idx}

    faces: list[list[int]] = []
    regions: list[tuple[int, int]] = []

    def add_triangle(tri: list[int], ra: int, rb: int) -> None:
        v0, v1, v2 = v[tri[0]], v[tri[1]], v[tri[2]]
        order = _orient_outward_from_lower_label(v0, v1, v2, ra, rb, interior_point)
        faces.append([tri[k] for k in order])
        regions.append((min(ra, rb), max(ra, rb)))

    for i in idx:
        a, b, c = [j for j in idx if j != i]
        add_triangle([a, b, c], 0, i)

    for a, b in itertools.combinations(idx, 2):
        c, d = [j for j in idx if j not in (a, b)]
        add_triangle([0, a, b], c, d)

    return v, np.array(faces, dtype=np.int64), np.array(regions, dtype=np.int64)


def tetrahedral_vertex_ground_truth_tensions(gamma: float = 1.0) -> dict[tuple[int, int], float]:
    """Ground-truth tension dict matching `tetrahedral_vertex_mesh`'s labelling (all interfaces equal).

    Args:
        gamma (float, optional): Common tension value (the construction is only exactly
            at equilibrium, with 120/109.47-degree angles, when every interface carries
            the same tension). Defaults to 1.0.

    Returns:
        dict[tuple[int, int], float]: One entry per of the 10 interfaces (4 outer + 6 inner).
    """
    idx = [1, 2, 3, 4]
    tensions = {(0, i): gamma for i in idx}
    tensions.update({(min(a, b), max(a, b)): gamma for a, b in itertools.combinations(idx, 2)})
    return tensions


# =====================================================================
# Kelvin cell (BCC-lattice Voronoi tiling) — topology/performance stress test only
# =====================================================================
def kelvin_label_at(points: NDArray[np.float64], cell_size: float = 1.0) -> NDArray[np.int64]:
    """Vectorised point classifier for a periodic Kelvin (BCC Voronoi) foam.

    Not a mechanical equilibrium: real Kelvin cells need slightly curved faces to meet
    Plateau's laws exactly (this is precisely why the flat-faced tetrakaidecahedron was
    superseded, for the minimal-area problem, by Weaire-Phelan). This classifier's cell
    boundaries are flat (a BCC lattice's Voronoi diagram), so its trijunction angles are
    NOT 120/109.47 degrees and must not be used for an angle-convergence check — only
    for topology (valence, junction count) and performance stress-testing.

    Labels 1..15 index the origin lattice point and its 14 BCC Voronoi neighbours (see
    `_kelvin_kept_mask`), in a fixed enumeration order; 0 is exterior (everything past
    that fixed 15-cell block). This block size is currently the only one implemented.

    Args:
        points (NDArray[np.float64]): Query points, shape `(n, 3)`.
        cell_size (float, optional): BCC lattice spacing. Defaults to 1.0.

    Returns:
        NDArray[np.int64]: Label per point, shape `(n,)`.
    """
    centers = _kelvin_lattice_points(cell_size)
    d2 = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    nearest = np.argmin(d2, axis=1)
    kept = _kelvin_kept_mask(centers, cell_size)
    label_of_center = np.zeros(len(centers), dtype=np.int64)
    label_of_center[kept] = np.arange(1, kept.sum() + 1)
    return label_of_center[nearest]


def _kelvin_lattice_points(cell_size: float, padding_shells: int = 2) -> NDArray[np.float64]:
    """BCC lattice points (integer and half-integer-offset) in `[-padding_shells, padding_shells]^3`.

    `padding_shells` must be large enough that the 15 kept cells' Voronoi regions (see
    `_kelvin_kept_mask`) are bounded; 2 was checked to give a bounded, no-`-1`-vertex
    Voronoi diagram for the fixed 15-cell block (`tests/test_phantoms.py`).
    """
    axis = np.arange(-padding_shells, padding_shells + 1)
    integer_pts = np.array(list(itertools.product(axis, axis, axis)), dtype=float)
    half_pts = integer_pts + 0.5
    return np.vstack([integer_pts, half_pts]) * cell_size


def _kelvin_kept_mask(centers: NDArray[np.float64], cell_size: float) -> NDArray[np.bool_]:
    """True for the origin and its 14 nearest BCC neighbours — the 15-cell labelled block.

    Nearest by Euclidean distance `<= cell_size`: the 6 axis-aligned integer neighbours
    are at distance exactly `cell_size`, the 8 half-integer-offset neighbours at
    `cell_size * sqrt(3)/2`. The rest of `centers` exist only to bound their Voronoi
    cells. A plain Chebyshev cutoff would over-include: e.g. the integer point `(1,1,0)`
    has Chebyshev norm 1 but Euclidean distance `sqrt(2) * cell_size`, and is not a
    Voronoi neighbour of the origin.
    """
    return np.linalg.norm(centers, axis=1) <= cell_size + 1e-9


def kelvin_mesh(
    cell_size: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64], list[int]]:
    """Exact analytic mesh of a small cluster of Kelvin (BCC Voronoi) cells.

    Built as the actual Voronoi diagram of a padded BCC lattice (`scipy.spatial.Voronoi`
    on exact lattice coordinates), not a hand-placed truncated-octahedron template — the
    faces are therefore exactly the flat bisector polygons the classifier in
    `kelvin_label_at` implies, fan-triangulated. See `kelvin_label_at`'s docstring for
    why this is a topology/performance stress test, not an angle-accuracy phantom.
    Labels the fixed 15-cell block (origin + its 14 BCC Voronoi neighbours; see
    `_kelvin_kept_mask`) — a different block size is not currently implemented.

    Args:
        cell_size (float, optional): BCC lattice spacing. Defaults to 1.0.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64], list[int]]:
            `(V, F, regions, cell_labels)` — `cell_labels` lists the labels actually
            present as bounded interior cells (1-indexed; region 0 is exterior).
    """
    centers = _kelvin_lattice_points(cell_size)
    kept_mask = _kelvin_kept_mask(centers, cell_size)
    kept_indices = np.where(kept_mask)[0]
    label_of_point = {int(pt_idx): i + 1 for i, pt_idx in enumerate(kept_indices)}

    vor = Voronoi(centers)

    interior_point = {label: centers[pt_idx] for pt_idx, label in label_of_point.items()}

    verts: list[NDArray[np.float64]] = list(vor.vertices)
    faces: list[list[int]] = []
    regions: list[tuple[int, int]] = []

    for (pt_a, pt_b), ridge in zip(vor.ridge_points, vor.ridge_vertices, strict=True):
        if -1 in ridge:
            continue  # unbounded ridge, only possible for a point outside the padded block
        label_a = label_of_point.get(int(pt_a))
        label_b = label_of_point.get(int(pt_b))
        if label_a is None and label_b is None:
            continue  # ridge between two un-labelled padding points: not part of the block
        ra = label_a if label_a is not None else 0
        rb = label_b if label_b is not None else 0
        if ra == rb:
            continue  # both sides mapped to exterior (shouldn't happen: pt_a != pt_b), skip defensively
        for k in range(1, len(ridge) - 1):
            tri = [ridge[0], ridge[k], ridge[k + 1]]
            v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
            order = _orient_outward_from_lower_label(v0, v1, v2, ra, rb, interior_point)
            faces.append([tri[o] for o in order])
            regions.append((min(ra, rb), max(ra, rb)))

    v_arr, f_arr = np.array(verts), np.array(faces, dtype=np.int64)
    used = np.unique(f_arr)
    remap = -np.ones(len(v_arr), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return v_arr[used], remap[f_arr], np.array(regions, dtype=np.int64), sorted(label_of_point.values())
