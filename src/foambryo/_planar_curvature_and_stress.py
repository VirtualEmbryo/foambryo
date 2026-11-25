r"""Curvature of the interfaces and mean stress of the cells of a 2-D mesh (private module; provisional).

The 2-D counterparts of `foambryo.curvature` (the mean curvature of a 3-D mesh's interfaces) and of
`foambryo._cell_stress` (each cell's mean stress tensor), from the same physics and in the same conventions. The
public names dispatch here when given a 2-D mesh (`foambryo.PlanarMesh`, delaunay-watershed-2d's `DCEL_Data`, a
`(points, segments, labels)` triple or a `(vertices, edges)` pair): `foambryo.curvature.compute_curvature_interfaces`,
`foambryo.curvature.compute_curvature_vertices_cotan`, and the force viewer's stress (`foambryo._cell_stress`).

**Frame and units.** Everything is in the mesh's own frame, `(row, column)`, and its own length unit (pixels, or the
unit of a mesh built in physical units). A curvature is in inverse length units; a stress has the units of a pressure,
which in 2D is a force per length, the units of a tension divided by a length. Tensions and pressures are taken as
given, in one consistent unit (relative units when they come from foambryo's inference).

**The curvature of an interface, and Laplace's law.** A 2-D interface is a curve; it has one curvature, `kappa`. The
interface between labels `a < b` gets the signed curvature

.. math:: \kappa_{ab} = \pm 1/R, \qquad \kappa_{ab} > 0 \text{ when the centre of curvature lies on } b\text{'s side},

that is, positive along the unit normal from `a` into `b` when that normal points to the centre. With this sign,
Laplace's law reads, as in 3D,

.. math:: p_a - p_b = -\gamma_{ab}\, \kappa_{ab} \qquad (\text{3D: } P_a - P_b = -2 H_{ab} T_{ab}),

where `gamma_ab` is the interface's tension (a force) and `p` the pressures (force per length): the curve's one
curvature takes the place of the surface's `2 H` (the sum of its two principal curvatures), and the pressure is higher
on the side that holds the centre of curvature (a cell alone in the medium has `kappa_01 = +1/R` and
`p_1 - p_0 = gamma / R`), as foambryo's 3-D `compute_curvature_interfaces` gives `H_01 = +1/R` for a sphere (measured:
0.109 for a ball of radius 9 voxels). It is the same law as the 2-D inference's `gamma * kappa = p_left - p_right`
(`foambryo.InterfaceArc.curvature`, positive when the circle's centre is on the left of the arc's traversal in
`(x, y) = (column, row)`): `kappa_ab = -arc.curvature` when `a` is on the left of the arc, `+arc.curvature` when `b`
is.

Two readings of `kappa_ab` (`reading`):

* `"arc"` (the default): the circle the 2-D inference fits to each interface (`foambryo.interface_arcs`, the reading
  the Laplace pressures use): `1/R`, 0 for a straight interface, `nan` for one read by its chord (fewer than 4
  vertices). An interface in several pieces (split by the image border) is the length-weighted mean of its pieces'
  curvatures (`weighted=True`), or their plain mean.
* `"local"`: the mean of the vertex curvatures below over the interface's inner vertices (not the junctions, the ends of
  an interface or the image border's vertices), weighted by each vertex's dual length (`weighted=True`, the 2-D
  counterpart of 3-D's area weighting, `PressureComputationMethod.WeightedLaplace`) or equally (`Laplace`); `nan` for
  an interface without inner vertex.

**The curvature at a vertex** (`vertex_curvatures`, the 2-D form of the cotangent Laplacian of
`foambryo.curvature.compute_curvature_vertices_cotan`). At an inner vertex `x_i` of an interface, between its
neighbours `x_{i-1}` and `x_{i+1}` along it, with edge lengths `l_1 = |x_i - x_{i-1}|`, `l_2 = |x_{i+1} - x_i|`, unit
tangents `t_1`, `t_2` and dual length `l_i = (l_1 + l_2) / 2`, the discrete Laplace-Beltrami operator of the curve
applied to its position (the cotangent Laplacian reduces to weights `1 / l` on a curve) is the curvature vector

.. math:: \mathbf{K}_i = \frac{\mathbf{t}_2 - \mathbf{t}_1}{\ell_i}, \qquad
          |\mathbf{K}_i| = \frac{2\sin(\phi_i/2)}{\ell_i},

with `phi_i` the turning angle at the vertex; it points to the centre of curvature. It is exact on a circle sampled at
equal arc steps: `l = 2 R sin(delta / 2)`, `phi = delta`, so `|K| = 1/R` whatever the step (the turning-angle form
`phi / l_i` would read `delta / (2 R sin(delta / 2))`, above `1/R`). The junctions (three or more materials), the ends
of an interface and the image border's vertices have no vertex curvature (`nan`).

**The mean stress of a cell** (`cell_mean_stress_tensors`; Batchelor G. K., J. Fluid Mech. 41, 545, 1970, as in 3D).
The stress averaged over cell `c` of area `A_c` and pressure `p_c`, bounded by interfaces `m` of tension `t_m`, is

.. math:: \bar\sigma_c = -p_c\, \mathbf{I} + \frac{1}{A_c} \sum_m t_m^{(c)} \int_m (\mathbf{I} - \mathbf{n}\mathbf{n})\,
          \mathrm{d}l = -p_c\, \mathbf{I}
          + \frac{1}{A_c} \sum_m t_m^{(c)} \sum_{e \in m} l_e\, \mathbf{t}_e \mathbf{t}_e,

a symmetric 2 x 2 tensor in the `(row, column)` frame. Derivation: inside the cell the cytoplasm is a fluid at rest,
`sigma = -p I`; an interface is a line under tension, whose stress is `t (I - n n)` concentrated on the line (in 2D,
`I - n n = t t`, the tangent's projector); averaging over the area gives the two terms. An interface between two cells
is shared, `t_m^(c) = t_m / 2`; one against a medium (`lumen_materials`, the medium 0) counts in full,
`t_m^(c) = t_m`, as in 3D. Sign convention: tension positive (a negative principal value is compressive). Units: the
pressure's, `t l / A` being a force per length. A body free of external load carries no mean stress: one circular cell
at its Laplace pressure has `sigma = 0` exactly in the continuum (`t (pi R) / (pi R^2) = t / R = p`), and a cluster of
cells in a medium at pressure 0 in mechanical equilibrium has `sum_c A_c sigma_c = 0`.

**When the pressures, hence the stresses, mean something.** Laplace's law, and the pressures inferred from it, hold only
for cells deformed by nothing but their own tensions and pressures. A tissue under an external load (an embryo confined
by its eggshell, cells on a substrate or against a wall) has interfaces whose curvature is imposed by that load, and may
change along one interface: Laplace's law does not give the pressures, and the stress tensor's pressure term, `-p I`,
is not a measurement either; only the tensions, read from the junction angles, may still be. The curvature itself is
geometry and is read the same way in every case; its relation to the pressures is the part that needs unconfined
cells. Nothing here can tell a confined tissue from a free one: that is the user's knowledge of the experiment.

**The deviatoric stress** (`part="deviatoric"`) holds there: `sigma_dev = sigma - (tr sigma / 2) I = T - (tr T / 2) I`
with `T = (1/A_c) sum_m t_m^(c) sum_e l_e t_e t_e`, the pressure term being isotropic. It needs no pressure and is
computed without one. It shows the cell's anisotropy (eigenvalues `+s` and `-s`, `s` = half the difference of the full
tensor's principal stresses) and its principal directions (those of the full tensor); it leaves out the isotropic part,
the mean normal stress `-p + tr T / 2`. With tensions read near the junctions without assuming circular interfaces
(`angle_reading="fitted_tangent"`; `"arc"` fits each interface to a circle, i.e. assumes Laplace's law), it is a
measurement in a confined embryo too, provided each interface's tension is uniform along it (no tangential traction
from the shell). Definition and reasons in `foambryo._cell_stress`.

Cells cut by the image border get no tensor: part of their boundary is the image's edge, not a membrane, and the
traction across it is unknown (in 3D, cut cells have no pressure, hence no stress, for the same reason).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Iterable

import numpy as np
from numpy.typing import NDArray

from foambryo._two_dimensional import _chains, _excluded, _mesh_arrays, interface_arcs

__all__ = [
    "CURVATURE_READINGS_2D",
    "DEFAULT_CURVATURE_READING_2D",
    "cell_areas_and_centroids",
    "cell_mean_stress_tensors",
    "interface_curvatures",
    "vertex_curvatures",
]

#: The two readings of a 2-D interface's curvature: the fitted circle, and the mean of the vertex curvatures.
CURVATURE_READINGS_2D = ("arc", "local")
DEFAULT_CURVATURE_READING_2D = "arc"

Interface = tuple[int, int]


def _inner_vertex_triples(edges: NDArray[np.int64], excluded: set[int]) -> list[tuple[Interface, int, int, int, int]]:
    """Every inner vertex of every interface: `(pair, left material, previous, vertex, next)` along the traversal.

    The traversal is that of the 2-D inference's interfaces (`_two_dimensional._chains`): chains of one material pair
    between junctions and excluded vertices, `left` on their left in `(x, y) = (column, row)`. A closed chain (a cell
    enclosed by one material) has every vertex inner.
    """
    out = []
    for pair, path, left, closed, _ in _chains(edges, excluded):
        if closed:
            ring = path[:-1]
            n = len(ring)
            if n < 3:
                continue
            out.extend((pair, left, ring[k - 1], ring[k], ring[(k + 1) % n]) for k in range(n))
        else:
            out.extend((pair, left, path[k - 1], path[k], path[k + 1]) for k in range(1, len(path) - 1))
    return out


def _curvature_vector(previous: NDArray, here: NDArray, following: NDArray) -> tuple[NDArray, float]:
    """`(K, dual length)` at `here`: `K = (t_2 - t_1) / l_dual`, `l_dual = (l_1 + l_2) / 2` (module docstring)."""
    first, second = here - previous, following - here
    l1, l2 = float(np.hypot(*first)), float(np.hypot(*second))
    dual = 0.5 * (l1 + l2)
    if l1 == 0.0 or l2 == 0.0:
        return np.full(2, np.nan), dual
    return (second / l2 - first / l1) / dual, dual


def _into_second_label(pair: Interface, left: int, tangent: NDArray) -> NDArray:
    """The unit normal of a traversal `tangent` (in `(row, column)`) that points into `pair[1]`, the larger label.

    The traversal has `left` on its left in `(x, y) = (column, row)`, where the left normal of `(t_x, t_y)` is
    `(-t_y, t_x)`; back in `(row, column)` it is `(t_column, -t_row)`.
    """
    into_left = np.array([tangent[1], -tangent[0]]) / np.hypot(*tangent)
    return into_left if left == pair[1] else -into_left


def vertex_curvatures(
    mesh: object,
    *,
    exclude_image_border: bool = True,
    excluded_vertices: Collection[int] = (),
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """The curvature at every vertex of a 2-D mesh: the 2-D form of `compute_curvature_vertices_cotan`'s three arrays.

    Args:
        mesh: a 2-D mesh (`foambryo.PlanarMesh`, dw2d's `DCEL_Data`, a triple or a pair).
        exclude_image_border: give the image border's vertices no curvature (`nan`), as the 2-D inference stops its
            interfaces there. Keyword-only.
        excluded_vertices: further vertices without curvature. Keyword-only.

    Returns:
        `(curvature, inverse_dual_lengths, curvature_vectors)`: `|K_i|` per vertex (inverse length units), `1 / l_i`
        (inverse length units), and `K_i` `(n, 2)` in `(row, column)`, pointing to the centre of curvature (module
        docstring); `nan` at the junctions, the ends of the interfaces and the excluded vertices.
    """
    vertices, edges = _mesh_arrays(mesh)
    excluded = _excluded(mesh, exclude_image_border, excluded_vertices)
    n = len(vertices)
    vectors = np.full((n, 2), np.nan)
    inverse_dual = np.full(n, np.nan)
    for _pair, _left, previous, here, following in _inner_vertex_triples(edges, excluded):
        vector, dual = _curvature_vector(vertices[previous], vertices[here], vertices[following])
        vectors[here] = vector
        inverse_dual[here] = 1.0 / dual if dual > 0 else np.nan
    return np.hypot(vectors[:, 0], vectors[:, 1]), inverse_dual, vectors


def _check_reading(reading: str) -> None:
    if reading not in CURVATURE_READINGS_2D:
        message = f"reading must be one of {CURVATURE_READINGS_2D} for a 2-D mesh, got {reading!r}"
        raise ValueError(message)


def interface_curvatures(
    mesh: object,
    weighted: bool = True,
    *,
    reading: str = DEFAULT_CURVATURE_READING_2D,
    exclude_image_border: bool = True,
    excluded_vertices: Collection[int] = (),
) -> dict[Interface, float]:
    """The signed curvature `kappa_ab` of every interface of a 2-D mesh, with Laplace's law `p_a - p_b = -gamma kappa`.

    Args:
        mesh: a 2-D mesh (`foambryo.PlanarMesh`, dw2d's `DCEL_Data`, a triple or a pair).
        weighted: weight the pieces of an interface by their lengths (`"arc"`) or its vertices by their dual lengths
            (`"local"`), rather than equally. Defaults to True.
        reading: `"arc"` (the circle the Laplace pressures read) or `"local"` (the vertex curvatures). Keyword-only.
        exclude_image_border: stop the interfaces at the image border's vertices, as the 2-D inference does.
            Keyword-only.
        excluded_vertices: further vertices where interfaces stop. Keyword-only.

    Returns:
        dict[tuple[int, int], float]: every material pair `(a, b)`, `a < b`, of the mesh's segments -> `kappa_ab`, in
            inverse length units, positive when the centre of curvature lies on `b`'s side; `nan` where the reading
            has no curvature (module docstring).

    Raises:
        ValueError: for an unknown `reading`.
    """
    _check_reading(reading)
    vertices, edges = _mesh_arrays(mesh)
    excluded = _excluded(mesh, exclude_image_border, excluded_vertices)
    pairs = sorted({(int(min(a, b)), int(max(a, b))) for a, b in edges[:, 2:4].tolist()})
    sums: dict[Interface, float] = defaultdict(float)
    weights: dict[Interface, float] = defaultdict(float)
    if reading == "arc":
        for arc in interface_arcs((vertices, edges), exclude_image_border=False, excluded_vertices=excluded):
            if not np.isfinite(arc.curvature):
                continue
            kappa = arc.curvature if arc.left == arc.materials[1] else -arc.curvature
            w = arc.length if weighted else 1.0
            sums[arc.materials] += w * kappa
            weights[arc.materials] += w
    else:
        for pair, left, previous, here, following in _inner_vertex_triples(edges, excluded):
            vector, dual = _curvature_vector(vertices[previous], vertices[here], vertices[following])
            if not np.all(np.isfinite(vector)):
                continue
            normal = _into_second_label(pair, left, vertices[following] - vertices[previous])
            kappa = float(np.sign(vector @ normal) * np.hypot(*vector))
            w = dual if weighted else 1.0
            sums[pair] += w * kappa
            weights[pair] += w
    return {pair: (sums[pair] / weights[pair] if weights[pair] > 0 else np.nan) for pair in pairs}


def cell_areas_and_centroids(mesh: object) -> tuple[dict[int, float], dict[int, NDArray[np.float64]]]:
    """The area and the area centroid `(row, column)` of every cell (every material but 0), from its boundary.

    Green's theorem on the segments: under the orientation rule each segment `p -> q` is the positively oriented
    boundary of its first label in `(row, column)` (its normal `(dc, -dr)` points out of it) and the negatively
    oriented boundary of its second, so `A_c = 1/2 sum (p_r q_c - p_c q_r)` over the segments of `c`, each with the
    sign `+1` where `c` is first and `-1` where it is second, and `centroid_c = 1/(6 A_c) sum (p + q) (p_r q_c - p_c
    q_r)` with the same signs. A cut cell is closed along the image border by its border segments.

    Returns:
        `(areas, centroids)`: cell -> area (squared length units, positive), cell -> `(2,)` centroid.
    """
    vertices, edges = _mesh_arrays(mesh)
    p, q = vertices[edges[:, 0]], vertices[edges[:, 1]]
    cross = p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0]
    moment = (p + q) * cross[:, None]
    areas: dict[int, float] = defaultdict(float)
    moments: dict[int, NDArray[np.float64]] = defaultdict(lambda: np.zeros(2))
    for k, (first, second) in enumerate(edges[:, 2:4].tolist()):
        for material, sign in ((first, 1.0), (second, -1.0)):
            if material == 0:
                continue
            areas[material] += sign * 0.5 * cross[k]
            moments[material] = moments[material] + sign * moment[k]
    centroids = {c: moments[c] / (6.0 * areas[c]) if areas[c] != 0 else np.full(2, np.nan) for c in areas}
    return dict(sorted(areas.items())), dict(sorted(centroids.items()))


def cell_mean_stress_tensors(
    mesh: object,
    dict_tensions: dict[Interface, float],
    dict_pressure: dict[int, float],
    areas: dict[int, float],
    cells: Iterable[int],
    lumen_materials: Iterable[int] = (0,),
    *,
    part: str = "full",
) -> dict[int, NDArray[np.float64]]:
    """Area-averaged stress tensor of each requested cell of a 2-D mesh (tension positive), 2 x 2 in `(row, column)`.

    Args:
        mesh: a 2-D mesh (`foambryo.PlanarMesh`, dw2d's `DCEL_Data`, a triple or a pair).
        dict_tensions: tension of every interface, keyed by its sorted material pair (an interface absent from it, or
            with a non-finite tension, makes the tensors of its cells `nan`).
        dict_pressure: pressure of each cell, in the units of the tensions divided by a length. Not read with
            `part="deviatoric"` (it may be `None` then).
        areas: area of each cell (positive; `cell_areas_and_centroids`).
        cells: the cells to compute. Leave out the cells cut by the image border (module docstring).
        lumen_materials: materials treated as exterior: an interface against one counts in full, any other half.
            Defaults to `(0,)`.
        part: `"full"` (the default) or `"deviatoric"`, the traceless part `sigma - (tr sigma / 2) I`, from the
            tensions and the geometry alone (module docstring). Keyword-only.

    Returns:
        dict[int, NDArray[np.float64]]: cell -> symmetric 2 x 2 mean stress tensor
            `-p I + (1/A) sum_m t_m sum_e l_e t_e t_e`, or its deviatoric part (module docstring).

    Raises:
        ValueError: for a `part` not in `foambryo._cell_stress.STRESS_PARTS`.
    """
    from foambryo._cell_stress import check_stress_part, deviatoric_part

    check_stress_part(part)
    lumen_materials = set(lumen_materials)
    vertices, edges = _mesh_arrays(mesh)
    d = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    lengths = np.hypot(d[:, 0], d[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        # l t t = d d / l: the tangent's projector times the length (a zero-length segment adds nothing)
        projectors = np.where(lengths[:, None, None] > 0, d[:, :, None] * d[:, None, :] / lengths[:, None, None], 0.0)
    projector_of_pair: dict[Interface, NDArray[np.float64]] = defaultdict(lambda: np.zeros((2, 2)))
    for k, (a, b) in enumerate(edges[:, 2:4].tolist()):
        key = (min(a, b), max(a, b))
        projector_of_pair[key] = projector_of_pair[key] + projectors[k]
    pairs_of_cell: dict[int, list[Interface]] = defaultdict(list)
    for key in projector_of_pair:
        pairs_of_cell[key[0]].append(key)
        pairs_of_cell[key[1]].append(key)

    tensors = {}
    for c in cells:
        mean_stress = -float(dict_pressure[c]) * np.identity(2) if part == "full" else np.zeros((2, 2))
        for key in pairs_of_cell[c]:
            tension = float(dict_tensions.get(key, np.nan))
            share = tension if (key[0] in lumen_materials or key[1] in lumen_materials) else tension / 2
            mean_stress = mean_stress + share / areas[c] * projector_of_pair[key]
        tensors[c] = mean_stress if part == "full" else deviatoric_part(mean_stress)
    return tensors
