"""Mean stress tensor of each cell, and its split into principal parts, for the force viewer.

Pure numerics (no Polyscope), so they can be tested without a display. Only
`foambryo.viewing` uses them; nothing in the inference does.

Formula (G. K. Batchelor, J. Fluid Mech. 41, 545, 1970, for a suspension of drops): the
volume-averaged stress in a cell of volume V, pressure p, bounded by membranes m of tension
t_m, is

    sigma = -p I + (1 / V) * sum_m t_m * integral_m (I - n n) dA,

with n the unit normal of the membrane. Sign convention: **tension positive** — a negative
principal value is compressive, a positive one tensile. `I - n n` is unchanged when n is
reversed, so the orientation of a membrane's triangles does not enter. A membrane shared by
two cells is split between them (t_m / 2 each); a membrane against an exterior material
counts in full for the one cell it bounds.

Units: those of the inputs — pressure, and tension divided by length (tension x area /
volume), which agree when tension and pressure come from the same inference.
"""

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numpy.typing import NDArray

from foambryo.dcel import compute_faces_areas, compute_normal_faces

if TYPE_CHECKING:
    from collections.abc import Iterable

    from foambryo.dcel import DcelData


def cell_mean_stress_tensors(
    mesh: "DcelData",
    dict_tensions: dict[tuple[int, int], float],
    dict_pressure: dict[int, float],
    volumes: dict[int, float],
    cells: "Iterable[int]",
    lumen_materials: "Iterable[int]" = (0,),
) -> dict[int, NDArray[np.float64]]:
    """Volume-averaged stress tensor of each requested cell (tension positive).

    Args:
        mesh (DcelData): The mesh; `mesh.f` rows are `(v0, v1, v2, material_1, material_2)`.
        dict_tensions (dict[tuple[int, int], float]): Tension of every interface of the mesh,
            keyed by its sorted material pair.
        dict_pressure (dict[int, float]): Pressure of each cell, in the units of the tensions
            divided by a length. A cell whose pressure is `nan` (undetermined by a Laplace
            method) gets a `nan` tensor: leave it out of `cells`, as the stress viewer does.
        volumes (dict[int, float]): Volume of each cell (positive).
        cells (Iterable[int]): The cells to compute.
        lumen_materials (Iterable[int]): Materials treated as exterior: a membrane against one
            of them counts in full, any other membrane half. Defaults to `(0,)`.

    Returns:
        dict[int, NDArray[np.float64]]: Cell -> symmetric 3x3 mean stress tensor,
            `-p I + (1/V) sum_m t_m sum_faces A (I - n n)`.
    """
    lumen_materials = set(lumen_materials)
    delta = np.identity(3)

    membranes_of_cell: dict[int, list[tuple[int, int]]] = {key: [] for key in mesh.materials}
    for key in dict_tensions:
        a, b = key
        membranes_of_cell[a].append(key)
        membranes_of_cell[b].append(key)

    faces_of_membrane: dict[tuple[int, int], list[int]] = {key: [] for key in dict_tensions}
    for i, face in enumerate(mesh.f):
        a, b = face[3:]
        # A crop interface (only crop faces) is absent from the tension map: no membrane, no stress.
        if (a, b) in faces_of_membrane:
            faces_of_membrane[(a, b)].append(i)

    areas = compute_faces_areas(mesh.v, mesh.f)
    normals = compute_normal_faces(mesh.v, mesh.f)
    area_weighted_projectors = np.array([delta - np.tensordot(n, n, axes=0) for n in normals])
    for i in range(len(area_weighted_projectors)):
        area_weighted_projectors[i] *= areas[i]

    tensors = {}
    for c in cells:
        mean_stress = np.zeros((3, 3))
        mean_stress += -dict_pressure[c] * delta
        v = volumes[c]
        for m in membranes_of_cell[c]:
            a, b = m
            t = dict_tensions[m] if a in lumen_materials or b in lumen_materials else dict_tensions[m] / 2
            projector_sum = 0
            for nt in faces_of_membrane[m]:
                projector_sum += area_weighted_projectors[nt]
            mean_stress += t / v * projector_sum
        tensors[c] = mean_stress
    return tensors


class PrincipalStressAxes(NamedTuple):
    """A stress tensor's principal axes, split by the sign of their principal value.

    Attributes:
        compression (NDArray[np.float64]): 3x3; column k is the k-th unit eigenvector times its
            eigenvalue when that eigenvalue is < 0 (compressive, tension-positive convention), else 0.
        tension (NDArray[np.float64]): 3x3; the same for eigenvalues >= 0 (tensile).
    """

    compression: NDArray[np.float64]
    tension: NDArray[np.float64]


def split_principal_stresses(tensor: NDArray[np.float64]) -> PrincipalStressAxes:
    """Principal directions of a symmetric stress tensor, scaled by their principal values.

    Args:
        tensor (NDArray[np.float64]): A symmetric 3x3 tensor (it is symmetrised first, which
            leaves an exactly symmetric one unchanged).

    Returns:
        PrincipalStressAxes: `(compression, tension)`, two real 3x3 arrays. Column k of
            `compression` is the k-th unit eigenvector times its eigenvalue when that eigenvalue
            is < 0 (compressive), else zero; `tension` holds the non-negative ones (tensile) the
            same way. Every eigenvalue lands in exactly one of the two, scaling its own column.
            The columns follow `numpy.linalg.eigh`'s ascending order, and each eigenvector's sign
            is arbitrary: read the columns as a set of axes, not by index or orientation.
    """
    vals, vects = np.linalg.eigh((tensor + tensor.T) / 2)
    compression = vects.copy()
    tension = vects.copy()
    for k in range(3):
        if vals[k] < 0:
            compression[:, k] *= vals[k]
            tension[:, k] *= 0
        else:
            compression[:, k] *= 0
            tension[:, k] *= vals[k]
    return PrincipalStressAxes(compression, tension)
