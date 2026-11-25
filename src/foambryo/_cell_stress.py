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

A 2-D mesh gives the 2-D form, 2 x 2 in `(row, column)`: `sigma = -p I + (1 / A) * sum_m t_m *
integral_m (I - n n) dl`, the area `A` in `volumes`' place (definition, derivation and the cut
cells in `foambryo._planar_curvature_and_stress`). `split_principal_stresses` takes a tensor of
either size.

**The deviatoric stress** (`part="deviatoric"`, `deviatoric_part`). In `d` dimensions (2 or 3),

    sigma_dev = sigma - (tr sigma / d) I = T - (tr T / d) I,    T = (1 / V) * sum_m t_m * integral_m (I - n n) dA,

since the pressure term `-p I` is isotropic and drops out: the deviatoric stress depends only on the tensions and on
the geometry of the cell's boundary, never on the pressures, and is computed without them. It is traceless; it shows
the cell's anisotropy, how much more the cell is pulled along one direction than across it (its eigenvalues, `+s`
and `-s` in 2D), and the principal directions of its mean stress, which are the full stress's own (removing a
multiple of `I` changes no eigenvector). What it leaves out is the isotropic part, `(tr sigma / d) I`, the mean
normal stress `-p + tr T / d`: whether the cell as a whole is under tension or compression, which needs the pressure.

Why it holds where the full stress does not. A cell confined by an eggshell, or pressed on a substrate, has its
interfaces' curvature imposed by the load: Laplace's law does not give its pressure, and the full stress is not a
measurement. The tensions may still be, when they are read from the junction angles alone, near the junctions,
without assuming a circular interface (in 2D the `"fitted_tangent"` reading; the `"arc"` reading fits each interface
to a circle, i.e. assumes Laplace's law): force balance at a junction does not involve the pressures. With such
tensions, the deviatoric stress is a measurement in a confined embryo too. Assumptions: each interface's tension is
uniform along it (no tangential traction from the shell or substrate), and a membrane against the confining body is
counted as one against the medium (in full).
"""

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numpy.typing import NDArray

from foambryo.dcel import compute_faces_areas, compute_normal_faces

if TYPE_CHECKING:
    from collections.abc import Iterable

    from foambryo.dcel import DcelData

#: The parts of a cell's mean stress that can be asked for: the full tensor (the default), or its deviatoric part.
STRESS_PARTS = ("full", "deviatoric")


def check_stress_part(part: str) -> None:
    """Raise `ValueError` unless `part` is one of `STRESS_PARTS`."""
    if part not in STRESS_PARTS:
        message = f"part (stress) must be one of {STRESS_PARTS}, got {part!r}"
        raise ValueError(message)


def deviatoric_part(tensor: NDArray[np.float64]) -> NDArray[np.float64]:
    """The deviatoric part of a `d x d` tensor, `tensor - (tr tensor / d) I` (traceless; module docstring)."""
    tensor = np.asarray(tensor, dtype=np.float64)
    return tensor - np.trace(tensor) / len(tensor) * np.identity(len(tensor))


def cell_mean_stress_tensors(
    mesh: "DcelData",
    dict_tensions: dict[tuple[int, int], float],
    dict_pressure: dict[int, float],
    volumes: dict[int, float],
    cells: "Iterable[int]",
    lumen_materials: "Iterable[int]" = (0,),
    *,
    part: str = "full",
) -> dict[int, NDArray[np.float64]]:
    """Volume-averaged stress tensor of each requested cell (tension positive), or its deviatoric part.

    Args:
        mesh (DcelData): The mesh; `mesh.f` rows are `(v0, v1, v2, material_1, material_2)`.
        dict_tensions (dict[tuple[int, int], float]): Tension of every interface of the mesh,
            keyed by its sorted material pair.
        dict_pressure (dict[int, float]): Pressure of each cell, in the units of the tensions
            divided by a length. A cell whose pressure is `nan` (undetermined by a Laplace
            method) gets a `nan` tensor: leave it out of `cells`, as the stress viewer does.
            Not read with `part="deviatoric"` (it may be `None` then).
        volumes (dict[int, float]): Volume of each cell (positive).
        cells (Iterable[int]): The cells to compute.
        lumen_materials (Iterable[int]): Materials treated as exterior: a membrane against one
            of them counts in full, any other membrane half. Defaults to `(0,)`.
        part (str): `"full"` (the default) or `"deviatoric"`: the tensor's deviatoric part,
            `sigma - (tr sigma / d) I`, computed from the tensions and the geometry alone (module
            docstring: what it shows, why it holds in a confined embryo, what it leaves out).
            Keyword-only.

    Returns:
        dict[int, NDArray[np.float64]]: Cell -> symmetric 3x3 mean stress tensor,
            `-p I + (1/V) sum_m t_m sum_faces A (I - n n)`, or its traceless deviatoric part; for a
            2-D mesh (whose `volumes` are the cells' areas), 2x2 in `(row, column)`.

    Raises:
        ValueError: for a `part` not in `STRESS_PARTS`.
    """
    check_stress_part(part)
    if not hasattr(mesh, "f"):
        from foambryo._planar_curvature_and_stress import cell_mean_stress_tensors as planar_tensors

        return planar_tensors(mesh, dict_tensions, dict_pressure, volumes, cells, lumen_materials, part=part)
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
        if part == "full":
            mean_stress += -dict_pressure[c] * delta
        v = volumes[c]
        for m in membranes_of_cell[c]:
            a, b = m
            t = dict_tensions[m] if a in lumen_materials or b in lumen_materials else dict_tensions[m] / 2
            projector_sum = 0
            for nt in faces_of_membrane[m]:
                projector_sum += area_weighted_projectors[nt]
            mean_stress += t / v * projector_sum
        tensors[c] = mean_stress if part == "full" else deviatoric_part(mean_stress)
    return tensors


class PrincipalStressAxes(NamedTuple):
    """A stress tensor's principal axes, split by the sign of their principal value.

    Attributes:
        compression (NDArray[np.float64]): 3x3 (2x2 in 2D); column k is the k-th unit eigenvector times its
            eigenvalue when that eigenvalue is < 0 (compressive, tension-positive convention), else 0.
        tension (NDArray[np.float64]): 3x3 (2x2 in 2D); the same for eigenvalues >= 0 (tensile).
    """

    compression: NDArray[np.float64]
    tension: NDArray[np.float64]


def split_principal_stresses(tensor: NDArray[np.float64]) -> PrincipalStressAxes:
    """Principal directions of a symmetric stress tensor, scaled by their principal values.

    Args:
        tensor (NDArray[np.float64]): A symmetric 3x3 tensor, or 2x2 in 2D (it is symmetrised first,
            which leaves an exactly symmetric one unchanged).

    Returns:
        PrincipalStressAxes: `(compression, tension)`, two real arrays of the tensor's size. Column k of
            `compression` is the k-th unit eigenvector times its eigenvalue when that eigenvalue
            is < 0 (compressive), else zero; `tension` holds the non-negative ones (tensile) the
            same way. Every eigenvalue lands in exactly one of the two, scaling its own column.
            The columns follow `numpy.linalg.eigh`'s ascending order, and each eigenvector's sign
            is arbitrary: read the columns as a set of axes, not by index or orientation.
    """
    vals, vects = np.linalg.eigh((tensor + tensor.T) / 2)
    compression = vects.copy()
    tension = vects.copy()
    for k in range(len(vals)):
        if vals[k] < 0:
            compression[:, k] *= vals[k]
            tension[:, k] *= 0
        else:
            compression[:, k] *= 0
            tension[:, k] *= vals[k]
    return PrincipalStressAxes(compression, tension)
