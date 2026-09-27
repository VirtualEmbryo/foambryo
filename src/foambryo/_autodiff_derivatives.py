"""Autodiff (PyTorch) reference implementations of the area/volume/length derivatives.

These are the original float32 autodiff derivative paths (Sacha Ichbiah 2021,
Matthieu Perez 2024). As of the sparse-assembly work they are **no longer on any inference code path**:
`DcelData.compute_area_derivatives` / `compute_volume_derivatives` now return the
closed-form float64 dicts, and the variational solve assembles its operator sparsely
(`foambryo.geometry.compute_area_derivative_sparse` / `compute_volume_derivative_sparse`).

They are retained here only as a **gradient check**: an independent, autodiff-computed
derivative that the closed-form path is asserted to agree with to ~1e-6
(`tests/test_variational_sparse_assembly.py`). That is their real value — cross-checking
the hand-derived gradients — which is why `torch` is now an **optional** dependency
(`pip install foambryo[autodiff]`) rather than a runtime requirement. Importing this
module requires `torch`; importing `foambryo` itself does not.

This autodiff path casts to `torch.float` (float32) while the closed form is float64, so
the two are *not* bit equivalent and the autodiff path was the *less* accurate default. The
sparse-assembly work measured the discrepancy at ~1e-5 relative on the interface/cell
gradients and ~2e-7 relative on the resulting variational tensions, and made the closed form
the default accordingly.
"""

from typing import TYPE_CHECKING

import numpy as np
import torch
from numpy.typing import NDArray

from foambryo.geometry import extract_edges_trijunctions

if TYPE_CHECKING:
    from foambryo.dcel import DcelData

# The original module set this at import time as a global side effect. It is kept here,
# where it fires only when the autodiff path is explicitly imported (the gradient-check
# test), instead of on every `import foambryo`.
torch.use_deterministic_algorithms(True)


def compute_area_derivative_autodiff(
    mesh: "DcelData",
    device: str = "cpu",
) -> dict[tuple[int, int], NDArray[np.float64]]:
    """Compute dict that maps interface (label1, label2) to array of change of area per point."""
    # Faces_membranes = extract_faces_membranes(Mesh)
    key_mult = np.amax(mesh.f[:, 3:]) + 1
    keys = mesh.f[:, 3] + key_mult * mesh.f[:, 4]
    faces_membranes = {}
    for key in np.unique(keys):
        tup = (key % key_mult, key // key_mult)
        faces_membranes[tup] = mesh.f[:, :3][np.arange(len(keys))[keys == key]]

    verts = torch.tensor(mesh.v, dtype=torch.float, requires_grad=True).to(device)
    optimizer = torch.optim.SGD([verts], lr=1)  # Useless, here just to reset the grad

    areas_derivatives = {}
    for tup in sorted(faces_membranes.keys()):
        loss_area = (compute_area_faces_torch(verts, torch.tensor(faces_membranes[tup]))).sum()
        loss_area.backward()
        areas_derivatives[tup] = (verts.grad).numpy().copy()
        optimizer.zero_grad()

    return areas_derivatives


def compute_volume_derivative_autodiff_dict(mesh: "DcelData", device: str = "cpu") -> dict[int, NDArray[np.float64]]:
    """Compute map cell number -> derivative of volume wrt to each point."""
    # Faces_manifolds = extract_faces_manifolds(Mesh)
    faces_manifolds = {key: [] for key in mesh.materials}
    for face in mesh.f:
        a, b, c, m1, m2 = face
        faces_manifolds[m1].append([a, b, c])
        faces_manifolds[m2].append([a, c, b])

    verts = torch.tensor(mesh.v, dtype=torch.float, requires_grad=True).to(device)
    optimizer = torch.optim.SGD([verts], lr=1)  # Useless, here just to reset the grad

    volumes_derivatives = {}
    for key in mesh.materials:  # 1:] :
        faces = faces_manifolds[key]
        assert len(faces) > 0
        loss_volume = -compute_volume_manifold_torch(verts, torch.tensor(faces))
        loss_volume.backward()
        volumes_derivatives[key] = verts.grad.numpy().copy()
        optimizer.zero_grad()

    return volumes_derivatives


def compute_length_derivative_autodiff(
    mesh: "DcelData",
    device: str = "cpu",
) -> dict[tuple[int, int], NDArray[np.float64]]:
    """Compute map trijunction edge (V1, V2) -> change of length wrt to points."""
    edges_trijunctions = extract_edges_trijunctions(mesh)

    verts = torch.tensor(mesh.v, dtype=torch.float, requires_grad=True).to(device)
    optimizer = torch.optim.SGD([verts], lr=1)  # Useless, here just to reset the grad

    length_derivatives: dict[tuple[int, int], NDArray[np.float64]] = {}
    for tup in sorted(edges_trijunctions.keys()):
        loss_length = (compute_length_edges_trijunctions_torch(verts, torch.tensor(edges_trijunctions[tup]))).sum()
        loss_length.backward()
        length_derivatives[tup] = (verts.grad).numpy().copy()
        optimizer.zero_grad()

    return length_derivatives


def compute_length_edges_trijunctions_torch(
    points: "torch.FloatTensor",
    edges_trijunctions: "torch.IntTensor",
) -> "torch.FloatTensor":
    """Compute the length of each edge of a trijunction using torch because why not."""
    positions = points[edges_trijunctions]
    return torch.norm(positions[:, 0] - positions[:, 1], dim=1)


def compute_area_faces_torch(
    points: "torch.FloatTensor",
    triangles: "torch.IntTensor",
) -> "torch.FloatTensor":
    """Compute area of every triangle using torch."""
    positions = points[triangles]
    sides = positions - positions[:, [2, 0, 1]]

    lengths_sides = torch.norm(sides, dim=2)
    half_perimeters = torch.sum(lengths_sides, axis=1) / 2
    diffs = torch.zeros(lengths_sides.shape)
    diffs[:, 0] = half_perimeters - lengths_sides[:, 0]
    diffs[:, 1] = half_perimeters - lengths_sides[:, 1]
    diffs[:, 2] = half_perimeters - lengths_sides[:, 2]
    return (half_perimeters * diffs[:, 0] * diffs[:, 1] * diffs[:, 2]) ** (0.5)


def compute_volume_manifold_torch(
    points: "torch.FloatTensor",
    triangles: "torch.IntTensor",
) -> "torch.FloatTensor":
    """Compute volume of each cell."""
    coords = points[triangles]
    cross_prods = torch.cross(coords[:, 1], coords[:, 2], axis=1)
    dots = torch.sum(cross_prods * coords[:, 0], axis=1)
    return -torch.sum(dots) / 6
