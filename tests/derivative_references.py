"""Float64 reference derivatives of the interface areas, cell volumes and trijunction lengths.

An independent check of the closed-form derivatives in `foambryo.geometry`: each triangle's area and signed
volume term, and each trijunction edge's length, are differentiated numerically with respect to their own
vertex coordinates, then summed per key the way the closed form is keyed. Two schemes:

* the complex step, `f'(x) = Im f(x + i h) / h`: no subtraction, so accurate to rounding (`h = 1e-30`);
* central finite differences, `(f(x + h) - f(x - h)) / (2 h)`, `h` about `6e-6` times the mean edge length
  (the cube root of the double-precision epsilon, which balances truncation and rounding).

Norms are written `sqrt(v . v)`, never `abs`, so that the functions stay analytic for the complex step.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from foambryo.geometry import extract_edges_trijunctions

COMPLEX_STEP = 1e-30
RELATIVE_FINITE_DIFFERENCE_STEP = 6e-6


def _triangle_area(x: NDArray, y: NDArray, z: NDArray) -> NDArray:
    cross = np.cross(y - x, z - x)
    return np.sqrt(np.sum(cross * cross, axis=1)) / 2


def _triangle_volume_term(x: NDArray, y: NDArray, z: NDArray) -> NDArray:
    return np.sum(x * np.cross(y, z), axis=1) / 6


def _edge_length(p: NDArray, q: NDArray) -> NDArray:
    difference = p - q
    return np.sqrt(np.sum(difference * difference, axis=1))


def _local_derivatives(
    function: Callable[..., NDArray], corners: NDArray[np.float64], scheme: str, step: float,
) -> NDArray[np.float64]:
    """`(m, k, 3)` derivatives of `function` (one value per row) w.r.t. each corner coordinate of each row."""
    derivative = np.empty(corners.shape)
    for corner in range(corners.shape[1]):
        for component in range(3):
            if scheme == "complex_step":
                moved = corners.astype(np.complex128)
                moved[:, corner, component] += 1j * COMPLEX_STEP
                derivative[:, corner, component] = np.imag(function(*np.moveaxis(moved, 1, 0))) / COMPLEX_STEP
            elif scheme == "central_difference":
                plus, minus = corners.copy(), corners.copy()
                plus[:, corner, component] += step
                minus[:, corner, component] -= step
                difference = function(*np.moveaxis(plus, 1, 0)) - function(*np.moveaxis(minus, 1, 0))
                derivative[:, corner, component] = difference / (2 * step)
            else:
                message = f"unknown scheme {scheme!r}"
                raise ValueError(message)
    return derivative


def _scatter(n_points: int, indices: NDArray[np.int64], local: NDArray[np.float64]) -> NDArray[np.float64]:
    gradient = np.zeros((n_points, 3))
    for corner in range(indices.shape[1]):
        np.add.at(gradient, indices[:, corner], local[:, corner])
    return gradient


def finite_difference_step(mesh: object) -> float:
    """`RELATIVE_FINITE_DIFFERENCE_STEP` times the mean edge length of the mesh's triangles."""
    corners = mesh.v[mesh.f[:, :3]]
    lengths = np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=2)
    return float(RELATIVE_FINITE_DIFFERENCE_STEP * lengths.mean())


def reference_area_derivatives(mesh: object, scheme: str) -> dict[tuple[int, int], NDArray[np.float64]]:
    """Derivative of each interface's area w.r.t. every point, keyed `(label_a, label_b)` like the closed form."""
    triangles, labels = mesh.f[:, :3], mesh.f[:, 3:]
    local = _local_derivatives(_triangle_area, mesh.v[triangles], scheme, finite_difference_step(mesh))
    result = {}
    for key in sorted({tuple(int(v) for v in pair) for pair in labels}):
        rows = np.flatnonzero((labels[:, 0] == key[0]) & (labels[:, 1] == key[1]))
        result[key] = _scatter(len(mesh.v), triangles[rows], local[rows])
    return result


def reference_volume_derivatives(mesh: object, scheme: str) -> dict[int, NDArray[np.float64]]:
    """Derivative of each cell's volume w.r.t. every point: `+` for the triangle's first label, `-` for its second."""
    triangles, labels = mesh.f[:, :3], mesh.f[:, 3:]
    local = _local_derivatives(_triangle_volume_term, mesh.v[triangles], scheme, finite_difference_step(mesh))
    result = {}
    for cell in mesh.materials:
        first, second = labels[:, 0] == cell, labels[:, 1] == cell
        result[int(cell)] = _scatter(len(mesh.v), triangles[first], local[first]) - _scatter(
            len(mesh.v), triangles[second], local[second],
        )
    return result


def reference_length_derivatives(mesh: object, scheme: str) -> dict[tuple[int, int, int], NDArray[np.float64]]:
    """Derivative of each trijunction's length w.r.t. every point, keyed like the closed form."""
    edges = extract_edges_trijunctions(mesh)
    step = finite_difference_step(mesh)
    result = {}
    for key in sorted(edges):
        local = _local_derivatives(_edge_length, mesh.v[edges[key]], scheme, step)
        result[key] = _scatter(len(mesh.v), edges[key], local)
    return result


def relative_normwise_disagreement(closed_form: dict, reference: dict) -> float:
    """`||reference - closed_form|| / ||closed_form||` over the concatenation of every key of the closed form."""
    keys = sorted(closed_form)
    closed = np.concatenate([np.asarray(closed_form[key]).ravel() for key in keys])
    other = np.concatenate([np.asarray(reference[key]).ravel() for key in keys])
    return float(np.linalg.norm(other - closed) / np.linalg.norm(closed))
