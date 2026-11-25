"""2-D meshes sampled exactly on the circular arcs of an equilibrium foam, with its exact tensions and pressures.

The foam is a Möbius tiling (Noll N. et al., Phys. Rev. X 10, 011072, 2020): label `k` carries a lifted vector
`v_k = (A, B_row, B_column, C)` and owns the points where `f_k(x) = A |x|^2 + B . x + C` is the smallest; the medium is
label 0, `v_0 = 0`. Every interface `a|b` lies on the circle `f_a = f_b`, of radius `R = T_ab / |A_a - A_b|`, where
`T_ab = sqrt(tension_squared(v_a - v_b))` is its tension; the pressures are `p_k = A_k` on the same scale (Laplace's
law and Young's law hold exactly; `foambryo._global_fit_2d`'s docstring). `_random_tiling` of
`test_two_dimensional_global_fit` draws the foams.

`exact_arc_mesh` samples each interface at equal angular steps between its two junctions (the three-way ties of
`foambryo._global_fit_2d.fitted_junctions`, shared by the three interfaces that meet there), so every vertex lies on
its circle up to rounding, and orients each segment by the one orientation rule (its normal `(dc, -dr)` points out of
its first label), found from the sign of `f_a - f_b` beside the segment. Nothing is rasterised.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from foambryo import _global_fit_2d as gf
from foambryo.planar_mesh import PlanarMesh

#: The angular step of the samples is this arc length divided by the circle's radius (in the tiling's pixels).
DEFAULT_STEP = 0.05


def exact_tensions_and_pressures(v: np.ndarray) -> tuple[dict[tuple[int, int], float], dict[int, float]]:
    """Every pair's exact tension `sqrt(tension_squared(v_a - v_b))` and every label's pressure `A_k` (medium 0)."""
    n = len(v)
    tensions = {
        (a, b): math.sqrt(max(float(gf.tension_squared(v[a] - v[b])), 0.0)) for a in range(n) for b in range(a + 1, n)
    }
    return tensions, {k: float(v[k, 0]) for k in range(n)}


def _circle(v: np.ndarray, a: int, b: int) -> tuple[np.ndarray, float]:
    d = v[a] - v[b]
    centre = -d[1:3] / (2.0 * d[0])
    return centre, math.sqrt(float(gf.tension_squared(d))) / abs(float(d[0]))


def _owns(v: np.ndarray, a: int, b: int, point: np.ndarray) -> bool:
    """Whether `point` (on the circle `f_a = f_b`) belongs to the interface `a|b`: no other label is smaller there."""
    f = gf.evaluate(v, point[None])[0]
    level = 0.5 * (f[a] + f[b])
    others = np.delete(f, [a, b])
    return bool(np.all(others > level + 1e-9 * (1.0 + abs(level))))


def _arcs_of_pair(v: np.ndarray, a: int, b: int, ends: list[tuple[float, int]]) -> list[tuple]:
    """The arcs `(angle_0, angle_1, junction_0, junction_1)` of the circle `a|b` that belong to the interface.

    Between consecutive junctions on the circle (`ends`, sorted by angle), an arc belongs to it when its middle does;
    a circle without junction is the whole interface when one of its points belongs to it (junctions `None`).
    """
    centre, radius = _circle(v, a, b)
    if not ends:
        return [(0.0, 2 * math.pi, None, None)] if _owns(v, a, b, centre + np.array([radius, 0.0])) else []
    arcs = []
    for (t0, k0), (t1, k1) in zip(ends, ends[1:] + ends[:1], strict=True):
        t1 = t1 if t1 > t0 else t1 + 2 * math.pi
        middle = 0.5 * (t0 + t1)
        if _owns(v, a, b, centre + radius * np.array([math.cos(middle), math.sin(middle)])):
            arcs.append((t0, t1, k0, k1))
    return arcs


def exact_arc_mesh(v: np.ndarray, window: tuple[float, float, float, float], step: float = DEFAULT_STEP) -> PlanarMesh:
    """A `PlanarMesh` whose vertices lie exactly on the foam's arcs, junctions shared, in the one orientation rule."""
    junctions = gf.fitted_junctions(v, window, 0.0)
    points: list[np.ndarray] = [np.array(j["position"], dtype=np.float64) for j in junctions]
    segments: list[tuple[int, int]] = []
    labels: list[tuple[int, int]] = []
    pairs = {tuple(sorted(p)) for j in junctions for p in itertools.combinations(j["rows"], 2)}
    pairs |= gf._adjacent_pairs(gf.raster_rows(v, window))
    for a, b in sorted(pairs):
        if abs(v[a, 0] - v[b, 0]) < 1e-14:
            continue  # a straight interface: none in these foams
        centre, radius = _circle(v, a, b)
        ends = sorted((math.atan2(points[k][1] - centre[1], points[k][0] - centre[0]), k)
                      for k, j in enumerate(junctions) if a in j["rows"] and b in j["rows"])
        for t0, t1, k0, k1 in _arcs_of_pair(v, a, b, ends):
            n = max(4, math.ceil(radius * (t1 - t0) / step))
            index = []
            for i, t in enumerate(t0 + (t1 - t0) * np.arange(n + 1) / n):
                if i in (0, n) and (k0, k1)[i // n] is not None:
                    index.append((k0, k1)[i // n])
                elif i == n:
                    index.append(index[0])  # a closed circle
                else:
                    points.append(centre + radius * np.array([math.cos(t), math.sin(t)]))
                    index.append(len(points) - 1)
            for p, q in itertools.pairwise(index):
                segments.append(_oriented(v, a, b, points[p], points[q], p, q, step))
                labels.append((a, b))
    return PlanarMesh(np.array(points), np.array(segments), np.array(labels), image_border_vertices=[])


def _oriented(v: np.ndarray, a: int, b: int, start: np.ndarray, end: np.ndarray, p: int, q: int,
              step: float) -> tuple[int, int]:
    """`(p, q)` or `(q, p)`: the segment whose normal `(dc, -dr)` points out of `a`, the first label."""
    d = end - start
    normal = np.array([d[1], -d[0]]) / math.hypot(*d)
    beside = 0.5 * (start + end) - 1e-3 * step * normal  # inside the label the normal of p -> q leaves
    f = gf.evaluate(v, beside[None])[0]
    return (p, q) if f[a] < f[b] else (q, p)


def regular_polygon_cell(radius: float, n: int, centre: tuple[float, float] = (0.0, 0.0)) -> PlanarMesh:
    """One cell (label 1) in the medium (0), bounded by `n` vertices at equal angles on a circle of `radius`."""
    angles = 2 * math.pi * np.arange(n) / n
    points = np.column_stack([centre[0] + radius * np.cos(angles), centre[1] + radius * np.sin(angles)])
    # counter-clockwise in (row, column): the normal (dc, -dr) points out of the cell, label 1 is not first though:
    # labels are (0, 1), so the segments run clockwise, their normal pointing out of the medium (into the cell).
    segments = np.column_stack([(np.arange(n) + 1) % n, np.arange(n)])
    return PlanarMesh(points, segments, np.tile([0, 1], (n, 1)), image_border_vertices=[])
