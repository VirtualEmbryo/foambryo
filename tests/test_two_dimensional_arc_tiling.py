"""`foambryo.infer_forces_by_equilibrium_tiling`: the global fit of an equilibrium tiling of circular arcs (opt-in).

The label images are rasterised Möbius tilings (`_random_tiling`: every interface a circle, Young's and Laplace's laws
exact), so the truth is known. The mesh the entry point needs for its start is built here from the label image's cracks
(every side shared by two pixels of different labels is an edge, oriented with dw2d's contract: `m1 < m2` and `m1` on
the left of `v1 -> v2` in `(x, y) = (column, row)`), so these tests need no dw2d.

Tolerance of the accuracy test: the entry point and a fit started from the true tiling minimise the same cost on the
same points, so they must stop at the same minimum. The fit stops when an accepted step lowers the cost by less than
`FIT_FTOL` (1e-12) of it, which leaves the parameters within about `sqrt(2 FIT_FTOL cost / lambda_min)` of the minimum
(`lambda_min` the smallest eigenvalue of `J^T J` there); the test computes that distance, relative to the parameters,
and allows 100 times it for the map from parameters to tensions.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import linalg as sparse_linalg

from foambryo import _global_fit_2d as gf
from foambryo import _two_dimensional as fb2
from foambryo.junction_graph import ComponentTensionSolution

from .test_two_dimensional_global_fit import _random_tiling


def _crack_mesh(labels: np.ndarray) -> SimpleNamespace:
    """A polyline mesh of a label image's cracks, with dw2d's orientation contract and its image-border mark."""
    h, w = labels.shape
    index: dict[tuple[float, float], int] = {}

    def vertex(r: float, c: float) -> int:
        return index.setdefault((r, c), len(index))

    edges = []
    for i, j in zip(*np.nonzero(labels[:, :-1] != labels[:, 1:]), strict=True):
        a, b = int(labels[i, j]), int(labels[i, j + 1])  # a in column j: on the left of the direction +row
        p, q = vertex(i - 0.5, j + 0.5), vertex(i + 0.5, j + 0.5)
        edges.append((p, q, a, b) if a < b else (q, p, b, a))
    for i, j in zip(*np.nonzero(labels[:-1, :] != labels[1:, :]), strict=True):
        a, b = int(labels[i, j]), int(labels[i + 1, j])  # b in row i + 1: on the left of the direction +column
        p, q = vertex(i + 0.5, j - 0.5), vertex(i + 0.5, j + 0.5)
        edges.append((p, q, b, a) if b < a else (q, p, a, b))
    v = np.array(sorted(index, key=index.get), float)
    border = [k for k, (r, c) in enumerate(v) if r in (-0.5, h - 0.5) or c in (-0.5, w - 0.5)]
    return SimpleNamespace(v=v, e=np.array(edges, np.int64), image_border_vertices=border)


def _labels_of(v: np.ndarray, window: tuple[float, float, float, float]) -> np.ndarray:
    return np.asarray(gf.raster_rows(v, window), np.int64)


def _fit_from_the_truth(labels: np.ndarray, v: np.ndarray) -> tuple[dict, float]:
    """Tensions (mean 1) of the fit of the main block started from the true tiling, and the stopping-rule bound.

    The main block as `fit_tiling` makes it: interfaces with at least `MIN_POINTS_PER_INTERFACE` points, the largest
    biconnected block that holds the medium.
    """
    data = gf.data_from_label_image(labels)
    pairs, counts = np.unique(data.pairs, axis=0, return_counts=True)
    kept = [(int(a), int(b)) for (a, b), n in zip(pairs, counts, strict=True) if n >= gf.MIN_POINTS_PER_INTERFACE]
    block = set(max((b for b in gf.biconnected_blocks(kept) if any(0 in q for q in b)), key=len))
    data = gf._subset(data, np.array([(int(a), int(b)) in block for a, b in data.pairs.tolist()]))
    problem = gf._problem(data)
    problem.basis, problem.basis_inverse = gf._local_basis(problem)
    s, c = problem.scale, problem.centre
    rows = np.array([v[m] for m in problem.labels])
    model = np.column_stack(
        [
            rows[:, 0] * s * s,
            s * (rows[:, 1:3] + 2 * rows[:, :1] * c[None]),
            rows[:, 3] + rows[:, 1:3] @ c + rows[:, 0] * (c @ c),
        ],
    )
    theta = (model[1:] - model[0]).ravel()
    w0 = problem.basis_inverse @ theta
    u = (gf._gauge_rows(problem, theta) @ problem.basis).tocsr()
    gauge = (u, u @ w0, math.sqrt(float(np.sum(problem.sqrt_w**2))))
    w, status = gf.levenberg_marquardt(problem, w0, gauge)
    assert status.converged
    _, jac = gf._residuals_and_jacobian(problem, w, gauge)
    lam = sparse_linalg.eigsh((jac.T @ jac).tocsc(), k=1, sigma=0.0, which="LM", return_eigenvectors=False)[0]
    bound = 100 * math.sqrt(2 * gf.FIT_FTOL * status.final_cost / lam) / np.linalg.norm(w)
    vv = gf._to_pixels(problem, gf._full(problem.basis @ w, len(problem.labels)))
    pairs = {(int(a), int(b)) for a, b in data.pairs.tolist()}
    row = {m: k for k, m in enumerate(problem.labels)}
    t = {p: math.sqrt(gf.tension_squared(vv[row[p[0]]] - vv[row[p[1]]])) for p in pairs}
    mean = np.mean(list(t.values()))
    return {p: x / mean for p, x in t.items()}, bound


def test_the_entry_point_reaches_the_minimum_the_truth_reaches_and_keeps_its_contract():
    v = _random_tiling(16, 2, radius=16.0)
    labels = _labels_of(v, (0.0, 400.0, 0.0, 400.0))
    mesh = _crack_mesh(labels)
    solution, pressures, report = fb2.infer_forces_by_equilibrium_tiling(mesh, labels, mean_tension=2.0, diagnostics=True)
    assert isinstance(solution, ComponentTensionSolution)
    assert isinstance(report, fb2.EquilibriumTilingReport)
    assert report.converged
    assert report.valid
    assert pressures[0] == 0.0
    for block in {b for b in solution.component_of.values() if b is not None}:
        finite = [x for p, x in solution.dict_tensions.items() if solution.component_of[p] == block]
        assert np.mean(finite) == pytest.approx(2.0, rel=1e-12)
    for p, x in solution.dict_tensions.items():
        assert math.isfinite(x) or p in report.refused
    for m, x in pressures.items():
        assert math.isfinite(x) or m in report.pressure_refused
    reference, bound = _fit_from_the_truth(labels, v)
    assert bound < 1e-3  # the tolerance must stay meaningful
    block = [p for p, b in solution.component_of.items() if b == 0]
    mean = np.mean([reference[p] for p in block])
    for p in block:
        assert solution.dict_tensions[p] / 2.0 == pytest.approx(reference[p] / mean, rel=bound)


def test_a_crop_of_a_tiling_gives_tensions_and_a_reason_for_every_nan():
    """A window that cuts the tissue (the medium on two sides only); refusals are local, never the whole image."""
    v = _random_tiling(32, 4, radius=16.0)
    labels = _labels_of(v, (0.0, 400.0, 0.0, 400.0))
    rows, cols = np.nonzero(labels)
    crop = labels[: int(rows.mean()) + 10, : int(cols.mean()) + 10]
    solution, pressures, report = fb2.infer_forces_by_equilibrium_tiling(_crack_mesh(crop), crop, diagnostics=True)
    finite = [x for x in solution.dict_tensions.values() if math.isfinite(x)]
    assert len(finite) >= 0.5 * len(solution.dict_tensions)
    for p, x in solution.dict_tensions.items():
        assert math.isfinite(x) or p in report.refused
    for m, x in pressures.items():
        assert math.isfinite(x) or m in report.pressure_refused


def test_the_junction_search_finds_every_junction_of_a_4096_cell_tiling():
    """The check's route above 16 materials (the lower convex hull of the lifted vectors) on the true tiling.

    4 096 discs of random pressures on a jittered triangular lattice (a Möbius tiling of near-hexagonal cells, whose
    junctions are well apart); every junction of its label image (a pixel corner where three labels meet, away from the
    window border) must be a junction the search finds. On a square lattice the junctions come in near-coincident pairs
    and the pixel corners then show triples that are not junctions of the tiling (measured: 337 of 7 281, all within
    1.5 px of a true junction, none found by the exhaustive search either), so the lattice is triangular.
    """
    rng = np.random.default_rng(7)
    n_side, spacing = 64, 12.0
    centres = np.array(
        [
            (j * spacing * math.sqrt(3) / 2, i * spacing + (j % 2) * spacing / 2)
            for i in range(n_side)
            for j in range(n_side)
        ],
    )
    centres += 20.0 + rng.uniform(-1.0, 1.0, centres.shape)
    a = rng.uniform(0.8, 1.2, len(centres))
    v = np.zeros((len(centres) + 1, 4))
    v[1:, 0] = a
    v[1:, 1:3] = -2 * a[:, None] * centres
    v[1:, 3] = a * (np.sum(centres**2, axis=1) - 8.0**2)
    window = (0.0, n_side * spacing + 40.0, 0.0, n_side * spacing + 40.0)
    labels = _labels_of(v, window)
    data = gf.data_from_label_image(labels)
    present = sorted(set(np.unique(labels).tolist()))
    assert len(present) > gf._ALL_TRIPLES_UP_TO
    rows = np.array([v[m] for m in present])
    found = gf.fitted_junctions(
        gf._to_model(rows, data),
        gf._window_to_model(data),
        gf.BORDER_BAND_PX / gf._scale_of(data),
    )
    triples = {tuple(sorted(present[k] for k in j["rows"])) for j in found}
    inside = {m for m, p in data.junctions if gf._inside(p, data.window, gf.BORDER_BAND_PX)}
    assert len(inside) > 1.5 * len(present)
    assert inside <= triples
