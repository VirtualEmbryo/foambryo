"""The global 2D fit (an equilibrium tiling of circular arcs, `foambryo._global_fit_2d`) on exact and cut tilings.

Exact tilings: points sampled exactly on the interfaces of a random Möbius tiling (every interface a circle, Young's and
Laplace's laws exact). The fit must return the tiling's tensions and pressures. Tolerance: the fit stops when its scaled
step is below `FIT_XTOL` (1e-12) of the scaled parameters; a parameter error `delta` of the converged fit is bounded by
about `xtol * cond(J)` relative, so the tests compute the Jacobian's condition number at the solution and require the
relative error of every tension and pressure below `100 * FIT_XTOL * cond(J)` (the factor 100 covers the map from
parameters to tensions), never a tolerance chosen in advance.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from foambryo import _global_fit_2d as gf
from foambryo import _two_dimensional as fb2

DATA = Path(__file__).resolve().parent / "data"


def _random_tiling(n_cells: int, seed: int, radius: float = 12.0) -> np.ndarray:
    """Lifted vectors (medium row 0 at 0) of `n_cells` overlapping discs of random pressures: a Möbius tiling."""
    rng = np.random.default_rng(seed)
    centres: list[np.ndarray] = []
    while len(centres) < n_cells:
        p = rng.uniform(-1, 1, 2) * math.sqrt(n_cells) * 10 + 200.0
        if all(np.linalg.norm(p - q) > 14 for q in centres):
            centres.append(p)
    c = np.array(centres)
    a = rng.uniform(0.8, 1.2, n_cells)
    v = np.zeros((n_cells + 1, 4))
    v[1:, 0] = a
    v[1:, 1:3] = -2 * a[:, None] * c
    v[1:, 3] = a * (np.sum(c**2, 1) - radius**2)
    return v


def _exact_data(v: np.ndarray, window: tuple[float, float, float, float], spacing: float = 1.0) -> gf.FitData:
    """Points every `spacing` px on each pair's circle where that pair has the smallest `f` (the interface itself)."""
    pts, prs, nrm = [], [], []
    for a, b in itertools.combinations(range(len(v)), 2):
        d = v[a] - v[b]
        t2 = gf.tension_squared(d)
        if t2 <= 0 or abs(d[0]) < 1e-14:
            continue
        c = -d[1:3] / (2 * d[0])
        r = math.sqrt(t2) / abs(d[0])
        k = max(64, int(2 * math.pi * r / spacing))
        th = (np.arange(k) + 0.5) * 2 * math.pi / k
        x = c[None] + r * np.column_stack([np.cos(th), np.sin(th)])
        f = gf.evaluate(v, x)
        own = f[:, a].copy()
        f[:, [a, b]] = np.inf
        r0, r1, c0, c1 = window
        keep = (f.min(1) > own + 1e-9) & (x[:, 0] >= r0) & (x[:, 0] <= r1) & (x[:, 1] >= c0) & (x[:, 1] <= c1)
        x = x[keep]
        g = 2 * d[0] * x + d[1:3]
        pts.append(x)
        prs.append(np.tile([a, b], (len(x), 1)))
        nrm.append(g / np.linalg.norm(g, axis=1)[:, None])
    junctions = gf.fitted_junctions(v, window, 0.0)
    return gf.FitData(
        points=np.vstack(pts),
        pairs=np.vstack(prs).astype(np.int64),
        weights=np.ones(sum(len(p) for p in pts)),
        normals=np.vstack(nrm),
        window=window,
        junctions=[(tuple(sorted(j["rows"])), j["position"]) for j in junctions],
        source="exact",
    )


def _jacobian_condition(data: gf.FitData, result: gf.GlobalFitResult) -> float:
    """Condition number of the fit's Jacobian (data rows, local frames) at the fitted tiling of the largest block."""
    problem = gf._problem(data)
    problem.basis, problem.basis_inverse = gf._local_basis(problem)
    s, c = problem.scale, problem.centre
    v = np.array([result.lifted[m] for m in problem.labels])
    model = np.column_stack(
        [v[:, 0] * s * s, s * (v[:, 1:3] + 2 * v[:, :1] * c[None]), v[:, 3] + v[:, 1:3] @ c + v[:, 0] * (c @ c)],
    )
    w = problem.basis_inverse @ model[1:].ravel()
    gauge = (gf._gauge_rows(problem, model[1:].ravel()) @ problem.basis, np.zeros(1), 1.0)
    _, jac = gf._residuals_and_jacobian(problem, w, (gauge[0].tocsr(), gauge[0] @ w, 1.0))
    sv = np.linalg.svd(jac.toarray(), compute_uv=False)
    return float(sv[0] / sv[-1])


@pytest.mark.parametrize(("n_cells", "seed"), [(4, 1), (16, 2), (64, 3)])
def test_exact_tilings_are_recovered_to_the_solvers_precision(n_cells, seed):
    v = _random_tiling(n_cells, seed)
    data = _exact_data(v, (0.0, 400.0, 0.0, 400.0))
    res = gf.fit_tiling(data)
    assert res.report["converged"]
    assert res.report["validity"]["valid"]
    big = res.report["largest_component"]
    pairs = [p for p, x in res.tensions.items() if math.isfinite(x) and res.report["component_of"][p] == big]
    true = {p: math.sqrt(gf.tension_squared(v[p[0]] - v[p[1]])) for p in pairs}
    mean = np.mean([true[p] for p in pairs])
    bound = 100 * gf.FIT_XTOL * _jacobian_condition(data, res)
    assert bound < 1e-6  # the tolerance must stay meaningful
    for p in pairs:
        assert abs(res.tensions[p] - true[p] / mean) <= bound * true[p] / mean
    cells = {m for p in pairs for m in p} - {0}
    for m in cells:
        assert abs(res.pressures[m] - v[m, 0] / mean) <= bound * abs(v[m, 0] / mean) + bound


def test_the_residual_is_the_signed_distance_to_the_circle():
    """`g / (|grad g| / 2 + T)` is `|x - c| - R` (sign of `d_A`) for a circle, the signed distance for a line."""
    rng = np.random.default_rng(0)
    c, r, a = np.array([3.0, -2.0]), 5.0, 0.7
    d = np.array([[a, -2 * a * c[0], -2 * a * c[1], a * (c @ c - r * r)]])
    x = rng.normal(size=(20, 2)) * 4
    rho, _ = gf.distances_and_derivatives(np.repeat(d, 20, 0), gf.lifted(x), x, with_derivatives=False)
    np.testing.assert_allclose(rho, np.linalg.norm(x - c, axis=1) - r, rtol=1e-12, atol=1e-12)
    n, x0 = np.array([0.6, 0.8]), np.array([1.0, 2.0])
    line = np.array([[0.0, 2 * n[0], 2 * n[1], -2 * n @ x0]])
    rho, _ = gf.distances_and_derivatives(np.repeat(line, 20, 0), gf.lifted(x), x, with_derivatives=False)
    np.testing.assert_allclose(rho, (x - x0) @ n, rtol=1e-12, atol=1e-12)


def test_the_analytic_jacobian_equals_the_complex_step():
    v = _random_tiling(8, 5)
    data = _exact_data(v, (0.0, 400.0, 0.0, 400.0), spacing=3.0)
    problem = gf._problem(data)
    problem.basis, problem.basis_inverse = gf._local_basis(problem)
    theta0 = gf.appendix_b_start(problem)
    w = (problem.basis_inverse @ theta0) * (1 + 1e-2 * np.random.default_rng(1).standard_normal(problem.n_free))
    u = (gf._gauge_rows(problem, theta0) @ problem.basis).tocsr()
    gauge = (u, u @ w, 2.0)
    _, jac = gf._residuals_and_jacobian(problem, w, gauge)
    jac = jac.toarray()
    for k in range(problem.n_free):
        e = np.zeros(problem.n_free, complex)
        e[k] = 1e-30j
        column = gf._residuals(problem, w + e, gauge).imag / 1e-30
        np.testing.assert_allclose(jac[:, k], column, rtol=1e-12, atol=1e-12 * np.abs(column).max())


def test_a_cell_touching_only_the_medium_is_its_own_block_and_refused():
    """Its tension shares a scale with no other interface: it is refused; the other block is recovered."""
    v = _random_tiling(4, 1)
    lonely = np.array([1.0, -2 * 330.0, -2 * 330.0, 2 * 330.0**2 - 12.0**2])
    v = np.vstack([v, lonely])
    data = _exact_data(v, (0.0, 400.0, 0.0, 400.0))
    res = gf.fit_tiling(data)
    assert (0, 5) in res.refused
    assert "block by itself" in res.refused[(0, 5)]
    assert math.isnan(res.tensions[(0, 5)])
    assert sum(math.isfinite(x) for x in res.tensions.values()) == len(res.tensions) - 1


def test_a_window_without_medium_is_gauged_on_a_cell():
    """Data without the medium (as in a window inside a tissue): gauged on its best-sampled cell, ratios kept."""
    v = _random_tiling(16, 2)
    full = _exact_data(v, (0.0, 400.0, 0.0, 400.0))
    data = gf._subset(full, (full.pairs != 0).all(axis=1))
    data = gf.FitData(**{**data.__dict__, "junctions": [j for j in data.junctions if 0 not in j[0]]})
    assert 0 not in set(np.unique(data.pairs).tolist())
    res = gf.fit_tiling(data)
    assert res.reference != 0
    big = res.report["largest_component"]
    pairs = [p for p, x in res.tensions.items() if math.isfinite(x) and res.report["component_of"][p] == big]
    assert len(pairs) >= 3
    true = {p: math.sqrt(gf.tension_squared(v[p[0]] - v[p[1]])) for p in pairs}
    mean = np.mean(list(true.values()))
    for p in pairs:
        assert res.tensions[p] == pytest.approx(true[p] / mean, rel=1e-6)


def test_the_lens_of_the_cropped_foam_gets_finite_tensions():
    """R196's crop whose two-sided cell 14 the arc reading refuses: the global fit reads its sides as two circles."""
    with np.load(DATA / "two_dimensional_mesh_of_a_cropped_foam_with_a_lens.npz") as z:
        mesh = SimpleNamespace(v=z["vertices"], e=z["edges"], image_border_vertices=z["image_border_vertices"].tolist())
    data = gf.data_from_mesh_vertices(mesh.v, mesh.e, fb2.image_border_vertices(mesh))
    res = gf.fit_tiling(data)
    for p in ((9, 14), (12, 14)):
        assert p in res.tensions
        if p not in res.refused:
            assert 0.1 < res.tensions[p] < 10.0


def test_crack_points_and_their_normals():
    labels = np.array([[1, 1, 2], [1, 1, 2], [0, 0, 2]])
    data = gf.data_from_label_image(labels)
    pairs = {tuple(p) for p in data.pairs.tolist()}
    assert pairs == {(1, 2), (0, 1), (0, 2)}
    for x, p, n in zip(data.points, data.pairs, data.normals, strict=True):
        lo_side = np.round(x - 0.5 * n).astype(int)
        assert labels[lo_side[0], lo_side[1]] == p[0]
    assert [m for m, _ in data.junctions] == [(0, 1, 2)]
