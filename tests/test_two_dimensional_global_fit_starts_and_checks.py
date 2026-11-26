"""The global 2D fit's anchored start, its anchors and its second validity check (`foambryo._global_fit_2d`).

Exact tilings as in `test_two_dimensional_global_fit.py`: points sampled exactly on the circles of a random Möbius
tiling, normals the exact gradients. With readings that carry the exact tensions, every row of the anchored start's
least-squares problem vanishes at the true tiling (on a circle `grad g . n = |grad g| = 2 T`), so the start is the
tiling up to the solve's error: a ridge of 1e-12 of the largest diagonal entry and round-off, bounded by
`1e-12 * cond(N)` relative with `N` the normal matrix; the tests compute `cond(N)` and use that bound.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from foambryo import _global_fit_2d as gf

from .test_two_dimensional_global_fit import _exact_data, _jacobian_condition, _random_tiling

WINDOW = (0.0, 400.0, 0.0, 400.0)


def _reading(a: int, b: int, tension: float) -> gf.LocalReading:
    """A reading of interface `a|b` that carries only a tension (all the anchored start uses)."""
    return gf.LocalReading(
        left=a,
        right=b,
        curvature=0.0,
        centre=None,
        point=(0.0, 0.0),
        normal=(1.0, 0.0),
        tension=tension,
    )


def _exact_readings(v: np.ndarray, data: gf.FitData) -> dict:
    """One reading per interface of the data, with its exact tension (the start uses only the tension)."""
    out = {}
    for a, b in {(int(p), int(q)) for p, q in data.pairs.tolist()}:
        t = math.sqrt(gf.tension_squared(v[a] - v[b]))
        out[(a, b)] = [_reading(a, b, t)]
    return out


def _tensions_of(theta: np.ndarray, problem) -> dict:
    """The tension of every interface of the problem from free parameters `theta` (model frame -> pixels)."""
    v = gf._to_pixels(problem, gf._full(theta, len(problem.labels)))
    rows = {(int(a), int(b)) for a, b in zip(problem.row_a.tolist(), problem.row_b.tolist(), strict=True)}
    return {
        tuple(sorted((problem.labels[a], problem.labels[b]))): math.sqrt(
            max(float(gf.tension_squared(v[a] - v[b])), 0.0),
        )
        for a, b in rows
    }


@pytest.mark.parametrize(("n_cells", "seed"), [(4, 1), (16, 2)])
def test_the_anchored_start_is_the_tiling_when_the_readings_are_exact(n_cells, seed):
    v = _random_tiling(n_cells, seed)
    data = _exact_data(v, WINDOW)
    problem = gf._problem(data)
    readings = _exact_readings(v, data)
    theta = gf.anchored_start(problem, data.normals, readings)
    # the bound: the normal matrix of the start's rows (data rows and anchors), dense at these sizes
    t_point = gf._reading_tensions(problem, readings)
    rows = gf._block_matrix(problem, (problem.sqrt_w / (2.0 * t_point * problem.scale))[:, None] * problem.phi)
    m, _ = gf.anchor_rows(problem, data.normals, t_point, gf.ANCHORED_START_KAPPA_PX)
    a = np.vstack([rows.toarray(), m.toarray()])
    bound = 1e-12 * np.linalg.cond(a.T @ a)
    assert bound < 1e-4  # the tolerance must stay meaningful
    for (p, q), t in _tensions_of(theta, problem).items():
        true = math.sqrt(gf.tension_squared(v[p] - v[q]))
        assert abs(t - true) <= bound * true


def test_anchors_kept_in_the_fit_leave_an_exact_tiling_where_it_is():
    """At the true tiling the anchor rows are zero, so keeping them does not move the minimum.

    Bound: the fit's own (`test_exact_tilings_are_recovered_to_the_solvers_precision`), `100 * FIT_XTOL * cond(J)`.
    """
    v = _random_tiling(16, 2)
    data = _exact_data(v, WINDOW)
    res = gf.fit_tiling(data, starts=("anchored",), readings=_exact_readings(v, data), anchor_kappa_px=0.1)
    assert res.report["converged"]
    assert res.report["validity"]["valid"]
    finite = {p: x for p, x in res.tensions.items() if math.isfinite(x)}
    true = {p: math.sqrt(gf.tension_squared(v[p[0]] - v[p[1]])) for p in finite}
    mean = np.mean(list(true.values()))
    bound = 100 * gf.FIT_XTOL * _jacobian_condition(data, res)
    assert bound < 1e-6
    for p, x in finite.items():
        assert abs(x - true[p] / mean) <= bound * true[p] / mean


def test_a_cell_enclosed_by_the_medium_alone_is_refused_without_hanging():
    """A cell whose only interface is a closed loop with the medium, read from a label image's crack points.

    Crack-point normals are axis-aligned, so on a closed loop they sum to exactly zero. Before the fix the anchor row
    at the mean point along the mean normal was then zero, the start of that one-interface block was 0, its scale row
    `nan`, and the solver retried a singular factorisation for ever. The block is refused; the other cells are fitted.
    """
    lab = np.zeros((80, 80), np.int64)
    yy, xx = np.mgrid[:80, :80]
    lab[(yy - 20) ** 2 + (xx - 20) ** 2 <= 8**2] = 1
    lab[40:75, 40:57] = 2
    lab[40:57, 57:75] = 3
    lab[57:75, 57:75] = 4
    data = gf.data_from_label_image(lab)
    pairs = {(int(a), int(b)) for a, b in data.pairs.tolist()}
    assert np.allclose(data.normals[data.pairs[:, 1] == 1].sum(axis=0), 0.0, atol=0.0)  # exactly zero
    readings = {p: [_reading(p[0], p[1], 1.0)] for p in pairs}
    res = gf.fit_tiling(data, starts=("anchored",), readings=readings)
    assert math.isnan(res.tensions[(0, 1)])
    assert "a block by itself" in res.refused[(0, 1)]
    assert all(math.isfinite(res.tensions[p]) for p in pairs if 1 not in p)


def test_the_solver_stops_on_a_non_finite_start():
    v = _random_tiling(4, 1)
    data = _exact_data(v, WINDOW)
    problem = gf._problem(data)
    problem.basis, problem.basis_inverse = gf._local_basis(problem)
    theta = np.full(problem.n_free, np.nan)
    u = (gf._gauge_rows(problem, np.ones(problem.n_free)) @ problem.basis).tocsr()
    _, status = gf.levenberg_marquardt(problem, theta, (u, 1.0, 1.0))
    assert not status.converged
    assert status.reason == "non-finite"


def test_the_region_check_flags_a_region_beyond_its_data_and_nothing_within_the_band():
    labels = np.zeros((40, 40), np.int64)
    labels[5:20, 5:20] = 1
    labels[5:20, 20:35] = 2
    labels[20:35, 5:35] = 3
    assert gf._regions_beyond_their_data(labels, labels) == {}
    shifted = labels.copy()
    shifted[5:20, 20:22] = 1  # cell 1 two pixels into cell 2: within the band
    assert gf._regions_beyond_their_data(shifted, labels) == {}
    taken = labels.copy()
    taken[20:35, 5:20] = 1  # cell 1 owns half of cell 3: far beyond its data
    assert set(gf._regions_beyond_their_data(taken, labels)) == {1}
