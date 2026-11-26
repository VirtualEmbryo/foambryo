"""The geometric circle fit of the 2D arc reading (`foambryo._two_dimensional._fit_circle`), written in NumPy.

Pinned: exact circles and lines are recovered; Pratt's algebraic fit by Newton's iteration on its quartic is the
eigenvector a LAPACK eigen-solver gives; on noisy arcs the returned circle is a stationary point of the sum of squared
distances to rounding (the refinement does not stop where only the sums of squares agree); the fit calls no SciPy
optimiser; and on the 18 golden meshes the arc reading's tensions and pressures are those the fit wrote
(`two_dimensional_arc_reading_numpy_golden.json`) to 1e-12 on every supported NumPy and SciPy, since nothing in the
fit depends on their versions.
"""

import inspect
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from foambryo import _two_dimensional as fb2
from foambryo.junction_graph import JunctionGraphGaugeWarning

DATA = Path(__file__).resolve().parent / "data"


def _arc(centre, radius, start, stop, n):
    t = np.linspace(start, stop, n)
    return np.column_stack([centre[0] + radius * np.cos(t), centre[1] + radius * np.sin(t)])


@pytest.mark.parametrize(
    ("radius", "span", "n"),
    [(5.0, 1.0, 6), (50.0, 0.3, 10), (1e4, 0.01, 8), (3.0, 6.2, 30), (2.0, 1.5, 3), (20.0, np.pi, 4)],
)
def test_an_exact_arc_gives_its_circle(radius, span, n):
    circle = fb2._fit_circle(_arc((2.0, -1.0), radius, 0.3, 0.3 + span, n))
    assert circle is not None
    assert circle.centre is not None
    assert circle.radius == pytest.approx(radius, rel=1e-9)
    assert np.allclose(circle.centre, (2.0, -1.0), rtol=0, atol=1e-9 * radius)


def test_exact_collinear_points_give_a_line():
    x = np.linspace(0.0, 10.0, 7)
    circle = fb2._fit_circle(np.column_stack([x, 2.0 * x + 1.0]))
    assert circle is not None
    assert circle.centre is None
    assert circle.radius == np.inf
    assert abs(float(np.dot(circle.normal, (1.0, 2.0)))) < 1e-12
    assert abs(float(np.dot(circle.point - (0.0, 1.0), circle.normal))) < 1e-12


def test_fewer_than_three_distinct_points_give_no_circle():
    assert fb2._fit_circle(np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])) is None


def _pratt_by_eigenvectors(points):
    """Pratt's fit as a generalised eigenproblem solved by LAPACK (`np.linalg.eig`), for comparison."""
    z = np.sum(points * points, axis=1)
    design = np.column_stack([z, points[:, 0], points[:, 1], np.ones(len(points))])
    moments = design.T @ design / len(points)
    constraint = np.array([[0, 0, 0, -2.0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [-2.0, 0, 0, 0]])
    eta, vectors = np.linalg.eig(np.linalg.inv(constraint) @ moments)
    eta, vectors = np.real(eta), np.real(vectors)
    best, best_eta = None, np.inf
    for k in range(4):
        v = vectors[:, k]
        norm = float(v @ constraint @ v)
        if norm > 0 and eta[k] >= -1e-10 and eta[k] < best_eta:
            best, best_eta = v / np.sqrt(norm), eta[k]
    return best


def test_pratt_by_newton_on_its_quartic_is_the_eigenvector():
    rng = np.random.default_rng(7)
    for _ in range(100):
        n = int(rng.integers(4, 40))
        points = _arc((0.0, 0.0), 10 ** rng.uniform(-1, 3), 0.0, rng.uniform(0.1, 6.2), n)
        points += rng.normal(0.0, rng.uniform(0.0, 0.3), points.shape)
        points -= points.mean(axis=0)
        points /= np.sqrt(np.mean(np.sum(points * points, axis=1)))
        newton = np.array(fb2._algebraic_circle(points))
        eig = _pratt_by_eigenvectors(points)
        sign = 1.0 if float(np.dot(newton, eig)) >= 0 else -1.0
        assert np.allclose(newton, sign * eig, rtol=0, atol=1e-9)


def _relative_gradient(points, circle):
    """|J^T d| / (|J|_F |d|) of the distances d_i = |x_i - c| - R in the coordinates (c, R)."""
    v = points - circle.centre
    rho = np.hypot(v[:, 0], v[:, 1])
    d = rho - circle.radius
    jac = np.column_stack([-v[:, 0] / rho, -v[:, 1] / rho, -np.ones(len(points))])
    return float(np.linalg.norm(jac.T @ d) / (np.linalg.norm(jac) * np.linalg.norm(d)))


def test_the_fitted_circle_is_a_stationary_point_on_noisy_arcs():
    """Where the sums of squares agree to rounding, the gradient is still resolved: the polish reaches it."""
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(200):
        n = int(rng.integers(4, 30))
        radius = 10 ** rng.uniform(0.5, 2.5)
        points = _arc((3.0, -2.0), radius, 0.0, rng.uniform(0.2, 3.0), n) + rng.normal(0.0, 0.3, (n, 2))
        circle = fb2._fit_circle(points)
        assert circle is not None
        if circle.centre is not None:
            worst = max(worst, _relative_gradient(np.unique(points, axis=0), circle))
    assert worst < 1e-10


def test_the_circle_fit_calls_no_scipy_optimiser():
    for function in (fb2._fit_circle, fb2._refine_circle, fb2._levenberg_marquardt, fb2._polish, fb2._algebraic_circle):
        source = inspect.getsource(function)
        assert "scipy" not in source
        assert "least_squares" not in source
    assert not hasattr(fb2, "least_squares")


GOLDEN = json.loads((DATA / "two_dimensional_arc_reading_numpy_golden.json").read_text())


@pytest.fixture(scope="module")
def golden_meshes():
    with np.load(DATA / "two_dimensional_reference_meshes.npz") as arrays:
        return {n: (arrays[f"{n}__vertices"], arrays[f"{n}__edges"]) for n in sorted(GOLDEN["cases"])}


@pytest.mark.parametrize("name", sorted(GOLDEN["cases"]))
def test_the_arc_reading_is_its_golden_master_on_every_supported_version(golden_meshes, name):
    """Tensions (to 1e-12 of the gauge) and pressures (to 1e-12 of the largest), on every NumPy and SciPy."""
    expected = GOLDEN["cases"][name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", JunctionGraphGaugeWarning)
        tensions = fb2.infer_tensions(golden_meshes[name])
        pressures = fb2.infer_pressures(golden_meshes[name], tensions)
    for a, b, value in expected["tensions"]:
        assert np.isnan(tensions[(a, b)]) == (value is None), (a, b)
        if value is not None:
            assert abs(tensions[(a, b)] - value) <= 1e-12, (a, b)
    scale = max([abs(v) for _, v in expected["pressures"] if v is not None] + [1e-300])
    for material, value in expected["pressures"]:
        assert np.isnan(pressures[material]) == (value is None), material
        if value is not None:
            assert abs(pressures[material] - value) <= 1e-12 * scale, material
