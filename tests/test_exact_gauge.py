"""The gauge of every tension method is exact, the Laplace pin too, and dense solves drop null directions.

What this pins
--------------
Until 0.6.0 the mean-tension gauge of the angle-reading, cotangent, per-edge, `Equilibrium` and
`Variational`-Gram methods was a least-squares row of ones of weight 1. Their physics rows are
homogeneous, so the row's weight only rescaled the solution: tension ratios were right, but the mean
missed `mean_tension` (by ~1e-6 for `YoungDupre`, up to 80 % for `Equilibrium`) and the row raised
the condition number by up to three orders of magnitude. Since 0.7.0 (`solve_with_exact_gauge`):

1. every component's tensions meet the gauge exactly (arithmetic mean; geometric for `LogLami`);
2. they are the exact constrained least-squares solution: equal, on the cut benchmark mesh, to an
   independent KKT solve of the same physics rows;
3. the gauge row's weight is the median physics-row norm, rounded to a power of two;
4. the Laplace pin is exact (the exterior's pressure is `base_pressure` exactly) and the dense and
   sparse paths agree on a mesh with an undetermined cell;

The rank-deficient dense pressure solves are fixed by construction, not by a rank cutoff: the Laplace
system is solved on the exterior's full-rank block (its record's rank is its number of unknowns, see
`test_solver_options_and_diagnostics.py`), and the cropped variational one from its own truncated SVD.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from foambryo import (
    PressureComputationMethod,
    TensionComputationMethod,
    infer_pressures,
    infer_tensions,
    infer_tensions_by_junction_component,
    junction_graph,
)
from foambryo.dcel import DcelData
from foambryo.tension_inference import (
    VariationalRoute,
    _build_matrix_tension_symmetrical_yd,
    _infer_tension_variational_yd,
)
from foambryo.wedge_refusal import line_averaged_angles

CUT_CASE = Path(__file__).parent / "data" / "cut_benchmark_case_000_mesh.npz"
GAUGED = (
    TensionComputationMethod.YoungDupre,
    TensionComputationMethod.YoungDupreLocal,
    TensionComputationMethod.ProjectionYoungDupre,
    TensionComputationMethod.Cotan,
    TensionComputationMethod.InvCotan,
    TensionComputationMethod.Lami,
    TensionComputationMethod.InvLami,
)


@pytest.fixture(scope="module")
def cut_case() -> DcelData:
    data = np.load(CUT_CASE)
    return DcelData(data["points"], data["triangles"], data["labels"])


def _quiet(function: object, *args: object, **kwargs: object) -> object:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


@pytest.mark.parametrize("mean_tension", [1.0, 0.37])
@pytest.mark.parametrize("mode", GAUGED)
def test_every_component_meets_its_mean_tension_exactly(cut_case, mode, mean_tension):
    solution = _quiet(infer_tensions_by_junction_component, cut_case, mean_tension, mode)
    for component in {c for c in solution.component_of.values() if c is not None}:
        values = [v for k, v in solution.dict_tensions.items() if solution.component_of[k] == component]
        assert np.mean(values) == pytest.approx(mean_tension, rel=1e-12)


def test_log_lami_meets_its_geometric_mean_exactly(cut_case):
    tensions = _quiet(infer_tensions, cut_case, 0.37, TensionComputationMethod.LogLami)
    values = np.array([v for v in tensions.values() if np.isfinite(v)])
    assert np.exp(np.mean(np.log(values))) == pytest.approx(0.37, rel=1e-12)


@pytest.mark.parametrize("mean_tension", [1.0, 2.5])
def test_equilibrium_meets_its_mean_tension_exactly(cut_case, mean_tension):
    """`Equilibrium`'s unweighted rows used to pull a soft row's mean off by 8 % typically."""
    tensions = _quiet(infer_tensions, cut_case, mean_tension, TensionComputationMethod.Equilibrium)
    assert np.mean(list(tensions.values())) == pytest.approx(mean_tension, rel=1e-12)


def test_the_variational_gram_route_meets_its_mean_tension_exactly(cut_case):
    _x, tensions, _r = _quiet(_infer_tension_variational_yd, cut_case, 1.0, route=VariationalRoute.Gram)
    assert np.mean(list(tensions.values())) == pytest.approx(1.0, rel=1e-12)


def test_young_dupre_is_the_constrained_least_squares_solution(cut_case):
    """Against an independent KKT solve of the builder's own physics rows, and on both paths."""
    angles = _quiet(line_averaged_angles, cut_case)
    areas = cut_case.compute_areas_interfaces()
    matrix, vector = _build_matrix_tension_symmetrical_yd(angles.dict_rad, areas, angles.dict_length)
    physics = matrix[:-1]
    n = physics.shape[1]
    kkt = np.zeros((n + 1, n + 1))
    kkt[:n, :n] = physics.T @ physics
    kkt[:n, n] = kkt[n, :n] = 1.0
    exact = np.linalg.solve(kkt, np.concatenate([np.zeros(n), [vector[-1]]]))[:n]
    keys = sorted(areas)
    for solver in ("dense", "sparse"):
        tensions = _quiet(infer_tensions, cut_case, solver=solver)
        got = np.array([tensions[k] for k in keys])
        np.testing.assert_allclose(got, exact, rtol=0, atol=1e-10 * np.abs(exact).max())


def test_the_gauge_row_weight_is_the_median_row_norm_to_a_power_of_two():
    physics = np.diag([3.0, 4.0, 100.0, 0.0])
    weight = junction_graph.gauge_row_weight(physics, n_unknowns=4)
    assert np.log2(weight) == int(np.log2(weight))
    assert weight * np.sqrt(4) == pytest.approx(4.0, rel=np.sqrt(2) - 1)  # median of 3, 4, 100
    assert junction_graph.gauge_row_weight(np.zeros((3, 4)), n_unknowns=4) == 1.0


@pytest.mark.parametrize("mode", [PressureComputationMethod.Laplace, PressureComputationMethod.WeightedLaplace])
@pytest.mark.parametrize("base_pressure", [0.0, 0.25])
def test_the_laplace_pin_is_exact_and_both_paths_agree(cut_case, mode, base_pressure):
    tensions = _quiet(infer_tensions, cut_case)
    dense = _quiet(infer_pressures, cut_case, tensions, mode=mode, base_pressure=base_pressure, solver="dense")
    sparse = _quiet(infer_pressures, cut_case, tensions, mode=mode, base_pressure=base_pressure, solver="sparse")
    assert dense[0] == sparse[0] == base_pressure
    assert np.isnan(dense[2])
    assert np.isnan(sparse[2])
    finite = [k for k in dense if np.isfinite(dense[k])]
    scale = max(abs(dense[k]) for k in finite)
    assert max(abs(dense[k] - sparse[k]) for k in finite) <= 1e-12 * scale
