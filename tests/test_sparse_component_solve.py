"""The tension solve switches to sparse `lsqr` on large systems and stays exactly dense on small ones.

`solve_gauged_least_squares` runs `scipy.linalg.lstsq` on each junction-graph component. On a
1 866-cell tissue that is 230 s and 25 GiB for one component of 11 742 interfaces, because the dense
solve grows as the cube of the unknowns. At or above `SPARSE_SOLVE_MIN_UNKNOWNS` it now runs `lsqr`
on the same rows and columns, in CSR form, with no condition-number stop. These tests pin the
contract:

1. **below the threshold nothing changes**: the result is bit-identical to `lstsq`, fragmented or
   not, and the record says the dense solver ran;
2. **above it `lsqr` solves the same system**: same components, gauges and undetermined
   interfaces, tensions within 1e-10 of the dense solve on well-conditioned systems and on two
   analytic doublets;
3. **a failure to converge is not silent**: the dense solve runs instead, the result is exactly
   the dense one, a `SparseSolveFallbackWarning` is emitted and the record says so.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import linalg

from foambryo import junction_graph
from foambryo.junction_graph import (
    JunctionGraphPartition,
    SparseSolveFallbackWarning,
    solve_gauged_least_squares,
)
from foambryo.tension_inference import TensionComputationMethod, infer_tensions

from .test_junction_graph import _two_separated_doublets


def a_system(n: int, rows_per_unknown: int = 4, seed: int = 0):
    """A sparse, well-conditioned, overdetermined system with the global gauge row last.

    Its physics rows are homogeneous (zero right-hand side), as every tension builder's are; their
    least-squares solution under the gauge is non-trivial because the rows have full column rank.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(rows_per_unknown * n):
        row = np.zeros(n)
        columns = rng.choice(n, size=3, replace=False)
        row[columns] = rng.normal(size=3)
        rows.append(row)
    matrix_m = np.vstack([*rows, np.ones(n)])
    vector_b = np.zeros(len(matrix_m))
    vector_b[-1] = float(n)
    keys = [(i, i + 1) for i in range(n)]
    return matrix_m, vector_b, keys


def one_component(keys):
    return JunctionGraphPartition(interfaces=tuple(keys), components=(tuple(keys),),
                                  component_of=dict.fromkeys(keys, 0), isolated=())


def two_components_and_an_isolated_interface(n: int, seed: int = 1):
    """Two independent blocks of `n` unknowns each, plus one interface with no physics row."""
    a, b_a, _keys_a = a_system(n, seed=seed)
    b, b_b, _ = a_system(n, seed=seed + 1)
    keys = [(i, i + 1) for i in range(2 * n + 1)]
    physics = np.zeros((len(a) - 1 + len(b) - 1, 2 * n + 1))
    physics[: len(a) - 1, :n] = a[:-1]
    physics[len(a) - 1 :, n : 2 * n] = b[:-1]
    matrix_m = np.vstack([physics, np.ones(2 * n + 1)])
    vector_b = np.concatenate([b_a[:-1], b_b[:-1], [float(2 * n + 1)]])
    partition = JunctionGraphPartition(
        interfaces=tuple(keys), components=(tuple(keys[:n]), tuple(keys[n : 2 * n])),
        component_of={**dict.fromkeys(keys[:n], 0), **dict.fromkeys(keys[n : 2 * n], 1)},
        isolated=(keys[-1],),
    )
    return matrix_m, vector_b, keys, partition


def test_below_the_threshold_the_solve_is_dense_and_meets_the_gauge_exactly():
    """Below the threshold: dense `lstsq` on the well-scaled system, then the exact gauge.

    Until 0.6.0 this was bit-identical to `lstsq` on the soft row of ones; the ratios are unchanged
    (to 1e-12 here) and the scale now meets the gauge.
    """
    matrix_m, vector_b, keys = a_system(60)
    assert len(keys) < junction_graph.SPARSE_SOLVE_MIN_UNKNOWNS
    solution = solve_gauged_least_squares(matrix_m, vector_b, keys, 1.0, one_component(keys))
    soft, *_ = linalg.lstsq(matrix_m, vector_b)
    assert solution.tensions.sum() == pytest.approx(60.0, abs=1e-11)
    np.testing.assert_allclose(solution.tensions, soft * (60.0 / soft.sum()), rtol=0, atol=1e-12)
    note = solution.solver_notes[0]
    assert note["solver"] == "dense lstsq"
    assert note["n_unknowns"] == 60
    assert note["gauge"].startswith("exact")


def test_below_the_threshold_a_fragmented_solve_is_bit_identical_too(monkeypatch):
    matrix_m, vector_b, keys, partition = two_components_and_an_isolated_interface(40)
    dense = solve_gauged_least_squares(matrix_m, vector_b, keys, 1.0, partition)
    monkeypatch.setattr(junction_graph, "SPARSE_SOLVE_MIN_UNKNOWNS", 41)
    again = solve_gauged_least_squares(matrix_m, vector_b, keys, 1.0, partition)
    assert dense.tensions.tobytes() == again.tensions.tobytes()
    assert {c: note["solver"] for c, note in dense.solver_notes.items()} == {0: "dense lstsq", 1: "dense lstsq"}


@pytest.mark.parametrize("threshold", [0, 40])
def test_above_the_threshold_lsqr_solves_the_same_component_wise_system(monkeypatch, threshold):
    matrix_m, vector_b, keys, partition = two_components_and_an_isolated_interface(40)
    dense = solve_gauged_least_squares(matrix_m, vector_b, keys, 1.0, partition)
    monkeypatch.setattr(junction_graph, "SPARSE_SOLVE_MIN_UNKNOWNS", threshold)
    sparse = solve_gauged_least_squares(matrix_m, vector_b, keys, 1.0, partition)
    finite = np.isfinite(dense.tensions)
    assert np.array_equal(finite, np.isfinite(sparse.tensions))
    assert finite.sum() == 80
    scale = np.abs(dense.tensions[finite]).max()
    assert np.abs(sparse.tensions[finite] - dense.tensions[finite]).max() <= 1e-10 * scale
    assert sparse.undetermined == dense.undetermined
    assert sparse.component_of == dense.component_of
    assert sparse.gauge_notes == dense.gauge_notes
    for component in (0, 1):
        note = sparse.solver_notes[component]
        assert note["solver"] == "sparse lsqr"
        assert note["lsqr_istop"] in (1, 2, 4, 5)
        assert note["lsqr_iterations"] > 0
    np.testing.assert_allclose(sparse.lstsq_residuals, dense.lstsq_residuals, rtol=1e-8)


def test_above_the_threshold_the_unfragmented_path_runs_lsqr(monkeypatch):
    matrix_m, vector_b, keys = a_system(50, seed=3)
    dense = solve_gauged_least_squares(matrix_m, vector_b, keys, 1.0, one_component(keys))
    monkeypatch.setattr(junction_graph, "SPARSE_SOLVE_MIN_UNKNOWNS", 50)
    sparse = solve_gauged_least_squares(matrix_m, vector_b, keys, 1.0, one_component(keys))
    assert sparse.solver_notes[0]["solver"] == "sparse lsqr"
    assert np.abs(sparse.tensions - dense.tensions).max() <= 1e-10 * np.abs(dense.tensions).max()


@pytest.mark.parametrize("mode", [TensionComputationMethod.YoungDupre, TensionComputationMethod.LogLami])
def test_on_two_analytic_doublets_lsqr_reproduces_the_dense_tensions(monkeypatch, mode):
    mesh, _truth_a, _truth_b = _two_separated_doublets()
    dense = infer_tensions(mesh, mode=mode)
    monkeypatch.setattr(junction_graph, "SPARSE_SOLVE_MIN_UNKNOWNS", 0)
    sparse = infer_tensions(mesh, mode=mode)
    assert sorted(dense) == sorted(sparse)
    for key, value in dense.items():
        assert sparse[key] == pytest.approx(value, rel=1e-10, abs=1e-12)


def test_a_solve_that_does_not_converge_falls_back_to_the_dense_one(monkeypatch):
    matrix_m, vector_b, keys, partition = two_components_and_an_isolated_interface(40)
    dense = solve_gauged_least_squares(matrix_m, vector_b, keys, 1.0, partition)
    monkeypatch.setattr(junction_graph, "SPARSE_SOLVE_MIN_UNKNOWNS", 0)
    monkeypatch.setattr(junction_graph, "_LSQR_ITERATIONS_PER_UNKNOWN", 0.05)  # 2 iterations for 40 unknowns
    with pytest.warns(SparseSolveFallbackWarning, match="dense least-squares solve was run instead"):
        fallen_back = solve_gauged_least_squares(matrix_m, vector_b, keys, 1.0, partition)
    assert fallen_back.tensions.tobytes() == dense.tensions.tobytes()
    # The physics residual is computed by a sparse product on the fallback path, a dense one on the
    # dense path: equal to rounding, not bit for bit.
    np.testing.assert_allclose(fallen_back.lstsq_residuals, dense.lstsq_residuals, rtol=1e-12)
    for note in fallen_back.solver_notes.values():
        assert note["solver"] == "dense lstsq after lsqr did not converge"
        assert note["lsqr_istop"] == 7
