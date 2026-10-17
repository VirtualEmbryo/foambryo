"""The solver override, the sparse pressure paths and the opt-in solve diagnostics.

What this pins
--------------
Very large tissues (10 000 cells and more) cannot be solved densely: the Young-Dupre system alone
would be some 120 GB, the variational pressure's Gram matrices as much. The public entry points
therefore take `solver="auto" | "dense" | "sparse"` and a `sparse_threshold`, and
`diagnostics=False | True | "spectrum"` (`foambryo.linear_solves`). The tests pin:

1. **the default is untouched**: below the threshold every output is bitwise what the dense path
   gives, and turning diagnostics on changes no output bit;
2. **the sparse paths solve the same systems**: forced sparse against forced dense, the tensions
   and the three pressure methods agree to 1e-9 of their scale and leave the same cells undetermined
   (`nan`), on a real cut benchmark mesh, on a mesh with an undetermined Laplace cell, on a
   fragmented junction graph and on a cropped mesh;
3. **the sparse assembly writes the same entries**: the Young-Dupre and Laplace builders' sparse
   matrices densify to the dense matrices bit for bit;
4. **the options are honoured and checked**: the threshold switches the path, invalid values raise,
   methods with no sparse path refuse `solver="sparse"`, a forced dense solve too large for memory
   warns before allocating, a sparse pressure solve that does not converge falls back with a warning;
5. **the diagnostics say what they claim**: one record per solve, the dense condition number is
   the 2-norm one, `"spectrum"` adds singular values to the sparse solves, and
   `save_solve_diagnostics` writes JSON and NPZ files that read back.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import issparse

from foambryo import (
    DenseSolveMemoryWarning,
    PressureComputationMethod,
    SolveDiagnostics,
    SparseSolveFallbackWarning,
    TensionComputationMethod,
    infer_forces,
    infer_pressures,
    infer_pressures_and_residuals,
    infer_tensions,
    infer_tensions_and_residuals,
    infer_tensions_and_residuals_with_report,
    infer_tensions_by_junction_component,
    junction_graph,
    linear_solves,
    save_solve_diagnostics,
)
from foambryo.dcel import DcelData
from foambryo.pressure_inference import _build_matrix_laplace
from foambryo.tension_inference import _build_matrix_tension_symmetrical_yd
from foambryo.wedge_refusal import line_averaged_angles

from .test_crop_faces import cropped, reconstruction  # noqa: F401  (fixtures)
from .test_junction_graph import _two_separated_doublets
from .test_laplace_pressure_undetermined_cells import _tetrahedron_with_three_capped_faces, _unit_tensions

CUT_CASE = Path(__file__).parent / "data" / "cut_benchmark_case_000_mesh.npz"
PRESSURE_MODES = (
    PressureComputationMethod.Laplace,
    PressureComputationMethod.WeightedLaplace,
    PressureComputationMethod.Variational,
)
TOLERANCE = 1e-9


@pytest.fixture(scope="module")
def cut_case() -> DcelData:
    data = np.load(CUT_CASE)
    return DcelData(data["points"], data["triangles"], data["labels"])


def _quiet(function: object, *args: object, **kwargs: object) -> object:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


def _bits(mapping: dict) -> bytes:
    keys = sorted(mapping)
    return repr(keys).encode() + np.array([mapping[k] for k in keys], dtype=np.float64).tobytes()


def _assert_agree(dense: dict, sparse: dict) -> None:
    """Same keys, same undetermined (`nan`) entries, and the rest within TOLERANCE of the scale."""
    assert sorted(dense) == sorted(sparse)
    keys = sorted(dense)
    d = np.array([dense[k] for k in keys], dtype=np.float64)
    s = np.array([sparse[k] for k in keys], dtype=np.float64)
    assert np.array_equal(np.isnan(d), np.isnan(s))
    finite = np.isfinite(d)
    scale = np.abs(d[finite]).max()
    assert np.abs(s[finite] - d[finite]).max() <= TOLERANCE * scale


# --- 1. the default is untouched ---------------------------------------------------------------
def test_below_the_threshold_the_default_is_bitwise_the_dense_path(cut_case):
    tensions = _quiet(infer_tensions, cut_case)
    assert _bits(tensions) == _bits(_quiet(infer_tensions, cut_case, solver="dense"))
    for mode in PRESSURE_MODES:
        default = _quiet(infer_pressures, cut_case, tensions, mode=mode)
        assert _bits(default) == _bits(_quiet(infer_pressures, cut_case, tensions, mode=mode, solver="dense"))


@pytest.mark.parametrize("diagnostics", [True, "spectrum"])
def test_diagnostics_change_no_output_bit(cut_case, diagnostics):
    tensions = _quiet(infer_tensions, cut_case)
    with_diagnostics, records = _quiet(infer_tensions, cut_case, diagnostics=diagnostics)
    assert _bits(with_diagnostics) == _bits(tensions)
    assert isinstance(records, SolveDiagnostics)
    for mode in PRESSURE_MODES:
        for solver in ("auto", "sparse"):
            plain = _quiet(infer_pressures, cut_case, tensions, mode=mode, solver=solver)
            diagnosed, _ = _quiet(
                infer_pressures, cut_case, tensions, mode=mode, solver=solver, diagnostics=diagnostics,
            )
            assert _bits(diagnosed) == _bits(plain)


def test_every_entry_point_keeps_its_return_shape_without_diagnostics(cut_case):
    assert isinstance(_quiet(infer_tensions, cut_case), dict)
    assert len(_quiet(infer_tensions_and_residuals, cut_case)) == 3
    assert len(_quiet(infer_tensions_and_residuals_with_report, cut_case)) == 4
    tensions = _quiet(infer_tensions, cut_case)
    assert isinstance(_quiet(infer_pressures, cut_case, tensions), dict)
    assert len(_quiet(infer_pressures_and_residuals, cut_case, tensions)) == 3
    assert len(_quiet(infer_forces, cut_case, mode_pressure=PressureComputationMethod.Laplace)) == 2


def test_with_diagnostics_each_entry_point_returns_them_last(cut_case):
    assert isinstance(_quiet(infer_tensions_and_residuals, cut_case, diagnostics=True)[3], SolveDiagnostics)
    assert isinstance(_quiet(infer_tensions_and_residuals_with_report, cut_case, diagnostics=True)[4], SolveDiagnostics)
    solution, records = _quiet(infer_tensions_by_junction_component, cut_case, diagnostics=True)
    assert solution.solver_notes[0]["solver"] == "dense lstsq"
    assert len(records) == 1
    tensions = _quiet(infer_tensions, cut_case)
    assert isinstance(_quiet(infer_pressures_and_residuals, cut_case, tensions, diagnostics=True)[3], SolveDiagnostics)
    dict_tensions, dict_pressures, records = _quiet(
        infer_forces, cut_case, mode_pressure=PressureComputationMethod.WeightedLaplace, diagnostics=True,
    )
    assert [r.system for r in records.records] == ["YoungDupre tension system", "WeightedLaplace pressure system"]
    assert _bits(dict_tensions) == _bits(tensions)
    assert np.isnan(dict_pressures[2])  # the cut case's undetermined Laplace cell, unchanged


# --- 2. the sparse paths solve the same systems -----------------------------------------------
def test_forced_sparse_agrees_with_forced_dense_on_a_cut_benchmark_mesh(cut_case):
    t_dense = _quiet(infer_tensions, cut_case, solver="dense")
    t_sparse = _quiet(infer_tensions, cut_case, solver="sparse")
    _assert_agree(t_dense, t_sparse)
    for mode in PRESSURE_MODES:
        p_dense = _quiet(infer_pressures, cut_case, t_dense, mode=mode, solver="dense")
        p_sparse = _quiet(infer_pressures, cut_case, t_dense, mode=mode, solver="sparse")
        _assert_agree(p_dense, p_sparse)
    # the Laplace methods leave cell 2 undetermined on both paths
    laplace = _quiet(infer_pressures, cut_case, t_dense, mode=PressureComputationMethod.Laplace, solver="sparse")
    assert np.isnan(laplace[2])


@pytest.mark.parametrize("mode", [PressureComputationMethod.Laplace, PressureComputationMethod.WeightedLaplace])
def test_the_sparse_laplace_path_reports_the_same_undetermined_cell(mode):
    mesh = _tetrahedron_with_three_capped_faces()
    tensions = _unit_tensions(mesh)
    dense = _quiet(infer_pressures, mesh, tensions, mode=mode, solver="dense")
    with pytest.warns(UserWarning, match="undetermined"):
        sparse = infer_pressures(mesh, tensions, mode=mode, solver="sparse")
    _assert_agree(dense, sparse)
    assert sum(np.isnan(v) for v in sparse.values()) == 1


def test_a_fragmented_junction_graph_is_solved_sparse_component_by_component():
    mesh, _truth_a, _truth_b = _two_separated_doublets()
    dense = _quiet(infer_tensions_by_junction_component, mesh, solver="dense")
    sparse = _quiet(infer_tensions_by_junction_component, mesh, solver="sparse")
    assert sparse.fragmented
    assert sparse.n_components == dense.n_components == 2
    assert {note["solver"] for note in sparse.solver_notes.values()} == {"sparse lsqr"}
    _assert_agree(dense.dict_tensions, sparse.dict_tensions)


def test_a_sparse_assembly_with_dense_components_is_bitwise_the_dense_solve():
    """Whole system at the threshold (sparse assembly), each component below it (dense solve)."""
    mesh, _truth_a, _truth_b = _two_separated_doublets()
    dense = _quiet(infer_tensions_by_junction_component, mesh)
    n_interfaces = len(dense.dict_tensions)
    largest_component = max(sum(1 for c in dense.component_of.values() if c == k) for k in range(2))
    assert largest_component < n_interfaces
    mixed = _quiet(infer_tensions_by_junction_component, mesh, sparse_threshold=n_interfaces)
    assert {note["solver"] for note in mixed.solver_notes.values()} == {"dense lstsq"}
    assert mixed.tensions.tobytes() == dense.tensions.tobytes()


def test_the_cropped_variational_pressure_takes_either_path_on_request(cropped):  # noqa: F811
    tensions = _quiet(infer_tensions, cropped)
    dense, records_dense = _quiet(infer_pressures, cropped, tensions, solver="dense", diagnostics=True)
    sparse, records_sparse = _quiet(infer_pressures, cropped, tensions, solver="sparse", diagnostics=True)
    assert records_dense.records[0].solver == "dense lstsq"
    assert records_sparse.records[0].solver == "sparse normal equations + dense eigh"
    _assert_agree(dense, sparse)


# --- 3. the sparse assembly writes the same entries -------------------------------------------
def test_the_young_dupre_builder_writes_the_same_entries_dense_and_sparse(cut_case):
    angles = _quiet(line_averaged_angles, cut_case)
    areas = cut_case.compute_areas_interfaces()
    dense, b_dense = _build_matrix_tension_symmetrical_yd(angles.dict_rad, areas, angles.dict_length)
    sparse, b_sparse = _build_matrix_tension_symmetrical_yd(angles.dict_rad, areas, angles.dict_length, as_sparse=True)
    assert issparse(sparse)
    assert sparse.toarray().tobytes() == dense.tobytes()
    assert b_sparse.tobytes() == b_dense.tobytes()


def test_the_laplace_builder_writes_the_same_entries_dense_and_sparse(cut_case):
    tensions = _quiet(infer_tensions, cut_case)
    curvature = cut_case.compute_curvatures_interfaces(weighted=True)
    areas = cut_case.compute_areas_interfaces()
    args = (curvature, areas, tensions, cut_case.n_materials, cut_case.materials)
    dense, b_dense = _build_matrix_laplace(*args)
    sparse, b_sparse = _build_matrix_laplace(*args, as_sparse=True)
    assert sparse.toarray().tobytes() == dense.tobytes()
    assert b_sparse.tobytes() == b_dense.tobytes()


# --- 4. the options are honoured and checked ---------------------------------------------------
def test_the_threshold_switches_the_path(cut_case):
    tensions = _quiet(infer_tensions, cut_case)
    n_materials = cut_case.n_materials
    _, below = _quiet(infer_pressures, cut_case, tensions, mode=PressureComputationMethod.Laplace,
                      sparse_threshold=n_materials + 1, diagnostics=True)
    _, at = _quiet(infer_pressures, cut_case, tensions, mode=PressureComputationMethod.Laplace,
                   sparse_threshold=n_materials, diagnostics=True)
    assert below.records[0].solver == "dense lstsq"
    assert at.records[0].solver == "sparse lsqr"
    _, records = _quiet(infer_tensions, cut_case, sparse_threshold=0, diagnostics=True)
    assert records.records[0].solver == "sparse lsqr"


def test_the_package_threshold_is_read_when_the_solve_runs(cut_case, monkeypatch):
    monkeypatch.setattr(junction_graph, "SPARSE_SOLVE_MIN_UNKNOWNS", 0)
    tensions = _quiet(infer_tensions, cut_case)
    _, records = _quiet(infer_pressures, cut_case, tensions, diagnostics=True)
    assert records.records[0].solver == "sparse direct (SuperLU)"


@pytest.mark.parametrize(
    ("solver", "threshold", "diagnostics"),
    [("lsqr", None, False), ("Dense", None, False), ("auto", -1, False), ("auto", 1.5, False), ("auto", None, "full")],
)
def test_invalid_options_raise(cut_case, solver, threshold, diagnostics):
    with pytest.raises(ValueError, match="must be"):
        infer_tensions(cut_case, solver=solver, sparse_threshold=threshold, diagnostics=diagnostics)
    with pytest.raises(ValueError, match="must be"):
        infer_forces(cut_case, solver=solver, sparse_threshold=threshold, diagnostics=diagnostics)


@pytest.mark.parametrize(
    "mode",
    [TensionComputationMethod.Equilibrium, TensionComputationMethod.Variational, TensionComputationMethod.ForceBalance],
)
def test_methods_without_a_sparse_path_refuse_a_forced_sparse_solver(cut_case, mode):
    with pytest.raises(ValueError, match="has no sparse solver"):
        infer_tensions(cut_case, mode=mode, solver="sparse")
    dense = _quiet(infer_tensions, cut_case, mode=mode, solver="dense")
    assert _bits(dense) == _bits(_quiet(infer_tensions, cut_case, mode=mode))


def test_a_forced_dense_solve_too_large_for_memory_warns_before_allocating(cut_case, monkeypatch):
    monkeypatch.setattr(linear_solves, "available_memory_bytes", lambda: 1024)
    tensions = _quiet(infer_tensions, cut_case)
    with pytest.warns(DenseSolveMemoryWarning, match=r"needs about .* GiB .* more than the"):
        infer_pressures(cut_case, tensions, mode=PressureComputationMethod.Laplace, solver="dense")
    with pytest.warns(DenseSolveMemoryWarning, match="YoungDupre tension system"):
        infer_tensions(cut_case, solver="dense")
    with warnings.catch_warnings(record=True) as caught:  # "auto" is not forced: it never warns
        warnings.simplefilter("always")
        infer_pressures(cut_case, tensions, mode=PressureComputationMethod.Laplace)
    assert not any(issubclass(w.category, DenseSolveMemoryWarning) for w in caught)


def test_a_sparse_pressure_solve_that_does_not_converge_falls_back_to_dense(cut_case, monkeypatch):
    tensions = _quiet(infer_tensions, cut_case)
    dense = _quiet(infer_pressures, cut_case, tensions, mode=PressureComputationMethod.Laplace, solver="dense")
    monkeypatch.setattr(junction_graph, "_LSQR_ITERATIONS_PER_UNKNOWN", 0.2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fallen_back, records = infer_pressures(
            cut_case, tensions, mode=PressureComputationMethod.Laplace, solver="sparse", diagnostics=True,
        )
    assert any(issubclass(w.category, SparseSolveFallbackWarning) for w in caught)
    assert records.records[0].solver == "dense lstsq after lsqr did not converge"
    assert records.records[0].lsqr_istop == 7
    _assert_agree(dense, fallen_back)


# --- 5. the diagnostics say what they claim ----------------------------------------------------
def test_dense_records_carry_the_two_norm_condition_number_and_every_singular_value(cut_case):
    tensions = _quiet(infer_tensions, cut_case)
    curvature = cut_case.compute_curvatures_interfaces(weighted=False)
    whole, _b = _build_matrix_laplace(curvature, cut_case.compute_areas_interfaces(), tensions,
                                      cut_case.n_materials, cut_case.materials)
    # The system solved: the exterior's pin eliminated (column 0, last row) and the undetermined
    # cell 2 left out with the rows that touch it.
    cell_2 = list(cut_case.materials).index(2)
    columns = [j for j in range(whole.shape[1]) if j not in (0, cell_2)]
    rows = [i for i in range(whole.shape[0] - 1) if whole[i, cell_2] == 0]
    matrix = whole[np.ix_(rows, columns)]
    _, records = _quiet(infer_pressures, cut_case, tensions, mode=PressureComputationMethod.Laplace, diagnostics=True)
    record = records.records[0]
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    assert (record.n_rows, record.n_unknowns) == matrix.shape
    assert record.rank == np.linalg.matrix_rank(matrix) == matrix.shape[1]  # full rank once reduced
    np.testing.assert_allclose(record.singular_values, singular_values, rtol=1e-10, atol=1e-14)
    assert record.condition_number == pytest.approx(singular_values[0] / singular_values[-1], rel=1e-8)
    assert record.residual_norm >= 0
    assert record.seconds >= 0


def test_sparse_records_carry_lsqr_s_estimate_and_the_spectrum_on_request(cut_case):
    _, plain = _quiet(infer_tensions, cut_case, solver="sparse", diagnostics=True)
    _, spectrum = _quiet(infer_tensions, cut_case, solver="sparse", diagnostics="spectrum")
    record = plain.records[0]
    assert record.solver == "sparse lsqr"
    assert record.lsqr_istop in (1, 2, 4, 5)
    assert record.lsqr_iterations > 0
    assert record.condition_number_source == "lsqr estimate (Frobenius norm)"
    assert record.singular_values is None
    with_spectrum = spectrum.records[0]
    assert with_spectrum.singular_values is not None
    assert with_spectrum.spectrum_seconds is not None
    assert np.all(np.diff(with_spectrum.singular_values) <= 0)
    # the largest singular value svds finds is the dense one
    _, dense = _quiet(infer_tensions, cut_case, solver="dense", diagnostics=True)
    assert with_spectrum.singular_values[0] == pytest.approx(dense.records[0].singular_values[0], rel=1e-8)


def test_the_variational_pressure_records_name_their_solver_and_condition_source(cut_case):
    tensions = _quiet(infer_tensions, cut_case)
    _, dense = _quiet(infer_pressures, cut_case, tensions, solver="dense", diagnostics=True)
    _, sparse = _quiet(infer_pressures, cut_case, tensions, solver="sparse", diagnostics="spectrum")
    assert dense.records[0].solver == "dense inv"
    assert dense.records[0].condition_number >= 1
    assert sparse.records[0].solver == "sparse direct (SuperLU)"
    assert sparse.records[0].condition_number_source.startswith("1-norm estimate")
    assert sparse.records[0].singular_values is not None


def test_saved_diagnostics_read_back(cut_case, tmp_path):
    _, _, records = _quiet(infer_forces, cut_case, solver="sparse", diagnostics="spectrum")
    records.records[0].condition_number = float("inf")  # the JSON must survive a non-finite number
    json_path, npz_path = save_solve_diagnostics(records, tmp_path / "solves")
    payload = json.loads(json_path.read_text())
    assert payload["options"] == {"solver": "sparse", "sparse_threshold": None, "diagnostics": "spectrum"}
    assert [s["system"] for s in payload["solves"]] == [r.system for r in records.records]
    assert payload["solves"][0]["condition_number"] == "inf"
    arrays = np.load(npz_path)
    for index, record in enumerate(records.records):
        if record.singular_values is not None:
            np.testing.assert_array_equal(arrays[f"singular_values_{index}"], record.singular_values)


def test_column_scaling_makes_lsqr_accurate_on_a_badly_scaled_system():
    """Cells whose every interface is small have tiny columns: column scaling fixes lsqr's accuracy.

    The Laplace rows carry interface area fractions, so a small cell's column is small throughout.
    Here: a well-connected random cell graph pinned to the exterior (each row `P_a - P_b`), whose
    columns are then multiplied by weights spanning four decades. Column-scaled `lsqr` must match
    the dense solution to 1e-11 of its scale, where unscaled `lsqr` does not converge at all.
    """
    from scipy.sparse import coo_matrix, diags

    rng = np.random.default_rng(0)
    n = 300
    pairs = {tuple(sorted(rng.choice(n, size=2, replace=False))) for _ in range(4 * n)}
    pairs |= {(i, i + 1) for i in range(n - 1)}  # connected
    rows, cols, values = [], [], []
    for row, (a, b) in enumerate(sorted(pairs)):
        rows += [row, row]
        cols += [a, b]
        values += [1.0, -1.0]
    pin = len(pairs)
    rows.append(pin)
    cols.append(0)
    values.append(1.0)
    incidence = coo_matrix((values, (rows, cols)), shape=(pin + 1, n)).tocsr()
    matrix = (incidence @ diags(10.0 ** rng.uniform(-4, 0, size=n))).tocsr()
    vector = matrix @ rng.normal(size=n) + 1e-3 * rng.normal(size=pin + 1)
    dense, _residuals, _record = junction_graph._solve_least_squares(matrix, vector, use_sparse=False)
    scaled, _r, scaled_record = junction_graph._solve_least_squares(matrix, vector, use_sparse=True, scale_columns=True)
    assert scaled_record["solver"] == "sparse lsqr"
    assert scaled_record["lsqr_columns_scaled"] is True
    assert np.abs(scaled - dense).max() <= 1e-11 * np.abs(dense).max()
    # unscaled, lsqr does not converge within its iteration limit and the dense solve is run instead
    with pytest.warns(SparseSolveFallbackWarning):
        _plain, _r, plain_record = junction_graph._solve_least_squares(matrix, vector, use_sparse=True)
    assert plain_record["solver"] == "dense lstsq after lsqr did not converge"
    assert scaled_record["lsqr_iterations"] < plain_record["lsqr_iterations"]


def test_a_dense_fallback_too_large_for_memory_is_not_run(cut_case, monkeypatch):
    """At tissue scale the dense fallback would allocate hundreds of GB: it is skipped, loudly."""
    tensions = _quiet(infer_tensions, cut_case)
    monkeypatch.setattr(junction_graph, "_LSQR_ITERATIONS_PER_UNKNOWN", 0.2)
    monkeypatch.setattr(linear_solves, "available_memory_bytes", lambda: 1024)
    with pytest.warns(SparseSolveFallbackWarning, match="dense fallback was not run"):
        pressures, records = infer_pressures(
            cut_case, tensions, mode=PressureComputationMethod.Laplace, solver="sparse", diagnostics=True,
        )
    assert records.records[0].solver == "sparse lsqr, not converged (dense fallback too large for memory)"
    assert records.records[0].lsqr_istop == 7
    dense = _quiet(infer_pressures, cut_case, tensions, mode=PressureComputationMethod.Laplace)
    assert sorted(pressures) == sorted(dense)


@pytest.mark.parametrize("symmetric_positive", [False, True])
def test_the_smallest_singular_values_fall_back_to_shift_invert(monkeypatch, symmetric_positive):
    """When ARPACK's smallest-value search does not converge, shift-invert at zero gives them."""
    from scipy.sparse import eye as sparse_eye
    from scipy.sparse import random as sparse_random

    rng = np.random.default_rng(1)
    matrix = sparse_random(400, 60, density=0.05, random_state=rng, format="csr")
    matrix = matrix + sparse_eye(400, 60, format="csr")
    if symmetric_positive:
        matrix = (matrix.T @ matrix).tocsr()
    exact = np.linalg.svd(matrix.toarray(), compute_uv=False)
    real_svds = linear_solves.svds

    def svds_that_fails_at_the_small_end(*args: object, **kwargs: object) -> object:
        if kwargs.get("which") == "SM":
            message = "forced for the test"
            raise linear_solves.ArpackNoConvergence(message, np.zeros(0), np.zeros(0))
        return real_svds(*args, **kwargs)

    monkeypatch.setattr(linear_solves, "svds", svds_that_fails_at_the_small_end)
    values, seconds, note = linear_solves.extreme_singular_values(matrix, symmetric_positive=symmetric_positive)
    assert seconds >= 0
    assert "smallest values by shift-invert" in note
    k = linear_solves.SPECTRUM_EXTREME_VALUES
    np.testing.assert_allclose(np.sort(values)[:k], np.sort(exact)[:k], rtol=1e-6)
