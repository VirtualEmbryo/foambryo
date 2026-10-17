"""The uncertainty work — calibrated uncertainty, and the diagnostics that say whether to trust it.

What is pinned here, and why each one is a guard against a specific way this feature could rot:

1. **The point estimates never move.** `infer_tensions_with_uncertainty` must return exactly
   what `infer_tensions` returns for the same method. An "uncertainty" feature that quietly
   perturbs the estimate would invalidate every number measured before it.
2. **The junction-length weighting of `YoungDupre` is untouched.** The per-edge Young-Dupre work measured that
   `YoungDupre`'s 2.3e-05 is *weighting-critical* — a factor-~4 change in inter-line weights
   costs two orders of magnitude — so this is the one thing the uncertainty work was told not to disturb as a
   side effect. Pinned by comparing the assembled operator to the shipped builder's, entry by
   entry, and by asserting the tensions are bit-identical.
3. **`IidGls` really is ~250x optimistic, and `CoherentResidual` really is O(1).** The
   headline result is asserted on the golden mesh, so a future change to the covariance path
   fails here rather than silently shipping a wrong error bar.
4. **The two models differ by exactly `sqrt(dof)`.** That is the whole content of the fix, and
   stating it as a test makes it impossible to "improve" one without understanding the other.
5. **The calibration notes match the committed measurement.** The module tells users a number;
   the benchmark measures it; if they disagree, the module is lying. Asserted against
   `foambryo_benchmarks/baseline/uncertainty_calibration.json`.
6. **An uncalibrated interval cannot leave the module.** Requesting one for a method the uncertainty work never
   measured raises rather than guessing.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from foambryo import TensionComputationMethod, dcel_mesh_from_file, infer_tensions
from foambryo.tension_inference import _build_matrix_tension_symmetrical_yd
from foambryo.uncertainty import (
    CALIBRATION_ERROR_OVER_STD,
    UncertaintyModel,
    compute_inference_diagnostics,
    error_over_std,
    infer_tensions_with_uncertainty,
)
from foambryo.validation import force_balance as fb
from tests.conftest import DATASET_DIR, require_substitute_mesh

CALIBRATION_RECORD = (
    Path(__file__).resolve().parent.parent / "foambryo_benchmarks" / "baseline" / "uncertainty_calibration.json"
)


@pytest.fixture(scope="module")
def golden_mesh():
    return dcel_mesh_from_file(str(require_substitute_mesh()))


@pytest.fixture(scope="module")
def golden_truth():
    path = DATASET_DIR / "000_dict_tensions.npy"
    if not path.exists():
        pytest.skip(f"{path} not found")
    return np.load(path, allow_pickle=True).item()


@pytest.fixture(scope="module")
def calibration_summary():
    if not CALIBRATION_RECORD.exists():
        pytest.skip(f"{CALIBRATION_RECORD} not found; run foambryo_benchmarks.run_uncertainty_calibration")
    return json.loads(CALIBRATION_RECORD.read_text())["summary"]["equilibrium"]


# --------------------------------------------------------------------------------------
# 1. The point estimate is never touched
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method",
    [TensionComputationMethod.ForceBalance, TensionComputationMethod.YoungDupre],
)
def test_point_estimate_is_bit_identical_to_infer_tensions(golden_mesh, method):
    result = infer_tensions_with_uncertainty(golden_mesh, 1.0, method=method)
    reference = infer_tensions(golden_mesh, mean_tension=1, mode=method)
    assert set(result.dict_tensions) == set(reference)
    for key, value in reference.items():
        assert result.dict_tensions[key] == pytest.approx(value, rel=0, abs=0), key


def test_variational_and_force_balance_are_now_the_same_estimator(golden_mesh):
    """Under route `Operator`, `Variational` *is* `ForceBalance` — to 1e-14, not to 1e-5.

    This is an uncertainty-work result worth pinning rather than a technicality. The force-balance work concluded
    "`ForceBalance` = `Variational`, they agree to a median 1.5e-05"; route P's row
    normalisation was the whole of that residual disagreement, and with route Q the default the
    two are the *same* constrained least-squares problem, so they agree to machine precision.
    That is why the uncertainty module can build one interval for both, and it is the reason the
    route flip removes a duplicated-but-subtly-different pair from the public API.

    Route `Gram` is kept reachable, and its disagreement (~4e-04 here) is asserted too — so a
    future reader can see that the unification is a property of route Q, not of the mesh.
    """
    from foambryo.tension_inference import VariationalRoute, _infer_tension_variational_yd

    balance = infer_tensions(golden_mesh, mean_tension=1, mode=TensionComputationMethod.ForceBalance)
    keys = sorted(balance)
    reference = np.array([balance[key] for key in keys])
    for route, tolerance, at_least in (
        (VariationalRoute.Operator, 1e-12, None),
        (VariationalRoute.Gram, None, 1e-6),
    ):
        _values, tensions, _resid = _infer_tension_variational_yd(golden_mesh, 1.0, route=route)
        got = np.array([tensions[tuple(key)] for key in keys])
        deviation = float(np.abs(got - reference).max())
        if tolerance is not None:
            assert deviation < tolerance, (route, deviation)
        else:
            assert deviation > at_least, (route, deviation)


def test_uncertainty_for_variational_matches_its_own_point_estimate(golden_mesh):
    """`Variational`'s interval is built on the same estimate `infer_tensions` returns.

    Agreement to 1e-12 rather than bit-identity, because the interval is built from the
    force-balance operator assembled by `validation.force_balance` while `Variational` assembles
    from `geometry`'s sparse operators — the same mathematics through two summation orders.
    """
    result = infer_tensions_with_uncertainty(
        golden_mesh,
        1.0,
        method=TensionComputationMethod.Variational,
    )
    reference = infer_tensions(golden_mesh, mean_tension=1, mode=TensionComputationMethod.Variational)
    for key, value in reference.items():
        assert result.dict_tensions[key] == pytest.approx(value, abs=1e-12), key


def test_young_dupre_weighting_is_untouched(golden_mesh):
    """The load-bearing junction-length weights (per-edge Young-Dupre work) are reused, not rebuilt differently."""
    from foambryo.uncertainty import _young_dupre_operator

    matrix_m, vector_b, _interfaces, _rad, _areas, _length = _young_dupre_operator(golden_mesh, 1.0)
    dict_rad, _deg, dict_length = golden_mesh.compute_angles_tri(unique=False)
    dict_areas = golden_mesh.compute_areas_interfaces()
    expected_m, expected_b = _build_matrix_tension_symmetrical_yd(dict_rad, dict_areas, dict_length, 1.0)
    np.testing.assert_array_equal(matrix_m, expected_m)
    np.testing.assert_array_equal(vector_b, expected_b)


# --------------------------------------------------------------------------------------
# 2. The headline calibration result
# --------------------------------------------------------------------------------------


def test_iid_gls_is_two_orders_optimistic_and_coherent_residual_is_order_one(golden_mesh, golden_truth):
    naive = infer_tensions_with_uncertainty(
        golden_mesh,
        1.0,
        method=TensionComputationMethod.ForceBalance,
        model=UncertaintyModel.IidGls,
    )
    coherent = infer_tensions_with_uncertainty(
        golden_mesh,
        1.0,
        method=TensionComputationMethod.ForceBalance,
        model=UncertaintyModel.CoherentResidual,
    )
    naive_ratio = error_over_std(naive, golden_truth)["median"]
    coherent_ratio = error_over_std(coherent, golden_truth)["median"]

    # The failure this asserts is the one the uncertainty work exists to prevent: shipping the naive interval.
    assert naive_ratio > 50, naive_ratio
    assert 0.1 <= coherent_ratio <= 10.0, coherent_ratio
    assert coherent_ratio < naive_ratio / 50


def test_the_two_models_differ_by_exactly_sqrt_dof(golden_mesh):
    """`CoherentResidual` is `IidGls` with the i.i.d. averaging factor removed. Nothing else."""
    naive = infer_tensions_with_uncertainty(
        golden_mesh,
        1.0,
        method=TensionComputationMethod.ForceBalance,
        model=UncertaintyModel.IidGls,
    )
    coherent = infer_tensions_with_uncertainty(
        golden_mesh,
        1.0,
        method=TensionComputationMethod.ForceBalance,
        model=UncertaintyModel.CoherentResidual,
    )
    diagnostics = coherent.diagnostics
    dof = coherent.n_rows - (diagnostics.n_interfaces + diagnostics.n_cells - 1)
    np.testing.assert_allclose(
        coherent.tension_std,
        naive.tension_std * np.sqrt(dof),
        rtol=1e-12,
    )


def test_coherent_residual_scale_is_the_measured_residual_norm(golden_mesh):
    """No free hyperparameter: the interval's scale is a measurement, not a choice."""
    result = infer_tensions_with_uncertainty(
        golden_mesh,
        1.0,
        method=TensionComputationMethod.ForceBalance,
        model=UncertaintyModel.CoherentResidual,
    )
    assert result.sigma_scale == pytest.approx(result.residual_norm, rel=1e-14)


def test_gauge_direction_has_zero_variance(golden_mesh):
    """The overall tension scale is a convention, so its variance must be exactly zero.

    The constrained covariance is singular along the constraint, which is the correct
    statement; a non-zero number there would be claiming to have measured a gauge choice.
    """
    result = infer_tensions_with_uncertainty(
        golden_mesh,
        1.0,
        method=TensionComputationMethod.ForceBalance,
        model=UncertaintyModel.CoherentResidual,
    )
    ni = result.diagnostics.n_interfaces
    m, _interfaces, _cells = __import__(
        "foambryo.force_balance_inference",
        fromlist=["build_force_balance_operator"],
    ).build_force_balance_operator(golden_mesh)
    anchor = np.zeros(m.shape[1])
    anchor[:ni] = 1.0 / ni
    cov = fb.constrained_covariance(m, anchor, 1.0)
    # Relative to the covariance's own scale: its diagonal is O(1e3) here, so "zero" means
    # zero to double precision, not zero as an absolute number.
    assert abs(float(anchor @ cov @ anchor)) < 1e-12 * float(np.abs(np.diag(cov)).max())


def _young_dupre_propagation(golden_mesh):
    from foambryo.junction_graph import solve_with_exact_gauge
    from foambryo.uncertainty import _angle_propagation_std, _young_dupre_operator

    matrix_m, vector_b, *_ = _young_dupre_operator(golden_mesh, 1.0)
    tensions, _res, _note = solve_with_exact_gauge(matrix_m[:-1], vector_b[:-1], float(vector_b[-1]))
    std, _se, jacobian, dp_gamma, dpt_residual = _angle_propagation_std(
        golden_mesh, matrix_m, tensions, return_jacobian=True,
    )
    return matrix_m, tensions, std, jacobian, dp_gamma.toarray(), dpt_residual.toarray()


def _gauge_basis(n):
    q, _r = np.linalg.qr(np.hstack([np.ones((n, 1)), np.eye(n)[:, : n - 1]]))
    return q[:, 1:]


def test_the_angle_propagation_is_the_full_derivative_of_the_exact_gauge_solve(golden_mesh):
    """`J = -Z (P Z)^+ dP gamma - Z ((P Z)^T P Z)^+ Z^T dP^T r`, `Z` any orthonormal basis of `1⊥`.

    Since 0.7.0 the propagation differentiates the exact-gauge solve (every Jacobian column sums to
    zero: the mean tension has no variance), and includes the residual term, which was dropped
    until 0.6.0. Checked here against an independent construction; on this mesh the term moves the
    standard errors, but by less than 10 % at the median.
    """
    matrix_m, _tensions, std, jacobian, dp_gamma, dpt_residual = _young_dupre_propagation(golden_mesh)
    physics = matrix_m[:-1]
    z = _gauge_basis(physics.shape[1])
    reduced = physics @ z
    expected = -z @ np.linalg.pinv(reduced) @ dp_gamma - z @ np.linalg.pinv(reduced.T @ reduced) @ z.T @ dpt_residual
    scale = np.abs(expected).max()
    np.testing.assert_allclose(jacobian, expected, rtol=0, atol=1e-10 * scale)
    assert np.abs(jacobian.sum(axis=0)).max() <= 1e-12 * scale
    without_term = -z @ np.linalg.pinv(reduced) @ dp_gamma
    relative = np.abs(np.linalg.norm(jacobian, axis=1) - np.linalg.norm(without_term, axis=1)) / np.linalg.norm(
        without_term, axis=1)
    assert 0 < np.median(relative) < 0.1
    assert np.all(std > 0)


def test_the_propagation_matches_finite_differences_of_the_constrained_solve():
    """The Jacobian against central differences of `solve_with_exact_gauge`, on random systems.

    Homogeneous physics rows `P`, perturbation directions `dP_k` with two entries each (as an angle
    enters the Young-Dupre rows): the derivative of the exact-gauge solution along every `dP_k`.
    """
    from scipy import sparse

    from foambryo.junction_graph import solve_with_exact_gauge
    from foambryo.uncertainty import _exact_gauge_propagated_variance

    rng = np.random.default_rng(7)
    n_rows, n, n_wedges = 40, 9, 12
    physics = rng.normal(size=(n_rows, n))
    gauge = float(n)
    gamma, *_ = solve_with_exact_gauge(physics, np.zeros(n_rows), gauge, use_sparse=False)
    residual = physics @ gamma
    directions = []
    dp_gamma = np.zeros((n_rows, n_wedges))
    dpt_residual = np.zeros((n, n_wedges))
    for k in range(n_wedges):
        d = np.zeros((n_rows, n))
        for row, col in zip(rng.choice(n_rows, 2, replace=False), rng.choice(n, 2, replace=False), strict=True):
            d[row, col] = rng.normal()
        directions.append(d)
        dp_gamma[:, k] = d @ gamma
        dpt_residual[:, k] = d.T @ residual
    _variance, jacobian = _exact_gauge_propagated_variance(
        physics, sparse.csc_matrix(dp_gamma), sparse.csc_matrix(dpt_residual), np.ones(n_wedges), return_jacobian=True,
    )
    step = 1e-6
    for k, d in enumerate(directions):
        up, *_ = solve_with_exact_gauge(physics + step * d, np.zeros(n_rows), gauge, use_sparse=False)
        down, *_ = solve_with_exact_gauge(physics - step * d, np.zeros(n_rows), gauge, use_sparse=False)
        finite = (up - down) / (2 * step)
        np.testing.assert_allclose(jacobian[:, k], finite, rtol=0, atol=1e-6 * np.abs(finite).max())


# --------------------------------------------------------------------------------------
# 3. Resampling cannot fix a bias — the reason the deterministic bound is preferred
# --------------------------------------------------------------------------------------


def test_resampling_models_remain_optimistic(golden_mesh, golden_truth):
    """Cluster-robust and cluster bootstrap model correlation; the dominant term is a bias.

    Both must land far above 1, and both must be *better* than the naive i.i.d. model — i.e.
    they recover part of the missing factor but nowhere near all of it. If a future change made
    one of them O(1), that would be a genuine finding and this test should be re-derived, not
    relaxed.
    """
    ratios = {}
    for model in (UncertaintyModel.ClusterRobust, UncertaintyModel.ClusterBootstrap, UncertaintyModel.IidGls):
        result = infer_tensions_with_uncertainty(
            golden_mesh,
            1.0,
            method=TensionComputationMethod.ForceBalance,
            model=model,
            n_resamples=50,
            seed=0,
        )
        ratios[model] = error_over_std(result, golden_truth)["median"]
    assert ratios[UncertaintyModel.ClusterRobust] > 5
    assert ratios[UncertaintyModel.ClusterBootstrap] > 5
    assert ratios[UncertaintyModel.ClusterRobust] < ratios[UncertaintyModel.IidGls]


def test_angle_family_intervals_are_conservative_not_optimistic(golden_mesh, golden_truth):
    """`YoungDupre`'s intervals err safe, by ~2 orders — the honest negative result.

    The force-balance work established the mechanism: the dataset's angle deviations from Neumann lie nearly
    orthogonal to the Young-Dupre operator's column space, so propagating their magnitude
    massively over-predicts the error.
    """
    for model in (UncertaintyModel.AnglePropagation, UncertaintyModel.LineBootstrap):
        result = infer_tensions_with_uncertainty(
            golden_mesh,
            1.0,
            method=TensionComputationMethod.YoungDupre,
            model=model,
            n_resamples=50,
            seed=0,
        )
        ratio = error_over_std(result, golden_truth)["median"]
        assert 0 < ratio < 0.5, (model, ratio)


# --------------------------------------------------------------------------------------
# 4. The module cannot lie about its calibration, and cannot ship an unmeasured one
# --------------------------------------------------------------------------------------


def test_calibration_notes_match_the_committed_measurement(calibration_summary):
    """Every advertised `error/std` agrees with the benchmark, to within a factor 1.5.

    A tolerance rather than equality, because the constant in the module is a rounded
    human-readable figure; a factor 1.5 is tight enough to catch a stale claim (the failures
    this guards against are order-of-magnitude ones) and loose enough not to churn on rounding.
    """
    measured = calibration_summary["variants"]
    name_of = {
        UncertaintyModel.ProbeResidual: "ProbeResidual",
        UncertaintyModel.CoherentResidual: "CoherentResidual",
        UncertaintyModel.IidGls: "IidGls",
        UncertaintyModel.ClusterRobust: "ClusterRobust",
        UncertaintyModel.ClusterBootstrap: "ClusterBootstrap",
        UncertaintyModel.AnglePropagation: "AnglePropagation",
        UncertaintyModel.LineBootstrap: "LineBootstrap",
    }
    for model, advertised in CALIBRATION_ERROR_OVER_STD.items():
        actual = measured[name_of[model]]["median_error_over_std"]
        assert actual == pytest.approx(advertised, rel=0.5), (model.name, advertised, actual)


def test_only_coherent_residual_is_order_one_among_the_hyperparameter_free_models(calibration_summary):
    """The headline verdict, asserted so it cannot decay into prose.

    `BayesianMap` also lands O(1), but its width is set by a prior s.d. nobody derived — the
    committed sweep shows `error/std` tracks `1/prior_std`, so its calibration was chosen. Among
    the models with no free knob, `CoherentResidual` is the only one.
    """
    variants = calibration_summary["variants"]
    hyperparameter_free = ("ProbeResidual", "CoherentResidual", "IidGls", "ClusterRobust",
                           "ClusterBootstrap", "AnglePropagation", "LineBootstrap")
    order_one = [name for name in hyperparameter_free if variants[name]["is_order_one"]]
    assert order_one == ["ProbeResidual", "CoherentResidual"], order_one
    # Only ProbeResidual is conservative on EVERY case, which is what makes it the default.
    assert variants["ProbeResidual"]["max_error_over_std"] <= 1.0
    assert variants["CoherentResidual"]["max_error_over_std"] > 10.0
    slope = calibration_summary["map_prior_sweep"]["log_log_slope_median"]
    assert slope < -0.7, slope


def test_an_uncalibrated_interval_cannot_be_requested(golden_mesh):
    with pytest.raises(NotImplementedError, match="no calibrated interval"):
        infer_tensions_with_uncertainty(golden_mesh, 1.0, method=TensionComputationMethod.Cotan)
    with pytest.raises(NotImplementedError, match="YoungDupreLocal"):
        infer_tensions_with_uncertainty(golden_mesh, 1.0, method=TensionComputationMethod.YoungDupreLocal)


def test_a_model_from_the_wrong_family_is_rejected(golden_mesh):
    with pytest.raises(ValueError, match=r"other\s+estimator family"):
        infer_tensions_with_uncertainty(
            golden_mesh,
            1.0,
            method=TensionComputationMethod.YoungDupre,
            model=UncertaintyModel.CoherentResidual,
        )


def test_every_result_carries_a_calibration_note(golden_mesh):
    for method, models in (
        (TensionComputationMethod.ForceBalance, (UncertaintyModel.CoherentResidual, UncertaintyModel.IidGls)),
        (TensionComputationMethod.YoungDupre, (UncertaintyModel.AnglePropagation,)),
    ):
        for model in models:
            result = infer_tensions_with_uncertainty(golden_mesh, 1.0, method=method, model=model)
            assert "error/std" in result.calibration_note
            assert result.model is model
    # The dangerous one must say so in words, not only in a number.
    naive = infer_tensions_with_uncertainty(
        golden_mesh,
        1.0,
        method=TensionComputationMethod.ForceBalance,
        model=UncertaintyModel.IidGls,
    )
    assert "DO NOT REPORT" in naive.calibration_note


def test_bootstrap_is_reproducible_from_its_seed(golden_mesh):
    kwargs = {
        "method": TensionComputationMethod.ForceBalance,
        "model": UncertaintyModel.ClusterBootstrap,
        "n_resamples": 30,
    }
    first = infer_tensions_with_uncertainty(golden_mesh, 1.0, seed=7, **kwargs)
    again = infer_tensions_with_uncertainty(golden_mesh, 1.0, seed=7, **kwargs)
    other = infer_tensions_with_uncertainty(golden_mesh, 1.0, seed=8, **kwargs)
    np.testing.assert_array_equal(first.tension_std, again.tension_std)
    assert not np.array_equal(first.tension_std, other.tension_std)


# --------------------------------------------------------------------------------------
# 5. The user-facing diagnostics
# --------------------------------------------------------------------------------------


def test_diagnostics_reproduce_the_harness_numbers(golden_mesh, golden_truth):
    """`compute_inference_diagnostics` is the harness's `mesh_diagnostics`, promoted to public.

    The point of the public function is that it can run without ground truth, so this checks
    both paths: `rho` is `None` when no tensions are given, and matches the harness when they
    are.
    """
    from foambryo_benchmarks import metrics as harness

    with_truth = compute_inference_diagnostics(golden_mesh, golden_truth)
    without = compute_inference_diagnostics(golden_mesh)
    assert without.rho is None
    assert without.rho_free == pytest.approx(with_truth.rho_free, rel=1e-12)

    reference = harness.mesh_diagnostics(golden_mesh.v, golden_mesh.f[:, :3], golden_mesh.f[:, 3:5], golden_truth)
    assert with_truth.rho == pytest.approx(reference["rho"], rel=1e-12)
    assert with_truth.sigma_soft == pytest.approx(reference["sigma_soft"], rel=1e-12)
    assert with_truth.n_interfaces == reference["n_interfaces"]
    assert with_truth.n_cells == reference["n_cells"]


def test_identifiability_report_is_printable_and_flags_a_bad_mesh(golden_mesh, golden_truth):
    text = compute_inference_diagnostics(golden_mesh, golden_truth).identifiability_report()
    assert "rho_free" in text
    assert "sigma_soft" in text
    assert "soft mode" in text
    assert "WARNING" not in text  # the golden mesh is one of the 42

    non_equilibrium = DATASET_DIR / "005_mesh.rec"
    if not non_equilibrium.exists():
        pytest.skip(f"{non_equilibrium} not found (sibling benchmarking-dataset/ missing)")
    bad = dcel_mesh_from_file(str(non_equilibrium))
    assert "WARNING" in compute_inference_diagnostics(bad).identifiability_report()


def test_error_over_std_gauge_matches_the_interval_too(golden_mesh, golden_truth):
    """Rescaling the tensions without rescaling the interval would be a units error.

    Checked by construction: multiplying the truth by a constant must leave `error/std`
    unchanged, since both error and interval scale with it.
    """
    result = infer_tensions_with_uncertainty(golden_mesh, 1.0, method=TensionComputationMethod.ForceBalance)
    base = error_over_std(result, golden_truth)
    scaled = error_over_std(result, {key: 3.0 * value for key, value in golden_truth.items()})
    assert scaled["median"] == pytest.approx(base["median"], rel=1e-12)
    assert scaled["gauge_matching_scale"] == pytest.approx(3.0 * base["gauge_matching_scale"], rel=1e-12)
