"""Tests for `foambryo.force_balance_inference` — the force-balance work's `ForceBalance` method.

What is pinned here, and why each test has teeth:

1. **The operator is the same whether it is assembled from `DcelData` or from the raw
   `.rec` arrays** (`test_operator_agrees_between_dcel_and_raw_rec`). `DcelData` relabels
   and re-winds triangles; the pressure signs depend on that winding, so this is the check
   that the new call site inherits the already-validated ported operator rather than a
   subtly different one.
2. **Closed-form pressure marginalisation reproduces the joint solve**
   (`test_marginalised_and_joint_solves_agree`). Eliminating the pressures in closed form is
   exact block elimination of a quadratic form (a Schur complement on the pressure block), so
   the tensions must equal those of the joint solve; this test makes it a standing check.
3. **The KKT gauge is exact and the soft row's is not** (`test_kkt_mean_tension_is_exact`,
   `test_soft_row_gauge_is_only_approximate`). This is the force-balance work's acceptance criterion
   (`mean(gamma) = 1` to `1e-12`) and the honest measurement of how much the soft row
   actually misses it by — which after the defect-cleanup work is *small*, so the test asserts the
   structural fact (KKT is exact, soft row is not bit-exact) rather than dramatising a
   defect that no longer exists.
4. **Pressures come out with the physical sign** (`test_inferred_pressures_match_the_truth`),
   with the opposite sign shown to be far worse, so the assertion cannot pass on a mesh
   where the sign is irrelevant.
5. **`ForceBalance` is the dataset-audit work's effective-tension diagnostic promoted to a method**
   (`test_force_balance_reproduces_effective_surface_tensions`). If those two ever diverge,
   one of them has changed convention.
6. **Structural invariants**: gauge homogeneity, the residual at the inferred tensions being
   a floor below `rho` at the nominal ones, and zero posterior variance along the fixed
   gauge direction.

**Gauge (the rule every accuracy comparison in this package follows).** Every error compared
against a ground truth here is gauge-matched first, except where the point of the test is the
gauge itself.
"""

import numpy as np
import pytest

from foambryo_benchmarks import phantoms as ph
from foambryo import (
    ForceBalanceNormalisation,
    TensionComputationMethod,
    dcel_mesh_from_file,
    infer_force_balance,
    infer_tensions,
)
from foambryo.dcel import DcelData
from foambryo.force_balance_inference import build_force_balance_operator
from foambryo.validation import force_balance as fb
from foambryo.validation import mesh_io
from tests.conftest import DATASET_DIR, require_substitute_mesh

# A spread of equilibrium cases (all five non-equilibrium cases are excluded, by the recorded
# equilibrium-split decision): the golden master, the smallest and the largest, and the two the
# per-method baseline table flags as worst for some method. Kept short so the file stays fast;
# the dataset-wide numbers are the benchmark harness's job, not pytest's.
EQUILIBRIUM_CASE_IDS = ("000", "008", "031", "044")


def _load_case(case_id: str) -> tuple[DcelData, dict, dict]:
    path = DATASET_DIR / f"{case_id}_mesh.rec"
    if not path.exists():
        pytest.skip(f"{path} not found (sibling benchmarking-dataset/ missing)")
    tensions = np.load(DATASET_DIR / f"{case_id}_dict_tensions.npy", allow_pickle=True).item()
    pressures = np.load(DATASET_DIR / f"{case_id}_dict_pressures.npy", allow_pickle=True).item()
    return dcel_mesh_from_file(str(path)), tensions, pressures


def _doublet(ratio: float = 1.0, n_rings: int = 24) -> DcelData:
    v, f, regions = ph.doublet_mesh(ratio, n_rings=n_rings, n_phi=2 * n_rings)
    return DcelData(v, f, regions)


def test_operator_agrees_between_dcel_and_raw_rec():
    path = require_substitute_mesh()
    v, f, regions = mesh_io.read_rec(path)
    interfaces = mesh_io.list_interfaces(regions)
    cells = mesh_io.cell_labels(regions)
    from_raw = fb.force_balance_operator(v, f, regions, interfaces, [], {}, cells)

    mesh = dcel_mesh_from_file(str(path))
    from_dcel, dcel_interfaces, dcel_cells = build_force_balance_operator(mesh)

    assert dcel_interfaces == interfaces
    assert dcel_cells == cells
    # Not bit-identical: `DcelData` reverses the winding of every triangle whose stored
    # label order it normalises, so the cross products are evaluated in a different order.
    # The operator is invariant in exact arithmetic, hence the tight absolute tolerance
    # against an operator whose entries are O(1e-4)-O(1e-2) here.
    assert np.allclose(from_raw, from_dcel, atol=1e-15, rtol=0)


@pytest.mark.parametrize("case_id", EQUILIBRIUM_CASE_IDS)
def test_marginalised_and_joint_solves_agree(case_id):
    """Eliminating the pressures in closed form changes neither the tensions nor the pressures."""
    mesh, _truth_tensions, _truth_pressures = _load_case(case_id)
    m, interfaces, _cells = build_force_balance_operator(mesh)
    ni = len(interfaces)

    joint = infer_force_balance(mesh, 1.0, ForceBalanceNormalisation.Kkt)

    reduced, pressure_map = fb.marginalise_pressures(m, ni)
    anchor = np.full(ni, 1.0 / ni)
    gamma_marginal = fb.solve_constrained(reduced, anchor, np.array([1.0]))
    pressure_marginal = -(pressure_map @ gamma_marginal)

    assert np.allclose(gamma_marginal, joint.tensions, rtol=1e-10, atol=1e-12)
    assert np.allclose(pressure_marginal, joint.pressures, rtol=1e-10, atol=1e-12)


def test_marginalisation_projector_annihilates_the_pressure_block():
    """`P_perp M_p = 0`: the reduced operator really is orthogonal to the eliminated block."""
    mesh, _t, _p = _load_case("000")
    m, interfaces, cells = build_force_balance_operator(mesh)
    ni = len(interfaces)
    m_p = m[:, ni:]
    q, _r = np.linalg.qr(m_p)
    projected = m_p - q @ (q.T @ m_p)
    assert np.linalg.norm(projected) < 1e-12 * np.linalg.norm(m_p)
    reduced, pressure_map = fb.marginalise_pressures(m, ni)
    assert reduced.shape == (m.shape[0], ni)
    assert pressure_map.shape == (len(cells), ni)


@pytest.mark.parametrize("case_id", EQUILIBRIUM_CASE_IDS)
@pytest.mark.parametrize("mean_tension", [1.0, 2.5])
def test_kkt_mean_tension_is_exact(case_id, mean_tension):
    """The force-balance work's acceptance criterion: `mean(gamma) = mean_tension` to 1e-12, by construction."""
    mesh, _t, _p = _load_case(case_id)
    result = infer_force_balance(mesh, mean_tension, ForceBalanceNormalisation.Kkt)
    assert abs(float(np.mean(result.tensions)) - mean_tension) < 1e-12 * mean_tension
    assert result.mean_tension_deviation < 1e-12


def test_soft_row_gauge_is_only_approximate():
    """The soft row misses the gauge — by little, but not by zero, and not by construction.

    After the defect-cleanup work's operator-precedence fix the soft row is accurate to ~1e-7
    here, so this test asserts the *structural* difference (KKT is exact to the last bit, the
    soft row is not) rather than a large error. That difference is the whole case for KKT after
    the defect-cleanup work.
    """
    mesh, _t, _p = _load_case("000")
    kkt = infer_force_balance(mesh, 1.0, ForceBalanceNormalisation.Kkt)
    soft = infer_force_balance(mesh, 1.0, ForceBalanceNormalisation.SoftRow)
    assert kkt.mean_tension_deviation < 1e-14
    assert 0 < soft.mean_tension_deviation < 1e-4
    assert soft.mean_tension_deviation > kkt.mean_tension_deviation


def test_soft_row_and_kkt_agree_on_the_tension_ratios():
    """The two gauges give the *same physics*: their tensions agree once gauge-matched.

    This is the measurement behind "adopt KKT for exactness, not for accuracy" — if the two
    disagreed materially, the choice would be an accuracy question instead.
    """
    mesh, _t, _p = _load_case("000")
    kkt = infer_force_balance(mesh, 1.0, ForceBalanceNormalisation.Kkt)
    soft = infer_force_balance(mesh, 1.0, ForceBalanceNormalisation.SoftRow)
    scale = float(np.mean(kkt.tensions) / np.mean(soft.tensions))
    assert np.allclose(scale * soft.tensions, kkt.tensions, rtol=1e-5)


@pytest.mark.parametrize("case_id", EQUILIBRIUM_CASE_IDS)
def test_inferred_pressures_match_the_truth(case_id):
    """Pressures are returned on foambryo's physical convention, and the sign is pinned."""
    mesh, truth_tensions, truth_pressures = _load_case(case_id)
    result = infer_force_balance(mesh, 1.0, ForceBalanceNormalisation.Kkt)

    gamma_truth = np.array([truth_tensions[key] for key in result.interfaces])
    scale = float(np.mean(gamma_truth) / np.mean(result.tensions))
    pressure_truth = np.array([truth_pressures[cell] for cell in result.cells])

    error = np.abs(scale * result.pressures - pressure_truth) / np.abs(pressure_truth)
    flipped = np.abs(-scale * result.pressures - pressure_truth) / np.abs(pressure_truth)
    assert np.median(error) < 0.1
    # The teeth: the wrong sign must be visibly wrong, so this is not a vacuous check on a
    # mesh whose pressures happen to be near zero.
    assert np.median(flipped) > 10 * np.median(error)
    assert result.dict_pressures[0] == 0.0


@pytest.mark.parametrize("case_id", EQUILIBRIUM_CASE_IDS)
def test_force_balance_reproduces_effective_surface_tensions(case_id):
    """`ForceBalance`+KKT *is* the dataset-audit work's `gamma_eff` diagnostic, promoted to a method.

    Same operator, same single mean-tension anchor, same KKT solve — so the two must agree
    exactly. Recording it as a test makes the connection non-accidental: the dataset audit's finding that
    `gamma_eff` deviates from nominal by ~0.023 absolute on this dataset is *already* a
    prediction of `ForceBalance`'s accuracy, and the two cannot drift apart silently.
    """
    mesh, _t, _p = _load_case(case_id)
    m, interfaces, _cells = build_force_balance_operator(mesh)
    gamma_eff, _p_eff, _residual = fb.effective_surface_tensions(m, len(interfaces), 0, mean_tension=1.0)
    result = infer_force_balance(mesh, 1.0, ForceBalanceNormalisation.Kkt)
    assert np.allclose(result.tensions, gamma_eff, rtol=1e-10, atol=1e-12)


def test_tensions_are_homogeneous_in_the_gauge():
    """Doubling `mean_tension` doubles every tension and every pressure, exactly."""
    mesh, _t, _p = _load_case("000")
    one = infer_force_balance(mesh, 1.0, ForceBalanceNormalisation.Kkt)
    two = infer_force_balance(mesh, 2.0, ForceBalanceNormalisation.Kkt)
    assert np.allclose(two.tensions, 2 * one.tensions, rtol=1e-12)
    assert np.allclose(two.pressures, 2 * one.pressures, rtol=1e-12)
    # rho is a ratio of two quantities linear in beta, so it is gauge-invariant.
    assert abs(two.rho - one.rho) < 1e-12


@pytest.mark.parametrize("case_id", EQUILIBRIUM_CASE_IDS)
def test_inferred_tensions_leave_less_force_unbalanced_than_the_nominal_ones(case_id):
    """`ForceBalance` minimises exactly the residual `rho` measures, so it cannot do worse.

    This is the sense in which `ForceBalance` is the *best possible* method under this
    operator — and, read the other way, the reason its error on a mesh that is not at this
    operator's discrete equilibrium is a floor set by the data, not by the solver.
    """
    mesh, truth_tensions, _p = _load_case(case_id)
    m, interfaces, _cells = build_force_balance_operator(mesh)
    gamma_nominal = np.array([truth_tensions[key] for key in interfaces])
    rho_nominal = fb.residual_at(m, len(interfaces), 0, gamma_nominal, np.zeros(0))
    result = infer_force_balance(mesh, 1.0, ForceBalanceNormalisation.Kkt)
    assert result.rho <= rho_nominal


def test_posterior_variance_vanishes_along_the_fixed_gauge():
    """The KKT error bars carry zero variance in the direction that was constrained."""
    mesh, _t, _p = _load_case("000")
    m, interfaces, _cells = build_force_balance_operator(mesh)
    ni = len(interfaces)
    anchor = np.zeros(m.shape[1])
    anchor[:ni] = 1.0 / ni
    beta = fb.solve_constrained(m, anchor, np.array([1.0]))
    sigma_f = fb.residual_noise_scale(m, beta)
    cov = fb.constrained_covariance(m, anchor, sigma_f)
    variance_along_gauge = float(anchor @ cov @ anchor)
    assert abs(variance_along_gauge) < 1e-12 * float(np.trace(cov))
    assert np.all(np.diag(cov)[:ni] >= -1e-18)


@pytest.mark.parametrize("normalisation", list(ForceBalanceNormalisation), ids=lambda n: n.name)
def test_every_normalisation_produces_finite_error_bars(normalisation):
    mesh, _t, _p = _load_case("000")
    result = infer_force_balance(mesh, 1.0, normalisation)
    assert result.tension_std.shape == result.tensions.shape
    assert np.all(np.isfinite(result.tension_std))
    assert np.all(result.tension_std >= 0)
    assert np.all(np.isfinite(result.tensions))
    assert result.normalisation is normalisation
    assert set(result.dict_tension_std) == set(result.interfaces)


@pytest.mark.parametrize(
    "mode",
    [
        TensionComputationMethod.ForceBalance,
        TensionComputationMethod.ForceBalanceSoftRow,
        TensionComputationMethod.ForceBalanceMAP,
    ],
    ids=lambda mode: mode.name,
)
def test_enum_dispatch_matches_the_direct_call(mode):
    """The three new enum members are exactly the three normalisations, and nothing else."""
    from foambryo.tension_inference import _FORCE_BALANCE_NORMALISATIONS

    mesh, _t, _p = _load_case("000")
    direct = infer_force_balance(mesh, 1.0, _FORCE_BALANCE_NORMALISATIONS[mode])
    dispatched = infer_tensions(mesh, 1, mode)
    assert dispatched == pytest.approx(direct.dict_tensions)
    assert set(dispatched) == set(mesh.compute_areas_interfaces())


@pytest.mark.parametrize("case_id", EQUILIBRIUM_CASE_IDS)
def test_force_balance_agrees_with_variational(case_id):
    """`Variational` and `ForceBalance` are the same estimator, and this pins that.

    Both invert the discrete area-gradient equilibrium: `Variational` assembles the
    per-interface Gram matrix `F = D D^T` of the same area gradients that make up `M_gamma`,
    where `ForceBalance` keeps `M` itself. Measured over the 42 equilibrium cases, their
    tensions agree to a median-of-medians of **1.5e-05** per interface while *both* differ
    from the ground truth by **2.1e-02** — i.e. the difference between them is three orders of
    magnitude smaller than the difference between either and the truth.

    That is the fact that reframes the dataset-audit work's headline: `ForceBalance` does not add a new
    physical model, it re-derives `Variational` in a form with an exact gauge, a sparse-ready
    operator and error bars. So the `Variational`-vs-`YoungDupre` inversion is a statement
    about *which balance a mesh satisfies*, not about one method's implementation.

    The tolerance here is loose (1e-3 median) because the two paths differ in weighting and
    in float32-vs-float64; the point is the order of magnitude,
    not bit-agreement.
    """
    mesh, _t, _p = _load_case(case_id)
    force_balance = infer_tensions(mesh, 1, TensionComputationMethod.ForceBalance)
    variational = infer_tensions(mesh, 1, TensionComputationMethod.Variational)
    keys = sorted(force_balance)
    a = np.array([force_balance[key] for key in keys])
    b = np.array([variational[key] for key in keys])
    assert np.median(np.abs(a - b) / np.abs(b)) < 1e-3


def test_force_balance_recovers_the_analytic_doublet():
    """Absolute accuracy on a mesh that *is* an exact discrete-equilibrium test case.

    The doublet is an analytic equilibrium (no voxelisation, no seed noise), so this is an
    absolute check rather than a comparison against another method. `1e-3` is a loose bound
    around the ~1e-5 measured at `n_rings = 24`
    (`foambryo_benchmarks/baseline/doublet_convergence.md`); the sweep, not this test, is where the
    convergence order is measured.
    """
    for ratio in (0.5, 1.0, 1.5):
        mesh = _doublet(ratio, n_rings=24)
        truth = ph.doublet_ground_truth_tensions(ratio, mean_tension=1.0)
        result = infer_force_balance(mesh, 1.0, ForceBalanceNormalisation.Kkt)
        gamma_truth = np.array([truth[key] for key in result.interfaces])
        scale = float(np.mean(gamma_truth) / np.mean(result.tensions))
        error = np.abs(scale * result.tensions - gamma_truth) / np.abs(gamma_truth)
        assert error.max() < 1e-3, f"ratio={ratio}: {error}"


def _rescaled(case_id: str, scale: float) -> DcelData:
    path = DATASET_DIR / f"{case_id}_mesh.rec"
    if not path.exists():
        pytest.skip(f"{path} not found (sibling benchmarking-dataset/ missing)")
    v, f, regions = mesh_io.read_rec(path)
    return DcelData(np.asarray(v, dtype=np.float64) * scale, f, regions)


@pytest.mark.parametrize("case_id", ["000", "001", "016", "046"])
def test_map_does_not_depend_on_the_length_unit(case_id):
    """`BayesianMap` gives the same relative tensions, and pressures times the scale, in any length unit.

    Multiplying every vertex by `l` scales the surface-tension columns of the operator by `l` and
    the pressure columns by `l**2`; `sigma_f` and the pressure prior follow (`_solve_bayesian_map`),
    so the negative log-posterior is unchanged and so is the estimate, up to rounding. Before that,
    the same rescaling moved the relative tensions by up to 1.6e-3 on these meshes and by up to
    43 % on reconstructions in voxel units. Checked at `l = 10` and `l = 0.1` (not powers of two,
    so the arithmetic is not exact) to 1e-12 of the largest value.
    """
    base = infer_force_balance(_rescaled(case_id, 1.0), 1.0, ForceBalanceNormalisation.BayesianMap)
    t1 = base.tensions / base.tensions.mean()
    p1 = base.pressures / base.tensions.mean()
    for scale in (10.0, 0.1):
        here = infer_force_balance(_rescaled(case_id, scale), 1.0, ForceBalanceNormalisation.BayesianMap)
        assert here.interfaces == base.interfaces
        assert here.cells == base.cells
        ts = here.tensions / here.tensions.mean()
        ps = scale * here.pressures / here.tensions.mean()
        assert np.max(np.abs(ts - t1)) <= 1e-12 * np.max(np.abs(t1)), (case_id, scale)
        assert np.max(np.abs(ps - p1)) <= 1e-12 * np.max(np.abs(p1)), (case_id, scale)
        assert here.sigma_f == pytest.approx(scale * base.sigma_f, rel=1e-12)


@pytest.mark.parametrize("case_id", ["000", "001", "016", "046"])
def test_map_does_not_depend_on_the_gauge_value(case_id):
    """`BayesianMap` gives the same relative tensions and pressures whatever `mean_tension` is.

    The prior's mean and widths are proportional to `mean_tension`, and so is `sigma_f`
    (`_solve_bayesian_map`), so with `beta = mean_tension * beta1` the negative log-posterior does
    not change and neither do `gamma / mean(gamma)` and `p / mean(gamma)`. Before `sigma_f` followed
    the gauge value, `mean_tension = 0.5` moved the relative tensions by up to 1.7e-2 on these
    meshes. Checked at 0.5, 2 and 10 against 1, to 1e-12 of the largest value.
    """
    mesh = _rescaled(case_id, 1.0)
    base = infer_force_balance(mesh, 1.0, ForceBalanceNormalisation.BayesianMap)
    t1 = base.tensions / base.tensions.mean()
    p1 = base.pressures / base.tensions.mean()
    for mean_tension in (0.5, 2.0, 10.0):
        here = infer_force_balance(mesh, mean_tension, ForceBalanceNormalisation.BayesianMap)
        ts = here.tensions / here.tensions.mean()
        ps = here.pressures / here.tensions.mean()
        assert np.max(np.abs(ts - t1)) <= 1e-12 * np.max(np.abs(t1)), (case_id, mean_tension)
        assert np.max(np.abs(ps - p1)) <= 1e-12 * np.max(np.abs(p1)), (case_id, mean_tension)
