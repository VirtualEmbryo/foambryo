"""Tests for `foambryo_benchmarks.attribution_metrics` (the attribution work).

Every item of the project's non-negotiable testing discipline gets its own test here: a
positive control with a hand-computed expected magnitude (not a magic number),
a non-vacuity check on the "how many cases the metric was present for" counters, and a
corruption-detection control for `rho_free` mirroring the dataset-audit work's own use of it.
"""

import numpy as np
import pytest

from foambryo_benchmarks import attribution_metrics as am
from foambryo_benchmarks import metrics as m
from foambryo.validation import mesh_io
from tests.conftest import require_substitute_mesh


def test_attribution_errors_perfect_p1_gives_zero_inference_error():
    """P1 == ground truth exactly => `e_inference` must be exactly 0, gauge scale exactly 1."""
    truth = {(1, 2): 1.0, (1, 3): 1.0, (2, 3): 1.0}
    result = am.attribution_errors(dict(truth), dict(truth), truth)
    assert result["e_inference"] == pytest.approx(0.0, abs=1e-12)
    assert result["gauge_matching_scale_p1"] == pytest.approx(1.0)


def test_attribution_errors_detects_an_injected_error_at_the_predicted_magnitude():
    """A hand-computed positive control: a known 20% error on one of three interfaces.

    `truth = {(1,2): 1, (1,3): 1, (2,3): 1}` (mean 1). `p2` injects +20% on `(1,2)` only:
    `{(1,2): 1.2, (1,3): 1, (2,3): 1}`, mean `3.2/3`. `gauge_matching_scale` (the gauge-matching
    rule every accuracy comparison follows) is `mean(truth)/mean(p2) = 1 / (3.2/3) = 15/16 = 0.9375`. After matching:
    `(1,2) -> 1.2 * 0.9375 = 1.125` (rel. error 0.125), `(1,3), (2,3) -> 0.9375` each (rel.
    error 0.0625 each). The median of `[0.125, 0.0625, 0.0625]` is **0.0625** - an
    independently derived number, not one read off the code under test.
    """
    truth = {(1, 2): 1.0, (1, 3): 1.0, (2, 3): 1.0}
    p1 = dict(truth)
    p2 = {(1, 2): 1.2, (1, 3): 1.0, (2, 3): 1.0}

    result = am.attribution_errors(p1, p2, truth)

    assert result["gauge_matching_scale_p2"] == pytest.approx(15 / 16)
    assert result["e_total"] == pytest.approx(0.0625, abs=1e-12)
    assert result["e_inference"] == pytest.approx(0.0, abs=1e-12)
    assert result["e_meshing"] == pytest.approx(0.0625, abs=1e-12)


def test_attribution_errors_scales_with_the_size_of_the_injected_error():
    """Doubling the injected error should roughly double the detected `e_total`.

    Not exact (the median and the gauge factor are both nonlinear in the perturbation),
    but the ordering and the rough magnitude are exactly what a working detector must show;
    a detector that returns the same number regardless of injected size, or a smaller
    number for a larger injection, is broken.
    """
    truth = {(1, 2): 1.0, (1, 3): 1.0, (2, 3): 1.0}
    small = am.attribution_errors(dict(truth), {(1, 2): 1.1, (1, 3): 1.0, (2, 3): 1.0}, truth)
    large = am.attribution_errors(dict(truth), {(1, 2): 1.4, (1, 3): 1.0, (2, 3): 1.0}, truth)
    assert 0 < small["e_total"] < large["e_total"]


def test_attribution_errors_reports_non_vacuous_presence_counts():
    """Guards against the offset-exclusion work's `None == None` failure mode: presence must be counted."""
    truth = {(1, 2): 1.0, (1, 3): 1.0, (2, 3): 1.0}
    p1 = dict(truth)
    p2 = {(1, 2): 1.2, (1, 3): 1.0}  # p2 is missing interface (2, 3) - a real dw3d outcome
    result = am.attribution_errors(p1, p2, truth)
    assert result["n_common_interfaces"] == {"p1": 3, "p2": 2}
    assert result["e_total_detail"]["gauge_matched"]["n"] == 2


def test_attribution_errors_raises_rather_than_silently_measuring_nothing():
    """Mismatched label alphabets (a harness bug, e.g. a stale mesh) must raise, not pass."""
    truth = {(1, 2): 1.0, (1, 3): 1.0}
    p1 = dict(truth)
    p2 = {(9, 10): 1.0}  # disjoint label alphabet - simulates a wrong-mesh / path bug
    with pytest.raises(ValueError, match="zero common interfaces"):
        am.attribution_errors(p1, p2, truth)


@pytest.fixture
def ground_truth_mesh():
    v, f, regions = mesh_io.read_rec(require_substitute_mesh())
    return v, f, regions


def test_rho_free_is_small_on_a_near_equilibrium_ground_truth_mesh(ground_truth_mesh):
    """Sanity range from the dataset-audit work: `rho_free` in 0.0007-0.036 over the 42 equilibrium cases."""
    v, f, regions = ground_truth_mesh
    result = am.rho_free_of_mesh(v, f, regions)
    assert 0.0 <= result["rho_free"] < 0.05
    assert result["n_interfaces"] > 0


def test_rho_free_detects_an_injected_geometric_corruption():
    """Positive control, in the dataset-audit work's own style: inject a known defect, check it is seen.

    Displacing every vertex of one interface far along a single fixed direction turns a
    balanced surface into a sheared one that no set of tensions can balance well, so
    `rho_free` (computed with NO ground truth at all) must rise sharply. This is the
    injected-defect control the project's testing discipline requires, on the dataset-audit
    work's precedent: a metric that cannot be shown to detect a defect is not evidence of absence.
    """
    v, f, regions = mesh_io.read_rec(require_substitute_mesh())
    baseline = am.rho_free_of_mesh(v, f, regions)

    interface_ab = tuple(sorted(mesh_io.list_interfaces(regions))[0])
    on_interface = np.any(np.isin(regions, interface_ab), axis=1)
    vertex_ids = np.unique(f[on_interface])

    v_corrupted = v.copy()
    edge_length = mesh_io.mean_edge_length(v, f)
    v_corrupted[vertex_ids] += np.array([5.0 * edge_length, 0.0, 0.0])

    corrupted = am.rho_free_of_mesh(v_corrupted, f, regions)

    assert corrupted["rho_free"] > 10 * max(baseline["rho_free"], 1e-6)


# ---------------------------------------------------------------------------------------
# interface-area fractions -- the predicate the benchmarks-package rename listed as
# missing, added by the second-order-angle-reading work
# ---------------------------------------------------------------------------------------
def test_interface_area_fractions_sum_to_one_and_are_scale_invariant():
    """The whole point of the fraction is that a uniform rescale of the mesh cannot move it."""
    v = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])
    f = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 4]])
    regions = np.array([[0, 1], [0, 2], [0, 1]])

    fractions = m.interface_area_fractions(v, f, regions)
    assert set(fractions) == {(0, 1), (0, 2)}
    assert sum(fractions.values()) == pytest.approx(1.0)

    scaled = m.interface_area_fractions(v * 7.5, f, regions)
    for key, value in fractions.items():
        assert scaled[key] == pytest.approx(value)


def test_interface_area_fraction_errors_count_identity_defects_separately():
    """An interface one mesh has and the other does not is a count, never an averaged error."""
    candidate = {(0, 1): 0.5, (0, 2): 0.3, (1, 2): 0.2}
    reference = {(0, 1): 0.4, (0, 2): 0.3, (0, 3): 0.3}
    result = m.interface_area_fraction_errors(candidate, reference)

    assert result["n_matched"] == 2
    assert result["n_invented"] == 1  # (1, 2) is in the candidate only
    assert result["n_missed"] == 1  # (0, 3) is in the reference only
    assert result["per_interface"] == {(0, 1): pytest.approx(0.1), (0, 2): pytest.approx(0.0)}
    assert result["max_abs_error"] == pytest.approx(0.1)


def test_closer_interface_area_fractions_reports_ties_rather_than_splitting_them():
    """Two readings of one mesh give identical areas; the predicate must say `tied`, not pick."""
    reference = {(0, 1): 0.5, (0, 2): 0.5}
    identical = m.closer_interface_area_fractions(reference, reference, reference)
    assert identical == {
        "n_matched": 2,
        "n_first_closer": 0,
        "n_second_closer": 0,
        "n_tied": 2,
        "first_mean_abs_error": 0.0,
        "second_mean_abs_error": 0.0,
    }

    first = {(0, 1): 0.51, (0, 2): 0.49}
    second = {(0, 1): 0.60, (0, 2): 0.40}
    verdict = m.closer_interface_area_fractions(first, second, reference)
    assert verdict["n_first_closer"] == 2
    assert verdict["n_second_closer"] == 0
    assert verdict["n_tied"] == 0


def test_median_of_determined_raises_on_an_undetermined_value_without_the_flag():
    """One `nan` among otherwise-clean values must refuse, not silently propagate or drop it."""
    with pytest.raises(ValueError, match="undetermined"):
        m.median_of_determined([1.0, 2.0, float("nan"), 3.0])


def test_median_of_determined_with_allow_partial_returns_the_determined_median_and_counts():
    """`allow_partial=True` computes the median over the finite values only, with counts attached."""
    result = m.median_of_determined([1.0, 2.0, float("nan"), 3.0], allow_partial=True)
    assert result == {"median": pytest.approx(2.0), "n_determined": 3, "n_total": 4}


def test_median_of_determined_all_finite_needs_no_flag_and_reports_full_counts():
    """No undetermined value at all: behaves like a plain median, flag or not."""
    result = m.median_of_determined([1.0, 2.0, 3.0])
    assert result == {"median": pytest.approx(2.0), "n_determined": 3, "n_total": 3}


def test_median_of_determined_all_undetermined_returns_none_not_nan():
    """Zero determined values: `median` is `None` (JSON `null`), never a propagated `nan`."""
    result = m.median_of_determined([float("nan"), float("nan")], allow_partial=True)
    assert result == {"median": None, "n_determined": 0, "n_total": 2}
