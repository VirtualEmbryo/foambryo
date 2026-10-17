"""Tests for `foambryo_benchmarks.analyze_robustness`'s aggregation logic (the robustness work).

Synthetic records only - the underlying metrics (`attribution_metrics`, `segmentation_
perturbation`, `junction_instability`) already carry their own positive controls; this
file's job is to catch a vacuous-aggregation bug in the *aggregation* itself (the class of
defect where an aggregation reports "0 differ" while comparing nothing).
"""

import pytest

from foambryo_benchmarks import analyze_robustness as analyze
from foambryo_benchmarks import mesh_populations as pop


def _record(case, condition, variant, method, e_total, e_inference=1e-4, rho_free_p2=0.5, instability=None):
    record = {
        "case": case,
        "condition": condition,
        "dw3d_variant": variant,
        "method": method,
        "e_total": e_total,
        "e_inference": e_inference,
        "e_meshing": e_total - e_inference,
        "rho_free_p2": rho_free_p2,
    }
    if instability is not None:
        record["instability"] = instability
    return pop.tag(record, pop.P2_DW3D_RECONSTRUCTION)


def test_degradation_by_condition_computes_the_ratio_to_its_own_baseline():
    records = [
        _record("000", "baseline", "v1", "YoungDupre", e_total=0.10),
        _record("001", "baseline", "v1", "YoungDupre", e_total=0.10),
        _record("000", "erode1", "v1", "YoungDupre", e_total=0.20),
        _record("001", "erode1", "v1", "YoungDupre", e_total=0.30),
    ]
    summary = analyze.degradation_by_condition(records)
    cell = summary["equilibrium"]["v1"]["YoungDupre"]
    assert cell["baseline"]["e_total_median"] == pytest.approx(0.10)
    assert cell["erode1"]["e_total_median"] == pytest.approx(0.25)
    assert cell["erode1"]["degradation_vs_baseline"] == pytest.approx(2.5)
    assert cell["baseline"]["n_present"] == 2


def test_degradation_by_condition_does_not_mix_two_methods_or_two_variants():
    records = [
        _record("000", "baseline", "v1", "YoungDupre", e_total=0.10),
        _record("000", "baseline", "v1", "ForceBalanceKKT", e_total=0.90),
        _record("000", "baseline", "v2", "YoungDupre", e_total=0.50),
    ]
    summary = analyze.degradation_by_condition(records)
    grid = summary["equilibrium"]
    assert grid["v1"]["YoungDupre"]["baseline"]["e_total_median"] == pytest.approx(0.10)
    assert grid["v1"]["ForceBalanceKKT"]["baseline"]["e_total_median"] == pytest.approx(0.90)
    assert grid["v2"]["YoungDupre"]["baseline"]["e_total_median"] == pytest.approx(0.50)


def test_method_ranking_by_condition_detects_a_ranking_swap():
    records = [
        # baseline: YoungDupre best, ForceBalanceKKT worst
        _record("000", "baseline", "v1", "YoungDupre", e_total=0.10),
        _record("000", "baseline", "v1", "ForceBalanceKKT", e_total=0.90),
        # erode1: ranking flips
        _record("000", "erode1", "v1", "YoungDupre", e_total=0.90),
        _record("000", "erode1", "v1", "ForceBalanceKKT", e_total=0.10),
    ]
    result = analyze.method_ranking_by_condition(records)
    cell = result["equilibrium"]["v1"]
    assert cell["baseline"]["ranking"] == ["YoungDupre", "ForceBalanceKKT"]
    assert cell["baseline"]["matches_baseline_order"] is True
    assert cell["erode1"]["ranking"] == ["ForceBalanceKKT", "YoungDupre"]
    assert cell["erode1"]["matches_baseline_order"] is False


def test_method_ranking_by_condition_flags_a_preserved_ranking_too():
    records = [
        _record("000", "baseline", "v1", "YoungDupre", e_total=0.10),
        _record("000", "baseline", "v1", "ForceBalanceKKT", e_total=0.90),
        _record("000", "erode1", "v1", "YoungDupre", e_total=0.20),
        _record("000", "erode1", "v1", "ForceBalanceKKT", e_total=0.95),
    ]
    result = analyze.method_ranking_by_condition(records)
    assert result["equilibrium"]["v1"]["erode1"]["matches_baseline_order"] is True


def test_instability_spatial_summary_pools_junction_counts_not_pre_averaged_fractions():
    """Two cases with very different junction counts must be pooled by junction, not by case.

    A case with 100 junctions must outweigh a case with 1, unlike a naive mean of the two
    cases' `fraction_unstable` values.
    """
    spatial_a = [{"bin": 0, "n": 100, "n_unstable": 50, "median_length": 1.0}]  # 50%
    spatial_b = [{"bin": 0, "n": 1, "n_unstable": 0, "median_length": 1.0}]  # 0%
    records = [
        _record(
            "000", "baseline", "v1", "YoungDupre", e_total=0.1,
            instability={"spatial": spatial_a},
        ),
        _record(
            "001", "baseline", "v1", "YoungDupre", e_total=0.1,
            instability={"spatial": spatial_b},
        ),
    ]
    result = analyze.instability_spatial_summary(records)
    bin0 = result["equilibrium"]["v1"]["baseline"]["YoungDupre"]["bins"][0]
    assert bin0["n"] == 101
    assert bin0["n_unstable"] == 50
    assert bin0["fraction_unstable"] == pytest.approx(50 / 101)  # not (0.5 + 0.0) / 2 = 0.25


def test_degradation_by_condition_raises_on_mixed_equilibrium_groups_without_a_split():
    records = [
        _record("000", "baseline", "v1", "YoungDupre", e_total=0.1),  # equilibrium
        _record("005", "baseline", "v1", "YoungDupre", e_total=0.1),  # non-equilibrium
    ]
    # require_single_population only guards P1/P2/P3 mixing; the equilibrium split itself
    # must still be reported separately in the output, never silently pooled.
    summary = analyze.degradation_by_condition(records)
    assert "equilibrium" in summary
    assert "non_equilibrium" in summary
    assert summary["equilibrium"]["v1"]["YoungDupre"]["baseline"]["n_present"] == 1
    assert summary["non_equilibrium"]["v1"]["YoungDupre"]["baseline"]["n_present"] == 1


def test_main_helpers_raise_on_an_empty_record_list():
    with pytest.raises(ValueError, match="cannot summarise an empty aggregate"):
        analyze._require_nonempty([], "test context")
