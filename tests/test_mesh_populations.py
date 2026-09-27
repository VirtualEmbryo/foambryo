"""Tests for `foambryo_benchmarks.mesh_populations` (the population-split work - the P1/P2/P3 split)."""

import pytest

from foambryo_benchmarks import mesh_populations as pop


def test_require_single_population_returns_the_shared_population():
    records = [
        {"case": "000", "population": pop.P1_SIMULATOR_GROUND_TRUTH},
        {"case": "001", "population": pop.P1_SIMULATOR_GROUND_TRUTH},
    ]
    assert pop.require_single_population(records) == pop.P1_SIMULATOR_GROUND_TRUTH


def test_require_single_population_raises_on_mixed_p1_p2():
    records = [
        {"case": "000", "population": pop.P1_SIMULATOR_GROUND_TRUTH},
        {"case": "000", "population": pop.P2_DW3D_RECONSTRUCTION},
    ]
    with pytest.raises(ValueError, match="refusing to aggregate"):
        pop.require_single_population(records)


def test_require_single_population_raises_on_all_three_mixed():
    records = [
        {"population": pop.P1_SIMULATOR_GROUND_TRUTH},
        {"population": pop.P2_DW3D_RECONSTRUCTION},
        {"population": pop.P3_ANALYTIC_PHANTOM},
    ]
    with pytest.raises(ValueError, match="refusing to aggregate"):
        pop.require_single_population(records)


def test_require_single_population_raises_on_empty_list():
    with pytest.raises(ValueError, match="empty record list"):
        pop.require_single_population([])


def test_require_single_population_raises_on_unknown_tag():
    """A typo in the population tag must not silently create a fourth population."""
    records = [{"population": "P2_dw3d_reconstruciton"}]  # deliberate typo
    with pytest.raises(ValueError, match="unrecognised population"):
        pop.require_single_population(records)


def test_require_single_population_respects_custom_key():
    records = [{"origin": pop.P3_ANALYTIC_PHANTOM}, {"origin": pop.P3_ANALYTIC_PHANTOM}]
    assert pop.require_single_population(records, population_key="origin") == pop.P3_ANALYTIC_PHANTOM


def test_tag_rejects_unknown_population():
    with pytest.raises(ValueError, match="unrecognised population"):
        pop.tag({"case": "000"}, "not_a_real_population")


def test_tag_does_not_mutate_the_input():
    record = {"case": "000"}
    tagged = pop.tag(record, pop.P1_SIMULATOR_GROUND_TRUTH)
    assert "population" not in record
    assert tagged["population"] == pop.P1_SIMULATOR_GROUND_TRUTH
    assert tagged["case"] == "000"
