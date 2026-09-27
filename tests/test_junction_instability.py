"""Positive-control tests for `foambryo_benchmarks.junction_instability` (the robustness work).

Every metric gets an injected-defect-of-known-effect control, per the project's
non-negotiable testing discipline: a hand-computed instability
boundary for `is_unstable`, and a real mesh with one interface's tension driven far past the
triangle-inequality boundary for `junction_instability_report`, checked to flag exactly the
junctions it should and count (never zero-fill) the ones it cannot evaluate.
"""

import pytest

from foambryo_benchmarks import junction_instability as ji
from foambryo import dcel_mesh_from_file, geometry
from tests.conftest import require_substitute_mesh


def test_is_unstable_equal_tensions_is_stable():
    assert ji.is_unstable(1.0, 1.0, 1.0) is False


def test_is_unstable_at_the_exact_boundary_is_not_flagged():
    """`2 == 1 + 1` degenerates to a straight line but is not yet violated.

    The boundary itself is stable; only strictly exceeding it is unstable.
    """
    assert ji.is_unstable(2.0, 1.0, 1.0) is False


def test_is_unstable_just_past_the_boundary_is_flagged():
    assert ji.is_unstable(2.0001, 1.0, 1.0) is True


def test_is_unstable_is_symmetric_in_its_three_arguments():
    values = (5.0, 1.0, 1.0)
    results = {ji.is_unstable(*perm) for perm in [(5, 1, 1), (1, 5, 1), (1, 1, 5)]}
    assert results == {True}
    assert ji.is_unstable(*values) == ji.is_unstable(*values[::-1])


@pytest.fixture
def ground_truth_mesh():
    return dcel_mesh_from_file(require_substitute_mesh())


def test_junction_instability_report_all_equal_tensions_is_fully_stable(ground_truth_mesh):
    """Sanity baseline: every interface at tension 1 can never violate the inequality.

    `2 * 1 = 2`, which is not `> 1 + 1 = 2`, so the report must show 0 unstable.
    """
    interfaces = geometry.extract_edges_trijunctions(ground_truth_mesh)
    all_pairs = {pair for triple in interfaces for pair in _pairs_of(triple)}
    tensions = dict.fromkeys(all_pairs, 1.0)

    report = ji.junction_instability_report(ground_truth_mesh, tensions)

    assert report["n_present"] > 0
    assert report["n_unstable"] == 0
    assert report["fraction_unstable"] == 0.0


def test_junction_instability_report_detects_an_injected_instability_at_the_right_junctions(
    ground_truth_mesh,
):
    """Positive control: drive ONE interface's tension to 10x the others.

    Every triple line touching that interface must now be flagged (`10 > 1 + 1`); every
    triple line that does NOT touch it must remain stable. A targeted check, not "count > 0".
    """
    interfaces = geometry.extract_edges_trijunctions(ground_truth_mesh)
    all_pairs = {pair for triple in interfaces for pair in _pairs_of(triple)}
    hot_pair = sorted(all_pairs)[0]
    tensions = dict.fromkeys(all_pairs, 1.0)
    tensions[hot_pair] = 10.0

    report = ji.junction_instability_report(ground_truth_mesh, tensions)

    touching = [r for r in report["junctions"] if hot_pair in _pairs_of(r["triple"])]
    not_touching = [r for r in report["junctions"] if hot_pair not in _pairs_of(r["triple"])]
    assert len(touching) > 0
    assert all(r["unstable"] for r in touching)
    assert len(not_touching) > 0
    assert not any(r["unstable"] for r in not_touching)
    assert report["n_unstable"] == len(touching)


def test_junction_instability_report_counts_absent_interfaces_rather_than_zero_filling(
    ground_truth_mesh,
):
    """Removing one interface must show up as `n_absent > 0`, not silently scored stable.

    Every triple line touching the removed interface must be excluded from `"junctions"` -
    the offset-exclusion work's `None == None` failure mode, restated for this metric.
    """
    interfaces = geometry.extract_edges_trijunctions(ground_truth_mesh)
    all_pairs = {pair for triple in interfaces for pair in _pairs_of(triple)}
    missing_pair = sorted(all_pairs)[0]
    tensions = dict.fromkeys(all_pairs, 1.0)
    del tensions[missing_pair]

    report = ji.junction_instability_report(ground_truth_mesh, tensions)

    assert report["n_absent"] > 0
    assert report["n_present"] + report["n_absent"] + report["n_non_triple"] == report["n_candidate_triples"]
    assert all(missing_pair not in _pairs_of(r["triple"]) for r in report["junctions"])


def test_junction_instability_report_raises_rather_than_reporting_a_vacuous_fraction(
    ground_truth_mesh,
):
    with pytest.raises(ValueError, match="undefined, not 0"):
        ji.junction_instability_report(ground_truth_mesh, {})


def test_spatial_summary_bins_cover_every_present_junction(ground_truth_mesh):
    interfaces = geometry.extract_edges_trijunctions(ground_truth_mesh)
    all_pairs = {pair for triple in interfaces for pair in _pairs_of(triple)}
    tensions = dict.fromkeys(all_pairs, 1.0)
    report = ji.junction_instability_report(ground_truth_mesh, tensions)

    summary = ji.spatial_summary(report, n_bins=3)

    assert len(summary) == 3
    assert sum(b["n"] for b in summary) == report["n_present"]
    # innermost bin's radii must not exceed the outermost bin's, by construction of the sort
    assert summary[0]["radial_distance_range"][1] <= summary[-1]["radial_distance_range"][0] + 1e-9


def _pairs_of(triple):
    a, b, c = sorted(int(x) for x in triple)
    return {(a, b), (b, c), (a, c)}
