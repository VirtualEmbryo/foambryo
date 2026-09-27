"""Tests for `foambryo_benchmarks.junction_component_metrics` (the junction-graph work)."""

from __future__ import annotations

import math

import pytest

from foambryo_benchmarks.junction_component_metrics import (
    per_component_gauge_matched_tension_errors,
    require_unfragmented_mesh,
)
from foambryo.junction_graph import JunctionGraphPartition


def _partition_two_components(isolated: tuple = ()) -> JunctionGraphPartition:
    components = (((0, 1), (0, 2), (1, 2)), ((0, 3), (0, 4), (3, 4)))
    component_of = {key: i for i, comp in enumerate(components) for key in comp}
    return JunctionGraphPartition(
        interfaces=tuple(k for comp in components for k in comp) + isolated,
        components=components,
        component_of=component_of,
        isolated=isolated,
    )


def test_require_unfragmented_mesh_raises_on_fragmented_partition():
    with pytest.raises(ValueError, match="refusing a single-gauge"):
        require_unfragmented_mesh(_partition_two_components())


def test_require_unfragmented_mesh_passes_single_component():
    partition = JunctionGraphPartition(
        interfaces=((0, 1), (0, 2), (1, 2)),
        components=(((0, 1), (0, 2), (1, 2)),),
        component_of={(0, 1): 0, (0, 2): 0, (1, 2): 0},
        isolated=(),
    )
    require_unfragmented_mesh(partition)  # must not raise


def test_per_component_gauge_matched_errors_recovers_perfect_fit_within_each_component():
    """Recover zero error in both components despite different arbitrary scale factors.

    Two components, each scaled by a DIFFERENT arbitrary factor relative to truth --
    per-component gauge matching must recover zero error in both, which a single global
    scale could not (that is the entire point of the decomposition).
    """
    truth = {(0, 1): 1.0, (0, 2): 1.0, (1, 2): 0.5, (0, 3): 2.0, (0, 4): 2.0, (3, 4): 3.0}
    inferred = {key: 7.0 * value for key, value in truth.items() if key in {(0, 1), (0, 2), (1, 2)}}
    inferred.update({key: 0.02 * value for key, value in truth.items() if key in {(0, 3), (0, 4), (3, 4)}})

    partition = _partition_two_components()
    result = per_component_gauge_matched_tension_errors(inferred, truth, partition)

    assert result["n"] == 6
    assert result["n_undetermined_excluded"] == 0
    assert result["n_components"] == 2
    for value in result["per_key"].values():
        assert value == pytest.approx(0.0, abs=1e-10)
    assert result["per_component"][0]["gauge_matching_scale"] == pytest.approx(1.0 / 7.0)
    assert result["per_component"][1]["gauge_matching_scale"] == pytest.approx(1.0 / 0.02)


def test_per_component_gauge_matched_errors_excludes_undetermined_interfaces():
    truth = {(0, 1): 1.0, (0, 2): 1.0, (1, 2): 1.0, (0, 5): 1.0}
    inferred = {(0, 1): 1.0, (0, 2): 1.0, (1, 2): 1.0, (0, 5): math.nan}
    partition = JunctionGraphPartition(
        interfaces=((0, 1), (0, 2), (1, 2), (0, 5)),
        components=(((0, 1), (0, 2), (1, 2)),),
        component_of={(0, 1): 0, (0, 2): 0, (1, 2): 0},
        isolated=((0, 5),),
    )
    result = per_component_gauge_matched_tension_errors(inferred, truth, partition)
    assert result["n"] == 3
    assert (0, 5) not in result["per_key"]
    assert result["n_undetermined_excluded"] == 1
    for value in result["per_key"].values():
        assert value == pytest.approx(0.0, abs=1e-10)
