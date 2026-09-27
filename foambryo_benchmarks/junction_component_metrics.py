"""Per-component gauge matching for junction-graph-fragmented meshes (the junction-graph work).

`foambryo_benchmarks.metrics.gauge_matching_scale`/`tension_relative_errors` compute **one** global
scale factor and error table for a whole mesh, which is only valid when every interface
shares a common gauge -- i.e. when the mesh's triple-junction graph
(`foambryo.junction_graph`) is a single connected component with no isolated interface. When
it is not, applying one scale across components silently blends incompatible normalisations
-- exactly the mechanism that pinned `YoungDupre`'s error at *exactly* 1.0 on 26 of the 72
converged-cohort meshes before per-component gauge matching, and exactly what
`foambryo.junction_graph.require_single_component_solution` refuses at the library level.

`require_unfragmented_mesh` is this module's `foambryo_benchmarks.case_groups.require_single_group`
analogue: call it before any single-gauge computation. `per_component_gauge_matched_tension_errors`
is the decomposition a fragmented mesh needs instead: it gauge-matches **each component
separately** against the same ground truth, then pools only the resulting *relative errors*
(a dimensionless, already gauge-invariant quantity) into one mesh-level summary -- never the
raw tensions and never a single scale factor. Undetermined (closed) interfaces are excluded
from the error table entirely (their inferred tension is `nan`, never compared to a number),
and their count is reported rather than silently dropped.
"""

from __future__ import annotations

import numpy as np

from foambryo_benchmarks import metrics as m
from foambryo.junction_graph import JunctionGraphPartition


def require_unfragmented_mesh(partition: JunctionGraphPartition) -> None:
    """Refuse a single-gauge computation on a fragmented mesh's tensions.

    Args:
        partition (JunctionGraphPartition): The mesh's triple-junction-graph partition.

    Raises:
        ValueError: If `partition.fragmented` -- i.e. more than one component or at least
            one isolated (closed) interface. `foambryo_benchmarks.metrics.gauge_matching_scale` and
            `tension_relative_errors` must not be called directly on such a mesh's tensions;
            use `per_component_gauge_matched_tension_errors` instead.
    """
    if partition.fragmented:
        message = (
            f"refusing a single-gauge computation on a fragmented mesh: "
            f"{partition.n_components} independently-gauged component(s) and "
            f"{len(partition.isolated)} undetermined (closed) interface(s) -> "
            f"{partition.isolated}. Use "
            "foambryo_benchmarks.junction_component_metrics.per_component_gauge_matched_tension_errors "
            "instead."
        )
        raise ValueError(message)


def per_component_gauge_matched_tension_errors(
    dict_inferred: dict[tuple[int, int], float],
    dict_truth: dict[tuple[int, int], float],
    partition: JunctionGraphPartition,
) -> dict:
    """Gauge-match and score each junction-graph component separately, then pool the errors.

    Args:
        dict_inferred (dict[tuple[int, int], float]): Inferred tension per interface. May
            contain `nan` at undetermined interfaces (`foambryo.junction_graph`'s
            convention) -- these are skipped, never compared to a ground-truth value.
        dict_truth (dict[tuple[int, int], float]): Ground-truth tension per interface.
        partition (JunctionGraphPartition): The mesh's triple-junction-graph partition,
            computed from the SAME mesh `dict_inferred` was solved on.

    Returns:
        dict: `{"per_key", "mean", "median", "p95", "n", "n_undetermined_excluded",
            "n_components", "per_component"}`. `per_key`/`mean`/`median`/`p95`/`n` pool every
            component's relative errors (dimensionless, so pooling is valid even though the
            components are not on a common tension scale); `per_component` gives each
            component's own gauge-matching scale and error stats, for a reader who wants the
            un-pooled detail.
    """
    per_key: dict[tuple[int, int], float] = {}
    per_component: dict[int, dict] = {}
    for component_id, component_keys in enumerate(partition.components):
        keys = [key for key in component_keys if key in dict_inferred and not np.isnan(dict_inferred[key])]
        inferred_c = {key: dict_inferred[key] for key in keys}
        truth_c = {key: dict_truth[key] for key in keys if key in dict_truth}
        scale = m.gauge_matching_scale(inferred_c, truth_c)
        matched = {key: scale * value for key, value in inferred_c.items()}
        errors = m.tension_relative_errors(matched, truth_c)
        per_component[component_id] = {
            "n": errors["n"],
            "median": errors["median"],
            "mean": errors["mean"],
            "p95": errors["p95"],
            "gauge_matching_scale": scale,
        }
        per_key.update(errors["per_key"])

    values = np.array(list(per_key.values())) if per_key else np.array([np.nan])
    return {
        "per_key": per_key,
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "n": len(per_key),
        "n_undetermined_excluded": len(partition.isolated),
        "n_components": partition.n_components,
        "per_component": per_component,
    }
