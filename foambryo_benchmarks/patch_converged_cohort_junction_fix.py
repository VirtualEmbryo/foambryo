#!/usr/bin/env python
r"""CLI: regenerate `converged_cohort.json`'s `YoungDupre`/`YoungDupreLocal` rows (the junction-graph work).

`foambryo_benchmarks/run_converged_cohort.py` re-runs **all 7** `METHOD_REGISTRY` methods on **all 72**
meshes (~35-190 minutes depending on machine load). The junction-graph work's fix
(`foambryo.junction_graph`) only changes output on the meshes whose triple-junction graph is
fragmented, and only for the two methods whose coupling is that graph (`YoungDupre`,
`YoungDupreLocal`) -- proved bit-identical on every other (mesh, method) pair by an actual A/B diff
against the pre-fix commit on all 47 dataset and all 72 cohort meshes, not by argument.

This script exploits that proof rather than re-deriving it: it re-computes only `YoungDupre`/
`YoungDupreLocal` on the meshes `foambryo.junction_graph.partition_from_mesh` finds fragmented,
patches those two methods' entries (plus a `junction_graph` field on every record, for
transparency, by design: the component structure must be surfaced), and
leaves every other method's cached numbers untouched. A full `run_converged_cohort` re-run
would reproduce this file exactly (mod float ordering in `ForceBalance`/`Variational`'s own
non-junction-graph solves, which this script does not touch at all).

Usage:
    uv run python -m foambryo_benchmarks.patch_converged_cohort_junction_fix
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from foambryo_benchmarks import metrics as m
from foambryo_benchmarks.junction_component_metrics import per_component_gauge_matched_tension_errors
from foambryo_benchmarks.run_attribution import METHOD_REGISTRY
from foambryo_benchmarks.run_converged_cohort import JUNCTION_GRAPH_METHOD_NAMES
from foambryo import dcel_mesh_from_file
from foambryo.junction_graph import partition_from_mesh

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent
CONVERGED_MESHES_DIR = WORKSPACE_ROOT / "converged-meshes"
DEFAULT_BASELINE = REPO_ROOT / "foambryo_benchmarks" / "baseline" / "converged_cohort.json"


def _json_safe(obj: object) -> object:
    if isinstance(obj, dict):
        return {(",".join(map(str, k)) if isinstance(k, tuple) else str(k)): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def patch_record(record: dict) -> bool:
    """Patch one mesh's record in place. Returns whether its junction graph is fragmented.

    Args:
        record (dict): One `converged_cohort.json` entry, modified in place.

    Returns:
        bool: Whether the mesh's triple-junction graph was fragmented (i.e. whether any
            method entry was actually recomputed).
    """
    folder, stage = record["folder"], record["stage"]
    mesh_path = CONVERGED_MESHES_DIR / folder / "converged" / f"mesh{stage:06d}.rec"
    tensions_path = mesh_path.with_name(mesh_path.stem + "_dict_tensions.npy")
    dict_truth = np.load(tensions_path, allow_pickle=True).item()

    mesh = dcel_mesh_from_file(str(mesh_path))
    partition = partition_from_mesh(mesh)
    record["junction_graph"] = {
        "fragmented": partition.fragmented,
        "n_components": partition.n_components,
        "component_sizes": sorted((len(c) for c in partition.components), reverse=True),
        "n_isolated": len(partition.isolated),
    }
    if not partition.fragmented:
        return False

    dict_tensions_used = {key: dict_truth[key] for key in partition.interfaces}
    for method_name in JUNCTION_GRAPH_METHOD_NAMES:
        t0 = time.perf_counter()
        try:
            dict_inferred = METHOD_REGISTRY[method_name](mesh)
        except Exception as exc:  # a single method failure must not stop the patch
            record["methods"][method_name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        wall_time_s = time.perf_counter() - t0
        n_undetermined = len(partition.isolated)
        determined = {key: value for key, value in dict_inferred.items() if not np.isnan(value)}
        error = m.tension_relative_errors(determined, dict_tensions_used)
        error_matched = per_component_gauge_matched_tension_errors(dict_inferred, dict_tensions_used, partition)
        record["methods"][method_name] = {
            "gauge_matching_scale": None,
            "gauge_matching_note": (
                f"fragmented junction graph: {error_matched['n_components']} component(s), "
                f"{n_undetermined} undetermined interface(s) excluded -- gauge-matched per "
                "component (foambryo_benchmarks.junction_component_metrics), not globally"
            ),
            "tension_relative_error": {
                **{k: v for k, v in error.items() if k != "per_key"},
                "n_undetermined_excluded": n_undetermined,
            },
            "tension_relative_error_matched_gauge": {
                k: v for k, v in error_matched.items() if k not in ("per_key", "per_component")
            },
            "per_component_gauge_matching": error_matched["per_component"],
            "wall_time_s": wall_time_s,
        }
    return True


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    args = parser.parse_args()

    records = json.loads(args.baseline.read_text())
    n_fragmented = 0
    for record in records:
        if "error" in record:
            continue
        if patch_record(record):
            n_fragmented += 1
            print(f"  patched {record['folder']} stage {record['stage']}: {record['junction_graph']}")

    args.baseline.write_text(json.dumps(_json_safe(records), indent=2))
    print(f"\n{n_fragmented} of {len(records)} meshes were fragmented and patched -> {args.baseline}")


if __name__ == "__main__":
    main()
