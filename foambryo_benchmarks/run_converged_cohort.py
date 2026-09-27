#!/usr/bin/env python
r"""The converged-mesh-cohort work: tensions-only benchmark, ground-truth meshes.

`converged-meshes/` holds 4 folders x 18 optimisation-stage meshes (90-165 cells) from the
same LosTopos/Simba generator as `benchmarking-dataset`, but run with `conjugate_gradient`
at `optimization-tolerance 1e-9` (`benchmarking-dataset`'s 47 cases used gradient descent).
Each folder carries **tensions only** (`multitensions.opt`, parsed by
`foambryo_benchmarks.converged_mesh_parser`) -- there are no ground-truth pressures, and none can be
honestly derived: `optimization-use-volume-constraint 1` means volume is enforced by force
projection with pressures as the projection's Lagrange multipliers, not as an energy term,
so a "pressure" read off this generator would be a
solver artifact, not an input. Every record below is tensions-only, and says so.

For each of the 72 meshes this emits:
  - `converged/meshNNNNNN_dict_tensions.npy`, keyed exactly like `benchmarking-dataset`
    (`tuple(min_label, max_label) -> float`), from the parsed `multitensions.opt`;
  - the dataset audit (`foambryo_benchmarks.metrics.dataset_audit`): integrity, `rho` at nominal
    tensions with pressures marginalised, `rho_free`, singular spectrum, `sigma_soft`,
    `gamma_eff / gamma_nominal`;
  - gauge-matched relative tension error for the same 7 methods as the attribution grid
    (`foambryo_benchmarks.run_attribution.METHOD_REGISTRY`) -- this is a P1-only measurement (no
    `dw3d`, no P2 mesh), i.e. the direct analogue of the attribution benchmark's `e_inference`;
  - the edge-valence report (`foambryo_benchmarks.valence_metrics`), for whether the simulator's own
    converged meshes carry valence->=4 edges (they do: 106-1558 per mesh, on all 72).

Usage:
    uv run python -m foambryo_benchmarks.run_converged_cohort
    uv run python -m foambryo_benchmarks.run_converged_cohort --folders 90LumenA_AllEqu_Base1B_pTEidown_mToup
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from foambryo_benchmarks import metrics as m
from foambryo_benchmarks.converged_mesh_parser import parse_multitensions, stage_groups, validate_coverage
from foambryo_benchmarks.junction_component_metrics import per_component_gauge_matched_tension_errors
from foambryo_benchmarks.run_attribution import METHOD_REGISTRY
from foambryo_benchmarks.valence_metrics import edge_valence_report
from foambryo import dcel_mesh_from_file
from foambryo.junction_graph import partition_from_mesh
from foambryo.validation import mesh_io

#: `METHOD_REGISTRY` entries whose tensions are coupled through the triple-junction graph
#: (`foambryo.junction_graph`'s "Scope: which methods") and therefore need per-component
#: gauge matching on a fragmented mesh, rather than the single global
#: `foambryo_benchmarks.metrics.gauge_matching_scale` every other method here still uses correctly
#: (measured, not assumed, that `ForceBalanceKKT` and
#: both `Variational` routes stay single-component even on the most fragmented cohort mesh).
JUNCTION_GRAPH_METHOD_NAMES = frozenset({"YoungDupre", "YoungDupreLocal"})

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent
CONVERGED_MESHES_DIR = WORKSPACE_ROOT / "converged-meshes"
OUTPUT_DIR = REPO_ROOT / "foambryo_benchmarks" / "baseline" / "converged_cohort"


def discover_folders(root: Path) -> list[Path]:
    """List converged-mesh folders (anything with a `multitensions.opt` and a `converged/`).

    Args:
        root (Path): `converged-meshes/`.

    Returns:
        list[Path]: Sorted folder paths.
    """
    return sorted(
        p for p in root.iterdir() if p.is_dir() and (p / "multitensions.opt").exists() and (p / "converged").is_dir()
    )


def _json_safe(obj: object) -> object:
    if isinstance(obj, dict):
        return {(",".join(map(str, k)) if isinstance(k, tuple) else str(k)): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def run_folder(folder: Path, output_dir: Path) -> list[dict]:
    """Process every stage of one converged-mesh folder.

    Args:
        folder (Path): e.g. `converged-meshes/90LumenA_.../`.
        output_dir (Path): Where to write per-mesh JSON records (baseline artefacts).

    Returns:
        list[dict]: One record per stage.
    """
    parsed = parse_multitensions(folder / "multitensions.opt")
    groups = stage_groups(parsed)
    stage_to_group = {stage: gi for gi, group in enumerate(groups) for stage in group}

    records = []
    for stage in range(parsed.n_stages):
        mesh_path = folder / "converged" / f"mesh{stage:06d}.rec"
        v, f, regions = mesh_io.read_rec(mesh_path)
        interfaces = mesh_io.list_interfaces(regions)
        dict_tensions = parsed.tensions_by_stage[stage]

        coverage = validate_coverage(dict_tensions, interfaces)
        if coverage["n_uncovered"] > 0:
            message = (
                f"{folder.name} stage {stage}: {coverage['n_uncovered']} interfaces have no "
                f"rule in multitensions.opt: {coverage['uncovered']}"
            )
            raise ValueError(message)

        # Emit the ground-truth tensions .npy, keyed exactly like benchmarking-dataset.
        tensions_npy_path = mesh_path.with_name(mesh_path.stem + "_dict_tensions.npy")
        np.save(tensions_npy_path, dict_tensions, allow_pickle=True)

        dict_tensions_used = {key: dict_tensions[key] for key in interfaces}
        audit = m.dataset_audit(v, f, regions, dict_tensions_used)
        valence = edge_valence_report(v, f, regions)

        mesh = dcel_mesh_from_file(str(mesh_path))
        partition = partition_from_mesh(mesh)
        methods = {}
        for method_name, method_fn in METHOD_REGISTRY.items():
            t0 = time.perf_counter()
            try:
                dict_inferred = method_fn(mesh)
            except Exception as exc:  # a single method failure must not stop the grid
                methods[method_name] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            wall_time_s = time.perf_counter() - t0

            is_junction_graph_method = method_name in JUNCTION_GRAPH_METHOD_NAMES
            if is_junction_graph_method and partition.fragmented:
                # The junction-graph work: this mesh's triple-junction graph is fragmented, so a single
                # global gauge_matching_scale would silently blend components on different
                # scales. Gauge-match each component
                # separately and pool only the (dimensionless, gauge-invariant) errors.
                # Undetermined (closed) interfaces are `nan` (foambryo.junction_graph's
                # convention) and are excluded here too, never left to poison the raw
                # (un-gauge-matched) aggregate with a silent NaN.
                n_undetermined = len(partition.isolated)
                determined = {key: value for key, value in dict_inferred.items() if not np.isnan(value)}
                error = m.tension_relative_errors(determined, dict_tensions_used)
                error_matched = per_component_gauge_matched_tension_errors(
                    dict_inferred,
                    dict_tensions_used,
                    partition,
                )
                methods[method_name] = {
                    "gauge_matching_scale": None,
                    "gauge_matching_note": (
                        f"fragmented junction graph: {error_matched['n_components']} component(s), "
                        f"{n_undetermined} undetermined interface(s) excluded -- "
                        "gauge-matched per component (foambryo_benchmarks.junction_component_metrics), not globally"
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
                continue

            scale = m.gauge_matching_scale(dict_inferred, dict_tensions_used)
            error = m.tension_relative_errors(dict_inferred, dict_tensions_used)
            error_matched = m.tension_relative_errors(
                {key: scale * value for key, value in dict_inferred.items()},
                dict_tensions_used,
            )
            methods[method_name] = {
                "gauge_matching_scale": scale,
                "tension_relative_error": {k: v for k, v in error.items() if k != "per_key"},
                "tension_relative_error_matched_gauge": {k: v for k, v in error_matched.items() if k != "per_key"},
                "wall_time_s": wall_time_s,
            }

        record = {
            "folder": folder.name,
            "stage": stage,
            "hold_last_value_group": stage_to_group[stage],
            "n_cells": len(mesh_io.cell_labels(regions)),
            "n_interfaces": len(interfaces),
            "n_junctions": audit["n_junctions"],
            "junction_graph": {
                "fragmented": partition.fragmented,
                "n_components": partition.n_components,
                "component_sizes": sorted((len(c) for c in partition.components), reverse=True),
                "n_isolated": len(partition.isolated),
            },
            "integrity": audit["integrity"],
            "rho": audit["rho"],
            "rho_free": audit["rho_free"],
            "sigma_gauge": audit["sigma_gauge"],
            "sigma_soft": audit["sigma_soft"],
            "soft_mode_fractions": audit["soft_mode_fractions"],
            "residual_localisation": audit["residual_localisation"],
            "effective_tensions_gauge_summary": {
                gauge: {k: v for k, v in stats.items() if k != "per_key"}
                for gauge, stats in audit["effective_tensions"]["gauges"].items()
            },
            "valence": valence,
            "methods": methods,
            "coverage": {"n_mesh_interfaces": coverage["n_mesh_interfaces"], "n_covered": coverage["n_covered"]},
            "tensions_only": True,
            "pressures_available": False,
        }
        records.append(record)
        (output_dir / f"{folder.name}__stage{stage:03d}.json").write_text(
            json.dumps(_json_safe(record), indent=2, sort_keys=True) + "\n",
        )
        print(
            f"  {folder.name} stage {stage:2d} (group {stage_to_group[stage]}): "
            f"n_cells={record['n_cells']:3d} n_iface={record['n_interfaces']:3d} "
            f"rho={record['rho']:.4f} rho_free={record['rho_free']:.4f} "
            f"valence>=4={valence['n_valence_ge4']:3d} 2mat-v4={valence['n_two_material_valence4']}",
        )
    return records


def main() -> None:
    """Run the whole cohort and write per-mesh JSON + one combined JSON list."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=CONVERGED_MESHES_DIR)
    parser.add_argument("--folders", nargs="*", default=None, help="Subset of folder names (default: all)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    folders = discover_folders(args.root)
    if args.folders:
        folders = [f for f in folders if f.name in args.folders]

    all_records = []
    for folder in folders:
        print(f"[{folder.name}]")
        all_records.extend(run_folder(folder, args.output_dir))

    combined_path = args.output_dir.parent / "converged_cohort.json"
    combined_path.write_text(json.dumps(_json_safe(all_records), indent=2))
    print(f"\n{len(all_records)} mesh records from {len(folders)} folders -> {combined_path}")


if __name__ == "__main__":
    main()
