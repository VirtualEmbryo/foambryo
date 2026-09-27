#!/usr/bin/env python
r"""CLI: the converged-mesh-cohort work's analyses over `foambryo_benchmarks/run_converged_cohort.py`'s output.

Four independent analyses, run together because they all read the same two committed
baselines (`foambryo_benchmarks/baseline/converged_cohort.json`, `foambryo_benchmarks/baseline/attribution.json`):

- **The pre-registered prediction test** (`rho_free_prediction_test`) -- pools the 42
  equilibrium `benchmarking-dataset` cases with the 72 converged-cohort meshes and reports the
  Spearman correlation between each method's error and `rho_free`, against criteria written
  *before* this script was run: confirmed if `ForceBalanceKKT`'s correlation is >= 0.60 with
  p < 0.01 *and* `YoungDupre`'s `|r|` is <= 0.30; falsified if `ForceBalanceKKT`'s is < 0.30;
  unresolved otherwise.
- **The hold-last-value spread analysis** (`hold_last_value_spread`): within each folder's
  hold-last-value stage groups (`foambryo_benchmarks.converged_mesh_parser.stage_groups`), the spread
  of each method's *inferred* tension across re-convergence/remeshing at fixed physics.
- **The valence-persistence and wall-time-scaling analysis** (`valence_persistence`,
  `scaling_report`): whether valence-≥4 edges persist across a re-convergence group, and how
  each inference method's wall time scales with interface count on these much larger meshes.
- **The interface-set attribution check** (`interface_set_attribution`): whether
  `foambryo_benchmarks/baseline/attribution.json` already carries enough information to say how many
  interfaces `dw3d` invents/misses relative to ground truth, without running `dw3d` again.

Usage:
    uv run python -m foambryo_benchmarks.analyze_converged_cohort
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = REPO_ROOT / "foambryo_benchmarks" / "baseline"


def _present(records: list[dict]) -> list[dict]:
    return [r for r in records if "error" not in r]


def rho_free_prediction_test(
    converged_records: list[dict],
    c1_records: list[dict],
    methods: tuple[str, ...] = ("ForceBalanceKKT", "YoungDupre", "Variational", "YoungDupreLocal", "BayesianMAP"),
) -> dict:
    """The pre-registered prediction test: pooled Spearman correlation of per-method error against `rho_free`.

    Args:
        converged_records (list[dict]): `converged_cohort.json` records (72 meshes).
        c1_records (list[dict]): `attribution.json` records (47 cases x 4 variants x 7
            methods); only `equilibrium_group == "equilibrium"` rows are used, deduplicated
            across `dw3d_variant` since `rho_free_p1`/`e_inference` are P1-only and
            variant-independent.
        methods (tuple[str, ...]): Methods to report (must exist in both sources).

    Returns:
        dict: `{method: {"n", "spearman_r", "spearman_p", "rho_free_range", "error_range"}}`.
    """
    # De-duplicate the 42-case side: rho_free_p1 / e_inference do not depend on dw3d_variant.
    seen = set()
    dataset_pairs: dict[str, list[tuple[float, float]]] = {m: [] for m in methods}
    for r in c1_records:
        if r.get("equilibrium_group") != "equilibrium" or "error" in r:
            continue
        key = (r["case"], r["method"])
        if key in seen:
            continue
        seen.add(key)
        if r["method"] in dataset_pairs:
            dataset_pairs[r["method"]].append((r["rho_free_p1"], r["e_inference"]))

    cohort_pairs: dict[str, list[tuple[float, float]]] = {m: [] for m in methods}
    for r in converged_records:
        rho_free = r["rho_free"]
        for method in methods:
            entry = r["methods"].get(method, {})
            if "error" in entry:
                continue
            cohort_pairs[method].append(
                (rho_free, entry["tension_relative_error_matched_gauge"]["median"]),
            )

    result = {}
    for method in methods:
        pairs = dataset_pairs[method] + cohort_pairs[method]
        rho = np.array([p[0] for p in pairs])
        err = np.array([p[1] for p in pairs])
        r_value, p_value = spearmanr(rho, err)
        result[method] = {
            "n": len(pairs),
            "n_dataset_42": len(dataset_pairs[method]),
            "n_cohort_72": len(cohort_pairs[method]),
            "spearman_r": float(r_value),
            "spearman_p": float(p_value),
            "rho_free_range": [float(rho.min()), float(rho.max())],
            "error_range": [float(err.min()), float(err.max())],
            "error_median": float(np.median(err)),
        }
    return result


def hold_last_value_spread(converged_records: list[dict], methods: tuple[str, ...]) -> dict:
    """The hold-last-value spread analysis: spread of inferred tension within each hold-last-value stage group.

    For every group of >=2 stages sharing one ground-truth tension map (computed by the
    parser, not hardcoded), this reads back each per-mesh JSON's own inferred tensions is
    NOT available here (the combined baseline drops `per_key`), so this instead uses the
    per-method *aggregate* error spread across the group's meshes as the empirical
    mesh-to-mesh variability estimate at fixed physics -- the direct quantity the
    uncertainty work's error/std calibration should be read against.

    Args:
        converged_records (list[dict]): `converged_cohort.json` records.
        methods (tuple[str, ...]): Methods to report.

    Returns:
        dict: `{folder: {group_id: {method: {"stages", "n", "error_median_spread",
            "error_median_values"}}}}`.
    """
    by_folder: dict[str, list[dict]] = {}
    for r in converged_records:
        by_folder.setdefault(r["folder"], []).append(r)

    result: dict = {}
    for folder, records in by_folder.items():
        records = sorted(records, key=lambda r: r["stage"])
        by_group: dict[int, list[dict]] = {}
        for r in records:
            by_group.setdefault(r["hold_last_value_group"], []).append(r)
        result[folder] = {}
        for group_id, group_records in by_group.items():
            if len(group_records) < 2:
                continue
            entry = {}
            for method in methods:
                values = [
                    r["methods"][method]["tension_relative_error_matched_gauge"]["median"]
                    for r in group_records
                    if "error" not in r["methods"].get(method, {})
                ]
                if len(values) < 2:
                    continue
                values = np.array(values)
                entry[method] = {
                    "stages": [r["stage"] for r in group_records],
                    "n": len(values),
                    "median": float(np.median(values)),
                    "std": float(np.std(values)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "spread": float(values.max() - values.min()),
                    "cv": float(np.std(values) / np.mean(values)) if np.mean(values) else None,
                }
            result[folder][group_id] = entry
    return result


def valence_persistence(converged_records: list[dict]) -> dict:
    """The valence-persistence analysis: valence-≥4 / 2-material-valence-4 edge counts, per mesh and group.

    Args:
        converged_records (list[dict]): `converged_cohort.json` records.

    Returns:
        dict: `{folder: {"per_stage": [...], "groups": {group_id: {"n_valence_ge4_values"}}}}`.
    """
    by_folder: dict[str, list[dict]] = {}
    for r in converged_records:
        by_folder.setdefault(r["folder"], []).append(r)

    result: dict = {}
    for folder, records in by_folder.items():
        records = sorted(records, key=lambda r: r["stage"])
        per_stage = [
            {
                "stage": r["stage"],
                "group": r["hold_last_value_group"],
                "n_valence_ge4": r["valence"]["n_valence_ge4"],
                "n_two_material_valence4": r["valence"]["n_two_material_valence4"],
                "n_cells": r["n_cells"],
                "n_interfaces": r["n_interfaces"],
            }
            for r in records
        ]
        by_group: dict[int, list[int]] = {}
        for s in per_stage:
            by_group.setdefault(s["group"], []).append(s["n_valence_ge4"])
        result[folder] = {
            "per_stage": per_stage,
            "groups": {
                gid: {"n_valence_ge4_values": vals, "constant": len(set(vals)) == 1}
                for gid, vals in by_group.items()
                if len(vals) >= 2
            },
            "total_two_material_valence4": sum(s["n_two_material_valence4"] for s in per_stage),
        }
    return result


def scaling_report(converged_records: list[dict], methods: tuple[str, ...]) -> dict:
    """The wall-time-scaling analysis: how per-method wall time scales with interface count here.

    Args:
        converged_records (list[dict]): `converged_cohort.json` records.
        methods (tuple[str, ...]): Methods to report.

    Returns:
        dict: `{"n_cells_range", "n_interfaces_range", "per_method": {method: {...}}}`.
    """
    n_cells = np.array([r["n_cells"] for r in converged_records])
    n_iface = np.array([r["n_interfaces"] for r in converged_records])
    per_method = {}
    for method in methods:
        times, ifaces, errors = [], [], []
        for r in converged_records:
            entry = r["methods"].get(method, {})
            if "error" in entry:
                continue
            times.append(entry["wall_time_s"])
            ifaces.append(r["n_interfaces"])
            errors.append(entry["tension_relative_error_matched_gauge"]["median"])
        times, ifaces = np.array(times), np.array(ifaces)
        # log-log slope: time ~ n_interfaces^slope.
        slope = float(np.polyfit(np.log(ifaces), np.log(times), 1)[0]) if len(times) > 2 else None
        per_method[method] = {
            "n": len(times),
            "wall_time_median_s": float(np.median(times)),
            "wall_time_max_s": float(np.max(times)),
            "log_log_slope_time_vs_n_interfaces": slope,
            "error_median": float(np.median(errors)),
        }
    return {
        "n_cells_range": [int(n_cells.min()), int(n_cells.max())],
        "n_interfaces_range": [int(n_iface.min()), int(n_iface.max())],
        "per_method": per_method,
    }


def sigma_soft_report(converged_records: list[dict]) -> dict:
    """The valence-persistence and wall-time-scaling analysis: sigma_soft distribution on
    the cohort, to compare against the 42's own range."""
    sigma_soft = np.array([r["sigma_soft"] for r in converged_records])
    rho_free = np.array([r["rho_free"] for r in converged_records])
    return {
        "n": len(sigma_soft),
        "sigma_soft_min": float(sigma_soft.min()),
        "sigma_soft_median": float(np.median(sigma_soft)),
        "sigma_soft_max": float(sigma_soft.max()),
        "rho_free_min": float(rho_free.min()),
        "rho_free_median": float(np.median(rho_free)),
        "rho_free_max": float(rho_free.max()),
    }


def interface_set_attribution(c1_records: list[dict]) -> dict:
    """The interface-set attribution check: invented/missed interface counts, derived from
    committed count fields only.

    `foambryo_benchmarks/baseline/attribution.json` strips `per_key` (`run_attribution._drop_per_key_detail`),
    so the *identity* of which interface is invented/missed is not recoverable from it -- but
    the *counts* this check actually asks for ("how many interfaces the reconstruction
    invents, how many it misses") are: `invented = n_interfaces_p2 - n_common_interfaces.p2`,
    `missed = n_interfaces_p1 - n_common_interfaces.p2` (n_interfaces_p1 stands in for the
    ground truth's own interface count, since P1 -- the simulator mesh -- reproduces every
    ground-truth interface by construction). No `dw3d` run is needed for this.

    Args:
        c1_records (list[dict]): `attribution.json` records.

    Returns:
        dict: `{"has_per_key": False, "per_variant": {variant: {...}}}`.
    """
    has_per_key = any(
        "per_key" in r.get("e_total_detail", {}).get("gauge_matched", {}) for r in c1_records if "error" not in r
    )

    # Interface-set counts are a property of the mesh (P1/P2), not of the tension method, so
    # collapse to one row per (case, variant) using the first method present, to avoid
    # counting the same geometric fact 7 times.
    seen: set[tuple[str, str]] = set()
    by_variant: dict[str, list[dict]] = {}
    for r in c1_records:
        if "error" in r or r.get("equilibrium_group") != "equilibrium":
            continue
        key = (r["case"], r["dw3d_variant"])
        if key in seen:
            continue
        seen.add(key)
        by_variant.setdefault(r["dw3d_variant"], []).append(r)

    per_variant = {}
    for variant, records in by_variant.items():
        invented = [r["n_interfaces_p2"] - r["n_common_interfaces"]["p2"] for r in records]
        missed = [r["n_interfaces_p1"] - r["n_common_interfaces"]["p2"] for r in records]
        per_variant[variant] = {
            "n_cases": len(records),
            "invented_total": int(sum(invented)),
            "invented_median": float(np.median(invented)),
            "invented_max": int(max(invented)),
            "n_cases_with_invented_interface": int(sum(x > 0 for x in invented)),
            "missed_total": int(sum(missed)),
            "missed_median": float(np.median(missed)),
            "missed_max": int(max(missed)),
            "n_cases_with_missed_interface": int(sum(x > 0 for x in missed)),
        }
    return {"has_per_key": has_per_key, "per_variant": per_variant}


def main() -> None:
    """CLI entry point: run all four analyses and write the combined JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converged-cohort", type=Path, default=BASELINE_DIR / "converged_cohort.json")
    parser.add_argument("--attribution", type=Path, default=BASELINE_DIR / "attribution.json")
    parser.add_argument("--out", type=Path, default=BASELINE_DIR / "converged_cohort_analysis.json")
    args = parser.parse_args()

    converged_records = _present(json.loads(args.converged_cohort.read_text()))
    c1_records = json.loads(args.attribution.read_text())

    methods = ("ForceBalanceKKT", "YoungDupre", "Variational", "YoungDupreLocal", "BayesianMAP")
    analysis = {
        "c4_3_rho_free_prediction_test": rho_free_prediction_test(converged_records, c1_records, methods),
        "c4_4_hold_last_value_spread": hold_last_value_spread(converged_records, methods),
        "c4_5_valence_persistence": valence_persistence(converged_records),
        "c4_5_scaling_report": scaling_report(converged_records, methods),
        "c4_5_sigma_soft_report": sigma_soft_report(converged_records),
        "c4_6_interface_set_attribution": interface_set_attribution(c1_records),
    }
    args.out.write_text(json.dumps(analysis, indent=2, sort_keys=True))
    print(f"wrote {args.out}")

    print("\nPre-registered prediction test:")
    for method, stats in analysis["c4_3_rho_free_prediction_test"].items():
        print(
            f"  {method}: n={stats['n']} ({stats['n_dataset_42']}+{stats['n_cohort_72']}) "
            f"spearman_r={stats['spearman_r']:.3f} p={stats['spearman_p']:.2e} "
            f"rho_free=[{stats['rho_free_range'][0]:.4g},{stats['rho_free_range'][1]:.4g}] "
            f"error_median={stats['error_median']:.4g}",
        )

    print("\nsigma_soft / rho_free on the cohort:")
    print(analysis["c4_5_sigma_soft_report"])

    has_per_key = analysis["c4_6_interface_set_attribution"]["has_per_key"]
    print(f"\nInterface-set attribution (has_per_key = {has_per_key}):")
    for variant, stats in analysis["c4_6_interface_set_attribution"]["per_variant"].items():
        print(f"  {variant}: {stats}")


if __name__ == "__main__":
    main()
