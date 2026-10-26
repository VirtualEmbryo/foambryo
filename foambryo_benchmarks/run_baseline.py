#!/usr/bin/env python
"""Generate the foambryo inference-harness baseline JSON over the benchmarking dataset.

Runs every existing `TensionComputationMethod` on every case's *ground-truth* `.rec` mesh
and records tension/pressure error (in two gauges) and wall time — see
`foambryo_benchmarks/run_case.py`.

Scope: the default is the full dataset. The harness was first restricted to a 3-case smoke
test, until the dataset had been audited for equilibrium; that audit has run
(`foambryo_benchmarks/run_dataset_audit.py`), so the restriction is lifted.

**The recorded decision to exclude the five non-equilibrium cases is wired in, not left to
the reader.** Five cases (`005`, `015`, `018`,
`021`, `043`) are not at mechanical equilibrium, so on those the error measured here is
dominated by the data, not by the method. `summarise_methods` therefore reports the 42 and
the 5 as two separate tables and refuses to produce a combined one; the same split governs
`summary.json` and `method_comparison.md`, both written next to the per-case records. See
`foambryo_benchmarks/case_groups.py`.

Usage:
    uv run python -m foambryo_benchmarks.run_baseline                    # all cases in the dataset
    uv run python -m foambryo_benchmarks.run_baseline --case-ids 000 001 # a subset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from foambryo_benchmarks import case_groups
from foambryo_benchmarks.run_case import _json_safe, run_case

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent
DATASET_DIR = WORKSPACE_ROOT / "benchmarking-dataset"
BASELINE_DIR = REPO_ROOT / "foambryo_benchmarks" / "baseline"

# The three cases the initial smoke test ran on, kept as a named set so a quick check is still
# one flag away now that the default is the whole dataset.
SMOKE_TEST_CASE_IDS = ["000", "001", "002"]


def discover_case_ids(dataset_dir: Path) -> list[str]:
    """List every case id in the dataset that has all three files `run_case` needs.

    Args:
        dataset_dir (Path): Directory holding `{case_id}_mesh.rec` etc.

    Returns:
        list[str]: Sorted case ids.
    """
    ids = []
    for mesh_path in sorted(dataset_dir.glob("*_mesh.rec")):
        case_id = mesh_path.name.removesuffix("_mesh.rec")
        needed = (
            dataset_dir / f"{case_id}_dict_tensions.npy",
            dataset_dir / f"{case_id}_dict_pressures.npy",
        )
        if all(path.exists() for path in needed):
            ids.append(case_id)
    return ids


def summarise_methods(records: list[dict]) -> dict:
    """Method-comparison summary, **computed separately for each equilibrium group**.

    This is the table that carries every accuracy claim about the inference methods, so it is
    the table that must not mix the 42 usable cases with the 5 non-equilibrium ones (the
    recorded decision to exclude the five non-equilibrium cases; see
    `foambryo_benchmarks/case_groups.py`). There is deliberately no all-47 row anywhere in the output.

    Args:
        records (list[dict]): Per-case baseline records from `run_case`.

    Returns:
        dict: `{"dataset": {...}, "groups": {group: {"methods": {name: {...}}}}}`.
    """
    grouped = case_groups.split_records(records)
    return {
        "dataset": {
            "n_cases": len(records),
            "n_by_group": {group: len(items) for group, items in grouped.items()},
            "group_notes": {group: case_groups.group_note(group) for group in case_groups.GROUP_ORDER},
        },
        "groups": {group: _summarise_methods_for_group(items) for group, items in grouped.items() if items},
    }


def _summarise_methods_for_group(records: list[dict]) -> dict:
    """Per-method error summary over records from a **single** equilibrium group.

    The headline statistic is the median over cases of each case's median relative tension
    error, in both gauges (as published and after matching the overall scale — the rule every
    accuracy comparison in this package follows). Reported alongside: the worst case,
    the spread of the gauge-matching factor, and `max |1 - mean(gamma)|`, which is what
    exposes a method whose soft normalisation row is not holding.

    Args:
        records (list[dict]): Per-case baseline records, all from one group.

    Returns:
        dict: Group name, case list and per-method statistics.

    Raises:
        ValueError: (from `case_groups.require_single_group`) if the records span both groups.
    """
    group = case_groups.require_single_group(records)
    method_names = sorted({name for r in records for name in r["methods"]})

    methods: dict[str, dict] = {}
    for name in method_names:
        entries = [r["methods"][name] for r in records if name in r["methods"]]
        errored = [r["case"] for r in records if "error" in r["methods"].get(name, {})]
        working = [(r["case"], r["methods"][name]) for r in records if "error" not in r["methods"].get(name, {})]
        if not working:
            methods[name] = {
                "n_errored": len(errored),
                "errored_cases": errored,
                "error_example": entries[0].get("error") if entries else None,
            }
            continue

        published = np.array([e["tension_relative_error"]["median"] for _, e in working])
        matched = np.array([e["tension_relative_error_matched_gauge"]["median"] for _, e in working])
        pressure = np.array(
            [
                e["pressure_relative_error_matched_gauge"]["median"]
                for _, e in working
                if e.get("pressure_relative_error_matched_gauge")
            ],
        )
        scales = np.array([e["gauge_matching_scale"] for _, e in working])
        mean_gamma = np.array([e["mean_inferred_tension"] for _, e in working])
        worst_index = int(np.argmax(matched))

        methods[name] = {
            "n_cases": len(working),
            "n_errored": len(errored),
            "errored_cases": errored,
            "tension_median_error_published": float(np.median(published)),
            "tension_median_error_matched_gauge": float(np.median(matched)),
            "tension_worst_case_matched_gauge": float(matched[worst_index]),
            "tension_worst_case": working[worst_index][0],
            "pressure_median_error_matched_gauge": float(np.median(pressure)) if pressure.size else None,
            "gauge_scale_median": float(np.median(scales)),
            "gauge_scale_max": float(scales.max()),
            "max_abs_mean_tension_deviation": float(np.abs(1.0 - mean_gamma).max()),
        }

    return {
        "group": group,
        "group_note": case_groups.group_note(group),
        "n_cases": len(records),
        "cases": [r["case"] for r in records],
        "methods": methods,
        "force_balance_normalisations": _summarise_normalisations(records),
    }


def _summarise_normalisations(records: list[dict]) -> dict:
    """The force-balance work's three-way normalisation comparison, aggregated over one group's cases.

    Answers, per variant, three questions: is
    `mean(gamma)` exactly 1, how well-conditioned is the system solved, and what are the
    error bars — plus the calibration ratio that says whether those error bars can be
    believed. The tension *error* per variant is already in the `methods` table under the
    matching `ForceBalance*` name; it is not repeated here.

    Args:
        records (list[dict]): Per-case records from one group.

    Returns:
        dict: `{variant: {statistic: value}}`, empty if no record carries the block (records
            generated before the force-balance work).
    """
    present = [r["force_balance_normalisations"] for r in records if "force_balance_normalisations" in r]
    if not present:
        return {}
    variants = sorted(set(present[0]) - {"n_interfaces"})
    summary = {}
    for variant in variants:
        entries = [p[variant] for p in present if variant in p]
        summary[variant] = {
            "n_cases": len(entries),
            "max_mean_tension_deviation": float(max(e["mean_tension_deviation"] for e in entries)),
            "median_mean_tension_deviation": float(np.median([e["mean_tension_deviation"] for e in entries])),
            "median_condition_number": float(np.median([e["condition_number"] for e in entries])),
            "max_condition_number": float(max(e["condition_number"] for e in entries)),
            "median_tension_std": float(np.median([e["tension_std_median"] for e in entries])),
            "median_error_over_std": float(np.median([e["error_over_std_median"] for e in entries])),
            "median_rho_at_inferred_tensions": float(np.median([e["rho_at_inferred_tensions"] for e in entries])),
        }
    return summary


def markdown_method_table(summary: dict) -> str:
    """Render one method-comparison table per equilibrium group.

    Args:
        summary (dict): Output of `summarise_methods`.

    Returns:
        str: GitHub-flavoured Markdown, one table per group, each with its own caption.
    """
    preamble = (
        "Generated by `foambryo_benchmarks/run_baseline.py`. Every number is the **median over cases** "
        "of that case's **median relative tension error** against the ground-truth tensions, "
        "on the ground-truth `.rec` mesh (no `dw3d` involved).\n\n"
        "**The two groups are never combined** — the recorded decision excluded the five non-equilibrium "
        "cases from every accuracy claim and kept them as a negative control. See "
        "`foambryo_benchmarks/case_groups.py` for the decision and the three measurements behind it.\n\n"
        "**Gauge** (the rule every accuracy comparison in this package follows): errors are "
        "given as published *and* after matching the overall tension scale. Only tension "
        "*ratios* are physical, so a method "
        "whose soft normalisation row does not hold can look far worse than it is — which the "
        "`max |1 - mean(g)|` column exposes.\n"
    )
    parts = ["# Tension-inference method comparison, by equilibrium group\n", preamble]
    for group in case_groups.GROUP_ORDER:
        if group not in summary["groups"]:
            continue
        group_summary = summary["groups"][group]
        parts.append(
            f"\n## {group} — {group_summary['n_cases']} cases\n\n"
            f"{group_summary['group_note']}\n\n"
            f"Cases: {', '.join(group_summary['cases'])}\n\n"
            "| method | as published | gauge-matched | worst case (matched) | pressure (matched) "
            "| gauge factor (median) | max \\|1 - mean(g)\\| | errored |\n"
            "|---|---|---|---|---|---|---|---|\n",
        )
        rows = sorted(
            group_summary["methods"].items(),
            key=lambda item: item[1].get("tension_median_error_matched_gauge", float("inf")),
        )
        for name, stats in rows:
            if "tension_median_error_matched_gauge" not in stats:
                parts.append(f"| `{name}` | — | — | — | — | — | — | **{stats['n_errored']} cases** |\n")
                continue
            pressure = stats["pressure_median_error_matched_gauge"]
            parts.append(
                f"| `{name}` | {stats['tension_median_error_published']:.2e} "
                f"| **{stats['tension_median_error_matched_gauge']:.2e}** "
                f"| {stats['tension_worst_case_matched_gauge']:.2e} ({stats['tension_worst_case']}) "
                f"| {'—' if pressure is None else f'{pressure:.2e}'} "
                f"| {stats['gauge_scale_median']:.4f} "
                f"| {stats['max_abs_mean_tension_deviation']:.2e} "
                f"| {stats['n_errored']} |\n",
            )
        parts.append(_markdown_normalisation_table(group_summary))
    return "".join(parts)


def _markdown_normalisation_table(group_summary: dict) -> str:
    """Render the force-balance work's normalisation comparison for one group, or nothing if absent."""
    normalisations = group_summary.get("force_balance_normalisations") or {}
    if not normalisations:
        return ""
    caption = (
        "\n**`ForceBalance` normalisation comparison (the force-balance work).** One estimator, three gauge "
        "treatments; the tension errors are the three `ForceBalance*` rows above. `error/std` is "
        "the median of `|error| / std` — near 1 would mean the i.i.d. force-noise model is "
        "calibrated, and far above 1 means the residual is systematic and the error bars are "
        "optimistic.\n\n"
        "**Warning.** On `dw3d`-reconstructed meshes the operator this family inverts has "
        "`rho_free` about 50x the reference meshes' own (median 0.546 against 0.0118 on the "
        "40-case equilibrium cohort, all 40 above the 0.2 alarm threshold), so every "
        "`ForceBalance*` row computed on a reconstruction, here or in any variant comparison, "
        "stands on a floor no gauge treatment can lower; the angle-reading rows do not.\n\n"
        "| normalisation | max \\|1 - mean(g)\\| | median cond | median std | median error/std | median rho |\n"
        "|---|---|---|---|---|---|\n"
    )
    parts = [caption]
    for name, stats in sorted(normalisations.items()):
        parts.append(
            f"| `{name}` | {stats['max_mean_tension_deviation']:.2e} "
            f"| {stats['median_condition_number']:.2e} "
            f"| {stats['median_tension_std']:.2e} "
            f"| {stats['median_error_over_std']:.1f} "
            f"| {stats['median_rho_at_inferred_tensions']:.4f} |\n",
        )
    return "".join(parts)


def main(case_ids: list[str] | None = None, dataset_dir: Path = DATASET_DIR) -> None:
    """Run the harness over `case_ids` and write one baseline JSON record per case.

    Args:
        case_ids (list[str] | None, optional): Cases to run. Defaults to every case found
            in `dataset_dir`.
        dataset_dir (Path, optional): Dataset directory. Defaults to `DATASET_DIR`.
    """
    case_ids = case_ids if case_ids is not None else discover_case_ids(dataset_dir)
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    written = []
    records = []
    for case_id in case_ids:
        mesh_path = dataset_dir / f"{case_id}_mesh.rec"
        tensions_path = dataset_dir / f"{case_id}_dict_tensions.npy"
        pressures_path = dataset_dir / f"{case_id}_dict_pressures.npy"
        if not (mesh_path.exists() and tensions_path.exists() and pressures_path.exists()):
            print(f"  SKIP {case_id}: missing input file(s) under {dataset_dir}")
            continue

        record = run_case(mesh_path, tensions_path, pressures_path)
        record["case"] = case_id
        out_path = BASELINE_DIR / f"case_{case_id}.json"
        out_path.write_text(json.dumps(_json_safe(record), indent=2, sort_keys=True) + "\n")
        rho = record["mesh_diagnostics"]["rho"]
        n_interfaces = record["mesh_diagnostics"]["n_interfaces"]
        n_failed = sum("error" in method for method in record["methods"].values())
        print(
            f"  {case_id}: [{case_groups.group_of(case_id)}] n_interfaces={n_interfaces:2d} "
            f"rho={rho:.4g} methods_errored={n_failed} -> {out_path.name}",
        )
        written.append(case_id)
        records.append(record)

    print(f"Wrote {len(written)} baseline JSON files to {BASELINE_DIR}")

    if not records:
        return

    # The method comparison is an accuracy claim, so it is written per equilibrium group and
    # never as one aggregate over the whole dataset (the recorded decision to exclude the five
    # non-equilibrium cases; foambryo_benchmarks/case_groups.py).
    summary = summarise_methods(records)
    (BASELINE_DIR / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n")
    (BASELINE_DIR / "method_comparison.md").write_text(markdown_method_table(summary))

    for group in case_groups.GROUP_ORDER:
        if group not in summary["groups"]:
            continue
        group_summary = summary["groups"][group]
        print()
        print(f"[{group}] {group_summary['n_cases']} cases — {group_summary['group_note']}")
        for name, stats in sorted(
            group_summary["methods"].items(),
            key=lambda item: item[1].get("tension_median_error_matched_gauge", float("inf")),
        ):
            if "tension_median_error_matched_gauge" not in stats:
                print(f"  {name:22s} ERRORED on all {stats['n_errored']} cases: {stats['error_example']}")
                continue
            print(
                f"  {name:22s} published {stats['tension_median_error_published']:.2e} "
                f"matched {stats['tension_median_error_matched_gauge']:.2e} "
                f"worst {stats['tension_worst_case_matched_gauge']:.2e} ({stats['tension_worst_case']}) "
                f"max|1-mean(g)| {stats['max_abs_mean_tension_deviation']:.2e}",
            )
    print(f"\nWrote summary.json + method_comparison.md to {BASELINE_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR, help="benchmarking-dataset directory")
    parser.add_argument(
        "--case-ids",
        type=str,
        nargs="*",
        default=None,
        help=f"subset of case ids (default: all). The initial smoke-test set was {SMOKE_TEST_CASE_IDS}.",
    )
    args = parser.parse_args()
    main(args.case_ids, args.dataset_dir)
