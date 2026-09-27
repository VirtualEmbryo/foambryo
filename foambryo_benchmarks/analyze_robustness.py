#!/usr/bin/env python
"""CLI: summarise `foambryo_benchmarks/run_robustness.py`'s records (the robustness work).

The attribution benchmark's headline, at the 2-10-cell scale of the 42 equilibrium cases: meshing
error dominates inference error (`E_inference` on the ground-truth mesh 2e-5 to 2e-2, `E_total` on
the `dw3d` reconstruction 0.05-0.80) -- by three to four orders of magnitude for the angle-reading
methods, but only by a factor of about 1.2-2.6 for `ForceBalanceKKT`/`Variational` on the dithered
meshes (`v0_3`, dw3d's default), and the ranking inverts at 91-166 cells. So
this reports degradation of `E_total`/`E_meshing` (not "which method wins") as the primary
question, and separately checks whether the *method ranking* established on the unperturbed
mask survives each perturbation - two different questions,
kept in two different functions so neither answer is read off the other's numbers.

Every summary reports `n_present` (records that carried a numeric `e_total`) and raises
rather than silently reporting a vacuous aggregate over zero records.

Usage:
    uv run python -m foambryo_benchmarks.analyze_robustness --records foambryo_benchmarks/baseline/robustness.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from foambryo_benchmarks import case_groups
from foambryo_benchmarks import mesh_populations as pop
from foambryo_benchmarks.run_robustness import CONDITIONS


def _present(records: list[dict]) -> list[dict]:
    """Records that carry a numeric `e_total` (i.e. did not fail)."""
    return [r for r in records if "e_total" in r and "error" not in r]


def _require_nonempty(records: list[dict], context: str) -> list[dict]:
    if not records:
        message = f"0 records present for {context} - cannot summarise an empty aggregate"
        raise ValueError(message)
    return records


def degradation_by_condition(records: list[dict]) -> dict:
    """Median `E_total`/`E_meshing`/`rho_free_p2` per (variant, method, condition).

    Also reports the ratio to that (variant, method)'s own `"baseline"` median - the
    degradation factor.

    Args:
        records (list[dict]): Robustness records, all tagged `P2_dw3d_reconstruction`.

    Returns:
        dict: `{group: {variant: {method: {condition: {...}}}}}`. `e_inference_median` is
            reported too, and by the attribution's non-additivity caveat should be ~constant across
            conditions for a fixed (variant, method) since P1 is never perturbed - a
            non-constant `e_inference` across conditions for the same method would itself be
            a bug worth flagging, so it is reported rather than assumed away.
    """
    pop.require_single_population(records)
    by_group_variant_method: dict[tuple[str, str, str], list[dict]] = {}
    for r in records:
        group = case_groups.group_of(r["case"])
        by_group_variant_method.setdefault((group, r["dw3d_variant"], r["method"]), []).append(r)

    summary: dict = {}
    for (group, variant, method), recs in by_group_variant_method.items():
        by_condition: dict[str, list[dict]] = {}
        for r in recs:
            by_condition.setdefault(r["condition"], []).append(r)

        baseline_present = _present(by_condition.get("baseline", []))
        baseline_median = float(np.median([r["e_total"] for r in baseline_present])) if baseline_present else None

        condition_summary = {}
        for condition, condition_records in by_condition.items():
            present = _present(condition_records)
            e_total_median = float(np.median([r["e_total"] for r in present])) if present else None
            condition_summary[condition] = {
                "n_cases": len(condition_records),
                "n_present": len(present),
                "e_total_median": e_total_median,
                "e_inference_median": float(np.median([r["e_inference"] for r in present])) if present else None,
                "e_meshing_median": float(np.median([r["e_meshing"] for r in present])) if present else None,
                "rho_free_p2_median": float(np.median([r["rho_free_p2"] for r in present])) if present else None,
                "degradation_vs_baseline": (
                    e_total_median / baseline_median if (e_total_median is not None and baseline_median) else None
                ),
            }
        summary.setdefault(group, {}).setdefault(variant, {})[method] = condition_summary
    return summary


def method_ranking_by_condition(records: list[dict]) -> dict:
    """Whether the baseline method ranking (by median `E_total`) survives each perturbation.

    Args:
        records (list[dict]): Robustness records.

    Returns:
        dict: `{group: {variant: {condition: {"ranking": [method, ...] sorted best-to-worst,
            "matches_baseline_order": bool, "n_methods": int}}}}`. `matches_baseline_order`
            compares the full ordering, not just the top method - a ranking that swaps the
            bottom two methods still "changes", and that is reported.
    """
    pop.require_single_population(records)
    by_group_variant_condition: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    for r in _present(records):
        key = (case_groups.group_of(r["case"]), r["dw3d_variant"], r["condition"])
        by_group_variant_condition.setdefault(key, {}).setdefault(r["method"], []).append(r["e_total"])

    result: dict = {}
    baseline_orderings: dict[tuple[str, str], list[str]] = {}
    # First pass: baselines, so every condition can be compared against its own variant's baseline.
    for (group, variant, condition), by_method in by_group_variant_condition.items():
        if condition != "baseline":
            continue
        ordering = sorted(by_method, key=lambda m: np.median(by_method[m]))
        baseline_orderings[(group, variant)] = ordering

    for (group, variant, condition), by_method in by_group_variant_condition.items():
        ordering = sorted(by_method, key=lambda m: np.median(by_method[m]))
        baseline_ordering = baseline_orderings.get((group, variant))
        result.setdefault(group, {}).setdefault(variant, {})[condition] = {
            "ranking": ordering,
            "matches_baseline_order": (ordering == baseline_ordering) if baseline_ordering else None,
            "n_methods": len(ordering),
        }
    return result


def instability_spatial_summary(records: list[dict]) -> dict:
    """Aggregate the spatial (inner/mid/outer) instability bins across cases.

    Tests the foambryo paper's claim that flagged-unstable
    junctions concentrate at the embryo centre, where junctions are short. Pools every
    junction from every case sharing a (group, variant, condition, method) rather than
    averaging pre-aggregated fractions, so a case with more junctions contributes
    proportionally more evidence.

    Args:
        records (list[dict]): Robustness records.

    Returns:
        dict: `{group: {variant: {condition: {method: {"n_cases_with_instability_data",
            "bins": [{"bin", "n", "n_unstable", "fraction_unstable", "median_length"}, ...]}}}}}`.
            A `(group, variant, condition, method)` cell with zero cases carrying instability
            data reports `"n_cases_with_instability_data": 0` and `"bins": None` rather than
            an empty-looking zero.
    """
    pop.require_single_population(records)
    by_key: dict[tuple[str, str, str, str], list[dict]] = {}
    for r in records:
        instability = r.get("instability")
        if not isinstance(instability, dict) or "spatial" not in instability:
            continue
        key = (case_groups.group_of(r["case"]), r["dw3d_variant"], r["condition"], r["method"])
        by_key.setdefault(key, []).append(instability)

    result: dict = {}
    for (group, variant, condition, method), instabilities in by_key.items():
        n_bins = len(instabilities[0]["spatial"])
        pooled_bins = []
        for bin_index in range(n_bins):
            entries = [inst["spatial"][bin_index] for inst in instabilities if inst["spatial"][bin_index]["n"] > 0]
            n_total = sum(e["n"] for e in entries)
            n_unstable_total = sum(e["n_unstable"] for e in entries)
            pooled_bins.append(
                {
                    "bin": bin_index,
                    "n": n_total,
                    "n_unstable": n_unstable_total,
                    "fraction_unstable": (n_unstable_total / n_total) if n_total > 0 else None,
                    "median_length": (
                        float(np.median([e["median_length"] for e in entries])) if entries else None
                    ),
                },
            )
        result.setdefault(group, {}).setdefault(variant, {}).setdefault(condition, {})[method] = {
            "n_cases_with_instability_data": len(instabilities),
            "bins": pooled_bins,
        }
    return result


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    default_records = Path(__file__).resolve().parent / "baseline" / "robustness.json"
    parser.add_argument("--records", type=Path, default=default_records)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    records = json.loads(args.records.read_text())
    _require_nonempty(records, "the whole robustness grid")
    n_present = len(_present(records))
    print(f"{len(records)} records loaded, {n_present} present (non-error), conditions: {CONDITIONS}")

    summary = {
        "n_records": len(records),
        "n_present": n_present,
        "degradation_by_condition": degradation_by_condition(records),
        "method_ranking_by_condition": method_ranking_by_condition(records),
        "instability_spatial_summary": instability_spatial_summary(records),
    }
    text = json.dumps(summary, indent=2)
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
