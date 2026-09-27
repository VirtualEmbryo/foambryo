#!/usr/bin/env python
"""CLI: summarise `foambryo_benchmarks/run_attribution.py`'s attribution records into the headline table.

Reports the 42 equilibrium cases (the headline) and the 5 non-equilibrium cases separately
by construction (the recorded decision to exclude the five non-equilibrium cases,
`foambryo_benchmarks/case_groups.py`) - never an aggregate over all 47.
Every record is asserted `population = P2_dw3d_reconstruction`
(`foambryo_benchmarks.mesh_populations.require_single_population`) before it is aggregated, and every
aggregate reports `n_present` (how many records actually carried a numeric `e_total`, not
an `"error"` key) rather than silently dividing by a count that includes failures.

**A `"present"` record can still be undetermined.** A mesh can have a fully closed,
triple-line-free interface, which some methods report as `nan` rather than a number
(`foambryo.junction_graph`) - that record still has a numeric `e_total` key (it is not an
`"error"` record), but the number is `nan`. `summarise_grid` never lets that `nan` reach a
bare `np.median` (which would silently turn the whole group into `nan`) or a silent
`np.nanmedian` (which would silently drop it); see `foambryo_benchmarks.metrics.
median_of_determined`, and every `*_n_determined`/`*_n_total` pair this function returns
alongside each median.

Usage:
    uv run python -m foambryo_benchmarks.analyze_attribution --records foambryo_benchmarks/baseline/attribution.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from foambryo_benchmarks import case_groups
from foambryo_benchmarks import mesh_populations as pop
from foambryo_benchmarks.metrics import median_of_determined


def _present(records: list[dict]) -> list[dict]:
    """Records that carry a numeric `e_total` (i.e. did not fail), asserted non-vacuously."""
    return [r for r in records if "e_total" in r and "error" not in r]


#: Fields `summarise_grid` medians over, each independently -- `rho_free_p1`/`rho_free_p2`
#: are intrinsic to a mesh and never `nan` even when a tension is undetermined, so a field's
#: own `n_determined` can differ from another field's on the same group of records.
_GRID_FIELDS = ("e_total", "e_inference", "e_meshing", "rho_free_p1", "rho_free_p2")


def summarise_grid(records: list[dict]) -> dict:
    """Median `E_total` / `E_inference` / `E_meshing`, split by (variant, method, group).

    Every `{field}_median` is computed by `median_of_determined(..., allow_partial=True)`,
    never a bare `np.median` or a silent `np.nanmedian` -- so a group containing one
    undetermined (`nan`) record still yields a real median over the rest, explicitly paired
    with `{field}_n_determined`/`{field}_n_total` rather than silently becoming `nan` itself
    or silently dropping the undetermined case with no trace. A caller that renders any of
    these medians into a table **must** show `n_determined`/`n_total` next to it whenever
    `n_determined < n_total` -- see this module's own docstring.

    Args:
        records (list[dict]): Attribution records, all tagged `P2_dw3d_reconstruction`.

    Returns:
        dict: `{group: {variant: {method: {...}}}}`.
    """
    pop.require_single_population(records)
    by_variant_method: dict[tuple[str, str], list[dict]] = {}
    for r in records:
        by_variant_method.setdefault((r["dw3d_variant"], r["method"]), []).append(r)

    summary: dict = {group: {} for group in case_groups.GROUP_ORDER}
    for (variant, method), group_records in by_variant_method.items():
        split = case_groups.split_records(group_records)
        for group, recs in split.items():
            present = _present(recs)
            entry: dict = {"n_cases": len(recs), "n_present": len(present)}
            for field in _GRID_FIELDS:
                agg = median_of_determined([r[field] for r in present], allow_partial=True)
                entry[f"{field}_median"] = agg["median"]
                entry[f"{field}_n_determined"] = agg["n_determined"]
                entry[f"{field}_n_total"] = agg["n_total"]
            summary[group].setdefault(variant, {})[method] = entry
    return summary


def rho_free_correlation(records: list[dict]) -> dict:
    """Spearman correlation of `E_total` with `rho_free_p2`, per (variant, method), per group.

    The prediction tested: a method whose error tracks `rho_free` is being limited by the
    mesh, not by itself. Computed per group
    because `rho_free_p2` may differ systematically between the 42 and the 5 in a way that
    would confound a pooled correlation.

    Args:
        records (list[dict]): Attribution records.

    Returns:
        dict: `{group: {variant: {method: {"n", "spearman_r", "spearman_p"}}}}`.
    """
    pop.require_single_population(records)
    by_variant_method: dict[tuple[str, str], list[dict]] = {}
    for r in records:
        by_variant_method.setdefault((r["dw3d_variant"], r["method"]), []).append(r)

    result: dict = {group: {} for group in case_groups.GROUP_ORDER}
    for (variant, method), group_records in by_variant_method.items():
        split = case_groups.split_records(group_records)
        for group, recs in split.items():
            present = _present(recs)
            entry = {"n": len(present), "spearman_r": None, "spearman_p": None}
            if len(present) >= 4:
                rho = [r["rho_free_p2"] for r in present]
                err = [r["e_total"] for r in present]
                r_value, p_value = spearmanr(rho, err)
                entry["spearman_r"] = float(r_value)
                entry["spearman_p"] = float(p_value)
            result[group].setdefault(variant, {})[method] = entry
    return result


def route_crossover(records: list[dict]) -> dict:
    """The route-crossover question, measured on P2: does route Q still beat route P?

    The uncertainty work found route Q beats route P on the 42 equilibrium cases (on P1) but is worse on the
    5 non-equilibrium ones - the population the 5 most resemble being `dw3d` output. This
    reports the paired per-case ratio `E_total(RouteQ) / E_total(RouteP)` on P2, by group
    and by `dw3d` variant: `< 1` means route Q is still better on `dw3d` meshes.

    Args:
        records (list[dict]): Attribution records.

    Returns:
        dict: `{group: {variant: {"n_paired", "median_ratio_q_over_p"}}}`.
    """
    pop.require_single_population(records)
    by_case_variant: dict[tuple[str, str], dict[str, dict]] = {}
    for r in records:
        if r["method"] not in ("VariationalRouteP", "VariationalRouteQ"):
            continue
        by_case_variant.setdefault((r["case"], r["dw3d_variant"]), {})[r["method"]] = r

    result: dict = {group: {} for group in case_groups.GROUP_ORDER}
    ratios_by_group_variant: dict[tuple[str, str], list[float]] = {}
    for (case_id, variant), by_method in by_case_variant.items():
        p_record = by_method.get("VariationalRouteP")
        q_record = by_method.get("VariationalRouteQ")
        if not p_record or not q_record or "e_total" not in p_record or "e_total" not in q_record:
            continue
        group = case_groups.group_of(case_id)
        ratios_by_group_variant.setdefault((group, variant), []).append(q_record["e_total"] / p_record["e_total"])

    for (group, variant), ratios in ratios_by_group_variant.items():
        result[group][variant] = {
            "n_paired": len(ratios),
            "median_ratio_q_over_p": float(np.median(ratios)),
        }
    return result


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    default_records = Path(__file__).resolve().parent / "baseline" / "attribution.json"
    parser.add_argument("--records", type=Path, default=default_records)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    records = json.loads(args.records.read_text())
    summary = {
        "grid": summarise_grid(records),
        "rho_free_correlation": rho_free_correlation(records),
        "route_crossover": route_crossover(records),
    }
    text = json.dumps(summary, indent=2)
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
