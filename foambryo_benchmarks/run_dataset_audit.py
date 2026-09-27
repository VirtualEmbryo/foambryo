#!/usr/bin/env python
r"""Integrity and equilibrium audit of `benchmarking-dataset`'s 47 ground-truth meshes.

This audit is what the recorded decision to exclude the five non-equilibrium cases rests on
(`foambryo_benchmarks.case_groups`). Runs, on every case, three checks in order — integrity,
equilibrium, identifiability, the order the *technical note on tension inference (H. Turlier,
in preparation)* recommends — plus the effective-tension comparison:

1. **integrity** — dangling vertex references; every triple line shared by exactly 3
   interfaces (`foambryo.validation.mesh_io.mesh_integrity`);
2. **equilibrium** — `rho = ||P_perp b|| / ||M_gamma gamma||` at the *nominal* tensions
   with pressures marginalised, and `rho_free`, the same residual once the tensions are
   themselves free (the floor `rho` cannot go below);
3. **identifiability** — the singular spectrum of the column-normalised operator,
   `sigma_soft`, and its block composition;
4. **effective tensions** — which `gamma` balance each mesh, compared to nominal per
   interface, in three explicitly-stated gauges, looking for a factor on a subset of
   interfaces (for example, some line tensions encoded at half their value by the program
   that generated the mesh): a subset at 0.5 or 2.0 against a bulk at 1.0.

Also reported, because it is what turns a large `rho` into a diagnosis:
**residual localisation** — whether the unbalanced force sits at the triple lines (expected;
Young-Dupré cannot hold exactly on a triangulation) or is spread over the interface interiors
(a surface relaxation that never converged). See `foambryo_benchmarks.metrics.residual_localisation`.

The operator is the harness default `beta = (gamma, p)`, `M = [M_gamma | M_p]` — no
line-tension block, by design.

**Gauge:** `rho`, `rho_free`, the spectrum and the localisation fractions are all
gauge-invariant; see `foambryo_benchmarks.metrics.dataset_audit` for the argument. The
`gamma_eff / gamma_nominal` ratio is not, and is reported in a `mean_one`, a
`least_squares` and a `median_one` gauge — the last being the one in which a factor applied
to a *minority* of interfaces is actually visible, which the other two dilute. See
`foambryo_benchmarks.metrics.effective_tension_ratios`.

Usage:
    uv run python -m foambryo_benchmarks.run_dataset_audit                    # all 47 cases
    uv run python -m foambryo_benchmarks.run_dataset_audit --case-ids 000 001 # a subset
    uv run python -m foambryo_benchmarks.run_dataset_audit --no-mask-checks   # skip reading the .tif masks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from foambryo_benchmarks import case_groups
from foambryo_benchmarks import metrics as m
from foambryo_benchmarks.run_case import _json_safe
from foambryo.validation import mesh_io

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent
DATASET_DIR = WORKSPACE_ROOT / "benchmarking-dataset"
AUDIT_DIR = REPO_ROOT / "foambryo_benchmarks" / "baseline" / "dataset_audit"

# Ratio histogram bins for the pooled gamma_eff/gamma_nominal distribution. Chosen so that
# 0.5, 1.0 and 2.0 each fall at a bin *centre* rather than on an edge, which is what makes a
# spike at the factor-2 signature visible instead of split across two bins.
RATIO_BIN_EDGES = np.round(np.arange(0.0, 2.60, 0.05), 4)


def discover_case_ids(dataset_dir: Path) -> list[str]:
    """List the case ids that have all three files this audit needs.

    Args:
        dataset_dir (Path): Directory holding `{case_id}_mesh.rec` etc.

    Returns:
        list[str]: Sorted case ids.
    """
    ids = []
    for mesh_path in sorted(dataset_dir.glob("*_mesh.rec")):
        case_id = mesh_path.name.removesuffix("_mesh.rec")
        tensions = dataset_dir / f"{case_id}_dict_tensions.npy"
        if tensions.exists():
            ids.append(case_id)
    return ids


def mask_checks(mask_path: Path) -> dict:
    """Report which labels touch the image border, from the segmentation mask.

    This is the candidate explanatory covariate "cells touching the image border" for any
    case that fails the equilibrium check: a cell clipped by the volume boundary is not a
    closed capillary body, so its share of the force balance is incomplete by construction
    and no tension can balance it.

    Read from the mask rather than inferred from the mesh deliberately — the `.rec` meshes
    are in normalised (roughly unit-cube) coordinates, so a mesh bounding box cannot be
    compared to the mask extent without reconstructing the normalisation.

    Args:
        mask_path (Path): Path to `{case_id}_labels_filled.tif`.

    Returns:
        dict: `{"mask_shape", "mask_labels", "border_labels", "border_cells",
            "n_border_cells"}`, where `border_cells` excludes the exterior medium (0).
    """
    from skimage import io as skio

    labels = skio.imread(mask_path)
    border: set[int] = set()
    for axis in range(labels.ndim):
        border |= set(np.unique(labels.take([0, -1], axis=axis)).tolist())
    border_cells = sorted(label for label in border if label != 0)
    return {
        "mask_shape": list(labels.shape),
        "mask_labels": sorted(np.unique(labels).tolist()),
        "border_labels": sorted(border),
        "border_cells": border_cells,
        "n_border_cells": len(border_cells),
    }


def audit_case(dataset_dir: Path, case_id: str, *, with_mask_checks: bool = True) -> dict:
    """Audit one case.

    Args:
        dataset_dir (Path): Directory holding `{case_id}_mesh.rec` etc.
        case_id (str): Case id, e.g. `"000"`.
        with_mask_checks (bool, optional): Also read `{case_id}_labels_filled.tif` to find
            cells touching the image border. Defaults to `True`.

    Returns:
        dict: The per-case audit record.
    """
    mesh_path = dataset_dir / f"{case_id}_mesh.rec"
    dict_truth_tensions = np.load(dataset_dir / f"{case_id}_dict_tensions.npy", allow_pickle=True).item()
    dict_truth_tensions = {tuple(int(x) for x in k): float(val) for k, val in dict_truth_tensions.items()}

    v, f, regions = mesh_io.read_rec(mesh_path)
    record = m.dataset_audit(v, f, regions, dict_truth_tensions)
    record["case"] = case_id
    record["nominal_tensions"] = dict_truth_tensions

    mask_path = dataset_dir / f"{case_id}_labels_filled.tif"
    if with_mask_checks and mask_path.exists():
        record["mask"] = mask_checks(mask_path)
    return record


def summarise(records: list[dict]) -> dict:
    """Aggregate the per-case records, **separately for each equilibrium group**.

    The recorded decision is that the five non-equilibrium cases are excluded from accuracy
    claims and kept as a negative control, so a single distribution over all 47 would mix a
    data defect into every statistic. This function therefore returns one summary per group
    (`foambryo_benchmarks.case_groups`), plus a `dataset` block holding only the genuinely
    group-independent counts. There is no all-47 aggregate of `rho`, the residual
    localisation, the identifiability spectrum or the effective-tension ratios.

    Args:
        records (list[dict]): Per-case audit records, either group or both.

    Returns:
        dict: `{"dataset": {...}, "groups": {"equilibrium": {...}, "non_equilibrium": {...}}}`.
    """
    grouped = case_groups.split_records(records)
    return {
        "dataset": {
            "n_cases": len(records),
            "n_by_group": {group: len(items) for group, items in grouped.items()},
            "group_notes": {group: case_groups.group_note(group) for group in case_groups.GROUP_ORDER},
            # Integrity is a property of the mesh file, not of whether it was relaxed, so it
            # is the one count that is meaningful over the whole dataset.
            "integrity": {
                "n_intact": sum(bool(r["integrity"]["intact"]) for r in records),
                "not_intact": [r["case"] for r in records if not r["integrity"]["intact"]],
                "n_with_boundary_edges": sum(r["integrity"]["n_boundary_edges"] > 0 for r in records),
                "n_with_quadjunction_edges": sum(r["integrity"]["n_quadjunction_edges"] > 0 for r in records),
                "n_with_duplicate_faces": sum(r["integrity"]["n_duplicate_faces"] > 0 for r in records),
                "n_with_degenerate_faces": sum(r["integrity"]["n_degenerate_faces"] > 0 for r in records),
                "n_with_nonmanifold_interface_edges": sum(
                    r["integrity"]["n_nonmanifold_interface_edges"] > 0 for r in records
                ),
            },
        },
        "groups": {
            group: _summarise_group(items) for group, items in grouped.items() if items
        },
    }


def _summarise_group(records: list[dict]) -> dict:
    """Summarise records from a **single** equilibrium group.

    Args:
        records (list[dict]): Per-case audit records, all from one group.

    Returns:
        dict: Distribution of `rho`, the pooled `gamma_eff/gamma_nominal` histogram in all
            three gauges, integrity counts, and identifiability summary.

    Raises:
        ValueError: (from `case_groups.require_single_group`) if the records span both groups.
    """
    group = case_groups.require_single_group(records)
    rho = np.array([r["rho"] for r in records])
    rho_free = np.array([r["rho_free"] for r in records])
    sigma_soft = np.array([r["sigma_soft"] for r in records])
    sigma_gauge = np.array([r["sigma_gauge"] for r in records])
    interior = np.array([r["residual_localisation"]["fraction_interface_interior"] for r in records])

    pooled: dict[str, dict] = {}
    for gauge in ("mean_one", "least_squares", "median_one"):
        ratios = np.array(
            [val for r in records for val in r["effective_tensions"]["gauges"][gauge]["per_key"].values()],
        )
        # gamma_nominal per interface, in the same order, so the ratio can be converted back
        # to an absolute deviation: a uniform additive error and a multiplicative defect look
        # identical in the ratio alone (see `metrics.effective_tension_ratios`).
        nominal = np.array(
            [
                r["nominal_tensions"][key] / np.mean(list(r["nominal_tensions"].values()))
                for r in records
                for key in r["effective_tensions"]["gauges"][gauge]["per_key"]
            ],
        )
        deviation = np.abs(ratios - 1.0) * nominal
        counts, _ = np.histogram(ratios, bins=RATIO_BIN_EDGES)
        pooled[gauge] = {
            "n": int(ratios.size),
            "abs_deviation_median": float(np.median(deviation)),
            "abs_deviation_p95": float(np.percentile(deviation, 95)),
            "abs_deviation_max": float(deviation.max()),
            "min": float(ratios.min()),
            "p05": float(np.percentile(ratios, 5)),
            "median": float(np.median(ratios)),
            "p95": float(np.percentile(ratios, 95)),
            "max": float(ratios.max()),
            "n_outside_0.05": int((np.abs(ratios - 1.0) >= 0.05).sum()),
            "n_near_half": int((np.abs(ratios - 0.5) < 0.05 * 0.5).sum()),
            "n_near_one": int((np.abs(ratios - 1.0) < 0.05).sum()),
            "n_near_two": int((np.abs(ratios - 2.0) < 0.05 * 2.0).sum()),
            "histogram": {"bin_edges": RATIO_BIN_EDGES.tolist(), "counts": counts.tolist()},
        }

    return {
        "group": group,
        "group_note": case_groups.group_note(group),
        "n_cases": len(records),
        "cases": [r["case"] for r in records],
        "integrity": {
            "n_intact": sum(bool(r["integrity"]["intact"]) for r in records),
            "not_intact": [r["case"] for r in records if not r["integrity"]["intact"]],
            "n_with_boundary_edges": sum(r["integrity"]["n_boundary_edges"] > 0 for r in records),
            "n_with_quadjunction_edges": sum(r["integrity"]["n_quadjunction_edges"] > 0 for r in records),
            "n_with_duplicate_faces": sum(r["integrity"]["n_duplicate_faces"] > 0 for r in records),
            "n_with_degenerate_faces": sum(r["integrity"]["n_degenerate_faces"] > 0 for r in records),
            "n_with_nonmanifold_interface_edges": sum(
                r["integrity"]["n_nonmanifold_interface_edges"] > 0 for r in records
            ),
        },
        "rho": {
            "min": float(rho.min()),
            "p25": float(np.percentile(rho, 25)),
            "median": float(np.median(rho)),
            "p75": float(np.percentile(rho, 75)),
            "p95": float(np.percentile(rho, 95)),
            "max": float(rho.max()),
            "mean": float(rho.mean()),
            "n_above_0.10": int((rho > 0.10).sum()),
            "n_above_0.20": int((rho > 0.20).sum()),
            "worst_cases": [records[i]["case"] for i in np.argsort(-rho)[:5]],
        },
        "rho_free": {
            "min": float(rho_free.min()),
            "median": float(np.median(rho_free)),
            "max": float(rho_free.max()),
        },
        "residual_localisation": {
            "fraction_interface_interior": {
                "min": float(interior.min()),
                "median": float(np.median(interior)),
                "max": float(interior.max()),
            },
            # The discriminator: cases whose residual sits in the interfaces rather than at
            # the triple lines have not converged. See `metrics.residual_localisation`.
            "n_interior_dominated": int((interior > 0.5).sum()),
            "interior_dominated_cases": [
                r["case"]
                for r in records
                if r["residual_localisation"]["fraction_interface_interior"] > 0.5
            ],
        },
        "identifiability": {
            "sigma_gauge_max": float(sigma_gauge.max()),
            "sigma_soft_min": float(sigma_soft.min()),
            "sigma_soft_median": float(np.median(sigma_soft)),
            "sigma_soft_max": float(sigma_soft.max()),
            "amplification_median": float(np.median(1.0 / sigma_soft)),
            "amplification_max": float((1.0 / sigma_soft).max()),
        },
        "effective_tension_ratio_pooled": pooled,
    }


def markdown_table(records: list[dict]) -> str:
    """Render the audit table, one row per case, **grouped by the equilibrium split**.

    The 42 equilibrium cases come first, then a separator row, then the 5 non-equilibrium
    cases, so the table cannot be read as one population by accident.

    Args:
        records (list[dict]): Per-case audit records.

    Returns:
        str: A GitHub-flavoured Markdown table.
    """
    header = (
        f"**Equilibrium split (the recorded decision to exclude the five non-equilibrium cases).** "
        f"Rows are grouped: first the cases at mechanical "
        f"equilibrium — {case_groups.group_note(case_groups.EQUILIBRIUM)} — then the "
        f"{case_groups.group_note(case_groups.NON_EQUILIBRIUM)}. Do not aggregate across the "
        "separator row; `summary.json` reports the two groups separately for the same reason. "
        "See `foambryo_benchmarks/case_groups.py`.\n\n"
        "Gauge: `rho`, `rho_free`, `sigma_soft` and `res interior` are gauge-invariant "
        "(each is a ratio of like quantities). The `ratio` columns are "
        "`gamma_eff/gamma_nominal` in the **mean-one gauge** (both vectors rescaled to mean "
        "1). The `least_squares` and `median_one` gauges are in the per-case JSON and in "
        "`summary.json`; `median_one` is the one that can see a factor applied to a minority "
        "of interfaces. See `foambryo_benchmarks.metrics.effective_tension_ratios`.\n\n"
        "`res interior` is the fraction of the squared residual carried by interface-interior "
        "vertices. High means the interfaces themselves are not at equilibrium (not "
        "converged); low means the residual sits at the triple lines, which is expected "
        "discretisation error. See `foambryo_benchmarks.metrics.residual_localisation`.\n\n"
        "| case | n_cells | n_iface | n_junc | intact? | rho | rho_free | res interior "
        "| sigma_soft | ratio min | ratio med | ratio max | border cells | bnd edges | quad edges |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    grouped = case_groups.split_records(records)
    rows = []
    for group in case_groups.GROUP_ORDER:
        if not grouped[group]:
            continue
        if rows:  # separator between the two populations
            rows.append(f"| **{group}** | | | | | | | | | | | | | | |")
        for r in grouped[group]:
            gauge = r["effective_tensions"]["gauges"]["mean_one"]
            mask = r.get("mask", {})
            interior = r["residual_localisation"]["fraction_interface_interior"]
            rows.append(
                f"| {r['case']} | {r['n_cells']} | {r['n_interfaces']} | {r['n_junctions']} "
                f"| {'yes' if r['integrity']['intact'] else '**NO**'} "
                f"| {r['rho']:.4f} | {r['rho_free']:.4f} | {interior:.2f} | {r['sigma_soft']:.4f} "
                f"| {gauge['min']:.3f} | {gauge['median']:.3f} | {gauge['max']:.3f} "
                f"| {mask.get('n_border_cells', '-')} "
                f"| {r['integrity']['n_boundary_edges']} | {r['integrity']['n_quadjunction_edges']} |",
            )
    return header + "\n".join(rows) + "\n"


def main() -> None:
    """Run the audit over the dataset and write per-case JSON, a summary and a table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR, help="benchmarking-dataset directory")
    parser.add_argument("--output-dir", type=Path, default=AUDIT_DIR, help="where to write the audit records")
    parser.add_argument("--case-ids", type=str, nargs="*", default=None, help="subset of case ids (default: all)")
    parser.add_argument(
        "--no-mask-checks",
        action="store_false",
        dest="with_mask_checks",
        help="skip reading the .tif masks (border-touching cells will not be reported)",
    )
    args = parser.parse_args()

    case_ids = args.case_ids if args.case_ids else discover_case_ids(args.dataset_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for case_id in case_ids:
        record = audit_case(args.dataset_dir, case_id, with_mask_checks=args.with_mask_checks)
        (args.output_dir / f"case_{case_id}.json").write_text(
            json.dumps(_json_safe(record), indent=2, sort_keys=True) + "\n",
        )
        records.append(record)
        gauge = record["effective_tensions"]["gauges"]["mean_one"]
        print(
            f"  {case_id}: n_cells={record['n_cells']:2d} n_iface={record['n_interfaces']:2d} "
            f"intact={record['integrity']['intact']!s:5s} rho={record['rho']:.4f} "
            f"rho_free={record['rho_free']:.4f} "
            f"res_interior={record['residual_localisation']['fraction_interface_interior']:.2f} "
            f"sigma_soft={record['sigma_soft']:.4f} "
            f"ratio=[{gauge['min']:.3f}, {gauge['max']:.3f}]",
        )

    summary = summarise(records)
    (args.output_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n")
    (args.output_dir / "table.md").write_text(markdown_table(records))

    print()
    dataset = summary["dataset"]
    print(f"{dataset['integrity']['n_intact']}/{dataset['n_cases']} cases intact (integrity is group-independent)")
    for group in case_groups.GROUP_ORDER:
        if group not in summary["groups"]:
            continue
        group_summary = summary["groups"][group]
        print()
        print(f"[{group}] {group_summary['n_cases']} cases — {group_summary['group_note']}")
        print(
            f"  rho: min {group_summary['rho']['min']:.4f} median {group_summary['rho']['median']:.4f} "
            f"max {group_summary['rho']['max']:.4f} | >0.10: {group_summary['rho']['n_above_0.10']} cases",
        )
        localisation = group_summary["residual_localisation"]
        print(
            f"  interior-dominated residual (not converged): {localisation['n_interior_dominated']} cases "
            f"{localisation['interior_dominated_cases']}",
        )
        for gauge, stats in group_summary["effective_tension_ratio_pooled"].items():
            print(
                f"  gamma_eff/gamma_nominal [{gauge} gauge]: n={stats['n']} "
                f"median {stats['median']:.4f} range [{stats['min']:.3f}, {stats['max']:.3f}] "
                f"| near 0.5: {stats['n_near_half']} near 1.0: {stats['n_near_one']} "
                f"near 2.0: {stats['n_near_two']} "
                f"| |d gamma| median {stats['abs_deviation_median']:.4f} p95 {stats['abs_deviation_p95']:.4f}",
            )
    print(f"\nWrote {len(records)} records + summary.json + table.md to {args.output_dir}")


if __name__ == "__main__":
    main()
