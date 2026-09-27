#!/usr/bin/env python
"""CLI: `error/std` for every uncertainty variant — the uncertainty work's calibration measurement.

The uncertainty work's deliverable is **not** "add error bars".
The force-balance work measured that the naive KKT posterior s.d. is `error/std ~ 246` — an interval wrong
by 250x *in the dangerous direction* — and the per-edge Young-Dupre work measured that an edge-resampling bootstrap
gives `error/std ~ 0.17`, about 6x conservative. Neither is right. So the deliverable is:

> produce an interval whose `error/std` is **O(1)**, or state plainly that we cannot — and
> report `error/std` for **every** variant as THE calibration metric.

This script is that report. Every variant, every case, both equilibrium groups, one JSON.

**`error/std`** is `median_i |gamma_hat_i - gamma*_i| / std_i`, gauge-matched (the rule
every accuracy comparison in this package follows: the inferred tensions *and their intervals*
are rescaled by the one factor matching the means, since only ratios are physical). 1 is calibrated; >1 optimistic; <1
conservative.

Variants, grouped by which estimator they put an interval on:

- force balance / `Variational`: `IidGls` (the naive KKT posterior), `CoherentResidual` (the
  coherent-residual candidate), `ClusterRobust`, `ClusterBootstrap`, and `BayesianMap` (the
  prior-based posterior, read
  from `foambryo.infer_force_balance` so it is the same number the baseline records carry).
- `YoungDupre`: `AnglePropagation`, `LineBootstrap`.
- `YoungDupreLocal`: `EdgeBootstrap` (the per-edge Young-Dupre work's, here extended from 2 cases to all 47).

Also reported per case: the **implied `n_eff`** — the number of independent observations the
operator's rows are *actually* worth, `dof / (error/std of IidGls)^2`. The i.i.d. model assumes
`n_eff = n_rows` (~34 000); `CoherentResidual` assumes `n_eff = 1`. The measured value is the
whole story of this phase.

Usage:
    uv run python -m foambryo_benchmarks.run_uncertainty_calibration
    uv run python -m foambryo_benchmarks.run_uncertainty_calibration --cases 000 011 --out /tmp/cal.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from foambryo_benchmarks import case_groups
from foambryo_benchmarks.run_baseline import discover_case_ids
from foambryo_benchmarks.run_per_edge_young_dupre import bootstrap_calibration
from foambryo import (
    ForceBalanceNormalisation,
    TensionComputationMethod,
    dcel_mesh_from_file,
    infer_force_balance,
)
from foambryo.uncertainty import (
    FORCE_BALANCE_MODELS,
    UncertaintyModel,
    compute_inference_diagnostics,
    error_over_std,
    infer_tensions_with_uncertainty,
)
from foambryo.validation import force_balance as fb

#: The models measured here, in report order, with the estimator each one attaches to.
VARIANTS: tuple[tuple[str, TensionComputationMethod, UncertaintyModel], ...] = (
    ("ProbeResidual", TensionComputationMethod.ForceBalance, UncertaintyModel.ProbeResidual),
    ("CoherentResidual", TensionComputationMethod.ForceBalance, UncertaintyModel.CoherentResidual),
    ("ClusterBootstrap", TensionComputationMethod.ForceBalance, UncertaintyModel.ClusterBootstrap),
    ("ClusterRobust", TensionComputationMethod.ForceBalance, UncertaintyModel.ClusterRobust),
    ("IidGls", TensionComputationMethod.ForceBalance, UncertaintyModel.IidGls),
    ("AnglePropagation", TensionComputationMethod.YoungDupre, UncertaintyModel.AnglePropagation),
    ("LineBootstrap", TensionComputationMethod.YoungDupre, UncertaintyModel.LineBootstrap),
)

#: Prior widths swept to test whether `BayesianMap`'s O(1) calibration is a choice or a result.
#: See `map_prior_sweep`. 0.3 is the shipped `MAP_PRIOR_TENSION_STD`.
MAP_PRIOR_STD_SWEEP: tuple[float, ...] = (0.03, 0.1, 0.3, 1.0)


def bayesian_map_calibration(mesh: object, truth: dict) -> dict:
    """`error/std` for the force-balance work's Gaussian-MAP posterior, and why its number is not a success.

    `ForceBalanceMAP`'s `error/std` is ~0.2 — the only variant predating this calibration work that looks calibrated.
    It is not: its interval is **prior-dominated**. `MAP_PRIOR_TENSION_STD` is 0.3 by choice,
    and the reported posterior s.d. is a large fraction of it, so the interval is mostly a
    restatement of the prior width rather than a measurement of this mesh. The diagnostic that
    shows this is `std / prior_std`: near 1 means the data added nothing.

    Args:
        mesh (object): The `DcelData` mesh.
        truth (dict): Ground-truth tensions.

    Returns:
        dict: The calibration ratios plus the prior-domination diagnostic.
    """
    from foambryo.force_balance_inference import MAP_PRIOR_TENSION_STD  # local import

    result = infer_force_balance(mesh, 1.0, ForceBalanceNormalisation.BayesianMap)
    truth_vector = np.array([truth[key] for key in result.interfaces])
    scale = float(truth_vector.mean() / result.tensions.mean())
    error = np.abs(scale * result.tensions - truth_vector)
    std = scale * result.tension_std
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(std > 0, error / std, np.nan)
    finite = ratio[np.isfinite(ratio)]
    return {
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
        "n": int(finite.size),
        "median_std": float(np.median(std)),
        "median_abs_error": float(np.median(error)),
        "prior_tension_std": MAP_PRIOR_TENSION_STD,
        "median_std_over_prior_std": float(np.median(std) / MAP_PRIOR_TENSION_STD),
    }


def coherent_bound_optimism(mesh: object, truth: dict) -> dict:
    """How much `CoherentResidual` under-states its own perturbation, measured.

    The bound uses `||M beta_hat||`, the residual at the *estimate*. The perturbation that
    actually displaced the estimate is the residual at the *truth*, `||P_perp M_gamma gamma*||`,
    which is necessarily larger because `beta_hat` minimises. This measures the ratio — the
    single quantity that explains why the bound's median `error/std` is below 1 while its p95
    is above 1.

    Args:
        mesh (object): The `DcelData` mesh.
        truth (dict): Ground-truth tensions.

    Returns:
        dict: The two residual norms and their ratio.
    """
    from foambryo.force_balance_inference import build_force_balance_operator  # local import

    m, interfaces, _cells = build_force_balance_operator(mesh)
    ni = len(interfaces)
    anchor = np.zeros(m.shape[1])
    anchor[:ni] = 1.0 / ni
    beta = fb.solve_constrained(m, anchor, np.array([1.0]))
    residual_estimate = float(np.linalg.norm(m @ beta))

    truth_vector = np.array([truth[key] for key in interfaces])
    m_gamma, _m_t, m_p = fb.split(m, ni, 0)
    load = m_gamma @ truth_vector
    pressures, *_ = np.linalg.lstsq(m_p, -load, rcond=None)
    residual_truth = float(np.linalg.norm(load + m_p @ pressures))
    return {
        "residual_at_estimate": residual_estimate,
        "residual_at_truth": residual_truth,
        "ratio_truth_over_estimate": residual_truth / residual_estimate if residual_estimate else float("nan"),
    }


def run_case(case_id: str, dataset_dir: Path, n_resamples: int, seed: int) -> dict:
    """Every calibration measurement for one case.

    Args:
        case_id (str): Case id, e.g. `"000"`.
        dataset_dir (Path): Directory holding the dataset files.
        n_resamples (int): Bootstrap draws for the bootstrap models.
        seed (int): Bootstrap seed, recorded.

    Returns:
        dict: One record.
    """
    truth = np.load(dataset_dir / f"{case_id}_dict_tensions.npy", allow_pickle=True).item()
    mesh = dcel_mesh_from_file(str(dataset_dir / f"{case_id}_mesh.rec"))
    diagnostics = compute_inference_diagnostics(mesh, truth)

    record: dict = {
        "case": case_id,
        "group": case_groups.group_of(case_id),
        "n_interfaces": diagnostics.n_interfaces,
        "n_cells": diagnostics.n_cells,
        "n_rows": diagnostics.n_rows,
        "rho": diagnostics.rho,
        "rho_free": diagnostics.rho_free,
        "sigma_soft": diagnostics.sigma_soft,
        "condition_number": diagnostics.condition_number,
        "variants": {},
    }

    for name, method, model in VARIANTS:
        start = time.perf_counter()
        result = infer_tensions_with_uncertainty(
            mesh,
            1.0,
            method=method,
            model=model,
            n_resamples=n_resamples,
            seed=seed,
        )
        calibration = error_over_std(result, truth)
        calibration["estimator"] = method.name
        calibration["wall_time_s"] = time.perf_counter() - start
        calibration["sigma_scale"] = result.sigma_scale
        if model in FORCE_BALANCE_MODELS:
            dof = max(result.n_rows - (diagnostics.n_interfaces + diagnostics.n_cells - 1), 1)
            calibration["dof"] = dof
            if model is UncertaintyModel.IidGls and calibration["median"] > 0:
                calibration["implied_n_eff"] = dof / calibration["median"] ** 2
        record["variants"][name] = calibration

    record["variants"]["BayesianMap"] = bayesian_map_calibration(mesh, truth) | {
        "estimator": "ForceBalanceMAP",
    }
    record["variants"]["EdgeBootstrap"] = _edge_bootstrap(mesh, truth, n_resamples, seed)
    record["coherent_bound_optimism"] = coherent_bound_optimism(mesh, truth)
    record["map_prior_sweep"] = map_prior_sweep(mesh, truth, MAP_PRIOR_STD_SWEEP)
    return record


def _edge_bootstrap(mesh: object, truth: dict, n_resamples: int, seed: int) -> dict:
    """The per-edge Young-Dupre work's per-edge bootstrap, on the uncertainty report's key names.

    The per-edge work produced this for two cases only; extending it to all 47 is part of this
    calibration work's remit, because it was the per-edge work's stated "most promising lead"
    for an interval with `error/std = O(1)` and a lead measured on two cases is not a
    measurement.

    Args:
        mesh (object): The `DcelData` mesh.
        truth (dict): Ground-truth tensions.
        n_resamples (int): Bootstrap draws.
        seed (int): Bootstrap seed.

    Returns:
        dict: The calibration entry.
    """
    raw = bootstrap_calibration(mesh, truth, n_resamples=n_resamples, seed=seed)
    return {
        "estimator": TensionComputationMethod.YoungDupreLocal.name,
        "median": raw["error_over_std_median"],
        "p95": raw["error_over_std_p95"],
        "max": raw["error_over_std_max"],
        "median_std": raw["mean_bootstrap_std"],
        "n_resamples": raw["n_resamples"],
        "seed": raw["seed"],
    }


def map_prior_sweep(mesh: object, truth: dict, prior_stds: tuple[float, ...]) -> dict:
    """Is `BayesianMap`'s O(1) calibration a result, or an artifact of the prior width?

    `ForceBalanceMAP` is the one variant predating this calibration work whose `error/std` is already O(1). This decides
    whether that means anything. Its interval's width is set by `MAP_PRIOR_TENSION_STD`, which
    is a **choice** (0.3, documented as "deliberately broad" but not derived). If `error/std`
    scales as `1 / prior_std`, then the variant can be made to show any calibration at all and
    its O(1) is not evidence of anything — the number was chosen, not measured.
    `CoherentResidual`, by contrast, has no free hyperparameter: its scale is the measured
    residual norm.

    Args:
        mesh (object): The `DcelData` mesh.
        truth (dict): Ground-truth tensions.
        prior_stds (tuple[float, ...]): Prior tension standard deviations to sweep.

    Returns:
        dict: `error/std` and the interval width per prior width, plus the fitted exponent of
            `error/std` against `prior_std` (about -1 means "purely the prior").
    """
    from foambryo import force_balance_inference as fbi  # local import

    original = fbi.MAP_PRIOR_TENSION_STD
    entries = {}
    try:
        for prior_std in prior_stds:
            fbi.MAP_PRIOR_TENSION_STD = prior_std
            entries[f"{prior_std:g}"] = bayesian_map_calibration(mesh, truth)
    finally:
        fbi.MAP_PRIOR_TENSION_STD = original

    widths = np.array([float(key) for key in entries])
    ratios = np.array([entry["median"] for entry in entries.values()])
    keep = np.isfinite(ratios) & (ratios > 0)
    slope = (
        float(np.polyfit(np.log(widths[keep]), np.log(ratios[keep]), 1)[0]) if keep.sum() > 1 else float("nan")
    )
    return {"by_prior_std": entries, "log_log_slope_error_over_std_vs_prior_std": slope}


def summarise(records: list[dict]) -> dict:
    """Per-group medians of every variant's `error/std`, ordered best-calibrated first.

    Args:
        records (list[dict]): Per-case records.

    Returns:
        dict: One entry per equilibrium group.
    """
    summary = {}
    for group, group_records in case_groups.split_records(records).items():
        if not group_records:
            continue
        case_groups.require_single_group(group_records)
        names = sorted({name for record in group_records for name in record["variants"]})
        variants = {}
        for name in names:
            entries = [record["variants"][name] for record in group_records if name in record["variants"]]
            medians = np.array([entry["median"] for entry in entries], dtype=float)
            finite = medians[np.isfinite(medians)]
            variants[name] = {
                "estimator": entries[0]["estimator"],
                "median_error_over_std": float(np.median(finite)) if finite.size else float("nan"),
                "min_error_over_std": float(np.min(finite)) if finite.size else float("nan"),
                "max_error_over_std": float(np.max(finite)) if finite.size else float("nan"),
                "median_p95_error_over_std": (
                    float(np.median([e["p95"] for e in entries if np.isfinite(e.get("p95", np.nan))]))
                    if any(np.isfinite(e.get("p95", np.nan)) for e in entries)
                    else float("nan")
                ),
                "median_std": float(np.median([entry["median_std"] for entry in entries])),
                "n_cases": int(finite.size),
                # "O(1)" is read as within a factor 10 of calibrated, in either direction.
                "is_order_one": bool(finite.size and 0.1 <= float(np.median(finite)) <= 10.0),
            }
        order = sorted(
            names,
            key=lambda name: abs(np.log10(max(variants[name]["median_error_over_std"], 1e-12))),
        )
        implied = [
            record["variants"]["IidGls"]["implied_n_eff"]
            for record in group_records
            if "implied_n_eff" in record["variants"].get("IidGls", {})
        ]
        summary[group] = {
            "n_cases": len(group_records),
            "cases": [record["case"] for record in group_records],
            "note": case_groups.group_note(group),
            "variants": variants,
            "ranking_best_calibrated_first": order,
            "implied_n_eff": {
                "median": float(np.median(implied)) if implied else float("nan"),
                "min": float(np.min(implied)) if implied else float("nan"),
                "max": float(np.max(implied)) if implied else float("nan"),
                "median_n_rows": float(np.median([record["n_rows"] for record in group_records])),
            },
            "coherent_bound_optimism_median": float(
                np.median([r["coherent_bound_optimism"]["ratio_truth_over_estimate"] for r in group_records]),
            ),
            "rho_free": {
                "median": float(np.median([record["rho_free"] for record in group_records])),
                "min": float(np.min([record["rho_free"] for record in group_records])),
                "max": float(np.max([record["rho_free"] for record in group_records])),
            },
            "map_prior_sweep": {
                "log_log_slope_median": float(
                    np.median(
                        [r["map_prior_sweep"]["log_log_slope_error_over_std_vs_prior_std"] for r in group_records],
                    ),
                ),
                "median_error_over_std_by_prior_std": {
                    f"{width:g}": float(
                        np.median(
                            [r["map_prior_sweep"]["by_prior_std"][f"{width:g}"]["median"] for r in group_records],
                        ),
                    )
                    for width in MAP_PRIOR_STD_SWEEP
                },
            },
        }
    return summary


def markdown_table(summary: dict) -> str:
    """The calibration table, best-calibrated first — the one table this calibration work is judged on.

    Args:
        summary (dict): Output of `summarise`.

    Returns:
        str: Markdown.
    """
    lines = [
        "# Uncertainty calibration — `error/std` for every variant",
        "",
        "Generated by `foambryo_benchmarks/run_uncertainty_calibration.py` (the uncertainty work).",
        "`error/std` is the median over interfaces of `|inferred - truth| / std`, **gauge-matched**",
        "(the interval is rescaled by the same factor as the tensions). **1 is calibrated**;",
        "above 1 the interval is optimistic (dangerous), below 1 conservative (safe, less useful).",
        "",
        (
            "The two equilibrium groups are never combined: by recorded decision, the five "
            "non-equilibrium cases are excluded from every accuracy claim."
        ),
        "",
    ]
    for group in case_groups.GROUP_ORDER:
        if group not in summary:
            continue
        entry = summary[group]
        lines += [
            f"## {group} — {entry['n_cases']} cases",
            "",
            entry["note"],
            "",
            "| variant | estimator | median error/std | median p95 | range over cases | median std | O(1)? |",
            "|---|---|---|---|---|---|---|",
        ]
        for name in entry["ranking_best_calibrated_first"]:
            variant = entry["variants"][name]
            lines.append(
                f"| `{name}` | `{variant['estimator']}` | **{variant['median_error_over_std']:.3g}** | "
                f"{variant['median_p95_error_over_std']:.3g} | "
                f"{variant['min_error_over_std']:.3g} - {variant['max_error_over_std']:.3g} | "
                f"{variant['median_std']:.2e} | {'yes' if variant['is_order_one'] else 'no'} |",
            )
        implied = entry["implied_n_eff"]
        lines += [
            "",
            (
                f"**Implied `n_eff`** (independent observations the operator rows are actually worth): "
                f"median **{implied['median']:.2f}**, range {implied['min']:.2f}-{implied['max']:.2f}, "
                f"against a median **{implied['median_n_rows']:.0f} rows**. The i.i.d. model assumes "
                f"`n_eff = n_rows`; `CoherentResidual` assumes `n_eff = 1`."
            ),
            "",
            (
                f"**`CoherentResidual`'s own optimism**: the residual at the truth is "
                f"**{entry['coherent_bound_optimism_median']:.2f}x** the residual at the estimate "
                "(median), which is why a median `error/std` below 1 coexists with a p95 above 1. "
                "`ProbeResidual` measures the same perturbation off an independent estimate instead."
            ),
            "",
            (
                f"**`rho_free`** on this group: median {entry['rho_free']['median']:.4g}, "
                f"range {entry['rho_free']['min']:.4g}-{entry['rho_free']['max']:.4g}."
            ),
            "",
            (
                "**`BayesianMap`'s O(1) is a choice, not a result.** Sweeping the prior tension s.d. "
                "(its `MAP_PRIOR_TENSION_STD`, shipped at 0.3):"
            ),
            "",
            "| prior s.d. | " + " | ".join(f"{w:g}" for w in MAP_PRIOR_STD_SWEEP) + " |",
            "|---" * (len(MAP_PRIOR_STD_SWEEP) + 1) + "|",
            "| median error/std | "
            + " | ".join(
                f"{entry['map_prior_sweep']['median_error_over_std_by_prior_std'][f'{w:g}']:.3g}"
                for w in MAP_PRIOR_STD_SWEEP
            )
            + " |",
            "",
            (
                f"log-log slope of `error/std` against the prior s.d.: "
                f"**{entry['map_prior_sweep']['log_log_slope_median']:.2f}** "
                "(-1 would mean the interval *is* the prior). The width is set by a hyperparameter "
                "nobody derived, so any calibration it shows was chosen. `ProbeResidual` and "
                "`CoherentResidual` have no such knob: their scale is a measured residual norm."
            ),
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    """Run the calibration study over the dataset and write the JSON + Markdown records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "benchmarking-dataset",
    )
    parser.add_argument("--cases", nargs="*", default=None, help="Case ids; default is all of them")
    parser.add_argument("--n-resamples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "baseline" / "uncertainty_calibration.json",
    )
    args = parser.parse_args()

    case_ids = args.cases if args.cases else discover_case_ids(args.dataset_dir)
    records = []
    for case_id in case_ids:
        start = time.perf_counter()
        records.append(run_case(case_id, args.dataset_dir, args.n_resamples, args.seed))
        print(f"  {case_id}: {time.perf_counter() - start:.1f}s", flush=True)

    summary = summarise(records)
    payload = {
        "records": records,
        "summary": summary,
        "n_resamples": args.n_resamples,
        "seed": args.seed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (args.out.parent / "uncertainty_calibration.md").write_text(markdown_table(summary) + "\n")
    print(f"wrote {args.out}")
    print(markdown_table(summary))


if __name__ == "__main__":
    main()
