#!/usr/bin/env python
"""CLI: decide `Variational`'s solve route on evidence — route P (Gram) vs route Q (Operator).

The sparse-assembly work sparsified the variational assembly but **retained** route P and left
the route choice to an explicit decision (a default that moves published numbers is flipped
only on benchmark evidence), because it moves published numbers by ~1.5e-4. This is the measurement that decides it.

The question, stated so it can be answered: **does route Q change ACCURACY on the 42
equilibrium cases?** If it does not, the tie-break is conditioning (route Q never forms
`M^T M`) and provenance (route P's per-row normalisation is an undocumented ad-hoc
reweighting). If accuracy degrades, route P stays and the reason is recorded.

Everything is gauge-matched (the rule every accuracy comparison in this package follows) and
the 42 and the 5 are reported separately by construction (`foambryo_benchmarks/case_groups.py`,
the recorded decision to exclude the five non-equilibrium cases).

Usage:
    uv run python -m foambryo_benchmarks.run_variational_route
    uv run python -m foambryo_benchmarks.run_variational_route --cases 000 011 --out /tmp/route.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from foambryo_benchmarks import case_groups
from foambryo_benchmarks.run_baseline import discover_case_ids
from foambryo import dcel_mesh_from_file
from foambryo.geometry import compute_area_derivative_sparse, compute_volume_derivative_sparse
from foambryo.tension_inference import (
    VariationalRoute,
    _build_matrix_discrete_sparse,
    _extract_submatrices,
    _infer_tension_variational_yd,
)

#: The ground truth's own precision floor. `benchmarking-dataset`'s `dict_tensions.npy` are
#: stored to 4 decimals, so each value carries a half-ulp of 5e-5 *absolute*; the relative
#: floor is therefore `5e-5 / gamma`, which is 5e-5 at the median tension (~1) and **6.7e-4**
#: at the smallest tension in the dataset (0.075). Any inference difference below the latter
#: cannot be adjudicated by this dataset at all. The force-balance work measured the *aggregate* form of the
#: same floor as `|1 - mean(gamma_nominal)| = 5.6e-06`.
GROUND_TRUTH_HALF_ULP = 5e-5


def route_conditioning(mesh: object) -> dict:
    """Condition numbers of the matrix each route actually inverts.

    Route P inverts the row-normalised Schur complement `mat_t` (an `nm x nm` dense block
    derived from the Gram matrix); route Q inverts `M Z`, the `3N x (n-1)` operator left after
    the gauge is eliminated by substitution. Reporting `cond` of the *solved* matrix is the
    only comparison that means anything — the sparse-assembly work's quoted "850 vs 7e5" is `cond(M)` against
    `cond(M^T M)`, which is a square root by construction rather than a property of a route.

    Args:
        mesh (object): The `DcelData` mesh.

    Returns:
        dict: Condition numbers and the row-normalisation divisor statistics.
    """
    matrix_m, _b, nm, _keys = _build_matrix_discrete_sparse(mesh, 1.0)
    f_g, f_p, g_g, g_p = _extract_submatrices(matrix_m, nm)
    mat_t = f_g - f_p @ np.linalg.inv(g_p) @ g_g

    _keys, matrix_d = compute_area_derivative_sparse(mesh)
    _materials, matrix_v = compute_volume_derivative_sparse(mesh)
    cells_v = matrix_v[1:]
    n_cols = nm + cells_v.shape[0]
    operator = np.empty((matrix_d.shape[1], n_cols))
    operator[:, :nm] = matrix_d.T.toarray()
    operator[:, nm:] = -cells_v.T.toarray()
    anchor = np.zeros(n_cols)
    anchor[:nm] = 1.0 / nm
    unit = anchor / np.linalg.norm(anchor)
    reflector = unit - np.eye(n_cols)[0]
    reflector /= np.linalg.norm(reflector)
    basis = (np.eye(n_cols) - 2 * np.outer(reflector, reflector))[:, 1:]

    gram = operator.T @ operator
    # The divisors route P applies: the row sums of the *un-normalised* Gram block system.
    dd = (matrix_d @ matrix_d.T).toarray()
    dv = (matrix_d @ cells_v.T).toarray()
    vv = (cells_v @ cells_v.T).toarray()
    raw = np.zeros((n_cols, n_cols))
    raw[:nm, :nm] = dd
    raw[:nm, nm:] = -dv
    raw[nm:, :nm] = dv.T
    raw[nm:, nm:] = -vv
    divisors = raw.sum(axis=1)

    return {
        "cond_route_p_solved": float(np.linalg.cond(mat_t)),
        "cond_route_q_solved": float(np.linalg.cond(operator @ basis)),
        "cond_operator_m": float(np.linalg.cond(operator)),
        "cond_gram_mtm": float(np.linalg.cond(gram)),
        "row_normalisation_divisor_min": float(divisors.min()),
        "row_normalisation_divisor_max": float(divisors.max()),
        "row_normalisation_divisor_abs_ratio": float(np.abs(divisors).max() / np.abs(divisors).min()),
        "row_normalisation_changes_sign": bool(divisors.min() < 0 < divisors.max()),
    }


def run_case(case_id: str, dataset_dir: Path) -> dict:
    """Both routes on one case, with the accuracy comparison and the conditioning.

    Args:
        case_id (str): Case id, e.g. `"000"`.
        dataset_dir (Path): Directory holding `{case}_mesh.rec` and `{case}_dict_tensions.npy`.

    Returns:
        dict: One record.
    """
    truth = np.load(dataset_dir / f"{case_id}_dict_tensions.npy", allow_pickle=True).item()
    mesh = dcel_mesh_from_file(str(dataset_dir / f"{case_id}_mesh.rec"))

    out: dict = {"case": case_id, "group": case_groups.group_of(case_id)}
    tensions = {}
    for route in VariationalRoute:
        start = time.perf_counter()
        _values, dict_tensions, _resid = _infer_tension_variational_yd(mesh, 1.0, route=route)
        elapsed = time.perf_counter() - start
        keys = sorted(dict_tensions)
        truth_vector = np.array([truth[key] for key in keys])
        inferred = np.array([dict_tensions[key] for key in keys])
        scale = float(truth_vector.mean() / inferred.mean())
        relative = np.abs(scale * inferred - truth_vector) / np.abs(truth_vector)
        tensions[route.name] = scale * inferred
        out[route.name] = {
            "median_relative_error_matched_gauge": float(np.median(relative)),
            "p95_relative_error_matched_gauge": float(np.percentile(relative, 95)),
            "max_relative_error_matched_gauge": float(relative.max()),
            "mean_inferred_tension": float(inferred.mean()),
            "gauge_matching_scale": scale,
            "wall_time_s": elapsed,
        }
        out["interfaces"] = [[int(label) for label in key] for key in keys]

    shift = np.abs(tensions["Gram"] - tensions["Operator"])
    truth_vector = np.array([truth[key] for key in sorted(truth)])
    out["route_shift"] = {
        "median_abs": float(np.median(shift)),
        "max_abs": float(shift.max()),
        "median_relative_to_truth": float(np.median(shift / np.abs(truth_vector))),
    }
    out["ground_truth_precision_floor"] = {
        "half_ulp_absolute": GROUND_TRUTH_HALF_ULP,
        "relative_at_median_tension": float(GROUND_TRUTH_HALF_ULP / np.median(truth_vector)),
        "relative_at_smallest_tension": float(GROUND_TRUTH_HALF_ULP / truth_vector.min()),
        "smallest_tension": float(truth_vector.min()),
    }
    out["conditioning"] = route_conditioning(mesh)
    out["operator_better"] = bool(
        out["Operator"]["median_relative_error_matched_gauge"]
        < out["Gram"]["median_relative_error_matched_gauge"],
    )
    return out


def summarise(records: list[dict]) -> dict:
    """Per-group aggregate: the median-of-medians each route achieves, and the verdict inputs.

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
        entry: dict = {
            "n_cases": len(group_records),
            "cases": [record["case"] for record in group_records],
            "note": case_groups.group_note(group),
        }
        for route in VariationalRoute:
            errors = [record[route.name]["median_relative_error_matched_gauge"] for record in group_records]
            entry[route.name] = {
                "median_of_case_medians": float(np.median(errors)),
                "worst_case": float(np.max(errors)),
                "median_wall_time_s": float(np.median([r[route.name]["wall_time_s"] for r in group_records])),
            }
        entry["operator_better_on_n_cases"] = int(sum(record["operator_better"] for record in group_records))
        entry["ratio_operator_over_gram"] = float(
            entry["Operator"]["median_of_case_medians"] / entry["Gram"]["median_of_case_medians"],
        )
        entry["route_shift_median_abs"] = float(
            np.median([record["route_shift"]["median_abs"] for record in group_records]),
        )
        entry["route_shift_max_abs"] = float(np.max([record["route_shift"]["max_abs"] for record in group_records]))
        entry["cond_route_p_median"] = float(
            np.median([record["conditioning"]["cond_route_p_solved"] for record in group_records]),
        )
        entry["cond_route_q_median"] = float(
            np.median([record["conditioning"]["cond_route_q_solved"] for record in group_records]),
        )
        entry["cond_gram_mtm_median"] = float(
            np.median([record["conditioning"]["cond_gram_mtm"] for record in group_records]),
        )
        entry["row_normalisation_divisor_abs_ratio_range"] = [
            float(np.min([r["conditioning"]["row_normalisation_divisor_abs_ratio"] for r in group_records])),
            float(np.max([r["conditioning"]["row_normalisation_divisor_abs_ratio"] for r in group_records])),
        ]
        entry["row_normalisation_changes_sign_on_n_cases"] = int(
            sum(record["conditioning"]["row_normalisation_changes_sign"] for record in group_records),
        )
        summary[group] = entry
    return summary


def main() -> None:
    """Run the route comparison over the dataset and write the JSON record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "benchmarking-dataset",
    )
    parser.add_argument("--cases", nargs="*", default=None, help="Case ids; default is all of them")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "baseline" / "variational_route.json",
    )
    args = parser.parse_args()

    case_ids = args.cases if args.cases else discover_case_ids(args.dataset_dir)
    records = []
    for case_id in case_ids:
        records.append(run_case(case_id, args.dataset_dir))
        print(f"  {case_id}: done", flush=True)

    payload = {"records": records, "summary": summarise(records)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    for group, entry in payload["summary"].items():
        print(
            f"{group}: Gram {entry['Gram']['median_of_case_medians']:.6e}  "
            f"Operator {entry['Operator']['median_of_case_medians']:.6e}  "
            f"ratio {entry['ratio_operator_over_gram']:.5f}  "
            f"Operator better on {entry['operator_better_on_n_cases']}/{entry['n_cases']}  "
            f"shift {entry['route_shift_median_abs']:.2e}  "
            f"cond P {entry['cond_route_p_median']:.2e} / Q {entry['cond_route_q_median']:.2e}",
        )


if __name__ == "__main__":
    main()
