#!/usr/bin/env python
r"""CLI: the per-edge Young-Dupre measurement.

Everything the per-edge measurement has to answer, in one regenerable record. Five blocks:

1. **The acceptance measurement.** `YoungDupreLocal` vs `YoungDupre`, gauge-matched median
   relative tension error, on the **42** equilibrium cases (the 5 non-equilibrium cases are
   reported separately and never mixed in — the recorded decision to exclude the five
   non-equilibrium cases, `foambryo_benchmarks/case_groups.py`). Acceptance
   asks for a >= 10 % reduction in the median; anything else is `FALSIFIED` with numbers.

2. **A factorial that says *why*, not just *whether*.** Per-edge Young-Dupre differs from the
   shipped `YoungDupre` in exactly two independent ways, and they are separable:

   | variant | rows | angles | isolates |
   |---|---|---|---|
   | `D_line_cosmean` | one block per triple line | `cos(mean angle)` | the control: **must** reproduce `YoungDupre` |
   | `C_line_meancos` | one block per triple line | length-weighted `mean(cos angle)` | the **Jensen gap** alone |
   | `B_edge_avgangle` | one block per edge | each edge given its line's mean angle | the **row weighting** alone |
   | `A_edge` | one block per edge | that edge's own angles | both together = `YoungDupreLocal` |

   `B'` repeats `B` with weights rescaled so each line keeps its total least-squares weight; it
   must reproduce `D` to ~1e-14, which is what proves the row split itself is neutral and the
   damage is the reweighting.

3. **The mechanism, quantified.** If the Jensen gap is the mechanism, the per-case excess error
   of the line-preserving per-edge variant should track the length-weighted mean of
   `0.5 * Var(angle)` along each line. Reported as a Spearman correlation over the 42, because a
   mechanism that is asserted and not measured is not a mechanism.

4. **Boundary multiplicity** — half-edge visit multiplicity for triple lines touching the image
   boundary. Measured over all 47 meshes, plus a *constructed* open-boundary junction, because
   the dataset audit found 0 of 47 cases with cells touching the image border so the dataset cannot exhibit it.

5. **`Equilibrium`'s two recorded non-changes** (the defect-cleanup work): no junction-length weight, and one
   3-row block per incident half-edge. All four combinations measured on the 42.

Usage:
    uv run python -m foambryo_benchmarks.run_per_edge_young_dupre
    uv run python -m foambryo_benchmarks.run_per_edge_young_dupre --case-ids 000 001
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, stats

from foambryo_benchmarks import case_groups
from foambryo_benchmarks import metrics as m
from foambryo_benchmarks.run_baseline import DATASET_DIR, discover_case_ids
from foambryo_benchmarks.run_case import _json_safe
from foambryo import dcel_mesh_from_file
from foambryo.dcel import DcelData
from foambryo.geometry import compute_angles_tri, compute_trijunction_edges
from foambryo.per_edge_tension_inference import (
    DEFAULT_HUBER_DELTA,
    PerEdgeWeighting,
    _assemble_per_edge_blocks,
    build_matrix_tension_local_yd,
    infer_tensions_young_dupre_local,
)
from foambryo.tension_inference import TensionComputationMethod, _infer_tension_equilibrium, infer_tensions

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = REPO_ROOT / "foambryo_benchmarks" / "baseline"


def _matched_median(dict_tensions: dict, truth: dict) -> float:
    """Gauge-matched median relative tension error (the rule every accuracy comparison follows).

    Args:
        dict_tensions (dict): Inferred tensions.
        truth (dict): Ground-truth tensions.

    Returns:
        float: median relative error after matching the overall tension scale.
    """
    scale = m.gauge_matching_scale(dict_tensions, truth)
    return m.tension_relative_errors({k: scale * v for k, v in dict_tensions.items()}, truth)["median"]


def _solve(
    blocks: NDArray[np.float64],
    columns: NDArray[np.int64],
    weights: NDArray[np.float64],
    n_interfaces: int,
    keys: list[tuple[int, int]],
    truth: dict[tuple[int, int], float],
) -> tuple[float, NDArray[np.float64]]:
    """Assemble, solve with the shipped soft gauge row, and score.

    Args:
        blocks (NDArray): `(n, 3, 3)` coefficient blocks.
        columns (NDArray): `(n, 3)` interface column indices.
        weights (NDArray): `(n,)` row weights.
        n_interfaces (int): number of interfaces.
        keys (list): interface keys in column order.
        truth (dict): ground-truth tensions.

    Returns:
        tuple[float, NDArray]: gauge-matched median error, and the tension vector.
    """
    matrix_m, vector_b = build_matrix_tension_local_yd(blocks, columns, weights, n_interfaces, 1.0)
    tensions, *_ = linalg.lstsq(matrix_m, vector_b)
    return _matched_median(dict(zip(keys, tensions, strict=True)), truth), tensions


def multiplicity_audit(mesh: DcelData) -> dict:
    """Is every trijunction edge counted exactly once in its line's length weight? On one mesh.

    Who is counted, who is dropped (an edge that has lost one of its three films, at an open mesh
    boundary), and how many times.

    Args:
        mesh (DcelData): Mesh to analyze.

    Returns:
        dict: incident-face histogram over geometric edges, visit multiplicities of the counted
            edges, counts of dropped quadrijunctional and mesh-boundary edges, and the junction
            length credited versus the length actually present in the mesh.
    """
    incident: dict[tuple[int, int], list[int]] = defaultdict(list)
    length_of: dict[tuple[int, int], float] = {}
    for edge in mesh.half_edges:
        key = (min(edge.origin.key, edge.destination.key), max(edge.origin.key, edge.destination.key))
        incident[key].append(edge.key)
        length_of[key] = edge.length

    histogram: dict[int, int] = defaultdict(int)
    for keys in incident.values():
        histogram[len(keys)] += 1

    counted = compute_trijunction_edges(mesh)
    counted_keys = {edge.vertices for edge in counted}
    # A "junction edge" is any geometric edge where more than two films meet. Length present is
    # over all of them; length credited is over the ones compute_angles_tri actually counts.
    junction_keys = {key for key, keys in incident.items() if len(keys) >= 3}
    length_present = sum(length_of[key] for key in junction_keys)
    length_credited = sum(edge.length for edge in counted)

    return {
        "incident_faces_histogram": {str(k): v for k, v in sorted(histogram.items())},
        "visit_multiplicities": sorted({len(edge.half_edge_keys) for edge in counted}),
        "n_geometric_edges": len(incident),
        "n_junction_edges": len(junction_keys),
        "n_counted_trijunctional_edges": len(counted),
        "n_dropped_quadrijunctional_edges": len(junction_keys - counted_keys),
        "n_mesh_boundary_edges": histogram.get(1, 0),
        "junction_length_present": length_present,
        "junction_length_credited": length_credited,
        "credited_fraction": (length_credited / length_present) if length_present else float("nan"),
    }


def constructed_open_boundary_case(n_segments: int = 6, intact_segments: int = 4) -> dict:
    """A triple line that runs off an open mesh boundary — the case the dataset cannot supply.

    Three rectangular sheets meet along the x axis at 120 degrees, separating regions 1, 2, 3.
    One sheet is truncated after `intact_segments` of the line's `n_segments` edges, so the
    remaining edges carry only **two** films. This is what "a triple line touching the image
    boundary" looks like to the DCEL: not a change of multiplicity, but a loss of the third film.

    Args:
        n_segments (int, optional): Edges along the triple line. Defaults to 6.
        intact_segments (int, optional): How many keep all three films. Defaults to 4.

    Returns:
        dict: the audit for the intact line and for the truncated one, side by side.
    """

    def build(truncate_from: int) -> DcelData:
        xs = np.linspace(0.0, 1.0, n_segments + 1)
        vertices = [np.array([x, 0.0, 0.0]) for x in xs]
        sheet_start = []
        for sheet in range(3):
            angle = np.deg2rad(120 * sheet)
            offset = np.array([0.0, np.cos(angle), np.sin(angle)])
            sheet_start.append(len(vertices))
            vertices.extend([np.array([x, 0.0, 0.0]) + offset for x in xs])
        pairs = [(1, 2), (2, 3), (1, 3)]
        faces = []
        for sheet in range(3):
            material_1, material_2 = pairs[sheet]
            row = sheet_start[sheet]
            for i in range(n_segments):
                if sheet == 0 and i >= truncate_from:
                    continue
                faces.append([i, i + 1, row + i, material_1, material_2])
                faces.append([i + 1, row + i + 1, row + i, material_1, material_2])
        faces = np.array(faces)
        return DcelData(np.array(vertices), faces[:, :3], faces[:, 3:])

    intact = multiplicity_audit(build(n_segments))
    truncated = multiplicity_audit(build(intact_segments))
    true_length = 1.0
    return {
        "n_segments": n_segments,
        "intact_segments": intact_segments,
        "intact": intact,
        "truncated": truncated,
        "true_line_length": true_length,
        "weight_bias_factor": truncated["junction_length_credited"] / intact["junction_length_credited"],
    }


def weight_perturbation_sensitivity(mesh: DcelData, truth: dict, factor: float) -> dict:
    """How much does under-weighting **one** triple line by `factor` move the tensions?

    This converts the edge-accounting question above -- a line whose boundary edges are dropped has
    its length weight under-stated -- from "a weight is wrong" into "and it would cost this much",
    using the real operator. Each triple line in turn has its weight multiplied by
    `factor` (the fraction of its length that would survive at an open boundary) and the
    gauge-matched median tension error is re-measured.

    Args:
        mesh (DcelData): Mesh to analyze.
        truth (dict): Ground-truth tensions.
        factor (float): Weight multiplier applied to one line at a time.

    Returns:
        dict: baseline error and the worst / median error over the single-line perturbations.
    """
    dict_areas = mesh.compute_areas_interfaces()
    keys = [tuple(int(label) for label in key) for key in sorted(dict_areas)]
    edges = compute_trijunction_edges(mesh)
    _, columns, regions, lengths, _ = _assemble_per_edge_blocks(edges, dict_areas)
    lines, line_index = np.unique(regions, axis=0, return_inverse=True)
    line_index = line_index.ravel()
    line_lengths = np.bincount(line_index, weights=lengths)

    # One block per line at weight l_k / L: the shipped YoungDupre system, rebuilt here so the
    # perturbation is applied to exactly the operator being discussed.
    total_length = lengths.sum()
    rad, _, _ = compute_angles_tri(mesh, unique=True)
    line_blocks = np.zeros((len(lines), 3, 3))
    line_columns = np.zeros((len(lines), 3), dtype=np.int64)
    for edge_i, line_i in enumerate(line_index):
        a, b, c = (int(label) for label in regions[edge_i])
        angle_a, angle_b, angle_c = (
            rad[(min(b, c), a, max(b, c))],
            rad[(min(a, c), b, max(a, c))],
            rad[(min(a, b), c, max(a, b))],
        )
        line_blocks[line_i] = [
            [1.0, np.cos(angle_b), np.cos(angle_a)],
            [np.cos(angle_b), 1.0, np.cos(angle_c)],
            [np.cos(angle_a), np.cos(angle_c), 1.0],
        ]
        line_columns[line_i] = columns[edge_i]

    base_weights = line_lengths / total_length
    baseline, _ = _solve(line_blocks, line_columns, base_weights, len(keys), keys, truth)
    perturbed = []
    for line_i in range(len(lines)):
        weights = base_weights.copy()
        weights[line_i] *= factor
        error, _ = _solve(line_blocks, line_columns, weights, len(keys), keys, truth)
        perturbed.append(error)
    perturbed = np.array(perturbed)
    return {
        "factor": factor,
        "baseline_median_error": baseline,
        "perturbed_median_error_median": float(np.median(perturbed)),
        "perturbed_median_error_max": float(perturbed.max()),
        "worst_relative_degradation": float(perturbed.max() / baseline),
    }


def jensen_predictor(mesh: DcelData) -> float:
    """Length-weighted mean of `0.5 * Var(angle)` over triple lines — the predicted Jensen bias.

    `mean(cos a) = cos(mean a) * (1 - Var(a) / 2) + O(Var^2)`, so a per-edge builder that takes
    the cosine before averaging carries a systematic relative bias of `Var / 2` on every
    off-diagonal entry. This is that prediction, as one number per mesh.

    Args:
        mesh (DcelData): Mesh to analyze.

    Returns:
        float: predicted relative bias.
    """
    per_key_angles: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    per_key_lengths: dict[tuple[int, int, int], float] = defaultdict(float)
    for edge in compute_trijunction_edges(mesh):
        for key, angle in zip(edge.keys, edge.angles, strict=True):
            per_key_angles[key].append(angle)
            per_key_lengths[key] += edge.length
    total = sum(per_key_lengths.values())
    return float(
        sum(0.5 * np.var(angles) * per_key_lengths[key] for key, angles in per_key_angles.items()) / total,
    )


def bootstrap_calibration(mesh: DcelData, truth: dict, n_resamples: int = 200, seed: int = 0) -> dict:
    """`error / std` for a per-edge bootstrap interval — the one interval the per-edge measurement produces.

    The promised pay-off of a per-edge operator is that the edges are resampleable, so an
    interval can be built without an i.i.d. Gaussian residual model. This resamples the
    trijunctional edges with replacement, re-solves, and reports **`error / std`** (the
    uncertainty work's calibration metric, per the force-balance work's finding that the
    naive posterior s.d. is ~250x optimistic). A
    well-calibrated interval gives `error / std` of order 1.

    Args:
        mesh (DcelData): Mesh to analyze.
        truth (dict): Ground-truth tensions.
        n_resamples (int, optional): Bootstrap resamples. Defaults to 200.
        seed (int, optional): RNG seed, recorded so the number is reproducible. Defaults to 0.

    Returns:
        dict: median over interfaces of `|error| / std`, and the mean bootstrap s.d.
    """
    dict_areas = mesh.compute_areas_interfaces()
    keys = [tuple(int(label) for label in key) for key in sorted(dict_areas)]
    edges = compute_trijunction_edges(mesh)
    blocks, columns, regions, lengths, _ = _assemble_per_edge_blocks(edges, dict_areas)
    _, line_index = np.unique(regions, axis=0, return_inverse=True)
    line_index = line_index.ravel()
    line_lengths = np.bincount(line_index, weights=lengths)
    total_length = lengths.sum()
    weights = (line_lengths[line_index] / total_length) * np.sqrt(lengths / line_lengths[line_index])

    _, point_estimate = _solve(blocks, columns, weights, len(keys), keys, truth)
    scale = m.gauge_matching_scale(dict(zip(keys, point_estimate, strict=True)), truth)
    truth_vector = np.array([truth[key] for key in keys])
    error = np.abs(scale * point_estimate - truth_vector)

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_resamples):
        pick = rng.integers(0, len(blocks), len(blocks))
        _, tensions = _solve(blocks[pick], columns[pick], weights[pick], len(keys), keys, truth)
        draws.append(tensions * m.gauge_matching_scale(dict(zip(keys, tensions, strict=True)), truth))
    spread = np.std(np.array(draws), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(spread > 0, error / spread, np.nan)
    return {
        "n_resamples": n_resamples,
        "seed": seed,
        "error_over_std_median": float(np.nanmedian(ratio)),
        # p95 added by the uncertainty work, which reports it for every variant in its calibration table.
        # Additive: no pre-existing key or number changes.
        "error_over_std_p95": float(np.nanpercentile(ratio, 95)),
        "error_over_std_max": float(np.nanmax(ratio)),
        "mean_bootstrap_std": float(np.mean(spread)),
    }


def run_case(case_id: str, dataset_dir: Path) -> dict:
    """Every per-edge measurement for one case.

    Args:
        case_id (str): Case id, e.g. `"000"`.
        dataset_dir (Path): Directory holding the dataset.

    Returns:
        dict: one record.
    """
    mesh_path = dataset_dir / f"{case_id}_mesh.rec"
    truth = np.load(dataset_dir / f"{case_id}_dict_tensions.npy", allow_pickle=True).item()
    mesh = dcel_mesh_from_file(str(mesh_path))

    dict_areas = mesh.compute_areas_interfaces()
    keys = [tuple(int(label) for label in key) for key in sorted(dict_areas)]
    n_interfaces = len(keys)
    edges = compute_trijunction_edges(mesh)
    blocks, columns, regions, lengths, _ = _assemble_per_edge_blocks(edges, dict_areas)
    total_length = lengths.sum()
    lines, line_index = np.unique(regions, axis=0, return_inverse=True)
    line_index = line_index.ravel()
    line_lengths = np.bincount(line_index, weights=lengths)
    line_counts = np.bincount(line_index)

    # Blocks with each edge carrying its LINE's averaged angles (variant B).
    rad, _, _ = compute_angles_tri(mesh, unique=True)
    blocks_averaged = np.empty_like(blocks)
    for edge_i in range(len(blocks)):
        a, b, c = (int(label) for label in regions[edge_i])
        angle_a, angle_b, angle_c = (
            rad[(min(b, c), a, max(b, c))],
            rad[(min(a, c), b, max(a, c))],
            rad[(min(a, b), c, max(a, b))],
        )
        blocks_averaged[edge_i] = [
            [1.0, np.cos(angle_b), np.cos(angle_a)],
            [np.cos(angle_b), 1.0, np.cos(angle_c)],
            [np.cos(angle_a), np.cos(angle_c), 1.0],
        ]

    # One block per line: mean(cos) (variant C) and cos(mean) (variant D, the control).
    line_blocks_meancos = np.zeros((len(lines), 3, 3))
    line_blocks_cosmean = np.zeros((len(lines), 3, 3))
    line_columns = np.zeros((len(lines), 3), dtype=np.int64)
    for edge_i, line_i in enumerate(line_index):
        line_blocks_meancos[line_i] += blocks[edge_i] * lengths[edge_i]
        line_blocks_cosmean[line_i] = blocks_averaged[edge_i]
        line_columns[line_i] = columns[edge_i]
    line_blocks_meancos /= line_lengths[:, None, None]

    edge_weight = lengths / total_length
    weight_matched = line_lengths[line_index] / total_length / np.sqrt(line_counts[line_index])

    variants = {}
    variants["D_line_cosmean"], tensions_d = _solve(
        line_blocks_cosmean, line_columns, line_lengths / total_length, n_interfaces, keys, truth,
    )
    variants["C_line_meancos"], _ = _solve(
        line_blocks_meancos, line_columns, line_lengths / total_length, n_interfaces, keys, truth,
    )
    variants["B_edge_avgangle"], _ = _solve(blocks_averaged, columns, edge_weight, n_interfaces, keys, truth)
    variants["Bprime_edge_avgangle_weightmatched"], tensions_bp = _solve(
        blocks_averaged, columns, weight_matched, n_interfaces, keys, truth,
    )
    variants["A_edge"], _ = _solve(blocks, columns, edge_weight, n_interfaces, keys, truth)

    shipped = {
        "YoungDupre": _matched_median(infer_tensions(mesh, 1, TensionComputationMethod.YoungDupre), truth),
        "YoungDupreLocal": _matched_median(
            infer_tensions(mesh, 1, TensionComputationMethod.YoungDupreLocal), truth,
        ),
    }
    for weighting in PerEdgeWeighting:
        shipped[f"YoungDupreLocal_{weighting.name}"] = _matched_median(
            infer_tensions_young_dupre_local(mesh, 1.0, weighting=weighting).dict_tensions, truth,
        )
    robust = infer_tensions_young_dupre_local(mesh, 1.0, huber_delta=DEFAULT_HUBER_DELTA)
    shipped["YoungDupreLocal_Huber"] = _matched_median(robust.dict_tensions, truth)

    equilibrium = {}
    for deduplicate in (False, True):
        for length_weighted in (False, True):
            name = f"dedup={deduplicate},weighted={length_weighted}"
            _, dict_tensions, _ = _infer_tension_equilibrium(
                mesh, 1, deduplicate=deduplicate, length_weighted=length_weighted,
            )
            equilibrium[name] = _matched_median(dict_tensions, truth)

    field = infer_tensions_young_dupre_local(mesh, 1.0).residual_field
    return {
        "case_id": case_id,
        "n_interfaces": n_interfaces,
        "n_trijunctional_edges": len(blocks),
        "n_triple_lines": len(lines),
        "edges_per_line_min_median_max": [
            int(line_counts.min()), float(np.median(line_counts)), int(line_counts.max()),
        ],
        "shipped_methods": shipped,
        "factorial": variants,
        "control_Bprime_matches_D_max_abs": float(np.abs(tensions_bp - tensions_d).max()),
        "jensen_predicted_bias": jensen_predictor(mesh),
        "angle_spread_deg": {
            "median": float(np.degrees(np.median([np.std(a) for a in _per_key_angles(edges).values()]))),
            "max": float(np.degrees(np.max([np.std(a) for a in _per_key_angles(edges).values()]))),
        },
        "equilibrium_variants": equilibrium,
        "multiplicity": multiplicity_audit(mesh),
        "irls_iterations": robust.irls_iterations,
        "residual_field": {
            "n_edges": len(field.residual_norms),
            "median": float(np.median(field.residual_norms)),
            "p95": float(np.quantile(field.residual_norms, 0.95)),
            "max": float(field.residual_norms.max()),
        },
    }


def _per_key_angles(edges: list) -> dict:
    """Per wedge key, the list of per-edge angles along that triple line.

    Args:
        edges (list): Output of `compute_trijunction_edges`.

    Returns:
        dict: wedge key -> list of angles in radians.
    """
    per_key: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    for edge in edges:
        for key, angle in zip(edge.keys, edge.angles, strict=True):
            per_key[key].append(angle)
    return per_key


def summarise(records: list[dict]) -> dict:
    """Group-split summary: medians over cases, plus the mechanism correlation.

    Args:
        records (list[dict]): Per-case records.

    Returns:
        dict: `{"groups": {group: {...}}, "mechanism": {...}}`.
    """
    summary: dict = {"groups": {}, "mechanism": {}}
    by_group: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_group[case_groups.group_of(record["case_id"])].append(record)

    for group in case_groups.GROUP_ORDER:
        group_records = by_group.get(group, [])
        if not group_records:
            continue
        methods = sorted(group_records[0]["shipped_methods"])
        summary["groups"][group] = {
            "n_cases": len(group_records),
            "shipped_methods": {
                name: float(np.median([r["shipped_methods"][name] for r in group_records])) for name in methods
            },
            "factorial": {
                name: float(np.median([r["factorial"][name] for r in group_records]))
                for name in sorted(group_records[0]["factorial"])
            },
            "equilibrium_variants": {
                name: float(np.median([r["equilibrium_variants"][name] for r in group_records]))
                for name in sorted(group_records[0]["equilibrium_variants"])
            },
            "control_Bprime_matches_D_max_abs": float(
                np.max([r["control_Bprime_matches_D_max_abs"] for r in group_records]),
            ),
        }

    equilibrium_records = by_group.get(case_groups.EQUILIBRIUM, [])
    if len(equilibrium_records) > 2:
        predicted = np.array([r["jensen_predicted_bias"] for r in equilibrium_records])
        excess = np.array(
            [
                r["shipped_methods"]["YoungDupreLocal"] - r["shipped_methods"]["YoungDupre"]
                for r in equilibrium_records
            ],
        )
        rho, pvalue = stats.spearmanr(predicted, excess)
        summary["mechanism"] = {
            "n_cases": len(equilibrium_records),
            "spearman_jensen_vs_excess_error": float(rho),
            "p_value": float(pvalue),
            "median_predicted_bias": float(np.median(predicted)),
            "median_excess_error": float(np.median(excess)),
        }

    baseline = summary["groups"].get(case_groups.EQUILIBRIUM, {}).get("shipped_methods", {})
    if "YoungDupre" in baseline:
        reduction = 1.0 - baseline["YoungDupreLocal"] / baseline["YoungDupre"]
        summary["acceptance"] = {
            "criterion": "YoungDupreLocal reduces median tension error vs YoungDupre by >= 10 % on the 42",
            "young_dupre": baseline["YoungDupre"],
            "young_dupre_local": baseline["YoungDupreLocal"],
            "fractional_reduction": float(reduction),
            "verdict": "PASS" if reduction >= 0.10 else "FALSIFIED",
        }
    return summary


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--case-ids", nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=BASELINE_DIR / "per_edge_young_dupre.json")
    parser.add_argument("--calibration-cases", nargs="*", default=["000", "011"])
    args = parser.parse_args()

    case_ids = args.case_ids or discover_case_ids(args.dataset_dir)
    records = []
    for case_id in case_ids:
        records.append(run_case(case_id, args.dataset_dir))
        print(f"  {case_id}: {records[-1]['shipped_methods']['YoungDupre']:.3e} -> "
              f"{records[-1]['shipped_methods']['YoungDupreLocal']:.3e}")

    calibration = {}
    for case_id in args.calibration_cases:
        if case_id in case_ids:
            mesh = dcel_mesh_from_file(str(args.dataset_dir / f"{case_id}_mesh.rec"))
            truth = np.load(args.dataset_dir / f"{case_id}_dict_tensions.npy", allow_pickle=True).item()
            calibration[case_id] = bootstrap_calibration(mesh, truth)
            calibration[case_id]["weight_perturbation"] = weight_perturbation_sensitivity(mesh, truth, 2.0 / 3.0)

    payload = {
        "records": records,
        "summary": summarise(records),
        "to_verify_3_constructed": constructed_open_boundary_case(),
        "calibration": calibration,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")
    print(json.dumps(_json_safe(payload["summary"]), indent=2, sort_keys=True))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
