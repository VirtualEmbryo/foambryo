#!/usr/bin/env python
r"""CLI: the angle-convention audit — is `compute_angles_tri` right?

The open question: `foambryo.geometry.compute_angles_tri` orients face normals by a
*label-comparison* heuristic (`geometry.py:397-403`), not by geometry, and an early phantom
measurement reported that on the doublet its **through-cell** trijunction keys converged to
`theta` (the supplement of the true wedge `180 - theta`), while its through-medium key
converged to the true wedge `2*theta`. If the heuristic were label-order-dependent, the
returned angle would be the wedge for some junction label patterns and its supplement for
others — a *sign flip* on off-diagonal tension-matrix entries (`cos(180 - phi) = -cos phi`),
read as mesh noise.

This script settles it with four measurements, in priority order:

1. **Relabelling invariance (the decisive experiment).** Build the doublet once; apply label
   permutations to the region columns *keeping the triangle winding fixed* (the SAME physical
   mesh, cells renamed); rebuild the `DcelData` (which re-normalises label order and winding);
   run `compute_angles_tri`; map each output key back through the permutation to identify the
   physical wedge. **If a physical wedge's measured angle changes under a permutation, the
   heuristic is not geometric — that is the defect.** It does not: all permutations agree to
   ~1e-13 degrees, so the measurement is geometric and label-invariant.

2. **Wedge convergence on both phantoms.** The doublet's through-medium and through-cell keys
   both converge to the *true wedge* (`2*theta` and `180 - theta`), not the supplement, at
   every ratio. The tetrahedral vertex's interior (physical) trijunctions read exactly 120.

3. **Sum-to-360 across the 42 equilibrium meshes.** The three wedges of each real junction
   sum to 360 degrees; a single supplement flip would give `|sum - 360| ~ 2*(90 - theta)`,
   tens of degrees. Measured worst is a fraction of a degree (real mesh non-flatness).

4. **Reproducing the early supplement reading, on demand.** Flipping only the doublet's
   `(1, 2)` flat-disk winding — which simulates the phantom generator's former
   `_orient_outward_from_lower_label` "both regions bounded" bug, fixed at the package's initial
   validation stage — turns the through-cell reading into the supplement while leaving the
   through-medium reading untouched. So the early observation was a *phantom-generator
   orientation bug*, not a defect in `compute_angles_tri`.

Plus a per-interface method table on the doublet (all `TensionComputationMethod` members,
gauge-matched, `(1, 2)` interface broken out) so the phantom `Cotan`/`InvCotan`/`Equilibrium`
numbers — stale since the operator-precedence fix in `Cotan`/`InvCotan`, with `Equilibrium`
never having run before its half-edge traversal was fixed — are current.

Usage:
    uv run python -m foambryo_benchmarks.run_angle_convention
    uv run python -m foambryo_benchmarks.run_angle_convention --quick   # skip the 42-case sweep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from foambryo_benchmarks import case_groups
from foambryo_benchmarks import metrics as m
from foambryo_benchmarks import phantoms as ph
from foambryo_benchmarks.run_case import _json_safe
from foambryo import TensionComputationMethod, dcel_mesh_from_file, infer_tensions
from foambryo.dcel import DcelData
from foambryo.geometry import compute_angles_tri

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT.parent / "benchmarking-dataset"
BASELINE_DIR = REPO_ROOT / "foambryo_benchmarks" / "baseline"

# The three doublet trijunction keys, and which physical wedge each is (region in slot 1):
#   (1, 0, 2) -> wedge of the medium (region 0)  = 2*theta  (through-medium)
#   (0, 1, 2) -> wedge of cell A     (region 1)  = 180-theta (through-cell)
#   (0, 2, 1) -> wedge of cell B     (region 2)  = 180-theta (through-cell)
DOUBLET_MEDIUM_KEY = (1, 0, 2)
DOUBLET_CELL_KEYS = ((0, 1, 2), (0, 2, 1))


def _relabel_regions(regions: np.ndarray, perm: dict[int, int]) -> np.ndarray:
    """Rename region labels by `perm`, keeping the row (triangle winding) untouched."""
    return np.array([[perm[a], perm[b]] for a, b in regions], dtype=np.int64)


def relabelling_invariance(ratio: float = 0.5, n_rings: int = 24) -> dict:
    """Measurement 1: does a pure relabelling change any measured wedge angle?"""
    v, f, regions = ph.doublet_mesh(ratio, n_rings=n_rings)
    perms = {
        "identity": {0: 0, 1: 1, 2: 2},
        "swap_1_2": {0: 0, 1: 2, 2: 1},
        "cycle": {0: 1, 1: 2, 2: 0},
        "medium_to_max": {0: 2, 1: 0, 2: 1},
        "medium_to_mid": {0: 1, 1: 0, 2: 2},
        "arbitrary": {0: 5, 1: 3, 2: 7},
    }
    # For each permutation, record the angle keyed by the *physical* wedge region (0/1/2).
    per_perm: dict[str, dict[int, float]] = {}
    for name, perm in perms.items():
        inv = {v_: k for k, v_ in perm.items()}
        mesh = DcelData(v.copy(), f.copy(), _relabel_regions(regions, perm))
        _, deg, _ = compute_angles_tri(mesh, unique=False)
        per_perm[name] = {inv[key[1]]: val for key, val in deg.items()}
    # Max spread across permutations, per physical wedge.
    spread = {}
    for wedge in (0, 1, 2):
        vals = [per_perm[name][wedge] for name in perms]
        spread[wedge] = float(max(vals) - min(vals))
    return {
        "ratio": ratio,
        "n_rings": n_rings,
        "angle_by_physical_wedge_deg": {name: {str(k): v for k, v in d.items()} for name, d in per_perm.items()},
        "max_spread_across_permutations_deg": {str(k): v for k, v in spread.items()},
        "invariant": bool(max(spread.values()) < 1e-9),
    }


def wedge_convergence(ratios: tuple[float, ...] = (0.5, 1.0, 1.5)) -> dict:
    """Measurement 2a: doublet through-medium/through-cell keys converge to the true wedge."""
    out: dict[str, dict] = {}
    for ratio in ratios:
        pred = ph.doublet_predicted_angles(ratio)
        rows = []
        for nr in (8, 16, 32, 64, 128):
            v, f, regions = ph.doublet_mesh(ratio, n_rings=nr, n_phi=2 * nr)
            mesh = DcelData(v.copy(), f.copy(), regions.copy())
            _, deg, _ = compute_angles_tri(mesh, unique=False)
            rows.append(
                {
                    "n_rings": nr,
                    "through_medium_deg": float(deg[DOUBLET_MEDIUM_KEY]),
                    "through_cell_deg": float(deg[DOUBLET_CELL_KEYS[0]]),
                },
            )
        out[f"ratio_{ratio}"] = {
            "theta_deg": pred["theta_deg"],
            "true_wedge_medium_deg": pred["angle_through_medium_deg"],
            "true_wedge_cell_deg": pred["angle_through_cell_deg"],
            "supplement_cell_deg": pred["theta_deg"],
            "convergence": rows,
        }
    return out


def tetrahedral_interior(scale: float = 1.0) -> dict:
    """Measurement 2b: the tetrahedral vertex's interior trijunctions read 120 degrees."""
    v, f, regions = ph.tetrahedral_vertex_mesh(scale)
    mesh = DcelData(v.copy(), f.copy(), regions.copy())
    _, deg, _ = compute_angles_tri(mesh, unique=False)
    interior_labels = {1, 2, 3, 4}
    interior = {str(k): float(val) for k, val in deg.items() if set(k) <= interior_labels}
    outer = {str(k): float(val) for k, val in deg.items() if not set(k) <= interior_labels}
    interior_vals = np.array(list(interior.values()))
    return {
        "interior_deg": interior,
        "outer_deg_non_physical_truncation_artifacts": outer,
        "interior_max_abs_error_from_120_deg": float(np.max(np.abs(interior_vals - 120.0))),
    }


def sum_to_360_over_dataset() -> dict:
    """Measurement 3: the 3 wedges of each junction sum to 360 across the 42 equilibrium meshes."""
    case_ids = sorted(p.name[:3] for p in DATASET_DIR.glob("*_mesh.rec"))
    eq_ids = [c for c in case_ids if case_groups.group_of(c) == case_groups.EQUILIBRIUM]
    worst = 0.0
    worst_case = None
    n_junctions = 0
    per_case = {}
    for cid in eq_ids:
        mesh = dcel_mesh_from_file(str(DATASET_DIR / f"{cid}_mesh.rec"))
        _, deg, _ = compute_angles_tri(mesh, unique=False)
        groups: dict[frozenset, list[float]] = {}
        for key, val in deg.items():
            groups.setdefault(frozenset(key), []).append(val)
        devs = [abs(sum(vals) - 360.0) for vals in groups.values() if len(vals) == 3]
        n_junctions += len(devs)
        case_worst = max(devs) if devs else 0.0
        per_case[cid] = {"n_complete_junctions": len(devs), "max_abs_sum_minus_360_deg": case_worst}
        if case_worst > worst:
            worst, worst_case = case_worst, cid
    return {
        "n_equilibrium_cases": len(eq_ids),
        "n_complete_junctions": n_junctions,
        "worst_abs_sum_minus_360_deg": worst,
        "worst_case": worst_case,
        "per_case": per_case,
    }


def disk_flip_reproduces_supplement(ratio: float = 0.5, n_rings: int = 32) -> dict:
    """Measurement 4: flipping only the (1,2) disk winding turns through-cell into the supplement.

    This reproduces the phantom generator's former `_orient_outward_from_lower_label` "both
    regions bounded" bug (the flat cell-cell disk is the only bounded-bounded interface in the
    doublet), which is what the early phantom measurement read and misread as a
    `compute_angles_tri` defect.
    """
    pred = ph.doublet_predicted_angles(ratio)
    v, f, regions = ph.doublet_mesh(ratio, n_rings=n_rings, n_phi=2 * n_rings)

    mesh = DcelData(v.copy(), f.copy(), regions.copy())
    _, deg_ok, _ = compute_angles_tri(mesh, unique=False)

    f_bug = f.copy()
    disk = np.array([tuple(r) == (1, 2) for r in regions])
    f_bug[disk] = f_bug[disk][:, [0, 2, 1]]  # reverse only the disk triangles' winding
    mesh_bug = DcelData(v.copy(), f_bug, regions.copy())
    _, deg_bug, _ = compute_angles_tri(mesh_bug, unique=False)

    return {
        "ratio": ratio,
        "n_rings": n_rings,
        "true_wedge_cell_deg": pred["angle_through_cell_deg"],
        "supplement_cell_deg": pred["theta_deg"],
        "correct_mesh": {
            "through_medium_deg": float(deg_ok[DOUBLET_MEDIUM_KEY]),
            "through_cell_deg": float(deg_ok[DOUBLET_CELL_KEYS[0]]),
        },
        "disk_flipped_mesh": {
            "through_medium_deg": float(deg_bug[DOUBLET_MEDIUM_KEY]),
            "through_cell_deg": float(deg_bug[DOUBLET_CELL_KEYS[0]]),
        },
    }


def doublet_per_interface_methods(ratios: tuple[float, ...] = (0.5, 1.0, 1.5), n_rings: int = 32) -> dict:
    """Per-interface, gauge-matched tension error for every method on the doublet.

    Re-measures `Cotan`/`InvCotan`/`Equilibrium` after those two fixes and breaks out the `(1, 2)`
    flat cell-cell interface — the interface the convention question bears on.
    """
    out: dict[str, dict] = {}
    for ratio in ratios:
        mesh = dcel_from_doublet(ratio, n_rings)
        truth = ph.doublet_ground_truth_tensions(ratio, mean_tension=1.0)
        methods: dict[str, dict] = {}
        for mode in TensionComputationMethod:
            try:
                inferred = infer_tensions(mesh, mean_tension=1, mode=mode)
            except Exception as exc:  # record any method failure, do not crash the sweep
                methods[mode.name] = {"error": repr(exc)}
                continue
            scale = m.gauge_matching_scale(inferred, truth)
            matched = {k: v * scale for k, v in inferred.items()}
            errs = m.tension_relative_errors(matched, truth)
            methods[mode.name] = {
                "median": errs["median"],
                "max": float(max(errs["per_key"].values())),
                "err_interface_1_2": errs["per_key"].get((1, 2)),
                "err_interface_0_1": errs["per_key"].get((0, 1)),
            }
        out[f"ratio_{ratio}"] = {"n_rings": n_rings, "methods": methods}
    return out


def dcel_from_doublet(ratio: float, n_rings: int) -> DcelData:
    """Build a `DcelData` doublet at the shipped default azimuthal resolution."""
    v, f, regions = ph.doublet_mesh(ratio, n_rings=n_rings, n_phi=2 * n_rings)
    return DcelData(v.copy(), f.copy(), regions.copy())


def main() -> None:
    """Run from the command line and write `baseline/angle_convention.json`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="skip the 42-case sum-to-360 sweep")
    parser.add_argument("--out", type=Path, default=BASELINE_DIR / "angle_convention.json")
    args = parser.parse_args()

    result = {
        "relabelling_invariance": relabelling_invariance(),
        "wedge_convergence": wedge_convergence(),
        "tetrahedral_interior": tetrahedral_interior(),
        "disk_flip_reproduces_supplement": disk_flip_reproduces_supplement(),
        "doublet_per_interface_methods": doublet_per_interface_methods(),
    }
    if not args.quick:
        result["sum_to_360_over_dataset"] = sum_to_360_over_dataset()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_json_safe(result), indent=2))

    ri = result["relabelling_invariance"]
    print(f"relabelling invariance: max spread {max(ri['max_spread_across_permutations_deg'].values()):.2e} deg "
          f"-> {'INVARIANT (geometric)' if ri['invariant'] else 'LABEL-DEPENDENT (DEFECT)'}")
    tet = result["tetrahedral_interior"]
    print(f"tetrahedral interior: max |angle - 120| = {tet['interior_max_abs_error_from_120_deg']:.2e} deg")
    df = result["disk_flip_reproduces_supplement"]
    print(f"disk flip: through-cell {df['correct_mesh']['through_cell_deg']:.3f} (correct) "
          f"-> {df['disk_flipped_mesh']['through_cell_deg']:.3f} (flipped); "
          f"true wedge {df['true_wedge_cell_deg']:.3f}, supplement {df['supplement_cell_deg']:.3f}")
    if "sum_to_360_over_dataset" in result:
        s = result["sum_to_360_over_dataset"]
        print(f"sum-to-360: {s['n_complete_junctions']} junctions over {s['n_equilibrium_cases']} cases, "
              f"worst |sum-360| = {s['worst_abs_sum_minus_360_deg']:.2e} deg (case {s['worst_case']})")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
