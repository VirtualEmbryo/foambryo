#!/usr/bin/env python
"""A large tissue: the cost estimate, the sparse solves, and the table of every linear solve.

Every tension and pressure foambryo infers is the solution of a linear least-squares system. Small systems are
solved densely, exactly; a dense system grows as the product of its dimensions in memory and as their cube in time,
so `solver="auto"` (the default) solves a system sparsely from `sparse_threshold` unknowns on (2 000 by default).
This example generates a foam of several hundred Voronoi cells, and:

1. estimates the reconstruction's time and memory before running it (`foambryo.reconstruction_cost`: power laws
   fitted on one machine to medians over the benchmark tissues of 2 to 10 cells and to single tissues of 90 to
   1 866 cells; the band is the fit's, not a guarantee);
2. reconstructs the mesh and infers tensions and pressures with `diagnostics=True`, which returns one record per
   linear solve (the solver used, the system's size, rank, residual, condition number, time). It calls
   `infer_tensions_then_pressures`, the two calls of `infer_forces` made one after the other, as `foambryo infer`
   does: if an interface has no triple line, its tension is undetermined (`nan`), the pressure solve refuses, and
   the tensions are kept instead of lost;
3. writes them as `solves.csv`, the table `foambryo infer --diagnostics basic` writes (from a direct call with
   `diagnostics=True`, `save_solve_diagnostics` writes the same records as JSON and NPZ files);
4. solves the tensions again with the dense and with the sparse solver forced, and compares them.

The foam is not at mechanical equilibrium: its tensions show the workflow and the solvers, not the method's
accuracy. On a 1 866-cell tissue the sparse and dense tensions agree to 1.5e-11 of their scale and the pressures to
1e-11; the dense solve took 6 to 9 minutes and 14-18 GiB there, the sparse one 9 s and 1 GiB.

    python large_tissue_solvers.py                # about a minute and 5 GiB; writes ./large-tissue-output/
    python large_tissue_solvers.py --cells 300    # smaller and faster
"""

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from foambryo import (
    PressureComputationMethod,
    dcel_mesh_from_segmentation_mask,
    infer_tensions,
    infer_tensions_then_pressures,
)
from foambryo.reconstruction_cost import estimate_reconstruction_cost, voxels_handed_to_dw3d
from foambryo.results import write_solve_table


def generated_foam(n_cells: int, cell_radius: float = 12.0, seed: int = 3) -> np.ndarray:
    """About `n_cells` Voronoi cells of radius ~`cell_radius` voxels filling a ball; label 0 is the medium."""
    rng = np.random.default_rng(seed)
    radius = cell_radius * n_cells ** (1 / 3) * 1.05
    n = int(2 * radius) + 8
    centre = (n - 1) / 2
    candidates = rng.uniform(-radius, radius, (40 * n_cells, 3))
    candidates = candidates[np.linalg.norm(candidates, axis=1) < radius - 0.6 * cell_radius]
    seeds: list[np.ndarray] = []
    for point in candidates:  # a simple dart throw: seeds at least 1.4 cell radii apart
        if len(seeds) == n_cells:
            break
        if not seeds or np.min(np.linalg.norm(np.asarray(seeds) - point, axis=1)) > 1.4 * cell_radius:
            seeds.append(point)
    grid = np.indices((n, n, n)).reshape(3, -1).T.astype(float)
    _, nearest = cKDTree(np.asarray(seeds) + centre).query(grid)
    inside = np.linalg.norm(grid - centre, axis=1) <= radius
    return np.where(inside, nearest + 1, 0).reshape(n, n, n).astype(np.uint32)


def scale_relative_difference(a: dict, b: dict) -> float:
    """The largest |a - b| over the shared finite values, relative to the largest |b| (the solution's scale)."""
    keys = [k for k in a if k in b and np.isfinite(a[k]) and np.isfinite(b[k])]
    return max(abs(a[k] - b[k]) for k in keys) / max(abs(b[k]) for k in keys)


def main() -> None:
    """Estimate, reconstruct, infer with diagnostics, and compare the dense and sparse solves."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cells", type=int, default=600)
    parser.add_argument("--output-dir", type=Path, default=Path("large-tissue-output"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    mask = generated_foam(args.cells)
    n_cells = len(np.unique(mask)) - 1
    print(f"generated foam: shape {mask.shape}, {n_cells} cells, {time.perf_counter() - started:.0f} s")

    estimate = estimate_reconstruction_cost(voxels_handed_to_dw3d(mask.shape, (1.0, 1.0, 1.0)))
    print(f"estimated reconstruction: {estimate.seconds[1]:.0f} s ({estimate.seconds[0]:.0f}-{estimate.seconds[2]:.0f})"
          f", {estimate.peak_gib[1]:.1f} GiB ({estimate.peak_gib[0]:.1f}-{estimate.peak_gib[2]:.1f})")
    started = time.perf_counter()
    mesh = dcel_mesh_from_segmentation_mask(mask)
    print(f"reconstructed: {len(mesh.v)} vertices, {len(mesh.f)} triangles, {time.perf_counter() - started:.0f} s")

    started = time.perf_counter()
    outcome = infer_tensions_then_pressures(
        mesh, mode_pressure=PressureComputationMethod.WeightedLaplace, diagnostics=True,
    )
    undetermined = sum(not np.isfinite(t) for t in outcome.tensions.values())
    print(f"{outcome.library_call}: {outcome.status}, {time.perf_counter() - started:.1f} s; "
          f"{len(outcome.tensions)} tensions ({undetermined} undetermined), "
          f"{'no' if outcome.pressures is None else len(outcome.pressures)} pressures")
    if outcome.error:
        print(f"  {outcome.error[:160]}...")
    print(f"  {'system':48s} {'solver':22s} {'rows':>7s} {'unknowns':>8s} {'condition':>10s} {'seconds':>8s}")
    for record in outcome.records:
        condition = f"{record.condition_number:.3g}" if record.condition_number is not None else "-"
        print(f"  {record.system[:48]:48s} {record.solver[:22]:22s} {record.n_rows:7d} {record.n_unknowns:8d} "
              f"{condition:>10s} {record.seconds:8.2f}")
    write_solve_table(args.output_dir / "solves.csv", outcome.records)
    print(f"wrote {args.output_dir / 'solves.csv'}")

    timings = {}
    solutions = {}
    for solver in ("dense", "sparse"):
        started = time.perf_counter()
        solutions[solver] = infer_tensions(mesh, solver=solver)
        timings[solver] = time.perf_counter() - started
    print(f"tensions, dense solve {timings['dense']:.1f} s, sparse solve {timings['sparse']:.1f} s; they differ by "
          f"{scale_relative_difference(solutions['sparse'], solutions['dense']):.1e} of the tensions' scale")


if __name__ == "__main__":
    main()
