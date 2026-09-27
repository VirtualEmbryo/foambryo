#!/usr/bin/env python
"""Infer tensions on one benchmark mesh both ways, and print the two contrasts side by side.

Runs in well under a minute on one case and needs nothing but this repository and its sibling
`benchmarking-dataset/` — no `dw3d`, no reconstruction, no network.

What it shows
-------------
`foambryo` reads a trijunction contact angle as the angle between two incident triangles' flat
faces. A flat triangle has no curvature to report, so on a curved interface that is the angle of
the *chord* to the triangle's far vertex, not the interface's *tangent* at the junction. The two
differ by `kappa * L / 2` for a circular arc, with `L` the distance from the junction to the
triangle's far vertex. `angle_reading="quadric"` fits a generic quadric to the interface's own
mesh vertices around each junction and reads the tangent instead.

The mesh is not modified either way: no vertex moves, no area or volume changes. Only the angle
is read differently.

**Read the result carefully — on THIS mesh the correction is expected to make things slightly
worse, and that is the lesson.** The only meshes that ship with this repository are the
`benchmarking-dataset` **ground-truth** meshes: simulator output, far finer than anything a
reconstruction produces. Their `L` is small, so their chord deficit is small — MEASURED, all 47
cases: the chord reading's contrast is already 1.000 to within 2e-5, and the quadric reading
moves it to a median of 1.080, worse on **47 of 47** cases. The correction over-applies at fine
spacing (about +26 % on a mesh where the truth is known, from dw3d's identifiability-floor work).

The reading is for **coarse reconstructed meshes**, where the chord deficit is large: on
`dw3d`-reconstructed meshes of these same 47 cases the contrast goes 0.738 -> 0.954 at
`min_distance = 3` and 0.722 -> 1.056 at 4. See `BENCHMARKS.md`. This example prints the
correction's own scale (the median rotation it applies) so the two regimes are distinguishable
by eye rather than by assertion.

What "contrast" means here
--------------------------
The ordinary-least-squares slope of gauge-matched inferred tension on ground-truth tension over
one case's interfaces. **1.000 is perfect.** Below 1 means the inference under-separates the
tensions — it compresses them toward their mean, which is what reading a chord where the geometry
has a tangent does.

Usage:
    foambryo/.venv/bin/python Examples/second_order_angle_reading.py
    foambryo/.venv/bin/python Examples/second_order_angle_reading.py --case 007
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

from foambryo import (
    AngleReadingCoverageWarning,
    TensionComputationMethod,
    dcel_mesh_from_file,
    infer_tensions_and_residuals_with_report,
)
from foambryo.second_order_angle_reading import junction_frames, quadric_angle_rotations

REPOSITORY = Path(__file__).resolve().parent.parent
DATASET = REPOSITORY.parent / "benchmarking-dataset"


def contrast(inferred: dict[tuple[int, int], float], truth: dict[tuple[int, int], float]) -> float:
    """Gauge-matched OLS slope of inferred tension on true tension. 1.000 is perfect.

    The gauge is matched first because only tension *ratios* are inferable: the mean of the
    inferred tensions is rescaled onto the mean of the true ones before the slope is taken, so
    the slope measures how well the *spread* is recovered rather than an arbitrary overall scale.

    Args:
        inferred (dict[tuple[int, int], float]): inferred tension per interface.
        truth (dict[tuple[int, int], float]): ground-truth tension per interface.

    Returns:
        float: the slope, or `nan` if fewer than three interfaces are shared and finite.
    """
    shared = sorted(
        key for key in set(inferred) & set(truth)
        if np.isfinite(inferred[key]) and np.isfinite(truth[key])
    )
    if len(shared) < 3:
        return float("nan")
    predicted = np.array([inferred[key] for key in shared], dtype=float)
    actual = np.array([truth[key] for key in shared], dtype=float)
    predicted = predicted * (actual.mean() / predicted.mean())
    return float(np.polyfit(actual, predicted, 1)[0])


def main() -> None:
    """Infer both ways on one case and print the comparison."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", default="000", help="benchmark case identifier, e.g. 000")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument(
        "--method", default="YoungDupre",
        help="any TensionComputationMethod that reads angles; YoungDupre is the recommended one",
    )
    options = parser.parse_args()

    mesh_path = options.dataset / f"{options.case}_mesh.rec"
    truth_path = options.dataset / f"{options.case}_dict_tensions.npy"
    if not mesh_path.exists():
        message = (
            f"{mesh_path} not found. This example reads the sibling `benchmarking-dataset/` "
            f"directory; pass --dataset if yours is elsewhere."
        )
        raise SystemExit(message)

    mesh = dcel_mesh_from_file(str(mesh_path))
    truth = {
        tuple(sorted(int(v) for v in key)): float(value)
        for key, value in np.load(truth_path, allow_pickle=True).item().items()
    }
    mode = getattr(TensionComputationMethod, options.method)

    results = {}
    for reading in ("chord", "quadric"):
        # The report-returning entry point hands back the coverage numbers instead of warning;
        # `infer_tensions(...)` would warn instead, which is the right choice for a one-liner but
        # not for a script that wants to print the coverage.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AngleReadingCoverageWarning)
            _tensions, dict_tensions, _residuals, report = infer_tensions_and_residuals_with_report(
                mesh, 1, mode, angle_reading=reading,
            )
        inferred = {tuple(sorted(int(v) for v in key)): float(value) for key, value in dict_tensions.items()}
        results[reading] = {"contrast": contrast(inferred, truth), "report": report, "tensions": inferred}

    print(f"case {options.case}, method {options.method}, {len(truth)} ground-truth interfaces")
    print()
    print(f"  {'reading':10s} {'contrast':>9s}   (1.000 is perfect)")
    for reading in ("chord", "quadric"):
        print(f"  {reading:10s} {results[reading]['contrast']:9.4f}")
    print()

    report = results["quadric"]["report"]
    if report.coverage_fraction is None:
        print("  the quadric reading found no trijunction to correct on this mesh")
    else:
        print(f"  quadric coverage: {report.n_corrected}/{report.n_incident_faces} trijunction faces "
              f"({report.coverage_fraction:.4f})")
        if report.uncorrected_lines:
            print(f"  {len(report.uncorrected_lines)} of {len(report.line_coverage)} trijunction lines are read "
                  f"with a MIXTURE of the two conventions, because the quadric fit was")
            print("  inadmissible on part of them and those faces kept the chord reading:")
            for line in report.uncorrected_lines[:5]:
                done, total = report.line_coverage[line]
                print(f"    line {line}: {done}/{total} faces corrected")
            if len(report.uncorrected_lines) > 5:
                print(f"    ... and {len(report.uncorrected_lines) - 5} more")
        else:
            print("  every trijunction line was corrected throughout")
    print()
    rotations, _report = quadric_angle_rotations(mesh)
    degrees = np.degrees(np.abs(np.array(list(rotations.values()), dtype=float)))
    levers = np.array([frame["lever"] for frame in junction_frames(mesh)], dtype=float)
    if len(degrees):
        print(f"  the correction applied: median {np.median(degrees):.3f} deg, "
              f"p90 {np.percentile(degrees, 90):.3f} deg, over a median lever of "
              f"{np.median(levers):.4f} mesh units")
    print()
    print("  The mesh is identical under both readings -- no vertex moved. Only the angle changed.")
    print()
    print("  NOTE. This is a GROUND-TRUTH mesh from the benchmark, and it is far finer than any")
    print("  reconstruction. Its chord deficit is already small, so the correction's known")
    print("  over-application at fine spacing dominates and the contrast gets slightly WORSE.")
    print("  Measured over all 47 reference meshes: chord |contrast-1| 1.9e-05, quadric 0.080,")
    print("  chord closer on 47 of 47. On dw3d-RECONSTRUCTED meshes of the same 47 cases the")
    print("  contrast goes 0.738 -> 0.954 (min_distance 3) and 0.722 -> 1.056 (4). See")
    print("  BENCHMARKS.md. Use angle_reading='quadric' on reconstructions, not on simulator")
    print("  output.")


if __name__ == "__main__":
    main()
