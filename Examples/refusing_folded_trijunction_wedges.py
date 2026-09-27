#!/usr/bin/env python
"""Run the three wedge-refusal policies on one mesh and print what each costs and buys.

Runs in a few seconds on one case and needs nothing but this repository and its sibling
`benchmarking-dataset/` — no `dw3d`, no reconstruction, no network.

What it shows
-------------
A **folded** (reflex) trijunctional edge is one whose interior wedge angle in some region is past a
straight angle: the surface is locally folded there, and the three films do not enclose the edge the
way a trijunction requires. The Young-Dupré balance every angle-reading method in this package
imposes presupposes the opposite, so such a row is an observation the geometry rules out.

It cannot be read away. `YoungDupreLocal`'s rows are built from `cos` of the wedge, and `cos` is
even about a full turn, so a folded wedge and its reflection are literally the same row — no
`angle_reading` reaches it. And `LogLami`'s rows take `log(sin(...))`, which is undefined for a
folded wedge: it does not degrade there, it raises.

`foambryo.wedge_refusal` lets the estimator **decline** the observation instead, three ways:

* `ExcludeEdgeFromLineMean` — the folded edge leaves its line's angle mean and its line's length;
  the line still reports, from what survives;
* `DropLine` — a line carrying any folded edge does not report at all;
* `DownweightLine` — nothing is removed; the line's mean is unchanged and only its row weight is
  scaled, by the fraction of its arclength that is not folded.

**The number to watch is the determined fraction, not the error.** Declining rows can leave an
interface with no surviving junction line, and it is then reported `nan` rather than given the
number least squares would have produced for a quantity nothing constrains. A policy that improves
an error statistic by declining the data it would have got wrong has improved nothing, so this
example prints the two together and refuses to print one without the other.

**Read the result carefully — on THIS mesh you will probably see nothing happen, and that is the
lesson.** The only meshes that ship with this repository are the `benchmarking-dataset`
**ground-truth** meshes: simulator output, which is barely folded at all — 5 folded edges in 14 183
over all 47 cases, on cases 004, 020 and 045. Every policy is then bitwise the shipped estimator,
which is the identity this machinery is built around. Pass `--case 004`, `--case 020` or
`--case 045` to see a mesh where something does change.

Folding is a property of *reconstructed* meshes: 2-3 % of trijunctional edges on the best
`dw3d` point placements and up to 25 % on the worst. There, excluding the folded edge lowers the
per-edge estimator's gauge-matched tension error by a fifth on average and by 41 % on one common
configuration, while `DropLine` removes a median 16-76 % of the junction *lines* — nothing to
`YoungDupreLocal`, whose row is one edge, and most of the data to every line-averaged method.

Usage:
    foambryo/.venv/bin/python Examples/refusing_folded_trijunction_wedges.py
    foambryo/.venv/bin/python Examples/refusing_folded_trijunction_wedges.py --case 020
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

from foambryo import (
    TensionComputationMethod,
    WedgeRefusal,
    WedgeRefusalPolicy,
    dcel_mesh_from_file,
    infer_tensions,
    reflex_trijunction_edges,
    refusal_report,
)

#: The policies, in the order the README introduces them.
POLICIES = (
    ("no refusal", WedgeRefusalPolicy.NoRefusal),
    ("exclude the folded edge", WedgeRefusalPolicy.ExcludeEdgeFromLineMean),
    ("drop the line", WedgeRefusalPolicy.DropLine),
    ("down-weight the line", WedgeRefusalPolicy.DownweightLine),
)

#: One per-edge method and one line-averaged one: the policies mean different things to the two.
METHODS = (TensionComputationMethod.YoungDupreLocal, TensionComputationMethod.YoungDupre)


def _dataset_dir() -> Path:
    """`benchmarking-dataset/`, as a sibling of this repository."""
    return Path(__file__).resolve().parent.parent.parent / "benchmarking-dataset"


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="004", help="benchmark case id; 004, 020 and 045 do fold")
    parser.add_argument("--dataset-dir", default=None)
    args = parser.parse_args()

    dataset = Path(args.dataset_dir) if args.dataset_dir else _dataset_dir()
    path = dataset / f"{args.case}_mesh.rec"
    if not path.exists():
        message = f"{path} not found; pass --dataset-dir"
        raise SystemExit(message)

    mesh = dcel_mesh_from_file(str(path))
    folded = reflex_trijunction_edges(mesh)
    total = refusal_report(mesh, WedgeRefusal(policy=WedgeRefusalPolicy.DropLine))
    print(f"case {args.case}: {total.n_edges} trijunctional edges on {total.n_lines} junction lines")
    print(f"  folded: {len(folded)} edges ({100 * len(folded) / total.n_edges:.2f} %), "
          f"carrying {100 * total.length_refused / total.length_total:.2f} % of the junction arclength")
    if not folded:
        print("  nothing to decline on this mesh, so every policy below must be bitwise the default.")
    print()

    for method in METHODS:
        unit = "edge" if method is TensionComputationMethod.YoungDupreLocal else "line"
        print(f"{method.name}  (its observation unit is one junction {unit})")
        print(f"  {'policy':26s} {'rows kept':>12s} {'determined':>12s}  tensions")
        baseline = None
        for label, policy in POLICIES:
            refusal = WedgeRefusal(policy=policy)
            report = refusal_report(mesh, refusal, per_edge_rows=unit == "edge")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tensions = infer_tensions(mesh, 1.0, method, angle_reading="oriented_chord", refusal=refusal)
            finite = [value for value in tensions.values() if np.isfinite(value)]
            determined = len(finite) / len(tensions)
            if baseline is None:
                baseline = tensions
                note = ""
            elif all(
                tensions[key] == baseline[key] or (np.isnan(tensions[key]) and np.isnan(baseline[key]))
                for key in baseline
            ):
                note = "  bitwise the default"
            else:
                note = f"  mean {np.mean(finite):.4f}, spread {np.ptp(finite):.4f}"
            print(
                f"  {label:26s} {report.n_units_used:6d}/{report.n_units:<5d} "
                f"{len(finite):6d}/{len(tensions):<5d} {determined:6.1%}{note}",
            )
        print()

    print("The determined fraction is not a formality: read it beside every error you quote.")
    print("An interface with no surviving junction line is reported nan, never a number.")


if __name__ == "__main__":
    main()
