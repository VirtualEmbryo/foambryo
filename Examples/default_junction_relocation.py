#!/usr/bin/env python
"""What dw3d 0.5.0's default junction relocation changes in the tensions foambryo infers.

Since `delaunay-watershed-3d` 0.5.0 every reconstruction moves the finished mesh's trijunction
vertices onto the trijunction curves the segmentation mask itself gives (triangles and labels are
untouched). This example reconstructs one timepoint twice -- with the default and with
`relocate_junctions = False`, the 0.4 behaviour -- infers Young-Dupre tensions on both meshes, and
draws the trijunction line the relocation moved most, beside the mask's own curve, with the three
tensions that meet there before and after.

Run it on your own label image (exterior `0`), for example one timepoint of a time series::

    python Examples/default_junction_relocation.py path/to/labels.tif --out junction.png

Needs `foambryo`, `delaunay-watershed-3d>=0.5.0` and, for the figure, `matplotlib`.
Tensions are relative (the gauge is fixed by the solver's own normalisation), so compare the two
columns, not the absolute values.
"""

from __future__ import annotations

import argparse

import numpy as np
import skimage.io as io
from dw3d import extract_junction_curves, get_default_mesh_reconstruction_algorithm
from dw3d.junction_curves import mesh_trijunction_components

from foambryo import TensionComputationMethod, infer_tensions
from foambryo.dcel import DcelData


def reconstruct(mask: np.ndarray, min_distance: int, relocate: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """dw3d's default reconstruction, with junction relocation set explicitly."""
    algorithm = get_default_mesh_reconstruction_algorithm(min_distance=min_distance)
    algorithm.relocate_junctions = relocate
    return algorithm.construct_mesh_from_segmentation_mask(mask)


def shared_limits(point_sets: list[np.ndarray], margin: float = 1.0) -> list[tuple[float, float]]:
    """One (low, high) per axis covering every point set, padded to a cube so both panels match."""
    everything = np.vstack(point_sets)
    low, high = everything.min(axis=0) - margin, everything.max(axis=0) + margin
    half = (high - low).max() / 2
    centre = (high + low) / 2
    return [(c - half, c + half) for c in centre]


def main() -> None:
    """Reconstruct twice, infer tensions twice, print and draw one junction."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mask", help="label image, exterior 0")
    parser.add_argument("--min-distance", type=int, default=3)
    parser.add_argument("--out", default="default_junction_relocation.png")
    arguments = parser.parse_args()
    mask = io.imread(arguments.mask)

    points, triangles, labels = reconstruct(mask, arguments.min_distance, relocate=True)
    base_points, base_triangles, base_labels = reconstruct(mask, arguments.min_distance, relocate=False)
    assert np.array_equal(triangles, base_triangles), "relocation must not change connectivity"
    assert np.array_equal(labels, base_labels), "relocation must not change labels"

    # One tension per interface, keyed by the (smaller, larger) label pair.
    tensions = infer_tensions(DcelData(points, triangles, labels), 1, TensionComputationMethod.YoungDupre)
    base_tensions = infer_tensions(
        DcelData(base_points, base_triangles, base_labels), 1, TensionComputationMethod.YoungDupre,
    )

    # The trijunction line the relocation moved most, among lines with one mesh counterpart.
    curves = extract_junction_curves(mask, base_points, base_triangles, base_labels)
    components = mesh_trijunction_components(base_points, base_triangles, base_labels)
    best = None
    for curve in curves.matched:
        parts = components.get(tuple(int(x) for x in curve.triple), [])
        if len(parts) != 1 or len(parts[0]["vertices"]) < 6:
            continue
        vertices = parts[0]["vertices"]
        moved_by = float(np.median(np.linalg.norm(points[vertices] - base_points[vertices], axis=1)))
        if best is None or moved_by > best[0]:
            best = (moved_by, curve, parts[0])
    moved_by, curve, line = best
    a, b, c = (int(x) for x in curve.triple)
    print(f"trijunction ({a}, {b}, {c}): median vertex displacement {moved_by:.2f} voxels")
    print(f"{'interface':>12}  {'relocation off':>14}  {'default':>10}")
    for pair in ((a, b), (a, c), (b, c)):
        before, after = base_tensions.get(pair, float("nan")), tensions.get(pair, float("nan"))
        print(f"{pair!s:>12}  {before:>14.4f}  {after:>10.4f}")

    import matplotlib.pyplot as plt  # optional dependency, only for the figure

    # Both panels share one set of limits and ticks, so the two lines can be compared by eye.
    vertices = line["vertices"]
    limits = shared_limits([base_points[vertices], points[vertices], curve.points])
    fig = plt.figure(figsize=(10, 4.5))
    for panel, (mesh_points, title) in enumerate(((base_points, "relocate_junctions=False (0.4)"),
                                                  (points, "default (dw3d 0.5.0)"))):
        ax = fig.add_subplot(1, 2, panel + 1, projection="3d")
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        ax.set_zlim(*limits[2])
        for axis, (low, high) in zip((ax.xaxis, ax.yaxis, ax.zaxis), limits, strict=True):
            axis.set_ticks(np.linspace(low, high, 5).round(1))
        for v1, v2 in line["edges"]:
            ax.plot(*np.stack([mesh_points[v1], mesh_points[v2]]).T, color="k", linewidth=2)
        ax.plot(*curve.points.T, color="tab:red", linestyle="--", label="mask trijunction curve")
        ax.set_title(title)
        ax.set_xlabel("x (voxels)")
        ax.set_ylabel("y (voxels)")
        ax.set_zlabel("z (voxels)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(arguments.out, dpi=120)
    print(f"saved {arguments.out}")


if __name__ == "__main__":
    main()
