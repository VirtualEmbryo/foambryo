#!/usr/bin/env python
"""A mask whose voxels are longer along z: give the spacing, and the mask is resampled to cubic voxels first.

A microscope stack usually has a z step larger than its pixel size. This example builds a foam of Voronoi cells on
cubic voxels of 0.5 um, keeps every second z slice (voxels of 1.0 x 0.5 x 0.5 um, an anisotropy of 2), and shows
what giving the spacing changes:

* the estimate of the reconstruction's time and memory counts the voxels **after** resampling (twice as many);
* `dcel_mesh_from_segmentation_mask(mask, spacing=(1.0, 0.5, 0.5))` resamples the mask to cubic voxels of 0.5 um
  by shape-based interpolation, meshes it, and returns the mesh **in um**; `mesh.source_spacing` records the
  spacing, and so does the `.rec` file that `reconstruct_mask` writes (the `foambryo mesh` route);
* the cells' volumes, in um^3, agree with their voxel counts times the voxel volume;
* without the spacing, the same stack is meshed in voxel units and squashed along z: every volume is half what it
  should be, and every contact angle is read on a deformed tissue.

The accuracy gain of resampling first is measured, not shown here: on 12 simulated tissues at equilibrium with cells
as small as those of real tissues, down-sampled along z, the contact-angle error grew by +0.83, +0.71 and +1.17
degrees at an anisotropy of 5/3, 2 and 3 when the mask is resampled first, against +1.38, +1.97 and +4.10 when the
native grid is meshed and the mesh scaled afterwards (each against the same tissue meshed at full resolution; see
BENCHMARKS.md). One small example cannot show a difference of a degree: its own seed-to-seed spread is larger.

    pip install "foambryo[viewing]"          # matplotlib, for the figure
    python anisotropic_voxels.py             # under a minute; writes ./anisotropic-voxels-output/
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
from dw3d import resample_to_isotropic
from dw3d.io import load_rec_appendix
from scipy.spatial import cKDTree

from foambryo import dcel_mesh_from_file, dcel_mesh_from_segmentation_mask, infer_tensions, reconstruct_mask
from foambryo.reconstruction_cost import estimate_reconstruction_cost, voxels_handed_to_dw3d

VOXEL_UM = 0.5  # the cubic voxels the foam is generated on


def generated_foam(n: int = 96, n_cells: int = 12, seed: int = 7) -> np.ndarray:
    """Voronoi cells filling a ball, on an `n`^3 grid of cubic voxels; label 0 is the medium."""
    rng = np.random.default_rng(seed)
    radius, centre = 0.46 * n, (n - 1) / 2
    seeds: list[np.ndarray] = []
    while len(seeds) < n_cells:  # well-separated seeds inside the ball
        point = rng.uniform(-0.75 * radius, 0.75 * radius, 3)
        if np.linalg.norm(point) < 0.75 * radius and all(np.linalg.norm(point - s) > 0.35 * radius for s in seeds):
            seeds.append(point)
    grid = np.indices((n, n, n)).reshape(3, -1).T.astype(float)
    _, nearest = cKDTree(np.asarray(seeds) + centre).query(grid)
    inside = np.linalg.norm(grid - centre, axis=1) <= radius
    return np.where(inside, nearest + 1, 0).reshape(n, n, n).astype(np.uint16)


def volume_ratios(mesh: object, mask: np.ndarray, voxel_volume: float) -> np.ndarray:
    """Each cell's mesh volume divided by its voxel count times the voxel volume (1 when the units agree)."""
    volumes = mesh.compute_volumes_cells()
    labels, counts = np.unique(mask[mask > 0], return_counts=True)
    return np.array([volumes[int(k)] / (c * voxel_volume) for k, c in zip(labels, counts, strict=True)])


def main() -> None:
    """Mesh the stack with and without its spacing, write its .rec file, and draw the resampling."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=Path("anisotropic-voxels-output"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stack = generated_foam()[::2]  # every second z slice: voxels twice as long along z
    spacing = (2 * VOXEL_UM, VOXEL_UM, VOXEL_UM)  # (dz, dy, dx) in um: axis 0 of the mask is z
    print(f"the stack: shape {stack.shape}, {len(np.unique(stack)) - 1} cells, spacing {spacing} um")

    voxels = voxels_handed_to_dw3d(stack.shape, spacing)
    estimate = estimate_reconstruction_cost(voxels)
    print(f"voxels handed to the reconstruction: {voxels:.3g} (the stack has {stack.size:.3g}); estimated "
          f"{estimate.seconds[1]:.1f} s and {estimate.peak_gib[1]:.2f} GiB (fitted on tissues of 2 to 1 866 cells)")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mesh = dcel_mesh_from_segmentation_mask(stack, spacing=spacing)
    for warning in caught:
        if issubclass(warning.category, DeprecationWarning):
            continue  # notices about the libraries' own internals
        print(f"  {warning.category.__name__}: {str(warning.message)[:110]}")
    extent = mesh.v.max(axis=0) - mesh.v.min(axis=0)
    print(f"with spacing: mesh in um, extent (z, y, x) = {np.round(extent, 1)}, source_spacing {mesh.source_spacing}")
    ratios = volume_ratios(mesh, stack, voxel_volume=float(np.prod(spacing)))
    print(f"  cell volume / (voxels x voxel volume): median {np.median(ratios):.3f}")

    squashed = dcel_mesh_from_segmentation_mask(stack)  # no spacing: voxel units, z not scaled
    extent = squashed.v.max(axis=0) - squashed.v.min(axis=0)
    ratios = volume_ratios(squashed, stack, voxel_volume=float(np.prod(spacing)) / VOXEL_UM**3)
    print(f"without spacing: mesh in voxels, extent (z, y, x) = {np.round(extent, 1)}: the tissue is squashed along z;"
          f"\n  cell volume / (voxels x true voxel volume): median {np.median(ratios):.3f}")

    # The `foambryo mesh` route: the same mesh, written as a .rec file that records its frame and spacing.
    reconstruction = reconstruct_mask(stack, spacing, spacing_unit="um")
    rec = args.output_dir / "stack.rec"
    reconstruction.write_rec(rec)
    appendix = load_rec_appendix(rec)
    print(f"{rec}: {reconstruction.summary()}; recorded "
          f"{ {k: appendix[k] for k in ('coordinate_frame', 'spacing', 'spacing_unit', 'resampling')} }")
    read_back = dcel_mesh_from_file(rec)
    print(f"  read back: source_spacing {read_back.source_spacing}")

    tensions = infer_tensions(mesh)
    print(f"tensions of {len(tensions)} interfaces, mean {np.nanmean(list(tensions.values())):.3f} (normalised to 1)")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed: no figure (pip install 'foambryo[viewing]')")
        return
    resampled = resample_to_isotropic(stack, spacing)
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    for axis, (title, image, aspect) in zip(axes, (
        ("the stack: voxels 1.0 x 0.5 um (z, x)", stack[:, stack.shape[1] // 2, :], 2.0),
        ("resampled to cubic voxels of 0.5 um", resampled[:, resampled.shape[1] // 2, :], 1.0),
    ), strict=True):
        colours = plt.get_cmap("tab20").with_extremes(bad="white")  # the medium, label 0, in white
        axis.imshow(np.ma.masked_equal(image, 0), cmap=colours, interpolation="nearest", aspect=aspect,
                    origin="lower")
        axis.set(title=title, xlabel="x (voxels)", ylabel="z (voxels)")
    figure.suptitle("A generated foam (12 cells), one xz section through its middle; colour = cell label")
    figure.tight_layout()
    figure.savefig(args.output_dir / "stack_and_its_resampling.png", dpi=150)
    print(f"figure: {args.output_dir / 'stack_and_its_resampling.png'}")


if __name__ == "__main__":
    main()
