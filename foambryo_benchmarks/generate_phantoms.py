#!/usr/bin/env python
"""CLI: generate the analytic phantoms for both harnesses.

For each phantom case, writes:
- an exact analytic `.rec` mesh + `dict_tensions.npy` ground truth (for foambryo's
  harness, `foambryo_benchmarks/run_case.py`);
- a voxelised segmentation mask `.tif` at each of 5 resolutions (for
  `delaunay-watershed-3d`'s harness — see `delaunay-watershed-3d/dw3d_benchmarks/phantoms.py`,
  which mirrors this file's label-classifier logic so it can voxelise independently).

The generator is checked in, not the volumes (the same rule as for `dw3d`'s scaling
dataset): this script is checked in; its output directory is gitignored and meant to be regenerated on demand.

Usage:
    uv run python -m foambryo_benchmarks.generate_phantoms --output-dir /tmp/phantoms
    uv run python -m foambryo_benchmarks.generate_phantoms --output-dir /tmp/phantoms --resolutions 16 32 64
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import skimage.io as io
from dw3d.io import save_rec
from numpy.typing import NDArray

from foambryo_benchmarks import phantoms as ph
from foambryo.validation import mesh_io

DEFAULT_RESOLUTIONS = (32, 64, 128, 256, 512)

# A phantom's point classifier: (n, 3) query points -> (n,) integer labels.
LabelFunction = Callable[[NDArray[np.float64]], NDArray[np.int64]]


def _voxelize(label_fn: LabelFunction, half_extent: float, resolution: int) -> np.ndarray:
    """Rasterise `label_fn` on a `resolution^3` grid covering `[-half_extent, half_extent]^3`.

    Evaluated one z-slice at a time so peak memory is `O(resolution^2)`, not
    `O(resolution^3)` — needed since `kelvin_label_at` computes a distance to every one
    of ~250 lattice points per query point.

    Args:
        label_fn: Callable `(points: (n, 3) array) -> labels: (n,) int array`.
        half_extent (float): Half the side length of the cubic domain.
        resolution (int): Voxels per side.

    Returns:
        np.ndarray: Label volume, shape `(resolution, resolution, resolution)`, `int32`.
    """
    step = 2 * half_extent / resolution
    coords = -half_extent + step * (np.arange(resolution) + 0.5)
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    xy_flat = np.stack([xx.ravel(), yy.ravel()], axis=1)
    volume = np.zeros((resolution, resolution, resolution), dtype=np.int32)
    for iz, z in enumerate(coords):
        pts = np.column_stack([xy_flat, np.full(len(xy_flat), z)])
        volume[:, :, iz] = label_fn(pts).reshape(resolution, resolution)
    return volume


def _save_dict(path: Path, mapping: dict) -> None:
    """Save a dict as a 0-d object array, the format `run_case.py` loads with `.item()`.

    Args:
        path (Path): Destination `.npy` file.
        mapping (dict): The dict to store.
    """
    arr = np.empty((), dtype=object)
    arr[()] = mapping
    np.save(path, arr)


def _write_case(
    output_dir: Path,
    name: str,
    v: np.ndarray,
    f: np.ndarray,
    regions: np.ndarray,
    tensions: dict,
    label_fn: LabelFunction,
    half_extent: float,
    resolutions: list[int],
    pressures: dict | None = None,
) -> None:
    case_dir = output_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)

    save_rec(case_dir / f"{name}_mesh.rec", v, f, regions, binary_mode=True)
    _save_dict(case_dir / f"{name}_dict_tensions.npy", tensions)
    # Only the doublet has a ground-truth pressure (see
    # phantoms.doublet_ground_truth_pressures for why the other two do not). Without this
    # file run_case.py's pressure-error columns are absent rather than misleading.
    if pressures is not None:
        _save_dict(case_dir / f"{name}_dict_pressures.npy", pressures)

    for resolution in resolutions:
        volume = _voxelize(label_fn, half_extent, resolution)
        io.imsave(
            case_dir / f"{name}_labels_res{resolution}.tif",
            volume.astype(np.int32),
            check_contrast=False,
        )
    print(f"  {name}: mesh {v.shape[0]} verts / {f.shape[0]} tris, {len(resolutions)} mask resolutions -> {case_dir}")


def generate_all(output_dir: Path, resolutions: list[int]) -> None:
    """Generate every analytic phantom case under `output_dir`.

    Args:
        output_dir (Path): Destination directory (created if missing).
        resolutions (list[int]): Voxel-grid resolutions for the mask sweep.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for ratio in (0.5, 1.0, 1.5):
        v, f, regions = ph.doublet_mesh(ratio, n_rings=24, n_phi=48)
        # mean_tension=1 matches the gauge run_case.py's infer_tensions(mean_tension=1)
        # imposes. Without it both the tension- and pressure-error columns are dominated
        # by the scale mismatch (20% at ratio=0.5, 14% at ratio=1.5) rather than by
        # inference accuracy -- see phantoms._doublet_gauge_factor.
        tensions = ph.doublet_ground_truth_tensions(ratio, mean_tension=1.0)
        pressures = ph.doublet_ground_truth_pressures(ratio, mean_tension=1.0)
        _, radius, zc = ph._doublet_geometry(ratio, r0=1.0)
        half_extent = 1.5 * (zc + radius)
        _write_case(
            output_dir,
            f"doublet_ratio{ratio}",
            v,
            f,
            regions,
            tensions,
            lambda pts, ratio=ratio: ph.doublet_label_at(pts, ratio),
            half_extent,
            resolutions,
            pressures=pressures,
        )

    v, f, regions = ph.tetrahedral_vertex_mesh(scale=1.0)
    tensions = ph.tetrahedral_vertex_ground_truth_tensions()
    _write_case(
        output_dir,
        "tetrahedral_vertex",
        v,
        f,
        regions,
        tensions,
        lambda pts: ph.tetrahedral_vertex_label_at(pts, scale=1.0),
        1.2,
        resolutions,
    )

    v, f, regions, _cell_labels = ph.kelvin_mesh(cell_size=1.0)
    # Uniform placeholder tension on every interface actually present in the mesh: the
    # flat-faced Kelvin tiling is not a real equilibrium (see kelvin_label_at's
    # docstring), so there is no true ground truth here — this only lets the harness
    # run (rho will be nonzero and that is expected, not a defect).
    tensions = dict.fromkeys(mesh_io.list_interfaces(regions), 1.0)
    _write_case(
        output_dir,
        "kelvin",
        v,
        f,
        regions,
        tensions,
        lambda pts: ph.kelvin_label_at(pts, cell_size=1.0),
        1.8,
        resolutions,
    )


def main() -> None:
    """Generate every phantom case from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolutions", type=int, nargs="+", default=list(DEFAULT_RESOLUTIONS))
    args = parser.parse_args()
    generate_all(args.output_dir, args.resolutions)


if __name__ == "__main__":
    main()
