#!/usr/bin/env python
"""A tissue the image cuts: crop faces, cut cells, `nan` pressures, and a pressure method that refuses.

When a cell of the mask reaches a face of the image box, the mesh closes it with triangles on that face. They are
not contact with the medium: the tissue goes on beyond the image. delaunay-watershed-3d marks them as **crop
faces**, and foambryo keeps them out of every equation:

* the tensions are inferred from the junctions the image shows; an interface made only of crop faces is absent;
* the pressures of the **cut cells** (the cells the crop cuts) are `nan`, with a `CutCellPressureWarning`, because
  part of their surface and volume was not observed;
* the Laplace pressure methods have no formulation for crop faces and raise `ValueError` on a cropped mesh
  (`PressureComputationMethod.has_crop_face_formulation` says which methods accept one: only `Variational`).

This example builds a foam of Voronoi cells, cuts away a third of it along x, and runs the inference on the cut
mask. The foam is not at mechanical equilibrium: its tensions show the workflow, not the method's accuracy. (On 433
plane crops of 47 simulated tissues at equilibrium, the tensions of the interfaces next to the crop had 14 %, 33 %
and 39 % error with 10 %, 25 % and 40 % of the tissue cut away when crop faces were read as medium contact, and 10 %,
9 % and 13 % with crop faces kept out.)

If the tissue really ends on a face of the image (for example on a coverslip), declare that face physical:
`dcel_mesh_from_segmentation_mask(mask, physical_image_faces=("axis2_low",))`; its triangles are then contact with
the medium.

    pip install "foambryo[viewing]"          # matplotlib, for the figure
    python cropped_tissue.py                 # under a minute; writes ./cropped-tissue-output/
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from foambryo import (
    CutCellPressureWarning,
    PressureComputationMethod,
    dcel_mesh_from_segmentation_mask,
    infer_pressures,
    infer_tensions,
)


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


def main() -> None:
    """Cut a foam, infer its tensions and pressures, and show what a cropped mesh refuses."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=Path("cropped-tissue-output"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    foam = generated_foam()
    cut_at = foam.shape[2] // 3
    mask = foam[:, :, cut_at:]  # the image box starts a third of the way into the tissue, along x
    print(f"the cut mask: shape {mask.shape}, {len(np.unique(mask)) - 1} cells")

    mesh = dcel_mesh_from_segmentation_mask(mask)
    print(f"{int(mesh.crop_faces.sum())} crop faces of {len(mesh.f)} triangles; cut cells {mesh.cut_cells}; "
          f"interfaces made only of crop faces: {mesh.crop_interfaces}")

    tensions = infer_tensions(mesh)
    crop_interface_kept = any(k in tensions for k in mesh.crop_interfaces)
    print(f"{len(tensions)} tensions; any crop interface among them: {crop_interface_kept}")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", CutCellPressureWarning)
        pressures = infer_pressures(mesh, tensions)  # Variational, the default
    for warning in caught:
        if issubclass(warning.category, CutCellPressureWarning):
            print(f"CutCellPressureWarning: {str(warning.message)[:120]}")
    reported = {c: p for c, p in pressures.items() if c != 0 and np.isfinite(p)}
    print(f"pressures: {len(reported)} cells reported, {sum(np.isnan(p) for p in pressures.values())} nan "
          f"(the cut cells)")

    for method in PressureComputationMethod:
        print(f"  {method.name:16s} has a formulation for crop faces: {method.has_crop_face_formulation}")
    try:
        infer_pressures(mesh, tensions, mode=PressureComputationMethod.Laplace)
    except ValueError as error:
        print(f"Laplace on a cropped mesh refuses: {error}")

    coverslip = dcel_mesh_from_segmentation_mask(mask, physical_image_faces=("axis2_low",))
    print(f"with the cut face declared physical (a coverslip): {int(coverslip.crop_faces.sum())} crop faces, "
          f"cut cells {coverslip.cut_cells}")

    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        print("matplotlib is not installed: no figure (pip install 'foambryo[viewing]')")
        return
    section = foam[foam.shape[0] // 2]
    status = np.zeros(section.shape)  # 0 medium, 1 an uncut cell, 2 a cut cell, 3 outside the image
    status[section > 0] = 1
    status[np.isin(section, mesh.cut_cells)] = 2
    status[:, :cut_at] = np.where(section[:, :cut_at] > 0, 3, 0)
    figure, axis = plt.subplots(figsize=(5.2, 4.6))
    axis.imshow(status, cmap=ListedColormap(["white", "#9ecae1", "#de2d26", "#d9d9d9"]), interpolation="nearest",
                origin="lower", vmin=0, vmax=3)
    axis.axvline(cut_at - 0.5, color="black", linewidth=1.2)
    axis.set_title("Mid z section; the image box starts at the black line.\nRed: cut cells (pressure nan). "
                   "Blue: pressure reported.\nGrey: not imaged.", fontsize=10)
    axis.set(xlabel="x (voxels)", ylabel="y (voxels)")
    figure.tight_layout()
    figure.savefig(args.output_dir / "cut_cells_in_a_section.png", dpi=150)
    print(f"figure: {args.output_dir / 'cut_cells_in_a_section.png'}")


if __name__ == "__main__":
    main()
