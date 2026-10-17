# Release notes — 0.7.0, 17 October 2026

`foambryo` and `delaunay-watershed-3d` ship together as 0.7.0, the first release on PyPI since 0.5.1. 0.6.0 was prepared but not published; its changes are part of 0.7.0 and are included below. This is what changes for you and what can mislead you; `CHANGELOG.md` has the full list, and dw3d's `RELEASE_NOTES.md` covers reconstruction. Install both with `pip install --upgrade foambryo delaunay-watershed-3d`.

## What changes for you: results that differ from 0.5.1

1. **Meshes of cells that touch the image border change, and so do the tensions and pressures inferred from them.** The border triangles were wound the wrong way (in delaunay-watershed-3d), and a cell closed by the image border read too small a volume, often a negative one. On ten benchmark masks cut through their middle, the inferred tensions changed by up to 199 %, and every cell pressure by 37–139 %. Reconstruct and infer again any tissue cropped through its cells.
2. **Cropped tissues: crop faces are kept out of the equations, and cut cells get `nan` pressures.** Where a cell reaches the image box, the mesh's crop faces used to be read as contact with the medium. On 433 plane crops of the benchmark tissues, the tensions of the interfaces next to the crop had 13–40 % error that way, and 9–12 % now; the cut cells' pressures, reported before, had 15–46 % error and are now `nan` with a `CutCellPressureWarning`. The Laplace pressure methods, `infer_tensions_with_uncertainty` and `compute_inference_diagnostics` refuse a cropped mesh. On a mesh without crop faces nothing changes.
3. **The normalisation is exact, so the scale of the tensions changes.** The returned tensions now have exactly the mean `mean_tension` (per connected piece of the junction graph). Tension ratios are unchanged; the overall scale changes by one factor per mesh: for `YoungDupre` by 3e-6 typically and up to 1.4 %, for `YoungDupreLocal` up to 12 %, for `InvCotan` up to 8 %, for `Equilibrium` by 10 % typically. Pressures computed from those tensions scale by the same factor.
4. **`ForceBalanceMAP` gives different tensions at a mesh's own scale.** It no longer depends on the mesh's length unit nor on `mean_tension` (rescaling a mesh used to move its relative tensions by up to 43 %). On 45 reconstructions with a simulation truth, the median of the per-case median relative tension error fell from 0.177 to 0.120 (better on 35, worse on 10).
5. **A cell that no Laplace equation determines gets a `nan` pressure**, with an `UndeterminedPressureWarning`, instead of silently the medium's pressure. The variational method (the default) is not affected.
6. **`Cotan`, `InvCotan`, `Lami` and `LogLami` leave out flat junction lines** (wedges of exactly 0 or 180 degrees), with a `DegenerateWedgeWarning`, instead of solving a system with infinite coefficients. On every mesh without such a line, every number is unchanged.
7. **The first-order error bars of `YoungDupre` are larger**: they include a term they missed (a median 3 % larger on reconstructions of the benchmark tissues, 3.9–4.7 % on the same tissues voxelised at real cell size, and 21–23 % on simulated tissues of 90 and 165 cells at real cell size), and they are documented as a lower bound (see below).
8. **The viewers' colours changed.** Tensions are drawn with `coolwarm`, light grey at the mean tension, blue below, red above; pressures with `magma` trimmed to its 10–90 % range; undetermined values in green. Figures made with 0.5 are not comparable colour for colour.
9. **dw3d's `set_pixel_size` scaled the wrong axes** when the xy and z sizes differ: re-run any analysis that used it (dw3d's notes).
10. **With `Laplace` or `WeightedLaplace` and a non-zero `base_pressure`, every cell's pressure is now `base_pressure` lower**: it used to be added twice. Nothing changes at the default `base_pressure = 0` or with the variational method.
11. **Higher minimum versions**, the oldest with which the test suite passes (on Python 3.11): delaunay-watershed-3d 0.7.0, numpy 2.0.0, scipy 1.13.0, networkx 3.2, trimesh 4.0.0; for the extras polyscope 2.6.1, matplotlib 3.8.4, tifffile 2023.1.23 and torch 2.3.0. The old floors (numpy 1.21.6, scipy 1.4.1, …) were never tested, and their releases have no wheels for Python 3.11. foambryo 0.7.0 needs delaunay-watershed-3d 0.7.0: the command's `mesh` and `infer-over-seeds` call its new physical-units reconstruction.

## New

* **Masks with anisotropic voxels**: `dcel_mesh_from_segmentation_mask(mask, spacing=(dz, dy, dx))` resamples the mask to cubic voxels first and returns the mesh in the unit of the spacing.
* **Large tissues**: sparse solves from 2 000 unknowns, `solver="auto" | "dense" | "sparse"`, `diagnostics=True`, and an estimate of the reconstruction's time and memory (`foambryo.reconstruction_cost`). Below 2 000 unknowns every result is bitwise what it was. On a 1 866-cell tissue meshed on its native grid, from mask to tensions, 0.7.0 took 9.5 min and 30.9 GiB of peak memory where 0.5.1 took 47.5 min and 54.7 GiB (one machine, nothing else running; the tension solve itself 0.4 s instead of 226 s). Resampled to cubic voxels first, the same tissue's reconstruction takes about 13.5 min and 38 GiB.
* **Seed ensembles**: `infer_tensions_over_seeds`.
* **The `foambryo` command**: `mesh`, `infer`, `infer-over-seeds`, `view`.
* **The viewer**: colour bars, length scale bar, explode and cut, backgrounds, saved figures, export, a window that starts from a mask, seed-ensemble results.
* **Public helpers for other front ends**: `foambryo.results`, `foambryo.colour_scales`, `foambryo.segmentation_masks`, `reconstruct_mask`, `infer_tensions_then_pressures`.
* **Documentation**: a user guide and examples in `docs/`, and a documentation website built from them.

## What can mislead you

1. **A seed ensemble's spread is a lower bound on the error, not an error bar.** A single seed's ±1 SD interval held the true tension for about half of the interfaces on synthetic tissues. Fix the seed list in advance.
2. **No error bar is calibrated on a reconstructed mesh.** One reconstruction's first-order `± std` held the true tension for 26–54 % of interfaces (nominal 68 %), `± 2 std` for 48–79 % (nominal 95 %).
3. **`combine="angles"` is experimental.** On synthetic masks degraded by a smooth random warp of about one voxel, it lost its advantage over the default on tissues of 90 and 165 cells while keeping it on small aggregates.
4. **Anisotropy above 3 was not measured**; delaunay-watershed-3d warns and builds the mesh anyway. Keep the default angle reading on such meshes: the quadric reading made every route worse at real cell size.
5. **Error bars stop at about 1 300 cells.** `infer_tensions_with_uncertainty` and `compute_inference_diagnostics` decompose a dense operator that scipy cannot take past `2**31 - 1` entries; on a larger mesh they refuse at once. Use `infer_tensions_over_seeds` for a seed spread (a lower bound on the error).
6. **The reconstruction estimate is a fit**, on one machine, to medians over the benchmark tissues (2 to 10 cells) and to single tissues of 90 to 1 866 cells; its band is the largest in-sample ratio, not a guarantee.
7. The limitations listed for 0.5.0 below still hold, except 7, fixed in 0.7.0 (item 4 above), and the end of 8: the faster neighbour search planned for 0.5.1 is in 0.7.0, and on a 1 866-cell tissue the relocation guard takes 45 s instead of 6.6 min, with the same meshes (dw3d's `CHANGELOG.md`).

## Credits

Recent work on dw3d and foambryo is part of a collaboration with Alejandro Aguilera Castrejon (Janelia Research Campus, Howard Hughes Medical Institute). We thank him and his Janelia colleagues Adam L. Taylor, Wenhong Li and Francesco Roncato. Adam L. Taylor found the border-triangle winding defect fixed in 0.6.0 and pointed us to the seed-extraction and dense tension-solve bottlenecks on large tissues.

# Release notes — 0.5.1

**0.5.1 is the final 0.5.0 (as on GitHub's `main` since 27 September), published again.** The 0.5.0 wheel and source archive on PyPI were built from an earlier snapshot of the release: their licence metadata says CC BY-NC-SA 4.0 instead of GPL-3.0-or-later, and the force viewer is the version before its fixes (it crashes with numpy 2). GitHub's `v0.5.0` tag points at that earlier snapshot too. Install 0.5.1 (`pip install --upgrade foambryo delaunay-watershed-3d`); 0.5.0 is withdrawn from PyPI (yanked). Inference is unchanged: every tension and pressure 0.5.1 infers is the one the final 0.5.0 infers. The 0.5.0 notes below apply to 0.5.1 as they are.

# Release notes — 0.5.0

`foambryo` and `delaunay-watershed-3d` ship together as 0.5.0; this foambryo requires `delaunay-watershed-3d>=0.5.0`. This is what changes for you and what can mislead you; `CHANGELOG.md` has the full list, and dw3d's `RELEASE_NOTES.md` covers reconstruction.

0.5.0 is the first release on PyPI since 0.3.5; `CHANGELOG.md` also lists the changes of 0.4.0–0.4.2, which were never published there.

## Licence

From 0.5.0 `foambryo` is licensed under the **GNU General Public License, version 3 or (at your option) any later version** (SPDX `GPL-3.0-or-later`), replacing CC BY-NC-SA 4.0: commercial use is allowed, under the GPL's conditions. Earlier releases keep the licence they were published under.

## What changes for you

- **Meshes reconstructed with dw3d 0.5.0 differ from 0.4 meshes**: junction relocation is on by default there, so angles and tensions inferred from a freshly reconstructed mesh move. Reconstruct with `relocate_junctions=False` to reproduce a 0.4 mesh.
- **`dcel_mesh_from_segmentation_mask` now defaults to `min_distance=3`** (was 5), dw3d's own default: a denser mesh, slower to build. Pass `min_distance=5` for the 0.4 spacing.
- **The pressure and force functions now raise instead of returning a non-finite field.** `infer_pressures`, `infer_pressures_and_residuals` and `infer_forces` raise `UndeterminedTensionError` (a `ValueError`) naming every interface whose tension is not finite; on the real embryo series this happens on 5 of 154 timepoints, all outside the segmentation-quality window, where 0.4 returned 54–71 non-finite pressures out of 55–72 without a warning. Tensions are unchanged, and pressures from finite tensions are bitwise unchanged.
- **The second-order reading (`angle_reading="quadric"`) is now independent of the mesh's units**, which moves previously published second-order numbers by up to **0.14°**.
- **`AngleReadingCoverageWarning` now carries its numbers**: the measured coverage fraction and how many trijunction lines were not fully corrected.

## Limitations

Every item is measured; each is one sentence and its number.

1. **The benchmark under-rewards tangent-based improvements.** Its ground-truth meshes are facet-equilibrated (Surface Evolver), so an improvement in the surface-*tangent* description of a contact angle is scored against a facet-based truth and can read as no change.
2. **The second-order angle reader has its own floor, and it cannot resolve junction relocation's effect.** On a perfectly smooth analytic surface it reads contact angles with a floor of **0.245°**, and at every point spacing measured (0 of 32 spacing-by-variant cells) its check could not resolve the effect of junction relocation on tangent-based angles, in either direction.
3. **On real embryos, do not use second-order line means at sub-degree scale.** The second-order reader covers a median **0.804** of incident trijunction faces on real embryo meshes (minimum **0.454**), and its line means move by **1.75° [1.41, 2.17]** when uncovered faces are excluded instead of filled by the chord reading.
4. **Real embryo cells are small in voxels, so every benchmark statement is about one spacing step optimistic for them.** Their equivalent radius at trijunctions is a median **17.6** voxels against **38.2** in the benchmark, so real data at `min_distance 3` sits where the benchmark sits at `min_distance 4`; for cells this small, consider a smaller `min_distance`.
5. **Folded trijunctions on real meshes are mostly real geometry.** Real embryo meshes carry **3.2×** the benchmark's fraction of reflex (folded) trijunctional edges, and more than half of them sit where the segmentation mask's own wedge already exceeds 180°.
6. **The force-balance residual `rho_free` alarms on every reconstructed mesh.** Its median is **0.578** on reconstructed meshes against **0.0118** on the reference meshes, so a high value on your mesh is expected and is not by itself a sign of a bad segmentation.
7. **`ForceBalanceMAP` is not scale-invariant**, contrary to its docstring: rescaling a mesh moves its relative tensions by up to **43 %**; plain `ForceBalance` is exact under rescaling.
8. **Junction relocation, now on by default, has a cost that grows faster than linearly with mesh size.** It adds about **+0.75 s** to a ~5 s benchmark reconstruction (**+15 %**) and **0.1–1.5 s** per timepoint to a ~0.6 s real-embryo reconstruction (median **0.34 s** and **0.48 s** on two developmental time series, about 65 s per 154-timepoint series); within one real series it grows with triangle count with a fitted exponent of **2.68** over a 2.2-fold range of mesh sizes, which is not a tissue-scale law. **A faster neighbour search is planned for 0.5.1**, and will leave every relocated mesh bit-for-bit unchanged.
