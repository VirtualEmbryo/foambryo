# Release notes — 0.8.0, 26 November 2026

`foambryo` 0.8.0 infers tensions and pressures of 2-D tissues as well as 3-D ones, through the same functions, command line and viewers. It ships together with delaunay-watershed 0.8.0 (which now meshes 2-D masks as well as 3-D ones) and napari-foambryo 0.2.0. This is what changes for you and what can mislead you; `CHANGELOG.md` has the full list, and delaunay-watershed's `RELEASE_NOTES.md` covers reconstruction. Install with:

```bash
pip install --upgrade foambryo delaunay-watershed
```

foambryo 0.8.0 requires `delaunay-watershed` 0.8.0, which installs both `dw3d` and `dw2d`, and Python 3.12 or later.

## What changes for you

1. **2-D tissues, through the same calls: the input decides the dimension.** `infer_tensions`, `infer_tensions_by_junction_component`, `infer_pressures`, `infer_forces` and `infer_tensions_then_pressures` take a 2-D mesh (`foambryo.PlanarMesh`, dw2d's mesh, or its arrays) as they take a 3-D one; `dcel_mesh_from_segmentation_mask`, `reconstruct_mask` and `dcel_mesh_from_file` take a 2-D mask or `.rec` file. A 3-D mesh gives bit for bit the numbers of before. In 2D the tensions come from Young-Dupré at each junction (each junction-graph component gauged alone) and the pressures from Laplace's law on circles fitted to the interfaces. The 2-D names are provisional: they may change in any release until they are declared stable. If you use the 2D inference, please cite the Developmental Cell 2026 paper (Citation, below).
2. **In 2D the junction angles are read on circular arcs by default (`angle_reading="arc"`).** Each interface is fitted by the least-squares circle through its vertices and its tangent is taken at the circle's point nearest the junction. On synthetic equilibrium foams with known tensions and pressures (53 held-out tissues, 661 images: exact masks, crops, boundary noise of 0.5 to 2 px, cells of 10 to 40 px radius), against the chord reading of foambryo2D, the relative tension error is lower by 0.068 (95 % interval 0.063–0.075) and the pressure error by 0.091 of the RMS pressure, with delaunay-watershed's junction relocation on (its default). `"chord"` and `"fitted_tangent"` remain. The circle fit is written in NumPy, so its circles are the same bit for bit on every supported NumPy and SciPy.
3. **Which reading, and which pressures, to trust.** The arc reading, Laplace's law and the global fit assume interfaces shaped by the cells' own tensions and pressures. In a *C. elegans* embryo inside its eggshell the cells are compressed against the shell: an interface's curvature follows Laplace's law, and its tangent obeys Young–Dupré, only close to the junctions. For such compressed cells (and cells on a substrate) there are no Laplace pressures, which would not make sense, nor full stress built on them; the arc reading and the global fit are not used; read the tensions with the chord (`angle_reading="chord"`), with `"fitted_tangent"`, the other reading local to the junctions, as a check. Measured on one embryo per condition (three frames each): the cell–medium interfaces depart from their fitted circle by 0.67–1.76 px RMS inside the eggshell against 0.06–1.04 px without it, and inside the eggshell the global fit spreads the tensions from 0.23 to 2.03 against 0.61–1.33 for the fitted tangent and 0.51–1.36 for the chord. The new **deviatoric stress** (`plot_force_inference(..., stress="deviatoric")`, 2D and 3D) is computed from the tensions and the cells' boundaries alone, never from the pressures: it is the contribution of the cortical tensions to each cell's stress anisotropy; under confinement the shell's contact forces also act on the cells and are not in it, so there it is not the cell's total stress anisotropy. Its view colours no cell by pressure, and its colour bars are the tension bar and a "Deviatoric stress" bar (red below 0, compressive; teal from 0, tensile; from the smallest to the largest principal value over the cells drawn). On the command line: `foambryo infer MESH --angle-reading chord --no-pressures` and `foambryo view MESH --angle-reading chord --stress deviatoric`. **For free tissues** (no shell), arcs and pressures apply in principle, but on one free embryo the chord gives three of the four cell–cell interfaces 0.39–0.61 times the arc reading's tension, and no real image has known tensions: which reading is closer to the truth on real free tissues is an open question. The documentation's 2-D pages and its two examples show the choice on the two embryos.
4. **Opt-in: a global fit of an equilibrium tiling of circular arcs**, `infer_forces_by_equilibrium_tiling(mesh, labels, ...)` (`foambryo infer --tension-method equilibrium-tiling --mask MASK`), after Noll, Streichan and Shraiman (2020). It fits the label image itself, with Young's and Laplace's laws exact, starting from the arc reading. On synthetic foams (307 held-out tissues of 4 to 64 cells) its median relative tension error is lower than the arc reading's by 0.03 to 0.11 in every tier and cell size except one (2 px of noise at 20 px radius, not resolved); on a 4 096-cell tissue 0.077 against 0.156, in 74 s and 0.6 GiB. It is provisional and not the default, because it still refuses whole images where the arc reading refuses none (11 of 453 crops). It assumes circular interfaces, so it is not for confined tissues.
5. **The command line and the viewers take 2-D input.** `foambryo mesh` meshes a 2-D mask (`--spacing DY DX`, square pixels only), `foambryo infer` infers a 2-D mesh (`--angle-reading`, in 2D `arc`, `chord` or `fitted_tangent`), `foambryo view` opens a 2-D window: interfaces coloured by tension, cells by pressure, the image underneath, shown as napari and matplotlib show it (row 0 at the top), never its mirror image. `foambryo infer-over-seeds` refuses a 2-D mask (dw2d has no dither seed). `run.json` is schema 2 and records `dimension`.
6. **Curvature and stress in 2D.** `compute_curvature_interfaces` and `compute_curvature_vertices_cotan` take a 2-D mesh; Laplace's law reads $p_a - p_b = -\gamma_{ab}\kappa_{ab}$ in 2D beside $P_a - P_b = -2H_{ab}T_{ab}$ in 3D, with the same sign convention ($\kappa_{ab} > 0$ when the centre of curvature lies on $b$'s side). The force window draws each cell's principal stresses in 2D as in 3D.
7. **A 2-D mesh's orientation is measured on input.** foambryo checks the orientation of every 2-D mesh from the signed areas of its cells: a mesh written by dw2d 0.1.5 or by 0.8.0's development versions (edges stored the other way round) gives the same tensions and pressures, bit for bit; a mesh whose cells disagree is refused with an error that names them.
8. **3-D numbers change only through delaunay-watershed 0.8.0's meshes.** Its padding of thin margins of medium changes 18 of the lab's 684 test masks; the 47 benchmark meshes are bit for bit those of 0.7.1, so every 3-D number of `BENCHMARKS.md` stands.
9. **torch is gone**: the `autodiff` extra is removed; the derivative check is a float64 complex-step check (agreement 3e-16 to 4e-14). The inference never used torch.
10. **Defaults that differ by dimension are written `None`** in the signatures (`angle_reading`, the pressure `mode`, `min_distance`); in 3D they resolve to what they were (`"chord"`, `Variational`, 3 voxels).

## New

* `foambryo.PlanarMesh`, `foambryo.dimension_of`, and the 2-D-only names at the top level: `interface_arcs`, `InterfaceArc`, `junction_table`, `image_border_vertices`, `TwoDimensionalReport`, `TwoDimensionalPressureReport`, `MAX_LENS_CONDITION`, `IllConditionedLensWarning`, `NoJunctionError`, `ANGLE_READINGS_2D`, `DEFAULT_ANGLE_READING_2D`; `infer_forces_by_equilibrium_tiling`, `EquilibriumTilingReport`, `EQUILIBRIUM_TILING_MAX_EVALUATIONS`.
* `foambryo.plotting_utilities.planar_mesh_drawing(mesh)`: what the viewers draw of a 2-D mesh (each cell cut into triangles by ear clipping, exact for non-convex cells), without Polyscope or a display; foambryo's window and napari-foambryo draw from it.
* `plot_force_inference(..., stress="deviatoric")`, in 2D and 3D, with its own "Deviatoric stress" colour bar in the window, in the saved figure and in the saved bars, and no pressure colours: the contribution of the cortical tensions to each cell's stress anisotropy; under confinement the shell's contact forces also act on the cells and are not in it, so there it is not the cell's total stress anisotropy.
* `foambryo view --stress full|deviatoric` (default `full`); refused with `--no-pressures` and for values without pressures.
* `view_mask(..., stress="full"|"deviatoric")`: the same choice from Python for the window `foambryo view MASK` opens, with the same refusals.
* `foambryo mesh --pad-thin-medium-margins` / `--no-pad-thin-medium-margins` and `reconstruct_mask(..., pad_thin_medium_margins=None | True | False)`: delaunay-watershed's padding of thin margins of medium, by default the library's own choice (on for cubic voxels and square pixels, off for a mask resampled to cubic voxels). `ThinMediumMarginWarning` as `foambryo mesh` prints it names the option.
* The public names that front ends use are listed in their module's `__all__` (`foambryo.io`, `pressure_inference`, `force_inference`, `second_order_angle_reading`, `seed_ensembles`, `dcel`); nothing is renamed or removed, and `from <module> import *` no longer brings the modules they import.
* `foambryo --version` names delaunay-watershed-2d and iorec.

## What can mislead you

1. **Pressures and the full stress of compressed or confined cells are not measurements, and their deviatoric stress is only the cortical tensions' contribution** (item 3). foambryo cannot tell a confined tissue from a free one: the choice is yours.
2. **On real free tissues the arc and the chord readings can disagree** (item 3): an open question, with no real ground truth.
3. **In 2D, junctions on the image border are excluded**, by delaunay-watershed's mark (`exclude_image_border=True`). A cell that rows of several junction-graph components determine gets `nan` in the flat pressure map, with a warning; its pressure in each component is in the report.
4. **A lens whose angles cannot separate its two tensions is refused** (`nan` and `IllConditionedLensWarning`, condition number above 20): a small two-sided cell read by its chords is the usual case. On 3 522 synthetic and real readings the guard fired on 2.6 %, almost all of crops.
5. **An interface with fewer than 4 mesh vertices is read by its chord and gives no Laplace row**: most interfaces at a cell radius of 10 px.
6. **Every 2-D accuracy number is measured on synthetic tissues**; no real 2-D image has ground truth. The boundary noise of 0.75 px was calibrated on seven real 2-D masks.
7. **Not every function has a 2-D counterpart yet**: `infer_force_balance`, `infer_tensions_young_dupre_local`, `infer_tensions_with_uncertainty`, `compute_inference_diagnostics` and the `..._and_residuals` functions raise `ValueError` on a 2-D mesh.
8. The points of 0.7.1's list still hold.

## Known issues

1. **A cell cut by the image box can be absent from delaunay-watershed's 3-D mesh, and `cut_cells` then names another cell.** Since 0.8.0 the absent cell is named (`UnresolvedCellWarning`, cause `"removed by mesh surgery"`), and `foambryo mesh` and `MaskReconstruction.unresolved` list it; it gets no pressure.
2. **A mask with voxels longer along one axis is not padded for thin margins of medium by default** (delaunay-watershed's known issue: there the padding made its watershed lose cells on 20 of 100 dither seeds against 3 of 100); `foambryo mesh` prints the `ThinMediumMarginWarning` that names such cells; `foambryo mesh --pad-thin-medium-margins` pads anyway. On the two real anisotropic stacks tested, a 110-cell and a 1 866-cell tissue (z/xy = 5/3), the default (no padding on a resampled mask) lost or inflated no cell near an image face. On the 1 866-cell tissue, whose z faces have thin margins (2 and 3 resampled voxels), the 9 cells within 8 voxels of a face kept a mesh volume of 0.88 to 1.05 times their voxel volume over 5 dither seeds, within the range of its interior cells (0.84 to 1.05); `ThinMediumMarginWarning` named 3 to 6 of them, depending on the seed, although their volumes were normal. Padding (`pad_thin_medium_margins=True`) lost no further cell on that tissue over the same seeds, and lowered the volume of the four cells nearest a face by 0.01 to 0.09 of their voxel volume; which of the two is closer to the true volume is not known. Two stacks give no rate over stacks, and five seeds bound a per-seed loss rate only to 0.52 or less (95 %).
3. **napari's default 3-D orientation shows the mirror image of a tissue.** `viewer.scene.camera.orientation = ('away', 'down', 'right')` shows the tissue the right way round (`viewer.camera` is deprecated since napari 0.9.0).
4. **Figures saved from the window on a high-density (Retina) display have small text.**

## Citation

If you use foambryo, in 3D or in 2D, please cite both articles:

* Ichbiah S., Delbary F., McDougall A., Dumollard R., Turlier H. *Embryo mechanics cartography: inference of 3D force atlases from fluorescence microscopy.* Nature Methods 20(12), 1989–1999 (2023), doi:10.1038/s41592-023-02084-7.
* Yamamoto K., Ichbiah S., Perez M., Borrego-Pinto J., Delbary F., Goehring N. W., Northrop P., Turlier H., Charras G. *Spatiotemporal mapping of the contractile and adhesive forces sculpting early C. elegans embryos.* Developmental Cell 61(6), 1273–1288.e4 (2026), doi:10.1016/j.devcel.2026.04.013.

Also cited:

* The global fit follows Noll N., Streichan S. J., Shraiman B. I. *Variational method for image-based inference of internal stress in epithelial tissues.* Physical Review X 10(1), 011072 (2020), doi:10.1103/PhysRevX.10.011072.
* The 2-D benchmark compares foambryo with the published CellFIT equations, re-implemented: Brodland G. W., Veldhuis J. H., Kim S., Perrone M., Mashburn D., Hutson M. S. *CellFIT: a cellular force-inference toolkit using curvilinear cell boundaries.* PLoS ONE 9(6), e99116 (2014), doi:10.1371/journal.pone.0099116.

## Data

* The 2-D examples use frames of 4-cell *C. elegans* embryos from the data of Yamamoto et al. (2026) above, deposited on Zenodo (doi:10.5281/zenodo.19627642) under CC BY-SA 4.0; the derived files carry the same licence.
* The README's image is the public Astec-Pm10 segmentation of a 64-cell *Phallusia mammillata* embryo, time point 10 (Lemaire 2020, doi:10.6084/m9.figshare.8235485, CC BY 4.0; Guignard et al., Science 2020), meshed by delaunay-watershed and drawn by foambryo's viewer; `Images_github_repo/make_the_readme_figure_of_a_64_cell_embryo.py` makes it from the deposit's files.

## Credits

Recent work on dw3d and foambryo is part of a collaboration with Alejandro Aguilera Castrejon (Janelia Research Campus, Howard Hughes Medical Institute). We thank him and his Janelia colleagues Adam L. Taylor, Wenhong Li and Francesco Roncato. Adam L. Taylor found the border-triangle winding defect fixed in 0.6.0 and pointed us to the seed-extraction and dense tension-solve bottlenecks on large tissues.

Development of this release was supported by Anthropic's AI for Science program; parts of the code were written with Claude and reviewed by the developers.

# Release notes — 0.7.1, 26 October 2026

`foambryo` and `delaunay-watershed-3d` ship together as 0.7.1. This is what changes for you and what can mislead you; `CHANGELOG.md` has the full list, and dw3d's `RELEASE_NOTES.md` covers reconstruction. Install both with `pip install --upgrade foambryo delaunay-watershed-3d`.

## What changes for you

1. **Tensions and pressures change slightly because delaunay-watershed-3d 0.7.1's meshes do**; foambryo's inference is unchanged. dw3d now orders the tetrahedra canonically, fixes its mesh surgery, and meshes enclosed media and unresolved regions correctly (its release notes). On the benchmark, `YoungDupre`'s median relative tension error goes from 0.0777 to 0.0763 on the default meshes; `ForceBalanceKKT`'s from 0.1150 to 0.1138 and `BayesianMAP`'s from 0.1211 to 0.1186, mostly through cases 031 and 038, whose spurious cells are gone.
2. **On a mesh with a medium besides 0 (a lumen), the medium is no longer a cell.** Its interfaces keep their own tensions and its pressure is inferred like a cell's, but it is not counted as a cell, not listed as a cut cell, and a triangle between a cell and the medium is coloured by the cell's pressure. dw3d 0.7.1 records the media in the `.rec` file; for a mesh made elsewhere, name them with `DcelData(..., media=[L])` or `foambryo infer --media L`. Every mesh without such a medium gives the same numbers and pictures as before.
3. **`foambryo view` drew the mirror image of the tissue**, its left and right exchanged, in every view and every figure saved from the window. It now draws the tissue the right way round; the mesh, its files and every number are unchanged. Figures saved with 0.7.0 show the mirror image.

## New

* `pressures.csv` has a `region_kind` column (`exterior`, `medium`, `cell`) when the mesh has a medium besides 0; `read_pressure_table` reads both forms and `read_pressure_region_kinds` the column. `run.json` records `mesh.media`, `mesh.cells`, `mesh.unresolved` and `options.media` (schema 1: keys added, none changed).
* `foambryo mesh` names the enclosed media it made and the labels of the mask the mesh could not hold, and relays dw3d's warnings about them; `DcelData.unresolved` holds those labels, and no count or table includes them.
* `segmentation_mask_report(mask, media=[...])` leaves the media out of the cell statistics.

## Corrected in the documentation

1. **The user guide's second-order angle-reading numbers were measured on meshes without junction relocation**, which is dw3d's default since 0.5.0. On relocated meshes the reading's excess over its own floor is +0.0112 [−0.0093, +0.0315] at `min_distance` 3 and −0.0221 [−0.0295, +0.0053] at 4: both intervals include zero, so the guide no longer says that a residual slope survives at 3. An interval that includes zero bounds the excess; it does not show that there is none.
2. **What the 0.5.0 repair of the second-order reading changed**: the 0.5.0 entry quoted a unit dependence measured on one rescaled mesh (0.14 degrees); on 40 benchmark reconstructions the repair moved the median line-mean angle error from 5.27 to 5.40 degrees.

## What can mislead you

1. **Numbers not measured again on dw3d 0.7.1's meshes say so where printed**: the timings, the 1 866-cell tissue, the anisotropy costs and the error-bar calibration on simulated tissues at a real cell size. They were measured on 0.6.0's or 0.7.0's meshes.
2. **Between machines, meshes may still differ at exactly cospherical points**, and so may the tensions read from them; on one machine a reconstruction is reproducible.
3. The points of 0.7.0's list still hold.

## Known issues

1. **foambryo 0.7.0 cannot read a `pressures.csv` written for a mesh with a lumen**: the file's third column, `region_kind`, is refused by 0.7.0's reader. Files of meshes without such a medium are unchanged.
2. **napari's default 3-D orientation shows the mirror image of a tissue**, mask and mesh alike: napari draws the array's axes in the order (z, y, x) and calls that orientation right-handed. `viewer.camera.orientation = ('away', 'down', 'right')` shows the tissue the right way round, with the plugin's surfaces shaded darker. A switch in napari-foambryo is planned; this is documented in the user guide (*Viewers*, "Which way round").
3. **A cell cut by the image box can be absent from delaunay-watershed-3d's mesh, and `cut_cells` then names another cell** (corrected on 4 November 2026: the first wording said cell 1 gets a pressure). On a small test mask of four cells in a ball cut by the image box, the cells with voxels on the box are 1, 3 and 4. At the default `min_distance` 3 on cubic voxels, cell 1 is absent from the mesh, so it gets no pressure; `DcelData.cut_cells` gives 2, 3 and 4, because cell 2 took most of cell 1, and cell 2's pressure is withheld as cut. Check the reconstruction's `CellVolumeDisagreementWarning` for cells absent from the mesh, and `cut_cells` against the mask's own border labels.
4. **Figures saved from the window on a high-density (Retina) display have small text**: the colour bars and captions are drawn at 150 dots per inch of the window's pixels, which such a display doubles, so their text comes out about half as large relative to the picture (measured on one Mac: a 2 560-pixel-wide figure). The values are unaffected.

## Credits

Recent work on dw3d and foambryo is part of a collaboration with Alejandro Aguilera Castrejon (Janelia Research Campus, Howard Hughes Medical Institute). We thank him and his Janelia colleagues Adam L. Taylor, Wenhong Li and Francesco Roncato. Adam L. Taylor found the border-triangle winding defect fixed in 0.6.0 and pointed us to the seed-extraction and dense tension-solve bottlenecks on large tissues.

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
- **The second-order reading (`angle_reading="quadric"`) is now independent of the mesh's units.** This changes second-order readings on reconstructed meshes: on 40 benchmark reconstructions (delaunay-watershed-3d 0.4.2, without junction relocation), the median line-mean angle error under the oriented second-order reading moved from 5.27° to 5.40°, single cases by up to 0.92°. (This note first said "by up to 0.14°", which is the unit dependence removed, measured on one rescaled mesh.)
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
