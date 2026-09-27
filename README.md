# foambryo

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]
[![DOI](https://zenodo.org/badge/625305276.svg)](https://zenodo.org/badge/latestdoi/625305276)

**foambryo** is a python package developed to infer relative surface tensions and cell pressures from the 3D geometry of cells in foam-like clusters, such as early-embryos, tissues or organoids.

<img src="https://raw.githubusercontent.com/VirtualEmbryo/foambryo/main/Images_github_repo/Window_76_cells.png" alt="drawing" width="650"/>

It was developed by Sacha Ichbiah during his PhD in [Turlier Lab](https://www.turlierlab.com), and is maintained by Matthieu Perez and Hervé Turlier. For support, please open an issue.
If you use our library in your work please cite the [paper](https://doi.org/10.1101/2023.04.12.536641).

If you are interested in inferring tensions in 2D, please look at the [foambryo2D](https://github.com/VirtualEmbryo/foambryo2D) package instead. 

### Biological and biophysical context 

Multicellular structures encountered in the field of developmental biology have various shapes and arrangements that reflectes their physiological functions. Mechanics is at the heart of the development of these structures, yet tools to investigate the forces at play in 3D remain scarse. Inferring forces or stresses from cell shapes only allows one to reveal the fundamental mechanics shaping cells from microscopy imaging. This software makes the hypothesis that cells in many early embryos, tissues and cell aggregates are mechanically akin to heterogeneous foam-like structures (see [Physical model](README.md/#Physical-model)).

### Prerequisites 

**foambryo** requires Python >3.6 with pip.

**foambryo** requires the prior segmentation of images of multicellular aggregates into cell segmentation masks using one's favorite algorithm (watershed algorithm, [cellpose](https://www.cellpose.org) or any preferred one). The quality of the segmentation and the size of the original image will directly affect the precision of the inference results.

#### Dependencies

All required dependencies are installed by pip automatically.

**foambryo** relies on a companion tool [**delaunay-watershed**](https://github.com/VirtualEmbryo/delaunay-watershed) that we developed to construct precise multimaterial meshes from instance segmentations. From these multimaterial meshes, one can efficiently and robustly measure junction angles and interface curvatures to invert the **Young-Dupré** and **Laplace** laws and infer the **surface tensions** $\gamma_{ij}$ and **cell pressures** $p_i$ underlying the mechanical equilibrium of foam-like cell aggregates.

The viewer is based on [**Polyscope**](https://github.com/nmwsharp/polyscope), a C++/Python viewer designed to visualize 3-dimensional geometry, in particular meshes.

#### Operating systems

This package should work on any OS and was tested on the following macOS and Linux systems:

macOS: BigSur (11.7.9) and Ventura (13.5)

Linux: Ubuntu 16.04, CentOS 7 and Manjaro 22.0


### Installation

We recommend to install **foambryo** from the PyPI repository directly.
```shell
pip install foambryo
```
To use the foambryo viewers, use instead:

```shell
pip install "foambryo[viewing]"
```
For developers, you may also install **foambryo** by cloning the source code and installing from the local directory.

First, make sure you have the LFS git extension, as there are some large data alongside the source code, necessary to run the example notebook.
```shell
git clone https://github.com/VirtualEmbryo/foambryo.git
pip install pathtopackage/foambryo
```

### Quick start example 



Load an instance segmentation, reconstruct its multimaterial mesh, infer and visualize the forces with Polyscope

```py
from foambryo import dcel_mesh_from_segmentation_mask
from foambryo.viewing import plot_force_inference, plot_tension_inference

# Load the labels
import skimage.io as io
segmentation = io.imread("Segmentation.tif")

# Reconstruct a multimaterial mesh from segmentation
mesh = dcel_mesh_from_segmentation_mask(segmentation_mask)  # DCEL mesh ready for force inference !

# Infer and view the forces
plot_force_inference(mesh)

#Or just the tensions
plot_tension_inference(mesh)
```

See [the introductory notebook](Examples/Example.ipynb) for further details.

### Physical model
We consider a tissue constituted of cells i. 

<img src="https://raw.githubusercontent.com/VirtualEmbryo/foambryo/main/Images_github_repo/Equilibrium.png" alt="drawing" width=“175”/>

They minimize, under conservation of volume an energy 
$\mathcal{E}=\underset{ij}{\sum}\gamma_{ij}$.

The two main laws underlying mechanical force balance are: 
- **Young-Dupré Law:** $\gamma_{ij} + \gamma_{ik} + \gamma_{jk} = 0$.
- **Laplace Law:** $p_j - p_i = 2 \gamma_{ij} H_{ij}$ where $H_{ij}$ is the mean curvature of the interface between the cell i and j.


---

### API and Documentation

See [the introductory notebook](Examples/Example.ipynb) to understand the API.

#### Infer tensions and pressures

The notebook shows how to create a mesh and how foambryo compute forces on the mesh.

We recall that the forces are relative, this is why you need to give the mean tension and base pressure.

- `base_pressure`: reference exterior pressure. All pressures inside cells are computed relative to this one.
- `mean_tension`: as one only infers ratios between tensions, it has to be given. You can set it to 1 for instance.

Foambryo has several strategies to compute tensions. They fall into two families that respond
very differently to mesh-reconstruction quality — see `BENCHMARKS.md` — angle-reading methods,
which read local contact angles at triple junctions, and operator-inverting methods, which
assemble and invert a global discrete operator over the whole mesh:

Angle-reading:
- `YoungDupre` (Young-Dupré with cosines only) — the recommended default when mesh quality is
  uncertain,
- `YoungDupreLocal` (the same balance imposed per trijunctional edge instead of once per triple
  line; a better diagnostic, a worse point estimate),
- `ProjectionYoungDupre` (Young-Dupré with cosines and sines),
- `Lami` ([Lami's theorem](https://en.wikipedia.org/wiki/Lami%27s_theorem)),
- `InvLami` (inverse of the Lami's relation),
- `LogLami` (logarithm of the Lami's relation).

Operator-inverting:
- `Cotan` (cotangent formula, see [Yamamoto et al. 2023](https://doi.org/10.1101/2023.03.07.531437)),
- `InvCotan` (inverse of the cotangent formula),
- `Variational` (variational formulation, see our [paper](https://doi.org/10.1101/2023.04.12.536641)),
- `ForceBalance`, `ForceBalanceSoftRow`, `ForceBalanceMAP` (the unified discrete force-balance
  operator under three different gauge treatments — hard KKT, soft least-squares row, and a
  Gaussian-prior MAP estimate with uncertainty).

Also: `Equilibrium`, a geometric-edge balance that fits neither family cleanly.

#### Choosing how the trijunction angle is read: `angle_reading`

Every method above that reads angles — the six angle-reading ones plus `Cotan` and `InvCotan` —
takes an `angle_reading` keyword. It selects **how a contact angle is measured on the mesh**, not
how tensions are solved for once it has been measured:

- `angle_reading="chord"` (**the default, and unchanged**) measures the angle between the two
  incident triangles' flat faces. A flat triangle has no curvature to report, so on a curved
  interface it reports the angle of the *chord* to the triangle's far vertex rather than the
  interface's *tangent* at the junction.
- `angle_reading="quadric"` fits a generic quadric surface to the interface's own mesh vertices
  in a small window around each junction edge and reads that surface's tangent instead. For a
  circular arc the two differ by exactly `κL/2`, with `κ` the interface's curvature and `L` the
  distance to the triangle's far vertex, and that is the correction applied.
- Nothing else changes: no vertex moves, no interface area or cell volume changes, and the mesh
  is untouched. Only the angle is read differently.

```python
from foambryo import TensionComputationMethod, dcel_mesh_from_file, infer_tensions

mesh = dcel_mesh_from_file("my_mesh.rec")
tensions = infer_tensions(mesh, 1, TensionComputationMethod.YoungDupre, angle_reading="quadric")
```

**The measured gain**, on 42 mechanically-equilibrated benchmark cases meshed by `dw3d` (see
`BENCHMARKS.md` for the full table and its provenance):

| `dw3d` `min_distance` | 3 | 4 |
|---|---:|---:|
| length-weighted junction-angle flattening slope, `chord` | −0.2343 | −0.2204 |
| the same, `quadric` | **−0.0848** | **−0.0170** |
| tension contrast (1.000 is perfect), `chord` | 0.738 | 0.722 |
| the same, `quadric` | **0.954** | **1.056** |
| median per-line angle error, `chord` | 7.35° | 8.22° |
| the same, `quadric` | **4.83°** | **5.43°** |

Contrast and per-line angle error improve *together*, which is what distinguishes a geometry fix
from a bias-for-variance trade.

**The window is fixed and is not a tuning parameter.** The quadric is fitted in a window of four
times the triangle's own lever, isotropic in the two surface directions. Those values were chosen
a priori and then re-selected independently by held-out fit residual under a pre-registered
coverage floor, which landed back on exactly them. They are module constants rather than
arguments on purpose: exposing them would invite tuning them against the very error they are
judged on, and a window tuned that way would flatter itself.

**Coverage, and what happens where the fit does not apply.** The quadric needs at least ten of
the interface's own mesh vertices inside its window. Where it has fewer, that face keeps the
chord reading, and a result can therefore mix the two conventions. This is never silent:

```python
from foambryo import infer_tensions_and_residuals_with_report

*_values, report = infer_tensions_and_residuals_with_report(
    mesh, 1, TensionComputationMethod.YoungDupre, angle_reading="quadric",
)
report.coverage_fraction    # e.g. 0.919 -- the fraction of trijunction faces corrected
report.uncorrected_lines    # the trijunction lines that are only partly corrected
report.is_complete          # False when any face fell back
```

Calling `infer_tensions` or `compute_angles_tri` directly, which drop the report, emits an
`AngleReadingCoverageWarning` carrying the same numbers instead. On the benchmark meshes coverage
is 0.919 of faces at `min_distance = 3` and 0.890 at 4 — but only **0.220** and **0.178** of
*lines* are corrected throughout, so most lines carry a mixture. Look at `uncorrected_lines`, not
only at the fraction.

**When *not* to use it, and this is the most important caveat.** The correction removes a
`κL/2` chord deficit, and on a **well-resolved** mesh that deficit is already small while the
reader's known over-application is not. Measured on the 47 ground-truth simulation meshes that
ship with the benchmark — far finer than any reconstruction — the chord reading's contrast is
already 1.000 to within 2×10⁻⁵, and `angle_reading="quadric"` moves it to a median of 1.080,
**worse on 47 of 47 cases**. Use it on `dw3d`-reconstructed meshes, not on simulator output or
any mesh whose elements are already much smaller than the interface's radius of curvature.

**The honest residual.** The correction is not exact, and four measured facts qualify it:

1. It **over-applies at fine spacing.** On a coarsened reference mesh, where the true angles are
   known, the reader removes about 26 % more rotation than the chord deficit it exists to remove
   at `min_distance = 3`; by `min_distance = 7` it under-applies instead. The reference-mesh
   result above is the same effect at its most extreme.
2. Consequently a residual slope survives: the corrected arm sits **+0.0487** above its own
   corrected floor at `min_distance = 3`, while at `min_distance = 4` the excess contains zero
   (−0.0102, CI [−0.0288, +0.0127]).
3. `YoungDupreLocal` is the one method whose **tension** error gets worse under the quadric
   reading even though its angles improve — it imposes the balance per edge and so never performs
   the along-line average that cancels the correction's per-edge scatter. Use `YoungDupre` with
   `angle_reading="quadric"`, not `YoungDupreLocal`.
4. It reads angles a second-order surface fit reports; it does **not** move any vertex, so it
   cannot repair a mesh whose junction line is in the wrong place.

A runnable side-by-side comparison on one benchmark case:
[`Examples/second_order_angle_reading.py`](Examples/second_order_angle_reading.py).

#### Declining a folded trijunction observation: `refusal`

A **folded** trijunction edge is one whose three interior wedge angles do not enclose the edge the
way a trijunction requires: one of them is past a straight angle. The Young-Dupré balance every
angle-reading method above imposes presupposes the opposite, so such a row is an observation the
geometry rules out. On `dw3d`-reconstructed meshes 2-25 % of trijunctional edges are folded,
depending on the point placement; on the simulator's own meshes essentially none are (5 in 14 183).

Two reasons this cannot be read away. `YoungDupreLocal`'s rows are built from `cos` of the wedge,
and `cos` is even about a full turn, so a folded wedge and its reflection are literally the same
row — no `angle_reading` reaches it. And `LogLami`'s rows take `log(sin(...))`, which is undefined
for a folded wedge, so it does not degrade there, it **raises**.

`infer_tensions` and its relatives therefore take a `refusal` keyword, which is **off by default**
and changes nothing until you pass it:

```python
from foambryo import (
    TensionComputationMethod, WedgeRefusal, WedgeRefusalPolicy,
    dcel_mesh_from_file, infer_tensions, reflex_trijunction_edges, refusal_report,
)

mesh = dcel_mesh_from_file("my_mesh.rec")
len(reflex_trijunction_edges(mesh))          # how many edges of this mesh are folded

refusal = WedgeRefusal(policy=WedgeRefusalPolicy.ExcludeEdgeFromLineMean)
tensions = infer_tensions(
    mesh, 1, TensionComputationMethod.YoungDupre, angle_reading="oriented_chord", refusal=refusal,
)
refusal_report(mesh, refusal)                # what was declined, in rows and in arclength
```

The three policies differ in what they do with a line that is only partly folded:

- `ExcludeEdgeFromLineMean` — the folded edge leaves its line's angle mean and its line's length;
  the line still reports, from what survives.
- `DropLine` — a line carrying any folded edge does not report at all.
- `DownweightLine` — nothing is removed from the measurement; the line's mean is unchanged and only
  its row weight is scaled, by the fraction of its arclength that is not folded.

**Report the refused share alongside any accuracy you gain from this.** A policy that improves an
error statistic by declining the data it would have got wrong has not improved anything, and
`refusal_report` is what makes that visible. Measured on 15 reconstruction variants x 40
mechanically-equilibrated benchmark cases: `ExcludeEdgeFromLineMean` lowers the gauge-matched
tension error on all 15 and on all eight angle-reading methods, and leaves every interface
determined; `DropLine` costs a median 16-76 % of the junction **lines**, which is nothing to
`YoungDupreLocal` (whose row is one edge) and is most of the data to every line-averaged method,
on some meshes all of it.

**What happens when a refusal leaves something unconstrained.** An interface with no surviving
junction line is reported `float("nan")`, never a number, with the same
`JunctionGraphGaugeWarning` a fully closed interface already raises; a mesh with no surviving
observation at all returns `nan` for every interface rather than raising. Use
`infer_tensions_by_junction_component` to see which interfaces those are and which gauge component
each surviving one belongs to.

**A method that reads no wedge angle refuses a policy** rather than ignoring it — `Equilibrium`,
`Variational` and the three `ForceBalance*` members raise, for the same reason they already refuse
a non-default `angle_reading`.

A runnable comparison of the three policies on one benchmark case:
[`Examples/refusing_folded_trijunction_wedges.py`](Examples/refusing_folded_trijunction_wedges.py).

And also to compute pressures:
- `Variational` (variational formulation, see our [paper](https://doi.org/10.1101/2023.04.12.536641)), 
- `Laplace` (Laplace's law)
- `WeightedLaplace` (Laplace's law with weight on curvature by area).

---
### Biological examples
#### *P. mammillata* early embryo
*Phallusia mammillata* is a solitary marine tunicate of the ascidian class known for its stereotypical development. As the embryo develops freely, without any constraint, we can do a full force inference and infer its tensions, pressures and stresses.
We use segmentation data from [Guignard, L., Fiúza, U. et al.](https://www.science.org/doi/10.1126/science.aar5663)

<img src="https://raw.githubusercontent.com/VirtualEmbryo/foambryo/main/Images_github_repo/Ascidians.png" alt="drawing" width="900"/>


#### *C. elegans* early embryo
*Caenorhabditis elegans* is a widely studied model organism, with one of the most reproducible development. The embryo of this earthworm is developing within a shell. As the shell shape and mechanics is unknown, the pressures are not accessible. However we can still use Young-Dupré relationships to retrieve surface tensions at cell membranes. Here we use segmentation data from [Cao, J., Guan, G., Ho, V.W.S. et al.](https://doi.org/10.1038/s41467-020-19863-x)

<img src="https://raw.githubusercontent.com/VirtualEmbryo/foambryo/main/Images_github_repo/CElegans.png" alt="drawing" width="900"/>

#### Plotting scalar quantities on surface meshes

Gaussian and mean curvatures can be plotted on our meshes, and may be useful to study the geometric properties of interfaces between cells. 
We can also plot the vertex area and volume derivatives, that appear in our variational formulas, the difference between the two principal curvatures and the residual of the best sphere-fit that can be used to detect non-spherical constant-mean-curvature surfaces.
They can be obtained by putting the option `scalar_quantities = True` when viewing the forces. 

<img src="https://raw.githubusercontent.com/VirtualEmbryo/foambryo/main/Images_github_repo/scalar_quantities.png" alt="drawing" width="900"/>

- Gaussian Curvature is computed using the angle defect formula.
- Mean Curvature is computed using the cotan formula.

To see the code of each of these use-cases, please load the associated jupyter notebooks in the folder Notebooks

---

### Repository layout

The project's development record (working notes, exploratory scripts and their results) is
kept in the development repository for provenance and is not part of the distribution:
`pip install` does not ship it, and nothing under `foambryo` imports from it.

### Credits, contact, citations
If you use this tool, please cite the associated preprint: 
Do not hesitate to contact Matthieu Perez and Hervé Turlier for practical questions and applications. 
We hope that **foambryo** could help biologists and physicists to shed light on the mechanical aspects of early development.



```
@article {Ichbiah2023.04.12.536641,
	author = {Sacha Ichbiah and Fabrice Delbary and Alex McDougall and R{\'e}mi Dumollard and Herv{\'e} Turlier},
	title = {Embryo mechanics cartography: inference of 3D force atlases from fluorescence microscopy},
	elocation-id = {2023.04.12.536641},
	year = {2023},
	doi = {10.1101/2023.04.12.536641},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {The morphogenesis of tissues and embryos results from a tight interplay between gene expression, biochemical signaling and mechanics. Although sequencing methods allow the generation of cell-resolved spatio-temporal maps of gene expression in developing tissues, creating similar maps of cell mechanics in 3D has remained a real challenge. Exploiting the foam-like geometry of cells in embryos, we propose a robust end-to-end computational method to infer spatiotemporal atlases of cellular forces from fluorescence microscopy images of cell membranes. Our method generates precise 3D meshes of cell geometry and successively predicts relative cell surface tensions and pressures in the tissue. We validate it with 3D active foam simulations, study its noise sensitivity, and prove its biological relevance in mouse, ascidian and C. elegans embryos. 3D inference allows us to recover mechanical features identified previously, but also predicts new ones, unveiling potential new insights on the spatiotemporal regulation of cell mechanics in early embryos. Our code is freely available and paves the way for unraveling the unknown mechanochemical feedbacks that control embryo and tissue morphogenesis.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2023/04/13/2023.04.12.536641},
	eprint = {https://www.biorxiv.org/content/early/2023/04/13/2023.04.12.536641.full.pdf},
	journal = {bioRxiv}
}
```

### License

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
