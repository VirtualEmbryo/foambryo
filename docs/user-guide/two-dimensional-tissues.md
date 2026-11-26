# Two-dimensional tissues

foambryo infers the relative tensions and pressures of a two-dimensional tissue (a section, a monolayer, an embryo imaged in one plane) with the same functions as in 3D: the input decides the dimension. A 2-D mask gives a 2-D mesh, a 2-D mesh gives 2-D inference.

**Provisional.** Everything 2D is provisional in 0.8.0: the 2-D names and signatures may change in any release until they are declared stable.

**Citation.** If you use foambryo, in 2D as in 3D, please cite both articles: Ichbiah S., Delbary F., McDougall A., Dumollard R., Turlier H. *Embryo mechanics cartography: inference of 3D force atlases from fluorescence microscopy.* Nature Methods **20**(12), 1989–1999 (2023), [doi:10.1038/s41592-023-02084-7](https://doi.org/10.1038/s41592-023-02084-7); and Yamamoto K., Ichbiah S., Perez M., Borrego-Pinto J., Delbary F., Goehring N. W., Northrop P., Turlier H., Charras G. *Spatiotemporal mapping of the contractile and adhesive forces sculpting early C. elegans embryos.* Developmental Cell **61**(6), 1273–1288.e4 (2026), [doi:10.1016/j.devcel.2026.04.013](https://doi.org/10.1016/j.devcel.2026.04.013).

## From a mask to tensions and pressures

```python
import foambryo

mesh = foambryo.dcel_mesh_from_segmentation_mask(mask, spacing=(dy, dx))  # a 2-D mask: a PlanarMesh
tensions, pressures = foambryo.infer_forces(mesh)                          # arc reading, Laplace pressures
```

`dcel_mesh_from_segmentation_mask` hands a 2-D label image (0 the medium, the cells 1, 2, …) to delaunay-watershed's 2D mesher (dw2d) and returns a `foambryo.PlanarMesh`: `points` `(n, 2)` in `(row, column)`, `segments` `(m, 2)`, `labels` `(m, 2)`, the vertices on the image border, the image shape and, with `spacing=(dy, dx)`, the pixel size (square pixels only). `min_distance=None` is 4 pixels in 2D (3 voxels in 3D). `foambryo.dimension_of(mesh)` says which dimension a mesh is. The package-level functions also take dw2d's own `DCEL_Data`, a `(points, segments, labels)` triple or a `(vertices, edges)` pair, and `dcel_mesh_from_file` reads a 2-D `.rec` file.

| function | in 2D |
|---|---|
| `infer_tensions(mesh, angle_reading=None)` | Young's law at each junction; `None` is `"arc"` in 2D, `"chord"` in 3D |
| `infer_tensions_by_junction_component(mesh)` | the same, with each interface's junction-graph component |
| `infer_pressures(mesh, tensions)` | Laplace's law on the interfaces read by a circle (`mode=None` is `Laplace` in 2D, the only one offered) |
| `infer_forces(mesh)`, `infer_tensions_then_pressures(mesh)` | the tensions of each component gauged alone, then the pressures of each component on its own scale |
| `infer_forces_by_equilibrium_tiling(mesh, mask)` | the opt-in global fit (below) |
| `interface_arcs(mesh)`, `junction_table(mesh)` | each interface with the circle read on it (`InterfaceArc`); each junction with its wedges |

Functions without a 2-D counterpart yet (`infer_force_balance`, `infer_tensions_young_dupre_local`, `infer_tensions_with_uncertainty`, `compute_inference_diagnostics`, the `…_and_residuals` functions, `infer_tensions_over_seeds` given a 2-D mask) raise `ValueError` saying so. With `diagnostics=True`, `infer_tensions` and `infer_tensions_by_junction_component` also return a `TwoDimensionalReport` and `infer_pressures` a `TwoDimensionalPressureReport`, each holding the solves' `SolveDiagnostics`; `infer_forces` returns one `SolveDiagnostics` of its solves, as in 3D.

## Orientation, and the sign of Laplace's law

These two conventions are stated here once; the rest of the documentation refers to them.

**Orientation.** Every mesh element is the positively oriented boundary of its first label, in the mask's index frame. In 2D: on every segment `labels[k, 0] < labels[k, 1]`, and `labels[k, 0]` lies on the left of the segment `segments[k, 0] → segments[k, 1]` in `(row, column)`, so that the segment's normal `(dc, −dr)` points out of the first label. In 3D the triangle's right-hand normal in `(z, y, x)` points out of its first label: it is one rule. delaunay-watershed's 2D meshes follow it since 0.8.0 (their edges ran the other way before). foambryo does not take a 2-D mesh's orientation on trust: it measures it from the signed area of each cell's boundary, reverses a mesh in the former orientation (dw2d 0.1.5's, for instance), which then gives the same tensions and pressures bit for bit, and refuses with a `ValueError` a mesh whose cells disagree.

**Laplace's law.** For an interface between labels $a < b$, with tension $\gamma_{ab}$ and signed curvature $\kappa_{ab}$ (in 1/length, positive when the centre of curvature lies on $b$'s side),

$$p_a - p_b = -\gamma_{ab}\,\kappa_{ab},$$

the two-dimensional form of the 3D law $p_a - p_b = -2 H_{ab}\, \gamma_{ab}$ with the mean curvature $H_{ab}$ positive under the same convention. A cell bulging into the medium ($b = $ the cell, $a = 0$): the centre of curvature lies inside the cell, $\kappa_{0b} > 0$, so $p_b = p_0 + \gamma_{0b}\kappa_{0b}$ is above the medium's pressure. `foambryo.curvature.compute_curvature_interfaces` returns $\kappa_{ab}$ with this sign; `InterfaceArc.curvature` is signed along the interface's own traversal instead (positive when the circle's centre is on the traversal's left), with `tension * curvature = p_left - p_right`.

The pressures are in units of the tension per unit length of the mesh: relative tension per pixel for a mesh in pixels, per micrometre for a mesh built with `spacing` in micrometres.

## Which angle reading

Young's law at a junction where three interfaces meet, $\sum_k \gamma_k \mathbf{t}_k = 0$ with $\mathbf{t}_k$ the unit tangent of interface $k$ leaving the junction, needs the three tangents. In a 2D mesh each interface is a polyline whose vertices are about 5 px apart at the default `min_distance` of 4 pixels, so the tangent must be read; `angle_reading` chooses how.

| reading | what it reads | assumes | use it for |
|---|---|---|---|
| `"arc"` (default) | the geometric least-squares circle through all the interface's vertices; the tangent at the circle's point nearest the junction (an interface with fewer than 4 vertices: its chord, junction to other end) | every interface is a circular arc: uniform tension and pressure difference, the tissue deformed by its own forces only, near mechanical equilibrium | free tissues near equilibrium; **not** compressed or confined cells |
| `"chord"` | the direction of each interface's first edge at the junction (foambryo2D's reading) | nothing beyond the first edge | **compressed or confined cells** (an embryo in its eggshell, cells on a substrate): the reading to use there |
| `"fitted_tangent"` | the tangent at the junction of a parabola through the junction and the next four vertices of the interface | the interface is smooth over its first four edges | the other reading local to the junctions, to check how much the choice matters under confinement |

**Why confinement changes the reading.** In a *C. elegans* embryo inside its eggshell, the cells are compressed against the shell. The shell's contact forces act on the cells where they touch it, so an interface's curvature follows Laplace's law, and its tangent obeys Young–Dupré, only close to the junctions; further along, the interface takes the shape the shell imposes. For compressed cells, therefore:

- **no Laplace pressures**: they would be read from curvatures that the shell, not the cells' tensions and pressures, sets, and would not make sense;
- **no arc reading and no global fit**: both fit a circle to the whole interface, including the part the shell shapes;
- **the chord**, which reads only each interface's first edge at the junction; `"fitted_tangent"`, which reads the junction and its next four vertices, is the other reading local to the junctions.

**What was measured, on one embryo per condition** (a 4-cell *C. elegans* embryo inside its eggshell and one whose eggshell was removed, three frames each, dw2d's mesh at its defaults; see the [eggshell example](../examples/choosing-an-angle-reading-inside-an-eggshell.md)):

| quantity | inside the eggshell | eggshell removed |
|---|---|---|
| RMS distance of the cell–medium interfaces' vertices to their fitted circle | 0.67–1.76 px | 0.06–1.04 px |
| the same, cell–cell interfaces | 0.13–0.48 px | 0.12–0.58 px |
| range of the relative tensions (mean 1), one frame: the global fit | 0.23–2.03 | 0.54–1.62 |
| the same, the two local readings (chord; fitted tangent) | 0.51–1.36; 0.61–1.33 | 0.53–1.41; 0.59–1.54 |

Pressed against the shell, the cell–medium interfaces depart from a circle about twice as much as without it, and the global fit, which forces them onto circles, spreads the tensions over a range more than twice as wide as the local readings' (1.80 against 0.86 and 0.72). These numbers come from one embryo per condition: they show which assumption fails inside an eggshell, not how accurate any reading is.

**Free tissues: arcs and pressures apply in principle.** On synthetic equilibrium foams the arc reading lowers the relative tension error by 0.068 and the pressure error by 0.091 of the RMS pressure against the chord (with the mesher's default junction relocation), and is significantly worse in no tier (BENCHMARKS.md, *Two-dimensional tissues*); it is the default for that reason. **Open question:** on the free embryo of the [worked example](../examples/two-dimensional-tissue.md) (one embryo, one frame), the chord gives three of the four cell–cell interfaces 0.39 to 0.61 times the tension the arc reading gives them, and no real image has known tensions. Which reading is closer to the truth on real free tissues is not known.

## Which pressures and stresses to trust

| tissue | junction angles | tensions | pressures | full stress | deviatoric stress |
|---|---|---|---|---|---|
| deformed by its own forces only (free, near equilibrium) | `"arc"` (default); the global fit, opt-in | yes | yes | yes | yes: the anisotropic part of the cell's stress |
| compressed or confined (eggshell, substrate) | `"chord"`; `"fitted_tangent"` as a check | yes, read near the junctions | **no** | **no** | the contribution of the cortical tensions only, not the cell's total stress anisotropy |

**Pressures** come from Laplace's law on the interfaces' curvatures. In a compressed or confined tissue the curvature of an interface is imposed partly by the shell or the substrate, so Laplace's law does not give the cells' pressures: foambryo computes numbers, and they are not measurements. Only the tensions read near the junctions may be read there. foambryo cannot tell a confined tissue from a free one: this is the user's call.

**The full stress** of a cell, its mean stress tensor $\sigma = -p\,I + \frac{1}{A}\sum_m \gamma_m \sum_e l_e\, \mathbf{t}_e \mathbf{t}_e^{\top}$ (a sum over the cell's interfaces $m$ and their segments $e$, of length $l_e$ and unit tangent $\mathbf{t}_e$; $A$ the cell's area; 2 × 2 in `(row, column)`, in units of a pressure), contains the pressure, so it is not a measurement in a confined tissue either.

**The deviatoric stress**, $\sigma - \frac{\operatorname{tr}\sigma}{2} I$, drops the isotropic term $-p\,I$: it is computed from the tensions and the cell's boundary alone, and is the contribution of the cortical tensions to the cell's stress anisotropy (its principal directions and the difference of its principal values; not the mean normal stress). In a free tissue that is the anisotropic part of the cell's stress. Under confinement, the shell's contact forces also act on the cells and are not in it: there it is the cortical tensions' contribution only, not the cell's total stress anisotropy, and it holds as such when the tensions are read near the junctions and each interface's tension is uniform along it. `plot_force_inference(mesh, tensions, pressures, stress="deviatoric")` draws it in 2D and in 3D, with its own colour bar and without colouring the cells by pressure (on the command line, `foambryo view MESH --stress deviatoric`); `stress="full"` (the default) draws the full stress.

A cell that rows of several junction-graph components determine has no single pressure scale: its pressure is `nan` in the flat map, with a warning (its pressure in each component is in the report). A cell cut by the image border is reported; its pressure is not withheld, unlike a cut cell in 3D, and no stress glyph is drawn for it, full or deviatoric.

## Junctions on the image border, and lenses

**The image border.** Where the image border cuts a cell–cell interface, the medium along the border makes a straight 180° wedge that is not a contact angle. By default (`exclude_image_border=True`) the vertices on the image border are excluded: on a mesh from dw2d 0.8.0 they are dw2d's own mark, taken before any vertex moves and never moved by its junction relocation (`PlanarMesh.image_border_vertices`, `foambryo.image_border_vertices(mesh)`); on bare arrays, the vertices on the image rectangle. `excluded_vertices` excludes more.

**Lenses.** When two interfaces take their Young's-law rows from the same junctions and no other (a small lens, a cell with two sides, often on the border of a crop), their tensions are determined only as far as their two columns are well conditioned; read by chords, the two columns are equal and the system is singular, and the solve would return two opposite tensions of order 1e11 to 1e12. Where the condition number of the two columns exceeds `MAX_LENS_CONDITION` (20: an angle error of 1/20 rad could then change the two tensions by their own size), those junctions give no row, the two tensions are `nan`, `IllConditionedLensWarning` names the interfaces, and `TwoDimensionalReport.refused_lenses` records them. On 3 522 readings of synthetic and real meshes the guard fired on 2.6 %, almost all crops, and changed nothing elsewhere, bit for bit.

## The global fit (opt-in, provisional)

```python
solution, pressures, report = foambryo.infer_forces_by_equilibrium_tiling(mesh, mask, diagnostics=True)
tensions = solution.dict_tensions
```

Instead of reading each junction and each interface alone, `infer_forces_by_equilibrium_tiling(mesh, labels, mean_tension=1.0, *, max_evaluations=EQUILIBRIUM_TILING_MAX_EVALUATIONS, diagnostics=False)` fits the label image's crack points (the middles of the sides shared by two pixels of different labels) by one equilibrium tiling of circular arcs, in which every interface is a circle and Young's and Laplace's laws hold exactly (Noll N., Streichan S. J., Shraiman B. I., *Variational method for image-based inference of internal stress in epithelial tissues*, Phys. Rev. X 10, 011072 (2020), [doi:10.1103/PhysRevX.10.011072](https://doi.org/10.1103/PhysRevX.10.011072)). It starts from the arc reading of the mesh. It needs the label image the mesh was built from as well as the mesh, and refuses one that cannot be the mesh's. It returns a `ComponentTensionSolution` (each component a biconnected block of the graph of materials, gauged to `mean_tension`), the pressures relative to the medium and, with `diagnostics=True`, an `EquilibriumTilingReport` with the reason for every `nan`. On the command line: `foambryo infer MESH --tension-method equilibrium-tiling --mask MASK`.

- **Opt-in.** On synthetic equilibrium foams it is more accurate than the arc reading in almost every tier and scale (BENCHMARKS.md), including a 4 096-cell tissue, but it refuses some images whole where the arc reading refuses none (11 of 453 crops). Its refusals are `nan` with a reason, not wrong numbers.
- **Provisional**, as every 2-D name.
- **Not for compressed or confined cells.** It assumes circular interfaces and Laplace's law for every cell; inside an eggshell or on a substrate neither holds away from the junctions, and its tensions are not the cells' (on one embryo inside its eggshell it spread the tensions from 0.23 to 2.03, against 0.51 to 1.36 for the chord).

## Curvature and stress in 2D

`foambryo.curvature.compute_curvature_interfaces(mesh, reading="arc")` returns each interface's signed curvature $\kappa_{ab}$ (convention above), read from the circle the 2-D pressures fit (`reading="arc"`, the default) or from the vertex curvatures (`"local"`); `compute_curvature_vertices_cotan` returns the vertex curvatures, $K_i = (t_2 - t_1)/l_i$ (the cotangent Laplacian's one-dimensional form, exact on a circle sampled at equal steps; the junctions, the interfaces' ends and the image border's vertices have none, `nan`). `compute_gaussian_curvature_vertices` refuses a 2-D mesh: a curve has one curvature. The cells' stresses are drawn by the viewer (`plot_force_inference`, above); `foambryo.plotting_utilities.planar_mesh_drawing(mesh)` returns what the viewers draw of a 2-D mesh (each cell cut into triangles by ear clipping, its area and centroid, the interfaces and junctions), for figures of your own; it needs neither Polyscope nor a display.

## The command line in 2D

```bash
foambryo mesh mask.tif --spacing 0.2457 0.2457 -o embryo.rec
foambryo infer embryo.rec -o results
foambryo infer embryo.rec -o results-chord --angle-reading chord --no-pressures
foambryo infer embryo.rec -o results-global-fit --tension-method equilibrium-tiling --mask mask.tif
foambryo view embryo.rec --image membranes.tif --angle-reading chord --stress deviatoric
```

`foambryo mesh` meshes a 2-D mask (`--spacing DY DX`, the two equal; `--min-distance` 4 pixels by default; thin margins of medium padded as delaunay-watershed does, `--no-pad-thin-medium-margins` to turn it off) and writes a `.rec` file that records `dimension = 2`. `foambryo infer` infers a 2-D mesh with Young's law and Laplace's law (`--pressure-method laplace`, the only one in 2D), takes `--angle-reading arc|chord|fitted_tangent`, and with `--tension-method equilibrium-tiling --mask MASK` runs the global fit; `run.json` records `dimension`. `foambryo infer-over-seeds` refuses a 2-D mask (dw2d has no dither seed), and `--media` is refused on a 2-D mesh. `foambryo view` opens the 2-D window on a 2-D mesh or mask, with `--image` under it and, with `--stress deviatoric`, the deviatoric stress's glyphs. See the [command-line specification](../command_line_specification.md).

## The viewers in 2D

`foambryo view` and `plot_force_inference`, `plot_tension_inference`, `plot_pressure_inference` and `view_mask` open a 2-D window on 2-D input: each interface a curve coloured by its tension, each cell a polygon coloured by its pressure, the junctions as points, and each uncut cell's stress glyph; rows run downward, as in napari and matplotlib. See [Viewers](viewers.md). The napari plugin, napari-foambryo 0.2.0, has 2-D docks too.

## Coming from foambryo2D

foambryo2D is frozen; its inference runs in foambryo since 0.8.0, on foambryo's own Young's-law rows, junction graph and solves. Its per-junction Young–Dupré inference is `foambryo.infer_tensions(mesh, angle_reading="chord")`, which gives its tensions to 1e-12 of the mean tension on 18 reference meshes, on the largest junction-graph component (the only one foambryo2D solved; foambryo gauges each component and returns the others too). foambryo adds the arc reading (the default), pressures, the image-border mark, the lens guard, the global fit and the viewers. Meshes from dw2d 0.1.5 are read in their former orientation and give the same results.

## Examples

- [A free 2D tissue: mask → mesh → tensions, pressures and stresses](../examples/two-dimensional-tissue.md): a 4-cell embryo without eggshell, the arc reading and the global fit.
- [Choosing an angle reading inside an eggshell](../examples/choosing-an-angle-reading-inside-an-eggshell.md): the chord, with `fitted_tangent` as a check; why the arc reading and the global fit do not apply; no pressures; the deviatoric stress as the cortical tensions' contribution.
