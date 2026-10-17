# Tension and pressure methods

## Relative tensions, relative pressures

Contact angles fix the **ratios** of the surface tensions, not their scale, and the Laplace law fixes the **differences** of pressures, not their level. So foambryo returns:

* tensions **normalised** so that their mean is `mean_tension` (1 by default). The normalisation is exact: the mean of the returned tensions is `mean_tension` to rounding, in each connected piece of the mesh's junction graph (`LogLami` normalises the geometric mean instead; `ForceBalanceSoftRow` and `ForceBalanceMAP` hold it as a soft constraint, by design). If the junction graph is in several pieces, each piece has its own scale and a `JunctionGraphGaugeWarning` says so: use `infer_tensions_by_junction_component` to see which interface belongs to which piece, and never compare tensions across pieces;
* pressures relative to `base_pressure`, the pressure of the medium (label 0; 0 by default), in units of tension per length unit of the mesh (per voxel for a mesh in voxel units, per um for a mesh in um).

An interface with no triple line (a closed surface: an engulfed cell, a sealed lumen) has no measurable tension and is reported as `nan`, never as a number. `infer_forces` and `infer_pressures` then refuse the pressures (`UndeterminedTensionError`), because one undetermined tension makes nearly every pressure undetermined; `infer_tensions_then_pressures` keeps the tensions and returns the refusal instead of raising it.

```python
from foambryo import dcel_mesh_from_file, infer_forces, infer_pressures, infer_tensions

mesh = dcel_mesh_from_file("tissue.rec")
tensions, pressures = infer_forces(mesh)               # mean_tension=1, base_pressure=0
tensions = infer_tensions(mesh, mean_tension=1)        # the same two calls, one after the other
pressures = infer_pressures(mesh, tensions, base_pressure=0)
```

`tensions` maps each interface `(label_1, label_2)`, with `label_1 < label_2`, to its tension; `pressures` maps each material (the medium, 0, included) to its pressure.

## Tension methods

foambryo has several strategies to compute tensions. They fall into two families that respond very differently to mesh-reconstruction quality — see [`BENCHMARKS.md`](https://github.com/VirtualEmbryo/foambryo/blob/v0.7.0/BENCHMARKS.md) — angle-reading methods, which read local contact angles at triple junctions, and operator-inverting methods, which assemble and invert a global discrete operator over the whole mesh:

Angle-reading:

- `YoungDupre` (Young-Dupré with cosines only) — the recommended default when mesh quality is uncertain,
- `YoungDupreLocal` (the same balance imposed per trijunctional edge instead of once per triple line; a better diagnostic, a worse point estimate),
- `ProjectionYoungDupre` (Young-Dupré with cosines and sines),
- `Lami` ([Lami's theorem](https://en.wikipedia.org/wiki/Lami%27s_theorem)),
- `InvLami` (inverse of the Lami's relation),
- `LogLami` (logarithm of the Lami's relation).

Operator-inverting:

- `Cotan` (cotangent formula, see [Yamamoto et al. 2026](https://doi.org/10.1016/j.devcel.2026.04.013)),
- `InvCotan` (inverse of the cotangent formula),
- `Variational` (variational formulation, see our [article](https://doi.org/10.1038/s41592-023-02084-7)),
- `ForceBalance`, `ForceBalanceSoftRow`, `ForceBalanceMAP` (the unified discrete force-balance operator under three treatments of the normalisation: exact (a KKT system), a soft least-squares row, and a Gaussian-prior MAP estimate).

Also: `Equilibrium`, a geometric-edge balance that fits neither family cleanly.

**On a large tissue, use the default.** `Equilibrium`, `Variational` and `ForceBalance` solve dense systems only: on a 1 866-cell tissue each took one to two hours with a peak of about 68 GiB (`Equilibrium` about 1 hour, `Variational` about 70 minutes, `ForceBalance` about 110 minutes), where `YoungDupre`, the default, takes seconds. See [large tissues](large-tissues.md).

## Pressure methods

Three methods infer the pressures, from the tensions:

- `Variational` (variational formulation, see our [article](https://doi.org/10.1038/s41592-023-02084-7)),
- `Laplace` (Laplace's law),
- `WeightedLaplace` (Laplace's law with weight on curvature by area).

Only `Variational`, the default, has a formulation for cropped tissues (see [Cropped tissues](cropped-tissues.md)); `PressureComputationMethod.has_crop_face_formulation` says so for each method. The Laplace methods read each interface's mean curvature; an interface all of whose vertices lie on junction lines has none, and a cell that no equation then determines gets a `nan` pressure with an `UndeterminedPressureWarning`.

## Flat junction lines

`Cotan`, `InvCotan`, `Lami` and `LogLami` have coefficients that are infinite at a wedge of exactly 0 or 180 degrees (`1/sin`, `cot(phi/2)`, `log sin`). Such wedges occur on short triple lines whose faces are exactly coplanar, because mesh vertices sit on the voxel lattice. These four methods leave such a line's equations out, with a `DegenerateWedgeWarning` that says how many lines were left out and how many interfaces were left without an equation (those are `nan`). The other methods' coefficients stay finite there.

## Error bars and diagnostics

`infer_tensions_with_uncertainty` returns the tensions of `infer_tensions` with an interval, and `compute_inference_diagnostics` says how well the mesh determines them. **No interval is a calibrated error bar on a reconstructed mesh.** The first-order interval of the `YoungDupre` tensions (`UncertaintyModel.AnglePropagation`) is a **lower bound** on the error: on 2 400 reconstructions with known tensions, one reconstruction's `± std` held the true tension for 26–54 % of the interfaces (68 % for a calibrated interval) and `± 2 std` for 48–79 % (95 %), because the error is mostly the reconstruction's bias, which no propagated scatter measures. Neither function has a formulation for crop faces yet: both raise `ValueError` on a cropped mesh. **Both stop at about 1 300 cells** of real size: they compute the diagnostics from the dense force-balance operator (three rows per mesh vertex, one column per interface and per cell) and its singular value decomposition, which scipy cannot take past `2**31 - 1` entries, so both raise `ValueError` at once on a larger mesh, whatever the method ([large tissues](large-tissues.md#error-bars)). See the `foambryo.uncertainty` module in the [API reference](../api/uncertainty.md) for every interval and its measured calibration.

## How the contact angle is read: `angle_reading`

Every tension method that reads angles — the six angle-reading ones plus `Cotan` and `InvCotan` — takes an `angle_reading` keyword. It selects **how a contact angle is measured on the mesh**, not how tensions are solved for once it has been measured:

- `angle_reading="chord"` (**the default, and unchanged**) measures the angle between the two incident triangles' flat faces. A flat triangle has no curvature to report, so on a curved interface it reports the angle of the *chord* to the triangle's far vertex rather than the interface's *tangent* at the junction.
- `angle_reading="quadric"` fits a generic quadric surface to the interface's own mesh vertices in a small window around each junction edge and reads that surface's tangent instead. For a circular arc the two differ by exactly `κL/2`, with `κ` the interface's curvature and `L` the distance to the triangle's far vertex, and that is the correction applied.
- Nothing else changes: no vertex moves, no interface area or cell volume changes, and the mesh is untouched. Only the angle is read differently.

```python
from foambryo import TensionComputationMethod, dcel_mesh_from_file, infer_tensions

mesh = dcel_mesh_from_file("my_mesh.rec")
tensions = infer_tensions(mesh, 1, TensionComputationMethod.YoungDupre, angle_reading="quadric")
```

**The measured gain**, on 42 mechanically-equilibrated benchmark cases meshed by `dw3d` with its defaults (see [`BENCHMARKS.md`](https://github.com/VirtualEmbryo/foambryo/blob/v0.7.0/BENCHMARKS.md) for the full table and its provenance):

| `dw3d` `min_distance` | 3 | 4 |
|---|---:|---:|
| length-weighted junction-angle flattening slope, `chord` | −0.2135 | −0.2176 |
| the same, `quadric` | **−0.0496** | **−0.0072** |
| tension contrast (1.000 is perfect), `chord` | 0.791 | 0.766 |
| the same, `quadric` | **1.022** | **1.059** |
| median per-line angle error, `chord` | 6.00° | 7.03° |
| the same, `quadric` | **4.73°** | **5.46°** |

Contrast and per-line angle error improve *together*, which is what distinguishes a geometry fix from a bias-for-variance trade.

**The window is fixed and is not a tuning parameter.** The quadric is fitted in a window of four times the triangle's own lever, isotropic in the two surface directions. Those values were chosen a priori and then re-selected independently by held-out fit residual under a pre-registered coverage floor, which landed back on exactly them. They are module constants rather than arguments on purpose: exposing them would invite tuning them against the very error they are judged on, and a window tuned that way would flatter itself.

**Coverage, and what happens where the fit does not apply.** The quadric needs at least ten of the interface's own mesh vertices inside its window. Where it has fewer, that face keeps the chord reading, and a result can therefore mix the two conventions. This is never silent:

```python
from foambryo import infer_tensions_and_residuals_with_report

*_values, report = infer_tensions_and_residuals_with_report(
    mesh, 1, TensionComputationMethod.YoungDupre, angle_reading="quadric",
)
report.coverage_fraction    # e.g. 0.919 -- the fraction of trijunction faces corrected
report.uncorrected_lines    # the trijunction lines that are only partly corrected
report.is_complete          # False when any face fell back
```

Calling `infer_tensions` or `compute_angles_tri` directly, which drop the report, emits an `AngleReadingCoverageWarning` carrying the same numbers instead. On the benchmark meshes coverage is 0.919 of faces at `min_distance = 3` and 0.890 at 4 — but only **0.220** and **0.178** of *lines* are corrected throughout, so most lines carry a mixture. Look at `uncorrected_lines`, not only at the fraction.

**When *not* to use it, and this is the most important caveat.** The correction removes a `κL/2` chord deficit, and on a **well-resolved** mesh that deficit is already small while the reader's known over-application is not. Measured on the 47 ground-truth simulation meshes that ship with the benchmark — far finer than any reconstruction — the chord reading's contrast is already 1.000 to within 2×10⁻⁵, and `angle_reading="quadric"` moves it to a median of 1.080, **worse on 47 of 47 cases**. Use it on `dw3d`-reconstructed meshes, not on simulator output or any mesh whose elements are already much smaller than the interface's radius of curvature.

**The honest residual.** The correction is not exact, and four measured facts qualify it:

1. It **over-applies at fine spacing.** On a coarsened reference mesh, where the true angles are known, the reader removes about 26 % more rotation than the chord deficit it exists to remove at `min_distance = 3`; by `min_distance = 7` it under-applies instead. The reference-mesh result above is the same effect at its most extreme.
2. Consequently a residual slope survives: the corrected arm sits **+0.0487** above its own corrected floor at `min_distance = 3`, while at `min_distance = 4` the excess contains zero (−0.0102, CI [−0.0288, +0.0127]).
3. `YoungDupreLocal` is the one method whose **tension** error gets worse under the quadric reading even though its angles improve — it imposes the balance per edge and so never performs the along-line average that cancels the correction's per-edge scatter. Use `YoungDupre` with `angle_reading="quadric"`, not `YoungDupreLocal`.
4. It reads angles a second-order surface fit reports; it does **not** move any vertex, so it cannot repair a mesh whose junction line is in the wrong place.

A runnable side-by-side comparison on one benchmark case: [`Examples/second_order_angle_reading.py`](../examples/second-order-angle-reading.md).

## Declining folded trijunction observations: `refusal`

A **folded** trijunction edge is one whose three interior wedge angles do not enclose the edge the way a trijunction requires: one of them is past a straight angle. The Young-Dupré balance every angle-reading method above imposes presupposes the opposite, so such a row is an observation the geometry rules out. On `dw3d`-reconstructed meshes 2-25 % of trijunctional edges are folded, depending on the point placement; on the simulator's own meshes essentially none are (5 in 14 183).

Two reasons this cannot be read away. `YoungDupreLocal`'s rows are built from `cos` of the wedge, and `cos` is even about a full turn, so a folded wedge and its reflection are literally the same row — no `angle_reading` reaches it. And `LogLami`'s rows take `log(sin(...))`, which is undefined for a folded wedge, so it does not degrade there, it **raises**.

`infer_tensions` and its relatives therefore take a `refusal` keyword, which is **off by default** and changes nothing until you pass it:

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

- `ExcludeEdgeFromLineMean` — the folded edge leaves its line's angle mean and its line's length; the line still reports, from what survives.
- `DropLine` — a line carrying any folded edge does not report at all.
- `DownweightLine` — nothing is removed from the measurement; the line's mean is unchanged and only its row weight is scaled, by the fraction of its arclength that is not folded.

**Report the refused share alongside any accuracy you gain from this.** A policy that improves an error statistic by declining the data it would have got wrong has not improved anything, and `refusal_report` is what makes that visible. Measured on 15 reconstruction variants x 40 mechanically-equilibrated benchmark cases: `ExcludeEdgeFromLineMean` lowers the gauge-matched tension error on all 15 and on all eight angle-reading methods, and leaves every interface determined; `DropLine` costs a median 16-76 % of the junction **lines**, which is nothing to `YoungDupreLocal` (whose row is one edge) and is most of the data to every line-averaged method, on some meshes all of it.

**What happens when a refusal leaves something unconstrained.** An interface with no surviving junction line is reported `float("nan")`, never a number, with the same `JunctionGraphGaugeWarning` a fully closed interface already raises; a mesh with no surviving observation at all returns `nan` for every interface rather than raising. Use `infer_tensions_by_junction_component` to see which interfaces those are and which gauge component each surviving one belongs to.

**A method that reads no wedge angle refuses a policy** rather than ignoring it — `Equilibrium`, `Variational` and the three `ForceBalance*` members raise, for the same reason they already refuse a non-default `angle_reading`.

A runnable comparison of the three policies on one benchmark case: [`Examples/refusing_folded_trijunction_wedges.py`](../examples/refusing-folded-trijunction-wedges.md).
