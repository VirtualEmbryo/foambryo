# Methods note: what foambryo computes

This note says what foambryo infers from a multimaterial mesh and how, so that its results can be read correctly. It describes what is implemented, not how well each method performs: for measured accuracy see `BENCHMARKS.md` and the [methods guide](user-guide/methods.md); for every argument, the [API reference](api/index.md). The method itself is published in Ichbiah et al., *Nature Methods* (2023), cited at the end.

## Unknowns and conventions

The mesh is a set of triangles, each separating two **regions**: cells, labelled by positive integers, and the medium, label 0. The triangles between regions $i$ and $j$ form the **interface** $(i, j)$, written with $i < j$; the edges where three regions meet form the **triple lines** (junction lines).

foambryo infers

* one **surface tension** $\gamma_{ij}$ per interface (an energy per unit area), and
* one **pressure** $p_i$ per cell; the medium's pressure is `base_pressure` (0 by default).

It assumes the mesh is at mechanical equilibrium for the capillary energy

$$
E = \sum_{i<j} \gamma_{ij} A_{ij} \;-\; \sum_{i \ge 1} p_i V_i ,
$$

with $A_{ij}$ the area of interface $(i, j)$ and $V_i$ the volume of cell $i$. **Line tensions are not inferred**: every triple line contributes no energy of its own.

A shape fixes only the **ratios** of the tensions, not their scale, and only the **differences** of pressures, not their level. So the tensions are **normalised**: their mean is `mean_tension` (1 by default), and the pressures are relative to the medium's. With the default normalisation the tensions are dimensionless (in units of the mean tension) and the pressures are in units of tension per length unit of the mesh: per voxel for a mesh in voxel coordinates, per micrometre for a mesh in micrometres. The normalisation holds in each connected piece of the junction graph separately; tensions in different pieces cannot be compared (a `JunctionGraphGaugeWarning` says when there are several). An interface that touches no triple line (a closed surface) has no measurable tension and is reported as `nan`.

## Angle-based tension methods

**Young–Dupré at a triple line.** Where regions $a$, $b$ and $c$ meet, the three tensions pull along their interfaces, perpendicular to the line, and balance: $\gamma_{ab}\,\mathbf t_{ab} + \gamma_{bc}\,\mathbf t_{bc} + \gamma_{ac}\,\mathbf t_{ac} = 0$, with $\mathbf t$ the unit vector in each interface, normal to the line and pointing away from it. Let $\theta_f$ be the **contact angle** of region $f$: the angle between the two interfaces that bound $f$, measured inside $f$, so that $\theta_a + \theta_b + \theta_c = 2\pi$. Projecting the balance on each of the three directions gives three equations,

$$
\gamma_{ab} + \gamma_{bc}\cos\theta_b + \gamma_{ac}\cos\theta_a = 0, \quad
\gamma_{ab}\cos\theta_b + \gamma_{bc} + \gamma_{ac}\cos\theta_c = 0, \quad
\gamma_{ab}\cos\theta_a + \gamma_{bc}\cos\theta_c + \gamma_{ac} = 0 .
$$

**`YoungDupre`, the default.** On the mesh, each edge of a triple line carries three wedges, one per region. Each contact angle is averaged over the edges of its line (an arithmetic mean), the three equations of each line are weighted by the line's share of the total junction length, and all lines are solved together by least squares, with the mean tension imposed exactly.

**How an angle is read.** By default (`angle_reading="chord"`) the wedge angle on an edge is the angle between the two flat triangles that share it. Because a triangle is flat, on a curved interface this is the angle of the chord to the triangle's far vertex, not of the tangent at the line. `angle_reading="quadric"` (opt-in) fits a quadric surface to the interface's own vertices in a small window around the edge and reads that surface's tangent instead; where the window holds too few vertices, the edge keeps the chord reading, and a report (or an `AngleReadingCoverageWarning`) says how many. Either way, no vertex moves.

**`YoungDupreLocal`** imposes the same three equations once per mesh edge instead of once per line, so least squares does the averaging along the line; the row weights keep each line's total weight equal to `YoungDupre`'s.

Other members write the same balance differently: with sines as well as cosines (`ProjectionYoungDupre`), through Lami's theorem (`Lami`, `InvLami`, `LogLami`), or through the cotangent formula of Yamamoto et al. (2026) (`Cotan`, `InvCotan`); `Equilibrium` imposes the vector balance itself, component by component, at every junction edge. The [methods guide](user-guide/methods.md) lists them, with their limits.

## Force-balance tension methods

**The virtual-work balance.** At equilibrium, moving any one mesh vertex $\mathbf x_v$ does no work:

$$
\frac{\partial E}{\partial \mathbf x_v}
= \sum_{i<j} \gamma_{ij}\,\frac{\partial A_{ij}}{\partial \mathbf x_v}
- \sum_{i \ge 1} p_i\,\frac{\partial V_i}{\partial \mathbf x_v} = 0
\qquad \text{for every vertex } v .
$$

The derivatives are exact for the triangulated surface: a triangle's area gradient at one corner is $\tfrac12\,\mathbf n \times \mathbf e$, with $\mathbf n$ its unit normal and $\mathbf e$ the opposite edge, and a cell's volume gradient at one corner is $\tfrac16\,\mathbf x_b \times \mathbf x_c$ from the two other corners, each summed over the triangles that contain the vertex. Stacking three rows per vertex gives a linear system $M\beta = 0$ in the unknowns $\beta = (\gamma, p)$, with one column per interface and one per cell (the medium has no column: its pressure is the reference). A reconstructed mesh is never exactly at equilibrium, so foambryo minimises $\lVert M\beta \rVert^2$, and since $\beta = 0$ would minimise it, it fixes the scale with the mean tension. The three members differ only in how:

* **`ForceBalance`**: $\overline{\gamma} = $ `mean_tension` exactly, as a constraint (a Lagrange multiplier: a KKT system).
* **`ForceBalanceSoftRow`**: the same condition as one more least-squares row, so the mean is met only approximately.
* **`ForceBalanceMAP`**: a Gaussian prior on the tensions (mean `mean_tension`, standard deviation 0.3 × `mean_tension`) and a broad one on the pressures; the result is the posterior's maximum, with its covariance. The prior anchors the scale softly, so the mean tension is not exactly `mean_tension`; the relative tensions do not depend on the mesh's length unit.

`foambryo.force_balance_inference.infer_force_balance` returns the tensions and the pressures of the same solve.

**`Variational`** minimises the same quantity, $\lVert M\beta\rVert^2$ with the mean tension met exactly, by a different factorisation (the normalisation is eliminated by substitution and the reduced problem solved by QR), and agrees with `ForceBalance` to rounding. foambryo's original route, which divides each normal equation by its row sum, weights the equations differently; it is kept inside the library so that published numbers can be reproduced.

## Pressures

Given the tensions, `infer_pressures` computes the pressures with one of three methods:

* **`Variational`** (the default): the pressures that best balance every vertex at those tensions, the least-squares solution of $\sum_i p_i\,\partial V_i/\partial\mathbf x_v = \sum_{i<j}\gamma_{ij}\, \partial A_{ij}/\partial\mathbf x_v$. It reads no curvature, and it is the only method that handles cropped tissues.
* **`Laplace`**: Laplace's law on each interface, $p_a - p_b = 2\gamma_{ab}H_{ab}$, with $H_{ab}$ the interface's mean curvature, positive when its centre of curvature lies on $a$'s side, averaged over the interface's vertices off the triple lines; all interfaces are solved together by least squares.
* **`WeightedLaplace`**: the same, with the curvature averaged with area weights.

Every pressure is relative to the medium, whose pressure is `base_pressure`: a non-zero `base_pressure` shifts all pressures by that amount and changes no difference. An interface whose vertices all lie on triple lines has no curvature, and a cell that no remaining Laplace equation connects to the medium gets `nan`, with an `UndeterminedPressureWarning`.

## Diagnostics, error bars and seed spread

`compute_inference_diagnostics` describes how well the mesh determines the force-balance solution:

* $\rho$, at given tensions, is the fraction of the surface-tension force left unbalanced once the best pressures are fitted, $\lVert M\beta\rVert / \lVert M_\gamma\gamma \rVert$ ($M_\gamma$: the tension columns of $M$);
* $\rho_\text{free}$ is the same ratio with the tensions free (at the `ForceBalance` solution): how far the mesh is from an equilibrium of any tensions; above 0.2, no tensions balance the mesh and the force-balance methods should not be trusted on it;
* $\sigma_\text{soft}$ is the smallest singular value of $M$ (columns scaled to unit norm) after the one that belongs to the overall scale: the least well determined combination of unknowns; and $\operatorname{cond}(M)$ is the condition number of $M$.

`infer_tensions_with_uncertainty` adds an interval to each tension. For `YoungDupre` it propagates the measured spread of each contact angle along its line through the solve, to first order: this is a **lower bound** on the error, not a calibrated error bar, because most of a reconstruction's error is a bias that no scatter measures. For the force-balance family the interval is scaled by the force residual. Neither function handles crop faces, and both stop at about 1 300 cells: they need the dense operator $M$ and its singular value decomposition.

`infer_tensions_over_seeds` reconstructs the same mask at a fixed list of seeds of the point placement and averages the tensions; their standard deviation over the seeds measures only the part of the error that changes with the seed, so it too is a lower bound.

## Cropped tissues

Where a cell reaches the image box, the mesh closes it with **crop faces**: triangles on the box, which are not contact with the medium. Every tension method leaves them, and every junction edge and vertex that touches them, out of its equations; an interface made only of crop faces has no tension. The pressures of the cells the crop cuts are `nan`, with a `CutCellPressureWarning`, because part of their surface and volume was not observed. The Laplace methods and the uncertainty functions refuse a cropped mesh. A tissue that really ends on an image face (a coverslip) is declared with `physical_image_faces`. See [cropped tissues](user-guide/cropped-tissues.md).

## References

* Ichbiah S., Delbary F., McDougall A., Dumollard R., Turlier H. *Embryo mechanics cartography: inference of 3D force atlases from fluorescence microscopy.* Nature Methods **20**(12), 1989–1999 (2023), [doi:10.1038/s41592-023-02084-7](https://doi.org/10.1038/s41592-023-02084-7).
* Yamamoto et al. (2026), for the cotangent formula, [doi:10.1016/j.devcel.2026.04.013](https://doi.org/10.1016/j.devcel.2026.04.013).
* foambryo's API reference: [tension inference](api/tension-inference.md), [force-balance inference](api/force-balance-inference.md), [pressure inference](api/pressure-inference.md), [uncertainty](api/uncertainty.md), [crop faces](api/crop-faces.md), [validation](api/validation.md).
