# Changelog

User-facing changes only, organised by what a user experiences.

## 0.5.0

**New licence: from 0.5.0, `foambryo` is licensed under the GNU General Public License, version 3
or (at your option) any later version** (SPDX `GPL-3.0-or-later`; the text of version 3 is in
`License`), replacing CC BY-NC-SA 4.0. Commercial use is
allowed, under the GPL's conditions (a distributed derivative work must be released under the same
licence, with its source). Earlier releases keep the licence they were published under.

Released together with `delaunay-watershed-3d` 0.5.0, which foambryo now requires
(`delaunay-watershed-3d>=0.5.0`). See `RELEASE_NOTES.md` for what to run and the limitations that
matter on real data.

**Python 3.11 or later is required.** The package metadata said `>=3.6`, which was wrong: the source
needs 3.10 and its dependency `delaunay-watershed-3d` 0.5.0 needs 3.11, so `pip` on an older Python
now refuses to install instead of failing later.

### Meshes you reconstruct with dw3d 0.5.0 are different: junction relocation is on by default

- **dw3d 0.5.0 moves every reconstructed mesh's trijunction vertices onto the trijunction curves the
  segmentation mask itself gives**, in every reconstruction configuration. Triangles and material
  labels are unchanged; vertex positions on the trijunction lines move, so every angle and tension
  foambryo infers from a freshly reconstructed mesh moves too. To reproduce a 0.4 mesh exactly,
  reconstruct with `relocate_junctions=False` (or `set_junction_relocation(False)` on dw3d's
  factory). Meshes you already have on disk are not affected.
- **`dcel_mesh_from_segmentation_mask` now reconstructs at `min_distance=3` by default, not 5** —
  dw3d's own default. The mesh is denser and takes longer to build. Pass `min_distance=5` to get the
  0.4 spacing back; at 5 or more dw3d emits `CoarseSpacingRelocationWarning` once per
  reconstruction (the effect of relocation on surface-tangent contact angles there lies within the
  check's own noise, while facet-based angles improve), so the default call no longer warns.

### The second-order angle reading no longer depends on the units a mesh is expressed in

- **`angle_reading="quadric"` (and `"oriented_quadric"`) now gives the same angles whatever the
  mesh's units.** The fit used to decide whether the two endpoints of the junction edge entered it
  by the sign of a rounding error, which changes with scale: rescaling one benchmark mesh by 117x
  moved its line-mean wedge angles by up to 0.14 degrees. The endpoints are now always included,
  decided by vertex identity. After the repair the same rescaling moves the reading by about
  2e-13 degrees, the same rounding level as the chord reading.
- **This moves previously published second-order numbers by up to 0.14 degrees.** On a smooth
  analytic surface over fourteen triangulations the repair moves the reading itself by at most
  0.016 degrees (about 0.001 at typical spacings): its value is the invariance, not an accuracy gain.
- The historical behaviour stays reachable, for comparison only, through
  `quadric_angle_rotations(..., junction_endpoints="rounding")`; with that and
  `fit_constant_term=False` the reading is bit-identical to 0.4.2. The fit still omits a constant
  term; the docstring now gives the correct reason and what would settle it.

### Two reading selectors that report a folded wedge as folded

- **`angle_reading="oriented_chord"` and `"oriented_quadric"`** report a folded (reflex) trijunction
  wedge as its oriented angle in `[0, 360)` degrees instead of reading it as `360 - theta`. The
  default reading is unchanged (`"chord"`).

### `AngleReadingCoverageWarning` now says how much was covered

- **The warning's message now leads with the measured coverage fraction and the number of
  trijunction lines that were not fully corrected, out of how many**, then lists those lines. When
  and how often it fires is unchanged: it fired on all 199 meshes it was measured on (154 real embryo
  timepoints and 45 benchmark cases), so its firing alone tells you nothing -- the numbers in the
  message are what to read.

### The force viewer works with numpy 2 and draws each cell's stress correctly

Only the display changes: no inferred tension or pressure moves.

- **`plot_force_inference` crashed with numpy 2** (`UFuncOutputCastingError`, on every mesh tried),
  which is what `pip install` gives you. The per-cell stress tensor is symmetric, and is now
  diagonalised as such, so its principal values are real under every numpy version.
- **One principal direction was drawn at the wrong length.** When the second principal value was
  non-negative, its axis was drawn as a unit vector and the third axis was scaled twice. Fixed.
- **A membrane shared by two cells counted with opposite signs in the two cells**, depending only on
  which cell had the lower label: it pulled on one and pushed on the other. It now counts the same
  way in both, as in Batchelor's formula for the mean stress. Checked on two cases with a known
  answer: a single bubble at its Laplace pressure shows no stress, and two mirror-image cells show
  mirror-image stresses. Cells touching only the exterior were already right.
- **The force viewer drew tensile directions as compressive, and compressive ones as tensile.** The
  stress is tension-positive, so a negative principal value is compression; the viewer drew the
  negative ones as "principal axes" and the non-negative ones in red as "compressive". Each set is now
  named and coloured for what it holds — compression in red, tension in teal. The Polyscope vector
  quantities on the point cloud "Stress tensors principal directions" are renamed, for k = 0, 1, 2:
  `principal_axes_k` → `compression_axes_k`, `principal_axes_k_down` → `compression_axes_k_down`,
  `compressive_stress__axes_k` → `tension_axes_k`, `compressive_stress__axes_k_down` →
  `tension_axes_k_down`. The arrows themselves are unchanged; only their names and colours are.
- **`plot_tension_inference(scalar_quantities=True)` raised `TypeError`**: it passed an argument
  (`clean_before`) that the curvature and derivative views do not take. Fixed; the option shows the
  same scalar quantities as `plot_force_inference(scalar_quantities=True)`.

### User-visible text

- The uncertainty calibration notes no longer name internal development steps; each states its
  measurement and points to `foambryo_benchmarks/run_uncertainty_calibration.py`.
- `BENCHMARKS.md` is now self-contained: every number states its population and how it was
  measured, without citing files that are not distributed. Those numbers were measured on meshes
  reconstructed with dw3d **before** junction relocation became its default.
- The README and the docstrings cite the published article (Ichbiah et al., *Nature Methods* 2023,
  doi:10.1038/s41592-023-02084-7) instead of its preprint, and the `Cotan` method's published
  source (Yamamoto et al., *Developmental Cell* 2026, doi:10.1016/j.devcel.2026.04.013). New
  `CITATION.cff`.
- The introductory notebook `Examples/Example.ipynb` says where to download its data (the link in
  `data/alternative_link_to_data`, unzipped into `data/`) instead of asking for Git LFS, which no
  file in the repository uses.
- `Examples/actual.ipynb` is removed: a regression check with no explanatory text, which compared
  floating-point tensions for exact equality. `tests/test_golden_master.py` makes the same check with
  a stated tolerance. `Examples/reference.ipynb` and the two `reference_*.txt` files it wrote stay.

### Pressure inference refuses an undetermined tension instead of returning a non-finite field

- **`infer_pressures`, `infer_pressures_and_residuals` and `infer_forces` now raise
  `UndeterminedTensionError` when any tension they would use is not a finite number**, naming every
  such interface. `UndeterminedTensionError` is new, exported from `foambryo`, and subclasses
  `ValueError`.

  Why: an interface with no triple line at all — a closed surface such as an engulfed cell, a sealed
  lumen, or a fragment a segmentation left inside another cell — has no angle from which its tension
  can be measured, and `infer_tensions` reports it as `nan` (with a `JunctionGraphGaugeWarning`).
  The pressure solve couples every cell to every tension, so that one `nan` used to come back as a
  pressure field in which nearly every cell was non-finite, without any warning from the pressure
  stage. Measured on five real embryo timepoints carrying one to four such interfaces, all but one
  cell's pressure was non-finite each time (54 of 55 up to 71 of 72). That field carried no usable
  value; the pressure inference now says why instead of returning it.

  What does not change: every tension, and every pressure inferred from finite tensions, is
  **bitwise** what it was — verified on 45 benchmark reconstructions and 149 real embryo timepoints.
  `infer_tensions` still returns `nan` for an undetermined interface and the other tensions of such
  a mesh remain valid; use `infer_tensions_by_junction_component` to see which interfaces are
  undetermined.

### The force-balance family now warns that it is unreliable on reconstructed meshes

No behaviour changes and no default moves. What changed is that the estimators say what was
measured about them.

- **`infer_force_balance`, the `ForceBalance*` members of `TensionComputationMethod`, and the
  `rho_free` diagnostic now carry the measurement.** On reconstructed meshes `rho_free` — the
  distance of a mesh from **any** equilibrium of the force-balance operator, the floor no tension
  assignment can get below — has median `0.578` against the reference meshes' `0.0118`, a factor of
  about 50, with per-case ratios from 17 to 850 and **every one** of the 40 equilibrium benchmark
  cases above the `0.2` threshold at which the diagnostics already reported that no surface tensions
  balance the mesh. `rho` at the `Kkt` solution equals `rho_free` by construction, so the solve
  cannot lower it.

  What this means for a user: every `ForceBalance*` tension and pressure inferred on a reconstructed
  mesh, and every comparison of one reconstruction against another through this family, stands on
  that floor. **Read `rho` before reading the tensions.** The angle-reading family (`YoungDupre` and
  relatives) never inverts the operator and is not bounded this way. Whether the family is repairable
  on reconstructed meshes is an open scientific question and is not addressed here — the method is
  published and stays, with the warning it needed.


### The angle-reading methods can now decline a folded trijunction observation

- **New `refusal` keyword on `infer_tensions`, `infer_tensions_and_residuals`,
  `infer_tensions_and_residuals_with_report`, `infer_tensions_by_junction_component` and
  `infer_tensions_young_dupre_local`, default off.** A folded (reflex) trijunctional edge is one
  whose interior wedge angle in some region is past a straight angle; the Young-Dupré balance
  presupposes the opposite, and no angle reading can repair such a row. `foambryo.wedge_refusal`
  offers three ways to decline it — `ExcludeEdgeFromLineMean`, `DropLine`, `DownweightLine` —
  together with `reflex_trijunction_edges`, which lists a mesh's folded edges from the mesh alone,
  and `refusal_report`, which counts what a policy declines in rows and in arclength.
  With the default `NO_REFUSAL` every result is **bitwise** what it was before.
- **An interface left with no surviving junction line is reported `float("nan")`**, with the
  existing `JunctionGraphGaugeWarning`, rather than given the number ordinary least squares would
  have produced for a quantity nothing constrains. The gauge partition is rebuilt over the
  surviving lines; a mesh left with no observation at all returns `nan` everywhere rather than
  raising. New `foambryo.junction_graph.partition_over_surviving_junctions`.
- A method that builds no row from a measured wedge angle — `Equilibrium`, `Variational`,
  `ForceBalance`, `ForceBalanceSoftRow`, `ForceBalanceMAP` — **raises** on a non-default policy
  instead of ignoring it.
- `compute_angles_tri_with_report` gains a `skip_edges` keyword, which is how a policy declines an
  edge's contribution; with `None` (the default) its result is bitwise unchanged.
- **No policy is a default, and one of them is a trap.** `DropLine` removes a whole junction line
  when any one of its edges is folded, which costs a median 16-76 % of the lines on reconstructed
  meshes: harmless to `YoungDupreLocal`, whose row is a single edge, and ruinous to every
  line-averaged method. See README, *Declining a folded trijunction observation*, and report the
  refused share beside any accuracy number.

## 0.4.2

### Fixed a crash on some real meshes in the default tension-inference path

- **`infer_tensions(mesh, mode=YoungDupre)` — the default call in the default mode — no
  longer raises an unhandled `KeyError` on a mesh with a non-manifold triple-line edge.**
  `dcel_mesh_from_multimaterial_mesh` and `dcel_mesh_from_file` reach the same fixed code
  path. The edge that used to crash the solver is now excluded from the tension system
  instead, with a new `foambryo.validation.mesh_io.MalformedTripleLineWarning` naming how
  many edges were excluded and why, so this is never silent. On a mesh with no such edge,
  every affected function is bitwise identical to before (checked on the 42-case equilibrium
  benchmark cohort — one case, already flagged elsewhere in this project as carrying 2
  malformed triple-line edges, is the one exception and now emits the new warning; on that
  mesh only, the excluded edges' fictitious junction no longer contributes to the tension
  system, which changes the returned numbers there, by construction).
- A new `foambryo.junction_graph.MalformedTripleLineError`, carrying the offending edge and
  region-triple, replaces what used to be a bare `KeyError` naming an internal dict key, for
  the (now believed unreachable in practice) case where a genuine inconsistency remains after
  the exclusion above.
- A new `foambryo.junction_graph.JunctionGraphGaugeWarning` is emitted from
  `infer_tensions_and_residuals` (and so from `infer_tensions`) whenever a mesh's
  triple-junction graph is fragmented into more than one gauge component, or has an
  undetermined (fully closed) interface. Use `infer_tensions_by_junction_component` to get
  each interface's own component and gauge instead of the flat, potentially
  scale-mismatched result these entry points return; that function itself does not emit the
  warning, since it already hands the caller the information the warning would describe.
  Does not fire on any of the 42-case equilibrium benchmark cohort (none of those meshes has
  a fragmented junction graph).
- Docstrings for `infer_tensions`, `infer_tensions_and_residuals` and `infer_forces` now
  state plainly that an undetermined interface is reported as `nan`, and that a fragmented
  junction graph's flat return value blends per-component gauges that are not on the same
  scale.

### Fixed a performance regression the `JunctionGraphGaugeWarning` check above introduced

- `infer_tensions_and_residuals` (and so `infer_tensions`) used to recompute the mesh's
  triple-junction partition a second time, from scratch, purely to decide whether to emit
  `JunctionGraphGaugeWarning` — even though the tension solve just above it had already
  computed the same partition and discarded it. This added a measured **~2x per-call
  overhead** (median 2.04x, range 1.82x–2.18x across the benchmark cohort) to `YoungDupre`,
  `YoungDupreLocal`, `ProjectionYoungDupre`, `Cotan`, `InvCotan`, `Lami`, `InvLami` and
  `LogLami` (the 8 methods `JunctionGraphGaugeWarning` applies to). Fixed by reusing the
  already-computed partition instead of recomputing it. Every tension, residual and
  warning-emission decision is bitwise identical to before (checked on all 42 equilibrium
  cases and on the real-embryo frames used to verify the crash fix above); only the
  wall-clock cost changed.

### Added `line_coverage_fraction` to the second-order angle reading's coverage report

- `AngleReadingReport.line_coverage_fraction`: the fraction of trijunction *lines* every one
  of whose faces was corrected by the `"quadric"` reading, as opposed to the existing
  `coverage_fraction`, which is per-*face* (a line with 2 of 3 faces corrected counts as 2/3
  there, but as not-fully-covered here). `None` on a mesh with no trijunctional edge,
  mirroring `coverage_fraction`'s own convention. Included in the coverage warning's message
  alongside the existing per-face number. A trijunction line with partial coverage is not,
  by itself, a known source of extra angle error — see the field's own docstring for the
  measurement this states. Purely additive: every existing field, warning trigger and
  numeric value of `AngleReadingReport` is unchanged.

### Dropped internal report citations from user-visible error text

- Two `ValueError`s (`junction_graph.require_single_component_solution`,
  `tension_inference.infer_tensions_by_junction_component`) and one `NotImplementedError`
  (`uncertainty.compute_inference_diagnostics`) no longer cite internal development reports,
  which are not distributed with the package and so were a dead reference for anyone reading the
  error outside this project. No change to when each is raised.

### Corrected one measurement affected by the crash fix

- The dataset-audit measurement for one benchmark case (the one already known to carry two
  non-manifold triple-line edges, see the crash fix above) recorded a triple line's length
  as `0.14288904810756733`; with those two edges correctly excluded, it is
  `0.08233713750534329` (about 42% shorter) — the two excluded edges were previously being
  summed into that line's length. No other measurement for that case, or any other case,
  changes.

### A note on this release's published accuracy numbers

The `dw3d`-meshing accuracy table in `BENCHMARKS.md` is out of date for reasons unrelated to
this release (an unrelated change to how `dw3d` reconstructs meshes since that table was
last measured) and is being re-measured and republished in its own follow-up — this
release's fixes do not move any number in that table. Do not read today's `BENCHMARKS.md`
table as reflecting 0.4.2.

## 0.4.1

Release hygiene only. No change to inference behaviour: every default is bitwise identical to
0.4.0 on all 47 benchmark meshes.

- **Development artefacts no longer ship or return.** The project's development record is kept
  in the development repository and has never been part of the packaged distribution;
  confirmed by building the wheel and listing its contents. Development notes and scratch scripts
  are now gitignored so they cannot be re-added by accident.
- Internal benchmark filenames and test names that were named after a development phase (e.g.
  `c1_attribution.json`, `run_c2_robustness.py`) are renamed to describe what they measure. This
  does not change any benchmark's recorded numbers, and `foambryo_benchmarks` remains a
  development tool, not part of the installed package.
- `delaunay-watershed-3d`'s three phase-named factory methods
  (`get_v0_3_algorithm`/`get_a1b_algorithm`/`get_a3_a5_algorithm`) are renamed on that side to
  describe their mechanism; the old names still work as deprecated aliases until `dw3d` 0.5.0.
  Nothing here depends on the old names, so no change is needed in this package.

## 0.4.0

### New capability — an opt-in second-order trijunction angle reading

- **`angle_reading="quadric"`** on `infer_tensions`, `infer_tensions_and_residuals`,
  `infer_tensions_young_dupre_local` and `compute_residual_junctions_dict`, and `reading=` on
  `compute_angles_tri` / `compute_trijunction_edges` / `DcelData.compute_angles_tri`. It changes
  **how a contact angle is measured**, not how tensions are solved for: instead of the angle
  between two flat triangles — which is the *chord* to each triangle's far vertex — it fits a
  generic quadric to the interface's own mesh vertices around the junction and reads that
  surface's *tangent*. No vertex moves; no area, volume or interface identity changes.

  On `dw3d`-reconstructed benchmark meshes this takes the gauge-matched tension contrast from
  0.738 to 0.954 (`min_distance = 3`) and 0.722 to 1.056 (4), and the median per-line
  junction-angle error from 7.35° to 4.83° and 8.22° to 5.43°, on all eight methods that read
  angles. Full table, provenance and the residual: `BENCHMARKS.md`. Worked example:
  `Examples/second_order_angle_reading.py`.

- **Nothing changes unless you ask for it.** `"chord"` remains the default and is pinned
  **bitwise** against the previous release's behaviour on all 47 benchmark meshes — every angle,
  length and tension, digested over raw IEEE-754 bits, by
  `tests/test_second_order_angle_reading.py`. If you do not pass `angle_reading`, no number you
  have ever computed with this package moves.

- **The fallback is never silent.** The quadric fit is not admissible everywhere; where it is
  not, that face keeps the chord reading. New `AngleReadingReport` (from
  `infer_tensions_and_residuals_with_report`, `compute_angles_tri_with_report`, or
  `YoungDupreLocalResult.angle_reading_report`) carries the coverage fraction and names every
  trijunction line that is read with a mixture of the two conventions. The report-less entry
  points emit an `AngleReadingCoverageWarning` carrying the same numbers instead.

- **Where it does not help, stated plainly.** On a mesh whose elements are already much smaller
  than the interface's radius of curvature, the chord deficit is small and this correction
  over-applies. On the 47 ground-truth simulation meshes in the benchmark it makes the contrast
  *worse* on 47 of 47 cases. Use it on reconstructions.

- **`YoungDupreLocal` refuses `angle_reading="quadric"` outright**, raising `ValueError` rather
  than returning a degraded result: its tension error gets *worse* under the quadric reading
  even though its angles improve (+0.0290 relative RMS tension error at `min_distance=3`, +0.97
  at 4), because a per-edge balance never performs the along-line average that cancels the
  correction's per-edge scatter. Use `angle_reading="quadric"` with a line-averaged method
  instead (`YoungDupre`, `ProjectionYoungDupre`, `Lami`, `InvLami`, `LogLami`), or keep
  `YoungDupreLocal` on the default `"chord"` reading.

- **A method that cannot honour it now says so**: `Variational`, `Equilibrium` and the three
  `ForceBalance*` members raise `ValueError` on a non-default `angle_reading` rather than
  silently ignoring it.

### New in the benchmark harness

- `foambryo_benchmarks.metrics.interface_area_fractions`, `interface_area_fraction_errors` and
  `closer_interface_area_fractions`: interface area as a fraction of the mesh's own total, which
  is scale-invariant and needs no registration between a candidate mesh and its reference.

### Default behaviour changes — several methods now give correct numbers instead of wrong or missing ones

- **`TensionComputationMethod.Equilibrium` used to crash on every input.** It now runs and
  gives a scientifically sensible result. If you were catching the exception, you no longer
  need to.
- **`Cotan` and `InvCotan` had an operator-precedence bug** that over-weighted short
  junctions on the diagonal of their linear system. Fixed; both methods' output has changed
  (previously published results using them, if any, were unaffected — neither was the
  default).
- **`Variational` now solves through a different, better-justified linear-algebra route by
  default** (routing through the operator directly, `VariationalRoute.Operator`, rather than
  forming and row-normalising the Gram matrix). The previous route (`VariationalRoute.Gram`)
  remains available for anyone who needs to reproduce a number computed under it.
- **A silent aggregation defect is fixed for eight of the nine tension methods**
  (`YoungDupre`, `ProjectionYoungDupre`, `Cotan`, `InvCotan`, `Lami`, `InvLami`, `LogLami`,
  `YoungDupreLocal`): on a mesh whose triple-junction graph is disconnected — a mesh with a
  fully engulfed cell or a sealed lumen, or one large enough to genuinely fragment — these
  methods used to silently drive some interfaces' tensions toward zero while inflating
  others, with no warning. They now solve per connected component with a per-component
  gauge, and interfaces with no triple line at all are reported as undetermined (`nan`)
  rather than guessed. New: `infer_tensions_by_junction_component`, for when you want that
  structure exposed explicitly rather than collapsed into one flat result. This does not
  affect meshes whose triple-junction graph is already fully connected, which is the common
  case for the small-to-medium cell counts most users work with.
- **`torch` is now an optional dependency.** The closed-form (non-autodiff) derivatives were
  already the more numerically accurate path and are now the only one installed by default;
  install `foambryo[autodiff]` if you specifically need the autodiff cross-check.

### New capability

- **Uncertainty intervals on inferred tensions**: `infer_tensions_with_uncertainty`, with
  several interval models (`UncertaintyModel`) and a calibration note on every result stating
  how conservative or optimistic that model measured as on a 42-case benchmark. `ProbeResidual`
  is the recommended default.
- **Mesh-level identifiability diagnostics** (`compute_inference_diagnostics`,
  `InferenceDiagnostics`): before trusting a tension inference on a mesh with no ground
  truth, check `rho_free` (how far the mesh sits from any mechanical equilibrium) and the
  operator's singular-value spectrum.
- **Per-edge Young–Dupré** (`infer_tensions_young_dupre_local`,
  `TensionComputationMethod.YoungDupreLocal`): imposes the balance once per trijunctional
  edge instead of once per triple line on line-averaged angles, giving a per-edge residual
  field usable for diagnostics and a calibrated bootstrap uncertainty. On its own it is
  *less* accurate as a point estimate than the line-averaged `YoungDupre` (averaging first
  is a noise filter, not a bias); it is kept for the diagnostic it enables, not as a
  replacement.

### Documentation corrected

- The capillary energy's pressure term has the sign `E = Σ γA + Σ tL − Σ pV`, not `+ΣpV`,
  in `foambryo.validation.force_balance`'s module docstring — it previously contradicted
  `foambryo.force_balance_inference`'s docstring, which had the correct sign. No numerical
  result was affected: the code was already self-consistent, and this is purely a
  documentation fix.
- The tension method enum (`TensionComputationMethod`) now states which methods read
  measured angles and which invert a global operator, since mesh-reconstruction quality
  affects the two families very differently — see `dw3d`'s and this package's
  `BENCHMARKS.md`.

### Compatibility

`delaunay-watershed-3d`'s reconstruction default has reverted to its published (`dithered`)
configuration in this release cycle (see `dw3d`'s `CHANGELOG.md`); no code change is
required here, but tensions inferred from freshly-reconstructed meshes will differ slightly
from ones inferred while `dw3d`'s revised meshing pipeline was the default, because the mesh
itself is different. This is expected and is the change the accuracy finding above concerns.
