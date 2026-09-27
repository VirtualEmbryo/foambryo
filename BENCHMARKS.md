# Benchmarks: which tension method to use, and how mesh quality affects it

> **Measured on dw3d meshes reconstructed before dw3d 0.5.0's default change.** Every mesh behind
> these numbers was reconstructed with dw3d's junction relocation off, the default up to dw3d 0.4.2.
> Since dw3d 0.5.0 relocation is on by default; reconstruct with `relocate_junctions=False` to
> reproduce the meshes measured here. See `CHANGELOG.md` (0.5.0) and `RELEASE_NOTES.md`.

Short, honest comparison. Numbers below are measured, not estimated; each section states the
population, the statistic and the configuration its numbers were measured on.

**Conventions shared by every section.** Unless a section says otherwise, the numbers come from
the benchmarking dataset: 47 simulated cell aggregates of 2 to 10 cells, each with a ground-truth
mesh, ground-truth tensions and a segmentation mask. 42 of the cases are at mechanical
equilibrium and carry every accuracy claim below; the other 5 (`005`, `015`, `018`, `021`, `043`)
are not, are kept as a negative control, and are never pooled with the 42. Every tension error is
**gauge-matched**: only tension ratios are physical, so before any error is computed the inferred
tensions are rescaled by the single factor `mean(truth) / mean(inferred)` taken over the
interfaces the two share (`foambryo_benchmarks.metrics.gauge_matching_scale`). A relative tension
error is `|γ̂ − γ| / |γ|`, per interface.

## Accuracy vs `dw3d`'s two meshing variants

Gauge-matched median relative tension error, 42 mechanically-equilibrated cases, meshed both
ways and paired per case. `v0_3` is `dw3d`'s default; `link_checked` is its opt-in "revised
meshing" variant (see `dw3d`'s `BENCHMARKS.md`).

**How it was measured.** Each of the 42 equilibrium cases is reconstructed from its
segmentation mask by `dw3d` in both configurations, at `min_distance = 3`, and each method infers
tensions on each reconstruction. A case's error is the median, over the interfaces the
reconstruction shares with the ground truth, of the gauge-matched relative tension error; the
table gives the median of that per-case error over the 42 cases. "Which is better" counts the
cases in which the `v0_3` mesh gives the lower per-case error (no case ties). The grid is
produced by `foambryo_benchmarks/run_attribution.py` and summarised by
`foambryo_benchmarks/analyze_attribution.py`; its per-case records are in
`foambryo_benchmarks/baseline/attribution.json`, where `link_checked` is recorded under its
longer name `offset_excluded_linkcheck`.

| method | on `v0_3` meshes | on `link_checked` meshes | which is better |
|---|---:|---:|---:|
| `YoungDupre` | **0.0882** | 0.1049 | `v0_3`, 29/42 cases |
| `YoungDupreLocal` | **0.0949** | 0.1245 | `v0_3`, 29/42 cases |
| `ForceBalanceKKT` | **0.1299** | 0.3436 | `v0_3`, 38/42 cases |
| `BayesianMAP` | **0.1799** | 0.1962 | `v0_3`, 22/42 cases |

`v0_3` wins on every method — this is why it is `dw3d`'s default. On the *ground-truth* mesh
(no reconstruction error at all: the same statistic, with the simulator's own mesh of each case in
place of the reconstruction), the same four methods reach 2×10⁻⁵ to 2×10⁻² (medians: `YoungDupre` 2.3×10⁻⁵,
`YoungDupreLocal` 4.3×10⁻⁴, `BayesianMAP` 0.020, `ForceBalanceKKT` 0.021) — reconstruction
error, not this package's inference, dominates the error budget at the 2–10-cell scale these
42 cases cover.

## Which method to pick, and why mesh quality matters differently to each

`ForceBalanceKKT` (and `Variational`, the same operator) invert a global operator built from
the mesh; their error tracks how far the mesh sits from mechanical equilibrium (`rho_free`, which
`foambryo.compute_inference_diagnostics` computes from the mesh alone, without ground truth),
measured at Spearman `r = 0.89` within a 72-mesh cohort. That cohort is 72 ground-truth simulator
meshes of larger aggregates (91–166 labelled regions; four simulations, 18 optimisation stages
each; no reconstruction step involved), and the correlation is between `ForceBalanceKKT`'s
gauge-matched median relative tension error per mesh and that mesh's `rho_free` (`r = 0.894`,
`p = 4.8×10⁻²⁶`, over a `rho_free` range of 0.037–0.490). It is a within-population result:
across the 42 benchmark cases alone, whose `rho_free` spans only 0.0007–0.036, the same
correlation is `r = 0.066`, and pooling the two populations hides it.

`YoungDupre` reads local contact angles only and is comparatively insensitive to this — its
accuracy degrades far less between a good mesh and a poor one. Measured on the 42 cases with the
statistic of the table above, between `v0_3` meshes and the `dw3d` configuration whose
reconstructions sit furthest from equilibrium (median `rho_free` 0.583 → 0.992), `YoungDupre`'s
error rises from 0.088 to 0.111 (+26 %) while `ForceBalanceKKT`'s rises from 0.130 to 0.715
(+450 %); within the 72-mesh cohort, `YoungDupre`'s correlation with `rho_free` stays weak
(`r = 0.252`). Practically: if you cannot control or verify your mesh's quality,
`YoungDupre` is the more robust default; if you have reason to trust your mesh's geometry (a
ground-truth simulation, or a mesh you have otherwise validated), the `ForceBalance*` family
uses more of the mesh's information and can do better. See `TensionComputationMethod`'s
docstring for the full angle-reading/operator-inverting split.

## Correctness fixes since the published version

Three defects in the published version are fixed and are worth knowing about if you compare
old and new results on the same mesh:

- `TensionComputationMethod.Equilibrium` raised on every input; it now runs.
- `Cotan`/`InvCotan` had an operator-precedence bug that over-weighted short junctions; fixed
  (changes their output; neither was the default).
- On a mesh with a disconnected triple-junction graph (an engulfed cell, a sealed lumen, or
  simply large enough to fragment), eight of the nine tension methods used to silently drive
  some interfaces' tensions toward zero. Now solved per connected component with its own
  gauge; undetermined interfaces are reported as `nan` rather than guessed. Unaffected: any
  mesh whose triple-junction graph is already fully connected, the common case for small
  cell counts.

Speed is a secondary concern at the sizes measured, but not a negligible one. On the 42
equilibrium cases above (3–34 interfaces) every method runs in well under a second. On 72
converged simulator meshes of 91–166 cells (433–760 interfaces), the median wall time per
inference is 0.43 s for `YoungDupre` (max 0.72 s), 0.41 s for `YoungDupreLocal` (max 0.70 s), 1.16 s
for `Variational` (max 4.2 s), 3.20 s for `ForceBalanceKKT` (max 10.3 s) and 3.74 s for
`BayesianMAP` (max 12.6 s); the force-balance family grows with the interface count with a log-log
slope of about 2.4, the Young-Dupré methods with about 1. The accuracy numbers above are what
should drive a method choice; these tell you what it costs.

## The second-order (`quadric`) trijunction angle reading

**Opt-in, off by default.** `angle_reading="quadric"` on `infer_tensions` and its relatives
replaces the chord each flat triangle reports with the tangent of a generic quadric fitted to the
interface's own mesh vertices around the junction. See `README.md` for how to invoke it and
`foambryo.second_order_angle_reading` for what enters the fit.

**Provenance.** The estimator is the mesh-patch quadric fit developed in a curvature-from-data
study on `dw3d` reconstructions, ported into foambryo's production code. The port reproduces
that study's numbers **bit-identically** on all 47 cases at both mesh spacings it was measured
at (`min_distance` 3 and 4), and the default (`chord`) path is pinned bitwise against the code
as it was before the feature was added by `tests/test_second_order_angle_reading.py`.

**Measurement.** 47 `dw3d`-reconstructed meshes of the benchmarking dataset (`dw3d`'s `v0_3`
configuration, seed 42) at `min_distance` 3 and 4, written `md=3` and `md=4` below; 42
mechanically-equilibrated and 5 non-equilibrium cases reported separately and never pooled;
case-level bootstrap (whole cases resampled with replacement), 10 000 draws, seed `20260901`,
paired (the same resampled cases drive both readings on every draw), intervals in brackets are
the 2.5–97.5 % percentiles. Every number is measured through foambryo's public API
(`infer_tensions(..., angle_reading=...)`). The measurement scripts and per-case results are not
distributed with this package.

### Junction-angle error, 42 equilibrium cases

Length-weighted regression of signed line-mean angle error on the line's true deviation from
120°. There is one row per wedge of each trijunction line: `y` is the wedge angle read on the
reconstruction, averaged along the line, minus its true value (the Neumann angle implied by the
ground-truth tensions), and `x` is that true angle minus 120°. The slope is a weighted
least-squares fit with an intercept, pooled over all lines of the 42 cases, each row weighted by
the junction length foambryo accumulates for it. A slope of 0 means no systematic pull toward
120°; a negative slope means angles are flattened toward 120°. Each reading is quoted against
**its own** floor — a floor read with the chord reader does not bound a reading with the
quadric one.

- **Matched-spacing floor**: the ground-truth reference mesh, registered to the reconstruction
  and coarsened until its median transverse lever — the distance from a junction triangle's apex
  to the midpoint of its junction edge — matches the reconstruction's (8.693 voxels at `md=3`,
  11.236 at `md=4`), then read with the stated reader. It is what a geometrically perfect mesh at
  the same spacing reads.
- **Excess over own floor**: |slope| − |floor|, paired per bootstrap draw.
- **Shuffled-curvature null**: the same correction with each junction face's fitted curvature
  replaced by another face's, randomly permuted within the mesh (seed `20260902`). It should
  do nothing.

| | `md=3` | `md=4` |
|---|---:|---:|
| slope, `chord` | −0.2343 [−0.2556, −0.2111] | −0.2204 [−0.2423, −0.1968] |
| slope, **`quadric`** | **−0.0848** [−0.1020, −0.0676] | **−0.0170** [−0.0366, +0.0036] |
| slope, shuffled-curvature null | −0.2305 | −0.2218 |
| matched-spacing floor, `chord` | −0.1405 | −0.1952 |
| matched-spacing floor, `quadric` | +0.0361 | +0.0272 |
| excess over own floor, `chord` | +0.0938 [+0.0768, +0.1107] | +0.0252 [+0.0037, +0.0488] |
| excess over own floor, **`quadric`** | **+0.0487** [+0.0304, +0.0676] | **−0.0102** [−0.0288, +0.0127] |

Paired improvement in \|slope\| (the mean over bootstrap draws of \|slope, `quadric`\| −
\|slope, `chord`\|, both recomputed on the same resampled cases): **−0.1495** [−0.1713, −0.1262]
at `md=3` and **−0.2029** [−0.2310, −0.1722] at `md=4`, on **100 %** of bootstrap draws. The
null, measured the same way against `chord`, moves −0.0037 and +0.0015 — indistinguishable from
doing nothing.

### End to end, every method that reads angles, 42 equilibrium cases

`contrast` is the gauge-matched OLS slope of inferred on true tension; **1.000 is perfect**. It
is computed per case (ordinary least squares with an intercept over the interfaces shared with
the ground truth, after gauge matching) for the cases with at least 4 matched interfaces — 36 of
the 42 at `md=3` and 37 at `md=4`, for every method — and the table gives its median over those
cases. `median |c−1|` is the robust companion, the median over the same cases of
\|contrast − 1\| (one case, 038, reaches \|contrast\| > 3 under *both* readings and dominates any
squared statistic). The per-line angle error is the absolute value of the regression's `y` above;
it is a property of the *reading* and is therefore the same for every method: **7.35° → 4.83°**
at `md=3` and **8.22° → 5.43°** at `md=4` (medians over lines within each case, then medians
over the 42 cases), p90 17.65° → 14.65° and 18.41° → 13.20° (each case's 90th percentile, median
over cases).

| method | `md=3` contrast | median \|c−1\| | `md=4` contrast | median \|c−1\| |
|---|---|---|---|---|
| `YoungDupre` | 0.738 → **0.954** | 0.262 → **0.073** | 0.722 → **1.056** | 0.278 → **0.080** |
| `ProjectionYoungDupre` | 0.731 → **0.935** | 0.269 → **0.078** | 0.714 → **1.018** | 0.286 → **0.066** |
| `Lami` | 0.725 → **0.941** | 0.275 → **0.074** | 0.734 → **1.008** | 0.266 → **0.092** |
| `InvLami` | 0.712 → **0.922** | 0.288 → **0.086** | 0.711 → **1.000** | 0.289 → **0.065** |
| `LogLami` | 0.691 → **0.919** | 0.309 → **0.082** | 0.697 → **0.980** | 0.304 → **0.100** |
| `YoungDupreLocal` | 0.799 → 1.029 | 0.201 → 0.089 | 0.829 → 1.135 | 0.208 → 0.188 |
| `Cotan` | 0.721 → **0.930** | 0.279 → **0.077** | 0.723 → **1.016** | 0.277 → **0.068** |
| `InvCotan` | 0.734 → **0.943** | 0.266 → **0.060** | 0.724 → **1.015** | 0.276 → **0.072** |

Contrast and per-line angle error improve **together**, which is what separates a geometry fix
from a bias-for-variance trade.

**`YoungDupreLocal` is the exception and it is a real one.** Its relative RMS tension error (per
case, the root-mean-square of the gauge-matched `γ̂ − γ` divided by the mean true tension) gets
*worse* under the quadric reading — `+0.0290` [`+0.0019`, `+0.0603`] at `md=3` and `+0.97`
[`+0.035`, `+2.808`] at `md=4`, each the mean over the 42 cases of the paired per-case change,
quadric minus chord — while its angles improve like everyone else's. It imposes the balance per
trijunctional edge and so never performs the along-line average that cancels the correction's
per-edge scatter. Prefer `YoungDupre`.

### Coverage, and where it does not apply

Pooled over the 42 equilibrium cases: the fraction of trijunction-incident faces whose quadric
fit had enough samples to be applied, and the fraction of trijunction lines on which every
incident face was corrected.

| | `md=3` | `md=4` |
|---|---:|---:|
| trijunction **faces** corrected | 0.919 | 0.890 |
| trijunction **lines** corrected throughout | **0.220** | **0.178** |

The two differ by a factor of four and both matter. Uncorrected faces keep the chord reading, so
most *lines* carry a mixture of the two conventions. `AngleReadingReport.uncorrected_lines` names
them; the report-less entry points emit an `AngleReadingCoverageWarning` instead.

### The regime where it does not help

On the 47 **ground-truth** simulation meshes — much finer than any reconstruction — the chord
reading is already exact to within 2×10⁻⁵ of contrast and the quadric reading is **worse on
47 of 47 cases** (median contrast 1.080). Both figures are for `YoungDupre`, as medians over the
47 cases. The correction over-applies at fine spacing. Use it on reconstructions.

### Cost, and what it does not change

Median wall time per case over the 42 equilibrium cases: the quadric fit is 0.024 s, paid once
per mesh and cached on it; applying the rotations costs about as much again as the chord read on
every `compute_angles_tri` call (+103 % at `md=3`, +116 % at `md=4`). Nothing else moves: the
mesh is untouched, the mesh-validity check is identical under both readings (3 abnormal
non-manifold edges at `md=3`, 0 at `md=4`, 47/47 watertight), and all 676 matched interfaces have
identical areas under the two readings in 47 of 47 cases (areas recomputed from the mesh after
each reading has run).

### A predicate that used to be missing

`foambryo_benchmarks.metrics.interface_area_fractions`, `interface_area_fraction_errors` and
`closer_interface_area_fractions` compare interface areas **as a fraction of the mesh's own
total**, which is scale-invariant and needs no registration. Interface-area error previously had
no predicate in the benchmark suite; this is it. Measured on the reconstructions above (`v0_3`,
42 equilibrium cases), against each case's ground-truth mesh: for each case the median, over the
interfaces present in both meshes, of \|Δfraction\|, then the median over cases — **0.00070** at
`md=3` and **0.00083** at `md=4`. Interfaces present in only one of the two meshes are counted
rather than scored: 5 and 4 invented and 7 and 7 missed interfaces over the 42 equilibrium
cases.

## Package renamed `benchmarks` -> `foambryo_benchmarks`

The top-level `benchmarks/` package was renamed to `foambryo_benchmarks/` because both
this repository and `delaunay-watershed-3d` shipped a package named
`benchmarks`, and a script that put both repositories on `sys.path` could silently resolve
the name to either one.

**Migration hazard:** pulling this rename into an existing checkout
leaves a stray `benchmarks/__pycache__/` directory on disk — `git mv` moves tracked
files, not gitignored build artifacts. With `benchmarks/__init__.py` gone but the
directory still present, Python imports it as an empty implicit namespace package (PEP
420): `import benchmarks` still succeeds, with no `__file__` and no real submodules.
`tests/test_no_ambiguous_benchmarks_package.py` now tolerates this case (only a
*real*, `__file__`-bearing `benchmarks` package fails it), but the stray directory
should still be deleted on any checkout that hits it — `rm -rf benchmarks/` once
`git status` confirms it holds nothing tracked.
