# Benchmarks: which tension method to use, and how mesh quality affects it

> **Measured on meshes from delaunay-watershed-3d (`dw3d`) 0.7.1 with its defaults**, junction relocation included, and foambryo's inference, which is unchanged since 0.7.0. dw3d 0.7.1 puts the tetrahedra in a canonical order before reading them and its mesh surgery no longer leaves abnormal edges, so its meshes differ slightly from 0.7.0's: every section below that reads dw3d meshes was measured again on them, case by case against 0.7.0, with the same scripts. **Exceptions**, said where they are printed: the *0.5.1* columns (0.5.1 is the last published version before 0.7.0; a 0.5.1 cell stands where 0.5.1 gave a different number from 0.7.0, and they were not measured again), the section "What 0.7.0 changes, against 0.5.1" (a record of 0.7.0, apart from its seed-ensemble paragraph and its benchmark anisotropy columns), the quadric reading's cost (measured before 0.5.0), and the timing paragraph (measured in early October 2026 on the code prepared for 0.6.0). 0.5.1 republished 0.5.0's code unchanged, so its numbers are 0.5.0's.

Short, honest comparison. Numbers below are measured, not estimated; each section states the population, the statistic and the configuration its numbers were measured on.

**Conventions shared by every section.** Unless a section says otherwise, the numbers come from the benchmarking dataset: 47 simulated cell aggregates of 2 to 10 cells, each with a ground-truth mesh, ground-truth tensions and a segmentation mask. 42 of the cases are at mechanical equilibrium and carry every accuracy claim below; the other 5 (`005`, `015`, `018`, `021`, `043`) are not, are kept as a negative control, and are never pooled with the 42. Every tension error is **gauge-matched**: only tension ratios are physical, so before any error is computed the inferred tensions are rescaled by the single factor `mean(truth) / mean(inferred)` taken over the interfaces the two share (`foambryo_benchmarks.metrics.gauge_matching_scale`). A relative tension error is `|γ̂ − γ| / |γ|`, per interface.

## Accuracy vs `dw3d`'s two meshing variants

Gauge-matched median relative tension error, 42 mechanically-equilibrated cases, meshed both ways with dw3d 0.7.1 and paired per case. `dithered` is `dw3d`'s default; `link_checked` is its opt-in "revised meshing" variant (see `dw3d`'s `BENCHMARKS.md`).

**How it was measured.** Each of the 42 equilibrium cases is reconstructed from its segmentation mask by `dw3d` in both configurations, at `min_distance = 3`, and each method infers tensions on each reconstruction. A case's error is the median, over the interfaces the reconstruction shares with the ground truth, of the gauge-matched relative tension error; the table gives the median of that per-case error over the 42 cases. Every cell is a median over 42 cases. Before dw3d 0.7.1, case 038's `link_checked` mesh had an interface without a triple line, whose tension was undetermined (`nan`), and the case was scored on its other interfaces: that interface was the membrane of a 26-voxel cell no tetrahedron lies inside, which 0.7.1 leaves out of the mesh (dw3d's `CHANGELOG.md`, 0.7.1). "Which is better" counts the cases in which the `dithered` mesh gives the lower per-case error (no case ties). The grid is produced by `foambryo_benchmarks/run_attribution.py` and summarised by `foambryo_benchmarks/analyze_attribution.py`; in its records `dithered` is named `v0_3` and `link_checked` `offset_excluded_linkcheck`, their names when the grid was first run.

| method | on `dithered` meshes | on `link_checked` meshes | which is better | 0.7.0 (`dithered` / `link_checked` / better) | 0.5.1 (`dithered` / `link_checked` / better) |
|---|---:|---:|---:|---:|---:|
| `YoungDupre` | **0.0763** | 0.1003 | `dithered`, 31/42 cases | 0.0777 / 0.0986 / 30 | 0.0777 / 0.0975 (41 cases) / 29 |
| `YoungDupreLocal` | **0.0911** | 0.1266 | `dithered`, 31/42 cases | 0.1039 / 0.1220 / 30 | 0.1039 / 0.1198 (41 cases) / 29 |
| `ForceBalanceKKT` | **0.1138** | 0.3262 | `dithered`, 41/42 cases | 0.1150 / 0.3327 / 38 | 0.1298 / 0.3357 / 39 |
| `BayesianMAP` (`ForceBalanceMAP`) | **0.1186** | 0.3149 | `dithered`, 40/42 cases | 0.1211 / 0.3265 / 39 | 0.1841 / 0.2010 / 25 |

0.7.1's meshes move the medians by up to 0.013 (`YoungDupreLocal` on `dithered` meshes, a jump between two ordered cases across a gap: its paired median change over the 42 cases is 0) and leave the conclusion as it was. Paired per case, 18 of the 42 `dithered` cases change for `YoungDupre`, by at most 0.036, and the median change is 0. Two cases change much more for the force-balance rows: 031 and 038, whose masks hold a cell that no tetrahedron lies inside (62 and 26 voxels), which 0.7.1 leaves out of the mesh instead of letting its seed grow a few spurious triangles; `ForceBalanceKKT`'s error on their `dithered` meshes falls from 0.208 to 0.122 and from 0.801 to 0.109 (against 0.7.1's meshes before that fix), `YoungDupre`'s from 0.122 to 0.102 on 038 only. The 0.5.1 column (and 0.7.0's when it was published) left case 038 out of the `link_checked` cells of the two Young-Dupré rows, which were then medians over 41 cases.

`dithered` wins on every method — this is why it is `dw3d`'s default. On the *ground-truth* mesh (no reconstruction error at all: the same statistic, with the simulator's own mesh of each case in place of the reconstruction), the same four methods reach 2×10⁻⁵ to 2×10⁻² (medians: `YoungDupre` 2.3×10⁻⁵, `YoungDupreLocal` 4.3×10⁻⁴, `BayesianMAP` 0.020, `ForceBalanceKKT` 0.021) — reconstruction error, not this package's inference, dominates the error budget at the 2–10-cell scale these 42 cases cover.

Between 0.5.1 and 0.7.0 the two force-balance rows changed for two reasons. `ForceBalanceMAP` no longer depends on the mesh's length unit or on `mean_tension` (`CHANGELOG.md`, 0.6.0), which changes its estimate on every mesh: it is now much better on `dithered` meshes and worse on `link_checked` ones. And `dw3d` 0.7.0 winds correctly the few triangles that lie on a flat tetrahedron (10 triangles in 7 of the 47 `dithered` meshes, 55 in 33 of the `link_checked` ones; no vertex moves; counted on 0.7.0's meshes). The force-balance operator reads cell volumes, so those cases change for `ForceBalanceKKT` and `ForceBalanceMAP`. Between 0.5.1 and 0.7.0 the angle-reading methods kept their medians; case by case they moved on one of the seven rewound `dithered` meshes only (case 032: `YoungDupre`'s error by −0.0015, `YoungDupreLocal`'s by −0.024), and on every other case they agreed with 0.5.1 to 1e-10.

## Which method to pick, and why mesh quality matters differently to each

`ForceBalanceKKT` (and `Variational`, the same operator) invert a global operator built from the mesh; their error tracks how far the mesh sits from mechanical equilibrium (`rho_free`, which `foambryo.compute_inference_diagnostics` computes from the mesh alone, without ground truth), measured at Spearman `r = 0.89` within a 72-mesh cohort. That cohort is 72 ground-truth simulator meshes of larger aggregates (91–166 labelled regions; four simulations, 18 optimisation stages each; no reconstruction step involved), and the correlation is between `ForceBalanceKKT`'s gauge-matched median relative tension error per mesh and that mesh's `rho_free` (`r = 0.894`, `p = 4.8×10⁻²⁶`, over a `rho_free` range of 0.037–0.490). It is a within-population result: across the 42 benchmark cases alone, whose `rho_free` spans only 0.0007–0.036, the same correlation is `r = 0.066`, and pooling the two populations hides it.

`YoungDupre` reads local contact angles only and is comparatively insensitive to this — its accuracy degrades far less between a good mesh and a poor one. Measured on the 42 cases with the statistic of the table above, between `dithered` meshes and the `dw3d` configuration whose reconstructions sit furthest from equilibrium (`get_junction_protected_mesh_reconstruction_algorithm` at its defaults; median `rho_free` 0.541 → 0.989), `YoungDupre`'s error rises from 0.076 to 0.097 (+27 %) while `ForceBalanceKKT`'s rises from 0.114 to 0.714 (+527 %; on 0.7.0, 0.078 to 0.099 and 0.115 to 0.713, with `rho_free` 0.551 → 0.989; on 0.5.1, 0.130 to 0.712, with `rho_free` 0.556 → 0.992); within the 72-mesh cohort, `YoungDupre`'s correlation with `rho_free` stays weak (`r = 0.252`). Practically: if you cannot control or verify your mesh's quality, `YoungDupre` is the more robust default; if you have reason to trust your mesh's geometry (a ground-truth simulation, or a mesh you have otherwise validated), the `ForceBalance*` family uses more of the mesh's information and can do better. See `TensionComputationMethod`'s docstring for the full angle-reading/operator-inverting split.

## Correctness fixes since the published version

Three defects in the published version are fixed and are worth knowing about if you compare old and new results on the same mesh:

- `TensionComputationMethod.Equilibrium` raised on every input; it now runs.
- `Cotan`/`InvCotan` had an operator-precedence bug that over-weighted short junctions; fixed (changes their output; neither was the default).
- On a mesh with a disconnected triple-junction graph (an engulfed cell, a sealed lumen, or simply large enough to fragment), eight of the nine tension methods used to silently drive some interfaces' tensions toward zero. Now solved per connected component with its own gauge; undetermined interfaces are reported as `nan` rather than guessed. Unaffected: any mesh whose triple-junction graph is already fully connected, the common case for small cell counts.

Since 0.5.1, `ForceBalanceMAP` no longer depends on the mesh's length unit or on the value of `mean_tension` (`CHANGELOG.md`); its numbers above changed with it, which is why they carry a 0.5.1 column.

**Speed** is a secondary concern at the sizes measured, but not a negligible one. On 72 converged simulator meshes of 91–166 cells (433–760 interfaces), the median wall time per inference is 1.01 s for `YoungDupre` (max 2.0 s), 1.02 s for `YoungDupreLocal` (max 2.1 s), 1.43 s for `Variational` (max 5.2 s), 3.16 s for `ForceBalanceKKT` (max 10.3 s) and 3.77 s for `BayesianMAP` (`ForceBalanceMAP`; max 12.4 s); the force-balance family grows with the interface count with a log-log slope of about 2.5, the Young-Dupré methods with about 1.3. On 0.5.1 the medians are the same within 2 % and the maxima within 7 %. These times were measured in early October 2026 on the code prepared for 0.6.0 (never published; its changes are part of 0.7.0), before 0.7.0's exact gauge, and were not measured again for 0.7.0. Measured single-threaded on one machine (Apple M3 Max, macOS 14.8.4) with a 1-minute load average between 1.2 and 3.9, one run per version, each method called as `foambryo_benchmarks/run_converged_cohort.py` calls it; before 0.5.0 the same paragraph read 0.43 s for `YoungDupre`, on a different day and code. The accuracy numbers above are what should drive a method choice; these tell you what it costs.

## The second-order (`quadric`) trijunction angle reading

**Opt-in, off by default.** `angle_reading="quadric"` on `infer_tensions` and its relatives replaces the chord each flat triangle reports with the tangent of a generic quadric fitted to the interface's own mesh vertices around the junction. See `README.md` for how to invoke it and `foambryo.second_order_angle_reading` for what enters the fit.

**Provenance.** The estimator is the mesh-patch quadric fit developed in a curvature-from-data study on `dw3d` reconstructions, ported into foambryo's production code. The port reproduces that study's numbers **bit-identically** on all 47 cases at both mesh spacings it was measured at (`min_distance` 3 and 4), and the default (`chord`) path is pinned bitwise against the code as it was before the feature was added by `tests/test_second_order_angle_reading.py`.

**Measurement.** 47 `dw3d`-reconstructed meshes of the benchmarking dataset (`dw3d` 0.7.1's default `dithered` configuration, seed 42, relocation on) at `min_distance` 3 and 4, written `md=3` and `md=4` below; 42 mechanically-equilibrated and 5 non-equilibrium cases reported separately and never pooled; case-level bootstrap (whole cases resampled with replacement), 10 000 draws, seed `20260901`, paired (the same resampled cases drive both readings on every draw), intervals in brackets are the 2.5–97.5 % percentiles. Every number is measured through foambryo's public API (`infer_tensions(..., angle_reading=...)`). The measurement scripts and per-case results are not distributed with this package.

### Junction-angle error, 42 equilibrium cases

Length-weighted regression of signed line-mean angle error on the line's true deviation from 120°. There is one row per wedge of each trijunction line: `y` is the wedge angle read on the reconstruction, averaged along the line, minus its true value (the Neumann angle implied by the ground-truth tensions), and `x` is that true angle minus 120°. The slope is a weighted least-squares fit with an intercept, pooled over all lines of the 42 cases, each row weighted by the junction length foambryo accumulates for it. A slope of 0 means no systematic pull toward 120°; a negative slope means angles are flattened toward 120°. Each reading is quoted against **its own** floor — a floor read with the chord reader does not bound a reading with the quadric one.

- **Matched-spacing floor**: the ground-truth reference mesh, registered to the reconstruction and coarsened until its median transverse lever — the distance from a junction triangle's apex to the midpoint of its junction edge — matches the reconstruction's (8.706 voxels at `md=3`, 11.213 at `md=4`), then read with the stated reader. It is what a geometrically perfect mesh at the same spacing reads.
- **Excess over own floor**: |slope| − |floor|, paired per bootstrap draw.
- **Shuffled-curvature null**: the same correction with each junction face's fitted curvature replaced by another face's, randomly permuted within the mesh (seed `20260902`). It should do nothing.

| | `md=3` | `md=4` | 0.7.0 (`md=3` / `md=4`) | 0.5.1, where it differed from 0.7.0 |
|---|---:|---:|---:|---:|
| slope, `chord` | −0.2118 [−0.2266, −0.1948] | −0.2169 [−0.2342, −0.1982] | −0.2135 / −0.2176 | − / −0.2174 |
| slope, **`quadric`** | **−0.0475** [−0.0643, −0.0300] | **−0.0051** [−0.0294, +0.0202] | −0.0496 / −0.0072 | −0.0498 / −0.0069 |
| slope, shuffled-curvature null | −0.2166 | −0.2255 | −0.2198 / −0.2244 | −0.2197 / −0.2246 |
| matched-spacing floor, `chord` | −0.1411 | −0.2022 | −0.1401 / −0.1963 | − |
| matched-spacing floor, `quadric` | +0.0363 | +0.0272 | +0.0365 / +0.0272 | − |
| excess over own floor, `chord` | +0.0707 [+0.0569, +0.0838] | +0.0147 [−0.0036, +0.0322] | +0.0733 / +0.0213 | +0.0734 / +0.0211 |
| excess over own floor, **`quadric`** | **+0.0112** [−0.0093, +0.0315] | **−0.0221** [−0.0295, +0.0053] | +0.0131 / −0.0200 | +0.0133 / −0.0203 |

Paired improvement in \|slope\| (the mean over bootstrap draws of \|slope, `quadric`\| − \|slope, `chord`\|, both recomputed on the same resampled cases): **−0.1643** [−0.1870, −0.1408] at `md=3` and **−0.2058** [−0.2279, −0.1791] at `md=4`, on **100 %** of bootstrap draws (0.7.0: −0.1640 and −0.2057; 0.5.1: −0.1638 and −0.2056). The null, measured the same way against `chord`, moves +0.0048 [−0.0010, +0.0105] and +0.0086 [−0.0018, +0.0192] (0.7.0: +0.0063 [−0.0002, +0.0129] and +0.0068 [−0.0037, +0.0173]; 0.5.1: +0.0062 and +0.0073): its intervals include zero at both spacings, so no effect of the null is resolved; both point estimates are less than a twentieth of the correction. The quadric reading's excess over its own floor is not resolved at either spacing: both intervals include zero, which bounds the excess (at most +0.0315 at `md=3` and +0.0053 at `md=4`) but does not show that it is zero. Since 0.7.1's enclosed-media fix the chord reading's excess interval at `md=4` includes zero too (before it: +0.0215 [+0.0001, +0.0449]); its point estimate, +0.0147, is of the same sign, and the shift is no evidence that the chord reading has reached its floor. The 0.5.1 values differed from 0.7.0's, in the fourth decimal, because `dw3d` 0.7.0 winds a few triangles differently and, at `md=4`, leaves one vertex of one case (038) where it was, a junction move its relocation guard refuses.

### End to end, every method that reads angles, 42 equilibrium cases

`contrast` is the gauge-matched OLS slope of inferred on true tension; **1.000 is perfect**. It is computed per case (ordinary least squares with an intercept over the interfaces shared with the ground truth, after gauge matching) for the cases with at least 4 matched interfaces — 36 of the 42 at both spacings, for every method — and the table gives its median over those cases. `median |c−1|` is the robust companion, the median over the same cases of \|contrast − 1\| (on meshes before 0.7.1's enclosed-media fix, case 038 was the outlier: its 26-voxel cell 7, which no tetrahedron lies inside, made a few spurious triangles, and its `YoungDupre` contrast at `md=4` was −1.67 under `chord` and −1.33 under `quadric`; it now reads 0.77 and 0.97. Case 031 at `md=4` lost two matched interfaces with its own spurious cell and fell below the 4 needed). The per-line angle error is the absolute value of the regression's `y` above; it is a property of the *reading* and is therefore the same for every method: **6.00° → 4.76°** at `md=3` and **7.03° → 5.46°** at `md=4` (medians over lines within each case, then medians over the 42 cases), p90 15.69° → 12.24° and 15.30° → 11.91° (each case's 90th percentile, median over cases). On 0.7.0: 6.00° → 4.73°, 7.03° → 5.46°, p90 16.11° → 12.24° and 16.98° → 11.98° (0.5.1 read 16.21° for the `md=4` chord p90).

| method | `md=3` contrast | median \|c−1\| | `md=4` contrast | median \|c−1\| |
|---|---|---|---|---|
| `YoungDupre` | 0.787 → **1.024** | 0.213 → **0.053** | 0.770 → **1.066** | 0.230 → **0.085** |
| `ProjectionYoungDupre` | 0.783 → **1.013** | 0.217 → **0.052** | 0.763 → **1.046** | 0.237 → **0.088** |
| `Lami` | 0.759 → **0.999** | 0.241 → **0.049** | 0.736 → **1.027** | 0.264 → **0.081** |
| `InvLami` | 0.752 → **0.993** | 0.248 → **0.054** | 0.752 → **1.017** | 0.248 → **0.070** |
| `LogLami` | 0.729 → **0.983** | 0.271 → **0.050** | 0.733 → **1.006** | 0.267 → **0.062** |
| `Cotan` | 0.755 → **0.997** | 0.245 → **0.056** | 0.755 → **1.026** | 0.245 → **0.083** |
| `InvCotan` | 0.758 → **1.021** | 0.242 → **0.049** | 0.757 → **1.052** | 0.243 → **0.081** |

On 0.7.0 the cells differed by at most 0.020 (`InvLami`'s `md=4` quadric \|c−1\|, 0.090; `YoungDupre`, `md=3`: 0.791 → 1.022, median \|c−1\| 0.209 → 0.058; at `md=4` its quadric \|c−1\| was 0.092). 0.5.1 differed from 0.7.0 in one cell: `YoungDupre`'s `md=4` chord contrast was 0.771. Contrast and per-line angle error improve **together**, which is what separates a geometry fix from a bias-for-variance trade. At `md=4` the quadric reading overshoots slightly (contrast 1.01–1.07).

**`YoungDupreLocal` is refused with the quadric reading** (`ValueError`, since 0.5.0). In the measurement that led to the refusal (on `dw3d` meshes without relocation), its relative RMS tension error (per case, the root-mean-square of the gauge-matched `γ̂ − γ` divided by the mean true tension) got *worse* under the quadric reading — `+0.0290` [`+0.0019`, `+0.0603`] at `md=3` and `+0.97` [`+0.035`, `+2.808`] at `md=4`, each the mean over the 42 cases of the paired per-case change — while its angles improved like everyone else's. It imposes the balance per trijunctional edge and so never performs the along-line average that cancels the correction's per-edge scatter. Prefer `YoungDupre`.

### Coverage, and where it does not apply

Pooled over the 42 equilibrium cases: the fraction of trijunction-incident faces whose quadric fit had enough samples to be applied, and the fraction of trijunction lines on which every incident face was corrected.

| | `md=3` | `md=4` |
|---|---:|---:|
| trijunction **faces** corrected | 0.932 | 0.913 |
| trijunction **lines** corrected throughout | **0.328** | **0.288** |

(0.7.0: 0.931 and 0.324 at `md=3`, 0.911 and 0.287 at `md=4`.)

The two differ by a factor of about three and both matter. Uncorrected faces keep the chord reading, so most *lines* carry a mixture of the two conventions. `AngleReadingReport.uncorrected_lines` names them; the report-less entry points emit an `AngleReadingCoverageWarning` instead.

### The regime where it does not help

On the 47 **ground-truth** simulation meshes — much finer than any reconstruction — the chord reading is already exact to within 2×10⁻⁵ of contrast and the quadric reading is **worse on 47 of 47 cases** (median contrast 1.081). Both figures are for `YoungDupre`, as medians over the 47 cases. The correction over-applies at fine spacing. Use it on reconstructions.

### Cost, and what it does not change

Measured before 0.5.0 and not yet repeated on 0.7.0: median wall time per case over the 42 equilibrium cases, the quadric fit is 0.024 s, paid once per mesh and cached on it; applying the rotations costs about as much again as the chord read on every `compute_angles_tri` call (+103 % at `md=3`, +116 % at `md=4`). Nothing else moves: the mesh is untouched, the mesh-validity check is identical under both readings (no abnormal non-manifold edge at `md=3` or `md=4`, 42/42 equilibrium meshes watertight; measured again on 0.7.1's meshes), and all 608 matched interfaces of the 42 equilibrium cases have identical areas under the two readings (areas recomputed from the mesh after each reading has run).

### A predicate that used to be missing

`foambryo_benchmarks.metrics.interface_area_fractions`, `interface_area_fraction_errors` and `closer_interface_area_fractions` compare interface areas **as a fraction of the mesh's own total**, which is scale-invariant and needs no registration. Interface-area error previously had no predicate in the benchmark suite; this is it. Measured on the reconstructions above (`dithered`, 42 equilibrium cases), against each case's ground-truth mesh: for each case the median, over the interfaces present in both meshes, of \|Δfraction\|, then the median over cases — **0.00045** at `md=3` and **0.00054** at `md=4` (0.7.0: 0.00046 and 0.00054). Interfaces present in only one of the two meshes are counted rather than scored: 5 and 4 invented and 9 and 11 missed interfaces over the 42 equilibrium cases (7 and 7 before 0.7.1's enclosed-media fix: the membranes of the cells of cases 031 and 038 that no tetrahedron lies inside, which earlier meshes covered with a few spurious triangles, are now missing).

## What 0.7.0 changes, against 0.5.1

Each paragraph names the population, the statistic and how 0.5.1 was compared. This section is a record of 0.7.0, measured on 0.7.0's meshes, except where a paragraph says it was measured again on 0.7.1's. Times were measured on one machine (Apple M3 Max, 16 cores, 128 GiB, macOS 14.8.4), one process at a time, with a 1-minute load average between 3 and 7, unless a paragraph says otherwise.

**The exact gauge: the scale each method returns.** Tensions are defined up to one factor; 0.7.0 meets `mean_tension` exactly in every connected piece of the junction graph instead of through a weighted extra row of the system. Every gauge-matched error above is unchanged by it. What changes is the scale a method returns: on the 42 equilibrium `dithered` reconstructions, the mean tension `YoungDupre` returns moved by a median 1.5e-6 relative to 0.5.1 (at most 0.19 %, on one case), `YoungDupreLocal`'s by a median 3.5e-5 (at most 3.4 %), and the force-balance family's not at all, except on case 032 (0.2 %, the rewound triangles above). `ForceBalanceMAP`'s moved on every case (median 91 %), because it no longer depends on the mesh's units. On a 1 866-cell tissue, the `Variational` (operator route) and `ForceBalance` tensions with the exact gauge differ from the old weighted row by one factor (0.987), the same for every interface to 2e-10.

**Non-finite coefficients.** On a few real meshes a triple line has a wedge of exactly 0 or 180 degrees (two films in one plane); the `Cotan`, `InvCotan`, `Lami` and `LogLami` systems then hold infinite coefficients, or rows about 1e15 times the others. 0.7.0 leaves such a line out of those four systems, with a warning, and every mesh without such a line gives bitwise the same numbers. None of the 47 benchmark meshes, the 72 converged simulator meshes or a 110-cell tissue has such a line; 6 of 484 frames of a real embryo series and a 1 866-cell tissue have (compared with the code just before the guard, not with 0.5.1).

**Large tissues.** A 1 866-cell tissue, from its mask to its tensions (the mask meshed on its native grid with the default reconstruction, the mesh scaled to the voxel size, `YoungDupre`), one run of 0.5.1 and two of 0.7.0, alternated:

| | 0.5.1 | 0.7.0 |
|---|---:|---:|
| whole run | 47.5 min | 9.5 min (both runs) |
| peak memory | 54.7 GiB | 30.9 GiB |
| reconstruction | 43.5 min | 9.2 min |
| — choosing the watershed's seeds | 27.9 min | 5.6 s |
| — moving the junctions (relocation and its guard) | 8.5 min | 2.2 min |
| — junction curves | 77 s | 50 s |
| tension solve | 226 s (dense) | 0.4 s (sparse) |

Both meshes have 93 097 points, 233 675 triangles and the same labels, but they are not the same mesh: 2 vertex positions differ (relocation's guard against inverting a cell, new in 0.6.0, refuses one move) and 166 triangles are wound the other way. Both refuse the pressures for the same reason (4 interfaces without a measurable tension). The tension and pressure systems are solved sparsely from 2 000 unknowns (`solver="auto"`). On synthetic Voronoi foams of 10 000 and 20 000 cells (70 555 and 142 017 interfaces), the `YoungDupre` tensions took 403 s and 1.7 GiB, and 602 s and 3.1 GiB, sparsely, and the pressures 1.5 to 6 s, measured on a shared machine (load 10 to 15), so these times are upper bounds; 0.5.1 solves only densely, which there needs an estimated 382 GiB and 1 549 GiB, more than the machine has. The methods with no sparse path are slow at this size: on the 1 866-cell tissue `Equilibrium` took about 1 hour, `Variational` about 70 minutes and `ForceBalance` about 110 minutes, each with a peak of 68 GiB.

**Seed ensembles.** `infer_tensions_over_seeds(mask)` (new in 0.7.0; eight seeds, the default) against one reconstruction at the default seed, which is what 0.5.1 returns, with the tension table's statistic, measured again on 0.7.1's meshes: on the 42 equilibrium cases the median error falls from 0.0763 to 0.0628, lower on 23 cases and higher on 19 (paired median change −3.0 %); on the 5 non-equilibrium cases from 0.0689 to 0.0320, lower on 4. (On 0.7.0's meshes: 0.0777 to 0.0638, 23 and 19, −4.9 %; 0.0689 to 0.0321.) Most of the error on these small aggregates is bias that every seed shares, which an ensemble cannot remove.

**Anisotropic voxels.** 0.5.1 could only mesh an anisotropic mask on its own grid and scale the mesh afterwards. 0.7.0 resamples it to cubic voxels first (`dcel_mesh_from_segmentation_mask(mask, spacing=...)`, and `dw3d`'s `construct_mesh_in_physical_units`). Measured on masks down-sampled along z by a factor `k` (each coarse voxel takes the label covering most of it), with the mask at full resolution as the control, one seed, the shipped code: the median over cases of each mesh's median absolute line-mean angle error, as the difference with the control, paired per case:

| `k` | 12 simulated tissues of 90 and 165 cells at real cell size (control 5.57°), 0.7.0's meshes | | 42 benchmark cases (control 6.00°): resampled on 0.7.1's meshes, native grid on 0.8.0's | |
|---|---:|---:|---:|---:|
| | resampled (0.7.0) | native grid (0.5.1's only route) | resampled | native grid |
| 5/3 | +0.83° | +1.38° | +0.43° | +1.59° |
| 2 | +0.71° | +1.97° | +1.45° | +2.25° |
| 3 | +1.17° | +4.10° | +1.27° | +5.91° |

The benchmark columns were measured again on 0.7.1's meshes (on 0.7.0's: +0.43°, +1.54°, +1.27° resampled and +1.43°, +2.43°, +5.58° native), and the native-grid column again on 0.8.0's, which pads thin margins of medium on the native grid (the resampled route is not padded by default). With the padding turned off, every case gives 0.7.1's number bit for bit; with it, no mask is padded at `k` = 5/3 and 2, and at `k` = 3, 18 of the 42 masks are padded and 17 of them change, which moves the cost from +5.85° (0.7.1) to +5.91° (per case −1.72° to +3.96°, median 0). The padding loses no cell and rescues none: case 038 loses its 26-voxel cell 7 with and without it at every `k`, and case 031 its cell 2 at `k` = 5/3 and 2. The tissue columns are 0.7.0's, measured before 0.8.0's padding: most of their meshes need more than 8 GiB to reconstruct and were not measured again; on the five of the six 90-cell tissues that were measured on 0.7.1, the costs move by at most 0.13°.

The resampled route keeps a valid mesh: on both populations and at every `k`, every mesh is watertight, with no nonlocal self-intersection and no degenerate triangle. Since 0.7.1 mesh surgery leaves no abnormal non-manifold edge on any route: none on the 294 benchmark meshes of the table (measured again), and none on the tissues' meshes, where 0.7.0 left one edge on 4 of 36 resampled meshes, one on one control mesh and 28 on 15 native-grid meshes (`dw3d`'s `CHANGELOG.md`, 0.7.1). On the tissues (0.7.0's meshes), the resampled route's trijunction lines are rougher than the control's (tangent error +1.1° to +1.8°, position +0.15 to +0.40 voxel) and smoother than the native grid's at `k` = 2 and 3 (tangent error −1.3° and −4.2°, lower on all 12 tissues). Above `k` = 3 nothing was measured, and a warning says so.

**The Laplace pressures.** 0.5.1's `Laplace` and `WeightedLaplace` pressures added `base_pressure` twice: every cell came out `base_pressure` too high. With the default `base_pressure=0`, no number changes.

## Two-dimensional tissues

**Every tissue in this section is synthetic.** No real two-dimensional image has known tensions or pressures, so no accuracy number below was measured on real data; the noise tiers are calibrated on real masks for boundary roughness only.

**Which code.** Measured during the development of 0.8.0, with foambryo's 2D inference and delaunay-watershed's 2D mesher (dw2d) at their 0.8.0 defaults: arc reading, junction relocation on, junctions on the image border excluded by the mesher's mark, ill-conditioned lenses refused. Two later changes came after these runs. dw2d's edges were reversed to follow the orientation rule shared with 3D, which changes no tension or pressure, bit for bit. Thin margins of medium are padded, which changes a mesh only where a cell lies within `2 * min_distance + 2` pixels (10 at the default 4) of an image side that no cell reaches; the decision is read on the mask, so it was evaluated on every image: of the 661 images of the comparison, it pads 11, all with 2 px of noise (8 at r = 20 px, 3 at r = 40 px), and of the 614 images with 2 px of noise of the global fit's held-out set it pads 41; it pads no image of any other tier, crop, Cellpose segmentation or large tissue. The comparison's noise tier was measured again with 0.8.0's code: without the padding it gives the earlier numbers bit for bit on all 318 images, and with it 2 images change (both at 2 px and r = 40 px), which moves one number below, the paired pressure difference against the CellFIT equations at 2 px and r = 40 px, from −0.084 to −0.087 (its interval unchanged). The arc-against-chord table and the global fit's 2 px rows were measured again in the same way, on the tissues of the padded images, with the padding off as the control: the arc readings' tensions and angles of the first table's runs and every record of the global fit's runs came out bit for bit, so an unpadded image keeps its numbers. In the comparison set the same 2 images change, and the arc-against-chord table does not move (tension −0.0603, pressure −0.0853, angle −3.47° with relocation off; −0.0682, −0.0908, −4.23° with it). In the global fit's set 6 of the 41 padded images change; its 2 px rows move by less than their printed digits (the paired difference at r = 40 px from −0.0596 to −0.0595, its interval unchanged). The research code behind the arc-against-chord table scores pressures with 0.1.5's edge orientation, which 0.8.0 reversed: its pressures were therefore scored on the edges reversed back, and with them every arc reading reproduced its earlier records bit for bit, pressures included.

**What is measured.** 53 simulated two-dimensional foams at mechanical equilibrium, 4 to 64 cells, with known tensions, pressures and junction angles, drawn as label masks with a mean cell radius r of 10, 20 or 40 px, in five tiers. *exact*: the masks of the true circular-arc interfaces; *crop*: the same masks cut by a straight line (26 crops per scale); *noise*: each mask warped by a smooth random displacement field (correlation length 4 px, RMS displacement a = 0.5, 1 or 2 px); *calibrated*: the warp at a = 0.75 px, the level whose boundary roughness matches the median of seven real two-dimensional segmentations from three sources (per mask 0.52, 0.66, 0.67, beyond 2 and beyond 2 px; the upper quartile lies beyond 2 px, which is why the 2 px tier is kept). Every image is meshed by dw2d and then read by foambryo, so the numbers include the mesh's own error: they are reconstructions end to end, not readings on exact geometry. None of these foams was used to choose the reading: the choice was made on 17 other foams (211 images) and fixed before these were read.

**Statistics.** Tension error: |γ̂ − γ| / γ per interface between two cells, after rescaling the inferred tensions to the true mean; pressure error: |p̂ − p| over the root-mean-square true pressure; angle error: the mean of a junction's three absolute wedge errors, in degrees. Each is the median over the image's interfaces, cells or junctions, then the median over the images of a tier, with a 95 % bootstrap interval over the foams. Differences are paired per image, on the interfaces both methods score, with a 95 % bootstrap interval over the foams (10 000 draws); a difference is "not resolved" when its interval contains 0.

### The default reading: an arc per interface

foambryo 0.8.0 reads each interface of the 2D mesh as a circular arc (the geometric least-squares circle through the interface's mesh vertices, its tangent taken at the circle's point nearest the junction; an interface with fewer than 4 vertices is read by its chord). Against the chord reading of earlier versions (relocation off), over all 661 images:

| change | tension error | pressure error | angle error (°) |
|---|---|---|---|
| arc reading, relocation off | −0.060 [−0.067, −0.054] | −0.085 [−0.091, −0.081] | −3.5 [−3.8, −3.0] |
| arc reading, relocation on (the default) | −0.068 [−0.075, −0.063] | −0.091 [−0.098, −0.085] | −4.2 [−4.5, −4.0] |

No tier is significantly worse under the default. An interface with fewer than 4 mesh vertices (or whose circle fit fails) is read by its chord from end to end and gives no Laplace equation: 68 % of the interfaces at r = 10 px, where the mesh's vertices are about 5 px apart, 11 % at r = 20 px and 1 % at r = 40 px (medians over images). How much of the gain at r = 10 px comes from that chord rather than from the arcs is not measured.

### Against CellFIT and pycellfit

CellFIT (G. W. Brodland, J. H. Veldhuis, S. Kim, M. Perrone, D. Mashburn and M. S. Hutson, *PLoS ONE* 9(6): e99116, 2014, doi:10.1371/journal.pone.0099116) fits a circular arc to each interface of the segmented image and solves junction force balance and Laplace's law. Two implementations were run on the same images, from the masks: **the CellFIT equations**, re-implemented from the paper (this is not the authors' software), and **pycellfit** 0.2.3, the public implementation, which reads interfaces as chords and is not deterministic (one 1 024-cell image failed in two of three runs). pycellfit failed to build on 7, 4 and 1 of the 26 crops at r = 10, 20 and 40 px, and on 1 to 3 of the 53 images at 1 and 2 px of noise; foambryo and the CellFIT equations built on every image.

Medians per tier (tension error T, pressure error P, angle error A in degrees):

| tier | r (px) | T foambryo 0.8.0 | T pycellfit | T CellFIT equations | P foambryo 0.8.0 | P CellFIT equations | A foambryo 0.8.0 | A CellFIT equations |
|---|---|---|---|---|---|---|---|---|
| exact | 10 | 0.134 [0.115, 0.141] | 0.341 | 0.094 [0.074, 0.112] | 0.165 | 0.123 | 8.07 | 5.05 |
| exact | 20 | 0.070 [0.061, 0.088] | 0.350 | 0.037 [0.028, 0.043] | 0.060 | 0.064 | 3.77 | 1.81 |
| exact | 40 | 0.040 [0.030, 0.050] | 0.343 | 0.013 [0.010, 0.015] | 0.031 | 0.019 | 1.59 | 0.57 |
| calibrated 0.75 px | 20 | 0.092 [0.079, 0.126] | 0.349 | 0.120 [0.097, 0.133] | 0.089 | 0.143 | 6.03 | 6.97 |
| calibrated 0.75 px | 40 | 0.064 [0.054, 0.074] | 0.356 | 0.061 [0.054, 0.068] | 0.050 | 0.085 | 3.37 | 3.62 |
| noise 0.5 px | 20 | 0.094 [0.078, 0.116] | 0.348 | 0.072 [0.062, 0.084] | 0.087 | 0.095 | 4.76 | 4.73 |
| noise 0.5 px | 40 | 0.061 [0.043, 0.072] | 0.345 | 0.041 [0.033, 0.053] | 0.042 | 0.060 | 2.70 | 2.50 |
| noise 1 px | 20 | 0.113 [0.101, 0.143] | 0.347 | 0.147 [0.114, 0.155] | 0.103 | 0.162 | 7.52 | 9.17 |
| noise 1 px | 40 | 0.074 [0.059, 0.092] | 0.343 | 0.083 [0.070, 0.102] | 0.058 | 0.123 | 4.45 | 5.00 |
| noise 2 px | 20 | 0.216 [0.198, 0.240] | 0.337 | 0.284 [0.255, 0.331] | 0.170 | 0.310 | 13.57 | 17.38 |
| noise 2 px | 40 | 0.152 [0.118, 0.175] | 0.340 | 0.169 [0.149, 0.185] | 0.114 | 0.208 | 8.29 | 9.55 |
| crop | 10 | 0.162 [0.152, 0.193] | 0.376 | 0.159 [0.129, 0.196] | 0.222 | 0.209 | 8.65 | 6.01 |
| crop | 20 | 0.132 [0.112, 0.147] | 0.365 | 0.076 [0.054, 0.089] | 0.103 | 0.102 | 4.41 | 2.18 |
| crop | 40 | 0.070 [0.063, 0.082] | 0.388 | 0.028 [0.019, 0.033] | 0.053 | 0.035 | 2.15 | 0.73 |

Paired differences, foambryo 0.8.0 minus the other tool (negative: foambryo better):

| tier | r (px) | tension, vs pycellfit | tension, vs CellFIT equations | pressure, vs CellFIT equations | angle (°), vs CellFIT equations |
|---|---|---|---|---|---|
| exact | 10 / 20 / 40 | −0.193 / −0.262 / −0.297 | +0.033 [+0.013, +0.043] / +0.025 [+0.015, +0.043] / +0.026 [+0.017, +0.032] | +0.032 / +0.001 (not resolved) / +0.008 | +2.86 / +1.98 / +1.03 |
| calibrated 0.75 px | 20 / 40 | −0.215 / −0.278 | −0.002 [−0.009, +0.010] / −0.001 [−0.006, +0.004] (not resolved) | −0.037 [−0.051, −0.022] / −0.026 [−0.040, −0.018] | −0.95 / −0.20 |
| noise 0.5 px | 20 / 40 | −0.233 / −0.274 | +0.012 [+0.000, +0.034] / +0.013 [+0.008, +0.023] | −0.009 (not resolved) / −0.013 | −0.03 (not resolved) / +0.17 |
| noise 1 px | 20 / 40 | −0.198 / −0.249 | −0.013 [−0.027, −0.000] / −0.006 [−0.008, −0.002] | −0.046 / −0.044 | −1.28 / −0.37 |
| noise 2 px | 20 / 40 | −0.099 / −0.175 | −0.071 [−0.087, −0.045] / −0.021 [−0.035, −0.007] | −0.095 / −0.087 | −4.07 / −1.39 |
| crop | 10 / 20 / 40 | −0.195 / −0.246 / −0.308 | +0.016 [−0.025, +0.056] (not resolved) / +0.039 [+0.030, +0.082] / +0.042 [+0.032, +0.051] | +0.027 (not resolved) / +0.003 (not resolved) / +0.023 | +2.56 / +2.30 / +1.31 |

Every difference against pycellfit favours foambryo, with its interval below zero. Against the CellFIT equations: on exact masks and crops the equations are better, by 0.02 to 0.04 in tension error; at the noise measured on real masks (0.75 px) the tensions are tied and foambryo's pressures and angles are better; from 1 px of noise foambryo is better on every metric. Exact circular arcs are exactly the model the CellFIT equations fit, and they fit each arc to every pixel crack of the mask's boundary with the junctions on the mask's own corners, where foambryo reads the mesher's vertices, about 5 px apart at the default `min_distance` of 4. Under noise that sparser sampling smooths the boundary. A fit to the boundary cracks inside foambryo's reading closes the gap on exact masks (−0.000 [−0.001, +0.000] at r = 40 px) but is worse under 2 px of noise (+0.034 [+0.021, +0.055] in tension error at r = 20 px), so the default keeps the mesh's vertices: foambryo reads the mesh, not the mask. Given the same geometry, foambryo's inference matches or beats the CellFIT equations from r = 20 px; their remaining lead end to end on clean masks comes from their boundary extraction.

### The opt-in global fit (`infer_forces_by_equilibrium_tiling`)

The global fit adjusts one equilibrium tiling of circular arcs (every interface a circle, Young's and Laplace's laws exact; Noll, Streichan and Shraiman, *Phys. Rev. X* 10, 011072, 2020) to the label image's crack points, starting from the arc reading of dw2d's mesh. It was evaluated on a second held-out set, built and sealed before any fit: 307 synthetic equilibrium foams of 4 to 64 cells, with the tiers above, Cellpose segmentations of rendered images of the foams, and an off-equilibrium tier whose membranes are displaced smoothly by 0.75 px. Tension error (statistics as above), global fit against the arc reading, paired:

| tier | r (px) | global fit | arc reading | difference [95 %] |
|---|---|---|---|---|
| exact | 10 / 20 / 40 | 0.034 / 0.011 / 0.004 | 0.129 / 0.077 / 0.042 | −0.092 [−0.098, −0.086] / −0.064 [−0.068, −0.060] / −0.038 [−0.043, −0.032] |
| Cellpose segmentations | 10 / 20 / 40 | 0.111 / 0.043 / 0.024 | 0.227 / 0.119 / 0.074 | −0.111 [−0.117, −0.101] / −0.077 [−0.082, −0.072] / −0.051 [−0.054, −0.047] |
| calibrated 0.75 px | 20 / 40 | 0.070 / 0.029 | 0.106 / 0.065 | −0.035 [−0.040, −0.026] / −0.035 [−0.038, −0.030] |
| noise 0.5 px | 20 / 40 | 0.047 / 0.020 | 0.086 / 0.052 | −0.039 [−0.044, −0.033] / −0.031 [−0.035, −0.027] |
| noise 1 px | 20 / 40 | 0.097 / 0.039 | 0.126 / 0.076 | −0.030 [−0.034, −0.023] / −0.038 [−0.040, −0.034] |
| noise 2 px | 20 / 40 | 0.207 / 0.082 | 0.214 / 0.142 | −0.005 [−0.014, +0.005] / −0.060 [−0.067, −0.050] |
| crop | 10 / 20 / 40 | 0.055 / 0.020 / 0.007 | 0.163 / 0.128 / 0.082 | −0.107 [−0.117, −0.102] / −0.104 [−0.115, −0.093] / −0.072 [−0.081, −0.067] |
| off equilibrium, 0.75 px | 20 / 40 | 0.073 / 0.036 | 0.134 / 0.085 | −0.056 [−0.064, −0.052] / −0.047 [−0.050, −0.044] |

On larger synthetic tissues: 4 096 cells at r = 10 px, 0.077 against 0.156 for the arc reading, every one of the 12 285 interfaces kept, in 74 s and 0.57 GiB (dw2d's mesh, separately, 14 to 20 s in two measurements on a shared machine); 1 024 cells, 0.068 against 0.161 at r = 10 px and 0.023 against 0.248 at r = 20 px. Each large tissue is one draw.

**Why it is opt-in.** It refuses whole images where the arc reading refuses none: 11 of 453 crops (the arc reading: 0), and 4 more images than the arc reading on each of the Cellpose and the off-equilibrium tiers (4-cell tissues). A refusal is an honest `nan` with its reason in the report, not a wrong number. The fit assumes that every interface is a circular arc and that Laplace's law holds, so it is not for confined tissues (an embryo in its eggshell, cells on a substrate); it needs the label image as well as the mesh.

### Ill-conditioned lenses

On 70 synthetic tissues (the tiers and crops above at r = 10, 20 and 40 px, and the calibrated 0.75 px tier) and dw2d's five real masks, at `min_distance` 4 and 3, read with the arc and the chord readings (1 761 meshes, 3 522 readings), the 2D tension system was singular in 2 readings, both a crop at r = 10 px in which a lens (a cell with two sides) has a single junction on the crop's border; without the guard they returned two opposite tensions of about 1e11 to 1e12, without a warning. With the guard (`MAX_LENS_CONDITION = 20`) no reading keeps a component condition number above 1e6 (the largest is 214); the guard fires on 90 readings (2.6 %), almost all crops, where the median tension error moves by −0.001, and the 3 432 other readings are unchanged, bit for bit.

### What these numbers do not show

No real two-dimensional image has ground truth here: the calibrated tier matches real masks in boundary roughness only, measured on seven masks. The foams have 4 to 64 cells, apart from the large tissues of the global fit. The off-equilibrium tier displaces membranes smoothly; it is not a model of non-equilibrium mechanics. None of the tissues is confined: nothing here measures the readings inside an eggshell or on a substrate, where the cells are compressed and only the tensions read near the junctions (`angle_reading="chord"`, with `"fitted_tangent"` as a check) and the deviatoric stress, as the cortical tensions' contribution, hold (see the user guide's page on two-dimensional tissues). Run times against pycellfit were not measured with 0.8.0's defaults.

## Package renamed `benchmarks` -> `foambryo_benchmarks`

The top-level `benchmarks/` package was renamed to `foambryo_benchmarks/` because both this repository and `delaunay-watershed-3d` shipped a package named `benchmarks`, and a script that put both repositories on `sys.path` could silently resolve the name to either one.

**Migration hazard:** pulling this rename into an existing checkout leaves a stray `benchmarks/__pycache__/` directory on disk — `git mv` moves tracked files, not gitignored build artifacts. With `benchmarks/__init__.py` gone but the directory still present, Python imports it as an empty implicit namespace package (PEP 420): `import benchmarks` still succeeds, with no `__file__` and no real submodules. `tests/test_no_ambiguous_benchmarks_package.py` now tolerates this case (only a *real*, `__file__`-bearing `benchmarks` package fails it), but the stray directory should still be deleted on any checkout that hits it — `rm -rf benchmarks/` once `git status` confirms it holds nothing tracked.
