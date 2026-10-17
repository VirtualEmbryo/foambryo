# Benchmarks: which tension method to use, and how mesh quality affects it

> **Measured on 0.7.0, on meshes from delaunay-watershed-3d (`dw3d`) 0.7.0 with its defaults**, junction relocation included. Where 0.5.1, the last published version before 0.7.0, gives a different number, a *0.5.1* column stands beside it, compared case by case; everywhere else the two versions give the same number. How the 0.7.0 numbers were obtained: the tension table below was re-run on 0.7.0 and compared with 0.5.1 case by case; for the other sections, 0.7.0's default meshes are identical, bit for bit, to those the sections were measured on during 0.7.0's development (47 cases at each of the five spacings used), and 0.7.0's inference gives the same tension ratios on them, to 1e-12, for every method and both angle readings. 0.5.1 republished 0.5.0's code unchanged, so its numbers are 0.5.0's. **Exceptions**, said where they are printed: the quadric reading's cost (measured before 0.5.0) and the timing paragraph (measured in early October 2026 on the code prepared for 0.6.0, before 0.7.0's exact gauge). Neither was measured again for 0.7.0.

Short, honest comparison. Numbers below are measured, not estimated; each section states the population, the statistic and the configuration its numbers were measured on.

**Conventions shared by every section.** Unless a section says otherwise, the numbers come from the benchmarking dataset: 47 simulated cell aggregates of 2 to 10 cells, each with a ground-truth mesh, ground-truth tensions and a segmentation mask. 42 of the cases are at mechanical equilibrium and carry every accuracy claim below; the other 5 (`005`, `015`, `018`, `021`, `043`) are not, are kept as a negative control, and are never pooled with the 42. Every tension error is **gauge-matched**: only tension ratios are physical, so before any error is computed the inferred tensions are rescaled by the single factor `mean(truth) / mean(inferred)` taken over the interfaces the two share (`foambryo_benchmarks.metrics.gauge_matching_scale`). A relative tension error is `|γ̂ − γ| / |γ|`, per interface.

## Accuracy vs `dw3d`'s two meshing variants

Gauge-matched median relative tension error, 42 mechanically-equilibrated cases, meshed both ways and paired per case. `dithered` is `dw3d`'s default; `link_checked` is its opt-in "revised meshing" variant (see `dw3d`'s `BENCHMARKS.md`).

**How it was measured.** Each of the 42 equilibrium cases is reconstructed from its segmentation mask by `dw3d` in both configurations, at `min_distance = 3`, and each method infers tensions on each reconstruction. A case's error is the median, over the interfaces the reconstruction shares with the ground truth, of the gauge-matched relative tension error; the table gives the median of that per-case error over the 42 cases. "Which is better" counts the cases in which the `dithered` mesh gives the lower per-case error (no case ties). The grid is produced by `foambryo_benchmarks/run_attribution.py` and summarised by `foambryo_benchmarks/analyze_attribution.py`; in its records `dithered` is named `v0_3` and `link_checked` `offset_excluded_linkcheck`, their names when the grid was first run.

| method | on `dithered` meshes | on `link_checked` meshes | which is better | 0.5.1, where different (`dithered` / `link_checked` / better) |
|---|---:|---:|---:|---:|
| `YoungDupre` | **0.0777** | 0.0975 | `dithered`, 29/42 cases | same |
| `YoungDupreLocal` | **0.1039** | 0.1198 | `dithered`, 29/42 cases | same |
| `ForceBalanceKKT` | **0.1150** | 0.3327 | `dithered`, 38/42 cases | 0.1298 / 0.3357 / 39/42 |
| `BayesianMAP` (`ForceBalanceMAP`) | **0.1211** | 0.3265 | `dithered`, 39/42 cases | 0.1841 / 0.2010 / 25/42 |

`dithered` wins on every method — this is why it is `dw3d`'s default. On the *ground-truth* mesh (no reconstruction error at all: the same statistic, with the simulator's own mesh of each case in place of the reconstruction), the same four methods reach 2×10⁻⁵ to 2×10⁻² (medians: `YoungDupre` 2.3×10⁻⁵, `YoungDupreLocal` 4.3×10⁻⁴, `BayesianMAP` 0.020, `ForceBalanceKKT` 0.021) — reconstruction error, not this package's inference, dominates the error budget at the 2–10-cell scale these 42 cases cover.

The two force-balance rows differ from 0.5.1 for two reasons. `ForceBalanceMAP` no longer depends on the mesh's length unit or on `mean_tension` (`CHANGELOG.md`, 0.6.0), which changes its estimate on every mesh: it is now much better on `dithered` meshes and worse on `link_checked` ones. And `dw3d` now winds correctly the few triangles that lie on a flat tetrahedron (10 triangles in 7 of the 47 `dithered` meshes, 55 in 33 of the `link_checked` ones; no vertex moves). The force-balance operator reads cell volumes, so those cases change for `ForceBalanceKKT` and `ForceBalanceMAP`. The angle-reading methods keep their medians; case by case they move on one of the seven rewound `dithered` meshes only (case 032: `YoungDupre`'s error by −0.0015, `YoungDupreLocal`'s by −0.024), and on every other case they agree with 0.5.1 to 1e-10.

## Which method to pick, and why mesh quality matters differently to each

`ForceBalanceKKT` (and `Variational`, the same operator) invert a global operator built from the mesh; their error tracks how far the mesh sits from mechanical equilibrium (`rho_free`, which `foambryo.compute_inference_diagnostics` computes from the mesh alone, without ground truth), measured at Spearman `r = 0.89` within a 72-mesh cohort. That cohort is 72 ground-truth simulator meshes of larger aggregates (91–166 labelled regions; four simulations, 18 optimisation stages each; no reconstruction step involved), and the correlation is between `ForceBalanceKKT`'s gauge-matched median relative tension error per mesh and that mesh's `rho_free` (`r = 0.894`, `p = 4.8×10⁻²⁶`, over a `rho_free` range of 0.037–0.490). It is a within-population result: across the 42 benchmark cases alone, whose `rho_free` spans only 0.0007–0.036, the same correlation is `r = 0.066`, and pooling the two populations hides it.

`YoungDupre` reads local contact angles only and is comparatively insensitive to this — its accuracy degrades far less between a good mesh and a poor one. Measured on the 42 cases with the statistic of the table above, between `dithered` meshes and the `dw3d` configuration whose reconstructions sit furthest from equilibrium (`get_junction_protected_mesh_reconstruction_algorithm` at its defaults; median `rho_free` 0.551 → 0.989), `YoungDupre`'s error rises from 0.078 to 0.099 (+28 %) while `ForceBalanceKKT`'s rises from 0.115 to 0.713 (+520 %; on 0.5.1, 0.130 to 0.712, with `rho_free` 0.556 → 0.992); within the 72-mesh cohort, `YoungDupre`'s correlation with `rho_free` stays weak (`r = 0.252`). Practically: if you cannot control or verify your mesh's quality, `YoungDupre` is the more robust default; if you have reason to trust your mesh's geometry (a ground-truth simulation, or a mesh you have otherwise validated), the `ForceBalance*` family uses more of the mesh's information and can do better. See `TensionComputationMethod`'s docstring for the full angle-reading/operator-inverting split.

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

**Measurement.** 47 `dw3d`-reconstructed meshes of the benchmarking dataset (`dw3d`'s default `dithered` configuration, seed 42, relocation on) at `min_distance` 3 and 4, written `md=3` and `md=4` below; 42 mechanically-equilibrated and 5 non-equilibrium cases reported separately and never pooled; case-level bootstrap (whole cases resampled with replacement), 10 000 draws, seed `20260901`, paired (the same resampled cases drive both readings on every draw), intervals in brackets are the 2.5–97.5 % percentiles. Every number is measured through foambryo's public API (`infer_tensions(..., angle_reading=...)`). The measurement scripts and per-case results are not distributed with this package.

### Junction-angle error, 42 equilibrium cases

Length-weighted regression of signed line-mean angle error on the line's true deviation from 120°. There is one row per wedge of each trijunction line: `y` is the wedge angle read on the reconstruction, averaged along the line, minus its true value (the Neumann angle implied by the ground-truth tensions), and `x` is that true angle minus 120°. The slope is a weighted least-squares fit with an intercept, pooled over all lines of the 42 cases, each row weighted by the junction length foambryo accumulates for it. A slope of 0 means no systematic pull toward 120°; a negative slope means angles are flattened toward 120°. Each reading is quoted against **its own** floor — a floor read with the chord reader does not bound a reading with the quadric one.

- **Matched-spacing floor**: the ground-truth reference mesh, registered to the reconstruction and coarsened until its median transverse lever — the distance from a junction triangle's apex to the midpoint of its junction edge — matches the reconstruction's (8.722 voxels at `md=3`, 11.219 at `md=4`), then read with the stated reader. It is what a geometrically perfect mesh at the same spacing reads.
- **Excess over own floor**: |slope| − |floor|, paired per bootstrap draw.
- **Shuffled-curvature null**: the same correction with each junction face's fitted curvature replaced by another face's, randomly permuted within the mesh (seed `20260902`). It should do nothing.

| | `md=3` | `md=4` | 0.5.1, where different (`md=3` / `md=4`) |
|---|---:|---:|---:|
| slope, `chord` | −0.2135 [−0.2276, −0.1972] | −0.2176 [−0.2367, −0.1975] | same / −0.2174 |
| slope, **`quadric`** | **−0.0496** [−0.0663, −0.0323] | **−0.0072** [−0.0320, +0.0185] | −0.0498 / −0.0069 |
| slope, shuffled-curvature null | −0.2198 | −0.2244 | −0.2197 / −0.2246 |
| matched-spacing floor, `chord` | −0.1401 | −0.1963 | same |
| matched-spacing floor, `quadric` | +0.0365 | +0.0272 | same |
| excess over own floor, `chord` | +0.0733 [+0.0597, +0.0863] | +0.0213 [−0.0000, +0.0446] | +0.0734 / +0.0211 |
| excess over own floor, **`quadric`** | **+0.0131** [−0.0074, +0.0337] | **−0.0200** [−0.0295, +0.0078] | +0.0133 / −0.0203 |

Paired improvement in \|slope\| (the mean over bootstrap draws of \|slope, `quadric`\| − \|slope, `chord`\|, both recomputed on the same resampled cases): **−0.1640** [−0.1866, −0.1404] at `md=3` and **−0.2057** [−0.2290, −0.1779] at `md=4`, on **100 %** of bootstrap draws ( −0.1638 and −0.2056 on 0.5.1). The null, measured the same way against `chord`, moves +0.0063 [−0.0002, +0.0129] and +0.0068 [−0.0037, +0.0173] (0.5.1: +0.0062 and +0.0073): slightly the wrong way, by less than a twentieth of the correction, and both intervals include zero. On these meshes the quadric reading sits on its own floor at both spacings: its excess intervals include zero. The 0.5.1 values differ, in the fourth decimal, because `dw3d` now winds a few triangles differently and, at `md=4`, leaves one vertex of one case (038) where it was, a junction move its relocation guard now refuses.

### End to end, every method that reads angles, 42 equilibrium cases

`contrast` is the gauge-matched OLS slope of inferred on true tension; **1.000 is perfect**. It is computed per case (ordinary least squares with an intercept over the interfaces shared with the ground truth, after gauge matching) for the cases with at least 4 matched interfaces — 36 of the 42 at `md=3` and 37 at `md=4`, for every method — and the table gives its median over those cases. `median |c−1|` is the robust companion, the median over the same cases of \|contrast − 1\| (one case, 038, reaches \|contrast\| > 3 under *both* readings and dominates any squared statistic). The per-line angle error is the absolute value of the regression's `y` above; it is a property of the *reading* and is therefore the same for every method: **6.00° → 4.73°** at `md=3` and **7.03° → 5.46°** at `md=4` (medians over lines within each case, then medians over the 42 cases), p90 16.11° → 12.24° and 16.98° → 11.98° (each case's 90th percentile, median over cases; 0.5.1 reads 16.21° for the `md=4` chord p90).

| method | `md=3` contrast | median \|c−1\| | `md=4` contrast | median \|c−1\| |
|---|---|---|---|---|
| `YoungDupre` | 0.791 → **1.022** | 0.209 → **0.058** | 0.766 → **1.059** | 0.234 → **0.092** |
| `ProjectionYoungDupre` | 0.786 → **1.013** | 0.214 → **0.056** | 0.761 → **1.041** | 0.239 → **0.091** |
| `Lami` | 0.759 → **0.999** | 0.241 → **0.056** | 0.730 → **1.026** | 0.270 → **0.083** |
| `InvLami` | 0.753 → **0.993** | 0.247 → **0.063** | 0.752 → **1.013** | 0.248 → **0.090** |
| `LogLami` | 0.722 → **0.982** | 0.278 → **0.055** | 0.732 → **1.005** | 0.268 → **0.065** |
| `Cotan` | 0.756 → **0.997** | 0.244 → **0.056** | 0.752 → **1.022** | 0.248 → **0.085** |
| `InvCotan` | 0.755 → **1.019** | 0.245 → **0.049** | 0.751 → **1.052** | 0.249 → **0.088** |

0.5.1 differs in one cell: `YoungDupre`'s `md=4` chord contrast is 0.771. Contrast and per-line angle error improve **together**, which is what separates a geometry fix from a bias-for-variance trade. At `md=4` the quadric reading overshoots slightly (contrast 1.01–1.06).

**`YoungDupreLocal` is refused with the quadric reading** (`ValueError`, since 0.5.0). In the measurement that led to the refusal (on `dw3d` meshes without relocation), its relative RMS tension error (per case, the root-mean-square of the gauge-matched `γ̂ − γ` divided by the mean true tension) got *worse* under the quadric reading — `+0.0290` [`+0.0019`, `+0.0603`] at `md=3` and `+0.97` [`+0.035`, `+2.808`] at `md=4`, each the mean over the 42 cases of the paired per-case change — while its angles improved like everyone else's. It imposes the balance per trijunctional edge and so never performs the along-line average that cancels the correction's per-edge scatter. Prefer `YoungDupre`.

### Coverage, and where it does not apply

Pooled over the 42 equilibrium cases: the fraction of trijunction-incident faces whose quadric fit had enough samples to be applied, and the fraction of trijunction lines on which every incident face was corrected.

| | `md=3` | `md=4` |
|---|---:|---:|
| trijunction **faces** corrected | 0.931 | 0.911 |
| trijunction **lines** corrected throughout | **0.324** | **0.287** |

The two differ by a factor of about three and both matter. Uncorrected faces keep the chord reading, so most *lines* carry a mixture of the two conventions. `AngleReadingReport.uncorrected_lines` names them; the report-less entry points emit an `AngleReadingCoverageWarning` instead.

### The regime where it does not help

On the 47 **ground-truth** simulation meshes — much finer than any reconstruction — the chord reading is already exact to within 2×10⁻⁵ of contrast and the quadric reading is **worse on 47 of 47 cases** (median contrast 1.081). Both figures are for `YoungDupre`, as medians over the 47 cases. The correction over-applies at fine spacing. Use it on reconstructions.

### Cost, and what it does not change

Measured before 0.5.0 and not yet repeated on 0.7.0: median wall time per case over the 42 equilibrium cases, the quadric fit is 0.024 s, paid once per mesh and cached on it; applying the rotations costs about as much again as the chord read on every `compute_angles_tri` call (+103 % at `md=3`, +116 % at `md=4`). Nothing else moves: the mesh is untouched, the mesh-validity check is identical under both readings (no abnormal non-manifold edge at `md=3` or `md=4`, 42/42 equilibrium meshes watertight), and all 608 matched interfaces of the 42 equilibrium cases have identical areas under the two readings (areas recomputed from the mesh after each reading has run).

### A predicate that used to be missing

`foambryo_benchmarks.metrics.interface_area_fractions`, `interface_area_fraction_errors` and `closer_interface_area_fractions` compare interface areas **as a fraction of the mesh's own total**, which is scale-invariant and needs no registration. Interface-area error previously had no predicate in the benchmark suite; this is it. Measured on the reconstructions above (`dithered`, 42 equilibrium cases), against each case's ground-truth mesh: for each case the median, over the interfaces present in both meshes, of \|Δfraction\|, then the median over cases — **0.00046** at `md=3` and **0.00054** at `md=4`. Interfaces present in only one of the two meshes are counted rather than scored: 5 and 4 invented and 7 and 7 missed interfaces over the 42 equilibrium cases.

## What 0.7.0 changes, against 0.5.1

Each paragraph names the population, the statistic and how 0.5.1 was compared. Times were measured on one machine (Apple M3 Max, 16 cores, 128 GiB, macOS 14.8.4), one process at a time, with a 1-minute load average between 3 and 7, unless a paragraph says otherwise.

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

The same mesh comes out of both (93 097 points), and both refuse the pressures for the same reason (4 interfaces without a measurable tension). The tension and pressure systems are solved sparsely from 2 000 unknowns (`solver="auto"`). On synthetic Voronoi foams of 10 000 and 20 000 cells (70 555 and 142 017 interfaces), the `YoungDupre` tensions took 403 s and 1.7 GiB, and 602 s and 3.1 GiB, sparsely, and the pressures 1.5 to 6 s, measured on a shared machine (load 10 to 15), so these times are upper bounds; 0.5.1 solves only densely, which there needs an estimated 382 GiB and 1 549 GiB, more than the machine has. The methods with no sparse path are slow at this size: on the 1 866-cell tissue `Equilibrium` took about 1 hour, `Variational` about 70 minutes and `ForceBalance` about 110 minutes, each with a peak of 68 GiB.

**Seed ensembles.** `infer_tensions_over_seeds(mask)` (new in 0.7.0; eight seeds, the default) against one reconstruction at the default seed, which is what 0.5.1 returns (its number is the same in 0.7.0), with the tension table's statistic: on the 42 equilibrium cases the median error fell from 0.0777 to 0.0638, lower on 23 cases and higher on 19 (paired median change −4.9 %); on the 5 non-equilibrium cases from 0.0689 to 0.0321, lower on 4. Most of the error on these small aggregates is bias that every seed shares, which an ensemble cannot remove.

**Anisotropic voxels.** 0.5.1 could only mesh an anisotropic mask on its own grid and scale the mesh afterwards. 0.7.0 resamples it to cubic voxels first (`dcel_mesh_from_segmentation_mask(mask, spacing=...)`, and `dw3d`'s `construct_mesh_in_physical_units`). Measured on masks down-sampled along z by a factor `k` (each coarse voxel takes the label covering most of it), with the mask at full resolution as the control, one seed, the shipped code: the median over cases of each mesh's median absolute line-mean angle error, as the difference with the control, paired per case:

| `k` | 12 simulated tissues of 90 and 165 cells at real cell size (control 5.57°) | | 42 benchmark cases (control 6.00°) | |
|---|---:|---:|---:|---:|
| | resampled (0.7.0) | native grid (0.5.1's only route) | resampled | native grid |
| 5/3 | +0.83° | +1.38° | +0.43° | +1.43° |
| 2 | +0.71° | +1.97° | +1.54° | +2.43° |
| 3 | +1.17° | +4.10° | +1.27° | +5.58° |

The resampled route keeps a valid mesh: on both populations and at every `k`, every mesh is watertight, with no nonlocal self-intersection and no degenerate triangle; abnormal non-manifold edges stay at the control's level (one edge on 4 of 36 tissue meshes, against one on one control mesh and 28 on 15 meshes for the native grid; none on the benchmark). Its trijunction lines are rougher than the control's (tangent error +1.1° to +1.8°, position +0.15 to +0.40 voxel on the tissues) and smoother than the native grid's at `k` = 2 and 3 (tangent error −1.3° and −4.2°, lower on all 12 tissues). Above `k` = 3 nothing was measured, and a warning says so.

**The Laplace pressures.** 0.5.1's `Laplace` and `WeightedLaplace` pressures added `base_pressure` twice: every cell came out `base_pressure` too high. With the default `base_pressure=0`, no number changes.

## Package renamed `benchmarks` -> `foambryo_benchmarks`

The top-level `benchmarks/` package was renamed to `foambryo_benchmarks/` because both this repository and `delaunay-watershed-3d` shipped a package named `benchmarks`, and a script that put both repositories on `sys.path` could silently resolve the name to either one.

**Migration hazard:** pulling this rename into an existing checkout leaves a stray `benchmarks/__pycache__/` directory on disk — `git mv` moves tracked files, not gitignored build artifacts. With `benchmarks/__init__.py` gone but the directory still present, Python imports it as an empty implicit namespace package (PEP 420): `import benchmarks` still succeeds, with no `__file__` and no real submodules. `tests/test_no_ambiguous_benchmarks_package.py` now tolerates this case (only a *real*, `__file__`-bearing `benchmarks` package fails it), but the stray directory should still be deleted on any checkout that hits it — `rm -rf benchmarks/` once `git status` confirms it holds nothing tracked.
