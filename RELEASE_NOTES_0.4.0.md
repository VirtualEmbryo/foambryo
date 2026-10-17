# Release notes — 0.4.0

`foambryo` (this repository) and `delaunay-watershed-3d` ship together as 0.4.0. This is not a changelog (see `CHANGELOG.md` for that) — it is what to run and what will mislead you, for a team testing this on real data for the first time. `dw3d`'s own notes are `delaunay-watershed-3d/RELEASE_NOTES_0.4.0.md`; read both — the anisotropy caveat below is measured on `dw3d`-reconstructed meshes and interacts with the angle reading here.

**dw3d 0.3.6 is what is currently published. This release supersedes it.**

## What is new

Two opt-in features in this repository (a third, junction-curve extraction, ships in `dw3d` — see its notes).

| feature | invocation | gain |
|---|---|---|
| second-order contact angles | `infer_tensions(..., angle_reading="quadric")` | tension contrast 0.738 → 0.954 at `min_distance=3` (`YoungDupre`); per-line angle error 7.35° → 4.83° |
| junction-curve extraction (`dw3d`) | `extract_junction_curves(mask, points, triangles, labels)` | tortuosity and curvature at 1.001× / 1.013× of the identifiability floor; line-position error 0.585 vs the mesh's own 0.864 |
| `.rec` metadata appendix (`dw3d`) | `save_rec(..., metadata={...})` | records spacing, units, coordinate frame, crop offset |

Plus one behavioural change in `dw3d`, not opt-in: zero abnormal non-manifold edges at every reconstruction spacing tested, where 0.3.6 leaves three at the shipped default. See `dw3d`'s notes for the numbers — it changes the mesh a tension inference here consumes, not anything in this repository's own code.

**One new hard error, worth knowing before you hit it.** `TensionComputationMethod.YoungDupreLocal` combined with `angle_reading="quadric"` now raises `ValueError` instead of silently returning a degraded result. See item 3 below.

## What is unchanged

Every default is bitwise identical to 0.3.6's angle reading and to the pre-merge tension inference — pinned by a SHA-256 digest of the raw IEEE-754 bits of every angle, length and tension this package computes, over all 47 benchmark meshes (`tests/test_second_order_angle_reading.py`). If you do not pass `angle_reading`, no number you have ever computed with this package moves. This is what lets your team adopt this release without re-validating past results — an existing script produces the same numbers it always has.

## What will mislead you

In this order, because the first one is the one that will actually bite.

1. **Anisotropy. `dw3d` ignores voxel spacing entirely**, and this repository's tension inference consumes whatever mesh `dw3d` hands it. A 2:1 z-anisotropic stack yields a mesh that is 42/42 watertight with zero abnormal edges — **every validity check passes** — and gives tension contrast **0.610 against 0.729** isotropic, with junction lines at 1.22 voxels instead of 0.86 and median angle error 10.5° instead of 7.8°. At 3:1: contrast **0.484**. **No diagnostic fires anywhere in either package.** Anyone meshing anisotropic data must know its spacing and treat the numbers as unvalidated. Full detail: `delaunay-watershed-3d/RELEASE_NOTES_0.4.0.md`.
2. **The quadric reading is calibrated at `dw3d`'s benchmark lever**, `min_distance` 3–4. Its over-application is lever-dependent — positive (helps) at fine spacing, crossing zero near `min_distance=5`. On the 47 *ground-truth* reference meshes (no reconstruction noise to correct) it is *worse* than the `"chord"` reading on **47 of 47** cases (median `|contrast-1|` 0.0803 against ~0.0000). On real cells of a different size than this benchmark's, the achieved lever differs from 3–4, so **report both readings side by side for the first cases** you run rather than trusting the quadric reading by default.
3. **`YoungDupreLocal` cannot be combined with `angle_reading="quadric"`** — an explicit `ValueError` as of this release (previously silent), because it gave +0.0290 relative RMS tension error at `min_distance=3` and **+0.97** at `min_distance=4`, even though the angles themselves improve under the quadric reading. A per-edge balance never performs the along-line average that cancels the quadric correction's per-face scatter, so it eats the scatter directly. Use the quadric reading with a line-averaged method instead — `YoungDupre`, `ProjectionYoungDupre`, `Lami`, `InvLami`, or `LogLami` — or keep `YoungDupreLocal` on the default `"chord"` reading.
4. **The curves need the mask, not just the mesh** (`dw3d` feature, consumed here for tension inference if you key interfaces off them). Match rate to the mesh's own trijunctions is 0.900; matched and invented curves are separate fields and **only the matched set is usable** (matched median 0.68 voxels from their interfaces, invented median 29.7, none within a voxel).
5. **Costs.** The quadric reading roughly doubles the angle-reading step (the quadric fit itself is ~0.024 s, cached per mesh) — negligible next to a tension solve. Junction-curve extraction costs are on the `dw3d` side: 0.77 s/case against a ~1.5 s reconstruction, or 4.47 s with `estimator="anchor"`.

## Reproduction

```python
from foambryo import infer_tensions, TensionComputationMethod
from foambryo.dcel import DcelData
from dw3d.io import load_rec

mesh = DcelData(*load_rec("case_mesh.rec"))
chord = infer_tensions(mesh, 1, TensionComputationMethod.YoungDupre, angle_reading="chord")
quadric = infer_tensions(mesh, 1, TensionComputationMethod.YoungDupre, angle_reading="quadric")
# compare chord vs quadric on your own data before trusting either by default (item 2 above)
```

The bit-identity gate (`tests/test_second_order_angle_reading.py`) and the full benchmark table (`BENCHMARKS.md`) are the source for every number above.

## Push (not yet done)

```bash
git -C foambryo push origin main
git -C foambryo push origin v0.4.0
```
