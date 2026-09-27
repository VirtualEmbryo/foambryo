# Release notes — 0.5.0

`foambryo` and `delaunay-watershed-3d` ship together as 0.5.0; this foambryo requires
`delaunay-watershed-3d>=0.5.0`. This is what changes for you and what can mislead you;
`CHANGELOG.md` has the full list, and dw3d's `RELEASE_NOTES.md` covers reconstruction.

0.5.0 is the first release on PyPI since 0.3.5; `CHANGELOG.md` also lists the changes of 0.4.0–0.4.2,
which were never published there.

## Licence

From 0.5.0 `foambryo` is licensed under the **GNU General Public License, version 3 or (at your
option) any later version** (SPDX `GPL-3.0-or-later`), replacing CC BY-NC-SA 4.0: commercial use is allowed, under the GPL's conditions.
Earlier releases keep the licence they were published under.

## What changes for you

- **Meshes reconstructed with dw3d 0.5.0 differ from 0.4 meshes**: junction relocation is on by
  default there, so angles and tensions inferred from a freshly reconstructed mesh move.
  Reconstruct with `relocate_junctions=False` to reproduce a 0.4 mesh.
- **`dcel_mesh_from_segmentation_mask` now defaults to `min_distance=3`** (was 5), dw3d's own
  default: a denser mesh, slower to build. Pass `min_distance=5` for the 0.4 spacing.
- **The pressure and force functions now raise instead of returning a non-finite field.**
  `infer_pressures`, `infer_pressures_and_residuals` and `infer_forces` raise
  `UndeterminedTensionError` (a `ValueError`) naming every interface whose tension is not finite;
  on the real embryo series this happens on 5 of 154 timepoints, all outside the
  segmentation-quality window, where 0.4 returned 54–71 non-finite pressures out of 55–72 without
  a warning. Tensions are unchanged, and pressures from finite tensions are bitwise unchanged.
- **The second-order reading (`angle_reading="quadric"`) is now independent of the mesh's units**,
  which moves previously published second-order numbers by up to **0.14°**.
- **`AngleReadingCoverageWarning` now carries its numbers**: the measured coverage fraction and how
  many trijunction lines were not fully corrected.

## Limitations

Every item is measured; each is one sentence and its number.

1. **The benchmark under-rewards tangent-based improvements.** Its ground-truth meshes are
   facet-equilibrated (Surface Evolver), so an improvement in the surface-*tangent* description of
   a contact angle is scored against a facet-based truth and can read as no change.
2. **The second-order angle reader has its own floor, and it cannot resolve junction relocation's
   effect.** On a perfectly smooth analytic surface it reads contact angles with a floor of
   **0.245°**, and at every point spacing measured (0 of 32 spacing-by-variant cells) its check could
   not resolve the effect of junction relocation on tangent-based angles, in either direction.
3. **On real embryos, do not use second-order line means at sub-degree scale.** The second-order
   reader covers a median **0.804** of incident trijunction faces on real embryo meshes (minimum
   **0.454**), and its line means move by **1.75° [1.41, 2.17]** when uncovered faces are excluded
   instead of filled by the chord reading.
4. **Real embryo cells are small in voxels, so every benchmark statement is about one spacing step
   optimistic for them.** Their equivalent radius at trijunctions is a median **17.6** voxels against
   **38.2** in the benchmark, so real data at `min_distance 3` sits where the benchmark sits at
   `min_distance 4`; for cells this small, consider a smaller `min_distance`.
5. **Folded trijunctions on real meshes are mostly real geometry.** Real embryo meshes carry
   **3.2×** the benchmark's fraction of reflex (folded) trijunctional edges, and more than half of
   them sit where the segmentation mask's own wedge already exceeds 180°.
6. **The force-balance residual `rho_free` alarms on every reconstructed mesh.** Its median is
   **0.578** on reconstructed meshes against **0.0118** on the reference meshes, so a high value on
   your mesh is expected and is not by itself a sign of a bad segmentation.
7. **`ForceBalanceMAP` is not scale-invariant**, contrary to its docstring: rescaling a mesh moves
   its relative tensions by up to **43 %**; plain `ForceBalance` is exact under rescaling.
8. **Junction relocation, now on by default, has a cost that grows faster than linearly with mesh
   size.** It adds about **+0.75 s** to a ~5 s benchmark reconstruction (**+15 %**) and **0.1–1.5 s**
   per timepoint to a ~0.6 s real-embryo reconstruction (median **0.34 s** and **0.48 s** on two
   developmental time series, about 65 s per 154-timepoint series); within one real series it grows
   with triangle count with a fitted exponent of **2.68** over a 2.2-fold range of mesh sizes, which
   is not a tissue-scale law. **A faster neighbour search is planned for 0.5.1**, and will leave every
   relocated mesh bit-for-bit unchanged.
