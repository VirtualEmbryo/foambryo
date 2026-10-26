# Several reconstructions: seed ensembles

delaunay-watershed-3d places its tessellation points with a small random dither, fixed by a seed (42 by default, so every result is reproducible on one machine; between machines, remaining differences are expected only at exactly cospherical points, confirmed on Linux x86-64 in continuous integration). Another seed gives a slightly different mesh of the same mask, and slightly different tensions. With the default reconstruction and `YoungDupre`, a tension typically varies from seed to seed by about 5 % on the benchmarking dataset, whose cells are large; by about 13–14 % on tissues with cells of real size (a radius of ~25 voxels); and by about 21 % on a 110-cell and a 1 866-cell tissue meshed on their native anisotropic grid (measured on meshes from dw3d's 0.6.0 code, not on 0.7.1's, which differ slightly; not measured again: one reconstruction of the larger tissue needs about 31 GiB).

`infer_tensions_over_seeds` reconstructs the mask at a fixed list of seeds (0 to 7 by default), infers the tensions of each mesh and averages them per interface:

```py
from foambryo import infer_tensions_over_seeds

result = infer_tensions_over_seeds(segmentation_mask)          # seeds 0..7, mean of the tensions
result.tensions                                                # interface -> ensemble tension
result.tension_seed_sd                                         # interface -> SD over the seeds
result.relative_tension_seed_sd                                # interface -> that SD / |tension|
result.found_by_fewer_than_half_the_seeds                      # interface -> flag
```

Against ground truth, an ensemble of 8 seeds lowered the median relative tension error from 0.152 to 0.123 on 12 synthetic tissues with cells of real size (lower on all 12), and from 0.077 to 0.064 on the benchmarking dataset (lower on 35 of 42 cases; both measured on meshes from dw3d's 0.6.0 code, never published and part of 0.7.0; not measured again on 0.7.1's meshes), where most of the error is bias rather than seed noise. Those baselines are a single seed on average. Against one reconstruction at the default seed, which is what `infer_tensions` and earlier versions return, the benchmark's median fell from 0.076 to 0.063, but the ensemble was lower on 23 of the 42 cases only (dw3d 0.7.1's meshes; 0.078 to 0.064 and 23 cases on 0.7.0's). The contact-angle error fell by about 2 degrees at real cell size. The cost is one reconstruction per seed: about a minute and 8 GiB (up to 11) per seed for a 165-cell tissue, about 11 minutes and 30–35 GiB for a 1 866-cell tissue. `max_workers` runs seeds in parallel processes, each holding one reconstruction in memory; the result does not depend on it.

Three things to know:

- **The seed list is fixed in advance.** Never choose seeds by their result: that tunes the measurement on itself. `seeds=[42]` reproduces the single-seed `infer_tensions` result bit for bit.
- **The seed spread is a lower bound on the error, not an error bar.** `tension_seed_sd` sees only the part of the error that changes with the seed. On the synthetic tissues a single seed's ±1 SD interval held the true tension for about half of the interfaces, against 68 % for a calibrated interval.
- **Interfaces found by only some seeds** are flagged, not removed. On the synthetic tissues most of the interfaces found by fewer than half of the seeds do not exist in the true tissue.

**`combine="angles"` is experimental.** It averages each trijunction's contact angles over the seeds and infers the tensions once, instead of averaging the tensions inferred from each seed. With `angle_reading="quadric"` it was the most accurate ensemble we measured on synthetic tissues with known tensions, voxelised at the cell size of real embryos, but not by a margin consistent enough to make it the default:

* against the default ensemble (`combine="tensions"`, chord reading, 8 seeds) it lowered the median tension error by 45 % on 46 independent synthetic tissues (better on 40 of them), and by 52 % (41 of 46) when the voxels were 5/3 times longer along z and the mask was up-sampled before meshing;
* against averaging the tensions read with the quadric reading, its margin is small (3–7 %, better on 31–33 of the 46): most of its gain comes from the quadric reading, which a single reconstruction reads too noisily at this cell size and averaging makes usable;
* the quadric reading is not a safe choice on its own either: averaging tensions read with it halves the error on small aggregates of 2 to 10 cells but is 17–23 % worse than the default on tissues of 90 and 165 cells. Only `combine="angles"` with the quadric reading was better than the default on both;
* with the chord reading, `combine="angles"` is level with the default on small aggregates and inconsistent on the larger tissues.

On synthetic masks degraded by a smooth random warp of about one voxel, `combine="angles"` with the quadric reading lost its advantage over the default on larger tissues of 90 and 165 cells while keeping it on small aggregates.

The default stays `combine="tensions"` with the chord reading. Use `combine="angles", angle_reading="quadric"` as a comparison, not as a replacement, until it has been tested on tissues with known tensions that are closer to real segmentations. It warns with `SeedEnsembleExperimentalWarning`.

With the command: `foambryo infer-over-seeds MASK --spacing DZ DY DX -o OUTDIR` (see the [command line](../command_line_specification.md)). In the viewers, `foambryo view MESH --results OUTDIR` draws the ensemble tensions on the mesh, the seed spread as a second quantity, and the interfaces found by fewer than half of the seeds in magenta.

Example: [seed ensembles](../examples/seed-ensembles.md).
