# Seed ensembles

Tensions from eight reconstructions of one mask: the ensemble, its relative seed spread, its flags, its caveat.

delaunay-watershed-3d places its tessellation points with a small random dither, fixed by a seed (42 by default). Another seed gives a slightly different mesh of the same mask, and slightly different tensions. `infer_tensions_over_seeds` reconstructs the mask at a fixed list of seeds (0 to 7 by default), infers the tensions of each mesh and averages them per interface. This example runs it on two generated masks:

* **a doublet at equilibrium**: two equal spheres split by a plane, so that the three surfaces meet at 120 degrees and all three tensions are exactly 1. Its error against the truth can be compared with its seed spread: the spread is a **lower bound on the error, not an error bar** (it sees only the part of the error that changes with the seed, not the bias that every reconstruction shares);
* **a foam of 12 Voronoi cells** (not at equilibrium: its tensions show the workflow, not the accuracy), whose small facets are not found by every seed: such interfaces are **flagged**, not removed (`found_by_fewer_than_half_the_seeds`). On synthetic tissues, most interfaces found by fewer than half of the seeds did not exist in the true tissue.

Fix the seed list in advance, and never choose seeds by their result. `combine="angles"` (average the contact angles over the seeds, then infer once) is experimental and warns with `SeedEnsembleExperimentalWarning`; it is run here once, with the quadric reading, for comparison only.

    pip install "foambryo[viewing]"          # matplotlib, for the figure
    python seed_ensembles.py                 # one to two minutes; writes ./seed-ensembles-output/

The script: [`Examples/seed_ensembles.py`](https://github.com/VirtualEmbryo/foambryo/blob/v0.7.0/Examples/seed_ensembles.py). It needs only foambryo (and matplotlib, from `foambryo[viewing]`, for its figure); it generates its own tissue.

![Left: the doublet's three tensions at each of the 8 seeds (grey) and their ensemble with its seed SD (black); the true tension is 1 (red line). The error is two to three times the spread. Right: the foam's 49 ensemble tensions with their seed SD; in red the interfaces found by fewer than half of the seeds.](figures/seed_ensembles.png)

*Left: the doublet's three tensions at each of the 8 seeds (grey) and their ensemble with its seed SD (black); the true tension is 1 (red line). The error is two to three times the spread. Right: the foam's 49 ensemble tensions with their seed SD; in red the interfaces found by fewer than half of the seeds.*

## Output of one run

On macOS 14 (Apple M3 Max), Python 3.12; times vary with the machine.

```text
doublet at equilibrium, 8 seeds (0..7); every true tension is 1
  interface   ensemble  |error|  seed SD  relative SD
  (0, 1)        1.069    0.069    0.029        2.7%
  (0, 2)        1.076    0.076    0.022        2.0%
  (1, 2)        0.855    0.145    0.044        5.2%
foam of 12 cells, 8 seeds
  49 interfaces; relative seed SD median 30.7%, 90th percentile 77.6%
  found by fewer than half of the seeds (flagged, kept): [(3, 6), (7, 12), (9, 12)] (seeds that found each: [3, 2, 1])
  combine='angles' warned: SeedEnsembleExperimentalWarning
  combine='angles' (quadric) against the default ensemble: median relative difference 14.1% over 49 interfaces (a comparison, not an accuracy)
figure: seed-ensembles-output/seed_ensembles.png
```

## The script

```python
--8<-- "Examples/seed_ensembles.py"
```
