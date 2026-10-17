# Large tissue: estimate, solvers, diagnostics

A large tissue: the cost estimate, the sparse solves, and the table of every linear solve.

Every tension and pressure foambryo infers is the solution of a linear least-squares system. Small systems are solved densely, exactly; a dense system grows as the product of its dimensions in memory and as their cube in time, so `solver="auto"` (the default) solves a system sparsely from `sparse_threshold` unknowns on (2 000 by default). This example generates a foam of several hundred Voronoi cells, and:

1. estimates the reconstruction's time and memory before running it (`foambryo.reconstruction_cost`: power laws fitted on one machine to medians over the benchmark tissues of 2 to 10 cells and to single tissues of 90 to 1 866 cells; the band is the fit's, not a guarantee);
2. reconstructs the mesh and infers tensions and pressures with `diagnostics=True`, which returns one record per linear solve (the solver used, the system's size, rank, residual, condition number, time). It calls `infer_tensions_then_pressures`, the two calls of `infer_forces` made one after the other, as `foambryo infer` does: if an interface has no triple line, its tension is undetermined (`nan`), the pressure solve refuses, and the tensions are kept instead of lost;
3. writes them as `solves.csv`, the table `foambryo infer --diagnostics basic` writes (from a direct call with `diagnostics=True`, `save_solve_diagnostics` writes the same records as JSON and NPZ files);
4. solves the tensions again with the dense and with the sparse solver forced, and compares them.

The foam is not at mechanical equilibrium: its tensions show the workflow and the solvers, not the method's accuracy. On a 1 866-cell tissue the sparse and dense tensions agree to 1.5e-11 of their scale and the pressures to 1e-11; the dense solve took 6 to 9 minutes and 14-18 GiB there, the sparse one 9 s and 1 GiB.

    python large_tissue_solvers.py                # about a minute and 5 GiB; writes ./large-tissue-output/
    python large_tissue_solvers.py --cells 300    # smaller and faster

The script: [`Examples/large_tissue_solvers.py`](https://github.com/VirtualEmbryo/foambryo/blob/v0.7.0/Examples/large_tissue_solvers.py). It needs only foambryo; it generates its own tissue.

## Output of one run

On macOS 14 (Apple M3 Max), Python 3.12; times vary with the machine.

```text
generated foam: shape (220, 220, 220), 524 cells, 4 s
estimated reconstruction: 5 s (3-8), 1.8 GiB (1.2-2.7)
reconstructed: 7171 vertices, 21248 triangles, 16 s
infer_forces(mesh, mean_tension=1, base_pressure=0, mode_tension=TensionComputationMethod.YoungDupre, mode_pressure=PressureComputationMethod.WeightedLaplace, solver='auto', diagnostics=True): complete, 1.3 s; 3110 tensions (0 undetermined), 525 pressures
  system                                           solver                    rows unknowns  condition  seconds
  YoungDupre tension system                        sparse lsqr              12613     3110        605     0.03
  WeightedLaplace pressure system                  dense lstsq               3108      520       18.5     0.06
wrote large-tissue-output/solves.csv
tensions, dense solve 5.1 s, sparse solve 0.9 s; they differ by 8.2e-13 of the tensions' scale
       34.71 real        27.46 user         1.76 sys
          4544233472  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
              289045  page reclaims
                  10  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                   0  messages sent
                   0  messages received
                   0  signals received
                 297  voluntary context switches
               49144  involuntary context switches
        392591154825  instructions retired
        103362655852  cycles elapsed
          1199106048  peak memory footprint
```

## The script

```python
--8<-- "Examples/large_tissue_solvers.py"
```
