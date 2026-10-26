# Large tissues: time, memory and solvers

## Before the reconstruction: the estimate

Reconstructing the mesh is the expensive step. `foambryo.reconstruction_cost` estimates its wall time and peak memory from the number of voxels handed to delaunay-watershed-3d (an anisotropic mask counts its voxels after resampling to cubic voxels):

```py
from foambryo.reconstruction_cost import estimate_reconstruction_cost, voxels_handed_to_dw3d

estimate = estimate_reconstruction_cost(voxels_handed_to_dw3d(mask.shape, spacing))
print(estimate.text())  # for a 1 866-cell tissue at cubic voxels: "about 13 min (8 min-20 min) and 39 GiB (26 GiB-58 GiB)"
```

The estimate is a power law fitted to seven sets of measured reconstructions, from aggregates of 2 to 10 cells to a 1 866-cell tissue, on one machine (Apple M3 Max, 128 GiB); its band is the largest in-sample ratio, not a guarantee. Measured there with nothing else running: a 1 866-cell tissue took 9 minutes and 31 GiB on its native grid, and 13.5 minutes and 38 GiB once resampled to cubic voxels. The `foambryo view MASK` window and the napari plugin show this estimate before they reconstruct. On a machine with too little memory, run `foambryo mesh` on a larger one and open the `.rec` file it writes.

## The linear solves: `solver`, `sparse_threshold`, `diagnostics`

Every tension and pressure is the solution of a linear least-squares system. Small systems are solved densely, exactly; a dense system grows as the product of its dimensions in memory and as their cube in time, so large ones are solved sparsely. `infer_tensions*`, `infer_pressures*` and `infer_forces` take three keyword arguments:

- `solver="auto"` (the default) solves a system densely below `sparse_threshold` unknowns and sparsely at or above; below the threshold the result is bitwise what it always was. `solver="dense"` and `solver="sparse"` force one path whatever the size. Forcing `"dense"` on a system too large for the memory available warns (`DenseSolveMemoryWarning`) before anything is allocated; a method with no sparse path (`Equilibrium`, `Variational` and `ForceBalance*` tensions) refuses `"sparse"`. These methods are slow on a large tissue: on a 1 866-cell tissue, one run each on a machine with nothing else running, `Equilibrium` took about 1 hour, `Variational` about 70 minutes and `ForceBalance` about 110 minutes, each with a peak of 68 GiB; `YoungDupre` takes seconds there.
- `sparse_threshold` (default 2 000): the number of unknowns at which `"auto"` switches.
- `diagnostics=True` returns one more value, last: a `SolveDiagnostics` with, for every solve, the solver used, the system's shape, its rank (dense solves), the residual norm, the condition number and the time. `diagnostics="spectrum"` adds the extreme singular values of the sparse systems (seconds on a 1 866-cell tissue); a dense solve always records all of them, at no cost. `save_solve_diagnostics(diagnostics, "path/stem")` writes them to a JSON and an NPZ file. The outputs are bitwise the same with and without diagnostics.

```py
tensions, pressures, diagnostics = infer_forces(
    mesh, mode_pressure=PressureComputationMethod.WeightedLaplace, diagnostics=True,
)
for record in diagnostics.records:
    print(record.system, record.solver, record.condition_number)
save_solve_diagnostics(diagnostics, "my_tissue_solves")
```

The sparse paths: `lsqr` for the Young-Dupré-family tensions and the Laplace pressures (the latter with its columns scaled to unit norm), and a sparse direct factorisation for the variational pressures. On a 1 866-cell tissue the sparse and dense pressures agree to 1e-11 of their scale, and the Young-Dupré tensions to 1.5e-11 of theirs (the dense solve took 6 to 9 minutes and 14-18 GiB there, the sparse one 9 s and 1 GiB; measured on a mesh from dw3d's 0.6.0 code, not on 0.7.1's, which has 93 099 points where 0.7.0's has 93 097; not measured again: one run needs about 31 GiB). On a synthetic 10 000-cell foam the dense Young-Dupré system would need some 380 GiB; the sparse solve took 7 minutes and 1.7 GiB, and each pressure method a few seconds.

With the command: `foambryo infer --solver sparse --diagnostics basic` writes the same records as a table, `solves.csv`, one row per solve.

## Error bars

`infer_tensions_with_uncertainty` and `compute_inference_diagnostics` have no sparse path. Both compute the diagnostics from the dense force-balance operator, three rows per mesh vertex and one column per interface and per cell, and its singular value decomposition, for every method (`YoungDupre` included). scipy cannot decompose more than `2**31 - 1` entries (`foambryo.uncertainty.DENSE_OPERATOR_ENTRY_LIMIT`), so both raise `ValueError` at once on a larger mesh, before building anything, with the operator's size. A mesh of the default reconstruction reaches the limit at about 1 300 cells of real size (generated foams of about 55 000 voxels per cell: 0.80 of it at 1 200 cells, 1.10 at 1 400); the 1 866-cell tissue's operator is 1.8 times the limit (on dw3d 0.7.0's and on 0.7.1's mesh). Before 0.7.0 such a call failed only after building the operator: about 20 minutes and 55 GiB on the 1 866-cell tissue. Below the limit the cost is that of the operator: on the 1 200-cell foam (0.80 of the limit) `infer_tensions_with_uncertainty(mesh, 1, YoungDupre)` took about 30 minutes and 46 GiB (on a machine shared with other work, so the time is an upper bound). Instead, use `infer_tensions` for the tensions, and `infer_tensions_over_seeds` for their spread over reconstruction seeds, a lower bound on the error as the `YoungDupre` interval is; a mesh with fewer vertices (a larger `min_distance`) has a smaller operator.

Example: [large tissue](../examples/large-tissue.md).
