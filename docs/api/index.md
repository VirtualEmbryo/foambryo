# API reference

Generated from the docstrings. The names exported by `foambryo` itself are listed in the package's `__init__`; the modules below hold them.

| module | what |
|---|---|
| [`foambryo.io`](io.md) | `dcel_mesh_from_segmentation_mask`, `dcel_mesh_from_file`, `reconstruct_mask` |
| [`foambryo.dcel`](mesh.md) | `DcelData`: vertices, triangles, labels, crop faces, cut cells |
| [`foambryo.tension_inference`](tension-inference.md) | `infer_tensions` and its variants, `TensionComputationMethod` |
| [`foambryo.pressure_inference`](pressure-inference.md) | `infer_pressures`, `PressureComputationMethod` |
| [`foambryo.force_inference`](force-inference.md) | `infer_forces`, `infer_tensions_then_pressures` |
| [`foambryo.seed_ensembles`](seed-ensembles.md) | `infer_tensions_over_seeds` |
| [`foambryo.crop_faces`](crop-faces.md) | crop faces, cut cells and their warnings |
| [`foambryo.linear_solves`](linear-solves.md) | `solver`, `SolveDiagnostics`, `save_solve_diagnostics` |
| [`foambryo.junction_graph`](junction-graph.md) | components, undetermined interfaces, the exact normalisation |
| [`foambryo.uncertainty`](uncertainty.md) | `infer_tensions_with_uncertainty`, `compute_inference_diagnostics` |
| [`foambryo.second_order_angle_reading`](second-order-angle-reading.md) | `angle_reading` |
| [`foambryo.wedge_refusal`](wedge-refusal.md) | `refusal`, `WedgeRefusal`, `refusal_report` |
| [`foambryo.per_edge_tension_inference`](per-edge-tension-inference.md) | `YoungDupreLocal` |
| [`foambryo.force_balance_inference`](force-balance-inference.md) | `ForceBalance*`, `infer_force_balance` |
| [`foambryo.reconstruction_cost`](reconstruction-cost.md) | the time and memory estimate |
| [`foambryo.segmentation_masks`](segmentation-masks.md) | reading and checking masks, the mask report |
| [`foambryo.results`](results.md) | the tables and the run record of the command |
| [`foambryo.colour_scales`](colour-scales.md) | the viewers' colours |
| [`foambryo.viewing`](viewing.md) | `plot_force_inference`, `view_mask`, `save_colour_bars` |
| [`foambryo.geometry`](geometry.md) | areas, volumes, angles and their derivatives |
| [`foambryo.curvature`](curvature.md) | mean and Gaussian curvatures |
| [`foambryo.validation`](validation.md) | mesh integrity and force-balance checks |
