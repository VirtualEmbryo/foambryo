# Examples

Each example is a short script in the repository's [`Examples/`](https://github.com/VirtualEmbryo/foambryo/blob/v0.7.0/Examples) folder. The first six generate their own tissue (foams of Voronoi cells, or a doublet at equilibrium whose answer is known), so they need nothing but foambryo; each writes its outputs in a folder of the current directory.

| example | what it shows | time |
|---|---|---|
| [Anisotropic voxels](anisotropic-voxels.md) | `spacing`, the resampling to cubic voxels, what forgetting the spacing does | under a minute |
| [Cropped tissue](cropped-tissue.md) | crop faces, cut cells, `nan` pressures, a pressure method that refuses | under a minute |
| [Seed ensembles](seed-ensembles.md) | eight seeds, the relative spread, the flags, the caveat | under a minute |
| [Large tissue](large-tissue.md) | the cost estimate, sparse and dense solves, the diagnostics table | about a minute |
| [The command line](command-line.md) | `foambryo mesh`, `infer`, `view`, `infer-over-seeds`, `--no-pressures`, `--diagnostics` | under a minute |
| [The Polyscope viewer](polyscope-viewer.md) | the viewer from Python, saved figures, a seed ensemble, the window that starts from a mask | under a minute |
| [Second-order angle reading](second-order-angle-reading.md) | `angle_reading="quadric"` on a benchmark mesh | seconds |
| [Folded trijunction observations](refusing-folded-trijunction-wedges.md) | the wedge-refusal policies | seconds |
| [Junction relocation](default-junction-relocation.md) | what delaunay-watershed-3d's relocation changes, on your own mask | seconds |

The [introductory notebook](https://github.com/VirtualEmbryo/foambryo/blob/v0.7.0/Examples/Example.ipynb) walks through the API on a benchmark aggregate; its data are downloaded separately (see `data/alternative_link_to_data`). The napari plugin has its own examples: a quick start on benchmark case 000, and *From a segmentation plugin to foambryo*.

Times were measured on one machine (Apple M3 Max); the Polyscope examples need a display.
