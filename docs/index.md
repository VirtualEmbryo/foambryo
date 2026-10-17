# foambryo

**foambryo** infers the relative surface tensions and cell pressures of foam-like cell aggregates (early embryos, tissues, organoids) from the 3-D geometry of their cells. It starts from a segmentation mask, reconstructs its multimaterial mesh with [delaunay-watershed-3d](https://github.com/VirtualEmbryo/delaunay-watershed), reads the contact angles and curvatures of the mesh, and inverts the Young-Dupré and Laplace laws.

![Benchmark case 000 in foambryo's Polyscope window](images/viewer_polyscope_case_000_tensions.jpg)

* [Getting started](getting-started.md): installation and three quick starts (Python, command line, napari).
* [Methods note](methods-note.md): what foambryo computes, and how to read its results.
* User guide: [methods](user-guide/methods.md), [cropped tissues](user-guide/cropped-tissues.md), [anisotropic voxels](user-guide/anisotropic-voxels.md), [large tissues](user-guide/large-tissues.md), [seed ensembles](user-guide/seed-ensembles.md), [viewers](user-guide/viewers.md).
* [Examples](examples/index.md), one page each, every one runnable on generated data.
* [The command line](command_line_specification.md): every option, file, column, exit code and message.
* [API reference](api/index.md), generated from the docstrings.
* [Changelog](changelog.md) and [release notes](release-notes.md).

If you use foambryo, please cite: Ichbiah S., Delbary F., McDougall A., Dumollard R., Turlier H. *Embryo mechanics cartography: inference of 3D force atlases from fluorescence microscopy.* Nature Methods **20**(12), 1989–1999 (2023), [doi:10.1038/s41592-023-02084-7](https://doi.org/10.1038/s41592-023-02084-7).

foambryo is free software, under the GNU General Public License, version 3 or (at your option) any later version (`GPL-3.0-or-later`).
