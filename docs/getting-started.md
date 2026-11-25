# Getting started

## Installation

foambryo needs Python 3.12 or later. Install it from PyPI, in a virtual environment:

```shell
pip install foambryo                 # the library (delaunay-watershed, its 3D and 2D meshers, comes with it)
pip install "foambryo[viewing]"      # and the Polyscope viewer (Polyscope, matplotlib)
pip install "foambryo[cli]"          # and TIFF masks for the foambryo command (tifffile)
```

The napari plugin: `pip install "napari[all]" napari-foambryo` (its version 0.2.0, with 2D docks, is released with foambryo 0.8.0).

## The input: a segmentation mask

A 3-D array of non-negative integer labels: one label per cell, 0 for the medium around the tissue. Axis 0 is z, axes 1 and 2 are y and x. Its voxel size is `(dz, dy, dx)`, in any one unit; one axis (the z step) may be longer than the other two, which must be equal. foambryo never guesses it: give it, or work in voxel units.

A **2-D mask** (axis 0 y, axis 1 x) is a section or a monolayer imaged in one plane; its pixel size is `(dy, dx)`, the two equal. The same functions mesh it and infer its tensions and pressures; see [two-dimensional tissues](user-guide/two-dimensional-tissues.md), where it says which numbers hold for a confined tissue (an embryo in its eggshell, cells on a substrate).

## In Python

```py
import tifffile
from foambryo import dcel_mesh_from_segmentation_mask, infer_forces
from foambryo.viewing import plot_force_inference, save_colour_bars

mask = tifffile.imread("segmentation.tif")
mesh = dcel_mesh_from_segmentation_mask(mask, spacing=(1.0, 0.5, 0.5))  # (dz, dy, dx) in um; mesh in um
tensions, pressures = infer_forces(mesh)       # {(label_1, label_2): tension}, {label: pressure}
save_colour_bars(mesh, tensions, pressures, "colour_bars.png")   # no window needed
plot_force_inference(mesh, tensions, pressures, length_unit="um")  # needs a display
```

The tensions are normalised to a mean of 1 (`mean_tension`), the pressures are relative to the medium's (0, `base_pressure`), in tension per um here. Undetermined values are `nan`. See the [methods](user-guide/methods.md).

In 2D, the same calls:

```py
mask_2d = tifffile.imread("segmentation_2d.tif")                          # axis 0 is y
mesh_2d = dcel_mesh_from_segmentation_mask(mask_2d, spacing=(0.25, 0.25))  # a PlanarMesh, in um
tensions, pressures = infer_forces(mesh_2d)    # junction angles read on circular arcs (angle_reading="arc")
```

## On the command line

```shell
foambryo mesh segmentation.tif --spacing 1.0 0.5 0.5 -o tissue.rec   # the mesh, in um
foambryo infer tissue.rec -o results                                 # tensions.csv, pressures.csv, run.json
foambryo view tissue.rec --results results                           # the Polyscope window
foambryo mesh section.tif --spacing 0.25 0.25 -o section.rec         # a 2-D mask: a 2-D mesh
```

`foambryo infer` gives exactly the numbers of `infer_forces(dcel_mesh_from_file("tissue.rec"))`. See the [command-line specification](command_line_specification.md) and the [example](examples/command-line.md).

## In napari

With the napari-foambryo plugin: *File › Open Sample › foambryo › Benchmark case 000*; *Plugins › foambryo › Reconstruct* (the layer and its voxel size are chosen; click **Reconstruct**); *Infer*; *Results*; *Export*. The plugin's numbers are those of the library, bit for bit. See [Viewers](user-guide/viewers.md).
