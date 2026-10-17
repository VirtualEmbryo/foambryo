# Getting started

## Installation

foambryo needs Python 3.11 or later. Install it from PyPI, in a virtual environment:

```shell
pip install foambryo                 # the library (delaunay-watershed-3d comes with it)
pip install "foambryo[viewing]"      # and the Polyscope viewer (Polyscope, matplotlib)
pip install "foambryo[cli]"          # and TIFF masks for the foambryo command (tifffile)
```

The napari plugin: `pip install "napari[all]" napari-foambryo` (its version 0.1.0 is released with foambryo 0.7.0).

## The input: a segmentation mask

A 3-D array of non-negative integer labels: one label per cell, 0 for the medium around the tissue. Axis 0 is z, axes 1 and 2 are y and x. Its voxel size is `(dz, dy, dx)`, in any one unit; one axis (the z step) may be longer than the other two, which must be equal. foambryo never guesses it: give it, or work in voxel units.

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

## On the command line

```shell
foambryo mesh segmentation.tif --spacing 1.0 0.5 0.5 -o tissue.rec   # the mesh, in um
foambryo infer tissue.rec -o results                                 # tensions.csv, pressures.csv, run.json
foambryo view tissue.rec --results results                           # the Polyscope window
```

`foambryo infer` gives exactly the numbers of `infer_forces(dcel_mesh_from_file("tissue.rec"))`. See the [command-line specification](command_line_specification.md) and the [example](examples/command-line.md).

## In napari

With the napari-foambryo plugin: *File › Open Sample › foambryo › Benchmark case 000*; *Plugins › foambryo › Reconstruct* (the layer and its voxel size are chosen; click **Reconstruct**); *Infer*; *Results*; *Export*. The plugin's numbers are those of the library, bit for bit. See [Viewers](user-guide/viewers.md).
