# Masks with anisotropic voxels: `spacing`

A microscope stack usually has a z step larger than its pixel size. Give the voxel size and the mask is meshed through delaunay-watershed-3d's `construct_mesh_in_physical_units`: **resampled to cubic voxels** of the shortest spacing by shape-based interpolation, meshed, and returned in the unit of the spacing.

```py
mesh = dcel_mesh_from_segmentation_mask(mask, spacing=(1.0, 0.5, 0.5))  # (dz, dy, dx), in um
mesh.source_spacing                                                      # (1.0, 0.5, 0.5)
```

One axis may be longer than the other two, which must be equal. Meshing the native voxels and scaling the mesh afterwards samples the surfaces coarsely along the long axis and tilts the junction triangles. Measured on 12 simulated tissues of 90 and 165 cells voxelised at a real cell size and down-sampled along z, one meshing seed, as the median over tissues of each mesh's median line-mean contact-angle error minus that of the same tissue meshed at full resolution: resampling first costs +0.83, +0.71 and +1.17 degrees at an anisotropy of 5/3, 2 and 3; meshing the native voxels, +1.38, +1.97 and +4.10 (`BENCHMARKS.md`). Above 3 nothing was measured at that cell size, and delaunay-watershed-3d warns (`UnmeasuredAnisotropyWarning`) but builds the mesh. Keep the default angle reading: the quadric reading made every route worse at that cell size. The cost is that of a mask acquired at the short spacing, the anisotropy times the voxels, and `min_distance` counts cubic voxels.

Without `spacing` nothing changes: the mesh is reconstructed on the voxel grid, in voxel units, bit for bit as before. The spacing travels with the mesh: delaunay-watershed-3d records it in the `.rec` file, and `dcel_mesh_from_file` reads it back as `mesh.source_spacing`. It changes no number; the mesh is already in physical units.

The numbers above were measured on simulated tissues at mechanical equilibrium whose cells are as small, in voxels, as those of real tissues, with foambryo's default angle reading.

A mesh made without `spacing` and scaled afterwards: dw3d's `set_pixel_size(xy_pixel_size, z_pixel_size)` scales the mesh's column 0 (the mask's axis 0, z) by `z_pixel_size` and columns 1 and 2 by `xy_pixel_size`. Before 0.7.0 it scaled columns 0 and 1 by the xy size and column 2 by the z size: re-run any analysis that used it with different sizes.

Example: [anisotropic voxels](../examples/anisotropic-voxels.md).
