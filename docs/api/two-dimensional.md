# Two-dimensional tissues

The package-level functions (`infer_tensions`, `infer_tensions_by_junction_component`, `infer_pressures`, `infer_forces`, `infer_tensions_then_pressures`, `dcel_mesh_from_segmentation_mask`, `dcel_mesh_from_file`) take 2-D meshes and masks: their pages are those of the 3-D functions ([tension inference](tension-inference.md), [pressure inference](pressure-inference.md), [force inference](force-inference.md), [input and output](io.md)). This page lists the names that exist only in 2D. **Provisional:** they may change in any release until they are declared stable. The user guide's page on [two-dimensional tissues](../user-guide/two-dimensional-tissues.md) explains them.

## The 2-D mesh

::: foambryo.planar_mesh

## Interfaces, junctions and the image border

::: foambryo.interface_arcs
    options:
      show_root_heading: true
      heading_level: 3

::: foambryo.InterfaceArc
    options:
      show_root_heading: true
      heading_level: 3

::: foambryo.junction_table
    options:
      show_root_heading: true
      heading_level: 3

::: foambryo.image_border_vertices
    options:
      show_root_heading: true
      heading_level: 3

## Reports, warnings and constants

::: foambryo.TwoDimensionalReport
    options:
      show_root_heading: true
      heading_level: 3

::: foambryo.TwoDimensionalPressureReport
    options:
      show_root_heading: true
      heading_level: 3

::: foambryo.IllConditionedLensWarning
    options:
      show_root_heading: true
      heading_level: 3

::: foambryo.NoJunctionError
    options:
      show_root_heading: true
      heading_level: 3

::: foambryo.MAX_LENS_CONDITION
    options:
      show_root_heading: true
      heading_level: 3

::: foambryo.ANGLE_READINGS_2D
    options:
      show_root_heading: true
      heading_level: 3

::: foambryo.DEFAULT_ANGLE_READING_2D
    options:
      show_root_heading: true
      heading_level: 3

## The global fit (opt-in)

::: foambryo.infer_forces_by_equilibrium_tiling
    options:
      show_root_heading: true
      heading_level: 3

::: foambryo.EquilibriumTilingReport
    options:
      show_root_heading: true
      heading_level: 3

::: foambryo.EQUILIBRIUM_TILING_MAX_EVALUATIONS
    options:
      show_root_heading: true
      heading_level: 3

## Drawing a 2-D mesh

`planar_mesh_drawing` and `PlanarMeshDrawing`, what the viewers draw of a 2-D mesh, are on the [plotting utilities](plotting-utilities.md) page.
