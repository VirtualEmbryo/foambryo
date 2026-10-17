# Cropped tissues: crop faces and cut cells

A tissue imaged in part has cells that the image boundary cuts. The mesh closes them with **crop faces**: triangles on the boundary, labelled like contact with the medium although nothing beyond them was observed. delaunay-watershed-3d marks them, and foambryo reads the mark from the reconstruction (`dcel_mesh_from_segmentation_mask`) or from a `.rec` or `.vtk` file that records it (`dcel_mesh_from_file`); you can also pass it yourself, `DcelData(points, triangles, labels, crop_faces=mask)`. Then every tension method keeps the crop faces, and every junction edge and vertex that touches them, out of its equations; an interface made only of crop faces is absent from the tension map; and the pressures of the cells the crop cuts (`mesh.cut_cells`) are returned as `nan`, with a `CutCellPressureWarning`, because part of their surface and volume was not observed. The viewers draw the crop faces as their own surface.

```py
mesh = dcel_mesh_from_segmentation_mask(mask)              # a mask whose cells touch the image box
mesh.cut_cells                                             # the cells the crop cuts
tensions = infer_tensions(mesh)                            # no crop interface in it
pressures = infer_pressures(mesh, tensions)                # nan for every cut cell
```

If the tissue really ends on an image face (a coverslip), pass `physical_image_faces=("axis0_low",)` to `dcel_mesh_from_segmentation_mask`. A mesh file written before crop faces were recorded reads them as medium contact; when it looks cropped, `dcel_mesh_from_file` warns (`UnrecordedCropFacesWarning`). The Laplace pressure methods and the uncertainty functions have no formulation for crop faces yet and raise on a cropped mesh.

What each step does with a cropped mesh:

| step | on a cropped mesh |
|---|---|
| reconstruction (`dcel_mesh_from_segmentation_mask`, `foambryo mesh`) | marks the crop faces; `mesh.crop_faces` (one boolean per triangle), `mesh.cut_cells`, `mesh.crop_interfaces` |
| `.rec` and `.vtk` files | record the crop faces; `dcel_mesh_from_file` reads them back |
| every tension method | keeps crop faces, and every junction edge and vertex that touches them, out of its equations; the rest of a junction line that ends on the crop still counts |
| `PressureComputationMethod.Variational` (the default) | infers the pressures; those of the cut cells are `nan`, with a `CutCellPressureWarning` naming them |
| `Laplace`, `WeightedLaplace` | refuse (`ValueError`); `has_crop_face_formulation` is `False` |
| `infer_tensions_with_uncertainty`, `compute_inference_diagnostics` | refuse (`ValueError`) |
| wedge refusal (`refusal=`) | a crop edge is always left out of its line's mean; the policy acts on the rest of the line |
| viewers | draw the crop faces as see-through glass with their edges; cut cells as undetermined pressures |

**What it buys**, measured on 433 plane crops of the 47 benchmark tissues (`YoungDupre`, relative error of the tensions of the interfaces that touch the crop, median over the 42 tissues at equilibrium): 13 %, 33 % and 40 % error with 10 %, 25 % and 40 % of the tissue cropped away when crop faces were read as medium contact, 10 %, 9 % and 12 % with crop faces kept out. The cut cells' pressures, reported before crop faces were recorded, had 15 % to 46 % error. On a mesh without crop faces every number is what it was before crop faces existed, bit for bit.

Example: [cropped tissue](../examples/cropped-tissue.md).
