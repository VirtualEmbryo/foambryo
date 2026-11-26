# Example data: two 4-cell *C. elegans* embryos in 2D

Two small files used by the two-dimensional examples (`Examples/two_dimensional_tissue_without_eggshell.ipynb` and `Examples/choosing_an_angle_reading_inside_an_eggshell.ipynb`). They are derived from the public data of Yamamoto et al., *Developmental Cell* 2026, and are distributed under that data's licence, **CC BY-SA 4.0**, not under foambryo's licence.

| file | embryo | frames | size (pixels) |
|---|---|---|---|
| `c_elegans_4_cell_with_eggshell.npz` | a non-perturbed embryo inside its eggshell (movie 190429-2 of the deposit) | movie layers 139, 153, 168 (normalised time 0.25, 0.49, 0.75) | 229 × 146 |
| `c_elegans_4_cell_without_eggshell.npz` | an embryo whose eggshell was removed (movie 190430-9) | movie layers 86, 105, 125 (normalised time 0.25, 0.49, 0.75) | 195 × 231 |

Each file holds:

* `images`: `(3, rows, columns)` uint16, the membrane channel (channel 0) of the movie at the three layers, cropped, values unchanged;
* `masks`: `(3, rows, columns)` uint8, the cells segmented with Cellpose-SAM and tracked over the movie: 0 is the medium, 1 to 4 the cells;
* `cell_names`: the name of each label, `cell_names[label]`. With eggshell: 1 EMS, 2 ABa, 3 ABp, 4 P2. Without eggshell the two AB daughters are both ABx and cannot usually be told apart; where they must be distinguished they are ABx1 (label 2) and ABx2 (label 3), as in the deposit's tension files: 1 EMS, 2 ABx1, 3 ABx2, 4 P2;
* `movie_layers` (0-based) and `normalised_times`: 0 is the end of the furrowing of P1's cytokinesis, 1 the start of the furrowing of the ABx cells' cytokinesis; the three frames are those closest to 0.25, 0.5 and 0.75, the window the paper analysed and its middle;
* `crop_in_movie`: the crop, `(first row, last row + 1, first column, last column + 1)` in the movie's pixels;
* `pixels_per_micrometre`: 4.0702, as the deposit states for every movie (one pixel is 0.2457 µm);
* `source`, `licence`, `citation`: the provenance as text.

The masks keep at least 16 pixels of medium between the embryo and every side of the image, so delaunay-watershed's 2D mesher meshes them without padding a thin margin.

## How they were made

From the deposit's movies (`raw_data/embryos_gfp_tif/inside_eggshell/190429-2.tif` and `raw_data/embryos_gfp_tif/wo_eggshell/190430-9.tif`): channel 0 cropped to the paper's field (rows 150–450 and columns 150–450 for 190429-2; rows 100–390 and columns 150–450 for 190430-9), segmented on every layer of the paper's 4-cell window with Cellpose-SAM (Cellpose 4.2.1.1, model `cpsam`, flow threshold 0.8, cell-probability threshold 0, diameter twice the median cell radius: 72.6 px with eggshell, 76.4 px without; labels then expanded by 20 px and kept inside the embryo's outline, the Otsu foreground of the image smoothed by a Gaussian of 1.5 px), tracked from frame to frame by the assignment of largest total overlap, and named by the paper's watershed seed positions; then three layers were taken and cropped again to the embryo with a margin of 16 pixels. No frame of either series has a number of cells other than four, a fragment, or an ambiguous link. The cell–medium boundary of the masks lies on the outer flank of the membrane, about 1 to 2 px outside its intensity maximum (seen on the frames, not measured); cell–cell boundaries lie on the membrane. With eggshell, a small bright body at the anterior tip (probably the polar body, not identified) is inside ABa's label.

## Licence and citation

The data are distributed under the Creative Commons Attribution-ShareAlike 4.0 International licence (CC BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/), as the deposit: Zenodo, doi:10.5281/zenodo.19627642. The frames were cropped and their segmentations made as above; the image values are unchanged.

When you use them, please cite:

Yamamoto K., Ichbiah S., Perez M., Borrego-Pinto J., Delbary F., Goehring N. W., Northrop P., Turlier H., Charras G. Spatiotemporal mapping of the contractile and adhesive forces sculpting early *C. elegans* embryos. *Developmental Cell* 61(6), 1273–1288.e4 (2026). doi:10.1016/j.devcel.2026.04.013.
