# The Polyscope viewer from Python

The Polyscope viewer from Python: tensions and pressures on the tissue, saved figures, and a seed ensemble.

foambryo's viewer draws the mesh coloured by the inferred values, with a panel that shows their colour bars, which call produced them, a length scale bar, and controls to explode and cut the tissue, change the methods and recompute, change the background, and export the values. Every colour is computed by the library's own colour functions (`foambryo.colour_scales`), the ones the napari plugin uses too:

* tensions: `coolwarm`, light grey at the mean tension (1 by default), blue below, red above, over a range centred on the mean;
* pressures: `magma`, light for low pressures, trimmed to its 10-90 % range so that no cell is drawn black or white;
* an undetermined value (`nan`) is green, and an undetermined tension's interface is drawn as its outline; crop faces are see-through glass with their edges.

This example generates a foam of Voronoi cells imaged with voxels of 1.0 x 0.5 x 0.5 um (not at mechanical equilibrium: the colours show the viewer, not the method's accuracy), and

1. writes the colour bars without any window (`save_colour_bars`: matplotlib only);
2. writes the viewer's first view of the tensions and pressures as a figure (`plot_force_inference(..., save_figure=..., show=False)`), then of the tensions exploded, then of tensions averaged over four reconstructions with their seed spread (`plot_tension_inference(..., seed_ensemble=...)`). Saving a figure needs Polyscope and a display; the window closes once the figure is written;
3. with `--show`, opens the same views as windows to explore (close each window to go on);
4. with `--from-mask`, opens the window on the mask itself (`view_mask`): the panel reconstructs it and infers, in separate processes, with their elapsed time, an estimate and Cancel. This is what `foambryo view MASK --spacing DZ DY DX` opens.

    pip install "foambryo[viewing]" python polyscope_viewer.py                   # writes ./polyscope-viewer-output/ python polyscope_viewer.py --show            # and opens the windows python polyscope_viewer.py --from-mask       # the window that starts from the mask

The script: [`Examples/polyscope_viewer.py`](https://github.com/VirtualEmbryo/foambryo/blob/v0.7.0/Examples/polyscope_viewer.py). It needs only foambryo and `foambryo[viewing]`; it generates its own tissue.

![The figure `plot_tension_inference(..., scattered=True, save_figure=...)` writes: the generated foam's tensions, exploded, with the colour bar and the length scale bar.](figures/polyscope_viewer_tensions_exploded.png)

*The figure `plot_tension_inference(..., scattered=True, save_figure=...)` writes: the generated foam's tensions, exploded, with the colour bar and the length scale bar.*

## Output of one run

On macOS 14 (Apple M3 Max), Python 3.12, delaunay-watershed-3d 0.7.1; times vary with the machine.

```text
wrote polyscope-viewer-output/colour_bars.png: ColourBarRanges(tension=(-0.12555029199922796, 2.125550291999228), pressure=(0.0, 0.2526588795647705), alpha=0.05)
wrote polyscope-viewer-output/tensions_and_pressures.png
wrote polyscope-viewer-output/tensions_exploded.png
wrote polyscope-viewer-output/tensions_over_four_seeds.png (interfaces found by fewer than half of the seeds: 5, drawn in magenta)
```

## The script

```python
--8<-- "Examples/polyscope_viewer.py"
```
