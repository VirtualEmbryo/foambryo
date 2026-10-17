"""The colour scales of the force viewer, for every front end that colours tensions and pressures.

The Polyscope viewer (`foambryo.viewing`), `foambryo view --colour-bars` and the napari plugin colour with these
functions, so one value has one colour in all of them, and a colour bar spans exactly the range the colours were
clipped to:

* tensions: `coolwarm` (`TENSION_COLOUR_MAP`: blue for low, light grey for the mean, red for high), over a range
  centred on the mean of the finite tensions and symmetric around it, reaching the farther of the `alpha` and
  `1 - alpha` quantiles (`tension_colour_range`, `tension_colour_fractions`, `tension_colours`);
* pressures: `magma` reversed and trimmed to its 10-90 % range (`PRESSURE_COLOUR_MAP`,
  `PRESSURE_COLOUR_MAP_RANGE`: no pressure is black or near-white, so none vanishes on a black or a white
  background), from 0 to the `1 - alpha` quantile of the finite pressures over the surface's triangles
  (`surface_pressure_colour_upper`, `pressure_colour_fractions`, `pressure_colours`);
* other per-interface quantities (a seed spread, a residual): `jet` between the `alpha` and `1 - alpha` quantiles
  (`quantile_colour_range`, `clipped_colour_fractions`, `quantity_colours`);
* undetermined values (`nan`): `UNDETERMINED_COLOR`, green, outside every map; a viewer that can draw outlines draws an
  undetermined interface as its outline in this green, without fill;
* crop faces: see-through glass, `CROP_FACE_COLOR` at `CROP_FACE_OPACITY` with edges in `CROP_FACE_EDGE_COLOR`;
* the backgrounds a viewer offers, by name (`BACKGROUND_COLOURS`: black, dark grey, light grey, white).

`alpha` is `DEFAULT_CLIP_QUANTILE` (0.05) unless given. `save_colour_bars` writes the bars to an image file and
returns the values at their ends (`ColourBarRanges`). This module needs matplotlib (installed by
`foambryo[viewing]`), but neither Polyscope nor a display. The functions are implemented in
`foambryo._colour_scales`; this module is their public interface.
"""

from foambryo._colour_scales import (
    BACKGROUND_COLOURS,
    CROP_FACE_COLOR,
    CROP_FACE_EDGE_COLOR,
    CROP_FACE_OPACITY,
    DEFAULT_CLIP_QUANTILE,
    PRESSURE_COLOUR_MAP,
    PRESSURE_COLOUR_MAP_RANGE,
    TENSION_COLOUR_MAP,
    UNDETERMINED_COLOR,
    ColourBarRanges,
    clipped_colour_fractions,
    colour_bar_ranges,
    pressure_colour_fractions,
    pressure_colour_upper,
    pressure_colours,
    quantile_colour_range,
    quantity_colours,
    save_colour_bars,
    surface_face_pressures,
    surface_pressure_colour_upper,
    tension_colour_fractions,
    tension_colour_range,
    tension_colours,
)

__all__ = [
    "BACKGROUND_COLOURS",
    "CROP_FACE_COLOR",
    "CROP_FACE_EDGE_COLOR",
    "CROP_FACE_OPACITY",
    "DEFAULT_CLIP_QUANTILE",
    "PRESSURE_COLOUR_MAP",
    "PRESSURE_COLOUR_MAP_RANGE",
    "TENSION_COLOUR_MAP",
    "UNDETERMINED_COLOR",
    "ColourBarRanges",
    "clipped_colour_fractions",
    "colour_bar_ranges",
    "pressure_colour_fractions",
    "pressure_colour_upper",
    "pressure_colours",
    "quantile_colour_range",
    "quantity_colours",
    "save_colour_bars",
    "surface_face_pressures",
    "surface_pressure_colour_upper",
    "tension_colour_fractions",
    "tension_colour_range",
    "tension_colours",
]
