"""The colour scales of the force viewer, shared by the viewer and by its colour bars.

The viewer (`foambryo.viewing`) colours the surface tensions with `coolwarm` (blue for low, light grey for
the mean tension, red for high) over a range centred on the mean tension and symmetric around it
(`tension_colour_range`), and the pressures with `magma`, reversed and trimmed to its 10-90 % range
(`PRESSURE_COLOUR_MAP_RANGE`), between 0 and an upper quantile of the pressures over the surface's triangles
(`surface_pressure_colour_upper`), on the surface and on the graph view's cell nodes alike. Other per-interface
quantities (a seed spread, a residual) keep `jet` (`quantity_colours`): they are not tensions, and a diverging map
centred on their mean would mean nothing.

Two things on a surface are not values and are drawn so that no colour of either map can be read for them:

* an **undetermined** value (`nan`): `UNDETERMINED_COLOR`, green, outside both maps on Polyscope's white and
  napari's black backgrounds. A viewer that can draw outlines draws an undetermined interface as its outline
  in this green, without fill; a viewer that cannot fills it with the green;
* a **crop face** (where the image box cut the tissue): see-through "glass", `CROP_FACE_COLOR` at
  `CROP_FACE_OPACITY`, with its triangles' edges in `CROP_FACE_EDGE_COLOR`.

A colour without its scale cannot be read, so `save_colour_bars` draws the scales, and both the viewer and the
bars compute them with the functions of this module: a bar's ends are then the viewer's clipped range by
construction, not by a second implementation that could drift.

This module imports matplotlib but not Polyscope, so the colour bars can be written on a machine
with no display and no Polyscope installed (the `foambryo view --no-show` command). It draws on a
`matplotlib.figure.Figure` with its non-interactive canvas, never through `pyplot`, so no window
backend is ever selected.

Every front end (the Polyscope viewer, `foambryo view --colour-bars`, the napari plugin) takes its colours
from these functions, so one value has one colour in all of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import cm, colormaps
from numpy.typing import NDArray

if TYPE_CHECKING:
    import os

    from matplotlib.axes import Axes

    from foambryo.dcel import DcelData

#: The tension colour map: matplotlib's `coolwarm`, blue (low) through light grey (the mean) to red (high).
TENSION_COLOUR_MAP = "coolwarm"

#: The pressure colour map: matplotlib's `magma`, reversed (light for low pressures, dark for high ones), and
#: trimmed: a pressure fraction `x` in [0, 1] is drawn `magma(lo + (hi - lo) x)` with `(lo, hi)` this range. The
#: whole map runs from black to near-white, so one end vanished on a black background and the other on a white
#: one; trimmed, no colour is pure black or near-white. The nearest colour of the trimmed map is 31 (CIELAB) from
#: black and 40 from white, against 1.4 and 31 for the whole map.
PRESSURE_COLOUR_MAP = "magma"
PRESSURE_COLOUR_MAP_RANGE = (0.10, 0.90)

#: The colour of an undetermined (nan) value, in the viewers and on the bars: green (#1b9e77). Its CIELAB distance
#: to the nearest colour of `coolwarm` is 54 and of the trimmed `magma` 64 (the former grey, 0.6, was 25 from
#: `coolwarm`'s centre); its contrast is 3.4 against white and 6.2 against black.
UNDETERMINED_COLOR = np.array([27, 158, 119]) / 255

#: Crop faces are see-through glass: this fill at `CROP_FACE_OPACITY`, with their edges drawn in
#: `CROP_FACE_EDGE_COLOR` (contrast 4.0 against white, 5.3 against black). The fill is close to `coolwarm`'s centre:
#: what tells a crop face from a value is that it is see-through and edged, not its colour.
CROP_FACE_COLOR = np.array([0.80, 0.80, 0.80])
CROP_FACE_OPACITY = 0.15
CROP_FACE_EDGE_COLOR = np.array([0.50, 0.50, 0.50])

#: The backgrounds every viewer offers (the Polyscope window's panel, the napari plugin's View dock), by name: one
#: background hides colours that another shows. The two greys are chosen against the maps: `coolwarm`'s middle (the
#: mean tension) is light grey, 12 (CIELAB) from white and 1.4 from a 0.85 grey, so the light grey is 0.70, the
#: lightest no closer to it than white (15); the dark grey, 0.20, is at least 32 from every colour of both maps, from
#: the undetermined green and from the crop faces' edges.
BACKGROUND_COLOURS: dict[str, tuple[float, float, float]] = {
    "black": (0.0, 0.0, 0.0),
    "dark grey": (0.20, 0.20, 0.20),
    "light grey": (0.70, 0.70, 0.70),
    "white": (1.0, 1.0, 1.0),
}

#: Quantile clipped at each end of the colour scales, the viewer's default `alpha`.
DEFAULT_CLIP_QUANTILE = 0.05

#: Samples along a bar: enough that neighbouring samples are not distinguishable by eye.
_BAR_SAMPLES = 512


def quantile_colour_range(values: NDArray[np.float64], alpha: float) -> tuple[np.float64, np.float64]:
    """The range the viewer colours a per-interface quantity other than a tension over: two quantiles.

    The `alpha` and `1 - alpha` quantiles of the finite values.

    Used for the per-interface quantities the viewer draws with `jet` (`quantity_colours`): a seed spread, a
    residual. Tensions have their own range, `tension_colour_range`.

    Args:
        values: the values, for tensions one per interface; `nan` values take no part in the range.
        alpha: quantile clipped at each end, in [0, 0.5).

    Returns:
        tuple: `(lowest, highest)`, the two quantiles of the finite values.

    Raises:
        IndexError: if no value is finite (numpy's error on an empty quantile).
    """
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return np.quantile(finite, alpha), np.quantile(finite, 1 - alpha)


def tension_colour_range(
    values: NDArray[np.float64],
    alpha: float = DEFAULT_CLIP_QUANTILE,
    *,
    centre: float | None = None,
) -> tuple[float, float]:
    """The range the viewers colour tensions over: centred on the mean tension, symmetric around it.

    The centre is the mean of the finite tensions, which is `mean_tension` (1 by default) for every method that
    meets its gauge exactly (`infer_tensions`; per junction-graph component, so also over several components). The
    half-width is the larger of the distances from the centre to the `alpha` and the `1 - alpha` quantiles, so that
    both quantiles lie within the range and a tension equal to the mean is drawn in the map's middle colour (light
    grey).

    Args:
        values: the tensions, one per interface; `nan` values take no part.
        alpha: quantile at each end, in [0, 0.5) (the viewer's `alpha`, default 0.05).
        centre: the centre instead of the mean (for example a known `mean_tension`). Keyword-only.

    Returns:
        tuple: `(centre - half, centre + half)`. With every finite tension equal, `(centre, centre)`.

    Raises:
        ValueError: if no tension is finite.
    """
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not finite.size:
        message = "no tension is finite: there is no tension colour range"
        raise ValueError(message)
    middle = float(np.mean(finite)) if centre is None else float(centre)
    low, high = np.quantile(finite, alpha), np.quantile(finite, 1 - alpha)
    half = float(max(middle - low, high - middle, 0.0))
    return middle - half, middle + half


def tension_colour_fractions(values: NDArray[np.float64], colour_range: tuple[float, float]) -> NDArray[np.float64]:
    """Where each tension falls on the tension map, in [0, 1] (0.5 is the mean); `nan` stays `nan`.

    `clipped_colour_fractions` on `tension_colour_range`'s range; when the range has no width (every tension equal),
    every finite tension is at the middle, 0.5, rather than undetermined.
    """
    lowest, highest = colour_range
    values = np.asarray(values, dtype=np.float64)
    if not highest > lowest:
        return np.where(np.isfinite(values), 0.5, np.nan)
    return clipped_colour_fractions(values, colour_range)


def clipped_colour_fractions(values: NDArray[np.float64], colour_range: tuple[float, float]) -> NDArray[np.float64]:
    """Where each value falls on a colour scale, in [0, 1]: clipped to the range, then stretched.

    Args:
        values: the values to colour (per triangle on the viewer's surface); `nan` stays `nan`.
        colour_range: `(lowest, highest)`, from `quantile_colour_range` or `tension_colour_range`.

    Returns:
        NDArray[np.float64]: `(clip(v, lowest, highest) - lowest) / (highest - lowest)`. A range of zero width
        gives `nan` (and numpy's division warning); `tension_colour_fractions` handles that case for tensions.
    """
    lowest, highest = colour_range
    fractions = np.asarray(values, dtype=np.float64).clip(lowest, highest)
    fractions -= np.amin(lowest)
    fractions /= np.amax(highest - lowest)
    return fractions


def pressure_colour_upper(values: NDArray[np.float64], alpha: float) -> np.float64:
    """The top of the range the viewer colours pressures over: the `1 - alpha` quantile.

    The bottom is 0: the viewer clips negative pressures to 0.

    Args:
        values: pressures; on the viewer's surface, one per triangle (the pressure of the
            higher-labelled of its two materials). `nan` values take no part.
        alpha: quantile clipped at the top.

    Returns:
        np.float64: the `1 - alpha` quantile of the finite values.

    Raises:
        IndexError: if no value is finite (numpy's error on an empty quantile).
    """
    values = np.asarray(values, dtype=np.float64)
    return np.quantile(values[np.isfinite(values)], 1 - alpha)


def pressure_colour_fractions(values: NDArray[np.float64], upper: float) -> NDArray[np.float64]:
    """Where each pressure falls on the reversed pressure scale: 0 for the top, 1 for 0 and below.

    Args:
        values: the pressures to colour; `nan` stays `nan`.
        upper: the top of the range, from `surface_pressure_colour_upper`.

    Returns:
        NDArray[np.float64]: `1 - clip(v, 0, upper) / upper`, the fraction `pressure_colours` draws
        (on the trimmed `magma`, `PRESSURE_COLOUR_MAP_RANGE`). Dividing by `upper` itself, rather than by the
        largest clipped value (the same number whenever a value reaches `upper`, as on the surface), gives one
        pressure one colour wherever it is drawn, also on the graph view, whose cells need not reach the
        surface's top. With `upper` 0 (no positive pressure) the fractions are `nan`.
    """
    fractions = np.asarray(values, dtype=np.float64).clip(0, upper)
    fractions /= upper
    return 1 - fractions


def tension_colours(fractions: NDArray[np.float64]) -> NDArray[np.float64]:
    """RGB colours of tension fractions (`tension_colour_fractions`): `coolwarm`; `nan` is `UNDETERMINED_COLOR`."""
    colours = cm.coolwarm(fractions)[:, :3]
    colours[np.isnan(fractions)] = UNDETERMINED_COLOR
    return colours


def quantity_colours(fractions: NDArray[np.float64]) -> NDArray[np.float64]:
    """RGB colours of another per-interface quantity's fractions (a seed spread, a residual): `jet`; `nan` green."""
    colours = cm.jet(fractions)[:, :3]
    colours[np.isnan(fractions)] = UNDETERMINED_COLOR
    return colours


def pressure_colours(fractions: NDArray[np.float64]) -> NDArray[np.float64]:
    """RGB colours of pressure fractions (`pressure_colour_fractions`), as every front end draws them.

    A fraction `x` in [0, 1] is `magma(lo + (hi - lo) x)`, `(lo, hi) = PRESSURE_COLOUR_MAP_RANGE` (0.1, 0.9): 0 (the
    top of the range) is dark purple, 1 (a pressure of 0 or below) light orange, and no pressure is black or
    near-white. `nan` is `UNDETERMINED_COLOR`.
    """
    lo, hi = PRESSURE_COLOUR_MAP_RANGE
    colours = colormaps[PRESSURE_COLOUR_MAP](lo + (hi - lo) * np.asarray(fractions, dtype=np.float64))[:, :3]
    colours[np.isnan(fractions)] = UNDETERMINED_COLOR
    return colours


def surface_face_pressures(
    mesh: DcelData,
    dict_pressures: dict[int, float],
    *,
    scattered: bool = False,
) -> NDArray[np.float64]:
    """The pressure the viewer's surface gives each triangle: that of its higher-labelled material.

    Args:
        mesh: the mesh.
        dict_pressures: pressure per material (0 = medium).
        scattered: the triangles of the scattered view (`mesh.f_scattered`, each carrying the
            pressure of the cell it was scattered with), as the viewer draws them with
            `scattered=True`. Keyword-only. Defaults to `False`.

    Returns:
        NDArray[np.float64]: one pressure per row of `mesh.f` (or of `mesh.f_scattered`).

    Raises:
        KeyError: if a material of the mesh has no pressure in `dict_pressures`.
    """
    if scattered:
        return np.array([dict_pressures[idx] for idx in mesh.cluster_idx_scattered], dtype=np.float64)
    return np.array([dict_pressures[max(face[3], face[4])] for face in mesh.f], dtype=np.float64)


def surface_pressure_colour_upper(
    mesh: DcelData,
    dict_pressures: dict[int, float],
    alpha: float = DEFAULT_CLIP_QUANTILE,
    *,
    scattered: bool = False,
) -> np.float64 | None:
    """The top of the viewer's pressure scale, for every view that colours pressures.

    The one place the pressure range is computed: the viewer's surface ("Pressures Cells"), its
    graph view's cell nodes ("Graph Nodes") and the colour bars all take it from here, so one
    pressure has one colour in all three. It is the `1 - alpha` quantile of the finite pressures
    over the triangles the surface draws (`surface_face_pressures`); the bottom is 0.

    Args:
        mesh: the mesh the pressures belong to.
        dict_pressures: pressure per material (0 = medium).
        alpha: quantile clipped at the top.
        scattered: over the scattered view's triangles, as the surface drawn with `scattered=True`.
            Keyword-only. Defaults to `False`.

    Returns:
        np.float64 | None: the top of the scale, or `None` when no pressure is finite.
    """
    face_pressures = surface_face_pressures(mesh, dict_pressures, scattered=scattered)
    if not np.isfinite(face_pressures).any():
        return None
    return pressure_colour_upper(face_pressures, alpha)


@dataclass(frozen=True)
class ColourBarRanges:
    """The values at the two ends of each colour bar, which are the viewer's clipped ranges.

    Attributes:
        tension: `(lowest, highest)` of the tension bar (`tension_colour_range`: centred on the mean tension), or
            `None` when no tension is finite.
        pressure: `(0, highest)` of the pressure bar, or `None` when no pressure was given or none
            is finite.
        alpha: the quantile clipped at each end.
    """

    tension: tuple[float, float] | None
    pressure: tuple[float, float] | None
    alpha: float


def colour_bar_ranges(
    mesh: DcelData,
    dict_tensions: dict[tuple[int, int], float],
    dict_pressures: dict[int, float] | None = None,
    alpha: float = DEFAULT_CLIP_QUANTILE,
) -> ColourBarRanges:
    """The ranges the viewer's surface colours tensions and pressures over, without drawing anything.

    Args:
        mesh: the mesh the values belong to (its triangles weight the pressure quantile).
        dict_tensions: tension per interface, keyed `(label_1, label_2)` with `label_1 < label_2`.
        dict_pressures: pressure per material (0 = medium), or `None` for no pressure bar.
        alpha: quantile clipped at each end (the viewer's `alpha`). Defaults to 0.05.

    Returns:
        ColourBarRanges: the two ranges.
    """
    tensions = np.array(list(dict_tensions.values()), dtype=np.float64)
    tension = None
    if np.isfinite(tensions).any():
        tension = tension_colour_range(tensions, alpha)
    pressure = None
    if dict_pressures is not None:
        upper = surface_pressure_colour_upper(mesh, dict_pressures, alpha)
        if upper is not None:
            pressure = (0.0, float(upper))
    return ColourBarRanges(tension=tension, pressure=pressure, alpha=alpha)


def save_colour_bars(
    mesh: DcelData,
    dict_tensions: dict[tuple[int, int], float],
    dict_pressures: dict[int, float] | None,
    path: str | os.PathLike[str],
    alpha: float = DEFAULT_CLIP_QUANTILE,
    *,
    tension_label: str = "Surface tension",
    pressure_label: str = "Pressure",
) -> ColourBarRanges:
    """Write the colour bars of the force viewer's surface colours to an image file.

    One horizontal bar per quantity, each spanning exactly the range the viewer colours over
    (`colour_bar_ranges`), drawn with the viewer's own colour function, with its two end values
    written under it:

    * **surface tension**: `coolwarm`, centred on the mean of the finite tensions (one per interface; light grey),
      symmetric, its half-width the farther of the `alpha` and `1 - alpha` quantiles from the mean
      (`tension_colour_range`); the viewer draws tensions beyond either end in the end's colour; the bar's middle
      tick is the mean;
    * **pressure**: `magma` reversed (light for low) and trimmed to its 10-90 % range, so that neither end is black
      or near-white, from 0 to the `1 - alpha` quantile of the
      finite pressures over the mesh's triangles, each carrying the pressure of the higher-labelled
      of its two materials, as on the viewer's surface; negative pressures are drawn as 0.

    A green swatch, "undetermined", is added when any value is `nan`, and a glass swatch, "crop face", when the mesh
    has crop faces. The viewer's graph view (cell nodes) colours pressures on this same scale
    (`surface_pressure_colour_upper`).

    Headless: matplotlib's non-interactive canvas, no window, no Polyscope.

    Args:
        mesh: the mesh the values belong to.
        dict_tensions: tension per interface (`(label_1, label_2) -> tension`, `label_1 < label_2`).
        dict_pressures: pressure per material (`label -> pressure`, 0 = medium), or `None` for the
            tension bar only.
        path: the image file; matplotlib chooses the format from the suffix (`.png`, `.pdf`, `.svg`).
        alpha: quantile clipped at each end, the viewer's `alpha` (default 0.05).
        tension_label: the tension bar's title (for example with a unit). Keyword-only.
        pressure_label: the pressure bar's title. Keyword-only.

    Returns:
        ColourBarRanges: the values at the ends of each bar (`pressure` is `None` when no pressure bar was
        drawn).

    Raises:
        ValueError: if no tension is finite (there is no scale to draw), or matplotlib does not know the
            file's suffix.
    """
    from matplotlib.figure import Figure

    ranges = colour_bar_ranges(mesh, dict_tensions, dict_pressures, alpha)
    if ranges.tension is None:
        message = "no tension is finite: there is no colour scale to draw"
        raise ValueError(message)
    any_undetermined = not np.isfinite(np.array(list(dict_tensions.values()), dtype=np.float64)).all()
    bars = [(tension_label, ranges.tension, _tension_bar_colours(ranges.tension))]
    if ranges.pressure is not None:
        bars.append((pressure_label, ranges.pressure, _pressure_bar_colours(ranges.pressure[1])))
    if dict_pressures is not None:
        any_undetermined |= not np.isfinite(np.array(list(dict_pressures.values()), dtype=np.float64)).all()
    with_crop = bool(getattr(mesh, "has_crop_faces", False))
    legend = any_undetermined or with_crop

    figure = Figure(figsize=(5.0, 1.15 * len(bars) + (0.35 if legend else 0.0)), dpi=200)
    rows = len(bars) + (1 if legend else 0)
    heights = [1.0] * len(bars) + ([0.45] if legend else [])
    grid = figure.add_gridspec(rows, 1, height_ratios=heights, hspace=1.6, left=0.08, right=0.92, top=0.82,
                               bottom=0.18)
    for row, (label, (lowest, highest), colours) in enumerate(bars):
        axes = figure.add_subplot(grid[row])
        # A zero-width range (every value equal, or no positive pressure) is drawn over a unit
        # interval around it so that the image has an extent; its end labels still say the range.
        left, right = (lowest, highest) if highest > lowest else (lowest - 0.5, lowest + 0.5)
        axes.imshow(colours[np.newaxis], aspect="auto", extent=(left, right, 0, 1), interpolation="nearest")
        axes.set_xlim(left, right)
        axes.set_yticks([])
        if row == 0:
            middle = (lowest + highest) / 2
            axes.set_xticks([left, (left + right) / 2, right])
            axes.set_xticklabels([f"{lowest:.4g}", f"{middle:.4g}", f"{highest:.4g}"])
            title = (f"{label} (centred on the mean, {middle:.4g}; symmetric, to the farther of the "
                     f"{100 * alpha:g}% and {100 * (1 - alpha):g}% quantiles)")
        else:
            axes.set_xticks([left, right])
            axes.set_xticklabels([f"{lowest:.4g}", f"{highest:.4g}"])
            title = f"{label} (from 0, clipped at the {100 * (1 - alpha):g}% quantile over triangles)"
        axes.set_title(title, fontsize=7 if row else 6.2, loc="left")
        axes.tick_params(labelsize=7)
    if legend:
        _draw_legend(figure.add_subplot(grid[len(bars)]), undetermined=any_undetermined, crop=with_crop)
    figure.savefig(Path(path))
    return ranges


def _draw_legend(axes: Axes, *, undetermined: bool, crop: bool) -> None:
    """The swatches under the bars: undetermined (green, outlined, no fill) and crop face (glass with edges)."""
    from matplotlib.patches import Rectangle

    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.set_axis_off()
    x = 0.0
    if undetermined:
        axes.add_patch(Rectangle((x, 0.1), 0.06, 0.8, facecolor="none", edgecolor=UNDETERMINED_COLOR, linewidth=1.5))
        axes.text(x + 0.08, 0.5, "undetermined (nan): green outline, or green", fontsize=7, va="center")
        x = 0.55
    if crop:
        axes.add_patch(Rectangle((x, 0.1), 0.06, 0.8, facecolor=(*CROP_FACE_COLOR, CROP_FACE_OPACITY),
                                 edgecolor=CROP_FACE_EDGE_COLOR, linewidth=1.0, hatch="//"))
        axes.text(x + 0.08, 0.5, "crop face (no value)", fontsize=7, va="center")


def _tension_bar_colours(colour_range: tuple[float, float]) -> NDArray[np.float64]:
    """The colours along the tension bar, from its lowest to its highest value."""
    lowest, highest = colour_range
    samples = np.linspace(lowest, highest, _BAR_SAMPLES)
    return tension_colours(tension_colour_fractions(samples, colour_range))


def _quantity_bar_colours(colour_range: tuple[float, float]) -> NDArray[np.float64]:
    """The colours along another quantity's bar (`jet`), from its lowest to its highest value."""
    lowest, highest = colour_range
    samples = np.linspace(lowest, highest, _BAR_SAMPLES)
    with np.errstate(invalid="ignore", divide="ignore"):  # a zero-width range is undetermined, as the viewer draws it
        return quantity_colours(clipped_colour_fractions(samples, colour_range))


def _pressure_bar_colours(upper: float) -> NDArray[np.float64]:
    """The colours along the pressure bar, from 0 to `upper`."""
    samples = np.linspace(0.0, upper, _BAR_SAMPLES)
    with np.errstate(invalid="ignore", divide="ignore"):  # a zero-width range is undetermined, as the viewer draws it
        return pressure_colours(pressure_colour_fractions(samples, upper))
