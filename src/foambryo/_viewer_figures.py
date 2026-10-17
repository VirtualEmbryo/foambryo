"""What the Polyscope viewer adds to its window and its saved figures, computed without Polyscope.

The viewer (`foambryo.viewing`) colours its surfaces with the colour functions of `foambryo._colour_scales`
and hands Polyscope the colours, not the values: Polyscope would draw an undetermined value (`nan`) in the top
colour of a scalar map, where the library draws it in its own colour,
outside every map. So Polyscope cannot draw the colour bars itself. This
module computes, with no Polyscope and no display:

* **the colour scales** of a view (`viewer_colour_scales`): one per coloured quantity, each the range the
  library's functions clip that quantity to, and each sampled with the library's own bar colours (the samples
  `save_colour_bars` draws); the viewer draws them in its panel, and `compose_figure` under a saved view;
* **a length scale bar** (`scale_bar_for_points`): a segment of round length in the mesh's unit, placed under the
  tissue, and where it falls on a rendered image (`project_to_pixels`);
* **a saved figure** (`compose_figure`): a rendered view of the window (an RGB array) with its colour bars and
  the scale bar's length written beside the segment, through matplotlib's non-interactive canvas;
* **seed-ensemble tensions on one mesh** (`ensemble_on_mesh`): the tensions combined over seeds, matched to
  the interfaces of the mesh being viewed by their label pair.

Nothing here changes a colour: the scales are read from `colour_bar_ranges` and
`surface_pressure_colour_upper`, the colours from the library's colour functions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np

from foambryo._colour_scales import (
    CROP_FACE_COLOR,
    CROP_FACE_EDGE_COLOR,
    CROP_FACE_OPACITY,
    UNDETERMINED_COLOR,
    _pressure_bar_colours,
    _quantity_bar_colours,
    _tension_bar_colours,
    colour_bar_ranges,
    quantile_colour_range,
    surface_pressure_colour_upper,
)

if TYPE_CHECKING:
    import os
    from collections.abc import Iterable, Mapping

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from foambryo.dcel import DcelData


#: How the viewer names its two scales; also the titles of the bars.
TENSION_TITLE = "Surface tension"
PRESSURE_TITLE = "Pressure"
SEED_SPREAD_TITLE = "Tension spread over the seeds (SD)"

#: Units written differently on screen than in a file (`.rec` appendices hold ASCII words).
_DISPLAY_UNITS = {"um": "µm", "micron": "µm", "microns": "µm", "micrometer": "µm", "micrometre": "µm"}

#: The lengths a scale bar may have, per decade.
_ROUND_LENGTHS = (1.0, 2.0, 5.0)


@dataclass(frozen=True)
class ColourScale:
    """One colour bar: a quantity, the range its colours span, and which library colour function colours it.

    Attributes:
        title: the quantity, as the bar is titled.
        low: the value at the bar's left end (the colour of every value at or below it).
        high: the value at the bar's right end (the colour of every value at or above it).
        colour_function: `"tension"` (`coolwarm` between `low` and `high`, centred on the mean tension,
            `tension_colours`), `"pressure"` (reversed `magma` from 0 to `high`, `pressure_colours`) or
            `"quantity"` (`jet` between `low` and `high`, `quantity_colours`: a seed spread).
        clipping: how the range was chosen, in words, for the bar's caption.
    """

    title: str
    low: float
    high: float
    colour_function: Literal["tension", "pressure", "quantity"]
    clipping: str

    def colours(self) -> NDArray[np.float64]:
        """The bar's colours from `low` to `high`: the samples `save_colour_bars` draws, `(512, 3)` RGB in [0, 1]."""
        if self.colour_function == "tension":
            return _tension_bar_colours((self.low, self.high))
        if self.colour_function == "quantity":
            return _quantity_bar_colours((self.low, self.high))
        return _pressure_bar_colours(self.high)


@dataclass(frozen=True)
class ViewerColourScales:
    """The colour bars of one view, and which swatches of what has no value belong with them.

    Attributes:
        scales: one `ColourScale` per coloured quantity, tension first.
        undetermined: whether any value drawn is `nan`, or an interface of the mesh has no tension (drawn in
            `UNDETERMINED_COLOR`: an interface as its outline, a cell filled).
        alpha: the quantile clipped at each end.
        crop: whether the mesh has crop faces (drawn as glass, `CROP_FACE_COLOR`).
    """

    scales: tuple[ColourScale, ...]
    undetermined: bool
    alpha: float
    crop: bool = False

    def scale(self, title: str) -> ColourScale | None:
        """The scale titled `title`, or `None`."""
        return next((scale for scale in self.scales if scale.title == title), None)


def _any_nan(values: Iterable[float]) -> bool:
    return not np.isfinite(np.array(list(values), dtype=np.float64)).all()


def viewer_colour_scales(
    mesh: DcelData,
    dict_tensions: Mapping[tuple[int, int], float] | None,
    dict_pressures: Mapping[int, float] | None,
    alpha: float,
    *,
    scattered: bool = False,
) -> ViewerColourScales:
    """The colour scales of the force viewer for these values: the ranges its surfaces and graph are coloured over.

    The tension range and, without `scattered`, the pressure range are `colour_bar_ranges`' (so the bars equal
    those `save_colour_bars` writes); with `scattered`, the pressure range is `surface_pressure_colour_upper` over
    the scattered view's triangles, as that view colours them. A quantity with no finite value has no scale.

    Args:
        mesh: the mesh the values belong to.
        dict_tensions: tension per interface, or `None` for no tension scale.
        dict_pressures: pressure per material (0 = medium), or `None` for no pressure scale.
        alpha: the quantile clipped at each end (the viewer's `alpha`).
        scattered: the scales of the scattered view. Keyword-only.

    Returns:
        ViewerColourScales: the scales, tension first, and whether a value is undetermined.
    """
    pressures = None if scattered or dict_pressures is None else dict(dict_pressures)
    ranges = colour_bar_ranges(mesh, dict(dict_tensions or {}), pressures, alpha)
    pressure = ranges.pressure
    if scattered and dict_pressures is not None:
        upper = surface_pressure_colour_upper(mesh, dict(dict_pressures), alpha, scattered=True)
        pressure = None if upper is None else (0.0, float(upper))
    percent = f"{100 * alpha:g}%"
    top = f"{100 * (1 - alpha):g}%"
    scales = []
    if dict_tensions is not None and ranges.tension is not None:
        middle = (ranges.tension[0] + ranges.tension[1]) / 2
        scales.append(ColourScale(TENSION_TITLE, *ranges.tension, "tension",
                                  f"centred on the mean, {middle:.4g}; to the farther of the {percent} and {top} "
                                  "quantiles"))
    if pressure is not None:
        over = "the scattered view's triangles" if scattered else "triangles"
        scales.append(ColourScale(PRESSURE_TITLE, *pressure, "pressure",
                                  f"from 0, clipped at the {top} quantile over {over}"))
    undetermined = (dict_tensions is not None and (_any_nan(dict_tensions.values()) or bool(
        interfaces_on_mesh(mesh) - set(dict_tensions)))) or (
        dict_pressures is not None and _any_nan(dict_pressures.values()))
    crop = bool(getattr(mesh, "has_crop_faces", False))
    return ViewerColourScales(scales=tuple(scales), undetermined=bool(undetermined), alpha=alpha, crop=crop)


def quantity_colour_scale(title: str, values: Mapping[tuple[int, int], float], alpha: float) -> ColourScale | None:
    """The scale of another per-interface quantity the viewer colours with `jet`, such as a seed spread.

    Args:
        title: the bar's title.
        values: one value per interface; `nan` takes no part.
        alpha: the quantile clipped at each end.

    Returns:
        ColourScale | None: `jet` between the `alpha` and `1 - alpha` quantiles, or `None` with no finite value.
    """
    array = np.array(list(values.values()), dtype=np.float64)
    if not np.isfinite(array).any():
        return None
    low, high = quantile_colour_range(array, alpha)
    return ColourScale(title, float(low), float(high), "quantity",
                       f"clipped at the {100 * alpha:g}% and {100 * (1 - alpha):g}% quantiles")


# ---------------------------------------------------------------------------------------------
# the length scale bar
# ---------------------------------------------------------------------------------------------
def display_unit(unit: str) -> str:
    """How a length unit is written on screen: `um` (the `.rec` spelling) as `µm`; any other as it is."""
    return _DISPLAY_UNITS.get(unit, unit)


def format_length(length: float, unit: str) -> str:
    """`10 µm`: a scale bar's label."""
    return f"{length:g} {display_unit(unit)}"


def round_scale_bar_length(extent: float) -> float:
    """The longest of 1, 2 or 5 times a power of ten that is at most a quarter of `extent`.

    Args:
        extent: the tissue's largest side, in the mesh's unit; finite and positive.

    Returns:
        float: the scale bar's length, between an eighth and a quarter of `extent`.

    Raises:
        ValueError: if `extent` is not finite and positive.
    """
    if not (math.isfinite(extent) and extent > 0):
        message = f"the extent must be finite and positive, not {extent!r}"
        raise ValueError(message)
    target = extent / 4
    decade = 10.0 ** math.floor(math.log10(target))
    return max(step * decade for step in _ROUND_LENGTHS if step * decade <= target * (1 + 1e-12))


@dataclass(frozen=True)
class ScaleBar:
    """A length scale bar drawn in the scene: a segment of `length` mesh units, from `start` to `end`.

    Attributes:
        length: the segment's length, in `unit`.
        unit: the mesh's length unit, as recorded (`um`, ...).
        start: the segment's first end, in mesh coordinates.
        end: the segment's other end: `start + (length, 0, 0)`.
    """

    length: float
    unit: str
    start: NDArray[np.float64]
    end: NDArray[np.float64]

    @property
    def label(self) -> str:
        """The length written beside the segment, `10 µm`."""
        return format_length(self.length, self.unit)


def scale_bar_for_points(points: NDArray[np.float64], unit: str, length: float | None = None) -> ScaleBar:
    """A scale bar under a tissue: along the first coordinate axis, below the points along the second.

    Polyscope's home view looks along its third axis with its second axis up, so the segment is horizontal and
    under the tissue in that view; its length is exact in 3-D, and foreshortened on screen like any length in a
    perspective view of the scene.

    Args:
        points: the mesh's points, `(n, 3)`, in the mesh's unit.
        unit: that unit.
        length: the segment's length; `None` (the default) chooses `round_scale_bar_length` of the largest side.

    Returns:
        ScaleBar: the segment, starting at the tissue's lowest first coordinate, 8 % of the largest side below
        its lowest second coordinate, half-way along the third.

    Raises:
        ValueError: if `length` is given and is not finite and positive, or the points have no extent.
    """
    points = np.asarray(points, dtype=np.float64)
    low, high = points.min(axis=0), points.max(axis=0)
    extent = float(np.max(high - low))
    if length is None:
        length = round_scale_bar_length(extent)
    elif not (math.isfinite(length) and length > 0):
        message = f"the scale bar's length must be finite and positive, not {length!r}"
        raise ValueError(message)
    start = np.array([low[0], low[1] - 0.08 * extent, 0.5 * (low[2] + high[2])])
    end = start + np.array([float(length), 0.0, 0.0])
    return ScaleBar(length=float(length), unit=unit, start=start, end=end)


def project_to_pixels(
    points: NDArray[np.float64],
    view_matrix: NDArray[np.float64],
    fov_vertical_deg: float,
    width: int,
    height: int,
) -> NDArray[np.float64]:
    """Where points of the scene fall on a perspective image, in pixels (x to the right, y down).

    The OpenGL convention Polyscope uses: the view matrix takes a point to camera coordinates, the camera looks
    along its -z axis with +y up, and the vertical field of view spans the image's height.

    Args:
        points: `(n, 3)` scene coordinates.
        view_matrix: the camera's 4 x 4 view matrix (`CameraParameters.get_view_mat()`).
        fov_vertical_deg: the vertical field of view, in degrees.
        width: the image's width, in pixels.
        height: the image's height, in pixels.

    Returns:
        NDArray[np.float64]: `(n, 2)` pixel coordinates; `nan` for a point behind the camera.
    """
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    homogeneous = np.c_[points, np.ones(len(points))]
    camera = homogeneous @ np.asarray(view_matrix, dtype=np.float64).T
    depth = -camera[:, 2]
    focal = 1.0 / math.tan(math.radians(fov_vertical_deg) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_ndc = focal * (height / width) * camera[:, 0] / depth
        y_ndc = focal * camera[:, 1] / depth
    pixels = np.c_[(x_ndc + 1) / 2 * width, (1 - y_ndc) / 2 * height]
    pixels[depth <= 0] = np.nan
    return pixels


# ---------------------------------------------------------------------------------------------
# a saved figure: the view, its colour bars and its scale bar
# ---------------------------------------------------------------------------------------------
def _draw_bar(axes: Axes, scale: ColourScale) -> None:
    """One bar, as `save_colour_bars` draws it: the samples over `[low, high]`, the two ends written under it."""
    low, high = scale.low, scale.high
    left, right = (low, high) if high > low else (low - 0.5, low + 0.5)
    axes.imshow(scale.colours()[np.newaxis], aspect="auto", extent=(left, right, 0, 1), interpolation="nearest")
    axes.set_xlim(left, right)
    axes.set_yticks([])
    if scale.colour_function == "tension":  # the middle is the mean tension
        axes.set_xticks([left, (left + right) / 2, right])
        axes.set_xticklabels([f"{low:.4g}", f"{(low + high) / 2:.4g}", f"{high:.4g}"])
    else:
        axes.set_xticks([left, right])
        axes.set_xticklabels([f"{low:.4g}", f"{high:.4g}"])
    axes.set_title(f"{scale.title} ({scale.clipping})", fontsize=7, loc="left")
    axes.tick_params(labelsize=7)


def compose_figure(
    image: NDArray[np.uint8] | None,
    scales: ViewerColourScales,
    path: str | os.PathLike[str],
    **options: Any,  # noqa: ANN401 - figure_of_the_view's keyword arguments, passed through
) -> Path:
    """Write a rendered view with its colour bars under it, and its scale bar's length beside the segment.

    Headless: matplotlib's non-interactive canvas, no window. The view is embedded at one image pixel per
    figure pixel in a PNG (`dpi` figure pixels per inch), and as an image in a PDF or an SVG; the bars and the
    text are vector graphics in those. The arguments are `figure_of_the_view`'s, and `path`, the file:
    matplotlib chooses the format from its suffix (`.png`, `.pdf`, `.svg`).

    Returns:
        Path: the file written.
    """
    path = Path(path)
    figure_of_the_view(image, scales, **options).savefig(path)
    return path


def figure_of_the_view(
    image: NDArray[np.uint8] | None,
    scales: ViewerColourScales,
    *,
    scale_bar: ScaleBar | None = None,
    scale_bar_pixels: NDArray[np.float64] | None = None,
    caption: str | None = None,
    labels: Iterable[tuple[str, float, float]] = (),
    dpi: int = 150,
) -> Figure:
    """The figure `compose_figure` writes: the view, one bar per scale, the swatches of what has no value, the caption.

    Args:
        image: the rendered view, `(height, width, 3)` (or 4, the alpha dropped) 8-bit RGB, top row first;
            `None` writes the colour bars alone (5 inches wide).
        scales: the colour bars to draw under it.
        scale_bar: the scale bar drawn in the view, whose length is written beside it. Keyword-only.
        scale_bar_pixels: where its two ends fall on `image` (`project_to_pixels`), `(2, 2)`; without them, or
            when an end is off the image, the length is written in the image's lower left corner. Keyword-only.
        caption: a line written under the bars (for example which methods produced the values). Keyword-only.
        labels: `(text, x, y)` written on the view at those pixels (the x, y, z frame's). Keyword-only.
        dpi: figure pixels per inch. Keyword-only.

    Returns:
        Figure: a matplotlib figure on its non-interactive canvas (no `pyplot`, no window).
    """
    from matplotlib.figure import Figure

    bars = list(scales.scales)
    if image is None:
        width_inches, view_inches = 5.0, 0.0
    else:
        image = np.asarray(image)[..., :3]
        height, width = image.shape[:2]
        width_inches, view_inches = width / dpi, height / dpi
    # Heights in inches, from the top: the view, then per bar its title (0.22), the bar (0.16) and its end values
    # (0.27), the swatches of what has no value, the caption.
    bar_row, bar_height, swatch_row = 0.65, 0.16, 0.3
    caption_row = 0.5 if caption else 0.0
    legend = scales.undetermined or scales.crop
    total = view_inches + bar_row * len(bars) + (swatch_row if legend else 0.0) + caption_row + 0.1
    figure = Figure(figsize=(width_inches, total), dpi=dpi)
    side = min(0.08, 0.5 / width_inches)

    def rect(
        top_inches: float, height_inches: float, left: float = side, right: float = side,
    ) -> tuple[float, float, float, float]:
        return (left, 1 - (top_inches + height_inches) / total, 1 - left - right, height_inches / total)

    if image is not None:
        view = figure.add_axes(rect(0.0, view_inches, 0.0, 0.0))
        view.imshow(image, interpolation="nearest")
        view.set_axis_off()
        if scale_bar is not None:
            _write_scale_bar_label(view, scale_bar, scale_bar_pixels, width, height)
        _write_labels(view, labels, width, height)
    top = view_inches
    for scale in bars:
        _draw_bar(figure.add_axes(rect(top + 0.22, bar_height)), scale)
        top += bar_row
    if legend:
        draw_no_value_legend(figure.add_axes(rect(top + 0.05, 0.2)), undetermined=scales.undetermined,
                             crop=scales.crop)
        top += swatch_row
    if caption:
        axes = figure.add_axes(rect(top, caption_row))
        axes.text(0.0, 1.0, caption, fontsize=6, va="top", ha="left", wrap=True)
        axes.set_axis_off()
    return figure


def draw_no_value_legend(axes: Axes, *, undetermined: bool, crop: bool) -> None:
    """The swatches of what has no value: an undetermined interface (a green outline) and a crop face (glass)."""
    from matplotlib.patches import Rectangle

    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.set_axis_off()
    x = 0.0
    if undetermined:
        axes.add_patch(Rectangle((x, 0.1), 0.04, 0.8, facecolor="none", edgecolor=UNDETERMINED_COLOR, linewidth=1.5))
        axes.text(x + 0.06, 0.5, "undetermined (nan): green outline (a cell: green)", fontsize=7, va="center")
        x = 0.55
    if crop:
        axes.add_patch(Rectangle((x, 0.1), 0.04, 0.8, facecolor=(*CROP_FACE_COLOR, CROP_FACE_OPACITY),
                                 edgecolor=CROP_FACE_EDGE_COLOR, linewidth=1.0, hatch="//"))
        axes.text(x + 0.06, 0.5, "crop face (no value)", fontsize=7, va="center")


def _write_scale_bar_label(
    view: Axes,
    scale_bar: ScaleBar,
    scale_bar_pixels: NDArray[np.float64] | None,
    width: int,
    height: int,
) -> None:
    """Write the scale bar's length just under the segment, or in the lower left corner when it is off the view."""
    from matplotlib import patheffects

    ends = None if scale_bar_pixels is None else np.asarray(scale_bar_pixels, dtype=np.float64)
    inside = ends is not None and bool(np.isfinite(ends).all()) and bool(
        (ends[:, 0] >= 0).all() and (ends[:, 0] <= width).all() and (ends[:, 1] >= 0).all()
        and (ends[:, 1] <= height).all())
    outline = [patheffects.withStroke(linewidth=2.5, foreground="white")]
    if inside:
        x, y = ends.mean(axis=0)
        view.text(x, y + 0.02 * height, scale_bar.label, ha="center", va="top", fontsize=9, path_effects=outline)
    else:
        view.text(0.02 * width, 0.97 * height, f"scale bar: {scale_bar.label}", ha="left", va="bottom", fontsize=9,
                  path_effects=outline)


def _write_labels(view: Axes, labels: Iterable[tuple[str, float, float]], width: int, height: int) -> None:
    """Write each label centred on its pixel, when the pixel is on the view."""
    from matplotlib import patheffects

    outline = [patheffects.withStroke(linewidth=2.5, foreground="white")]
    for text, x, y in labels:
        if 0 <= x <= width and 0 <= y <= height:
            view.text(x, y, text, ha="center", va="center", fontsize=9, path_effects=outline)


def next_free_path(path: str | os.PathLike[str]) -> Path:
    """`path` if no file has that name, else `stem_2.suffix`, `stem_3.suffix`, ...: the window never overwrites."""
    path = Path(path)
    if not path.exists():
        return path
    number = 2
    while (candidate := path.with_name(f"{path.stem}_{number}{path.suffix}")).exists():
        number += 1
    return candidate


# ---------------------------------------------------------------------------------------------
# seed ensembles on one mesh
# ---------------------------------------------------------------------------------------------
class SeedEnsembleMaps(Protocol):
    """What the viewer reads from a seed ensemble: `foambryo.SeedEnsembleResult`'s three per-interface maps."""

    tensions: Mapping[tuple[int, int], float]
    tension_seed_sd: Mapping[tuple[int, int], float]
    found_by_fewer_than_half_the_seeds: Mapping[tuple[int, int], bool]


def interfaces_on_mesh(mesh: DcelData) -> set[tuple[int, int]]:
    """The label pairs of the mesh's interfaces that can carry a tension: its triangles' pairs, crop faces excluded."""
    triangles = np.asarray(mesh.f)
    crop = np.asarray(getattr(mesh, "crop_faces", np.zeros(len(triangles), dtype=bool)), dtype=bool)
    return {(int(a), int(b)) for a, b in triangles[~crop][:, 3:5]}


@dataclass(frozen=True)
class EnsembleOnMesh:
    """Tensions combined over seeds, on the interfaces of one mesh (matched by label pair).

    Attributes:
        tensions: the ensemble tension of each interface of the mesh the ensemble has; the mesh's other
            interfaces are absent (drawn as an undetermined tension).
        tension_seed_sd: the seeds' spread for the same interfaces (`nan` where fewer than two seeds found one).
        found_by_fewer_than_half_the_seeds: the interfaces of the mesh flagged by the ensemble.
        not_on_the_mesh: how many of the ensemble's interfaces the mesh does not have (not drawn).
        missing_from_the_ensemble: how many of the mesh's interfaces the ensemble does not have (drawn undetermined).
    """

    tensions: dict[tuple[int, int], float]
    tension_seed_sd: dict[tuple[int, int], float]
    found_by_fewer_than_half_the_seeds: tuple[tuple[int, int], ...]
    not_on_the_mesh: int
    missing_from_the_ensemble: int


def ensemble_on_mesh(
    mesh: DcelData,
    tensions: Mapping[tuple[int, int], float],
    tension_seed_sd: Mapping[tuple[int, int], float],
    found_by_fewer_than_half_the_seeds: Mapping[tuple[int, int], bool],
) -> EnsembleOnMesh:
    """Restrict an ensemble's per-interface maps to the interfaces of the mesh being viewed.

    An ensemble combines several reconstructions of one mask; the mesh shown is one reconstruction of it (for
    example `foambryo mesh`'s). Its interfaces are matched to the ensemble's by label pair `(label 1, label 2)`,
    `label 1 < label 2`. The colour scale is then taken over what is drawn, not over interfaces the mesh does not
    have.

    Args:
        mesh: the mesh shown.
        tensions: the ensemble tension per interface (`SeedEnsembleResult.tensions`).
        tension_seed_sd: the seeds' standard deviation per interface.
        found_by_fewer_than_half_the_seeds: the ensemble's flag per interface.

    Returns:
        EnsembleOnMesh: the restricted maps, and the counts of unmatched interfaces on either side.
    """
    on_mesh = interfaces_on_mesh(mesh)
    keys = [(int(a), int(b)) for a, b in tensions]
    shared = sorted(key for key in keys if key in on_mesh)
    return EnsembleOnMesh(
        tensions={key: float(tensions[key]) for key in shared},
        tension_seed_sd={key: float(tension_seed_sd[key]) for key in shared},
        found_by_fewer_than_half_the_seeds=tuple(key for key in shared if found_by_fewer_than_half_the_seeds[key]),
        not_on_the_mesh=len(keys) - len(shared),
        missing_from_the_ensemble=len(on_mesh - set(keys)),
    )
