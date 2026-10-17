"""The Polyscope viewer: a mesh coloured by its tensions and pressures, in a window with their colour bars.

Sacha Ichbiah 2021
Matthieu Perez 2024

This module needs Polyscope (`pip install "foambryo[viewing]"`), and its viewers need a display. The colour bars
alone are written without either by `save_colour_bars` (also in `foambryo.colour_scales`).

The three force viewers (`plot_force_inference`, `plot_tension_inference`, `plot_pressure_inference`) draw the
mesh coloured with the library's colour functions (`foambryo.colour_scales`):

* **tensions**: `coolwarm` (blue for low, light grey for the mean tension, red for high), over a range centred on
  the mean of the finite tensions and symmetric around it, its half-width the farther of the `alpha` and
  `1 - alpha` quantiles from the mean;
* **pressures**: `magma`, reversed (light for low) and trimmed to its 10-90 % range, from 0 to the `1 - alpha`
  quantile of the pressures over the surface's triangles;
* **undetermined values** (`nan`): green, `#1b9e77` (`UNDETERMINED_COLOR`). In the tension view an interface whose
  tension is undetermined is drawn as its outline in this green, without fill; a cell whose pressure is
  undetermined is filled green;
* **crop faces** (where the image box cut the tissue): see-through glass with its edges drawn; they carry no value;
* other quantities drawn on the surface (a seed spread, the sphere-fit residual, the curvatures) keep `jet`.

With tensions and pressures, the surface shows the tensions first. The viewers with tensions also draw a graph
of the tissue (a node at each cell's centroid, an edge per interface coloured by its tension; with pressures, the
nodes coloured by pressure) and, with pressures, each cell's principal stresses. Polyscope's window gets a panel,
"foambryo", that shows:

* **the colour bars** of what is drawn: the library's colours over the library's ranges (the ones
  `save_colour_bars` writes), with a green swatch when a value is undetermined and a glass one for crop
  faces. Polyscope is handed the colours, not the values: its own scalar colour maps would draw `nan` in their
  top colour;
* **which values** the colours show: the inference call that produced them (the library's options are the
  viewers' own, `mode_tension`, `mode_pressure`, `mean_tension`, `base_pressure`, `solver`, with the library's
  defaults), or where they were read from;
* with tensions and pressures, **which one colours the surface** (the tensions by default);
* **a length scale bar** when the mesh's unit is known (`length_unit`): a segment of round length in the scene,
  its length written beside it; and an optional **x, y, z frame**, three labelled segments along the mesh's
  columns (z, y, x);
* **the background**: white (Polyscope's), light grey, dark grey or black (`BACKGROUND_COLOURS`, the napari plugin's
  choices too), or a custom colour; on a dark one, the scale bar, the frame, their labels and the graph's black
  nodes are drawn light;
* **Explode** (the scattered view: each cell moved away from the tissue's centre by a coefficient; the graph's
  nodes stay on their exploded cells) and **Cut** (a scene slice plane normal to z, y or x, at a chosen position,
  keeping either side);
* **Save figure** (the view with its colour bars and scale bar, PNG / PDF / SVG), **Save colour bars**, and
  **Export**: `foambryo infer`'s files for the values shown (`tensions.csv`, `pressures.csv`, `solves.csv`,
  `run.json`, `colour_bars.png`), in a folder; a file is never overwritten without asking;
* a choice of tension method, pressure method or no pressures, solver and diagnostics, **Recompute**, and the
  `foambryo infer` command line that gives the same numbers.

`view_mask` opens the window on a segmentation mask instead (`foambryo view MASK --spacing DZ DY DX`): the mask
first, then its reconstruction and the inference, run in child processes that the panel follows and can cancel.

`save_figure=PATH` writes the same figure from Python, without keeping the window open when `show=False`
(Polyscope still needs a display to render the view).
"""

import inspect
import time
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import polyscope.imgui as psim
from matplotlib import cm

from foambryo._cell_stress import cell_mean_stress_tensors, split_principal_stresses
from foambryo._colour_scales import (
    BACKGROUND_COLOURS,
    CROP_FACE_COLOR,
    CROP_FACE_EDGE_COLOR,
    pressure_colour_fractions,
    pressure_colours,
    surface_face_pressures,
    surface_pressure_colour_upper,
    tension_colour_fractions,
    tension_colour_range,
    tension_colours,
)

# `save_colour_bars` and `ColourBarRanges` are this module's public API, implemented where they need
# no Polyscope (so that the colour bars can be written on a machine without a display).
from foambryo._colour_scales import ColourBarRanges as ColourBarRanges
from foambryo._colour_scales import save_colour_bars as save_colour_bars
from foambryo._polyscope_recording import Recorder, recording

# `ps` is Polyscope, or, on a thread that prepares a scene (`_ForceWindow.prepare`), the recorder of its calls.
from foambryo._polyscope_recording import polyscope_target as ps
from foambryo._viewer_defaults import DEFAULT_SCATTER_COEFF
from foambryo._viewer_figures import (
    SEED_SPREAD_TITLE,
    TENSION_TITLE,
    ColourScale,
    EnsembleOnMesh,
    ScaleBar,
    SeedEnsembleMaps,
    ViewerColourScales,
    compose_figure,
    ensemble_on_mesh,
    format_length,
    next_free_path,
    project_to_pixels,
    quantity_colour_scale,
    round_scale_bar_length,
    scale_bar_for_points,
    viewer_colour_scales,
)
from foambryo._window_export import WindowInference, earlier_window_outputs, export_window_values
from foambryo.curvature import (
    compute_curvature_vertices_cotan,
    compute_gaussian_curvature_vertices,
    compute_sphere_fit_residues_dict,
)
from foambryo.dcel import update_graph_with_scattered_values
from foambryo.force_inference import InferenceOutcome, infer_forces, infer_tensions_then_pressures
from foambryo.plotting_utilities import (
    CROP_FACES_HIDDEN,
    UNDETERMINED_COLOR,
    hide_crop_faces,
    plot_trijunctions,
    plot_trijunctions_topo_change_viewer,
    view_crop_faces,
    view_dict_values_on_mesh,
    view_tensions_on_mesh,
    view_vertex_values_on_embryo,
)
from foambryo.pressure_inference import PressureComputationMethod, infer_pressures
from foambryo.tension_inference import (
    TensionComputationMethod,
    compute_residual_junctions_dict,
    infer_tensions,
)

if TYPE_CHECKING:
    import os
    from collections.abc import Callable, Mapping

    from foambryo._mask_workflow import MaskWorkflow
    from foambryo.dcel import DcelData


# `UNDETERMINED_COLOR` (green, #1b9e77, defined in `foambryo._colour_scales`) fills a cell whose pressure is
# undetermined (nan): a Laplace method could not fix it, or the cell is cut by the crop and its pressure is not
# reported. An undetermined tension's interface is drawn as its outline in that green, without fill.

#: The library's defaults for the inference options the viewers take: those of `infer_forces`.
_INFER_FORCES_DEFAULTS = {
    name: parameter.default for name, parameter in inspect.signature(infer_forces).parameters.items()
}

#: The name of the surface the viewers colour, and of the structures this module adds.
_SURFACE = "Embryo"
_FEW_SEEDS_SURFACE = "Interfaces found by fewer than half of the seeds"  # a curve network: their edges
_FEW_SEEDS_COLOR = (0.85, 0.15, 0.85)
_SCALE_BAR_COLOR = (0.08, 0.08, 0.08)
_SOLVERS = ("auto", "dense", "sparse")
_MASK = "Mask (cell voxels)"
_RECONSTRUCTION = "Reconstruction"
#: An undetermined tension is drawn as its interface's outline (its triangles' edges), without fill.
_UNDETERMINED_OUTLINE = "Undetermined tensions (outline)"
#: The per-face transparency of the tension view: 0 on the undetermined interfaces and the crop faces, else 1.
_TENSION_FILL = "tension fill (undetermined and crop faces hidden)"
_TENSIONS, _PRESSURES = "Surface Tensions", "Pressures Cells"
#: The x, y, z frame: three segments from the drawn box's lowest corner, along the mesh's columns (z, y, x).
_FRAME = "Frame (z, y, x)"
_FRAME_COLOR = (0.15, 0.15, 0.15)
#: The point clouds drawn black on Polyscope's white background: the graph's nodes and the stress tensors' centres.
_BLACK_POINT_CLOUDS = ("Graph Nodes", "Stress tensors principal directions")
#: The window's background: Polyscope's (white) until one is chosen in a panel. Polyscope keeps one background for
#: the process, so this does too; the drawing reads it on the worker thread, where Polyscope cannot be asked.
_BACKGROUND = {"rgb": (1.0, 1.0, 1.0), "custom": False}
#: Polyscope 2.6.1 tone-maps its background, not the colours of its structures: a background value `v` is shown as
#: `min(1, (1.1 v) ** (1 / 2.2))` (exposure 1.1, gamma 2.2; measured on 21 values of a screenshot, to the 8-bit level:
#: 0.20 was shown 0.50 and 0.70 was shown 0.89). The window hands it the value shown as the colour chosen, so that a
#: background reads as its name says, and as in napari. Polyscope's own Render menu can change the exposure and the
#: gamma; the window assumes Polyscope's defaults.
_POLYSCOPE_EXPOSURE, _POLYSCOPE_GAMMA = 1.1, 2.2


def _polyscope_background_value(rgb: "tuple[float, float, float]") -> tuple[float, float, float]:
    """The value to give `ps.set_background_color` for Polyscope to show `rgb` (white: 1 / 1.1, shown 1)."""
    return tuple(min(1.0, float(c) ** _POLYSCOPE_GAMMA / _POLYSCOPE_EXPOSURE) for c in rgb[:3])


def _shown_background(value: "tuple[float, float, float] | np.ndarray") -> tuple[float, float, float]:
    """The colour Polyscope shows for a background value (the inverse of `_polyscope_background_value`)."""
    return tuple(min(1.0, (_POLYSCOPE_EXPOSURE * float(c)) ** (1 / _POLYSCOPE_GAMMA)) for c in value[:3])


def _relative_luminance(rgb: "tuple[float, float, float] | np.ndarray") -> float:
    """The WCAG relative luminance of an sRGB colour in [0, 1]."""
    rgb = np.asarray(rgb, dtype=np.float64)[:3]
    return float(np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4) @ [0.2126, 0.7152, 0.0722])


def _on_background(dark: "tuple[float, float, float]") -> tuple[float, float, float]:
    """The colour to draw `dark` (a grey meant for a white background) in, on the window's background.

    `dark` itself on a light background, its mirror `1 - dark` (a light grey) on a dark one: the one of the two
    with the larger WCAG contrast against the background (light from a luminance below 0.18, which dark grey 0.20
    and black are and light grey 0.70 and white are not).
    """
    if _relative_luminance(_BACKGROUND["rgb"]) >= 0.179:
        return tuple(float(c) for c in dark)
    return tuple(1.0 - float(c) for c in dark)


def _set_background(rgb: "tuple[float, float, float]", window: "_ForceWindow | None") -> None:
    """Paint Polyscope's background `rgb` and redraw in the matching ink what the window draws in a fixed grey."""
    _BACKGROUND["rgb"] = tuple(float(c) for c in rgb[:3])
    ps.set_background_color(_polyscope_background_value(_BACKGROUND["rgb"]))
    if window is not None:
        window.apply_ink()


def _panel_background(window: "_ForceWindow | None") -> None:
    """The background choice: the library's named colours (`BACKGROUND_COLOURS`) or a custom one.

    A background set in Polyscope's own menu (Appearance) is followed too: the ink is redrawn to match it.
    """
    shown = _shown_background(ps.get_background_color())
    if not np.allclose(shown, _BACKGROUND["rgb"], atol=1e-6):
        _BACKGROUND["rgb"] = shown
        if window is not None:
            window.apply_ink()
    names = [*BACKGROUND_COLOURS, "custom"]
    named = [i for i, rgb in enumerate(BACKGROUND_COLOURS.values()) if np.allclose(rgb, _BACKGROUND["rgb"])]
    index = len(names) - 1 if _BACKGROUND["custom"] or not named else named[0]
    changed, chosen = psim.Combo("background", index, names)
    if changed:
        _BACKGROUND["custom"] = chosen == len(names) - 1
        if not _BACKGROUND["custom"]:
            _set_background(tuple(BACKGROUND_COLOURS[names[chosen]]), window)
    if _BACKGROUND["custom"]:
        changed, rgb = psim.ColorEdit3("custom background", _BACKGROUND["rgb"])
        if changed:
            _set_background(tuple(rgb), window)
_DIAGNOSTICS = ("none", "basic", "spectrum")
#: While a scene is prepared on the worker thread, each frame of the window sleeps this long in its panel. Polyscope's
#: render loop holds Python's lock (the GIL) for the whole frame outside the panel, so without this pause the worker
#: barely runs (measured: 55 s instead of 4 s for a 2 000-cell foam). The pause fits within Polyscope's 60-frames-per-
#: second budget, so the frame rate does not drop.
_YIELD_TO_THE_WORKER_SECONDS = 0.012


def plot_force_inference(
    mesh: "DcelData",
    dict_tensions: dict[tuple[int, int], float] | None = None,
    dict_pressure: dict[int, float] | None = None,
    alpha: float = 0.05,
    scalar_quantities: bool = False,
    scattered: bool = False,
    scatter_coeff: float = DEFAULT_SCATTER_COEFF,
    *,
    mean_tension: float = _INFER_FORCES_DEFAULTS["mean_tension"],
    base_pressure: float = _INFER_FORCES_DEFAULTS["base_pressure"],
    mode_tension: TensionComputationMethod = _INFER_FORCES_DEFAULTS["mode_tension"],
    mode_pressure: PressureComputationMethod = _INFER_FORCES_DEFAULTS["mode_pressure"],
    solver: str = _INFER_FORCES_DEFAULTS["solver"],
    length_unit: str | None = None,
    scale_bar: bool | float = True,
    save_figure: "str | os.PathLike[str] | None" = None,
    show: bool = True,
    values_from: str | None = None,
    mesh_file: str | None = None,
) -> None:
    """Show a mesh's tensions and pressures in Polyscope's window, with their colour bars; what `foambryo view` opens.

    The surface is coloured by the tensions first (`coolwarm`, centred on the mean tension); the panel switches it
    to the pressures (reversed `magma`, trimmed to its 10-90 % range). An undetermined tension (`nan`) is drawn as
    its interface's green outline, without fill, and an undetermined pressure fills its cell green; crop faces
    are see-through glass. The graph of the tissue and each cell's principal stresses are drawn too. The module
    docstring lists what the panel offers.

    Needs Polyscope (`pip install "foambryo[viewing]"`) and a display, also with `show=False` (Polyscope renders
    the figure). Returns when the window is closed.

    Args:
        mesh (DcelData): the mesh, in its own length unit.
        dict_tensions (dict[tuple[int, int], float] | None, optional): tension per interface, keyed
            `(label_1, label_2)` with `label_1 < label_2` (0 is the medium). Defaults to None: the tensions and
            the pressures are inferred with `infer_forces` and the options below.
        dict_pressure (dict[int, float] | None, optional): pressure per label, the medium (0) included. Defaults
            to None: inferred with `infer_forces`, or, when `dict_tensions` is given, with `infer_pressures` from
            those tensions and the options below.
        alpha (float, optional): the quantile clipped at each end of the colour scales, in [0, 0.5): the tension
            range reaches the farther of the `alpha` and `1 - alpha` quantiles from the mean, the pressure range
            runs from 0 to the `1 - alpha` quantile. Defaults to 0.05.
        scalar_quantities (bool, optional): also add the sphere-fit residual, the area and volume derivatives
            and the curvatures as surface quantities (`jet`). Defaults to False.
        scattered (bool, optional): open the window exploded: each cell moved away from the tissue's centre by
            `scatter_coeff` times (its centroid - the tissue's centroid). The panel changes it. Defaults to False.
        scatter_coeff (float, optional): the explode coefficient. Defaults to 0.2 (`DEFAULT_SCATTER_COEFF`).
        mean_tension (float, optional): the mean the tensions are gauged to. Defaults to 1. Keyword-only.
        base_pressure (float, optional): the pressure of the medium, label 0. Defaults to 0. Keyword-only.
        mode_tension (TensionComputationMethod, optional): the tension method. Defaults to `YoungDupre`.
            Keyword-only.
        mode_pressure (PressureComputationMethod, optional): the pressure method. Defaults to `Variational`.
            Keyword-only.
        solver (str, optional): `"auto"`, `"dense"` or `"sparse"`. Defaults to `"auto"`. Keyword-only.
            These five are `infer_forces`' options, with its defaults, passed to it unchanged when the viewer
            infers (bitwise `foambryo infer`'s numbers); the panel names the call that produced the colours.
        length_unit (str | None, optional): the unit of the mesh's coordinates (`"um"`, ...); with a unit, a
            length scale bar is drawn. Defaults to None: no scale bar. Keyword-only.
        scale_bar (bool | float, optional): draw the scale bar (`True`, its length 1, 2 or 5 times a power of
            ten, at most a quarter of the tissue's largest side), not (`False`), or with this length, in
            `length_unit`. Defaults to True. Keyword-only.
        save_figure (str | os.PathLike | None, optional): write the first view, with its colour bars and scale
            bar, to this file (`.png`, `.pdf`, `.svg`; matplotlib chooses the format from the suffix), before the
            window is shown. Defaults to None. Keyword-only.
        show (bool, optional): open the window; `False` with `save_figure` only writes the figure. Defaults to
            True. Keyword-only.
        values_from (str | None, optional): where given values come from, shown in the panel and under a saved
            figure. Defaults to None. Keyword-only.
        mesh_file (str | None, optional): the mesh's file name, for the `foambryo infer` command line the panel
            shows for values inferred in the window. Defaults to None (no command line). Keyword-only.

    Raises:
        ValueError: when the viewer infers and the library refuses the tensions or the pressures (the library's
            message; `UndeterminedTensionError`, a `ValueError`, when the given tensions are not all finite on a
            mesh without crop faces).
    """
    options = _InferenceOptions(mode_tension, mode_pressure, mean_tension, base_pressure, solver)
    tensions, pressures, values_from = options.infer(mesh, "force", dict_tensions, dict_pressure, values_from)
    window = _ForceWindow(
        mesh, "force", tensions, pressures, options, values_from,
        alpha=alpha, scalar_quantities=scalar_quantities, scattered=scattered, scatter_coeff=scatter_coeff,
        length_unit=length_unit, scale_bar=scale_bar, mesh_name=mesh_file,
    )
    window.open(save_figure=save_figure, show=show)


def plot_pressure_inference(
    mesh: "DcelData",
    dict_pressure: dict[int, float] | None = None,
    alpha: float = 0.05,
    scattered: bool = False,
    scatter_coeff: float = DEFAULT_SCATTER_COEFF,
    *,
    mean_tension: float = _INFER_FORCES_DEFAULTS["mean_tension"],
    base_pressure: float = _INFER_FORCES_DEFAULTS["base_pressure"],
    mode_tension: TensionComputationMethod = _INFER_FORCES_DEFAULTS["mode_tension"],
    mode_pressure: PressureComputationMethod = _INFER_FORCES_DEFAULTS["mode_pressure"],
    solver: str = _INFER_FORCES_DEFAULTS["solver"],
    length_unit: str | None = None,
    scale_bar: bool | float = True,
    save_figure: "str | os.PathLike[str] | None" = None,
    show: bool = True,
    values_from: str | None = None,
) -> None:
    """Show a mesh's cell pressures in Polyscope's window, with their colour bar.

    Each triangle carries the pressure of the higher-labelled of its two materials, coloured with reversed
    `magma` trimmed to its 10-90 % range, from 0 to the `1 - alpha` quantile over the triangles (negative
    pressures drawn as 0). An undetermined pressure (`nan`) fills its cell green; crop faces are see-through glass.
    Polyscope's structures are removed first.

    Needs Polyscope (`pip install "foambryo[viewing]"`) and a display, also with `show=False` (Polyscope renders
    the figure). Returns when the window is closed.

    Args:
        mesh (DcelData): the mesh, in its own length unit.
        dict_pressure (dict[int, float] | None, optional): pressure per label, the medium (0) included. Defaults
            to None: the tensions and pressures are inferred with `infer_forces` and the options below, and the
            pressures are shown.
        alpha (float, optional): the quantile clipped at the top of the pressure scale. Defaults to 0.05.
        scattered (bool, optional): open the window exploded: each cell moved away from the tissue's centre by
            `scatter_coeff` times (its centroid - the tissue's centroid). Defaults to False.
        scatter_coeff (float, optional): the explode coefficient. Defaults to 0.2 (`DEFAULT_SCATTER_COEFF`).
        mean_tension (float, optional): the mean the tensions are gauged to. Defaults to 1. Keyword-only.
        base_pressure (float, optional): the pressure of the medium, label 0. Defaults to 0. Keyword-only.
        mode_tension (TensionComputationMethod, optional): the tension method. Defaults to `YoungDupre`.
            Keyword-only.
        mode_pressure (PressureComputationMethod, optional): the pressure method. Defaults to `Variational`.
            Keyword-only.
        solver (str, optional): `"auto"`, `"dense"` or `"sparse"`. Defaults to `"auto"`. Keyword-only.
            These five are `infer_forces`' options, with its defaults, passed to it unchanged when the viewer
            infers; the panel names the call that produced the colours. Unused when `dict_pressure` is given.
        length_unit (str | None, optional): the unit of the mesh's coordinates (`"um"`, ...); with a unit, a
            length scale bar is drawn. Defaults to None: no scale bar. Keyword-only.
        scale_bar (bool | float, optional): draw the scale bar (`True`, its length chosen round), not
            (`False`), or with this length, in `length_unit`. Defaults to True. Keyword-only.
        save_figure (str | os.PathLike | None, optional): write the first view, with its colour bar and scale
            bar, to this file (`.png`, `.pdf`, `.svg`), before the window is shown. Defaults to None.
            Keyword-only.
        show (bool, optional): open the window; `False` with `save_figure` only writes the figure. Defaults to
            True. Keyword-only.
        values_from (str | None, optional): where given values come from, shown in the panel and under a saved
            figure. Defaults to None. Keyword-only.

    Raises:
        ValueError: when the viewer infers and the library refuses the tensions or the pressures (the library's
            message).
    """
    ps.remove_all_structures()
    options = _InferenceOptions(mode_tension, mode_pressure, mean_tension, base_pressure, solver)
    tensions, pressures, values_from = options.infer(mesh, "pressure", None, dict_pressure, values_from)
    window = _ForceWindow(
        mesh, "pressure", tensions, pressures, options, values_from,
        alpha=alpha, scattered=scattered, scatter_coeff=scatter_coeff, length_unit=length_unit, scale_bar=scale_bar,
    )
    window.open(save_figure=save_figure, show=show)


def plot_tension_inference(
    mesh: "DcelData",
    dict_tensions: dict[tuple[int, int], float] | None = None,
    alpha: float = 0.05,
    scalar_quantities: bool = False,
    scattered: bool = False,
    scatter_coeff: float = DEFAULT_SCATTER_COEFF,
    *,
    mean_tension: float = _INFER_FORCES_DEFAULTS["mean_tension"],
    mode_tension: TensionComputationMethod = _INFER_FORCES_DEFAULTS["mode_tension"],
    solver: str = _INFER_FORCES_DEFAULTS["solver"],
    seed_ensemble: SeedEnsembleMaps | None = None,
    length_unit: str | None = None,
    scale_bar: bool | float = True,
    save_figure: "str | os.PathLike[str] | None" = None,
    show: bool = True,
    values_from: str | None = None,
    mesh_file: str | None = None,
) -> None:
    """Show a mesh's surface tensions in Polyscope's window, with their colour bar.

    The surface is coloured with `coolwarm` over a range centred on the mean tension and symmetric around it; an
    undetermined tension (`nan`), or an interface absent from the map, is drawn as its interface's green outline,
    without fill; crop faces are see-through glass. The graph of the tissue is drawn too, its edges coloured by
    tension. Polyscope's structures are removed first.

    Needs Polyscope (`pip install "foambryo[viewing]"`) and a display, also with `show=False` (Polyscope renders
    the figure). Returns when the window is closed.

    Args:
        mesh (DcelData): the mesh, in its own length unit.
        dict_tensions (dict[tuple[int, int], float] | None, optional): tension per interface, keyed
            `(label_1, label_2)` with `label_1 < label_2` (0 is the medium). Defaults to None: inferred with
            `infer_tensions` and the options below.
        alpha (float, optional): the quantile at each end of the colour scale, in [0, 0.5): the range reaches the
            farther of the `alpha` and `1 - alpha` quantiles from the mean. Defaults to 0.05.
        scalar_quantities (bool, optional): also add the sphere-fit residual, the area and volume derivatives
            and the curvatures as surface quantities (`jet`). Defaults to False.
        scattered (bool, optional): open the window exploded: each cell moved away from the tissue's centre by
            `scatter_coeff` times (its centroid - the tissue's centroid). Defaults to False.
        scatter_coeff (float, optional): the explode coefficient. Defaults to 0.2 (`DEFAULT_SCATTER_COEFF`).
        mean_tension (float, optional): the mean the tensions are gauged to. Defaults to 1. Keyword-only.
        mode_tension (TensionComputationMethod, optional): the tension method. Defaults to `YoungDupre`.
            Keyword-only.
        solver (str, optional): `"auto"`, `"dense"` or `"sparse"`. Defaults to `"auto"`. Keyword-only.
            These three are `infer_forces`' tension options, with its defaults, passed to `infer_tensions`
            unchanged when the viewer infers.
        seed_ensemble (SeedEnsembleMaps | None, optional): tensions combined over several reconstructions of
            the mesh's mask (`foambryo.infer_tensions_over_seeds`' result, or any object with its `tensions`,
            `tension_seed_sd` and `found_by_fewer_than_half_the_seeds` maps). They are drawn on this mesh's
            interfaces, matched by label pair (the mesh's other interfaces drawn as undetermined); the seeds'
            spread is a second quantity with its own colour bar (`jet`, between the `alpha` and `1 - alpha`
            quantiles); the interfaces found by fewer than half of the seeds are drawn over the surface in
            magenta (their triangles' edges). Not with `dict_tensions`. Defaults to None. Keyword-only.
        length_unit (str | None, optional): the unit of the mesh's coordinates (`"um"`, ...); with a unit, a
            length scale bar is drawn. Defaults to None: no scale bar. Keyword-only.
        scale_bar (bool | float, optional): draw the scale bar (`True`, its length chosen round), not
            (`False`), or with this length, in `length_unit`. Defaults to True. Keyword-only.
        save_figure (str | os.PathLike | None, optional): write the first view, with its colour bars and scale
            bar, to this file (`.png`, `.pdf`, `.svg`), before the window is shown. Defaults to None.
            Keyword-only.
        show (bool, optional): open the window; `False` with `save_figure` only writes the figure. Defaults to
            True. Keyword-only.
        values_from (str | None, optional): where given values come from, shown in the panel and under a saved
            figure. Defaults to None. Keyword-only.
        mesh_file (str | None, optional): the mesh's file name, for the `foambryo infer` command line the panel
            shows for values inferred in the window. Defaults to None (no command line). Keyword-only.

    Raises:
        ValueError: if both `dict_tensions` and `seed_ensemble` are given, or when the viewer infers and the
            library refuses the tensions (the library's message).
    """
    ps.remove_all_structures()
    options = _InferenceOptions(mode_tension, _INFER_FORCES_DEFAULTS["mode_pressure"], mean_tension,
                                _INFER_FORCES_DEFAULTS["base_pressure"], solver, pressures=False)
    ensemble = None
    if seed_ensemble is not None:
        if dict_tensions is not None:
            message = "give dict_tensions or seed_ensemble, not both"
            raise ValueError(message)
        ensemble = ensemble_on_mesh(mesh, seed_ensemble.tensions, seed_ensemble.tension_seed_sd,
                                    seed_ensemble.found_by_fewer_than_half_the_seeds)
        dict_tensions = ensemble.tensions
        values_from = values_from or "tensions combined over seeds (infer_tensions_over_seeds)"
    tensions, _, values_from = options.infer(mesh, "tension", dict_tensions, None, values_from)
    window = _ForceWindow(
        mesh, "tension", tensions, None, options, values_from,
        alpha=alpha, scalar_quantities=scalar_quantities, scattered=scattered, scatter_coeff=scatter_coeff,
        length_unit=length_unit, scale_bar=scale_bar, ensemble=ensemble, mesh_name=mesh_file,
    )
    window.open(save_figure=save_figure, show=show)


class _InferenceOptions:
    """The inference options of a viewer, and the library calls they make.

    Every option is passed to the library unchanged; the defaults are `infer_forces`' own.
    """

    def __init__(
        self,
        mode_tension: TensionComputationMethod,
        mode_pressure: PressureComputationMethod,
        mean_tension: float,
        base_pressure: float,
        solver: str,
        pressures: bool = True,
        diagnostics: str = "none",
    ) -> None:
        self.mode_tension = mode_tension
        self.mode_pressure = mode_pressure
        self.mean_tension = mean_tension
        self.base_pressure = base_pressure
        self.solver = solver
        self.pressures = pressures
        self.diagnostics = diagnostics
        #: The last inference made with these options (for the panel's Export), or `None`.
        self.inference: WindowInference | None = None

    def record_options(self, with_pressures: bool) -> dict:
        """`run.json`'s `options`, as `foambryo infer` writes them for these options."""
        from foambryo.results import option_name

        options = {"tension_method": option_name(self.mode_tension), "pressure_method": option_name(self.mode_pressure),
                   "mean_tension": self.mean_tension, "base_pressure": self.base_pressure, "solver": self.solver,
                   "diagnostics": self.diagnostics}
        if not with_pressures:
            options.update(pressure_method=None, base_pressure=None, no_pressures=True)
        return options

    def _infer(self, mesh: "DcelData", *, with_pressures: bool) -> InferenceOutcome:
        """`infer_tensions_then_pressures`, `foambryo infer`'s call (bitwise `infer_forces`), with its record kept.

        The warnings are counted for the record and still shown. A refused tension solve raises its message, as
        `infer_forces` would.
        """
        from foambryo.results import DIAGNOSTICS

        date = datetime.now(UTC).isoformat(timespec="seconds")
        clock = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for category in (DeprecationWarning, PendingDeprecationWarning, ImportWarning, ResourceWarning):
                warnings.simplefilter("ignore", category)
            outcome = infer_tensions_then_pressures(
                mesh, self.mean_tension, self.base_pressure, self.mode_tension, self.mode_pressure, solver=self.solver,
                diagnostics=DIAGNOSTICS[self.diagnostics], with_pressures=with_pressures)
        counts: dict[tuple[str, str], int] = {}
        for warning in caught:
            warnings.showwarning(warning.message, warning.category, warning.filename, warning.lineno)
            key = (warning.category.__name__, str(warning.message))
            counts[key] = counts.get(key, 0) + 1
        if outcome.tensions is None or (with_pressures and outcome.pressures is None):
            raise ValueError(outcome.error or outcome.status)
        self.inference = WindowInference(
            outcome, self.record_options(with_pressures), date, round(time.perf_counter() - clock, 3),
            [{"category": c, "message": m, "count": n} for (c, m), n in counts.items()])
        return outcome

    def infer_arguments(self) -> list[str]:
        """`foambryo infer`'s options that give these numbers (`--no-pressures` without pressures)."""
        from foambryo.results import option_name

        pressure = (["--pressure-method", option_name(self.mode_pressure), "--base-pressure", repr(self.base_pressure)]
                    if self.pressures else ["--no-pressures"])
        return ["--tension-method", option_name(self.mode_tension), *pressure, "--mean-tension",
                repr(self.mean_tension), "--solver", self.solver]

    def tension_call(self) -> str:
        """The tension call, as text."""
        return (f"infer_tensions(mesh, mean_tension={self.mean_tension!r}, mode=TensionComputationMethod."
                f"{self.mode_tension.name}, solver={self.solver!r})")

    def pressure_call(self) -> str:
        """The pressure call, as text."""
        return (f"infer_pressures(mesh, tensions, mode=PressureComputationMethod.{self.mode_pressure.name}, "
                f"base_pressure={self.base_pressure!r}, solver={self.solver!r})")

    def forces_call(self) -> str:
        """The `infer_forces` call, as text."""
        return (f"infer_forces(mesh, mean_tension={self.mean_tension!r}, base_pressure={self.base_pressure!r}, "
                f"mode_tension=TensionComputationMethod.{self.mode_tension.name}, "
                f"mode_pressure=PressureComputationMethod.{self.mode_pressure.name}, solver={self.solver!r})")

    def infer(
        self,
        mesh: "DcelData",
        kind: str,
        dict_tensions: "Mapping[tuple[int, int], float] | None",
        dict_pressures: "Mapping[int, float] | None",
        values_from: str | None,
    ) -> tuple[dict | None, dict | None, str]:
        """The values a viewer of `kind` shows: those given, else inferred with these options.

        `"force"` and `"pressure"`: with no tensions, `infer_forces` (tensions and pressures together); with
        tensions and no pressures, `infer_pressures` from them. `"tension"`: `infer_tensions` when no tension is
        given. A pressure viewer given its pressures infers nothing.

        Returns:
            tuple: the tensions (`None` for a pressure viewer given its pressures), the pressures (`None` for a
            tension viewer) and the text the panel shows about where they come from.
        """
        if kind == "tension":
            if dict_tensions is None:
                return self._infer(mesh, with_pressures=False).tensions, None, f"inferred here: {self.tension_call()}"
            return dict(dict_tensions), None, values_from or "tensions given to the viewer"
        if kind == "pressure" and dict_pressures is not None:
            return None, dict(dict_pressures), values_from or "pressures given to the viewer"
        if dict_tensions is None:
            outcome = self._infer(mesh, with_pressures=True)
            return outcome.tensions, outcome.pressures, f"inferred here: {self.forces_call()}"
        if dict_pressures is None:
            pressures = infer_pressures(mesh, dict_tensions, mode=self.mode_pressure, base_pressure=self.base_pressure,
                                        solver=self.solver)
            given = values_from or "tensions given to the viewer"
            return dict(dict_tensions), pressures, f"{given}; pressures inferred here: {self.pressure_call()}"
        return dict(dict_tensions), dict(dict_pressures), values_from or "tensions and pressures given to the viewer"


class _ForceWindow:
    """One force viewer: the scene it draws, its colour bars, its scale bar and its panel.

    **Drawing.** `prepare` runs the drawing code (the three viewer functions' structures, their colours bitwise the
    library's, the coloured surface's back faces drawn like its front faces, the scale bar, the seed-ensemble
    quantities, the undetermined interfaces' outlines) against a recorder of Polyscope's calls, on any thread;
    `install` replays the recorded calls on the window's thread. `draw` does both at once; `request_redraw`
    prepares on a worker thread and the panel installs the result when it is ready, so the window keeps turning
    while a large tissue's arrays are computed (the explode coefficient is followed live this way).

    **What the surface shows** (`showing`): the tensions by default, also when pressures were inferred; the panel
    switches to the pressures. With the tensions, an undetermined tension's interface is drawn as its outline in
    `UNDETERMINED_COLOR`, without fill, and the crop faces are hidden from the coloured surface (they are glass).

    `panel` is Polyscope's user callback, run every frame.
    """

    def __init__(
        self,
        mesh: "DcelData",
        kind: str,
        tensions: dict | None,
        pressures: dict | None,
        options: _InferenceOptions,
        values_from: str,
        *,
        alpha: float = 0.05,
        scalar_quantities: bool = False,
        scattered: bool = False,
        scatter_coeff: float = DEFAULT_SCATTER_COEFF,
        length_unit: str | None = None,
        scale_bar: bool | float = True,
        ensemble: EnsembleOnMesh | None = None,
        mesh_name: str | None = None,
        inference: WindowInference | None = None,
        rec_bytes: bytes | None = None,
    ) -> None:
        self.mesh = mesh
        self.kind = kind
        self.base_kind = kind  # a force viewer drawn with the tensions only can draw its pressures again
        self.mesh_name = mesh_name
        #: Called at the end of every `draw`: a host window adds its own structures (the mask) there.
        self.after_draw: Callable[[], None] | None = None
        #: The cut: `None`, or `(axis, fraction along the drawn box, keep_higher)`; `slice_plane` is Polyscope's.
        self.cut: tuple[int, float, bool] | None = None
        self.slice_plane = None
        self.tensions = tensions
        self.pressures = pressures
        self.options = options
        self.values_from = values_from
        self.alpha = alpha
        self.scalar_quantities = scalar_quantities
        self.scattered = scattered
        self.scatter_coeff = scatter_coeff
        self.ensemble = ensemble
        self.length_unit = length_unit
        self.scale_bar: ScaleBar | None = None
        if length_unit is not None and scale_bar is not False:
            length = None if scale_bar is True else float(scale_bar)
            points = mesh.v
            self.scale_bar = scale_bar_for_points(points, length_unit, length)
        self.show_scale_bar = self.scale_bar is not None
        self.flat_colours = False
        self.scales: ViewerColourScales | None = None
        self.figure_path = "foambryo_figure.png"
        self.bars_path = "foambryo_colour_bars.png"
        self.status = ""
        self.tension_choice = list(TensionComputationMethod).index(options.mode_tension)
        self.pressure_choice = list(PressureComputationMethod).index(options.mode_pressure)
        self.solver_choice = _SOLVERS.index(options.solver)
        self.diagnostics_choice = _DIAGNOSTICS.index(options.diagnostics)
        self.pressures_choice = kind != "tension"
        #: What the surface is coloured by: the tensions (the default whenever there are tensions) or the pressures.
        self.showing = "pressure" if kind == "pressure" else "tension"
        self.show_outlines = True
        self.show_frame = False
        #: The frame drawn: `(origin, length)` in the mesh's coordinates, or `None`.
        self.frame: tuple[np.ndarray, float] | None = None
        #: The values' inference, for Export (the window's own, or that of the values given to it).
        self.inference = inference if inference is not None else options.inference
        self.rec_bytes = rec_bytes
        self.export_folder = "foambryo_export"
        #: foambryo's files found in the export folder, waiting for Replace or Cancel; `None` when not asking.
        self.export_asking: list[str] | None = None
        #: The colour arrays handed to the coloured surface, by quantity, and whether it has a tension fill.
        self.colours: dict[str, np.ndarray] = {}
        self.tension_fill = False
        self._installed: set[tuple[str, str]] = set()
        self._executor: ThreadPoolExecutor | None = None
        self._future: Future | None = None
        self._queued = False

    # -----------------------------------------------------------------------------------------
    # the scene: prepared off the window's thread, installed on it
    # -----------------------------------------------------------------------------------------
    def prepare(self, scattered: bool, coeff: float) -> tuple[Recorder, ViewerColourScales]:
        """Compute the scene for this explode state and record Polyscope's calls; any thread, no Polyscope call."""
        recorder = Recorder()
        with recording(recorder):
            self._draw_scene(scattered, coeff)
        return recorder, self._colour_scales(scattered)

    def install(self, recorder: Recorder, scales: ViewerColourScales) -> None:
        """Replace this window's structures by a prepared scene's, on the window's thread; then what is shown."""
        for kind, name in self._installed:
            getattr(ps, f"remove_{kind}")(name, error_if_absent=False)
        recorder.replay()
        self._installed = set(recorder.structures)
        self.scales = scales
        self.colours = {args[0]: args[1] for args, _ in recorder.recorded_calls(_SURFACE, "add_color_quantity")}
        self.tension_fill = any(args[0] == _TENSION_FILL
                                for args, _ in recorder.recorded_calls(_SURFACE, "add_scalar_quantity"))
        self.apply_showing()
        self._draw_frame()
        if self.after_draw is not None:
            self.after_draw()
        self.apply_cut()

    def draw(self) -> None:
        """Register every structure of this viewer, and compute its colour scales (on this thread)."""
        self.install(*self.prepare(self.scattered, self.scatter_coeff))

    def redraw(self) -> None:
        """Draw again, at once (after the values changed)."""
        self.draw()

    def request_redraw(self) -> None:
        """Prepare the scene again on a worker thread; the panel installs it when ready (`poll_redraw`).

        A request made while one is being prepared is kept, and prepared next with the window's state then: the
        last request is always drawn, and no more than one is prepared at a time.
        """
        if self._future is not None and not self._future.done():
            self._queued = True
            return
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="foambryo-scene")
        self._queued = False
        self._future = self._executor.submit(self.prepare, self.scattered, self.scatter_coeff)

    def poll_redraw(self) -> bool:
        """Install a scene prepared in the background, if one is ready; returns whether one was installed.

        While one is being prepared, the frame pauses (`_YIELD_TO_THE_WORKER_SECONDS`) so that the worker runs.
        """
        if self._future is None:
            return False
        if not self._future.done():
            time.sleep(_YIELD_TO_THE_WORKER_SECONDS)
            return False
        future, self._future = self._future, None
        try:
            recorder, scales = future.result()
        except (ValueError, ArithmeticError, KeyError) as error:
            self.status = f"{type(error).__name__}: {error}"
            installed = False
        else:
            self.install(recorder, scales)
            installed = True
        if self._queued:
            self.request_redraw()
        return installed

    @property
    def preparing(self) -> bool:
        """A scene is being prepared in the background."""
        return self._future is not None

    def close(self) -> None:
        """Stop the worker thread (a scene being prepared is abandoned)."""
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def set_explode(self, scattered: bool, coeff: float | None = None) -> None:
        """Draw the scattered view (`DcelData.compute_scattered_arrays(coeff)`) or the plain one, at once."""
        self.scattered = bool(scattered)
        if coeff is not None:
            self.scatter_coeff = float(coeff)
        self.redraw()

    def _draw_scene(self, scattered: bool, coeff: float) -> None:
        """Every structure of this viewer, through `ps` (recorded by `prepare`)."""
        mesh = self.mesh
        nx_graph = None
        if self.kind in ("force", "tension"):
            nx_graph = mesh.compute_networkx_graph()
            nx.set_edge_attributes(nx_graph, self.tensions, "tension")
        if self.kind == "force":
            nx.set_node_attributes(nx_graph, self.pressures, "pressure")
        if self.kind == "pressure":
            nx_graph = mesh.compute_networkx_graph()
            nx.set_node_attributes(nx_graph, self.pressures, "pressure")
        # Fitted before exploding, on the mesh's own interfaces: an exploded interface with the medium is translated,
        # which changes its residual by rounding (about 1e-12 relative). Only when the panel shows them.
        residuals = None
        if self.scalar_quantities and nx_graph is not None:
            residuals = compute_sphere_fit_residues_dict(nx_graph)
        if scattered:
            mesh.compute_scattered_arrays(coeff=coeff)
            nx_graph = update_graph_with_scattered_values(nx_graph, mesh)

        if self.kind == "force":
            self._draw_forces(nx_graph, scattered, residuals)
        elif self.kind == "tension":
            self._draw_tensions(nx_graph, scattered, residuals)
        else:
            _view_pressures_on_mesh(mesh, self.pressures, alpha=self.alpha, scattered=scattered)
        view_crop_faces(mesh, scattered=scattered)

        if ps.has_surface_mesh(_SURFACE):
            surface = ps.get_surface_mesh(_SURFACE)
            # Polyscope shades a back face darker by default; the triangles' winding follows the label order, so
            # the outside of a tissue faces away from the camera and every colour was drawn darker than its bar.
            surface.set_back_face_policy("identical")
            surface.set_material("flat" if self.flat_colours else "clay")
            if self.tensions is not None:
                self._draw_undetermined(surface, scattered)
        self._draw_scale_bar()
        self.apply_ink()

    def apply_ink(self) -> None:
        """Draw the scale bar, the frame and the black point clouds in the ink of the background (`_on_background`)."""
        for name in _BLACK_POINT_CLOUDS:
            if ps.has_point_cloud(name):
                ps.get_point_cloud(name).set_color(_on_background((0.0, 0.0, 0.0)))
        if ps.has_curve_network(_FRAME):
            ps.get_curve_network(_FRAME).set_color(_on_background(_FRAME_COLOR))
        if self.scale_bar is not None and ps.has_curve_network(f"Scale bar ({self.scale_bar.label})"):
            ps.get_curve_network(f"Scale bar ({self.scale_bar.label})").set_color(_on_background(_SCALE_BAR_COLOR))

    def _draw_undetermined(self, surface: "ps.SurfaceMesh", scattered: bool) -> None:
        """The tension view's fill (0 on undetermined interfaces and crop faces) and the undetermined outlines.

        An interface whose tension is `nan`, or absent from the map (an interface a seed ensemble does not have),
        has no value: in the tension view it is not filled, and its triangles' edges are drawn in
        `UNDETERMINED_COLOR`, a curve network over the surface (a wireframe only of those triangles).
        """
        mesh = self.mesh
        v, f, idx = ((mesh.v_scattered, mesh.f_scattered, mesh.idx_scattered) if scattered
                     else (mesh.v, mesh.f, np.arange(len(mesh.f))))
        pairs = np.asarray(mesh.f)[idx][:, 3:5]
        values = np.array([self.tensions.get((int(a), int(b)), np.nan) for a, b in pairs], dtype=np.float64)
        crop = np.asarray(mesh.crop_faces, dtype=bool)[idx]
        undetermined = ~np.isfinite(values) & ~crop
        if undetermined.any() or crop.any():
            surface.add_scalar_quantity(_TENSION_FILL, np.where(undetermined | crop, 0.0, 1.0), defined_on="faces")
        if undetermined.any():
            triangles = np.asarray(f)[undetermined][:, :3]
            edges = np.unique(np.sort(triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0)
            nodes, edges = np.unique(edges, return_inverse=True)
            ps.register_curve_network(_UNDETERMINED_OUTLINE, np.asarray(v)[nodes], edges.reshape(-1, 2),
                                      color=tuple(float(c) for c in UNDETERMINED_COLOR), radius=0.0015)

    def apply_showing(self) -> None:
        """Colour the surface by `showing`, the tensions or the pressures.

        With the tensions, the undetermined interfaces' outlines are shown and they and the crop faces are not
        filled; with the pressures, the crop faces are not filled.
        """
        if not ps.has_surface_mesh(_SURFACE):
            return
        surface = ps.get_surface_mesh(_SURFACE)
        tension = self.showing == "tension" and _TENSIONS in self.colours
        if tension:
            surface.add_color_quantity(_TENSIONS, self.colours[_TENSIONS], defined_on="faces", enabled=True)
            if self.tension_fill:
                surface.set_transparency_quantity(_TENSION_FILL)
        elif _PRESSURES in self.colours:
            surface.add_color_quantity(_PRESSURES, self.colours[_PRESSURES], defined_on="faces", enabled=True)
            if getattr(self.mesh, "has_crop_faces", False):
                surface.set_transparency_quantity(CROP_FACES_HIDDEN)
            else:
                surface.clear_transparency_quantity()
        if ps.has_curve_network(_UNDETERMINED_OUTLINE):
            ps.get_curve_network(_UNDETERMINED_OUTLINE).set_enabled(tension and self.show_outlines)

    def _draw_frame(self) -> None:
        """The x, y, z frame: three segments of one round length from just outside the drawn box's lowest corner.

        The origin is the box's lowest corner moved outwards by 10 % of the box's largest side along each axis, so
        that the segments are not hidden inside the tissue (`show_frame`).

        The mesh's columns are (z, y, x), delaunay-watershed-3d's order: the segment along column 0 is labelled z.
        Its length is the scale bar's rule (1, 2 or 5 times a power of ten, at most a quarter of the box), in the
        mesh's unit. The labels are written by the panel (`_label_overlays`) and in saved figures.
        """
        low, high = self.drawn_box()
        extent = float(np.max(high - low))
        if not (np.isfinite(extent) and extent > 0):
            self.frame = None
            return
        length = round_scale_bar_length(extent)
        origin = low - 0.1 * extent
        nodes = np.vstack([origin, origin + length * np.eye(3)])
        network = ps.register_curve_network(_FRAME, nodes, np.array([[0, 1], [0, 2], [0, 3]]),
                                            color=_on_background(_FRAME_COLOR), radius=0.005)
        network.set_enabled(self.show_frame)
        self._installed.add(("curve_network", _FRAME))
        self.frame = (origin, length)

    def frame_labels(self, width: int, height: int) -> list[tuple[str, float, float]]:
        """The frame's labels and where they fall on an image `width` x `height`.

        `z`, `y`, `x` at the tips, the length at the corner. Empty without the frame shown or outside a perspective
        view.
        """
        if self.frame is None or not self.show_frame or ps.get_view_projection_mode() != "perspective":
            return []
        origin, length = self.frame
        points = np.vstack([origin, origin + length * np.eye(3)])
        camera = ps.get_view_camera_parameters()
        pixels = project_to_pixels(points, camera.get_view_mat(), camera.get_fov_vertical_deg(), width, height)
        unit = format_length(length, self.length_unit) if self.length_unit else f"{length:g} (mesh units)"
        texts = [unit, "z", "y", "x"]
        return [(text, float(x), float(y)) for text, (x, y) in zip(texts, pixels, strict=True)
                if np.isfinite(x) and np.isfinite(y)]

    def drawn_box(self) -> np.ndarray:
        """The `(2, 3)` box of the surface as drawn (scattered or not), in the mesh's coordinates."""
        points = self.mesh.v_scattered if self.scattered and hasattr(self.mesh, "v_scattered") else self.mesh.v
        points = np.asarray(points, dtype=np.float64)
        return np.stack([points.min(axis=0), points.max(axis=0)])

    def cut_pose(self) -> tuple[np.ndarray, np.ndarray] | None:
        """The slice plane's point and normal for the cut, or `None` without a cut.

        Polyscope keeps the side of a slice plane its normal points to (measured in a real window, Polyscope
        2.6.1): the normal is `+axis` to keep the higher side, `-axis` for the lower one.
        """
        if self.cut is None:
            return None
        axis, fraction, keep_higher = self.cut
        low, high = self.drawn_box()
        point = (low + high) / 2
        point[axis] = low[axis] + fraction * (high[axis] - low[axis])
        normal = np.zeros(3)
        normal[axis] = 1.0 if keep_higher else -1.0
        return point, normal

    def apply_cut(self) -> None:
        """Pose Polyscope's scene slice plane as `cut` says, or deactivate it."""
        pose = self.cut_pose()
        if pose is None:
            if self.slice_plane is not None:
                self.slice_plane.set_active(False)
            return
        if self.slice_plane is None:
            self.slice_plane = ps.add_scene_slice_plane()
            self.slice_plane.set_draw_widget(False)
        self.slice_plane.set_active(True)
        self.slice_plane.set_pose(pose[0], pose[1])

    def _draw_forces(self, nx_graph: nx.Graph, scattered: bool, dict_sphere_fit_residual: dict | None) -> None:
        mesh, alpha = self.mesh, self.alpha
        ps_mesh = view_tensions_on_mesh(mesh, self.tensions, alpha=alpha, scattered=scattered, name_values=_TENSIONS)
        ps_mesh = _view_pressures_on_mesh(
            mesh,
            self.pressures,
            ps_mesh=ps_mesh,
            alpha=alpha,
            scattered=scattered,
        )
        _plot_stress_tensor(
            mesh,
            nx_graph,
            self.tensions,
            self.pressures,
            clean_before=False,
            clean_after=False,
            show=False,
        )
        _display_embryo_graph_forces(
            nx_graph,
            alpha=alpha,
            clean_before=False,
            clean_after=False,
            show=False,
            base_pressure=0,
            plot_pressures=True,
            pressure_upper=surface_pressure_colour_upper(mesh, self.pressures, alpha, scattered=scattered),
        )
        if self.scalar_quantities:
            _view_scalar_quantities(mesh, ps_mesh, dict_sphere_fit_residual, alpha, scattered)

    def _draw_tensions(self, nx_graph: nx.Graph, scattered: bool, dict_sphere_fit_residual: dict | None) -> None:
        mesh, alpha = self.mesh, self.alpha
        # A seed ensemble's spread is added before the tensions: Polyscope shows the last colour quantity enabled.
        ps_mesh = self._draw_seed_spread(self.ensemble, scattered) if self.ensemble is not None else None
        ps_mesh = view_tensions_on_mesh(mesh, self.tensions, alpha=alpha, ps_mesh=ps_mesh, scattered=scattered,
                                        name_values=_TENSIONS)
        _display_embryo_graph_forces(
            nx_graph,
            alpha=alpha,
            clean_before=False,
            clean_after=False,
            show=False,
            base_pressure=0,
            plot_pressures=False,
        )
        if self.scalar_quantities:
            _view_scalar_quantities(mesh, ps_mesh, dict_sphere_fit_residual, alpha, scattered)
        if self.ensemble is not None:
            self._draw_flagged_interfaces(self.ensemble, scattered)

    def _draw_seed_spread(self, ensemble: EnsembleOnMesh, scattered: bool) -> ps.SurfaceMesh | None:
        """The seeds' spread (SD) on the surface, with `jet` between two quantiles (`quantity_colours`)."""
        spread = ensemble.tension_seed_sd
        if not np.isfinite(np.array(list(spread.values()), dtype=np.float64)).any():
            return None
        return view_dict_values_on_mesh(
            self.mesh, spread, alpha=self.alpha, ps_mesh=None, scattered=scattered,
            clean_before=False, clean_after=False, show=False, name_values=SEED_SPREAD_TITLE,
        )

    def _draw_flagged_interfaces(self, ensemble: EnsembleOnMesh, scattered: bool) -> None:
        """The interfaces found by fewer than half of the seeds: their triangles' edges, in magenta, over the surface.

        A wireframe rather than a second surface: a surface on the same triangles as the coloured one would fight
        it for depth, and would hide the interface's own colour.
        """
        flagged = set(ensemble.found_by_fewer_than_half_the_seeds)
        if not flagged:
            return
        v, f, idx = ((self.mesh.v_scattered, self.mesh.f_scattered, self.mesh.idx_scattered) if scattered
                     else (self.mesh.v, self.mesh.f, np.arange(len(self.mesh.f))))
        pairs = np.asarray(self.mesh.f)[idx][:, 3:5]
        marked = np.array([(int(a), int(b)) in flagged for a, b in pairs], dtype=bool)
        if marked.any():
            triangles = np.asarray(f)[marked][:, :3]
            edges = np.unique(np.sort(triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0)
            # Only these triangles' vertices are nodes: Polyscope draws every node of a curve network.
            nodes, edges = np.unique(edges, return_inverse=True)
            ps.register_curve_network(_FEW_SEEDS_SURFACE, np.asarray(v)[nodes], edges.reshape(-1, 2),
                                      color=_FEW_SEEDS_COLOR, radius=0.0015)

    def _draw_scale_bar(self) -> None:
        if self.scale_bar is None:
            return
        nodes = np.array([self.scale_bar.start, self.scale_bar.end])
        network = ps.register_curve_network(f"Scale bar ({self.scale_bar.label})", nodes, np.array([[0, 1]]),
                                            color=_on_background(_SCALE_BAR_COLOR), radius=0.004)
        network.set_enabled(self.show_scale_bar)

    def _colour_scales(self, scattered: bool | None = None) -> ViewerColourScales:
        scattered = self.scattered if scattered is None else scattered
        tensions = self.tensions if self.kind != "pressure" else None
        pressures = self.pressures if self.kind != "tension" else None
        scales = viewer_colour_scales(self.mesh, tensions, pressures, self.alpha, scattered=scattered)
        if self.ensemble is not None:
            spread = quantity_colour_scale(SEED_SPREAD_TITLE, self.ensemble.tension_seed_sd, self.alpha)
            if spread is not None:
                scales = replace(scales, scales=(*scales.scales, spread))
        return scales

    # -----------------------------------------------------------------------------------------
    # the window
    # -----------------------------------------------------------------------------------------
    def open(self, *, save_figure: "str | os.PathLike[str] | None", show: bool) -> None:
        """Draw, write the figure if asked, then show the window with its panel until it is closed."""
        self.draw()
        if save_figure is not None:
            self.save_figure(save_figure, render_first=True)
        if not show:
            return
        ps.set_open_imgui_window_for_user_callback(False)
        ps.set_user_callback(self.panel)
        try:
            ps.show()
        finally:
            ps.clear_user_callback()
            self.close()

    def caption(self) -> str:
        """What a saved figure says about its values, under its colour bars."""
        return f"Values: {self.values_from}"

    def scale_bar_pixels(self, width: int, height: int) -> np.ndarray | None:
        """Where the scale bar's ends fall in the current view, on an image `width` x `height` (perspective only)."""
        if self.scale_bar is None or ps.get_view_projection_mode() != "perspective":
            return None
        camera = ps.get_view_camera_parameters()
        return project_to_pixels(np.array([self.scale_bar.start, self.scale_bar.end]), camera.get_view_mat(),
                                 camera.get_fov_vertical_deg(), width, height)

    def save_figure(self, path: "str | os.PathLike[str]", *, render_first: bool = False) -> Path:
        """Write the current view with its colour bars, its scale bar's length, the frame's labels and its caption.

        The view is Polyscope's own rendering of the scene without the user interface
        (`screenshot_to_buffer(include_UI=False)`); the bars and the text are drawn by matplotlib.
        """
        if render_first:
            # Polyscope keeps its camera from one viewer call to the next in a process; a figure written on opening
            # is framed on this mesh, as the first window of a new process would be.
            ps.reset_camera_to_home_view()
            ps.show(forFrames=1)
        image = ps.screenshot_to_buffer(transparent_bg=False, include_UI=False)
        height, width = image.shape[:2]
        shown_bar = self.scale_bar if self.show_scale_bar else None
        return compose_figure(image, self.scales, path, scale_bar=shown_bar,
                              scale_bar_pixels=self.scale_bar_pixels(width, height) if shown_bar else None,
                              caption=self.caption(), labels=self.frame_labels(width, height))

    def save_colour_bars(self, path: "str | os.PathLike[str]") -> Path:
        """Write the window's colour bars: `save_colour_bars` when they are its bars, else the panel's bars.

        `save_colour_bars` draws the tension bar and the surface's pressure bar of an unscattered view; the
        scattered view's pressure scale, a pressure viewer's lone bar and a seed ensemble's spread are written by
        `compose_figure` without a view, with the same colours and ranges.
        """
        titles = [scale.title for scale in self.scales.scales]
        if not self.scattered and self.ensemble is None and self.kind in ("force", "tension") and (
                titles and titles[0] == TENSION_TITLE):
            save_colour_bars(self.mesh, self.tensions, self.pressures, path, self.alpha)
            return Path(path)
        return compose_figure(None, self.scales, path, caption=self.caption())

    def recompute(self) -> None:
        """Infer again with the methods, solver, diagnostics and pressures chosen in the panel, and redraw.

        A force viewer with *pressures* unticked draws the tensions only (`infer_tensions`, as `foambryo infer
        --no-pressures`), and its pressures again once ticked.
        """
        with_pressures = self.pressures_choice and self.base_kind != "tension"
        options = _InferenceOptions(
            list(TensionComputationMethod)[self.tension_choice], list(PressureComputationMethod)[self.pressure_choice],
            self.options.mean_tension, self.options.base_pressure, _SOLVERS[self.solver_choice],
            pressures=with_pressures, diagnostics=_DIAGNOSTICS[self.diagnostics_choice])
        kind = ("force" if self.base_kind != "tension" else "tension") if with_pressures else "tension"
        tensions, pressures, values_from = options.infer(self.mesh, kind, None, None, None)
        self.options, self.tensions, self.values_from = options, tensions, values_from
        self.pressures = pressures if with_pressures else None
        self.kind = kind if self.base_kind != "pressure" or not with_pressures else "pressure"
        self.inference = options.inference
        if self.kind == "tension":
            self.showing = "tension"
        self.redraw()

    def export(self, folder: "str | os.PathLike[str]", *, overwrite: bool = False) -> list[str]:
        """Write `foambryo infer`'s files for the values shown in `folder` (`foambryo._window_export`).

        Raises:
            FileExistsError: when the folder holds foambryo's files and `overwrite` is false.
            ValueError: for a seed ensemble (written by `foambryo infer-over-seeds`), or without tensions.
        """
        if self.ensemble is not None:
            message = "a seed ensemble's files are foambryo infer-over-seeds' (its command line is in the panel)"
            raise ValueError(message)
        inference = self.inference
        if inference is None:  # values given to the window: their tables, and a record that says so
            outcome = InferenceOutcome(self.tensions, self.pressures, library_call=self.values_from)
            inference = WindowInference(outcome, self.options.record_options(self.pressures is not None),
                                        given=True)
        return export_window_values(folder, self.mesh, inference, mesh_file=self.mesh_name, rec_bytes=self.rec_bytes,
                                    overwrite=overwrite)

    def panel(self) -> None:
        """The "foambryo" panel: Polyscope's user callback, drawn every frame."""
        _begin_panel()
        self.panel_body(methods=self.ensemble is None)
        psim.End()
        self._label_overlays()

    def panel_body(self, *, methods: bool = True) -> None:
        """The panel's contents: values, bars, display, explode and cut, saving, export, and the methods.

        The methods (and Recompute) only with `methods`.
        """
        self.poll_redraw()
        psim.TextWrapped(f"Values: {self.values_from}")
        if self.preparing:
            psim.TextDisabled("drawing in the background ...")
        if self.scales is None:  # the first scene is still being prepared
            return
        psim.Separator()
        self._panel_showing()
        self._panel_bars()
        psim.Separator()
        self._panel_display()
        if psim.CollapsingHeader("Explode and cut", psim.ImGuiTreeNodeFlags_DefaultOpen):
            self._panel_explode_and_cut()
        if psim.CollapsingHeader("Save figure, colour bars", psim.ImGuiTreeNodeFlags_DefaultOpen):
            self._panel_saving()
        if psim.CollapsingHeader("Export (foambryo infer's files)", psim.ImGuiTreeNodeFlags_DefaultOpen):
            self._panel_export()
        if methods and psim.CollapsingHeader("Methods, solver, Recompute", psim.ImGuiTreeNodeFlags_DefaultOpen):
            self._panel_methods()
        if self.status:
            psim.TextWrapped(self.status)

    def _panel_bars(self) -> None:
        """The colour bars, and the swatches of what has no value and of the ensemble's flags."""
        for scale in self.scales.scales:
            _imgui_colour_bar(scale)
        if self.scales.undetermined:
            _imgui_swatch(UNDETERMINED_COLOR, "undetermined (nan): green outline, no fill (a cell: green)",
                          outline=True)
        if self.scales.crop:
            _imgui_swatch(CROP_FACE_COLOR, "crop face (no value): see-through, edged", edge=CROP_FACE_EDGE_COLOR)
        ensemble = self.ensemble
        if ensemble is not None:
            _imgui_swatch(np.array(_FEW_SEEDS_COLOR), f"found by fewer than half of the seeds "
                          f"({len(ensemble.found_by_fewer_than_half_the_seeds)} interfaces)")
            psim.TextWrapped(f"{ensemble.missing_from_the_ensemble} interfaces of the mesh are not in the "
                             f"ensemble (green outlines); {ensemble.not_on_the_mesh} of the ensemble are not on the "
                             "mesh.")

    def _panel_showing(self) -> None:
        """With tensions and pressures: which one colours the surface (the tensions first)."""
        if self.tensions is None or self.pressures is None or self.kind != "force":
            return
        psim.TextUnformatted("colour the surface by")
        for value, name in (("tension", "tensions"), ("pressure", "pressures")):
            psim.SameLine()
            if psim.RadioButton(name, self.showing == value) and self.showing != value:
                self.showing = value
                self.apply_showing()

    def _panel_explode_and_cut(self) -> None:
        changed, scattered = psim.Checkbox("explode (each cell moved away from the centre)", self.scattered)
        changed_coeff, coeff = psim.SliderFloat("explode coefficient", self.scatter_coeff, 0.0, 1.0)
        if changed_coeff:
            self.scatter_coeff = coeff
        if changed or (changed_coeff and self.scattered):  # followed live: drawn as soon as it is prepared
            self.scattered = scattered
            self.request_redraw()
        psim.TextUnformatted("cut with a plane normal to")
        axis = None if self.cut is None else self.cut[0]
        for index, name in enumerate(("z", "y", "x")):
            psim.SameLine()
            if psim.RadioButton(name, axis == index):
                fraction, keep = (0.5, False) if self.cut is None else self.cut[1:]
                self.cut = (index, fraction, keep)
                self.apply_cut()
        psim.SameLine()
        if psim.RadioButton("no cut", axis is None):
            self.cut = None
            self.apply_cut()
        if self.cut is not None:
            index, fraction, keep = self.cut
            low, high = self.drawn_box()
            unit = self.scale_bar.unit if self.scale_bar is not None else "mesh units"
            changed, position = psim.SliderFloat(f"position ({unit})", float(low[index] + fraction * (high[index] -
                                                 low[index])), float(low[index]), float(high[index]))
            changed_keep, keep = psim.Checkbox("keep the higher side", keep)
            if changed or changed_keep:
                span = high[index] - low[index]
                self.cut = (index, (position - low[index]) / span if span else 0.5, keep)
                self.apply_cut()

    def _explode_and_say(self, scattered: bool, coeff: float) -> str:
        self.set_explode(scattered, coeff)
        return f"exploded by {coeff:g}" if scattered else "not exploded"

    def _panel_display(self) -> None:
        changed, self.flat_colours = psim.Checkbox("flat colours (as on the bars, no shading)", self.flat_colours)
        if changed and ps.has_surface_mesh(_SURFACE):
            ps.get_surface_mesh(_SURFACE).set_material("flat" if self.flat_colours else "clay")
        if self.scale_bar is not None:
            changed, self.show_scale_bar = psim.Checkbox(f"scale bar, {self.scale_bar.label}", self.show_scale_bar)
            if changed:
                ps.get_curve_network(f"Scale bar ({self.scale_bar.label})").set_enabled(self.show_scale_bar)
        else:
            psim.TextDisabled("no scale bar: the mesh's length unit is not known")
        unit = f"{self.length_unit}" if self.length_unit else "mesh units"
        changed, self.show_frame = psim.Checkbox(f"x, y, z frame ({unit})", self.show_frame)
        if changed and ps.has_curve_network(_FRAME):
            ps.get_curve_network(_FRAME).set_enabled(self.show_frame)
        if ps.has_curve_network(_UNDETERMINED_OUTLINE):
            changed, self.show_outlines = psim.Checkbox("undetermined tensions' outlines", self.show_outlines)
            if changed:
                self.apply_showing()
        if ps.has_surface_mesh("Crop faces"):
            crop = ps.get_surface_mesh("Crop faces")
            changed, shown = psim.Checkbox("crop faces (glass)", crop.is_enabled())
            if changed:
                crop.set_enabled(shown)
        _panel_background(self)

    def _panel_saving(self) -> None:
        _, self.figure_path = psim.InputText("##figure", self.figure_path)
        psim.SameLine()
        if psim.Button("Save figure"):
            self._try(lambda: f"wrote {self.save_figure(next_free_path(self.figure_path))}")
        _, self.bars_path = psim.InputText("##bars", self.bars_path)
        psim.SameLine()
        if psim.Button("Save colour bars"):
            self._try(lambda: f"wrote {self.save_colour_bars(next_free_path(self.bars_path))}")
        psim.TextDisabled("PNG, PDF or SVG, by the suffix; never overwrites")

    def _panel_export(self) -> None:
        """A folder, Export, and Replace / Cancel when it holds foambryo's files (any other file is never touched)."""
        _, self.export_folder = psim.InputText("##export", self.export_folder)
        psim.SameLine()
        if psim.Button("Export") and self.export_asking is None:
            existing = earlier_window_outputs(self.export_folder)
            if existing:
                self.export_asking = existing
            else:
                self._try(self._export_and_say)
        if self.export_asking is not None:
            psim.TextWrapped(f"{self.export_folder} already holds {', '.join(self.export_asking)}. Replace them?")
            if psim.Button("Cancel##export"):  # the safe choice first
                self.export_asking, self.status = None, "export cancelled: nothing was written"
            psim.SameLine()
            if psim.Button("Replace##export"):
                self.export_asking = None
                self._try(lambda: self._export_and_say(overwrite=True))
        psim.TextDisabled("tensions.csv, pressures.csv, solves.csv (with diagnostics), run.json, colour_bars.png")

    def _export_and_say(self, *, overwrite: bool = False) -> str:
        written = self.export(self.export_folder, overwrite=overwrite)
        return f"wrote {', '.join(written)} in {self.export_folder}"

    def _panel_methods(self) -> None:
        names = [member.name for member in TensionComputationMethod]
        _, self.tension_choice = psim.Combo("tension method", self.tension_choice, names)
        if self.base_kind != "tension":
            _, self.pressures_choice = psim.Checkbox("pressures (off: the tensions only)", self.pressures_choice)
            if self.pressures_choice:
                names = [member.name for member in PressureComputationMethod]
                _, self.pressure_choice = psim.Combo("pressure method", self.pressure_choice, names)
        _, self.solver_choice = psim.Combo("solver", self.solver_choice, list(_SOLVERS))
        _, self.diagnostics_choice = psim.Combo("diagnostics (for Export)", self.diagnostics_choice,
                                                list(_DIAGNOSTICS))
        if psim.Button("Recompute"):
            self._try(self._recompute_and_say)
        command = self.command_line()
        if command:
            psim.TextDisabled("the same numbers as files:")
            psim.TextWrapped(command)

    def command_line(self) -> str:
        """`foambryo infer MESH -o results ...` for the values drawn, when the viewer inferred them from a mesh file."""
        if self.mesh_name is None or not self.values_from.startswith("inferred"):
            return ""
        import shlex

        arguments = self.options.infer_arguments()
        if self.options.diagnostics != "none":
            arguments += ["--diagnostics", self.options.diagnostics]
        return " ".join(["foambryo infer", shlex.quote(self.mesh_name), "-o", "results", *arguments])

    def _recompute_and_say(self) -> str:
        self.recompute()
        return "recomputed"

    def _try(self, action: "Callable[[], str]") -> None:
        """Run a panel action; a refusal is shown in the panel, not raised into Polyscope's frame."""
        try:
            self.status = action()
        except (ValueError, OSError, ArithmeticError) as error:
            self.status = f"{type(error).__name__}: {error}"

    def _label_overlays(self) -> None:
        """Write the scale bar's length beside it and the frame's labels at its tips, in the window.

        Not in screenshots: a saved figure writes its own.
        """
        width, height = psim.GetIO().DisplaySize
        labels = list(self.frame_labels(int(width), int(height)))
        if self.scale_bar is not None and self.show_scale_bar:
            ends = self.scale_bar_pixels(int(width), int(height))
            if ends is not None and np.isfinite(ends).all():
                x, y = ends.mean(axis=0)
                labels.append((self.scale_bar.label, float(x), float(y) + 6.0 + psim.CalcTextSize("0")[1] / 2))
        draw = psim.GetForegroundDrawList()
        for text, x, y in labels:
            size = psim.CalcTextSize(text)
            draw.AddText((x - size[0] / 2, y - size[1] / 2), psim.GetColorU32((*_on_background((0.0, 0.0, 0.0)), 1.0)),
                         text)


class _MaskWindow:
    """The window of `_view_mask_workflow`: the mask first, then its mesh, then the values, all in one window.

    The workflow (`foambryo._mask_workflow.MaskWorkflow`) runs the reconstruction and the inference in child
    processes; the panel polls it every frame (the window keeps drawing while they run) and draws, once values
    exist, a `_ForceWindow` for them, with the mask kept in the scene (`after_draw`) and the panel's explode and
    cut. The inference choices are the workflow's, so the window's own method choice and Recompute are not shown.
    """

    def __init__(
        self,
        workflow: "MaskWorkflow",
        *,
        scattered: bool = False,
        scatter_coeff: float = DEFAULT_SCATTER_COEFF,
        scale_bar: bool | float = True,
        alpha: float = 0.05,
    ) -> None:
        from foambryo._mask_workflow import mask_point_cloud

        self.workflow = workflow
        self.cloud = mask_point_cloud(workflow.mask, workflow.spacing)
        self.show_mask = True
        self.scattered, self.scatter_coeff = scattered, scatter_coeff
        self.scale_bar, self.alpha = scale_bar, alpha
        self.window: _ForceWindow | None = None
        self.cut: tuple[int, float, bool] | None = None
        self.slice_plane = None

    # -- the scene ------------------------------------------------------------------------------
    def draw_mask(self) -> None:
        """The mask's cell voxels as points, one colour per label (the medium left out)."""
        cloud = ps.register_point_cloud(_MASK, self.cloud.points, radius=self.cloud.radius, point_render_mode="quad",
                                        enabled=self.show_mask)
        cloud.set_radius(self.cloud.radius, relative=False)
        cloud.add_color_quantity("label", self.cloud.colours, enabled=True)

    def draw_mesh(self) -> None:
        """The reconstruction, plain and half transparent, over its mask, before any value exists."""
        mesh = self.workflow.mesh
        surface = ps.register_surface_mesh(_RECONSTRUCTION, np.asarray(mesh.v), np.asarray(mesh.f)[:, :3])
        surface.set_color((0.85, 0.85, 0.85))
        surface.set_transparency(0.6)
        surface.set_back_face_policy("identical")

    def draw(self) -> None:
        """Everything the workflow has: the mask, then the mesh or the values' window."""
        ps.remove_all_structures()
        if self.window is not None:
            self.window.draw()  # draws the mask too, through after_draw
            return
        self.draw_mask()
        if self.workflow.mesh is not None:
            self.draw_mesh()
        self._apply_cut()

    def show_values(self) -> None:
        """Make the values' window from the workflow's last inference, and draw it."""
        from foambryo.results import PRESSURE_METHODS, TENSION_METHODS, ensemble_library_call

        workflow, choices = self.workflow, self.workflow.inferred_with
        options = _InferenceOptions(TENSION_METHODS[choices.tension_method], PRESSURE_METHODS[choices.pressure_method],
                                    choices.mean_tension, choices.base_pressure, choices.solver,
                                    pressures=choices.pressures, diagnostics=choices.diagnostics)
        unit = workflow.unit
        common = {"alpha": self.alpha, "scattered": self.scattered, "scatter_coeff": self.scatter_coeff,
                  "length_unit": unit, "scale_bar": self.scale_bar}
        if workflow.ensemble is not None:
            result = workflow.ensemble
            ensemble = ensemble_on_mesh(workflow.mesh, result.tensions, result.tension_seed_sd,
                                        result.found_by_fewer_than_half_the_seeds)
            call = ensemble_library_call(n_seeds=choices.seeds, min_distance=workflow.min_distance,
                                         mode=TENSION_METHODS[choices.tension_method],
                                         angle_reading=choices.angle_reading, combine=choices.combine,
                                         max_workers=choices.max_workers, spacing=workflow.spacing)
            window = _ForceWindow(workflow.mesh, "tension", ensemble.tensions, None, options,
                                  f"inferred here (shown on the reconstruction's mesh, seed 42): {call}",
                                  ensemble=ensemble, **common)
        else:
            outcome = workflow.outcome
            if outcome.tensions is None:
                return
            kind = "force" if outcome.pressures is not None else "tension"
            inference = WindowInference(outcome, choices.record_options(), workflow.inferred_date,
                                        workflow.inferred_seconds, workflow.warnings)
            window = _ForceWindow(workflow.mesh, kind, outcome.tensions, outcome.pressures, options,
                                  f"inferred here: {outcome.library_call}", inference=inference,
                                  rec_bytes=workflow.reconstruction.rec_bytes, **common)
        window.after_draw = self._after_values_drawn
        window.cut, window.slice_plane = self.cut, self.slice_plane
        self.window = window
        # The mask's points would speckle the coloured surface and do not explode with it: hidden, not removed.
        self.show_mask = False
        if ps.has_point_cloud(_MASK):
            ps.get_point_cloud(_MASK).set_enabled(False)
        # Prepared off the window's thread: the mask and the reconstruction stay on screen until the values replace
        # the reconstruction.
        window.request_redraw()

    def _after_values_drawn(self) -> None:
        """Once the values are drawn: the plain reconstruction removed, the mask kept (drawn once, then reused)."""
        if ps.has_surface_mesh(_RECONSTRUCTION):
            ps.remove_surface_mesh(_RECONSTRUCTION)
        if not ps.has_point_cloud(_MASK):
            self.draw_mask()

    def _apply_cut(self) -> None:
        """Before any value: the cut on the mask's (or the mesh's) box."""
        if self.cut is None:
            if self.slice_plane is not None:
                self.slice_plane.set_active(False)
            return
        axis, fraction, keep_higher = self.cut
        points = self.workflow.mesh.v if self.workflow.mesh is not None else self.cloud.points
        low, high = np.min(points, axis=0), np.max(points, axis=0)
        point = (low + high) / 2
        point[axis] = low[axis] + fraction * (high[axis] - low[axis])
        normal = np.zeros(3)
        normal[axis] = 1.0 if keep_higher else -1.0
        if self.slice_plane is None:
            self.slice_plane = ps.add_scene_slice_plane()
            self.slice_plane.set_draw_widget(False)
        self.slice_plane.set_active(True)
        self.slice_plane.set_pose(point, normal)

    # -- the window -------------------------------------------------------------------------------
    def open(self) -> None:
        """Show the mask and the panel until the window is closed; a job still running is cancelled then."""
        ps.init()
        self.draw()
        ps.reset_camera_to_home_view()
        ps.set_open_imgui_window_for_user_callback(False)
        ps.set_user_callback(self.panel)
        try:
            ps.show()
        finally:
            ps.clear_user_callback()
            self.workflow.cancel()
            if self.window is not None:
                self.window.close()

    def panel(self) -> None:
        """The panel: the workflow's steps, then the values' window's own sections."""
        event = self.workflow.poll()
        if event == "reconstructed":
            self.window = None
            self.draw()
        elif event == "inferred":
            self.show_values()
        _begin_panel()
        psim.TextWrapped(self.workflow.progress_text() or self.workflow.status)
        if self.window is not None:  # the values first, with their colour bars; the steps below
            if psim.CollapsingHeader("Values", psim.ImGuiTreeNodeFlags_DefaultOpen):
                self.window.panel_body(methods=False)
            self.cut, self.slice_plane = self.window.cut, self.window.slice_plane
            self.scattered, self.scatter_coeff = self.window.scattered, self.window.scatter_coeff
        for title, section in (("Mask and reconstruction", self._panel_mask), ("Inference", self._panel_inference)):
            if psim.CollapsingHeader(title, psim.ImGuiTreeNodeFlags_DefaultOpen):
                section()
        if self.window is None and psim.CollapsingHeader("Cut", psim.ImGuiTreeNodeFlags_DefaultOpen):
            self._panel_cut()
        psim.End()
        if self.window is not None:
            self.window._label_overlays()

    def _panel_mask(self) -> None:
        workflow = self.workflow
        spacing = " x ".join(f"{v:g}" for v in workflow.spacing)
        psim.TextWrapped(f"Mask {workflow.mask_name}: {workflow.mask.shape[0]} x {workflow.mask.shape[1]} x "
                         f"{workflow.mask.shape[2]} voxels of {spacing} {workflow.unit}")
        stride = f", every {self.cloud.stride}th voxel" if self.cloud.stride > 1 else ""
        changed, self.show_mask = psim.Checkbox(f"show the mask ({len(self.cloud.points)} points{stride})",
                                                self.show_mask)
        if changed and ps.has_point_cloud(_MASK):
            ps.get_point_cloud(_MASK).set_enabled(self.show_mask)
        if ps.has_surface_mesh(_RECONSTRUCTION):
            psim.TextDisabled("the reconstruction is drawn half transparent over the mask: Infer colours it")
        busy = workflow.busy
        _, workflow.min_distance = psim.InputInt("min distance (cubic voxels)", workflow.min_distance)
        workflow.min_distance = max(1, workflow.min_distance)
        psim.TextDisabled(f"estimate: {workflow.estimate().text()}")
        if psim.Button("Reconstruct") and not busy:
            workflow.start_reconstruction()
        psim.SameLine()
        if psim.Button("Cancel") and busy:
            workflow.cancel()
        if self.window is None:  # with values, the background is chosen in their section
            _panel_background(None)

    def _panel_inference(self) -> None:
        from foambryo.results import PRESSURE_METHODS, TENSION_METHODS
        from foambryo.seed_ensembles import COMBINE_OPTIONS

        workflow = self.workflow
        choices = workflow.choices
        modes = ("one mesh", "seed ensemble")
        mode = modes.index(choices.mode)
        for index, name in enumerate(modes):
            if index:
                psim.SameLine()
            if psim.RadioButton(name, mode == index):
                mode = index
        tensions, pressures = list(TENSION_METHODS), list(PRESSURE_METHODS)
        _, tension = psim.Combo("tension method", tensions.index(choices.tension_method), tensions)
        with_pressures, pressure = choices.pressures, pressures.index(choices.pressure_method)
        if mode == 0:
            _, with_pressures = psim.Checkbox("pressures (off: the tensions only)", choices.pressures)
            if with_pressures:
                _, pressure = psim.Combo("pressure method", pressure, pressures)
        _, solver = psim.Combo("solver", _SOLVERS.index(choices.solver), list(_SOLVERS))
        diagnostics = _DIAGNOSTICS.index(choices.diagnostics)
        if mode == 0:
            _, diagnostics = psim.Combo("diagnostics (for Export)", diagnostics, list(_DIAGNOSTICS))
        seeds, workers, combine = choices.seeds, choices.max_workers, list(COMBINE_OPTIONS).index(choices.combine)
        if mode == 1:
            _, seeds = psim.InputInt("seeds (0 ... N-1), N", choices.seeds)
            _, workers = psim.InputInt("workers (processes)", choices.max_workers)
            _, combine = psim.Combo("combine", combine, list(COMBINE_OPTIONS))
            psim.TextDisabled("the seed spread is a lower bound on the error, not an error bar")
        workflow.choices = replace(choices, mode=modes[mode], tension_method=tensions[tension],
                                  pressure_method=pressures[pressure], pressures=with_pressures,
                                  solver=_SOLVERS[solver], diagnostics=_DIAGNOSTICS[diagnostics],
                                  seeds=max(2, seeds), max_workers=max(1, workers),
                                  combine=list(COMBINE_OPTIONS)[combine])
        self._panel_infer_buttons()

    def _panel_infer_buttons(self) -> None:
        """Infer and Cancel, and the command lines that give the same numbers."""
        workflow = self.workflow
        ready = workflow.mesh is not None or workflow.choices.mode == "seed ensemble"
        if psim.Button("Infer") and ready and not workflow.busy:
            workflow.start_inference()
        psim.SameLine()
        if psim.Button("Cancel##inference") and workflow.busy:
            workflow.cancel()
        if not ready:
            psim.SameLine()
            psim.TextDisabled("reconstruct first (or choose a seed ensemble)")
        psim.TextDisabled("the same numbers from the command line:")
        for line in workflow.commands():
            psim.TextWrapped(line)

    def _panel_cut(self) -> None:
        psim.TextUnformatted("cut with a plane normal to")
        axis = None if self.cut is None else self.cut[0]
        for index, name in enumerate(("z", "y", "x")):
            psim.SameLine()
            if psim.RadioButton(name, axis == index):
                self.cut = (index, 0.5 if self.cut is None else self.cut[1], False if self.cut is None else self.cut[2])
                self._apply_cut()
        psim.SameLine()
        if psim.RadioButton("no cut", axis is None):
            self.cut = None
            self._apply_cut()
        if self.cut is not None:
            index, fraction, keep = self.cut
            changed, fraction = psim.SliderFloat("position (fraction of the box)", fraction, 0.0, 1.0)
            changed_keep, keep = psim.Checkbox("keep the higher side", keep)
            if changed or changed_keep:
                self.cut = (index, fraction, keep)
                self._apply_cut()


def _view_mask_workflow(
    workflow: "MaskWorkflow",
    *,
    scattered: bool = False,
    scatter_coeff: float = DEFAULT_SCATTER_COEFF,
    scale_bar: bool | float = True,
) -> None:
    """Open the viewer on a segmentation mask (`foambryo view MASK --spacing ...`) until the window is closed.

    The mask is shown first, as its cell voxels in one colour per label; the panel reconstructs it
    (`foambryo.io.reconstruct_mask`, `foambryo mesh`'s call) and infers (`infer_tensions_then_pressures` on the
    mesh, or `infer_tensions_over_seeds` on the mask), each in a child process with its elapsed time and estimate
    and a Cancel; the values are then drawn as by `plot_force_inference`, with their colour bars, explode and cut.

    Args:
        workflow: the mask, its voxel size and the choices (`foambryo._mask_workflow.MaskWorkflow`).
        scattered: draw the values exploded, as `plot_force_inference(scattered=...)`.
        scatter_coeff: how far, as `plot_force_inference(scatter_coeff=...)`.
        scale_bar: as for `plot_force_inference` (the length unit is the workflow's).
    """
    _MaskWindow(workflow, scattered=scattered, scatter_coeff=scatter_coeff, scale_bar=scale_bar).open()


def view_mask(
    mask: "np.ndarray",
    spacing: tuple[float, float, float],
    *,
    spacing_unit: str = "um",
    min_distance: int | None = None,
    mean_tension: float = _INFER_FORCES_DEFAULTS["mean_tension"],
    base_pressure: float = _INFER_FORCES_DEFAULTS["base_pressure"],
    mode_tension: TensionComputationMethod = _INFER_FORCES_DEFAULTS["mode_tension"],
    mode_pressure: PressureComputationMethod = _INFER_FORCES_DEFAULTS["mode_pressure"],
    solver: str = _INFER_FORCES_DEFAULTS["solver"],
    pressures: bool = True,
    scattered: bool = False,
    scatter_coeff: float = DEFAULT_SCATTER_COEFF,
    scale_bar: bool | float = True,
    mask_name: str = "mask.npz",
) -> None:
    """Open the viewer on a segmentation mask: the window `foambryo view MASK --spacing DZ DY DX` opens.

    The mask is shown first (each cell's voxels as points in its own colour); the panel reconstructs it as
    `foambryo mesh` does (`foambryo.io.reconstruct_mask`) and infers as `foambryo infer` does
    (`infer_tensions_then_pressures`) or `foambryo infer-over-seeds` (`infer_tensions_over_seeds`), each in a child
    process with its elapsed time, the estimate (`foambryo.reconstruction_cost`) and Cancel; the values are then
    drawn with their colour bars, explode, cut and export, as by `plot_force_inference`. Returns when the window is
    closed (a job still running is cancelled then).

    Needs Polyscope (`pip install "foambryo[viewing]"`), delaunay-watershed-3d and a display.

    Args:
        mask: the segmentation, a 3-D array of non-negative integer labels, axis 0 = z, axis 1 = y, axis 2 = x,
            0 the medium (`check_segmentation_mask`'s checks).
        spacing: the voxel size `(dz, dy, dx)`, three finite numbers > 0, in `spacing_unit`; an anisotropic mask
            is resampled to cubic voxels of the shortest spacing before meshing, as by `foambryo mesh` (only one
            axis may be longer than the other two). There is no default: a wrong spacing changes every angle.
        spacing_unit: the spacing's unit, the mesh's length unit, written on the scale bar (`--spacing-unit`).
            Defaults to `"um"`. Keyword-only.
        min_distance: the reconstruction's `--min-distance`, in cubic voxels of the shortest spacing; `None` (the
            default) is delaunay-watershed-3d's default (3). The panel changes it. Keyword-only.
        mean_tension: `infer_forces`' option, with its default (1); the panel starts with it. Keyword-only.
        base_pressure: likewise (default 0). Keyword-only.
        mode_tension: likewise (default `TensionComputationMethod.YoungDupre`). Keyword-only.
        mode_pressure: likewise (default `PressureComputationMethod.Variational`). Keyword-only.
        solver: likewise (default `"auto"`). Keyword-only.
        pressures: `False` infers the tensions only (`--no-pressures`). Defaults to True. Keyword-only.
        scattered: draw the values exploded (`--scattered`). Defaults to False. Keyword-only.
        scatter_coeff: by this coefficient (`--scattered COEFF`). Defaults to 0.2. Keyword-only.
        scale_bar: as for `plot_force_inference`, in `spacing_unit`. Defaults to True. Keyword-only.
        mask_name: the mask's file name in the equivalent command lines the panel shows and in the messages of a
            refused mask. Defaults to `"mask.npz"`. Keyword-only.

    Raises:
        ValueError: if the mask or the spacing is refused (the messages of `foambryo mesh`'s checks; a refused
            mask raises `foambryo.segmentation_masks.SegmentationMaskError`, a `ValueError`).
    """
    from dw3d import get_default_mesh_reconstruction_algorithm
    from dw3d.mask_preprocessing import coarse_axis_and_factor

    from foambryo._mask_workflow import MaskWorkflow, ViewerChoices
    from foambryo.results import option_name
    from foambryo.segmentation_masks import check_segmentation_mask

    mask = check_segmentation_mask(mask, name=mask_name)
    spacing = tuple(float(value) for value in spacing)
    if len(spacing) != 3 or not all(np.isfinite(value) and value > 0 for value in spacing):
        message = f"spacing must be three positive finite numbers (dz, dy, dx), not {spacing!r}"
        raise ValueError(message)
    coarse_axis_and_factor(spacing)  # refuses a spacing the resampling cannot handle, with its message
    if min_distance is None:
        min_distance = inspect.signature(get_default_mesh_reconstruction_algorithm).parameters["min_distance"].default
    choices = ViewerChoices.library(
        tension_method=option_name(mode_tension), pressure_method=option_name(mode_pressure),
        mean_tension=float(mean_tension), base_pressure=float(base_pressure), solver=solver, pressures=pressures)
    workflow = MaskWorkflow(mask, spacing, spacing_unit, mask_name, int(min_distance), choices)
    _view_mask_workflow(workflow, scattered=scattered, scatter_coeff=scatter_coeff, scale_bar=scale_bar)


def _begin_panel() -> None:
    """Open the "foambryo" ImGui window at the top right, as tall as the window (it scrolls), the first time."""
    width, height = ps.get_window_size()
    psim.SetNextWindowPos((width - 400.0, 10.0), psim.ImGuiCond_FirstUseEver)
    psim.SetNextWindowSize((390.0, max(200.0, height - 20.0)), psim.ImGuiCond_FirstUseEver)
    psim.Begin("foambryo")


def _imgui_colour_bar(scale: ColourScale, width: float = 300.0, height: float = 16.0) -> None:
    """One colour bar in the panel: the library's bar colours as flat rectangles, its two ends written under it."""
    psim.TextUnformatted(scale.title)
    psim.TextDisabled(scale.clipping)
    draw = psim.GetWindowDrawList()
    x0, y0 = psim.GetCursorScreenPos()
    colours = scale.colours()
    n = len(colours)
    for i, (r, g, b) in enumerate(colours):
        draw.AddRectFilled((x0 + width * i / n, y0), (x0 + width * (i + 1) / n, y0 + height),
                           psim.GetColorU32((float(r), float(g), float(b), 1.0)))
    start = psim.GetCursorPosX()
    psim.Dummy((width, height))
    low, high = f"{scale.low:.4g}", f"{scale.high:.4g}"
    psim.TextUnformatted(low)
    if scale.colour_function == "tension":  # the middle is the mean tension
        middle = f"{(scale.low + scale.high) / 2:.4g}"
        psim.SameLine(start + (width - psim.CalcTextSize(middle)[0]) / 2)
        psim.TextUnformatted(middle)
    psim.SameLine(start + width - psim.CalcTextSize(high)[0])
    psim.TextUnformatted(high)


def _imgui_swatch(colour: np.ndarray, label: str, *, outline: bool = False, edge: np.ndarray | None = None) -> None:
    """A swatch and its label: filled, or (`outline`) only its outline in the colour; `edge` draws a border."""
    draw = psim.GetWindowDrawList()
    x0, y0 = psim.GetCursorScreenPos()
    r, g, b = (float(c) for c in colour)
    if outline:
        draw.AddRect((x0 + 1.0, y0 + 1.0), (x0 + 15.0, y0 + 13.0), psim.GetColorU32((r, g, b, 1.0)), 0.0, 0, 2.0)
    else:
        draw.AddRectFilled((x0, y0), (x0 + 16.0, y0 + 14.0), psim.GetColorU32((r, g, b, 1.0)))
    if edge is not None:
        er, eg, eb = (float(c) for c in edge)
        draw.AddRect((x0, y0), (x0 + 16.0, y0 + 14.0), psim.GetColorU32((er, eg, eb, 1.0)))
    psim.Dummy((16.0, 14.0))
    psim.SameLine()
    psim.TextUnformatted(label)


def _view_scalar_quantities(
    mesh: "DcelData",
    ps_mesh: ps.SurfaceMesh,
    dict_sphere_fit_residual: dict,
    alpha: float,
    scattered: bool,
) -> None:
    """The curvature and derivative quantities `scalar_quantities=True` adds, in the order the viewers add them."""
    ps_mesh = view_dict_values_on_mesh(
        mesh,
        dict_sphere_fit_residual,
        alpha=alpha,
        ps_mesh=ps_mesh,
        scattered=scattered,
        clean_before=False,
        clean_after=False,
        show=False,
        name_values="Sphere fit residual",
    )
    for view in (
        _view_area_derivatives,
        _view_volume_derivatives,
        _view_mean_curvature_cotan,
        _view_gaussian_curvature,
        _view_discrepancy_of_principal_curvatures,
    ):
        view(mesh, alpha=alpha, ps_mesh=ps_mesh, clean_after=False, show=False, scattered=scattered)

def _view_pressures_on_mesh(
    mesh: "DcelData",
    dict_pressures: tuple[int, float],
    alpha: float = 0.05,
    ps_mesh: ps.SurfaceMesh | None = None,
    scattered: bool = False,
) -> ps.SurfaceMesh:
    """Get a polyscope SurfaceMesh with pressures data on cells.

    Args:
        mesh (DcelData): Mesh to show.
        dict_pressures (tuple[int, float]): Pressures already computed.
        alpha (float, optional): Quantile to ignore extreme values. Defaults to 0.05.
        ps_mesh (ps.SurfaceMesh | None, optional): SurfaceMesh to add the pressure data if it exists. Defaults to None.
        scattered (bool, optional): Scatters cells. Defaults to False.

    Returns:
        ps.SurfaceMesh: A Polyscope SurfaceMesh with a pressure data on cells.
    """
    v, f = (mesh.v_scattered, mesh.f_scattered) if scattered else (mesh.v, mesh.f)
    # Each triangle carries its higher-labelled material's pressure (scattered: its cell's).
    pressures = surface_face_pressures(mesh, dict_pressures, scattered=scattered)

    # An undetermined pressure (nan, from a Laplace method) takes no part in the scale and is
    # drawn `UNDETERMINED_COLOR`. With every pressure finite, the scale and the colours are exactly as before.
    finite = np.isfinite(pressures)
    if not finite.any():
        colors_face = np.tile(UNDETERMINED_COLOR, (len(pressures), 1))
    else:
        # The one pressure scale: the graph view's cell nodes and `save_colour_bars` use it too.
        maxp = surface_pressure_colour_upper(mesh, dict_pressures, alpha, scattered=scattered)

        print("Extremas of pressures plotted: ", 0, maxp)

        colors_face = pressure_colours(pressure_colour_fractions(pressures, maxp))
    # ps_mesh = view_faces_values_on_embryo(Mesh,pressures,ps_mesh = ps_mesh,name_values = "Pressures Cells",colormap = cm.magma,clean_before = clean_before, clean_after=clean_after,show=show,adapt_values=False,scattered = False)  # noqa: E501
    if ps_mesh is None:
        ps.init()
        ps_mesh = ps.register_surface_mesh("Embryo", v, f[:, [0, 1, 2]])
    ps_mesh.set_color((0.3, 0.6, 0.8))  # rgb triple on [0,1]
    ps_mesh.add_color_quantity("Pressures Cells", colors_face, defined_on="faces", enabled=True)
    hide_crop_faces(mesh, ps_mesh, scattered=scattered)
    return ps_mesh


# def view_pressures_on_mesh(Mesh,dict_pressures,alpha = 0.05,ps_mesh = None, clean_before = True, clean_after=True,show=True, scattered = False):  # noqa: E501

#     v,f = Mesh.v,Mesh.f

#     def find_pressure(face):
#         return(dict_pressures[max(face[3],face[4])])

#     pressures = np.array(list(map(find_pressure,f)))
#     pressures_values = np.array(list(dict_pressures.values()))
#     maxp = np.quantile(pressures_values,1-alpha)

#     print("Extremas of pressures plotted: ",0,maxp)

#     pressures = pressures.clip(0,maxp)
#     pressures/=np.amax(pressures)
#     pressures = 1-pressures
#     ps_mesh = view_faces_values_on_embryo(Mesh,pressures,ps_mesh = ps_mesh,name_values = "Pressures Cells",colormap = cm.magma,clean_before = clean_before, clean_after=clean_after,show=show,adapt_values=False,scattered = scattered)  # noqa: E501

#     return(ps_mesh)


# def view_faces_values_on_embryo(Mesh,Vf,name_values = "Values",ps_mesh = None,colormap = cm.jet,min_to_zero=True,clean_before = True, clean_after=True,show=True,highlight_junctions=False,adapt_values = True, scattered = False):  # noqa: E501

#     if scattered :
#         v,f,idx = Mesh.v_scattered,Mesh.f_scattered, Mesh.idx_scattered

#     else :
#         v,f,idx = Mesh.v,Mesh.f, np.arange(len(Mesh.f))

#     Values = Vf.copy()
#     if adapt_values :
#         if min_to_zero :
#             Values-=np.amin(Values)
#         Values/=(np.amax(Values)-np.amin(Values))

#     Values = Values[idx]
#     colors_face = colormap(Values)[:,:3]

#     ps.init()

#     if clean_before :
#         ps.remove_all_structures()


#     if ps_mesh == None :
#         ps_mesh = ps.register_surface_mesh("Embryo", v,f[:,[0,1,2]])

#     ps_mesh.set_color((0.3, 0.6, 0.8)) # rgb triple on [0,1]
#     #ps_mesh.set_transparency(0.2)
#     ps_mesh.add_color_quantity(name_values, colors_face, defined_on='faces',enabled=True)


#     if highlight_junctions :
#         plot_trijunctions(Mesh,clean_before = False, clean_after = False, show=False, color = "uniform",value_color = np.ones(3))  # noqa: E501

#     ps.set_ground_plane_mode("none")

#     if show :
#         ps.show()

#     if clean_after :
#         ps.remove_all_structures()

#     return(ps_mesh)


###
# Scalar quantities
###


def _view_area_derivatives(
    mesh: "DcelData",
    alpha: bool = 0.05,
    ps_mesh: ps.SurfaceMesh | None = None,
    remove_trijunctions: bool = True,
    clean_after: bool = True,
    show: bool = True,
    scattered: bool = False,
) -> ps.SurfaceMesh:
    """View or add area derivatives data on a Polyscope mesh.

    Args:
        mesh (DcelData): Mesh to analayze.
        alpha (bool, optional): Quantile to ignore extreme values. Defaults to 0.05.
        ps_mesh (ps.SurfaceMesh | None, optional): Add data to this mesh if it exists. Defaults to None.
        remove_trijunctions (bool, optional): Do not show ill-defined data on trijunctions. Defaults to True.
        clean_after (bool, optional): Clean polyscope after this view. Defaults to True.
        show (bool, optional): Show the polyscope viewer. Defaults to True.
        scattered (bool, optional): Scatter cells. Defaults to False.

    Returns:
        ps.SurfaceMesh: Mesh with the area derivatives added.
    """
    area_derivatives = mesh.compute_area_derivatives()
    derivatives_verts = np.zeros(mesh.v.shape)
    for key in area_derivatives:
        derivatives_verts += np.abs(area_derivatives[key])

    vmin, vmax = (
        np.quantile(derivatives_verts, alpha),
        np.quantile(derivatives_verts, 1 - alpha),
    )
    print("Extremas of area derivatives plotted: ", vmin, vmax)
    derivatives_verts = derivatives_verts.clip(vmin, vmax)

    ps_mesh = view_vertex_values_on_embryo(
        mesh,
        derivatives_verts,
        name_values="Area Derivatives",
        ps_mesh=ps_mesh,
        remove_trijunctions=remove_trijunctions,
        clean_after=clean_after,
        show=show,
        scattered=scattered,
    )

    return ps_mesh


def _view_volume_derivatives(
    mesh: "DcelData",
    alpha: bool = 0.05,
    ps_mesh: ps.SurfaceMesh | None = None,
    remove_trijunctions: bool = True,
    clean_after: bool = True,
    show: bool = True,
    scattered: bool = False,
) -> ps.SurfaceMesh:
    """View or add volume derivatives data on a Polyscope mesh.

    Args:
        mesh (DcelData): Mesh to analayze.
        alpha (bool, optional): Quantile to ignore extreme values. Defaults to 0.05.
        ps_mesh (ps.SurfaceMesh | None, optional): Add data to this mesh if it exists. Defaults to None.
        remove_trijunctions (bool, optional): Do not show ill-defined data on trijunctions. Defaults to True.
        clean_after (bool, optional): Clean polyscope after this view. Defaults to True.
        show (bool, optional): Show the polyscope viewer. Defaults to True.
        scattered (bool, optional): Scatter cells. Defaults to False.

    Returns:
        ps.SurfaceMesh: Mesh with the volume derivatives added.
    """
    volume_derivatives = mesh.compute_volume_derivatives()
    derivatives_verts = np.zeros(mesh.v.shape)
    for dv in volume_derivatives:
        derivatives_verts += np.abs(volume_derivatives[dv])

    vmin, vmax = (
        np.quantile(derivatives_verts, alpha),
        np.quantile(derivatives_verts, 1 - alpha),
    )
    print("Extremas of volume derivatives plotted: ", vmin, vmax)
    derivatives_verts = derivatives_verts.clip(vmin, vmax)

    ps_mesh = view_vertex_values_on_embryo(
        mesh,
        derivatives_verts,
        ps_mesh=ps_mesh,
        name_values="Volume Derivatives",
        remove_trijunctions=remove_trijunctions,
        clean_after=clean_after,
        show=show,
        scattered=scattered,
    )

    return ps_mesh


def _view_mean_curvature_cotan(
    mesh: "DcelData",
    alpha: bool = 0.05,
    ps_mesh: ps.SurfaceMesh | None = None,
    remove_trijunctions: bool = True,
    clean_after: bool = True,
    show: bool = True,
    scattered: bool = False,
) -> ps.SurfaceMesh:
    """View or add mean curvature data on a Polyscope mesh.

    Args:
        mesh (DcelData): Mesh to analayze.
        alpha (bool, optional): Quantile to ignore extreme values. Defaults to 0.05.
        ps_mesh (ps.SurfaceMesh | None, optional): Add data to this mesh if it exists. Defaults to None.
        remove_trijunctions (bool, optional): Do not show ill-defined data on trijunctions. Defaults to True.
        clean_after (bool, optional): Clean polyscope after this view. Defaults to True.
        show (bool, optional): Show the polyscope viewer. Defaults to True.
        scattered (bool, optional): Scatter cells. Defaults to False.

    Returns:
        ps.SurfaceMesh: Mesh with the mean curvature added.
    """
    mean_curvature, _, _ = compute_curvature_vertices_cotan(mesh)

    vmin, vmax = np.quantile(mean_curvature, alpha), np.quantile(mean_curvature, 1 - alpha)
    print("Extremas of mean curvature (cotan) plotted: ", vmin, vmax)
    mean_curvature = mean_curvature.clip(vmin, vmax)

    ps_mesh = view_vertex_values_on_embryo(
        mesh,
        mean_curvature,
        ps_mesh=ps_mesh,
        name_values="Mean Curvature Cotan",
        remove_trijunctions=remove_trijunctions,
        clean_after=clean_after,
        show=show,
        scattered=scattered,
    )

    return ps_mesh


def _view_gaussian_curvature(
    mesh: "DcelData",
    alpha: bool = 0.05,
    ps_mesh: ps.SurfaceMesh | None = None,
    remove_trijunctions: bool = True,
    clean_after: bool = True,
    show: bool = True,
    scattered: bool = False,
) -> ps.SurfaceMesh:
    """View or add Gaussian curvature data on a Polyscope mesh.

    Args:
        mesh (DcelData): Mesh to analayze.
        alpha (bool, optional): Quantile to ignore extreme values. Defaults to 0.05.
        ps_mesh (ps.SurfaceMesh | None, optional): Add data to this mesh if it exists. Defaults to None.
        remove_trijunctions (bool, optional): Do not show ill-defined data on trijunctions. Defaults to True.
        clean_after (bool, optional): Clean polyscope after this view. Defaults to True.
        show (bool, optional): Show the polyscope viewer. Defaults to True.
        scattered (bool, optional): Scatter cells. Defaults to False.

    Returns:
        ps.SurfaceMesh: Mesh with the gaussian curvature added.
    """
    gaussian_curvature = compute_gaussian_curvature_vertices(mesh)

    vmin, vmax = np.quantile(gaussian_curvature, alpha), np.quantile(gaussian_curvature, 1 - alpha)
    print("Extremas of gaussian curvature plotted: ", vmin, vmax)
    gaussian_curvature = gaussian_curvature.clip(vmin, vmax)

    ps_mesh = view_vertex_values_on_embryo(
        mesh,
        gaussian_curvature,
        ps_mesh=ps_mesh,
        name_values="Gaussian Curvature",
        remove_trijunctions=remove_trijunctions,
        clean_after=clean_after,
        show=show,
        scattered=scattered,
    )

    return ps_mesh


def _view_discrepancy_of_principal_curvatures(
    mesh: "DcelData",
    alpha: bool = 0.05,
    ps_mesh: ps.SurfaceMesh | None = None,
    remove_trijunctions: bool = True,
    clean_after: bool = True,
    show: bool = True,
    scattered: bool = False,
) -> ps.SurfaceMesh:
    """View or add discrepancy of principal curvatures on a Polyscope mesh.

    If H is the mean curvature and G the gaussian curvature, it shows (H²-G)/H².

    Args:
        mesh (DcelData): Mesh to analayze.
        alpha (bool, optional): Quantile to ignore extreme values. Defaults to 0.05.
        ps_mesh (ps.SurfaceMesh | None, optional): Add data to this mesh if it exists. Defaults to None.
        remove_trijunctions (bool, optional): Do not show ill-defined data on trijunctions. Defaults to True.
        clean_after (bool, optional): Clean polyscope after this view. Defaults to True.
        show (bool, optional): Show the polyscope viewer. Defaults to True.
        scattered (bool, optional): Scatter cells. Defaults to False.

    Returns:
        ps.SurfaceMesh: Mesh with the discrepancy of principal curvatures added.
    """
    mean_curvature, _, _ = compute_curvature_vertices_cotan(mesh)
    mean_curvature /= 9
    mean_curvature[mean_curvature == 0] = np.mean(mean_curvature)

    gaussian_curvature = compute_gaussian_curvature_vertices(mesh)
    kdiff = np.abs((mean_curvature**2 - gaussian_curvature) / (mean_curvature**2))

    vmin, vmax = np.quantile(kdiff, alpha), np.quantile(kdiff, 1 - alpha)
    print("Extremas of discrepancy of principal curvatures plotted: ", vmin, vmax)
    kdiff = kdiff.clip(vmin, vmax)

    ps_mesh = view_vertex_values_on_embryo(
        mesh,
        kdiff,
        ps_mesh=ps_mesh,
        name_values="Principal curvature discrepancy",
        remove_trijunctions=remove_trijunctions,
        clean_after=clean_after,
        show=show,
        scattered=scattered,
    )

    return ps_mesh


###
# Stress and graph
###


def _plot_stress_tensor(
    mesh: "DcelData",
    nx_graph: nx.Graph,
    dict_tensions: dict[tuple[int, int], float],
    dict_pressure: dict[int, float],
    clean_before: bool = True,
    clean_after: bool = True,
    show: bool = True,
    lumen_materials: int | list[int] = 0,
) -> None:
    """Show stress tensor on a polyscope viewer.

    Args:
        mesh (DcelData): Mesh to analyze.
        nx_graph (nx.Graph): Graph from mesh with data
        dict_tensions (dict[tuple[int, int], float]): Map interface to tension.
        dict_pressure (dict[int, float]): Map cell to pressure.
        clean_before (bool, optional): Clean viewer before this function. Defaults to True.
        clean_after (bool, optional): Clean viewer after this function. Defaults to True.
        show (bool, optional): Show viewer. Defaults to True.
        lumen_materials (int | list[int], optional): Materials considered exterior. Defaults to 0.
    """
    # Formula from G. K. BATCHELOR, J Fluid Mech, 1970 -- see foambryo._cell_stress.
    if isinstance(lumen_materials, int):
        lumen_materials = [lumen_materials]

    volumes = mesh.compute_volumes_cells()
    # A cell with an undetermined pressure (nan, from a Laplace method) has no stress: none is drawn.
    cells = [
        c
        for c in mesh.materials[1 : mesh.n_materials]
        if nx_graph.nodes.data("volume")[c] > 0 and np.isfinite(dict_pressure[c])
    ]
    mean_stresses = cell_mean_stress_tensors(
        mesh,
        dict_tensions,
        dict_pressure,
        volumes,
        cells,
        lumen_materials=lumen_materials,
    )

    # Tension-positive convention: negative principal values are compressive, non-negative ones
    # tensile. Column k of each array is the k-th principal axis scaled by its principal value.
    compression_vectors = np.zeros((mesh.n_materials - 1, 3, 3))
    tension_vectors = np.zeros((mesh.n_materials - 1, 3, 3))
    for i in range(1, mesh.n_materials):
        c = mesh.materials[i]
        if c not in mean_stresses:
            continue
        axes = split_principal_stresses(mean_stresses[c])
        compression_vectors[i - 1], tension_vectors[i - 1] = axes.compression, axes.tension

    ###
    # PLOTING
    ###

    ps.init()
    ps.set_ground_plane_mode("none")
    if clean_before:
        ps.remove_all_structures()

    # register a point cloud
    centroids = np.array([a[1] for a in nx_graph.nodes.data("centroid")])[
        np.array(nx_graph.nodes.data("volume"))[:, 1] > 0
    ]

    # ps_mesh = ps.register_surface_mesh("volume mesh", Mesh.v, Mesh.f[:,[0,1,2]])

    # ps_mesh.set_enabled() # default is true

    # ps_mesh.set_color((0.3, 0.6, 0.8)) # rgb triple on [0,1]
    # ps_mesh.set_transparency(0.2)

    ps_cloud = ps.register_point_cloud("Stress tensors principal directions", centroids, color=(0, 0, 0), radius=0.004)

    # Compression in red, tension in teal; each axis drawn both ways from the cell centroid.
    has_volume = np.array(nx_graph.nodes.data("volume"))[1:, 1] > 0
    radius = 0.005
    length = 0.1
    for name, vectors, color in (
        ("compression", compression_vectors, (0.7, 0.0, 0.0)),
        ("tension", tension_vectors, (0.2, 0.5, 0.5)),
    ):
        for k in range(3):
            axis = vectors[:, :, k][has_volume]
            for suffix, sign in (("", 1), ("_down", -1)):
                ps_cloud.add_vector_quantity(
                    f"{name}_axes_{k}{suffix}",
                    sign * axis,
                    enabled=True,
                    radius=radius,
                    length=length,
                    color=color,
                )

    if show:
        ps.show()
    if clean_after:
        ps.remove_all_structures()


def _display_embryo_graph_forces(
    nx_graph: nx.Graph,
    alpha: float = 0.05,
    clean_before: bool = True,
    clean_after: bool = True,
    show: bool = True,
    base_pressure: float = 0,
    plot_pressures: bool = True,
    pressure_upper: float | None = None,
) -> None:
    """Show polyscope viewer with embryo as a graph colored wrt to tensions and pressures.

    Args:
        nx_graph (nx.Graph): Graph from a mesh.
        alpha (float, optional): Quantile to ignore extreme values. Defaults to 0.05.
        clean_before (bool, optional): Clean viewer before this function. Defaults to True.
        clean_after (bool, optional): Clean viewer after this function. Defaults to True.
        show (bool, optional): Show viewer. Defaults to True.
        base_pressure (float, optional): Base value for exterior pressure. Defaults to 0.
        plot_pressures (bool, optional): Show pressures on cell nodess. Defaults to True.
        pressure_upper (float | None, optional): the top of the pressure scale. `plot_force_inference`
            passes the surface's (`surface_pressure_colour_upper`), so that a cell node has the colour of
            its cell on the surface and on the colour bar. Defaults to None: the `1 - alpha` quantile
            over the graph's cells, a different scale, kept only for a direct call without the mesh.
    """
    ps.init()
    ps.set_ground_plane_mode("none")

    if clean_before:
        ps.remove_all_structures()

    keys_edges = list(nx_graph.edges.keys())
    vertices = []
    edges = []
    values = []
    v_p0_list = []
    for i, key in enumerate(keys_edges):
        a, b = key

        v2 = nx_graph.nodes[b]["centroid"]
        if a == 0:
            v1 = np.mean(nx_graph.edges[key]["verts"], axis=0)  # + (np.mean(G.edges[key]['verts'],axis=0)-v2)*1.7
            v_p0_list.append(v1.copy())
        else:
            v1 = nx_graph.nodes[a]["centroid"]

        vertices.append(v1)
        vertices.append(v2)

        edges.append([2 * i, 2 * i + 1])
        # A crop interface carries no tension (it is absent from the map): drawn undetermined.
        values.append(nx_graph.edges[key].get("tension", np.nan))

    vertices = np.array(vertices)
    edges = np.array(edges)
    values = np.array(values, dtype=np.float64)
    v_p0_list = np.array(v_p0_list)

    # The library's tension colours, over the surface's range (`tension_colour_range` of the same tensions): an
    # undetermined tension (nan) takes no part in the range and is drawn `UNDETERMINED_COLOR`.
    finite_t = np.isfinite(values)
    if not finite_t.any():
        colors_tensions = np.tile(UNDETERMINED_COLOR, (len(values), 1))
    else:
        colors_tensions = tension_colours(tension_colour_fractions(values, tension_colour_range(values, alpha)))

    ps_net = ps.register_curve_network(
        "Graph Edges",
        vertices,
        edges,
        radius=0.007,
        color=np.array([92, 85, 141]) / 255,
    )  # ,color = "000000")
    ps_net.add_color_quantity("Surface tensions", colors_tensions, defined_on="edges", enabled=True)

    centroids_for_plotting = np.array([a[1] for a in nx_graph.nodes.data("centroid")])[
        np.array(nx_graph.nodes.data("volume"))[:, 1] > 0
    ]  # [1:]
    ps_cloud = ps.register_point_cloud("Graph Nodes", centroids_for_plotting, radius=0.02, color=[0, 0, 0])  # )

    if plot_pressures:
        pressure_for_plotting = np.array([a[1] for a in nx_graph.nodes.data("pressure")])[
            np.array(nx_graph.nodes.data("volume"))[:, 1] > 0
        ]  # [1:]
        values_p = pressure_for_plotting.copy()
        values_p -= base_pressure

        # An undetermined pressure (nan) takes no part in the scale and is drawn green.
        finite = np.isfinite(np.asarray(values_p, dtype=np.float64))
        if not finite.any():
            colors_pressure = np.tile(UNDETERMINED_COLOR, (len(values_p), 1))
        else:
            upper = pressure_upper
            if upper is None:
                upper = np.quantile(np.asarray(values_p, dtype=np.float64)[finite], 1 - alpha)
            colors_pressure = pressure_colours(pressure_colour_fractions(values_p, upper))
        ps_cloud.add_color_quantity("Pressure", colors_pressure, enabled=True)

    if show:
        ps.show()

    if clean_after:
        ps.remove_all_structures()


####
# Residuals
####


def plot_valid_junctions(
    mesh: "DcelData",
    dict_tensions: dict[tuple[int, int], float] | None = None,
) -> None:
    """Show in Polyscope's window which triple lines the tensions can balance: green if they can, red if not.

    At a triple line between cells `a`, `b` and `c`, three tensions can balance only if each is at most the sum of
    the other two (they form a triangle). A line where one tension exceeds the sum of the other two is drawn red
    ("bad trijunctions"), every other line green ("valid trijunctions"), over the mesh drawn white and mostly
    see-through (opacity 0.1). A line with a crop interface (absent from the tension map) is not judged and is
    drawn green. When every line is judged and bad, only the red lines are drawn. Used to check an inference. When
    no line is judged (the mesh has no triple line, or every line has a
    crop interface), no line is drawn: the window shows the mesh alone, with a `UserWarning` that says so.

    Needs Polyscope (`pip install "foambryo[viewing]"`) and a display. Polyscope's structures are removed first;
    returns when the window is closed, and removes them then.

    Args:
        mesh (DcelData): the mesh.
        dict_tensions (dict[tuple[int, int], float] | None, optional): tension per interface, keyed
            `(label_1, label_2)` with `label_1 < label_2`. Defaults to None: `infer_tensions(mesh, mean_tension=1)`,
            with that function's default method.

    Warns:
        UserWarning: when no triple line can be judged, and none is drawn.
    """
    if dict_tensions is None:
        dict_tensions = infer_tensions(mesh, mean_tension=1)
    dict_length = mesh.compute_length_trijunctions()
    dict_validity = {}
    for key in dict_length:
        a, b, c = key
        if any(pair not in dict_tensions for pair in ((a, b), (a, c), (b, c))):
            continue  # a junction with a crop interface (absent from the map) lies on the crop: not judged
        ga, gb, gc = dict_tensions[(a, b)], dict_tensions[(a, c)], dict_tensions[(b, c)]
        if gc > ga + gb or ga > gc + gb or gb > gc + ga:
            dict_validity[key] = 1.0
        else:
            dict_validity[key] = 0.0

    ps.set_ground_plane_mode("none")
    ps.remove_all_structures()
    ps.register_surface_mesh("mesh", mesh.v, mesh.f[:, :3], color=(1, 1, 1), transparency=0.1)
    if not dict_validity:
        # Nothing to colour: a mesh without triple lines, or one whose every line has an interface absent from the
        # tension map (a crop interface). Show the mesh alone rather than fail on an empty maximum.
        warnings.warn(
            "plot_valid_junctions: no triple line could be judged (the mesh has none, or every one has an interface "
            "without a tension, such as a crop interface); no line is drawn",
            UserWarning,
            stacklevel=2,
        )
        ps.show()
        ps.remove_all_structures()
        return
    if np.amax(list(dict_validity.values())) == 0:
        plot_trijunctions_topo_change_viewer(
            mesh,
            dict_trijunctional_values=dict_validity,
            color="uniform",
            value_color=(0, 1, 0),
            clean_before=False,
        )
    else:
        plot_trijunctions_topo_change_viewer(
            mesh,
            dict_trijunctional_values=dict_validity,
            color="values",
            clean_before=False,
        )


def plot_residual_junctions(
    mesh: "DcelData",
    dict_tensions: dict[tuple[int, int], float] | None = None,
    alpha: float = 0.05,
) -> None:
    """Show in Polyscope's window how far each triple line is from force balance under the tensions.

    The residuals are `foambryo.tension_inference.compute_residual_junctions_dict(mesh, dict_tensions, alpha)`
    (read with the chord angle reading): per triple line, the sum of the squared residuals of its three balance
    equations, each clipped between the `alpha` and `1 - alpha` quantiles over all lines. The lines are coloured
    with `jet` from 0 to the largest residual, over the mesh drawn white and mostly see-through (opacity 0.1); the
    range is printed.

    Needs Polyscope (`pip install "foambryo[viewing]"`) and a display. Polyscope's structures are removed first;
    returns when the window is closed, and removes them then.

    Args:
        mesh (DcelData): the mesh.
        dict_tensions (dict[tuple[int, int], float] | None, optional): tension per interface, keyed
            `(label_1, label_2)` with `label_1 < label_2`. Defaults to None: `infer_tensions(mesh, mean_tension=1)`,
            with that function's default method.
        alpha (float, optional): the quantile at which the residuals are clipped at each end. Defaults to 0.05.
    """
    if dict_tensions is None:
        dict_tensions = infer_tensions(mesh, mean_tension=1)
    dict_residuals = compute_residual_junctions_dict(mesh, dict_tensions, alpha=alpha)

    print(
        "Extremas of the residuals plotted : ",
        np.amin(list(dict_residuals.values())),
        np.amax(list(dict_residuals.values())),
    )

    ps.set_ground_plane_mode("none")
    ps.remove_all_structures()
    ps.register_surface_mesh("mesh", mesh.v, mesh.f[:, :3], color=(1, 1, 1), transparency=0.1)

    plot_trijunctions(mesh, dict_trijunctional_values=dict_residuals, clean_before=False, cmap=cm.jet)


# def plot_force_inference_with_lt(Mesh, force_inference_dicts=None, alpha=0.05, scalar_quantities=False):
#     if force_inference_dicts == None:
#         _, dict_tensions, dict_line_tensions, dict_pressure, _ = infer_forces_variational_lt(Mesh, mean_tension=1)
#     else:
#         dict_tensions, dict_line_tensions, dict_pressure = force_inference_dicts

#     G = Mesh.compute_networkx_graph()
#     nx.set_edge_attributes(G, dict_tensions, "tension")
#     nx.set_node_attributes(G, dict_pressure, "pressure")

#     ps_mesh = view_dict_values_on_mesh(
#         Mesh,
#         dict_tensions,
#         alpha=alpha,
#         ps_mesh=None,
#         clean_before=False,
#         clean_after=False,
#         show=False,
#         name_values="Surface Tensions",
#     )
#     ps_mesh = view_pressures_on_mesh(
#         Mesh,
#         dict_pressure,
#         ps_mesh=ps_mesh,
#         alpha=alpha,
#         clean_before=False,
#         clean_after=False,
#         show=False,
#     )
#     ps_mesh = plot_trijunctions(
#         Mesh,
#         Dict_trijunctional_values=dict_line_tensions,
#         clean_before=False,
#         clean_after=False,
#         show=False,
#         cmap=cm.jet,
#     )

#     plot_stress_tensor(Mesh, G, dict_tensions, dict_pressure, clean_before=False, clean_after=False, show=False)
#     display_embryo_graph_forces(
#         G,
#         alpha=alpha,
#         clean_before=False,
#         clean_after=False,
#         show=False,
#         P0=0,
#         Plot_pressures=True,
#     )
#     if scalar_quantities:
#         view_area_derivatives(Mesh, alpha=alpha, ps_mesh=ps_mesh, clean_before=False, clean_after=False, show=False)
#         view_volume_derivatives(Mesh, alpha=alpha, ps_mesh=ps_mesh, clean_before=False, clean_after=False, show=False)
#         view_mean_curvature_cotan(Mesh, alpha=alpha, ps_mesh=ps_mesh, clean_before=False, clean_after=False, show=False)  # noqa: E501
#         # view_mean_curvature_robust(
#         #     Mesh,
#         #     alpha=alpha,
#         #     ps_mesh=ps_mesh,
#         #     clean_before=False,
#         #     clean_after=False,
#         #     show=False,
#         # )
#         view_gaussian_curvature(Mesh, alpha=alpha, ps_mesh=ps_mesh, clean_before=False, clean_after=False, show=False)
#         view_discrepancy_of_principal_curvatures(
#             Mesh,
#             alpha=alpha,
#             ps_mesh=ps_mesh,
#             clean_before=False,
#             clean_after=False,
#             show=False,
#         )

#     ps.show()


# from foambryo.curvature import compute_curvature_vertices_robust_laplacian


# def view_mean_curvature_robust(
#     Mesh,
#     alpha=0.05,
#     ps_mesh=None,
#     remove_trijunctions=True,
#     clean_before=True,
#     clean_after=True,
#     show=True,
#     scattered=False,
# ):
#     H, _, _ = compute_curvature_vertices_robust_laplacian(Mesh)

#     vmin, vmax = np.quantile(H, alpha), np.quantile(H, 1 - alpha)
#     print("Extremas of mean curvature (robust) plotted: ", vmin, vmax)
#     H = H.clip(vmin, vmax)

#     ps_mesh = view_vertex_values_on_embryo(
#         Mesh,
#         H,
#         ps_mesh=ps_mesh,
#         name_values="Mean Curvature Robust",
#         remove_trijunctions=remove_trijunctions,
#         clean_before=clean_before,
#         clean_after=clean_after,
#         show=show,
#         scattered=scattered,
#     )

#     return ps_mesh
