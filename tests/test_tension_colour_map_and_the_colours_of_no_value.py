"""Tensions are coloured blue - light grey - red around the mean tension; no value is ever drawn in a map's colour.

What this pins, and why:

* the tension map is matplotlib's `coolwarm`, over a range **centred on the mean tension and symmetric** around it,
  wide enough to hold the 5 % and 95 % quantiles: a tension equal to the mean is the map's middle (light grey),
  so "higher than average" is red and "lower" is blue in every front end. The mean of `infer_tensions`' tensions is
  its `mean_tension` (1 by default), so the centre is the gauge;
* every tension colour comes from these functions: the Polyscope viewer's surface and graph, the bars of
  `save_colour_bars`, so no front end can drift from the bar it shows;
* an **undetermined** value is green, far from every colour of the tension and pressure maps and visible on a white
  (Polyscope) and a black (napari) background; **crop faces** are see-through, edged glass, and are hidden on the
  coloured surface, so that a crop face never shows a value's colour (nor the undetermined green);
* other per-interface quantities (a seed spread, a residual) keep `jet`: they are not tensions;
* the pressure map is `magma` reversed and **trimmed to its 10-90 % range**: the whole map ran from black (high
  pressure) to near-white (low), so one end vanished on napari's black canvas and the other on a white one; no
  pressure colour is now within 30 CIELAB units of black or white (the whole map came within 1.4 of black);
* the **backgrounds** the viewers offer (`BACKGROUND_COLOURS`): on the two greys, no colour of either map, nor the
  undetermined green, nor the crop faces' edges, comes closer than on white (the tension map's light-grey middle
  is what limits the light grey).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from matplotlib import cm

from foambryo import dcel_mesh_from_file, infer_forces, infer_tensions
from foambryo.colour_scales import (
    BACKGROUND_COLOURS,
    CROP_FACE_COLOR,
    CROP_FACE_EDGE_COLOR,
    CROP_FACE_OPACITY,
    PRESSURE_COLOUR_MAP,
    PRESSURE_COLOUR_MAP_RANGE,
    TENSION_COLOUR_MAP,
    UNDETERMINED_COLOR,
    colour_bar_ranges,
    pressure_colour_fractions,
    pressure_colours,
    quantity_colours,
    save_colour_bars,
    tension_colour_fractions,
    tension_colour_range,
    tension_colours,
)
from foambryo.dcel import DcelData

from .conftest import require_substitute_mesh


def _cielab(rgb: np.ndarray) -> np.ndarray:
    """CIELAB (D65) of sRGB colours in [0, 1]."""
    rgb = np.asarray(rgb, dtype=np.float64)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = linear @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def _contrast(rgb: np.ndarray, background: float) -> float:
    """WCAG contrast ratio of a colour against a grey background (0 black, 1 white)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    luminance = float(np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4) @ [0.2126, 0.7152, 0.0722])
    lighter, darker = max(luminance, background), min(luminance, background)
    return (lighter + 0.05) / (darker + 0.05)


@pytest.fixture(scope="module")
def case_000():
    mesh = dcel_mesh_from_file(require_substitute_mesh())
    return mesh, infer_forces(mesh)


def test_the_tension_map_is_coolwarm_and_the_mean_is_its_middle():
    assert TENSION_COLOUR_MAP == "coolwarm"
    fractions = np.linspace(0, 1, 11)
    assert np.array_equal(tension_colours(fractions), cm.coolwarm(fractions)[:, :3])
    middle = tension_colours(np.array([0.5]))[0]
    assert np.all(middle > 0.8)  # light grey
    assert np.ptp(middle) < 0.05
    assert np.array_equal(quantity_colours(fractions), cm.jet(fractions)[:, :3])


def test_the_range_is_centred_on_the_mean_tension_and_holds_both_quantiles(case_000):
    _, (tensions, _) = case_000
    values = np.array(list(tensions.values()))
    low, high = tension_colour_range(values, 0.05)
    assert (low + high) / 2 == pytest.approx(1.0, abs=1e-12)  # the default gauge: the mean tension is 1
    assert low <= np.quantile(values, 0.05)
    assert np.quantile(values, 0.95) <= high
    assert min(np.quantile(values, 0.05) - low, high - np.quantile(values, 0.95)) == pytest.approx(0.0, abs=1e-12)
    assert tension_colour_fractions(np.array([1.0]), (low, high))[0] == pytest.approx(0.5)
    assert colour_bar_ranges(case_000[0], tensions).tension == (low, high)


def test_the_centre_follows_mean_tension(case_000):
    mesh, _ = case_000
    tensions = infer_tensions(mesh, mean_tension=2.5)
    low, high = tension_colour_range(np.array(list(tensions.values())))
    assert (low + high) / 2 == pytest.approx(2.5, rel=1e-12)
    assert tension_colour_range(np.array(list(tensions.values())), centre=3.0)[0] + tension_colour_range(
        np.array(list(tensions.values())), centre=3.0)[1] == pytest.approx(6.0)


def test_undetermined_and_equal_tensions():
    values = np.array([1.0, np.nan, 1.0])
    colour_range = tension_colour_range(values)
    assert colour_range == (1.0, 1.0)
    fractions = tension_colour_fractions(values, colour_range)
    assert fractions[0] == 0.5
    assert np.isnan(fractions[1])
    colours = tension_colours(fractions)
    assert np.array_equal(colours[1], UNDETERMINED_COLOR)
    with pytest.raises(ValueError, match="no tension is finite"):
        tension_colour_range(np.array([np.nan]))


def test_the_undetermined_green_is_outside_every_map_on_both_backgrounds():
    samples = np.linspace(0, 1, 512)
    green = _cielab(UNDETERMINED_COLOR)
    for name, colours in (("coolwarm", cm.coolwarm(samples)[:, :3]), ("pressures", pressure_colours(samples))):
        distances = np.linalg.norm(_cielab(colours) - green, axis=1)
        assert distances.min() > 40, name  # a CIELAB distance of ~2 is just noticeable
    for colour in (UNDETERMINED_COLOR, CROP_FACE_EDGE_COLOR):
        assert min(_contrast(colour, 1.0), _contrast(colour, 0.0)) > 3
    assert 0 < CROP_FACE_OPACITY <= 0.25
    assert np.linalg.norm(_cielab(CROP_FACE_EDGE_COLOR) - green) > 40


def _cropped_case_000(mesh):
    """Case 000 with the triangles of one interface with the medium marked as crop faces (a colour fixture)."""
    labels = np.asarray(mesh.f[:, 3:5])
    crop = (labels[:, 0] == 0) & (labels[:, 1] == labels[labels[:, 0] == 0, 1].min())
    return DcelData(np.asarray(mesh.v), np.asarray(mesh.f[:, :3]), labels, crop_faces=crop), crop


def test_the_bars_have_a_middle_tick_and_a_legend_for_what_has_no_value(case_000, tmp_path):
    mesh, (tensions, pressures) = case_000
    cropped, _ = _cropped_case_000(mesh)
    ranges = save_colour_bars(cropped, tensions, pressures, tmp_path / "bars.svg")
    svg = (tmp_path / "bars.svg").read_text()
    assert "centred on the mean" in svg
    assert "crop face" in svg
    assert f"{(ranges.tension[0] + ranges.tension[1]) / 2:.4g}" in svg
    with_nan = {**tensions, next(iter(tensions)): float("nan")}
    save_colour_bars(mesh, with_nan, None, tmp_path / "nan.svg")
    assert "undetermined" in (tmp_path / "nan.svg").read_text()


@pytest.fixture
def headless_polyscope(monkeypatch):
    """Polyscope's mock backend, recording the colour, scalar and transparency calls of every surface."""
    ps = pytest.importorskip("polyscope")
    real_init, real_show = ps.init, ps.show
    try:
        real_init("openGL_mock")
    except RuntimeError as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"Polyscope's mock backend is unavailable: {error}")
    monkeypatch.setattr(ps, "init", lambda *_args, **_kwargs: real_init("openGL_mock"))
    monkeypatch.setattr(ps, "show", lambda *_args, **_kwargs: real_show(forFrames=1))
    recorded: dict[str, object] = {}
    for method in ("add_color_quantity", "add_scalar_quantity", "set_transparency_quantity", "set_transparency",
                   "set_edge_width", "set_edge_color", "set_color"):
        original = getattr(ps.SurfaceMesh, method)

        def recording(self, *args: object, _method=method, _original=original, **kwargs: object):
            recorded[(self.get_name(), _method, args[0] if _method.startswith("add") else None)] = (
                args[1] if _method.startswith("add") else args[0])
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(ps.SurfaceMesh, method, recording)
    yield recorded
    ps.remove_all_structures()


def test_crop_faces_are_glass_and_hidden_on_the_coloured_surface(headless_polyscope, case_000):
    import foambryo.viewing as viewing
    from foambryo.plotting_utilities import CROP_FACES_HIDDEN

    mesh, _ = case_000
    cropped, crop = _cropped_case_000(mesh)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the cut cell's pressure is withheld, with a warning
        kept, pressures = infer_forces(cropped)
    viewing.plot_force_inference(cropped, kept, pressures)
    hidden = np.asarray(headless_polyscope[("Embryo", "add_scalar_quantity", CROP_FACES_HIDDEN)])
    assert np.array_equal(hidden, np.where(crop, 0.0, 1.0))
    # The window shows the tensions first: their fill also hides the undetermined interfaces, and the crop faces.
    from foambryo.viewing import _TENSION_FILL

    assert headless_polyscope[("Embryo", "set_transparency_quantity", None)] == _TENSION_FILL
    tension_fill = np.asarray(headless_polyscope[("Embryo", "add_scalar_quantity", _TENSION_FILL)])
    assert (tension_fill[crop] == 0).all()
    assert headless_polyscope[("Crop faces", "set_transparency", None)] == CROP_FACE_OPACITY
    assert headless_polyscope[("Crop faces", "set_edge_width", None)] > 0
    assert np.allclose(headless_polyscope[("Crop faces", "set_color", None)], CROP_FACE_COLOR)
    assert np.allclose(headless_polyscope[("Crop faces", "set_edge_color", None)], CROP_FACE_EDGE_COLOR)
    face_tensions = np.array([kept.get((int(f[3]), int(f[4])), np.nan) for f in cropped.f])
    expected = tension_colours(tension_colour_fractions(face_tensions, tension_colour_range(
        np.array(list(kept.values())))))
    assert np.array_equal(headless_polyscope[("Embryo", "add_color_quantity", "Surface Tensions")], expected)


def test_the_pressure_map_is_magma_reversed_and_trimmed_so_no_pressure_vanishes_on_black_or_white():
    assert PRESSURE_COLOUR_MAP == "magma"
    assert PRESSURE_COLOUR_MAP_RANGE == (0.10, 0.90)
    fractions = np.linspace(0, 1, 512)
    colours = pressure_colours(fractions)
    assert np.array_equal(colours, cm.magma(0.1 + 0.8 * fractions)[:, :3])
    # the top of the range (fraction 0) is the dark end, a pressure of 0 (fraction 1) the light end
    assert np.array_equal(colours[[0, -1]], cm.magma(np.array([0.1, 0.9]))[:, :3])
    for name, background in (("black", 0.0), ("white", 1.0)):
        distances = np.linalg.norm(_cielab(colours) - _cielab(np.full(3, background)), axis=1)
        assert distances.min() > 30, name
    whole = cm.magma(fractions)[:, :3]
    assert np.linalg.norm(_cielab(whole) - _cielab(np.zeros(3)), axis=1).min() < 2  # what the trim is for
    # the fractions, and so the scale and every value, are unchanged: only their colours are
    pressures = np.array([-1.0, 0.0, 0.5, 1.0, 3.0, np.nan])
    assert np.array_equal(pressure_colour_fractions(pressures, 1.0), [1.0, 1.0, 0.5, 0.0, 0.0, np.nan],
                          equal_nan=True)
    assert np.array_equal(pressure_colours(pressure_colour_fractions(pressures, 1.0))[-1], UNDETERMINED_COLOR)


def test_the_backgrounds_are_named_and_no_colour_vanishes_on_the_greys():
    assert list(BACKGROUND_COLOURS) == ["black", "dark grey", "light grey", "white"]
    samples = np.linspace(0, 1, 512)
    colours = {
        "tensions": cm.coolwarm(samples)[:, :3], "pressures": pressure_colours(samples),
        "undetermined": UNDETERMINED_COLOR[np.newaxis], "crop face edges": CROP_FACE_EDGE_COLOR[np.newaxis],
    }

    def nearest(name: str, background: str) -> float:
        return float(np.linalg.norm(_cielab(colours[name]) - _cielab(np.array(BACKGROUND_COLOURS[background])),
                                    axis=1).min())

    for name in colours:
        assert nearest(name, "dark grey") > 30, name
    assert nearest("tensions", "light grey") >= nearest("tensions", "white")  # the mean tension's light grey
    for name in ("pressures", "undetermined"):
        assert nearest(name, "light grey") > 30, name
