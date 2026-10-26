"""The viewer's window shows the colour bars of what it draws, a length scale bar, and the library's options.

What this pins, and why:

* **The bars in the window are the library's colours over the library's ranges.** Polyscope is handed colours,
  not values (its scalar maps draw `nan` in their top colour), so the window draws the bars itself, with ImGui.
  The colours handed to ImGui are the library's bar colours (the samples `save_colour_bars` draws) and the
  ranges are `colour_bar_ranges`' (for the scattered view, the scattered surface's own pressure range). The
  graph view's edges are coloured over the same tension range.
* **Back faces are drawn like front faces** on the coloured surface: the outside of a tissue faces away from
  Polyscope's camera, and its default policy drew every colour ~15 % darker than its bar.
* **The library's options are the viewers' own**: their defaults are `infer_forces`' and they are passed to the
  library unchanged; the command offers `infer`'s options when it infers and refuses them with `--results`.
* **A seed ensemble** is drawn on the viewed mesh's interfaces matched by label pair, its spread with a bar of its
  own, the interfaces found by fewer than half of the seeds marked; the command reads `infer-over-seeds`' tables.
* **A saved figure** shows the view with its bars and the scale bar's length beside the segment, whose ends are
  projected with Polyscope's camera convention; the window never overwrites a file.

Polyscope runs on its mock backend (no display); tests that need it are skipped without Polyscope.
"""

from __future__ import annotations

import inspect
import math
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from foambryo import (
    InferenceOutcome,
    PressureComputationMethod,
    TensionComputationMethod,
    cli,
    dcel_mesh_from_file,
    dcel_mesh_from_multimaterial_mesh,
    infer_forces,
    infer_pressures,
    infer_tensions,
)
from foambryo import _result_tables as tables
from foambryo._colour_scales import (
    _pressure_bar_colours,
    _tension_bar_colours,
    clipped_colour_fractions,
    colour_bar_ranges,
    surface_pressure_colour_upper,
    tension_colours,
)
from foambryo._viewer_figures import (
    ColourScale,
    ViewerColourScales,
    compose_figure,
    ensemble_on_mesh,
    figure_of_the_view,
    format_length,
    interfaces_on_mesh,
    next_free_path,
    project_to_pixels,
    round_scale_bar_length,
    scale_bar_for_points,
    viewer_colour_scales,
)

from .conftest import require_substitute_mesh

CUT_CASE = Path(__file__).parent / "data" / "cut_benchmark_case_000_mesh.npz"


def _benchmark_case():
    mesh = dcel_mesh_from_file(require_substitute_mesh())
    tensions, pressures = infer_forces(mesh)
    return mesh, tensions, pressures


def _cut_case_with_laplace_pressures():
    arrays = np.load(CUT_CASE)
    mesh = dcel_mesh_from_multimaterial_mesh(arrays["points"], arrays["triangles"], arrays["labels"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tensions = infer_tensions(mesh)
        pressures = infer_pressures(mesh, tensions, mode=PressureComputationMethod.Laplace)
    return mesh, tensions, pressures


CASES = {"benchmark case 000": _benchmark_case, "cut case 000, Laplace pressures": _cut_case_with_laplace_pressures}


@pytest.fixture
def mock_polyscope(monkeypatch):
    """Polyscope's mock backend; `show` runs one frame (so the panel, the user callback, runs once)."""
    ps = pytest.importorskip("polyscope")
    real_init, real_show = ps.init, ps.show
    try:
        real_init("openGL_mock")
    except RuntimeError as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"Polyscope's mock backend is unavailable: {error}")
    monkeypatch.setattr(ps, "init", lambda *_args, **_kwargs: real_init("openGL_mock"))
    monkeypatch.setattr(ps, "show", lambda *_args, **_kwargs: real_show(forFrames=1))
    yield ps
    ps.remove_all_structures()


# ---------------------------------------------------------------------------------------------
# the colour scales and the bars in the window
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("case", list(CASES))
def test_the_window_scales_are_the_colour_bar_ranges(case):
    mesh, tensions, pressures = CASES[case]()
    scales = viewer_colour_scales(mesh, tensions, pressures, 0.05)
    ranges = colour_bar_ranges(mesh, tensions, pressures, 0.05)
    assert [s.title for s in scales.scales] == ["Surface tension", "Pressure"]
    assert (scales.scales[0].low, scales.scales[0].high) == ranges.tension
    assert (scales.scales[1].low, scales.scales[1].high) == ranges.pressure
    assert scales.undetermined == any(math.isnan(p) for p in pressures.values())
    assert np.array_equal(scales.scales[0].colours(), _tension_bar_colours(ranges.tension))
    assert np.array_equal(scales.scales[1].colours(), _pressure_bar_colours(ranges.pressure[1]))


def test_the_scattered_window_takes_the_scattered_pressure_range():
    mesh, tensions, pressures = _benchmark_case()
    mesh.compute_scattered_arrays(coeff=0.2)
    scales = viewer_colour_scales(mesh, tensions, pressures, 0.05, scattered=True)
    assert scales.scales[1].high == float(surface_pressure_colour_upper(mesh, pressures, 0.05, scattered=True))
    assert "scattered" in scales.scales[1].clipping


class _DrawListRecorder:
    def __init__(self) -> None:
        self.rectangles: list[tuple] = []
        self.outlines: list[tuple] = []

    def AddRectFilled(self, p_min, p_max, col, *_args: object):  # noqa: N802 - ImGui's name
        self.rectangles.append((p_min, p_max, col))

    def AddRect(self, p_min, p_max, col, *_args: object):  # noqa: N802 - ImGui's name
        self.outlines.append((p_min, p_max, col))

    def AddText(self, *_args: object) -> None:  # noqa: N802 - ImGui's name
        pass


@pytest.mark.usefixtures("mock_polyscope")
@pytest.mark.parametrize("case", list(CASES))
def test_the_panel_draws_the_library_bar_colours(monkeypatch, case):
    """The rectangles ImGui is handed are the library's bar samples, in order; the undetermined swatch an outline."""
    import foambryo.viewing as viewing

    recorder = _DrawListRecorder()
    monkeypatch.setattr(viewing.psim, "GetWindowDrawList", lambda: recorder)
    monkeypatch.setattr(viewing.psim, "GetColorU32", lambda colour, *_args: tuple(colour))
    mesh, tensions, pressures = CASES[case]()
    viewing.plot_force_inference(mesh, tensions, pressures)
    ranges = colour_bar_ranges(mesh, tensions, pressures)
    expected = [*_tension_bar_colours(ranges.tension), *_pressure_bar_colours(ranges.pressure[1])]
    drawn = [rectangle[2][:3] for rectangle in recorder.rectangles]
    assert np.array_equal(np.array(drawn[: len(expected)]), np.array(expected))
    undetermined = any(math.isnan(p) for p in pressures.values())
    assert len(drawn) == len(expected)
    outlines = [rectangle[2][:3] for rectangle in recorder.outlines]
    assert outlines == ([tuple(float(c) for c in viewing.UNDETERMINED_COLOR)] if undetermined else [])


@pytest.mark.parametrize("case", list(CASES))
def test_the_graph_edges_are_coloured_over_the_bars_tension_range(mock_polyscope, monkeypatch, case):
    import foambryo.viewing as viewing

    recorded: dict[str, np.ndarray] = {}
    original = mock_polyscope.CurveNetwork.add_color_quantity

    def recording(self, name, values, *args: object, **kwargs: object):
        recorded[name] = np.array(values, dtype=np.float64)
        return original(self, name, values, *args, **kwargs)

    monkeypatch.setattr(mock_polyscope.CurveNetwork, "add_color_quantity", recording)
    mesh, tensions, pressures = CASES[case]()
    viewing.plot_force_inference(mesh, tensions, pressures)
    graph = mesh.compute_networkx_graph()
    edge_tensions = np.array([tensions.get(key, np.nan) for key in graph.edges], dtype=np.float64)
    expected = tension_colours(clipped_colour_fractions(edge_tensions, colour_bar_ranges(mesh, tensions).tension))
    assert np.array_equal(recorded["Surface tensions"], expected)


@pytest.mark.parametrize("call", ["plot_force_inference", "plot_tension_inference", "plot_pressure_inference"])
def test_the_coloured_surface_draws_back_faces_like_front_faces(mock_polyscope, call):
    import foambryo.viewing as viewing

    mesh = dcel_mesh_from_file(require_substitute_mesh())
    getattr(viewing, call)(mesh)
    assert mock_polyscope.get_surface_mesh("Embryo").get_back_face_policy() == "identical"


# ---------------------------------------------------------------------------------------------
# the library's options
# ---------------------------------------------------------------------------------------------
def test_the_viewers_inference_defaults_are_infer_forces():
    pytest.importorskip("polyscope")
    import foambryo.viewing as viewing

    library = inspect.signature(infer_forces).parameters
    for function in (viewing.plot_force_inference, viewing.plot_pressure_inference, viewing.plot_tension_inference):
        parameters = inspect.signature(function).parameters
        for name in ("mean_tension", "base_pressure", "mode_tension", "mode_pressure", "solver"):
            if name in parameters:
                assert parameters[name].default == library[name].default, (function.__name__, name)
                assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.usefixtures("mock_polyscope")
def test_the_viewers_pass_the_options_to_the_library_unchanged(monkeypatch):
    import foambryo.viewing as viewing

    """The window infers with `foambryo infer`'s call, `infer_tensions_then_pressures` (bitwise `infer_forces`)."""
    calls: list[tuple] = []
    real = viewing.infer_tensions_then_pressures

    def recorded(*args: object, **kwargs: object):
        calls.append((args[1:], kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(viewing, "infer_tensions_then_pressures", recorded)
    mesh = dcel_mesh_from_file(require_substitute_mesh())
    options = {"mean_tension": 2.0, "base_pressure": 0.5, "mode_tension": TensionComputationMethod.Lami,
               "mode_pressure": PressureComputationMethod.Laplace, "solver": "dense"}
    viewing.plot_force_inference(mesh, **options)
    assert calls[-1] == ((2.0, 0.5, TensionComputationMethod.Lami, PressureComputationMethod.Laplace),
                         {"solver": "dense", "diagnostics": False, "with_pressures": True})
    viewing.plot_tension_inference(mesh, mean_tension=2.0, mode_tension=TensionComputationMethod.Lami, solver="dense")
    assert calls[-1][0][0] == 2.0
    assert calls[-1][0][2] == TensionComputationMethod.Lami
    assert calls[-1][1] == {"solver": "dense", "diagnostics": False, "with_pressures": False}


@pytest.mark.usefixtures("mock_polyscope")
def test_the_panel_names_the_call_that_produced_the_colours(monkeypatch):
    import foambryo.viewing as viewing

    windows = []
    real_open = viewing._ForceWindow.open
    monkeypatch.setattr(viewing._ForceWindow, "open", lambda self, **kw: (windows.append(self), real_open(self, **kw)))
    mesh = dcel_mesh_from_file(require_substitute_mesh())
    viewing.plot_force_inference(mesh, mode_tension=TensionComputationMethod.Lami)
    assert windows[-1].values_from.startswith("inferred here: infer_forces(mesh, ")
    assert "TensionComputationMethod.Lami" in windows[-1].values_from
    tensions, pressures = infer_forces(mesh)
    viewing.plot_force_inference(mesh, tensions, pressures, values_from="read from results/")
    assert windows[-1].values_from == "read from results/"


@pytest.mark.usefixtures("mock_polyscope")
def test_recompute_infers_again_with_the_methods_chosen_in_the_panel(monkeypatch):
    import foambryo.viewing as viewing

    windows = []
    real_open = viewing._ForceWindow.open
    monkeypatch.setattr(viewing._ForceWindow, "open", lambda self, **kw: (windows.append(self), real_open(self, **kw)))
    mesh = dcel_mesh_from_file(require_substitute_mesh())
    viewing.plot_force_inference(mesh)
    window = windows[-1]
    window.tension_choice = list(TensionComputationMethod).index(TensionComputationMethod.Lami)
    window.pressure_choice = list(PressureComputationMethod).index(PressureComputationMethod.Laplace)
    window.recompute()
    expected_t, expected_p = infer_forces(mesh, mode_tension=TensionComputationMethod.Lami,
                                          mode_pressure=PressureComputationMethod.Laplace)
    assert window.tensions == expected_t
    assert window.pressures == expected_p
    assert window.scales.scales[0].high == colour_bar_ranges(mesh, expected_t).tension[1]


# ---------------------------------------------------------------------------------------------
# the length scale bar, and where it falls on the image
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize(("extent", "length"), [(100.0, 20.0), (37.0, 5.0), (1.3, 0.2), (4.0, 1.0), (8.0, 2.0)])
def test_the_scale_bar_length_is_round_and_at_most_a_quarter(extent, length):
    assert round_scale_bar_length(extent) == length
    assert extent / 8 <= length <= extent / 4 * (1 + 1e-12)


@pytest.mark.parametrize("extent", [0.0, -1.0, math.inf, math.nan])
def test_no_scale_bar_length_for_a_degenerate_extent(extent):
    with pytest.raises(ValueError, match="finite and positive"):
        round_scale_bar_length(extent)


def test_the_scale_bar_lies_under_the_tissue_with_its_length_in_its_unit():
    points = np.array([[0.0, 0.0, 0.0], [80.0, 40.0, 20.0]])
    bar = scale_bar_for_points(points, "um")
    assert bar.length == 20.0
    assert bar.label == "20 µm" == format_length(20.0, "um")
    assert np.allclose(bar.end - bar.start, [0.0, 0.0, 20.0])  # along x, the mesh's column 2: horizontal when drawn
    assert bar.start[1] < 0.0  # below the tissue along y, drawn up
    assert scale_bar_for_points(points, "voxel", length=7.5).label == "7.5 voxel"
    with pytest.raises(ValueError, match="finite and positive"):
        scale_bar_for_points(points, "um", length=0.0)


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    view = np.eye(4)
    view[0, :3], view[1, :3], view[2, :3] = right, true_up, -forward
    view[:3, 3] = -view[:3, :3] @ eye
    return view


def test_points_project_where_a_perspective_camera_sees_them():
    view = _look_at(np.array([0.0, 0.0, 10.0]), np.zeros(3), np.array([0.0, 1.0, 0.0]))
    width, height, fov = 800, 600, 45.0
    pixels = project_to_pixels(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 20.0]]),
                               view, fov, width, height)
    focal = 1 / math.tan(math.radians(fov) / 2)
    assert np.allclose(pixels[0], [400.0, 300.0])
    assert np.isclose(pixels[1, 0] - 400.0, focal * 0.1 * height / 2)  # x scales with the height: square pixels
    assert np.isclose(300.0 - pixels[2, 1], focal * 0.1 * height / 2)  # +y is up, image rows go down
    assert np.isnan(pixels[3]).all()  # behind the camera


# ---------------------------------------------------------------------------------------------
# saved figures and colour bars
# ---------------------------------------------------------------------------------------------
def test_a_figure_writes_the_scale_bar_length_beside_the_segment(tmp_path):
    scales = viewer_colour_scales(*_benchmark_case(), 0.05)
    image = np.full((120, 200, 3), 255, dtype=np.uint8)
    bar = scale_bar_for_points(np.array([[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]]), "um")
    ends = np.array([[40.0, 100.0], [80.0, 100.0]])
    inside = figure_of_the_view(image, scales, scale_bar=bar, scale_bar_pixels=ends, caption="Values: test")
    label = next(text for text in inside.axes[0].texts if text.get_text() == "20 µm")
    assert label.get_position() == pytest.approx((60.0, 100.0 + 0.02 * 120))  # under the segment's middle
    assert any(text.get_text() == "Values: test" for axes in inside.axes for text in axes.texts)
    ends[0, 0] = -40.0  # one end off the image
    outside = figure_of_the_view(image, scales, scale_bar=bar, scale_bar_pixels=ends)
    assert [text.get_text() for text in outside.axes[0].texts] == ["scale bar: 20 µm"]
    for suffix in (".png", ".pdf"):
        assert compose_figure(image, scales, tmp_path / f"f{suffix}", scale_bar=bar).stat().st_size > 0
    bars_only = compose_figure(None, scales, tmp_path / "bars.png")
    assert bars_only.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_a_figure_draws_the_undetermined_swatch_when_a_value_is_undetermined(tmp_path):
    import matplotlib

    scales = ViewerColourScales((ColourScale("Surface tension", 0.5, 1.5, "tension", "clipped"),), True, 0.05)
    with matplotlib.rc_context({"svg.fonttype": "none"}):
        path = compose_figure(None, scales, tmp_path / "bars.svg")
    assert ">undetermined (nan): green outline (a cell: green)<" in path.read_text(encoding="utf-8")


def test_the_window_never_overwrites_a_file(tmp_path):
    first = tmp_path / "figure.png"
    assert next_free_path(first) == first
    first.write_bytes(b"x")
    assert next_free_path(first) == tmp_path / "figure_2.png"
    (tmp_path / "figure_2.png").write_bytes(b"x")
    assert next_free_path(first) == tmp_path / "figure_3.png"


def test_save_figure_from_python_writes_the_view_with_its_bars(mock_polyscope, monkeypatch, tmp_path):
    import foambryo._viewer_figures as figures
    import foambryo.viewing as viewing

    composed = {}
    real_compose = figures.compose_figure

    def recording(image, scales, path, **kwargs: object):
        composed.update(image=image, scales=scales, kwargs=kwargs)
        return real_compose(image, scales, path, **kwargs)

    monkeypatch.setattr(viewing, "compose_figure", recording)
    shown = []
    monkeypatch.setattr(mock_polyscope, "show", lambda *_args, **kwargs: shown.append(kwargs.get("forFrames")))
    mesh, tensions, pressures = _benchmark_case()
    viewing.plot_force_inference(mesh, tensions, pressures, length_unit="um", save_figure=tmp_path / "f.png",
                                 show=False)
    assert (tmp_path / "f.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert shown == [1]  # one frame rendered for the figure; no window left open
    assert composed["scales"] == viewer_colour_scales(mesh, tensions, pressures, 0.05)
    assert composed["kwargs"]["scale_bar"].unit == "um"
    assert composed["image"].ndim == 3


@pytest.mark.parametrize("scattered", [False, True], ids=["surface", "scattered"])
@pytest.mark.usefixtures("mock_polyscope")
def test_the_panels_colour_bars_are_save_colour_bars_when_they_are_its_bars(monkeypatch, tmp_path, scattered):
    import foambryo.viewing as viewing

    windows = []
    real_open = viewing._ForceWindow.open
    monkeypatch.setattr(viewing._ForceWindow, "open", lambda self, **kw: (windows.append(self), real_open(self, **kw)))
    called = []
    monkeypatch.setattr(viewing, "save_colour_bars", lambda *args, **_kwargs: called.append(args))
    mesh, tensions, pressures = _benchmark_case()
    viewing.plot_force_inference(mesh, tensions, pressures, scattered=scattered)
    path = windows[-1].save_colour_bars(tmp_path / "bars.png")
    assert path == tmp_path / "bars.png"
    assert bool(called) is not scattered
    if scattered:
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------------------------
# seed ensembles
# ---------------------------------------------------------------------------------------------
def _ensemble_for(mesh, flagged=(), extra=((7, 8),)):
    interfaces = sorted(interfaces_on_mesh(mesh))
    keys = [*interfaces[1:], *extra]  # one interface of the mesh missing, others not on the mesh
    return SimpleNamespace(
        tensions={key: 1.0 + 0.01 * i for i, key in enumerate(keys)},
        tension_seed_sd={key: (math.nan if i == 0 else 0.02 + 0.001 * i) for i, key in enumerate(keys)},
        found_by_fewer_than_half_the_seeds={key: key in flagged for key in keys},
    )


def test_an_ensemble_is_matched_to_the_mesh_by_label_pair():
    mesh = dcel_mesh_from_file(require_substitute_mesh())
    interfaces = sorted(interfaces_on_mesh(mesh))
    ensemble = _ensemble_for(mesh, flagged=(interfaces[2], (7, 8)))
    matched = ensemble_on_mesh(mesh, ensemble.tensions, ensemble.tension_seed_sd,
                               ensemble.found_by_fewer_than_half_the_seeds)
    assert set(matched.tensions) == set(interfaces[1:])
    assert matched.not_on_the_mesh == 1
    assert matched.missing_from_the_ensemble == 1
    assert matched.found_by_fewer_than_half_the_seeds == (interfaces[2],)
    assert all(matched.tensions[k] == ensemble.tensions[k] for k in matched.tensions)


def test_the_ensemble_view_draws_its_tensions_spread_and_flagged_interfaces(mock_polyscope, monkeypatch):
    import foambryo.viewing as viewing

    recorded: dict[str, np.ndarray] = {}
    original = mock_polyscope.SurfaceMesh.add_color_quantity

    def recording(self, name, values, *args: object, **kwargs: object):
        recorded[name] = np.array(values, dtype=np.float64)
        return original(self, name, values, *args, **kwargs)

    monkeypatch.setattr(mock_polyscope.SurfaceMesh, "add_color_quantity", recording)
    windows = []
    real_open = viewing._ForceWindow.open
    monkeypatch.setattr(viewing._ForceWindow, "open", lambda self, **kw: (windows.append(self), real_open(self, **kw)))
    mesh = dcel_mesh_from_file(require_substitute_mesh())
    interfaces = sorted(interfaces_on_mesh(mesh))
    ensemble = _ensemble_for(mesh, flagged=(interfaces[2],))
    viewing.plot_tension_inference(mesh, seed_ensemble=ensemble)
    on_mesh = {k: v for k, v in ensemble.tensions.items() if k in set(interfaces)}
    faces = np.array([on_mesh.get((int(f[3]), int(f[4])), np.nan) for f in mesh.f])
    expected = tension_colours(clipped_colour_fractions(faces, colour_bar_ranges(mesh, on_mesh).tension))
    assert np.array_equal(recorded["Surface Tensions"], expected)
    assert "Tension spread over the seeds (SD)" in recorded
    marked = mock_polyscope.get_curve_network("Interfaces found by fewer than half of the seeds")
    triangles = np.asarray(mesh.f)[[(int(f[3]), int(f[4])) == interfaces[2] for f in mesh.f]][:, :3]
    edges = {tuple(sorted((int(t[i]), int(t[(i + 1) % 3])))) for t in triangles for i in range(3)}
    assert marked.n_edges() == len(edges)
    assert marked.n_nodes() == len({vertex for edge in edges for vertex in edge})  # no node off the marked triangles
    titles = [scale.title for scale in windows[-1].scales.scales]
    assert titles == ["Surface tension", "Tension spread over the seeds (SD)"]
    with pytest.raises(ValueError, match="not both"):
        viewing.plot_tension_inference(mesh, on_mesh, seed_ensemble=ensemble)


# ---------------------------------------------------------------------------------------------
# foambryo view
# ---------------------------------------------------------------------------------------------
def _run(capsys, *arguments: object):
    code = cli.main([str(argument) for argument in arguments])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.fixture
def recorded_viewer(monkeypatch):
    """Replace the three viewer functions by recorders: the command's arguments to them, without a window."""
    pytest.importorskip("polyscope")
    import foambryo.viewing as viewing

    calls: list[tuple[str, tuple, dict]] = []
    for name in ("plot_force_inference", "plot_tension_inference", "plot_pressure_inference"):
        monkeypatch.setattr(viewing, name, lambda *args, _name=name, **kwargs: calls.append((_name, args, kwargs)))
    return calls


def test_view_infers_as_infer_does_with_the_same_options(capsys, recorded_viewer):
    mesh_path = require_substitute_mesh()
    assert _run(capsys, "view", mesh_path)[0] == cli.EXIT_OK
    name, args, kwargs = recorded_viewer[-1]
    tensions, pressures = infer_forces(dcel_mesh_from_file(mesh_path))
    assert name == "plot_force_inference"
    assert args[1] == tensions  # bitwise: the same keys, and floats that compare equal
    assert args[2] == pressures
    assert kwargs["mode_tension"] is TensionComputationMethod.YoungDupre
    assert kwargs["length_unit"] is None  # the benchmark mesh records no unit: no scale bar
    assert _run(capsys, "view", mesh_path, "--tension-method", "lami", "--pressure-method", "laplace",
                "--solver", "dense", "--mean-tension", "2")[0] == cli.EXIT_OK
    name, args, kwargs = recorded_viewer[-1]
    tensions, pressures = infer_forces(dcel_mesh_from_file(mesh_path), 2.0, 0.0, TensionComputationMethod.Lami,
                                       PressureComputationMethod.Laplace, solver="dense")
    assert args[1] == tensions
    assert args[2] == pressures
    assert kwargs["mode_pressure"] is PressureComputationMethod.Laplace
    assert kwargs["solver"] == "dense"
    assert "mode_tension=TensionComputationMethod.Lami" in kwargs["values_from"]


def test_view_inference_defaults_are_the_librarys():
    args = cli.build_parser().parse_args(["view", "m.rec"])
    resolved = cli._view_inference_args(args)
    library = inspect.signature(infer_forces).parameters
    assert cli.TENSION_METHODS[resolved.tension_method] is library["mode_tension"].default
    assert cli.PRESSURE_METHODS[resolved.pressure_method] is library["mode_pressure"].default
    assert resolved.mean_tension == library["mean_tension"].default
    assert resolved.base_pressure == library["base_pressure"].default
    assert resolved.solver == library["solver"].default


def test_view_refuses_inference_options_with_results(capsys, tmp_path):
    mesh_path = require_substitute_mesh()
    assert _run(capsys, "infer", mesh_path, "-o", tmp_path / "out")[0] == cli.EXIT_OK
    message = ("foambryo view: error: --tension-method applies when 'foambryo view' infers: with --results the "
               "values are read from OUTDIR\n")
    assert _run(capsys, "view", mesh_path, "--results", tmp_path / "out", "--tension-method", "lami") == (
        cli.EXIT_USAGE, "", message)


def test_view_figure_without_a_window(capsys, tmp_path, recorded_viewer):
    mesh_path = require_substitute_mesh()
    code, out, _ = _run(capsys, "view", mesh_path, "--figure", tmp_path / "f.png", "--no-show", "--no-scale-bar")
    assert code == cli.EXIT_OK
    kwargs = recorded_viewer[-1][2]
    assert kwargs["save_figure"] == str(tmp_path / "f.png")
    assert kwargs["show"] is False
    assert kwargs["scale_bar"] is False
    assert out.endswith(f"wrote {tmp_path / 'f.png'}\n")


def test_view_draws_a_scale_bar_in_the_unit_the_mesh_records(capsys, tmp_path, recorded_viewer):
    from tests.test_command_line import _four_cells_in_a_ball

    np.save(tmp_path / "mask.npy", _four_cells_in_a_ball())
    assert _run(capsys, "mesh", tmp_path / "mask.npy", "-o", tmp_path / "m.rec", "--spacing", 0.5, 0.5, 0.5,
                "--spacing-unit", "um")[0] == cli.EXIT_OK
    assert _run(capsys, "view", tmp_path / "m.rec")[0] == cli.EXIT_OK
    assert recorded_viewer[-1][2]["length_unit"] == "um"
    assert recorded_viewer[-1][2]["scale_bar"] is True
    assert _run(capsys, "view", tmp_path / "m.rec", "--scale-bar", 2)[0] == cli.EXIT_OK
    assert recorded_viewer[-1][2]["scale_bar"] == 2.0
    assert _run(capsys, "view", require_substitute_mesh(), "--scale-bar", 0.2)[0] == cli.EXIT_OK
    assert recorded_viewer[-1][2]["length_unit"] == "mesh units"


def _write_ensemble_table(outdir: Path, rows: dict) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    lines = [",".join(tables.ENSEMBLE_TENSION_COLUMNS)]
    for (a, b), (tension, sd, n, flag) in sorted(rows.items()):
        lines.append(f"{a},{b},{tension!r},{sd!r},{n},{'true' if flag else 'false'}")
    (outdir / tables.TENSION_TABLE).write_text("\n".join(lines) + "\n", encoding="ascii")


def test_view_reads_what_infer_over_seeds_wrote(capsys, tmp_path, recorded_viewer):
    mesh_path = require_substitute_mesh()
    mesh = dcel_mesh_from_file(mesh_path)
    interfaces = sorted(interfaces_on_mesh(mesh))
    rows = {key: (1.0 + 0.01 * i, 0.03, 8, i == 2) for i, key in enumerate([*interfaces[1:], (7, 8)])}
    _write_ensemble_table(tmp_path / "ensemble", rows)
    bars = tmp_path / "bars.png"
    code, out, err = _run(capsys, "view", mesh_path, "--results", tmp_path / "ensemble", "--colour-bars", bars)
    assert code == cli.EXIT_OK, err
    assert "warning: UserWarning: 1 interfaces of the ensemble are not on" in err
    assert "1 interfaces of 000_mesh.rec are not in the ensemble (drawn undetermined)" in err
    on_mesh = {key: row[0] for key, row in rows.items() if key in set(interfaces)}
    low, high = colour_bar_ranges(mesh, on_mesh).tension
    assert f"tension {low:.6g} to {high:.6g}\n" in out
    name, _, kwargs = recorded_viewer[-1]
    assert name == "plot_tension_inference"
    assert kwargs["seed_ensemble"].tensions == {key: row[0] for key, row in rows.items()}
    assert kwargs["values_from"].startswith("tensions combined over seeds, read from")


def test_view_refuses_an_ensemble_of_another_mask(capsys, tmp_path):
    _write_ensemble_table(tmp_path / "ensemble", {(70, 80): (1.0, 0.1, 8, False)})
    code, _, err = _run(capsys, "view", require_substitute_mesh(), "--results", tmp_path / "ensemble",
                        "--colour-bars", tmp_path / "b.png", "--no-show")
    assert code == cli.EXIT_FILE
    assert "no interface of 000_mesh.rec is in" in err
    assert not (tmp_path / "b.png").exists()


def test_view_reports_a_refused_inference(capsys, monkeypatch, recorded_viewer):
    """Tensions refused: exit 4 with the library's message. Pressures refused: the tensions alone, with a warning."""
    mesh_path = require_substitute_mesh()
    monkeypatch.setattr(cli, "infer_tensions_then_pressures",
                        lambda *_args, **_kwargs: InferenceOutcome(None, None, [], "failed", "no triple line"))
    assert _run(capsys, "view", mesh_path) == (cli.EXIT_REFUSED, "", "foambryo view: error: failed: no triple line\n")
    tensions = {(0, 1): 1.0}
    monkeypatch.setattr(cli, "infer_tensions_then_pressures", lambda *_args, **_kwargs: InferenceOutcome(
        tensions, None, [], "pressures refused", "undetermined tension"))
    code, _, err = _run(capsys, "view", mesh_path)
    assert code == cli.EXIT_OK
    assert "warning: UserWarning: the pressures are not shown: undetermined tension" in err
    assert recorded_viewer[-1][0] == "plot_tension_inference"
    assert recorded_viewer[-1][1][1] == tensions
