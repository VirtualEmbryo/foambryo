"""The viewer's window: tensions first, undetermined outlines, a frame, live explode, drawing off its thread, Export.

What this pins, and why (each the answer to a request about the window):

* with tensions and pressures, the surface shows the **tensions** first; the panel switches to the pressures;
* an **undetermined** tension's interface is drawn as its outline in the undetermined green, without fill, and the
  crop faces are never filled with a value's colour (a per-face transparency); the pressure view fills every cell;
* an optional **x, y, z frame**: three segments of one round length from the drawn box's lowest corner, along the
  mesh's columns (z, y, x), labelled at their tips;
* **explode is live**: changing the coefficient draws the scattered view at the new coefficient, the library's
  `compute_scattered_arrays(coeff)` positions bitwise; and the graph view's nodes and the stress tensors' centres
  stay at their exploded cells' barycentres however often the coefficient changes (they used to run away: each
  explode added its displacement to the mesh's own centroids), with the unexploded sphere-fit residuals;
* the scene is **prepared off the window's thread**: the worker computes and records, Polyscope is called only when
  the window installs the result, and the installed colours are the library's (the colour gates elsewhere);
* **the background**: the library's named backgrounds (`BACKGROUND_COLOURS`, the napari plugin's choices) paint
  Polyscope's, which is handed the value it shows as that colour (it tone-maps its background); on a dark one, the
  frame, the scale bar, their labels and the graph's black nodes are drawn light, on a light one dark, also in a
  scene drawn after the choice;
* **Export** writes `foambryo infer`'s files: `tensions.csv`, `pressures.csv`, `solves.csv` (with diagnostics) byte for
  byte the command's, `colour_bars.png` byte for byte `foambryo view --colour-bars`', `run.json` the command's record;
  it never replaces a file without being told to, and never touches another file.
"""

from __future__ import annotations

import inspect
import json
import threading

import numpy as np
import pytest

from foambryo import TensionComputationMethod, dcel_mesh_from_file, infer_forces, infer_tensions
from foambryo._colour_scales import UNDETERMINED_COLOR
from foambryo.dcel import DcelData

from .conftest import require_substitute_mesh
from .test_command_line import _run


@pytest.fixture
def mock_polyscope(monkeypatch):
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
    ps.remove_all_slice_planes()


@pytest.fixture
def enabled_colours(monkeypatch, mock_polyscope):
    """The colour quantities added to the coloured surface, in order, with `enabled`."""
    added: list[tuple[str, np.ndarray, bool]] = []
    original = mock_polyscope.SurfaceMesh.add_color_quantity

    def recording(self, name, values, *args: object, **kwargs: object):
        if self.get_name() == "Embryo":
            added.append((name, np.array(values), bool(kwargs.get("enabled", False))))
        return original(self, name, values, *args, **kwargs)

    monkeypatch.setattr(mock_polyscope.SurfaceMesh, "add_color_quantity", recording)
    return added


def _window(kind="force", tensions=None, pressures=None, mesh=None, **kwargs: object):
    from foambryo.viewing import _ForceWindow, _InferenceOptions

    mesh = mesh if mesh is not None else dcel_mesh_from_file(require_substitute_mesh())
    if tensions is None:
        tensions, pressures = infer_forces(mesh)
    options = _InferenceOptions(TensionComputationMethod.YoungDupre, inspect.signature(infer_forces)
                                .parameters["mode_pressure"].default, 1.0, 0.0, "auto")
    return _ForceWindow(mesh, kind, tensions, pressures if kind == "force" else None, options, "given",
                        mesh_name=str(require_substitute_mesh()), **kwargs)


def test_tensions_are_shown_first_and_the_panel_switches(enabled_colours):
    window = _window()
    window.draw()
    shown = [name for name, _, enabled in enabled_colours if enabled]
    assert shown[-1] == "Surface Tensions"
    window.showing = "pressure"
    window.apply_showing()
    assert [name for name, _, enabled in enabled_colours if enabled][-1] == "Pressures Cells"


def test_an_undetermined_interface_is_an_outline_without_fill(mock_polyscope, monkeypatch):
    from foambryo.viewing import _TENSION_FILL, _UNDETERMINED_OUTLINE

    fills = {}
    original = mock_polyscope.SurfaceMesh.add_scalar_quantity

    def recording(self, name, values, *args: object, **kwargs: object):
        fills[name] = np.array(values)
        return original(self, name, values, *args, **kwargs)

    monkeypatch.setattr(mock_polyscope.SurfaceMesh, "add_scalar_quantity", recording)
    mesh = dcel_mesh_from_file(require_substitute_mesh())
    tensions = infer_tensions(mesh)
    missing = next(iter(tensions))
    tensions[missing] = float("nan")
    window = _window("tension", tensions, None, mesh)
    window.draw()
    undetermined = np.array([(int(a), int(b)) == missing for a, b in mesh.f[:, 3:5]])
    assert np.array_equal(fills[_TENSION_FILL], np.where(undetermined, 0.0, 1.0))
    outline = mock_polyscope.get_curve_network(_UNDETERMINED_OUTLINE)
    assert outline.is_enabled()
    assert np.allclose(outline.get_color(), UNDETERMINED_COLOR)
    assert window.scales.undetermined


def test_the_pressure_view_hides_no_cell_and_no_outline(mock_polyscope):
    from foambryo.viewing import _UNDETERMINED_OUTLINE

    mesh = dcel_mesh_from_file(require_substitute_mesh())
    tensions, pressures = infer_forces(mesh)
    tensions = dict(tensions)
    del tensions[next(iter(tensions))]  # absent from the map: undetermined (a nan tension has no pressures)
    window = _window("force", tensions, pressures, mesh)
    window.draw()
    window.showing = "pressure"
    window.apply_showing()
    assert not mock_polyscope.get_curve_network(_UNDETERMINED_OUTLINE).is_enabled()


def test_the_frame_is_three_round_segments_from_the_lowest_corner(mock_polyscope):
    from foambryo._viewer_figures import round_scale_bar_length
    from foambryo.viewing import _FRAME

    window = _window()
    window.show_frame = True
    window.draw()
    low, high = window.drawn_box()
    origin, length = window.frame
    extent = float(np.max(high - low))
    assert np.array_equal(origin, low - 0.1 * extent)  # just outside the box, so that the tissue does not hide it
    assert length == round_scale_bar_length(extent)
    assert mock_polyscope.get_curve_network(_FRAME).is_enabled()
    labels = [text for text, _, _ in window.frame_labels(800, 600)]
    assert labels[1:] == ["z", "y", "x"]  # columns 0, 1, 2 of the mesh's points are z, y, x


def test_explode_follows_the_coefficient_live_off_the_windows_thread(mock_polyscope, monkeypatch):
    window = _window()
    window.draw()
    worker_calls = []
    original = mock_polyscope.register_surface_mesh

    def register(*args: object, **kwargs: object):
        worker_calls.append(threading.current_thread() is threading.main_thread())
        return original(*args, **kwargs)

    monkeypatch.setattr(mock_polyscope, "register_surface_mesh", register)
    window.scattered, window.scatter_coeff = True, 0.45
    window.request_redraw()
    window._future.result(timeout=120)
    assert worker_calls == []  # prepared on the worker: Polyscope not called yet
    assert window.poll_redraw()
    assert worker_calls
    assert all(worker_calls)  # installed on the window's thread
    expected = DcelData(np.asarray(window.mesh.v), np.asarray(window.mesh.f[:, :3]), np.asarray(window.mesh.f[:, 3:5]))
    expected.compute_scattered_arrays(0.45)
    drawn = mock_polyscope.get_surface_mesh("Embryo")
    assert drawn.n_vertices() == len(expected.v_scattered)
    assert np.array_equal(window.mesh.v_scattered, expected.v_scattered)
    window.close()


def test_live_explode_keeps_the_graph_and_the_stress_tensors_on_the_cells(mock_polyscope, monkeypatch):
    import foambryo.viewing as viewing
    from foambryo.curvature import compute_sphere_fit_residues_dict

    points: dict[str, np.ndarray] = {}
    original = mock_polyscope.register_point_cloud

    def register(name, positions, *args: object, **kwargs: object):
        points[name] = np.array(positions, dtype=np.float64)
        return original(name, positions, *args, **kwargs)

    residuals: list[dict] = []
    original_scalars = viewing._view_scalar_quantities

    def scalars(mesh, ps_mesh, dict_sphere_fit_residual, *args: object):
        residuals.append(dict(dict_sphere_fit_residual))
        return original_scalars(mesh, ps_mesh, dict_sphere_fit_residual, *args)

    monkeypatch.setattr(mock_polyscope, "register_point_cloud", register)
    monkeypatch.setattr(viewing, "_view_scalar_quantities", scalars)
    window = _window(scalar_quantities=True)
    window.draw()
    mesh = window.mesh
    centroids = {cell: value.copy() for cell, value in mesh.centroids.items()}
    unexploded = compute_sphere_fit_residues_dict(mesh.compute_networkx_graph())
    graph = mesh.compute_networkx_graph()
    drawn_cells = [cell for cell, volume in graph.nodes.data("volume") if volume > 0 and cell != 0]
    tolerance = 1e-12 * float(np.ptp(mesh.v, axis=0).max())
    for coeff in (0.2, 0.5, 1.0, 0.5, 0.2, 0.8, 0.1, 0.6):
        window.scattered, window.scatter_coeff = True, coeff
        window.request_redraw()
        window._future.result(timeout=120)
        assert window.poll_redraw()
        cell_of_vertex = np.zeros(len(mesh.v_scattered), dtype=np.int64)
        cell_of_vertex[mesh.f_scattered[:, :3].ravel()] = np.repeat(mesh.cluster_idx_scattered.astype(np.int64), 3)
        barycentres = np.array([mesh.v_scattered[cell_of_vertex == cell].mean(axis=0) for cell in drawn_cells])
        for name in ("Graph Nodes", "Stress tensors principal directions"):
            assert np.linalg.norm(points[name] - barycentres, axis=1).max() <= tolerance, (coeff, name)
        assert residuals[-1] == unexploded
        assert all(np.array_equal(mesh.centroids[cell], centroids[cell]) for cell in centroids)
    window.close()


def test_the_background_paints_polyscope_and_the_ink_follows_it(mock_polyscope):
    import foambryo.viewing as viewing
    from foambryo.colour_scales import BACKGROUND_COLOURS

    window = _window(length_unit="um")
    window.draw()
    scale_bar = f"Scale bar ({window.scale_bar.label})"
    try:
        for name, rgb in BACKGROUND_COLOURS.items():
            viewing._set_background(rgb, window)
            # Polyscope is handed the value it shows as `rgb` (it tone-maps its background)
            assert np.allclose(mock_polyscope.get_background_color()[:3], viewing._polyscope_background_value(rgb))
            assert np.allclose(viewing._shown_background(mock_polyscope.get_background_color()), rgb)
            dark = name in ("light grey", "white")
            for structure, grey in ((mock_polyscope.get_curve_network(viewing._FRAME), viewing._FRAME_COLOR),
                                    (mock_polyscope.get_curve_network(scale_bar), viewing._SCALE_BAR_COLOR),
                                    (mock_polyscope.get_point_cloud("Graph Nodes"), (0.0, 0.0, 0.0))):
                expected = grey if dark else tuple(1 - c for c in grey)
                assert np.allclose(structure.get_color(), expected), (name, structure.get_name())
        viewing._set_background(BACKGROUND_COLOURS["black"], window)
        window.scattered, window.scatter_coeff = True, 0.3  # a scene drawn after the choice takes the light ink
        window.request_redraw()
        window._future.result(timeout=120)
        assert window.poll_redraw()
        assert np.allclose(mock_polyscope.get_curve_network(viewing._FRAME).get_color(),
                           tuple(1 - c for c in viewing._FRAME_COLOR))
        assert np.allclose(mock_polyscope.get_point_cloud("Graph Nodes").get_color(), (1.0, 1.0, 1.0))
    finally:
        viewing._set_background(BACKGROUND_COLOURS["white"], None)
        window.close()


@pytest.mark.usefixtures("mock_polyscope")
def test_a_second_request_during_a_preparation_is_drawn_next():
    window = _window()
    window.draw()
    window.scattered = True
    window.request_redraw()
    window.scatter_coeff = 0.7
    window.request_redraw()  # queued: one preparation at a time
    window._future.result(timeout=120)
    window.poll_redraw()
    window._future.result(timeout=120)
    window.poll_redraw()
    assert np.allclose(window.drawn_box(), _scattered_box(window.mesh, 0.7))
    window.close()


def _scattered_box(mesh, coeff):
    copy = DcelData(np.asarray(mesh.v), np.asarray(mesh.f[:, :3]), np.asarray(mesh.f[:, 3:5]))
    copy.compute_scattered_arrays(coeff)
    return np.stack([copy.v_scattered.min(axis=0), copy.v_scattered.max(axis=0)])


@pytest.mark.usefixtures("mock_polyscope")
def test_export_writes_the_commands_files_byte_for_byte(capsys, tmp_path):
    from foambryo._viewer_figures import TENSION_TITLE  # noqa: F401 - the window's module imports cleanly
    from foambryo.viewing import _ForceWindow, _InferenceOptions

    mesh_path = require_substitute_mesh()
    mesh = dcel_mesh_from_file(mesh_path)
    options = _InferenceOptions(TensionComputationMethod.YoungDupre, inspect.signature(infer_forces)
                                .parameters["mode_pressure"].default, 1.0, 0.0, "auto", diagnostics="basic")
    tensions, pressures, values_from = options.infer(mesh, "force", None, None, None)
    window = _ForceWindow(mesh, "force", tensions, pressures, options, values_from, mesh_name=str(mesh_path))
    written = window.export(tmp_path / "window")
    assert written == ["tensions.csv", "pressures.csv", "solves.csv", "colour_bars.png", "run.json"]
    assert _run(capsys, "infer", mesh_path, "-o", tmp_path / "command", "--diagnostics", "basic")[0] == 0
    assert _run(capsys, "view", mesh_path, "--results", tmp_path / "command", "--colour-bars",
                tmp_path / "command" / "colour_bars.png", "--no-show")[0] == 0
    for name in ("tensions.csv", "pressures.csv", "colour_bars.png"):
        assert (tmp_path / "window" / name).read_bytes() == (tmp_path / "command" / name).read_bytes(), name
    # solves.csv: every column byte for byte but the solves' wall times, which differ between any two runs
    assert _without_times(tmp_path / "window" / "solves.csv") == _without_times(tmp_path / "command" / "solves.csv")
    window_record = json.loads((tmp_path / "window" / "run.json").read_text())
    command_record = json.loads((tmp_path / "command" / "run.json").read_text())
    for key in ("schema", "options", "library_call", "mesh", "results", "units"):
        assert window_record[key] == command_record[key], key
    assert window_record["input"]["sha256"] == command_record["input"]["sha256"]


def _without_times(path):
    rows = [line.split(",") for line in path.read_text().splitlines()]
    timed = [i for i, column in enumerate(rows[0]) if column in ("seconds", "spectrum_seconds")]
    return [[value for i, value in enumerate(row) if i not in timed] for row in rows]


@pytest.mark.usefixtures("mock_polyscope")
def test_export_asks_before_replacing_and_touches_no_other_file(tmp_path):
    window = _window()
    window.draw()
    folder = tmp_path / "out"
    folder.mkdir()
    (folder / "notes.txt").write_text("mine")
    window.export(folder)
    with pytest.raises(FileExistsError, match="already holds"):
        window.export(folder)
    window._panel_export  # noqa: B018 - the panel's Replace / Cancel calls export(overwrite=True) or nothing
    assert window.export(folder, overwrite=True)
    assert (folder / "notes.txt").read_text() == "mine"
    record = json.loads((folder / "run.json").read_text())
    assert record["values_given_to_the_window"] == "given"  # given values say so; no solve record
    assert not (folder / "solves.csv").exists()


def test_view_mask_takes_the_commands_options_with_the_librarys_defaults():
    from dw3d import get_default_mesh_reconstruction_algorithm

    import foambryo.viewing as viewing

    parameters = inspect.signature(viewing.view_mask).parameters
    library = inspect.signature(infer_forces).parameters
    for name in ("mean_tension", "base_pressure", "mode_tension", "mode_pressure", "solver"):
        assert parameters[name].default == library[name].default
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["min_distance"].default is None  # resolved to delaunay-watershed-3d's own default
    assert "min_distance" in inspect.signature(get_default_mesh_reconstruction_algorithm).parameters
    assert parameters["pressures"].default is True


@pytest.mark.usefixtures("mock_polyscope")
def test_the_window_pauses_its_frames_while_a_scene_is_prepared(monkeypatch):
    """The panel sleeps while a scene is being prepared, so that the worker thread runs.

    Polyscope's loop holds the GIL outside the panel: without a pause each frame, the worker starves (55 s instead
    of 4 s for a 2 000-cell foam, measured).
    """
    import foambryo.viewing as viewing

    window = _window()
    window.draw()
    release = threading.Event()
    sleeps: list[float] = []
    monkeypatch.setattr(viewing.time, "sleep", sleeps.append)
    real_prepare = window.prepare
    monkeypatch.setattr(window, "prepare", lambda *args: (release.wait(60), real_prepare(*args))[1])
    window.request_redraw()
    assert not window.poll_redraw()
    assert sleeps == [viewing._YIELD_TO_THE_WORKER_SECONDS]
    release.set()
    window._future.result(timeout=120)
    assert window.poll_redraw()
    assert len(sleeps) == 1
    window.close()


def test_a_replay_initialises_polyscope_first(mock_polyscope, monkeypatch):
    """The recorder drops the drawing code's `init`; the replay makes it.

    So a first window opens in a fresh process: the kit's `figures` run failed with "Polyscope has not been
    initialized" before.
    """
    from foambryo._polyscope_recording import Recorder, polyscope_target, recording

    calls: list[str] = []
    real_init = mock_polyscope.init
    monkeypatch.setattr(mock_polyscope, "init", lambda *a, **k: (calls.append("init"), real_init(*a, **k))[1])
    recorder = Recorder()
    with recording(recorder):
        polyscope_target.init()
        polyscope_target.register_curve_network("a", np.zeros((2, 3)), np.array([[0, 1]]))
    assert calls == []  # recorded, not made
    recorder.replay()
    assert calls == ["init"]
    assert mock_polyscope.has_curve_network("a")
