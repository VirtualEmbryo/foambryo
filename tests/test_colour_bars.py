"""The colour bars say which value each of the viewer's colours is.

What this pins, and why: a coloured mesh cannot be read without its scale, and a scale computed
apart from the viewer drifts from it. So:

* the ends of each bar are **exactly** the range the viewer prints for the same input ("Extremas
  ..."), on a mesh with every value finite and on a cropped mesh with an undetermined (`nan`) Laplace
  pressure;
* the viewer's surface colours are the shared functions' colours, array for array, so the bars and
  the viewer cannot be changed one without the other;
* **one pressure scale**: the graph view's cell nodes take the surface's range (over triangles), so a
  cell's node has exactly the colour of its cell on the surface and on the bar (it used to take its
  own quantile over cells: 5.383 against the surface's 5.472 on case 000);
* the image is written with no display, with Polyscope absent and without `matplotlib.pyplot`;
* degenerate scales (every tension equal, no positive pressure) still give an image, and no finite
  tension is an error, not a blank bar.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pytest

from foambryo import (
    PressureComputationMethod,
    dcel_mesh_from_file,
    dcel_mesh_from_multimaterial_mesh,
    infer_forces,
    infer_pressures,
    infer_tensions,
)
from foambryo._colour_scales import (
    colour_bar_ranges,
    pressure_colour_fractions,
    pressure_colour_upper,
    pressure_colours,
    save_colour_bars,
    surface_face_pressures,
    surface_pressure_colour_upper,
    tension_colour_fractions,
    tension_colour_range,
    tension_colours,
)

from .conftest import require_substitute_mesh

CUT_CASE = Path(__file__).parent / "data" / "cut_benchmark_case_000_mesh.npz"


def _benchmark_case():
    mesh = dcel_mesh_from_file(require_substitute_mesh())
    tensions, pressures = infer_forces(mesh)
    return mesh, tensions, pressures


def _cut_case_with_laplace_pressures():
    """The cut benchmark case 000 with Laplace pressures: one cell's pressure is undetermined (nan)."""
    arrays = np.load(CUT_CASE)
    mesh = dcel_mesh_from_multimaterial_mesh(arrays["points"], arrays["triangles"], arrays["labels"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tensions = infer_tensions(mesh)
        pressures = infer_pressures(mesh, tensions, mode=PressureComputationMethod.Laplace)
    assert any(np.isnan(p) for p in pressures.values())
    return mesh, tensions, pressures


CASES = {"benchmark case 000": _benchmark_case, "cut case 000, Laplace pressures": _cut_case_with_laplace_pressures}


@pytest.fixture
def headless_viewer(monkeypatch):
    """Polyscope's mock backend, and a record of every colour array handed to it."""
    ps = pytest.importorskip("polyscope")
    real_init, real_show = ps.init, ps.show
    try:
        real_init("openGL_mock")
    except RuntimeError as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"Polyscope's mock backend is unavailable: {error}")
    monkeypatch.setattr(ps, "init", lambda *_args, **_kwargs: real_init("openGL_mock"))
    monkeypatch.setattr(ps, "show", lambda *_args, **_kwargs: real_show(forFrames=1))
    recorded: dict[str, np.ndarray] = {}
    original = ps.SurfaceMesh.add_color_quantity

    def recording(self, name, values, *args: object, **kwargs: object):
        recorded[name] = np.array(values, dtype=np.float64)
        return original(self, name, values, *args, **kwargs)

    monkeypatch.setattr(ps.SurfaceMesh, "add_color_quantity", recording)
    yield recorded
    ps.remove_all_structures()


@pytest.mark.usefixtures("headless_viewer")
@pytest.mark.parametrize("case", list(CASES))
def test_bar_ends_are_the_range_the_viewer_prints(capsys, tmp_path, case):
    import foambryo.viewing as viewing

    mesh, tensions, pressures = CASES[case]()
    viewing.plot_force_inference(mesh, tensions, pressures)
    printed = {line.split(":")[0]: line.split()[-2:] for line in capsys.readouterr().out.splitlines()
               if line.startswith("Extremas")}
    ranges = viewing.save_colour_bars(mesh, tensions, pressures, tmp_path / "bars.png")
    assert ranges.tension == tuple(float(v) for v in printed["Extremas of the Surface Tensions plotted "])
    assert ranges.pressure == tuple(float(v) for v in printed["Extremas of pressures plotted"])
    assert (tmp_path / "bars.png").stat().st_size > 0


@pytest.mark.parametrize("case", list(CASES))
def test_the_viewers_surface_colours_are_the_shared_functions(headless_viewer, case):
    import foambryo.viewing as viewing

    mesh, tensions, pressures = CASES[case]()
    viewing.plot_force_inference(mesh, tensions, pressures)
    face_tensions = np.array([tensions.get((int(f[3]), int(f[4])), np.nan) for f in mesh.f])
    colour_range = tension_colour_range(np.array(list(tensions.values())), 0.05)
    expected_tensions = tension_colours(tension_colour_fractions(face_tensions, colour_range))
    assert np.array_equal(headless_viewer["Surface Tensions"], expected_tensions)
    face_pressures = surface_face_pressures(mesh, pressures)
    upper = pressure_colour_upper(face_pressures, 0.05)
    expected_pressures = pressure_colours(pressure_colour_fractions(face_pressures, upper))
    assert np.array_equal(headless_viewer["Pressures Cells"], expected_pressures)
    ranges = colour_bar_ranges(mesh, tensions, pressures)
    assert ranges.tension == tuple(float(v) for v in colour_range)
    assert ranges.pressure == (0.0, float(upper))


@pytest.mark.parametrize("scattered", [False, True], ids=["surface", "scattered surface"])
@pytest.mark.parametrize("case", list(CASES))
def test_the_graph_views_cells_have_their_surface_colour(headless_viewer, monkeypatch, case, scattered):
    import polyscope as ps

    import foambryo.viewing as viewing

    original = ps.PointCloud.add_color_quantity

    def recording(self, name, values, *args: object, **kwargs: object):
        headless_viewer[f"points: {name}"] = np.array(values, dtype=np.float64)
        return original(self, name, values, *args, **kwargs)

    monkeypatch.setattr(ps.PointCloud, "add_color_quantity", recording)
    mesh, tensions, pressures = CASES[case]()
    viewing.plot_force_inference(mesh, tensions, pressures, scattered=scattered)
    upper = surface_pressure_colour_upper(mesh, pressures, 0.05, scattered=scattered)
    graph = mesh.compute_networkx_graph()
    cells = [cell for cell, volume in graph.nodes.data("volume") if volume > 0]
    node_pressures = np.array([pressures[cell] for cell in cells], dtype=np.float64)
    nodes = headless_viewer["points: Pressure"]
    assert np.array_equal(nodes, pressure_colours(pressure_colour_fractions(node_pressures, upper)))
    if not scattered:  # each cell's node is the colour of the triangles the surface gives that cell
        face_colours = headless_viewer["Pressures Cells"]
        owner = np.max(mesh.f[:, 3:5], axis=1)
        assert all((owner == cell).any() for cell in cells)
        for row, cell in enumerate(cells):
            assert (face_colours[owner == cell] == nodes[row]).all()
    assert colour_bar_ranges(mesh, tensions, pressures).pressure == (0.0, float(
        surface_pressure_colour_upper(mesh, pressures, 0.05)))


def test_bars_are_written_without_a_display_polyscope_or_pyplot(tmp_path):
    """In a fresh interpreter where importing Polyscope fails: PNG, PDF and SVG all written."""
    script = textwrap.dedent(f"""
        import sys
        sys.modules["polyscope"] = None
        from foambryo import dcel_mesh_from_file, infer_forces
        from foambryo._colour_scales import save_colour_bars
        mesh = dcel_mesh_from_file({str(require_substitute_mesh())!r})
        tensions, pressures = infer_forces(mesh)
        for suffix in ("png", "pdf", "svg"):
            save_colour_bars(mesh, tensions, pressures, {str(tmp_path)!r} + "/bars." + suffix)
        assert "matplotlib.pyplot" not in sys.modules
        assert "foambryo.viewing" not in sys.modules
    """)
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path), "MPLBACKEND": "",
                   "DISPLAY": "", "MPLCONFIGDIR": str(tmp_path / "mplconfig")}
    environment.pop("MPLBACKEND")
    completed = subprocess.run(  # noqa: S603 — fixed script + trusted paths, a test subprocess
        [sys.executable, "-c", script], env=environment, capture_output=True, text=True,
                               check=False, timeout=600)
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "bars.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert (tmp_path / "bars.pdf").read_bytes()[:5] == b"%PDF-"
    assert b"<svg" in (tmp_path / "bars.svg").read_bytes()


def test_degenerate_scales_still_give_an_image(tmp_path):
    mesh, tensions, pressures = _benchmark_case()
    equal = dict.fromkeys(tensions, 1.0)
    zero = dict.fromkeys(pressures, 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no division warning leaks out of the drawing
        ranges = save_colour_bars(mesh, equal, zero, tmp_path / "bars.png")
    assert ranges.tension == (1.0, 1.0)
    assert ranges.pressure == (0.0, 0.0)
    assert (tmp_path / "bars.png").stat().st_size > 0
    assert save_colour_bars(mesh, tensions, None, tmp_path / "tension_only.png").pressure is None


def test_no_finite_tension_is_an_error(tmp_path):
    mesh, tensions, _ = _benchmark_case()
    with pytest.raises(ValueError, match="no tension is finite"):
        save_colour_bars(mesh, dict.fromkeys(tensions, np.nan), None, tmp_path / "bars.png")
    assert not (tmp_path / "bars.png").exists()
