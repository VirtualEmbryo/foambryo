"""Every public function of `foambryo.viewing` runs once on a real mesh, without a display.

Polyscope's headless mock backend (`openGL_mock`) stands in for the window: every `ps.init`
is routed to it (the viewer calls `ps.init()` with the default backend, and Polyscope refuses
to re-initialise with another one), and every `ps.show` renders two frames instead of waiting
for a window that is never closed (on the mock backend a plain `show()` never returns).
Skipped when Polyscope is not installed (it is the optional `viewing` extra) or its mock
backend cannot start. A `ComplexWarning` fails the test: complex results where real ones are
expected are what broke the force viewer under numpy 2.
"""

import numpy as np
import pytest

from foambryo import dcel_mesh_from_file

from .conftest import require_substitute_mesh

ps = pytest.importorskip("polyscope")


@pytest.fixture(scope="module")
def headless():
    real_init, real_show = ps.init, ps.show
    try:
        real_init("openGL_mock")
    except RuntimeError as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"Polyscope's mock backend is unavailable: {error}")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ps, "init", lambda *_args, **_kwargs: real_init("openGL_mock"))
        patch.setattr(ps, "show", lambda *_args, **_kwargs: real_show(forFrames=2))
        yield
    ps.remove_all_structures()


@pytest.fixture(scope="module")
def mesh():
    return dcel_mesh_from_file(str(require_substitute_mesh()))


CALLS = {
    "plot_force_inference": {},
    "plot_force_inference(scalar_quantities=True)": {"scalar_quantities": True},
    "plot_force_inference(scattered=True)": {"scattered": True},
    "plot_pressure_inference": {},
    "plot_tension_inference": {},
    "plot_tension_inference(scalar_quantities=True)": {"scalar_quantities": True},
    "plot_valid_junctions": {},
    "plot_residual_junctions": {},
}


def test_every_public_function_is_called():
    import foambryo.viewing as viewing

    public = {
        name
        for name, value in vars(viewing).items()
        if callable(value) and not name.startswith("_") and getattr(value, "__module__", "") == viewing.__name__
    }
    assert public == {call.split("(")[0] for call in CALLS} | {"view_mask"}


@pytest.mark.usefixtures("headless")
def test_view_mask_runs_headless():
    """`view_mask` (`foambryo view MASK`'s window from Python) opens on a mask and closes, with no Polyscope error."""
    import foambryo.viewing as viewing

    mask = np.zeros((12, 12, 12), dtype=np.uint8)
    mask[3:9, 3:6, 3:9] = 1
    mask[3:9, 6:9, 3:9] = 2
    viewing.view_mask(mask, (1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="spacing"):
        viewing.view_mask(mask, (1.0, 0.0, 1.0))


@pytest.mark.usefixtures("headless")
@pytest.mark.filterwarnings("error::numpy.exceptions.ComplexWarning")
@pytest.mark.parametrize("call", list(CALLS))
def test_viewer_function_runs_headless(mesh, call):
    import foambryo.viewing as viewing

    getattr(viewing, call.split("(")[0])(mesh, **CALLS[call])
    assert np.isfinite(mesh.v).all()


# ---------------------------------------------------------------------------------------------
# what the stress viewer draws, and under which name and colour
# ---------------------------------------------------------------------------------------------
RED, TEAL = (0.7, 0.0, 0.0), (0.2, 0.5, 0.5)


@pytest.mark.usefixtures("headless")
@pytest.mark.parametrize(
    ("pressure_over_laplace", "drawn", "empty"),
    [(0.0, "tension", "compression"), (2.0, "compression", "tension")],
)
def test_stress_viewer_draws_each_sign_under_its_own_name_and_colour(monkeypatch, pressure_over_laplace, drawn, empty):
    """A bubble with no pressure is purely tensile, one above its Laplace pressure purely compressive."""
    import foambryo.viewing as viewing
    from tests.test_cell_stress import _bubble

    recorded: dict[str, tuple[np.ndarray, tuple[float, float, float]]] = {}

    class Recorder:
        def add_vector_quantity(self, name: str, vectors: np.ndarray, **kwargs: object) -> None:
            recorded[name] = (np.asarray(vectors), tuple(kwargs["color"]))

    monkeypatch.setattr(ps, "register_point_cloud", lambda *_args, **_kwargs: Recorder())
    gamma, radius = 1.0, 1.0
    mesh = _bubble(3, radius)
    pressure = pressure_over_laplace * 2 * gamma / radius
    viewing._plot_stress_tensor(
        mesh,
        mesh.compute_networkx_graph(),
        {(0, 1): gamma},
        {0: 0.0, 1: pressure},
        show=False,
    )

    assert set(recorded) == {
        f"{n}_axes_{k}{s}" for n in ("compression", "tension") for k in range(3) for s in ("", "_down")
    }
    for name, (vectors, colour) in recorded.items():
        assert colour == (RED if name.startswith("compression") else TEAL)
        if name.startswith(drawn):
            assert np.linalg.norm(vectors) > 0.1 * gamma / radius
        else:
            assert name.startswith(empty)
            assert not np.any(vectors)


# ---------------------------------------------------------------------------------------------
# an undetermined pressure (nan, from a Laplace method): drawn in the undetermined colour, never breaks a viewer
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def forces(mesh):
    from foambryo import infer_forces

    return infer_forces(mesh)


def _with_one_undetermined_cell(pressures: dict[int, float], cell: int = 1) -> dict[int, float]:
    return {c: (float("nan") if c == cell else p) for c, p in pressures.items()}


@pytest.mark.usefixtures("headless")
@pytest.mark.parametrize("call", ["plot_force_inference", "plot_pressure_inference"])
def test_pressure_viewers_run_with_an_undetermined_pressure(mesh, forces, call):
    import foambryo.viewing as viewing

    tensions, pressures = forces
    kwargs = {"dict_pressure": _with_one_undetermined_cell(pressures)}
    if call == "plot_force_inference":
        kwargs["dict_tensions"] = tensions
    getattr(viewing, call)(mesh, **kwargs)


class _ColourRecorder:
    def __init__(self, recorded: dict) -> None:
        self.recorded = recorded

    def add_color_quantity(self, name: str, colours: np.ndarray, **_kwargs: object) -> None:
        self.recorded[name] = np.array(colours)

    def __getattr__(self, _name: str) -> object:  # set_color, add_vector_quantity, ...: not under test
        return lambda *_args, **_kwargs: None


@pytest.mark.usefixtures("headless")
def test_the_pressure_mesh_colours_an_undetermined_cell_and_is_unchanged_otherwise(monkeypatch, mesh, forces):
    import matplotlib.cm as cm

    import foambryo.viewing as viewing

    recorded: dict[str, np.ndarray] = {}
    monkeypatch.setattr(ps, "register_surface_mesh", lambda *_args, **_kwargs: _ColourRecorder(recorded))
    _, pressures = forces
    alpha = 0.05

    viewing._view_pressures_on_mesh(mesh, pressures, alpha=alpha)
    face_pressures = np.array([pressures[max(f[3], f[4])] for f in mesh.f])
    clipped = face_pressures.clip(0, np.quantile(face_pressures, 1 - alpha))
    expected = cm.magma(0.1 + 0.8 * (1 - clipped / np.amax(clipped)))[:, :3]  # magma reversed, trimmed to 10-90 %
    assert recorded["Pressures Cells"].tobytes() == expected.tobytes()

    viewing._view_pressures_on_mesh(mesh, _with_one_undetermined_cell(pressures), alpha=alpha)
    undetermined = np.max(mesh.f[:, 3:], axis=1) == 1
    assert undetermined.any()
    assert (recorded["Pressures Cells"][undetermined] == viewing.UNDETERMINED_COLOR).all()
    assert np.isfinite(recorded["Pressures Cells"]).all()


@pytest.mark.usefixtures("headless")
def test_the_force_graph_draws_an_undetermined_cell_in_its_colour(monkeypatch, mesh, forces):
    import networkx as nx

    import foambryo.viewing as viewing

    recorded: dict[str, np.ndarray] = {}
    monkeypatch.setattr(ps, "register_point_cloud", lambda *_args, **_kwargs: _ColourRecorder(recorded))
    monkeypatch.setattr(ps, "register_curve_network", lambda *_args, **_kwargs: _ColourRecorder({}))
    tensions, pressures = forces
    graph = mesh.compute_networkx_graph()
    nx.set_edge_attributes(graph, tensions, "tension")
    nx.set_node_attributes(graph, _with_one_undetermined_cell(pressures), "pressure")

    viewing._display_embryo_graph_forces(graph, show=False)
    nodes = [n for n, v in graph.nodes.data("volume") if v > 0]
    colours = recorded["Pressure"]
    assert (colours[nodes.index(1)] == viewing.UNDETERMINED_COLOR).all()
    assert np.isfinite(colours).all()
    assert sum((c == viewing.UNDETERMINED_COLOR).all() for c in colours) == 1


@pytest.mark.usefixtures("headless")
def test_the_stress_viewer_draws_no_stress_for_an_undetermined_cell(monkeypatch, mesh, forces):
    import foambryo.viewing as viewing

    asked: list[list[int]] = []
    real = viewing.cell_mean_stress_tensors

    def recording(*args: object, **kwargs: object) -> dict:
        asked.append([int(c) for c in args[4]])
        return real(*args, **kwargs)

    monkeypatch.setattr(viewing, "cell_mean_stress_tensors", recording)
    tensions, pressures = forces
    viewing._plot_stress_tensor(
        mesh,
        mesh.compute_networkx_graph(),
        tensions,
        _with_one_undetermined_cell(pressures),
        show=False,
    )
    assert 1 not in asked[0]
    assert len(asked[0]) > 0
