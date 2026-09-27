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
    assert public == {call.split("(")[0] for call in CALLS}


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
