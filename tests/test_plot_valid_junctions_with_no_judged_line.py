"""`plot_valid_junctions` shows the mesh alone, and says so, when no triple line can be judged.

Before 0.7.0 it took the maximum of an empty list and raised a numpy `ValueError` when no triple line was judged: a
mesh without triple lines, or one whose every line has an interface absent from the tension map (as a crop
interface is). Pins both cases, on Polyscope's headless mock backend, and that a mesh with judged lines draws them
without the warning. Skipped when Polyscope is not installed (the optional `viewing` extra).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from foambryo import dcel_mesh_from_file, dcel_mesh_from_segmentation_mask

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


def one_cell_mesh():
    z, y, x = np.mgrid[:20, :20, :20].astype(float)
    mask = ((z - 10) ** 2 + (y - 10) ** 2 + (x - 10) ** 2 < 6**2).astype(np.uint8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dcel_mesh_from_segmentation_mask(mask)


@pytest.mark.usefixtures("headless")
def test_a_mesh_without_triple_lines_is_shown_alone_with_a_warning() -> None:
    from foambryo.viewing import plot_valid_junctions

    mesh = one_cell_mesh()
    assert mesh.compute_length_trijunctions() == {}
    with pytest.warns(UserWarning, match="no triple line could be judged"):
        plot_valid_junctions(mesh, {(0, 1): 1.0})


@pytest.mark.usefixtures("headless")
def test_lines_whose_interfaces_have_no_tension_are_not_judged() -> None:
    from foambryo.viewing import plot_valid_junctions

    mesh = dcel_mesh_from_file(str(require_substitute_mesh()))
    with pytest.warns(UserWarning, match="no triple line could be judged"):
        plot_valid_junctions(mesh, {})


@pytest.mark.usefixtures("headless")
def test_judged_lines_are_drawn_without_the_warning() -> None:
    from foambryo.viewing import plot_valid_junctions

    mesh = dcel_mesh_from_file(str(require_substitute_mesh()))
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="plot_valid_junctions", category=UserWarning)
        plot_valid_junctions(mesh)
