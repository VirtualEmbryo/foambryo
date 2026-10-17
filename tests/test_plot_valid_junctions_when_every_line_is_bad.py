"""`plot_valid_junctions` draws the bad lines in red, and no green group, when every judged line is bad.

Until 0.7.0 the drawing helper stacked the green group whatever its size, so a tension map under which every triple
line of the mesh is judged and found bad (one tension above the sum of the other two at every line) raised numpy's
`ValueError: need at least one array to concatenate` instead of showing the lines. The map used here,
`gamma_ij = 10 ** (i + j)` for labels `i < j`, makes every line bad: at a line `a < b < c`,
`gamma_bc = 10 ** (b + c) >= 10 * 10 ** (a + c) > gamma_ab + gamma_ac`. Pins that the red group is drawn and the empty
green group is not, on Polyscope's headless mock backend, and the mirror case (every line valid: green only).
Skipped when Polyscope is not installed (the optional `viewing` extra).
"""

from __future__ import annotations

import warnings

import pytest

from foambryo import dcel_mesh_from_file

from .conftest import require_substitute_mesh

ps = pytest.importorskip("polyscope")


@pytest.fixture
def drawn_curve_networks():
    """Polyscope on its mock backend; returns the names of the curve networks present when the window is shown."""
    real_init, real_show = ps.init, ps.show
    try:
        real_init("openGL_mock")
    except RuntimeError as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"Polyscope's mock backend is unavailable: {error}")
    seen: list[set[str]] = []
    names = ("valid trijunctions", "bad trijunctions", "trijunctions")

    def show(*_args: object, **_kwargs: object) -> None:
        seen.append({name for name in names if ps.has_curve_network(name)})
        real_show(forFrames=2)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ps, "init", lambda *_args, **_kwargs: real_init("openGL_mock"))
        patch.setattr(ps, "show", show)
        yield seen
    ps.remove_all_structures()


def every_line_bad(mesh) -> dict[tuple[int, int], float]:
    labels = sorted({int(label) for line in mesh.compute_length_trijunctions() for label in line})
    return {(i, j): 10.0 ** (i + j) for i in labels for j in labels if i < j}


def test_every_judged_line_bad_draws_the_red_group_only(drawn_curve_networks) -> None:
    from foambryo.viewing import plot_valid_junctions

    mesh = dcel_mesh_from_file(str(require_substitute_mesh()))
    tensions = every_line_bad(mesh)
    lines = mesh.compute_length_trijunctions()
    assert lines
    assert all(all(pair in tensions for pair in ((a, b), (a, c), (b, c))) for a, b, c in lines)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="plot_valid_junctions", category=UserWarning)
        plot_valid_junctions(mesh, tensions)
    assert drawn_curve_networks == [{"bad trijunctions"}]


def test_every_judged_line_valid_draws_the_green_lines_only(drawn_curve_networks) -> None:
    from foambryo.viewing import plot_valid_junctions

    mesh = dcel_mesh_from_file(str(require_substitute_mesh()))
    labels = sorted({int(label) for line in mesh.compute_length_trijunctions() for label in line})
    plot_valid_junctions(mesh, {(i, j): 1.0 for i in labels for j in labels if i < j})
    assert drawn_curve_networks == [{"trijunctions"}]
