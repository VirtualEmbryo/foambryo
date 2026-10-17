"""Triple lines with a wedge at 0 or 180 degrees in the methods whose coefficients are singular there.

`Cotan`, `InvCotan`, `Lami` and `LogLami` divide by, or take the logarithm of, a quantity that vanishes
when a wedge is exactly flat (180 degrees) or closed (0 degrees). On real meshes this happens on short
triple lines whose faces are exactly coplanar on the voxel lattice; on a 1 866-cell tissue it made all
four methods' systems non-finite, so `lsqr` ran to its iteration limit and the dense fallback ran on
`nan`. These tests pin the guard: such lines are left out with a `DegenerateWedgeWarning`, interfaces left
without an equation are `nan`, everything else is untouched, and the four methods never solve a
non-finite system.

The fixture is a row of unit cubes (cells 1, 2, ..., exterior 0). Between cubes `c` and `c + 1` the
triple line `(0, c, c + 1)` runs round the dividing square; its exterior wedge is exactly 180 degrees on
every edge (the two cells' outer faces are coplanar), and the two cell wedges are exactly 90 degrees.
The builders read each triple line's mean angle over its edges, so lifting a corner of an end face that a
line's triangles share, (n, 1, 1) for line (0, n - 1, n) and (0, 0, 1) for line (0, 1, 2), bends that
line's mean away from 180 degrees and leaves the other lines flat.
"""

import warnings

import numpy as np
import pytest

from foambryo import TensionComputationMethod, infer_tensions
from foambryo.dcel import DcelData
from foambryo.tension_inference import (
    DEGENERATE_WEDGE_TOLERANCE,
    DegenerateWedgeWarning,
    _is_degenerate_wedge,
    _without_degenerate_wedges,
)
from foambryo.wedge_refusal import line_averaged_angles

GUARDED = ("Cotan", "InvCotan", "Lami", "LogLami")
UNGUARDED = ("YoungDupre", "ProjectionYoungDupre", "InvLami", "YoungDupreLocal")


def row_of_cubes(n_cells: int, lifted: tuple[tuple[float, float, float], ...] = ()) -> DcelData:
    """`n_cells` unit cubes along x; each grid corner in `lifted` moved by (0, +-0.2, +-0.2), away from the box."""
    xs = [float(x) for x in range(n_cells + 1)]
    points = np.array([[x, y, z] for x in xs for y in (0.0, 1.0) for z in (0.0, 1.0)])
    index = {tuple(p): i for i, p in enumerate(points.tolist())}
    for corner in lifted:
        points[index[corner]] += [0.0, 0.2 if corner[1] else -0.2, 0.2 if corner[2] else -0.2]

    def square(corners, labels, towards):
        out = []
        for tri in ([corners[0], corners[1], corners[2]], [corners[0], corners[2], corners[3]]):
            p = points[[index[t] for t in tri]]
            if np.dot(np.cross(p[1] - p[0], p[2] - p[0]), np.asarray(towards) - p.mean(axis=0)) < 0:
                tri = [tri[0], tri[2], tri[1]]
            out.append([index[t] for t in tri] + list(labels))
        return out

    rows = []
    for cell in range(1, n_cells + 1):
        x0, x1 = float(cell - 1), float(cell)
        centre = (x0 + 0.5, 0.5, 0.5)
        for fixed in ((2, 0.0), (2, 1.0), (1, 0.0), (1, 1.0)):  # z = 0, z = 1, y = 0, y = 1
            axis, value = fixed
            other = 1 if axis == 2 else 2
            corners = []
            for a, b in ((x0, 0.0), (x1, 0.0), (x1, 1.0), (x0, 1.0)):
                p = [a, 0.0, 0.0]
                p[axis], p[other] = value, b
                corners.append(tuple(p))
            rows += square(corners, (0, cell), centre)
        if cell == 1:
            rows += square([(0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 1.0), (0.0, 0.0, 1.0)], (0, 1), centre)
        if cell == n_cells:
            rows += square([(x1, 0.0, 0.0), (x1, 1.0, 0.0), (x1, 1.0, 1.0), (x1, 0.0, 1.0)], (0, cell), centre)
        else:
            rows += square(
                [(x1, 0.0, 0.0), (x1, 1.0, 0.0), (x1, 1.0, 1.0), (x1, 0.0, 1.0)],
                (cell, cell + 1),
                (x1 + 0.5, 0.5, 0.5),
            )
    rows = np.array(rows, dtype=np.int64)
    return DcelData(points, rows[:, :3], rows[:, 3:])


def tensions(mesh, method):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = infer_tensions(mesh, mode=getattr(TensionComputationMethod, method))
    return result, [w for w in caught if issubclass(w.category, DegenerateWedgeWarning)]


def test_the_criterion_covers_rounding_at_both_ends():
    eps = np.finfo(float).eps
    for dot in (-1.0, -1.0 + eps / 2, -1.0 + eps, -1.0 + 2 * eps, 1.0, 1.0 - eps / 2, 1.0 - eps):
        assert _is_degenerate_wedge(float(np.arccos(dot)))
    assert _is_degenerate_wedge(float("nan"))
    assert not _is_degenerate_wedge(np.pi - 10 * DEGENERATE_WEDGE_TOLERANCE)
    assert not _is_degenerate_wedge(np.radians(0.001))


@pytest.mark.parametrize("method", GUARDED)
def test_a_flat_line_alone_leaves_every_interface_undetermined(method):
    result, caught = tensions(row_of_cubes(2), method)
    assert len(caught) == 1
    assert "1 triple line(s)" in str(caught[0].message)
    assert "3 interface(s)" in str(caught[0].message)
    assert all(np.isnan(v) for v in result.values())


@pytest.mark.parametrize("method", GUARDED)
def test_the_flat_line_is_left_out_and_the_rest_is_solved(method):
    result, caught = tensions(row_of_cubes(3, lifted=((3.0, 1.0, 1.0),)), method)
    assert len(caught) == 1
    assert "1 triple line(s)" in str(caught[0].message)
    assert "2 interface(s)" in str(caught[0].message)
    assert np.isnan(result[(0, 1)])
    assert np.isnan(result[(1, 2)])
    assert all(np.isfinite(result[key]) for key in ((0, 2), (0, 3), (2, 3)))


@pytest.mark.parametrize("method", UNGUARDED)
def test_methods_with_bounded_coefficients_are_not_guarded(method):
    result, caught = tensions(row_of_cubes(2), method)
    assert caught == []
    assert all(np.isfinite(v) for v in result.values())


@pytest.mark.parametrize("method", GUARDED)
def test_a_mesh_without_a_degenerate_wedge_is_untouched(method):
    mesh = row_of_cubes(3, lifted=((3.0, 1.0, 1.0), (0.0, 0.0, 1.0)))
    angles = line_averaged_angles(mesh)
    # the very same object comes back, so every coefficient is the one the unguarded builder writes
    assert _without_degenerate_wedges(mesh, angles, method) is angles
    result, caught = tensions(mesh, method)
    assert caught == []
    assert all(np.isfinite(v) for v in result.values())
