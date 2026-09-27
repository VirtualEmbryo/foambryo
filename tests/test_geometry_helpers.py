"""Unit tests for pure-geometry helpers in foambryo.geometry (written at the initial validation stage).

Uses the substitute benchmark mesh (see tests/conftest.py) as a real, non-trivial input
rather than a synthetic DCEL, since building a DCEL by hand is more machinery than the
formulas being tested need. `compute_areas_faces` runs automatically on mesh construction
(DCEL.__init__ calls `self.compute_areas_faces()`).
"""

import pytest

from foambryo import dcel_mesh_from_file
from foambryo.geometry import compute_angles_tri, compute_areas_cells, compute_areas_interfaces
from tests.conftest import require_substitute_mesh


@pytest.fixture(scope="module")
def mesh():
    return dcel_mesh_from_file(str(require_substitute_mesh()))


def test_areas_interfaces_are_positive_and_match_cell_totals(mesh):
    interface_areas = compute_areas_interfaces(mesh)
    cell_areas = compute_areas_cells(mesh)

    assert all(area > 0 for area in interface_areas.values())

    # Every cell's total area must equal the sum of areas of interfaces touching it
    # (each face contributes its area to both adjacent materials, geometry.py:221-223).
    for cell, total_area in cell_areas.items():
        expected = sum(area for (a, b), area in interface_areas.items() if a == cell or b == cell)
        assert total_area == pytest.approx(expected, rel=1e-10)


def test_compute_angles_tri_unique_parameter_deduplicates(mesh):
    """`unique` was dead; **the per-edge Young-Dupre work gave it a meaning.** Updated, not deleted.

    The original version of this test pinned the dead behaviour: `unique=True` and
    `unique=False` differed only by a uniform factor 2, so the flag changed no result. That work
    implemented de-duplication for real — `unique=True` now counts each *geometric* edge once
    instead of once per incident half-edge — so the factor is 6, not 2. The mean angles are
    still identical, which is what keeps every production call site (all of which pass
    `unique=False`) bit-unchanged.

    `tests/test_per_edge_young_dupre.py` carries the rest of the resolution: that
    `sum(dict_length) / 3` is now genuinely the mesh's total junction length, and that
    `unique=False` still reproduces the historical accumulation.
    """
    _, angles_deg_unique, lengths_unique = compute_angles_tri(mesh, unique=True)
    _, angles_deg_not_unique, lengths_not_unique = compute_angles_tri(mesh, unique=False)

    assert set(angles_deg_unique) == set(angles_deg_not_unique)
    for key in angles_deg_unique:
        assert angles_deg_unique[key] == pytest.approx(angles_deg_not_unique[key], rel=1e-10)
        # 3 incident half-edges per geometric edge, times the historical 2x append.
        assert lengths_not_unique[key] == pytest.approx(6 * lengths_unique[key], rel=1e-10)


def test_compute_angles_tri_trijunction_keys_have_three_distinct_labels(mesh):
    _, angles_deg, _ = compute_angles_tri(mesh, unique=True)

    for a, b, c in angles_deg:
        assert len({a, b, c}) == 3


def test_compute_angles_tri_angles_are_valid_degrees(mesh):
    _, angles_deg, lengths = compute_angles_tri(mesh, unique=True)

    for key, angle in angles_deg.items():
        assert 0.0 <= angle <= 180.0
        assert lengths[key] > 0
