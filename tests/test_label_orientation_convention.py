"""Tests for the label-order / triangle-winding convention foambryo depends on.

Background (established while building the analytic phantoms, at the initial validation stage).
`foambryo.geometry.compute_volume_cells` accumulates `+det([v0, v1, v2])` into a
triangle's `material_1` and `-det(...)` into its `material_2`. By the divergence theorem
that sum is a cell's true volume only if every face of that cell is wound outward from
it — so a triangle's right-hand-rule normal must point away from `material_1` and into
`material_2`.

The point of these tests is that **sorted label columns are not that invariant**, and
checking sortedness would catch nothing:

- Every one of the 47 `benchmarking-dataset` `.rec` ground-truth meshes stores roughly
  half its rows with `material_1 > material_2`, and they are all valid, because their
  winding is tied to the first-listed label. `DcelData` normalises them by swapping the
  labels *and* reversing the winding, which is why they give correct volumes.
- Conversely, a mesh whose label columns are already sorted but whose triangles were
  never reoriented passes any `material_1 < material_2` assertion and still produces
  silently negative volumes.

So the label-order invariant is enforced structurally in `DcelData`
(`_normalize_label_order`, which also now rejects same-material-both-sides faces), and
the failure that actually bites is detected geometrically by
`foambryo.validation.mesh_io.require_label_orientation_convention`.
"""

import numpy as np
import pytest

from foambryo.dcel import DcelData, _normalize_label_order
from foambryo.validation import mesh_io
from tests.conftest import require_substitute_mesh

EXTERIOR = 0


@pytest.fixture(scope="module")
def ground_truth_mesh():
    """A real, correctly-wound multimaterial mesh: `(v, f, regions)`."""
    return mesh_io.read_rec(require_substitute_mesh())


# ---------------------------------------------------------------------------
# The structural invariant: DcelData normalises label order, and says so
# ---------------------------------------------------------------------------
def test_reference_mesh_really_does_have_unsorted_label_columns(ground_truth_mesh):
    """Guards the premise of every other test here: unsorted input is the normal case."""
    _, _, regions = ground_truth_mesh
    n_unsorted = int((regions[:, 0] > regions[:, 1]).sum())

    assert n_unsorted > 0, "premise broken: this mesh no longer exercises the unsorted path"
    assert n_unsorted < len(regions), "expected a mix of sorted and unsorted rows"


def test_dcel_construction_establishes_material_1_lt_material_2(ground_truth_mesh):
    v, f, regions = ground_truth_mesh
    mesh = DcelData(v, f.copy(), regions.copy())

    assert (mesh.f[:, 3] < mesh.f[:, 4]).all()
    for face in mesh.faces:
        assert face.material_1 < face.material_2


def test_normalize_label_order_flips_winding_with_the_labels():
    """Swapping the labels without reversing the winding would break the volume sum."""
    rows = np.array([[10, 11, 12, 5, 2]], dtype=np.int64)

    out = _normalize_label_order(rows.copy())

    # labels sorted, and the last two vertices exchanged (which reverses the normal)
    np.testing.assert_array_equal(out, np.array([[10, 12, 11, 2, 5]], dtype=np.int64))


def test_normalize_label_order_leaves_already_sorted_rows_untouched():
    rows = np.array([[10, 11, 12, 2, 5]], dtype=np.int64)

    np.testing.assert_array_equal(_normalize_label_order(rows.copy()), rows)


def test_dcel_rejects_same_material_on_both_sides(ground_truth_mesh):
    """A face with one material on both sides is not an interface, and used to pass silently.

    Its `+det` and `-det` contributions land in the same cell and cancel, and it registers
    a spurious `(m, m)` interface key.
    """
    v, f, regions = ground_truth_mesh
    f_bad = np.vstack([f, f[:1]])
    regions_bad = np.vstack([regions, np.array([[3, 3]])])

    with pytest.raises(ValueError, match="same material on both sides"):
        DcelData(v, f_bad, regions_bad)


# ---------------------------------------------------------------------------
# The geometric invariant: what sortedness cannot detect
# ---------------------------------------------------------------------------
def test_signed_cell_volumes_match_dcel_on_the_reference_mesh(ground_truth_mesh):
    """The raw-array volume must equal DcelData's, so it can be checked before construction."""
    v, f, regions = ground_truth_mesh

    raw = mesh_io.signed_cell_volumes(v, f, regions)
    dcel = DcelData(v, f.copy(), regions.copy()).compute_volumes_cells()

    assert set(raw) == {int(k) for k in dcel}
    for label, volume in dcel.items():
        assert raw[int(label)] == pytest.approx(float(volume), rel=1e-9, abs=1e-12)


def test_reference_mesh_passes_the_orientation_check(ground_truth_mesh):
    v, f, regions = ground_truth_mesh

    report = mesh_io.check_label_orientation_convention(v, f, regions, exterior=EXTERIOR)

    assert report["ok"]
    assert report["nonpositive_cells"] == []
    assert report["n_rows_label_unsorted"] > 0  # valid despite unsorted columns
    assert report["volumes"][EXTERIOR] < 0  # the unbounded exterior is wound inward
    for label, volume in report["volumes"].items():
        if label != EXTERIOR:
            assert volume > 0
    mesh_io.require_label_orientation_convention(v, f, regions, exterior=EXTERIOR)


def test_mis_ordered_mesh_is_caught_by_the_orientation_check(ground_truth_mesh):
    """The deliberately mis-ordered mesh: label columns sorted, triangles NOT reoriented.

    This is what a `.rec` from another source can look like. It satisfies
    `material_1 < material_2` everywhere, so no label-order assertion can see it, and it
    produces negative cell volumes.
    """
    v, f, regions = ground_truth_mesh
    regions_mis_ordered = np.sort(regions, axis=1)  # sort labels, leave winding alone

    # the label-order invariant holds, before and after DcelData's normalisation
    assert (regions_mis_ordered[:, 0] < regions_mis_ordered[:, 1]).all()
    mesh = DcelData(v, f.copy(), regions_mis_ordered.copy())
    assert (mesh.f[:, 3] < mesh.f[:, 4]).all()

    # ... and yet the volumes are wrong, which is exactly the silent failure
    volumes = mesh.compute_volumes_cells()
    n_negative = sum(1 for label, volume in volumes.items() if label != EXTERIOR and volume <= 0)
    assert n_negative > 0, "premise broken: mis-ordering no longer corrupts the volumes"

    report = mesh_io.check_label_orientation_convention(v, f, regions_mis_ordered, exterior=EXTERIOR)
    assert not report["ok"]
    assert report["n_nonpositive_cells"] == n_negative
    assert report["n_rows_label_unsorted"] == 0  # sorted, and still wrong

    with pytest.raises(ValueError, match="non-positive volume"):
        mesh_io.require_label_orientation_convention(v, f, regions_mis_ordered, exterior=EXTERIOR)


def test_orientation_check_catches_a_single_reversed_triangle(ground_truth_mesh):
    """A subtler corruption: swap one row's labels only, leaving its winding alone.

    Only that row's two cells are affected, so this checks the detector localises rather
    than merely reporting a global sign problem.
    """
    v, f, regions = ground_truth_mesh
    regions_one_flipped = regions.copy()
    regions_one_flipped[0] = regions_one_flipped[0, ::-1]

    good = mesh_io.signed_cell_volumes(v, f, regions)
    bad = mesh_io.signed_cell_volumes(v, f, regions_one_flipped)

    affected = {int(regions[0, 0]), int(regions[0, 1])}
    for label in good:
        if label in affected:
            assert bad[label] != pytest.approx(good[label], rel=1e-12)
        else:
            assert bad[label] == pytest.approx(good[label], rel=1e-12)
