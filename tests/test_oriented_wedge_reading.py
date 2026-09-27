r"""The oriented wedge reading: what it pins, and the regression it exists to prevent.

`_wedges_at_half_edge` reads a trijunction wedge as `arccos` of a dot product between two
incident face normals. `arccos` has range `[0, pi]`, so a wedge whose interior angle exceeds a
straight angle -- a **reflex** wedge -- comes back as `360 - theta`, and the three wedges of that
junction edge sum to `360 - 2 (theta - 180)` rather than to 360. Every average along the line
inherits the deficit, and the deficit is not rare: on reconstructed benchmark meshes 2 % to 25 %
of trijunctional edges carry a reflex wedge, at a rate that differs by a factor of six between
reconstruction strategies. Two meshes compared through the unoriented reading are therefore
partly being compared on how much each one was clipped.

`"oriented_chord"` and `"oriented_quadric"` return the wedge in `[0, 360)`. The regressions
pinned here, in the order they matter:

1. **Identity where `arccos` is already right.** The oriented readings are *bitwise* identical to
   their unoriented counterparts on every wedge that is not reflex -- including every wedge of
   every ground-truth benchmark mesh, none of which has a reflex edge. A reading that changed a
   correct number would be a regression however good its reflex handling.
2. **The three wedges of an edge sum to a full turn**, which is the invariant
   `compute_angles_tri`'s docstring has always claimed and only the oriented reading makes true.
3. **The angles do not depend on how the mesh is numbered**: not on the direction of the junction
   edge, not on which half-edge is visited, not on the material labels.
4. **A known reflex wedge is read back exactly**, on a phantom whose three wedges are analytic.
5. **The default is untouched.** `"chord"` remains the default of every entry point and is not
   reachable by accident from the new selectors.
"""

import itertools

import numpy as np
import pytest

from foambryo import TensionComputationMethod, infer_tensions
from foambryo.dcel import DcelData
from foambryo.geometry import compute_angles_tri, compute_angles_tri_with_report, compute_trijunction_edges_with_report
from foambryo.io import dcel_mesh_from_file
from foambryo.second_order_angle_reading import (
    ANGLE_READINGS,
    CHORD_READING,
    ORIENTED_CHORD_READING,
    ORIENTED_QUADRIC_READING,
    QUADRIC_READING,
    QUADRIC_WINDOW_FACTOR,
    REFLEX_TOLERANCE_DEG,
    fits_quadric_tangents,
    reads_oriented_wedges,
)
from tests.conftest import DATASET_DIR, require_substitute_mesh

#: A junction edge's three wedges must sum to this, in degrees.
FULL_TURN_DEG = 360.0
#: Tolerance on that sum. The oriented reading takes its magnitude from `arccos` and only the
#: branch from the cross product, so the sum is exact analytically and holds to rounding.
SUM_TOLERANCE_DEG = 1e-9
#: Tolerance on a multiset of angles compared with itself after a relabelling of the mesh.
SET_TOLERANCE_DEG = 1e-12
#: The three film directions of the reflex phantom, in degrees about the junction axis. The
#: interior wedge of region `k` is the gap from ray `k` to ray `k + 1`, so these enclose 100, 60
#: and **200** degrees: region 2 is reflex.
REFLEX_PHANTOM_RAYS_DEG = (0.0, 100.0, 160.0)
#: `material -> its interior wedge, degrees` for `REFLEX_PHANTOM_RAYS_DEG`.
REFLEX_PHANTOM_WEDGES_DEG = {0: 100.0, 1: 60.0, 2: 200.0}
#: The one reflex wedge of that phantom.
REFLEX_PHANTOM_REFLEX_DEG = 200.0
#: MEASURED on the 47 in-dataset simulator meshes: which cases carry a reflex junction edge, and
#: how many. Five edges out of 14 183 -- the reference geometry is nearly, not exactly, free of
#: them, and "nearly" is what is pinned. See
#: `test_the_ground_truth_meshes_carry_the_recorded_handful_of_reflex_edges`.
GROUND_TRUTH_REFLEX_EDGES_BY_CASE = {"004": 2, "020": 2, "045": 1}
GROUND_TRUTH_REFLEX_WEDGES = sum(GROUND_TRUTH_REFLEX_EDGES_BY_CASE.values())


# ---------------------------------------------------------------------------------------
# Phantoms with analytically known wedges
# ---------------------------------------------------------------------------------------
def _three_film_phantom(rays_deg=(0.0, 120.0, 240.0), rings: int = 6, n_axis: int = 9) -> DcelData:
    """Three planar films leaving one straight junction line at prescribed in-plane angles.

    The junction runs along `+z`; film `k` is the half-plane leaving it in the direction
    `rays_deg[k]`, measured counter-clockwise about `+z` from `+x`. Region `k` is the sector
    between rays `k` and `k + 1`, and carries material `k`, so **the interior wedge of region
    `k` is exactly the gap between those two rays**: the three wedge angles are known
    analytically, and one of them is made reflex simply by spacing the rays unevenly. Exactly
    flat everywhere, so the quadric reading has nothing to correct either.

    **The winding, which is the whole reason this builder exists rather than reusing
    `test_second_order_angle_reading._flat_three_film_phantom`.** foambryo's convention is that
    a triangle's right-hand-rule normal points *away from* `material_1` and *into* `material_2`
    (`tests/test_label_orientation_convention.py`). Each strip here is wound so that its normal
    is the ray direction turned a quarter turn clockwise about `+z`, i.e. into the sector
    *before* the ray, so film `k` must be labelled `(material of sector k, material of sector
    k - 1)`. Labelling all three films `(0, 1), (0, 2), (1, 2)` with one uniform winding — as the
    older phantom does — reverses one film's normal and reads the symmetric junction as
    60/60/120 instead of 120/120/120, which is invisible to a test that only compares two
    readings with each other. `test_the_phantom_reads_a_symmetric_junction_at_120_degrees`
    guards this.

    Args:
        rays_deg (tuple[float, ...], optional): the three in-plane film directions, degrees,
            strictly increasing in `[0, 360)`. Defaults to the symmetric 120-degree junction.
        rings (int, optional): triangle rings per film away from the junction. Defaults to 6.
        n_axis (int, optional): vertices along the junction line itself. Defaults to 9.

    Returns:
        DcelData: the phantom mesh.
    """
    axis_positions = np.linspace(0.0, 40.0, n_axis)
    directions = [np.array([np.cos(np.radians(a)), np.sin(np.radians(a))]) for a in rays_deg]
    spacing = 6.0
    junction = np.stack([np.zeros(n_axis), np.zeros(n_axis), axis_positions], axis=1)
    points = [junction]
    offset = n_axis
    film_columns = []
    for direction in directions:
        columns = []
        for ring in range(1, rings + 1):
            radius = ring * spacing
            points.append(
                np.stack(
                    [np.full(n_axis, radius * direction[0]), np.full(n_axis, radius * direction[1]), axis_positions],
                    axis=1,
                ),
            )
            columns.append(np.arange(offset, offset + n_axis))
            offset += n_axis
        film_columns.append(columns)

    # film k separates sector k - 1 (clockwise of the ray, where its normal points) from
    # sector k, so its label pair is (material of sector k, material of sector k - 1).
    pairs = [(k, (k - 1) % 3) for k in range(3)]
    triangles, labels = [], []
    for columns, pair in zip(film_columns, pairs, strict=True):
        strips = [np.arange(n_axis), *columns]
        for left, right in itertools.pairwise(strips):
            for i in range(n_axis - 1):
                triangles.append([left[i], right[i], left[i + 1]])
                labels.append(list(pair))
                triangles.append([right[i], right[i + 1], left[i + 1]])
                labels.append(list(pair))
    return DcelData(
        np.concatenate(points, axis=0),
        np.asarray(triangles, dtype=np.int64),
        np.asarray(labels, dtype=np.int64),
    )


def _wedge_angles_deg(mesh: DcelData, reading: str) -> list[tuple[tuple[int, int], tuple[float, ...]]]:
    """`(edge vertices, three wedge angles in degrees)` per geometric trijunctional edge."""
    edges, _report = compute_trijunction_edges_with_report(mesh, reading=reading)
    return [(edge.vertices, tuple(float(np.degrees(a)) for a in edge.angles)) for edge in edges]


def _benchmark_meshes(limit: int | None = None):
    """The in-dataset ground-truth meshes, or a skip when the sibling dataset is absent."""
    if not DATASET_DIR.exists():
        pytest.skip(f"{DATASET_DIR} not found (sibling benchmarking-dataset/ missing)")
    paths = sorted(DATASET_DIR.glob("*_mesh.rec"))
    assert len(paths) == 47, f"expected 47 reference meshes, found {len(paths)}"
    return paths[:limit] if limit else paths


# ---------------------------------------------------------------------------------------
# 1 -- a known reflex wedge, read back exactly
# ---------------------------------------------------------------------------------------
def test_the_phantom_reads_a_symmetric_junction_at_120_degrees():
    """The phantom's own correctness check, and the reason it is not the older one.

    A three-film junction with rays 120 degrees apart has three 120-degree wedges. If the
    builder's winding did not match foambryo's label-order convention one film's normal would be
    reversed and this would read 60/60/120 — a reversal invisible to any test that only compares
    two readings with each other. Everything else in this module is measured on this phantom, so
    it is checked against its analytic answer first.
    """
    mesh = _three_film_phantom()
    _rad, degrees, _length = compute_angles_tri(mesh, unique=True)
    assert len(degrees) == 3
    for key, angle in degrees.items():
        assert angle == pytest.approx(120.0, abs=1e-9), key


def test_the_oriented_reading_recovers_an_analytically_known_reflex_wedge():
    """Rays at 0, 100 and 160 degrees enclose wedges of 100, 60 and 200 degrees.

    The unoriented reading must return `360 - 200 = 160` for the third and a sum of 320; the
    oriented reading must return 200 and a sum of 360. This is the whole defect and the whole
    fix, on a mesh whose answer is known without any reference geometry.
    """
    mesh = _three_film_phantom(REFLEX_PHANTOM_RAYS_DEG)

    per_material = compute_angles_tri(mesh, unique=True, reading=ORIENTED_CHORD_READING)[1]
    assert len(per_material) == 3
    for key, angle in per_material.items():
        assert angle == pytest.approx(REFLEX_PHANTOM_WEDGES_DEG[int(key[1])], abs=1e-9), key

    unoriented = _wedge_angles_deg(mesh, CHORD_READING)
    oriented = _wedge_angles_deg(mesh, ORIENTED_CHORD_READING)
    assert unoriented
    assert len(unoriented) == len(oriented)

    wedges = list(REFLEX_PHANTOM_WEDGES_DEG.values())
    expected_unoriented = sorted(min(w, FULL_TURN_DEG - w) for w in wedges)
    expected_oriented = sorted(wedges)
    for (_vertices, angles_u), (_v2, angles_o) in zip(unoriented, oriented, strict=True):
        assert sorted(angles_u) == pytest.approx(expected_unoriented, abs=1e-9)
        assert sorted(angles_o) == pytest.approx(expected_oriented, abs=1e-9)
        assert sum(angles_u) == pytest.approx(FULL_TURN_DEG - 2 * (REFLEX_PHANTOM_REFLEX_DEG - 180.0), abs=1e-9)
        assert sum(angles_o) == pytest.approx(FULL_TURN_DEG, abs=SUM_TOLERANCE_DEG)


def test_the_two_readings_agree_on_a_junction_with_no_reflex_wedge():
    """The symmetric 120-degree phantom has no reflex wedge, so the oriented reading is inert."""
    mesh = _three_film_phantom()
    assert _wedge_angles_deg(mesh, CHORD_READING) == _wedge_angles_deg(mesh, ORIENTED_CHORD_READING)


# ---------------------------------------------------------------------------------------
# 2 -- identity where `arccos` is already right, bitwise
# ---------------------------------------------------------------------------------------
def test_the_oriented_chord_reading_is_bitwise_identical_on_every_non_reflex_wedge():
    """Across all 47 reference meshes, only the five reflex wedges move, and they move exactly.

    Bitwise, not approximately: the oriented reading takes its magnitude from the unchanged
    `arccos` expression and uses the cross product only to choose a branch, so where the branch
    is not taken it returns the identical float. Both entry points are checked -- the per-edge
    one and the line-averaged one, including the accumulated lengths, which no reading may move.

    The line-averaged dicts are compared only on the lines carrying no reflex edge; a line that
    carries one *must* move, and that is the point of the reading.
    """
    n_moved = 0
    for path in _benchmark_meshes():
        mesh = dcel_mesh_from_file(str(path))
        unoriented = _wedge_angles_deg(mesh, CHORD_READING)
        oriented = _wedge_angles_deg(mesh, ORIENTED_CHORD_READING)
        moved_lines = set()
        for (vertices, angles_u), (_v2, angles_o) in zip(unoriented, oriented, strict=True):
            for unoriented_angle, oriented_angle in zip(angles_u, angles_o, strict=True):
                if unoriented_angle != oriented_angle:
                    n_moved += 1
                    moved_lines.add(vertices)
                    assert oriented_angle > 180.0, (path.name, vertices)

        chord_rad, chord_deg, chord_len, _r = compute_angles_tri_with_report(mesh, unique=False)
        oriented_rad, oriented_deg, oriented_len, _r2 = compute_angles_tri_with_report(
            mesh,
            unique=False,
            reading=ORIENTED_CHORD_READING,
        )
        assert chord_len == oriented_len, f"{path.name}: no reading may move an accumulated length"
        assert chord_rad.keys() == oriented_rad.keys(), path.name
        if not moved_lines:
            assert chord_rad == oriented_rad, path.name
            assert chord_deg == oriented_deg, path.name
    assert n_moved == GROUND_TRUTH_REFLEX_WEDGES, (
        f"{n_moved} ground-truth wedges moved, expected {GROUND_TRUTH_REFLEX_WEDGES}"
    )


def test_the_ground_truth_meshes_carry_the_recorded_handful_of_reflex_edges():
    """Nearly free of reflex edges is not the same as free of them, and the difference is pinned.

    MEASURED here, on all 47 in-dataset simulator meshes: **5 reflex junction edges out of
    14 183**, on cases `004`, `020` and `045`, the widest being 184.83 degrees. So a converged
    simulator mesh at its own spacing is essentially, but not exactly, free of the defect --
    against 2-25 % of edges on the reconstructed meshes. The count is pinned rather than
    asserted to be zero because zero would be false, and because a reconstruction change that
    raised it on the *reference* geometry would mean the reference had moved.
    """
    found = {}
    for path in _benchmark_meshes():
        mesh = dcel_mesh_from_file(str(path))
        angles = [a for _v, wedges in _wedge_angles_deg(mesh, ORIENTED_CHORD_READING) for a in wedges]
        reflex = [a for a in angles if a > 180.0 + REFLEX_TOLERANCE_DEG]
        if reflex:
            found[path.stem.removesuffix("_mesh")] = len(reflex)
    assert found == GROUND_TRUTH_REFLEX_EDGES_BY_CASE


def test_only_reflex_wedges_differ_between_the_two_readings():
    """Wherever the readings disagree the oriented angle is exactly the unoriented one reflected.

    Checked on the reflex phantom, where a disagreement certainly exists, and on the reference
    meshes, where none does. A reading that moved a non-reflex wedge -- or that moved a reflex
    one by anything other than `360 - theta` -- fails here.
    """
    meshes = [_three_film_phantom(REFLEX_PHANTOM_RAYS_DEG)]
    meshes += [dcel_mesh_from_file(str(p)) for p in _benchmark_meshes(limit=4)]
    n_reflex_seen = 0
    for mesh in meshes:
        for (_v, angles_u), (_v2, angles_o) in zip(
            _wedge_angles_deg(mesh, CHORD_READING),
            _wedge_angles_deg(mesh, ORIENTED_CHORD_READING),
            strict=True,
        ):
            for unoriented, oriented in zip(angles_u, angles_o, strict=True):
                if unoriented == oriented:
                    continue
                n_reflex_seen += 1
                assert oriented > 180.0
                assert oriented == pytest.approx(FULL_TURN_DEG - unoriented, abs=1e-12)
    assert n_reflex_seen > 0, "the phantom must contribute at least one reflex wedge"


# ---------------------------------------------------------------------------------------
# 3 -- the sum-to-a-full-turn invariant
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("reading", [ORIENTED_CHORD_READING, ORIENTED_QUADRIC_READING])
def test_every_junction_edge_sums_to_a_full_turn(reading):
    """The invariant `compute_angles_tri`'s docstring claims, made true, on real meshes.

    Both oriented readings, including the quadric one -- rotating normals about the junction
    edge keeps them coplanar, so the invariant is not a property of the chord reading alone.
    """
    for mesh in [
        _three_film_phantom(REFLEX_PHANTOM_RAYS_DEG),
        *[dcel_mesh_from_file(str(p)) for p in _benchmark_meshes(limit=4)],
    ]:
        for vertices, angles in _wedge_angles_deg(mesh, reading):
            assert sum(angles) == pytest.approx(FULL_TURN_DEG, abs=SUM_TOLERANCE_DEG), vertices


def test_the_unoriented_readings_break_that_sum_on_a_reflex_edge():
    """The control for the test above: the defect is present, and it is what is claimed.

    Without this a passing sum test could mean the invariant holds either way, and the oriented
    reading would be pinning nothing.
    """
    mesh = _three_film_phantom(REFLEX_PHANTOM_RAYS_DEG)
    deficits = [FULL_TURN_DEG - sum(angles) for _v, angles in _wedge_angles_deg(mesh, CHORD_READING)]
    assert deficits
    assert min(deficits) == pytest.approx(2 * (REFLEX_PHANTOM_REFLEX_DEG - 180.0), abs=1e-9)


# ---------------------------------------------------------------------------------------
# 4 -- independence of how the mesh is numbered
# ---------------------------------------------------------------------------------------
def _multiset(mesh: DcelData, reading: str) -> list[float]:
    return sorted(angle for _v, angles in _wedge_angles_deg(mesh, reading) for angle in angles)


def test_reversing_the_vertex_numbering_does_not_change_the_angles():
    """The orienting axis runs from the lower to the higher vertex key, so this reverses it.

    On every junction edge. If the axis sign were asserted rather than resolved from the
    geometry, each reflex wedge would swap with its complement and this multiset would change.
    """
    for mesh in [
        _three_film_phantom(REFLEX_PHANTOM_RAYS_DEG),
        *[dcel_mesh_from_file(str(p)) for p in _benchmark_meshes(limit=3)],
    ]:
        n = len(mesh.v)
        renumbered = DcelData(mesh.v[::-1].copy(), (n - 1 - mesh.f[:, :3]).copy(), mesh.f[:, 3:5].copy())
        assert _multiset(renumbered, ORIENTED_CHORD_READING) == pytest.approx(
            _multiset(mesh, ORIENTED_CHORD_READING),
            abs=SET_TOLERANCE_DEG,
        )


def test_permuting_the_material_labels_does_not_change_the_angles():
    """Relabelling the regions permutes the wedge *keys*; it may not move a wedge *angle*."""
    mesh = _three_film_phantom(REFLEX_PHANTOM_RAYS_DEG)
    before = _multiset(mesh, ORIENTED_CHORD_READING)
    for permutation in ({0: 0, 1: 2, 2: 1}, {0: 2, 1: 0, 2: 1}):
        relabelled = DcelData(
            mesh.v.copy(),
            mesh.f[:, :3].copy(),
            np.vectorize(permutation.get)(mesh.f[:, 3:5]).astype(np.int64),
        )
        assert _multiset(relabelled, ORIENTED_CHORD_READING) == pytest.approx(before, abs=SET_TOLERANCE_DEG)


def test_all_three_half_edges_of_a_geometric_edge_read_the_same_wedges():
    """`compute_angles_tri` visits a geometric edge once per half-edge when `unique=False`.

    The orienting axis is taken from the edge's vertex *keys* precisely so those three visits
    cannot disagree; if it were taken from the visiting half-edge's own direction they would.
    """
    mesh = _three_film_phantom(REFLEX_PHANTOM_RAYS_DEG)
    once = compute_angles_tri(mesh, unique=True, reading=ORIENTED_CHORD_READING)[1]
    thrice = compute_angles_tri(mesh, unique=False, reading=ORIENTED_CHORD_READING)[1]
    assert once.keys() == thrice.keys()
    for key in once:
        assert once[key] == pytest.approx(thrice[key], abs=SET_TOLERANCE_DEG)


# ---------------------------------------------------------------------------------------
# 5 -- the reflex flag agrees with the angle-sum deficit
# ---------------------------------------------------------------------------------------
def test_a_reflex_wedge_is_exactly_an_edge_with_an_angle_sum_deficit():
    """Two independent ways of finding a reflex edge, which must select the same edges.

    The oriented reading finds one by the sign of an in-plane cross product; the unoriented
    reading betrays one by its three angles falling short of a full turn. They must agree, and
    the reflex wedge must be the one the unoriented reading reads *largest* -- because the other
    two sum to exactly that value.
    """
    for mesh in [
        _three_film_phantom(REFLEX_PHANTOM_RAYS_DEG),
        *[dcel_mesh_from_file(str(p)) for p in _benchmark_meshes(limit=4)],
    ]:
        for (vertices, angles_u), (_v2, angles_o) in zip(
            _wedge_angles_deg(mesh, CHORD_READING),
            _wedge_angles_deg(mesh, ORIENTED_CHORD_READING),
            strict=True,
        ):
            by_deficit = (FULL_TURN_DEG - sum(angles_u)) > REFLEX_TOLERANCE_DEG
            reflex_positions = [i for i, a in enumerate(angles_o) if a > 180.0 + REFLEX_TOLERANCE_DEG]
            assert bool(reflex_positions) == by_deficit, vertices
            assert len(reflex_positions) <= 1, vertices
            if reflex_positions:
                assert angles_u[reflex_positions[0]] == max(angles_u), vertices


# ---------------------------------------------------------------------------------------
# 6 -- the selector, the default, and the refusals
# ---------------------------------------------------------------------------------------
def test_the_selector_traits_are_declared_for_every_reading():
    """No selector may acquire a default answer to "does it fit?" or "is it oriented?"."""
    assert ANGLE_READINGS == (CHORD_READING, QUADRIC_READING, ORIENTED_CHORD_READING, ORIENTED_QUADRIC_READING)
    assert [fits_quadric_tangents(r) for r in ANGLE_READINGS] == [False, True, False, True]
    assert [reads_oriented_wedges(r) for r in ANGLE_READINGS] == [False, False, True, True]
    for bad in ("oriented", "reflex", "chord_oriented", ""):
        with pytest.raises(ValueError, match="unknown angle reading"):
            reads_oriented_wedges(bad)


def test_the_default_is_still_the_unoriented_chord_reading():
    """Adding the oriented selectors may not make any entry point oriented by default."""
    mesh = _three_film_phantom(REFLEX_PHANTOM_RAYS_DEG)
    assert compute_angles_tri(mesh, unique=False) == compute_angles_tri(mesh, unique=False, reading=CHORD_READING)
    oriented = compute_angles_tri(mesh, unique=False, reading=ORIENTED_CHORD_READING)
    # The phantom is reflex, so the two readings must differ -- otherwise this test is vacuous.
    assert compute_angles_tri(mesh, unique=False)[1] != oriented[1]


def test_the_oriented_quadric_reading_reports_itself_and_keeps_the_window():
    """The coverage report must name the reading actually used, not the one it shares a fit with.

    The rotations are cached per mesh and shared with `"quadric"`, so the only thing that may
    differ between the two reports is the label -- and it must.
    """
    mesh = _three_film_phantom(REFLEX_PHANTOM_RAYS_DEG)
    _a, _d, _l, quadric = compute_angles_tri_with_report(mesh, unique=False, reading=QUADRIC_READING)
    _a2, _d2, _l2, oriented = compute_angles_tri_with_report(mesh, unique=False, reading=ORIENTED_QUADRIC_READING)
    assert quadric.reading == QUADRIC_READING
    assert oriented.reading == ORIENTED_QUADRIC_READING
    assert oriented.n_incident_faces == quadric.n_incident_faces
    assert oriented.n_corrected == quadric.n_corrected
    assert oriented.window_factor == QUADRIC_WINDOW_FACTOR


def test_young_dupre_local_refuses_the_oriented_quadric_and_accepts_the_oriented_chord():
    """The refusal is a property of the quadric tangent fit, so it carries to its oriented form.

    The per-edge balance never performs the along-line average that cancels the quadric
    correction's per-face scatter; reorienting a wedge does not change that, so the same
    combination is refused. `"oriented_chord"` has no scatter to eat and is accepted.
    """
    mesh = dcel_mesh_from_file(str(require_substitute_mesh()))
    with pytest.raises(ValueError, match="cannot be combined with"):
        infer_tensions(mesh, 1.0, TensionComputationMethod.YoungDupreLocal, angle_reading=ORIENTED_QUADRIC_READING)
    tensions = infer_tensions(mesh, 1.0, TensionComputationMethod.YoungDupreLocal, angle_reading=ORIENTED_CHORD_READING)
    assert tensions
    assert all(np.isfinite(v) for v in tensions.values())


def test_a_non_angle_method_refuses_the_oriented_readings():
    """`ForceBalance*` and `Variational` never read a wedge, so an oriented reading is a lie."""
    mesh = dcel_mesh_from_file(str(require_substitute_mesh()))
    for name in ("ForceBalance", "ForceBalanceMAP", "Variational"):
        mode = getattr(TensionComputationMethod, name)
        for reading in (ORIENTED_CHORD_READING, ORIENTED_QUADRIC_READING):
            with pytest.raises(ValueError, match="does not build its rows from measured"):
                infer_tensions(mesh, 1.0, mode, angle_reading=reading)


def test_the_oriented_reading_reads_nothing_but_the_mesh():
    """Safeguard 1, operationally: two meshes that are equal as arrays read equal angles.

    `compute_angles_tri_with_report` takes a `DcelData` and a selector string and nothing else,
    so there is no parameter through which a reference mesh, a tension or a target angle could
    reach the reader; this pins that a freshly constructed copy, carrying no cached state and no
    provenance, reads the same numbers.
    """
    mesh = _three_film_phantom(REFLEX_PHANTOM_RAYS_DEG)
    copy = DcelData(mesh.v.copy(), mesh.f[:, :3].copy(), mesh.f[:, 3:5].copy())
    assert _wedge_angles_deg(copy, ORIENTED_CHORD_READING) == _wedge_angles_deg(mesh, ORIENTED_CHORD_READING)
