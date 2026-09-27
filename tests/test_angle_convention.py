"""The angle-convention work: pin the resolved angle convention of `foambryo.geometry.compute_angles_tri`.

The angle-convention work settled an open question, raised by a code review and by an observation
on the analytic phantoms during the initial validation stage: `compute_angles_tri` returns the
**interior wedge angle** of the shared (middle-key) region, it is **relabel-invariant**
(geometric, not label-order-dependent), and the label-swap heuristic (`geometry.py:397-403`)
exists precisely to return the wedge regardless of where the shared region sits in the label
order. The YD matrix builders consume `cos(that wedge)`, matching Neumann physics. There was
no defect; the phantom-stage "through-cell converges to the supplement theta" observation was an
artifact of a pre-fix `_orient_outward_from_lower_label` winding on the flat cell-cell disk,
already fixed during the initial validation stage. See `foambryo_benchmarks/run_angle_convention.py`.

These tests fail loudly if a future change reintroduces a label-dependent or supplement-valued
angle, which would silently flip signs on off-diagonal tension-matrix entries.
"""

import numpy as np
import pytest

from foambryo_benchmarks import phantoms as ph
from foambryo import dcel_mesh_from_file
from foambryo.dcel import DcelData
from foambryo.geometry import compute_angles_tri

from .conftest import DATASET_DIR, require_substitute_mesh

# Doublet trijunction keys and the physical wedge each names (region in the middle slot):
#   (1, 0, 2) -> medium wedge (region 0) = 2*theta;  (0, 1, 2)/(0, 2, 1) -> cell wedges = 180-theta
_MEDIUM_KEY = (1, 0, 2)
_CELL_KEYS = ((0, 1, 2), (0, 2, 1))


def _doublet_dcel(ratio: float, n_rings: int, perm: dict[int, int] | None = None) -> DcelData:
    v, f, regions = ph.doublet_mesh(ratio, n_rings=n_rings, n_phi=2 * n_rings)
    if perm is not None:
        regions = np.array([[perm[a], perm[b]] for a, b in regions], dtype=np.int64)
    return DcelData(v.copy(), f.copy(), regions.copy())


# =====================================================================
# 1. Relabelling invariance — the decisive experiment
# =====================================================================
@pytest.mark.parametrize(
    "perm",
    [
        {0: 0, 1: 2, 2: 1},   # swap the two cells (the literal 1,2 -> 2,1 relabelling)
        {0: 1, 1: 2, 2: 0},   # cyclic: medium becomes label 1
        {0: 2, 1: 0, 2: 1},   # medium becomes the *largest* label
        {0: 5, 1: 3, 2: 7},   # arbitrary, non-contiguous labels
    ],
)
def test_relabelling_does_not_change_measured_wedges(perm):
    """Renaming cells (same physical mesh) must not change any measured wedge angle.

    If it did, `compute_angles_tri`'s normal-flip heuristic would be label-order-dependent
    rather than geometric — a sign flip on off-diagonal tension entries for some junctions.
    """
    ratio, n_rings = 0.5, 24
    inv = {v: k for k, v in perm.items()}
    base = compute_angles_tri(_doublet_dcel(ratio, n_rings), unique=False)[1]
    relabelled = compute_angles_tri(_doublet_dcel(ratio, n_rings, perm), unique=False)[1]
    base_by_wedge = {key[1]: val for key, val in base.items()}
    relabelled_by_wedge = {inv[key[1]]: val for key, val in relabelled.items()}
    assert base_by_wedge.keys() == relabelled_by_wedge.keys()
    for wedge, angle in base_by_wedge.items():
        assert relabelled_by_wedge[wedge] == pytest.approx(angle, abs=1e-9)


# =====================================================================
# 2. Both phantoms reproduce their analytic wedges (not the supplement)
# =====================================================================
@pytest.mark.parametrize("ratio", [0.5, 1.0, 1.5])
def test_doublet_converges_to_true_wedge_not_supplement(ratio):
    """Through-medium -> 2*theta and through-cell -> 180-theta (the wedges), refining."""
    pred = ph.doublet_predicted_angles(ratio)
    wedge_medium = pred["angle_through_medium_deg"]
    wedge_cell = pred["angle_through_cell_deg"]
    supplement_cell = pred["theta_deg"]
    coarse = compute_angles_tri(_doublet_dcel(ratio, 16), unique=False)[1]
    fine = compute_angles_tri(_doublet_dcel(ratio, 64), unique=False)[1]
    # through-cell must sit near the true wedge, and refining must move it *toward* the wedge
    # and *away* from the supplement — the decisive discriminator (wedge and supplement differ
    # for every ratio in (0, 2): 180-theta vs theta are equal only at theta=90, i.e. ratio=0).
    assert abs(wedge_cell - supplement_cell) > 1.0
    assert abs(fine[_CELL_KEYS[0]] - wedge_cell) < abs(fine[_CELL_KEYS[0]] - supplement_cell)
    assert abs(fine[_CELL_KEYS[0]] - wedge_cell) < abs(coarse[_CELL_KEYS[0]] - wedge_cell)
    # Absolute proximity at n_rings=64 (worst residual discretisation ~2.2 deg at ratio 1.5);
    # abs=3.0 stays far below the wedge-vs-supplement gap, which is >>3 deg for every ratio.
    assert fine[_MEDIUM_KEY] == pytest.approx(wedge_medium, abs=3.0)
    assert fine[_CELL_KEYS[0]] == pytest.approx(wedge_cell, abs=3.0)
    assert fine[_CELL_KEYS[1]] == pytest.approx(wedge_cell, abs=3.0)


def test_tetrahedral_vertex_interior_junctions_are_120():
    """The physical (interior) Plateau trijunctions read exactly 120 degrees."""
    v, f, regions = ph.tetrahedral_vertex_mesh()
    deg = compute_angles_tri(DcelData(v.copy(), f.copy(), regions.copy()), unique=False)[1]
    interior = [val for key, val in deg.items() if set(key) <= {1, 2, 3, 4}]
    assert interior, "expected interior trijunctions among cells 1..4"
    for angle in interior:
        assert angle == pytest.approx(120.0, abs=1e-6)


# =====================================================================
# 3. The three wedges of a real junction sum to 360 (all wedges, no flip)
# =====================================================================
def test_real_mesh_wedges_sum_to_360():
    """On a real equilibrium mesh (000), each junction's 3 wedges sum to 360 degrees.

    A single supplement flip on any wedge would give |sum - 360| ~ 2*(90 - theta), tens of
    degrees; the measured worst is a fraction of a degree (real mesh non-flatness).
    """
    require_substitute_mesh()  # skips with reason if benchmarking-dataset/ is absent
    mesh = dcel_mesh_from_file(str(DATASET_DIR / "000_mesh.rec"))
    deg = compute_angles_tri(mesh, unique=False)[1]
    groups: dict[frozenset, list[float]] = {}
    for key, val in deg.items():
        groups.setdefault(frozenset(key), []).append(val)
    complete = [vals for vals in groups.values() if len(vals) == 3]
    assert complete, "expected complete trijunctions on 000_mesh.rec"
    worst = max(abs(sum(vals) - 360.0) for vals in complete)
    assert worst < 5.0  # comfortably below a supplement flip; measured ~1e-13 on 000


# =====================================================================
# 4. The mechanism: flipping only the flat-disk winding yields the supplement
# =====================================================================
def test_disk_flip_reproduces_the_supplement():
    """Reversing only the (1,2) flat-disk winding yields the supplement, not a geometry defect.

    Reversing only the (1,2) flat-disk winding (the orientation bug fixed during the initial
    validation stage) turns the through-cell reading into the supplement while leaving
    through-medium untouched — pinning that the phantom-stage observation was a
    phantom-orientation artifact, not a geometry defect.
    """
    ratio, n_rings = 0.5, 32
    pred = ph.doublet_predicted_angles(ratio)
    v, f, regions = ph.doublet_mesh(ratio, n_rings=n_rings, n_phi=2 * n_rings)
    ok = compute_angles_tri(DcelData(v.copy(), f.copy(), regions.copy()), unique=False)[1]
    f_bug = f.copy()
    disk = np.array([tuple(r) == (1, 2) for r in regions])
    f_bug[disk] = f_bug[disk][:, [0, 2, 1]]
    bug = compute_angles_tri(DcelData(v.copy(), f_bug, regions.copy()), unique=False)[1]
    # correct: near the true wedge; flipped: near the supplement; through-medium: unchanged
    assert abs(ok[_CELL_KEYS[0]] - pred["angle_through_cell_deg"]) < abs(ok[_CELL_KEYS[0]] - pred["theta_deg"])
    assert abs(bug[_CELL_KEYS[0]] - pred["theta_deg"]) < abs(bug[_CELL_KEYS[0]] - pred["angle_through_cell_deg"])
    assert bug[_MEDIUM_KEY] == pytest.approx(ok[_MEDIUM_KEY], abs=1e-9)
    assert ok[_CELL_KEYS[0]] + bug[_CELL_KEYS[0]] == pytest.approx(180.0, abs=1e-6)
