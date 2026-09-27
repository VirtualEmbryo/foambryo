"""Tests for the analytic phantoms (written at the package's initial validation stage).

Every claim in `foambryo_benchmarks/phantoms.py`'s module docstring is checked here against an
independent computation (direct point-classification or a from-scratch symmetry
argument), not merely asserted — per the rule: "do not assert a contact-angle
formula you cannot verify."
"""

import numpy as np
import pytest

from foambryo_benchmarks import phantoms as ph
from foambryo import PressureComputationMethod
from foambryo.dcel import DcelData
from foambryo.pressure_inference import infer_pressures
from foambryo.tension_inference import infer_tensions
from foambryo.validation import mesh_io


# =====================================================================
# Doublet
# =====================================================================
def test_doublet_ratio_one_gives_120_degrees_through_medium():
    """Required limit check: gamma_cc/gamma_cm = 1 must give 120 degrees.

    That is Plateau's equal-tension Neumann triangle.
    """
    angles = ph.doublet_predicted_angles(1.0)
    assert angles["angle_through_medium_deg"] == pytest.approx(120.0)
    assert angles["angle_through_cell_deg"] == pytest.approx(120.0)


def test_doublet_predicted_angles_domain_validation():
    with pytest.raises(ValueError, match="physical domain"):
        ph.doublet_predicted_angles(2.5)
    with pytest.raises(ValueError, match="physical domain"):
        ph.doublet_predicted_angles(0.0)


@pytest.mark.parametrize("ratio", [0.3, 0.7, 1.0, 1.3, 1.8])
def test_doublet_predicted_angles_match_direct_point_classification(ratio):
    """Independent check of `doublet_predicted_angles`, bypassing the mesh entirely.

    Sweeps a fine ring of points around the rim and finds where the label changes.
    """
    predicted = ph.doublet_predicted_angles(ratio)
    r0 = 1.0
    eps = 1e-5
    n = 200_000
    angle_deg = np.linspace(0, 360, n, endpoint=False)
    angle_rad = np.radians(angle_deg)
    points = np.stack(
        [
            r0 + eps * np.cos(angle_rad),
            np.zeros(n),
            eps * np.sin(angle_rad),
        ],
        axis=1,
    )
    labels = ph.doublet_label_at(points, ratio, r0=r0)

    # medium (0) occupies angles in (-theta, +theta) i.e. (360-theta, 360) U (0, theta)
    is_medium = labels == 0
    medium_width = is_medium.sum() * (360.0 / n)
    assert medium_width == pytest.approx(predicted["angle_through_medium_deg"], abs=0.2)

    is_a = labels == 1
    a_width = is_a.sum() * (360.0 / n)
    assert a_width == pytest.approx(predicted["angle_through_cell_deg"], abs=0.2)

    is_b = labels == 2
    b_width = is_b.sum() * (360.0 / n)
    assert b_width == pytest.approx(predicted["angle_through_cell_deg"], abs=0.2)

    assert medium_width + a_width + b_width == pytest.approx(360.0, abs=0.5)


@pytest.mark.parametrize("ratio", [0.5, 1.0, 1.5])
def test_doublet_mesh_is_a_valid_closed_multimaterial_mesh(ratio):
    v, f, regions = ph.doublet_mesh(ratio, n_rings=10, n_phi=20)
    assert mesh_io.is_intact(v, f)
    assert mesh_io.list_interfaces(regions) == [(0, 1), (0, 2), (1, 2)]

    mesh = DcelData(v, f, regions)
    volumes = mesh.compute_volumes_cells()
    assert volumes[1] > 0
    assert volumes[2] > 0
    assert volumes[1] == pytest.approx(volumes[2], rel=1e-9)  # symmetric doublet

    junctions, _ = mesh_io.list_junctions(f, regions)
    assert junctions == [(0, 1, 2)]  # exactly one triple line


def test_doublet_mesh_through_medium_angle_converges_to_prediction():
    """The through-medium angle should approach the predicted wedge as resolution rises.

    This is the module docstring's verified claim about
    `foambryo.geometry.compute_angles_tri`. The through-cell entries are deliberately
    not checked here — see that docstring's convention note.
    """
    ratio = 0.5
    predicted = ph.doublet_predicted_angles(ratio)["angle_through_medium_deg"]

    errors = []
    for n_rings in (12, 48):
        v, f, regions = ph.doublet_mesh(ratio, n_rings=n_rings, n_phi=4 * n_rings)
        mesh = DcelData(v, f, regions)
        _, angles_deg, _ = mesh.compute_angles_tri(unique=False)
        through_medium = angles_deg[(1, 0, 2)]
        errors.append(abs(through_medium - predicted))

    assert errors[1] < errors[0]  # error shrinks as resolution increases
    assert errors[1] < errors[0] / 1.5  # by roughly the expected O(h) factor
    assert errors[1] < 3.0  # and is small at the finer (but still test-fast) resolution


def test_doublet_ground_truth_tensions_match_mesh_regions():
    ratio = 0.7
    _, _, regions = ph.doublet_mesh(ratio, n_rings=6, n_phi=12)
    tensions = ph.doublet_ground_truth_tensions(ratio)
    assert set(tensions) == set(mesh_io.list_interfaces(regions))
    assert tensions[(1, 2)] == pytest.approx(ratio * tensions[(0, 1)])


# ---------------------------------------------------------------------
# Doublet — Young-Laplace pressure ground truth
# ---------------------------------------------------------------------
@pytest.mark.parametrize("ratio", [0.5, 1.0, 1.5])
def test_doublet_pressures_are_two_gamma_over_the_cap_radius(ratio):
    """`Delta_P = 2 * gamma / R` per spherical cap, against the geometry's own radius."""
    _, radius, _ = ph._doublet_geometry(ratio, r0=1.0)
    pressures = ph.doublet_ground_truth_pressures(ratio, gamma_cm=1.0, r0=1.0)

    assert pressures[0] == 0.0  # exterior reference, foambryo's base_pressure convention
    assert pressures[1] == pytest.approx(2.0 / radius)
    assert pressures[2] == pytest.approx(2.0 / radius)


def test_doublet_pressures_satisfy_young_laplace_on_the_flat_interface():
    """The over-determined consistency check that makes this a real equilibrium.

    The A-B interface is flat, so Young-Laplace across it demands `p_A - p_B = 0`
    independently of the two outer caps. Three interfaces therefore constrain two unknown
    pressures, and the system closes only because `theta` came from the triple-line force
    balance in the first place.
    """
    for ratio in (0.3, 0.5, 1.0, 1.5, 1.9):
        pressures = ph.doublet_ground_truth_pressures(ratio)
        assert pressures[1] == pytest.approx(pressures[2], rel=0, abs=0)


def test_doublet_pressure_limit_check_ratio_one_is_sqrt_three():
    """Required limit check: at `ratio = 1`, `theta = 60` so `p = 2 * sin(60) = sqrt(3)`."""
    pressures = ph.doublet_ground_truth_pressures(1.0, gamma_cm=1.0, r0=1.0)

    assert pressures[1] == pytest.approx(np.sqrt(3.0))


def test_doublet_pressures_scale_with_gamma_and_inversely_with_size():
    """Dimensional check: `p` is linear in tension and inverse in length."""
    base = ph.doublet_ground_truth_pressures(0.8, gamma_cm=1.0, r0=1.0)[1]

    assert ph.doublet_ground_truth_pressures(0.8, gamma_cm=3.0, r0=1.0)[1] == pytest.approx(3.0 * base)
    assert ph.doublet_ground_truth_pressures(0.8, gamma_cm=1.0, r0=4.0)[1] == pytest.approx(base / 4.0)


def test_doublet_ground_truth_pressures_domain_validation():
    for bad_ratio in (0.0, 2.0, -0.5, 2.5):
        with pytest.raises(ValueError, match="outside the physical domain"):
            ph.doublet_ground_truth_pressures(bad_ratio)


@pytest.mark.parametrize("ratio", [0.5, 1.0, 1.5])
def test_doublet_mesh_outer_caps_have_the_radius_the_pressure_assumes(ratio):
    """The pressure truth must describe the *discretised* mesh, not just the ideal shape.

    Every dome vertex of `doublet_mesh` must lie exactly on the sphere of radius `R` whose
    curvature `2/R` the pressure ground truth uses.
    """
    _, radius, zc = ph._doublet_geometry(ratio, r0=1.0)
    v, _, _ = ph.doublet_mesh(ratio, n_rings=8, n_phi=16)

    on_upper_cap = np.linalg.norm(v - np.array([0.0, 0.0, zc]), axis=1)
    on_lower_cap = np.linalg.norm(v - np.array([0.0, 0.0, -zc]), axis=1)
    is_cap_vertex = np.isclose(on_upper_cap, radius) | np.isclose(on_lower_cap, radius)
    # the only non-cap vertices are the flat disk's centre and its rim ring
    assert is_cap_vertex.sum() > 0
    non_cap = v[~is_cap_vertex]
    assert np.allclose(non_cap[:, 2], 0.0), "unexpected off-plane vertex that is not on a cap"


@pytest.mark.parametrize("method", list(PressureComputationMethod))
def test_infer_pressures_reproduces_the_doublet_young_laplace_pressure(method):
    """End-to-end: foambryo's pressure inference against the analytic ground truth.

    Fed the *ground-truth* tensions, so this isolates pressure inference from tension
    inference. Measured relative error at this resolution is 2e-4 to 1.3e-3 depending on
    method; the tolerance is an order of magnitude looser so the test pins correctness
    rather than the exact discretisation error.
    """
    ratio = 1.0
    truth = ph.doublet_ground_truth_pressures(ratio)
    v, f, regions = ph.doublet_mesh(ratio, n_rings=32, n_phi=64)
    mesh = DcelData(v, f, regions)

    inferred = infer_pressures(mesh, ph.doublet_ground_truth_tensions(ratio), mode=method)

    assert inferred[0] == pytest.approx(truth[0])
    for cell in (1, 2):
        assert inferred[cell] == pytest.approx(truth[cell], rel=1e-2)


# ---------------------------------------------------------------------
# Doublet — the tension gauge, which the error columns are measured in
# ---------------------------------------------------------------------
@pytest.mark.parametrize("ratio", [0.5, 1.0, 1.5])
def test_doublet_gauge_rescaling_preserves_ratios_and_sets_the_mean(ratio):
    """Rescaling is a pure gauge change: ratios fixed, mean set to the requested value."""
    physical = ph.doublet_ground_truth_tensions(ratio)
    gauged = ph.doublet_ground_truth_tensions(ratio, mean_tension=1.0)

    assert np.mean(list(gauged.values())) == pytest.approx(1.0)
    assert gauged[(1, 2)] / gauged[(0, 1)] == pytest.approx(physical[(1, 2)] / physical[(0, 1)])
    assert gauged[(0, 1)] == pytest.approx(gauged[(0, 2)])


@pytest.mark.parametrize("ratio", [0.5, 1.0, 1.5])
def test_doublet_pressures_take_the_same_gauge_factor_as_the_tensions(ratio):
    """Pressures are linear in tension, so one shared factor keeps them one equilibrium."""
    t_physical = ph.doublet_ground_truth_tensions(ratio)
    t_gauged = ph.doublet_ground_truth_tensions(ratio, mean_tension=1.0)
    p_physical = ph.doublet_ground_truth_pressures(ratio)
    p_gauged = ph.doublet_ground_truth_pressures(ratio, mean_tension=1.0)

    factor = t_gauged[(0, 1)] / t_physical[(0, 1)]
    assert p_gauged[1] == pytest.approx(factor * p_physical[1])
    assert p_gauged[0] == 0.0  # the exterior reference is gauge-independent


def test_doublet_gauge_factor_is_one_at_ratio_one_and_not_otherwise():
    """Explains why ratio=1 looked fine and the other ratios did not.

    The physical mean over the three interfaces is `(2 + ratio) / 3`, so it equals 1 only
    at `ratio = 1`. Comparing mean-1-normalised inferred tensions against the physical
    ground truth therefore carries a spurious relative error of `|1/mean - 1|`: zero at
    ratio 1, but 20% at ratio 0.5 and 14.3% at ratio 1.5.
    """
    assert ph._doublet_gauge_factor(1.0, 1.0, 1.0) == pytest.approx(1.0)
    assert ph._doublet_gauge_factor(0.5, 1.0, 1.0) == pytest.approx(1.2)
    assert ph._doublet_gauge_factor(1.5, 1.0, 1.0) == pytest.approx(3.0 / 3.5)
    assert ph._doublet_gauge_factor(0.5, 1.0, None) == 1.0  # opted out


@pytest.mark.parametrize("ratio", [0.5, 1.0, 1.5])
def test_doublet_end_to_end_error_is_small_in_the_matched_gauge(ratio):
    """The point of the gauge: tension AND pressure error are small at every ratio.

    Inference normalises tensions to mean 1, so the ground truth must be stated in that
    gauge. Measured at n_rings=24, across ratios 0.5/1.0/1.5: mean tension error
    6.2%/3.6%/2.3%, worst-interface 12.3%/5.3%/2.7% (always the small cell-cell interface
    (1, 2)), pressure error 1.5%/1.8%/1.9%. Against the *physical*-gauge truth the same run
    reports 17.5%/3.6%/13.9% and 21.8%/1.8%/12.7% — almost all of it scale mismatch rather
    than inference error, and misleadingly clean at ratio 1 only because the gauge factor
    happens to be 1 there.

    Every one of these errors halves when n_rings doubles (checked separately), i.e. O(h)
    discretisation error, so the tolerances below are resolution-specific, not universal.
    """
    v, f, regions = ph.doublet_mesh(ratio, n_rings=24, n_phi=48)
    mesh = DcelData(v, f, regions)
    truth_tensions = ph.doublet_ground_truth_tensions(ratio, mean_tension=1.0)
    truth_pressures = ph.doublet_ground_truth_pressures(ratio, mean_tension=1.0)

    inferred_tensions = infer_tensions(mesh, mean_tension=1)
    inferred_pressures = infer_pressures(mesh, inferred_tensions, base_pressure=0)

    errors = {key: abs(inferred_tensions[key] - x) / x for key, x in truth_tensions.items()}
    assert max(errors.values()) < 0.15
    assert np.mean(list(errors.values())) < 0.07
    for cell in (1, 2):
        assert inferred_pressures[cell] == pytest.approx(truth_pressures[cell], rel=0.03)


def test_doublet_pressure_error_converges_with_resolution():
    """The error is discretisation error, so refining the mesh must reduce it.

    Measured mean relative error for the default (variational) method at n_rings
    8/16/32/64 is 1.69e-2, 4.22e-3, 1.06e-3, 2.7e-4 — a factor of ~4 per doubling, i.e.
    second order. Asserted as a factor of >= 3 per doubling to leave headroom.
    """
    ratio = 1.0
    truth = ph.doublet_ground_truth_pressures(ratio)
    tensions = ph.doublet_ground_truth_tensions(ratio)

    errors = []
    for n_rings in (8, 16, 32):
        v, f, regions = ph.doublet_mesh(ratio, n_rings=n_rings, n_phi=2 * n_rings)
        inferred = infer_pressures(DcelData(v, f, regions), tensions)
        errors.append(abs(inferred[1] - truth[1]) / truth[1])

    assert errors[1] < errors[0] / 3.0
    assert errors[2] < errors[1] / 3.0


def test_doublet_ground_truth_pressures_keys_match_mesh_cells():
    ratio = 0.7
    _, _, regions = ph.doublet_mesh(ratio, n_rings=6, n_phi=12)
    pressures = ph.doublet_ground_truth_pressures(ratio)

    # every region of the mesh, including the exterior, has a ground-truth pressure
    assert set(pressures) == {0, *mesh_io.cell_labels(regions)}


# =====================================================================
# Tetrahedral vertex
# =====================================================================
def test_tetrahedral_point_angle_is_arccos_minus_one_third():
    """Required limit check: the angle between any two of the 4 rays at the point."""
    assert pytest.approx(109.47122063449069) == ph.TETRAHEDRAL_POINT_ANGLE_DEG

    directions = ph._TETRA_DIRECTIONS
    assert directions.sum(axis=0) == pytest.approx(np.zeros(3), abs=1e-12)
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            cos_angle = np.dot(directions[i], directions[j])
            assert cos_angle == pytest.approx(-1.0 / 3.0, abs=1e-12)
            assert np.degrees(np.arccos(cos_angle)) == pytest.approx(ph.TETRAHEDRAL_POINT_ANGLE_DEG)


def test_tetrahedral_vertex_mesh_is_valid_and_has_the_predicted_ray_topology():
    v, f, regions = ph.tetrahedral_vertex_mesh(scale=3.0)
    assert mesh_io.is_intact(v, f)
    assert v.shape == (5, 3)
    assert f.shape == (10, 3)

    mesh = DcelData(v, f, regions)
    volumes = mesh.compute_volumes_cells()
    for cell in (1, 2, 3, 4):
        assert volumes[cell] > 0

    junctions, _ = mesh_io.list_junctions(f, regions)
    # 10 triple lines total: 6 outer (tetrahedron edges, 2 cells + exterior) + the 4
    # interior rays, each where 3 of the 4 cells meet (cell m absent from ray m's triple)
    assert len(junctions) == 10
    interior_rays = [j for j in junctions if 0 not in j]
    assert sorted(interior_rays) == [(1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)]


def test_tetrahedral_vertex_120_degrees_along_each_line():
    """Required limit check: 120 degrees along the lines, via foambryo's own angle code.

    Read from `compute_angles_tri` rather than re-derived here. That is trustworthy for
    this phantom because, unlike the doublet, every trijunction here is between three
    flat interfaces of the same kind.
    """
    v, f, regions = ph.tetrahedral_vertex_mesh(scale=5.0)
    mesh = DcelData(v, f, regions)
    _, angles_deg, lengths = mesh.compute_angles_tri(unique=False)

    junctions, _ = mesh_io.list_junctions(f, regions)
    interior_rays = [j for j in junctions if 0 not in j]
    assert len(interior_rays) == 4
    for a, b, c in interior_rays:
        for key in ((a, b, c), (b, a, c), (a, c, b)):
            if key in angles_deg:
                assert angles_deg[key] == pytest.approx(ph.TETRAHEDRAL_LINE_ANGLE_DEG, abs=1e-6)
        assert lengths[next(k for k in angles_deg if set(k) == {a, b, c})] > 0


def test_tetrahedral_vertex_label_at_matches_mesh_cell_interior_points():
    """Self-consistency check between the analytic mesh and the voxel-mask classifier.

    The two representations of the same phantom must agree.
    """
    scale = 4.0
    idx = [1, 2, 3, 4]
    vertices = ph._TETRA_DIRECTIONS * scale
    origin = np.zeros(3)
    for i in idx:
        others = [j for j in idx if j != i]
        centroid = (origin + sum(vertices[j - 1] for j in others)) / 4
        label = ph.tetrahedral_vertex_label_at(centroid[None, :], scale=scale)[0]
        assert label == i


def test_tetrahedral_vertex_ground_truth_tensions_match_mesh_regions():
    _, _, regions = ph.tetrahedral_vertex_mesh(scale=2.0)
    tensions = ph.tetrahedral_vertex_ground_truth_tensions()
    assert set(tensions) == set(mesh_io.list_interfaces(regions))


# =====================================================================
# Kelvin cell
# =====================================================================
def test_kelvin_mesh_builds_a_valid_multimaterial_mesh():
    v, f, regions, cell_labels = ph.kelvin_mesh()
    assert mesh_io.is_intact(v, f)
    assert len(cell_labels) == 15  # origin cell + its 14 Voronoi (BCC) neighbours

    mesh = DcelData(v, f, regions)
    volumes = mesh.compute_volumes_cells()
    for label in cell_labels:
        assert volumes[label] > 0


def test_kelvin_label_at_matches_lattice_points():
    """Self-consistency check between the Kelvin mesh and the voxel-mask classifier.

    Every kept lattice point must classify to its own (unique) label, and there must be
    exactly 15 of them (the origin + its 14 BCC Voronoi neighbours).
    """
    cell_size = 1.0
    centers = ph._kelvin_lattice_points(cell_size)
    kept = ph._kelvin_kept_mask(centers, cell_size)
    assert kept.sum() == 15

    kept_points = centers[kept]
    labels = ph.kelvin_label_at(kept_points, cell_size=cell_size)
    assert sorted(labels.tolist()) == list(range(1, 16))  # each kept point gets its own distinct label
