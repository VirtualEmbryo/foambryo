"""Tests for the dataset-audit machinery.

Covers the code added by the dataset-audit work:
`foambryo.validation.mesh_io.mesh_integrity` / `vertex_region_counts`,
`foambryo.validation.force_balance.effective_surface_tensions` / `residual_force_field`,
and `foambryo_benchmarks.metrics.residual_localisation` / `effective_tension_ratios` /
`interface_areas` / `triple_line_lengths`.

Two of these tests exist specifically to pin down known failure modes rather than to cover
lines:

- `test_effective_tension_ratios_are_gauge_invariant_under_a_uniform_factor` and
  `test_effective_tension_ratios_detect_a_factor_two_on_a_subset` are the pair that make
  the ratio detector trustworthy. The initial validation stage established that a single-ratio check cannot tell
  a gauge factor from a real defect; these assert that a *uniform* factor is invisible (it
  is a gauge choice) while a factor on a *subset* is visible in both gauges (it is a bug).
- `test_residual_localisation_on_the_golden_master_sits_on_the_triple_lines` pins the
  physical expectation that a converged mesh's residual concentrates at triple lines, which
  is the discriminator the dataset audit used to separate the dataset's five non-converged cases.
"""

import numpy as np
import pytest

from foambryo_benchmarks import metrics as m
from foambryo_benchmarks import phantoms
from foambryo.validation import force_balance as fb
from foambryo.validation import mesh_io
from tests.conftest import DATASET_DIR, require_substitute_mesh


@pytest.fixture
def golden_master():
    """`(v, f, regions)` of the golden-master mesh (see `conftest`'s module docstring)."""
    return mesh_io.read_rec(require_substitute_mesh())


@pytest.fixture
def golden_master_operator(golden_master):
    """`(v, f, regions, interfaces, m)` for the golden master, no line-tension block."""
    v, f, regions = golden_master
    interfaces = mesh_io.list_interfaces(regions)
    cells = mesh_io.cell_labels(regions)
    operator = fb.force_balance_operator(v, f, regions, interfaces, [], {}, cells)
    return v, f, regions, interfaces, operator


def _single_triangle():
    """One triangle separating regions 0 and 1: an open mesh, three boundary edges."""
    v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    f = np.array([[0, 1, 2]])
    regions = np.array([[0, 1]])
    return v, f, regions


def _minimal_triple_line():
    """Three triangles sharing edge (0, 1), separating regions 0/1, 0/2 and 1/2.

    The smallest possible well-formed triple line: one edge, three distinct regions, three
    interfaces, one triangle per interface.
    """
    v = np.array(
        [
            [0.0, 0.0, 0.0],  # 0 - on the triple line
            [0.0, 0.0, 1.0],  # 1 - on the triple line
            [1.0, 0.0, 0.0],
            [-0.5, 0.866, 0.0],
            [-0.5, -0.866, 0.0],
        ],
    )
    f = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]])
    regions = np.array([[0, 1], [0, 2], [1, 2]])
    return v, f, regions


# --------------------------------------------------------------------------------------
# mesh_integrity
# --------------------------------------------------------------------------------------


def test_mesh_integrity_accepts_a_well_formed_triple_line():
    v, f, regions = _minimal_triple_line()
    report = mesh_io.mesh_integrity(v, f, regions)

    assert report["intact"]
    assert report["n_triple_line_edges"] == 1
    assert report["n_malformed_triple_line_edges"] == 0
    assert report["n_quadjunction_edges"] == 0
    assert report["dangling_vertex_reference"] is False
    # The three outer edges of each triangle are boundary edges; only (0, 1) is shared.
    assert report["n_boundary_edges"] == 6


def test_mesh_integrity_rejects_a_triple_line_missing_one_of_its_three_interfaces():
    """Two of the three interfaces present: three regions on the edge, but only 2 interfaces.

    Built by relabelling the third triangle so the edge still sees regions {0, 1, 2} but
    the interface (1, 2) is absent — the case the operator would otherwise assemble a
    junction force balance for.
    """
    v, f, regions = _minimal_triple_line()
    regions = regions.copy()
    regions[2] = [0, 2]  # was [1, 2]

    report = mesh_io.mesh_integrity(v, f, regions)

    assert not report["intact"]
    assert report["n_triple_line_edges"] == 1
    assert report["n_malformed_triple_line_edges"] == 1


def test_mesh_integrity_detects_a_dangling_vertex_reference():
    v, f, regions = _minimal_triple_line()
    f = f.copy()
    f[0, 2] = len(v)  # one past the end

    report = mesh_io.mesh_integrity(v, f, regions)

    assert not report["intact"]
    assert report["dangling_vertex_reference"] is True


def test_mesh_integrity_counts_boundary_edges_on_an_open_mesh():
    v, f, regions = _single_triangle()
    report = mesh_io.mesh_integrity(v, f, regions)

    assert report["n_edges"] == 3
    assert report["n_boundary_edges"] == 3
    assert report["n_triple_line_edges"] == 0
    # No triple line to malform, so the two integrity conditions both hold: an open
    # mesh is a mesh-quality problem, deliberately not an integrity failure here.
    assert report["intact"]


def test_mesh_integrity_detects_duplicate_and_degenerate_faces():
    v, f, regions = _minimal_triple_line()
    f = np.vstack([f, f[0], [[0, 0, 1]]])
    regions = np.vstack([regions, regions[0], regions[0]])

    report = mesh_io.mesh_integrity(v, f, regions)

    assert report["n_duplicate_faces"] == 1
    assert report["n_degenerate_faces"] == 1


def test_mesh_integrity_triple_line_count_agrees_with_list_junctions(golden_master):
    """The two independent implementations of "which edges are triple lines" must agree."""
    v, f, regions = golden_master
    _junctions, edge_to_triple = mesh_io.list_junctions(f, regions)

    report = mesh_io.mesh_integrity(v, f, regions)

    assert report["n_triple_line_edges"] == len(edge_to_triple)


def test_mesh_integrity_passes_on_the_golden_master(golden_master):
    v, f, regions = golden_master
    report = mesh_io.mesh_integrity(v, f, regions)

    assert report["intact"]
    assert report["n_boundary_edges"] == 0, "the golden master is a closed mesh"
    assert report["n_degenerate_faces"] == 0
    assert report["n_duplicate_faces"] == 0


# --------------------------------------------------------------------------------------
# vertex_region_counts
# --------------------------------------------------------------------------------------


def test_vertex_region_counts_on_the_minimal_triple_line():
    v, f, regions = _minimal_triple_line()
    counts = mesh_io.vertex_region_counts(v, f, regions)

    # Vertices 0 and 1 are the triple line: they touch all three regions.
    assert counts[0] == 3
    assert counts[1] == 3
    # Vertices 2, 3, 4 each belong to one triangle, so they touch that triangle's two regions.
    assert list(counts[2:]) == [2, 2, 2]


def test_vertex_region_counts_are_bounded_by_the_number_of_regions(golden_master):
    v, f, regions = golden_master
    counts = mesh_io.vertex_region_counts(v, f, regions)

    n_regions = len(np.unique(regions))
    assert counts.min() >= 2, "every referenced vertex lies on at least one interface"
    assert counts.max() <= n_regions


# --------------------------------------------------------------------------------------
# residual_force_field
# --------------------------------------------------------------------------------------


def test_residual_force_field_norm_reproduces_residual_at(golden_master_operator):
    """`residual_at` must be exactly the norm ratio of the field `residual_force_field` returns."""
    _v, _f, _regions, interfaces, operator = golden_master_operator
    ni = len(interfaces)
    gamma = np.linspace(0.5, 1.5, ni)

    rho = fb.residual_at(operator, ni, 0, gamma, np.zeros(0))
    field = fb.residual_force_field(operator, ni, 0, gamma, np.zeros(0))
    load = operator[:, :ni] @ gamma

    assert field.shape == (operator.shape[0] // 3, 3)
    assert np.linalg.norm(field) / np.linalg.norm(load) == pytest.approx(rho, rel=1e-12)


def test_residual_at_is_gauge_invariant(golden_master_operator):
    """`rho` must not change when the tensions are rescaled — the whole point of using it."""
    _v, _f, _regions, interfaces, operator = golden_master_operator
    ni = len(interfaces)
    gamma = np.linspace(0.5, 1.5, ni)

    rho = fb.residual_at(operator, ni, 0, gamma, np.zeros(0))
    for scale in (0.1, 3.0, 1000.0):
        scaled = fb.residual_at(operator, ni, 0, scale * gamma, np.zeros(0))
        assert scaled == pytest.approx(rho, rel=1e-10)


# --------------------------------------------------------------------------------------
# effective_surface_tensions
# --------------------------------------------------------------------------------------


def test_effective_surface_tensions_respect_the_mean_tension_gauge(golden_master_operator):
    """The gauge is a hard constraint (KKT), so it must hold to machine precision."""
    _v, _f, _regions, interfaces, operator = golden_master_operator
    ni = len(interfaces)

    for mean_tension in (1.0, 2.5):
        gamma_eff, _p_eff, _residual = fb.effective_surface_tensions(operator, ni, 0, mean_tension)
        assert gamma_eff.mean() == pytest.approx(mean_tension, abs=1e-12)


def test_effective_surface_tensions_are_gauge_covariant(golden_master_operator):
    """Doubling the gauge doubles the tensions and leaves the residual fraction unchanged."""
    _v, _f, _regions, interfaces, operator = golden_master_operator
    ni = len(interfaces)

    gamma_1, p_1, residual_1 = fb.effective_surface_tensions(operator, ni, 0, 1.0)
    gamma_2, p_2, residual_2 = fb.effective_surface_tensions(operator, ni, 0, 2.0)

    np.testing.assert_allclose(gamma_2, 2 * gamma_1, rtol=1e-9)
    np.testing.assert_allclose(p_2, 2 * p_1, rtol=1e-9)
    assert residual_2 == pytest.approx(residual_1, rel=1e-9)


def test_effective_surface_tensions_recover_the_doublet_phantom_ground_truth():
    """On an analytic equilibrium, the tensions that balance the mesh are the true ones.

    The doublet is an exact mechanical equilibrium by construction
    (`foambryo_benchmarks/phantoms.py`), so this is the one place in this file where
    `effective_surface_tensions` can be checked against a *known* answer rather than
    against an internal consistency property. Discretisation is the only error source, so
    the tolerance is set by `n_rings`, not by the solver.
    """
    ratio = 0.7
    v, f, regions = phantoms.doublet_mesh(ratio=ratio, n_rings=48)
    truth = phantoms.doublet_ground_truth_tensions(ratio=ratio, mean_tension=1.0)

    interfaces = mesh_io.list_interfaces(regions)
    cells = mesh_io.cell_labels(regions)
    operator = fb.force_balance_operator(v, f, regions, interfaces, [], {}, cells)
    gamma_eff, _p_eff, residual = fb.effective_surface_tensions(operator, len(interfaces), 0, 1.0)

    expected = np.array([truth[key] for key in interfaces])
    # Same gauge on both sides (mean 1), so this is a genuine per-interface comparison.
    np.testing.assert_allclose(gamma_eff, expected, rtol=0.05)
    assert residual < 0.05


# --------------------------------------------------------------------------------------
# effective_tension_ratios — the gauge pair
# --------------------------------------------------------------------------------------


def test_effective_tension_ratios_are_gauge_invariant_under_a_uniform_factor():
    """A uniform factor is a gauge choice, not a defect: every ratio must come out 1.

    This is the failure hit during the initial validation stage: an
    arbitrary overall scale dominated the reported error. Both gauges must absorb it
    completely, for any factor — including a factor of exactly 2, which must *not* be
    mistaken for the defect this detector looks for: a factor on a subset of interfaces
    (for example, some line tensions encoded at half their value by the program that
    generated the mesh).
    """
    interfaces = [(0, 1), (0, 2), (1, 2), (0, 3)]
    nominal = np.array([0.4, 0.9, 1.3, 1.4])

    for factor in (0.5, 2.0, 17.0):
        report = m.effective_tension_ratios(factor * nominal, nominal, interfaces)
        for gauge in ("mean_one", "least_squares", "median_one"):
            stats = report["gauges"][gauge]
            np.testing.assert_allclose(list(stats["per_key"].values()), 1.0, rtol=1e-12)
            assert stats["n_near_one"] == len(interfaces)
            assert stats["n_near_half"] == 0
            assert stats["n_near_two"] == 0
            assert stats["abs_deviation_max"] == pytest.approx(0.0, abs=1e-12)


def test_effective_tension_ratios_detect_a_factor_two_on_a_subset():
    """A factor on a *subset* is a real defect, and `median_one` is the gauge that shows it.

    This is the signature of a factor on a subset of interfaces (for example, some line
    tensions encoded at half their value by the program that generated the mesh). Two of
    six interfaces are halved here.

    The test also pins the **dilution** that makes the obvious check wrong: in the
    `mean_one` gauge the halved subset lands at `0.6`, not `0.5`, because the affected
    interfaces move the mean that defines the gauge. So `n_near_half` is legitimately 0
    there — a literal "is anything near 0.5" test on `mean_one` would *miss this bug*. Only
    `median_one`, whose gauge is set by the unaffected majority, puts it at exactly 0.5.
    """
    interfaces = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (3, 4)]
    nominal = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    effective = nominal.copy()
    effective[:2] *= 0.5

    report = m.effective_tension_ratios(effective, nominal, interfaces)
    ratios = {gauge: np.array(list(stats["per_key"].values())) for gauge, stats in report["gauges"].items()}

    # median_one: the bulk is 1 by construction and the defect sits at its true factor.
    np.testing.assert_allclose(sorted(ratios["median_one"]), [0.5, 0.5, 1.0, 1.0, 1.0, 1.0], rtol=1e-12)
    assert report["gauges"]["median_one"]["n_near_half"] == 2
    assert report["gauges"]["median_one"]["n_near_one"] == 4

    # mean_one: same defect, diluted to 0.6 / 1.2 by the gauge. Documented, not a bug.
    np.testing.assert_allclose(sorted(ratios["mean_one"]), [0.6, 0.6, 1.2, 1.2, 1.2, 1.2], rtol=1e-12)
    assert report["gauges"]["mean_one"]["n_near_half"] == 0

    # Every gauge still shows the bimodality itself, which is the gauge-free signature: two
    # clusters whose ratio is exactly 2.
    for gauge, ratio in ratios.items():
        assert ratio.max() / ratio.min() == pytest.approx(2.0, rel=1e-12), gauge
        assert report["gauges"][gauge]["spread"] > 0.4


# --------------------------------------------------------------------------------------
# residual_localisation
# --------------------------------------------------------------------------------------


def test_residual_localisation_fractions_sum_to_one(golden_master_operator):
    v, f, regions, interfaces, operator = golden_master_operator
    gamma = np.ones(len(interfaces))

    report = m.residual_localisation(
        v, f, regions, operator, len(interfaces), gamma, mesh_io.mean_edge_length(v, f),
    )

    total = (
        report["fraction_interface_interior"]
        + report["fraction_triple_line"]
        + report["fraction_quadjunction"]
    )
    assert total == pytest.approx(1.0, abs=1e-10)
    assert report["n_interface_interior_vertices"] > report["n_triple_line_vertices"]


def test_residual_localisation_on_the_golden_master_sits_on_the_triple_lines():
    """A converged mesh at its own ground-truth tensions leaves its residual at the junctions.

    Young-Laplace is satisfied pointwise in the interface interiors of a converged mesh, so
    the residual has to live where Young-Dupré is imposed on a discrete triangulation. This
    is the physical expectation that the dataset audit's discriminator rests on; pinning it here means a
    future change that inverts it fails loudly instead of silently reinterpreting the five
    non-converged dataset cases.
    """
    mesh_path = require_substitute_mesh()
    tensions_path = DATASET_DIR / "000_dict_tensions.npy"
    if not tensions_path.exists():
        pytest.skip(f"{tensions_path} not found")

    v, f, regions = mesh_io.read_rec(mesh_path)
    truth = np.load(tensions_path, allow_pickle=True).item()
    truth = {tuple(int(x) for x in key): float(value) for key, value in truth.items()}

    interfaces = mesh_io.list_interfaces(regions)
    cells = mesh_io.cell_labels(regions)
    operator = fb.force_balance_operator(v, f, regions, interfaces, [], {}, cells)
    gamma = np.array([truth[key] for key in interfaces])

    report = m.residual_localisation(
        v, f, regions, operator, len(interfaces), gamma, mesh_io.mean_edge_length(v, f),
    )

    assert report["fraction_triple_line"] > 0.5
    assert report["fraction_interface_interior"] < 0.5
    # Triple-line vertices are ~2% of the mesh but carry most of the residual, so the
    # per-vertex amplitude there must be far larger than in the interiors.
    assert report["interior_over_triple_line"] < 0.2


# --------------------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------------------


def test_interface_areas_sum_to_the_total_triangle_area(golden_master):
    v, f, regions = golden_master
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    total = float(0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1).sum())

    areas = m.interface_areas(v, f, regions)

    assert set(areas) == set(mesh_io.list_interfaces(regions))
    assert sum(areas.values()) == pytest.approx(total, rel=1e-12)


def test_interface_areas_on_the_minimal_triple_line():
    v, f, regions = _minimal_triple_line()
    areas = m.interface_areas(v, f, regions)

    # Triangle (0, 1, 2) has legs of length 1 along z and x, so area 1/2.
    assert areas[(0, 1)] == pytest.approx(0.5)
    assert set(areas) == {(0, 1), (0, 2), (1, 2)}


def test_triple_line_lengths_on_the_minimal_triple_line():
    v, f, regions = _minimal_triple_line()
    _junctions, edge_to_triple = mesh_io.list_junctions(f, regions)

    lengths = m.triple_line_lengths(v, edge_to_triple)

    assert lengths == {(0, 1, 2): pytest.approx(1.0)}
