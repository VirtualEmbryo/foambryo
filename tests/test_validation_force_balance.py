"""Agreement tests: `foambryo.validation` vs. its reference implementation.

Written at the package's initial validation stage: `foambryo.validation.mesh_io` and
`foambryo.validation.force_balance` are a port of a standalone numpy/scipy reference
implementation, kept in the test tree as `tests/reference/force_balance_reference_implementation.py`,
with no algorithmic change (docstrings/type hints/naming only). These tests load the reference implementation
directly (`tests.conftest.load_reference_implementation`) and
assert the ported functions reproduce its outputs exactly (bit-identical, since both run
the same numpy operations in the same order on the same inputs) or, where floating-point
order of operations could plausibly differ, to a tight numerical tolerance.

Test mesh: `benchmarking-dataset/000_mesh.rec` (real ground-truth mesh, 6 cells) — the
same substitute mesh already used by `tests/test_golden_master.py`.
"""

import numpy as np
import pytest

from foambryo.validation import force_balance as fb
from foambryo.validation import mesh_io
from tests.conftest import DATASET_DIR, load_reference_implementation, require_substitute_mesh


@pytest.fixture(scope="module")
def reference():
    return load_reference_implementation()


@pytest.fixture(scope="module")
def mesh_path():
    return require_substitute_mesh()


@pytest.fixture(scope="module")
def reference_mesh(reference, mesh_path):
    return reference.read_rec(str(mesh_path))


@pytest.fixture(scope="module")
def reference_combinatorics(reference, reference_mesh):
    _, f, regions = reference_mesh
    interfaces = reference.list_interfaces(regions)
    junctions, edge_to_triple = reference.list_junctions(f, regions)
    cells = reference.cell_labels(regions)
    return interfaces, junctions, edge_to_triple, cells


@pytest.fixture(scope="module")
def reference_operator(reference, reference_mesh, reference_combinatorics):
    v, f, regions = reference_mesh
    interfaces, junctions, edge_to_triple, cells = reference_combinatorics
    return reference.force_balance_operator(v, f, regions, interfaces, junctions, edge_to_triple, cells)


def test_read_rec_matches_reference(reference, mesh_path):
    v_ref, f_ref, r_ref = reference.read_rec(str(mesh_path))
    v_ours, f_ours, r_ours = mesh_io.read_rec(mesh_path)

    np.testing.assert_array_equal(v_ours, v_ref)
    np.testing.assert_array_equal(f_ours, f_ref)
    np.testing.assert_array_equal(r_ours, r_ref)


def test_is_intact_matches_reference(reference, reference_mesh):
    v, f, _ = reference_mesh
    assert mesh_io.is_intact(v, f) == reference.is_intact(v, f)
    assert mesh_io.is_intact(v, f) is True  # a real ground-truth mesh must be intact


def test_cell_labels_matches_reference(reference, reference_mesh):
    _, _, regions = reference_mesh
    assert mesh_io.cell_labels(regions) == reference.cell_labels(regions)


def test_list_interfaces_matches_reference(reference, reference_mesh):
    _, _, regions = reference_mesh
    assert mesh_io.list_interfaces(regions) == reference.list_interfaces(regions)


def test_list_junctions_matches_reference(reference, reference_mesh):
    _, f, regions = reference_mesh
    junctions_ours, edge_to_triple_ours = mesh_io.list_junctions(f, regions)
    junctions_ref, edge_to_triple_ref = reference.list_junctions(f, regions)

    assert junctions_ours == junctions_ref
    assert edge_to_triple_ours == edge_to_triple_ref


def test_force_balance_operator_matches_reference(reference_mesh, reference_combinatorics, reference_operator):
    v, f, regions = reference_mesh
    interfaces, junctions, edge_to_triple, cells = reference_combinatorics

    m_ours = fb.force_balance_operator(v, f, regions, interfaces, junctions, edge_to_triple, cells)

    assert m_ours.shape == reference_operator.shape
    np.testing.assert_array_equal(m_ours, reference_operator)


def test_force_balance_operator_no_line_tension_block_is_m_gamma_m_p(reference_mesh, reference_combinatorics):
    """Check the harness default `beta = (gamma, p)` really is the general operator's two blocks.

    Passing empty `junctions`/`edge_to_triple` must give the same M_gamma and M_p blocks
    as the general operator, with nj == 0 and no line-tension columns in between.
    """
    v, f, regions = reference_mesh
    interfaces, junctions, edge_to_triple, cells = reference_combinatorics
    ni = len(interfaces)

    m_general = fb.force_balance_operator(v, f, regions, interfaces, junctions, edge_to_triple, cells)
    m_default = fb.force_balance_operator(v, f, regions, interfaces, [], {}, cells)

    assert m_default.shape == (m_general.shape[0], ni + len(cells))
    m_gamma_general, _, m_p_general = fb.split(m_general, ni, len(junctions))
    m_gamma_default, m_t_default, m_p_default = fb.split(m_default, ni, 0)

    assert m_t_default.shape[1] == 0
    np.testing.assert_array_equal(m_gamma_default, m_gamma_general)
    np.testing.assert_array_equal(m_p_default, m_p_general)


def test_split_matches_reference(reference, reference_operator, reference_combinatorics):
    interfaces, junctions, _, _ = reference_combinatorics
    ni, nj = len(interfaces), len(junctions)

    g_ours, l_ours, p_ours = fb.split(reference_operator, ni, nj)
    g_ref, l_ref, p_ref = reference.split(reference_operator, ni, nj)

    np.testing.assert_array_equal(g_ours, g_ref)
    np.testing.assert_array_equal(l_ours, l_ref)
    np.testing.assert_array_equal(p_ours, p_ref)


def test_residual_at_matches_reference(reference, reference_operator, reference_combinatorics):
    interfaces, junctions, _, _ = reference_combinatorics
    ni, nj = len(interfaces), len(junctions)
    rng = np.random.default_rng(0)
    gamma = rng.uniform(0.5, 1.5, size=ni)
    t = rng.uniform(-0.5, 0.5, size=nj)

    rho_ours = fb.residual_at(reference_operator, ni, nj, gamma, t)
    rho_ref = reference.residual_at(reference_operator, ni, nj, gamma, t)

    assert rho_ours == pytest.approx(rho_ref, rel=1e-10)


def test_residual_at_ground_truth_tensions_no_line_tension(
    reference,
    reference_mesh,
    reference_combinatorics,
    mesh_path,
):
    """Exercise rho at the mesh's own ground-truth tensions, harness-default operator.

    This is exactly the quantity the dataset audit later reports over the full 47-case
    dataset; here we only check it runs and matches the reference implementation on one
    case (an initial-validation smoke test, not a dataset conclusion).
    """
    case_id = mesh_path.stem.split("_")[0]
    tensions_path = DATASET_DIR / f"{case_id}_dict_tensions.npy"
    if not tensions_path.exists():
        pytest.skip(f"{tensions_path} not found")
    dict_tensions = np.load(tensions_path, allow_pickle=True).item()

    v, f, regions = reference_mesh
    interfaces, _, _, cells = reference_combinatorics
    m = fb.force_balance_operator(v, f, regions, interfaces, [], {}, cells)
    m_ref = reference.force_balance_operator(v, f, regions, interfaces, [], {}, cells)
    gamma = np.array([dict_tensions[key] for key in interfaces])

    rho_ours = fb.residual_at(m, len(interfaces), 0, gamma, np.zeros(0))
    rho_ref = reference.residual_at(m_ref, len(interfaces), 0, gamma, np.zeros(0))

    assert np.isfinite(rho_ours)
    assert rho_ours == pytest.approx(rho_ref, rel=1e-10)


def test_singular_values_matches_reference(reference, reference_operator):
    s_ours = fb.singular_values(reference_operator)
    s_ref = reference.singular_values(reference_operator)

    np.testing.assert_allclose(s_ours, s_ref, rtol=1e-10)


def test_soft_mode_matches_reference(reference, reference_operator, reference_combinatorics):
    interfaces, junctions, _, _ = reference_combinatorics
    ni, nj = len(interfaces), len(junctions)

    sigma_ours, frac_ours = fb.soft_mode(reference_operator, ni, nj)
    sigma_ref, frac_ref = reference.soft_mode(reference_operator, ni, nj)

    assert sigma_ours == pytest.approx(sigma_ref, rel=1e-10)
    np.testing.assert_allclose(frac_ours, frac_ref, rtol=1e-10)


def test_block_anchors_matches_reference(reference, reference_combinatorics, reference_operator):
    interfaces, junctions, _, _ = reference_combinatorics
    ni, nj = len(interfaces), len(junctions)
    ncols = reference_operator.shape[1]

    a_ours = fb.block_anchors(ni, nj, ncols)
    a_ref = reference.block_anchors(ni, nj, ncols)

    for got, expected in zip(a_ours, a_ref, strict=True):
        np.testing.assert_array_equal(got, expected)


def test_solve_constrained_matches_reference(reference, reference_operator, reference_combinatorics):
    interfaces, junctions, _, _ = reference_combinatorics
    ni, nj = len(interfaces), len(junctions)
    ncols = reference_operator.shape[1]
    anchors = fb.block_anchors(ni, nj, ncols)
    # mean gamma anchored to 1; the mean-line-tension anchor is the zero vector when
    # nj == 0, so its target is 0.0 in both cases (see block_anchors)
    targets = np.array([1.0, 0.0])

    beta_ours = fb.solve_constrained(reference_operator, anchors, targets)
    beta_ref = reference.solve_constrained(reference_operator, anchors, targets)

    np.testing.assert_allclose(beta_ours, beta_ref, rtol=1e-8)


def test_map_estimate_matches_reference(reference, reference_operator):
    ncols = reference_operator.shape[1]
    prior_mean = np.ones(ncols)
    prior_cov = np.eye(ncols) * 10.0

    beta_ours, cov_ours = fb.map_estimate(reference_operator, prior_mean, prior_cov)
    beta_ref, cov_ref = reference.map_estimate(reference_operator, prior_mean, prior_cov)

    np.testing.assert_allclose(beta_ours, beta_ref, rtol=1e-8)
    np.testing.assert_allclose(cov_ours, cov_ref, rtol=1e-8)
