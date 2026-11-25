"""The sparse-assembly work — sparse variational assembly, closed-form derivatives, float64 gradient check.

Three things are pinned here:

1. **Sparsification is exact.** The sparse operators
   (`compute_area_derivative_sparse` / `compute_volume_derivative_sparse`) reproduce the
   dense closed-form dicts, and `_build_matrix_discrete_sparse` reproduces the dense
   `_build_matrix_discrete`, to rtol=1e-10. This is the acceptance criterion "variational
   tensions agree with baseline to rtol=1e-10", read the only way it can be self-consistent:
   *given identical (closed-form) derivatives*, the sparse route equals the dense route.

2. **The closed-form derivatives are correct**, cross-checked in float64 against the complex-step
   derivatives of each triangle's area and volume term and each trijunction edge's length
   (`tests/derivative_references.py`), on the three golden-master meshes. They agree to 1e-13
   normwise or better (measured 3e-16 to 4e-14); a planted sign error or factor of 2 gives 0.3 to 2.
   This replaced a float32 PyTorch autodiff check, which agreed only to about 1e-5.

3. **The variational tensions moved only by the float32->float64 derivative switch** (~2e-7),
   documented and expected, not by the sparse assembly.
"""

import numpy as np
import pytest

from foambryo import TensionComputationMethod, dcel_mesh_from_file
from foambryo.geometry import (
    compute_area_derivative_dict,
    compute_area_derivative_sparse,
    compute_length_derivative_dict,
    compute_volume_derivative_dict,
    compute_volume_derivative_sparse,
)
from foambryo.tension_inference import (
    _build_matrix_discrete,
    _build_matrix_discrete_sparse,
    infer_tensions_and_residuals,
)
from tests.conftest import GOLDEN_MASTER_CASES, golden_master_mesh, require_substitute_mesh
from tests.derivative_references import (
    reference_area_derivatives,
    reference_length_derivatives,
    reference_volume_derivatives,
    relative_normwise_disagreement,
)


@pytest.fixture(scope="module")
def golden_mesh():
    return dcel_mesh_from_file(str(require_substitute_mesh()))


def test_sparse_area_operator_reproduces_dict(golden_mesh):
    keys, matrix_d = compute_area_derivative_sparse(golden_mesh)
    reference = compute_area_derivative_dict(golden_mesh)
    n_points = len(golden_mesh.v)
    assert keys == sorted(reference.keys())
    dense = matrix_d.toarray()
    for i, key in enumerate(keys):
        got = dense[i].reshape(n_points, 3)
        np.testing.assert_allclose(got, reference[key], rtol=1e-10, atol=1e-12)


def test_sparse_volume_operator_reproduces_dict(golden_mesh):
    materials, matrix_v = compute_volume_derivative_sparse(golden_mesh)
    reference = compute_volume_derivative_dict(golden_mesh)
    n_points = len(golden_mesh.v)
    assert list(materials) == list(golden_mesh.materials)
    dense = matrix_v.toarray()
    for j, material in enumerate(materials):
        got = dense[j].reshape(n_points, 3)
        np.testing.assert_allclose(got, reference[int(material)], rtol=1e-10, atol=1e-12)


def test_sparse_operator_is_actually_sparse(golden_mesh):
    """Each interface gradient touches only its own vertices — the point of the sparse assembly."""
    _keys, matrix_d = compute_area_derivative_sparse(golden_mesh)
    density = matrix_d.nnz / (matrix_d.shape[0] * matrix_d.shape[1])
    # ~6 % here (a code review measured mean interface support at 6.7 %); far below the dense dict's 100 %, and
    # it shrinks as the mesh grows. The guard is that the operator is genuinely sparse.
    assert density < 0.10, f"area operator unexpectedly dense: {density:.3%}"


def test_sparse_matrix_m_matches_dense_build(golden_mesh):
    """The assembled normal-equations matrix is identical to the dense dict path to rtol=1e-10."""
    area = compute_area_derivative_dict(golden_mesh)
    volume = compute_volume_derivative_dict(golden_mesh)
    dense, _, nm_dense = _build_matrix_discrete(area, volume, golden_mesh.materials)
    sparse, _, nm_sparse, _keys = _build_matrix_discrete_sparse(golden_mesh)
    assert nm_dense == nm_sparse
    np.testing.assert_allclose(sparse, dense, rtol=1e-10, atol=1e-12)


def _dense_route_p_tensions(golden_mesh):
    """The historical dense route-P solve, rebuilt here as the reproducibility reference."""
    from scipy import linalg

    from foambryo.tension_inference import _extract_submatrices

    area = compute_area_derivative_dict(golden_mesh)
    volume = compute_volume_derivative_dict(golden_mesh)
    dense, _, nm = _build_matrix_discrete(area, volume, golden_mesh.materials)
    f_g, f_p, g_g, g_p = _extract_submatrices(dense, nm)
    mat_t = f_g - f_p @ np.linalg.inv(g_p) @ g_g
    mat_inference = np.zeros((nm + 1, nm))
    mat_inference[:nm, :nm] = mat_t
    mat_inference[nm:] = 1
    b_inf = np.zeros(nm + 1)
    b_inf[-1] = nm
    x_reference, _, _, _ = linalg.lstsq(mat_inference, b_inf)
    return x_reference


def test_route_gram_tensions_match_dense_route(golden_mesh):
    """Route `Gram` still equals the dense closed-form solve to rtol=1e-10, once gauge-matched.

    **Updated by the uncertainty work, deliberately.** This test originally asserted that the
    *shipped* `Variational` reproduces the dense route-P solve. That work flipped the default
    route to `Operator`
    on measured evidence (`foambryo_benchmarks/run_variational_route.py`), which moves the
    tensions by ~4e-04 — so the assertion is now made against route `Gram` explicitly. That
    keeps exactly what the sparse-assembly work's criterion was for
    (every published number stays reproducible from this repository) while letting the default
    change be visible rather than silently tolerated.
    """
    from foambryo.tension_inference import VariationalRoute, _infer_tension_variational_yd

    _values, dict_tensions, _resid = _infer_tension_variational_yd(
        golden_mesh,
        1,
        route=VariationalRoute.Gram,
    )
    keys = sorted(dict_tensions)
    x_got = np.array([dict_tensions[tuple(k)] for k in keys])
    # Since 0.7.0 the gauge is exact (`mean = 1`); the historical solve used a soft row of ones,
    # which only rescales a homogeneous system's solution. The ratios are what is reproduced.
    reference = _dense_route_p_tensions(golden_mesh)
    assert x_got.mean() == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(x_got, reference * (len(reference) / reference.sum()), rtol=1e-10, atol=1e-12)


def test_the_default_route_differs_from_route_p_by_the_measured_shift(golden_mesh):
    """The route flip's effect is pinned, so it can never be mistaken for float noise.

    The uncertainty work measured the per-tension shift as median 1.4e-05 over the 42 equilibrium cases, up to
    6.4e-03 on the worst interface. On the golden mesh it is ~4e-04: four orders above float
    noise and below the ground truth's own worst-case quantisation floor (6.7e-04).
    """
    _, dict_tensions, _ = infer_tensions_and_residuals(golden_mesh, 1, TensionComputationMethod.Variational)
    keys = sorted(dict_tensions)
    x_got = np.array([dict_tensions[k] for k in keys])
    shift = np.abs(x_got - _dense_route_p_tensions(golden_mesh))
    assert 1e-6 < shift.max() < 1e-2, shift.max()


CLOSED_FORMS = {
    "area": (compute_area_derivative_dict, reference_area_derivatives),
    "volume": (compute_volume_derivative_dict, reference_volume_derivatives),
    "length": (compute_length_derivative_dict, reference_length_derivatives),
}


@pytest.fixture(scope="module", params=GOLDEN_MASTER_CASES)
def committed_golden_mesh(request, tmp_path_factory):
    """Each committed golden-master mesh (000, 002, 011), so the check runs without the benchmarking dataset."""
    return dcel_mesh_from_file(str(golden_master_mesh(request.param, tmp_path_factory.mktemp(request.param))))


@pytest.mark.parametrize("quantity", sorted(CLOSED_FORMS))
def test_closed_form_derivatives_match_the_complex_step(committed_golden_mesh, quantity):
    """The float64 closed form agrees with the complex step to 1e-13 normwise (measured 3e-16 to 4e-14)."""
    closed_form, reference = CLOSED_FORMS[quantity]
    disagreement = relative_normwise_disagreement(
        closed_form(committed_golden_mesh), reference(committed_golden_mesh, "complex_step"),
    )
    assert disagreement <= 1e-13, disagreement


@pytest.mark.parametrize("quantity", sorted(CLOSED_FORMS))
@pytest.mark.parametrize("planted", ["sign flipped", "factor of 2"])
def test_the_gradient_check_catches_a_planted_error(committed_golden_mesh, quantity, planted):
    """Negative control: one key's derivative negated or doubled is far outside the bound (measured 0.3 to 2)."""
    closed_form, reference = CLOSED_FORMS[quantity]
    derivatives = closed_form(committed_golden_mesh)
    largest = max(derivatives, key=lambda key: np.linalg.norm(derivatives[key]))
    factor = -1.0 if planted == "sign flipped" else 2.0
    wrong = {**derivatives, largest: factor * derivatives[largest]}
    assert relative_normwise_disagreement(wrong, reference(committed_golden_mesh, "complex_step")) > 1e-3
