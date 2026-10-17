"""The sparse-assembly work — sparse variational assembly, closed-form derivatives, autodiff gradient check.

Three things are pinned here:

1. **Sparsification is exact.** The sparse operators
   (`compute_area_derivative_sparse` / `compute_volume_derivative_sparse`) reproduce the
   dense closed-form dicts, and `_build_matrix_discrete_sparse` reproduces the dense
   `_build_matrix_discrete`, to rtol=1e-10. This is the acceptance criterion "variational
   tensions agree with baseline to rtol=1e-10", read the only way it can be self-consistent:
   *given identical (closed-form) derivatives*, the sparse route equals the dense route.

2. **The closed-form derivatives are correct**, cross-checked against the independent
   PyTorch autodiff path (`foambryo._autodiff_derivatives`). They agree to ~1e-5 normwise —
   float32 autodiff is the limiter, not the closed form. A sign or
   factor error in the closed form would show up as an O(1) disagreement here, not 1e-5.
   `torch` is optional, so this test skips cleanly when it is absent.

3. **The variational tensions moved only by the float32->float64 derivative switch** (~2e-7),
   documented and expected, not by the sparse assembly.
"""

import numpy as np
import pytest

from foambryo import TensionComputationMethod, dcel_mesh_from_file
from foambryo.geometry import (
    compute_area_derivative_dict,
    compute_area_derivative_sparse,
    compute_volume_derivative_dict,
    compute_volume_derivative_sparse,
)
from foambryo.tension_inference import (
    _build_matrix_discrete,
    _build_matrix_discrete_sparse,
    infer_tensions_and_residuals,
)
from tests.conftest import require_substitute_mesh


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


def test_closed_form_derivatives_match_autodiff_gradient_check(golden_mesh):
    """Autodiff (float32) cross-checks the closed-form (float64) derivatives.

    Agreement is limited to ~1e-5 normwise by float32 autodiff, not by the closed form. A wrong
    closed-form derivation would disagree at O(1); 1e-4 is a comfortable guard against that while
    tolerating the float32 noise.
    """
    torch = pytest.importorskip("torch")  # noqa: F841 — optional autodiff dependency
    from foambryo._autodiff_derivatives import (
        compute_area_derivative_autodiff,
        compute_volume_derivative_autodiff_dict,
    )

    area_cf = compute_area_derivative_dict(golden_mesh)
    area_ad = compute_area_derivative_autodiff(golden_mesh)
    volume_cf = compute_volume_derivative_dict(golden_mesh)
    volume_ad = compute_volume_derivative_autodiff_dict(golden_mesh)

    a_cf = np.concatenate([area_cf[k].ravel() for k in sorted(area_cf)])
    a_ad = np.concatenate([area_ad[k].ravel() for k in sorted(area_cf)])
    v_cf = np.concatenate([volume_cf[k].ravel() for k in sorted(volume_cf)])
    v_ad = np.concatenate([volume_ad[k].ravel() for k in sorted(volume_cf)])

    assert np.linalg.norm(a_ad - a_cf) / np.linalg.norm(a_cf) < 1e-4
    assert np.linalg.norm(v_ad - v_cf) / np.linalg.norm(v_cf) < 1e-4
