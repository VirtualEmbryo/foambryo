"""The error bars and the diagnostics refuse, at once, a mesh whose dense operator scipy cannot decompose.

`compute_inference_diagnostics`, which `infer_tensions_with_uncertainty` always calls, takes a thin SVD of the
dense force-balance operator `(3 n_vertices, n_interfaces + n_cells)`. Up to 0.7.0 a 1 866-cell tissue reached
that SVD after about 20 minutes and 55 GiB, and scipy then raised its LAPACK integer-overflow `ValueError`. Both
functions now raise before anything is built, with a message saying what the limit is, why, and what to use
instead. The limit itself is lowered here so that the golden mesh crosses it.
"""

import numpy as np
import pytest
from scipy import linalg

from foambryo import TensionComputationMethod, dcel_mesh_from_file, uncertainty
from foambryo.force_balance_inference import build_force_balance_operator
from foambryo.uncertainty import compute_inference_diagnostics, infer_tensions_with_uncertainty
from tests.conftest import require_substitute_mesh


@pytest.fixture(scope="module")
def golden_mesh():
    return dcel_mesh_from_file(str(require_substitute_mesh()))


def test_the_limit_is_where_scipy_refuses_a_thin_svd():
    """2**31 - 1 entries, the most scipy's standard LAPACK can index: one more and its thin SVD refuses."""
    assert uncertainty.DENSE_OPERATOR_ENTRY_LIMIT == 2**31 - 1
    from scipy.linalg import _decomp_svd

    if getattr(_decomp_svd, "HAS_ILP64", False):
        pytest.skip("this scipy is built with 64-bit LAPACK integers")
    columns = 1000
    rows = uncertainty.DENSE_OPERATOR_ENTRY_LIMIT // columns + 1
    too_large = np.broadcast_to(np.float64(0.0), (rows, columns))  # a view: no memory is allocated
    with pytest.raises(ValueError, match="integer overflow in LAPACK"):
        linalg.svd(too_large, full_matrices=False, check_finite=False)


def test_the_shape_is_counted_without_building_the_operator(golden_mesh):
    operator, _interfaces, _cells = build_force_balance_operator(golden_mesh)
    assert uncertainty._dense_operator_shape(golden_mesh) == operator.shape


@pytest.mark.parametrize(
    ("call", "name"),
    [
        (lambda mesh: compute_inference_diagnostics(mesh), "compute_inference_diagnostics"),
        (
            lambda mesh: infer_tensions_with_uncertainty(mesh, 1.0, method=TensionComputationMethod.YoungDupre),
            "infer_tensions_with_uncertainty",
        ),
        (
            lambda mesh: infer_tensions_with_uncertainty(mesh, 1.0, method=TensionComputationMethod.ForceBalance),
            "infer_tensions_with_uncertainty",
        ),
    ],
)
def test_a_mesh_over_the_limit_is_refused_before_the_operator_is_built(golden_mesh, monkeypatch, call, name):
    rows, columns = uncertainty._dense_operator_shape(golden_mesh)
    monkeypatch.setattr(uncertainty, "DENSE_OPERATOR_ENTRY_LIMIT", rows * columns - 1)

    def must_not_be_called(*_args: object, **_kwargs: object):
        message = "the operator was built before the size was checked"
        raise AssertionError(message)

    monkeypatch.setattr(uncertainty, "build_force_balance_operator", must_not_be_called)
    with pytest.raises(ValueError, match=f"{name} cannot run on this mesh") as refused:
        call(golden_mesh)
    text = str(refused.value)
    assert f"{rows} x {columns}" in text
    assert "2**31 - 1" in text
    assert "foambryo.infer_tensions " in text
    assert "foambryo.infer_tensions_over_seeds" in text


def test_a_mesh_at_the_limit_still_runs(golden_mesh, monkeypatch):
    rows, columns = uncertainty._dense_operator_shape(golden_mesh)
    monkeypatch.setattr(uncertainty, "DENSE_OPERATOR_ENTRY_LIMIT", rows * columns)
    diagnostics = compute_inference_diagnostics(golden_mesh)
    assert diagnostics.n_rows == rows
