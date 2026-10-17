"""Every pressure method adds the medium's pressure `base_pressure` once, and only once.

The regression this pins
------------------------
The Laplace solve pins the exterior at `x_0 = base_pressure` and eliminates the pin into the
right-hand side, so it returns pressures that already include `base_pressure`; until 0.7.0 the
caller added `base_pressure` to every cell a second time. From `base_pressure=0` to `1`, every cell
of benchmark case 000 moved by 2 with `Laplace` and `WeightedLaplace` and by 1 with `Variational`,
while the medium was reported at 1 by all three. With the default `base_pressure=0` nothing was
wrong; the differences between cells were always right.

What is asserted
----------------
For every pressure method, on the dense and on the sparse path: `p(b) - p(0) == b` for every cell
whose pressure is determined (to 1e-12 of the larger of `|b|` and the pressures' scale), the medium is
`b` exactly, and the undetermined cells stay `nan`. With a Laplace method the returned array holds
the same pressures as the map. On benchmark case 000 (`benchmarking-dataset/000_mesh.rec`), on the cut
benchmark case 000 (`tests/data/`, which has an undetermined cell under the Laplace methods), on a
cropped three-cell ball reconstructed by dw3d (the variational method, the only one with a crop-face
formulation, on both of its solve paths), and through `infer_forces`.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from dw3d import get_default_mesh_reconstruction_algorithm

from foambryo import (
    PressureComputationMethod,
    dcel_mesh_from_file,
    infer_forces,
    infer_pressures_and_residuals,
    infer_tensions,
)
from foambryo.dcel import DcelData

from .conftest import require_substitute_mesh

CUT_CASE = Path(__file__).parent / "data" / "cut_benchmark_case_000_mesh.npz"
BASES = (1.0, 0.25, -3.5)
SOLVERS = ("dense", "sparse")


def _quiet(function, *args: object, **kwargs: object):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


@pytest.fixture(scope="module")
def case_000() -> DcelData:
    return dcel_mesh_from_file(str(require_substitute_mesh()))


@pytest.fixture(scope="module")
def cut_case() -> DcelData:
    data = np.load(CUT_CASE)
    return DcelData(data["points"], data["triangles"], data["labels"])


@pytest.fixture(scope="module")
def cropped_ball() -> DcelData:
    """Cells 1 and 2 side by side, cell 3 below both, the image cutting cells 1 and 2 (as in `test_crop_faces`)."""
    n, radius = 60, 20
    z, y, x = np.mgrid[:n, :n, :n]
    centre = n / 2
    inside = (z - centre) ** 2 + (y - centre) ** 2 + (x - centre) ** 2 <= radius**2
    mask = np.zeros((n, n, n), dtype=np.uint16)
    mask[inside & (y < centre) & (z >= centre - 8)] = 1
    mask[inside & (y >= centre) & (z >= centre - 8)] = 2
    mask[inside & (z < centre - 8)] = 3
    algorithm = get_default_mesh_reconstruction_algorithm()
    points, triangles, labels = _quiet(
        algorithm.construct_mesh_from_segmentation_mask, np.ascontiguousarray(mask[: int(centre) + 10]),
    )
    return DcelData(
        np.asarray(points, dtype=np.float64), np.asarray(triangles, dtype=np.int64),
        np.asarray(labels, dtype=np.int64), crop_faces=np.asarray(algorithm.crop_faces),
    )


def _assert_added_once(mesh: DcelData, mode: PressureComputationMethod, solver: str) -> None:
    tensions = _quiet(infer_tensions, mesh)
    _array_0, at_0, _residuals = _quiet(infer_pressures_and_residuals, mesh, tensions, mode, 0.0, solver=solver)
    assert at_0[0] == 0.0
    cells = [c for c in at_0 if c != 0 and np.isfinite(at_0[c])]
    undetermined = [c for c in at_0 if c != 0 and not np.isfinite(at_0[c])]
    assert cells
    scale = max(abs(at_0[c]) for c in cells)
    for base in BASES:
        array_b, at_b, _residuals = _quiet(infer_pressures_and_residuals, mesh, tensions, mode, base, solver=solver)
        assert at_b[0] == base
        shift = np.array([at_b[c] - at_0[c] for c in cells])
        np.testing.assert_allclose(shift, base, rtol=0, atol=1e-12 * max(abs(base), scale))
        assert all(np.isnan(at_b[c]) for c in undetermined)
        if mode != PressureComputationMethod.Variational:
            assert array_b[0] == base
            index = {int(m): i for i, m in enumerate(mesh.materials)}
            assert all(array_b[index[int(c)]] == at_b[c] for c in cells)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("mode", list(PressureComputationMethod))
def test_case_000_every_cell_moves_by_the_base_pressure(case_000, mode, solver):
    _assert_added_once(case_000, mode, solver)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("mode", list(PressureComputationMethod))
def test_cut_case_every_determined_cell_moves_by_the_base_pressure(cut_case, mode, solver):
    _assert_added_once(cut_case, mode, solver)


@pytest.mark.parametrize("solver", SOLVERS)
def test_cropped_mesh_every_determined_cell_moves_by_the_base_pressure(cropped_ball, solver):
    _assert_added_once(cropped_ball, PressureComputationMethod.Variational, solver)


@pytest.mark.parametrize("mode", list(PressureComputationMethod))
def test_infer_forces_passes_the_base_pressure_through_once(case_000, mode):
    _tensions, at_0 = _quiet(infer_forces, case_000, mode_pressure=mode, base_pressure=0.0)
    _tensions, at_1 = _quiet(infer_forces, case_000, mode_pressure=mode, base_pressure=1.0)
    assert at_1[0] == 1.0
    np.testing.assert_allclose([at_1[c] - at_0[c] for c in at_0 if c != 0], 1.0, rtol=0, atol=1e-12)
