"""The Laplace pressure reports as nan the cells its equations do not determine, and warns.

The regression this pins
------------------------
The Laplace methods write one equation per interface, `P_a - P_b = -2 H_ab T_ab`. An interface
all of whose vertices lie on junction lines has no curvature (`nan`) and no equation. When the
equations left do not connect a cell to the exterior, the least-squares solve used to return its
minimum-norm value for that cell -- exactly the exterior's pressure -- without a word. On the cut
benchmark case 000, cell 2 came back with the pressure 0.0.

What is asserted
----------------
* on a minimal synthetic mesh, a tetrahedral cell each of whose four faces is a separate
  one-triangle interface has no curvature on any of them: its pressure is `nan`, the warning
  names it, and the other cells' pressures are finite and the least-squares solution of the
  remaining equations (to 1e-12: since 0.7.0 the exterior's pin is eliminated rather than solved
  as a row, which changes the last bits), in both Laplace modes;
* an infinite curvature is treated like `nan` (the equation is dropped), not passed to the solve;
* a mesh with no undetermined cell gets no warning, and its pressures are unchanged;
* on the cut benchmark case 000 (`tests/data/cut_benchmark_case_000_mesh.npz`: the benchmark's
  mask 000 cut in half along axis 0, meshed with dw3d's defaults at commit 99342c3), cell 2 is
  `nan` and nothing else is, through `infer_pressures` and through `infer_forces`;
* the variational method is untouched: it returns a number for every cell.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy import linalg

from foambryo import (
    PressureComputationMethod,
    UndeterminedPressureWarning,
    infer_forces,
    infer_pressures,
    infer_pressures_and_residuals,
)
from foambryo.dcel import DcelData
from foambryo.pressure_inference import _build_matrix_laplace
from foambryo_benchmarks import phantoms as ph

LAPLACE_MODES = (PressureComputationMethod.Laplace, PressureComputationMethod.WeightedLaplace)
CUT_CASE = Path(__file__).parent / "data" / "cut_benchmark_case_000_mesh.npz"


def _oriented(triangle: list[int], points: np.ndarray, labels: tuple[int, int], cell: int, centre: np.ndarray) -> list:
    """The triangle wound so that its normal points into the higher label, `cell` lying at `centre`."""
    a, b, c = (points[i] for i in triangle)
    normal = np.cross(b - a, c - a)
    towards_cell = np.dot(normal, centre - (a + b + c) / 3) > 0
    if towards_cell != (cell == max(labels)):
        triangle = [triangle[0], triangle[2], triangle[1]]
    return [*triangle, min(labels), max(labels)]


def _tetrahedron_with_three_capped_faces() -> DcelData:
    """Cell 2 is a tetrahedron ABCD; cells 1, 3 and 4 cap its faces ABC, ABD, ACD; BCD faces the exterior.

    Every face of cell 2 is its own one-triangle interface, all of whose vertices lie on junction
    lines, so none of cell 2's four interfaces has a curvature. Each cap's three exterior faces
    meet at its apex, which lies on no junction, so the caps' interfaces with the exterior do.
    """
    tetra = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
    centre = tetra.mean(axis=0)
    points = [*tetra]
    faces = {1: (0, 1, 2), 3: (0, 1, 3), 4: (0, 2, 3)}
    rows = []
    for cap, face in faces.items():
        corners = tetra[list(face)]
        middle = corners.mean(axis=0)
        apex = middle + 1.2 * (middle - centre) / np.linalg.norm(middle - centre)
        points.append(apex)
        p = len(points) - 1
        cap_centre = (corners.sum(axis=0) + apex) / 4
        rows.append(_oriented(list(face), np.array(points), (2, cap), 2, centre))
        for i, j in ((0, 1), (1, 2), (2, 0)):
            rows.append(_oriented([face[i], face[j], p], np.array(points), (0, cap), cap, cap_centre))
    rows.append(_oriented([1, 2, 3], np.array(points), (0, 2), 2, centre))
    rows = np.array(rows, dtype=np.int64)
    return DcelData(np.array(points), rows[:, :3], rows[:, 3:])


def _unit_tensions(mesh: DcelData) -> dict[tuple[int, int], float]:
    return dict.fromkeys(mesh.compute_areas_interfaces(), 1.0)


def _least_squares_of_the_remaining_equations(mesh: DcelData, tensions: dict, weighted: bool) -> np.ndarray:
    curvatures = mesh.compute_curvatures_interfaces(weighted=weighted)
    matrix, rhs = _build_matrix_laplace(
        curvatures,
        mesh.compute_areas_interfaces(),
        tensions,
        mesh.n_materials,
        mesh.materials,
        0,
    )
    return linalg.lstsq(matrix, rhs)[0]


def _bits(x: float) -> bytes:
    return np.float64(x).tobytes()


def test_the_synthetic_cell_has_no_curvature_on_any_interface() -> None:
    mesh = _tetrahedron_with_three_capped_faces()
    for weighted in (False, True):
        curvatures = mesh.compute_curvatures_interfaces(weighted=weighted)
        assert all(math.isnan(curvatures[k]) for k in ((1, 2), (2, 3), (2, 4), (0, 2)))
        assert all(math.isfinite(curvatures[k]) for k in ((0, 1), (0, 3), (0, 4)))


@pytest.mark.parametrize("mode", LAPLACE_MODES)
def test_an_undetermined_cell_is_nan_and_named_in_the_warning(mode: PressureComputationMethod) -> None:
    mesh = _tetrahedron_with_three_capped_faces()
    tensions = _unit_tensions(mesh)
    with pytest.warns(UndeterminedPressureWarning, match=r"1 cell\(s\) is undetermined and reported as nan: 2\."):
        relative, pressures, _ = infer_pressures_and_residuals(mesh, tensions, mode=mode)
    assert math.isnan(pressures[2])
    assert math.isnan(relative[list(mesh.materials).index(2)])
    expected = _least_squares_of_the_remaining_equations(
        mesh,
        tensions,
        mode == PressureComputationMethod.WeightedLaplace,
    )
    for cell in (1, 3, 4):
        assert math.isfinite(pressures[cell])
        assert pressures[cell] == pytest.approx(expected[list(mesh.materials).index(cell)], rel=1e-12, abs=1e-14)
    assert pressures[0] == 0


@pytest.mark.parametrize("mode", LAPLACE_MODES)
def test_an_infinite_curvature_is_dropped_like_a_nan(monkeypatch: pytest.MonkeyPatch, mode) -> None:
    mesh = DcelData(*ph.doublet_mesh(0.8, r0=1.0, n_rings=24, n_phi=48))
    tensions = _unit_tensions(mesh)
    original = mesh.compute_curvatures_interfaces
    results = {}
    for name in ("nan", "inf", "-inf"):

        def with_one_bad_curvature(weighted: bool = True, bad: float = float(name)) -> dict:
            curvatures = dict(original(weighted=weighted))
            curvatures[(1, 2)] = bad
            return curvatures

        monkeypatch.setattr(mesh, "compute_curvatures_interfaces", with_one_bad_curvature)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UndeterminedPressureWarning)  # (1, 2) dropped, nothing disconnected
            results[name] = infer_pressures(mesh, tensions, mode=mode)
    assert all(math.isfinite(v) for v in results["nan"].values())
    for name in ("inf", "-inf"):
        assert {k: _bits(v) for k, v in results[name].items()} == {k: _bits(v) for k, v in results["nan"].items()}


@pytest.mark.parametrize("mode", LAPLACE_MODES)
def test_no_warning_and_no_change_where_every_cell_is_determined(mode: PressureComputationMethod) -> None:
    mesh = DcelData(*ph.doublet_mesh(0.8, r0=1.0, n_rings=24, n_phi=48))
    tensions = _unit_tensions(mesh)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UndeterminedPressureWarning)
        pressures = infer_pressures(mesh, tensions, mode=mode)
    expected = _least_squares_of_the_remaining_equations(
        mesh,
        tensions,
        mode == PressureComputationMethod.WeightedLaplace,
    )
    for i, cell in enumerate(mesh.materials[1:]):
        assert pressures[cell] == pytest.approx(expected[i + 1], rel=1e-12, abs=1e-14)


def _cut_case() -> DcelData:
    data = np.load(CUT_CASE)
    return DcelData(data["points"], data["triangles"], data["labels"])


@pytest.mark.parametrize("mode", LAPLACE_MODES)
def test_the_cut_benchmark_case_000_reports_cell_2_as_nan_not_zero(mode: PressureComputationMethod) -> None:
    mesh = _cut_case()
    with pytest.warns(UndeterminedPressureWarning, match=r": 2\."):
        pressures = infer_pressures(mesh, _unit_tensions(mesh), mode=mode)
    assert math.isnan(pressures[2])
    assert all(math.isfinite(p) for c, p in pressures.items() if c != 2)


def test_infer_forces_passes_the_nan_through() -> None:
    mesh = _cut_case()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # tension-side warnings of a cut mesh are not under test here
        tensions, pressures = infer_forces(mesh, mode_pressure=PressureComputationMethod.Laplace)
    assert all(math.isfinite(t) for t in tensions.values())
    assert math.isnan(pressures[2])
    assert all(math.isfinite(p) for c, p in pressures.items() if c != 2)


def test_the_variational_method_still_returns_a_number_for_every_cell() -> None:
    mesh = _cut_case()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UndeterminedPressureWarning)
        pressures = infer_pressures(mesh, _unit_tensions(mesh), mode=PressureComputationMethod.Variational)
    assert all(math.isfinite(p) for p in pressures.values())
