"""Pressure inference refuses a non-finite interface tension instead of returning a non-finite field.

The regression this pins
------------------------
An interface with no triple line -- a closed surface such as an engulfed cell, a sealed lumen, or a
fragment a segmentation left inside another cell -- has no angle from which its tension can be
measured, and `infer_tensions` reports it as `nan` by design (and warns). Before this guard,
`infer_pressures` accepted that `nan` without a word and returned a pressure field in which nearly
every cell was non-finite, because the variational pressure solve couples every cell to every
tension; `infer_forces` returned the same field. Measured on real embryo segmentations, all but one
cell's pressure came back non-finite on each of five such timepoints.

What is asserted
----------------
* a mesh with a provably closed interface makes `infer_forces` raise `UndeterminedTensionError`
  naming that interface, for every pressure method, while `infer_tensions` still returns the other
  tensions finite and the closed one as `nan` and still warns (unchanged behaviour);
* `infer_pressures` raises for a `nan` and for an infinite tension, whichever pressure method is
  asked for, and the error is a `ValueError`, so callers that caught the least-squares solve's
  generic error keep working;
* on a healthy mesh the guard changes nothing: the pressures are bitwise those of the private
  solvers called without it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from foambryo import JunctionGraphGaugeWarning, UndeterminedTensionError, infer_forces, infer_pressures, infer_tensions
from foambryo.dcel import DcelData
from foambryo.pressure_inference import (
    PressureComputationMethod,
    _infer_pressures_laplace,
    _infer_pressures_variational,
    infer_pressures_and_residuals,
)
from foambryo.tension_inference import TensionComputationMethod
from foambryo_benchmarks import phantoms as ph
from tests.test_junction_graph import _combine, _uv_sphere

ALL_METHODS = tuple(PressureComputationMethod)


def _doublet() -> DcelData:
    """A healthy analytic doublet: one junction-graph component, no closed interface."""
    return DcelData(*ph.doublet_mesh(0.8, r0=1.0, n_rings=24, n_phi=48))


def _doublet_with_a_closed_sphere() -> DcelData:
    """The doublet plus a detached closed sphere whose interface `(0, 5)` has no triple line."""
    doublet = ph.doublet_mesh(1.0, r0=1.0, n_rings=24, n_phi=48)
    sphere = _uv_sphere(label_outside=0, label_inside=5, center=np.array([0.0, 0.0, 50.0]), radius=1.0)
    return DcelData(*_combine(doublet, sphere))


@pytest.mark.parametrize("mode", ALL_METHODS)
def test_infer_forces_refuses_the_pressures_of_a_mesh_with_a_closed_interface(mode: PressureComputationMethod) -> None:
    mesh = _doublet_with_a_closed_sphere()
    with pytest.warns(JunctionGraphGaugeWarning), pytest.raises(UndeterminedTensionError) as raised:
        infer_forces(mesh, mode_pressure=mode)
    assert raised.value.interfaces == ((0, 5),)
    assert "(0, 5)" in str(raised.value)


def test_the_tensions_of_that_mesh_are_unchanged_nan_only_where_undetermined() -> None:
    mesh = _doublet_with_a_closed_sphere()
    with pytest.warns(JunctionGraphGaugeWarning):
        tensions = infer_tensions(mesh, mode=TensionComputationMethod.YoungDupre)
    assert math.isnan(tensions[(0, 5)])
    assert all(math.isfinite(tensions[key]) for key in ((0, 1), (0, 2), (1, 2)))


@pytest.mark.parametrize("mode", ALL_METHODS)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_infer_pressures_refuses_any_non_finite_tension(mode: PressureComputationMethod, bad: float) -> None:
    mesh = _doublet()
    tensions = infer_tensions(mesh, mode=TensionComputationMethod.YoungDupre)
    tensions[(1, 2)] = bad
    with pytest.raises(UndeterminedTensionError) as raised:
        infer_pressures(mesh, tensions, mode=mode)
    assert raised.value.interfaces == ((1, 2),)
    assert isinstance(raised.value, ValueError)
    with pytest.raises(UndeterminedTensionError):
        infer_pressures_and_residuals(mesh, tensions, mode=mode)


def test_the_guard_changes_nothing_where_every_tension_is_finite() -> None:
    mesh = _doublet()
    tensions = infer_tensions(mesh, mode=TensionComputationMethod.YoungDupre)
    guarded = {mode: infer_pressures_and_residuals(mesh, tensions, mode=mode) for mode in ALL_METHODS}
    unguarded = {
        PressureComputationMethod.Variational: _infer_pressures_variational(mesh, tensions),
        PressureComputationMethod.Laplace: _infer_pressures_laplace(mesh, tensions, weighted=False),
        PressureComputationMethod.WeightedLaplace: _infer_pressures_laplace(mesh, tensions, weighted=True),
    }
    for mode in ALL_METHODS:
        got, expected = guarded[mode][1], unguarded[mode][1]
        assert got.keys() == expected.keys()
        assert all(np.float64(got[k]).tobytes() == np.float64(expected[k]).tobytes() for k in got), mode
