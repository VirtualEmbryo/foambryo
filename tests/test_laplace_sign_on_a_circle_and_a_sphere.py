"""Laplace's law has the sign of physics in both dimensions: a cell in the medium has the higher pressure.

The two docstrings write Laplace's law in different forms: 3D `P_a - P_b = -2 H_ab T_ab` (normal from `a` into `b`,
the orientation rule of the triangles), 2D `gamma kappa = p_left - p_right` (`kappa > 0` when the circle's centre is on
the left of the interface's traversal in `(x, y) = (column, row)`; the edges follow the same rule as the triangles in
index order, and the traversal is chosen with `m1` on its left in `(x, y)`).
Both must mean: the pressure is higher on the side that holds the centre of curvature, by `2 T / R` for a sphere and
`gamma / R` for a circle. Checked end to end, from a mask through the reconstruction's own orientation to the inferred
pressures, on one cell in the medium with a tension of 1: a ball of radius 9 voxels (both 3-D methods, the variational
and the Laplace ones) and a disk of radius 20 pixels. The relative tolerance (15 % in 3D, 5 % in 2D) is the
discretisation of a voxelised sphere and a pixelised circle at that radius, not a fitted margin: measured 4.5 % and
5.1 % above 2/R in 3D, 0.2 % below 1/R in 2D.
"""

from __future__ import annotations

import numpy as np
import pytest

import foambryo
from foambryo import PressureComputationMethod

RADIUS_3D = 9.0
RADIUS_2D = 20.0


@pytest.fixture(scope="module")
def sphere() -> foambryo.DcelData:
    n = 28
    z, y, x = np.indices((n, n, n))
    c = (n - 1) / 2
    ball = ((z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2 <= RADIUS_3D**2).astype(np.int64)
    return foambryo.dcel_mesh_from_segmentation_mask(ball)


@pytest.fixture(scope="module")
def circle() -> foambryo.PlanarMesh:
    n = 60
    r, c = np.indices((n, n))
    centre = (n - 1) / 2
    disk = ((r - centre) ** 2 + (c - centre) ** 2 <= RADIUS_2D**2).astype(np.int64)
    return foambryo.dcel_mesh_from_segmentation_mask(disk)


@pytest.mark.parametrize("mode", [PressureComputationMethod.Variational, PressureComputationMethod.Laplace])
def test_a_sphere_has_the_higher_pressure_inside_by_two_tension_over_radius(sphere, mode):
    pressures = foambryo.infer_pressures(sphere, {(0, 1): 1.0}, mode)
    difference = pressures[1] - pressures[0]
    assert difference > 0
    assert difference == pytest.approx(2.0 / RADIUS_3D, rel=0.15)


def test_a_circle_has_the_higher_pressure_inside_by_tension_over_radius(circle):
    pressures = foambryo.infer_pressures(circle, {(0, 1): 1.0})
    difference = pressures[1] - pressures[0]
    assert difference > 0
    assert difference == pytest.approx(1.0 / RADIUS_2D, rel=0.05)


def test_the_circle_s_curvature_sign_follows_dw2d_s_orientation(circle):
    """The medium (`m1 = 0`) is on the left of the traversal in `(x, y)`, so the centre is on the right: kappa < 0."""
    (arc,) = foambryo.interface_arcs(circle)
    assert arc.materials == (0, 1) and arc.left == 0 and arc.right == 1
    assert arc.curvature < 0
    assert -1.0 / arc.curvature == pytest.approx(RADIUS_2D, rel=0.05)
