"""The curvature of a 2-D mesh's interfaces and the mean stress of its cells, on exact cases.

Definitions and derivation in `foambryo._planar_curvature_and_stress`. Two kinds of exact case:

* **a circle**: one cell in the medium bounded by `n` vertices at equal angles on a circle of radius `R`. The vertex
  curvature `|K| = 2 sin(phi / 2) / l_dual` is exactly `1/R` there whatever `n`; the circle fit is exact; the cell's
  area is the polygon's, `(n/2) R^2 sin(2 pi / n)`; and its mean stress is exactly `(-p + gamma / (R cos(pi / n))) I`,
  since `sum_e l_e t_e t_e = (n R sin(pi / n)) I` for a regular polygon.
* **an equilibrium foam**: an 8-cell Möbius tiling (`_random_tiling`, the global fit's tests) with its exact tensions
  `T_ab` and pressures `p_k = A_k`, sampled on its arcs at equal angular steps (`exact_circular_arc_foams`). Laplace's
  law holds on every interface, `p_a - p_b = -T_ab kappa_ab`; the tissue, a body free of external load (medium at
  pressure 0), carries no total stress, `sum_c A_c sigma_c = 0`, up to replacing each arc by its chords. That
  replacement changes `integral t t dl` over an arc of angle `delta` by at most `delta^2 / 12` of its length
  (perpendicular to the chord: `r (delta - sin delta) / 2` against `r delta`), and the areas by the same order: the
  bound below, with `delta = step / r_min`. Measured: 0.54 to 0.58 of it on the three foams at steps 0.4 to 0.025 px,
falling as `step^2`.

And the sign shared with 3D: a disk meshed from a mask has `kappa_01 > 0`, as a ball has `H_01 > 0`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import foambryo
from foambryo import _two_dimensional as fb2
from foambryo._cell_stress import cell_mean_stress_tensors, split_principal_stresses
from foambryo._planar_curvature_and_stress import cell_areas_and_centroids
from foambryo.curvature import (
    compute_curvature_interfaces,
    compute_curvature_vertices_cotan,
    compute_gaussian_curvature_vertices,
)

from .exact_circular_arc_foams import (
    DEFAULT_STEP,
    _circle,
    exact_arc_mesh,
    exact_tensions_and_pressures,
    regular_polygon_cell,
)
from .test_two_dimensional_global_fit import _random_tiling

WINDOW = (150.0, 250.0, 150.0, 250.0)
SEEDS = (0, 1, 2)


@pytest.fixture(scope="module", params=SEEDS)
def foam(request):
    v = _random_tiling(8, request.param)
    tensions, pressures = exact_tensions_and_pressures(v)
    mesh = exact_arc_mesh(v, WINDOW)
    r_min = min(_circle(v, a, b)[1] for a, b in {tuple(p) for p in mesh.labels.tolist()})
    return mesh, tensions, pressures, r_min


# ---------------------------------------------------------------------------------------------------- a circle
@pytest.mark.parametrize(("radius", "n"), [(3.0, 7), (10.0, 64), (25.0, 200)])
def test_the_vertex_curvature_is_one_over_r_on_a_circle_sampled_at_equal_steps(radius, n):
    mesh = regular_polygon_cell(radius, n, centre=(40.0, 60.0))
    curvature, inverse_dual, vectors = compute_curvature_vertices_cotan(mesh)
    np.testing.assert_allclose(curvature, 1.0 / radius, rtol=1e-12)
    np.testing.assert_allclose(inverse_dual, 1.0 / (2 * radius * math.sin(math.pi / n)), rtol=1e-12)
    towards_centre = np.array([40.0, 60.0]) - mesh.points
    cosine = np.sum(vectors * towards_centre, axis=1) / (curvature * np.hypot(*towards_centre.T))
    np.testing.assert_allclose(cosine, 1.0, rtol=1e-12)  # K points to the centre


@pytest.mark.parametrize("reading", ["arc", "local"])
@pytest.mark.parametrize("weighted", [True, False])
def test_a_cell_in_the_medium_has_plus_one_over_r_with_either_reading(reading, weighted):
    mesh = regular_polygon_cell(10.0, 64)
    kappa = compute_curvature_interfaces(mesh, weighted, reading=reading)
    assert kappa.keys() == {(0, 1)}
    assert kappa[(0, 1)] == pytest.approx(0.1, rel=1e-12)  # the centre on label 1's side: positive, as in 3D


@pytest.mark.parametrize("n", [8, 64])
def test_one_cell_has_the_closed_form_area_centroid_and_mean_stress(n):
    radius, tension, pressure = 10.0, 1.3, 0.2
    mesh = regular_polygon_cell(radius, n, centre=(5.0, -3.0))
    areas, centroids = cell_areas_and_centroids(mesh)
    assert areas[1] == pytest.approx(0.5 * n * radius**2 * math.sin(2 * math.pi / n), rel=1e-13)
    np.testing.assert_allclose(centroids[1], [5.0, -3.0], atol=1e-12)
    sigma = cell_mean_stress_tensors(mesh, {(0, 1): tension}, {0: 0.0, 1: pressure}, areas, [1])[1]
    expected = -pressure + tension / (radius * math.cos(math.pi / n))
    np.testing.assert_allclose(sigma, expected * np.identity(2), rtol=0, atol=1e-14)
    # at its Laplace pressure the continuum cell carries no stress: the polygon's is the chord's excess only
    at_laplace = cell_mean_stress_tensors(mesh, {(0, 1): tension}, {0: 0.0, 1: tension / radius}, areas, [1])[1]
    assert np.max(np.abs(at_laplace)) == pytest.approx(tension / radius * (1 / math.cos(math.pi / n) - 1), rel=1e-9)


# ---------------------------------------------------------------------------------------------------- a foam
@pytest.mark.parametrize("reading", ["arc", "local"])
def test_laplaces_law_holds_with_the_exact_tensions_and_pressures(foam, reading):
    mesh, tensions, pressures, _ = foam
    kappa = compute_curvature_interfaces(mesh, reading=reading)
    assert len(kappa) >= 14
    for (a, b), value in kappa.items():
        expected = -(pressures[a] - pressures[b]) / tensions[(a, b)]
        assert value == pytest.approx(expected, rel=1e-7), (a, b)


def test_the_curvature_is_the_inferences_arc_curvature_with_the_sign_of_its_left_material(foam):
    mesh, _, _, _ = foam
    kappa = compute_curvature_interfaces(mesh, reading="arc")
    pieces: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for arc in fb2.interface_arcs(mesh):
        a, b = arc.materials
        pieces.setdefault((a, b), []).append((arc.length, -arc.curvature if arc.left == a else arc.curvature))
    assert pieces.keys() == kappa.keys()
    for pair, found in pieces.items():
        expected = sum(length * value for length, value in found) / sum(length for length, _ in found)
        assert kappa[pair] == pytest.approx(expected, rel=1e-14), pair


@pytest.mark.filterwarnings("ignore::foambryo.JunctionGraphGaugeWarning")  # some foams hold a cell alone in the medium
def test_laplaces_law_holds_with_the_inferred_forces(foam):
    mesh, _, _, _ = foam
    tensions, pressures = foambryo.infer_forces(mesh)
    kappa = compute_curvature_interfaces(mesh)
    determined = [pair for pair in kappa if np.isfinite(tensions[pair])]
    assert len(determined) >= len(kappa) - 1  # a cell alone in the medium has no junction, hence no tension
    for a, b in determined:
        value = kappa[(a, b)]
        assert pressures[a] - pressures[b] == pytest.approx(-tensions[(a, b)] * value, rel=1e-7, abs=1e-12)


def test_the_tissue_carries_no_total_stress(foam):
    mesh, tensions, pressures, r_min = foam
    areas, _ = cell_areas_and_centroids(mesh)
    assert all(area > 0 for area in areas.values())
    sigma = cell_mean_stress_tensors(mesh, tensions, pressures, areas, list(areas))
    total = sum(areas[c] * sigma[c] for c in areas)
    lengths = np.hypot(*(mesh.points[mesh.segments[:, 1]] - mesh.points[mesh.segments[:, 0]]).T)
    scale = float(sum(lengths[k] * tensions[tuple(pair)] for k, pair in enumerate(mesh.labels.tolist())))
    bound = (DEFAULT_STEP / r_min) ** 2 / 12
    assert np.max(np.abs(total)) / scale < bound
    # each tensor alone is not zero: the cells pull on one another
    assert max(np.max(np.abs(s)) for s in sigma.values()) > 1e3 * bound * max(abs(p) for p in pressures.values())


def test_a_shared_interface_is_split_between_its_two_cells(foam):
    mesh, tensions, pressures, _ = foam
    areas, _ = cell_areas_and_centroids(mesh)
    halves = cell_mean_stress_tensors(mesh, tensions, pressures, areas, list(areas))
    whole = cell_mean_stress_tensors(mesh, tensions, pressures, areas, list(areas), lumen_materials=range(100))
    # counted in full on both sides, every cell-cell interface adds its stress twice: the total moves off zero
    total_halves = sum(areas[c] * halves[c] for c in areas)
    total_whole = sum(areas[c] * whole[c] for c in areas)
    assert np.max(np.abs(total_whole)) > 1e3 * np.max(np.abs(total_halves))


def test_the_principal_split_takes_a_two_by_two_tensor():
    tensor = np.array([[-1.0, 0.5], [0.5, 2.0]])
    axes = split_principal_stresses(tensor)
    assert axes.compression.shape == axes.tension.shape == (2, 2)
    values = np.linalg.eigvalsh(tensor)
    assert np.linalg.norm(axes.compression[:, 0]) == pytest.approx(-values[0])
    assert np.linalg.norm(axes.tension[:, 1]) == pytest.approx(values[1])
    np.testing.assert_array_equal(axes.compression[:, 1], 0.0)
    np.testing.assert_array_equal(axes.tension[:, 0], 0.0)


def test_an_undetermined_tension_gives_an_undetermined_tensor(foam):
    mesh, tensions, pressures, _ = foam
    areas, _ = cell_areas_and_centroids(mesh)
    pair = next(p for p in tensions if p[0] != 0 and p in {tuple(q) for q in mesh.labels.tolist()})
    sigma = cell_mean_stress_tensors(mesh, {**tensions, pair: math.nan}, pressures, areas, list(areas))
    assert np.all(np.isnan(sigma[pair[0]]))
    assert np.all(np.isnan(sigma[pair[1]]))
    others = [c for c in areas if c not in pair]
    assert all(np.all(np.isfinite(sigma[c])) for c in others)


def test_physical_units_scale_curvature_and_stress_by_the_pixel_size():
    h = 0.25
    pixels = regular_polygon_cell(10.0, 64)
    physical = foambryo.PlanarMesh(pixels.points * h, pixels.segments, pixels.labels, image_border_vertices=[])
    assert compute_curvature_interfaces(physical)[(0, 1)] == pytest.approx(compute_curvature_interfaces(pixels)[(0, 1)]
                                                                           / h, rel=1e-12)
    areas_px, _ = cell_areas_and_centroids(pixels)
    areas_h, _ = cell_areas_and_centroids(physical)
    sigma_px = cell_mean_stress_tensors(pixels, {(0, 1): 1.0}, {0: 0.0, 1: 0.05}, areas_px, [1])[1]
    sigma_h = cell_mean_stress_tensors(physical, {(0, 1): 1.0}, {0: 0.0, 1: 0.05 / h}, areas_h, [1])[1]
    np.testing.assert_allclose(sigma_h, sigma_px / h, rtol=1e-12)


# ---------------------------------------------------------------------------------------------------- 2D and 3D
def test_a_disk_from_a_mask_and_a_ball_have_the_same_sign():
    n = 60
    r, c = np.indices((n, n))
    disk = ((r - (n - 1) / 2) ** 2 + (c - (n - 1) / 2) ** 2 <= 20.0**2).astype(np.int64)
    kappa = compute_curvature_interfaces(foambryo.dcel_mesh_from_segmentation_mask(disk))
    # the discretisation of a pixelised circle of radius 20 (R203's Laplace sign test: 0.2 % below 1/R, 5 % allowed)
    assert kappa[(0, 1)] == pytest.approx(1 / 20.0, rel=0.05)
    m = 28
    z, y, x = np.indices((m, m, m))
    centre = (m - 1) / 2
    ball = ((z - centre) ** 2 + (y - centre) ** 2 + (x - centre) ** 2 <= 9.0**2).astype(np.int64)
    curvature_3d = compute_curvature_interfaces(foambryo.dcel_mesh_from_segmentation_mask(ball))
    assert curvature_3d[(0, 1)] == pytest.approx(1 / 9.0, rel=0.05)  # measured 0.109 (2 % below)


def test_readings_and_quantities_without_a_meaning_in_the_dimension_raise_naming_them():
    mesh = regular_polygon_cell(10.0, 16)
    with pytest.raises(ValueError, match="reading"):
        compute_curvature_interfaces(mesh, reading="cotan")
    with pytest.raises(ValueError, match="no 2-D counterpart"):
        compute_gaussian_curvature_vertices(mesh)
    m = 20
    z, y, x = np.indices((m, m, m))
    ball = ((z - 9.5) ** 2 + (y - 9.5) ** 2 + (x - 9.5) ** 2 <= 36).astype(np.int64)
    with pytest.raises(ValueError, match="reading"):
        compute_curvature_interfaces(foambryo.dcel_mesh_from_segmentation_mask(ball), reading="arc")
