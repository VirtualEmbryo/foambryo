"""`infer_tensions_over_seeds`: several dither seeds of one mask, combined.

Small synthetic masks only (a ball cut into three cells, whole or cropped by the image). What is pinned:

* `seeds=[42]` is the single-seed path, bit for bit, with both `combine` options and both readings;
* each seed's tensions are `infer_tensions` on that seed's own dw3d reconstruction;
* `combine="tensions"` is the per-interface mean of the seeds, `tension_seed_sd` their SD, and the flag counts;
* `max_workers` does not change the result;
* crop faces are handled as the single-seed path handles them;
* the argument checks and the experimental warning.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from dw3d import get_dithered_mesh_reconstruction_algorithm

from foambryo import (
    DEFAULT_ENSEMBLE_SEEDS,
    SeedEnsembleExperimentalWarning,
    TensionComputationMethod,
    dcel_mesh_from_multimaterial_mesh,
    dcel_mesh_from_segmentation_mask,
    infer_tensions,
    infer_tensions_over_seeds,
)


def _ball_in_three_cells(cropped: bool = False) -> np.ndarray:
    """Cells 1 and 2 side by side, cell 3 below both; cropped, the image cuts cells 1 and 2 (as in test_crop_faces)."""
    n, radius = 60, 20
    z, y, x = np.mgrid[:n, :n, :n]
    centre = n / 2
    inside = (z - centre) ** 2 + (y - centre) ** 2 + (x - centre) ** 2 <= radius**2
    mask = np.zeros((n, n, n), dtype=np.uint16)
    mask[inside & (y < centre) & (z >= centre - 8)] = 1
    mask[inside & (y >= centre) & (z >= centre - 8)] = 2
    mask[inside & (z < centre - 8)] = 3
    return np.ascontiguousarray(mask[: int(centre) + 10]) if cropped else mask


def _quiet(function, *args: object, **kwargs: object):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


def _keyed(tensions: dict) -> dict[tuple[int, int], float]:
    return {tuple(sorted(int(x) for x in k)): float(v) for k, v in tensions.items()}


def _bitwise_equal(a: dict, b: dict) -> bool:
    return set(a) == set(b) and all(a[k] == b[k] or (np.isnan(a[k]) and np.isnan(b[k])) for k in a)


@pytest.fixture(scope="module")
def mask() -> np.ndarray:
    return _ball_in_three_cells()


@pytest.fixture(scope="module")
def ensemble(mask):
    """Seeds 0, 1 and 2, combined over tensions."""
    return _quiet(infer_tensions_over_seeds, mask, seeds=(0, 1, 2))


def test_the_default_seed_list_is_zero_to_seven_and_excludes_the_single_seed_default():
    assert tuple(range(8)) == DEFAULT_ENSEMBLE_SEEDS
    assert 42 not in DEFAULT_ENSEMBLE_SEEDS


@pytest.mark.parametrize("combine", ["tensions", "angles"])
@pytest.mark.parametrize("reading", ["chord", "quadric"])
def test_seed_42_alone_is_the_single_seed_path_bitwise(mask, combine, reading):
    single = _keyed(_quiet(infer_tensions, _quiet(dcel_mesh_from_segmentation_mask, mask), 1, angle_reading=reading))
    result = _quiet(infer_tensions_over_seeds, mask, seeds=[42], angle_reading=reading, combine=combine)
    assert _bitwise_equal(result.tensions, single)
    assert _bitwise_equal(result.per_seed_tensions[42], single)
    assert all(np.isnan(v) for v in result.tension_seed_sd.values())
    assert all(np.isnan(v) for v in result.relative_tension_seed_sd.values())
    assert set(result.n_seeds_found.values()) == {1}
    assert not any(result.found_by_fewer_than_half_the_seeds.values())


def test_each_seed_is_the_dw3d_reconstruction_at_that_seed(mask, ensemble):
    algorithm = get_dithered_mesh_reconstruction_algorithm(min_distance=3, seed=1)
    points, triangles, labels = _quiet(algorithm.construct_mesh_from_segmentation_mask, mask)
    own = _keyed(_quiet(infer_tensions, dcel_mesh_from_multimaterial_mesh(points, triangles, labels)))
    assert _bitwise_equal(ensemble.per_seed_tensions[1], own)


def test_the_seeds_give_different_meshes(ensemble):
    assert ensemble.per_seed_tensions[0] != ensemble.per_seed_tensions[1]


def test_combine_tensions_is_the_mean_and_the_spread_is_the_sd_over_seeds(ensemble):
    assert ensemble.seeds == (0, 1, 2)
    assert ensemble.combine == "tensions"
    assert ensemble.mode == TensionComputationMethod.YoungDupre
    assert [r.seed for r in ensemble.reconstructions] == [0, 1, 2]
    for key in ensemble.interfaces:
        values = [t[key] for t in ensemble.per_seed_tensions.values() if key in t and np.isfinite(t[key])]
        assert ensemble.n_seeds_found[key] == len(values)
        assert ensemble.tensions[key] == np.mean(values)
        assert ensemble.tension_seed_sd[key] == np.std(values, ddof=1)
        assert ensemble.found_by_fewer_than_half_the_seeds[key] == (len(values) < len(ensemble.seeds) / 2)
    # every seed is exact-gauged (mean 1), so where all seeds found the same interfaces the ensemble mean is 1 too
    if set(ensemble.n_seeds_found.values()) == {3}:
        assert np.mean(list(ensemble.tensions.values())) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("combine", ["tensions", "angles"])
def test_parallel_workers_give_the_same_result(mask, combine):
    serial = _quiet(infer_tensions_over_seeds, mask, seeds=(0, 1), combine=combine)
    parallel = _quiet(infer_tensions_over_seeds, mask, seeds=(0, 1), combine=combine, max_workers=2)
    assert _bitwise_equal(serial.tensions, parallel.tensions)
    assert _bitwise_equal(serial.tension_seed_sd, parallel.tension_seed_sd)
    assert serial.n_seeds_found == parallel.n_seeds_found
    for seed in (0, 1):
        assert _bitwise_equal(serial.per_seed_tensions[seed], parallel.per_seed_tensions[seed])
    assert [r.warnings for r in serial.reconstructions] == [r.warnings for r in parallel.reconstructions]


def test_combine_angles_solves_once_on_the_averaged_angles(mask):
    with pytest.warns(SeedEnsembleExperimentalWarning):
        result = infer_tensions_over_seeds(mask, seeds=(0, 1, 2), combine="angles")
    assert result.combine == "angles"
    assert set(result.tensions) == set().union(*(set(t) for t in result.per_seed_tensions.values()))
    values = np.array(list(result.tensions.values()))
    assert np.all(np.isfinite(values))
    assert np.mean(values) == pytest.approx(1.0, abs=1e-12)  # one exact-gauged solve
    # not the mean of the tensions: the solve is non-linear in the angles
    means = {k: np.mean([t[k] for t in result.per_seed_tensions.values()]) for k in result.interfaces}
    assert any(result.tensions[k] != means[k] for k in result.interfaces)


def test_crop_faces_are_handled_as_on_a_single_seed():
    cropped = _ball_in_three_cells(cropped=True)
    single = _keyed(_quiet(infer_tensions, _quiet(dcel_mesh_from_segmentation_mask, cropped)))
    for combine in ("tensions", "angles"):
        result = _quiet(infer_tensions_over_seeds, cropped, seeds=[42], combine=combine)
        assert _bitwise_equal(result.tensions, single)
        assert result.reconstructions[0].n_crop_faces > 0
    physical = _quiet(infer_tensions_over_seeds, cropped, seeds=[42], physical_image_faces=("axis0_high",))
    assert physical.reconstructions[0].n_crop_faces == 0


def test_other_methods_combine_over_tensions(mask):
    result = _quiet(infer_tensions_over_seeds, mask, seeds=(0, 1), mode=TensionComputationMethod.ForceBalance)
    assert result.mode == TensionComputationMethod.ForceBalance
    mesh = _quiet(dcel_mesh_from_segmentation_mask, mask)
    single = _keyed(_quiet(infer_tensions, mesh, 1, TensionComputationMethod.ForceBalance))
    assert set(result.tensions) == set(single)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"seeds": []}, ValueError),
        ({"seeds": [1, 1]}, ValueError),
        ({"combine": "median"}, ValueError),
        ({"combine": "angles", "mode": TensionComputationMethod.ForceBalance}, ValueError),
        ({"angle_reading": "spline"}, ValueError),
        ({"max_workers": 0}, ValueError),
        ({"seeds": [0], "smoothing": True}, TypeError),
    ],
)
def test_arguments_are_checked(mask, kwargs, error):
    with pytest.raises(error):
        _quiet(infer_tensions_over_seeds, mask, **kwargs)


def test_the_relative_spread_is_the_sd_over_the_absolute_ensemble_tension(ensemble):
    assert set(ensemble.relative_tension_seed_sd) == set(ensemble.interfaces)
    for key in ensemble.interfaces:
        sd, tension = ensemble.tension_seed_sd[key], ensemble.tensions[key]
        assert ensemble.relative_tension_seed_sd[key] == sd / abs(tension)
        assert ensemble.relative_tension_seed_sd[key] >= 0
    # a few percent on this ball, as the seed spread is on meshes of large cells
    assert 0 < np.median(list(ensemble.relative_tension_seed_sd.values())) < 0.5


def test_the_relative_spread_is_nan_where_it_cannot_be_formed():
    from foambryo.seed_ensembles import _relative

    assert _relative(0.1, 2.0) == 0.05
    assert _relative(0.1, -2.0) == 0.05
    assert np.isnan(_relative(0.1, 0.0))
    assert np.isnan(_relative(float("nan"), 1.0))
    assert np.isnan(_relative(0.1, float("nan")))


@pytest.mark.parametrize("combine", ["tensions", "angles"])
def test_seed_42_with_a_spacing_is_the_single_seed_path_with_that_spacing_bitwise(mask, combine):
    spacing = (5 / 3, 1.0, 1.0)  # one longer axis: the mask is resampled to cubic voxels before meshing
    mesh = _quiet(dcel_mesh_from_segmentation_mask, mask, spacing=spacing)
    single = _keyed(_quiet(infer_tensions, mesh))
    result = _quiet(infer_tensions_over_seeds, mask, seeds=[42], spacing=spacing, combine=combine)
    assert _bitwise_equal(result.tensions, single)
    assert result.reconstructions[0].n_points == len(mesh.v)
    without = _quiet(infer_tensions_over_seeds, mask, seeds=[42], combine=combine)
    assert not _bitwise_equal(without.tensions, single)  # the spacing reached the reconstruction
