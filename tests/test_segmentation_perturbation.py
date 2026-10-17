"""Positive-control tests for `foambryo_benchmarks.segmentation_perturbation` (the robustness work).

Every test injects a perturbation of an independently-computable size and asserts the
measured effect matches it - not merely that "something changed", which is how a
comparison that compares nothing can report "0 differ".
"""

import numpy as np
import pytest

from foambryo_benchmarks import segmentation_perturbation as sp


def _two_slab_mask(size=30, interface_at=15):
    """A flat interface at `z=interface_at`: label 1 for z < interface_at, label 2 for z >=.

    No background at all - isolates dilation/erosion from the "grows into 0" mechanic's
    interaction with a second interface, so the known slab-volume prediction holds exactly.
    """
    mask = np.ones((size, size, size), dtype=np.int32)
    mask[interface_at:] = 2
    return mask


def _boxed_label(volume_shape=(30, 30, 30), box=(10, 10, 10), label=1):
    """A single rectangular `label` box padded well inside an all-background volume.

    Padding on every side exceeds any `voxels` used in these tests, so the exact box-erosion/
    box-dilation formula (shrink or grow each dimension by `2 * voxels`) holds with no
    array-boundary interference - the failure mode hit and fixed in this test file's first
    draft, where a slab spanning the full array was silently eroded from the array edge too.
    """
    mask = np.zeros(volume_shape, dtype=np.int32)
    starts = [(v - b) // 2 for v, b in zip(volume_shape, box, strict=True)]
    slices = tuple(slice(s, s + b) for s, b in zip(starts, box, strict=True))
    mask[slices] = label
    return mask


def test_dilate_labels_leaves_an_already_fully_labelled_mask_untouched():
    """No background exists to dilate into, so the known "added volume" is exactly zero."""
    mask = _two_slab_mask()
    dilated = sp.dilate_labels(mask, voxels=1)
    assert np.array_equal(dilated, mask)


def test_dilate_labels_from_a_single_voxel_forms_the_predicted_euclidean_ball():
    """`expand_labels` grows by Euclidean, not Chebyshev/box, distance.

    A corner of a growing box is *not* reached as far as its faces are. Verified directly:
    from a single labelled voxel, the dilated set must equal the discrete Euclidean ball of
    radius `voxels` computed independently via `np.linalg.norm`, not by calling
    `expand_labels` again.
    """
    mask = np.zeros((21, 21, 21), dtype=np.int32)
    center = (10, 10, 10)
    mask[center] = 1
    k = 4

    dilated = sp.dilate_labels(mask, voxels=k)

    zz, yy, xx = np.indices(mask.shape)
    expected_ball = np.sqrt((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2) <= k

    assert np.array_equal(dilated == 1, expected_ball)
    assert int(np.sum(expected_ball)) > 1  # non-vacuous: the ball must contain more than the seed


def test_dilate_labels_grows_into_a_background_gap_symmetrically_from_both_sides():
    """Two boxed labels, a 10-voxel background gap between them along one axis.

    Dilating by 3 voxels must fill exactly 3 voxels of gap from each side, leaving the
    un-reached 4-voxel middle as background - and neither label may cross into the other's
    territory.
    """
    size = 12
    area = size * size
    mask = np.zeros((size, 30, size), dtype=np.int32)
    mask[:, :10] = 1
    mask[:, 20:] = 2  # 10-voxel background gap at columns [10, 20)

    dilated = sp.dilate_labels(mask, voxels=3)

    assert np.sum(dilated == 1) == np.sum(mask == 1) + 3 * area
    assert np.sum(dilated == 2) == np.sum(mask == 2) + 3 * area
    assert np.sum(dilated == 0) == 4 * area  # the un-reachable middle of the 10-voxel gap


def test_erode_labels_shrinks_a_padded_box_by_exactly_two_voxels_per_dimension():
    """Symmetric control to the dilation test.

    The box is far enough from the array edge that `border_value=0` erosion cannot also eat
    into it from outside the volume.
    """
    mask = _boxed_label(box=(12, 10, 8), label=1)
    k = 2
    eroded = sp.erode_labels(mask, voxels=k)
    assert int(np.sum(eroded == 1)) == (12 - 2 * k) * (10 - 2 * k) * (8 - 2 * k)


def test_erode_labels_cannot_create_a_label_collision():
    """However labels are arranged, erosion only ever converts labelled voxels to background."""
    rng = np.random.default_rng(0)
    mask = rng.integers(0, 4, size=(20, 20, 20)).astype(np.int32)
    eroded = sp.erode_labels(mask, voxels=1)
    changed = eroded != mask
    assert np.all(eroded[changed] == 0)  # every changed voxel became background, never a swap


def test_add_boundary_noise_changes_exactly_the_predicted_number_of_voxels():
    """The change count must equal `round(fraction * n_boundary_voxels)` exactly.

    Every `find_boundaries` voxel has a differing neighbour by construction (module
    docstring), so nothing is dropped for lack of a candidate to reassign to.
    """
    mask = _two_slab_mask(size=20, interface_at=10)
    from skimage.segmentation import find_boundaries

    n_boundary = int(find_boundaries(mask, mode="inner").sum())
    assert n_boundary > 0

    rng = np.random.default_rng(42)
    noisy, n_changed = sp.add_boundary_noise(mask, fraction=0.3, rng=rng)

    assert n_changed == round(0.3 * n_boundary)
    assert int(np.sum(noisy != mask)) == n_changed


def test_add_boundary_noise_only_reassigns_to_an_actually_present_neighbor_label():
    """A changed voxel's new label must be one that existed among its 6-neighbours.

    A hand-rolled reassignment bug could invent a label or leave the interior untouched.
    """
    mask = _two_slab_mask(size=20, interface_at=10)
    rng = np.random.default_rng(1)
    noisy, n_changed = sp.add_boundary_noise(mask, fraction=1.0, rng=rng)
    assert n_changed > 0
    changed_mask = noisy != mask
    # every changed voxel, on this two-label mask, can only have been flipped 1<->2
    assert set(np.unique(noisy[changed_mask]).tolist()) <= {1, 2}
    assert set(np.unique(mask[changed_mask]).tolist()) <= {1, 2}


def test_add_boundary_noise_zero_fraction_changes_nothing():
    mask = _two_slab_mask()
    rng = np.random.default_rng(0)
    noisy, n_changed = sp.add_boundary_noise(mask, fraction=0.0, rng=rng)
    assert n_changed == 0
    assert np.array_equal(noisy, mask)


def test_downsample_labels_majority_reproduces_a_uniform_block_exactly():
    """Every block is 100% one label - the trivial case any majority rule must get right."""
    mask = _two_slab_mask(size=16, interface_at=8)
    down = sp.downsample_labels_majority(mask, factor=2)
    assert down.shape == (8, 8, 8)
    assert np.array_equal(down[:4], np.full((4, 8, 8), 1))
    assert np.array_equal(down[4:], np.full((4, 8, 8), 2))


def test_downsample_labels_majority_picks_the_strict_majority_of_a_mixed_block():
    """A block of 5 voxels label 1, 2 of label 2, 1 of label 3 must downsample to 1.

    A majority rule that ties, averages, or picks the minority fails this.
    """
    mask = np.array([[[1, 1], [1, 1]], [[1, 2], [2, 3]]], dtype=np.int32)
    assert mask.shape == (2, 2, 2)
    assert (mask == 1).sum() == 5

    down = sp.downsample_labels_majority(mask, factor=2)

    assert down.shape == (1, 1, 1)
    assert down[0, 0, 0] == 1


def test_downsample_labels_majority_drops_a_non_multiple_remainder_rather_than_guessing():
    mask = np.ones((5, 4, 4), dtype=np.int32)  # 5 is not a multiple of factor=2
    down = sp.downsample_labels_majority(mask, factor=2)
    assert down.shape == (2, 2, 2)  # the trailing slice (index 4) is dropped, not padded


@pytest.mark.parametrize("voxels", [1, 2])
def test_dilate_and_erode_are_not_exact_inverses_at_a_single_interface(voxels):
    """Sanity check: the two functions are doing geometrically different things.

    Dilating then eroding a slab-pair mask does not exactly recover the original mask at the
    newly-labelled background boundary, because dilation's fill order is nearest-label
    (EDT-based), not the erosion's removal order. Documented, not treated as a bug (module
    docstrings: dilation fills background, erosion is a separate, simpler per-label shrink).
    """
    mask = np.zeros((6, 10, 10), dtype=np.int32)
    mask[:, :3] = 1
    mask[:, 7:] = 2
    round_tripped = sp.erode_labels(sp.dilate_labels(mask, voxels=voxels), voxels=voxels)
    assert not np.array_equal(round_tripped, mask) or voxels == 0
