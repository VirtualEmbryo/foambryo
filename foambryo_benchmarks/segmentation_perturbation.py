"""Segmentation-mask perturbations for the robustness benchmark.

The perturbations are morphological dilate and erode by +/-1 voxel and label-boundary noise,
chosen because they are closer to real-data failure than vertex-position noise. Every function here perturbs the
**segmentation mask** (`*_labels_filled.tif`, `int` labels, `0` = background/exterior) that
feeds `dw3d`, not the mesh `dw3d` produces — the perturbation happens *before* the pipeline
under test, exactly where a real segmentation error would.

**Determinism.** `add_boundary_noise` is the only stochastic one; it takes an explicit
`numpy.random.Generator` rather than touching the global RNG, so a case's noise is
reproducible given a seed and does not perturb any other call's randomness (the same class of
determinism defect `dw3d` removed when it retired its dithered point placement).

**Testing discipline.** Every function's docstring
states the *known magnitude* its positive control checks: `tests/test_segmentation_
perturbation.py` constructs a case where that magnitude is analytically predictable and
asserts the measured effect matches it, rather than merely asserting "something changed".
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_erosion
from skimage.segmentation import expand_labels, find_boundaries

_NEIGHBOR_OFFSETS: tuple[tuple[int, int, int], ...] = (
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
)


def dilate_labels(mask: NDArray[np.integer], voxels: int = 1) -> NDArray[np.integer]:
    """Grow every label into the background by `voxels`, simulating segmentation over-reach.

    Thin wrapper on `skimage.segmentation.expand_labels`: background (`0`) voxels within
    `voxels` of a labelled region are reassigned to the *nearest* label (ties broken by
    `scipy`'s EDT, deterministic). Already-labelled voxels are never touched, so this cannot
    create new label-label overlaps or change which of two touching cells "wins" a boundary
    voxel - it only consumes background. **Known magnitude for the positive control:** on a
    single flat interface far from any other feature, the added volume is `voxels * area`
    (a slab of thickness `voxels`), to `O(voxels/curvature_radius)`.

    Args:
        mask (NDArray[np.integer]): Segmentation mask, `0` = background.
        voxels (int, optional): Dilation distance in voxels. Defaults to 1.

    Returns:
        NDArray[np.integer]: The dilated mask, same shape and dtype.
    """
    return expand_labels(mask, distance=voxels).astype(mask.dtype)


def erode_labels(mask: NDArray[np.integer], voxels: int = 1) -> NDArray[np.integer]:
    """Shrink every label away from its boundary into the background by `voxels`.

    Per-label binary erosion (`scipy.ndimage.binary_erosion`): for each label, the voxels
    within `voxels` of that label's own boundary revert to `0`. Since erosion only *removes*
    voxels from a label's own region, two labels can never collide by construction - this is
    simpler than dilation, which is why it needs no tie-break. **Known magnitude for the
    positive control:** on a single flat interface, the removed volume from each side is
    `voxels * area`, symmetric with `dilate_labels`.

    Args:
        mask (NDArray[np.integer]): Segmentation mask, `0` = background.
        voxels (int, optional): Erosion distance in voxels. Defaults to 1.

    Returns:
        NDArray[np.integer]: The eroded mask, same shape and dtype.
    """
    out = mask.copy()
    for label in np.unique(mask):
        if label == 0:
            continue
        binary = mask == label
        eroded = binary_erosion(binary, iterations=voxels, border_value=0)
        out[binary & ~eroded] = 0
    return out


def add_boundary_noise(
    mask: NDArray[np.integer],
    fraction: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.integer], int]:
    """Reassign a random `fraction` of label-boundary voxels to a differing neighbour's label.

    Simulates the voxel-scale misclassification a real segmentation algorithm makes at
    material interfaces (as opposed to `dilate_labels`/`erode_labels`'s systematic bias).
    Boundary voxels are `skimage.segmentation.find_boundaries(mask, mode="inner")`: voxels
    that have at least one 6-connected neighbour with a different label. Each selected voxel
    is reassigned to one of its differing neighbours' labels, chosen uniformly at random by
    `rng`. **Known magnitude for the positive control:** every `find_boundaries` voxel has at
    least one differing neighbour by construction, so the number of voxels actually changed
    equals `round(fraction * n_boundary_voxels)` exactly - the second return value - and the
    test asserts this count, not just "some voxels changed".

    Args:
        mask (NDArray[np.integer]): Segmentation mask, `0` = background.
        fraction (float): Fraction of boundary voxels to reassign, in `[0, 1]`.
        rng (numpy.random.Generator): Explicit generator - never the global numpy RNG, so a
            case's noise is reproducible independent of call order elsewhere (see module
            docstring).

    Returns:
        tuple[NDArray[np.integer], int]: `(noisy_mask, n_voxels_changed)`.
    """
    boundary = find_boundaries(mask, mode="inner")
    coords = np.argwhere(boundary)
    n_select = round(fraction * len(coords))
    selected = coords[rng.permutation(len(coords))[:n_select]]

    out = mask.copy()
    shape = mask.shape
    n_changed = 0
    for z, y, x in selected:
        current = mask[z, y, x]
        neighbor_labels = []
        for dz, dy, dx in _NEIGHBOR_OFFSETS:
            zz, yy, xx = z + dz, y + dy, x + dx
            if 0 <= zz < shape[0] and 0 <= yy < shape[1] and 0 <= xx < shape[2]:
                neighbor_label = mask[zz, yy, xx]
                if neighbor_label != current:
                    neighbor_labels.append(neighbor_label)
        if neighbor_labels:
            out[z, y, x] = neighbor_labels[rng.integers(len(neighbor_labels))]
            n_changed += 1
    return out, n_changed


def downsample_labels_majority(mask: NDArray[np.integer], factor: int) -> NDArray[np.integer]:
    """Downsample a label image by `factor` per axis, each output voxel the majority label.

    Non-overlapping `factor**3` blocks, majority vote per block (ties broken by the lower
    label id via `argmax`'s first-max convention) - not nearest-neighbour decimation
    (`mask[::factor]`), which can silently drop a label that never lands on a sampled voxel.
    Any trailing voxels that do not fill a whole block are dropped, matching how a real
    lower-resolution acquisition would not include a fractional voxel either. **Known
    magnitude for the positive control:** a block that is more than half one label is
    guaranteed, by the majority rule, to output that label regardless of the rest of the
    block's composition.

    Args:
        mask (NDArray[np.integer]): Segmentation mask, `0` = background.
        factor (int): Downsampling factor per axis (e.g. `2`, `4`).

    Returns:
        NDArray[np.integer]: Downsampled mask, shape `tuple(s // factor for s in mask.shape)`.
    """
    trimmed_shape = tuple((s // factor) * factor for s in mask.shape)
    trimmed = mask[: trimmed_shape[0], : trimmed_shape[1], : trimmed_shape[2]]
    new_shape = tuple(s // factor for s in trimmed_shape)
    reshaped = trimmed.reshape(new_shape[0], factor, new_shape[1], factor, new_shape[2], factor)

    labels = np.unique(mask)
    counts = np.empty((len(labels), *new_shape), dtype=np.int32)
    for i, label in enumerate(labels):
        counts[i] = (reshaped == label).sum(axis=(1, 3, 5))
    return labels[counts.argmax(axis=0)].astype(mask.dtype)
