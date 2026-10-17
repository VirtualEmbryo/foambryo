"""Segmentation masks as front ends receive them: read, checked, and described before a reconstruction.

A **mask** is a 3-D array of non-negative integer labels, 0 the medium, axis 0 = z, axis 1 = y, axis 2 = x.
The `foambryo` command and the napari plugin read and check masks with these functions, so both accept and
refuse the same arrays with the same messages:

* `read_segmentation_mask(path)`: a `.npy`, `.npz` (array `mask`, optionally `spacing` and `spacing_unit`) or
  `.tif`/`.tiff` file (needs `tifffile`, `foambryo[cli]`), checked;
* `check_segmentation_mask(mask)`: the checks alone, for an array already in memory;
* `segmentation_mask_report(mask)`: what a reconstruction will meet, counted and changed in nothing: the cells,
  the cells touching the image box (they will be cut cells), the cells in several pieces, the pieces of medium
  the tissue encloses, and the very small labels.

Labels need not be consecutive: a label is a name, and the mesh keeps it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "MASK_SUFFIXES",
    "SPACING_UNIT_PATTERN",
    "SegmentationMaskError",
    "SegmentationMaskReport",
    "check_segmentation_mask",
    "read_segmentation_mask",
    "segmentation_mask_report",
]

#: Suffixes a mask file is read from.
MASK_SUFFIXES = (".npy", ".npz", ".tif", ".tiff")

#: What a spacing unit may be: it is written into a `.rec` file's appendix as `key = value` text.
SPACING_UNIT_PATTERN = re.compile(r"[A-Za-z0-9_.\-/^]+")

_NPZ_SPACING_KEYS = ("spacing", "spacing_unit")

#: A label smaller than this fraction of the median cell (in voxels) is reported as very small.
DEFAULT_SMALL_LABEL_FRACTION = 0.05

#: Voxels per slab when labels are counted slab by slab, so that counting never copies the whole mask.
_SLAB_VOXELS = 1 << 24


class SegmentationMaskError(ValueError):
    """A mask file that cannot be read, or an array that is not a mask; the message says which and why."""


def read_segmentation_mask(path: str | Path) -> tuple[np.ndarray, tuple[float, float, float] | None, str | None]:
    """Read a mask file: the label array, and the spacing and unit an `.npz` records (else `None`).

    Args:
        path: a `.npy`, `.npz` or `.tif`/`.tiff` file (the suffix in any case). An `.npz` holds the mask as `mask`
            (or as its only array other than `spacing` and `spacing_unit`), optionally `spacing` (three finite
            numbers > 0, `(dz, dy, dx)` along axes 0, 1, 2) and `spacing_unit` (one ASCII word of letters, digits
            and `_ . - / ^`). A TIFF's resolution tags are not read, and neither `.npy` nor TIFF records a spacing.

    Returns:
        tuple: the mask (checked by `check_segmentation_mask`, axis 0 = z), the recorded spacing `(dz, dy, dx)`
        or `None`, the recorded unit or `None`.

    Raises:
        SegmentationMaskError: on an unreadable file, an unsupported suffix, a missing `tifffile` for a TIFF, or an
            array that is not a mask. The message names the file.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    spacing = unit = None
    try:
        if suffix == ".npy":
            mask = np.load(path, allow_pickle=False)
        elif suffix == ".npz":
            mask, spacing, unit = _read_npz_mask(path)
        elif suffix in (".tif", ".tiff"):
            try:
                import tifffile
            except ImportError:
                message = "reading a TIFF mask needs tifffile: pip install 'foambryo[cli]'"
                raise SegmentationMaskError(message) from None
            mask = tifffile.imread(path)
        else:
            message = f"{path.name}: unsupported mask format {suffix!r}; use " + ", ".join(MASK_SUFFIXES)
            raise SegmentationMaskError(message)
    except SegmentationMaskError:
        raise
    except Exception as error:  # every reader error is the same failure for the user
        message = f"cannot read {path.name}: {error}"
        raise SegmentationMaskError(message) from error
    return check_segmentation_mask(mask, name=path.name), spacing, unit


def _read_npz_mask(path: Path) -> tuple[np.ndarray, tuple[float, float, float] | None, str | None]:
    with np.load(path, allow_pickle=False) as data:
        names = list(data.keys())
        candidates = [name for name in names if name not in _NPZ_SPACING_KEYS]
        if "mask" in names:
            mask_name = "mask"
        elif len(candidates) == 1:
            mask_name = candidates[0]
        else:
            message = f"{path.name} holds the arrays {names}: name the mask 'mask'"
            raise SegmentationMaskError(message)
        mask = data[mask_name]
        spacing = unit = None
        if "spacing" in names:
            values = np.asarray(data["spacing"], dtype=np.float64).ravel()
            if values.size != 3 or not np.all(np.isfinite(values)) or np.any(values <= 0):
                message = f"{path.name}: 'spacing' must be three finite numbers > 0, not {values}"
                raise SegmentationMaskError(message)
            spacing = (float(values[0]), float(values[1]), float(values[2]))
        if "spacing_unit" in names:
            unit = str(data["spacing_unit"])
            if not SPACING_UNIT_PATTERN.fullmatch(unit):
                message = f"{path.name}: 'spacing_unit' must be one ASCII word, not {unit!r}"
                raise SegmentationMaskError(message)
    return mask, spacing, unit


def check_segmentation_mask(mask: Any, *, name: str = "the mask") -> np.ndarray:  # noqa: ANN401 - any array-like
    """Check that an array is a mask: 3-D, integer labels, none negative.

    Args:
        mask: the array (anything `np.asarray` accepts; a lazy array is read into memory).
        name: how the messages name it, for example a file name. Keyword-only.

    Returns:
        np.ndarray: `np.asarray(mask)`, unchanged.

    Raises:
        SegmentationMaskError: if it is not 3-D, its values are not integers (floating-point and boolean values
            are refused), or a label is negative.
    """
    mask = np.asarray(mask)
    if mask.ndim != 3:
        message = f"{name} holds a {mask.ndim}-D array; a mask is 3-D"
        raise SegmentationMaskError(message)
    if not np.issubdtype(mask.dtype, np.integer):
        message = f"{name} holds {mask.dtype} values; a mask holds integer labels"
        raise SegmentationMaskError(message)
    if mask.size and mask.min() < 0:
        message = f"{name} holds negative labels; labels are 0 (medium) or positive"
        raise SegmentationMaskError(message)
    return mask


@dataclass(frozen=True)
class SegmentationMaskReport:
    """What a reconstruction of a mask will meet, counted; nothing in the mask is changed.

    Connectivity is by shared voxel faces (6 neighbours): two pieces of a label that touch only along an edge or
    at a corner are two pieces, as the mesh will see them.

    Attributes:
        shape: the mask's shape.
        n_cells: the number of labels other than 0.
        medium_voxels: the voxels of label 0.
        cells_touching_the_image_box: the labels with a voxel on a face of the array; the reconstruction makes
            them cut cells, whose pressures are not determined.
        cells_in_several_pieces: label -> number of pieces, for the labels in more than one piece.
        enclosed_medium_pieces: the pieces of label 0 that touch no face of the array, so that the tissue
            encloses them (a hole in the segmentation, read as medium).
        enclosed_medium_voxels: their voxels.
        small_label_fraction: a label below this fraction of the median cell is very small.
        median_cell_voxels: the median number of voxels of a cell (0 without cells).
        small_labels: label -> voxels, for the labels below `small_label_fraction` of the median cell.
    """

    shape: tuple[int, ...]
    n_cells: int
    medium_voxels: int
    cells_touching_the_image_box: tuple[int, ...]
    cells_in_several_pieces: dict[int, int] = field(default_factory=dict)
    enclosed_medium_pieces: int = 0
    enclosed_medium_voxels: int = 0
    small_label_fraction: float = DEFAULT_SMALL_LABEL_FRACTION
    median_cell_voxels: float = 0.0
    small_labels: dict[int, int] = field(default_factory=dict)

    def summary(self) -> str:
        """One line, as `foambryo mesh` prints it before the reconstruction.

        Returns:
            str: `mask, for information (nothing is changed): <C> cells; <B> touch the image box (cut cells); <S>
            in several pieces; <H> pieces of medium enclosed by the tissue; <N> smaller than 5% of the median
            cell (<V> voxels)`, with this report's fraction in place of 5%.
        """
        threshold = self.small_label_fraction * self.median_cell_voxels
        return (
            f"mask, for information (nothing is changed): {self.n_cells} cells; "
            f"{len(self.cells_touching_the_image_box)} touch the image box (cut cells); "
            f"{len(self.cells_in_several_pieces)} in several pieces; {self.enclosed_medium_pieces} pieces of medium "
            f"enclosed by the tissue; {len(self.small_labels)} smaller than {100 * self.small_label_fraction:g}% of "
            f"the median cell ({threshold:.6g} voxels)"
        )

    def as_record(self) -> dict[str, Any]:
        """The report as plain JSON values, for a run record (the labels of the two maps as strings)."""
        return {
            "shape": list(self.shape),
            "cells": self.n_cells,
            "medium_voxels": self.medium_voxels,
            "cells_touching_the_image_box": list(self.cells_touching_the_image_box),
            "cells_in_several_pieces": {str(k): v for k, v in self.cells_in_several_pieces.items()},
            "enclosed_medium_pieces": self.enclosed_medium_pieces,
            "enclosed_medium_voxels": self.enclosed_medium_voxels,
            "small_label_fraction": self.small_label_fraction,
            "median_cell_voxels": self.median_cell_voxels,
            "small_labels": {str(k): v for k, v in self.small_labels.items()},
        }


def segmentation_mask_report(
    mask: Any,  # noqa: ANN401 - any array-like
    *,
    small_label_fraction: float = DEFAULT_SMALL_LABEL_FRACTION,
) -> SegmentationMaskReport:
    """Describe a mask before it is reconstructed: a report only, nothing is changed.

    Args:
        mask: the mask; checked with `check_segmentation_mask` first.
        small_label_fraction: a label with fewer voxels than this fraction of the median cell is reported as very
            small. Defaults to 0.05 (`DEFAULT_SMALL_LABEL_FRACTION`). Keyword-only.

    Returns:
        SegmentationMaskReport: the counts.

    Raises:
        SegmentationMaskError: if the array is not a mask.
    """
    from scipy import ndimage

    mask = check_segmentation_mask(mask)
    labels, voxels = _label_voxel_counts(mask)
    medium_voxels = int(voxels[labels == 0].sum())
    cells, cell_voxels = labels[labels > 0], voxels[labels > 0]

    faces = [mask[0], mask[-1], mask[:, 0], mask[:, -1], mask[:, :, 0], mask[:, :, -1]]
    on_box = np.unique(np.concatenate([np.unique(face) for face in faces]))
    touching = tuple(int(label) for label in on_box if label > 0)

    six = ndimage.generate_binary_structure(3, 1)
    several: dict[int, int] = {}
    for label, box in _bounding_boxes(mask, cells):
        _, pieces = ndimage.label(mask[box] == label, structure=six)
        if pieces > 1:
            several[int(label)] = int(pieces)

    enclosed_pieces = enclosed_voxels = 0
    if medium_voxels:
        components, count = ndimage.label(mask == 0, structure=six, output=np.int32)
        if count:
            on_border = np.unique(np.concatenate([np.unique(face) for face in (
                components[0], components[-1], components[:, 0], components[:, -1], components[:, :, 0],
                components[:, :, -1],
            )]))
            sizes = np.bincount(components.ravel(), minlength=count + 1)
            inside = np.setdiff1d(np.arange(1, count + 1), on_border)
            enclosed_pieces, enclosed_voxels = int(inside.size), int(sizes[inside].sum())
        del components

    median = float(np.median(cell_voxels)) if cell_voxels.size else 0.0
    small = {int(label): int(n) for label, n in zip(cells, cell_voxels, strict=True)
             if n < small_label_fraction * median}
    return SegmentationMaskReport(
        shape=tuple(int(n) for n in mask.shape),
        n_cells=int(cells.size),
        medium_voxels=medium_voxels,
        cells_touching_the_image_box=touching,
        cells_in_several_pieces=several,
        enclosed_medium_pieces=enclosed_pieces,
        enclosed_medium_voxels=enclosed_voxels,
        small_label_fraction=float(small_label_fraction),
        median_cell_voxels=median,
        small_labels=small,
    )


def _label_voxel_counts(mask: np.ndarray) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """The labels present and their voxel counts, counted slab by slab along axis 0."""
    if not mask.size:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    highest = int(mask.max())
    rows = max(1, _SLAB_VOXELS // max(1, mask[0].size))
    if highest <= max(1 << 20, mask.size):  # a count per possible label is no larger than the mask
        counts = np.zeros(highest + 1, dtype=np.int64)
        for start in range(0, mask.shape[0], rows):
            slab = mask[start:start + rows].ravel().astype(np.intp, copy=False)  # bincount refuses uint64
            counts += np.bincount(slab, minlength=highest + 1)
        present = np.flatnonzero(counts)
        return present.astype(np.int64), counts[present]
    found: dict[int, int] = {}
    for start in range(0, mask.shape[0], rows):
        values, counts = np.unique(mask[start:start + rows], return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist(), strict=True):
            found[value] = found.get(value, 0) + count
    present = np.array(sorted(found), dtype=np.int64)
    return present, np.array([found[v] for v in present.tolist()], dtype=np.int64)


def _bounding_boxes(mask: np.ndarray, cells: NDArray[np.int64]) -> list[tuple[int, tuple[slice, ...]]]:
    """Each cell's bounding box, `(label, slices)`, with `scipy.ndimage.find_objects` on consecutive labels."""
    from scipy import ndimage

    if not cells.size:
        return []
    highest = int(cells[-1])
    if highest <= 4 * cells.size + 1024 and mask.dtype.itemsize <= 4:
        boxes = ndimage.find_objects(mask)
        return [(int(label), boxes[int(label) - 1]) for label in cells]
    # Sparse, large label values: find the boxes on consecutive indices, through a compact copy.
    compact = np.searchsorted(cells, mask).astype(np.int32) + 1
    compact[mask == 0] = 0
    boxes = ndimage.find_objects(compact)
    return [(int(label), boxes[index]) for index, label in enumerate(cells)]
