"""A segmentation mask that is not its own mirror image, to check which way round a viewer draws a tissue.

The mask's axes are (z, y, x) = (slice, row, column), as every mask foambryo reads; its voxels are cubic. It holds:

* label 1, a cube of 16 voxels, the **body**, near the low corner (8 voxels from the image's faces);
* three **arms** from the body, of different lengths, so that each can be told apart in any picture: along x (the
  mask's column, axis 2) 56 voxels (label 2), along y (row, axis 1) 36 voxels (label 3), along z (slice, axis 0)
  16 voxels (label 4), each 16 x 16 voxels across. Seen from outside the corner, x -> y -> z turn anticlockwise in
  a right-handed picture and clockwise in its mirror image;
* label 5, a letter **F**, 32 x 32 voxels and 12 slices thick (slices 10 to 21), in the plane of the x and y arms,
  its top bar against the x arm (so that it has triple lines, and a tension). It reads as an F in a slice shown
  as an image (row down, column right, as `matplotlib.pyplot.imshow` shows it: the view from the low slices
  towards the high ones), and as a mirrored F from the other side.

No label reaches a face of the image, so the mesh has no crop faces. delaunay-watershed-3d's default reconstruction
(`min_distance` 3) gives every region a mesh volume within 0.87-1.05 of its voxel count.
"""

from __future__ import annotations

import numpy as np

#: The mask's shape, (z, y, x).
SHAPE = (48, 72, 88)
BODY, X_ARM, Y_ARM, Z_ARM, LETTER = 1, 2, 3, 4, 5
#: The arms' labels by the axis they run along, and their lengths in voxels.
ARMS = {"x": X_ARM, "y": Y_ARM, "z": Z_ARM}
ARM_LENGTHS = {"x": 56, "y": 36, "z": 16}
#: The slice in which the letter is shown.
LETTER_SLICE = 16


def chirality_phantom() -> np.ndarray:
    """The phantom mask, `(z, y, x)`, `uint16`: medium 0, body 1, arms 2 (x), 3 (y), 4 (z), letter F 5."""
    mask = np.zeros(SHAPE, dtype=np.uint16)
    low, high = 8, 24  # the body's extent along each axis
    mask[low:high, low:high, low:high] = BODY
    mask[low:high, low:high, high:high + ARM_LENGTHS["x"]] = X_ARM
    mask[low:high, high:high + ARM_LENGTHS["y"], low:high] = Y_ARM
    mask[high:high + ARM_LENGTHS["z"], low:high, low:high] = Z_ARM
    letter = np.zeros(SHAPE[1:], dtype=bool)  # (y, x): row down, column right
    letter[24:56, 32:40] = True  # the upright, on the left
    letter[24:32, 32:64] = True  # the top bar, to the right, against the x arm (rows 8 to 23)
    letter[38:46, 32:56] = True  # the middle bar, shorter
    mask[10:22][:, letter] = LETTER
    return mask
