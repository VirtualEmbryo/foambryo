"""Module for input/output in foambryo.

Inputs can be:
- a segmentation mask (3D uint array) ;
- a multimaterial mesh: points, triangles, labels ;
- a file (.rec, or opened by meshio) containing a multimaterial mesh.

Output is always a DcelData mesh.

Matthieu Perez 2024.
"""

from pathlib import Path

import numpy as np
from dw3d import get_default_mesh_reconstruction_algorithm
from dw3d.io import load_meshio_mesh, load_rec
from numpy.typing import NDArray

from foambryo.dcel import DcelData

# `load_rec_appendix` only exists on dw3d >= 0.4.0's `dw3d.io`. An unguarded
# `from dw3d.io import ... load_rec_appendix` here would make the whole `foambryo`
# package fail to import against an older dw3d, since `foambryo/__init__.py` imports from
# this module. Guard just this one name instead of reimplementing or dropping the feature:
# everything else in foambryo works normally against an older dw3d, and only a call to
# `load_rec_appendix` itself fails, with a message naming the required dw3d version
# instead of a bare ImportError at package import time.
try:
    from dw3d.io import load_rec_appendix
except ImportError:

    def load_rec_appendix(*_args: object, **_kwargs: object) -> None:
        """Raise because the installed dw3d does not provide the .rec metadata appendix reader.

        This stands in for `dw3d.io.load_rec_appendix`, which is only available on
        dw3d >= 0.4.0.

        Raises:
            ImportError: always; the installed dw3d lacks `load_rec_appendix`.
        """
        msg = (
            "load_rec_appendix() requires dw3d >= 0.4.0, which is not installed. "
            "Upgrade dw3d to use this function."
        )
        raise ImportError(msg)


# Re-exported, not reimplemented -- dw3d.io is the single implementation of the .rec
# metadata appendix format. Not consumed by dcel_mesh_from_file: coordinate_frame /
# spacing are recorded but not yet used by any foambryo geometry code.
__all__ = [
    "dcel_mesh_from_file",
    "dcel_mesh_from_multimaterial_mesh",
    "dcel_mesh_from_segmentation_mask",
    "load_rec_appendix",
]


def dcel_mesh_from_segmentation_mask(
    segmentation_mask: NDArray[np.uint],
    min_distance: int = 3,
    print_info: bool = False,
) -> DcelData:
    """Obtain a DcelData mesh from a segmentation mask using Delaunay-Watershed 3D.

    The default mesh reconstruction algorithm is used, with its junction relocation (on by default
    since dw3d 0.5.0).

    Args:
        segmentation_mask (NDArray[np.uint]): Instance segmentation image.
        min_distance (int, optional): minimum distance between points of the output mesh, in voxels.
            Defaults to 3, dw3d's own default (it was 5 up to foambryo 0.4.2). Pass 5 to reproduce the
            0.4 spacing; at 5 or more dw3d emits `CoarseSpacingRelocationWarning` once per
            reconstruction.
        print_info (bool, optional): print mesh construction details. Defaults to False.

    Returns:
        DcelData: The mesh reconstructed from the segmentation and ready for force inference.
    """
    # Get a mesh reconstruction algorithm
    mesh_reconstruction_algorithm = get_default_mesh_reconstruction_algorithm(min_distance, print_info)

    # Reconstruct a multimaterial mesh from the mask using the mesh reconstruction algorithm
    points, triangles, labels = mesh_reconstruction_algorithm.construct_mesh_from_segmentation_mask(segmentation_mask)
    return DcelData(points, triangles, labels)


# For consistency
def dcel_mesh_from_multimaterial_mesh(
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
    labels: NDArray[np.int64],
) -> DcelData:
    """Obtain a DcelData mesh for force inference from raw multimaterial mesh data.

    Args:
        points (NDArray[np.float64]): Points of the multimaterial mesh.
        triangles (NDArray[np.int64]): Triangles of the multimaterial mesh.
        labels (NDArray[np.int64]): materials on each side of triangles.

    Returns:
        DcelData: The mesh ready for force inference.
    """
    return DcelData(points, triangles, labels)


def dcel_mesh_from_file(filename: str | Path) -> DcelData:
    """Obtain a DcelData mesh for force inference from a multimaterial mesh file.

    It can be either a .rec file created from Delaunay-Watershed 3D, or a mesh
    that can be opened by meshio and contains a "label1" and "label2" data.
    The .vtk meshes saved by Delaunay-Watershed 3D are an example of such meshes.

    Args:
        filename (str | Path): Path to the file on disk.

    Returns:
        DcelData: The mesh ready for force inference.
    """
    filename = Path(filename)
    if filename.suffix in (".rec", ".arec"):
        return DcelData(*load_rec(filename))
    else:
        points, triangles, labels = load_meshio_mesh(filename)

        # ensure endian and type
        points = points.astype(np.float64)
        triangles = triangles.astype(np.int64)
        labels = labels.astype(np.int64)
        return DcelData(points, triangles, labels)
