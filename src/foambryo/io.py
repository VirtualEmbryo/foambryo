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


def dcel_mesh_from_segmentation_mask(
    segmentation_mask: NDArray[np.uint],
    min_distance: int = 5,
    print_info: bool = False,
) -> DcelData:
    """Obtain a DcelData mesh from a segmentation mask using Delaunay-Watershed 3D.

    The default mesh reconstruction algorithm is used.

    Args:
        segmentation_mask (NDArray[np.uint]): Instance segmentation image.
        min_distance (int, optional): minimum distance between points of the output mesh. Defaults to 5.
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
