"""IO for .rec meshes.

Copy of private iorec package v1.0.5.
Matthieu Perez 2024
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def load_rec(
    filename: str | Path,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Read a rec mesh file.

    Args:
        filename (str | Path): The path to the file on disk.

    Raises:
        ValueError: If the path given is not a .rec or .arec file.
        FileNotFoundError: If the file doesn't exist.

    Returns:
        tuple[np.array, np.array, np.array]: (points, triangles, labels), with:
        - points: np.array of shape (num_points, 3) (float64)
        - triangles: np.array of shape (num_triangles, 3) (ulonglong)
        - labels: np.array of shape (num_triangles, 2) (uint)
    """
    filename = Path(filename).resolve()
    if not filename.exists():
        error = f"{filename} not found."
        raise FileNotFoundError(error)
    if filename.suffix == ".rec" or filename.suffix == ".arec":
        return _load_rec_file(filename)
    else:
        error = f"{filename} is not a .rec or .arec file."
        raise ValueError(error)


def _load_rec_file(
    filename: Path,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    if _is_binary(filename):
        return _load_binary_rec_file(filename)
    else:
        return _load_text_rec_file(filename)


def _load_binary_rec_file(
    filename: Path,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    with Path.open(filename, "rb") as f:
        # First we read the number of vertices
        (number_of_points,) = struct.unpack("Q", f.read(8))
        # Then we read the vertices : they are in a 3D space
        points = np.fromfile(f, count=3 * number_of_points, dtype=np.float64).reshape(
            (number_of_points, 3),
        )

        # Number of triangles
        (number_of_triangles,) = struct.unpack("Q", f.read(8))
        # Each line defines a triangle and 2 regions labels
        dtype_triangles_and_labels = np.dtype(
            [("triangles", np.int64, (3,)), ("labels", np.int32, (2,))],
        )
        triangles_and_labels = np.fromfile(
            f,
            count=number_of_triangles,
            dtype=dtype_triangles_and_labels,
        )

    return (
        points,
        triangles_and_labels["triangles"].astype(np.int64),
        triangles_and_labels["labels"].astype(np.int64),
    )


def _load_text_rec_file(
    filename: Path,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    with Path.open(filename) as f:
        # First we read the number of vertices
        number_of_points = int(f.readline().strip())
        # Then we read the vertices : they are in a 3D space
        points = np.fromfile(
            f,
            count=3 * number_of_points,
            sep="\n",
            dtype=np.float64,
        ).reshape((number_of_points, 3))

        # Number of triangles
        number_of_triangles = int(f.readline().strip())
        # Each line defines a triangle and 2 regions labels
        triangles_and_labels = np.fromfile(
            f,
            count=5 * number_of_triangles,
            sep="\n",
            dtype=np.int64,
        ).reshape(
            (number_of_triangles, 5),
        )

    return (
        points,
        triangles_and_labels[:, 0:3].astype(np.int64),
        triangles_and_labels[:, 3:5].astype(np.int64),
    )


def _is_binary(file_name: Path) -> bool:
    try:
        with Path.open(file_name, encoding="utf-8") as f:
            f.read()
            return False
    except UnicodeDecodeError:
        return True


def save_rec(
    filename: str | Path,
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
    labels: NDArray[np.int64],
    binary_mode: bool = False,
) -> None:
    """Save a rec mesh file at filename. Binary or text mode are available.

    Binary mode is more compact but might not be universally understood.

    Args:
        filename (str | Path): where to save the mesh file.
        points (NDArray[np.float64]): Mesh points.
        triangles (NDArray[np.int64]): Mesh triangles (topology).
        labels (NDArray[np.int64]): Triangles multimaterial labels.
        binary_mode (bool, optional): Choose binary mode. Defaults to False.
    """
    if binary_mode:
        _save_rec_binary(filename, points, triangles, labels)
    else:
        _save_rec_text(filename, points, triangles, labels)


def _save_rec_binary(
    filename: str | Path,
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
    labels: NDArray[np.int64],
) -> None:
    """Save a rec file at filename. Binary only."""
    bytes_content = struct.pack("Q", len(points))
    bytes_content += points.flatten().tobytes()
    bytes_content += struct.pack("Q", len(triangles))
    dt = np.dtype([("triangles", np.int64, (3,)), ("labels", np.int32, (2,))])

    def map_tri_labels_func(
        i: int,
    ) -> tuple[Any, Any]:
        # -> tuple[np.int64, np.int64, np.int64, np.int64, np.int64]:
        return triangles[i], labels[i]

    triangles_and_labels = np.fromiter(
        (map_tri_labels_func(xi) for xi in np.arange(len(triangles))),
        dtype=dt,
        count=len(triangles),
    )

    bytes_content += triangles_and_labels.tobytes()
    with Path(filename).open("wb") as file:
        file.write(bytes_content)


def _save_rec_text(
    filename: str | Path,
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
    labels: NDArray[np.int64],
) -> None:
    """Save a rec file at filename. Text only."""
    content = ""
    content += str(len(points)) + "\n"

    for i in range(len(points)):
        # content += f"{points[i][0]:.5f} {points[i][1]:.5f} {points[i][2]:.5f}\n"
        content += f"{points[i][0]} {points[i][1]} {points[i][2]}\n"

    content += str(len(triangles)) + "\n"

    for i in range(len(triangles)):
        content += f"{triangles[i][0]} {triangles[i][1]} {triangles[i][2]} "
        content += f"{labels[i][0]} {labels[i][1]}\n"

    with Path(filename).open("w") as file:
        file.write(content)
