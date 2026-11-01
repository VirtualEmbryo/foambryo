"""The frame the Polyscope viewer draws in: the tissue itself, not its mirror image.

foambryo's meshes, like delaunay-watershed-3d's, are in the mask's index order: column 0 is the mask's axis 0 (z,
the slice), column 1 its axis 1 (y, the row), column 2 its axis 2 (x, the column). Polyscope reads the three
columns of the arrays it is handed as its own x, y, z, a right-handed frame. Up to 0.7.0 the viewer handed it the
mesh's columns as they are, so z was drawn as x and x as z: the exchange of two axes is a reflection (determinant
-1), and every view of the window, and every figure saved from it, was the mirror image of the tissue.

The viewer now draws (x, y, z) = the mesh's columns (2, 1, 0), that is (column, row, slice) of the mask times the
voxel size: the identity, up to that scale, in the image's own frame (determinant +1). In the mesh's column
order the map is the permutation matrix

    P = [[0, 0, 1], [0, 1, 0], [1, 0, 0]],   drawn = P stored,   det P = -1,

itself a reflection of the stored columns. A reflection reverses the orientation of every triangle: the normal
Polyscope computes for the corners (a, b, c), the cross product (b - a) x (c - a) of the drawn points, is
(det P) P n = -P n, where n is the normal the library computes from the stored points. Each triangle is therefore
handed to Polyscope as (a, c, b), whose drawn normal is +P n: the side a triangle faces, its shading, the outward
side of a cell and any colour given to one side of a face keep their meaning. A vector drawn at a point (a normal,
a principal axis of stress, a force) is a direction in the same space and is drawn as P v; a slice plane's point
and normal as P x and P n; a per-axis voxel size (dz, dy, dx) as (dx, dy, dz).

Only what is handed to Polyscope changes. The stored mesh, its winding, the files written (`.rec`, `.vtk` and the
other formats meshio writes, which keep the mesh's column order) and every number are those of the mesh.

`foambryo._polyscope_recording.polyscope_target`, the `ps` of the drawing modules, applies this map on the way to
Polyscope (`in_drawn_frame`), whether a scene is drawn at once or recorded and replayed, so the drawing code works
in the mesh's own coordinates throughout. Code that looks at the scene through Polyscope's camera (where the scale
bar and the frame's labels fall on an image) maps its points with `drawn_points` first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike, NDArray

__all__ = ["DRAWN_COLUMNS", "DRAWN_CORNERS", "drawn_points", "drawn_triangles", "in_drawn_frame"]

#: Polyscope's x, y, z are the mesh's columns 2, 1, 0: the mask's column, row and slice.
DRAWN_COLUMNS = (2, 1, 0)
#: A triangle `(a, b, c)` of the mesh is handed to Polyscope as `(a, c, b)`, so that it faces the same side.
DRAWN_CORNERS = (0, 2, 1)

#: The module functions that register a structure: the arguments that hold its positions and its triangles, as
#: `(position, keyword)` (the structure's name is argument 0).
_REGISTERED_POSITIONS = {
    "register_surface_mesh": (1, "vertices"),
    "register_point_cloud": (1, "points"),
    "register_curve_network": (1, "nodes"),
}
_REGISTERED_TRIANGLES = {"register_surface_mesh": (2, "faces")}
#: The module functions that return a structure registered before, whose methods are mapped too.
_STRUCTURE_GETTERS = frozenset({
    *_REGISTERED_POSITIONS,
    "get_surface_mesh",
    "get_point_cloud",
    "get_curve_network",
    "add_scene_slice_plane",
})
#: The methods of a structure (or of a slice plane) whose arguments are positions or vectors, as
#: `(position, keyword)` pairs.
_STRUCTURE_METHOD_ARGUMENTS = {
    "add_vector_quantity": ((1, "values"),),
    "update_vertex_positions": ((0, "vertices"),),
    "update_point_positions": ((0, "points"),),
    "update_node_positions": ((0, "nodes"),),
    "set_pose": ((0, "plane_position"), (1, "plane_normal")),
}


def drawn_points(points: ArrayLike) -> NDArray:
    """Positions, vectors or a per-axis voxel size in the mesh's column order, in the drawn frame (a copy).

    The last axis holds the three columns (z, y, x); the result holds (x, y, z), the columns (2, 1, 0).

    Raises:
        ValueError: if the last axis does not have three entries.
    """
    array = np.asarray(points)
    if array.shape[-1:] != (3,):
        message = f"the drawn frame maps three columns (z, y, x), not an array of shape {array.shape}"
        raise ValueError(message)
    return array[..., list(DRAWN_COLUMNS)]


def drawn_triangles(triangles: ArrayLike) -> NDArray:
    """The triangles `(a, b, c)` as handed to Polyscope, `(a, c, b)`: each faces the same side once drawn (a copy).

    Raises:
        ValueError: if the triangles are not an `(n, 3)` array.
    """
    array = np.asarray(triangles)
    if array.ndim != 2 or array.shape[1] != 3:
        message = f"the drawn frame maps triangles as an (n, 3) array, not an array of shape {array.shape}"
        raise ValueError(message)
    return array[:, list(DRAWN_CORNERS)]


def _mapped(args: tuple, kwargs: dict, where: tuple[int, str], transform: Callable[[Any], Any]) -> tuple[tuple, dict]:
    """`args` and `kwargs` with the argument at `where` (by position or by keyword) passed through `transform`."""
    index, keyword = where
    if keyword in kwargs:
        kwargs = {**kwargs, keyword: transform(kwargs[keyword])}
    elif len(args) > index:
        args = (*args[:index], transform(args[index]), *args[index + 1:])
    return args, kwargs


class DrawnStructure:
    """A Polyscope structure (or a recorded one, or a slice plane) whose position and vector arguments are mapped.

    Every other attribute is the structure's own.
    """

    def __init__(self, structure: Any) -> None:  # noqa: ANN401 - a Polyscope or recorded structure
        self._structure = structure

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - whatever the structure holds
        attribute = getattr(self._structure, name)
        arguments = _STRUCTURE_METHOD_ARGUMENTS.get(name)
        if arguments is None:
            return attribute

        def mapped(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            for where in arguments:
                args, kwargs = _mapped(args, kwargs, where, drawn_points)
            return attribute(*args, **kwargs)

        return mapped


def in_drawn_frame(name: str, attribute: Any) -> Any:  # noqa: ANN401 - a Polyscope function or anything else
    """Polyscope's (or a recorder's) module attribute `name`, with its positions and triangles in the drawn frame.

    A registering function gets its positions through `drawn_points` and its triangles through `drawn_triangles`,
    and returns its structure as a `DrawnStructure`; a function that returns a structure registered before returns
    it as one too. Anything else is returned unchanged.
    """
    if name not in _STRUCTURE_GETTERS or not callable(attribute):
        return attribute

    def mapped(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        if name in _REGISTERED_POSITIONS:
            args, kwargs = _mapped(args, kwargs, _REGISTERED_POSITIONS[name], drawn_points)
        if name in _REGISTERED_TRIANGLES:
            args, kwargs = _mapped(args, kwargs, _REGISTERED_TRIANGLES[name], drawn_triangles)
        structure = attribute(*args, **kwargs)
        if structure is None or isinstance(structure, DrawnStructure):
            return structure
        return DrawnStructure(structure)

    return mapped
