"""The Polyscope viewer draws the tissue, not its mirror image, and keeps every triangle facing its own side.

foambryo's meshes are in the mask's order, columns (z, y, x). Up to 0.7.0 the viewer handed those columns to
Polyscope as its (x, y, z): an exchange of two axes, a reflection, so every view was the tissue's mirror image. It
now draws (x, y, z) = the columns (2, 1, 0), the mask's (column, row, slice), and hands each triangle `(a, b, c)` as
`(a, c, b)` so that it faces the same side (`foambryo._display_frame`). What this pins:

* the map from the mask's (column, row, slice) to what is drawn is the voxel size times the identity, of
  determinant +1; the former map had determinant -1;
* on a chirality phantom (`tests/chirality_phantom.py`: three arms of different lengths along z, y and x, and a
  letter F), the arms are drawn right-handed, told apart by their lengths alone, and the F reads as an F;
* triangle orientation is kept: each drawn normal is the mesh's normal mapped into the drawn frame, and the signed
  volume of every cell computed from the drawn arrays is the library's;
* vectors (the principal stress axes), slice planes and updated positions are mapped the same way, and every array
  of positions the scene hands Polyscope goes through the map;
* the mesh itself, its points and its winding, is never changed by drawing.
"""

from __future__ import annotations

import numpy as np
import pytest

from foambryo import infer_forces
from foambryo._display_frame import DRAWN_COLUMNS, DrawnStructure, drawn_points, drawn_triangles, in_drawn_frame
from foambryo._mask_workflow import mask_point_cloud
from foambryo.io import dcel_mesh_from_segmentation_mask
from tests.chirality_phantom import ARM_LENGTHS, ARMS, BODY, LETTER, LETTER_SLICE, chirality_phantom

#: The calls that take positions, triangles or vectors, all mapped; and those that take colours or values per
#: element, which are not positions.
_MAPPED_CALLS = frozenset({"register_surface_mesh", "register_point_cloud", "register_curve_network",
                           "add_vector_quantity", "update_vertex_positions", "update_point_positions",
                           "update_node_positions", "set_pose"})
_PER_ELEMENT_COLOUR_CALLS = frozenset({"add_color_quantity"})


@pytest.fixture(scope="module")
def phantom():
    return chirality_phantom()


@pytest.fixture(scope="module")
def phantom_mesh(phantom):
    return dcel_mesh_from_segmentation_mask(phantom, spacing=(1.0, 1.0, 1.0))


@pytest.fixture
def mock_polyscope(monkeypatch):
    """Polyscope's mock backend, and the arrays handed to Polyscope's registering functions, by structure name."""
    ps = pytest.importorskip("polyscope")
    real_init, real_show = ps.init, ps.show
    try:
        real_init("openGL_mock")
    except RuntimeError as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"Polyscope's mock backend is unavailable: {error}")
    monkeypatch.setattr(ps, "init", lambda *_args, **_kwargs: real_init("openGL_mock"))
    monkeypatch.setattr(ps, "show", lambda *_args, **_kwargs: real_show(forFrames=1))
    handed: dict[str, tuple] = {}
    for function in ("register_surface_mesh", "register_point_cloud", "register_curve_network"):
        original = getattr(ps, function)

        def recording(name: str, *args: object, _original=original, **kwargs: object):
            handed[name] = tuple(np.array(a) for a in args[:2])
            return _original(name, *args, **kwargs)

        monkeypatch.setattr(ps, function, recording)
    vectors: dict[str, np.ndarray] = {}
    original_vectors = ps.PointCloud.add_vector_quantity

    def vector_recording(self, name: str, values: object, *args: object, **kwargs: object):
        vectors[name] = np.array(values)
        return original_vectors(self, name, values, *args, **kwargs)

    monkeypatch.setattr(ps.PointCloud, "add_vector_quantity", vector_recording)
    yield ps, handed, vectors
    ps.remove_all_structures()
    ps.remove_all_slice_planes()


def _signed_volumes(points: np.ndarray, triangles: np.ndarray, labels: np.ndarray) -> dict[int, float]:
    """Each region's signed volume, `compute_volume_cells`' sum: `+det` to the first label, `-det` to the second."""
    increments = np.linalg.det(points[triangles]) / 6
    volumes: dict[int, float] = {}
    for (first, second), increment in zip(labels, increments, strict=True):
        volumes[int(first)] = volumes.get(int(first), 0.0) + increment
        volumes[int(second)] = volumes.get(int(second), 0.0) - increment
    return volumes


# ---------------------------------------------------------------------------------------------
# the map itself
# ---------------------------------------------------------------------------------------------
def test_the_drawn_frame_is_the_images_column_row_and_slice_with_determinant_plus_one():
    spacing = np.array([2.0, 0.5, 0.25])  # (dz, dy, dx): anisotropic, so that a swapped axis shows
    rng = np.random.default_rng(0)
    index = rng.integers(0, 50, size=(200, 3)).astype(np.float64)  # (slice, row, column), the mask's order
    stored = index * spacing  # the mesh's columns (z, y, x)
    column_row_slice = index[:, ::-1]
    for drawn, expected_determinant in ((drawn_points(stored), +1.0), (stored, -1.0)):  # the fix, and up to 0.7.0
        linear_map, *_ = np.linalg.lstsq(column_row_slice, drawn, rcond=None)
        assert np.allclose(column_row_slice @ linear_map, drawn, rtol=0, atol=1e-12)
        determinant = np.linalg.det(linear_map.T) / np.prod(spacing)
        assert determinant == pytest.approx(expected_determinant, abs=1e-12)
    assert np.array_equal(drawn_points(stored), column_row_slice * spacing[::-1])  # x = column * dx, ...
    assert drawn_points(spacing).tolist() == [0.25, 0.5, 2.0]  # the voxel size as drawn: (dx, dy, dz)
    assert DRAWN_COLUMNS == (2, 1, 0)


def test_the_map_copies_and_refuses_what_is_not_three_columns():
    points = np.arange(12.0).reshape(4, 3)
    drawn = drawn_points(points)
    drawn[0, 0] = -1.0
    assert points[0, 2] == 2.0  # a copy: the mesh's array is never changed
    triangles = np.array([[0, 1, 2], [3, 4, 5]])
    assert drawn_triangles(triangles).tolist() == [[0, 2, 1], [3, 5, 4]]
    with pytest.raises(ValueError, match="three columns"):
        drawn_points(np.zeros((4, 2)))
    with pytest.raises(ValueError, match=r"\(n, 3\)"):
        drawn_triangles(np.zeros((4, 4), dtype=int))


def test_a_reversed_triangle_faces_the_same_side_once_drawn():
    rng = np.random.default_rng(1)
    points = rng.normal(size=(30, 3))
    triangles = rng.permutation(30)[:27].reshape(9, 3)
    stored_normals = np.cross(points[triangles[:, 1]] - points[triangles[:, 0]],
                              points[triangles[:, 2]] - points[triangles[:, 0]])
    drawn, faces = drawn_points(points), drawn_triangles(triangles)
    drawn_normals = np.cross(drawn[faces[:, 1]] - drawn[faces[:, 0]], drawn[faces[:, 2]] - drawn[faces[:, 0]])
    assert np.array_equal(drawn_normals, drawn_points(stored_normals))  # +P n, bitwise: the same products
    unreversed = np.cross(drawn[triangles[:, 1]] - drawn[triangles[:, 0]],
                          drawn[triangles[:, 2]] - drawn[triangles[:, 0]])
    assert np.array_equal(unreversed, -drawn_points(stored_normals))  # without the reversal: -P n, the wrong side


class _Structure:
    """A stand-in for a Polyscope structure: records its calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name: str) -> object:
        return lambda *args, **kwargs: self.calls.append((name, args, kwargs))


def test_vectors_slice_planes_and_updated_positions_are_mapped():
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    structure = _Structure()
    registered = []

    def register(name: str, *args: object, **kwargs: object) -> _Structure:
        registered.append((name, args, kwargs))
        return structure

    drawn = in_drawn_frame("register_point_cloud", register)("cloud", points, radius=0.1)
    assert isinstance(drawn, DrawnStructure)
    assert np.array_equal(registered[0][1][0], points[:, ::-1])
    drawn.add_vector_quantity("axes", points, enabled=True)
    drawn.update_point_positions(points=points)
    drawn.set_pose(points[0], plane_normal=points[1])
    drawn.set_radius(0.2)
    calls = {name: (args, kwargs) for name, args, kwargs in structure.calls}
    assert np.array_equal(calls["add_vector_quantity"][0][1], points[:, ::-1])
    assert np.array_equal(calls["update_point_positions"][1]["points"], points[:, ::-1])
    assert np.array_equal(calls["set_pose"][0][0], [3.0, 2.0, 1.0])
    assert np.array_equal(calls["set_pose"][1]["plane_normal"], [6.0, 5.0, 4.0])
    assert calls["set_radius"] == ((0.2,), {})  # anything else is the structure's own
    found = []

    def register_surface(_name: str, _vertices: np.ndarray, faces: np.ndarray) -> _Structure:
        found.append(faces)
        return structure

    in_drawn_frame("register_surface_mesh", register_surface)("surface", points, faces=np.array([[0, 1, 2]]))
    assert found[0].tolist() == [[0, 2, 1]]
    assert in_drawn_frame("set_ground_plane_mode", print) is print  # no position: unchanged


# ---------------------------------------------------------------------------------------------
# the phantom through the viewer
# ---------------------------------------------------------------------------------------------
def test_the_mask_is_drawn_as_column_row_slice_and_the_letter_reads_as_an_f(mock_polyscope, phantom):
    from foambryo._polyscope_recording import polyscope_target as ps

    _, handed, _ = mock_polyscope
    spacing = (2.0, 0.5, 0.25)
    cloud = mask_point_cloud(phantom, spacing)
    ps.register_point_cloud("mask", cloud.points)
    drawn = handed["mask"][0]
    index = np.argwhere(phantom > 0)  # (slice, row, column), in the cloud's order
    assert np.array_equal(drawn, index[:, ::-1] * np.array(spacing)[::-1])
    letter = drawn[(cloud.labels == LETTER) & (index[:, 0] == LETTER_SLICE)]
    low, high = letter.min(axis=0), letter.max(axis=0)
    centre = letter.mean(axis=0)
    # An F: more ink on the side of its upright (low x, the image's left) and of its top bar (low y, the image's top).
    assert centre[0] < (low[0] + high[0]) / 2
    assert centre[1] < (low[1] + high[1]) / 2
    upright = letter[letter[:, 0] == low[0]]
    assert np.ptp(upright[:, 1]) == pytest.approx(high[1] - low[1])  # the upright spans the letter's height


def _drawn_regions(points: np.ndarray, triangles: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    """The drawn vertices of each region's triangles, by label."""
    regions = {}
    for label in np.unique(labels):
        rows = np.any(labels == label, axis=1)
        regions[int(label)] = points[np.unique(triangles[rows])]
    return regions


def test_the_phantoms_arms_are_drawn_right_handed(mock_polyscope, phantom_mesh):
    from foambryo.viewing import plot_force_inference

    _, handed, _ = mock_polyscope
    mesh = phantom_mesh
    tensions, pressures = infer_forces(mesh)
    plot_force_inference(mesh, tensions, pressures, show=False)
    points, triangles = handed["Embryo"]
    regions = _drawn_regions(points, triangles, np.asarray(mesh.f)[:, 3:5])
    body = regions[BODY].mean(axis=0)
    arms = {label: regions[label].mean(axis=0) - body for label in ARMS.values()}
    # Told apart by length alone, as in a picture: the longest arm is x, then y, then z.
    by_length = sorted(arms, key=lambda label: -np.linalg.norm(arms[label]))
    assert by_length == [ARMS["x"], ARMS["y"], ARMS["z"]]
    assert sorted(ARM_LENGTHS, key=ARM_LENGTHS.get, reverse=True) == ["x", "y", "z"]
    directions = np.array([arms[label] / np.linalg.norm(arms[label]) for label in by_length])
    assert np.linalg.det(directions) > 0.95  # right-handed (the mirror image gives about -1)
    # x, y, z each mostly along Polyscope's own x, y, z, in the positive direction
    assert np.argmax(np.abs(directions), axis=1).tolist() == [0, 1, 2]
    assert (np.diag(directions) > 0.9).all()


def test_every_triangle_faces_its_own_side_and_every_cell_keeps_its_volume(mock_polyscope, phantom_mesh):
    from foambryo.viewing import plot_force_inference

    _, handed, _ = mock_polyscope
    mesh = phantom_mesh
    points_before, faces_before = np.array(mesh.v), np.array(mesh.f)
    tensions, pressures = infer_forces(mesh)
    plot_force_inference(mesh, tensions, pressures, show=False)
    points, triangles = handed["Embryo"]
    assert np.array_equal(points, drawn_points(mesh.v))
    assert np.array_equal(triangles, drawn_triangles(np.asarray(mesh.f)[:, :3]))
    stored = np.asarray(mesh.v)[np.asarray(mesh.f)[:, :3]]
    stored_normals = np.cross(stored[:, 1] - stored[:, 0], stored[:, 2] - stored[:, 0])
    drawn = points[triangles]
    drawn_normals = np.cross(drawn[:, 1] - drawn[:, 0], drawn[:, 2] - drawn[:, 0])
    assert np.array_equal(drawn_normals, drawn_points(stored_normals))
    volumes = _signed_volumes(points, triangles, np.asarray(mesh.f)[:, 3:5])
    library = mesh.compute_volumes_cells()
    for label in mesh.materials:
        if label != 0:
            assert library[label] > 0
            assert volumes[label] == pytest.approx(library[label], rel=1e-12)
    # Drawing changes nothing of the mesh: its points and its winding are its own.
    assert np.array_equal(mesh.v, points_before)
    assert np.array_equal(mesh.f, faces_before)


def test_the_stress_axes_and_the_graph_are_drawn_in_the_same_frame(mock_polyscope, phantom_mesh):
    import foambryo.viewing as viewing

    _, handed, vectors = mock_polyscope
    mesh = phantom_mesh
    tensions, pressures = infer_forces(mesh)
    stored_vectors = {}
    original = viewing.split_principal_stresses
    axes_by_call = []

    def splitting(tensor):
        axes = original(tensor)
        axes_by_call.append(axes)
        return axes

    viewing.split_principal_stresses, previous = splitting, viewing.split_principal_stresses
    try:
        viewing.plot_force_inference(mesh, tensions, pressures, show=False)
    finally:
        viewing.split_principal_stresses = previous
    stored_vectors["compression_axes_0"] = np.array([axes.compression[:, 0] for axes in axes_by_call])
    stored_vectors["tension_axes_2"] = np.array([axes.tension[:, 2] for axes in axes_by_call])
    for name, stored in stored_vectors.items():
        assert np.array_equal(vectors[name], drawn_points(stored)), name
    graph = mesh.compute_networkx_graph()
    centroids = np.array([centroid for _, centroid in graph.nodes.data("centroid")])
    has_volume = np.array([volume for _, volume in graph.nodes.data("volume")]) > 0
    assert np.allclose(handed["Graph Nodes"][0], drawn_points(centroids[has_volume]), rtol=0, atol=1e-12)


@pytest.mark.usefixtures("mock_polyscope")
def test_every_array_of_positions_the_scene_hands_polyscope_is_mapped(phantom_mesh):
    """A guard: a future Polyscope call with positions that bypasses the drawn frame fails here."""
    import inspect

    from foambryo.viewing import _ForceWindow, _InferenceOptions

    mesh = phantom_mesh
    tensions, pressures = infer_forces(mesh)
    options = _InferenceOptions(inspect.signature(infer_forces).parameters["mode_tension"].default,
                                inspect.signature(infer_forces).parameters["mode_pressure"].default, 1.0, 0.0, "auto")
    window = _ForceWindow(mesh, "force", tensions, pressures, options, "test", scalar_quantities=True,
                          length_unit="um")
    for scattered in (False, True):
        recorder, _ = window.prepare(scattered, 0.3)
        names = set()
        for _target, name, args, kwargs in recorder.calls:
            arrays = [np.asarray(value) for value in (*args, *kwargs.values())
                      if isinstance(value, np.ndarray | list | tuple) and not isinstance(value, str)]
            for array in arrays:
                if array.dtype.kind == "f" and array.ndim == 2 and array.shape[1] == 3:
                    assert name in _MAPPED_CALLS | _PER_ELEMENT_COLOUR_CALLS, name
                    names.add(name)
        assert {"register_surface_mesh", "register_point_cloud", "register_curve_network",
                "add_vector_quantity"} <= names
        surface = next(args for _, name, args, _ in recorder.calls
                       if name == "register_surface_mesh" and args[0] == "Embryo")
        expected = (mesh.v_scattered, mesh.f_scattered) if scattered else (mesh.v, mesh.f)
        assert np.array_equal(surface[1], drawn_points(expected[0]))
        assert np.array_equal(surface[2], drawn_triangles(np.asarray(expected[1])[:, :3]))
