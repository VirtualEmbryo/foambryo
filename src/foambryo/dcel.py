"""Module dedicated to the computation of geometrical quantities on a 3D Mesh.

Will probably be moved to foambryo in the future.

Sacha Ichbiah, 2021.
Matthieu Perez, 2024.
"""

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import networkx
import numpy as np
from numpy.typing import NDArray

from dw3d.curvature import compute_curvature_interfaces
from dw3d.geometry import (
    compute_angles_tri,
    compute_area_derivative_autodiff,
    compute_area_derivative_dict,
    compute_areas_cells,
    compute_areas_faces,
    compute_areas_interfaces,
    compute_length_derivative_autodiff,
    compute_length_trijunctions,
    compute_volume_cells,
    compute_volume_derivative_autodiff_dict,
    compute_volume_derivative_dict,
)


def separate_faces_dict(triangles_and_labels: NDArray[np.uint]) -> dict[int, NDArray[np.uint]]:
    """Construct a dictionnary that maps a region id to the array of triangles forming this region."""
    nb_regions = np.amax(triangles_and_labels[:, [3, 4]]) + 1

    occupancy = np.zeros(nb_regions, dtype=np.int64)
    triangles_of_region: dict[int, list[int]] = {}
    for face in triangles_and_labels:
        triangle = face[:3]
        region1, region2 = face[3:]
        if region1 >= 0:
            if occupancy[region1] == 0:
                triangles_of_region[region1] = [triangle]
                occupancy[region1] += 1
            else:
                triangles_of_region[region1].append(triangle)

        if region2 >= 0:
            if occupancy[region2] == 0:
                triangles_of_region[region2] = [triangle]
                occupancy[region2] += 1
            else:
                triangles_of_region[region2].append(triangle)

    faces_separated: dict[int, NDArray[np.uint]] = {}
    for i in sorted(triangles_of_region.keys()):
        faces_separated[i] = np.array(triangles_of_region[i])

    return faces_separated


def renormalize_verts(
    points: NDArray[np.float64],
    triangles: NDArray[np.ulonglong],
) -> tuple[NDArray[np.float64], NDArray[np.ulonglong]]:
    """Take a mesh made from points and triangles and remove points not indexed in triangles. Re-index triangles.

    Return the filtered points and reindexed triangles.
    """
    used_points_id = np.unique(triangles)
    used_points = np.copy(points[used_points_id])
    idx_mapping = np.arange(len(used_points))
    mapping = dict(zip(used_points_id, idx_mapping, strict=True))

    reindexed_triangles = np.fromiter(
        (mapping[xi] for xi in triangles.reshape(-1)),
        dtype=np.ulonglong,
        count=3 * len(triangles),
    ).reshape((-1, 3))

    return (used_points, reindexed_triangles)


def _find_key_multiplier(num_points: int) -> int:
    key_multiplier = 1
    while num_points // key_multiplier != 0:
        key_multiplier *= 10
    return key_multiplier


@dataclass
class Vertex:
    """Vertex in 2D."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    key: int = 0
    on_trijunction = False


@dataclass
class HalfEdge:
    """Half-Edge of a DCEL graph."""

    origin: Vertex = None
    destination: Vertex = None
    material_1: int = 0
    material_2: int = 0
    twin: "HalfEdge" = None
    incident_face: "Face" = None
    prev_he: "HalfEdge" = None
    next_he: "HalfEdge" = None
    attached: dict = field(default_factory=dict)
    key: int = 0

    def compute_length(self) -> None:
        """Compute length of edge. Keep value in self.length."""
        v = np.zeros(3)
        v[0] = self.origin.x - self.destination.x
        v[1] = self.origin.y - self.destination.y
        v[2] = self.origin.z - self.destination.z
        self.length = np.linalg.norm(v)

    def set_face(self, face: "Face") -> None:
        """Set incident face of this edge."""
        if self.incident_face is not None:
            print("Error : the half-edge already has a face.")
            return
        self.incident_face = face
        if self.incident_face.outer_component is None:
            face.outer_component = self

    def set_prev(self, other: "HalfEdge") -> None:
        """Set previous HalfEdge."""
        if other.incident_face is not self.incident_face:
            print("Error setting prev relation : edges must share the same face.")
            return
        self.prev_he = other
        other.next_he = self

    def set_next(self, other: "HalfEdge") -> None:
        """Set next HalfEdge."""
        if other.incident_face is not self.incident_face:
            print("Error setting next relation : edges must share the same face.")
            return
        self.next_he = other
        other.prev_he = self

    def set_twin(self, other: "HalfEdge") -> None:
        """Set twin HalfEdge."""
        self.twin = other
        other.twin = other

    def return_vector(self) -> NDArray[np.float64]:
        """Return a normalized vector from origin to destination."""
        xo, yo = self.origin.x, self.origin.y
        xt, yt = self.destination.x, self.destination.y
        vect = np.array([xt - xo, yt - yo])
        vect /= np.linalg.norm(vect)
        return vect

    def __repr__(self) -> str:
        """Debug representation."""
        ox = "None"
        oy = "None"
        dx = "None"
        dy = "None"
        if self.origin is not None:
            ox = str(self.origin.x)
            oy = str(self.origin.y)
        if self.destination is not None:
            dx = str(self.destination.x)
            dy = str(self.destination.y)
        return f"origin : ({ox}, {oy}) ; destination : ({dx}, {dy})"


@dataclass
class Face:
    """Face of a DCEL graph."""

    attached: dict = field(default_factory=dict)
    outer_component: HalfEdge = None
    _closed: bool = True
    material_1: int = 0
    material_2: int = 0
    normal = None
    key: int = 0

    # def set_outer_component(self, half_edge):
    #     if half_edge.incident_face is not self:
    #         print("Error : the edge must have the same incident face.")
    #         return
    #     self.outer_component = half_edge

    def first_half_edge(self) -> HalfEdge | None:
        """Get the first HalfEdge of the face."""
        self._closed = False
        first_half_edge = self.outer_component
        if first_half_edge is None:
            return None
        while first_half_edge.prev_he is not None:
            first_half_edge = first_half_edge.prev_he
            if first_half_edge is self.outer_component:
                self._closed = True
                break
        return first_half_edge

    def last_half_edge(self) -> HalfEdge | None:
        """Get the last HalfEdge of the face."""
        self._closed = False
        last_half_edge = self.outer_component
        if last_half_edge is None:
            return None
        while last_half_edge.next_he is not None:
            last_half_edge = last_half_edge.next_he
            if last_half_edge is self.outer_component:
                self._closed = True
                last_half_edge = self.outer_component.prev_he
                break
        return last_half_edge

    def closed(self) -> bool:
        """Whether the face is closed or not (can we loop throught HalfEdges ?)."""
        self.first_half_edge()
        return self._closed

    def get_edges(self) -> list[HalfEdge]:
        """Get the ordered list of HalfEdges forming the face."""
        edges = []
        if self.outer_component is None:
            return edges

        first_half_edge = self.first_half_edge()
        last_half_edge = self.last_half_edge()
        edge = first_half_edge
        while True:
            edges.append(edge)
            if edge is last_half_edge:
                break
            else:
                edge = edge.next_he
        return edges

    def get_materials(self) -> None:
        """Set the materials of the face from its outer component HalfEdge."""
        if self.outer_component is not None:
            self.material_1 = self.outer_component.material_1
            self.material_2 = self.outer_component.material_2

    def get_vertices(self) -> list[Vertex]:
        """Get the list of Vertices forming the Face."""
        return [edge.origin for edge in self.get_edges() if edge.origin is not None]

    def get_area(self) -> float | None:
        """Get area of face. None if face is not closed."""
        if not self.closed():
            return None
        else:

            def _distance(p1: Vertex, p2: Vertex) -> float:
                return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + ((p1.z - p2.z) ** 2))

            area = 0
            vertices = self.get_vertices()
            p1 = vertices[0]
            for i in range(1, len(vertices) - 1):
                p2 = vertices[i]
                p3 = vertices[i + 1]
                a = _distance(p1, p2)
                b = _distance(p2, p3)
                c = _distance(p3, p1)
                s = (a + b + c) / 2.0
                area += math.sqrt(s * (s - a) * (s - b) * (s - c))
            return area


def separate_faces_dict_keep_idx(
    triangles_and_labels: NDArray[np.uint],
) -> tuple[dict[int, NDArray[np.uint]], dict[int, NDArray[np.uint]]]:
    """Construct two dictionnaries to map region -> triangles.

    One that maps a region id to the array of triangles forming this region.
    The other maps a region id to the indices of triangles in the triangles array.
    """
    nb_regions = np.amax(triangles_and_labels[:, [3, 4]]) + 1

    occupancy = np.zeros(nb_regions, dtype=np.int64)
    triangles_of_region: dict[int, list[int]] = {}
    indices_of_triangles_of_region: dict[int, list[int]] = {}
    for idx, face in enumerate(triangles_and_labels):
        triangle = face[:3]
        region1, region2 = face[3:]
        if region1 >= 0:
            if occupancy[region1] == 0:
                triangles_of_region[region1] = [triangle]
                indices_of_triangles_of_region[region1] = [idx]
                occupancy[region1] += 1
            else:
                triangles_of_region[region1].append(triangle)
                indices_of_triangles_of_region[region1].append(idx)

        if region2 >= 0:
            if occupancy[region2] == 0:
                triangles_of_region[region2] = [triangle]
                indices_of_triangles_of_region[region2] = [idx]
                occupancy[region2] += 1
            else:
                triangles_of_region[region2].append(triangle)
                indices_of_triangles_of_region[region2].append(idx)

    faces_separated: dict[int, NDArray[np.uint]] = {}
    indices_faces_separated: dict[int, NDArray[np.uint]] = {}
    for i in sorted(triangles_of_region.keys()):
        faces_separated[i] = np.array(triangles_of_region[i])
        indices_faces_separated[i] = np.array(indices_of_triangles_of_region[i])

    return (faces_separated, indices_faces_separated)


def compute_scattered_arrays(mesh: "DcelData", coeff: float) -> None:
    """Modify the mesh to obtain the alternative "scattered" mesh, with each region separated from others.

    coeff = 0 means no displacement.
    """
    points, triangles_and_labels = mesh.v, mesh.f

    clusters, clusters_idx = separate_faces_dict_keep_idx(triangles_and_labels)

    all_verts = []
    all_faces = []
    all_idx = []
    all_faces_cluster_idx = []
    offset = 0
    embryo_centroid = np.mean(points, axis=0)
    clusters_displacements = {}

    for key in sorted(clusters.keys()):
        if key == 0:
            continue
        faces = np.array(clusters[key])

        vn, fn = renormalize_verts(points, faces)

        # to change with a formula from DDG
        array_centroid = np.mean(vn, axis=0)
        vn = vn + coeff * (array_centroid - embryo_centroid)
        clusters_displacements[key] = (coeff * (array_centroid - embryo_centroid)).copy()

        all_verts.append(vn.copy())
        all_faces.append(fn.copy() + offset)
        all_idx.append(clusters_idx[key])
        all_faces_cluster_idx.append(np.ones(len(fn)) * key)

        offset += len(vn)
    all_verts = np.vstack(all_verts)
    all_faces = np.vstack(all_faces)
    all_faces_cluster_idx = np.hstack(all_faces_cluster_idx)
    all_idx = np.hstack(all_idx)
    mesh.v_scattered = all_verts
    mesh.f_scattered = all_faces
    mesh.idx_scattered = all_idx
    mesh.clusters_displacements = clusters_displacements
    mesh.cluster_idx_scattered = all_faces_cluster_idx


class DcelData:
    """DCEL Graph containing faces, half-edges and vertices."""

    def __init__(self, points: NDArray[np.float64], triangles_and_labels: NDArray[np.uint]) -> None:
        """Take a multimaterial mesh as input."""
        for i, f in enumerate(triangles_and_labels):
            if f[3] > f[4]:
                triangles_and_labels[i] = triangles_and_labels[i, [0, 2, 1, 4, 3]]
        points, triangles_and_labels = remove_unused_vertices(points, triangles_and_labels)
        self.v = points
        self.f = triangles_and_labels
        # self.n_materials = np.amax(Faces[:,[3,4]])+1
        self.materials: NDArray[np.ulonglong] = np.unique(triangles_and_labels[:, [3, 4]])
        self.n_materials = len(self.materials)
        vertices_list, halfedges_list, faces_list = build_lists(points, triangles_and_labels)
        self.vertices = vertices_list
        self.faces = faces_list
        self.half_edges = halfedges_list
        self.compute_areas_faces()
        self.compute_centroids_cells()
        self.mark_trijunctional_vertices()
        self.compute_length_halfedges()

    def compute_scattered_arrays(self, coeff: float) -> None:
        """Modify the mesh to obtain the alternative "scattered" mesh, with each region separated from others.

        coeff = 0 means no displacement.
        """
        compute_scattered_arrays(self, coeff)

    def compute_length_halfedges(self) -> None:
        """Compute all lengths of HalfEdges."""
        compute_length_halfedges(self)

    def compute_areas_faces(self) -> None:
        """Compute the area of every triangle using Heron's formula."""
        compute_areas_faces(self)

    def compute_vertex_normals(self) -> NDArray[np.float64]:
        """Compute normals at every vertex.

        Args:
            points (NDArray[np.float64]): points of the mesh
            triangles (NDArray[np.ulonglong]): triangles of the mesh

        Returns:
            NDArray[np.float64]: normal for every vertex
        """
        return compute_vertex_normals(self.v, self.f)

    def compute_verts_faces_interfaces(
        self,
    ) -> tuple[dict[tuple[int, int], NDArray[np.float64]], dict[tuple[int, int], NDArray[np.ulonglong]]]:
        """Returns two dicts that maps (label1, label2) of an interface to array of vertices & triangles."""
        return compute_verts_and_faces_interfaces(self)

    def compute_networkx_graph(self) -> networkx.Graph:
        """Compute a graph with values extracted from the mesh. Nodes = regions, edges = interfaces."""
        return compute_networkx_graph(self)

    def find_trijunctional_edges(self) -> NDArray[np.uint]:
        """Return an array of edges on trijunction.

        Each edge is [index of point 1, index of point 2].
        """
        return find_trijunctional_edges(self)

    def compute_centroids_cells(self) -> None:
        """Compute internally an array of each region's centroid."""
        self.centroids = {}
        separated_faces = separate_faces_dict(self.f)
        for i in separated_faces:
            self.centroids[i] = np.mean(self.v[np.unique(separated_faces[i]).astype(int)], axis=0)

    def mark_trijunctional_vertices(self, return_list: bool = False) -> NDArray[np.uint]:
        """Check for vertices on trijunction and mark them as such. Can return the keys of those vertices."""
        return mark_trijunctional_vertices(self, return_list)

    def compute_length_trijunctions(self, prints: bool = False) -> dict[tuple[int, int, int], float]:
        """Compute the length of trijunctions, identified by their 3 adjacent regions."""
        return compute_length_trijunctions(self, prints)

    # TODO:  FIND AN EFFICIENT IMPLEMENTATION OF THE TRIJUNCTIONAL LENGTH DERIVATIVES

    def compute_areas_cells(self) -> dict[int, float]:
        """Compute the area of each region."""
        return compute_areas_cells(self)

    def compute_areas_interfaces(self) -> dict[tuple[int, int], float]:
        """Compute area of every interface (label1, label2) in mesh."""
        return compute_areas_interfaces(self)

    def compute_area_derivatives_slow(self) -> dict[tuple[int, int], NDArray[np.float64]]:
        """Compute dict that maps interface (label1, label2) to array of change of area per point."""
        return compute_area_derivative_dict(self)

    def compute_area_derivatives(self) -> dict[tuple[int, int], NDArray[np.float64]]:
        """Compute dict that maps interface (label1, label2) to array of change of area per point."""
        return compute_area_derivative_autodiff(self)

    def compute_volumes_cells(self) -> dict[int, float]:
        """Compute map cell number -> volume."""
        return compute_volume_cells(self)

    def compute_volume_derivatives_slow(self) -> dict[int, NDArray[np.float64]]:
        """Compute map cell number -> derivative of volume wrt to each point."""
        return compute_volume_derivative_dict(self)

    def compute_volume_derivatives(self) -> dict[int, NDArray[np.float64]]:
        """Compute map cell number -> derivative of volume wrt to each point."""
        return compute_volume_derivative_autodiff_dict(self)

    def compute_length_derivatives(self) -> dict[tuple[int, int], NDArray[np.float64]]:
        """Compute map trijunction edge (V1, V2) -> change of length wrt to points."""
        return compute_length_derivative_autodiff(self)

    def compute_angles_junctions(
        self,
        unique: bool = True,
    ) -> dict[tuple[int, int, int], float]:
        """Compute a map trijunction (id reg 1, id reg 2, id reg 3) to mean angle in radians."""
        return compute_angles_tri(self, unique=unique)[0]

    def compute_angles_tri(
        self,
        unique: bool = True,
    ) -> tuple[dict[tuple[int, int, int], float], dict[tuple[int, int, int], float], dict[tuple[int, int, int], float]]:
        """Compute three maps trijunction (id reg 1, id reg 2, id reg 3) to mean angle, mean angle (deg), length."""
        return compute_angles_tri(self, unique=unique)

    def compute_curvatures_interfaces(self, weighted: bool = True) -> dict[tuple[int, int], float]:
        """Compute the mean curvature on the interfaces of the mesh."""
        # "robust" or "cotan"
        return compute_curvature_interfaces(self, weighted=weighted)

    def save(self, filename: str | Path) -> None:
        """Pickle list of vertices, half edges and faces."""
        with Path.open(filename, "wb") as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self.vertices, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.half_edges, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.faces, f, pickle.HIGHEST_PROTOCOL)

    def load(self, filename: str | Path) -> None:
        """Unpickle list of vertices, half edges and faces (might be dangerous)."""
        with Path.open(filename, "rb") as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            self.vertices = pickle.load(f)  # noqa: S301
            self.half_edges = pickle.load(f)  # noqa: S301
            self.faces = pickle.load(f)  # noqa: S301


"""
DCEL BUILDING FUNCTIONS
"""


def compute_normal_faces(points: NDArray[np.float64], triangles: NDArray[np.ulonglong]) -> NDArray[np.float64]:
    """Compute normals on every triangle."""
    positions = points[triangles[:, :3]]
    sides1 = positions[:, 1] - positions[:, 0]
    sides2 = positions[:, 2] - positions[:, 1]
    normal_faces = np.cross(sides1, sides2, axis=1)
    norms = np.linalg.norm(normal_faces, axis=1)  # *(1+1e-8)
    normal_faces /= np.array([norms] * 3).transpose()
    return normal_faces


def build_lists(
    points: NDArray[np.float64],
    triangles_and_labels: NDArray[np.uint],
) -> tuple[list[Vertex], list[HalfEdge], list[Face]]:
    """Build lists of DCEL objects from mesh."""
    normals = compute_normal_faces(points, triangles_and_labels)
    vertices_list = make_vertices_list(points)
    halfedge_list: list[HalfEdge] = []
    for i in range(len(triangles_and_labels)):
        a, b, c, _, _ = triangles_and_labels[i]
        halfedge_list.append(HalfEdge(origin=vertices_list[a], destination=vertices_list[b], key=3 * i + 0))
        halfedge_list.append(HalfEdge(origin=vertices_list[c], destination=vertices_list[a], key=3 * i + 1))
        halfedge_list.append(HalfEdge(origin=vertices_list[b], destination=vertices_list[c], key=3 * i + 2))

    for i in range(len(triangles_and_labels)):
        index = 3 * i
        halfedge_list[index].next_he = halfedge_list[index + 1]
        halfedge_list[index].prev_he = halfedge_list[index + 2]

        halfedge_list[index + 1].next_he = halfedge_list[index + 2]
        halfedge_list[index + 1].prev_he = halfedge_list[index]

        halfedge_list[index + 2].next_he = halfedge_list[index]
        halfedge_list[index + 2].prev_he = halfedge_list[index + 1]

    faces_list: list[Face] = []
    for i in range(len(triangles_and_labels)):
        faces_list.append(
            Face(
                outer_component=halfedge_list[i + 3],
                material_1=triangles_and_labels[i, 3],
                material_2=triangles_and_labels[i, 4],
                key=i,
            ),
        )
        faces_list[i].normal = normals[i]

    for i in range(len(triangles_and_labels)):
        halfedge_list[3 * i + 0].incident_face = faces_list[i]
        halfedge_list[3 * i + 1].incident_face = faces_list[i]
        halfedge_list[3 * i + 2].incident_face = faces_list[i]

    # find twins
    triangles = triangles_and_labels.copy()[:, [0, 1, 2]]
    edges = np.hstack((triangles, triangles)).reshape(-1, 2)
    edges = np.sort(edges, axis=1)
    key_mult = _find_key_multiplier(np.amax(triangles))
    keys = edges[:, 1] * key_mult + edges[:, 0]
    dict_twins = {}
    for i, key in enumerate(keys):
        dict_twins[key] = [*dict_twins.get(key, []), i]
    list_twins = []

    for i in range(len(edges)):
        key = keys[i]
        list_to_filter = dict_twins[key].copy()
        list_to_filter.remove(i)
        list_twins.append(list_to_filter)

    for i, list_twin in enumerate(list_twins):
        halfedge_list[i].twin = list_twin

    return (vertices_list, halfedge_list, faces_list)


def make_vertices_list(points: NDArray[np.float64]) -> list[Vertex]:
    """Build the list of Vertex from the mesh's points."""
    vertices_list = []
    for i, vertex_coords in enumerate(points):
        x, y, z = vertex_coords
        vertices_list.append(Vertex(x=x, y=y, z=z, key=i))
    return vertices_list


def mark_trijunctional_vertices(mesh: DcelData, return_list: bool = False) -> NDArray[np.uint]:
    """Check for vertices on trijunction and mark them as such. Can return the keys of those vertices."""
    list_trijunctional_vertices = []
    for edge in mesh.half_edges:
        if len(edge.twin) > 1:
            mesh.vertices[edge.origin.key].on_trijunction = True
            mesh.vertices[edge.destination.key].on_trijunction = True
            list_trijunctional_vertices.append(edge.origin.key)
            list_trijunctional_vertices.append(edge.destination.key)
    if return_list:
        return np.unique(list_trijunctional_vertices)


"""
DCEL Geometry functions
"""


def find_trijunctional_edges(mesh: DcelData) -> NDArray[np.uint]:
    """Return an array of edges on trijunction.

    Each edge is [index of point 1, index of point 2].
    """
    triangles_and_labels = mesh.f
    edges = np.vstack(
        (triangles_and_labels[:, [0, 1]], triangles_and_labels[:, [0, 2]], triangles_and_labels[:, [1, 2]]),
    )
    edges = np.sort(edges, axis=1)
    key_mult = _find_key_multiplier(len(mesh.v) + 1)
    edge_keys = (edges[:, 0] + 1) + (edges[:, 1] + 1) * key_mult
    _, index_first_occurence, index_counts = np.unique(
        edge_keys,
        return_index=True,
        return_counts=True,
    )
    print("Number of trijunctional edges :", np.sum(index_counts == 3))
    return edges[index_first_occurence[index_counts == 3]]


def compute_length_halfedges(mesh: DcelData) -> None:
    """Compute all lengths of HalfEdges."""
    for edge in mesh.half_edges:
        edge.compute_length()


def compute_faces_areas(points: NDArray[np.float64], triangles: NDArray[np.ulonglong]) -> NDArray[np.float64]:
    """Compute the area of every triangle using Heron's formula."""
    positions = points[triangles[:, :3]]
    sides = positions - positions[:, [2, 0, 1]]
    lengths_sides = np.linalg.norm(sides, axis=2)
    half_perimeters = np.sum(lengths_sides, axis=1) / 2

    diffs = np.array([half_perimeters] * 3).transpose() - lengths_sides
    return (half_perimeters * diffs[:, 0] * diffs[:, 1] * diffs[:, 2]) ** (0.5)


def compute_vertex_normals(
    points: NDArray[np.float64],
    triangles: NDArray[np.ulonglong],
) -> NDArray[np.float64]:
    """Compute normals at every vertex.

    Args:
        points (NDArray[np.float64]): points of the mesh
        triangles (NDArray[np.ulonglong]): triangles of the mesh

    Returns:
        NDArray[np.float64]: normal for every vertex
    """
    faces_on_verts = [[] for x in range(len(points))]
    for i, f in enumerate(triangles):
        faces_on_verts[f[0]].append(i)
        faces_on_verts[f[1]].append(i)
        faces_on_verts[f[2]].append(i)

    positions = points[triangles]
    side1 = positions[:, 0] - positions[:, 1]
    side2 = positions[:, 0] - positions[:, 2]
    faces_normals = np.cross(side1, side2, axis=1)
    norms = np.linalg.norm(faces_normals, axis=1)
    faces_normals *= np.array([1 / norms] * 3).transpose()
    faces_areas = compute_faces_areas(points, triangles)
    vertex_normals = np.zeros(points.shape)

    for i, f_list in enumerate(faces_on_verts):
        c = 0
        n = 0
        for f_idx in f_list:
            n += faces_normals[f_idx] * faces_areas[f_idx]
            c += faces_areas[f_idx]
        n /= c
        vertex_normals[i] = n
    return vertex_normals


def remove_unused_vertices(
    points: NDArray[np.float64],
    triangles_and_labels: NDArray[np.uint],
) -> tuple[NDArray[np.float64], NDArray[np.ulonglong]]:
    """From points and triangles with labels ; remove points not indexed in triangles. Re-index triangles and labels.

    Return the filtered points and reindexed triangles and labels.
    """
    # Some unused vertices appears after the tetrahedral remeshing. We need to remove them.
    filtered_points, reindexed_triangles = renormalize_verts(points, triangles_and_labels[:, :3])
    reindexed_triangles_and_labels = np.hstack((reindexed_triangles, triangles_and_labels[:, 3:]))
    return (filtered_points, reindexed_triangles_and_labels)


def compute_centroids_graph(mesh: DcelData) -> NDArray[np.float64]:
    """Return an array of each region's centroid."""
    centroids = np.zeros((mesh.n_materials, 3))
    faces_dict = separate_faces_dict(mesh.f)
    for index_region in faces_dict:
        centroids[index_region] = np.mean(mesh.v[faces_dict[index_region]].reshape(-1, 3), axis=0)
    return centroids


def compute_networkx_graph(mesh: DcelData) -> networkx.Graph:
    """Compute a graph with values extracted from the mesh. Nodes = regions, edges = interfaces."""
    verts_interfaces, faces_interfaces = mesh.compute_verts_faces_interfaces()
    areas = mesh.compute_areas_cells()
    volumes = mesh.compute_volumes_cells()
    areas_interfaces = mesh.compute_areas_interfaces()
    curvatures = mesh.compute_curvatures_interfaces()

    # Mesh.compute_centroids_cells()

    centroids = mesh.centroids
    # Centroids = compute_centroids_graph(Mesh)

    graph = networkx.Graph()
    data_dicts = [{"area": areas[x], "volume": volumes[x], "centroid": centroids[x]} for x in mesh.materials]
    graph.add_nodes_from(zip(mesh.materials, data_dicts, strict=False))

    edges_array = [
        (
            tup[0],
            tup[1],
            {
                "mean_curvature": curvatures[tup],
                "area": areas_interfaces[tup],
                "verts": verts_interfaces[tup],
                "faces": faces_interfaces[tup],
            },
        )
        for tup in curvatures
    ]
    graph.add_edges_from(edges_array)
    return graph


def update_graph_with_scattered_values(graph: networkx.Graph, mesh: DcelData) -> networkx.Graph:
    """Update graph after scattering regions (it moves vertices and centroids)."""
    new_centroids = dict(graph.nodes.data("centroid"))
    for key in new_centroids:
        if key == 0:
            continue
        new_centroids[key] += mesh.clusters_displacements[key]
        # print(new_centroids[key])
    networkx.set_node_attributes(graph, new_centroids, "centroid")

    verts_faces_dict = {}
    for elmt in graph.edges.data("verts"):
        a, b, v = elmt
        if a == 0:
            verts_faces_dict[(a, b)] = v.copy() + mesh.clusters_displacements[b]
        else:
            verts_faces_dict[(a, b)] = v.copy()

    networkx.set_edge_attributes(graph, verts_faces_dict, "verts")

    return graph


def compute_verts_and_faces_interfaces(
    mesh: DcelData,
) -> tuple[dict[tuple[int, int], NDArray[np.float64]], dict[tuple[int, int], NDArray[np.ulonglong]]]:
    """Returns two dicts that maps (label1, label2) of an interface to array of vertices & triangles."""
    # encode interface (label1, label2) to unique id
    key_mult = np.amax(mesh.f[:, [3, 4]]) + 1
    keys, inv_1 = np.unique(mesh.f[:, 3] + mesh.f[:, 4] * key_mult, return_inverse=True)

    interfaces: list[tuple[int, int]] = [(key % key_mult, key // key_mult) for key in keys]
    faces_dict: dict[tuple[int, int], NDArray[np.uint]] = {
        interfaces[i]: mesh.f[:, :3][keys[inv_1] == keys[i]] for i in range(len(keys))
    }
    faces_interfaces: dict[tuple[int, int], NDArray[np.ulonglong]] = {}
    verts_interfaces: dict[tuple[int, int], NDArray[np.float64]] = {}

    for key in faces_dict:
        v, f = renormalize_verts(mesh.v, faces_dict[key])
        faces_interfaces[key] = f.copy()
        verts_interfaces[key] = v.copy()

    return (verts_interfaces, faces_interfaces)
