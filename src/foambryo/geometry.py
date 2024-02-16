"""Geometry on a DCEL Mesh.

computation of trijunction mean angles and lengths, interfaces/regions areas, regions volumes ; and their derivatives.

Sacha Ichbiah, Sept 2021.
Matthieu Perez, 2024.
"""
from typing import TYPE_CHECKING

import numpy as np
import torch
from numpy.typing import NDArray
from tqdm import tqdm

if TYPE_CHECKING:
    from foambryo.dcel import DcelData


def _find_key_multiplier(num_points: int) -> int:
    key_multiplier = 1
    while num_points // key_multiplier != 0:
        key_multiplier *= 10
    return key_multiplier


## LENGTHS AND DERIVATIVES


def compute_area_derivative_autodiff(
    mesh: "DcelData",
    device: str = "cpu",
) -> dict[tuple[int, int], NDArray[np.float64]]:
    """Compute dict that maps interface (label1, label2) to array of change of area per point."""
    # Faces_membranes = extract_faces_membranes(Mesh)
    key_mult = np.amax(mesh.f[:, 3:]) + 1
    keys = mesh.f[:, 3] + key_mult * mesh.f[:, 4]
    faces_membranes = {}
    for key in np.unique(keys):
        tup = (key % key_mult, key // key_mult)
        faces_membranes[tup] = mesh.f[:, :3][np.arange(len(keys))[keys == key]]

    verts = torch.tensor(mesh.v, dtype=torch.float, requires_grad=True).to(device)
    optimizer = torch.optim.SGD([verts], lr=1)  # Useless, here just to reset the grad

    areas_derivatives = {}
    for tup in sorted(faces_membranes.keys()):
        loss_area = (
            compute_area_faces_torch(verts, torch.tensor(faces_membranes[tup]))
        ).sum()
        loss_area.backward()
        areas_derivatives[tup] = (verts.grad).numpy().copy()
        optimizer.zero_grad()

    return areas_derivatives


def compute_volume_derivative_autodiff_dict(
    mesh: "DcelData", device: str = "cpu"
) -> dict[int, NDArray[np.float64]]:
    """Compute map cell number -> derivative of volume wrt to each point."""
    # Faces_manifolds = extract_faces_manifolds(Mesh)
    faces_manifolds = {key: [] for key in mesh.materials}
    for face in mesh.f:
        a, b, c, m1, m2 = face
        faces_manifolds[m1].append([a, b, c])
        faces_manifolds[m2].append([a, c, b])

    verts = torch.tensor(mesh.v, dtype=torch.float, requires_grad=True).to(device)
    optimizer = torch.optim.SGD([verts], lr=1)  # Useless, here just to reset the grad

    volumes_derivatives = {}
    for key in mesh.materials:  # 1:] :
        faces = faces_manifolds[key]
        assert len(faces) > 0
        loss_volume = -compute_volume_manifold_torch(verts, torch.tensor(faces))
        loss_volume.backward()
        volumes_derivatives[key] = verts.grad.numpy().copy()
        optimizer.zero_grad()

    return volumes_derivatives


def compute_length_derivative_autodiff(
    mesh: "DcelData",
    device: str = "cpu",
) -> dict[tuple[int, int], NDArray[np.float64]]:
    """Compute map trijunction edge (V1, V2) -> change of length wrt to points."""
    edges_trijunctions = extract_edges_trijunctions(mesh)

    verts = torch.tensor(mesh.v, dtype=torch.float, requires_grad=True).to(device)
    optimizer = torch.optim.SGD([verts], lr=1)  # Useless, here just to reset the grad

    length_derivatives: dict[tuple[int, int], NDArray[np.float64]] = {}
    for tup in sorted(edges_trijunctions.keys()):
        loss_length = (
            compute_length_edges_trijunctions_torch(
                verts, torch.tensor(edges_trijunctions[tup])
            )
        ).sum()
        loss_length.backward()
        length_derivatives[tup] = (verts.grad).numpy().copy()
        optimizer.zero_grad()

    return length_derivatives


def compute_length_edges_trijunctions_torch(
    points: torch.FloatTensor,
    edges_trijunctions: torch.IntTensor,
) -> torch.FloatTensor:
    """Compute the length of each edge of a trijunction using torch because why not."""
    positions = points[edges_trijunctions]
    return torch.norm(positions[:, 0] - positions[:, 1], dim=1)


def compute_length_trijunctions(
    mesh: "DcelData", prints: bool = False
) -> dict[tuple[int, int, int], float]:
    """Compute the total length of each trijunction of the mesh in a map (id reg 1, id reg 2, id reg 3) -> length."""
    length_trijunctions: dict[tuple[int, int, int], float] = {}
    edges_trijunctions = extract_edges_trijunctions(mesh, prints)
    for key in edges_trijunctions:
        length_trijunctions[key] = np.sum(
            compute_length_edges_trijunctions(mesh.v, edges_trijunctions[key])
        )
    return length_trijunctions


def compute_length_edges_trijunctions(
    points: NDArray[np.float64],
    edges_trijunctions: NDArray[np.uint],
) -> NDArray[np.float64]:
    """Compute the length of each edge of a trijunction ."""
    positions = points[edges_trijunctions]
    return np.linalg.norm(positions[:, 0] - positions[:, 1], axis=1)


def extract_edges_trijunctions(
    mesh: "DcelData", prints: bool = False
) -> dict[tuple[int, int, int], NDArray[np.uint]]:
    """Extract a dict that maps trijunctions (id reg 1, id reg 2, id reg 3) to a list of edges."""
    triangles_and_labels = mesh.f
    edges = np.vstack(
        (
            triangles_and_labels[:, [0, 1]],
            triangles_and_labels[:, [0, 2]],
            triangles_and_labels[:, [1, 2]],
        ),
    )
    edges = np.sort(edges, axis=1)
    zones = np.vstack(
        (
            triangles_and_labels[:, [3, 4]],
            triangles_and_labels[:, [3, 4]],
            triangles_and_labels[:, [3, 4]],
        ),
    )
    key_mult = _find_key_multiplier(len(mesh.v) + 1)
    keys = (edges[:, 0] + 1) + (edges[:, 1] + 1) * key_mult
    _, index_first_occurence, index_inverse, index_counts = np.unique(
        keys,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    if prints:
        print("Number of trijunctional edges :", np.sum(index_counts == 3))

    indices = np.arange(len(index_counts))
    trijunction_indices_to_edge_key = {key: [] for key in indices[index_counts == 3]}
    table = np.zeros(len(index_counts))
    table[index_counts == 3] += 1

    for i in range(len(index_inverse)):
        inverse = index_inverse[i]
        if table[inverse] == 1:
            trijunction_indices_to_edge_key[inverse].append(i)

    trijunctional_line = {}
    for key in sorted(trijunction_indices_to_edge_key.keys()):
        x = trijunction_indices_to_edge_key[key]
        regions = np.hstack((zones[x[0]], zones[x[1]], zones[x[2]]))
        u = np.unique(regions)
        if len(u) > 4:
            print("oui")
            continue
        else:
            trijunctional_line[tuple(u)] = trijunctional_line.get(tuple(u), [])
            trijunctional_line[tuple(u)].append(edges[x[0]])
            assert edges[x[0]][0] == edges[x[1]][0] == edges[x[2]][0]
            assert edges[x[0]][1] == edges[x[1]][1] == edges[x[2]][1]

    output_dict: dict[tuple[int, int, int], NDArray[np.uint]] = {}
    for key in sorted(trijunctional_line.keys()):
        output_dict[key] = np.vstack(trijunctional_line[key])
    return output_dict


## AREAS AND DERIVATIVES
def compute_area_faces_torch(
    points: torch.FloatTensor,
    triangles: torch.IntTensor,
) -> torch.FloatTensor:
    """Compute area of every triangle using torch."""
    positions = points[triangles]
    sides = positions - positions[:, [2, 0, 1]]

    lengths_sides = torch.norm(sides, dim=2)
    half_perimeters = torch.sum(lengths_sides, axis=1) / 2
    diffs = torch.zeros(lengths_sides.shape)
    diffs[:, 0] = half_perimeters - lengths_sides[:, 0]
    diffs[:, 1] = half_perimeters - lengths_sides[:, 1]
    diffs[:, 2] = half_perimeters - lengths_sides[:, 2]
    return (half_perimeters * diffs[:, 0] * diffs[:, 1] * diffs[:, 2]) ** (0.5)


def compute_areas_faces(mesh: "DcelData") -> None:
    """Compute area of every triangle in the mesh. Modify the info of the mesh."""
    positions = mesh.v[mesh.f[:, [0, 1, 2]]]
    sides = positions - positions[:, [2, 0, 1]]
    lengths_sides = np.linalg.norm(sides, axis=2)
    half_perimeters = np.sum(lengths_sides, axis=1) / 2

    diffs = np.array([half_perimeters] * 3).transpose() - lengths_sides
    areas = (half_perimeters * diffs[:, 0] * diffs[:, 1] * diffs[:, 2]) ** (0.5)
    for i, face in enumerate(mesh.faces):
        face.area = areas[i]


def compute_areas_cells(mesh: "DcelData") -> dict[int, float]:
    """Compute area of each cells."""
    areas: dict[int, float] = {key: 0 for key in mesh.materials}
    for face in mesh.faces:
        areas[face.material_1] += face.area
        areas[face.material_2] += face.area
    return areas


def compute_areas_interfaces(mesh: "DcelData") -> dict[tuple[int, int], float]:
    """Compute area of every interface (label1, label2) in mesh."""
    interfaces = {}
    for face in mesh.faces:
        materials = (face.material_1, face.material_2)
        key = (min(materials), max(materials))
        interfaces[key] = interfaces.get(key, 0) + face.area
    return interfaces


def compute_area_derivative_dict(
    mesh: "DcelData",
) -> dict[tuple[int, int], NDArray[np.float64]]:
    """Compute dict that maps interface (label1, label2) to array of change of area per point."""
    interfaces_keys: NDArray[np.int64] = np.array(
        sorted(compute_areas_interfaces(mesh).keys())
    )
    points, triangles, labels = mesh.v, mesh.f[:, :3], mesh.f[:, 3:]
    area_derivatives: dict[tuple[int, int], NDArray[np.float64]] = {
        tuple(t): np.zeros((len(points), 3)) for t in interfaces_keys
    }

    coords = points[triangles]
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    e3 = np.cross(z - y, x - z) / (
        np.linalg.norm(np.cross(z - y, x - z), axis=1).reshape(-1, 1)
    )
    cross_e3_x = np.cross(e3, z - y) / 2
    cross_e3_y = np.cross(e3, x - z) / 2
    cross_e3_z = np.cross(e3, y - x) / 2

    list_indices_faces_per_vertices_x = {
        key: [[] for i in range(len(points))] for key in area_derivatives
    }
    list_indices_faces_per_vertices_y = {
        key: [[] for i in range(len(points))] for key in area_derivatives
    }
    list_indices_faces_per_vertices_z = {
        key: [[] for i in range(len(points))] for key in area_derivatives
    }

    for i in range(len(triangles)):
        i_x, i_y, i_z = triangles[i]
        a, b = labels[i]
        list_indices_faces_per_vertices_x[(a, b)][i_x].append(i)
        list_indices_faces_per_vertices_y[(a, b)][i_y].append(i)
        list_indices_faces_per_vertices_z[(a, b)][i_z].append(i)

    for key in tqdm(area_derivatives.keys()):
        for iv in range(len(points)):
            area_derivatives[key][iv] = np.vstack(
                (
                    cross_e3_x[list_indices_faces_per_vertices_x[key][iv]],
                    cross_e3_y[list_indices_faces_per_vertices_y[key][iv]],
                    cross_e3_z[list_indices_faces_per_vertices_z[key][iv]],
                ),
            ).sum(axis=0)

    return area_derivatives


##VOLUMES AND DERIVATIVES


def compute_volume_manifold_torch(
    points: torch.FloatTensor,
    triangles: torch.IntTensor,
) -> torch.FloatTensor:
    """Compute volume of each cell."""
    coords = points[triangles]
    cross_prods = torch.cross(coords[:, 1], coords[:, 2], axis=1)
    dots = torch.sum(cross_prods * coords[:, 0], axis=1)
    return -torch.sum(dots) / 6


def compute_volume_cells(mesh: "DcelData") -> dict[int, float]:
    """Compute map cell number -> volume."""
    volumes: dict[int, float] = {m: 0 for m in mesh.materials}
    for i, face in enumerate(mesh.faces):
        index = mesh.f[i, [0, 1, 2]]
        coords = mesh.v[index]
        inc = np.linalg.det(coords)
        volumes[face.material_1] += inc
        volumes[face.material_2] -= inc

    for key in volumes:
        volumes[key] = volumes[key] / 6
    return volumes


def compute_volume_derivative_dict(mesh: "DcelData") -> dict[int, NDArray[np.float64]]:
    """Compute map cell number -> derivative of volume wrt to each point."""
    points, triangles, labels = mesh.v, mesh.f[:, :3], mesh.f[:, 3:]
    materials = mesh.materials

    volumes_derivatives: dict[int, NDArray[np.float64]] = {
        key: np.zeros((len(points), 3)) for key in materials
    }

    coords = points[triangles]
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    cross_xy = np.cross(x, y) / 6
    cross_yz = np.cross(y, z) / 6
    cross_zx = np.cross(z, x) / 6
    faces_material = {mat: np.zeros(len(triangles)) for mat in materials}
    for n in materials:
        faces_material[n][labels[:, 0] == 1] = 1
        faces_material[n][labels[:, 1] == 1] = -1

    list_indices_faces_per_vertices_pos_x = {
        key: [[] for i in range(len(points))] for key in materials
    }
    list_indices_faces_per_vertices_pos_y = {
        key: [[] for i in range(len(points))] for key in materials
    }
    list_indices_faces_per_vertices_pos_z = {
        key: [[] for i in range(len(points))] for key in materials
    }
    list_indices_faces_per_vertices_neg_x = {
        key: [[] for i in range(len(points))] for key in materials
    }
    list_indices_faces_per_vertices_neg_y = {
        key: [[] for i in range(len(points))] for key in materials
    }
    list_indices_faces_per_vertices_neg_z = {
        key: [[] for i in range(len(points))] for key in materials
    }

    for i in range(len(triangles)):
        i_x, i_y, i_z = triangles[i]
        a, b = labels[i]
        # print(i_x,i_y,i_z)
        list_indices_faces_per_vertices_pos_x[a][i_x].append(i)
        list_indices_faces_per_vertices_pos_y[a][i_y].append(i)
        list_indices_faces_per_vertices_pos_z[a][i_z].append(i)

        list_indices_faces_per_vertices_neg_x[b][i_x].append(i)
        list_indices_faces_per_vertices_neg_y[b][i_y].append(i)
        list_indices_faces_per_vertices_neg_z[b][i_z].append(i)

    for n in tqdm(materials):
        for iv in range(len(points)):
            volumes_derivatives[n][iv] = np.vstack(
                (
                    cross_yz[list_indices_faces_per_vertices_pos_x[n][iv]],
                    cross_zx[list_indices_faces_per_vertices_pos_y[n][iv]],
                    cross_xy[list_indices_faces_per_vertices_pos_z[n][iv]],
                    -cross_yz[list_indices_faces_per_vertices_neg_x[n][iv]],
                    -cross_zx[list_indices_faces_per_vertices_neg_y[n][iv]],
                    -cross_xy[list_indices_faces_per_vertices_neg_z[n][iv]],
                ),
            ).sum(axis=0)

    return volumes_derivatives


##ANGLES


def compute_angles_tri(  # noqa: C901
    mesh: "DcelData",
    unique: bool = True,
) -> tuple[
    dict[tuple[int, int, int], float],
    dict[tuple[int, int, int], float],
    dict[tuple[int, int, int], float],
]:
    """Compute three maps trijunction (id reg 1, id reg 2, id reg 3) to mean angle, mean angle (deg), length."""
    ##We compute the angles at each trijunctions. If we fall onto a quadrijunction, we skip it

    dict_length: dict[tuple[int, int, int], float] = {}
    dict_angles: dict[tuple[int, int, int], list[float]] = {}
    for edge in mesh.half_edges:
        if len(edge.twin) > 1:
            face = edge.incident_face
            faces = [face]
            sources = [edge.origin.key - edge.destination.key]
            normals = [face.normal]
            materials = [[face.material_1, face.material_2]]

            for neighbor in edge.twin:
                face_attached = mesh.half_edges[neighbor].incident_face
                faces.append(face_attached)
                sources.append(
                    mesh.half_edges[neighbor].origin.key
                    - mesh.half_edges[neighbor].destination.key
                )
                materials.append([face_attached.material_1, face_attached.material_2])
                normals.append(face_attached.normal)

            regions_id = np.array(materials)
            if len(regions_id) != 3:
                continue
                ## If we fall onto a quadrijunction, we skip it.

            normals = np.array(normals).copy()

            if (
                regions_id[0, 0] == regions_id[1, 0]
                or regions_id[0, 1] == regions_id[1, 1]
            ):
                regions_id[1] = regions_id[1][[1, 0]]
                normals[1] *= -1

            if (
                regions_id[0, 0] == regions_id[2, 0]
                or regions_id[0, 1] == regions_id[2, 1]
            ):
                regions_id[2] = regions_id[2][[1, 0]]
                normals[2] *= -1

            pairs = [[0, 1], [1, 2], [2, 0]]

            for pair in pairs:
                i1, i2 = pair
                # if np.isnan(np.arccos(np.dot(normals[i1],normals[i2]))) :
                #    print("Isnan")
                # if np.dot(normals[i1],normals[i2])>1 or np.dot(normals[i1],normals[i2])<-1 :
                #    print("Alert",np.dot(normals[i1],normals[i2]))
                angle = np.arccos(np.clip(np.dot(normals[i1], normals[i2]), -1, 1))

                if regions_id[i1][1] == regions_id[i2][0]:
                    e, f, g = regions_id[i1][0], regions_id[i1][1], regions_id[i2][1]

                elif regions_id[i1][0] == regions_id[i2][1]:
                    e, f, g = regions_id[i2][0], regions_id[i2][1], regions_id[i1][1]

                dict_angles[(min(e, g), f, max(e, g))] = dict_angles.get(
                    (min(e, g), f, max(e, g)), []
                )
                dict_angles[(min(e, g), f, max(e, g))].append(angle)
                dict_length[(min(e, g), f, max(e, g))] = dict_length.get(
                    (min(e, g), f, max(e, g)), 0
                )
                dict_length[(min(e, g), f, max(e, g))] += edge.length
                if not unique:
                    dict_angles[(min(e, g), f, max(e, g))] = dict_angles.get(
                        (min(e, g), f, max(e, g)), []
                    )
                    dict_angles[(min(e, g), f, max(e, g))].append(angle)
                    dict_length[(min(e, g), f, max(e, g))] = dict_length.get(
                        (min(e, g), f, max(e, g)), 0
                    )
                    dict_length[(min(e, g), f, max(e, g))] += edge.length

    dict_mean_angles: dict[tuple[int, int, int], float] = {}
    dict_mean_angles_deg: dict[tuple[int, int, int], float] = {}
    for key in dict_angles:
        dict_mean_angles[key] = np.mean(dict_angles[key])
        dict_mean_angles_deg[key] = np.mean(dict_mean_angles[key] * 180 / np.pi)

    return (dict_mean_angles, dict_mean_angles_deg, dict_length)
