"""Module to infer tensions on a mesh.

Sacha Ichbiah 2021
Matthieu Perez 2024
"""
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

if TYPE_CHECKING:
    from foambryo.dcel import DcelData


class TensionComputationMethod(Enum):
    """Describe the formula used to infer the tensions from contact angles at each junction.

    It can be:
    - YoungDupre (Young-Dupré with cosines only),
    - ProjectionYoungDupre (Young-Dupré with cosines and sines),
    - Equilibrium,
    - Cotan (cotangent formula, see [Yamamoto et al. 2023](https://doi.org/10.1101/2023.03.07.531437)),
    - InvCotan (inverse of the cotangent formula),
    - Lami ([Lami's theorem](https://en.wikipedia.org/wiki/Lami%27s_theorem)),
    - InvLami (inverse of the Lami's relation),
    - LogLami (logarithm of the Lami's relation),
    - Variational (variational formulation, see our [paper](https://doi.org/10.1101/2023.04.12.536641)).
    """

    YoungDupre = 0
    ProjectionYoungDupre = 1
    Equilibrium = 2
    Cotan = 3
    InvCotan = 4
    Lami = 5
    InvLami = 6
    LogLami = 7
    Variational = 8


# FORCE INFERENCE ONLY TAKING INTO ACCOUNT TRIJUNCTIONS :
# THERE ARE A LOT OF QUADRIJUNCTIONS IN THE MESH
# HOWEVER, TAKING ONLY THE TRIJUNCTIONS INTO ACCOUNT SHOULD SUFFICE IN MOST CASES, ESPECIALLY WITH A FEW CELLS
# BECAUSE :
# THE PROBLEM IS HIGHLY OVERDETERMINED
# AND THE QUADRIJUNCTIONS ARE NOT VERY PRESENT (THEY OCCUPY A LITTLE LENGTH)
def infer_tensions(
    mesh: "DcelData",
    mean_tension: float = 1,
    mode: TensionComputationMethod = TensionComputationMethod.YoungDupre,
) -> dict[tuple[int, int], float]:
    """Infer tensions using chosen method.

    Args:
        mesh (DcelData): Mesh to analyze.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.
        mode (TensionComputationMethod, optional): method to infer tensions.

    Returns:
        dict[tuple[int, int], float]:
            - map interface id (label 1, label 2) -> tension on this interface
    """
    _, dict_tensions, _ = infer_tensions(mesh, mean_tension, mode)
    return dict_tensions


def infer_tensions(
    mesh: "DcelData",
    mean_tension: float = 1,
    mode: TensionComputationMethod = TensionComputationMethod.YoungDupre,
) -> tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
    """Infer tensions using chosen method. Get residuals too.

    Args:
        mesh (DcelData): Mesh to analyze.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.
        mode (TensionComputationMethod, optional): method to infer tensions.

    Returns:
        tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
            - array of relative tensions
            - map interface id (label 1, label 2) -> tension on this interface
            - residuals of the least square method.
    """
    if mode == TensionComputationMethod.ProjectionYoungDupre:
        return _infer_tension_projection_yd(mesh, mean_tension)
    elif mode == TensionComputationMethod.Equilibrium:
        return _infer_tension_equilibrium(mesh, mean_tension)
    elif mode == TensionComputationMethod.Cotan:
        return _infer_tension_cotan(mesh, mean_tension)
    elif mode == TensionComputationMethod.InvCotan:
        return _infer_tension_inv_cotan(mesh, mean_tension)
    elif mode == TensionComputationMethod.Lami:
        return _infer_tension_lami(mesh, mean_tension)
    elif mode == TensionComputationMethod.InvLami:
        return _infer_tension_inv_lami(mesh, mean_tension)
    elif mode == TensionComputationMethod.LogLami:
        return _infer_tension_lami_log(mesh, mean_tension)
    elif mode == TensionComputationMethod.Variational:
        return _infer_tension_variational_yd(mesh, mean_tension)
    else:  # if mode == TensionComputationMethod.YoungDupre:
        return _infer_tension_symmetrical_yd(mesh, mean_tension)


def _infer_tension_symmetrical_yd(
    mesh: "DcelData",
    mean_tension: float = 1,
) -> tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
    """Infer tensions using Symmetrical Young-Dupré method.

    Args:
        mesh (DcelData): Mesh to analyze.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.

    Returns:
        tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
            - array of relative tensions
            - map interface id (label 1, label 2) -> tension on this interface
            - residuals of the least square method.
    """
    dict_rad, _, dict_length = mesh.compute_angles_tri(unique=False)
    dict_areas = mesh.compute_areas_interfaces()
    matrix_m, vector_b = _build_matrix_tension_symmetrical_yd(dict_rad, dict_areas, dict_length, mean_tension)
    tensions, residuals, rank, sigma = linalg.lstsq(matrix_m, vector_b)
    dict_tensions = {}
    key_interfaces = np.array(sorted(dict_areas.keys()))
    nm = len(key_interfaces)
    for i in range(nm):
        dict_tensions[tuple(key_interfaces[i])] = tensions[i]

    return (tensions, dict_tensions, residuals)


def _build_matrix_tension_symmetrical_yd(
    dict_rad: dict[tuple[int, int, int], float],
    dict_areas: dict[tuple[int, int], float],
    dict_length_tri: dict[tuple[int, int, int], float],
    mean_tension: float = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Get matrix M and vector B of system MX=B to solve to get relative tensions.

    Args:
        dict_rad (dict[tuple[int, int, int], float]): Mean angles in radian at one side of a trijunction.
        dict_areas (dict[tuple[int, int], float]): Area of interfaces.
        dict_length_tri (dict[tuple[int, int, int], float]): Length of trijunctions.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]:
            - M
            - B
    """
    total_length_mesh = (
        sum(dict_length_tri.values()) / 3
    )  # total length of all the junctions on the mesh (on which are evaluated the surface tensions)

    # Index interfaces :
    keys_t = np.array(sorted(dict_areas.keys()))
    nb_zones = np.amax(keys_t) + 1
    linear_keys_t = keys_t[:, 0] * nb_zones + keys_t[:, 1]
    reverse_map_t = dict(zip(linear_keys_t, np.arange(len(linear_keys_t)), strict=False))

    # Index angles :
    keys_angles = np.array(list(dict_rad.keys()))
    keys_angles = -np.sort(-keys_angles, axis=1)
    ka = np.amax(keys_angles)
    linear_keys_angles = keys_angles[:, 0] * ka**2 + keys_angles[:, 1] * ka + keys_angles[:, 2]
    linear_keys_angles, index = np.unique(linear_keys_angles, return_index=True)
    keys_angles = keys_angles[index]

    # MX = B
    # x : structure : [0,nm[ : tensions
    nm = len(keys_t)
    nj = len(keys_angles)
    vector_b = np.zeros(3 * nj + 1)
    vector_b[-1] = nm * mean_tension  # Sum of tensions = number of membrane <=> mean of tensions = 1
    matrix_m = np.zeros((3 * nj + 1, nm))

    # CLASSICAL
    # YOUNG-DUPRÉ
    for i, key in enumerate(linear_keys_angles):
        c = key // ka**2
        b = (key - (c * (ka**2))) // ka
        a = key % ka

        angle_b = dict_rad[(a, b, c)]  # O_b = abc angle
        angle_a = dict_rad[(b, a, c)]  # O_a = cab angle
        angle_c = dict_rad[(a, c, b)]  # O_c = acb angle

        idx_tab = reverse_map_t[min(a, b) * nb_zones + max(a, b)]
        idx_tbc = reverse_map_t[min(b, c) * nb_zones + max(b, c)]
        idx_tac = reverse_map_t[min(a, c) * nb_zones + max(a, c)]

        factor = dict_length_tri[(a, b, c)] / total_length_mesh
        matrix_m[3 * i, idx_tab] = 1 * factor
        matrix_m[3 * i, idx_tbc] = np.cos(angle_b) * factor
        matrix_m[3 * i, idx_tac] = np.cos(angle_a) * factor

        matrix_m[3 * i + 1, idx_tab] = np.cos(angle_b) * factor
        matrix_m[3 * i + 1, idx_tbc] = 1 * factor
        matrix_m[3 * i + 1, idx_tac] = np.cos(angle_c) * factor

        matrix_m[3 * i + 2, idx_tab] = np.cos(angle_a) * factor
        matrix_m[3 * i + 2, idx_tbc] = np.cos(angle_c) * factor
        matrix_m[3 * i + 2, idx_tac] = 1 * factor

    matrix_m[3 * nj, :nm] = 1  # Enforces mean of tensions = 1 (see B[-1])

    return (matrix_m, vector_b)


def _infer_tension_projection_yd(
    mesh: "DcelData",
    mean_tension: float = 1,
) -> tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
    """Infer tensions using projection Young-Dupré method.

    Args:
        mesh (DcelData): Mesh to analyze.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.

    Returns:
        tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
            - array of relative tensions
            - map interface id (label 1, label 2) -> tension on this interface
            - residuals of the least square method.
    """
    dict_rad, _, dict_length = mesh.compute_angles_tri(unique=False)
    dict_areas = mesh.compute_areas_interfaces()
    matrix_m, vector_b = _build_matrix_tension_projection_yd(dict_rad, dict_areas, dict_length, mean_tension)
    x, resid, rank, sigma = linalg.lstsq(matrix_m, vector_b)
    dict_tensions = {}
    key_interfaces = np.array(sorted(dict_areas.keys()))
    nm = len(key_interfaces)
    for i in range(nm):
        dict_tensions[tuple(key_interfaces[i])] = x[i]

    return (x, dict_tensions, resid)


def _build_matrix_tension_projection_yd(
    dict_rad: dict[tuple[int, int, int], float],
    dict_areas: dict[tuple[int, int], float],
    dict_length_tri: dict[tuple[int, int, int], float],
    mean_tension: float = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Get matrix M and vector B of system MX=B to solve to get relative tensions.

    Args:
        dict_rad (dict[tuple[int, int, int], float]): Mean angles in radian at one side of a trijunction.
        dict_areas (dict[tuple[int, int], float]): Area of interfaces.
        dict_length_tri (dict[tuple[int, int, int], float]): Length of trijunctions.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]:
            - M
            - B
    """
    total_length_mesh = (
        sum(dict_length_tri.values()) / 3
    )  # total length of all the junctions on the mesh (on which are evaluated the surface tensions)

    # Index interfaces :
    keys_t = np.array(sorted(dict_areas.keys()))
    nb_zones = np.amax(keys_t) + 1
    linear_keys_t = keys_t[:, 0] * nb_zones + keys_t[:, 1]
    reverse_map_t = dict(zip(linear_keys_t, np.arange(len(linear_keys_t)), strict=False))

    # Index angles :
    keys_angles = np.array(list(dict_rad.keys()))
    keys_angles = -np.sort(-keys_angles, axis=1)
    ka = np.amax(keys_angles)
    linear_keys_a = keys_angles[:, 0] * ka**2 + keys_angles[:, 1] * ka + keys_angles[:, 2]
    linear_keys_a, index = np.unique(linear_keys_a, return_index=True)
    keys_angles = keys_angles[index]

    # MX = B
    # x : structure : [0,nm[ : tensions
    nm = len(keys_t)
    nj = len(keys_angles)
    vector_b = np.zeros(2 * nj + 1)
    vector_b[-1] = nm * mean_tension  # Sum of tensions = number of membrane <=> mean of tensions = 1
    matrix_m = np.zeros((2 * nj + 1, nm))

    # CLASSICAL
    # YOUNG-DUPRÉ
    for i, key in enumerate(linear_keys_a):
        c = key // ka**2
        b = (key - (c * (ka**2))) // ka
        a = key % ka
        angle_b = dict_rad[(a, b, c)]  # O_b = abc angle
        angle_a = dict_rad[(b, a, c)]  # O_a = cab angle
        idx_tab = reverse_map_t[min(a, b) * nb_zones + max(a, b)]
        idx_tbc = reverse_map_t[min(b, c) * nb_zones + max(b, c)]
        idx_tac = reverse_map_t[min(a, c) * nb_zones + max(a, c)]

        factor = dict_length_tri[(a, b, c)] / total_length_mesh
        matrix_m[2 * i, idx_tab] = 1 * factor
        matrix_m[2 * i, idx_tbc] = np.cos(angle_b) * factor
        matrix_m[2 * i, idx_tac] = np.cos(angle_a) * factor
        matrix_m[2 * i + 1, idx_tac] = -np.sin(angle_a) * factor
        matrix_m[2 * i + 1, idx_tbc] = np.sin(angle_b) * factor

    matrix_m[2 * nj, :nm] = 1  # Enforces mean of tensions = 1 (see B[-1])

    return (matrix_m, vector_b)


def _infer_tension_equilibrium(
    mesh: "DcelData",
    mean_tension: float = 1,
) -> tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
    """Infer tensions using Equilibrium method.

    Args:
        mesh (DcelData): Mesh to analyze.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.

    Returns:
        tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
            - array of relative tensions
            - map interface id (label 1, label 2) -> tension on this interface
            - residuals of the least square method.
    """
    dict_areas = mesh.compute_areas_interfaces()
    matrix_m, vector_b = _build_matrix_equilibrium_tensions(mesh, dict_areas, mean_tension)
    x, resid, rank, sigma = linalg.lstsq(matrix_m, vector_b)
    dict_tensions = {}
    key_interface = np.array(sorted(dict_areas.keys()))
    nm = len(key_interface)
    for i in range(nm):
        dict_tensions[tuple(key_interface[i])] = x[i]

    return (x, dict_tensions, resid)


def _build_matrix_equilibrium_tensions(
    mesh: "DcelData",
    dict_areas: dict[tuple[int, int], float],
    mean_tension: float = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Get matrix M and vector B of system MX=B to solve to get relative tensions.

    Args:
        mesh (DcelData): Mesh to analyze. We need the half-edge structure.
        dict_areas (dict[tuple[int, int], float]): Area of interfaces.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]:
            - M
            - B
    """
    # Index interfaces :
    keys_t = np.array(sorted(dict_areas.keys()))
    nb_zones = np.amax(keys_t) + 1
    linear_keys_t = keys_t[:, 0] * nb_zones + keys_t[:, 1]
    reverse_map_t = dict(zip(linear_keys_t, np.arange(len(linear_keys_t)), strict=False))

    # MX = B
    # x : structure : [0,nm[ : tensions
    nm = len(keys_t)
    table = [edge.key for edge in mesh.half_edges if len(edge.twin) > 1]

    ne = len(table)

    vector_b = np.zeros(3 * ne + 1)
    vector_b[-1] = nm * mean_tension  # Sum of tensions = number of membrane <=> mean of tensions = 1
    matrix_m = np.zeros((3 * ne + 1, nm))
    matrix_m[-1] = 1

    ne_i = 0
    for edge in mesh.half_edges:
        if len(edge.twin) == 1:
            continue

        list_edges = [edge.key, *edge.twin]
        for current_edge_idx in list_edges:
            # print(current_edge_idx)
            current_edge = mesh.half_edges[current_edge_idx]
            # face_attached = current_edge.incident_face
            # Find the edges of the faces
            e = current_edge.incident_face.outer_component  # current_edge
            edges_face = []
            for _ in range(3):
                # print(e.incident_face,e)
                edges_face.append([e.origin.key, e.destination.key])
                e = e.next
            edges_face = np.array(edges_face)

            # find the direction of the force vector
            verts_edge = mesh.v[[current_edge.origin.key, current_edge.destination.key]]

            index_edge = []
            index_not = []
            for index in np.unique(edges_face):
                if index in [current_edge.origin.key, current_edge.destination.key]:
                    index_edge.append(index)
                else:
                    index_not.append(index)

            # np.unique(edges_face),index,[edge.origin.key,edge.destination.key]
            # index_edge,index_not
            vector_tension = mesh.v[index_not[0]] - mesh.v[index_edge[0]]
            edge_vect = verts_edge[1] - verts_edge[0]
            edge_vect /= np.linalg.norm(edge_vect)
            vector_tension -= np.dot(vector_tension, edge_vect) * edge_vect
            vector_tension /= np.linalg.norm(vector_tension)

            a, b = (
                current_edge.incident_face.material_1,
                current_edge.incident_face.material_2,
            )
            idx_tension = reverse_map_t[min(a, b) * nb_zones + max(a, b)]

            matrix_m[3 * ne_i, idx_tension] = vector_tension[0]
            matrix_m[3 * ne_i + 1, idx_tension] = vector_tension[1]
            matrix_m[3 * ne_i + 2, idx_tension] = vector_tension[2]

        ne_i += 1

    return (matrix_m, vector_b)


def _infer_tension_cotan(
    mesh: "DcelData",
    mean_tension: float = 1,
) -> tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
    """Infer tensions using cotangents method.

    Args:
        mesh (DcelData): Mesh to analyze.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.

    Returns:
        tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
            - array of relative tensions
            - map interface id (label 1, label 2) -> tension on this interface
            - residuals of the least square method.
    """
    dict_rad, _, dict_length = mesh.compute_angles_tri(unique=False)
    dict_areas = mesh.compute_areas_interfaces()
    matrix_m, vector_b = _build_matrix_tension_cotan(dict_rad, dict_areas, dict_length, mean_tension)
    x, resid, rank, sigma = linalg.lstsq(matrix_m, vector_b)
    dict_tensions = {}
    key_interface = np.array(sorted(dict_areas.keys()))
    nm = len(key_interface)
    for i in range(nm):
        dict_tensions[tuple(key_interface[i])] = x[i]

    return (x, dict_tensions, resid)


def _build_matrix_tension_cotan(
    dict_rad: dict[tuple[int, int, int], float],
    dict_areas: dict[tuple[int, int], float],
    dict_length_tri: dict[tuple[int, int, int], float],
    mean_tension: float = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Get matrix M and vector B of system MX=B to solve to get relative tensions.

    Args:
        dict_rad (dict[tuple[int, int, int], float]): Mean angles in radian at one side of a trijunction.
        dict_areas (dict[tuple[int, int], float]): Area of interfaces.
        dict_length_tri (dict[tuple[int, int, int], float]): Length of trijunctions.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]:
            - M
            - B
    """
    total_length_mesh = (
        sum(dict_length_tri.values()) / 3
    )  # total length of all the junctions on the mesh (on which are evaluated the surface tensions)

    # Index interfaces :
    keys_t = np.array(sorted(dict_areas.keys()))
    nb_zones = np.amax(keys_t) + 1
    linear_keys_t = keys_t[:, 0] * nb_zones + keys_t[:, 1]
    reverse_map_t = dict(zip(linear_keys_t, np.arange(len(linear_keys_t)), strict=False))

    # Index angles :
    keys_angles = np.array(list(dict_rad.keys()))
    keys_angles = -np.sort(-keys_angles, axis=1)
    ka = np.amax(keys_angles)
    linear_keys_a = keys_angles[:, 0] * ka**2 + keys_angles[:, 1] * ka + keys_angles[:, 2]
    linear_keys_a, index = np.unique(linear_keys_a, return_index=True)
    keys_angles = keys_angles[index]

    # MX = B
    # x : structure : [0,nm[ : tensions
    nm = len(keys_t)
    nj = len(keys_angles)
    vector_b = np.zeros(3 * nj + 1)
    vector_b[-1] = nm * mean_tension  # Sum of tensions = number of membrane <=> mean of tensions = 1
    matrix_m = np.zeros((3 * nj + 1, nm))

    # CLASSICAL
    # YOUNG-DUPRÉ
    for i, key in enumerate(linear_keys_a):
        c = key // ka**2
        b = (key - (c * (ka**2))) // ka
        a = key % ka

        angle_b = dict_rad[(a, b, c)]  # O_b = abc angle
        angle_a = dict_rad[(b, a, c)]  # O_a = cab angle
        angle_c = dict_rad[(a, c, b)]
        idx_tab = reverse_map_t[min(a, b) * nb_zones + max(a, b)]
        idx_tbc = reverse_map_t[min(b, c) * nb_zones + max(b, c)]
        idx_tac = reverse_map_t[min(a, c) * nb_zones + max(a, c)]

        z1, z2, z3 = _compute_z(angle_a, angle_b, angle_c)
        f1, f2, f3, s1, s2, s3 = _factors_z(z1, z2, z3)

        factor = dict_length_tri[(a, b, c)] / total_length_mesh
        matrix_m[3 * i, idx_tab] = -s3 - f3 * factor
        matrix_m[3 * i, idx_tbc] = s3 * factor
        matrix_m[3 * i, idx_tac] = s3 * factor

        matrix_m[3 * i + 1, idx_tab] = s1 * factor
        matrix_m[3 * i + 1, idx_tbc] = -s1 - f1 * factor
        matrix_m[3 * i + 1, idx_tac] = s1 * factor

        matrix_m[3 * i + 2, idx_tab] = s2 * factor
        matrix_m[3 * i + 2, idx_tbc] = s2 * factor
        matrix_m[3 * i + 2, idx_tac] = -s2 - f2 * factor

    matrix_m[3 * nj, :nm] = 1  # Enforces mean of tensions = 1 (see B[-1])

    return (matrix_m, vector_b)


def _infer_tension_inv_cotan(
    mesh: "DcelData",
    mean_tension: float = 1,
) -> tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
    """Infer tensions using inverse cotangents method.

    Args:
        mesh (DcelData): Mesh to analyze.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.

    Returns:
        tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
            - array of relative tensions
            - map interface id (label 1, label 2) -> tension on this interface
            - residuals of the least square method.
    """
    dict_rad, _, dict_length = mesh.compute_angles_tri(unique=False)
    dict_areas = mesh.compute_areas_interfaces()
    matrix_m, vector_b = _build_matrix_tension_inv_cotan(dict_rad, dict_areas, dict_length, mean_tension)
    x, resid, rank, sigma = linalg.lstsq(matrix_m, vector_b)
    dict_tensions = {}
    key_interface = np.array(sorted(dict_areas.keys()))
    nm = len(key_interface)
    for i in range(nm):
        dict_tensions[tuple(key_interface[i])] = x[i]

    return (x, dict_tensions, resid)


def _build_matrix_tension_inv_cotan(
    dict_rad: dict[tuple[int, int, int], float],
    dict_areas: dict[tuple[int, int], float],
    dict_length_tri: dict[tuple[int, int, int], float],
    mean_tension: float = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Get matrix M and vector B of system MX=B to solve to get relative tensions.

    Args:
        dict_rad (dict[tuple[int, int, int], float]): Mean angles in radian at one side of a trijunction.
        dict_areas (dict[tuple[int, int], float]): Area of interfaces.
        dict_length_tri (dict[tuple[int, int, int], float]): Length of trijunctions.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]:
            - M
            - B
    """
    total_length_mesh = (
        sum(dict_length_tri.values()) / 3
    )  # total length of all the junctions on the mesh (on which are evaluated the surface tensions)

    # Index interfaces :
    keys_t = np.array(sorted(dict_areas.keys()))
    nb_zones = np.amax(keys_t) + 1
    linear_keys_t = keys_t[:, 0] * nb_zones + keys_t[:, 1]
    reverse_map_t = dict(zip(linear_keys_t, np.arange(len(linear_keys_t)), strict=False))

    # Index angles :
    keys_angles = np.array(list(dict_rad.keys()))
    keys_angles = -np.sort(-keys_angles, axis=1)
    ka = np.amax(keys_angles)
    linear_keys_angles = keys_angles[:, 0] * ka**2 + keys_angles[:, 1] * ka + keys_angles[:, 2]
    linear_keys_angles, index = np.unique(linear_keys_angles, return_index=True)
    keys_angles = keys_angles[index]

    # MX = B
    # x : structure : [0,nm[ : tensions
    nm = len(keys_t)
    nj = len(keys_angles)
    vector_b = np.zeros(3 * nj + 1)
    vector_b[-1] = nm * mean_tension  # Sum of tensions = number of membrane <=> mean of tensions = 1
    matrix_m = np.zeros((3 * nj + 1, nm))

    # CLASSICAL
    # YOUNG-DUPRÉ
    for i, key in enumerate(linear_keys_angles):
        c = key // ka**2
        b = (key - (c * (ka**2))) // ka
        a = key % ka

        angle_b = dict_rad[(a, b, c)]  # O_b = abc angle
        angle_a = dict_rad[(b, a, c)]  # O_a = cab angle
        angle_c = dict_rad[(a, c, b)]
        idx_tab = reverse_map_t[min(a, b) * nb_zones + max(a, b)]
        idx_tbc = reverse_map_t[min(b, c) * nb_zones + max(b, c)]
        idx_tac = reverse_map_t[min(a, c) * nb_zones + max(a, c)]

        z1, z2, z3 = _compute_z(angle_a, angle_b, angle_c)
        f1, f2, f3, s1, s2, s3 = _factors_z(z1, z2, z3)

        factor = dict_length_tri[(a, b, c)] / total_length_mesh
        matrix_m[3 * i, idx_tab] = -1 / s3 - 1 / f3 * factor
        matrix_m[3 * i, idx_tbc] = 1 / f3 * factor
        matrix_m[3 * i, idx_tac] = 1 / f3 * factor

        matrix_m[3 * i + 1, idx_tab] = 1 / f1 * factor
        matrix_m[3 * i + 1, idx_tbc] = -1 / s1 - 1 / f1 * factor
        matrix_m[3 * i + 1, idx_tac] = 1 / f1 * factor

        matrix_m[3 * i + 2, idx_tab] = 1 / f2 * factor
        matrix_m[3 * i + 2, idx_tbc] = 1 / f2 * factor
        matrix_m[3 * i + 2, idx_tac] = -1 / f2 - 1 / s2 * factor

    matrix_m[3 * nj, :nm] = 1  # Enforces mean of tensions = 1 (see B[-1])

    return (matrix_m, vector_b)


def _infer_tension_lami(
    mesh: "DcelData",
    mean_tension: float = 1,
) -> tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
    """Infer tensions using Lamy method.

    Args:
        mesh (DcelData): Mesh to analyze.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.

    Returns:
        tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
            - array of relative tensions
            - map interface id (label 1, label 2) -> tension on this interface
            - residuals of the least square method.
    """
    dict_rad, _, dict_length = mesh.compute_angles_tri(unique=False)
    dict_areas = mesh.compute_areas_interfaces()
    matrix_m, vector_b = _build_matrix_tension_lami(dict_rad, dict_areas, dict_length, mean_tension)
    x, resid, rank, sigma = linalg.lstsq(matrix_m, vector_b)
    dict_tensions = {}
    key_interface = np.array(sorted(dict_areas.keys()))
    nm = len(key_interface)
    for i in range(nm):
        dict_tensions[tuple(key_interface[i])] = x[i]

    return (x, dict_tensions, resid)


def _build_matrix_tension_lami(
    dict_rad: dict[tuple[int, int, int], float],
    dict_areas: dict[tuple[int, int], float],
    dict_length_tri: dict[tuple[int, int, int], float],
    mean_tension: float = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Get matrix M and vector B of system MX=B to solve to get relative tensions.

    Args:
        dict_rad (dict[tuple[int, int, int], float]): Mean angles in radian at one side of a trijunction.
        dict_areas (dict[tuple[int, int], float]): Area of interfaces.
        dict_length_tri (dict[tuple[int, int, int], float]): Length of trijunctions.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]:
            - M
            - B
    """
    total_length_mesh = (
        sum(dict_length_tri.values()) / 3
    )  # total length of all the junctions on the mesh (on which are evaluated the surface tensions)

    # Index interfaces :
    keys_t = np.array(sorted(dict_areas.keys()))
    nb_zones = np.amax(keys_t) + 1
    linear_keys_t = keys_t[:, 0] * nb_zones + keys_t[:, 1]
    reverse_map_t = dict(zip(linear_keys_t, np.arange(len(linear_keys_t)), strict=False))

    # Index angles :
    keys_angles = np.array(list(dict_rad.keys()))
    keys_angles = -np.sort(-keys_angles, axis=1)
    ka = np.amax(keys_angles)
    linear_keys_angles = keys_angles[:, 0] * ka**2 + keys_angles[:, 1] * ka + keys_angles[:, 2]
    linear_keys_angles, index = np.unique(linear_keys_angles, return_index=True)
    keys_angles = keys_angles[index]

    # MX = B
    # x : structure : [0,nm[ : tensions
    nm = len(keys_t)
    nj = len(keys_angles)
    vector_b = np.zeros(3 * nj + 1)
    vector_b[-1] = nm * mean_tension  # Sum of tensions = number of membrane <=> mean of tensions = 1
    matrix_m = np.zeros((3 * nj + 1, nm))

    # CLASSICAL
    # YOUNG-DUPRÉ
    for i, key in enumerate(linear_keys_angles):
        c = key // ka**2
        b = (key - (c * (ka**2))) // ka
        a = key % ka

        angle_b = dict_rad[(a, b, c)]  # O_b = abc angle
        angle_a = dict_rad[(b, a, c)]  # O_a = cab angle
        angle_c = dict_rad[(a, c, b)]
        idx_tab = reverse_map_t[min(a, b) * nb_zones + max(a, b)]
        idx_tbc = reverse_map_t[min(b, c) * nb_zones + max(b, c)]
        idx_tac = reverse_map_t[min(a, c) * nb_zones + max(a, c)]

        factor = dict_length_tri[(a, b, c)] / total_length_mesh
        matrix_m[3 * i, idx_tab] = 1 / np.sin(angle_c) * factor
        matrix_m[3 * i, idx_tbc] = -1 / np.sin(angle_a) * factor

        matrix_m[3 * i + 1, idx_tab] = 1 / np.sin(angle_c) * factor
        matrix_m[3 * i + 1, idx_tac] = -1 / np.sin(angle_b) * factor

        matrix_m[3 * i + 2, idx_tbc] = 1 / np.sin(angle_a) * factor
        matrix_m[3 * i + 2, idx_tac] = -1 / np.sin(angle_b) * factor

    matrix_m[3 * nj, :nm] = 1  # Enforces mean of tensions = 1 (see B[-1])

    return (matrix_m, vector_b)


def _infer_tension_inv_lami(
    mesh: "DcelData",
    mean_tension: float = 1,
) -> tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
    """Infer tensions using inverse Lamy method.

    Args:
        mesh (DcelData): Mesh to analyze.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.

    Returns:
        tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
            - array of relative tensions
            - map interface id (label 1, label 2) -> tension on this interface
            - residuals of the least square method.
    """
    dict_rad, _, dict_length = mesh.compute_angles_tri(unique=False)
    dict_areas = mesh.compute_areas_interfaces()
    matrix_m, vector_b = _build_matrix_tension_inv_lami(dict_rad, dict_areas, dict_length, mean_tension)
    x, resid, rank, sigma = linalg.lstsq(matrix_m, vector_b)
    dict_tensions = {}
    interface_keys = np.array(sorted(dict_areas.keys()))
    nm = len(interface_keys)
    for i in range(nm):
        dict_tensions[tuple(interface_keys[i])] = x[i]

    return (x, dict_tensions, resid)


def _build_matrix_tension_inv_lami(
    dict_rad: dict[tuple[int, int, int], float],
    dict_areas: dict[tuple[int, int], float],
    dict_length_tri: dict[tuple[int, int, int], float],
    mean_tension: float = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Get matrix M and vector B of system MX=B to solve to get relative tensions.

    Args:
        dict_rad (dict[tuple[int, int, int], float]): Mean angles in radian at one side of a trijunction.
        dict_areas (dict[tuple[int, int], float]): Area of interfaces.
        dict_length_tri (dict[tuple[int, int, int], float]): Length of trijunctions.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]:
            - M
            - B
    """
    total_length_mesh = (
        sum(dict_length_tri.values()) / 3
    )  # total length of all the junctions on the mesh (on which are evaluated the surface tensions)

    # Index interfaces :
    keys_t = np.array(sorted(dict_areas.keys()))
    nb_zones = np.amax(keys_t) + 1
    linear_keys_t = keys_t[:, 0] * nb_zones + keys_t[:, 1]
    reverse_map_t = dict(zip(linear_keys_t, np.arange(len(linear_keys_t)), strict=False))

    # Index angles :
    keys_angles = np.array(list(dict_rad.keys()))
    keys_angles = -np.sort(-keys_angles, axis=1)
    ka = np.amax(keys_angles)
    linear_keys_angles = keys_angles[:, 0] * ka**2 + keys_angles[:, 1] * ka + keys_angles[:, 2]
    linear_keys_angles, index = np.unique(linear_keys_angles, return_index=True)
    keys_angles = keys_angles[index]

    # MX = B
    # x : structure : [0,nm[ : tensions
    nm = len(keys_t)
    nj = len(keys_angles)
    vector_b = np.zeros(3 * nj + 1)
    vector_b[-1] = nm * mean_tension  # Sum of tensions = number of membrane <=> mean of tensions = 1
    matrix_m = np.zeros((3 * nj + 1, nm))

    # CLASSICAL
    # YOUNG-DUPRÉ
    for i, key in enumerate(linear_keys_angles):
        c = key // ka**2
        b = (key - (c * (ka**2))) // ka
        a = key % ka

        angle_b = dict_rad[(a, b, c)]  # O_b = abc angle
        angle_a = dict_rad[(b, a, c)]  # O_a = cab angle
        angle_c = dict_rad[(a, c, b)]
        idx_tab = reverse_map_t[min(a, b) * nb_zones + max(a, b)]
        idx_tbc = reverse_map_t[min(b, c) * nb_zones + max(b, c)]
        idx_tac = reverse_map_t[min(a, c) * nb_zones + max(a, c)]

        factor = dict_length_tri[(a, b, c)] / total_length_mesh
        matrix_m[3 * i, idx_tab] = np.sin(angle_a) * factor
        matrix_m[3 * i, idx_tbc] = -np.sin(angle_c) * factor

        matrix_m[3 * i + 1, idx_tab] = np.sin(angle_b) * factor
        matrix_m[3 * i + 1, idx_tac] = -np.sin(angle_c) * factor

        matrix_m[3 * i + 2, idx_tbc] = np.sin(angle_b) * factor
        matrix_m[3 * i + 2, idx_tac] = -np.sin(angle_a) * factor

    matrix_m[3 * nj, :nm] = 1  # Enforces mean of tensions = 1 (see B[-1])

    return (matrix_m, vector_b)


def _infer_tension_lami_log(
    mesh: "DcelData",
    mean_tension: float = 1,
) -> tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
    """Infer tensions using log Lamy method.

    Args:
        mesh (DcelData): Mesh to analyze.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.

    Returns:
        tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
            - array of relative tensions
            - map interface id (label 1, label 2) -> tension on this interface
            - residuals of the least square method.
    """
    dict_rad, _, dict_length = mesh.compute_angles_tri(unique=False)
    dict_areas = mesh.compute_areas_interfaces()
    matrix_m, vector_b = _build_matrix_tension_lami_log(dict_rad, dict_areas, dict_length, mean_tension)
    x_log, resid, rank, sigma = linalg.lstsq(matrix_m, vector_b)
    x = np.exp(x_log)
    dict_tensions = {}
    key_interface = np.array(sorted(dict_areas.keys()))
    nm = len(key_interface)
    for i in range(nm):
        dict_tensions[tuple(key_interface[i])] = x[i]

    return (x, dict_tensions, resid)


def _build_matrix_tension_lami_log(
    dict_rad: dict[tuple[int, int, int], float],
    dict_areas: dict[tuple[int, int], float],
    dict_length_tri: dict[tuple[int, int, int], float],
    mean_tension: float = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Get matrix M and vector B of system MX=B to solve to get relative tensions.

    Args:
        dict_rad (dict[tuple[int, int, int], float]): Mean angles in radian at one side of a trijunction.
        dict_areas (dict[tuple[int, int], float]): Area of interfaces.
        dict_length_tri (dict[tuple[int, int, int], float]): Length of trijunctions.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]:
            - M
            - B
    """
    total_length_mesh = (
        sum(dict_length_tri.values()) / 3
    )  # total length of all the junctions on the mesh (on which are evaluated the surface tensions)

    # Index interfaces :
    keys_t = np.array(sorted(dict_areas.keys()))
    nb_zones = np.amax(keys_t) + 1
    linear_keys_t = keys_t[:, 0] * nb_zones + keys_t[:, 1]
    reverse_map_t = dict(zip(linear_keys_t, np.arange(len(linear_keys_t)), strict=False))

    # Index angles :
    keys_angles = np.array(list(dict_rad.keys()))
    keys_angles = -np.sort(-keys_angles, axis=1)
    ka = np.amax(keys_angles)
    linear_keys_angles = keys_angles[:, 0] * ka**2 + keys_angles[:, 1] * ka + keys_angles[:, 2]
    linear_keys_angles, index = np.unique(linear_keys_angles, return_index=True)
    keys_angles = keys_angles[index]

    # MX = B
    # x : structure : [0,nm[ : tensions
    nm = len(keys_t)
    nj = len(keys_angles)
    vector_b = np.zeros(3 * nj + 1)
    vector_b[-1] = nm * np.log(mean_tension)  # Sum of tensions = number of membrane <=> mean of tensions = 1
    matrix_m = np.zeros((3 * nj + 1, nm))

    # CLASSICAL
    # YOUNG-DUPRÉ
    for i, key in enumerate(linear_keys_angles):
        c = key // ka**2
        b = (key - (c * (ka**2))) // ka
        a = key % ka

        angle_b = dict_rad[(a, b, c)]  # O_b = abc angle
        angle_a = dict_rad[(b, a, c)]  # O_a = cab angle
        angle_c = dict_rad[(a, c, b)]
        idx_tab = reverse_map_t[min(a, b) * nb_zones + max(a, b)]
        idx_tbc = reverse_map_t[min(b, c) * nb_zones + max(b, c)]
        idx_tac = reverse_map_t[min(a, c) * nb_zones + max(a, c)]

        factor = dict_length_tri[(a, b, c)] / total_length_mesh
        matrix_m[3 * i, idx_tab] = 1 * factor
        matrix_m[3 * i, idx_tbc] = -1 * factor
        vector_b[3 * i] = (np.log(np.sin(angle_c)) - np.log(np.sin(angle_a))) * factor

        matrix_m[3 * i + 1, idx_tab] = 1 * factor
        matrix_m[3 * i + 1, idx_tac] = -1 * factor
        vector_b[3 * i + 1] = (np.log(np.sin(angle_c)) - np.log(np.sin(angle_b))) * factor

        matrix_m[3 * i + 2, idx_tbc] = 1 * factor
        matrix_m[3 * i + 2, idx_tac] = -1 * factor
        vector_b[3 * i + 2] = (np.log(np.sin(angle_a)) - np.log(np.sin(angle_b))) * factor

    matrix_m[3 * nj, :nm] = 1  # Enforces mean of tensions = 1 (see B[-1])

    return (matrix_m, vector_b)


def _infer_tension_variational_yd(
    mesh: "DcelData",
    mean_tension: float = 1,
) -> tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
    """Infer tensions using Variational Young-Dupré method.

    Args:
        mesh (DcelData): Mesh to analyze.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.

    Returns:
        tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
            - array of relative tensions
            - map interface id (label 1, label 2) -> tension on this interface
            - residuals of the least square method.
    """
    area_derivatives, volume_derivatives = mesh.compute_area_derivatives(), mesh.compute_volume_derivatives()

    matrix_m, vector_b, nm = _build_matrix_discrete(area_derivatives, volume_derivatives, mesh.materials)

    f_g, f_p, g_g, g_p = _extract_submatrices(matrix_m, nm)

    mat_t = f_g - f_p @ (np.linalg.inv(g_p)) @ g_g

    mat_inference: NDArray[np.float64] = np.zeros((nm + 1, nm))
    mat_inference[:nm, :nm] = mat_t
    mat_inference[nm:] = 1
    b_inf = np.zeros(nm + 1)
    b_inf[-1] = nm * mean_tension  # Matthieu Perez: added mean_tension for consistency ?
    x, resid, rank, sigma = linalg.lstsq(mat_inference, b_inf)

    dict_tensions = {}

    key_interface = np.array(list(area_derivatives.keys()))
    nm = len(key_interface)

    for i in range(nm):
        dict_tensions[tuple(key_interface[i])] = x[i]

    return (x, dict_tensions, resid)


def _build_matrix_discrete(
    area_derivatives: dict[tuple[int, int], NDArray[np.float64]],
    volume_derivatives: dict[int, NDArray[np.float64]],
    materials: list[int],
    mean_tension: float = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Get matrix M and vector B of system MX=B to solve to get relative tensions.

    Args:
        area_derivatives (dict[tuple[int, int], NDArray[np.float64]]): derivative of area wrt to each point's position.
        volume_derivatives (dict[int, NDArray[np.float64]]): derivative of volume wrt to each point's position.
        materials (list[int]): List of materials in mesh.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64], int]:
            - M
            - B
            - number of interfaces in the mesh
    """
    keys_t = np.array(list(area_derivatives.keys()))
    nb_zones = np.amax(keys_t) + 1
    linear_keys_t = keys_t[:, 0] + keys_t[:, 1] * nb_zones
    reverse_map_t = dict(zip(linear_keys_t, np.arange(len(linear_keys_t)), strict=False))

    # MX = B
    # x : structure : ([0,nm[ : tensions) ([nm:nm+nc] : pressions) : (ya,yb,yc,yd...p0,p1,p2..pnc)
    nm = len(keys_t)
    nc = len(materials) - 1
    vector_b = np.zeros(nm + nc + 1)
    vector_b[-1] = nm * mean_tension  # Sum of tensions = number of membrane <=> mean of tensions = mean_tensions
    matrix_m = np.zeros((nm + nc + 1, nm + nc))
    # print("n_j",nj,"\nn_m",nm,"\nn_c",nc,"\nNumber of unknowns ",nm+nc+1)

    ###VARIATIONAL
    ###YOUNG-DUPRÉ
    ###EQUATIONS

    for i, tup_l in enumerate(area_derivatives.keys()):
        for tup_m in area_derivatives:
            a_m, b_m = tup_m
            key_m = a_m + b_m * nb_zones
            idx_tm = reverse_map_t[key_m]
            matrix_m[i, idx_tm] = np.sum(area_derivatives[tup_m] * area_derivatives[tup_l])

        for n, key_n in enumerate(materials[1:]):
            matrix_m[i, nm + n] = -np.sum(volume_derivatives[key_n] * area_derivatives[tup_l])

    off = nm

    ###VARIATIONAL
    ###LAPLACE
    ###EQUATIONS
    for i, key_i in enumerate(materials[1:]):
        for tup in area_derivatives:
            a, b = tup
            key = a + b * nb_zones
            idx_tk = reverse_map_t[key]
            matrix_m[off + i, idx_tk] = np.sum(area_derivatives[(a, b)] * volume_derivatives[key_i])

        for n, key_n in enumerate(materials[1:]):
            matrix_m[off + i, nm + n] = -np.sum(volume_derivatives[key_n] * volume_derivatives[key_i])

    off += nc

    lines_sum = np.sum(matrix_m, axis=1)
    for i in range(len(matrix_m) - 1):
        if lines_sum[i] != 0:
            matrix_m[i] /= lines_sum[i]

    matrix_m[off, :nm] = 1  # Enforces mean of tensions = mean_tensions (see B[-1])

    return (matrix_m, vector_b, nm)


def _extract_submatrices(
    matrix: NDArray[np.float64],
    nm: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    f_g = matrix[:nm, :nm]
    f_p = -matrix[:nm, nm:]
    g_g = matrix[nm:-1, :nm]
    g_p = -matrix[nm:-1, nm:]
    return (f_g, f_p, g_g, g_p)


###RESIDUAL STUFF


def _build_matrix_tension_validity(
    dict_rad: dict[tuple[int, int, int], float],
    dict_areas: dict[tuple[int, int], float],
    mean_tension: float = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Get matrix M and vector B of system MX=B to solve to get relative tensions.

    Args:
        dict_rad (dict[tuple[int, int, int], float]): Mean angles in radian at one side of a trijunction.
        dict_areas (dict[tuple[int, int], float]): Area of interfaces.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]:
            - M
            - B
    """
    # Index interfaces :
    keys_t = np.array(sorted(dict_areas.keys()))
    nb_zones = np.amax(keys_t) + 1
    linear_keys_t = keys_t[:, 0] * nb_zones + keys_t[:, 1]
    reverse_map_t = dict(zip(linear_keys_t, np.arange(len(linear_keys_t)), strict=False))

    # Index angles :
    keys_angles = np.array(list(dict_rad.keys()))
    keys_angles = -np.sort(-keys_angles, axis=1)
    ka = np.amax(keys_angles)
    linear_keys_angles = keys_angles[:, 0] * ka**2 + keys_angles[:, 1] * ka + keys_angles[:, 2]
    linear_keys_angles, index = np.unique(linear_keys_angles, return_index=True)
    keys_angles = keys_angles[index]

    # MX = B
    # x : structure : [0,nm[ : tensions
    nm = len(keys_t)
    nj = len(keys_angles)
    vector_b = np.zeros(3 * nj + 1)
    vector_b[-1] = nm * mean_tension  # Sum of tensions = number of membrane <=> mean of tensions = 1
    matrix_m = np.zeros((3 * nj + 1, nm))

    # CLASSICAL
    # YOUNG-DUPRÉ
    for i, key in enumerate(linear_keys_angles):
        c = key // ka**2
        b = (key - (c * (ka**2))) // ka
        a = key % ka

        angle_b = dict_rad[(a, b, c)]  # O_b = abc angle
        angle_a = dict_rad[(b, a, c)]  # O_a = cab angle
        angle_c = dict_rad[(a, c, b)]  # O_c = acb angle

        idx_tab = reverse_map_t[min(a, b) * nb_zones + max(a, b)]
        idx_tbc = reverse_map_t[min(b, c) * nb_zones + max(b, c)]
        idx_tac = reverse_map_t[min(a, c) * nb_zones + max(a, c)]

        matrix_m[3 * i, idx_tab] = 1
        matrix_m[3 * i, idx_tbc] = np.cos(angle_b)
        matrix_m[3 * i, idx_tac] = np.cos(angle_a)

        matrix_m[3 * i + 1, idx_tab] = np.cos(angle_b)
        matrix_m[3 * i + 1, idx_tbc] = 1
        matrix_m[3 * i + 1, idx_tac] = np.cos(angle_c)

        matrix_m[3 * i + 2, idx_tab] = np.cos(angle_a)
        matrix_m[3 * i + 2, idx_tbc] = np.cos(angle_c)
        matrix_m[3 * i + 2, idx_tac] = 1

    matrix_m[3 * nj, :nm] = 1  # Enforces mean of tensions = 1 (see B[-1])

    return (matrix_m, vector_b)


def compute_residual_junctions_dict(
    mesh: "DcelData",
    dict_tensions: dict[tuple[int, int], float],
    alpha: float = 0.05,
) -> dict[tuple[int, int, int], float]:
    """Compute residuals on trijunctions.

    Args:
        mesh (DcelData): Mesh to analyze.
        dict_tensions (dict[tuple[int, int], float]): Inferred tensions.
        alpha (float, optional): Quantile to apply. Defaults to 0.05.

    Returns:
        dict[tuple[int, int, int], float]: Map trijunction -> residue.
    """
    dict_rad, _, dict_length = mesh.compute_angles_tri(unique=False)
    dict_areas = mesh.compute_areas_interfaces()
    matrix_m, vector_b = _build_matrix_tension_validity(
        dict_rad,
        dict_areas,
        mean_tension=1,
    )

    keys_interface = np.array(sorted(dict_areas.keys()))
    nm = len(keys_interface)
    x = np.zeros(nm)
    for i in range(nm):
        x[i] = dict_tensions[tuple(keys_interface[i])]

    keys_angles = np.array(sorted(dict_rad.keys()))
    keys_angles = -np.sort(-keys_angles, axis=1)
    ka = np.amax(keys_angles)
    linear_keys_angles = keys_angles[:, 0] * ka**2 + keys_angles[:, 1] * ka + keys_angles[:, 2]
    linear_keys_angles, index = np.unique(linear_keys_angles, return_index=True)
    keys_angles = keys_angles[index]
    dict_residuals = {}
    array_resid = (np.abs(matrix_m @ x - vector_b) ** 2)[:-1]

    array_resid = array_resid.clip(np.quantile(array_resid, alpha), np.quantile(array_resid, 1 - alpha))

    for i, key in enumerate(linear_keys_angles):
        c = key // ka**2
        b = (key - (c * (ka**2))) // ka
        a = key % ka

        dict_residuals[(a, b, c)] = array_resid[3 * i + 0] + array_resid[3 * i + 1] + array_resid[3 * i + 2]

    return dict_residuals


# def infer_forces_variational_lt(mesh: "DcelData"):
#     # TODO
#     return None


def _compute_z(phi1: float, phi2: float, phi3: float) -> tuple[float, float, float]:
    z1 = np.sin(phi1) / (1 - np.cos(phi1))
    z2 = np.sin(phi2) / (1 - np.cos(phi2))
    z3 = np.sin(phi3) / (1 - np.cos(phi3))
    return (z1, z2, z3)


def _factors_z(z1: float, z2: float, z3: float) -> tuple[float, float, float, float, float, float]:
    f1 = z2 * z3
    f2 = z1 * z3
    f3 = z1 * z2
    s1 = z1 * (z2 + z3) / 2
    s2 = z2 * (z1 + z3) / 2
    s3 = z3 * (z1 + z2) / 2
    return (f1, f2, f3, s1, s2, s3)
