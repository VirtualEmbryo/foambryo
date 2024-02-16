"""Module to infer pressures on a mesh.

Sacha Ichbiah 2021
Matthieu Perez 2024
"""
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

if TYPE_CHECKING:
    from foambryo.dcel import DcelData


def _infer_pressure_laplace(
    mesh: "DcelData",
    dict_tensions: dict[tuple[int, int], float],
    base_pressure: float = 0,
    weighted: bool = False,
) -> tuple[NDArray[np.float64], dict[int, float], NDArray[np.float64]]:
    """Infer pressure forces on each cell of a mesh using Laplace method.

    Args:
        mesh (DcelData): Mesh to analyze.
        dict_tensions (dict[tuple[int, int], float]): Map interface -> tensions
        base_pressure (float, optional): Base pressure for exterior. Defaults to 0.
        weighted (bool, optional): Curvature weighted by area ?. Defaults to False.

    Returns:
        tuple[NDArray[np.float64], dict[int, float], NDArray[np.float64]]:
            - array of relative pressures (first value unknown ?)
            - map cell id (0 = exterior) -> pressure
            - residuals of the least square method
    """
    dict_areas = mesh.compute_areas_interfaces()
    dict_curvature = mesh.compute_curvatures_interfaces(weighted=weighted)

    matrix_m, vector_b = _build_matrix_laplace(
        dict_curvature,
        dict_areas,
        dict_tensions,
        mesh.n_materials,
        mesh.materials,
        base_pressure,
    )
    relative_pressures, residuals, _, _ = linalg.lstsq(matrix_m, vector_b)

    dict_pressures = {0: base_pressure}
    for i, key in enumerate(mesh.materials[1:]):
        dict_pressures[key] = relative_pressures[i + 1] + base_pressure

    return (relative_pressures, dict_pressures, residuals)


def _infer_pressure_variational(
    mesh: "DcelData",
    dict_tensions: dict[tuple[int, int], float],
    base_pressure: float = 0,
    prints: bool = False,
) -> tuple[NDArray[np.float64], dict[int, float], Literal[0]]:
    """Infer pressure forces on each cell of a mesh using variational method.

    Args:
        mesh (DcelData): Mesh to analyze.
        dict_tensions (dict[tuple[int, int], float]): Map interface -> tensions
        base_pressure (float, optional): Base pressure for exterior. Defaults to 0.
        prints (bool, optional): Verbose mode. Defaults to False.

    Returns:
        tuple[NDArray[np.float64], dict[int, float], NDArray[np.float64]]:
            - array of tensions ??? why ?
            - map cell id (0 = exterior) -> pressure
            - 0 (no residuals)
    """
    area_derivatives, volume_derivatives = mesh.compute_area_derivatives(), mesh.compute_volume_derivatives()

    matrix_m, _, nm = _build_matrix_discrete(area_derivatives, volume_derivatives, mesh.materials)

    f_g, f_p, g_g, g_p = _extract_submatrices(matrix_m, nm)

    mat_t = f_g - f_p @ (np.linalg.inv(g_p)) @ g_g
    if prints:
        print(f_g.shape, f_p.shape, g_g.shape, g_p.shape)
        print("Det of F_g:", np.linalg.det(f_g))
        print("Det of G_p:", np.linalg.det(g_p))

    mat_inference = np.zeros((nm + 1, nm))
    mat_inference[:nm, :nm] = mat_t
    mat_inference[nm:] = 1
    b_inf = np.zeros(nm + 1)
    b_inf[-1] = nm

    dict_pressures = {0: base_pressure}

    x = np.zeros(len(dict_tensions.keys()))
    for i, key in enumerate(sorted(dict_tensions.keys())):
        x[i] = dict_tensions[key]

    relative_pressures = np.linalg.inv(g_p) @ g_g @ x

    for i, key in enumerate(mesh.materials[1:]):
        dict_pressures[key] = relative_pressures[i] + base_pressure

    return (x, dict_pressures, 0)


def infer_pressure(
    mesh: "DcelData",
    dict_tensions: dict[tuple[int, int], float],
    mode: Literal["Variational"] | Literal["Laplace"] = "Variational",
    base_pressure: float = 0,
    weighted: bool = False,
) -> tuple[NDArray[np.float64], dict[int, float], NDArray[np.float64] | Literal[0]]:
    """Infer pressure forces on each cell of a mesh using Laplace method.

    Args:
        mesh (DcelData): Mesh to analyze.
        dict_tensions (dict[tuple[int, int], float]): Map interface -> tensions
        mode (Literal["Variational"] | Literal["Laplace"], optional): Computation method. Defaults to "Variational".
        base_pressure (float, optional): Base pressure for exterior. Defaults to 0.
        weighted (bool, optional): Curvature weighted by area ? For "Laplace" mode. Defaults to False.

    Returns:
        tuple[NDArray[np.float64], dict[int, float], NDArray[np.float64]]:
            - array of relative pressures ("Laplace") or of tensions ("Variational").
            - map cell id (0 = exterior) -> pressure
            - residuals of the least square method or 0 if mode is "Variational"
    """
    if mode == "Laplace":
        return _infer_pressure_laplace(mesh, dict_tensions, base_pressure, weighted=weighted)
    elif mode == "Variational":
        return _infer_pressure_variational(mesh, dict_tensions, base_pressure)
    else:
        print("Unimplemented method")
        return None


def _build_matrix_laplace(
    dict_curvature: dict[tuple[int, int], float],
    dict_areas: dict[tuple[int, int], float],
    dict_tensions: dict[tuple[int, int], float],
    n_materials: int,
    materials: NDArray[np.int64],
    base_pressure: float = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Get matrix M and vector B of system MX=B to solve to get relative pressures.

    Args:
        dict_curvature(dict[tuple[int, int], float]): Mean curvature of interfaces.
        dict_areas(dict[tuple[int, int], float]): Area of interfaces.
        dict_tensions(dict[tuple[int, int], float]): Tensions of interfaces.
        n_materials(int): Number of cells + 1.
        materials(NDArray[np.float64]): label of cells.
        base_pressure(float): Base pressure of exterior region.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]:
        - M
        - B
    """
    total_area_mesh = sum(dict_areas.values())  # sum of the areas of all the surfaces of the mesh
    interfaces = np.array(list(dict_curvature.keys()))
    kt = np.amax(interfaces) + 1
    keys_t = interfaces[:, 0] * kt + interfaces[:, 1]

    # MX = B
    # x : structure : [0,nc[ : pressions
    nm = len(interfaces)
    nc = n_materials - 1

    reverse_map_materials = dict(zip(materials, np.arange(len(materials)), strict=False))

    vector_b = np.zeros(nm + 1)
    vector_b[-1] = base_pressure  # Po = Pb with Pb = 1
    matrix_m = np.zeros((nm + 1, nc + 1))

    # CLASSICAL
    # LAPLACE
    for i, key in enumerate(keys_t):
        a = key // kt
        b = key % kt
        curvature_k = dict_curvature[(a, b)]
        tension_k = dict_tensions[(a, b)]

        if np.isnan(curvature_k):
            continue

        area_k = dict_areas[(a, b)]
        factor = area_k / total_area_mesh

        vector_b[i] = -2 * curvature_k * tension_k * factor  # +2Hk*Tk
        #  -> Sign is negative because we choose the following convention :
        # If 1 goes inside 2 the interface viewed from 1 is concave, viewed from 2 it is convex.
        # The curvature is noted negatively : Hk<0. Besides,  P1 > P2 and Tk > 0 thus laplace relation writes :
        # P1 - P2 = -2*Hk*Tk  <=> P1 - P2 + 2*Hk*Tk = 0
        am = reverse_map_materials[a]
        bm = reverse_map_materials[b]
        matrix_m[i, am] = 1 * factor  # Pi
        matrix_m[i, bm] = -1 * factor  # Pj

    matrix_m[-1, 0] = 1  # Enforces Po = Pb (see B[-2])
    # print(B)
    """
    pM = np.linalg.pinv(M)
    x = pM@B"""

    return (matrix_m, vector_b)


def _build_matrix_discrete(
    area_derivatives: dict[tuple[int, int], NDArray[np.float64]],
    volume_derivatives: dict[tuple[int, int], NDArray[np.float64]],
    materials: NDArray[np.int64],
    mean_tension: float = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Return matrix M and vector B and number of interfaces in mesh. MX = B can be solved to get relative pressures.

    Args:
        area_derivatives (dict[tuple[int, int], NDArray[np.float64]]): Derivative of areas wrt vertices.
        volume_derivatives (dict[tuple[int, int], NDArray[np.float64]]): Derivative of volumes wrt vertices.
        materials (NDArray[np.int64]): Labels of cells.
        mean_tension (float, optional): Mean tension of interfaces. Defaults to 1.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64], int]: M, B, and number of interfaces.
    """
    interfaces = np.array(list(area_derivatives.keys()))
    kt = np.amax(interfaces) + 1
    keys_t = interfaces[:, 0] + interfaces[:, 1] * kt
    reverse_map_t = dict(zip(keys_t, np.arange(len(keys_t)), strict=False))

    # MX = B
    # x : structure : ([0,nm[ : tensions) ([nm:nm+nc] : pressions) : (ya,yb,yc,yd...p0,p1,p2..pnc)
    nm = len(interfaces)
    nc = len(materials) - 1
    vector_b = np.zeros(nm + nc + 1)
    vector_b[-1] = nm * mean_tension  # Sum of tensions = number of membrane <=> mean of tensions = mean_tensions
    matrix_m = np.zeros((nm + nc + 1, nm + nc))
    # print("n_j",nj,"\nn_m",nm,"\nn_c",nc,"\nNumber of unknowns ",nm+nc+1)

    ###DISCRETE
    ###YOUNG-DUPRÉ
    ###EQUATIONS

    for i, tup_l in enumerate(area_derivatives.keys()):
        a_l, b_l = tup_l

        for tup_m in area_derivatives:
            a_m, b_m = tup_m
            key_m = a_m + b_m * kt
            idx_tm = reverse_map_t[key_m]
            matrix_m[i, idx_tm] = np.sum(area_derivatives[tup_m] * area_derivatives[tup_l])

        for n, key_n in enumerate(materials[1:]):
            matrix_m[i, nm + n] = -np.sum(volume_derivatives[key_n] * area_derivatives[tup_l])

    off = nm

    ###DISCRETE
    ###LAPLACE
    ###EQUATIONS
    for i, key_i in enumerate(materials[1:]):
        for tup in area_derivatives:
            a, b = tup
            key = a + b * kt
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
