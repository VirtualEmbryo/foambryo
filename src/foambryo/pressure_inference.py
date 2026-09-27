"""Module to infer pressures on a mesh.

Sacha Ichbiah 2021
Matthieu Perez 2024
"""

from collections.abc import Mapping
from enum import Enum
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from foambryo.tension_inference import _build_matrix_discrete_sparse

if TYPE_CHECKING:
    from foambryo.dcel import DcelData


class UndeterminedTensionError(ValueError):
    """A pressure inference was handed an interface tension that is not a finite number.

    Every pressure method solves for the cell pressures from the interface tensions, and the
    default variational method couples every cell to every tension, so a single non-finite
    tension makes nearly every inferred pressure non-finite: measured on five real embryo
    timepoints with one to four closed interfaces, all but one cell's pressure came back
    non-finite each time (54 of 55 cells up to 71 of 72). Such a field carries no usable value, so
    the pressure inference refuses it rather than returning it.

    The usual cause is not a numerical failure. An interface with no triple line at all -- a closed
    surface, such as an engulfed cell or a sealed lumen, or a fragment that a segmentation left
    inside another cell -- has no angle from which its tension could be measured, and
    `foambryo.infer_tensions` reports it as `float("nan")` by design. The other tensions of such a
    mesh remain valid; use `foambryo.infer_tensions_by_junction_component` to see which interfaces
    are undetermined and which component each determined tension belongs to.

    Subclasses `ValueError`, so code that already caught the generic error the least-squares
    solve raised on a non-finite input keeps working.

    Attributes:
        interfaces: the interfaces whose tension is not finite, sorted.
    """

    def __init__(self, interfaces: tuple[tuple[int, int], ...]) -> None:
        """Build the error and its message from the offending interfaces.

        Args:
            interfaces (tuple[tuple[int, int], ...]): the interfaces whose tension is not finite.
        """
        self.interfaces = tuple(sorted(interfaces))
        shown = ", ".join(str(key) for key in self.interfaces[:8])
        if len(self.interfaces) > 8:
            shown += f", ... ({len(self.interfaces)} in total)"
        message = (
            f"{len(self.interfaces)} interface tension(s) are not finite: {shown}. Pressures are "
            "not inferred, because the pressure solve couples the cells through the tensions and "
            "one non-finite tension makes nearly every pressure non-finite. An interface with no "
            "triple line (a closed surface) has no measurable tension and is reported as nan by "
            "infer_tensions; use infer_tensions_by_junction_component to see which interfaces are "
            "undetermined. The other tensions remain valid."
        )
        super().__init__(message)


def _require_finite_tensions(dict_tensions: Mapping[tuple[int, int], float]) -> None:
    """Refuse a tension map holding any value that is not a finite number.

    Checks every value of the map, because the default variational method consumes every one of
    them. Adds no floating-point operation to the solve: where every tension is finite, the
    inferred pressures are bitwise what they were without the check.

    Args:
        dict_tensions (Mapping[tuple[int, int], float]): interface -> tension.

    Raises:
        UndeterminedTensionError: naming every interface whose tension is `nan` or infinite.
    """
    nonfinite = tuple(key for key, value in dict_tensions.items() if not np.isfinite(value))
    if nonfinite:
        raise UndeterminedTensionError(nonfinite)


class PressureComputationMethod(Enum):
    """Describe all pressure inference methods.

    Variational: uses variational formulation (see our [article](https://doi.org/10.1038/s41592-023-02084-7)),
    Laplace: uses Laplace's law,
    WeightedLaplace: uses Laplace's law and weights curvature by area.
    """

    Variational = 0
    Laplace = 1
    WeightedLaplace = 2


def infer_pressures(
    mesh: "DcelData",
    dict_tensions: dict[tuple[int, int], float],
    mode: PressureComputationMethod = PressureComputationMethod.Variational,
    base_pressure: float = 0,
) -> dict[int, float]:
    """Infer pressure forces on each cell of a mesh using Laplace or Variational method.

    Args:
        mesh (DcelData): Mesh to analyze.
        dict_tensions (dict[tuple[int, int], float]): Map interface -> tensions
        mode (PressureComputationMethod, optional): Computation method. Defaults to Variational.
        base_pressure (float, optional): Base pressure for exterior. Defaults to 0.


    Returns:
        dict[int, float]: map cell id (0 = exterior) to pressure in cell.

    Raises:
        UndeterminedTensionError: if any tension in `dict_tensions` is not a finite number -- for
            example the `nan` that `infer_tensions` reports for an interface with no triple line.
            See `UndeterminedTensionError`.
    """
    _, dict_pressures, _ = infer_pressures_and_residuals(mesh, dict_tensions, mode, base_pressure)
    return dict_pressures


def infer_pressures_and_residuals(
    mesh: "DcelData",
    dict_tensions: dict[tuple[int, int], float],
    mode: PressureComputationMethod = PressureComputationMethod.Variational,
    base_pressure: float = 0,
) -> tuple[NDArray[np.float64], dict[int, float], NDArray[np.float64] | Literal[0]]:
    """Infer pressure forces on each cell of a mesh using Laplace or Variational method. Get residuals too.

    Args:
        mesh (DcelData): Mesh to analyze.
        dict_tensions (dict[tuple[int, int], float]): Map interface -> tensions
        mode (PressureComputationMethod, optional): Computation method. Defaults to Variational.
        base_pressure (float, optional): Base pressure for exterior. Defaults to 0.

    Returns:
        tuple[NDArray[np.float64], dict[int, float], NDArray[np.float64]]:
            - array of relative pressures ("Laplace") or of tensions ("Variational").
            - map cell id (0 = exterior) -> pressure
            - residuals of the least square method or 0 if mode is "Variational"

    Raises:
        UndeterminedTensionError: if any tension in `dict_tensions` is not a finite number -- for
            example the `nan` that `infer_tensions` reports for an interface with no triple line.
            Checked before any method runs; where every tension is finite the result is bitwise
            unchanged by the check.
    """
    _require_finite_tensions(dict_tensions)
    if mode == PressureComputationMethod.Laplace:
        return _infer_pressures_laplace(mesh, dict_tensions, base_pressure, weighted=False)
    elif mode == PressureComputationMethod.WeightedLaplace:
        return _infer_pressures_laplace(mesh, dict_tensions, base_pressure, weighted=True)
    else:  # mode == PressureComputationMethod.Variational:
        return _infer_pressures_variational(mesh, dict_tensions, base_pressure)


def _infer_pressures_laplace(
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


def _infer_pressures_variational(
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
    matrix_m, _, nm, _keys = _build_matrix_discrete_sparse(mesh)

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


def _extract_submatrices(
    matrix: NDArray[np.float64],
    nm: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    f_g = matrix[:nm, :nm]
    f_p = -matrix[:nm, nm:]
    g_g = matrix[nm:-1, :nm]
    g_p = -matrix[nm:-1, nm:]
    return (f_g, f_p, g_g, g_p)
