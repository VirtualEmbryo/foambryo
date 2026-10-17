"""Cell pressures inferred from a multimaterial mesh and its interface tensions.

The pressures are solved from the tensions (`foambryo.tension_inference`), by the variational
method (the default) or by Laplace's law (`PressureComputationMethod`). They are in the unit of the
tensions divided by the mesh's length unit: the tensions being relative (normalised to
`mean_tension`), so are the pressures, which scale with the tensions. The exterior medium, label 0,
has the pressure `base_pressure`; a cell whose pressure the mesh does not determine, or a cell the
crop cuts, is `nan`, never a number.

Sacha Ichbiah 2021
Matthieu Perez 2024
"""

import time
import warnings
from collections.abc import Mapping
from enum import Enum
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, onenormest, splu

from foambryo import linear_solves
from foambryo.crop_faces import vertex_rows_off_the_crop, warn_on_cut_cell_pressures
from foambryo.geometry import compute_area_derivative_sparse, compute_volume_derivative_sparse
from foambryo.junction_graph import _solve_least_squares
from foambryo.tension_inference import _build_matrix_discrete_sparse

if TYPE_CHECKING:
    from scipy.sparse.linalg import SuperLU

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


class UndeterminedPressureWarning(UserWarning):
    """Emitted when the Laplace pressure leaves cells undetermined; their pressure is `nan`.

    Also emitted by the variational method on a cropped mesh, for a cell that the crop does not
    cut but whose every vertex lies on a crop face or on an interface without a tension.

    The Laplace methods write one equation per interface, `P_a - P_b = -2 H_ab T_ab`, which
    needs the interface's mean curvature `H_ab`. The curvature is read at the vertices of the
    interface that are not on a junction line, so an interface all of whose vertices lie on
    junctions has no curvature (`nan`), and its equation is left out. When the equations that
    remain do not connect a cell to the exterior, through a chain of interfaces, nothing fixes
    that cell's pressure: it is reported as `nan`, never as a number. This is common on coarse
    meshes of small cells, where up to about a third of the interfaces can have no curvature and a
    few cells per hundred can be left undetermined.

    The variational method (the default) does not use curvatures and is not affected.
    """


def warn_on_undetermined_pressures(cells: tuple[int, ...], stacklevel: int = 4) -> None:
    """Emit `UndeterminedPressureWarning` naming the cells whose Laplace pressure is `nan`.

    Args:
        cells (tuple[int, ...]): Labels of the undetermined cells (nothing is emitted if empty).
        stacklevel (int, optional): passed through to `warnings.warn`, so the warning points
            at the caller's own line rather than at this module. Defaults to 4.
    """
    if not cells:
        return
    shown = ", ".join(str(c) for c in cells[:20])
    if len(cells) > 20:
        shown += f", ... ({len(cells)} in total)"
    message = (
        f"the Laplace pressure of {len(cells)} cell(s) is undetermined and reported as nan: {shown}. "
        "Every interface linking these cells to the exterior has no measurable curvature (all of "
        "its vertices lie on junction lines), so no equation fixes their pressure. The other "
        "pressures are unaffected. The variational method (the default) does not need curvatures."
    )
    warnings.warn(message, UndeterminedPressureWarning, stacklevel=stacklevel)


def _laplace_undetermined_cells(
    dict_curvature: Mapping[tuple[int, int], float],
    materials: NDArray[np.int64],
) -> tuple[int, ...]:
    """The cells that the interfaces with a finite curvature do not connect to the exterior.

    The Laplace system's rows are the interfaces with a finite curvature plus the exterior's pin,
    so its solution is unique exactly on the connected component of the exterior in the graph
    of cells joined by those interfaces; every other cell's pressure is undetermined.

    Args:
        dict_curvature (Mapping[tuple[int, int], float]): Mean curvature of every interface.
        materials (NDArray[np.int64]): The mesh's materials, the exterior first.

    Returns:
        tuple[int, ...]: Labels of the undetermined cells, sorted (empty if none).
    """
    column = {int(m): i for i, m in enumerate(materials)}
    live = [(column[int(a)], column[int(b)]) for (a, b), k in dict_curvature.items() if np.isfinite(k)]
    rows = [a for a, _ in live]
    cols = [b for _, b in live]
    n = len(materials)
    _, component = connected_components(coo_matrix((np.ones(len(live)), (rows, cols)), shape=(n, n)), directed=False)
    return tuple(sorted(int(m) for m, c in zip(materials, component, strict=True) if c != component[0]))


class PressureComputationMethod(Enum):
    """The pressure inference methods.

    - Variational (the default): the least-squares solution of the vertex force balance at the
      given tensions (see our [article](https://doi.org/10.1038/s41592-023-02084-7)). It reads no
      curvature, gives a number for every cell of an uncropped mesh, and is the only method with a
      formulation for crop faces.
    - Laplace: Laplace's law, `P_a - P_b = -2 H_ab T_ab`, on each interface, with the interface's
      mean curvature `H_ab` averaged over its vertices off the junction lines.
    - WeightedLaplace: the same, with the curvature averaged with area weights.

    The two Laplace methods need each interface's mean curvature. An interface with no
    measurable curvature contributes no equation, and a cell that the remaining equations do
    not connect to the exterior gets the pressure `nan` (see `UndeterminedPressureWarning`).
    They refuse a cropped mesh (`ValueError`); `has_crop_face_formulation` says which methods
    accept one.
    """

    Variational = 0
    Laplace = 1
    WeightedLaplace = 2

    @property
    def has_crop_face_formulation(self) -> bool:
        """Whether the method infers pressures on a cropped mesh (one with crop faces, `foambryo.crop_faces`).

        Only `Variational` does: the two Laplace methods read each interface's curvature, which on an interface the
        crop cuts is read partly on the crop, and `infer_pressures` refuses them on a cropped mesh. A front end can
        offer only the methods that have the formulation instead of failing after the tensions.
        """
        return self is PressureComputationMethod.Variational


def infer_pressures(
    mesh: "DcelData",
    dict_tensions: dict[tuple[int, int], float],
    mode: PressureComputationMethod = PressureComputationMethod.Variational,
    base_pressure: float = 0,
    *,
    solver: str = "auto",
    sparse_threshold: int | None = None,
    diagnostics: bool | str = False,
) -> "dict[int, float] | tuple[dict[int, float], linear_solves.SolveDiagnostics]":
    """Infer the pressure of every cell of a mesh from its interface tensions.

    The pressures are in the unit of the tensions divided by the mesh's length unit, and scale
    with the tensions: tensions normalised to a mean of 1 give pressures relative to that scale.

    **Cropped meshes** (`foambryo.crop_faces`, variational method only): the pressure of every
    cut cell is solved for and returned as `nan`, with a `CutCellPressureWarning` naming the
    cells; the force balances of the vertices on crop faces, and of the vertices of interfaces
    whose tension is `nan`, are left out, so `nan` tensions are accepted there; a cell those
    balances leave undetermined is `nan` too (`UndeterminedPressureWarning`). The crop interfaces
    need no entry in `dict_tensions`.

    Args:
        mesh (DcelData): Mesh to analyze, in any length unit.
        dict_tensions (dict[tuple[int, int], float]): interface `(label_1, label_2)`,
            `label_1 < label_2`, label 0 the exterior medium -> tension, one entry per interface of
            the mesh, as `foambryo.infer_tensions` returns it; all on one scale (tensions from
            separately normalised pieces of the junction graph are used as given). On an uncropped
            mesh every value must be finite; on a cropped mesh the crop interfaces may be absent
            and undetermined tensions `nan`.
        mode (PressureComputationMethod, optional): Computation method. Defaults to
            `PressureComputationMethod.Variational`.
        base_pressure (float, optional): Pressure of the exterior medium (label 0), returned as
            its pressure. Defaults to 0.
        solver (str, optional): `"auto"` (the default), `"dense"` or `"sparse"`: how the
            pressure system is solved (`foambryo.linear_solves`). `"auto"` is dense below
            `sparse_threshold` unknowns (materials for the Laplace methods, interfaces plus cells for
            the variational one) and sparse at or above, and bitwise the dense result below it. On
            a cropped mesh `"auto"` decides by `DENSE_CROP_PRESSURE_MAX_ENTRIES` instead.
            Keyword-only.
        sparse_threshold (int | None, optional): unknowns at which `"auto"` goes sparse; `None`
            (the default) is `foambryo.junction_graph.SPARSE_SOLVE_MIN_UNKNOWNS` (2 000).
            Keyword-only.
        diagnostics (bool | str, optional): `False` (the default), `True` or `"spectrum"`: also
            return a `foambryo.linear_solves.SolveDiagnostics`, last, with one record per solve.
            Keyword-only.

    Returns:
        dict[int, float]: map cell label (0 = exterior, at `base_pressure`) -> pressure. With a
            Laplace method, a cell whose pressure no equation fixes is `nan` (and
            `UndeterminedPressureWarning` is emitted); the variational method returns a number for
            every cell of an uncropped mesh. With `diagnostics` on, the tuple
            `(map, SolveDiagnostics)`.

    Raises:
        UndeterminedTensionError: on an uncropped mesh, if any tension in `dict_tensions` is not a
            finite number -- for example the `nan` that `infer_tensions` reports for an interface
            with no triple line. Checked before any method runs; where every tension is finite
            the result is bitwise unchanged by the check. See `UndeterminedTensionError`.
        ValueError: on a cropped mesh with a Laplace method, which has no formulation for crop
            faces (`PressureComputationMethod.has_crop_face_formulation`); on an invalid
            `solver`, `sparse_threshold` or `diagnostics`; with the variational method, when
            `dict_tensions` does not hold one tension per interface of an uncropped mesh.
        KeyError: with a Laplace method, when `dict_tensions` misses an interface of the mesh.
        numpy.linalg.LinAlgError: if the variational system is singular.

    Warns:
        UndeterminedPressureWarning: when a cell's pressure is undetermined and reported `nan`.
        CutCellPressureWarning: on a cropped mesh, naming the cut cells reported `nan`.
        SparseSolveFallbackWarning: when `lsqr` did not converge on a large Laplace system.
        DenseSolveMemoryWarning: when `solver="dense"` is forced on a system too large for the
            memory available.
    """
    result = infer_pressures_and_residuals(
        mesh, dict_tensions, mode, base_pressure,
        solver=solver, sparse_threshold=sparse_threshold, diagnostics=diagnostics,
    )
    if diagnostics is False:
        return result[1]
    return (result[1], result[3])


def infer_pressures_and_residuals(
    mesh: "DcelData",
    dict_tensions: dict[tuple[int, int], float],
    mode: PressureComputationMethod = PressureComputationMethod.Variational,
    base_pressure: float = 0,
    *,
    solver: str = "auto",
    sparse_threshold: int | None = None,
    diagnostics: bool | str = False,
) -> tuple:
    """Infer the pressure of every cell of a mesh from its interface tensions, with the residuals.

    `infer_pressures` returns this function's second value; see it for the units and for cropped
    meshes.

    Args:
        mesh (DcelData): Mesh to analyze, in any length unit.
        dict_tensions (dict[tuple[int, int], float]): interface `(label_1, label_2)`,
            `label_1 < label_2`, label 0 the exterior medium -> tension, one entry per interface of
            the mesh, as `foambryo.infer_tensions` returns it; all on one scale (tensions from
            separately normalised pieces of the junction graph are used as given). On an uncropped
            mesh every value must be finite; on a cropped mesh the crop interfaces may be absent
            and undetermined tensions `nan`.
        mode (PressureComputationMethod, optional): Computation method. Defaults to
            `PressureComputationMethod.Variational`.
        base_pressure (float, optional): Pressure of the exterior medium (label 0), returned as
            its pressure. Defaults to 0.
        solver (str, optional): `"auto"` (the default), `"dense"` or `"sparse"`: how the
            pressure system is solved (`foambryo.linear_solves`). `"auto"` is dense below
            `sparse_threshold` unknowns (materials for the Laplace methods, interfaces plus cells for
            the variational one) and sparse at or above, and bitwise the dense result below it. On
            a cropped mesh `"auto"` decides by `DENSE_CROP_PRESSURE_MAX_ENTRIES` instead.
            Keyword-only.
        sparse_threshold (int | None, optional): unknowns at which `"auto"` goes sparse; `None`
            (the default) is `foambryo.junction_graph.SPARSE_SOLVE_MIN_UNKNOWNS` (2 000).
            Keyword-only.
        diagnostics (bool | str, optional): `False` (the default), `True` or `"spectrum"`: also
            return a `foambryo.linear_solves.SolveDiagnostics`, last, with one record per solve.
            Keyword-only.

    Returns:
        tuple[NDArray[np.float64], dict[int, float], NDArray[np.float64] | int]:
            - with a Laplace method, one value per material in `mesh.materials` order (the
              exterior first, at `base_pressure`), `nan` where undetermined; with the variational
              method, the tensions used, in sorted interface order (`nan` where unusable on a
              cropped mesh);
            - map cell label (0 = exterior, at `base_pressure`) -> pressure. With a Laplace
              method, a cell whose pressure no equation fixes is `nan` in this map and in the
              array (and `UndeterminedPressureWarning` is emitted); the variational method returns
              a number for every cell of an uncropped mesh;
            - with a Laplace method, the squared residual of the solved equations (a length-1
              array, or empty); with the variational method, `0`;
            - with `diagnostics` on, a fourth value: the `SolveDiagnostics`.

    Raises:
        UndeterminedTensionError: on an uncropped mesh, if any tension in `dict_tensions` is not a
            finite number -- for example the `nan` that `infer_tensions` reports for an interface
            with no triple line. Checked before any method runs; where every tension is finite
            the result is bitwise unchanged by the check. See `UndeterminedTensionError`.
        ValueError: on a cropped mesh with a Laplace method, which has no formulation for crop
            faces (`PressureComputationMethod.has_crop_face_formulation`); on an invalid
            `solver`, `sparse_threshold` or `diagnostics`; with the variational method, when
            `dict_tensions` does not hold one tension per interface of an uncropped mesh.
        KeyError: with a Laplace method, when `dict_tensions` misses an interface of the mesh.
        numpy.linalg.LinAlgError: if the variational system is singular.

    Warns:
        UndeterminedPressureWarning: when a cell's pressure is undetermined and reported `nan`.
        CutCellPressureWarning: on a cropped mesh, naming the cut cells reported `nan`.
        SparseSolveFallbackWarning: when `lsqr` did not converge on a large Laplace system.
        DenseSolveMemoryWarning: when `solver="dense"` is forced on a system too large for the
            memory available.
    """
    with linear_solves.solve_options(solver, sparse_threshold, diagnostics) as collected:
        result = _infer_pressures_and_residuals(mesh, dict_tensions, mode, base_pressure)
    return result if collected is None else (*result, collected)


def _infer_pressures_and_residuals(
    mesh: "DcelData",
    dict_tensions: dict[tuple[int, int], float],
    mode: PressureComputationMethod,
    base_pressure: float,
) -> tuple[NDArray[np.float64], dict[int, float], NDArray[np.float64] | Literal[0]]:
    """`infer_pressures_and_residuals` under the solver options already in force."""
    if getattr(mesh, "has_crop_faces", False):
        if not getattr(mode, "has_crop_face_formulation", False):
            message = (
                f"{mode} has no formulation for crop faces (the curvature of an interface that the "
                "crop cuts is read partly on the crop); use PressureComputationMethod.Variational"
            )
            raise ValueError(message)
        return _infer_pressures_variational_off_the_crop(mesh, dict_tensions, base_pressure)
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

    An interface whose curvature is not finite has no equation. The cells that the remaining
    equations do not connect to the exterior are undetermined: their pressure is `nan`, in the
    array and in the map, and `UndeterminedPressureWarning` names them. Every other pressure is
    exactly the least-squares solution of the remaining equations.

    The system has one unknown per material. The exterior's pin is applied exactly, by elimination,
    and the cells connected to the exterior are solved alone (`_solve_laplace`). Below the sparse
    threshold (`foambryo.linear_solves`; 2 000 by default) by `scipy.linalg.lstsq`; at or above it,
    assembled sparse and solved by `lsqr` with the tension solve's settings, its columns scaled to
    unit norm (dense fallback with `SparseSolveFallbackWarning` if `lsqr` does not converge). The two
    paths solve the same full-rank problem. The residuals are those of that reduced system.

    Returns:
        tuple[NDArray[np.float64], dict[int, float], NDArray[np.float64]]:
            - array of pressures, one per material (the exterior first, at `base_pressure`), `nan`
              where undetermined
            - map cell id (0 = exterior) -> pressure, `nan` where undetermined
            - residuals of the least square method
    """
    dict_areas = mesh.compute_areas_interfaces()
    dict_curvature = mesh.compute_curvatures_interfaces(weighted=weighted)

    system = f"{'WeightedLaplace' if weighted else 'Laplace'} pressure system"
    as_sparse = linear_solves.use_sparse_solver(mesh.n_materials)
    if not as_sparse:
        linear_solves.warn_if_forced_dense_exceeds_memory(
            len(dict_curvature) + 1, mesh.n_materials, system, stacklevel=5,
        )
    matrix_m, vector_b = _build_matrix_laplace(
        dict_curvature,
        dict_areas,
        dict_tensions,
        mesh.n_materials,
        mesh.materials,
        base_pressure,
        as_sparse=as_sparse,
    )
    undetermined = _laplace_undetermined_cells(dict_curvature, mesh.materials)
    relative_pressures, residuals = _solve_laplace(
        matrix_m, vector_b, mesh.materials, undetermined, base_pressure, system, as_sparse=as_sparse,
    )

    # No equation fixes a disconnected cell's pressure: report it as nan.
    if undetermined:
        index = {int(m): i for i, m in enumerate(mesh.materials)}
        relative_pressures = relative_pressures.copy()
        relative_pressures[[index[c] for c in undetermined]] = np.nan
        warn_on_undetermined_pressures(undetermined)

    # The solve pinned the exterior at `base_pressure`, so these pressures already include it (until
    # 0.7.0 it was added a second time here).
    dict_pressures = {0: base_pressure}
    for i, key in enumerate(mesh.materials[1:]):
        dict_pressures[key] = relative_pressures[i + 1]

    return (relative_pressures, dict_pressures, residuals)


def _solve_laplace(
    matrix_m: NDArray[np.float64] | csr_matrix,
    vector_b: NDArray[np.float64],
    materials: NDArray[np.int64],
    undetermined: tuple[int, ...],
    base_pressure: float,
    system: str,
    *,
    as_sparse: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The Laplace solve, dense or sparse, on the cells connected to the exterior, the pin exact.

    The builder's last row pins the exterior, `x_0 = base_pressure`. It is applied exactly by
    elimination: column 0 leaves the system and `M[:, 0] * base_pressure` moves to the right-hand
    side (until 0.6.0 the pin was a least-squares row; the Laplace rows are differences, so it was
    met exactly anyway, but it set the condition number: 1.9e5 against 604 eliminated on a
    1 866-cell tissue). The columns of undetermined cells, and the rows that touch them, are left
    out too: they decouple from the exterior's block, and the caller reports them `nan`. What
    remains has full column rank (a connected graph's incidence rows), so the dense `lstsq` and the
    sparse `lsqr` (columns scaled to unit norm first) solve the same well-posed problem; on the
    whole singular system, dense `lstsq` used to invert a zero singular value it computed as a few
    `eps` and move the determined pressures by up to 9e-5 of their scale.

    Returns:
        tuple: one value per material (`base_pressure` for the exterior, `0` on the left-out columns,
        overwritten by the caller), and the squared residual of the reduced system as a length-1
        array.
    """
    index = {int(m): i for i, m in enumerate(materials)}
    keep_column = np.ones(len(materials), dtype=bool)
    keep_column[0] = False
    keep_column[[index[c] for c in undetermined]] = False
    dropped_cells = np.flatnonzero(~keep_column[1:]) + 1
    n_rows = matrix_m.shape[0] - 1  # without the pin row
    if as_sparse:
        csr = matrix_m.tocsr()[:n_rows]
        touches = (np.asarray(csr[:, dropped_cells].getnnz(axis=1)).ravel() > 0) if len(dropped_cells) else (
            np.zeros(n_rows, dtype=bool)
        )
        rows = np.flatnonzero(~touches)
        exterior = np.asarray(csr[rows][:, [0]].toarray()).ravel()
        reduced = csr[rows][:, np.flatnonzero(keep_column)]
    else:
        physics = matrix_m[:n_rows]
        touches = np.any(physics[:, dropped_cells] != 0, axis=1) if len(dropped_cells) else np.zeros(n_rows, bool)
        rows = np.flatnonzero(~touches)
        exterior = physics[rows, 0]
        reduced = physics[np.ix_(rows, np.flatnonzero(keep_column))]
    rhs = vector_b[rows] - exterior * base_pressure
    relative_pressures = np.zeros(len(materials))
    relative_pressures[0] = base_pressure
    if not keep_column.any():
        return relative_pressures, np.array([float(rhs @ rhs)])
    solved, residuals, _record = _solve_least_squares(
        reduced, rhs, system=system, use_sparse=as_sparse, scale_columns=as_sparse,
    )
    relative_pressures[keep_column] = solved
    return relative_pressures, np.atleast_1d(residuals) if np.size(residuals) else np.zeros(0)


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

    The pressures are the variational method's `inv(G_p) G_g x`, with `x` the tensions in sorted
    interface order: the cell rows of the row-normalised Gram system of `_build_matrix_discrete_sparse`.
    The row normalisation cancels in that product, so they are exactly `(V V^T)^-1 V D^T x`, `D` the
    interfaces' area gradients and `V` the cells' volume gradients: the least-squares solution of
    the vertex force balance `V^T p = D^T x`. Below the sparse threshold (`foambryo.linear_solves`,
    counted as interfaces plus cells, the size of the Gram system; 2 000 by default) the dense path
    runs exactly as it always did. At or above it, the `cells x cells` sparse symmetric system
    `V V^T p = V D^T x` is factorised by SuperLU (`scipy.sparse.linalg.splu`), a direct solve of the
    same matrix the dense path inverts, without forming any `interfaces x interfaces` matrix
    (126 GB at 10 000 cells).

    Returns:
        tuple[NDArray[np.float64], dict[int, float], NDArray[np.float64]]:
            - the tensions used, in sorted interface order
            - map cell id (0 = exterior) -> pressure
            - 0 (no residuals)

    Raises:
        numpy.linalg.LinAlgError: if `V V^T` is singular (sparse path; the dense path's `inv` raises
            the same on an exactly singular matrix).
    """
    n_interfaces = len(dict_tensions)
    n_cells = mesh.n_materials - 1
    system = "Variational pressure system"
    if linear_solves.use_sparse_solver(n_interfaces + n_cells):
        return _infer_pressures_variational_sparse(mesh, dict_tensions, base_pressure, system)
    linear_solves.warn_if_forced_dense_exceeds_memory(
        n_interfaces + n_cells + 1, n_interfaces + n_cells, system, copies=4, stacklevel=5,
    )
    start = time.perf_counter()
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

    if linear_solves.collecting_diagnostics():
        _record_variational_dense(g_p, g_g @ x, relative_pressures, system, start)

    for i, key in enumerate(mesh.materials[1:]):
        dict_pressures[key] = relative_pressures[i] + base_pressure

    return (x, dict_pressures, 0)


def _record_variational_dense(
    g_p: NDArray[np.float64],
    rhs: NDArray[np.float64],
    relative_pressures: NDArray[np.float64],
    system: str,
    start: float,
) -> None:
    """Diagnostics of the dense variational pressure solve (an SVD of `G_p`, only when asked)."""
    singular_values = np.linalg.svd(g_p, compute_uv=False)
    difference = rhs - g_p @ relative_pressures
    tolerance = max(g_p.shape) * np.finfo(float).eps * (singular_values[0] if len(singular_values) else 0.0)
    linear_solves.record_solve(linear_solves.SolveRecord(
        system=system,
        solver="dense inv",
        n_rows=int(g_p.shape[0]),
        n_unknowns=int(g_p.shape[1]),
        rank=int(np.sum(singular_values > tolerance)),
        residual_norm=float(np.sqrt(difference @ difference)),
        condition_number=linear_solves.condition_from_singular_values(singular_values),
        condition_number_source="SVD of the cells x cells matrix (2-norm; computed for diagnostics only)",
        seconds=time.perf_counter() - start,
        singular_values=singular_values,
    ))


def _infer_pressures_variational_sparse(
    mesh: "DcelData",
    dict_tensions: dict[tuple[int, int], float],
    base_pressure: float,
    system: str,
) -> tuple[NDArray[np.float64], dict[int, float], Literal[0]]:
    """The variational pressures by a sparse direct solve of `V V^T p = V D^T x` (cells x cells).

    See `_infer_pressures_variational`. Returns the same three values.
    """
    start = time.perf_counter()
    keys, matrix_d = compute_area_derivative_sparse(mesh)
    _materials, matrix_v = compute_volume_derivative_sparse(mesh)
    cells_v = matrix_v[1:]
    x = np.array([dict_tensions[key] for key in sorted(dict_tensions.keys())], dtype=np.float64)
    if len(x) != len(keys):
        message = f"{len(dict_tensions)} tensions for the mesh's {len(keys)} interfaces"
        raise ValueError(message)
    normal = (cells_v @ cells_v.T).tocsc()
    rhs = cells_v @ (matrix_d.T @ x)
    try:
        factor = splu(normal)
    except RuntimeError as error:  # SuperLU: "Factor is exactly singular"
        message = f"the variational pressure system V V^T is singular ({error})"
        raise np.linalg.LinAlgError(message) from error
    relative_pressures = factor.solve(rhs)
    if not np.all(np.isfinite(relative_pressures)):
        message = "the variational pressure system V V^T is singular: its solution is not finite"
        raise np.linalg.LinAlgError(message)
    if linear_solves.collecting_diagnostics():
        _record_variational_sparse(normal, factor, rhs, relative_pressures, system, start)
    dict_pressures = {0: base_pressure}
    for i, key in enumerate(mesh.materials[1:]):
        dict_pressures[key] = relative_pressures[i] + base_pressure
    return (x, dict_pressures, 0)


def _record_variational_sparse(
    normal: csc_matrix,
    factor: "SuperLU",
    rhs: NDArray[np.float64],
    relative_pressures: NDArray[np.float64],
    system: str,
    start: float,
) -> None:
    """Diagnostics of the sparse variational pressure solve: 1-norm condition estimate, spectrum if asked."""
    inverse = LinearOperator(normal.shape, matvec=factor.solve, rmatvec=lambda v: factor.solve(v, trans="T"))
    condition = float(onenormest(normal) * onenormest(inverse))
    difference = rhs - normal @ relative_pressures
    singular_values = spectrum_seconds = spectrum_note = None
    if linear_solves.spectrum_requested():
        singular_values, spectrum_seconds, spectrum_note = linear_solves.extreme_singular_values(
            normal, symmetric_positive=True,
        )
    linear_solves.record_solve(linear_solves.SolveRecord(
        system=system,
        solver="sparse direct (SuperLU)",
        n_rows=int(normal.shape[0]),
        n_unknowns=int(normal.shape[1]),
        rank=None,
        residual_norm=float(np.sqrt(difference @ difference)),
        condition_number=condition,
        condition_number_source="1-norm estimate from the factorisation (Hager-Higham onenormest)",
        seconds=time.perf_counter() - start,
        spectrum_seconds=spectrum_seconds,
        spectrum_note=spectrum_note,
        singular_values=singular_values,
    ))


def _infer_pressures_variational_off_the_crop(
    mesh: "DcelData",
    dict_tensions: Mapping[tuple[int, int], float],
    base_pressure: float = 0,
) -> tuple[NDArray[np.float64], dict[int, float], Literal[0]]:
    """The variational pressures of a cropped mesh: cut cells solved for and withheld.

    The variational method's pressures are the least-squares solution of the force balance
    `V^T p = D^T gamma` at the vertices, with the tensions given (on an uncropped mesh the
    Schur-complement solve `_infer_pressures_variational` computes the same solution). Here the
    rows of two kinds of vertex are left out: those on a crop face, held by tissue that was cut
    away, and those of the interfaces without a finite tension (nothing is known of their pull).
    Every cell keeps its column, the cut cells included, so that their force is not moved onto
    their neighbours. A cell the remaining rows do not determine (its column has a component in
    the null space of the system, beyond `1e-8`) is `nan`; the cut cells are `nan` whatever the
    solve gives them. Tensions are used as given, all on one scale: a mesh whose tensions come in
    separately gauged components (`JunctionGraphGaugeWarning`) mixes those scales here as on an
    uncropped mesh.

    Args:
        mesh (DcelData): a mesh with crop faces.
        dict_tensions (Mapping[tuple[int, int], float]): interface -> tension; crop interfaces may
            be absent, undetermined interfaces `nan`.
        base_pressure (float, optional): the exterior's pressure. Defaults to 0.

    Returns:
        tuple: the tensions used (in sorted interface order, `nan` where unusable), the pressure
        map (exterior first, `nan` for every cut or undetermined cell) and 0, as
        `_infer_pressures_variational` returns.
    """
    keys, matrix_d = compute_area_derivative_sparse(mesh)
    materials, matrix_v = compute_volume_derivative_sparse(mesh)
    usable = {
        key
        for key in (tuple(int(x) for x in k) for k in keys)
        if key in dict_tensions and np.isfinite(dict_tensions[key])
    }
    unusable_faces = [i for i, (a, b) in enumerate(mesh.f[:, 3:5]) if (int(a), int(b)) not in usable]
    rows = vertex_rows_off_the_crop(mesh, np.unique(mesh.f[unusable_faces, :3]))
    keys_as_tuples = [tuple(int(x) for x in key) for key in keys]
    gamma = np.array([dict_tensions[key] if key in usable else 0.0 for key in keys_as_tuples])
    rhs = matrix_d.T.tocsr()[rows] @ gamma
    cell_index = [j for j, m in enumerate(materials) if m != 0]
    sparse_system = matrix_v.T.tocsr()[rows][:, cell_index]
    solved, null = _least_squares_with_null_space(sparse_system, rhs)
    cut = set(mesh.cut_cells)
    dict_pressures: dict[int, float] = {0: base_pressure}
    undetermined = []
    for local, j in enumerate(cell_index):
        cell = int(materials[j])
        if cell in cut:
            dict_pressures[cell] = float("nan")
        elif null.shape[0] > 0 and np.any(np.abs(null[:, local]) > 1e-8):
            dict_pressures[cell] = float("nan")
            undetermined.append(cell)
        else:
            dict_pressures[cell] = float(solved[local]) + base_pressure
    warn_on_cut_cell_pressures(tuple(sorted(cut)), stacklevel=5)
    _warn_on_cells_without_balance(tuple(undetermined))
    used = np.array([dict_tensions.get(key, np.nan) for key in keys_as_tuples], dtype=np.float64)
    return (used, dict_pressures, 0)


#: Above this many entries (rows x cells) the cropped variational pressure system is solved through
#: its normal equations, sparse, instead of densely: a 1 866-cell tissue has some 350 000
#: rows, and the dense system alone would take 5 GB.
DENSE_CROP_PRESSURE_MAX_ENTRIES = 20_000_000

#: Relative eigenvalue below which a direction of the normal matrix `A^T A` counts as null in the
#: large-system path. The normal matrix squares the conditioning, so the null eigenvalues come out at
#: rounding level, about `1e-13` times the largest; real pressure directions of these systems sit far
#: above `1e-10` of it.
NORMAL_EQUATIONS_NULL_TOLERANCE = 1e-10


def _least_squares_with_null_space(
    system: "csr_matrix",
    rhs: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Minimum-norm least squares `min ||A p - rhs||`, and an orthonormal basis of `null(A)` (rows).

    Under `solver="auto"` (`foambryo.linear_solves`), small systems (at most
    `DENSE_CROP_PRESSURE_MAX_ENTRIES` entries; this rule, not `sparse_threshold`, decides here) are
    solved densely: the minimum-norm solution of a reduced SVD truncated at the relative tolerance
    `max(shape) * eps`, the null space read from the same SVD. Larger ones through the normal
    equations `A^T A p = A^T rhs`, assembled sparse (`A^T A` is cells x cells), solved by a symmetric eigendecomposition with eigenvalues
    below `NORMAL_EQUATIONS_NULL_TOLERANCE` times the largest treated as null. Measured on 903
    plane, corner and oblique crops of the benchmark tissues, both paths forced: the same cells
    reported on every one, and pressures within 1.4e-12 of their mean magnitude. `solver="dense"`
    and `"sparse"` force the first and the second path.

    Args:
        system: `A`, sparse, rows = kept vertex coordinates, columns = cells.
        rhs: the right-hand side, one entry per row.

    Returns:
        tuple: the solution (one entry per column) and the null-space basis, shape `(k, n_columns)`.
    """
    n_rows, n_cols = system.shape
    start = time.perf_counter()
    label = "Variational pressure system, cropped mesh"
    solver = linear_solves.current_solve_options().solver
    dense_path = n_rows * n_cols <= DENSE_CROP_PRESSURE_MAX_ENTRIES if solver == "auto" else solver == "dense"
    if dense_path:
        linear_solves.warn_if_forced_dense_exceeds_memory(n_rows, n_cols, label, copies=3, stacklevel=6)
        dense = system.toarray()
        # A reduced SVD gives every right singular vector when there are at least as many rows as
        # columns; the full one is needed only in the opposite case.
        u, singular, vt = np.linalg.svd(dense, full_matrices=n_rows < n_cols)
        tolerance = max(dense.shape) * np.finfo(float).eps * (singular[0] if len(singular) else 0.0)
        rank = int(np.sum(singular > tolerance))
        # The minimum-norm solution from the same SVD and the same tolerance that define the null
        # space. Until 0.6.0 `lstsq` solved it with its own cutoff (`eps`), kept a zero singular value
        # computed as ~1e-14 of the largest, and moved the determined pressures by up to 1.7e-4 of
        # their scale (a corner crop of a converged foam).
        solved = vt[:rank].T @ ((u[:, :rank].T @ rhs) / singular[:rank])
        if linear_solves.collecting_diagnostics():
            difference = rhs - dense @ solved
            linear_solves.record_solve(linear_solves.SolveRecord(
                system=label, solver="dense lstsq", n_rows=n_rows, n_unknowns=n_cols, rank=rank,
                residual_norm=float(np.sqrt(difference @ difference)),
                condition_number=linear_solves.condition_from_singular_values(singular),
                condition_number_source="singular values of the dense system (2-norm)",
                seconds=time.perf_counter() - start, singular_values=singular.copy(),
            ))
        return solved, vt[rank:]
    normal = (system.T @ system).toarray()
    projected = system.T @ rhs
    eigenvalues, eigenvectors = np.linalg.eigh(normal)
    keep = eigenvalues > NORMAL_EQUATIONS_NULL_TOLERANCE * max(float(eigenvalues[-1]), 0.0)
    basis = eigenvectors[:, keep]
    solved = basis @ ((basis.T @ projected) / eigenvalues[keep])
    if linear_solves.collecting_diagnostics():
        singular = np.sqrt(np.clip(eigenvalues[::-1], 0.0, None))
        difference = rhs - system @ solved
        kept = singular[: int(keep.sum())]
        linear_solves.record_solve(linear_solves.SolveRecord(
            system=label, solver="sparse normal equations + dense eigh", n_rows=n_rows, n_unknowns=n_cols,
            rank=int(keep.sum()), residual_norm=float(np.sqrt(difference @ difference)),
            condition_number=(float(kept[0] / kept[-1]) if len(kept) else None),
            condition_number_source="square roots of the normal matrix's eigenvalues, over the numerical rank",
            seconds=time.perf_counter() - start, singular_values=singular,
        ))
    return solved, eigenvectors[:, ~keep].T


def _warn_on_cells_without_balance(cells: tuple[int, ...]) -> None:
    """`UndeterminedPressureWarning` for the cells a cropped mesh's force balances leave undetermined."""
    if not cells:
        return
    shown = ", ".join(str(c) for c in cells[:20])
    message = (
        f"the pressure of {len(cells)} cell(s) that the crop does not cut is undetermined and reported "
        f"as nan: {shown}. Every vertex of these cells lies on a crop face or on an interface whose "
        "tension is undetermined, so no force balance fixes their pressure."
    )
    warnings.warn(message, UndeterminedPressureWarning, stacklevel=5)


def _build_matrix_laplace(
    dict_curvature: dict[tuple[int, int], float],
    dict_areas: dict[tuple[int, int], float],
    dict_tensions: dict[tuple[int, int], float],
    n_materials: int,
    materials: NDArray[np.int64],
    base_pressure: float = 0,
    *,
    as_sparse: bool = False,
) -> tuple[NDArray[np.float64] | csr_matrix, NDArray[np.float64]]:
    """Get matrix M and vector B of system MX=B to solve to get relative pressures.

    The entries are written as triplets, then into a dense matrix by assignment (bitwise the
    historical matrix) or, with `as_sparse`, into a CSR matrix holding the same values, without ever
    allocating the dense `(n_interfaces + 1) x n_materials` array.

    Args:
        dict_curvature(dict[tuple[int, int], float]): Mean curvature of interfaces.
        dict_areas(dict[tuple[int, int], float]): Area of interfaces.
        dict_tensions(dict[tuple[int, int], float]): Tensions of interfaces.
        n_materials(int): Number of cells + 1.
        materials(NDArray[np.float64]): label of cells.
        base_pressure(float): Base pressure of exterior region.
        as_sparse(bool): return M as a `scipy.sparse.csr_matrix`. Keyword-only. Defaults to False.

    Returns:
        tuple[NDArray[np.float64] | csr_matrix, NDArray[np.float64]]:
        - M, `(n_interfaces + 1, n_materials)`, the exterior's pin row last
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
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []

    # CLASSICAL
    # LAPLACE
    for i, key in enumerate(keys_t):
        a = key // kt
        b = key % kt
        curvature_k = dict_curvature[(a, b)]
        tension_k = dict_tensions[(a, b)]

        if not np.isfinite(curvature_k):  # no measurable curvature: no equation
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
        rows += [i, i]
        cols += [am, bm]
        values += [1 * factor, -1 * factor]  # Pi, Pj

    if as_sparse:
        matrix_sparse = coo_matrix(
            (np.array([*values, 1.0]), (np.array([*rows, nm], dtype=np.int64), np.array([*cols, 0], dtype=np.int64))),
            shape=(nm + 1, nc + 1),
        ).tocsr()
        return (matrix_sparse, vector_b)

    matrix_m = np.zeros((nm + 1, nc + 1))
    if rows:
        matrix_m[np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)] = np.array(values, dtype=np.float64)
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
