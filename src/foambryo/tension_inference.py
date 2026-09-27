"""Module to infer tensions on a mesh.

Sacha Ichbiah 2021
Matthieu Perez 2024
"""
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from foambryo.force_balance_inference import ForceBalanceNormalisation, infer_tensions_force_balance
from foambryo.geometry import compute_area_derivative_sparse, compute_volume_derivative_sparse
from foambryo.junction_graph import (
    ComponentTensionSolution,
    solve_gauged_least_squares,
    warn_on_junction_graph_fragmentation,
)
from foambryo.per_edge_tension_inference import infer_tensions_young_dupre_local
from foambryo.second_order_angle_reading import (
    CHORD_READING,
    AngleReadingReport,
    rotations_for_reading,
    validate_reading,
    warn_on_partial_coverage,
)
from foambryo.wedge_refusal import (
    NO_REFUSAL,
    WedgeRefusal,
    all_interfaces_undetermined,
    line_averaged_angles,
    require_refusal_is_honoured,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from foambryo.dcel import DcelData
    from foambryo.per_edge_tension_inference import YoungDupreLocalResult


class TensionComputationMethod(Enum):
    """Describe the formula used to infer the tensions from contact angles at each junction.

    It can be:
    - YoungDupre (Young-Dupré with cosines only),
    - YoungDupreLocal (the same balance imposed **per trijunctional edge** instead of once per
      triple line on angles averaged along it — see `foambryo.per_edge_tension_inference`),
    - ProjectionYoungDupre (Young-Dupré with cosines and sines),
    - Equilibrium,
    - Cotan (cotangent formula, see [Yamamoto et al. 2023](https://doi.org/10.1101/2023.03.07.531437)),
    - InvCotan (inverse of the cotangent formula),
    - Lami ([Lami's theorem](https://en.wikipedia.org/wiki/Lami%27s_theorem)),
    - InvLami (inverse of the Lami's relation),
    - LogLami (logarithm of the Lami's relation),
    - Variational (variational formulation, see our [paper](https://doi.org/10.1101/2023.04.12.536641)),
    - ForceBalance (the unified discrete force-balance operator with the mean-tension gauge
      imposed exactly, through a KKT system),
    - ForceBalanceSoftRow (the same operator, with the gauge as one more least-squares row —
      the convention all eight methods above use),
    - ForceBalanceMAP (the same operator, with a Gaussian prior; the gauge is anchored only
      softly, so its mean tension is deliberately not exactly `mean_tension`).

    The three `ForceBalance*` members are **one estimator under three gauge treatments**, not
    three physical models; they exist as separate members so that the comparison between the
    three gauge treatments is a regenerable benchmark row rather than a one-off table. See
    `foambryo.force_balance_inference`.

    **Which methods read angles and which invert an operator, and why the distinction now has
    measured consequences.** `YoungDupre`, `YoungDupreLocal`, `ProjectionYoungDupre`, `Lami`,
    `InvLami` and `LogLami` are all built from *measured contact angles* at triple junctions —
    they read the mesh's local geometry pointwise and never assemble or invert a global
    operator. `Cotan`, `InvCotan`, `Variational` and the three `ForceBalance*` members instead
    assemble a discrete operator over the whole mesh (an area/volume-gradient system, or its
    force-balance generalisation) and solve or invert it. `Equilibrium` is a geometric-edge
    balance and does not cleanly fall in either family (see its own note in
    `foambryo.validation.force_balance`). This split matters because mesh-reconstruction
    quality affects the two families very differently: on `dw3d`-reconstructed meshes, the
    operator-inverting family's error tracks the mesh's distance from mechanical equilibrium
    (`rho_free`) far more strongly than the angle-reading family's does, so an operator-
    inverting method can be dramatically more or less accurate depending on which meshing
    pipeline produced its input, while an angle-reading method is comparatively insensitive to
    it. See `dw3d`'s `BENCHMARKS.md` and this package's own `BENCHMARKS.md` for the numbers.

    **The operator-inverting family is unreliable on reconstructed meshes, measured.** On the
    benchmark cohort's shipped-default reconstructions `rho_free` -- the distance of the mesh
    from any equilibrium of the force-balance operator -- has median 0.578 against the
    reference meshes' 0.0118 (a factor of about 50), and all 40 equilibrium cases exceed the
    0.2 threshold at which `foambryo.uncertainty` reports that no surface tensions balance the
    mesh at all. Every `ForceBalance*` number obtained on a `dw3d` mesh, including every
    published comparison between reconstruction variants through this family, should be read
    with that floor in mind. See `foambryo.force_balance_inference`'s module docstring.
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
    ForceBalance = 9
    ForceBalanceSoftRow = 10
    ForceBalanceMAP = 11
    YoungDupreLocal = 12


class VariationalRoute(Enum):
    """Which linear algebra `TensionComputationMethod.Variational` uses to reach its answer.

    Both routes minimise the *same* discrete energy gradient; they differ in how, and the
    sparse-assembly work deferred the choice to the uncertainty work because it moves published
    numbers.

    - `Gram` (**route P**, the original): form the Gram normal equations `M^T M`, divide every
      row by **its own row sum**, Schur-eliminate the pressures, then least-squares with the
      mean-tension gauge as a soft extra row.
    - `Operator` (**route Q**): minimise `||M beta||^2` directly through `M`, with the gauge
      eliminated by substitution. `M^T M` is never formed and there is no row normalisation.

    **The row normalisation is the substantive difference, and it has never been justified.**
    Dividing normal-equation row `i` by `sum_j (M^T M)_ij` is not a symmetric preconditioner
    (it destroys the symmetry of the Gram matrix), it is not scale-invariant in any documented
    sense, and it has no derivation in the paper, the Supplementary Note or the
    *technical note on tension inference (H. Turlier, in preparation)*. Measured on
    `benchmarking-dataset`: the divisors span
    a factor **57-140 within a single mesh and change sign**, so it is a genuine, arbitrary
    reweighting of the physics rows — not a numerical nicety. That is why the two routes give
    different answers at all (~1.4e-4 per tension).

    **The uncertainty work's decision, on the evidence: `Operator` is the default.** See
    `VARIATIONAL_ROUTE_DEFAULT` and `foambryo_benchmarks/run_variational_route.py`. `Gram`
    stays reachable so every published number
    remains reproducible.
    """

    Gram = 0
    Operator = 1


#: Route used by `TensionComputationMethod.Variational` when no route is given.
#:
#: **The uncertainty work's decision (2026-07-31), taken on measured evidence, flipping the
#: sparse-assembly work's retained default.** Over the 42 equilibrium cases, gauge-matched
#: median relative tension error is
#: `Gram` 2.0800e-02 vs `Operator` 2.0793e-02 — a 0.03 % difference, `Operator` better on 30
#: of 42 cases, and the per-tension shift (median 1.1e-04) sits below the ground truth's own
#: worst-case quantisation floor (6.7e-04, the 4-decimal rounding of the smallest stored
#: tension). So **accuracy is neutral**, and the tie is broken by the two things that are not
#: neutral: `Operator` never forms `M^T M` (`cond` 24 against 7e5, a ~30 000x reduction on the
#: matrix actually inverted) and it drops an **undocumented ad-hoc row reweighting** that
#: spans a factor 57-140 and changes sign within one mesh. Route `Gram` remains available and
#: its numbers are pinned by a test, so nothing published becomes irreproducible.
VARIATIONAL_ROUTE_DEFAULT = VariationalRoute.Operator


#: The `ForceBalance*` members and the gauge treatment each one selects. Kept next to the
#: enum so a new normalisation cannot be added to one without the other.
_FORCE_BALANCE_NORMALISATIONS = {
    TensionComputationMethod.ForceBalance: ForceBalanceNormalisation.Kkt,
    TensionComputationMethod.ForceBalanceSoftRow: ForceBalanceNormalisation.SoftRow,
    TensionComputationMethod.ForceBalanceMAP: ForceBalanceNormalisation.BayesianMap,
}


# FORCE INFERENCE ONLY TAKING INTO ACCOUNT TRIJUNCTIONS :
# THERE ARE A LOT OF QUADRIJUNCTIONS IN THE MESH
# HOWEVER, TAKING ONLY THE TRIJUNCTIONS INTO ACCOUNT SHOULD SUFFICE IN MOST CASES, ESPECIALLY WITH A FEW CELLS
# BECAUSE :
# THE PROBLEM IS HIGHLY OVERDETERMINED
# AND THE QUADRIJUNCTIONS ARE NOT VERY PRESENT (THEY OCCUPY A LITTLE LENGTH)
#: Every method whose rows are built from measured trijunction angles, and which therefore
#: responds to `angle_reading`. Kept as an explicit registry, like
#: `_FORCE_BALANCE_NORMALISATIONS` and `_JUNCTION_GRAPH_GAUGED_BUILDERS`, so a newly-added
#: member cannot silently accept an `angle_reading` it does not honour.
#:
#: The six the second-order-angle-reading work's gate measures are the angle-reading *family* named in
#: `TensionComputationMethod`'s docstring — `YoungDupre`, `YoungDupreLocal`,
#: `ProjectionYoungDupre`, `Lami`, `InvLami`, `LogLami`. `Cotan` and `InvCotan` are classed
#: there as operator-inverting because they also assemble an area/volume system, but their
#: rows *do* read `compute_angles_tri`, so the selector reaches them too and that work reports
#: them separately rather than pretending it does not.
ANGLE_READING_METHODS = frozenset({
    TensionComputationMethod.YoungDupre,
    TensionComputationMethod.YoungDupreLocal,
    TensionComputationMethod.ProjectionYoungDupre,
    TensionComputationMethod.Lami,
    TensionComputationMethod.InvLami,
    TensionComputationMethod.LogLami,
    TensionComputationMethod.Cotan,
    TensionComputationMethod.InvCotan,
})


def _require_reading_is_honoured(
    mode: TensionComputationMethod,
    angle_reading: str,
    refusal: WedgeRefusal = NO_REFUSAL,
) -> None:
    """Refuse a non-default `angle_reading` or refusal policy on a method that reads no angle.

    Silently ignoring either would hand the caller a result they believe is corrected, or is
    refusing its infeasible observations, and is not — exactly the failure mode the
    second-order-angle-reading work exists to remove, and the one `foambryo.wedge_refusal`
    inherits.

    Args:
        mode (TensionComputationMethod): the requested method.
        angle_reading (str): the requested reading.
        refusal (WedgeRefusal, optional): the requested refusal policy. Defaults to refusing
            nothing, which every method accepts.

    Raises:
        ValueError: if `angle_reading` is not the default, or `refusal` is not `NO_REFUSAL`'s
            policy, and `mode` is not in `ANGLE_READING_METHODS`.
    """
    validate_reading(angle_reading)
    if mode not in ANGLE_READING_METHODS:
        require_refusal_is_honoured(mode.name, refusal)
    if angle_reading != CHORD_READING and mode not in ANGLE_READING_METHODS:
        message = (
            f"{mode} does not build its rows from measured trijunction angles, so "
            f"angle_reading={angle_reading!r} would have no effect on it. Refusing rather "
            f"than returning a result that looks corrected and is not. The methods that "
            f"honour it are {sorted(m.name for m in ANGLE_READING_METHODS)}."
        )
        raise ValueError(message)


def infer_tensions(
    mesh: "DcelData",
    mean_tension: float = 1,
    mode: TensionComputationMethod = TensionComputationMethod.YoungDupre,
    *,
    angle_reading: str = CHORD_READING,
    refusal: WedgeRefusal = NO_REFUSAL,
) -> dict[tuple[int, int], float]:
    """Infer tensions using chosen method.

    Args:
        mesh (DcelData): Mesh to analyze.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.
        mode (TensionComputationMethod, optional): method to infer tensions.
        angle_reading (str, optional): which trijunction angle convention the angle-reading
            methods consume — `"chord"` (the default, bit-identical to every published number)
            or `"quadric"`, the second-order reading of
            `foambryo.second_order_angle_reading`. Keyword-only. A `"quadric"` reading that
            could not cover every trijunction face falls back to the chord there and emits an
            `AngleReadingCoverageWarning`; use
            :func:`infer_tensions_and_residuals_with_report` to get the coverage numbers
            instead of the warning.
        refusal (WedgeRefusal, optional): which trijunction observations the angle-reading
            methods decline to use, and how -- see `foambryo.wedge_refusal`. Keyword-only;
            defaults to refusing nothing, which is **bitwise** the historical behaviour. A
            non-default policy on a method that reads no wedge angle raises rather than being
            ignored. Refusing rows can leave an interface, or a whole mesh, with no surviving
            junction line: those interfaces are reported `float("nan")`, never a number, and how
            often a policy does that is a number to report beside any accuracy it buys.

    Returns:
        dict[tuple[int, int], float]:
            - map interface id (label 1, label 2) -> tension on this interface. An
              interface with no triple line at all (a fully closed surface -- an engulfed
              cell, a sealed lumen) is undetermined and reported as `float("nan")`, never a
              number. If the mesh's triple-junction graph is fragmented into more than one
              disconnected piece, or has any undetermined interface, this flat return value
              blends per-component gauges that are not on the same scale (a
              `JunctionGraphGaugeWarning` is emitted in that case); use
              :func:`infer_tensions_by_junction_component` instead to get each interface's
              component and per-component gauge explicitly, and never average or compare
              tensions across components from this function's own return value.
    """
    _, dict_tensions, _ = infer_tensions_and_residuals(
        mesh, mean_tension, mode, angle_reading=angle_reading, refusal=refusal,
    )
    return dict_tensions


def infer_tensions_and_residuals(
    mesh: "DcelData",
    mean_tension: float = 1,
    mode: TensionComputationMethod = TensionComputationMethod.YoungDupre,
    *,
    angle_reading: str = CHORD_READING,
    refusal: WedgeRefusal = NO_REFUSAL,
) -> tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
    """Infer tensions using chosen method. Get residuals too.

    Args:
        mesh (DcelData): Mesh to analyze.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.
        mode (TensionComputationMethod, optional): method to infer tensions.
        angle_reading (str, optional): one of
            `foambryo.second_order_angle_reading.ANGLE_READINGS`; `"chord"` is the default.
            Keyword-only.
            See :func:`infer_tensions`.
        refusal (WedgeRefusal, optional): which trijunction observations the angle-reading
            methods decline to use, and how -- see `foambryo.wedge_refusal`. Keyword-only;
            defaults to refusing nothing, which is **bitwise** the historical behaviour. A
            non-default policy on a method that reads no wedge angle raises rather than being
            ignored. Refusing rows can leave an interface, or a whole mesh, with no surviving
            junction line: those interfaces are reported `float("nan")`, never a number, and how
            often a policy does that is a number to report beside any accuracy it buys.

    Returns:
        tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
            - array of relative tensions
            - map interface id (label 1, label 2) -> tension on this interface. See
              :func:`infer_tensions`'s return value for the undetermined-interface
              (`nan`) and fragmented-junction-graph (per-component gauge) caveats -- they
              apply here identically, and this function is where the corresponding
              `JunctionGraphGaugeWarning` is actually emitted.
            - residuals of the least square method.
    """
    tensions, dict_tensions, residuals, report, solution = _infer_tensions_and_residuals_with_report_and_solution(
        mesh, mean_tension, mode, angle_reading=angle_reading, refusal=refusal,
    )
    warn_on_partial_coverage(report)
    if solution is not None:
        # `solution` is the same `ComponentTensionSolution` (or `YoungDupreLocalResult`) the
        # builder above already computed the partition to produce -- reading
        # `fragmented`/`n_components`/`undetermined` off it, rather than calling
        # `partition_from_mesh` a second time, avoids a redundant per-call junction-graph
        # computation on every one of these 8 methods.
        warn_on_junction_graph_fragmentation(solution.fragmented, solution.n_components, solution.undetermined)
    return (tensions, dict_tensions, residuals)


def infer_tensions_and_residuals_with_report(
    mesh: "DcelData",
    mean_tension: float = 1,
    mode: TensionComputationMethod = TensionComputationMethod.YoungDupre,
    *,
    angle_reading: str = CHORD_READING,
    refusal: WedgeRefusal = NO_REFUSAL,
) -> tuple[
    NDArray[np.float64],
    dict[tuple[int, int], float],
    NDArray[np.float64],
    AngleReadingReport,
]:
    """:func:`infer_tensions_and_residuals`, plus which trijunction faces the reading covered.

    The reporting entry point, and the one to use for anything that will be published or
    benchmarked. It does **not** emit `AngleReadingCoverageWarning`: the caller is holding the
    numbers the warning would have described.

    Args:
        mesh (DcelData): Mesh to analyze.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.
        mode (TensionComputationMethod, optional): method to infer tensions.
        angle_reading (str, optional): one of
            `foambryo.second_order_angle_reading.ANGLE_READINGS`; `"chord"` is the default.
            Keyword-only.
        refusal (WedgeRefusal, optional): which trijunction observations the angle-reading
            methods decline to use, and how -- see `foambryo.wedge_refusal`. Keyword-only;
            defaults to refusing nothing, which is **bitwise** the historical behaviour. A
            non-default policy on a method that reads no wedge angle raises rather than being
            ignored. Refusing rows can leave an interface, or a whole mesh, with no surviving
            junction line: those interfaces are reported `float("nan")`, never a number, and how
            often a policy does that is a number to report beside any accuracy it buys.

    Returns:
        tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64],
            AngleReadingReport]: the three values :func:`infer_tensions_and_residuals` returns,
            plus the coverage report. For a method outside `ANGLE_READING_METHODS`, or for the
            chord reading, the report is the trivial complete-coverage one.

    Raises:
        ValueError: if `angle_reading` is not `"chord"` and `mode` does not read angles.
    """
    tensions, dict_tensions, residuals, report, _solution = _infer_tensions_and_residuals_with_report_and_solution(
        mesh, mean_tension, mode, angle_reading=angle_reading, refusal=refusal,
    )
    return (tensions, dict_tensions, residuals, report)


def _infer_tensions_and_residuals_with_report_and_solution(
    mesh: "DcelData",
    mean_tension: float = 1,
    mode: TensionComputationMethod = TensionComputationMethod.YoungDupre,
    *,
    angle_reading: str = CHORD_READING,
    refusal: WedgeRefusal = NO_REFUSAL,
) -> tuple[
    NDArray[np.float64],
    dict[tuple[int, int], float],
    NDArray[np.float64],
    AngleReadingReport,
    "ComponentTensionSolution | YoungDupreLocalResult | None",
]:
    """`infer_tensions_and_residuals_with_report`, plus the junction-graph-gauged solution.

    Internal. Exists so `infer_tensions_and_residuals` can read `fragmented`/`n_components`/
    `undetermined` off the solution the builder already computed for
    `JunctionGraphGaugeWarning`, instead of calling `partition_from_mesh` a second time --
    avoiding a redundant per-call junction-graph computation on every one of the 8 methods
    in `_JUNCTION_GRAPH_GAUGED_BUILDERS`. Threads the solution through rather than caching or
    memoising it: the warning must still fire in exactly the same cases, from the same data,
    on every call.

    Returns:
        The same 4-tuple `infer_tensions_and_residuals_with_report` returns, plus a fifth
        element: the solution `mode`'s builder computed when `mode` is a member of
        `_JUNCTION_GRAPH_GAUGED_BUILDERS`, else `None`.

    Raises:
        ValueError: if `angle_reading` is not `"chord"` and `mode` does not read angles.
    """
    _require_reading_is_honoured(mode, angle_reading, refusal)
    _rotations, report = rotations_for_reading(mesh, angle_reading if mode in ANGLE_READING_METHODS else CHORD_READING)

    solution: ComponentTensionSolution | YoungDupreLocalResult | None = None
    if mode == TensionComputationMethod.YoungDupreLocal:
        solution = infer_tensions_young_dupre_local(mesh, mean_tension, angle_reading=angle_reading, refusal=refusal)
        return (solution.tensions, solution.dict_tensions, solution.lstsq_residuals, report, solution)
    elif mode == TensionComputationMethod.ProjectionYoungDupre:
        solution = _infer_tension_projection_yd(mesh, mean_tension, angle_reading, refusal=refusal)
        legacy = _as_legacy_tuple(solution)
    elif mode == TensionComputationMethod.Equilibrium:
        legacy = _infer_tension_equilibrium(mesh, mean_tension)
    elif mode == TensionComputationMethod.Cotan:
        solution = _infer_tension_cotan(mesh, mean_tension, angle_reading, refusal=refusal)
        legacy = _as_legacy_tuple(solution)
    elif mode == TensionComputationMethod.InvCotan:
        solution = _infer_tension_inv_cotan(mesh, mean_tension, angle_reading, refusal=refusal)
        legacy = _as_legacy_tuple(solution)
    elif mode == TensionComputationMethod.Lami:
        solution = _infer_tension_lami(mesh, mean_tension, angle_reading, refusal=refusal)
        legacy = _as_legacy_tuple(solution)
    elif mode == TensionComputationMethod.InvLami:
        solution = _infer_tension_inv_lami(mesh, mean_tension, angle_reading, refusal=refusal)
        legacy = _as_legacy_tuple(solution)
    elif mode == TensionComputationMethod.LogLami:
        solution = _infer_tension_lami_log(mesh, mean_tension, angle_reading, refusal=refusal)
        legacy = _as_legacy_tuple(solution)
    elif mode == TensionComputationMethod.Variational:
        legacy = _infer_tension_variational_yd(mesh, mean_tension)
    elif mode in _FORCE_BALANCE_NORMALISATIONS:
        legacy = infer_tensions_force_balance(mesh, mean_tension, _FORCE_BALANCE_NORMALISATIONS[mode])
    else:  # if mode == TensionComputationMethod.YoungDupre:
        solution = _infer_tension_symmetrical_yd(mesh, mean_tension, angle_reading, refusal=refusal)
        legacy = _as_legacy_tuple(solution)
    return (*legacy, report, solution)


def _as_legacy_tuple(
    solution: ComponentTensionSolution,
) -> tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
    """Adapt a `ComponentTensionSolution` to the pre-existing 3-tuple return contract.

    Used by every builder the junction-graph work wired through `foambryo.junction_graph` (see that
    module's "Scope: which methods"). When the mesh's junction graph is not fragmented this
    is bit-identical to the old direct-`lstsq` return; the per-component/gauge/undetermined
    detail `ComponentTensionSolution` carries is only reachable through
    `infer_tensions_by_junction_component`.

    Args:
        solution (ComponentTensionSolution): A component-aware solve.

    Returns:
        tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
            `(tensions, dict_tensions, lstsq_residuals)`.
    """
    return (solution.tensions, solution.dict_tensions, solution.lstsq_residuals)


def _infer_tension_symmetrical_yd(
    mesh: "DcelData",
    mean_tension: float = 1,
    angle_reading: str = CHORD_READING,
    *,
    refusal: WedgeRefusal = NO_REFUSAL,
) -> ComponentTensionSolution:
    """Infer tensions using Symmetrical Young-Dupré method.

    Args:
        mesh (DcelData): Mesh to analyze.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.
        angle_reading (str, optional): which trijunction angle convention to read, `"chord"`
            (the default, and bit-identical to the historical behaviour) or `"quadric"`. See
            `foambryo.second_order_angle_reading`.
        refusal (WedgeRefusal, optional): which junction lines this estimator declines to use,
            and how (`foambryo.wedge_refusal`). Keyword-only; refusing nothing by default,
            which takes the untouched code path and is bitwise the historical behaviour.

    Returns:
        ComponentTensionSolution: Gauged per junction-graph component if the mesh's triple-
            junction graph is fragmented (the junction-graph work, `foambryo.junction_graph`); bit-identical
            to the direct `lstsq` solve otherwise.
    """
    angles = line_averaged_angles(mesh, reading=angle_reading, refusal=refusal)
    dict_areas = mesh.compute_areas_interfaces()
    keys_interface = [tuple(key) for key in sorted(dict_areas.keys())]
    if not angles.dict_rad:
        return all_interfaces_undetermined(keys_interface, mean_tension, angles.partition)
    matrix_m, vector_b = _build_matrix_tension_symmetrical_yd(
        angles.dict_rad,
        dict_areas,
        angles.dict_length,
        mean_tension,
    )
    return solve_gauged_least_squares(matrix_m, vector_b, keys_interface, mean_tension, angles.partition)


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
    angle_reading: str = CHORD_READING,
    *,
    refusal: WedgeRefusal = NO_REFUSAL,
) -> ComponentTensionSolution:
    """Infer tensions using projection Young-Dupré method.

    Args:
        mesh (DcelData): Mesh to analyze.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.
        angle_reading (str, optional): which trijunction angle convention to read, `"chord"`
            (the default, and bit-identical to the historical behaviour) or `"quadric"`. See
            `foambryo.second_order_angle_reading`.
        refusal (WedgeRefusal, optional): which junction lines this estimator declines to use,
            and how (`foambryo.wedge_refusal`). Keyword-only; refusing nothing by default,
            which takes the untouched code path and is bitwise the historical behaviour.

    Returns:
        ComponentTensionSolution: Gauged per junction-graph component if the mesh's triple-
            junction graph is fragmented (the junction-graph work, `foambryo.junction_graph`); bit-identical
            to the direct `lstsq` solve otherwise.
    """
    angles = line_averaged_angles(mesh, reading=angle_reading, refusal=refusal)
    dict_areas = mesh.compute_areas_interfaces()
    keys_interface = [tuple(key) for key in sorted(dict_areas.keys())]
    if not angles.dict_rad:
        return all_interfaces_undetermined(keys_interface, mean_tension, angles.partition)
    matrix_m, vector_b = _build_matrix_tension_projection_yd(
        angles.dict_rad,
        dict_areas,
        angles.dict_length,
        mean_tension,
    )
    return solve_gauged_least_squares(matrix_m, vector_b, keys_interface, mean_tension, angles.partition)


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
    deduplicate: bool = False,
    length_weighted: bool = False,
) -> tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
    """Infer tensions using Equilibrium method.

    Args:
        mesh (DcelData): Mesh to analyze.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.
        deduplicate (bool, optional): Emit each geometric edge's 3-row block once instead of
            once per incident half-edge. Defaults to False — see
            `_build_matrix_equilibrium_tensions`, this is a measured non-change.
        length_weighted (bool, optional): Weight each block by `length_e / L_total`. Defaults to
            False, likewise.

    Returns:
        tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
            - array of relative tensions
            - map interface id (label 1, label 2) -> tension on this interface
            - residuals of the least square method.
    """
    dict_areas = mesh.compute_areas_interfaces()
    matrix_m, vector_b = _build_matrix_equilibrium_tensions(
        mesh,
        dict_areas,
        mean_tension,
        deduplicate=deduplicate,
        length_weighted=length_weighted,
    )
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
    deduplicate: bool = False,
    length_weighted: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Get matrix M and vector B of system MX=B to solve to get relative tensions.

    **The two structural properties the defect-cleanup work recorded and deliberately did
    not change, now measured by the per-edge Young-Dupre work and still not changed by
    default.** `Equilibrium` is the only method
    that (i) carries **no junction-length weight** — every row is a unit tension direction — and
    (ii) emits each geometric edge's 3-row block **once per incident half-edge**, so on a closed
    mesh each block appears three times identically. Both are exposed here as flags so the
    effect is a regenerable measurement rather than an argument:

    - `deduplicate=True` emits one block per geometric edge. This is **not** a no-op even though
      the triplication is uniform, because the soft gauge row is *not* triplicated: removing the
      triplication makes the gauge row three times more influential relative to the physics rows.
    - `length_weighted=True` scales each block by `length_e / L_total`, making the method
      commensurate with the other eight.

    The per-edge Young-Dupre work measured both on the 42 equilibrium cases and **kept the defaults unchanged**
    (`False`, `False`) because both changes make the method *worse*. Public behaviour is
    preserved until a benchmark justifies flipping it, so the defaults stay, and the numbers are
    recorded rather than tuned away. Note also, recorded and unchanged:
    the guard is `len(edge.twin) == 1`, not `< 2`, so a **quadrijunctional** edge (twin length 3)
    is *not* skipped here although `geometry.compute_angles_tri` skips it, and a mesh-boundary
    edge (twin length 0) would be admitted with a single film. Neither occurs on the 47-case
    dataset (measured by the per-edge Young-Dupre work: no half-edge has an empty twin list, and
    quadrijunctional edges are 0-6 per mesh).

    Args:
        mesh (DcelData): Mesh to analyze. We need the half-edge structure.
        dict_areas (dict[tuple[int, int], float]): Area of interfaces.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.
        deduplicate (bool, optional): One block per geometric edge instead of per half-edge.
        length_weighted (bool, optional): Scale each block by `length_e / L_total`.

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
    contributing = [edge for edge in mesh.half_edges if len(edge.twin) != 1]
    if deduplicate:
        seen: set[tuple[int, int]] = set()
        unique_edges = []
        for edge in contributing:
            geometric_edge = (min(edge.origin.key, edge.destination.key), max(edge.origin.key, edge.destination.key))
            if geometric_edge not in seen:
                seen.add(geometric_edge)
                unique_edges.append(edge)
        contributing = unique_edges

    total_length = sum(edge.length for edge in contributing) if length_weighted else 1.0
    ne = len(contributing)

    vector_b = np.zeros(3 * ne + 1)
    vector_b[-1] = nm * mean_tension  # Sum of tensions = number of membrane <=> mean of tensions = 1
    matrix_m = np.zeros((3 * ne + 1, nm))
    matrix_m[-1] = 1

    ne_i = 0
    for edge in contributing:
        factor = (edge.length / total_length) if length_weighted else 1.0

        list_edges = [edge.key, *edge.twin]
        for current_edge_idx in list_edges:
            # print(current_edge_idx)
            current_edge = mesh.half_edges[current_edge_idx]
            # face_attached = current_edge.incident_face
            # Find the edges of the faces
            e = current_edge.incident_face.outer_component  # current_edge
            edges_face = []
            for _ in range(3):
                edges_face.append([e.origin.key, e.destination.key])
                # (Defect-cleanup fix.) Was `e = e.next`. `HalfEdge` has no `next` — the field is
                # `next_he` (dcel.py) — so this raised `AttributeError` on the first
                # iteration and `TensionComputationMethod.Equilibrium` had never run.
                e = e.next_he
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

            matrix_m[3 * ne_i, idx_tension] = vector_tension[0] * factor
            matrix_m[3 * ne_i + 1, idx_tension] = vector_tension[1] * factor
            matrix_m[3 * ne_i + 2, idx_tension] = vector_tension[2] * factor

        ne_i += 1

    return (matrix_m, vector_b)


def _infer_tension_cotan(
    mesh: "DcelData",
    mean_tension: float = 1,
    angle_reading: str = CHORD_READING,
    *,
    refusal: WedgeRefusal = NO_REFUSAL,
) -> ComponentTensionSolution:
    """Infer tensions using cotangents method.

    Args:
        mesh (DcelData): Mesh to analyze.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.
        angle_reading (str, optional): which trijunction angle convention to read, `"chord"`
            (the default, and bit-identical to the historical behaviour) or `"quadric"`. See
            `foambryo.second_order_angle_reading`.
        refusal (WedgeRefusal, optional): which junction lines this estimator declines to use,
            and how (`foambryo.wedge_refusal`). Keyword-only; refusing nothing by default,
            which takes the untouched code path and is bitwise the historical behaviour.

    Returns:
        ComponentTensionSolution: Gauged per junction-graph component if the mesh's triple-
            junction graph is fragmented (the junction-graph work, `foambryo.junction_graph`); bit-identical
            to the direct `lstsq` solve otherwise.
    """
    angles = line_averaged_angles(mesh, reading=angle_reading, refusal=refusal)
    dict_areas = mesh.compute_areas_interfaces()
    keys_interface = [tuple(key) for key in sorted(dict_areas.keys())]
    if not angles.dict_rad:
        return all_interfaces_undetermined(keys_interface, mean_tension, angles.partition)
    matrix_m, vector_b = _build_matrix_tension_cotan(angles.dict_rad, dict_areas, angles.dict_length, mean_tension)
    return solve_gauged_least_squares(matrix_m, vector_b, keys_interface, mean_tension, angles.partition)


def _build_matrix_tension_cotan(
    dict_rad: dict[tuple[int, int, int], float],
    dict_areas: dict[tuple[int, int], float],
    dict_length_tri: dict[tuple[int, int, int], float],
    mean_tension: float = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Get matrix M and vector B of system MX=B to solve to get relative tensions.

    Each trijunction contributes three rows, one per incident interface. Writing
    `z_i = sin(phi_i) / (1 - cos(phi_i)) = cot(phi_i / 2)` for the wedge angle `phi_i`
    *opposite* the interface carrying tension `gamma_i`, and
    `f_i = z_j z_k`, `s_i = z_i (z_j + z_k) / 2` (`_factors_z`), the row for `gamma_i` is

        -(s_i + f_i) gamma_i + s_i (gamma_j + gamma_k) = 0

    which is an exact identity whenever the junction is at equilibrium — i.e. whenever
    `phi_1 + phi_2 + phi_3 = 2 pi` and Lami's relation `gamma_i / sin(phi_i) = const` holds.
    Verified numerically over random equilibrium junctions to machine precision, and pinned
    by `tests/test_tension_matrix_properties.py`. The row is homogeneous of degree 1 in the
    tensions, so weighting the *whole* row by the junction-length fraction `factor` leaves
    the identity intact — which is what makes the weight a weight.

    (Defect fixed by the defect-cleanup work.) The diagonal entries used to read `-s3 - f3 * factor`, i.e.
    `(-s3) - (f3 * factor)`: the `s` half of the diagonal escaped `factor` while all six
    other entries carried it. On an exactly balanced junction the resulting row does not
    vanish; its residual is `s_i * gamma_i * (factor - 1)`, so it is correct only for a
    junction carrying the whole mesh's junction length (`factor = 1`) and the relative
    over-weighting of the diagonal grows like `1 / factor` — a factor of ~100 on a junction
    holding 0.5 % of the total length. This fix changes `Cotan`'s and `InvCotan`'s output.

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

        # (Defect-cleanup fix.) The three diagonal terms were written `-s3 - f3 * factor`, which Python
        # parses as `(-s3) - (f3 * factor)` — so the `s` half of the diagonal escaped the
        # junction-length weight while every other entry in the row carried it. See this
        # function's docstring for the identity and for what the old form cost.
        factor = dict_length_tri[(a, b, c)] / total_length_mesh
        matrix_m[3 * i, idx_tab] = (-s3 - f3) * factor
        matrix_m[3 * i, idx_tbc] = s3 * factor
        matrix_m[3 * i, idx_tac] = s3 * factor

        matrix_m[3 * i + 1, idx_tab] = s1 * factor
        matrix_m[3 * i + 1, idx_tbc] = (-s1 - f1) * factor
        matrix_m[3 * i + 1, idx_tac] = s1 * factor

        matrix_m[3 * i + 2, idx_tab] = s2 * factor
        matrix_m[3 * i + 2, idx_tbc] = s2 * factor
        matrix_m[3 * i + 2, idx_tac] = (-s2 - f2) * factor

    matrix_m[3 * nj, :nm] = 1  # Enforces mean of tensions = 1 (see B[-1])

    return (matrix_m, vector_b)


def _infer_tension_inv_cotan(
    mesh: "DcelData",
    mean_tension: float = 1,
    angle_reading: str = CHORD_READING,
    *,
    refusal: WedgeRefusal = NO_REFUSAL,
) -> ComponentTensionSolution:
    """Infer tensions using inverse cotangents method.

    Args:
        mesh (DcelData): Mesh to analyze.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.
        angle_reading (str, optional): which trijunction angle convention to read, `"chord"`
            (the default, and bit-identical to the historical behaviour) or `"quadric"`. See
            `foambryo.second_order_angle_reading`.
        refusal (WedgeRefusal, optional): which junction lines this estimator declines to use,
            and how (`foambryo.wedge_refusal`). Keyword-only; refusing nothing by default,
            which takes the untouched code path and is bitwise the historical behaviour.

    Returns:
        ComponentTensionSolution: Gauged per junction-graph component if the mesh's triple-
            junction graph is fragmented (the junction-graph work, `foambryo.junction_graph`); bit-identical
            to the direct `lstsq` solve otherwise.
    """
    angles = line_averaged_angles(mesh, reading=angle_reading, refusal=refusal)
    dict_areas = mesh.compute_areas_interfaces()
    keys_interface = [tuple(key) for key in sorted(dict_areas.keys())]
    if not angles.dict_rad:
        return all_interfaces_undetermined(keys_interface, mean_tension, angles.partition)
    matrix_m, vector_b = _build_matrix_tension_inv_cotan(angles.dict_rad, dict_areas, angles.dict_length, mean_tension)
    return solve_gauged_least_squares(matrix_m, vector_b, keys_interface, mean_tension, angles.partition)


def _build_matrix_tension_inv_cotan(
    dict_rad: dict[tuple[int, int, int], float],
    dict_areas: dict[tuple[int, int], float],
    dict_length_tri: dict[tuple[int, int, int], float],
    mean_tension: float = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Get matrix M and vector B of system MX=B to solve to get relative tensions.

    The reciprocal of `_build_matrix_tension_cotan`'s row: with the same `f_i`, `s_i` and the
    same "`phi_i` is opposite `gamma_i`" convention,

        -(1 / s_i + 1 / f_i) gamma_i + (1 / f_i) (gamma_j + gamma_k) = 0

    which is likewise an exact identity at an equilibrium junction (checked numerically to
    machine precision alongside the direct form). See `_build_matrix_tension_cotan`'s
    docstring for the shared derivation and for the operator-precedence defect fixed here; the
    residual its old form left on a balanced junction was `(gamma_i / s_i) * (factor - 1)`
    for rows 0 and 1 and `(gamma_i / f_i) * (factor - 1)` for row 2, which had the two terms
    written in the opposite order.

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

        # (Defect-cleanup fix.) Same precedence defect as `_build_matrix_tension_cotan`, plus a second
        # slip: row 2 read `-1 / f2 - 1 / s2`, the other two rows `-1 / s - 1 / f`. Once the
        # parentheses are in place that ordering is irrelevant — IEEE-754 addition is
        # commutative, so `(-1 / s2 - 1 / f2)` and `(-1 / f2 - 1 / s2)` are bit-identical —
        # but under the old form it decided *which* of the two terms escaped the weight, and
        # so changed the size of row 2's error. It is normalised here for readability only.
        factor = dict_length_tri[(a, b, c)] / total_length_mesh
        matrix_m[3 * i, idx_tab] = (-1 / s3 - 1 / f3) * factor
        matrix_m[3 * i, idx_tbc] = 1 / f3 * factor
        matrix_m[3 * i, idx_tac] = 1 / f3 * factor

        matrix_m[3 * i + 1, idx_tab] = 1 / f1 * factor
        matrix_m[3 * i + 1, idx_tbc] = (-1 / s1 - 1 / f1) * factor
        matrix_m[3 * i + 1, idx_tac] = 1 / f1 * factor

        matrix_m[3 * i + 2, idx_tab] = 1 / f2 * factor
        matrix_m[3 * i + 2, idx_tbc] = 1 / f2 * factor
        matrix_m[3 * i + 2, idx_tac] = (-1 / s2 - 1 / f2) * factor

    matrix_m[3 * nj, :nm] = 1  # Enforces mean of tensions = 1 (see B[-1])

    return (matrix_m, vector_b)


def _infer_tension_lami(
    mesh: "DcelData",
    mean_tension: float = 1,
    angle_reading: str = CHORD_READING,
    *,
    refusal: WedgeRefusal = NO_REFUSAL,
) -> ComponentTensionSolution:
    """Infer tensions using Lamy method.

    Args:
        mesh (DcelData): Mesh to analyze.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.
        angle_reading (str, optional): which trijunction angle convention to read, `"chord"`
            (the default, and bit-identical to the historical behaviour) or `"quadric"`. See
            `foambryo.second_order_angle_reading`.
        refusal (WedgeRefusal, optional): which junction lines this estimator declines to use,
            and how (`foambryo.wedge_refusal`). Keyword-only; refusing nothing by default,
            which takes the untouched code path and is bitwise the historical behaviour.

    Returns:
        ComponentTensionSolution: Gauged per junction-graph component if the mesh's triple-
            junction graph is fragmented (the junction-graph work, `foambryo.junction_graph`); bit-identical
            to the direct `lstsq` solve otherwise.
    """
    angles = line_averaged_angles(mesh, reading=angle_reading, refusal=refusal)
    dict_areas = mesh.compute_areas_interfaces()
    keys_interface = [tuple(key) for key in sorted(dict_areas.keys())]
    if not angles.dict_rad:
        return all_interfaces_undetermined(keys_interface, mean_tension, angles.partition)
    matrix_m, vector_b = _build_matrix_tension_lami(angles.dict_rad, dict_areas, angles.dict_length, mean_tension)
    return solve_gauged_least_squares(matrix_m, vector_b, keys_interface, mean_tension, angles.partition)


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
    angle_reading: str = CHORD_READING,
    *,
    refusal: WedgeRefusal = NO_REFUSAL,
) -> ComponentTensionSolution:
    """Infer tensions using inverse Lamy method.

    Args:
        mesh (DcelData): Mesh to analyze.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.
        angle_reading (str, optional): which trijunction angle convention to read, `"chord"`
            (the default, and bit-identical to the historical behaviour) or `"quadric"`. See
            `foambryo.second_order_angle_reading`.
        refusal (WedgeRefusal, optional): which junction lines this estimator declines to use,
            and how (`foambryo.wedge_refusal`). Keyword-only; refusing nothing by default,
            which takes the untouched code path and is bitwise the historical behaviour.

    Returns:
        ComponentTensionSolution: Gauged per junction-graph component if the mesh's triple-
            junction graph is fragmented (the junction-graph work, `foambryo.junction_graph`); bit-identical
            to the direct `lstsq` solve otherwise.
    """
    angles = line_averaged_angles(mesh, reading=angle_reading, refusal=refusal)
    dict_areas = mesh.compute_areas_interfaces()
    keys_interface = [tuple(key) for key in sorted(dict_areas.keys())]
    if not angles.dict_rad:
        return all_interfaces_undetermined(keys_interface, mean_tension, angles.partition)
    matrix_m, vector_b = _build_matrix_tension_inv_lami(angles.dict_rad, dict_areas, angles.dict_length, mean_tension)
    return solve_gauged_least_squares(matrix_m, vector_b, keys_interface, mean_tension, angles.partition)


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
    angle_reading: str = CHORD_READING,
    *,
    refusal: WedgeRefusal = NO_REFUSAL,
) -> ComponentTensionSolution:
    """Infer tensions using log Lamy method.

    The junction-graph gauge split (the junction-graph work, `foambryo.junction_graph`) is applied in
    `log`-space, exactly where this method itself works: `solve_gauged_least_squares` reads
    the per-component gauge value directly off `vector_b[-1]` (here
    `nm * log(mean_tension)`) rather than recomputing it from `mean_tension`, so the
    `exp` back-transform below is unaffected by whether the mesh fragments.

    Args:
        mesh (DcelData): Mesh to analyze.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.
        angle_reading (str, optional): which trijunction angle convention to read, `"chord"`
            (the default, and bit-identical to the historical behaviour) or `"quadric"`. See
            `foambryo.second_order_angle_reading`.
        refusal (WedgeRefusal, optional): which junction lines this estimator declines to use,
            and how (`foambryo.wedge_refusal`). Keyword-only; refusing nothing by default,
            which takes the untouched code path and is bitwise the historical behaviour.

    Returns:
        ComponentTensionSolution: Gauged per junction-graph component if the mesh's triple-
            junction graph is fragmented; bit-identical to the pre-junction-graph-work direct `lstsq`
            (+ `exp`) solve otherwise.
    """
    angles = line_averaged_angles(mesh, reading=angle_reading, refusal=refusal)
    dict_areas = mesh.compute_areas_interfaces()
    keys_interface = [tuple(key) for key in sorted(dict_areas.keys())]
    if not angles.dict_rad:
        return all_interfaces_undetermined(keys_interface, mean_tension, angles.partition)
    matrix_m, vector_b = _build_matrix_tension_lami_log(angles.dict_rad, dict_areas, angles.dict_length, mean_tension)
    solution = solve_gauged_least_squares(matrix_m, vector_b, keys_interface, mean_tension, angles.partition)
    tensions = np.exp(solution.tensions)
    return ComponentTensionSolution(
        tensions=tensions,
        dict_tensions=dict(zip(keys_interface, tensions.tolist(), strict=True)),
        component_of=solution.component_of,
        gauge_notes=solution.gauge_notes,
        undetermined=solution.undetermined,
        lstsq_residuals=solution.lstsq_residuals,
        fragmented=solution.fragmented,
        n_components=solution.n_components,
    )


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


def _infer_tension_young_dupre_local_component(
    mesh: "DcelData",
    mean_tension: float = 1,
    angle_reading: str = CHORD_READING,
    *,
    refusal: WedgeRefusal = NO_REFUSAL,
) -> ComponentTensionSolution:
    """Adapt `infer_tensions_young_dupre_local`'s result onto `ComponentTensionSolution`.

    Drops the per-edge residual field (still reachable via
    `foambryo.per_edge_tension_inference.infer_tensions_young_dupre_local` directly for a
    caller that wants it); keeps everything `infer_tensions_by_junction_component` needs.

    Args:
        mesh (DcelData): Mesh to analyze.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.
        angle_reading (str, optional): which trijunction angle convention to read, `"chord"`
            (the default, and bit-identical to the historical behaviour) or `"quadric"`. See
            `foambryo.second_order_angle_reading`.
        refusal (WedgeRefusal, optional): which trijunctional edges the estimator declines to
            use, and how (`foambryo.wedge_refusal`). Keyword-only; refusing nothing by default.

    Returns:
        ComponentTensionSolution: See `infer_tensions_young_dupre_local`'s own component
            fields (the junction-graph work, `foambryo.junction_graph`).
    """
    result = infer_tensions_young_dupre_local(mesh, mean_tension, angle_reading=angle_reading, refusal=refusal)
    return ComponentTensionSolution(
        tensions=result.tensions,
        dict_tensions=result.dict_tensions,
        component_of=result.component_of,
        gauge_notes=result.gauge_notes,
        undetermined=result.undetermined,
        lstsq_residuals=result.lstsq_residuals,
        fragmented=result.fragmented,
        n_components=result.n_components,
    )


#: Methods whose assembled system is `[physics rows; one trailing global gauge row]` with
#: physics rows keyed by triple junctions -- i.e. every method
#: `foambryo.junction_graph.solve_gauged_least_squares` can be safely applied to, and the
#: only methods `infer_tensions_by_junction_component` accepts. Kept as an explicit registry,
#: like `_FORCE_BALANCE_NORMALISATIONS`, so a newly-added `TensionComputationMethod` member
#: does not silently fall through either dispatch table. See `foambryo.junction_graph`'s
#: "Scope: which methods" for why `Equilibrium`, `ForceBalance*` and `Variational` are absent
#: (measured by the junction-graph work, not assumed).
_JUNCTION_GRAPH_GAUGED_BUILDERS: dict[
    TensionComputationMethod,
    "Callable[..., ComponentTensionSolution]",
] = {
    TensionComputationMethod.YoungDupre: _infer_tension_symmetrical_yd,
    TensionComputationMethod.ProjectionYoungDupre: _infer_tension_projection_yd,
    TensionComputationMethod.Cotan: _infer_tension_cotan,
    TensionComputationMethod.InvCotan: _infer_tension_inv_cotan,
    TensionComputationMethod.Lami: _infer_tension_lami,
    TensionComputationMethod.InvLami: _infer_tension_inv_lami,
    TensionComputationMethod.LogLami: _infer_tension_lami_log,
    TensionComputationMethod.YoungDupreLocal: _infer_tension_young_dupre_local_component,
}


def infer_tensions_by_junction_component(
    mesh: "DcelData",
    mean_tension: float = 1,
    mode: TensionComputationMethod = TensionComputationMethod.YoungDupre,
    *,
    refusal: WedgeRefusal = NO_REFUSAL,
) -> ComponentTensionSolution:
    """Infer tensions with the triple-junction-graph fragmentation surfaced explicitly.

    The junction-graph work; see `foambryo.junction_graph`'s module docstring for the
    mechanism and the API decision this
    implements (H. Turlier, 2026-08-02): infer per component, with a per-component gauge,
    and surface that structure rather than collapsing it into one number.

    Use this instead of `infer_tensions`/`infer_tensions_and_residuals` whenever a mesh might
    have a disconnected triple-junction graph (e.g. the `converged-meshes/` cohort, where
    12 of 72 meshes do, as the converged-mesh-cohort work measured) or a fully closed interface,
    and the
    caller needs to know *which* interfaces are (or are not) on a common scale before
    comparing or aggregating their tensions.

    Args:
        mesh (DcelData): Mesh to analyze.
        mean_tension (float, optional): Expected mean tension, applied **per component**.
            Defaults to 1.
        mode (TensionComputationMethod, optional): Which method to use. Must be a member of
            `_JUNCTION_GRAPH_GAUGED_BUILDERS` -- the angle/cotangent/Lami family plus
            `YoungDupreLocal`. Defaults to `YoungDupre`.
        refusal (WedgeRefusal, optional): which trijunction observations the estimator declines,
            and how (`foambryo.wedge_refusal`). Keyword-only; defaults to refusing nothing, which
            is bitwise the historical behaviour. This is the entry point to prefer under a
            refusal policy: refusing rows is itself a way to fragment the gauge structure, and
            this is the function that hands the caller the resulting components and
            undetermined interfaces rather than blending them.

    Returns:
        ComponentTensionSolution: `dict_tensions` maps every interface to its inferred
            tension, or `float("nan")` if it is undetermined (no triple line at all);
            `component_of` maps every interface to its gauge component, or `None` if
            undetermined; `gauge_notes` states, per component, which gauge was applied.

    Raises:
        ValueError: If `mode` does not use a single global junction-graph-coupled gauge
            (`Equilibrium`, any `ForceBalance*` member, or `Variational`) -- these methods'
            coupling structure is not the triple-junction graph (see
            `foambryo.junction_graph`'s "Scope: which methods"), so this decomposition does
            not apply to them; use `infer_tensions(mesh, mean_tension, mode)` instead.
    """
    if mode not in _JUNCTION_GRAPH_GAUGED_BUILDERS:
        message = (
            f"{mode} is not decomposed by the triple-junction graph -- its coupling "
            "structure is not the junction graph alone (see foambryo.junction_graph's "
            "'Scope: which methods'). Use infer_tensions(mesh, mean_tension, mode) instead."
        )
        raise ValueError(message)
    return _JUNCTION_GRAPH_GAUGED_BUILDERS[mode](mesh, mean_tension, refusal=refusal)


def _infer_tension_variational_yd(
    mesh: "DcelData",
    mean_tension: float = 1,
    route: "VariationalRoute | None" = None,
) -> tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
    """Infer tensions using the Variational method, by either solve route.

    Args:
        mesh (DcelData): Mesh to analyze.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.
        route (VariationalRoute | None, optional): Which solve route to use. Defaults to
            `None`, meaning `VARIATIONAL_ROUTE_DEFAULT` — see that constant and
            `VariationalRoute` for the uncertainty work's decision and its evidence.

    Returns:
        tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
            - array of relative tensions
            - map interface id (label 1, label 2) -> tension on this interface
            - residuals of the least square method.
    """
    if (route or VARIATIONAL_ROUTE_DEFAULT) is VariationalRoute.Operator:
        return _infer_tension_variational_operator(mesh, mean_tension)

    matrix_m, _vector_b, nm, keys = _build_matrix_discrete_sparse(mesh, mean_tension)

    f_g, f_p, g_g, g_p = _extract_submatrices(matrix_m, nm)

    mat_t = f_g - f_p @ (np.linalg.inv(g_p)) @ g_g

    mat_inference: NDArray[np.float64] = np.zeros((nm + 1, nm))
    mat_inference[:nm, :nm] = mat_t
    mat_inference[nm:] = 1
    b_inf = np.zeros(nm + 1)
    b_inf[-1] = nm * mean_tension  # Matthieu Perez: added mean_tension for consistency ?
    x, resid, rank, sigma = linalg.lstsq(mat_inference, b_inf)

    dict_tensions = {tuple(keys[i]): x[i] for i in range(nm)}

    return (x, dict_tensions, resid)


def _infer_tension_variational_operator(
    mesh: "DcelData",
    mean_tension: float = 1,
) -> tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
    """Route Q: solve `min ||M beta||^2` s.t. `mean(gamma) = mean_tension`, **through** `M`.

    `M = [D^T | -V^T]` is the `3N x (nm + nc)` energy-gradient operator whose columns are the
    sparse area gradients `D` and the negated cell-volume gradients `V` (the energy is
    `sum gamma A - sum p V`, so `grad_v E = D^T gamma - V^T p`). The gauge is eliminated by
    substitution rather than added as a row: with `a` the mean-tension constraint row,

        beta = beta_0 + Z y,   a^T beta_0 = mean_tension,   a^T Z = 0,

    with `Z` an orthonormal basis of `null(a^T)` from a single Householder reflector. What is
    left is an *unconstrained* least-squares problem in `y` on the operator `M Z`, solved by
    QR. **`M^T M` is never formed**, which is the whole point: on the golden mesh
    `cond(M^T M) ~ 7e5` while `cond(M Z) ~ 24` — the normal equations square the conditioning
    *and* the gauge direction is what makes `M` itself ill-conditioned (`cond(M) ~ 850`), so
    removing the gauge by substitution beats both.

    Args:
        mesh (DcelData): Mesh to analyze.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.

    Returns:
        tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
            tensions, the interface -> tension map, and `||M beta||^2` as a length-1 array
            (the residual sum of squares, matching what the other methods' `lstsq` returns).
    """
    keys, matrix_d = compute_area_derivative_sparse(mesh)
    _materials, matrix_v = compute_volume_derivative_sparse(mesh)
    cells_v = matrix_v[1:]  # drop the exterior row: its pressure is the gauge, fixed at 0
    nm = len(keys)
    n_cols = nm + cells_v.shape[0]

    # M as a dense (3N, nm + nc) block: nm + nc is a few tens, so this is a thin matrix, and
    # it is assembled from the *sparse* operators (the sparse-assembly work) rather than from dense gradient dicts.
    operator = np.empty((matrix_d.shape[1], n_cols))
    operator[:, :nm] = matrix_d.T.toarray()
    operator[:, nm:] = -cells_v.T.toarray()

    anchor = np.zeros(n_cols)
    anchor[:nm] = 1.0 / nm
    # Householder reflector mapping anchor/||anchor|| onto e_0, so H[:, 1:] spans null(a^T).
    unit_anchor = anchor / np.linalg.norm(anchor)
    reflector = unit_anchor - np.eye(n_cols)[0]
    reflector_norm = np.linalg.norm(reflector)
    basis = (
        np.eye(n_cols)[:, 1:]
        if reflector_norm < 1e-14
        else (np.eye(n_cols) - 2 * np.outer(reflector / reflector_norm, reflector / reflector_norm))[:, 1:]
    )
    particular = anchor * (mean_tension / (anchor @ anchor))

    y, _resid, _rank, _sv = linalg.lstsq(operator @ basis, -(operator @ particular))
    beta = particular + basis @ y
    tensions = beta[:nm]
    residual = operator @ beta

    dict_tensions = {tuple(keys[i]): tensions[i] for i in range(nm)}
    return (tensions, dict_tensions, np.array([float(residual @ residual)]))


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


def _build_matrix_discrete_sparse(
    mesh: "DcelData",
    mean_tension: float = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int, list[tuple[int, int]]]:
    """Sparse-operator assembly of the variational normal-equations matrix.

    Returns the same ``(matrix_m, vector_b, nm)`` as :func:`_build_matrix_discrete` (plus the
    ordered interface ``keys``), but assembles the Gram blocks
    ``F_gamma = D D^T``, ``D V^T``, ``V V^T`` from the **sparse** area/volume operators
    ``D`` (``n_interfaces x 3N``) and ``V`` (``n_cells x 3N``) via sparse matrix products --
    each interface/cell gradient is ~6 % dense.

    This is the sparse-assembly work's architectural fix: the dense per-interface ``(N, 3)`` gradient dicts
    (an object estimated at ~9 GB at 500 cells) and the ``O(n_interfaces^2 * N)`` Python double loop
    are gone. The only dense matrices formed are the tiny ``(nm+nc) x (nm+nc)`` Gram blocks -- the
    ``3N``-scale operator is never densified. The row normalisation, pressure sign convention and
    soft mean-tension row are byte-for-byte the same as :func:`_build_matrix_discrete`, so the
    returned matrix agrees with it to rtol=1e-10 (only the dot-product summation order differs);
    this is pinned in ``tests/test_variational_sparse_assembly.py``.

    Note: the returned ``matrix_m``'s soft row and ``vector_b`` are kept only for parity with
    :func:`_build_matrix_discrete`; both variational solves rebuild their own constraint row and
    right-hand side and read only ``matrix_m[:nm]`` / ``matrix_m[nm:-1]``.

    Args:
        mesh (DcelData): Mesh to analyze.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64], int, list[tuple[int, int]]]:
            M, B, number of interfaces, and the ordered interface keys (row order of the tensions).
    """
    keys, matrix_d = compute_area_derivative_sparse(mesh)  # D: (nm, 3N) CSR
    _materials, matrix_v = compute_volume_derivative_sparse(mesh)  # V: (n_materials, 3N) CSR
    cells_v = matrix_v[1:]  # drop the exterior row -> (nc, 3N)

    nm = len(keys)
    nc = cells_v.shape[0]

    # Tiny dense Gram blocks from sparse products; the 3N-scale object is never formed.
    f_gamma = (matrix_d @ matrix_d.T).toarray()  # <D_l, D_m>
    d_vt = (matrix_d @ cells_v.T).toarray()  # <D_l, V_n>
    v_vt = (cells_v @ cells_v.T).toarray()  # <V_i, V_n>

    matrix_m = np.zeros((nm + nc + 1, nm + nc))
    matrix_m[:nm, :nm] = f_gamma
    matrix_m[:nm, nm:] = -d_vt
    matrix_m[nm : nm + nc, :nm] = d_vt.T
    matrix_m[nm : nm + nc, nm:] = -v_vt

    lines_sum = np.sum(matrix_m, axis=1)
    for i in range(nm + nc):  # every row except the soft constraint row
        if lines_sum[i] != 0:
            matrix_m[i] /= lines_sum[i]

    matrix_m[nm + nc, :nm] = 1  # soft mean(tensions) = mean_tension row (kept for parity)

    vector_b = np.zeros(nm + nc + 1)
    vector_b[-1] = nm * mean_tension
    return (matrix_m, vector_b, nm, keys)


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
    *,
    angle_reading: str = CHORD_READING,
) -> dict[tuple[int, int, int], float]:
    """Compute residuals on trijunctions.

    Args:
        mesh (DcelData): Mesh to analyze.
        dict_tensions (dict[tuple[int, int], float]): Inferred tensions.
        alpha (float, optional): Quantile to apply. Defaults to 0.05.
        angle_reading (str, optional): one of
            `foambryo.second_order_angle_reading.ANGLE_READINGS`. Keyword-only. Defaults to
            `"chord"`. Pass the reading the tensions were inferred with, or the residual is
            measured against a different angle convention than the one it is judging.

    Returns:
        dict[tuple[int, int, int], float]: Map trijunction -> residue.
    """
    dict_rad, _, dict_length, report = mesh.compute_angles_tri_with_report(unique=False, reading=angle_reading)
    warn_on_partial_coverage(report)
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
