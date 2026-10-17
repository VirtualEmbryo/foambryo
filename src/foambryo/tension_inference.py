"""Surface tensions inferred from a multimaterial mesh, by any of the 13 methods of `TensionComputationMethod`.

The entry points are `infer_tensions` (a map interface -> tension), `infer_tensions_and_residuals`
(the same, with the solve's residuals), `infer_tensions_and_residuals_with_report` (plus the angle
reading's coverage, the entry point to use for anything published) and
`infer_tensions_by_junction_component` (each interface's connected piece of the junction graph made
explicit).

Conventions shared by every function here:

* an interface is keyed `(label_1, label_2)` with `label_1 < label_2`; label 0 is the exterior
  medium;
* tensions are **relative**: only their ratios are determined by the mesh. They are normalised so
  that their mean is `mean_tension`, exactly, in each connected piece of the junction graph (see
  `TensionComputationMethod` for the methods that normalise otherwise). Pressures computed from them
  (`foambryo.pressure_inference`) are in the same tension unit divided by the mesh's length unit;
* an interface whose tension the mesh does not determine is `float("nan")`, never a number;
* angles are read in radians.

Sacha Ichbiah 2021
Matthieu Perez 2024
"""
import time
import warnings
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse

from foambryo import linear_solves
from foambryo.crop_faces import remove_crop_interfaces, remove_crop_interfaces_from_map, vertex_rows_off_the_crop
from foambryo.force_balance_inference import ForceBalanceNormalisation, infer_tensions_force_balance
from foambryo.geometry import compute_area_derivative_sparse, compute_volume_derivative_sparse
from foambryo.junction_graph import (
    ComponentTensionSolution,
    partition_over_surviving_junctions,
    solve_gauged_least_squares,
    solve_with_exact_gauge,
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
    LineAveragedAngles,
    WedgeRefusal,
    all_interfaces_undetermined,
    line_averaged_angles,
    refusal_with_crop,
    require_refusal_is_honoured,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from foambryo.dcel import DcelData
    from foambryo.linear_solves import SolveDiagnostics
    from foambryo.per_edge_tension_inference import YoungDupreLocalResult


class TensionComputationMethod(Enum):
    """Describe the formula used to infer the tensions from contact angles at each junction.

    It can be:
    - YoungDupre (Young-Dupré with cosines only),
    - YoungDupreLocal (the same balance imposed **per trijunctional edge** instead of once per
      triple line on angles averaged along it — see `foambryo.per_edge_tension_inference`),
    - ProjectionYoungDupre (Young-Dupré with cosines and sines),
    - Equilibrium,
    - Cotan (cotangent formula, see [Yamamoto et al. 2026](https://doi.org/10.1016/j.devcel.2026.04.013)),
    - InvCotan (inverse of the cotangent formula),
    - Lami ([Lami's theorem](https://en.wikipedia.org/wiki/Lami%27s_theorem)),
    - InvLami (inverse of the Lami's relation),
    - LogLami (logarithm of the Lami's relation),
    - Variational (variational formulation, see our [article](https://doi.org/10.1038/s41592-023-02084-7)),
    - ForceBalance (the unified discrete force-balance operator with the mean-tension
      normalisation imposed exactly, through a KKT system),
    - ForceBalanceSoftRow (the same operator, with the normalisation as one more least-squares
      row, so its mean tension is only approximately `mean_tension`),
    - ForceBalanceMAP (the same operator, with a Gaussian prior; the normalisation is anchored only
      softly, so its mean tension is deliberately not exactly `mean_tension`; its relative tensions
      do not depend on the mesh's length unit or on the value of `mean_tension`, see
      `foambryo.force_balance_inference._solve_bayesian_map`).

    **The normalisation.** Only tension ratios are determined by a mesh, so every method fixes the
    overall scale (the "gauge") by one more condition: the tensions are normalised so that their
    mean is `mean_tension`, exactly, in each connected piece of the junction graph. This holds
    exactly for every method but two: `YoungDupre`, `YoungDupreLocal`, `ProjectionYoungDupre`,
    `Cotan`, `InvCotan`, `Lami` and `InvLami` meet it in each piece of the junction graph
    (`foambryo.junction_graph.solve_gauged_least_squares`); `LogLami` meets it on the geometric
    mean; `Equilibrium`, `Variational` (both routes) and `ForceBalance` meet it over the whole mesh
    at once; `ForceBalanceSoftRow` (a least-squares row by definition) and `ForceBalanceMAP` (a
    prior) do not meet it exactly. On a cropped mesh the mean runs over the interfaces whose
    tension is determined. Until 0.6.0 the first group, `Equilibrium` and the `Gram` route of
    `Variational` imposed it as a least-squares row of ones too; that row changed only the overall
    scale of their tensions, never a ratio (see `foambryo.junction_graph`).

    **Cost on large tissues.** `Equilibrium`, `Variational` and `ForceBalance` solve dense systems only: on a
    1 866-cell tissue each took one to two hours with a peak of about 68 GiB, where `YoungDupre`, the default, takes
    seconds (see the user guide's page on large tissues).

    The three `ForceBalance*` members are **one estimator under three normalisations**, not
    three physical models; they exist as separate members so that the comparison between the
    three is a regenerable benchmark row rather than a one-off table. See
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

    **Degenerate junctions.** `Cotan`, `InvCotan`, `Lami` and `LogLami` have coefficients that are
    infinite at a wedge of exactly 0 or 180 degrees (`1/sin`, `cot(phi/2)`, `log sin`). Such wedges occur
    on short triple lines whose faces are exactly coplanar (mesh vertices on the voxel lattice). These
    four methods leave such a line's equations out, with a `DegenerateWedgeWarning`; an interface left
    without any equation is `nan`. The other methods' coefficients stay finite and bounded there, and
    they are unchanged. See `DEGENERATE_WEDGE_TOLERANCE`.

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

    Both routes minimise the *same* discrete energy gradient; they differ in how, and the choice
    moves published numbers, which is why both are kept.

    - `Gram` (**route P**, the original): form the Gram normal equations `M^T M`, divide every
      row by **its own row sum**, Schur-eliminate the pressures, then least-squares with the
      mean-tension normalisation met exactly (`foambryo.junction_graph.solve_with_exact_gauge`;
      until 0.6.0 it was a least-squares row of ones). It has no formulation for crop faces.
    - `Operator` (**route Q**, the default): minimise `||M beta||^2` directly through `M`, with the
      normalisation eliminated by substitution. `M^T M` is never formed and there is no row
      normalisation.

    `TensionComputationMethod.Variational` always takes the default route,
    `VARIATIONAL_ROUTE_DEFAULT`; `Gram` is reachable only through the internal
    `_infer_tension_variational_yd(mesh, mean_tension, route=VariationalRoute.Gram)`.

    **The row normalisation is the substantive difference, and it has never been justified.**
    Dividing normal-equation row `i` by `sum_j (M^T M)_ij` is not a symmetric preconditioner
    (it destroys the symmetry of the Gram matrix), it is not scale-invariant in any documented
    sense, and it has no derivation in the paper or its Supplementary Note. Measured on
    `benchmarking-dataset`: the divisors span
    a factor **57-140 within a single mesh and change sign**, so it is a genuine, arbitrary
    reweighting of the physics rows — not a numerical nicety. That is why the two routes give
    different answers at all (~1.4e-4 per tension).

    **`Operator` is the default, chosen on measured evidence.** See `VARIATIONAL_ROUTE_DEFAULT`
    and `foambryo_benchmarks/run_variational_route.py`. `Gram` stays reachable so every published
    number remains reproducible.
    """

    Gram = 0
    Operator = 1


#: Route used by `TensionComputationMethod.Variational` when no route is given.
#:
#: **`Operator`, chosen on measured evidence over the former default `Gram`.** Over the 42
#: equilibrium cases, gauge-matched
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
#: Six of them are the angle-reading *family* named in `TensionComputationMethod`'s docstring —
#: `YoungDupre`, `YoungDupreLocal`, `ProjectionYoungDupre`, `Lami`, `InvLami`, `LogLami`. `Cotan`
#: and `InvCotan` are classed there as operator-inverting because they also assemble an
#: area/volume system, but their rows *do* read `compute_angles_tri`, so the selector reaches them
#: too, and their measurements are reported separately rather than pretending it does not.
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
    refusing its infeasible observations, and is not — exactly the failure mode the second-order
    angle reading exists to remove, and the one `foambryo.wedge_refusal` inherits.

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
    solver: str = "auto",
    sparse_threshold: int | None = None,
    diagnostics: bool | str = False,
) -> "dict[tuple[int, int], float] | tuple[dict[tuple[int, int], float], SolveDiagnostics]":
    """Infer the relative surface tension of every interface of a mesh, by the chosen method.

    Args:
        mesh (DcelData): Mesh to analyze, in any length unit (the tensions do not depend on it).
        mean_tension (float, optional): The normalisation: the tensions are normalised so that
            their mean is `mean_tension`, exactly, in each connected piece of the junction graph
            (for `LogLami`, their geometric mean; `Equilibrium`, `Variational` and `ForceBalance`
            normalise the whole mesh at once; `ForceBalanceSoftRow` meets it only approximately
            and `ForceBalanceMAP` deliberately not exactly; see `TensionComputationMethod`).
            Only tension ratios are measured, so this sets the unit of the tensions. Defaults
            to 1.
        mode (TensionComputationMethod, optional): method to infer tensions. Defaults to
            `TensionComputationMethod.YoungDupre`.
        angle_reading (str, optional): which trijunction angle convention the angle-reading
            methods (`ANGLE_READING_METHODS`) consume, one of
            `foambryo.second_order_angle_reading.ANGLE_READINGS`: `"chord"` (the default,
            bit-identical to every published number), `"quadric"` (the second-order reading),
            `"oriented_chord"` or `"oriented_quadric"` (the same readings with reflex wedges
            reported above 180 degrees). Keyword-only. A quadric reading that could not cover
            every trijunction face falls back to the chord there and emits an
            `AngleReadingCoverageWarning`; use :func:`infer_tensions_and_residuals_with_report`
            to get the coverage numbers instead of the warning.
        refusal (WedgeRefusal, optional): which trijunction observations the angle-reading
            methods decline to use, and how -- see `foambryo.wedge_refusal`. Keyword-only;
            defaults to `NO_REFUSAL` (refusing nothing), which is **bitwise** the historical
            behaviour. A non-default policy on a method that reads no wedge angle raises rather
            than being ignored. Refusing rows can leave an interface, or a whole mesh, with no
            surviving junction line: those interfaces are reported `float("nan")`, never a
            number, and how often a policy does that is a number to report beside any accuracy
            it buys.
        solver (str, optional): `"auto"` (the default), `"dense"` or `"sparse"`: how the linear
            systems are solved (`foambryo.linear_solves`). `"auto"` is dense below
            `sparse_threshold` unknowns and sparse at or above, and bitwise the dense result
            below it. `"sparse"` raises `ValueError` for a method with no sparse path
            (`Equilibrium`, `Variational`, `ForceBalance*`). Keyword-only.
        sparse_threshold (int | None, optional): unknowns (interfaces) at which `"auto"` goes
            sparse; `None` (the default) is `foambryo.junction_graph.SPARSE_SOLVE_MIN_UNKNOWNS`
            (2 000). Keyword-only.
        diagnostics (bool | str, optional): `False` (the default), `True` or `"spectrum"`: also
            return a `foambryo.linear_solves.SolveDiagnostics`, last, with one record per solve
            (solver, shape, rank, residual norm, condition number, time; `"spectrum"` adds the
            extreme singular values of the sparse solves). Keyword-only.

    Returns:
        dict[tuple[int, int], float]: map interface `(label_1, label_2)`, `label_1 < label_2`,
            label 0 the exterior medium -> relative tension on this interface. An interface whose
            tension the mesh does not determine is `float("nan")`, never a number: one with no
            triple line at all (a fully closed surface -- an engulfed cell, a sealed lumen), one
            whose every line was refused or left out, or, for `Cotan`, `InvCotan`, `Lami` and
            `LogLami`, one left without any equation by a degenerate wedge
            (`DegenerateWedgeWarning`). On a cropped mesh the interfaces made only of crop faces
            (`DcelData.crop_interfaces`) are absent. If the mesh's triple-junction graph is in
            more than one disconnected piece, or has any undetermined interface, this flat map
            holds tensions normalised separately in each piece, which are not on the same scale
            (a `JunctionGraphGaugeWarning` is emitted in that case); use
            :func:`infer_tensions_by_junction_component` to get each interface's piece
            explicitly, and never average or compare tensions across pieces from this map.
            With `diagnostics` on, the tuple `(map, SolveDiagnostics)`.

    Raises:
        ValueError: on an invalid `solver`, `sparse_threshold` or `diagnostics`; on an unknown
            `angle_reading`; on a non-default `angle_reading` or `refusal` with a method outside
            `ANGLE_READING_METHODS`; on `solver="sparse"` with `Equilibrium`, `Variational` or a
            `ForceBalance*` member; with `YoungDupreLocal`, on a quadric reading or on a mesh with
            no trijunctional edge; with `Cotan`, `InvCotan`, `Lami` or `LogLami`, if the system
            still holds a non-finite coefficient after the degenerate lines are left out.
        MalformedTripleLineError: if a triple line names an interface the mesh does not have
            (`foambryo.junction_graph`; not expected on a well-formed mesh).

    Warns:
        JunctionGraphGaugeWarning: when the junction graph is in several pieces or an interface
            is undetermined (the methods normalised per piece of the junction graph).
        AngleReadingCoverageWarning: when a quadric reading fell back to the chord on part of
            the mesh.
        DegenerateWedgeWarning: when `Cotan`, `InvCotan`, `Lami` or `LogLami` left out lines with a
            wedge at 0 or 180 degrees.
        SparseSolveFallbackWarning: when `lsqr` did not converge on a large system (the dense
            solve runs instead, or `lsqr`'s last iterate is returned if the dense one cannot fit
            in memory).
        DenseSolveMemoryWarning: when `solver="dense"` is forced on a system too large for the
            memory available.
    """
    result = infer_tensions_and_residuals(
        mesh, mean_tension, mode, angle_reading=angle_reading, refusal=refusal,
        solver=solver, sparse_threshold=sparse_threshold, diagnostics=diagnostics,
    )
    if diagnostics is False:
        return result[1]
    return (result[1], result[3])


def infer_tensions_and_residuals(
    mesh: "DcelData",
    mean_tension: float = 1,
    mode: TensionComputationMethod = TensionComputationMethod.YoungDupre,
    *,
    angle_reading: str = CHORD_READING,
    refusal: WedgeRefusal = NO_REFUSAL,
    solver: str = "auto",
    sparse_threshold: int | None = None,
    diagnostics: bool | str = False,
) -> tuple:
    """Infer the relative surface tensions by the chosen method, with the solve's residuals.

    `infer_tensions` returns this function's second value. Every argument, warning and exception
    is `infer_tensions`'s; this function is where its `JunctionGraphGaugeWarning` and
    `AngleReadingCoverageWarning` are emitted.

    Args:
        mesh (DcelData): Mesh to analyze, in any length unit.
        mean_tension (float, optional): The normalisation: the tensions are normalised so that
            their mean is `mean_tension`, exactly, in each connected piece of the junction graph
            (see :func:`infer_tensions` for the methods that normalise otherwise). Defaults to 1.
        mode (TensionComputationMethod, optional): method to infer tensions. Defaults to
            `TensionComputationMethod.YoungDupre`.
        angle_reading (str, optional): one of
            `foambryo.second_order_angle_reading.ANGLE_READINGS`; `"chord"` is the default.
            Keyword-only. See :func:`infer_tensions`.
        refusal (WedgeRefusal, optional): which trijunction observations the angle-reading
            methods decline to use, and how -- see `foambryo.wedge_refusal`. Keyword-only;
            defaults to `NO_REFUSAL` (refusing nothing), which is **bitwise** the historical
            behaviour. See :func:`infer_tensions`.
        solver (str, optional): `"auto"` (the default), `"dense"` or `"sparse"`, as for
            :func:`infer_tensions`. Keyword-only.
        sparse_threshold (int | None, optional): unknowns at which `"auto"` goes sparse; `None`
            (the default) is `foambryo.junction_graph.SPARSE_SOLVE_MIN_UNKNOWNS` (2 000).
            Keyword-only.
        diagnostics (bool | str, optional): `False` (the default), `True` or `"spectrum"`: also
            return a `foambryo.linear_solves.SolveDiagnostics`, last. Keyword-only.

    Returns:
        tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
            - the relative tensions as an array, in the map's key order (sorted interfaces);
            - map interface `(label_1, label_2)` -> tension on this interface. See
              :func:`infer_tensions`'s return value for the undetermined-interface (`nan`) and
              junction-graph (normalised per piece) caveats -- they apply here identically;
            - the squared norm of the residual of the solved equations, without the
              normalisation (which is met exactly and adds none): a length-1 array, one entry per
              connected piece of the junction graph when there are several, empty when no
              interface is determined; in log space for `LogLami`;
            - with `diagnostics` on, a fourth value: the `SolveDiagnostics`.

    Raises:
        ValueError: as :func:`infer_tensions`.
        MalformedTripleLineError: as :func:`infer_tensions`.
    """
    with linear_solves.solve_options(solver, sparse_threshold, diagnostics) as collected:
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
    if collected is None:
        return (tensions, dict_tensions, residuals)
    return (tensions, dict_tensions, residuals, collected)


def infer_tensions_and_residuals_with_report(
    mesh: "DcelData",
    mean_tension: float = 1,
    mode: TensionComputationMethod = TensionComputationMethod.YoungDupre,
    *,
    angle_reading: str = CHORD_READING,
    refusal: WedgeRefusal = NO_REFUSAL,
    solver: str = "auto",
    sparse_threshold: int | None = None,
    diagnostics: bool | str = False,
) -> tuple:
    """:func:`infer_tensions_and_residuals`, plus which trijunction faces the reading covered.

    The reporting entry point, and the one to use for anything that will be published or
    benchmarked. It emits neither `AngleReadingCoverageWarning` nor `JunctionGraphGaugeWarning`:
    the caller holds the coverage report, and can read the junction graph's pieces with
    :func:`infer_tensions_by_junction_component`. Every other warning is
    :func:`infer_tensions`'s.

    Args:
        mesh (DcelData): Mesh to analyze, in any length unit.
        mean_tension (float, optional): The normalisation: the tensions are normalised so that
            their mean is `mean_tension`, exactly, in each connected piece of the junction graph
            (see :func:`infer_tensions` for the methods that normalise otherwise). Defaults to 1.
        mode (TensionComputationMethod, optional): method to infer tensions. Defaults to
            `TensionComputationMethod.YoungDupre`.
        angle_reading (str, optional): one of
            `foambryo.second_order_angle_reading.ANGLE_READINGS`; `"chord"` is the default.
            Keyword-only. See :func:`infer_tensions`.
        refusal (WedgeRefusal, optional): which trijunction observations the angle-reading
            methods decline to use, and how -- see `foambryo.wedge_refusal`. Keyword-only;
            defaults to `NO_REFUSAL` (refusing nothing), which is **bitwise** the historical
            behaviour. See :func:`infer_tensions`.
        solver (str, optional): `"auto"` (the default), `"dense"` or `"sparse"`, as for
            :func:`infer_tensions`. Keyword-only.
        sparse_threshold (int | None, optional): unknowns at which `"auto"` goes sparse; `None`
            (the default) is `foambryo.junction_graph.SPARSE_SOLVE_MIN_UNKNOWNS` (2 000).
            Keyword-only.
        diagnostics (bool | str, optional): `False` (the default), `True` or `"spectrum"`: also
            return a `foambryo.linear_solves.SolveDiagnostics`, last. Keyword-only.

    Returns:
        tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64],
            AngleReadingReport]: the three values :func:`infer_tensions_and_residuals` returns,
            plus the coverage report. For a method outside `ANGLE_READING_METHODS`, or for a
            chord reading, the report is the trivial complete-coverage one. With `diagnostics`
            on, a fifth value: the `SolveDiagnostics`.

    Raises:
        ValueError: as :func:`infer_tensions`; in particular if `angle_reading` is not `"chord"`
            and `mode` does not read angles.
        MalformedTripleLineError: as :func:`infer_tensions`.
    """
    with linear_solves.solve_options(solver, sparse_threshold, diagnostics) as collected:
        tensions, dict_tensions, residuals, report, _solution = _infer_tensions_and_residuals_with_report_and_solution(
            mesh, mean_tension, mode, angle_reading=angle_reading, refusal=refusal,
        )
    if collected is None:
        return (tensions, dict_tensions, residuals, report)
    return (tensions, dict_tensions, residuals, report, collected)


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
    # On a cropped mesh the angle-reading methods decline every junction edge that touches a crop
    # face, whatever the caller's refusal (`foambryo.crop_faces`); without crop faces this returns
    # `refusal` itself and nothing below changes.
    refusal = refusal_with_crop(mesh, refusal) if mode in ANGLE_READING_METHODS else refusal

    solution: ComponentTensionSolution | YoungDupreLocalResult | None = None
    if mode == TensionComputationMethod.YoungDupreLocal:
        solution = infer_tensions_young_dupre_local(mesh, mean_tension, angle_reading=angle_reading, refusal=refusal)
        return (solution.tensions, solution.dict_tensions, solution.lstsq_residuals, report, solution)
    elif mode == TensionComputationMethod.ProjectionYoungDupre:
        solution = _infer_tension_projection_yd(mesh, mean_tension, angle_reading, refusal=refusal)
    elif mode == TensionComputationMethod.Equilibrium:
        linear_solves.require_dense_only_method_is_not_forced_sparse(str(mode))
        legacy = _infer_tension_equilibrium(mesh, mean_tension)
    elif mode == TensionComputationMethod.Cotan:
        solution = _infer_tension_cotan(mesh, mean_tension, angle_reading, refusal=refusal)
    elif mode == TensionComputationMethod.InvCotan:
        solution = _infer_tension_inv_cotan(mesh, mean_tension, angle_reading, refusal=refusal)
    elif mode == TensionComputationMethod.Lami:
        solution = _infer_tension_lami(mesh, mean_tension, angle_reading, refusal=refusal)
    elif mode == TensionComputationMethod.InvLami:
        solution = _infer_tension_inv_lami(mesh, mean_tension, angle_reading, refusal=refusal)
    elif mode == TensionComputationMethod.LogLami:
        solution = _infer_tension_lami_log(mesh, mean_tension, angle_reading, refusal=refusal)
    elif mode == TensionComputationMethod.Variational:
        linear_solves.require_dense_only_method_is_not_forced_sparse(str(mode))
        legacy = _infer_tension_variational_yd(mesh, mean_tension)
    elif mode in _FORCE_BALANCE_NORMALISATIONS:
        linear_solves.require_dense_only_method_is_not_forced_sparse(str(mode))
        legacy = infer_tensions_force_balance(mesh, mean_tension, _FORCE_BALANCE_NORMALISATIONS[mode])
    else:  # if mode == TensionComputationMethod.YoungDupre:
        solution = _infer_tension_symmetrical_yd(mesh, mean_tension, angle_reading, refusal=refusal)
    solution, legacy = _without_crop_interfaces(mesh, solution, legacy if solution is None else None)
    return (*legacy, report, solution)


def _without_crop_interfaces(
    mesh: "DcelData",
    solution: "ComponentTensionSolution | None",
    legacy: tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]] | None,
) -> tuple[
    "ComponentTensionSolution | None",
    tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]],
]:
    """Remove the crop interfaces from a method's result, as `foambryo.crop_faces` prescribes.

    Takes either a junction-graph solution (and derives the legacy 3-tuple from it) or a legacy
    3-tuple alone. Without crop faces both are returned as they are, bit for bit.
    """
    if solution is not None:
        solution = remove_crop_interfaces(solution, mesh.crop_interfaces)
        return solution, _as_legacy_tuple(solution)
    if mesh.has_crop_faces:
        tensions, dict_tensions = remove_crop_interfaces_from_map(legacy[1], mesh.crop_interfaces)
        legacy = (tensions, dict_tensions, legacy[2])
    return None, legacy


def _as_legacy_tuple(
    solution: ComponentTensionSolution,
) -> tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
    """Adapt a `ComponentTensionSolution` to the pre-existing 3-tuple return contract.

    Used by every builder solved through `foambryo.junction_graph` (see that
    module's "Scope: which methods"). The per-piece, normalisation and undetermined-interface
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
        ComponentTensionSolution: Normalised in each connected piece of the triple-junction graph
            when it is in several (`foambryo.junction_graph`), over the whole mesh otherwise; the
            mean-tension normalisation is met exactly either way.
    """
    angles = line_averaged_angles(mesh, reading=angle_reading, refusal=refusal)
    dict_areas = mesh.compute_areas_interfaces()
    keys_interface = [tuple(key) for key in sorted(dict_areas.keys())]
    if not angles.dict_rad:
        return all_interfaces_undetermined(keys_interface, mean_tension, angles.partition)
    # Sparse assembly when the whole system takes the sparse path: at tissue scale the dense matrix
    # alone (3 lines + 1 rows by interfaces) is several GiB before any solve (`foambryo.linear_solves`).
    as_sparse = linear_solves.use_sparse_solver(len(keys_interface))
    if not as_sparse:
        n_lines = len({tuple(sorted(key)) for key in angles.dict_rad})
        linear_solves.warn_if_forced_dense_exceeds_memory(
            3 * n_lines + 1, len(keys_interface), "YoungDupre tension system", stacklevel=4,
        )
    matrix_m, vector_b = _build_matrix_tension_symmetrical_yd(
        angles.dict_rad,
        dict_areas,
        angles.dict_length,
        mean_tension,
        as_sparse=as_sparse,
    )
    return solve_gauged_least_squares(
        matrix_m, vector_b, keys_interface, mean_tension, angles.partition, system="YoungDupre tension system",
    )


def _build_matrix_tension_symmetrical_yd(
    dict_rad: dict[tuple[int, int, int], float],
    dict_areas: dict[tuple[int, int], float],
    dict_length_tri: dict[tuple[int, int, int], float],
    mean_tension: float = 1,
    *,
    as_sparse: bool = False,
) -> tuple[NDArray[np.float64] | sparse.csr_matrix, NDArray[np.float64]]:
    """Get matrix M and vector B of system MX=B to solve to get relative tensions.

    The entries are computed once, as `(row, column, value)` triplets, and written either into a
    dense matrix (by assignment, exactly as this builder always wrote them: the dense matrix is
    bitwise the historical one) or into a CSR matrix holding the same values (`as_sparse`), which
    never allocates the `(3 n_lines + 1) x n_interfaces` dense array: 4.7 GB on a 1 866-cell tissue.

    Args:
        dict_rad (dict[tuple[int, int, int], float]): Mean angles in radian at one side of a trijunction.
        dict_areas (dict[tuple[int, int], float]): Area of interfaces.
        dict_length_tri (dict[tuple[int, int, int], float]): Length of trijunctions.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.
        as_sparse (bool, optional): return M as a `scipy.sparse.csr_matrix`. Keyword-only.
            Defaults to False.

    Returns:
        tuple[NDArray[np.float64] | scipy.sparse.csr_matrix, NDArray[np.float64]]:
            - M, `(3 n_lines + 1, n_interfaces)`, the gauge row (all ones) last
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

    # CLASSICAL
    # YOUNG-DUPRÉ
    # Nine entries per line, in the order the dense builder has always assigned them.
    rows = np.empty(9 * nj, dtype=np.int64)
    cols = np.empty(9 * nj, dtype=np.int64)
    values = np.empty(9 * nj, dtype=np.float64)
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
        span = slice(9 * i, 9 * i + 9)
        rows[span] = (3 * i, 3 * i, 3 * i, 3 * i + 1, 3 * i + 1, 3 * i + 1, 3 * i + 2, 3 * i + 2, 3 * i + 2)
        cols[span] = (idx_tab, idx_tbc, idx_tac) * 3
        values[span] = (
            1 * factor, np.cos(angle_b) * factor, np.cos(angle_a) * factor,
            np.cos(angle_b) * factor, 1 * factor, np.cos(angle_c) * factor,
            np.cos(angle_a) * factor, np.cos(angle_c) * factor, 1 * factor,
        )

    if as_sparse:
        gauge = np.arange(nm)
        matrix_sparse = sparse.coo_matrix(
            (
                np.concatenate([values, np.ones(nm)]),
                (np.concatenate([rows, np.full(nm, 3 * nj)]), np.concatenate([cols, gauge])),
            ),
            shape=(3 * nj + 1, nm),
        ).tocsr()
        return (matrix_sparse, vector_b)

    matrix_m = np.zeros((3 * nj + 1, nm))
    matrix_m[rows, cols] = values
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
        ComponentTensionSolution: Normalised in each connected piece of the triple-junction graph
            when it is in several (`foambryo.junction_graph`), over the whole mesh otherwise; the
            mean-tension normalisation is met exactly either way.
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
    if mesh.has_crop_faces:
        return _infer_tension_equilibrium_off_the_crop(mesh, dict_areas, mean_tension, deduplicate, length_weighted)
    matrix_m, vector_b = _build_matrix_equilibrium_tensions(
        mesh,
        dict_areas,
        mean_tension,
        deduplicate=deduplicate,
        length_weighted=length_weighted,
    )
    # Exact gauge (`solve_with_exact_gauge`): the builder's last row is read for its value only. Until
    # 0.6.0 it was a soft row of ones, which the unweighted Equilibrium rows pulled the mean off by 8 %
    # typically and up to 80 %.
    x, resid, _note = solve_with_exact_gauge(
        matrix_m[:-1], vector_b[:-1], float(vector_b[-1]), system="Equilibrium tension system", use_sparse=False,
    )
    dict_tensions = {}
    key_interface = np.array(sorted(dict_areas.keys()))
    nm = len(key_interface)
    for i in range(nm):
        dict_tensions[tuple(key_interface[i])] = x[i]

    return (x, dict_tensions, resid)


def _infer_tension_equilibrium_off_the_crop(
    mesh: "DcelData",
    dict_areas: dict[tuple[int, int], float],
    mean_tension: float,
    deduplicate: bool,
    length_weighted: bool,
) -> tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
    """`Equilibrium` on a cropped mesh: no block for a junction edge with a vertex on a crop face.

    The same blocks as `_build_matrix_equilibrium_tensions`, minus those of the half-edges that
    touch a crop face (`foambryo.crop_faces`). An interface left without any block (a crop
    interface, or a real one whose every junction edge touches the crop) is undetermined: its
    column leaves the system and it is `nan`; the gauge then fixes the mean of the determined
    tensions exactly, `sum = n_determined * mean_tension`. One global gauge, as on an uncropped mesh.

    Returns:
        The same three values as `_infer_tension_equilibrium`, over every interface (crop ones
        included, `nan`); the caller removes the crop interfaces.
    """
    matrix_m, _vector_b = _build_matrix_equilibrium_tensions(
        mesh,
        dict_areas,
        mean_tension,
        deduplicate=deduplicate,
        length_weighted=length_weighted,
        excluded_vertices=frozenset(int(v) for v in mesh.crop_vertices),
    )
    physics = matrix_m[:-1]
    determined = np.flatnonzero(np.any(physics != 0, axis=0))
    solved, resid, _note = solve_with_exact_gauge(
        physics[:, determined], np.zeros(len(physics)), len(determined) * mean_tension,
        system="Equilibrium tension system, cropped mesh", use_sparse=False,
    )
    keys = sorted(dict_areas.keys())
    x = np.full(len(keys), np.nan)
    x[determined] = solved
    dict_tensions = {tuple(key): x[i] for i, key in enumerate(keys)}
    return (x, dict_tensions, resid)


def _build_matrix_equilibrium_tensions(
    mesh: "DcelData",
    dict_areas: dict[tuple[int, int], float],
    mean_tension: float = 1,
    deduplicate: bool = False,
    length_weighted: bool = False,
    excluded_vertices: frozenset[int] = frozenset(),
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Get matrix M and vector B of system MX=B to solve to get relative tensions.

    **Two structural properties, recorded, measured and deliberately not changed by default.**
    `Equilibrium` is the only method
    that (i) carries **no junction-length weight** — every row is a unit tension direction — and
    (ii) emits each geometric edge's 3-row block **once per incident half-edge**, so on a closed
    mesh each block appears three times identically. Both are exposed here as flags so the
    effect is a regenerable measurement rather than an argument:

    - `deduplicate=True` emits one block per geometric edge. Until 0.6.0 this was **not** a no-op
      even though the triplication is uniform, because the normalisation was a least-squares row
      of ones that was *not* triplicated: removing the triplication made it three times more
      influential relative to the physics rows. Since the normalisation is exact (0.7.0) a uniform
      triplication no longer moves the solution beyond rounding (3e-15 on benchmark case 000).
    - `length_weighted=True` scales each block by `length_e / L_total`, making the method
      commensurate with the other eight.

    Both were measured on the 42 equilibrium cases and **the defaults were kept unchanged**
    (`False`, `False`) because both changes made the method *worse*. Public behaviour is
    preserved until a benchmark justifies flipping it, so the defaults stay, and the numbers are
    recorded rather than tuned away. Note also, recorded and unchanged:
    the guard is `len(edge.twin) == 1`, not `< 2`, so a **quadrijunctional** edge (twin length 3)
    is *not* skipped here although `geometry.compute_angles_tri` skips it, and a mesh-boundary
    edge (twin length 0) would be admitted with a single film. Neither occurs on the 47-case
    dataset (measured: no half-edge has an empty twin list, and quadrijunctional edges are 0-6
    per mesh).

    Args:
        mesh (DcelData): Mesh to analyze. We need the half-edge structure.
        dict_areas (dict[tuple[int, int], float]): Area of interfaces.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.
        deduplicate (bool, optional): One block per geometric edge instead of per half-edge.
        length_weighted (bool, optional): Scale each block by `length_e / L_total`.
        excluded_vertices (frozenset[int], optional): vertex keys whose half-edges give no block
            (the crop faces' vertices on a cropped mesh). Defaults to none, the historical system.

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
    if excluded_vertices:
        contributing = [
            edge
            for edge in contributing
            if edge.origin.key not in excluded_vertices and edge.destination.key not in excluded_vertices
        ]
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


#: A trijunction wedge whose angle lies within this many radians of 0 or pi is **degenerate**, and the
#: `Cotan`, `InvCotan`, `Lami` and `LogLami` equations of its triple line are left out. Those four
#: methods' coefficients are singular there: `Lami` divides by `sin(theta)`, `Cotan` and `InvCotan` use
#: `z = sin(phi) / (1 - cos(phi))` (`0/0` at 0, and 0 at pi, where `InvCotan` divides by it) and `LogLami`
#: takes `log(sin(theta))`. The value is fixed by the arithmetic of the reading, not tuned: a wedge is
#: `arccos` of the dot product of two computed unit normals, whose rounding error is at most about
#: `4 eps` (the three-term sum and the two normalisations), and near ±1 `arccos` turns a dot-product
#: error `e` into an angle error `sqrt(2 e)`. Within `sqrt(8 eps)` ≈ 4.2e-8 rad (2.4e-6 degrees) of 0 or
#: pi a wedge cannot be told apart from exactly 0 or pi. Measured on 628 meshes (benchmark, crops,
#: embryo series, converged meshes, two large tissues, synthetic foams): every degenerate wedge on
#: them is 0, pi or `arccos(-1 + 1 ulp)` = pi - 1.49e-8, from exactly coplanar faces on short
#: triple lines, and the nearest other wedge on a real or benchmark mesh is 9.2e-4 rad away (1.0e-6 on
#: a synthetic foam), so the result does not depend on where in that gap the cut lies.
DEGENERATE_WEDGE_TOLERANCE = float(np.sqrt(8 * np.finfo(float).eps))


class DegenerateWedgeWarning(UserWarning):
    """Triple lines with a wedge at 0 or 180 degrees were left out of a method whose coefficients are singular there.

    Emitted by `Cotan`, `InvCotan`, `Lami` and `LogLami`, once per call, naming how many lines were left
    out and how many interfaces were left without any equation (reported as `nan`).
    """


def _is_degenerate_wedge(angle: float) -> bool:
    """Whether `angle` (radians) is non-finite or within `DEGENERATE_WEDGE_TOLERANCE` of 0 or pi."""
    return not np.isfinite(angle) or angle <= DEGENERATE_WEDGE_TOLERANCE or angle >= np.pi - DEGENERATE_WEDGE_TOLERANCE


def _without_degenerate_wedges(mesh: "DcelData", angles: LineAveragedAngles, method: str) -> LineAveragedAngles:
    """Leave out the triple lines with a degenerate wedge, for a method whose coefficients are singular there.

    A triple line is left out when any of its three wedge angles is degenerate
    (`DEGENERATE_WEDGE_TOLERANCE`): its three wedge keys leave `dict_rad`, so its equations are never
    written. `dict_length` is returned whole, so the total junction length that normalises every row
    is unchanged and every kept row is bitwise the row it was. The junction-graph partition is
    recomputed over the surviving lines (`junction_graph.partition_over_surviving_junctions`), so an
    interface left with no line is undetermined (`nan`) and a graph that splits is gauged per
    component, as for any fragmented mesh.

    With no degenerate wedge this returns `angles` itself: the untouched path is bitwise the
    historical one.

    Args:
        mesh (DcelData): the mesh the angles were read on.
        angles (LineAveragedAngles): `wedge_refusal.line_averaged_angles`'s result.
        method (str): the method's name, for the warning.

    Returns:
        LineAveragedAngles: `angles`, or a copy without the degenerate lines and with the partition
            over the remaining ones. Emits one `DegenerateWedgeWarning` when anything is left out.
    """

    def line(key: tuple[int, int, int]) -> tuple[int, ...]:
        return tuple(sorted(int(label) for label in key))

    degenerate = {line(key) for key, angle in angles.dict_rad.items() if _is_degenerate_wedge(angle)}
    if not degenerate:
        return angles
    kept = {key: angle for key, angle in angles.dict_rad.items() if line(key) not in degenerate}
    surviving = {line(key) for key in kept}
    partition = partition_over_surviving_junctions(mesh.f[:, :3], mesh.f[:, 3:5], surviving)
    newly_undetermined = len(set(partition.isolated) - set(angles.partition.isolated))
    message = (
        f"{method}: {len(degenerate)} triple line(s) have a wedge angle at 0 or 180 degrees (within "
        f"{DEGENERATE_WEDGE_TOLERANCE:.1e} rad), where this method's coefficients are infinite. Their equations "
        f"are left out; {newly_undetermined} interface(s) left without any equation are reported as nan."
    )
    warnings.warn(message, DegenerateWedgeWarning, stacklevel=3)
    return LineAveragedAngles(kept, angles.dict_length, partition, angles.report)


def _require_finite_system(
    matrix_m: "NDArray[np.float64] | sparse.spmatrix",
    vector_b: NDArray[np.float64],
    method: str,
) -> None:
    """Refuse to hand a system with a non-finite coefficient to the solver.

    After `_without_degenerate_wedges` no such coefficient is expected. Should one remain (a cause other
    than a degenerate wedge), neither `lsqr` nor the dense solve may run on it: a `nan` in the matrix
    gives no answer, only a long iteration or a large dense allocation first.

    Raises:
        ValueError: naming the method and the number of non-finite entries.
    """
    data = matrix_m.data if sparse.issparse(matrix_m) else matrix_m
    n_matrix = int(np.count_nonzero(~np.isfinite(data)))
    n_vector = int(np.count_nonzero(~np.isfinite(vector_b)))
    if n_matrix or n_vector:
        message = (
            f"{method}: the assembled system has {n_matrix} non-finite coefficient(s) and {n_vector} non-finite "
            "right-hand side value(s) after the degenerate triple lines were left out; it is not solved."
        )
        raise ValueError(message)


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

    A triple line with a wedge at 0 or 180 degrees (`DEGENERATE_WEDGE_TOLERANCE`) gives this method
    infinite coefficients and is left out, with a `DegenerateWedgeWarning`
    (`_without_degenerate_wedges`); the system is never solved with a non-finite coefficient
    (`_require_finite_system` raises `ValueError` instead).

    Returns:
        ComponentTensionSolution: Normalised in each connected piece of the triple-junction graph
            when it is in several (`foambryo.junction_graph`), over the whole mesh otherwise; the
            mean-tension normalisation is met exactly either way.
    """
    angles = line_averaged_angles(mesh, reading=angle_reading, refusal=refusal)
    angles = _without_degenerate_wedges(mesh, angles, "Cotan")
    dict_areas = mesh.compute_areas_interfaces()
    keys_interface = [tuple(key) for key in sorted(dict_areas.keys())]
    if not angles.dict_rad:
        return all_interfaces_undetermined(keys_interface, mean_tension, angles.partition)
    matrix_m, vector_b = _build_matrix_tension_cotan(angles.dict_rad, dict_areas, angles.dict_length, mean_tension)
    _require_finite_system(matrix_m, vector_b, "Cotan")
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

    (A fixed defect.) The diagonal entries used to read `-s3 - f3 * factor`, i.e.
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

    A triple line with a wedge at 0 or 180 degrees (`DEGENERATE_WEDGE_TOLERANCE`) gives this method
    infinite coefficients and is left out, with a `DegenerateWedgeWarning`
    (`_without_degenerate_wedges`); the system is never solved with a non-finite coefficient
    (`_require_finite_system` raises `ValueError` instead).

    Returns:
        ComponentTensionSolution: Normalised in each connected piece of the triple-junction graph
            when it is in several (`foambryo.junction_graph`), over the whole mesh otherwise; the
            mean-tension normalisation is met exactly either way.
    """
    angles = line_averaged_angles(mesh, reading=angle_reading, refusal=refusal)
    angles = _without_degenerate_wedges(mesh, angles, "InvCotan")
    dict_areas = mesh.compute_areas_interfaces()
    keys_interface = [tuple(key) for key in sorted(dict_areas.keys())]
    if not angles.dict_rad:
        return all_interfaces_undetermined(keys_interface, mean_tension, angles.partition)
    matrix_m, vector_b = _build_matrix_tension_inv_cotan(angles.dict_rad, dict_areas, angles.dict_length, mean_tension)
    _require_finite_system(matrix_m, vector_b, "InvCotan")
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

    A triple line with a wedge at 0 or 180 degrees (`DEGENERATE_WEDGE_TOLERANCE`) gives this method
    infinite coefficients and is left out, with a `DegenerateWedgeWarning`
    (`_without_degenerate_wedges`); the system is never solved with a non-finite coefficient
    (`_require_finite_system` raises `ValueError` instead).

    Returns:
        ComponentTensionSolution: Normalised in each connected piece of the triple-junction graph
            when it is in several (`foambryo.junction_graph`), over the whole mesh otherwise; the
            mean-tension normalisation is met exactly either way.
    """
    angles = line_averaged_angles(mesh, reading=angle_reading, refusal=refusal)
    angles = _without_degenerate_wedges(mesh, angles, "Lami")
    dict_areas = mesh.compute_areas_interfaces()
    keys_interface = [tuple(key) for key in sorted(dict_areas.keys())]
    if not angles.dict_rad:
        return all_interfaces_undetermined(keys_interface, mean_tension, angles.partition)
    matrix_m, vector_b = _build_matrix_tension_lami(angles.dict_rad, dict_areas, angles.dict_length, mean_tension)
    _require_finite_system(matrix_m, vector_b, "Lami")
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
        ComponentTensionSolution: Normalised in each connected piece of the triple-junction graph
            when it is in several (`foambryo.junction_graph`), over the whole mesh otherwise; the
            mean-tension normalisation is met exactly either way.
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

    The per-piece normalisation of the junction graph (`foambryo.junction_graph`) is applied in
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

    A triple line with a wedge at 0 or 180 degrees (`DEGENERATE_WEDGE_TOLERANCE`) gives this method
    infinite coefficients and is left out, with a `DegenerateWedgeWarning`
    (`_without_degenerate_wedges`); the system is never solved with a non-finite coefficient
    (`_require_finite_system` raises `ValueError` instead).

    Returns:
        ComponentTensionSolution: Normalised in each connected piece of the triple-junction graph
            when it is in several, over the whole mesh otherwise; the geometric mean of the tensions
            is `mean_tension` exactly, the normalisation being met in log space before the `exp`.
    """
    angles = line_averaged_angles(mesh, reading=angle_reading, refusal=refusal)
    angles = _without_degenerate_wedges(mesh, angles, "LogLami")
    dict_areas = mesh.compute_areas_interfaces()
    keys_interface = [tuple(key) for key in sorted(dict_areas.keys())]
    if not angles.dict_rad:
        return all_interfaces_undetermined(keys_interface, mean_tension, angles.partition)
    matrix_m, vector_b = _build_matrix_tension_lami_log(angles.dict_rad, dict_areas, angles.dict_length, mean_tension)
    _require_finite_system(matrix_m, vector_b, "LogLami")
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
            fields (`foambryo.junction_graph`).
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
#: (measured, not assumed).
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
    solver: str = "auto",
    sparse_threshold: int | None = None,
    diagnostics: bool | str = False,
) -> "ComponentTensionSolution | tuple[ComponentTensionSolution, SolveDiagnostics]":
    """Infer tensions with each connected piece of the triple-junction graph made explicit.

    Two interfaces are coupled only when they meet on a triple line. When the graph of these
    couplings falls into several pieces, each piece's tensions are determined only up to their own
    scale: this function normalises each piece separately (its mean tension is `mean_tension`,
    exactly) and says which interface belongs to which piece, rather than collapsing them into one
    map. See `foambryo.junction_graph`'s module docstring for the mechanism.

    Use this instead of `infer_tensions`/`infer_tensions_and_residuals` whenever a mesh might
    have a disconnected triple-junction graph (12 of the 72 simulator meshes of 91-166 cells in
    `BENCHMARKS.md` do) or a fully closed interface, and the caller needs to know *which* interfaces
    are (or are not) on a common scale before comparing or aggregating their tensions. It always reads
    the chord angles; it emits no `JunctionGraphGaugeWarning`, since the result carries what the warning would say.

    Args:
        mesh (DcelData): Mesh to analyze, in any length unit.
        mean_tension (float, optional): The normalisation, applied **per piece**: the tensions of
            each connected piece are normalised so that their mean is `mean_tension`, exactly (for
            `LogLami`, their geometric mean). Defaults to 1.
        mode (TensionComputationMethod, optional): Which method to use: one of `YoungDupre`,
            `YoungDupreLocal`, `ProjectionYoungDupre`, `Cotan`, `InvCotan`, `Lami`, `InvLami` and
            `LogLami`. Defaults to `YoungDupre`.
        refusal (WedgeRefusal, optional): which trijunction observations the estimator declines,
            and how (`foambryo.wedge_refusal`). Keyword-only; defaults to `NO_REFUSAL` (refusing
            nothing), which is bitwise the historical behaviour. This is the entry point to prefer
            under a refusal policy: refusing rows can itself split the junction graph, and this
            function hands the caller the resulting pieces and undetermined interfaces rather
            than blending them.
        solver (str, optional): `"auto"` (the default), `"dense"` or `"sparse"`: how the linear
            systems are solved (`foambryo.linear_solves`). `"auto"` is dense below
            `sparse_threshold` unknowns and sparse at or above, and bitwise the dense result
            below it. Every method accepted here has a sparse path. Keyword-only.
        sparse_threshold (int | None, optional): unknowns (interfaces of one piece) at which
            `"auto"` goes sparse; `None` (the default) is
            `foambryo.junction_graph.SPARSE_SOLVE_MIN_UNKNOWNS` (2 000). Keyword-only.
        diagnostics (bool | str, optional): `False` (the default), `True` or `"spectrum"`: also
            return a `foambryo.linear_solves.SolveDiagnostics`, last, with one record per solve
            (solver, shape, rank, residual norm, condition number, time; `"spectrum"` adds the
            extreme singular values of the sparse solves). Keyword-only.

    Returns:
        ComponentTensionSolution: `dict_tensions` maps every interface `(label_1, label_2)`,
            `label_1 < label_2`, to its relative tension, or `float("nan")` if it is undetermined
            (no surviving triple line); `component_of` maps every interface to its piece, or
            `None` if undetermined; `gauge_notes` states, per piece, the normalisation applied;
            `solver_notes` which solver ran on each piece. Tensions of different pieces are on
            different scales and must not be compared. On a cropped mesh the crop interfaces are
            absent. With `diagnostics` on, the tuple `(solution, SolveDiagnostics)`.

    Raises:
        ValueError: If `mode` is `Equilibrium`, `Variational` or a `ForceBalance*` member -- these
            methods' coupling structure is not the triple-junction graph (see
            `foambryo.junction_graph`'s "Scope: which methods"), so this decomposition does
            not apply to them; use `infer_tensions(mesh, mean_tension, mode)` instead. Also on an
            invalid solver option, and as :func:`infer_tensions` for the method itself.
        MalformedTripleLineError: as :func:`infer_tensions`.

    Warns:
        DegenerateWedgeWarning: as :func:`infer_tensions`.
        SparseSolveFallbackWarning: as :func:`infer_tensions`.
        DenseSolveMemoryWarning: as :func:`infer_tensions`.
    """
    if mode not in _JUNCTION_GRAPH_GAUGED_BUILDERS:
        message = (
            f"{mode} is not decomposed by the triple-junction graph -- its coupling "
            "structure is not the junction graph alone (see foambryo.junction_graph's "
            "'Scope: which methods'). Use infer_tensions(mesh, mean_tension, mode) instead."
        )
        raise ValueError(message)
    refusal = refusal_with_crop(mesh, refusal)
    with linear_solves.solve_options(solver, sparse_threshold, diagnostics) as collected:
        solution = _JUNCTION_GRAPH_GAUGED_BUILDERS[mode](mesh, mean_tension, refusal=refusal)
    solution = remove_crop_interfaces(solution, mesh.crop_interfaces)
    return solution if collected is None else (solution, collected)


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
            `VariationalRoute` for the choice and its evidence.

    Returns:
        tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
            - array of relative tensions
            - map interface id (label 1, label 2) -> tension on this interface
            - residuals of the least square method.
    """
    if (route or VARIATIONAL_ROUTE_DEFAULT) is VariationalRoute.Operator:
        return _infer_tension_variational_operator(mesh, mean_tension)
    if mesh.has_crop_faces:
        message = (
            "the Gram route of Variational has no formulation for crop faces; the Operator route "
            "(the default) keeps the crop out of its equations"
        )
        raise ValueError(message)

    matrix_m, _vector_b, nm, keys = _build_matrix_discrete_sparse(mesh, mean_tension)

    f_g, f_p, g_g, g_p = _extract_submatrices(matrix_m, nm)

    mat_t = f_g - f_p @ (np.linalg.inv(g_p)) @ g_g

    # Exact gauge `mean(gamma) = mean_tension` (`solve_with_exact_gauge`), on the homogeneous
    # Schur-complement rows; until 0.6.0 a soft row of ones appended to them.
    x, resid, _note = solve_with_exact_gauge(
        mat_t, np.zeros(nm), nm * mean_tension, system="Variational tension system (Gram route)", use_sparse=False,
    )

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
    if mesh.has_crop_faces:
        return _infer_tension_variational_operator_off_the_crop(mesh, mean_tension, keys, matrix_d, matrix_v)
    cells_v = matrix_v[1:]  # drop the exterior row: its pressure is the gauge, fixed at 0
    nm = len(keys)
    n_cols = nm + cells_v.shape[0]

    # M as a dense (3N, nm + nc) block: nm + nc is a few tens, so this is a thin matrix, and
    # it is assembled from the *sparse* operators rather than from dense gradient dicts.
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

    start = time.perf_counter()
    reduced = operator @ basis
    reduced_rhs = -(operator @ particular)
    y, reduced_resid, reduced_rank, reduced_sv = linalg.lstsq(reduced, reduced_rhs)
    if linear_solves.collecting_diagnostics():
        linear_solves.record_dense_lstsq(
            "Variational tension system (Operator route, gauge eliminated)", "dense lstsq",
            reduced, reduced_rhs, y, reduced_resid, reduced_rank, reduced_sv, start,
        )
    beta = particular + basis @ y
    tensions = beta[:nm]
    residual = operator @ beta

    dict_tensions = {tuple(keys[i]): tensions[i] for i in range(nm)}
    return (tensions, dict_tensions, np.array([float(residual @ residual)]))


def _householder_gauged_lstsq(operator: NDArray[np.float64], nm: int, mean_tension: float) -> NDArray[np.float64]:
    """`min ||operator beta||` subject to `mean(beta[:nm]) = mean_tension`, as route Q solves it.

    The same algebra as `_infer_tension_variational_operator`: the gauge removed by one
    Householder reflector, then an unconstrained least squares through the operator.
    """
    n_cols = operator.shape[1]
    anchor = np.zeros(n_cols)
    anchor[:nm] = 1.0 / nm
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
    return particular + basis @ y


def _infer_tension_variational_operator_off_the_crop(
    mesh: "DcelData",
    mean_tension: float,
    keys: NDArray[np.int64],
    matrix_d: "sparse.spmatrix",
    matrix_v: "sparse.spmatrix",
) -> tuple[NDArray[np.float64], dict[tuple[int, int], float], NDArray[np.float64]]:
    """Route Q on a cropped mesh: the force balances of the vertices on crop faces are left out.

    A vertex on a crop face is held by tissue that was cut away, so its three rows of
    `M = [D^T | -V^T]` are dropped (`foambryo.crop_faces`). A column left without any non-zero
    entry has no equation: a crop interface (all its vertices are on the crop), a real interface
    whose every vertex is, or a cell that has only such vertices. Those columns leave the system,
    and their tensions are `nan`. The cut cells keep their pressure columns: unknowns solved for,
    so that their force is not moved into the tensions, and not returned.

    Returns:
        The same three values as `_infer_tension_variational_operator`, over every interface (crop
        ones included, `nan`); the caller removes the crop interfaces.
    """
    rows = vertex_rows_off_the_crop(mesh)
    d_t = matrix_d.T.tocsr()[rows].toarray()
    v_t = matrix_v.T.tocsr()[rows].toarray()
    materials = [int(m) for m in mesh.materials]  # the row order of `matrix_v`
    gamma_cols = [i for i in range(len(keys)) if np.any(d_t[:, i] != 0)]
    cell_cols = [j for j, m in enumerate(materials) if m != 0 and np.any(v_t[:, j] != 0)]
    nm = len(gamma_cols)
    operator = np.hstack([d_t[:, gamma_cols], -v_t[:, cell_cols]])
    beta = _householder_gauged_lstsq(operator, nm, mean_tension)
    residual = operator @ beta
    tensions = np.full(len(keys), np.nan)
    tensions[gamma_cols] = beta[:nm]
    dict_tensions = {tuple(keys[i]): tensions[i] for i in range(len(keys))}
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

    This is the sparse assembly's architectural fix: the dense per-interface ``(N, 3)`` gradient dicts
    (an object estimated at ~9 GB at 500 cells) and the ``O(n_interfaces^2 * N)`` Python double loop
    are gone. The only dense matrices formed are the tiny ``(nm+nc) x (nm+nc)`` Gram blocks -- the
    ``3N``-scale operator is never densified. The row normalisation, pressure sign convention and
    mean-tension row of ones are byte-for-byte the same as :func:`_build_matrix_discrete`, so the
    returned matrix agrees with it to rtol=1e-10 (only the dot-product summation order differs);
    this is pinned in ``tests/test_variational_sparse_assembly.py``.

    Note: the returned ``matrix_m``'s row of ones and ``vector_b`` are kept only for parity with
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
        dict[tuple[int, int, int], float]: Map trijunction -> residue. On a cropped mesh
            (`foambryo.crop_faces`), only the junction lines the tension methods read: the
            edges that touch a crop face are left out of the line means, and a line with no
            other edge is absent.
    """
    if mesh.has_crop_faces:
        angles = line_averaged_angles(mesh, reading=angle_reading)
        dict_rad, report = angles.dict_rad, angles.report
        # A crop interface is absent from the tension map; no line left here involves it, so its
        # column is zero and the value given to it does not enter any residual.
        dict_tensions = {key: dict_tensions.get(key, 0.0) for key in mesh.compute_areas_interfaces()}
    else:
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
