r"""Per-edge Young-Dupre tension inference (the per-edge Young-Dupre work).

`TensionComputationMethod.YoungDupre` imposes the Young-Dupre balance **once per triple line,
on angles that have already been averaged along it** (`geometry.compute_angles_tri` returns
`np.mean(...)` per wedge key). Young-Dupre is a *pointwise* balance, so averaging first is valid
only when the angle varies little along the line — and a review of the code during development
measured per-line angle standard deviations of 0.7-14.1 degrees, so it does not. Averaging also
destroys any route to per-edge residuals, robust weighting or uncertainty.

This module imposes the same balance **once per geometric trijunctional edge**:

.. math::

    w_e \begin{pmatrix}
        1 & \cos\alpha^e_b & \cos\alpha^e_a \\
        \cos\alpha^e_b & 1 & \cos\alpha^e_c \\
        \cos\alpha^e_a & \cos\alpha^e_c & 1
    \end{pmatrix}
    \begin{pmatrix} \gamma_{ab} \\ \gamma_{bc} \\ \gamma_{ac} \end{pmatrix} = 0

with `a < b < c` the three regions meeting on edge `e`, :math:`\alpha^e_f` the **interior wedge
angle** in region `f` measured on that edge (the angle-convention work settled this convention; see
`geometry._wedges_at_half_edge`), and :math:`w_e = \ell_e / L` the edge's share of the mesh's
total junction length. Summing the rows of one triple line reproduces the averaged builder's
weight for that line exactly, so the two methods differ only in *where* the averaging happens:
`YoungDupre` averages the angles and then takes the cosine; `YoungDupreLocal` takes the cosine
per edge and lets least squares do the averaging. The difference is the Jensen gap of `cos`
along the line, which is second order in the angle spread — so on a line whose angle is
constant the two are identical, and they separate exactly where the angle varies along the
line, i.e. where the mesh is noisy.

**Row weighting is not a detail — it dominates the comparison, and the per-edge Young-Dupre
work measured that.** The naive choice :math:`w_e = \ell_e / L` (the literal per-edge
formulation) does
*not* keep a triple line's total influence fixed when its single row block is split into
:math:`n_e` blocks: :math:`n_e` identical rows of weight :math:`w` are equivalent to one row of
weight :math:`w\sqrt{n_e}`, so the naive choice silently reweights line `k` by
:math:`1/\sqrt{n_e}` — it *penalises well-resolved lines*. On the 42 equilibrium cases that
reweighting alone costs two orders of magnitude, more than the per-edge idea itself is worth.
Hence :class:`PerEdgeWeighting`:

- ``LinePreserving`` (the default) sets :math:`w_e = (\ell_k / L)\sqrt{\ell_e / \ell_k}`, so
  :math:`\sum_{e \in k} w_e^2 = (\ell_k / L)^2` — each line keeps exactly the total
  least-squares weight `YoungDupre` gives it, and the *only* remaining difference between the
  two methods is per-edge versus averaged angles. This is the variant that isolates the
  per-edge question; with the angles held fixed it reproduces `YoungDupre` to ~1e-14.
- ``Edge`` sets :math:`w_e = \ell_e / L`, the literal per-edge formulation, kept so the
  measurement is reproducible.
- ``Arclength`` sets :math:`w_e = \sqrt{\ell_e / L}`, which makes the objective a discretisation
  of :math:`\int |r|^2\,\mathrm{d}\ell`.

**Read this outcome before using this.** The per-edge Young-Dupre work measured
`YoungDupreLocal` to be **decisively worse** than `YoungDupre` on the 42 equilibrium cases and
marked its own hypothesis **FALSIFIED**, for a reason that turns out to be structural: at a
triple line in mechanical equilibrium the three wedge angles are fixed by the three (constant)
tensions alone, so **the true angle is constant along each line** and every bit of measured
along-line variation is error. Averaging the angles first is therefore a noise filter, while the
pointwise form rectifies that noise into a systematic bias of order
:math:`\tfrac{1}{2}\mathrm{Var}(\alpha)` through the Jensen gap of `cos`. The method is kept —
as an additional, non-default member — because its per-edge residual field is a
genuinely more informative diagnostic than the per-junction one, and because a negative result
that is reproducible is worth more than a deleted branch.

The per-edge Young-Dupre work, 2026-07-31.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from foambryo.geometry import TrijunctionEdge, compute_trijunction_edges_with_report
from foambryo.junction_graph import (
    Interface,
    JunctionGraphPartition,
    partition_from_mesh,
    partition_over_surviving_junctions,
    solve_gauged_least_squares,
)
from foambryo.second_order_angle_reading import (
    CHORD_READING,
    CHORD_READING_REPORT,
    ORIENTED_QUADRIC_READING,
    QUADRIC_READING,
    AngleReadingReport,
)
from foambryo.wedge_refusal import NO_REFUSAL, WedgeRefusal, per_edge_rows

if TYPE_CHECKING:
    from foambryo.dcel import DcelData

#: Default Huber threshold, in units of the robust scale estimate (median absolute deviation
#: of the per-row residual). 1.345 is the classical value giving 95 % efficiency against a
#: Gaussian; it is a *default*, not a tuned parameter.
DEFAULT_HUBER_DELTA = 1.345


class PerEdgeWeighting(Enum):
    """How each edge's row block is weighted. See the module docstring — this choice dominates.

    - `LinePreserving`: `w_e = (l_k / L) * sqrt(l_e / l_k)`, so each triple line `k` keeps the
      total least-squares weight `YoungDupre` gives it. **Default.**
    - `Edge`: `w_e = l_e / L`, the phase specification's literal reading. Silently reweights
      line `k` by `1 / sqrt(n_e)`, i.e. penalises well-resolved lines.
    - `Arclength`: `w_e = sqrt(l_e / L)`, making the objective a discretisation of the continuum
      `integral |r|^2 dl`.
    """

    LinePreserving = 0
    Edge = 1
    Arclength = 2


@dataclass(frozen=True)
class PerEdgeResidualField:
    """The Young-Dupre residual of every trijunctional edge, as a renderable field.

    The paper plots one residual per *junction*; this is one per *edge*, which is strictly more
    informative — it localises a bad junction to the part of the line that is bad. All arrays
    are aligned index-wise and ordered as `geometry.compute_trijunction_edges`.

    Attributes:
        vertices: `(n_edges, 2)` vertex-key pairs.
        midpoints: `(n_edges, 3)` edge midpoints, for rendering the field on the mesh.
        lengths: `(n_edges,)` edge lengths.
        regions: `(n_edges, 3)` the three region labels `a < b < c` meeting on each edge.
        residual_vectors: `(n_edges, 3)` the **unweighted** three-row Young-Dupre defect
            `[gamma_ab + gamma_bc cos a_b + gamma_ac cos a_a, ...]` at the inferred tensions.
            Unweighted because that is the physical defect; the weighted version is the
            least-squares contribution and is `residual_vectors * weights[:, None]`.
        residual_norms: `(n_edges,)` Euclidean norm of `residual_vectors`.
        weights: `(n_edges,)` the row weight `w_e` actually used in the solve, including any
            robust down-weighting.
        robust_weights: `(n_edges,)` the IRLS multiplier alone (all ones without robust
            weighting), so a caller can see which edges were down-weighted and by how much.
    """

    vertices: NDArray[np.int64]
    midpoints: NDArray[np.float64]
    lengths: NDArray[np.float64]
    regions: NDArray[np.int64]
    residual_vectors: NDArray[np.float64]
    residual_norms: NDArray[np.float64]
    weights: NDArray[np.float64]
    robust_weights: NDArray[np.float64]


@dataclass(frozen=True)
class YoungDupreLocalResult:
    """Everything the per-edge solve produces, including the diagnostics `infer_tensions` drops.

    Attributes:
        tensions: `(n_interfaces,)` inferred tensions, ordered by `sorted(dict_areas)`.
            `float("nan")` at any interface undetermined by the junction-graph gauge split
            (the junction-graph work, `foambryo.junction_graph`) -- an interface with no triple line at all.
        dict_tensions: the same, keyed by interface `(label_1, label_2)`.
        lstsq_residuals: `scipy.linalg.lstsq`'s residual output, for signature compatibility
            with the other `_infer_tension_*` functions.
        residual_field: the per-edge residual field.
        n_rows: number of rows in the assembled system, gauge row included.
        n_edges: number of geometric trijunctional edges that carried a row block.
        irls_iterations: number of IRLS passes actually run (0 without robust weighting).
        component_of: interface -> junction-graph gauge component, `None` if undetermined
            (the junction-graph work). `{key: 0 for every interface}` when the graph is not fragmented.
        gauge_notes: gauge component -> one-line description of the gauge applied to it.
        undetermined: interfaces with no triple line at all (closed surfaces).
        fragmented: whether the mesh's triple-junction graph required per-component gauging.
        n_components: number of gauged components (1 when not fragmented).
        angle_reading_report: which trijunction faces the angle reading actually covered
            (the second-order-angle-reading work). For the default `"chord"` reading this is the trivial
            complete-coverage report; for `"quadric"` it names every trijunction line whose
            angles are a mixture of the two conventions because the quadric fit was
            inadmissible on part of it. Carried on the result rather than warned about, so a
            caller who asked for the second-order reading always holds its coverage.
    """

    tensions: NDArray[np.float64]
    dict_tensions: dict[tuple[int, int], float]
    lstsq_residuals: NDArray[np.float64]
    residual_field: PerEdgeResidualField
    n_rows: int
    n_edges: int
    irls_iterations: int
    component_of: dict[tuple[int, int], int | None]
    gauge_notes: dict[int, str]
    undetermined: tuple[tuple[int, int], ...]
    fragmented: bool
    n_components: int
    angle_reading_report: AngleReadingReport = CHORD_READING_REPORT


def _edge_regions_and_angles(edge: TrijunctionEdge) -> tuple[tuple[int, int, int], tuple[float, float, float]] | None:
    """Sort one edge's three wedges into `(a, b, c)` ascending with their angles.

    Args:
        edge (TrijunctionEdge): One geometric trijunctional edge.

    Returns:
        tuple | None: `((a, b, c), (angle_a, angle_b, angle_c))` with `a < b < c` and
            `angle_f` the interior wedge angle in region `f`; `None` if the edge's three wedge
            keys do not name three distinct regions (which cannot happen on a well-formed mesh
            and is skipped rather than silently mis-assembled).
    """
    wedge_of_region = {int(key[1]): float(angle) for key, angle in zip(edge.keys, edge.angles, strict=True)}
    if len(wedge_of_region) != 3:
        return None
    a, b, c = sorted(wedge_of_region)
    return (a, b, c), (wedge_of_region[a], wedge_of_region[b], wedge_of_region[c])


def _assemble_per_edge_blocks(
    trijunction_edges: list[TrijunctionEdge],
    dict_areas: dict[tuple[int, int], float],
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64], NDArray[np.float64], list[int]]:
    """Assemble the unweighted per-edge Young-Dupre blocks.

    Args:
        trijunction_edges (list[TrijunctionEdge]): Output of `geometry.compute_trijunction_edges`.
        dict_areas (dict[tuple[int, int], float]): Interface areas — used only for the interface
            *indexing*, exactly as the averaged builder uses it.

    Returns:
        tuple: `(blocks, columns, regions, lengths, kept)` where `blocks` is `(n_kept, 3, 3)` of
            unweighted coefficients, `columns` is `(n_kept, 3)` of interface column indices
            (order `ab`, `bc`, `ac`), `regions` is `(n_kept, 3)` of `(a, b, c)`, `lengths` is
            `(n_kept,)` and `kept` indexes back into `trijunction_edges`.
    """
    keys_interface = np.array(sorted(dict_areas.keys()))
    nb_zones = np.amax(keys_interface) + 1
    reverse_map = {
        int(key[0]) * int(nb_zones) + int(key[1]): index for index, key in enumerate(keys_interface)
    }

    blocks, columns, regions, lengths, kept = [], [], [], [], []
    for index, edge in enumerate(trijunction_edges):
        sorted_wedges = _edge_regions_and_angles(edge)
        if sorted_wedges is None:
            continue
        (a, b, c), (angle_a, angle_b, angle_c) = sorted_wedges
        kept.append(index)

        # Same three projections as `_build_matrix_tension_symmetrical_yd`, per edge instead of
        # per averaged triple line: row 0 projects on the `ab` film, row 1 on `bc`, row 2 on `ac`.
        blocks.append(
            [
                [1.0, np.cos(angle_b), np.cos(angle_a)],
                [np.cos(angle_b), 1.0, np.cos(angle_c)],
                [np.cos(angle_a), np.cos(angle_c), 1.0],
            ],
        )
        columns.append(
            [
                reverse_map[min(a, b) * int(nb_zones) + max(a, b)],
                reverse_map[min(b, c) * int(nb_zones) + max(b, c)],
                reverse_map[min(a, c) * int(nb_zones) + max(a, c)],
            ],
        )
        regions.append([a, b, c])
        lengths.append(edge.length)

    return (
        np.array(blocks, dtype=np.float64).reshape(-1, 3, 3),
        np.array(columns, dtype=np.int64).reshape(-1, 3),
        np.array(regions, dtype=np.int64).reshape(-1, 3),
        np.array(lengths, dtype=np.float64),
        kept,
    )


def build_matrix_tension_local_yd(
    blocks: NDArray[np.float64],
    columns: NDArray[np.int64],
    weights: NDArray[np.float64],
    n_interfaces: int,
    mean_tension: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Scatter the weighted per-edge blocks into `M`, and append the soft gauge row.

    The gauge row is the same soft least-squares row every other builder in
    `foambryo.tension_inference` uses (`matrix_m[-1, :] = 1`, `vector_b[-1] = nm * mean_tension`),
    so `YoungDupreLocal` is comparable with `YoungDupre` row-normalisation for row-normalisation.
    The force-balance work measured that the soft row, hard KKT and Bayesian MAP are indistinguishable in
    accuracy once gauge-matched, so nothing here turns on that choice.

    Args:
        blocks (NDArray): `(n_edges, 3, 3)` unweighted coefficient blocks.
        columns (NDArray): `(n_edges, 3)` interface column indices.
        weights (NDArray): `(n_edges,)` row weights.
        n_interfaces (int): Number of interfaces (columns of `M`).
        mean_tension (float, optional): Expected mean tension. Defaults to 1.

    Returns:
        tuple[NDArray, NDArray]: `M` of shape `(3 * n_edges + 1, n_interfaces)` and `b`.
    """
    n_edges = len(blocks)
    matrix_m = np.zeros((3 * n_edges + 1, n_interfaces))
    vector_b = np.zeros(3 * n_edges + 1)
    vector_b[-1] = n_interfaces * mean_tension

    weighted = blocks * weights[:, None, None]
    block_first_row = np.arange(n_edges) * 3
    for local_row in range(3):
        # `np.add.at`, not assignment: two of an edge's three films can in principle be the same
        # interface (a degenerate mesh), and adding is the correct behaviour there where
        # assignment would silently drop a term.
        np.add.at(matrix_m, ((block_first_row + local_row)[:, None], columns), weighted[:, local_row, :])

    matrix_m[-1, :] = 1
    return matrix_m, vector_b


def _row_weights(
    weighting: PerEdgeWeighting,
    lengths: NDArray[np.float64],
    regions: NDArray[np.int64],
    total_length: float,
) -> NDArray[np.float64]:
    """Row weight per edge under the chosen scheme. See :class:`PerEdgeWeighting`.

    Args:
        weighting (PerEdgeWeighting): scheme.
        lengths (NDArray): `(n_edges,)` edge lengths.
        regions (NDArray): `(n_edges, 3)` region triple, used to group edges into triple lines.
        total_length (float): total junction length `L`.

    Returns:
        NDArray[np.float64]: `(n_edges,)` weights.
    """
    if weighting is PerEdgeWeighting.Edge:
        return lengths / total_length
    if weighting is PerEdgeWeighting.Arclength:
        return np.sqrt(lengths / total_length)

    # LinePreserving: sum of w_e^2 over each line must equal (l_k / L)^2.
    _, line_index = np.unique(regions, axis=0, return_inverse=True)
    line_index = line_index.ravel()
    line_lengths = np.bincount(line_index, weights=lengths)
    return (line_lengths[line_index] / total_length) * np.sqrt(lengths / line_lengths[line_index])


def _huber_weights(residual_norms: NDArray[np.float64], delta: float) -> NDArray[np.float64]:
    """Huber IRLS multipliers from per-edge residual norms.

    The scale is the median absolute deviation of the residual norms about their median,
    rescaled by 1.4826 to be a consistent estimator of a Gaussian standard deviation. A
    degenerate scale (all residuals equal, e.g. an exact phantom) yields all-ones, i.e. robust
    weighting switches itself off rather than dividing by zero.

    Args:
        residual_norms (NDArray): `(n_edges,)` per-edge residual norms.
        delta (float): Huber threshold in units of the robust scale.

    Returns:
        NDArray[np.float64]: `(n_edges,)` multipliers in `(0, 1]`.
    """
    median = np.median(residual_norms)
    scale = 1.4826 * np.median(np.abs(residual_norms - median))
    if not np.isfinite(scale) or scale <= 0:
        return np.ones_like(residual_norms)
    standardised = np.abs(residual_norms - median) / scale
    return np.where(standardised <= delta, 1.0, delta / np.maximum(standardised, 1e-300))


def _undetermined_result(
    keys_interface: list[Interface],
    mean_tension: float,
    partition: JunctionGraphPartition,
    angle_reading_report: AngleReadingReport,
) -> YoungDupreLocalResult:
    """The result when a refusal policy leaves this mesh no row at all: every interface `nan`.

    Built through `junction_graph.solve_gauged_least_squares` on the gauge row alone, so the
    undetermined interfaces come from the same machinery that already reports a closed surface as
    undetermined, and `undetermined` / `component_of` / `gauge_notes` mean exactly what they
    always mean. The residual field is empty rather than absent, so a caller can iterate it
    without a special case.

    Args:
        keys_interface (list[Interface]): interface key per column, in order.
        mean_tension (float): the gauge value the caller requested.
        partition (JunctionGraphPartition): the partition over the surviving lines — with none
            surviving, every interface is isolated.
        angle_reading_report (AngleReadingReport): the reading's coverage report, carried through.

    Returns:
        YoungDupreLocalResult: `float("nan")` at every interface, `n_edges` zero.
    """
    from foambryo.wedge_refusal import all_interfaces_undetermined

    solution = all_interfaces_undetermined(keys_interface, mean_tension, partition)
    empty_field = PerEdgeResidualField(
        vertices=np.zeros((0, 2), dtype=np.int64),
        midpoints=np.zeros((0, 3)),
        lengths=np.zeros(0),
        regions=np.zeros((0, 3), dtype=np.int64),
        residual_vectors=np.zeros((0, 3)),
        residual_norms=np.zeros(0),
        weights=np.zeros(0),
        robust_weights=np.zeros(0),
    )
    return YoungDupreLocalResult(
        tensions=solution.tensions,
        dict_tensions=solution.dict_tensions,
        lstsq_residuals=solution.lstsq_residuals,
        residual_field=empty_field,
        n_rows=1,
        n_edges=0,
        irls_iterations=0,
        component_of=solution.component_of,
        gauge_notes=solution.gauge_notes,
        undetermined=solution.undetermined,
        fragmented=solution.fragmented,
        n_components=solution.n_components,
        angle_reading_report=angle_reading_report,
    )


def infer_tensions_young_dupre_local(
    mesh: DcelData,
    mean_tension: float = 1.0,
    huber_delta: float | None = None,
    max_irls_iterations: int = 60,
    weighting: PerEdgeWeighting = PerEdgeWeighting.LinePreserving,
    irls_tolerance: float = 1e-10,
    *,
    angle_reading: str = CHORD_READING,
    refusal: WedgeRefusal = NO_REFUSAL,
) -> YoungDupreLocalResult:
    """Infer tensions with one Young-Dupre block per trijunctional edge.

    Args:
        mesh (DcelData): Mesh to analyze.
        mean_tension (float, optional): Expected mean tension. Defaults to 1.
        huber_delta (float | None, optional): Huber threshold, in units of the robust scale, for
            IRLS down-weighting of outlying edges. `None` (the default) disables robust weighting
            entirely, which is what `TensionComputationMethod.YoungDupreLocal` uses — the robust
            variant is an option, not the shipped behaviour. Pass `DEFAULT_HUBER_DELTA` for the
            classical 1.345.
        max_irls_iterations (int, optional): IRLS iteration cap. Defaults to 60 — the per-edge
            Young-Dupre work measured convergence to `irls_tolerance` at 45-50 iterations on the
            dataset meshes, and the answer at a 20-iteration cap already agrees with the converged
            one to ~1e-6.
        weighting (PerEdgeWeighting, optional): row-weighting scheme. Defaults to
            `LinePreserving`, the only one that keeps each triple line's total influence equal to
            `YoungDupre`'s — see the module docstring, this choice is worth two orders of
            magnitude.
        irls_tolerance (float, optional): Stop IRLS when the max tension change falls below this.
        angle_reading (str, optional): which trijunction angle convention to read, `"chord"`
            (the default, and bit-identical to the historical behaviour) or `"oriented_chord"`.
            Keyword-only. See `foambryo.second_order_angle_reading`.
        refusal (WedgeRefusal, optional): which trijunctional edges this estimator declines to
            use, and how. Keyword-only; defaults to refusing nothing, which is bitwise the
            historical behaviour. **This method's observation unit is a single edge**, so
            `ExcludeEdgeFromLineMean` and `DropLine` are the same intervention here and agree
            exactly, while `DownweightLine` keeps the folded rows and scales their whole line.
            Refusing rows can leave an interface with no surviving line: it is then reported
            `float("nan")`, never a number. See `foambryo.wedge_refusal`.

    Returns:
        YoungDupreLocalResult: tensions plus the per-edge residual field and solve diagnostics.
            Every interface is `float("nan")`, with an empty residual field, when the refusal
            leaves no row at all — an empty answer rather than an exception, so a caller can
            count how often a policy costs a whole mesh.

    Raises:
        ValueError: if `angle_reading` is `"quadric"` or `"oriented_quadric"` (the
            quadric-safeguard work found that the per-edge balance never performs the
            along-line average that cancels the quadric
            correction's per-face scatter, so it eats the scatter directly — +0.0290 relative
            RMS tension error at `md=3`, +0.97 at `md=4`).
        ValueError: if the mesh has no trijunctional edge at all, which would leave the system
            with nothing but the gauge row.
    """
    if angle_reading in (QUADRIC_READING, ORIENTED_QUADRIC_READING):
        message = (
            f"angle_reading={angle_reading!r} cannot be combined with "
            "infer_tensions_young_dupre_local (TensionComputationMethod.YoungDupreLocal). The "
            "per-edge balance never performs the along-line average that cancels the quadric "
            "correction's per-face scatter, so it eats the scatter directly instead: relative "
            "RMS tension error was measured at +0.0290 at md=3 and +0.97 at md=4. That is a "
            "property of the quadric tangent fit, so it applies to the oriented quadric reading "
            "for the same reason. Use a quadric reading with a line-averaged method instead — "
            "TensionComputationMethod.YoungDupre, ProjectionYoungDupre, Lami, InvLami, or "
            "LogLami — or keep angle_reading='chord' or 'oriented_chord' with YoungDupreLocal."
        )
        raise ValueError(message)
    dict_areas = mesh.compute_areas_interfaces()
    trijunction_edges, angle_reading_report = compute_trijunction_edges_with_report(mesh, reading=angle_reading)
    row_scale = None
    if refusal.refuses_nothing:
        selected = trijunction_edges
    else:
        rows = per_edge_rows(mesh, trijunction_edges, refusal)
        selected = [trijunction_edges[index] for index in rows.keep]
        row_scale = np.asarray(rows.weight_scale, dtype=np.float64)
    blocks, columns, regions, lengths, kept = _assemble_per_edge_blocks(selected, dict_areas)

    keys_interface = [tuple(int(label) for label in key) for key in sorted(dict_areas.keys())]
    n_interfaces = len(keys_interface)

    if len(blocks) == 0:
        if refusal.refuses_nothing:
            msg = "No trijunctional edge found: the per-edge Young-Dupre system would be empty."
            raise ValueError(msg)
        empty_partition = partition_over_surviving_junctions(mesh.f[:, :3], mesh.f[:, 3:5], ())
        return _undetermined_result(keys_interface, mean_tension, empty_partition, angle_reading_report)

    if refusal.refuses_nothing:
        partition = partition_from_mesh(mesh)
    else:
        surviving = {tuple(int(label) for label in row) for row in regions}
        partition = partition_over_surviving_junctions(mesh.f[:, :3], mesh.f[:, 3:5], surviving)

    total_length = float(lengths.sum())
    base_weights = _row_weights(weighting, lengths, regions, total_length)
    if row_scale is not None:
        base_weights = base_weights * row_scale[kept]

    robust_weights = np.ones_like(base_weights)
    tensions = np.zeros(n_interfaces)
    lstsq_residuals = np.zeros(0)
    solution = None
    iterations = 0
    while True:
        matrix_m, vector_b = build_matrix_tension_local_yd(
            blocks,
            columns,
            base_weights * robust_weights,
            n_interfaces,
            mean_tension,
        )
        # The junction-graph work (`foambryo.junction_graph`): bit-identical to the pre-existing direct
        # `linalg.lstsq(matrix_m, vector_b)` call whenever the mesh's triple-junction graph
        # is not fragmented; the partition is fixed for the whole IRLS loop (it depends only
        # on mesh topology, not on the current robust weights).
        solution = solve_gauged_least_squares(matrix_m, vector_b, keys_interface, mean_tension, partition)
        new_tensions, lstsq_residuals = solution.tensions, solution.lstsq_residuals
        converged = iterations > 0 and np.nanmax(np.abs(new_tensions - tensions)) < irls_tolerance
        tensions = new_tensions
        if huber_delta is None or converged or iterations >= max_irls_iterations:
            break
        residual_norms = np.linalg.norm(np.einsum("eij,ej->ei", blocks, tensions[columns]), axis=1)
        robust_weights = _huber_weights(residual_norms, huber_delta)
        iterations += 1

    residual_vectors = np.einsum("eij,ej->ei", blocks, tensions[columns])
    kept_edges = [selected[index] for index in kept]
    midpoints = np.array(
        [0.5 * (mesh.v[edge.vertices[0]] + mesh.v[edge.vertices[1]]) for edge in kept_edges],
    ).reshape(-1, 3)
    vertices = np.array([edge.vertices for edge in kept_edges], dtype=np.int64).reshape(-1, 2)

    residual_field = PerEdgeResidualField(
        vertices=vertices,
        midpoints=midpoints,
        lengths=lengths,
        regions=regions,
        residual_vectors=residual_vectors,
        residual_norms=np.linalg.norm(residual_vectors, axis=1),
        weights=base_weights * robust_weights,
        robust_weights=robust_weights,
    )

    return YoungDupreLocalResult(
        tensions=tensions,
        dict_tensions=solution.dict_tensions,
        lstsq_residuals=lstsq_residuals,
        residual_field=residual_field,
        n_rows=3 * len(blocks) + 1,
        n_edges=len(blocks),
        irls_iterations=iterations,
        component_of=solution.component_of,
        gauge_notes=solution.gauge_notes,
        undetermined=solution.undetermined,
        fragmented=solution.fragmented,
        n_components=solution.n_components,
        angle_reading_report=angle_reading_report,
    )
