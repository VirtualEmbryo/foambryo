r"""The junction graph of a mesh, and the tension solves normalised in each of its connected pieces.

**The normalisation (the "gauge").** A mesh determines tension *ratios* only, so every tension
solve adds one condition that fixes the overall scale: the tensions are normalised so that their
mean is `mean_tension`, exactly, in each connected piece of the junction graph. This module calls
that condition the gauge, and a connected piece of the graph a component.

**Why per piece.** A Young-Dupre-family solver couples two interfaces only when they share a
triple line. Build the graph with interfaces as nodes and an edge wherever two interfaces meet at
a triple line (`foambryo.validation.mesh_io.list_junctions`). If that graph has `k` connected
components, the tension system decomposes into `k` blocks each determined only up to **its own
scale**, while `mean(gamma) = mean_tension` supplies a single global constraint. Ordinary least
squares then returns the minimum-norm solution, which drives the `k - 1` unpinned blocks towards
zero -- giving a gauge-matched relative error of exactly 1.0, silently. This was found on 12 of the
72 ground-truth simulator meshes of 91-166 cells described in `BENCHMARKS.md`.

**A degenerate sub-case.** An interface that is a fully closed surface (an engulfed cell, a
sealed lumen) has no triple line at all, so it is an isolated node in *no* junction-graph
edge and *no* equation in any builder's physics rows. Its tension is not merely unpinned in
scale -- it is wholly undetermined, and this module reports it as such (`float("nan")`),
never as a number, never silently folded into a component.

**The design.** Infer per component, normalised per component: each block's tensions are
internally consistent and comparable only *within* the block. That is surfaced in the output, not
buried in a docstring -- `ComponentTensionSolution` carries, for every interface, which component
it belongs to and which normalisation was applied to that component; isolated/closed interfaces
are reported as undetermined, never as a component with a normalisation and never as a number;
and any code that would aggregate tensions *across* components must refuse rather than silently
blend two incompatible scales (`require_single_component_solution` below).

**The normalisation is exact** (since 0.7.0). The builders still append a row of ones, `sum(gamma) =
n * f(mean_tension)`; `solve_gauged_least_squares` reads its value and replaces the row by
`solve_with_exact_gauge`: a gauge row weighted to the physics rows' median norm (the
best-conditioned weight), then an exact rescale (homogeneous rows) or shift (difference rows), so
that each component's tensions meet their gauge exactly. Until 0.6.0 the row of ones was a soft
least-squares row of weight 1. Because the physics rows are homogeneous, its weight changed only the
overall scale of the solution, never a tension ratio, but it missed the requested mean (by ~1e-6 for
`YoungDupre`, up to 80 % for `Equilibrium`) and raised the condition number by up to three orders of
magnitude (2.8e7 against 311 on a 1 866-cell tissue's `YoungDupre` system), which is what limited
the iterative solve's accuracy to ~1e-6 of the tension scale.

**Scope: which methods.** This module decomposes any solver whose assembled system is
`[physics rows; one trailing global gauge row `sum(gamma) = n * f(mean_tension)`]`, which is
every method in `foambryo.tension_inference` and `foambryo.per_edge_tension_inference`
**except**:

- `TensionComputationMethod.Equilibrium`, whose physics rows are keyed by *geometric mesh
  edges* (`len(edge.twin) != 1`), which include quadrijunction edges that
  `mesh_io.list_junctions` does not enumerate (it only finds edges bordering exactly three
  regions) -- so this module's triple-line-only graph would not necessarily match
  `Equilibrium`'s actual row coupling. A known limitation: `Equilibrium` is normalised over the
  whole mesh at once, never decomposed;
- the `ForceBalance*` and `Variational` family, whose operator additionally carries area
  *and* volume gradients tied to shared cells and is column-coupled through the pressure
  block rather than through the triple-junction graph alone -- measured not to fragment the
  same way, so these are normalised over the whole mesh at once and never decomposed here.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse
from scipy.sparse.linalg import lsqr

from foambryo import linear_solves
from foambryo.validation import mesh_io

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

Interface = tuple[int, int]

#: A least-squares solve with at least this many unknowns (interfaces in one junction-graph
#: component, or in the whole mesh when it is not fragmented) runs the sparse iterative `lsqr`
#: instead of the dense `scipy.linalg.lstsq`. Below it the dense solve runs unchanged, so
#: every result below the threshold is bit-identical to the dense code path. The dense solve's cost
#: grows as the cube of the number of unknowns: about 1 s at 2 000, 3.5 s at 3 040, 29 s at 6 080
#: and 230 s (25 GiB) at 11 742 (a 1 866-cell tissue). At 2 000 the sparse path could save at most
#: about a second, so the exact dense solution is kept wherever it is cheap; every benchmark,
#: embryo and phantom mesh of this package's test data is far below it. Above the threshold `lsqr`
#: reproduces the dense solution to within its stopping tolerance, and falls back to the dense solve
#: when it does not converge. The same default threshold decides the pressure solves
#: (`foambryo.pressure_inference`); every public entry point can override it (`sparse_threshold`)
#: or force a path (`solver`), see `foambryo.linear_solves`.
SPARSE_SOLVE_MIN_UNKNOWNS = 2000

#: `lsqr`'s stopping tolerances (`atol`, `btol`): the relative residual and normal-equation
#: residual at which it stops. With `conlim = 0` it never stops on its condition estimate, which
#: on a 1 866-cell tissue's system (condition estimate 1e8) otherwise stops it far from the
#: solution.
_LSQR_TOLERANCE = 1e-14

#: `lsqr`'s iteration limit, as a multiple of the number of unknowns. A 1 866-cell tissue's system
#: of 11 742 unknowns converges in about 1 650 iterations; reaching the limit is a failure to
#: converge and triggers the dense fallback.
_LSQR_ITERATIONS_PER_UNKNOWN = 10

#: `lsqr` stop codes that mean it converged: 1 and 4 (a solution of `A x = b` to the tolerance or to
#: machine precision), 2 and 5 (a least-squares solution to the tolerance or to machine precision).
#: 3 and 6 (condition limit) cannot occur with `conlim = 0`; 7 is the iteration limit. 0 is excluded
#: although it can mean `b = 0` (then `x = 0` is exact), because `lsqr` also returns it when it ran
#: no iteration at all; the gauge row makes `b` non-zero for every tension system, and a zero `b`
#: would only cost the dense fallback.
_LSQR_CONVERGED = frozenset({1, 2, 4, 5})


class SparseSolveFallbackWarning(UserWarning):
    """`lsqr` did not converge on a large linear system.

    The dense least-squares solve was run instead; or, when the dense solve would not fit in the
    memory available, `lsqr`'s last iterate is returned and the message says it is not converged.
    """


def _solve_least_squares(
    matrix: NDArray[np.float64] | sparse.csr_matrix,
    vector: NDArray[np.float64],
    *,
    system: str = "tension system",
    use_sparse: bool | None = None,
    scale_columns: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, object]]:
    """Least-squares solution of `matrix @ x = vector`, dense or sparse as the solver options say.

    The path is chosen by `foambryo.linear_solves.use_sparse_solver` from the options in force
    (`solver`, `sparse_threshold`; by default dense below `SPARSE_SOLVE_MIN_UNKNOWNS` columns and
    sparse at or above), unless `use_sparse` forces it. The dense path is exactly
    `scipy.linalg.lstsq(matrix, vector)` (bit-identical: no other floating-point operation is
    performed on the solution). The sparse path runs `lsqr` on the matrix in compressed sparse row
    form with `atol = btol = _LSQR_TOLERANCE`, `conlim = 0` and an iteration limit of
    `_LSQR_ITERATIONS_PER_UNKNOWN` per unknown. If `lsqr` does not converge (stop code not in
    `_LSQR_CONVERGED`), the dense solve runs instead, a `SparseSolveFallbackWarning` is emitted and
    the returned record says so; unless the dense solve would not fit in the memory available
    (`foambryo.linear_solves.available_memory_bytes`): then `lsqr`'s last iterate is returned, the
    warning says it is not converged and why the fallback was not run, and the record's solver is
    `"sparse lsqr, not converged (dense fallback too large for memory)"`. With `solver="dense"`
    forced, a `DenseSolveMemoryWarning` is emitted first when the dense solve would not fit in
    memory.

    With `scale_columns`, the sparse path solves for `y` with every column of the matrix divided by
    its 2-norm (`x = y / norms`; a zero column is left as it is): the same least-squares solution,
    with a condition number that can be orders of magnitude smaller. `lsqr`'s error grows with the
    condition number, so this decides its accuracy on a system whose columns differ widely in norm:
    on a 1 866-cell tissue's weighted-Laplace system it took the 2-norm condition number from
    1.9e5 to 6.0, the iterations from 1 649 to 72, and the distance to an SVD solution from 1.2e-8 to
    1.7e-13 of the pressure scale. It does not help the tension systems, whose conditioning comes
    from their gauge row, not from their columns. The dense path ignores it.

    With diagnostics on (`foambryo.linear_solves`), a `SolveRecord` is appended to them and the
    returned record gains `n_rows`, `rank`, `residual_norm`, `condition_number`,
    `condition_number_source` and `seconds`; without, it gains none of them.

    Args:
        matrix: `(n_rows, n_unknowns)` system, dense or `scipy.sparse`.
        vector: `(n_rows,)` right-hand side.
        system: what the system is, for messages and diagnostics. Keyword-only.
        use_sparse: force the sparse (`True`) or dense (`False`) path; `None` (the default)
            follows the options in force. Keyword-only.
        scale_columns: on the sparse path, solve the column-scaled system (above). Keyword-only.

    Returns:
        tuple: `(x, residuals, record)`.
            - `x`: `(n_unknowns,)` solution.
            - `residuals`: as `scipy.linalg.lstsq` reports it: the squared 2-norm of `vector - matrix
              @ x`, shape `(1,)`, or shape `(0,)` when the dense solver returns none (rank-deficient
              or not overdetermined). The sparse path always reports the squared residual.
            - `record`: `solver` (`"dense lstsq"`, `"sparse lsqr"`, or `"dense lstsq after lsqr did
              not converge"`), `n_unknowns`, and for the sparse path the stop code `lsqr_istop`, the
              iteration count `lsqr_iterations`, the residual norm `lsqr_residual_norm` and the
              condition estimate `lsqr_condition_estimate`.
    """
    n_rows, n_unknowns = matrix.shape
    if use_sparse is None:
        use_sparse = linear_solves.use_sparse_solver(n_unknowns)
    diagnose = linear_solves.collecting_diagnostics()
    start = time.perf_counter()
    if not use_sparse:
        linear_solves.warn_if_forced_dense_exceeds_memory(n_rows, n_unknowns, system, stacklevel=4)
        dense = matrix.toarray() if sparse.issparse(matrix) else matrix
        x, residuals, rank, singular_values = linalg.lstsq(dense, vector)
        record: dict[str, object] = {"solver": "dense lstsq", "n_unknowns": int(n_unknowns)}
        if diagnose:
            _diagnose_dense(record, system, dense, vector, x, residuals, rank, singular_values, start)
        return x, residuals, record

    operator = matrix if sparse.issparse(matrix) else sparse.csr_matrix(matrix)
    column_scale = None
    if scale_columns:
        norms = np.sqrt(np.asarray(operator.multiply(operator).sum(axis=0)).ravel())
        column_scale = np.where(norms > 0, 1.0 / np.where(norms > 0, norms, 1.0), 1.0)
    result = lsqr(
        operator if column_scale is None else (operator @ sparse.diags(column_scale)).tocsr(),
        vector,
        atol=_LSQR_TOLERANCE,
        btol=_LSQR_TOLERANCE,
        conlim=0,
        iter_lim=_LSQR_ITERATIONS_PER_UNKNOWN * n_unknowns,
    )
    x, istop, iterations, residual_norm = result[0], int(result[1]), int(result[2]), float(result[3])
    if column_scale is not None:
        x = x * column_scale
    record = {
        "solver": "sparse lsqr",
        "n_unknowns": int(n_unknowns),
        "lsqr_istop": istop,
        "lsqr_iterations": iterations,
        "lsqr_residual_norm": residual_norm,
        "lsqr_condition_estimate": float(result[6]),
    }
    if column_scale is not None:
        record["lsqr_columns_scaled"] = True
    if istop not in _LSQR_CONVERGED:
        return _after_lsqr_did_not_converge(operator, vector, x, record, system, diagnose, start)
    residual = vector - operator @ x
    if diagnose:
        _diagnose_lsqr(record, system, operator, float(np.sqrt(residual @ residual)), start)
    return x, np.array([float(residual @ residual)]), record


def _after_lsqr_did_not_converge(
    operator: sparse.csr_matrix,
    vector: NDArray[np.float64],
    x: NDArray[np.float64],
    record: dict[str, object],
    system: str,
    diagnose: bool,
    start: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, object]]:
    """The dense fallback of `_solve_least_squares`, or `lsqr`'s last iterate if it cannot fit in memory."""
    n_rows, n_unknowns = operator.shape
    istop, iterations = record["lsqr_istop"], record["lsqr_iterations"]
    needed = linear_solves.dense_solve_bytes(n_rows, n_unknowns)
    available = linear_solves.available_memory_bytes()
    if available is not None and needed > available:
        residual = vector - operator @ x
        message = (
            f"lsqr stopped with code {istop} after {iterations} iterations on a {system} of "
            f"{n_unknowns} unknowns without converging, and the dense fallback was not run: it needs "
            f"about {needed / 2**30:.1f} GiB, more than the {available / 2**30:.1f} GiB available. The "
            f"returned solution is lsqr's last iterate (residual norm {np.sqrt(residual @ residual):.3e}), "
            "not a converged least-squares solution."
        )
        warnings.warn(message, SparseSolveFallbackWarning, stacklevel=4)
        record["solver"] = "sparse lsqr, not converged (dense fallback too large for memory)"
        if diagnose:
            _diagnose_lsqr(record, system, operator, float(np.sqrt(residual @ residual)), start)
        return x, np.array([float(residual @ residual)]), record
    message = (
        f"lsqr stopped with code {istop} after {iterations} iterations on a {system} of "
        f"{n_unknowns} unknowns without converging; the dense least-squares solve was run instead."
    )
    warnings.warn(message, SparseSolveFallbackWarning, stacklevel=4)
    dense = operator.toarray()
    x, residuals, rank, singular_values = linalg.lstsq(dense, vector)
    record["solver"] = "dense lstsq after lsqr did not converge"
    if diagnose:
        _diagnose_dense(record, system, dense, vector, x, residuals, rank, singular_values, start)
    return x, residuals, record


def _diagnose_dense(
    record: dict[str, object],
    system: str,
    dense: NDArray[np.float64],
    vector: NDArray[np.float64],
    x: NDArray[np.float64],
    residuals: NDArray[np.float64],
    rank: int,
    singular_values: NDArray[np.float64],
    start: float,
) -> None:
    """Fill `record` and append a `SolveRecord` for a dense `lstsq` solve. Reads, never alters, `x`."""
    record.update(linear_solves.record_dense_lstsq(
        system, str(record["solver"]), dense, vector, x, residuals, rank, singular_values, start,
        lsqr_istop=record.get("lsqr_istop"),  # type: ignore[arg-type]
        lsqr_iterations=record.get("lsqr_iterations"),  # type: ignore[arg-type]
    ))


def _diagnose_lsqr(
    record: dict[str, object],
    system: str,
    operator: sparse.csr_matrix,
    residual_norm: float,
    start: float,
) -> None:
    """Fill `record` and append a `SolveRecord` for a converged `lsqr` solve (spectrum if asked)."""
    singular_values = None
    spectrum_seconds = None
    spectrum_note = None
    if linear_solves.spectrum_requested():
        singular_values, spectrum_seconds, spectrum_note = linear_solves.extreme_singular_values(operator)
    seconds = time.perf_counter() - start
    condition = float(record["lsqr_condition_estimate"])  # type: ignore[arg-type]
    source = "lsqr estimate (Frobenius norm)"
    if record.get("lsqr_columns_scaled"):
        source = "lsqr estimate (Frobenius norm) of the column-scaled system"
    record.update({
        "n_rows": int(operator.shape[0]),
        "rank": None,
        "residual_norm": residual_norm,
        "condition_number": condition,
        "condition_number_source": source,
        "seconds": seconds,
    })
    linear_solves.record_solve(linear_solves.SolveRecord(
        system=system,
        solver=str(record["solver"]),
        n_rows=int(operator.shape[0]),
        n_unknowns=int(operator.shape[1]),
        rank=None,
        residual_norm=residual_norm,
        condition_number=condition,
        condition_number_source=source,
        seconds=seconds,
        lsqr_istop=int(record["lsqr_istop"]),  # type: ignore[arg-type]
        lsqr_iterations=int(record["lsqr_iterations"]),  # type: ignore[arg-type]
        spectrum_seconds=spectrum_seconds,
        spectrum_note=spectrum_note,
        singular_values=singular_values,
    ))


class MalformedTripleLineError(RuntimeError):
    """A triple-line edge names an interface pair absent from the mesh's own interface list.

    Structurally, this should not be reachable after `mesh_io.list_junctions`'s stricter
    admission rule: a genuine triple line's three sub-interfaces are, by construction, its
    three incident triangles' own region pairs (see that function's docstring). This is a
    named, catchable safety net standing in for what used to be a bare `KeyError` naming an
    internal union-find dict key -- for any future change to `list_junctions`'s guarantee, or
    any caller that builds `junctions`/`edge_to_triple` some other way.

    Attributes:
        edge: The mesh edge `(u, v)`, `u < v`, that synthesized `triple` -- `None` if no
            edge in `edge_to_triple` could be matched to it.
        triple: The region-triple `(a, b, c)` the edge synthesized.
        missing_interface: The interface pair from `triple` that has no key in the
            union-find parent map, i.e. does not occur as any triangle's actual region pair.
    """

    def __init__(
        self,
        edge: tuple[int, int] | None,
        triple: tuple[int, int, int] | None,
        missing_interface: Interface,
    ) -> None:
        """Build the error from the offending edge, its synthesized triple, and the missing interface."""
        self.edge = edge
        self.triple = triple
        self.missing_interface = missing_interface
        message = (
            f"triple-line edge {edge} synthesizes region-triple {triple}, whose interface "
            f"{missing_interface} does not occur as any triangle's actual region pair in this "
            "mesh. Refusing to guess a tension for an interface that does not exist, rather "
            "than raising a bare KeyError."
        )
        super().__init__(message)


@dataclass(frozen=True)
class JunctionGraphPartition:
    """Connected-component partition of a mesh's interfaces by its triple-junction graph.

    Attributes:
        interfaces: Every interface present on the mesh (`mesh_io.list_interfaces`).
        components: One tuple of interface keys per connected component, ordered largest
            first (ties broken by the component's smallest key); a mesh with a fully
            connected junction graph has exactly one component containing every touched
            interface.
        component_of: Map interface -> component index, for every interface **except** the
            isolated ones (which are never assigned a component -- see `isolated`).
        isolated: Interfaces touching no triple line at all (closed surfaces). Disjoint from
            every entry of `components`.
    """

    interfaces: tuple[Interface, ...]
    components: tuple[tuple[Interface, ...], ...]
    component_of: dict[Interface, int]
    isolated: tuple[Interface, ...]

    @property
    def n_components(self) -> int:
        """Number of proper (gaugeable) components -- excludes isolated interfaces."""
        return len(self.components)

    @property
    def fragmented(self) -> bool:
        """True iff a single global `mean(gamma) = mean_tension` normalisation no longer pins one scale.

        This is the case whenever there is more than one component, **or** at least one
        isolated interface -- an isolated interface is already a failure of the single-gauge
        assumption even if the rest of the mesh's junction graph is fully connected.
        """
        return self.n_components > 1 or bool(self.isolated)


class JunctionGraphGaugeWarning(UserWarning):
    """Emitted when a mesh's triple-junction graph is in several pieces or has an undetermined interface.

    Fires whenever `JunctionGraphPartition.fragmented` is true: more than one component (a
    connected piece of the junction graph, normalised on its own), at least one undetermined
    interface (no surviving triple line, e.g. a fully closed surface), or both. Tensions in
    different components sit on different, arbitrary relative scales -- comparing or averaging
    them directly is meaningless -- and an undetermined interface is reported as `nan`, never a
    number. Emitted by `infer_tensions`, `infer_tensions_and_residuals`, `infer_forces` and
    `infer_tensions_over_seeds`, whose flat maps hold the tensions of every component side by
    side; use `foambryo.tension_inference.infer_tensions_by_junction_component` to get each
    interface's component explicitly instead of blending them.
    """


def warn_on_junction_graph_fragmentation(
    fragmented: bool,
    n_components: int,
    isolated: tuple[Interface, ...],
    stacklevel: int = 3,
) -> None:
    """Emit `JunctionGraphGaugeWarning` when a mesh's junction-graph gauging was fragmented.

    Called only from the entry points that drop the underlying gauge structure on the floor --
    mirrors `second_order_angle_reading.warn_on_partial_coverage`'s precedent: the
    report/solution-returning entry point (`infer_tensions_by_junction_component`) hands the
    caller the `ComponentTensionSolution` this warning would have described, and stays
    silent; `infer_tensions_and_residuals` (and, through it, `infer_tensions`) does not, so
    it calls this instead.

    Takes the three scalars directly, rather than a `JunctionGraphPartition`, so a caller that
    already solved the system (and therefore already has a `ComponentTensionSolution` or
    `YoungDupreLocalResult` carrying these same three fields) does not need to recompute
    `partition_from_mesh` a second time just to decide whether to warn.

    Args:
        fragmented (bool): Whether the junction graph required per-component gauging.
        n_components (int): Number of gauged components (1 when not fragmented).
        isolated (tuple[Interface, ...]): Interfaces with no triple line at all.
        stacklevel (int, optional): passed through to `warnings.warn`, so the warning points
            at the caller's own line rather than at this module. Defaults to 3.
    """
    if not fragmented:
        return
    undetermined_note = (
        f" and {len(isolated)} undetermined interface(s) {list(isolated)}" if isolated else ""
    )
    message = (
        f"this mesh's triple-junction graph is fragmented into {n_components} gauge "
        f"component(s){undetermined_note}. Tensions in different components are on different, "
        "arbitrary relative scales and must not be averaged or compared directly against each "
        "other; an undetermined interface (no triple line at all -- a closed surface) has no "
        "tension at all and is reported as nan. Use infer_tensions_by_junction_component to see "
        "which interface belongs to which component, and each component's own gauge."
    )
    warnings.warn(message, JunctionGraphGaugeWarning, stacklevel=stacklevel)


def partition_by_junction_graph(
    f: NDArray[np.int64],
    regions: NDArray[np.int64],
) -> JunctionGraphPartition:
    """Connected components of the triple-junction graph, from raw mesh arrays.

    One node per interface (`mesh_io.list_interfaces`); an edge between two interfaces
    whenever they co-occur in a triple line (`mesh_io.list_junctions` gives each triple
    line's three constituent interfaces, all `C(3,2) = 3` pairs among them become edges).

    Args:
        f (NDArray[np.int64]): Mesh triangles (vertex indices), shape `(nf, 3)`.
        regions (NDArray[np.int64]): Per-triangle region pair, shape `(nf, 2)`.

    Returns:
        JunctionGraphPartition: The partition. A "healthy" mesh (every interface reachable
            from every other through some chain of shared triple lines, no closed interface)
            returns exactly one component holding every interface and an empty `isolated`.

    Raises:
        MalformedTripleLineError: If a triple line names an interface pair that is not
            present in `mesh_io.list_interfaces(regions)`. `mesh_io.list_junctions`'s
            admission rule (three interfaces on exactly three incident triangles) should
            make this unreachable in practice; this is the defensive, named replacement for
            what used to be a bare `KeyError`.
    """
    mesh_io.warn_on_malformed_triple_lines(f, regions, stacklevel=3)
    interfaces = mesh_io.list_interfaces(regions)
    junctions, edge_to_triple = mesh_io.list_junctions(f, regions)
    return _partition_over(interfaces, junctions, edge_to_triple)


def partition_over_surviving_junctions(
    f: NDArray[np.int64],
    regions: NDArray[np.int64],
    surviving: Collection[tuple[int, int, int]],
) -> JunctionGraphPartition:
    """:func:`partition_by_junction_graph`, restricted to the triple lines still carrying a row.

    An estimator that declines some of its observations no longer couples the interfaces those
    observations coupled, so the gauge structure has to be read off the *surviving* junction
    lines rather than off the mesh. An interface left touching no surviving line is then
    `isolated` — exactly what this module already reports as `float("nan")` for a
    fully closed surface — and a survivor graph that falls into several pieces is gauged per
    component by :func:`solve_gauged_least_squares`, as a fragmented mesh already is.

    `surviving` only ever **removes** lines: a triple that this mesh's own
    `mesh_io.list_junctions` does not admit as a genuine triple line is absent from the graph
    here exactly as it is absent from :func:`partition_by_junction_graph`'s, so an estimator
    whose rows outrun that admission rule is treated identically with and without refusal.

    Args:
        f (NDArray[np.int64]): Mesh triangles (vertex indices), shape `(nf, 3)`.
        regions (NDArray[np.int64]): Per-triangle region pair, shape `(nf, 2)`.
        surviving (Collection[tuple[int, int, int]]): sorted region-triples whose row survived.

    Returns:
        JunctionGraphPartition: the partition of the same interfaces by the surviving lines
            alone. Passing every triple the mesh has reproduces
            :func:`partition_by_junction_graph` exactly.
    """
    mesh_io.warn_on_malformed_triple_lines(f, regions, stacklevel=3)
    interfaces = mesh_io.list_interfaces(regions)
    junctions, edge_to_triple = mesh_io.list_junctions(f, regions)
    kept = set(surviving)
    return _partition_over(interfaces, [triple for triple in junctions if triple in kept], edge_to_triple)


def _partition_over(
    interfaces: Sequence[Interface],
    junctions: Sequence[tuple[int, int, int]],
    edge_to_triple: dict[tuple[int, int], tuple[int, int, int]],
) -> JunctionGraphPartition:
    """Union-find over a given interface set and a given list of triple lines.

    Shared by :func:`partition_by_junction_graph` and
    :func:`partition_over_surviving_junctions` so that the two cannot drift apart; it is pure
    integer bookkeeping, so restricting `junctions` is the only thing that distinguishes them.

    Args:
        interfaces (Sequence[Interface]): every interface present on the mesh — the graph's nodes.
        junctions (Sequence[tuple[int, int, int]]): the region-triples that contribute edges.
        edge_to_triple (dict): mesh edge -> region-triple, used only to name the offending edge
            in `MalformedTripleLineError`.

    Returns:
        JunctionGraphPartition: See :func:`partition_by_junction_graph`.

    Raises:
        MalformedTripleLineError: If a triple line names an interface pair absent from
            `interfaces`.
    """
    parent: dict[Interface, Interface] = {key: key for key in interfaces}

    def find(x: Interface, triple: tuple[int, int, int] | None = None) -> Interface:
        if x not in parent:
            edge = next((e for e, t in edge_to_triple.items() if t == triple), None)
            raise MalformedTripleLineError(edge, triple, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: Interface, b: Interface, triple: tuple[int, int, int]) -> None:
        ra, rb = find(a, triple), find(b, triple)
        if ra != rb:
            parent[ra] = rb

    touched: set[Interface] = set()
    for a, b, c in junctions:
        pairs = ((min(a, b), max(a, b)), (min(b, c), max(b, c)), (min(a, c), max(a, c)))
        touched.update(pairs)
        union(pairs[0], pairs[1], (a, b, c))
        union(pairs[0], pairs[2], (a, b, c))

    groups: dict[Interface, list[Interface]] = {}
    for key in touched:
        groups.setdefault(find(key), []).append(key)
    components = tuple(
        tuple(sorted(group)) for group in sorted(groups.values(), key=lambda group: (-len(group), min(group)))
    )
    component_of = {key: index for index, component in enumerate(components) for key in component}
    isolated = tuple(sorted(set(interfaces) - touched))

    return JunctionGraphPartition(
        interfaces=tuple(interfaces),
        components=components,
        component_of=component_of,
        isolated=isolated,
    )


def partition_from_mesh(mesh: object) -> JunctionGraphPartition:
    """Convenience: partition a `DcelData` mesh directly (`mesh.f`'s vertex/label columns).

    Args:
        mesh: A `foambryo.dcel.DcelData` (typed as `object` to avoid a hard import cycle;
            only `mesh.f`, an `(nf, 5)` array of `[v0, v1, v2, label_a, label_b]`, is used).

    Returns:
        JunctionGraphPartition: See `partition_by_junction_graph`.
    """
    return partition_by_junction_graph(mesh.f[:, :3], mesh.f[:, 3:5])  # type: ignore[attr-defined]


@dataclass(frozen=True)
class ComponentTensionSolution:
    """One inference's tensions, decomposed by the junction graph.

    When `fragmented` is `False` this is the solve of the whole mesh with one normalisation,
    `mean(gamma) = mean_tension` met exactly. When `fragmented` is `True`, each connected
    component of the interfaces' triple-junction graph was solved with its **own**
    `mean(gamma) = mean_tension` normalisation, met exactly; a component's
    tensions are therefore comparable only to each other, never across components, and
    `gauge_notes` says so per component. Isolated interfaces (no triple line at all) are
    undetermined: `dict_tensions` maps them to `float("nan")`, and `component_of` maps them
    to `None`.

    Attributes:
        tensions: `(n_interfaces,)`, aligned with the `keys` the caller passed in;
            `float("nan")` at every undetermined interface.
        dict_tensions: The same, keyed by interface. `float("nan")` for undetermined
            interfaces -- never a number.
        component_of: Interface -> component index, or `None` for an undetermined interface.
        gauge_notes: Component index -> a one-line description of the gauge applied to it.
        undetermined: Interfaces with no triple line at all (closed surfaces).
        lstsq_residuals: Concatenation of each component's squared physics residual (a
            length-1 array per component; the normalisation, met exactly, adds none). When
            `fragmented` is `False`, the single whole-mesh solve's.
        fragmented: Whether the junction graph required per-component gauging.
        n_components: Number of gauged components (1 when not fragmented).
        solver_notes: Component index -> which least-squares solver ran on it and, for the sparse
            solver, its stop code, iteration count, residual norm and condition estimate
            (`_solve_least_squares`). Component 0 is the whole mesh when not fragmented. Empty for
            solutions not built by `solve_gauged_least_squares`, and for `LogLami`, which rebuilds
            its solution from the one this function returned.
    """

    tensions: NDArray[np.float64]
    dict_tensions: dict[Interface, float]
    component_of: dict[Interface, int | None]
    gauge_notes: dict[int, str]
    undetermined: tuple[Interface, ...]
    lstsq_residuals: NDArray[np.float64]
    fragmented: bool
    n_components: int
    solver_notes: dict[int, dict[str, object]] = field(default_factory=dict)


class GaugeNotExactWarning(UserWarning):
    """A gauge could not be made exact: the solution is the well-scaled soft row's.

    Raised only for a system whose physics rows are neither homogeneous (zero right-hand side) nor
    differences (each row sums to zero): no method of this package builds one.
    """


def _row_norms(physics: NDArray[np.float64] | sparse.csr_matrix) -> NDArray[np.float64]:
    """The 2-norm of every row, dense or sparse."""
    if sparse.issparse(physics):
        return np.sqrt(np.asarray(physics.multiply(physics).sum(axis=1)).ravel())
    return np.linalg.norm(physics, axis=1)


def gauge_row_weight(physics: NDArray[np.float64] | sparse.csr_matrix, n_unknowns: int) -> float:
    """Weight of the gauge row: its norm is the median non-zero physics-row norm (to a power of two).

    The row is `weight * 1^T`, norm `weight * sqrt(n)`. Matching it to the physics rows' typical norm
    puts the system at the bottom of the condition number's V in the row weight (measured on a
    1 866-cell tissue's Young-Dupre system: 311 against 2.8e7 for the unit row). 1.0 when no physics
    row has an entry.
    """
    norms = _row_norms(physics)
    norms = norms[norms > 0]
    if len(norms) == 0 or n_unknowns == 0:
        return 1.0
    # Rounded to a power of two: the same weight whatever the summation order of the norms (dense or
    # sparse rows), and an exact multiplication. It moves the condition number by at most sqrt(2).
    return float(2.0 ** np.round(np.log2(np.median(norms) / np.sqrt(n_unknowns))))


def solve_with_exact_gauge(
    physics: NDArray[np.float64] | sparse.csr_matrix,
    physics_b: NDArray[np.float64],
    gauge_total: float,
    *,
    system: str = "tension system",
    use_sparse: bool | None = None,
    scale_columns: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, object]]:
    """Least squares on `physics @ x = physics_b` with the gauge `sum(x) = gauge_total` met exactly.

    The gauge row is appended with the weight `gauge_row_weight` (its norm equal to the median
    physics-row norm, where the system is best conditioned), the system is solved by
    `_solve_least_squares` (dense or sparse as the options say), and the gauge is then made exact
    without moving the solution off the physics rows' least-squares set:

    * **homogeneous physics rows** (`physics_b == 0`: every Young-Dupre-family, cotangent, Lami,
      per-edge and `Equilibrium` system): the solution of a soft gauge row of any weight is the
      exact-gauge solution times one factor (Sherman-Morrison; the two were measured to agree to
      1e-13), so it is rescaled to `sum(x) = gauge_total`. Tension ratios are those of the exact
      constrained least-squares solution;
    * **difference rows** (each physics row sums to zero, e.g. `LogLami` in log space): adding a
      constant to every unknown leaves every physics residual unchanged, so the solution is shifted
      to meet the gauge;
    * otherwise (no builder of this package): the soft row's solution is returned with a
      `GaugeNotExactWarning`.

    Args:
        physics: `(n_rows, n)` physics rows, dense or CSR, without any gauge row.
        physics_b: `(n_rows,)` their right-hand side.
        gauge_total: the value `sum(x)` must take (`n * mean_tension` for the tension methods).
        system: label, for messages and diagnostics. Keyword-only.
        use_sparse: force a path, see `_solve_least_squares`. Keyword-only.
        scale_columns: see `_solve_least_squares`. Keyword-only.

    Returns:
        tuple: `(x, residuals, record)`. `residuals` is the squared 2-norm of the physics rows'
        residual, `physics @ x - physics_b`, as a length-1 array: the gauge row is met exactly and
        contributes nothing. `record` is `_solve_least_squares`'s, plus `gauge` (how the gauge was
        made exact) and `gauge_row_weight`.

    Nothing is raised beyond `_solve_least_squares`'s exceptions; a zero-sum solution of homogeneous
    rows (no scale to rescale) is returned as solved, with a `GaugeNotExactWarning`.
    """
    n = physics.shape[1]
    weight = gauge_row_weight(physics, n)
    if sparse.issparse(physics):
        matrix = sparse.vstack([physics, sparse.csr_matrix(np.full((1, n), weight))], format="csr")
    else:
        matrix = np.vstack([physics, np.full((1, n), weight)])
    vector = np.concatenate([physics_b, [weight * gauge_total]])
    x, _residuals, record = _solve_least_squares(
        matrix, vector, system=system, use_sparse=use_sparse, scale_columns=scale_columns,
    )
    total = float(np.sum(x))
    if not np.any(physics_b):
        if total != 0.0 and np.isfinite(total):
            x = x * (gauge_total / total)
            record["gauge"] = "exact: well-scaled gauge row, then the homogeneous solution rescaled"
        else:
            warnings.warn(f"the {system}'s solution sums to {total}: its gauge cannot be made exact by rescaling",
                          GaugeNotExactWarning, stacklevel=3)
            record["gauge"] = "not exact: zero-sum homogeneous solution"
    else:
        row_sums = np.abs(np.asarray(physics.sum(axis=1)).ravel())
        if np.all(row_sums <= 1e-12 * np.maximum(_row_norms(physics), np.finfo(np.float64).tiny)):
            x = x + (gauge_total - total) / n
            record["gauge"] = "exact: well-scaled gauge row, then shifted along the rows' null direction"
        else:
            warnings.warn(f"the {system}'s rows are neither homogeneous nor differences: its gauge is a "
                          "soft least-squares row, not exact", GaugeNotExactWarning, stacklevel=3)
            record["gauge"] = "soft (well-scaled row)"
    record["gauge_row_weight"] = weight
    difference = (physics @ x) - physics_b
    return x, np.array([float(difference @ difference)]), record


def _component_system(
    physics: NDArray[np.float64] | sparse.csr_matrix,
    row_mask: NDArray[np.bool_],
    cols: NDArray[np.intp],
    physics_sparse: list[sparse.csr_matrix],
) -> NDArray[np.float64] | sparse.csr_matrix:
    """One component's rows and columns of the physics block (no gauge row: see `solve_with_exact_gauge`).

    When the component takes the dense path (`foambryo.linear_solves.use_sparse_solver`; by default
    below `SPARSE_SOLVE_MIN_UNKNOWNS` columns) this is the dense sub-matrix (from a sparse `physics`,
    the same entries densified). On the sparse path, the same
    rows and columns in CSR form, never copied densely (at tissue scale the dense sub-matrix alone is
    several GiB); the CSR form of a dense physics block is built on first use and kept in
    `physics_sparse` (a one-element cache) for the next component.
    """
    if not linear_solves.use_sparse_solver(len(cols)):
        if sparse.issparse(physics):
            return physics[np.flatnonzero(row_mask)][:, cols].toarray()
        return physics[np.ix_(row_mask, cols)]
    if not physics_sparse:
        physics_sparse.append(physics.tocsr() if sparse.issparse(physics) else sparse.csr_matrix(physics))
    return physics_sparse[0][np.flatnonzero(row_mask)][:, cols].tocsr()


def _row_components(
    physics: NDArray[np.float64] | sparse.csr_matrix,
    col_component: NDArray[np.int_],
) -> NDArray[np.int_]:
    """The junction-graph component each physics row belongs to (`-1` for a row with no entry).

    Raises:
        ValueError: if a row's non-zero columns span more than one component (see
            `solve_gauged_least_squares`).
    """
    if sparse.issparse(physics):
        csr = physics.tocsr(copy=True)
        csr.eliminate_zeros()
        counts = np.diff(csr.indptr)
        has_support = counts > 0
        row_component = np.full(csr.shape[0], -1, dtype=col_component.dtype)
        row_component[has_support] = col_component[csr.indices[csr.indptr[:-1][has_support]]]
        entry_row = np.repeat(np.arange(csr.shape[0]), counts)
        mixed = bool(np.any(col_component[csr.indices] != row_component[entry_row]))
    else:
        nz_mask = physics != 0
        has_support = nz_mask.any(axis=1)
        first_col = np.argmax(nz_mask, axis=1)
        row_component = np.where(has_support, col_component[first_col], -1)
        # Defensive, cheap: the fragmented branch is rare, so a per-row Python check costs
        # nothing in practice and guards the decomposition's whole premise -- every physics row
        # must be confined to one component's columns.
        mixed = any(
            len({int(c) for c in col_component[np.nonzero(nz_mask[row_index])[0]]}) != 1
            for row_index in np.nonzero(has_support)[0]
        )
    if mixed:
        message = (
            "a physics row references interfaces from more than one triple-junction-graph "
            "component: this solver's row coupling does not match "
            "partition_by_junction_graph's triple-line-only graph, so the per-component "
            "decomposition must not be applied to it (see this module's docstring, "
            "'Scope: which methods')."
        )
        raise ValueError(message)
    return row_component


def solve_gauged_least_squares(
    matrix_m: NDArray[np.float64] | sparse.csr_matrix,
    vector_b: NDArray[np.float64],
    keys: Sequence[Interface],
    mean_tension: float,
    partition: JunctionGraphPartition,
    *,
    system: str = "tension system",
) -> ComponentTensionSolution:
    """Solve `matrix_m @ x = vector_b`, gauging per junction-graph component if fragmented.

    `matrix_m` must be exactly the shape every builder in `foambryo.tension_inference`
    produces: `(n_rows, len(keys))`, whose **last row** is the single global gauge
    `sum(gamma) = vector_b[-1]` (`vector_b[-1]` is `len(keys) * mean_tension` for every
    method except `LogLami`, which works in `log`-space -- this function reads
    `vector_b[-1]` directly rather than recomputing it, so it is agnostic to which
    convention the caller used). The row itself is not used: each component is solved by
    `solve_with_exact_gauge`, which meets its share of the gauge exactly (`sum(gamma)` over the
    component `= vector_b[-1] * n_component / n`, i.e. `mean(gamma) = mean_tension`). The residuals
    returned are the physics rows' (the gauge contributes none).

    Each component's solve is dense or sparse as `foambryo.linear_solves`'s options in force say
    (by default: dense below `SPARSE_SOLVE_MIN_UNKNOWNS` interfaces, `lsqr` at or above), see
    `_solve_least_squares`.

    Args:
        matrix_m (NDArray[np.float64] | scipy.sparse.csr_matrix): Assembled system, gauge row
            last. A sparse system is never densified as a whole: only the components that take the
            dense path are.
        vector_b (NDArray[np.float64]): Right-hand side, gauge value last.
        keys (Sequence[Interface]): Interface key per column of `matrix_m`, in order.
        mean_tension (float): The gauge value the caller requested (only used for the
            human-readable `gauge_notes`; the actual constraint value is read from
            `vector_b[-1]`).
        partition (JunctionGraphPartition): This mesh's junction-graph partition. Must cover
            exactly `keys` (checked).
        system (str, optional): what the system is, for messages and diagnostics. Keyword-only.

    Returns:
        ComponentTensionSolution: See its docstring.

    Raises:
        ValueError: If `partition`'s interfaces do not exactly match `keys`, or (defensive,
            fragmented case only) if a physics row's nonzero columns span more than one
            component -- meaning the caller's row coupling does not match the triple-junction
            graph, and this decomposition must not be trusted for that builder.
    """
    if set(partition.interfaces) != set(keys):
        message = (
            "partition_by_junction_graph's interfaces do not match the solver's own columns "
            "-- the mesh passed to partition_by_junction_graph must be the same mesh the "
            "caller built matrix_m from."
        )
        raise ValueError(message)

    nm = len(keys)

    if not partition.fragmented:
        tensions, residuals, solver_note = solve_with_exact_gauge(
            matrix_m[:-1], vector_b[:-1], float(vector_b[-1]), system=system,
        )
        return ComponentTensionSolution(
            tensions=tensions,
            dict_tensions=dict(zip(keys, tensions.tolist(), strict=True)),
            component_of=dict.fromkeys(keys, 0),
            gauge_notes={
                0: f"mean(gamma) = {mean_tension} exactly, over all {nm} interfaces (single connected junction graph)",
            },
            undetermined=(),
            lstsq_residuals=residuals,
            fragmented=False,
            n_components=1,
            solver_notes={0: solver_note},
        )

    index_of = {key: i for i, key in enumerate(keys)}
    physics = matrix_m[:-1]
    physics_b = vector_b[:-1]
    gauge_total = float(vector_b[-1])  # nm * f(mean_tension), whichever f the caller used

    col_component = np.full(nm, -1, dtype=int)
    for component_id, component_keys in enumerate(partition.components):
        for key in component_keys:
            col_component[index_of[key]] = component_id

    row_component = _row_components(physics, col_component)

    tensions = np.full(nm, np.nan)
    dict_tensions: dict[Interface, float] = {}
    component_of: dict[Interface, int | None] = {}
    gauge_notes: dict[int, str] = {}
    solver_notes: dict[int, dict[str, object]] = {}
    residual_chunks: list[NDArray[np.float64]] = []
    physics_sparse: list[sparse.csr_matrix] = []  # built once, only if a component takes the sparse path

    for key in partition.isolated:
        dict_tensions[key] = float("nan")
        component_of[key] = None

    for component_id, component_keys in enumerate(partition.components):
        cols = np.array([index_of[key] for key in component_keys])
        row_mask = row_component == component_id
        sub_b = physics_b[row_mask]
        gauge_b = gauge_total * (len(cols) / nm)
        sub_m = _component_system(physics, row_mask, cols, physics_sparse)
        sub_tensions, sub_resid, solver_notes[component_id] = solve_with_exact_gauge(
            sub_m, sub_b, gauge_b, system=f"{system}, junction-graph component {component_id}",
        )
        residual_chunks.append(np.atleast_1d(sub_resid) if sub_resid.size else np.zeros(0))
        for local_index, key in enumerate(component_keys):
            tensions[index_of[key]] = sub_tensions[local_index]
            dict_tensions[key] = float(sub_tensions[local_index])
            component_of[key] = component_id
        gauge_notes[component_id] = (
            f"mean(gamma) = {mean_tension} exactly, over this component's {len(cols)} interfaces only "
            "(disconnected from the rest of the mesh's triple-junction graph -- not comparable "
            "in scale to any other component)"
        )

    return ComponentTensionSolution(
        tensions=tensions,
        dict_tensions=dict_tensions,
        component_of=component_of,
        gauge_notes=gauge_notes,
        undetermined=partition.isolated,
        lstsq_residuals=(np.concatenate(residual_chunks) if residual_chunks else np.zeros(0)),
        fragmented=True,
        n_components=partition.n_components,
        solver_notes=solver_notes,
    )


def require_single_component_solution(solution: ComponentTensionSolution) -> int:
    """Refuse to let a caller silently aggregate a fragmented solution across components.

    Call this at the top of any function that reduces one mesh's tensions to a single number (a mean,
    a single gauge-matching scale, a pooled error). It raises rather than returning a number
    computed by blending two components that are not on the same scale.

    Args:
        solution (ComponentTensionSolution): A solve from `solve_gauged_least_squares`.

    Returns:
        int: `0`, the sole component index, when `solution` is not fragmented.

    Raises:
        ValueError: If `solution.fragmented` is `True` -- i.e. more than one gauge component,
            or at least one undetermined interface, is present.
    """
    if solution.fragmented:
        message = (
            f"refusing to aggregate across the junction-graph gauge split: this mesh's "
            f"tensions have {solution.n_components} independently-gauged component(s) and "
            f"{len(solution.undetermined)} undetermined interface(s) -> "
            f"{solution.undetermined}. Aggregate per component instead (see "
            f"foambryo.junction_graph)."
        )
        raise ValueError(message)
    return 0
