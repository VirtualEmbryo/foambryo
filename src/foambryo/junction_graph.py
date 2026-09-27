r"""Junction-graph fragmentation: per-component gauge for tension solvers.

The junction-graph work. Traced by the converged-mesh-cohort work: a Young-Dupre-family
solver couples two interfaces only when they share a triple line. Build the graph with interfaces as nodes and
an edge wherever two interfaces meet at a triple line
(`foambryo.validation.mesh_io.list_junctions`). If that graph has `k` connected components,
the tension system decomposes into `k` blocks each determined only up to **its own scale**,
while `mean(gamma) = mean_tension` supplies a single global constraint. Ordinary least
squares then returns the minimum-norm solution, which drives the `k - 1` unpinned blocks
towards zero -- giving a gauge-matched relative error of exactly 1.0, silently. The
converged-mesh-cohort work found this on 12 of 72 converged-cohort meshes.

**A degenerate sub-case.** An interface that is a fully closed surface (an engulfed cell, a
sealed lumen) has no triple line at all, so it is an isolated node in *no* junction-graph
edge and *no* equation in any builder's physics rows. Its tension is not merely unpinned in
scale -- it is wholly undetermined, and this module reports it as such (`float("nan")`),
never as a number, never silently folded into a component.

**API decision (H. Turlier, 2026-08-02; a design decision implemented here, not revisited).** Infer
per component, with a per-component gauge: each block's tensions are internally consistent
and comparable only *within* the block. That must be surfaced in the output, not buried in a
docstring -- `ComponentTensionSolution` therefore carries, for every interface, which
component it belongs to and which gauge was applied to that component; isolated/closed
interfaces are reported as undetermined, never as a component with a gauge and never as a
number; and any code that would aggregate tensions *across* components must refuse rather
than silently blend two incompatible scales. `foambryo_benchmarks.case_groups.require_single_group`
is this module's precedent for how an aggregation helper enforces that (see
`require_single_component_solution` below).

**Backward compatibility.** When the junction graph has exactly one component and no
isolated interface -- every mesh in `benchmarking-dataset` and every phantom, as measured
when this module was written --
`solve_gauged_least_squares` calls `scipy.linalg.lstsq` on exactly the `(matrix_m, vector_b)`
its caller already built, with no other floating-point operation in between. The result is
bit-identical to the pre-existing code path by construction, not merely by re-derivation.

**Scope: which methods.** This module decomposes any solver whose assembled system is
`[physics rows; one trailing global gauge row `sum(gamma) = n * f(mean_tension)`]`, which is
every method in `foambryo.tension_inference` and `foambryo.per_edge_tension_inference`
**except**:

- `TensionComputationMethod.Equilibrium`, whose physics rows are keyed by *geometric mesh
  edges* (`len(edge.twin) != 1`), which include quadrijunction edges that
  `mesh_io.list_junctions` does not enumerate (it only finds edges bordering exactly three
  regions) -- so this module's triple-line-only graph would not necessarily match
  `Equilibrium`'s actual row coupling. Left unfixed and undocumented as a known limitation,
  not silently misapplied;
- the `ForceBalance*` and `Variational` family, whose operator additionally carries area
  *and* volume gradients tied to shared cells and is column-coupled through the pressure
  block rather than through the triple-junction graph alone -- measured not to fragment the
  same way, so the fix does not apply there and was not built there.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from foambryo.validation import mesh_io

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

Interface = tuple[int, int]


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
        """True iff a single global `mean(gamma) = mean_tension` row no longer pins one scale.

        This is the case whenever there is more than one component, **or** at least one
        isolated interface -- an isolated interface is already a failure of the single-gauge
        assumption even if the rest of the mesh's junction graph is fully connected.
        """
        return self.n_components > 1 or bool(self.isolated)


class JunctionGraphGaugeWarning(UserWarning):
    """Raised when a mesh's triple-junction graph is fragmented or has an undetermined interface.

    Fires whenever `JunctionGraphPartition.fragmented` is true: more than one gauge
    component, at least one undetermined (fully closed) interface, or both. Tensions in
    different components sit on different, arbitrary relative scales -- comparing or
    averaging them directly is meaningless -- and an undetermined interface has no triple
    line at all and is reported as `nan`, never a number. The flat `infer_tensions` /
    `infer_tensions_and_residuals` entry points return exactly that flat, single-gauge
    picture regardless; use `foambryo.tension_inference.infer_tensions_by_junction_component`
    to get each interface's component and gauge explicitly instead of blending them.
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
    `isolated` — exactly what the 2026-08-02 API decision already reports as `float("nan")` for a
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

    When `fragmented` is `False` this is numerically identical to solving the whole mesh
    with the single global gauge (bit-for-bit) -- the backward-compatible case. When
    `fragmented` is `True`, each connected component of the interfaces' triple-junction
    graph was solved with its **own** `mean(gamma) = mean_tension` gauge; a component's
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
        lstsq_residuals: Concatenation of each component's own `scipy.linalg.lstsq` residual
            array. When `fragmented` is `False` this is exactly what the single whole-mesh
            solve returned.
        fragmented: Whether the junction graph required per-component gauging.
        n_components: Number of gauged components (1 when not fragmented).
    """

    tensions: NDArray[np.float64]
    dict_tensions: dict[Interface, float]
    component_of: dict[Interface, int | None]
    gauge_notes: dict[int, str]
    undetermined: tuple[Interface, ...]
    lstsq_residuals: NDArray[np.float64]
    fragmented: bool
    n_components: int


def solve_gauged_least_squares(
    matrix_m: NDArray[np.float64],
    vector_b: NDArray[np.float64],
    keys: Sequence[Interface],
    mean_tension: float,
    partition: JunctionGraphPartition,
) -> ComponentTensionSolution:
    """Solve `matrix_m @ x = vector_b`, gauging per junction-graph component if fragmented.

    `matrix_m` must be exactly the shape every builder in `foambryo.tension_inference`
    produces: `(n_rows, len(keys))`, whose **last row** is the single global gauge
    `sum(gamma) = vector_b[-1]` (`vector_b[-1]` is `len(keys) * mean_tension` for every
    method except `LogLami`, which works in `log`-space -- this function reads
    `vector_b[-1]` directly rather than recomputing it, so it is agnostic to which
    convention the caller used).

    Args:
        matrix_m (NDArray[np.float64]): Assembled system, gauge row last.
        vector_b (NDArray[np.float64]): Right-hand side, gauge value last.
        keys (Sequence[Interface]): Interface key per column of `matrix_m`, in order.
        mean_tension (float): The gauge value the caller requested (only used for the
            human-readable `gauge_notes`; the actual constraint value is read from
            `vector_b[-1]`).
        partition (JunctionGraphPartition): This mesh's junction-graph partition. Must cover
            exactly `keys` (checked).

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
        tensions, residuals, _rank, _sv = linalg.lstsq(matrix_m, vector_b)
        return ComponentTensionSolution(
            tensions=tensions,
            dict_tensions=dict(zip(keys, tensions.tolist(), strict=True)),
            component_of=dict.fromkeys(keys, 0),
            gauge_notes={
                0: f"mean(gamma) = {mean_tension} over all {nm} interfaces (single connected junction graph)",
            },
            undetermined=(),
            lstsq_residuals=residuals,
            fragmented=False,
            n_components=1,
        )

    index_of = {key: i for i, key in enumerate(keys)}
    physics = matrix_m[:-1]
    physics_b = vector_b[:-1]
    gauge_total = float(vector_b[-1])  # nm * f(mean_tension), whichever f the caller used

    col_component = np.full(nm, -1, dtype=int)
    for component_id, component_keys in enumerate(partition.components):
        for key in component_keys:
            col_component[index_of[key]] = component_id

    nz_mask = physics != 0
    has_support = nz_mask.any(axis=1)
    first_col = np.argmax(nz_mask, axis=1)
    row_component = np.where(has_support, col_component[first_col], -1)

    # Defensive, cheap: the fragmented branch is rare, so a per-row Python check costs
    # nothing in practice and guards the decomposition's whole premise -- every physics row
    # must be confined to one component's columns.
    for row_index in np.nonzero(has_support)[0]:
        touched_components = {int(component_id) for component_id in col_component[np.nonzero(nz_mask[row_index])[0]]}
        if len(touched_components) != 1:
            message = (
                "a physics row references interfaces from more than one triple-junction-graph "
                "component: this solver's row coupling does not match "
                "partition_by_junction_graph's triple-line-only graph, so the per-component "
                "decomposition must not be applied to it (see this module's docstring, "
                "'Scope: which methods')."
            )
            raise ValueError(message)

    tensions = np.full(nm, np.nan)
    dict_tensions: dict[Interface, float] = {}
    component_of: dict[Interface, int | None] = {}
    gauge_notes: dict[int, str] = {}
    residual_chunks: list[NDArray[np.float64]] = []

    for key in partition.isolated:
        dict_tensions[key] = float("nan")
        component_of[key] = None

    for component_id, component_keys in enumerate(partition.components):
        cols = np.array([index_of[key] for key in component_keys])
        row_mask = row_component == component_id
        sub_m = physics[np.ix_(row_mask, cols)]
        sub_b = physics_b[row_mask]
        gauge_row = np.ones((1, len(cols)))
        gauge_b = gauge_total * (len(cols) / nm)
        full_m = np.vstack([sub_m, gauge_row])
        full_b = np.concatenate([sub_b, [gauge_b]])
        sub_tensions, sub_resid, _rank, _sv = linalg.lstsq(full_m, full_b)
        residual_chunks.append(np.atleast_1d(sub_resid) if sub_resid.size else np.zeros(0))
        for local_index, key in enumerate(component_keys):
            tensions[index_of[key]] = sub_tensions[local_index]
            dict_tensions[key] = float(sub_tensions[local_index])
            component_of[key] = component_id
        gauge_notes[component_id] = (
            f"mean(gamma) = {mean_tension} over this component's {len(cols)} interfaces only "
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
    )


def require_single_component_solution(solution: ComponentTensionSolution) -> int:
    """Refuse to let a caller silently aggregate a fragmented solution across components.

    The junction-graph analogue of `foambryo_benchmarks.case_groups.require_single_group`: call this
    at the top of any function that reduces one mesh's tensions to a single number (a mean,
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
