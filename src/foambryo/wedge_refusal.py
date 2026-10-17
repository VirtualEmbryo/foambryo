r"""Refusal policies: declining a trijunction observation the mesh's own geometry makes infeasible.

A `WedgeRefusal` is passed as `refusal=` to `infer_tensions` and its relatives, and to
`infer_tensions_young_dupre_local`. Only the methods of `foambryo.tension_inference.ANGLE_READING_METHODS`
honour it; the others raise `ValueError` on any policy but `NO_REFUSAL`. `refusal_report` counts
what a policy declines.

The problem
-----------
At a trijunctional edge three films meet and the three interior wedge angles sum to a full turn.
A **reflex** wedge is one wider than a straight angle: the surface is locally folded there, the
three films do not enclose the edge the way a trijunction requires, and the Young-Dupré balance
every angle-reading estimator in this package imposes has no solution for such a triple. Feeding
the estimator that row is feeding it an observation that cannot have come from the object being
measured.

Two facts make this worth acting on rather than documenting:

* `foambryo.geometry`'s oriented reading can *see* the fold — but it cannot undo it. The per-edge
  estimator's rows are built from `cos` of the wedge, and `cos` is even about a full turn, so a
  reflex wedge and its reflection are literally the same row; no reading reaches it.
* The `log`-of-`sin` estimator does not merely degrade on such a row, it **fails**: a reflex wedge
  has a negative oriented sine, and the row's logarithm is not defined.

So the estimator should refuse the observation. This module is the machinery for that, and it is
**off by default** — `NO_REFUSAL` reproduces every existing number bitwise, and which policy (if
any) should become a default is a separate decision taken on its own evidence.

The three policies, and the unit each acts on
---------------------------------------------
Every angle-reading estimator here builds its rows from one of two observation units: a whole
**junction line** (the region triple), for the methods that average the wedge angles along the
line and weight the row by the line's arclength share; or a single **trijunctional edge**, for the
per-edge method, whose row *is* one edge. The policies are defined on the unit, so each is one
intervention with one meaning in both families:

* :attr:`WedgeRefusalPolicy.ExcludeEdgeFromLineMean` (`E`) — a refused edge contributes neither
  its angles nor its length to its line. The line keeps its row, computed from the surviving
  edges and weighted by their surviving arclength. A line that loses every edge has no mean and
  therefore no row.
* :attr:`WedgeRefusalPolicy.DropLine` (`D`) — a line carrying at least one refused edge loses its
  row entirely.
* :attr:`WedgeRefusalPolicy.DownweightLine` (`W`) — nothing is removed from the measurement: the
  line's angle means are what they always were, and only its row weight is scaled, by the
  fraction of its arclength that is not refused. A line with no unrefused arclength has weight
  zero, which is the same as having no row.

**A refused wedge takes the other two wedges of its own edge with it.** The three are one
observation of one edge and every row consumes all three; dropping one wedge key and keeping its
two partners would leave a line's three means computed over different edge subsets and no longer
summing to a full turn.

Two consequences follow from the definitions and are asserted as tests rather than assumed
(`tests/test_wedge_refusal.py`):

1. For the **per-edge** method the unit is one edge, so under `E` a refused edge's unit is empty
   and it has no row — `E` and `D` are the same intervention there and agree exactly. `W` is not:
   it keeps the refused rows and scales every row of their line instead.
2. For the **line-averaged** methods `E` and `W` give a surviving line the *same* row weight
   (`E`'s surviving arclength over the surviving total; `W`'s full arclength times the surviving
   fraction, over that same total). So `E` and `W` differ only in the angle a line reports, and
   `D` and `E` only in whether a partly folded line reports at all.

Underdetermination is an outcome, not an error
----------------------------------------------
Refusing rows can leave an interface with no surviving junction line at all, or split the
junction graph into pieces that no longer share a gauge. That is handled exactly as
`foambryo.junction_graph` already handles a closed interface and a fragmented mesh: the
partition is rebuilt over the surviving lines
(`junction_graph.partition_over_surviving_junctions`), an interface touched by no surviving line
is reported `float("nan")` and never a number, a survivor graph in several pieces is
normalised in each piece, and `JunctionGraphGaugeWarning` fires unchanged. A mesh all of whose lines are refused
returns every interface undetermined rather than raising: an empty answer is a result, and it is
one the caller can count.

**Report the refused counts next to any accuracy number.** A policy that improves an error
statistic by declining a large share of the data has not improved anything, and
:func:`refusal_report` is what makes that visible.

Two reasons to decline an edge: a fold, and the crop
----------------------------------------------------
A junction edge with a vertex on a crop face (`foambryo.crop_faces`) is not an observation of the
tissue either: the tissue beyond the crop was cut away. It is declined for that reason on every
cropped mesh, whatever the policy, and always edge by edge: it leaves its line's mean and length,
and the rest of the line still reports. The policy then acts on what remains, as if the crop
edges were not part of the line: under `DropLine` a line is dropped when one of its *other* edges
is refused, and under `DownweightLine` its weight is the unrefused fraction of the arclength
that is not on the crop. The two sets are held apart (`WedgeRefusal.crop_edges` beside the
policy's own `edges`) and counted apart in `RefusalReport`. On a mesh without crop faces the crop
set is empty and nothing here changes.

Reading nothing but the mesh
----------------------------
:func:`reflex_trijunction_edges` takes a mesh and a tolerance. There is no parameter through
which a reference geometry, a target angle or an expected value could reach it, and the decision
is therefore available on data that has no reference at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from foambryo.geometry import compute_angles_tri_with_report, compute_trijunction_edges_with_report
from foambryo.junction_graph import (
    ComponentTensionSolution,
    partition_from_mesh,
    partition_over_surviving_junctions,
    solve_gauged_least_squares,
)
from foambryo.second_order_angle_reading import (
    CHORD_READING,
    CHORD_READING_REPORT,
    ORIENTED_CHORD_READING,
    AngleReadingReport,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from foambryo.dcel import DcelData
    from foambryo.geometry import TrijunctionEdge
    from foambryo.junction_graph import Interface, JunctionGraphPartition

#: Excess past a straight angle, in degrees, above which a wedge counts as reflex.
#:
#: Not a free parameter: `arccos` is ill-conditioned at a straight wedge (its derivative diverges
#: as the dot product approaches `-1`), so a wedge that is geometrically straight is returned with
#: an error of order `1e-6` degrees. A smaller tolerance would count that rounding as geometry.
#: The same constant governs the mesh-quality count in the reconstruction package.
REFLEX_TOLERANCE_DEG = 1e-6

#: Region triple `(a, b, c)`, `a < b < c`, naming one junction line.
Triple = tuple[int, int, int]

#: Geometric edge as an ordered `(min, max)` vertex-key pair.
GeometricEdge = tuple[int, int]


class WedgeRefusalPolicy(Enum):
    """Which observations an angle-reading estimator declines, and how. See the module docstring.

    - `NoRefusal`: **the default.** Nothing is declined and every result is bitwise what it was
      before this module existed.
    - `ExcludeEdgeFromLineMean`: a refused edge leaves its line's angle mean and its line's
      length; the line still reports, from what survives.
    - `DropLine`: a line carrying any refused edge does not report at all.
    - `DownweightLine`: every line reports its unchanged mean, with its row weight scaled by the
      fraction of its arclength that is not refused.
    """

    NoRefusal = 0
    ExcludeEdgeFromLineMean = 1
    DropLine = 2
    DownweightLine = 3


@dataclass(frozen=True)
class WedgeRefusal:
    """A refusal policy together with the set of edges it applies to.

    Attributes:
        policy: which of the three interventions to apply. `NoRefusal` by default, which is the
            shipped behaviour and is bitwise identical to not passing a refusal at all.
        edges: the geometric edges to refuse, as `(min, max)` vertex-key pairs, or `None` (the
            default) to take them from the mesh itself through
            :func:`reflex_trijunction_edges`. An explicit set exists so that a caller with its
            own criterion — a control arm that refuses a matched random set, a diagnostic, a flag
            computed upstream — runs through exactly this code path rather than a second copy of
            it. It is never a way to feed in a reference geometry: whatever the caller supplies
            is a set of edge keys, and nothing in this module consults anything else.
        tolerance_deg: excess past a straight angle above which a wedge is reflex, used only when
            `edges` is `None`.
        crop_edges: the edges declined because they touch a crop face, excluded from their
            line's mean whatever `policy` is (see "Two reasons to decline an edge" in the module
            docstring). Filled from the mesh by every estimator (:func:`refusal_with_crop`); a
            caller leaves it empty.
    """

    policy: WedgeRefusalPolicy = WedgeRefusalPolicy.NoRefusal
    edges: frozenset[GeometricEdge] | None = None
    tolerance_deg: float = REFLEX_TOLERANCE_DEG
    crop_edges: frozenset[GeometricEdge] = frozenset()

    def __post_init__(self) -> None:
        """Normalise explicit edge collections to `frozenset`s of ordered pairs."""
        if self.edges is not None and not isinstance(self.edges, frozenset):
            object.__setattr__(
                self,
                "edges",
                frozenset((min(u, v), max(u, v)) for u, v in self.edges),
            )
        if not isinstance(self.crop_edges, frozenset):
            object.__setattr__(
                self,
                "crop_edges",
                frozenset((min(u, v), max(u, v)) for u, v in self.crop_edges),
            )

    @property
    def refuses_nothing(self) -> bool:
        """True when this refusal cannot change any number, so the untouched path is taken."""
        return self.policy is WedgeRefusalPolicy.NoRefusal and not self.crop_edges


#: The shipped default: refuse nothing, and take the pre-existing code path unchanged.
NO_REFUSAL = WedgeRefusal()


@dataclass(frozen=True)
class RefusalReport:
    """How much of a mesh one policy declined, in the units the estimator actually uses.

    Every field is a count or a length, so a caller can put the refused share next to the
    accuracy number it produced. `n_units_*` are in whichever unit the estimator's rows are —
    lines for the line-averaged methods, edges for the per-edge one — and the edge and line
    counts are always both present so the two families can be compared.

    Attributes:
        policy: the policy these counts describe.
        n_edges: trijunctional edges the mesh has.
        n_edges_refused: how many of them the refusal names.
        n_lines: junction lines the mesh has.
        n_lines_without_row: lines left with no row at all by this policy.
        n_lines_downweighted: lines that keep a row whose weight or angle the policy changed.
        length_total: total trijunction arclength, summed over geometric edges.
        length_refused: arclength carried by the refused edges.
        n_units: rows' observation units before refusal.
        n_units_refused: units the policy removed.
        n_units_used: units that still carry a row. `n_units_refused + n_units_used == n_units`.
        surviving_lines: the region triples that still carry a row, sorted.
        n_edges_touching_crop: edges declined because they touch a crop face (0 without crop
            faces). They are not counted in `n_edges_refused`, which counts the policy's own.
        length_touching_crop: arclength carried by those edges.
    """

    policy: WedgeRefusalPolicy
    n_edges: int
    n_edges_refused: int
    n_lines: int
    n_lines_without_row: int
    n_lines_downweighted: int
    length_total: float
    length_refused: float
    n_units: int
    n_units_refused: int
    n_units_used: int
    surviving_lines: tuple[Triple, ...] = field(default=())
    n_edges_touching_crop: int = 0
    length_touching_crop: float = 0.0


def triple_of_edge(edge: TrijunctionEdge) -> Triple:
    """The sorted region triple of one trijunctional edge, i.e. which junction line it lies on.

    Args:
        edge (TrijunctionEdge): one geometric trijunctional edge.

    Returns:
        Triple: `(a, b, c)` with `a < b < c`, the three regions meeting on it.
    """
    regions = sorted({int(key[1]) for key in edge.keys})
    return (regions[0], regions[1], regions[2])


def reflex_trijunction_edges(
    mesh: DcelData,
    tolerance_deg: float = REFLEX_TOLERANCE_DEG,
) -> frozenset[GeometricEdge]:
    """Every geometric trijunctional edge of `mesh` whose mesh is locally folded.

    An edge is reflex when its widest **oriented** wedge exceeds a straight angle by more than
    `tolerance_deg`. The oriented chord reading is used whatever reading the estimator itself
    consumes, so that "reflex" has exactly one definition here and the chord reading — still the
    shipped default — has a well-defined refused set too.

    Reads the mesh and nothing else.

    Args:
        mesh (DcelData): Mesh to analyze.
        tolerance_deg (float, optional): excess past a straight angle, in degrees, above which a
            wedge counts as reflex. Defaults to :data:`REFLEX_TOLERANCE_DEG`; see that constant
            for why it is not a free parameter.

    Returns:
        frozenset[GeometricEdge]: `(min, max)` vertex-key pairs, one per folded edge.
    """
    edges, _report = compute_trijunction_edges_with_report(mesh, reading=ORIENTED_CHORD_READING)
    return _reflex_among(edges, tolerance_deg)


def _reflex_among(
    edges: Sequence[TrijunctionEdge],
    tolerance_deg: float = REFLEX_TOLERANCE_DEG,
) -> frozenset[GeometricEdge]:
    """The folded edges among records already read under the oriented chord convention."""
    limit = np.pi + np.radians(tolerance_deg)
    return frozenset(edge.vertices for edge in edges if max(edge.angles) > limit)


def resolve_refused_edges(mesh: DcelData, refusal: WedgeRefusal) -> frozenset[GeometricEdge]:
    """The edges `refusal` declines on this mesh: its explicit set, or the folded ones.

    Args:
        mesh (DcelData): Mesh to analyze.
        refusal (WedgeRefusal): the policy and, optionally, an explicit edge set.

    Returns:
        frozenset[GeometricEdge]: empty under `NoRefusal`. The crop edges are not part of it:
        they are `refusal.crop_edges`.
    """
    if refusal.policy is WedgeRefusalPolicy.NoRefusal:
        return frozenset()
    if refusal.edges is not None:
        return refusal.edges
    return reflex_trijunction_edges(mesh, refusal.tolerance_deg)


def refusal_with_crop(mesh: DcelData, refusal: WedgeRefusal) -> WedgeRefusal:
    """`refusal`, with the mesh's crop edges added as the second reason to decline an edge.

    Every estimator calls this on its own refusal before reading any angle, so the crop is
    honoured whatever the caller passed. Idempotent; on a mesh without crop faces it returns
    `refusal` itself, unchanged, and the untouched code path stays untouched.

    Args:
        mesh (DcelData): the mesh, whose `crop_faces` decide.
        refusal (WedgeRefusal): the caller's refusal.

    Returns:
        WedgeRefusal: the same policy, explicit set and tolerance, with `crop_edges` set to every
        edge that has a vertex on a crop face (`foambryo.crop_faces.crop_edges`).
    """
    from foambryo.crop_faces import crop_edges  # local import: crop_faces is a leaf module

    if not getattr(mesh, "has_crop_faces", False) or refusal.crop_edges:
        return refusal
    return WedgeRefusal(refusal.policy, refusal.edges, refusal.tolerance_deg, crop_edges(mesh))


def _edge_records_and_refused(
    mesh: DcelData,
    refusal: WedgeRefusal,
) -> tuple[list[TrijunctionEdge], frozenset[GeometricEdge]]:
    """One oriented pass over the mesh giving both the edge records and the refused set.

    The edge set and the edge lengths do not depend on the angle reading — only the angles do
    (`geometry.compute_trijunction_edges_with_report`) — so these records serve as the line
    bookkeeping for any reading while carrying the oriented angles the refusal is defined on.
    """
    edges, _report = compute_trijunction_edges_with_report(mesh, reading=ORIENTED_CHORD_READING)
    if refusal.policy is WedgeRefusalPolicy.NoRefusal:
        refused: frozenset[GeometricEdge] = frozenset()
    elif refusal.edges is not None:
        refused = refusal.edges
    else:
        refused = _reflex_among(edges, refusal.tolerance_deg)
    return edges, refused


@dataclass
class _LineStats:
    """Per-junction-line edge counts and arclengths, split by what the refusal names."""

    n_edges: int = 0
    n_refused: int = 0
    length: float = 0.0
    length_refused: float = 0.0


def _line_stats(
    edges: Sequence[TrijunctionEdge],
    refused: Collection[GeometricEdge],
    crop: Collection[GeometricEdge] = frozenset(),
) -> dict[Triple, _LineStats]:
    """Edge counts and arclengths per junction line, split by what the refusal names.

    Survival is decided on the **edge count**, never on the arclength: a line all of whose edges
    are refused has no surviving observation whatever its lengths happen to be, and a comparison
    of two accumulated floats is not the right test for that.

    Args:
        edges (Sequence[TrijunctionEdge]): every trijunctional edge of the mesh.
        refused (Collection[GeometricEdge]): the edges the refusal's policy names.
        crop (Collection[GeometricEdge], optional): the edges that touch a crop face. They are
            left out of every line, as if absent; a line with no other edge is absent from the
            result. Defaults to none.

    Returns:
        dict[Triple, _LineStats]: one entry per junction line with an edge off the crop.
    """
    stats: dict[Triple, _LineStats] = {}
    for edge in edges:
        if edge.vertices in crop:
            continue
        entry = stats.setdefault(triple_of_edge(edge), _LineStats())
        entry.n_edges += 1
        entry.length += float(edge.length)
        if edge.vertices in refused:
            entry.n_refused += 1
            entry.length_refused += float(edge.length)
    return stats


def refusal_report(
    mesh: DcelData,
    refusal: WedgeRefusal,
    *,
    per_edge_rows: bool = False,
) -> RefusalReport:
    """Count what `refusal` declines on `mesh`, in the unit the estimator's rows use.

    Args:
        mesh (DcelData): Mesh to analyze.
        refusal (WedgeRefusal): the policy to account for.
        per_edge_rows (bool, optional): `True` when the estimator's observation unit is a single
            trijunctional edge (the per-edge method), `False` (the default) when it is a whole
            junction line. Keyword-only.

    Returns:
        RefusalReport: the counts, including which lines still carry a row.
    """
    refusal = refusal_with_crop(mesh, refusal)
    edges, refused = _edge_records_and_refused(mesh, refusal)
    crop = refusal.crop_edges
    stats = _line_stats(edges, refused, crop)
    lines_on_mesh = {triple_of_edge(edge) for edge in edges}
    # For a per-edge estimator the unit is one edge, so `DropLine` drops the refused rows and
    # leaves a line reporting from whatever survives -- the same line rule `ExcludeEdgeFromLineMean`
    # follows. See :func:`per_edge_rows`.
    effective = refusal.policy
    if per_edge_rows and effective is WedgeRefusalPolicy.DropLine:
        effective = WedgeRefusalPolicy.ExcludeEdgeFromLineMean
    keep_line = _surviving_line_scales(effective, stats)

    on_crop = [edge for edge in edges if edge.vertices in crop]
    if per_edge_rows:
        n_units = len(edges)
        if refusal.policy is WedgeRefusalPolicy.DownweightLine:
            n_units_used = sum(1 for edge in edges if edge.vertices not in crop and triple_of_edge(edge) in keep_line)
        else:
            n_units_used = sum(1 for edge in edges if edge.vertices not in refused and edge.vertices not in crop)
    else:
        n_units, n_units_used = len(lines_on_mesh), len(keep_line)

    return RefusalReport(
        policy=refusal.policy,
        n_edges=len(edges),
        n_edges_refused=sum(entry.n_refused for entry in stats.values()),
        n_lines=len(lines_on_mesh),
        n_lines_without_row=len(lines_on_mesh) - len(keep_line),
        n_lines_downweighted=sum(1 for triple in keep_line if stats[triple].n_refused),
        length_total=float(sum(entry.length for entry in stats.values()) + sum(float(e.length) for e in on_crop)),
        length_refused=float(sum(entry.length_refused for entry in stats.values())),
        n_units=n_units,
        n_units_refused=n_units - n_units_used,
        n_units_used=n_units_used,
        surviving_lines=tuple(sorted(keep_line)),
        n_edges_touching_crop=len(on_crop),
        length_touching_crop=float(sum(float(e.length) for e in on_crop)),
    )


def _surviving_line_scales(
    policy: WedgeRefusalPolicy,
    stats: dict[Triple, _LineStats],
) -> dict[Triple, float]:
    """Which lines keep a row under `policy`, and by what factor their weight is scaled.

    The scale is `1.0` for every policy but `DownweightLine`, where it is the line's unrefused
    arclength fraction. A line with no surviving edge is absent from the result: under
    `DownweightLine` its row weight would be zero, and a zero-weight row is the same as no row —
    treating it as no row is what keeps the gauge structure and the counts honest.

    Args:
        policy (WedgeRefusalPolicy): the policy.
        stats (dict[Triple, _LineStats]): per-line counts and arclengths.

    Returns:
        dict[Triple, float]: surviving line -> weight scale.
    """
    if policy is WedgeRefusalPolicy.NoRefusal:
        return dict.fromkeys(stats, 1.0)
    if policy is WedgeRefusalPolicy.DropLine:
        return {triple: 1.0 for triple, entry in stats.items() if entry.n_refused == 0}
    if policy is WedgeRefusalPolicy.ExcludeEdgeFromLineMean:
        return {triple: 1.0 for triple, entry in stats.items() if entry.n_refused < entry.n_edges}
    return {
        triple: (entry.length - entry.length_refused) / entry.length
        for triple, entry in stats.items()
        if entry.n_refused < entry.n_edges and entry.length > 0.0
    }


@dataclass(frozen=True)
class LineAveragedAngles:
    """The three maps a line-averaged builder consumes, after a refusal policy has been applied.

    Attributes:
        dict_rad: mean wedge angle in radians, keyed by wedge key, over the edges the policy
            left in the line.
        dict_length: accumulated length per wedge key, in the historical `unique=False`
            convention the builders expect, scaled by the policy's per-line weight factor.
        partition: the junction-graph partition **over the surviving lines**, so an interface
            left with no surviving line is reported undetermined rather than given a number.
        report: the angle reading's own coverage report, unchanged.
    """

    dict_rad: dict[Triple, float]
    dict_length: dict[Triple, float]
    partition: JunctionGraphPartition
    report: AngleReadingReport


def line_averaged_angles(
    mesh: DcelData,
    *,
    reading: str = CHORD_READING,
    refusal: WedgeRefusal = NO_REFUSAL,
) -> LineAveragedAngles:
    """Angles, lengths and gauge partition for a line-averaged builder, under a refusal policy.

    With :data:`NO_REFUSAL` this calls `geometry.compute_angles_tri_with_report(unique=False)`
    and `junction_graph.partition_from_mesh` and nothing else, in that order — so the untouched
    path really is untouched and every existing number is bitwise unchanged.

    Args:
        mesh (DcelData): Mesh to analyze.
        reading (str, optional): one of
            `foambryo.second_order_angle_reading.ANGLE_READINGS`. Keyword-only.
        refusal (WedgeRefusal, optional): which observations to decline. Keyword-only, defaults
            to refusing nothing.

    Returns:
        LineAveragedAngles: the maps, the partition and the coverage report. `dict_rad` is empty
            when the policy leaves no line reporting, which the builder turns into an
            all-undetermined answer rather than an exception.
    """
    refusal = refusal_with_crop(mesh, refusal)
    if refusal.refuses_nothing:
        dict_rad, _deg, dict_length, report = compute_angles_tri_with_report(mesh, unique=False, reading=reading)
        return LineAveragedAngles(dict_rad, dict_length, partition_from_mesh(mesh), report)

    edges, refused = _edge_records_and_refused(mesh, refusal)
    crop = refusal.crop_edges
    scales = _surviving_line_scales(refusal.policy, _line_stats(edges, refused, crop))

    # `E` is the only policy that changes what is measured, so it is the only one that re-runs the
    # accumulation; `D` and `W` read the untouched accumulation and then drop or rescale lines.
    # Crop edges change what is measured under every policy, so with any of them the accumulation
    # is re-run without them.
    skip = refused if refusal.policy is WedgeRefusalPolicy.ExcludeEdgeFromLineMean else None
    if crop:
        skip = crop | (skip or frozenset())
    dict_rad, _deg, dict_length, report = compute_angles_tri_with_report(
        mesh,
        unique=False,
        reading=reading,
        skip_edges=skip,
    )

    kept_rad: dict[Triple, float] = {}
    kept_length: dict[Triple, float] = {}
    surviving: set[Triple] = set()
    for key, angle in dict_rad.items():
        triple = tuple(sorted(int(label) for label in key))
        scale = scales.get(triple)
        if scale is None:
            continue
        kept_rad[key] = angle
        kept_length[key] = dict_length[key] * scale
        surviving.add(triple)

    partition = partition_over_surviving_junctions(mesh.f[:, :3], mesh.f[:, 3:5], surviving)
    return LineAveragedAngles(kept_rad, kept_length, partition, report)


@dataclass(frozen=True)
class PerEdgeRows:
    """Which of a mesh's trijunctional edges still carry a row, and how their weight is scaled.

    Attributes:
        keep: index, into the caller's own `trijunction_edges` list, of every edge that still
            carries a row, ascending.
        weight_scale: the multiplier on each kept row's weight, aligned with `keep`. All ones for
            every policy but `DownweightLine`, where it is the edge's line's unrefused arclength
            fraction.
        surviving_lines: the region triples still carrying at least one row.
        n_refused: how many edges lost their row, for either reason (the policy or the crop).
    """

    keep: tuple[int, ...]
    weight_scale: tuple[float, ...]
    surviving_lines: frozenset[Triple]
    n_refused: int


def per_edge_rows(
    mesh: DcelData,
    trijunction_edges: Sequence[TrijunctionEdge],
    refusal: WedgeRefusal = NO_REFUSAL,
) -> PerEdgeRows:
    """Apply a refusal policy to a per-edge estimator, whose observation unit is one edge.

    `ExcludeEdgeFromLineMean` and `DropLine` coincide here and the coincidence is exact, not
    approximate: the unit is a single edge, so excluding a folded edge from its own unit's mean
    empties that unit, and an empty unit has no row — which is what dropping it means. They are
    the same intervention. `DownweightLine` is genuinely different: it keeps every row, including
    the folded ones, and scales the whole line's rows by its unrefused arclength fraction.

    Args:
        mesh (DcelData): Mesh to analyze — read only to decide which edges are folded.
        trijunction_edges (Sequence[TrijunctionEdge]): the estimator's own edge list, under
            whichever reading it uses. The edge set and lengths do not depend on the reading.
        refusal (WedgeRefusal, optional): which observations to decline. Defaults to refusing
            nothing, in which case every edge is kept with scale `1.0`.

    Returns:
        PerEdgeRows: the surviving rows and their weight scales.
    """
    refusal = refusal_with_crop(mesh, refusal)
    if refusal.refuses_nothing:
        return PerEdgeRows(
            keep=tuple(range(len(trijunction_edges))),
            weight_scale=(1.0,) * len(trijunction_edges),
            surviving_lines=frozenset(triple_of_edge(edge) for edge in trijunction_edges),
            n_refused=0,
        )

    refused = resolve_refused_edges(mesh, refusal)
    crop = refusal.crop_edges
    scales = _surviving_line_scales(refusal.policy, _line_stats(trijunction_edges, refused, crop))
    downweighting = refusal.policy is WedgeRefusalPolicy.DownweightLine

    keep: list[int] = []
    weight_scale: list[float] = []
    surviving: set[Triple] = set()
    for index, edge in enumerate(trijunction_edges):
        if edge.vertices in crop:
            # Touches a crop face: never a row, whatever the policy.
            continue
        if downweighting:
            # The line, not the edge, is the thing down-weighted: every row survives, including
            # the folded ones, scaled by its line's unrefused arclength fraction.
            scale = scales.get(triple_of_edge(edge))
            if scale is None:
                continue
        else:
            # `ExcludeEdgeFromLineMean` and `DropLine` both reduce here to dropping the refused
            # row, because this estimator's unit *is* one edge: excluding a folded edge from its
            # own unit's mean empties that unit, and an empty unit has no row.
            if edge.vertices in refused:
                continue
            scale = 1.0
        keep.append(index)
        weight_scale.append(scale)
        surviving.add(triple_of_edge(edge))

    return PerEdgeRows(
        keep=tuple(keep),
        weight_scale=tuple(weight_scale),
        surviving_lines=frozenset(surviving),
        n_refused=len(trijunction_edges) - len(keep),
    )


def all_interfaces_undetermined(
    keys: Sequence[Interface],
    mean_tension: float,
    partition: JunctionGraphPartition,
) -> ComponentTensionSolution:
    """The answer when a policy leaves no observation at all: every interface `nan`.

    Assembles the normalisation's row alone and hands it to
    `junction_graph.solve_gauged_least_squares`, so the undetermined interfaces are produced by
    the same machinery that already produces them for a closed surface, and the result carries
    the same `undetermined` / `component_of` / `gauge_notes` structure every caller reads.

    Args:
        keys (Sequence[Interface]): interface key per column, in order.
        mean_tension (float): the gauge value the caller requested.
        partition (JunctionGraphPartition): the partition over the surviving lines — with no
            surviving line every interface is isolated.

    Returns:
        ComponentTensionSolution: `float("nan")` at every interface.
    """
    n_interfaces = len(keys)
    matrix_m = np.ones((1, n_interfaces))
    vector_b = np.array([n_interfaces * mean_tension], dtype=np.float64)
    return solve_gauged_least_squares(matrix_m, vector_b, keys, mean_tension, partition)


def require_refusal_is_honoured(method_name: str, refusal: WedgeRefusal) -> None:
    """Refuse a non-default policy on an estimator that reads no wedge angle.

    The counterpart of `tension_inference._require_reading_is_honoured`, and for the same reason:
    an estimator that assembles an area/volume or force-balance operator has no trijunction row
    to decline, so silently accepting a policy would hand the caller a result they believe
    refused its folded observations and which never had any.

    Args:
        method_name (str): the estimator's name, for the message.
        refusal (WedgeRefusal): the requested refusal.

    Raises:
        ValueError: if `refusal` is not `NO_REFUSAL`'s policy.
    """
    if refusal.refuses_nothing:
        return
    message = (
        f"{method_name} does not build its rows from measured trijunction wedge angles, so "
        f"{refusal.policy} would have nothing to decline. Refusing rather than returning a "
        "result that looks refused and is not. Use an angle-reading method, or "
        "WedgeRefusalPolicy.NoRefusal."
    )
    raise ValueError(message)


#: The trivial report a refusal-free call carries, re-exported so callers need not import two
#: modules to name it.
NO_REFUSAL_READING_REPORT = CHORD_READING_REPORT
